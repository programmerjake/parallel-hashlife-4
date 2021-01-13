#![no_std]
use array::{Array, ArrayRepr};
use core::{fmt::Debug, hash::Hash};
use index_vec::{IndexVec, IndexVecExt};

extern crate alloc;
#[cfg(any(test, feature = "std"))]
extern crate std;

pub mod array;
mod array_vec;
pub mod index_vec;
pub mod parallel;
pub mod parallel_hash_table;
pub mod simple;
#[cfg(any(test, feature = "std"))]
pub mod std_support;
pub mod traits;

pub trait HasErrorType {
    type Error;
}

impl<T: ?Sized + HasErrorType> HasErrorType for &'_ T {
    type Error = T::Error;
}

pub trait HasNodeType<const DIMENSION: usize>
where
    Array<Self::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    type NodeId: Clone + Debug + ArrayRepr<3, DIMENSION> + ArrayRepr<2, DIMENSION>;
}

impl<T, const DIMENSION: usize> HasNodeType<DIMENSION> for &'_ T
where
    T: ?Sized + HasNodeType<DIMENSION>,
    Array<T::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    type NodeId = T::NodeId;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct NodeAndLevel<Node> {
    pub node: Node,
    pub level: usize,
}

impl<T> NodeAndLevel<T> {
    pub fn map_node<R, F: FnOnce(T) -> R>(self, f: F) -> NodeAndLevel<R> {
        NodeAndLevel {
            node: f(self.node),
            level: self.level,
        }
    }
    pub fn try_map_node<R, E, F: FnOnce(T) -> Result<R, E>>(
        self,
        f: F,
    ) -> Result<NodeAndLevel<R>, E> {
        Ok(NodeAndLevel {
            node: f(self.node)?,
            level: self.level,
        })
    }
    pub const fn as_ref(&self) -> NodeAndLevel<&T> {
        let NodeAndLevel { ref node, level } = *self;
        NodeAndLevel { node, level }
    }
    pub fn as_mut(&mut self) -> NodeAndLevel<&mut T> {
        let NodeAndLevel {
            ref mut node,
            level,
        } = *self;
        NodeAndLevel { node, level }
    }
}

pub trait HasLeafType<const DIMENSION: usize>
where
    Array<Self::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    type Leaf: Clone + Debug + Default + ArrayRepr<3, DIMENSION> + ArrayRepr<2, DIMENSION>;
}

impl<T, const DIMENSION: usize> HasLeafType<DIMENSION> for &'_ T
where
    T: ?Sized + HasLeafType<DIMENSION>,
    Array<T::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    type Leaf = T::Leaf;
}

pub trait LeafStep<const DIMENSION: usize>: HasLeafType<DIMENSION> + HasErrorType
where
    Array<Self::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn leaf_step(
        &self,
        neighborhood: Array<Self::Leaf, 3, DIMENSION>,
    ) -> Result<Self::Leaf, Self::Error>;
}

impl<T, const DIMENSION: usize> LeafStep<DIMENSION> for &'_ T
where
    T: ?Sized + LeafStep<DIMENSION>,
    Array<T::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn leaf_step(
        &self,
        neighborhood: Array<Self::Leaf, 3, DIMENSION>,
    ) -> Result<Self::Leaf, Self::Error> {
        (**self).leaf_step(neighborhood)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum NodeOrLeaf<Node, Leaf> {
    Node(Node),
    Leaf(Leaf),
}

impl<Node, Leaf> NodeOrLeaf<Node, Leaf> {
    pub const fn as_ref(&self) -> NodeOrLeaf<&Node, &Leaf> {
        match self {
            NodeOrLeaf::Node(v) => NodeOrLeaf::Node(v),
            NodeOrLeaf::Leaf(v) => NodeOrLeaf::Leaf(v),
        }
    }
    pub fn as_mut(&mut self) -> NodeOrLeaf<&mut Node, &mut Leaf> {
        match self {
            NodeOrLeaf::Node(v) => NodeOrLeaf::Node(v),
            NodeOrLeaf::Leaf(v) => NodeOrLeaf::Leaf(v),
        }
    }
    pub const fn is_node(&self) -> bool {
        matches!(self, NodeOrLeaf::Node(_))
    }
    pub const fn is_leaf(&self) -> bool {
        matches!(self, NodeOrLeaf::Leaf(_))
    }
    pub fn node(self) -> Option<Node> {
        match self {
            NodeOrLeaf::Node(v) => Some(v),
            NodeOrLeaf::Leaf(_) => None,
        }
    }
    pub fn leaf(self) -> Option<Leaf> {
        match self {
            NodeOrLeaf::Leaf(v) => Some(v),
            NodeOrLeaf::Node(_) => None,
        }
    }
    pub fn map_node<T, F: FnOnce(Node) -> T>(self, f: F) -> NodeOrLeaf<T, Leaf> {
        match self {
            NodeOrLeaf::Node(v) => NodeOrLeaf::Node(f(v)),
            NodeOrLeaf::Leaf(v) => NodeOrLeaf::Leaf(v),
        }
    }
    pub fn map_leaf<T, F: FnOnce(Leaf) -> T>(self, f: F) -> NodeOrLeaf<Node, T> {
        match self {
            NodeOrLeaf::Node(v) => NodeOrLeaf::Node(v),
            NodeOrLeaf::Leaf(v) => NodeOrLeaf::Leaf(f(v)),
        }
    }
}

pub trait HashlifeData<'a, const DIMENSION: usize>:
    HasErrorType + LeafStep<DIMENSION> + HasNodeType<DIMENSION>
where
    IndexVec<DIMENSION>: IndexVecExt,
    Array<Self::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Self::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    /// key.level is the level of the nodes in key
    fn intern_non_leaf_node(
        &'a self,
        key: NodeAndLevel<Array<Self::NodeId, 2, DIMENSION>>,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error>;
    fn intern_leaf_node(
        &'a self,
        key: Array<Self::Leaf, 2, DIMENSION>,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error>;
    /// retval.node().level is the level of the nodes in retval.node()
    fn get_node_key(
        &'a self,
        node: NodeAndLevel<Self::NodeId>,
    ) -> NodeOrLeaf<NodeAndLevel<Array<Self::NodeId, 2, DIMENSION>>, Array<Self::Leaf, 2, DIMENSION>>;
    fn get_non_leaf_node_next(
        &'a self,
        node: NodeAndLevel<Self::NodeId>,
    ) -> Option<NodeAndLevel<Self::NodeId>>;
    fn fill_non_leaf_node_next(
        &'a self,
        node: NodeAndLevel<Self::NodeId>,
        new_next: NodeAndLevel<Self::NodeId>,
    );
    fn get_empty_node(&'a self, level: usize) -> Result<NodeAndLevel<Self::NodeId>, Self::Error>;
    fn get_center(
        &'a self,
        node: NodeAndLevel<Self::NodeId>,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        assert_ne!(node.level, 0, "leaf node has no center");
        let node_key = self.get_node_key(node).node().unwrap();
        if node_key.level == 0 {
            let key = Array::build_array(|index| {
                self.get_node_key(node_key.as_ref().map_node(|node| node[index].clone()))
                    .leaf()
                    .unwrap()[index.map(|v| 1 - v)]
                .clone()
            });
            self.intern_leaf_node(key)
        } else {
            let key = Array::build_array(|index| {
                self.get_node_key(node_key.as_ref().map_node(|node| node[index].clone()))
                    .node()
                    .unwrap()
                    .node[index.map(|v| 1 - v)]
                .clone()
            });
            self.intern_non_leaf_node(NodeAndLevel {
                node: key,
                level: node_key.level - 1,
            })
        }
    }
    fn expand_root(
        &'a self,
        node: NodeAndLevel<Self::NodeId>,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        let level = node.level;
        match self.get_node_key(node) {
            NodeOrLeaf::Leaf(node_key) => {
                let final_key =
                    Array::try_build_array(|outer_index| -> Result<Self::NodeId, Self::Error> {
                        let key = Array::build_array(|inner_index| {
                            if outer_index.map(|v| 1 - v) == inner_index {
                                node_key[outer_index].clone()
                            } else {
                                Self::Leaf::default()
                            }
                        });
                        Ok(self.intern_leaf_node(key)?.node)
                    })?;
                self.intern_non_leaf_node(NodeAndLevel {
                    node: final_key,
                    level: 0,
                })
            }
            NodeOrLeaf::Node(node_key) => {
                let empty_node = self.get_empty_node(level - 1)?.node;
                let final_key =
                    Array::try_build_array(|outer_index| -> Result<Self::NodeId, Self::Error> {
                        let key = Array::build_array(|inner_index| {
                            if outer_index.map(|v| 1 - v) == inner_index {
                                node_key.node[outer_index].clone()
                            } else {
                                empty_node.clone()
                            }
                        });
                        Ok(self
                            .intern_non_leaf_node(NodeAndLevel {
                                node: key,
                                level: node_key.level,
                            })?
                            .node)
                    })?;
                self.intern_non_leaf_node(NodeAndLevel {
                    node: final_key,
                    level,
                })
            }
        }
    }
}

impl<'a, 'b, T, const DIMENSION: usize> HashlifeData<'a, DIMENSION> for &'b T
where
    T: ?Sized + HashlifeData<'a, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    Array<T::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<T::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn intern_non_leaf_node(
        &'a self,
        key: NodeAndLevel<Array<Self::NodeId, 2, DIMENSION>>,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        (**self).intern_non_leaf_node(key)
    }
    fn intern_leaf_node(
        &'a self,
        key: Array<Self::Leaf, 2, DIMENSION>,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        (**self).intern_leaf_node(key)
    }
    fn get_node_key(
        &'a self,
        node: NodeAndLevel<Self::NodeId>,
    ) -> NodeOrLeaf<NodeAndLevel<Array<Self::NodeId, 2, DIMENSION>>, Array<Self::Leaf, 2, DIMENSION>>
    {
        (**self).get_node_key(node)
    }
    fn get_non_leaf_node_next(
        &'a self,
        node: NodeAndLevel<Self::NodeId>,
    ) -> Option<NodeAndLevel<Self::NodeId>> {
        (**self).get_non_leaf_node_next(node)
    }
    fn fill_non_leaf_node_next(
        &'a self,
        node: NodeAndLevel<Self::NodeId>,
        new_next: NodeAndLevel<Self::NodeId>,
    ) {
        (**self).fill_non_leaf_node_next(node, new_next)
    }
    fn get_empty_node(&'a self, level: usize) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        (**self).get_empty_node(level)
    }
    fn expand_root(
        &'a self,
        node: NodeAndLevel<Self::NodeId>,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        (**self).expand_root(node)
    }
}
