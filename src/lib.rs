#![cfg_attr(not(test), no_std)]
use array::{Array, ArrayRepr};
use core::{fmt::Debug, hash::Hash};
use index_vec::{IndexVec, IndexVecExt};

extern crate alloc;

pub mod array;
mod arrayvec;
pub mod index_vec;
pub mod parallel;
pub mod serial;
pub mod simple;

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

pub trait HashlifeData<const DIMENSION: usize>:
    HasErrorType + LeafStep<DIMENSION> + HasNodeType<DIMENSION>
where
    IndexVec<DIMENSION>: IndexVecExt,
    Array<Self::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Self::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn intern_node(
        &self,
        key: NodeOrLeaf<Array<Self::NodeId, 2, DIMENSION>, Array<Self::Leaf, 2, DIMENSION>>,
        level: usize,
    ) -> Result<Self::NodeId, Self::Error>;
    fn get_node_key(
        &self,
        node: Self::NodeId,
        level: usize,
    ) -> NodeOrLeaf<Array<Self::NodeId, 2, DIMENSION>, Array<Self::Leaf, 2, DIMENSION>>;
    fn get_node_next(&self, node: Self::NodeId, level: usize) -> Option<Self::NodeId>;
    fn fill_node_next(&self, node: Self::NodeId, level: usize, new_next: Self::NodeId);
    fn get_empty_node(&self, level: usize) -> Result<Self::NodeId, Self::Error>;
    fn expand_root(&self, node: Self::NodeId, level: usize) -> Result<Self::NodeId, Self::Error> {
        match self.get_node_key(node, level) {
            NodeOrLeaf::Leaf(node_key) => {
                assert_eq!(level, 0);
                let final_key =
                    Array::try_build_array(|outer_index| -> Result<Self::NodeId, Self::Error> {
                        let key = Array::build_array(|inner_index| {
                            if outer_index.map(|v| 1 - v) == inner_index {
                                node_key[outer_index].clone()
                            } else {
                                Self::Leaf::default()
                            }
                        });
                        self.intern_node(NodeOrLeaf::Leaf(key), 0)
                    })?;
                self.intern_node(NodeOrLeaf::Node(final_key), 1)
            }
            NodeOrLeaf::Node(node_key) => {
                assert_ne!(level, 0);
                let empty_node = self.get_empty_node(level - 1)?;
                let final_key =
                    Array::try_build_array(|outer_index| -> Result<Self::NodeId, Self::Error> {
                        let key = Array::build_array(|inner_index| {
                            if outer_index.map(|v| 1 - v) == inner_index {
                                node_key[outer_index].clone()
                            } else {
                                empty_node.clone()
                            }
                        });
                        self.intern_node(NodeOrLeaf::Node(key), level)
                    })?;
                self.intern_node(NodeOrLeaf::Node(final_key), level + 1)
            }
        }
    }
}

impl<T, const DIMENSION: usize> HashlifeData<DIMENSION> for &T
where
    T: ?Sized + HashlifeData<DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    Array<T::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<T::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn intern_node(
        &self,
        key: NodeOrLeaf<Array<Self::NodeId, 2, DIMENSION>, Array<Self::Leaf, 2, DIMENSION>>,
        level: usize,
    ) -> Result<Self::NodeId, Self::Error> {
        (**self).intern_node(key, level)
    }
    fn get_node_key(
        &self,
        node: Self::NodeId,
        level: usize,
    ) -> NodeOrLeaf<Array<Self::NodeId, 2, DIMENSION>, Array<Self::Leaf, 2, DIMENSION>> {
        (**self).get_node_key(node, level)
    }
    fn get_node_next(&self, node: Self::NodeId, level: usize) -> Option<Self::NodeId> {
        (**self).get_node_next(node, level)
    }
    fn fill_node_next(&self, node: Self::NodeId, level: usize, new_next: Self::NodeId) {
        (**self).fill_node_next(node, level, new_next)
    }
    fn get_empty_node(&self, level: usize) -> Result<Self::NodeId, Self::Error> {
        (**self).get_empty_node(level)
    }
    fn expand_root(&self, node: Self::NodeId, level: usize) -> Result<Self::NodeId, Self::Error> {
        (**self).expand_root(node, level)
    }
}
