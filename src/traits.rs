use core::fmt;

use crate::{
    array::{Array, ArrayRepr},
    index_vec::{IndexVec, IndexVecExt},
    NodeAndLevel, NodeOrLeaf,
};

pub mod parallel;
pub mod serial;

pub trait HasErrorType {
    type Error: 'static;
}

impl<'a, T: ?Sized + HasErrorType> HasErrorType for &'_ T {
    type Error = T::Error;
}

pub trait HasNodeType<'a, const DIMENSION: usize>
where
    Array<Self::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Self::NodeId, 2, DIMENSION>: 'a,
    Array<Self::NodeId, 3, DIMENSION>: 'a,
    Array<Array<Self::NodeId, 2, DIMENSION>, 2, DIMENSION>: 'a,
{
    type NodeId: Clone + fmt::Debug + ArrayRepr<3, DIMENSION> + ArrayRepr<2, DIMENSION> + 'a;
}

impl<'a, T, const DIMENSION: usize> HasNodeType<'a, DIMENSION> for &'_ T
where
    T: ?Sized + HasNodeType<'a, DIMENSION>,
    Array<T::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    type NodeId = T::NodeId;
}

pub trait HasLeafType<'a, const DIMENSION: usize>
where
    Array<Self::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Self::Leaf, 2, DIMENSION>: 'a,
    Array<Self::Leaf, 3, DIMENSION>: 'a,
    Array<Array<Self::Leaf, 2, DIMENSION>, 2, DIMENSION>: 'a,
{
    type Leaf: Clone + fmt::Debug + Default + ArrayRepr<3, DIMENSION> + ArrayRepr<2, DIMENSION> + 'a;
}

impl<'a, T, const DIMENSION: usize> HasLeafType<'a, DIMENSION> for &'_ T
where
    T: ?Sized + HasLeafType<'a, DIMENSION>,
    Array<T::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    type Leaf = T::Leaf;
}

pub trait LeafStep<'a, const DIMENSION: usize>: HasLeafType<'a, DIMENSION> + HasErrorType
where
    Array<Self::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn leaf_step(
        &'a self,
        neighborhood: Array<Self::Leaf, 3, DIMENSION>,
    ) -> Result<Self::Leaf, Self::Error>;
}

impl<'a, T, const DIMENSION: usize> LeafStep<'a, DIMENSION> for &'_ T
where
    T: ?Sized + LeafStep<'a, DIMENSION>,
    Array<T::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn leaf_step(
        &'a self,
        neighborhood: Array<Self::Leaf, 3, DIMENSION>,
    ) -> Result<Self::Leaf, Self::Error> {
        (**self).leaf_step(neighborhood)
    }
}

pub trait HashlifeData<'a, const DIMENSION: usize>:
    HasErrorType + HasLeafType<'a, DIMENSION> + HasNodeType<'a, DIMENSION>
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
