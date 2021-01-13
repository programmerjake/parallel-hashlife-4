use core::fmt;

use crate::{
    array::{Array, ArrayRepr},
    index_vec::{IndexVec, IndexVecExt},
    parallel_hash_table::{self, ParallelHashTable, WaitWake},
    traits::parallel::ParallelBuildArray,
    HasErrorType, HasLeafType, HasNodeType, HashlifeData, LeafStep, NodeAndLevel, NodeOrLeaf,
};
use crossbeam_utils::atomic::AtomicCell;
use hashbrown::hash_map::DefaultHashBuilder;

type Node<'a, Leaf, const DIMENSION: usize> =
    NodeOrLeaf<NodeData<'a, Leaf, DIMENSION>, Array<Leaf, 2, DIMENSION>>;

#[repr(transparent)]
pub struct NodeId<'a, Leaf, const DIMENSION: usize>(&'a Node<'a, Leaf, DIMENSION>)
where
    Leaf: ArrayRepr<2, DIMENSION>,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>;

impl<'a, Leaf, const DIMENSION: usize> fmt::Debug for NodeId<'a, Leaf, DIMENSION>
where
    Leaf: ArrayRepr<2, DIMENSION> + fmt::Debug,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.0, f)
    }
}

impl<'a, Leaf, const DIMENSION: usize> Clone for NodeId<'a, Leaf, DIMENSION>
where
    Leaf: ArrayRepr<2, DIMENSION> + fmt::Debug,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, Leaf, const DIMENSION: usize> Copy for NodeId<'a, Leaf, DIMENSION>
where
    Leaf: ArrayRepr<2, DIMENSION> + fmt::Debug,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
}

struct NodeData<'a, Leaf, const DIMENSION: usize>
where
    Leaf: ArrayRepr<2, DIMENSION>,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    key: Array<NodeId<'a, Leaf, DIMENSION>, 2, DIMENSION>,
    next: AtomicCell<Option<NodeId<'a, Leaf, DIMENSION>>>,
}

impl<'a, Leaf, const DIMENSION: usize> fmt::Debug for NodeData<'a, Leaf, DIMENSION>
where
    Leaf: ArrayRepr<2, DIMENSION> + fmt::Debug,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeData")
            .field("key", &self.key)
            .field("next", &self.next)
            .finish()
    }
}

pub struct Parallel<'a, Base, W, const DIMENSION: usize>
where
    Base: HasLeafType<DIMENSION>,
    NodeId<'a, Base::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Base::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    hash_table: ParallelHashTable<Node<'a, Base::Leaf, DIMENSION>, W>,
    hasher: DefaultHashBuilder,
    base: Base,
}

impl<'a, Base, W, const DIMENSION: usize> Parallel<'a, Base, W, DIMENSION>
where
    Base: HasLeafType<DIMENSION>,
    NodeId<'a, Base::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Base::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    pub fn new(base: Base, log2_capacity: u32, wait_waker: W) -> Self {
        Self {
            hash_table: ParallelHashTable::new(log2_capacity, wait_waker),
            hasher: DefaultHashBuilder::new(),
            base,
        }
    }
    pub fn with_probe_distance(
        base: Base,
        log2_capacity: u32,
        wait_waker: W,
        probe_distance: usize,
    ) -> Self {
        Self {
            hash_table: ParallelHashTable::with_probe_distance(
                log2_capacity,
                wait_waker,
                probe_distance,
            ),
            hasher: DefaultHashBuilder::new(),
            base,
        }
    }
    pub fn capacity(&self) -> usize {
        self.hash_table.capacity()
    }
    pub fn probe_distance(&self) -> usize {
        self.hash_table.probe_distance()
    }
    pub fn base(&self) -> &Base {
        &self.base
    }
    pub fn base_mut(&mut self) -> &mut Base {
        &mut self.base
    }
}

impl<'a, Base, W, const DIMENSION: usize> HasNodeType<DIMENSION>
    for Parallel<'a, Base, W, DIMENSION>
where
    Base: HasLeafType<DIMENSION>,
    NodeId<'a, Base::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<NodeId<'a, Base::Leaf, DIMENSION>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Base::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    type NodeId = NodeId<'a, Base::Leaf, DIMENSION>;
}

impl<'a, Base, W, const DIMENSION: usize> HasLeafType<DIMENSION>
    for Parallel<'a, Base, W, DIMENSION>
where
    Base: HasLeafType<DIMENSION>,
    NodeId<'a, Base::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Base::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    type Leaf = Base::Leaf;
}

impl<'a, Base, W, const DIMENSION: usize> HasErrorType for Parallel<'a, Base, W, DIMENSION>
where
    Base: HasLeafType<DIMENSION> + HasErrorType,
    NodeId<'a, Base::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Base::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    type Error = Base::Error;
}

impl<'a, Base, W, const DIMENSION: usize> LeafStep<DIMENSION> for Parallel<'a, Base, W, DIMENSION>
where
    Base: LeafStep<DIMENSION>,
    NodeId<'a, Base::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Base::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn leaf_step(
        &self,
        neighborhood: Array<Self::Leaf, 3, DIMENSION>,
    ) -> Result<Self::Leaf, Self::Error> {
        self.base.leaf_step(neighborhood)
    }
}

impl<'a, T, Base, W, const LENGTH: usize, const DIMENSION: usize>
    ParallelBuildArray<T, LENGTH, DIMENSION> for Parallel<'a, Base, W, DIMENSION>
where
    Base: HasLeafType<DIMENSION> + ParallelBuildArray<T, LENGTH, DIMENSION>,
    NodeId<'a, Base::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Base::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    T: ArrayRepr<LENGTH, DIMENSION> + Send,
    IndexVec<DIMENSION>: IndexVecExt,
    Base::Error: Send,
{
    fn parallel_build_array<F: Fn(IndexVec<DIMENSION>) -> Result<T, Self::Error> + Sync>(
        &self,
        f: F,
    ) -> Result<Array<T, LENGTH, DIMENSION>, Self::Error> {
        self.base.parallel_build_array(f)
    }
}

impl<'a, Base, W, const DIMENSION: usize> HashlifeData<DIMENSION>
    for Parallel<'a, Base, W, DIMENSION>
where
    Base: LeafStep<DIMENSION>,
    NodeId<'a, Base::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<NodeId<'a, Base::Leaf, DIMENSION>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Base::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    Base::Error: From<parallel_hash_table::NotEnoughSpace>,
    W: WaitWake,
{
    fn intern_non_leaf_node(
        &self,
        key: NodeAndLevel<Array<Self::NodeId, 2, DIMENSION>>,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        todo!()
    }

    fn intern_leaf_node(
        &self,
        key: Array<Self::Leaf, 2, DIMENSION>,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        todo!()
    }

    fn get_node_key(
        &self,
        node: NodeAndLevel<Self::NodeId>,
    ) -> NodeOrLeaf<NodeAndLevel<Array<Self::NodeId, 2, DIMENSION>>, Array<Self::Leaf, 2, DIMENSION>>
    {
        todo!()
    }

    fn get_non_leaf_node_next(
        &self,
        node: NodeAndLevel<Self::NodeId>,
    ) -> Option<NodeAndLevel<Self::NodeId>> {
        todo!()
    }

    fn fill_non_leaf_node_next(
        &self,
        node: NodeAndLevel<Self::NodeId>,
        new_next: NodeAndLevel<Self::NodeId>,
    ) {
        todo!()
    }

    fn get_empty_node(&self, level: usize) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        todo!()
    }
}
