use crate::{
    array::{Array, ArrayRepr},
    index_vec::{IndexVec, IndexVecExt},
    traits::{HasErrorType, HasNodeType, HashlifeData},
    NodeAndLevel,
};
use core::marker::{Send as TheSend, Sync as TheSync};

#[path = "hashlife_impl.rs"]
mod hashlife_impl;

pub trait ParallelBuildArray<T, const LENGTH: usize, const DIMENSION: usize>: HasErrorType
where
    T: ArrayRepr<LENGTH, DIMENSION> + Send,
    IndexVec<DIMENSION>: IndexVecExt,
    Self::Error: Send,
{
    fn parallel_build_array<F: Fn(IndexVec<DIMENSION>) -> Result<T, Self::Error> + Sync>(
        &self,
        f: F,
    ) -> Result<Array<T, LENGTH, DIMENSION>, Self::Error>;
}

impl<T, This, const LENGTH: usize, const DIMENSION: usize> ParallelBuildArray<T, LENGTH, DIMENSION>
    for &'_ This
where
    This: ?Sized + ParallelBuildArray<T, LENGTH, DIMENSION>,
    T: ArrayRepr<LENGTH, DIMENSION> + Send,
    IndexVec<DIMENSION>: IndexVecExt,
    Self::Error: Send,
{
    fn parallel_build_array<F: Fn(IndexVec<DIMENSION>) -> Result<T, Self::Error> + Sync>(
        &self,
        f: F,
    ) -> Result<Array<T, LENGTH, DIMENSION>, Self::Error> {
        (**self).parallel_build_array(f)
    }
}

pub trait Hashlife<'a, const DIMENSION: usize>: HashlifeData<'a, DIMENSION> + Sync
where
    IndexVec<DIMENSION>: IndexVecExt,
    Array<Self::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Self::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Self::Error: Send,
    Self::NodeId: Send + Sync,
    Self::Leaf: Send,
    Self: ParallelBuildArray<<Self as HasNodeType<DIMENSION>>::NodeId, 3, DIMENSION>,
    Self: ParallelBuildArray<<Self as HasNodeType<DIMENSION>>::NodeId, 2, DIMENSION>,
{
    fn recursive_hashlife_compute_node_next(
        &'a self,
        node: NodeAndLevel<Self::NodeId>,
        log2_step_size: usize,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        hashlife_impl::recursive_hashlife_compute_node_next(self, node, log2_step_size)
    }
}

impl<'a, T: ?Sized, const DIMENSION: usize> Hashlife<'a, DIMENSION> for T
where
    Self: HashlifeData<'a, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    Array<Self::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Self::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Self::Error: Send,
    Self::NodeId: Send + Sync,
    Self::Leaf: Send,
    Self: ParallelBuildArray<<T as HasNodeType<DIMENSION>>::NodeId, 3, DIMENSION>,
    Self: ParallelBuildArray<<T as HasNodeType<DIMENSION>>::NodeId, 2, DIMENSION>,
    Self: Sync,
{
}
