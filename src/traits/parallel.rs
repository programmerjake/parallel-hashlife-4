use crate::{
    array::{Array, ArrayRepr},
    index_vec::{IndexVec, IndexVecExt},
    traits::{HasErrorType, HasNodeType, HashlifeData},
    NodeAndLevel,
};
use core::marker::{Send as TheSend, Sync as TheSync};

use super::LeafStep;

#[path = "hashlife_impl.rs"]
mod hashlife_impl;

pub unsafe trait ParallelBuildArray<T, Error, const LENGTH: usize, const DIMENSION: usize>
where
    T: ArrayRepr<LENGTH, DIMENSION> + Send,
    IndexVec<DIMENSION>: IndexVecExt,
    Error: Send,
{
    fn parallel_build_array<F: Fn(IndexVec<DIMENSION>) -> Result<T, Error> + Sync>(
        &self,
        f: F,
    ) -> Result<Array<T, LENGTH, DIMENSION>, Error>;
}

unsafe impl<T, Error, This, const LENGTH: usize, const DIMENSION: usize>
    ParallelBuildArray<T, Error, LENGTH, DIMENSION> for &'_ This
where
    This: ?Sized + ParallelBuildArray<T, Error, LENGTH, DIMENSION>,
    T: ArrayRepr<LENGTH, DIMENSION> + Send,
    IndexVec<DIMENSION>: IndexVecExt,
    Error: Send,
{
    fn parallel_build_array<F: Fn(IndexVec<DIMENSION>) -> Result<T, Error> + Sync>(
        &self,
        f: F,
    ) -> Result<Array<T, LENGTH, DIMENSION>, Error> {
        (**self).parallel_build_array(f)
    }
}

pub trait Hashlife<const DIMENSION: usize>:
    HashlifeData<DIMENSION>
    + LeafStep<DIMENSION>
    + Sync
    + ParallelBuildArray<
        <Self as HasNodeType<DIMENSION>>::NodeId,
        <Self as HasErrorType>::Error,
        3,
        DIMENSION,
    > + ParallelBuildArray<
        <Self as HasNodeType<DIMENSION>>::NodeId,
        <Self as HasErrorType>::Error,
        2,
        DIMENSION,
    >
where
    IndexVec<DIMENSION>: IndexVecExt,
    Array<Self::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Self::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Self::NodeId: Send + Sync,
    Self::Leaf: Send,
    <Self as HasErrorType>::Error: Send,
{
    fn recursive_hashlife_compute_node_next(
        &self,
        node: NodeAndLevel<Self::NodeId>,
        log2_step_size: usize,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        hashlife_impl::recursive_hashlife_compute_node_next(self, node, log2_step_size)
    }
}

impl<T: ?Sized, const DIMENSION: usize> Hashlife<DIMENSION> for T
where
    Self: HashlifeData<DIMENSION>
        + LeafStep<DIMENSION>
        + ParallelBuildArray<<T as HasNodeType<DIMENSION>>::NodeId, Self::Error, 3, DIMENSION>
        + ParallelBuildArray<<T as HasNodeType<DIMENSION>>::NodeId, Self::Error, 2, DIMENSION>
        + Sync,
    IndexVec<DIMENSION>: IndexVecExt,
    Array<Self::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Self::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Self::Error: Send,
    Self::NodeId: Send + Sync,
    Self::Leaf: Send,
{
}
