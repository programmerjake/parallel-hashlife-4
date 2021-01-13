use crate::{
    array::{Array, ArrayRepr},
    index_vec::{IndexVec, IndexVecExt},
    HasErrorType, HashlifeData, NodeAndLevel,
};

#[path = "hashlife_impl.rs"]
mod hashlife_impl;

mod send_sync {
    pub trait Everything {}

    impl<T: ?Sized> Everything for T {}
}

use send_sync::{Everything as TheSend, Everything as TheSync};

trait ParallelBuildArray<T, const LENGTH: usize, const DIMENSION: usize>: HasErrorType
where
    T: ArrayRepr<LENGTH, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    fn parallel_build_array<F: Fn(IndexVec<DIMENSION>) -> Result<T, Self::Error>>(
        &self,
        f: F,
    ) -> Result<Array<T, LENGTH, DIMENSION>, Self::Error>;
}

impl<This, T, const LENGTH: usize, const DIMENSION: usize> ParallelBuildArray<T, LENGTH, DIMENSION>
    for This
where
    This: HasErrorType + ?Sized,
    T: ArrayRepr<LENGTH, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    fn parallel_build_array<F: Fn(IndexVec<DIMENSION>) -> Result<T, Self::Error>>(
        &self,
        f: F,
    ) -> Result<Array<T, LENGTH, DIMENSION>, Self::Error> {
        Array::try_build_array(f)
    }
}

pub trait Hashlife<'a, const DIMENSION: usize>: HashlifeData<'a, DIMENSION>
where
    IndexVec<DIMENSION>: IndexVecExt,
    Array<Self::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Self::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn recursive_hashlife_compute_node_next(
        &'a self,
        node: NodeAndLevel<Self::NodeId>,
        log2_step_size: usize,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        hashlife_impl::recursive_hashlife_compute_node_next(self, node, log2_step_size)
    }
}

impl<'a, T, const DIMENSION: usize> Hashlife<'a, DIMENSION> for T
where
    T: HashlifeData<'a, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    Array<Self::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Self::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
}
