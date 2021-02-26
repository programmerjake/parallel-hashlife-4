use crate::{
    array::{Array, ArrayRepr},
    index_vec::{IndexVec, IndexVecExt},
    traits::{HashlifeData, LeafStep},
    NodeAndLevel,
};

#[path = "hashlife_impl.rs"]
mod hashlife_impl;

mod send_sync {
    pub trait Everything {}

    impl<T: ?Sized> Everything for T {}
}

use send_sync::{Everything as TheSend, Everything as TheSync};

trait ParallelBuildArray<T, Error, const LENGTH: usize, const DIMENSION: usize>
where
    T: ArrayRepr<LENGTH, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    fn parallel_build_array<F: Fn(IndexVec<DIMENSION>) -> Result<T, Error>>(
        &self,
        f: F,
    ) -> Result<Array<T, LENGTH, DIMENSION>, Error>;
}

impl<This, T, Error, const LENGTH: usize, const DIMENSION: usize>
    ParallelBuildArray<T, Error, LENGTH, DIMENSION> for This
where
    This: ?Sized,
    T: ArrayRepr<LENGTH, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    #[inline(always)]
    fn parallel_build_array<F: Fn(IndexVec<DIMENSION>) -> Result<T, Error>>(
        &self,
        f: F,
    ) -> Result<Array<T, LENGTH, DIMENSION>, Error> {
        Array::try_build_array(f)
    }
}

pub trait Hashlife<const DIMENSION: usize>: HashlifeData<DIMENSION> + LeafStep<DIMENSION>
where
    IndexVec<DIMENSION>: IndexVecExt,
    Array<Self::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Self::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn recursive_hashlife_compute_node_next(
        &self,
        node: NodeAndLevel<Self::NodeId>,
        log2_step_size: usize,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        hashlife_impl::recursive_hashlife_compute_node_next(self, node, log2_step_size)
    }
}

impl<T, const DIMENSION: usize> Hashlife<DIMENSION> for T
where
    T: HashlifeData<DIMENSION> + LeafStep<DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    Array<Self::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Self::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
}
