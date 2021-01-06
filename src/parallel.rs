use crate::{
    array::{Array, ArrayRepr},
    index_vec::{IndexVec, IndexVecExt},
    HasErrorType, HasNodeType, HashlifeData,
};
use core::marker::{Send as TheSend, Sync as TheSync};

#[path = "hashlife.rs"]
mod hashlife;

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

/// # Safety
/// Implementation must not run code that would cross threads other than
/// through calling `self.parallel_build_array()`, since this implementation
/// is also used for the single-threaded version.
pub unsafe trait Hashlife<const DIMENSION: usize>: HashlifeData<DIMENSION> + Sync
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
        &self,
        node: Self::NodeId,
        level: usize,
        log2_step_size: usize,
    ) -> Result<Self::NodeId, Self::Error> {
        hashlife::recursive_hashlife_compute_node_next(self, node, level, log2_step_size)
    }
}

unsafe impl<T: ?Sized, const DIMENSION: usize> Hashlife<DIMENSION> for T
where
    Self: HashlifeData<DIMENSION>,
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
