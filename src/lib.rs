use std::{
    fmt::Debug,
    ops::{Add, Index, Sub},
};

pub trait NodeId: Copy + Debug + Send + Sync {}

pub trait IndexVec:
    Copy + Debug + Send + Sync + Add<Output = Self> + Sub<Output = Self> + From<usize>
{
}

pub trait NodeIdArray<IV: IndexVec, N: NodeId>:
    Copy + Debug + Send + Sync + Index<IV, Output = N>
{
    type Error: Debug + Send;
    type ParallelBuilder: Send + Sync;
    fn parallel_build_array(
        parallel_builder: &Self::ParallelBuilder,
        f: impl Fn(IV) -> Result<N, Self::Error> + Send,
    ) -> Result<Self, Self::Error>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HashlifeStepSize {
    Single,
    Double,
}

pub trait HashlifeData: Send + Sync {
    type NodeId: NodeId;
    type IndexVec: IndexVec;
    type Error: Debug + Send;
    type Array2: NodeIdArray<
        Self::IndexVec,
        Self::NodeId,
        Error = Self::Error,
        ParallelBuilder = Self,
    >;
    type Array3: NodeIdArray<
        Self::IndexVec,
        Self::NodeId,
        Error = Self::Error,
        ParallelBuilder = Self,
    >;
    type Array4: NodeIdArray<
        Self::IndexVec,
        Self::NodeId,
        Error = Self::Error,
        ParallelBuilder = Self,
    >;
    fn intern_node(&self, key: Self::Array2, level: usize) -> Self::NodeId;
    fn get_node_key(&self, node: Self::NodeId, level: usize) -> Option<Self::Array2>;
    fn get_or_compute_node_next(
        &self,
        node: Self::NodeId,
        level: usize,
        compute_next_nonleaf: impl FnOnce() -> Result<Self::NodeId, Self::Error>,
    ) -> Result<Self::NodeId, Self::Error>;
}

pub fn recursive_hashlife_compute_node_next<
    HD: HashlifeData,
    GetStepSizeForLevel: Send + Sync + Fn(usize) -> HashlifeStepSize,
>(
    hashlife_data: &HD,
    root_node: HD::NodeId,
    root_level: usize,
    get_step_size_for_level: &GetStepSizeForLevel,
) -> Result<HD::NodeId, HD::Error> {
    todo!()
}
