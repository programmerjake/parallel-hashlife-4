use core::hash::Hash;

use hash_table::NotEnoughSpace;
use serde::de::DeserializeOwned;

use crate::{
    array::{Array, ArrayRepr},
    hash_table,
    index_vec::{IndexVec, IndexVecExt},
    io::MacrocellReader,
    parallel, simple,
    std_support::{RayonParallel, StdWaitWake},
    traits::{HashlifeData, LeafStep},
    NodeAndLevel,
};

pub mod life;
pub mod wire_world;

pub fn run_serial_simple<LeafData, MapResult, R, const DIMENSION: usize>(
    macrocell_pattern: &str,
    leaf_data: LeafData,
    log2_step_size: Option<usize>,
    map_result: MapResult,
) -> Result<R, LeafData::Error>
where
    LeafData: LeafStep<DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    MapResult: FnOnce(
        &simple::Simple<LeafData, DIMENSION>,
        NodeAndLevel<simple::NodeId<LeafData::Leaf, DIMENSION>>,
    ) -> Result<R, LeafData::Error>,
    Array<LeafData::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    simple::NodeId<LeafData::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<simple::NodeId<LeafData::Leaf, DIMENSION>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    LeafData::Error: From<std::io::Error> + From<NotEnoughSpace>,
    LeafData::Leaf: Eq + Hash + DeserializeOwned,
{
    let hl = simple::Simple::new(leaf_data);
    let mut root = MacrocellReader::new(macrocell_pattern.as_bytes())?.read_body(&hl)?;
    if let Some(log2_step_size) = log2_step_size {
        while root.level < log2_step_size {
            root = hl.expand_root(root)?;
        }
        root = hl.expand_root(root)?;
        root = crate::traits::serial::Hashlife::recursive_hashlife_compute_node_next(
            &hl,
            root,
            log2_step_size,
        )?;
    }
    map_result(&hl, root)
}

pub fn run_serial<LeafData, MapResult, R, const DIMENSION: usize>(
    macrocell_pattern: &str,
    leaf_data: LeafData,
    log2_capacity: u32,
    log2_step_size: Option<usize>,
    map_result: MapResult,
) -> Result<R, LeafData::Error>
where
    LeafData: LeafStep<DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    MapResult: FnOnce(
        &parallel::Parallel<LeafData, (), hash_table::unsync::UnsyncHashTableImpl, DIMENSION>,
        NodeAndLevel<parallel::NodeId>,
    ) -> Result<R, LeafData::Error>,
    Array<LeafData::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    parallel::NodeId: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<parallel::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    LeafData::Error: From<std::io::Error> + From<NotEnoughSpace>,
    LeafData::Leaf: Eq + Hash + DeserializeOwned,
{
    let hl =
        parallel::Parallel::with_hash_table_impl(leaf_data, (), log2_capacity, Default::default());
    let mut root = MacrocellReader::new(macrocell_pattern.as_bytes())?.read_body(&hl)?;
    if let Some(log2_step_size) = log2_step_size {
        while root.level < log2_step_size {
            root = hl.expand_root(root)?;
        }
        root = hl.expand_root(root)?;
        root = crate::traits::serial::Hashlife::recursive_hashlife_compute_node_next(
            &hl,
            root,
            log2_step_size,
        )?;
    }
    map_result(&hl, root)
}

pub fn run_parallel<LeafData, MapResult, R, const DIMENSION: usize>(
    macrocell_pattern: &str,
    leaf_data: LeafData,
    log2_capacity: u32,
    log2_step_size: Option<usize>,
    map_result: MapResult,
) -> Result<R, LeafData::Error>
where
    LeafData: LeafStep<DIMENSION> + Sync,
    IndexVec<DIMENSION>: IndexVecExt,
    MapResult: FnOnce(
        &parallel::Parallel<
            LeafData,
            RayonParallel,
            hash_table::sync::SyncHashTableImpl<StdWaitWake>,
            DIMENSION,
        >,
        NodeAndLevel<parallel::NodeId>,
    ) -> Result<R, LeafData::Error>,
    Array<LeafData::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    parallel::NodeId: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Option<parallel::NodeId>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<parallel::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    LeafData::Error: From<std::io::Error> + From<NotEnoughSpace> + Send,
    LeafData::Leaf: Eq + Hash + DeserializeOwned + Sync + Send,
{
    let hl = parallel::Parallel::with_hash_table_impl(
        leaf_data,
        RayonParallel,
        log2_capacity,
        Default::default(),
    );
    let mut root = MacrocellReader::new(macrocell_pattern.as_bytes())?.read_body(&hl)?;
    if let Some(log2_step_size) = log2_step_size {
        while root.level < log2_step_size {
            root = hl.expand_root(root)?;
        }
        root = hl.expand_root(root)?;
        root = crate::traits::parallel::Hashlife::recursive_hashlife_compute_node_next(
            &hl,
            root,
            log2_step_size,
        )?;
    }
    map_result(&hl, root)
}
