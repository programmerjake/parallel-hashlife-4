use serde::de::DeserializeOwned;

use crate::{
    array::{Array, ArrayRepr},
    index_vec::{IndexVec, IndexVecExt, IndexVecNonzeroDimension},
    io::MacrocellReader,
    parallel, serial,
    serial_hash_table::NotEnoughSpace,
    std_support::{RayonParallel, StdWaitWake},
    traits::{HasErrorType, HasLeafType, HashlifeData, LeafStep},
    NodeAndLevel,
};

pub mod life;

pub fn run_serial<LeafData, MapResult, R, const DIMENSION: usize>(
    macrocell_pattern: &str,
    leaf_data: LeafData,
    log2_capacity: u32,
    log2_step_size: usize,
    map_result: MapResult,
) -> Result<R, LeafData::Error>
where
    for<'a> LeafData: LeafStep<'a, DIMENSION>,
    for<'a> Array<<LeafData as HasLeafType<'a, DIMENSION>>::Leaf, 2, DIMENSION>:
        ArrayRepr<2, DIMENSION>,
    for<'a> serial::NodeId<'a, <LeafData as HasLeafType<'a, DIMENSION>>::Leaf, DIMENSION>:
        ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    for<'a> Array<
        serial::NodeId<'a, <LeafData as HasLeafType<'a, DIMENSION>>::Leaf, DIMENSION>,
        2,
        DIMENSION,
    >: ArrayRepr<2, DIMENSION>,
    LeafData::Error: From<std::io::Error> + From<NotEnoughSpace>,
    for<'a> <LeafData as HasLeafType<'a, DIMENSION>>::Leaf: std::hash::Hash + Eq + DeserializeOwned,
    IndexVec<DIMENSION>: IndexVecExt,
    MapResult: for<'a> FnOnce(
        &'a serial::Serial<'a, LeafData, DIMENSION>,
        NodeAndLevel<serial::NodeId<'a, <LeafData as HasLeafType<'a, DIMENSION>>::Leaf, DIMENSION>>,
    ) -> Result<R, LeafData::Error>,
{
    let hl = serial::Serial::new(leaf_data, log2_capacity);
    let mut root = MacrocellReader::new(macrocell_pattern.as_bytes())?.read_body(&hl)?;
    while root.level <= log2_step_size {
        root = hl.expand_root(root)?;
    }
    root = crate::traits::serial::Hashlife::recursive_hashlife_compute_node_next(
        &hl,
        root,
        log2_step_size,
    )?;
    map_result(&hl, root)
}

pub fn run_parallel<LeafData, MapResult, R, const DIMENSION: usize, const PREV_DIMENSION: usize>(
    macrocell_pattern: &str,
    leaf_data: LeafData,
    log2_capacity: u32,
    log2_step_size: usize,
    map_result: MapResult,
) -> Result<R, LeafData::Error>
where
    for<'a> LeafData: LeafStep<'a, DIMENSION> + Sync + 'static,
    for<'a> Array<<LeafData as HasLeafType<'a, DIMENSION>>::Leaf, 2, DIMENSION>:
        ArrayRepr<2, DIMENSION>,
    for<'a> parallel::NodeId<'a, <LeafData as HasLeafType<'a, DIMENSION>>::Leaf, DIMENSION>:
        ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    for<'a> Option<parallel::NodeId<'a, <LeafData as HasLeafType<'a, DIMENSION>>::Leaf, DIMENSION>>:
        ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    for<'a> Array<
        parallel::NodeId<'a, <LeafData as HasLeafType<'a, DIMENSION>>::Leaf, DIMENSION>,
        2,
        DIMENSION,
    >: ArrayRepr<2, DIMENSION>,
    LeafData::Error: From<std::io::Error> + From<NotEnoughSpace> + Send,
    for<'a> <LeafData as HasLeafType<'a, DIMENSION>>::Leaf:
        std::hash::Hash + Eq + DeserializeOwned + Send + Sync,
    IndexVec<DIMENSION>: IndexVecNonzeroDimension,
    MapResult: for<'a> FnOnce(
        &'a parallel::Parallel<'a, LeafData, RayonParallel, StdWaitWake, DIMENSION>,
        NodeAndLevel<
            parallel::NodeId<'a, <LeafData as HasLeafType<'a, DIMENSION>>::Leaf, DIMENSION>,
        >,
    ) -> Result<R, LeafData::Error>,
{
    let hl = parallel::Parallel::new(leaf_data, RayonParallel, log2_capacity, StdWaitWake);
    hl.get_empty_node(0)?;
    let mut root = MacrocellReader::new(macrocell_pattern.as_bytes())?.read_body(&hl)?;
    while root.level <= log2_step_size {
        root = hl.expand_root(root)?;
    }
    root = crate::traits::parallel::Hashlife::recursive_hashlife_compute_node_next(
        &hl,
        root,
        log2_step_size,
    )?;
    map_result(&hl, root)
}
