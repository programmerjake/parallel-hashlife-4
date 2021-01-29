#![cfg(test)]
#![feature(test)]

extern crate test;

use parallel_hashlife::{
    array::Array,
    index_vec::{IndexVec, IndexVecForEach},
    io::MacrocellReader,
    parallel::Parallel,
    simple::Simple,
    std_support::{RayonParallel, StdWaitWake},
    traits::{parallel, serial, HasErrorType, HasLeafType, HashlifeData, LeafStep},
    NodeAndLevel,
};
use test::Bencher;

const PATTERN: &'static [u8] = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/patterns/spacefiller.mc"
))
.as_bytes();

const DIMENSION: usize = 2;

struct LeafData;

impl HasLeafType<DIMENSION> for LeafData {
    type Leaf = u8;
}

impl HasErrorType for LeafData {
    type Error = std::io::Error;
}

impl LeafStep<DIMENSION> for LeafData {
    fn leaf_step(
        &self,
        neighborhood: Array<Self::Leaf, 3, DIMENSION>,
    ) -> Result<Self::Leaf, Self::Error> {
        let mut sum = 0;
        IndexVec::<DIMENSION>::for_each_index(|index| sum += neighborhood[index], 3, ..);
        Ok(match sum {
            3 => 1,
            4 if neighborhood[IndexVec([1, 1])] != 0 => 1,
            _ => 0,
        })
    }
}

fn load_pattern<'a, HL: HashlifeData<'a, DIMENSION>>(
    hl: &'a HL,
) -> Result<NodeAndLevel<HL::NodeId>, HL::Error>
where
    HL::Error: From<std::io::Error>,
    HL::Leaf: serde::de::DeserializeOwned,
{
    MacrocellReader::new(test::black_box(PATTERN))?.read_body(hl)
}

#[bench]
fn serial_bench(bencher: &mut Bencher) {
    bencher.iter(|| -> Result<(), std::io::Error> {
        let hl = Simple::new(LeafData);
        let mut root = load_pattern(&hl)?;
        while root.level <= 50 {
            root = hl.expand_root(root)?;
        }
        root = serial::Hashlife::recursive_hashlife_compute_node_next(&hl, root, 50)?;
        test::black_box(root);
        Ok(())
    });
}

#[bench]
fn parallel_bench(bencher: &mut Bencher) {
    bencher.iter(|| -> Result<(), std::io::Error> {
        let hl = Parallel::new(
            LeafData,
            RayonParallel::<LeafData>::default(),
            12,
            StdWaitWake,
        );
        let mut root = load_pattern(&hl)?;
        while root.level <= 50 {
            root = hl.expand_root(root)?;
        }
        root = parallel::Hashlife::recursive_hashlife_compute_node_next(&hl, root, 50)?;
        test::black_box(root);
        Ok(())
    });
}
