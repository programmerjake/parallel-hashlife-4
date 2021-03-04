#![cfg(test)]
#![feature(test)]

extern crate test;

use parallel_hashlife::testing::{self, wire_world::LeafData};
use test::Bencher;

const PATTERN: &'static str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/patterns/primes.mc"));
const LOG2_CAPACITY: u32 = 20;
const LOG2_STEP_SIZE: usize = 8;

#[bench]
fn serial_simple_bench(bencher: &mut Bencher) {
    bencher.iter(|| {
        testing::run_serial_simple(
            test::black_box(PATTERN),
            LeafData,
            Some(LOG2_STEP_SIZE),
            |_hl, node| {
                test::black_box(node);
                Ok(())
            },
        )
        .unwrap()
    });
}

#[bench]
fn serial_bench(bencher: &mut Bencher) {
    bencher.iter(|| {
        testing::run_serial(
            test::black_box(PATTERN),
            LeafData,
            LOG2_CAPACITY,
            Some(LOG2_STEP_SIZE),
            |_hl, node| {
                test::black_box(node);
                Ok(())
            },
        )
        .unwrap()
    });
}

#[bench]
fn serial_bench_nostep(bencher: &mut Bencher) {
    bencher.iter(|| {
        testing::run_serial(
            test::black_box(PATTERN),
            LeafData,
            LOG2_CAPACITY,
            None,
            |_hl, node| {
                test::black_box(node);
                Ok(())
            },
        )
        .unwrap()
    });
}

#[bench]
fn parallel_bench(bencher: &mut Bencher) {
    bencher.iter(|| {
        testing::run_parallel(
            test::black_box(PATTERN),
            LeafData,
            LOG2_CAPACITY,
            Some(LOG2_STEP_SIZE),
            |_hl, node| {
                test::black_box(node);
                Ok(())
            },
        )
        .unwrap()
    });
}
