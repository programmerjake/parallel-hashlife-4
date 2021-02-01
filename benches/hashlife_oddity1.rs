#![cfg(test)]
#![feature(test)]

extern crate test;

use parallel_hashlife::testing;
use test::Bencher;

const PATTERN: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/patterns/hashlife-oddity1.mc"
));

#[bench]
fn serial_simple_bench(bencher: &mut Bencher) {
    bencher.iter(|| {
        testing::run_serial_simple(
            test::black_box(PATTERN),
            testing::life::LeafData,
            8,
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
            testing::life::LeafData,
            18,
            8,
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
            testing::life::LeafData,
            18,
            8,
            |_hl, node| {
                test::black_box(node);
                Ok(())
            },
        )
        .unwrap()
    });
}
