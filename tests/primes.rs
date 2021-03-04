use parallel_hashlife::{
    io::{MacrocellHeader, MacrocellReader, MacrocellWriter},
    testing,
    traits::HashlifeData,
};

const PATTERN: &'static str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/patterns/primes.mc"));

const LOG2_STEP_SIZE: usize = 8;

const EXPECTED_PATTERN: &'static str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/patterns/primes_after_256.mc"
));

#[test]
fn step_test() {
    testing::run_serial(
        PATTERN,
        testing::wire_world::LeafData,
        20,
        Some(LOG2_STEP_SIZE),
        |hl, mut node| {
            let mut expected = MacrocellReader::new(EXPECTED_PATTERN.as_bytes())?.read_body(hl)?;
            while expected.level < node.level {
                expected = hl.expand_root(expected)?;
            }
            while node.level < expected.level {
                node = hl.expand_root(node)?;
            }
            if node != expected {
                let mut header = MacrocellHeader::default();
                header.rule = Some("WireWorld".into());
                MacrocellWriter::with_header(std::io::stdout().lock(), header).write(hl, node)?;
            }
            assert_eq!(node, expected);
            Ok(())
        },
    )
    .unwrap();
}
