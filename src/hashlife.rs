use super::{Hashlife, ParallelBuildArray, TheSend, TheSync};
use crate::{
    array::{Array, ArrayRepr},
    index_vec::{IndexVec, IndexVecExt},
    NodeOrLeaf,
};

pub(crate) fn recursive_hashlife_compute_node_next<HL, const DIMENSION: usize>(
    hl: &HL,
    node: HL::NodeId,
    level: usize,
    log2_step_size: usize,
) -> Result<HL::NodeId, HL::Error>
where
    HL: Hashlife<DIMENSION> + ?Sized,
    HL::Error: TheSend,
    HL::NodeId: TheSend + TheSync,
    HL::Leaf: TheSend,
    IndexVec<DIMENSION>: IndexVecExt,
    Array<HL::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<HL::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    assert!(
        level > log2_step_size,
        "level too small to step with requested step size"
    );
    if let Some(next) = hl.get_node_next(node.clone(), level) {
        return Ok(next);
    }
    let next = match level {
        0 => unreachable!(),
        1 => {
            let node_key = hl.get_node_key(node.clone(), level).node().unwrap();
            let node_key_keys =
                Array::<Array<HL::Leaf, 2, DIMENSION>, 2, DIMENSION>::build_array(|index_vec| {
                    hl.get_node_key(node_key[index_vec].clone(), level - 1)
                        .leaf()
                        .unwrap()
                });
            let final_key = Array::try_build_array(|outer_index| -> Result<HL::Leaf, HL::Error> {
                let neighborhood = Array::<HL::Leaf, 3, DIMENSION>::build_array(|inner_index| {
                    let sum_vec = outer_index + inner_index;
                    let sum_div2 = sum_vec.map(|v| v / 2);
                    let sum_mod2 = sum_vec.map(|v| v % 2);
                    node_key_keys[sum_div2][sum_mod2].clone()
                });
                hl.leaf_step(neighborhood)
            })?;
            hl.intern_node(NodeOrLeaf::Leaf(final_key), level - 1)?
        }
        2 if log2_step_size == 0 => {
            todo!();
        }
        _ if level - 1 == log2_step_size => {
            let node_key = hl.get_node_key(node.clone(), level).node().unwrap();
            let node_key_keys =
                Array::<Array<HL::NodeId, 2, DIMENSION>, 2, DIMENSION>::build_array(|index_vec| {
                    hl.get_node_key(node_key[index_vec].clone(), level)
                        .node()
                        .unwrap()
                });
            let step1 = ParallelBuildArray::<HL::NodeId, 3, DIMENSION>::parallel_build_array(
                hl,
                |index_vec3| -> Result<HL::NodeId, HL::Error> {
                    let key = Array::build_array(|index_vec2| {
                        let sum_vec = index_vec3 + index_vec2;
                        let sum_div2 = sum_vec.map(|v| v / 2);
                        let sum_mod2 = sum_vec.map(|v| v % 2);
                        node_key_keys[sum_div2][sum_mod2].clone()
                    });
                    let temp = hl.intern_node(NodeOrLeaf::Node(key), level - 1)?;
                    hl.recursive_hashlife_compute_node_next(temp, level - 1, log2_step_size - 1)
                },
            )?;
            let final_key = ParallelBuildArray::<HL::NodeId, 2, DIMENSION>::parallel_build_array(
                hl,
                |outer_index| -> Result<HL::NodeId, HL::Error> {
                    let key =
                        Array::build_array(|inner_index| step1[outer_index + inner_index].clone());
                    let temp = hl.intern_node(NodeOrLeaf::Node(key), level - 1)?;
                    hl.recursive_hashlife_compute_node_next(temp, level - 1, log2_step_size - 1)
                },
            )?;
            hl.intern_node(NodeOrLeaf::Node(final_key), level - 1)?
        }
        _ => {
            todo!();
        }
    };
    hl.fill_node_next(node, level, next.clone());
    Ok(next)
}
