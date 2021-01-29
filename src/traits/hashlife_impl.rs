use super::{Hashlife, ParallelBuildArray, TheSend, TheSync};
use crate::{
    array::{Array, ArrayRepr},
    index_vec::{IndexVec, IndexVecExt},
    NodeAndLevel,
};

pub(crate) fn recursive_hashlife_compute_node_next<'a, HL, const DIMENSION: usize>(
    hl: &'a HL,
    node: NodeAndLevel<HL::NodeId>,
    log2_step_size: usize,
) -> Result<NodeAndLevel<HL::NodeId>, HL::Error>
where
    HL: Hashlife<'a, DIMENSION> + ?Sized,
    HL::Error: TheSend,
    HL::NodeId: TheSend + TheSync,
    HL::Leaf: TheSend,
    IndexVec<DIMENSION>: IndexVecExt,
    Array<HL::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<HL::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    assert!(
        node.level > log2_step_size,
        "level too small to step with requested step size"
    );
    if let Some(next) = hl.get_non_leaf_node_next(node.clone()) {
        return Ok(next);
    }
    let next = match node.level {
        0 => unreachable!(),
        1 => {
            let node_key = hl.get_node_key(node.clone()).node().unwrap();
            let node_key_keys =
                Array::<Array<HL::Leaf, 2, DIMENSION>, 2, DIMENSION>::build_array(|index_vec| {
                    hl.get_node_key(node_key.as_ref().map_node(|node| node[index_vec].clone()))
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
            hl.intern_leaf_node(final_key)?
        }
        _ if node.level - 1 == log2_step_size => {
            let node_key = hl.get_node_key(node.clone()).node().unwrap();
            let node_key_keys =
                Array::<Array<HL::NodeId, 2, DIMENSION>, 2, DIMENSION>::build_array(|index_vec| {
                    hl.get_node_key(node_key.as_ref().map_node(|node| node[index_vec].clone()))
                        .node()
                        .unwrap()
                        .node
                });
            let step1 =
                ParallelBuildArray::<HL::NodeId, HL::Error, 3, DIMENSION>::parallel_build_array(
                    hl,
                    |index_vec3| -> Result<HL::NodeId, HL::Error> {
                        let key = Array::build_array(|index_vec2| {
                            let sum_vec = index_vec3 + index_vec2;
                            let sum_div2 = sum_vec.map(|v| v / 2);
                            let sum_mod2 = sum_vec.map(|v| v % 2);
                            node_key_keys[sum_div2][sum_mod2].clone()
                        });
                        let temp = hl.intern_non_leaf_node(NodeAndLevel {
                            node: key,
                            level: node.level - 2,
                        })?;
                        Ok(hl
                            .recursive_hashlife_compute_node_next(temp, log2_step_size - 1)?
                            .node)
                    },
                )?;
            let final_key =
                ParallelBuildArray::<HL::NodeId, HL::Error, 2, DIMENSION>::parallel_build_array(
                    hl,
                    |outer_index| -> Result<HL::NodeId, HL::Error> {
                        let key = Array::build_array(|inner_index| {
                            step1[outer_index + inner_index].clone()
                        });
                        let temp = hl.intern_non_leaf_node(NodeAndLevel {
                            node: key,
                            level: node.level - 2,
                        })?;
                        Ok(hl
                            .recursive_hashlife_compute_node_next(temp, log2_step_size - 1)?
                            .node)
                    },
                )?;
            hl.intern_non_leaf_node(NodeAndLevel {
                node: final_key,
                level: node.level - 2,
            })?
        }
        _ => {
            let node_key = hl.get_node_key(node.clone()).node().unwrap();
            let node_key_keys =
                Array::<Array<HL::NodeId, 2, DIMENSION>, 2, DIMENSION>::build_array(|index_vec| {
                    hl.get_node_key(node_key.as_ref().map_node(|node| node[index_vec].clone()))
                        .node()
                        .unwrap()
                        .node
                });
            let step1 =
                ParallelBuildArray::<HL::NodeId, HL::Error, 3, DIMENSION>::parallel_build_array(
                    hl,
                    |index_vec3| -> Result<HL::NodeId, HL::Error> {
                        let key = Array::build_array(|index_vec2| {
                            let sum_vec = index_vec3 + index_vec2;
                            let sum_div2 = sum_vec.map(|v| v / 2);
                            let sum_mod2 = sum_vec.map(|v| v % 2);
                            node_key_keys[sum_div2][sum_mod2].clone()
                        });
                        let temp = hl.intern_non_leaf_node(NodeAndLevel {
                            node: key,
                            level: node.level - 2,
                        })?;
                        Ok(hl
                            .recursive_hashlife_compute_node_next(temp, log2_step_size)?
                            .node)
                    },
                )?;
            let final_key =
                Array::try_build_array(|outer_index| -> Result<HL::NodeId, HL::Error> {
                    let key =
                        Array::build_array(|inner_index| step1[outer_index + inner_index].clone());
                    let temp = hl.intern_non_leaf_node(NodeAndLevel {
                        node: key,
                        level: node.level - 2,
                    })?;
                    Ok(hl.get_center(temp)?.node)
                })?;
            hl.intern_non_leaf_node(NodeAndLevel {
                node: final_key,
                level: node.level - 2,
            })?
        }
    };
    hl.fill_non_leaf_node_next(node, next.clone());
    Ok(next)
}
