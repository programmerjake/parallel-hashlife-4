use crate::{
    array::{Array, ArrayRepr},
    index_vec::{IndexVec, IndexVecExt, IndexVecForEach},
    parallel_hash_table::{self, ParallelHashTable, WaitWake},
    traits::parallel::ParallelBuildArray,
    HasErrorType, HasLeafType, HasNodeType, HashlifeData, LeafStep, NodeAndLevel, NodeOrLeaf,
};
use alloc::{boxed::Box, vec::Vec};
use core::{
    convert::{TryFrom, TryInto},
    fmt,
    hash::{BuildHasher, Hash, Hasher},
    num::NonZeroU8,
    ptr,
};
use crossbeam_utils::atomic::AtomicCell;
use hashbrown::hash_map::DefaultHashBuilder;
use parallel_hash_table::LockResult;

type LevelType = u8;
type NonZeroLevelType = NonZeroU8;
const LEVEL_COUNT: usize = LevelType::MAX as usize + 1;

type Node<'a, Leaf, const DIMENSION: usize> =
    NodeOrLeaf<NodeData<'a, Leaf, DIMENSION>, Array<Leaf, 2, DIMENSION>>;

#[repr(transparent)]
pub struct NodeId<'a, Leaf, const DIMENSION: usize>(&'a Node<'a, Leaf, DIMENSION>)
where
    Leaf: ArrayRepr<2, DIMENSION>,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>;

impl<'a, Leaf, const DIMENSION: usize> NodeId<'a, Leaf, DIMENSION>
where
    Leaf: ArrayRepr<2, DIMENSION>,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn level(self) -> usize {
        match self.0 {
            NodeOrLeaf::Node(node) => usize::from(node.level.get()),
            NodeOrLeaf::Leaf(_) => 0,
        }
    }
}

impl<'a, Leaf, const DIMENSION: usize> fmt::Debug for NodeId<'a, Leaf, DIMENSION>
where
    Leaf: ArrayRepr<2, DIMENSION> + fmt::Debug,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.0, f)
    }
}

impl<'a, Leaf, const DIMENSION: usize> Clone for NodeId<'a, Leaf, DIMENSION>
where
    Leaf: ArrayRepr<2, DIMENSION> + fmt::Debug,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, Leaf, const DIMENSION: usize> Copy for NodeId<'a, Leaf, DIMENSION>
where
    Leaf: ArrayRepr<2, DIMENSION> + fmt::Debug,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
}

impl<'a, Leaf, const DIMENSION: usize> PartialEq for NodeId<'a, Leaf, DIMENSION>
where
    Leaf: ArrayRepr<2, DIMENSION>,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.0, other.0)
    }
}

impl<'a, Leaf, const DIMENSION: usize> Eq for NodeId<'a, Leaf, DIMENSION>
where
    Leaf: ArrayRepr<2, DIMENSION>,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
}

impl<'a, Leaf, const DIMENSION: usize> Hash for NodeId<'a, Leaf, DIMENSION>
where
    Leaf: ArrayRepr<2, DIMENSION>,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.0 as *const Node<'a, Leaf, DIMENSION>).hash(state);
    }
}

struct NodeData<'a, Leaf, const DIMENSION: usize>
where
    Leaf: ArrayRepr<2, DIMENSION>,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    key: Array<NodeId<'a, Leaf, DIMENSION>, 2, DIMENSION>,
    level: NonZeroLevelType,
    next: AtomicCell<Option<NodeId<'a, Leaf, DIMENSION>>>,
}

impl<'a, Leaf, const DIMENSION: usize> fmt::Debug for NodeData<'a, Leaf, DIMENSION>
where
    Leaf: ArrayRepr<2, DIMENSION> + fmt::Debug,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeData")
            .field("key", &self.key)
            .field("next", &self.next)
            .finish()
    }
}

impl<'a, Leaf, const DIMENSION: usize> PartialEq for NodeData<'a, Leaf, DIMENSION>
where
    Leaf: ArrayRepr<2, DIMENSION>,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<'a, Leaf, const DIMENSION: usize> Eq for NodeData<'a, Leaf, DIMENSION>
where
    Leaf: ArrayRepr<2, DIMENSION>,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
}

impl<'a, Leaf, const DIMENSION: usize> Hash for NodeData<'a, Leaf, DIMENSION>
where
    Leaf: ArrayRepr<2, DIMENSION>,
    NodeId<'a, Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

pub struct Parallel<'a, LeafStepT, ArrayBuilder, WaitWakeT, const DIMENSION: usize>
where
    LeafStepT: HasLeafType<DIMENSION>,
    NodeId<'a, LeafStepT::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    hash_table: ParallelHashTable<Node<'a, LeafStepT::Leaf, DIMENSION>, WaitWakeT>,
    hasher: DefaultHashBuilder,
    empty_nodes: Box<[AtomicCell<Option<NodeId<'a, LeafStepT::Leaf, DIMENSION>>>; LEVEL_COUNT]>,
    leaf_step: LeafStepT,
    array_builder: ArrayBuilder,
}

impl<'a, LeafStepT, ArrayBuilder, WaitWakeT, const DIMENSION: usize>
    Parallel<'a, LeafStepT, ArrayBuilder, WaitWakeT, DIMENSION>
where
    LeafStepT: HasLeafType<DIMENSION>,
    NodeId<'a, LeafStepT::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn from_hash_table(
        leaf_step: LeafStepT,
        array_builder: ArrayBuilder,
        hash_table: ParallelHashTable<Node<'a, LeafStepT::Leaf, DIMENSION>, WaitWakeT>,
    ) -> Self {
        let mut empty_nodes = Vec::with_capacity(LEVEL_COUNT);
        empty_nodes.resize_with(LEVEL_COUNT, || AtomicCell::new(None));
        Self {
            hash_table,
            hasher: DefaultHashBuilder::new(),
            empty_nodes: empty_nodes.into_boxed_slice().try_into().ok().unwrap(),
            leaf_step,
            array_builder,
        }
    }
    pub fn new(
        leaf_step: LeafStepT,
        array_builder: ArrayBuilder,
        log2_capacity: u32,
        wait_waker: WaitWakeT,
    ) -> Self {
        Self::from_hash_table(
            leaf_step,
            array_builder,
            ParallelHashTable::new(log2_capacity, wait_waker),
        )
    }
    pub fn with_probe_distance(
        leaf_step: LeafStepT,
        array_builder: ArrayBuilder,
        log2_capacity: u32,
        wait_waker: WaitWakeT,
        probe_distance: usize,
    ) -> Self {
        Self::from_hash_table(
            leaf_step,
            array_builder,
            ParallelHashTable::with_probe_distance(log2_capacity, wait_waker, probe_distance),
        )
    }
    pub fn capacity(&self) -> usize {
        self.hash_table.capacity()
    }
    pub fn probe_distance(&self) -> usize {
        self.hash_table.probe_distance()
    }
    pub fn leaf_step(&self) -> &LeafStepT {
        &self.leaf_step
    }
    pub fn leaf_step_mut(&mut self) -> &mut LeafStepT {
        &mut self.leaf_step
    }
    pub fn array_builder(&self) -> &ArrayBuilder {
        &self.array_builder
    }
    pub fn array_builder_mut(&mut self) -> &mut ArrayBuilder {
        &mut self.array_builder
    }
}

impl<'a, LeafStepT, ArrayBuilder, WaitWakeT, const DIMENSION: usize> HasNodeType<DIMENSION>
    for Parallel<'a, LeafStepT, ArrayBuilder, WaitWakeT, DIMENSION>
where
    LeafStepT: HasLeafType<DIMENSION>,
    NodeId<'a, LeafStepT::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<NodeId<'a, LeafStepT::Leaf, DIMENSION>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    type NodeId = NodeId<'a, LeafStepT::Leaf, DIMENSION>;
}

impl<'a, LeafStepT, ArrayBuilder, WaitWakeT, const DIMENSION: usize> HasLeafType<DIMENSION>
    for Parallel<'a, LeafStepT, ArrayBuilder, WaitWakeT, DIMENSION>
where
    LeafStepT: HasLeafType<DIMENSION>,
    NodeId<'a, LeafStepT::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    type Leaf = LeafStepT::Leaf;
}

impl<'a, LeafStepT, ArrayBuilder, WaitWakeT, const DIMENSION: usize> HasErrorType
    for Parallel<'a, LeafStepT, ArrayBuilder, WaitWakeT, DIMENSION>
where
    LeafStepT: HasLeafType<DIMENSION> + HasErrorType,
    ArrayBuilder: HasErrorType<Error = LeafStepT::Error>,
    NodeId<'a, LeafStepT::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    type Error = LeafStepT::Error;
}

impl<'a, LeafStepT, ArrayBuilder, WaitWakeT, const DIMENSION: usize> LeafStep<DIMENSION>
    for Parallel<'a, LeafStepT, ArrayBuilder, WaitWakeT, DIMENSION>
where
    LeafStepT: LeafStep<DIMENSION>,
    ArrayBuilder: HasErrorType<Error = LeafStepT::Error>,
    NodeId<'a, LeafStepT::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn leaf_step(
        &self,
        neighborhood: Array<Self::Leaf, 3, DIMENSION>,
    ) -> Result<Self::Leaf, Self::Error> {
        self.leaf_step.leaf_step(neighborhood)
    }
}

impl<'a, T, LeafStepT, ArrayBuilder, WaitWakeT, const LENGTH: usize, const DIMENSION: usize>
    ParallelBuildArray<T, LENGTH, DIMENSION>
    for Parallel<'a, LeafStepT, ArrayBuilder, WaitWakeT, DIMENSION>
where
    LeafStepT: HasLeafType<DIMENSION> + HasErrorType,
    ArrayBuilder: HasErrorType<Error = LeafStepT::Error> + ParallelBuildArray<T, LENGTH, DIMENSION>,
    NodeId<'a, LeafStepT::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    T: ArrayRepr<LENGTH, DIMENSION> + Send,
    IndexVec<DIMENSION>: IndexVecExt,
    ArrayBuilder::Error: Send,
{
    fn parallel_build_array<F: Fn(IndexVec<DIMENSION>) -> Result<T, Self::Error> + Sync>(
        &self,
        f: F,
    ) -> Result<Array<T, LENGTH, DIMENSION>, Self::Error> {
        self.array_builder.parallel_build_array(f)
    }
}

impl<'a, LeafStepT, ArrayBuilder, WaitWakeT, const DIMENSION: usize>
    Parallel<'a, LeafStepT, ArrayBuilder, WaitWakeT, DIMENSION>
where
    LeafStepT: LeafStep<DIMENSION>,
    ArrayBuilder: HasErrorType<Error = LeafStepT::Error>,
    LeafStepT::Leaf: Hash + Eq,
    NodeId<'a, LeafStepT::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<NodeId<'a, LeafStepT::Leaf, DIMENSION>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    LeafStepT::Error: From<parallel_hash_table::NotEnoughSpace>,
    WaitWakeT: WaitWake,
{
    fn intern_node(
        &'a self,
        node: Node<'a, LeafStepT::Leaf, DIMENSION>,
    ) -> Result<NodeId<'a, LeafStepT::Leaf, DIMENSION>, LeafStepT::Error> {
        let mut hasher = self.hasher.build_hasher();
        node.hash(&mut hasher);
        match self
            .hash_table
            .lock_entry(hasher.finish(), |v| *v == node)?
        {
            LockResult::Vacant(entry) => Ok(NodeId(entry.fill(node))),
            LockResult::Full(entry) => Ok(NodeId(entry)),
        }
    }
    #[cold]
    fn fill_empty_nodes(
        &'a self,
        target_level: usize,
    ) -> Result<NodeAndLevel<NodeId<'a, LeafStepT::Leaf, DIMENSION>>, LeafStepT::Error> {
        let mut start_node = None;
        for (i, node) in self.empty_nodes[0..target_level].iter().enumerate().rev() {
            if let Some(node) = node.load() {
                start_node = Some(NodeAndLevel { node, level: i });
                break;
            }
        }
        let mut node = match start_node {
            Some(node) => {
                self.intern_non_leaf_node(node.map_node(|node| Array::build_array(|_| node)))?
            }
            None => self.intern_leaf_node(Array::default())?,
        };
        loop {
            if let Some(old_node) = self.empty_nodes[node.level].swap(Some(node.node)) {
                assert_eq!(old_node, node.node);
            }
            if node.level == target_level {
                break;
            }
            node = self.intern_non_leaf_node(node.map_node(|node| Array::build_array(|_| node)))?;
        }
        Ok(node)
    }
}

impl<'a, LeafStepT, ArrayBuilder, WaitWakeT, const DIMENSION: usize> HashlifeData<'a, DIMENSION>
    for Parallel<'a, LeafStepT, ArrayBuilder, WaitWakeT, DIMENSION>
where
    LeafStepT: LeafStep<DIMENSION>,
    ArrayBuilder: HasErrorType<Error = LeafStepT::Error>,
    LeafStepT::Leaf: Hash + Eq,
    NodeId<'a, LeafStepT::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<NodeId<'a, LeafStepT::Leaf, DIMENSION>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    LeafStepT::Error: From<parallel_hash_table::NotEnoughSpace>,
    WaitWakeT: WaitWake,
{
    fn intern_non_leaf_node(
        &'a self,
        key: NodeAndLevel<Array<Self::NodeId, 2, DIMENSION>>,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        let level: NonZeroLevelType = LevelType::try_from(key.level + 1)
            .unwrap()
            .try_into()
            .unwrap();
        IndexVec::<DIMENSION>::for_each_index(
            |index| assert_eq!(key.node[index].level(), key.level),
            2,
            ..,
        );
        Ok(NodeAndLevel {
            node: self.intern_node(NodeOrLeaf::Node(NodeData {
                key: key.node,
                level,
                next: AtomicCell::new(None),
            }))?,
            level: level.get().into(),
        })
    }
    fn intern_leaf_node(
        &'a self,
        key: Array<Self::Leaf, 2, DIMENSION>,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        Ok(NodeAndLevel {
            node: self.intern_node(NodeOrLeaf::Leaf(key))?,
            level: 0,
        })
    }
    fn get_node_key(
        &'a self,
        node: NodeAndLevel<Self::NodeId>,
    ) -> NodeOrLeaf<NodeAndLevel<Array<Self::NodeId, 2, DIMENSION>>, Array<Self::Leaf, 2, DIMENSION>>
    {
        assert_eq!(node.node.level(), node.level);
        match node.node.0 {
            NodeOrLeaf::Node(node_data) => NodeOrLeaf::Node(NodeAndLevel {
                node: node_data.key.clone(),
                level: node.level - 1,
            }),
            NodeOrLeaf::Leaf(key) => NodeOrLeaf::Leaf(key.clone()),
        }
    }
    fn get_non_leaf_node_next(
        &'a self,
        node: NodeAndLevel<Self::NodeId>,
    ) -> Option<NodeAndLevel<Self::NodeId>> {
        assert_eq!(node.node.level(), node.level);
        match &*node.node.0 {
            NodeOrLeaf::Leaf(_) => panic!("leaf nodes don't have a next field"),
            NodeOrLeaf::Node(node_data) => node_data.next.load().map(|next| NodeAndLevel {
                node: next,
                level: node.level - 1,
            }),
        }
    }
    fn fill_non_leaf_node_next(
        &'a self,
        node: NodeAndLevel<Self::NodeId>,
        new_next: NodeAndLevel<Self::NodeId>,
    ) {
        assert_eq!(node.node.level(), node.level);
        assert_eq!(new_next.node.level(), new_next.level);
        match node.node.0 {
            NodeOrLeaf::Leaf(_) => panic!("leaf nodes don't have a next field"),
            NodeOrLeaf::Node(node_data) => {
                assert_eq!(node.level - 1, new_next.level);
                if let Some(old_next) = node_data.next.swap(Some(new_next.node)) {
                    assert_eq!(old_next, new_next.node);
                }
            }
        }
    }
    fn get_empty_node(&'a self, level: usize) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        if let Some(node) = self.empty_nodes[level].load() {
            Ok(NodeAndLevel { node, level })
        } else {
            self.fill_empty_nodes(level)
        }
    }
}

#[cfg(test)]
mod test {
    use super::Parallel;
    use crate::{
        array::Array,
        index_vec::{IndexVec, IndexVecForEach},
        parallel_hash_table::NotEnoughSpace,
        std_support::{RayonParallel, StdWaitWake},
        traits::parallel::Hashlife,
        HasErrorType, HasLeafType, HasNodeType, HashlifeData, LeafStep, NodeAndLevel, NodeOrLeaf,
    };
    use core::time::Duration;
    use std::{dbg, print, println, thread};

    const DIMENSION: usize = 2;

    struct LeafData {
        delay: bool,
    }

    impl HasErrorType for LeafData {
        type Error = NotEnoughSpace;
    }

    impl HasLeafType<DIMENSION> for LeafData {
        type Leaf = u8;
    }

    impl LeafStep<DIMENSION> for LeafData {
        fn leaf_step(
            &self,
            neighborhood: crate::array::Array<Self::Leaf, 3, DIMENSION>,
        ) -> Result<Self::Leaf, Self::Error> {
            if self.delay {
                thread::sleep(Duration::from_millis(1));
            }
            let mut sum = 0;
            IndexVec::<DIMENSION>::for_each_index(|index| sum += neighborhood[index], 3, ..);
            Ok(match sum {
                3 => 1,
                4 if neighborhood[IndexVec([1, 1])] != 0 => 1,
                _ => 0,
            })
        }
    }

    type HL<'a> = Parallel<'a, LeafData, RayonParallel<LeafData>, StdWaitWake, DIMENSION>;

    type NodeId<'a> = <HL<'a> as HasNodeType<DIMENSION>>::NodeId;

    fn get_leaf<'a>(
        hl: &'a HL<'a>,
        mut node: NodeAndLevel<NodeId<'a>>,
        mut location: IndexVec<DIMENSION>,
    ) -> u8 {
        loop {
            match hl.get_node_key(node) {
                NodeOrLeaf::Node(key) => {
                    let shift = key.level + 1;
                    node = key.map_node(|key| key[location.map(|v| v >> shift)].clone());
                    location = location.map(|v| v & ((1 << shift) - 1));
                }
                NodeOrLeaf::Leaf(key) => break key[location],
            }
        }
    }

    fn dump_2d<'a>(hl: &'a HL<'a>, node: NodeAndLevel<NodeId<'a>>, title: &str) {
        println!("{}:", title);
        let size = 2usize << node.level;
        for y in 0..size {
            for x in 0..size {
                match get_leaf(hl, node.clone(), IndexVec([y, x])) {
                    0 => print!("_ "),
                    leaf => print!("{} ", leaf),
                }
            }
            println!();
        }
    }

    fn build_2d_with_helper<'a>(
        hl: &'a HL<'a>,
        f: &mut impl FnMut(IndexVec<DIMENSION>) -> u8,
        outer_location: IndexVec<DIMENSION>,
        level: usize,
    ) -> NodeAndLevel<NodeId<'a>> {
        if level == 0 {
            hl.intern_leaf_node(Array::build_array(|index| {
                f(index + outer_location.map(|v| v * 2))
            }))
            .unwrap()
        } else {
            let key = Array::build_array(|index| {
                build_2d_with_helper(hl, f, index + outer_location.map(|v| v * 2), level - 1).node
            });
            hl.intern_non_leaf_node(NodeAndLevel {
                node: key,
                level: level - 1,
            })
            .unwrap()
        }
    }

    fn build_2d_with<'a>(
        hl: &'a HL<'a>,
        mut f: impl FnMut(IndexVec<DIMENSION>) -> u8,
        level: usize,
    ) -> NodeAndLevel<NodeId<'a>> {
        build_2d_with_helper(hl, &mut f, 0usize.into(), level)
    }

    fn build_2d<'a, const SIZE: usize>(
        hl: &'a HL<'a>,
        array: [[u8; SIZE]; SIZE],
    ) -> NodeAndLevel<NodeId<'a>> {
        assert!(SIZE.is_power_of_two());
        assert_ne!(SIZE, 1);
        let log2_size = SIZE.trailing_zeros();
        let level = log2_size as usize - 1;
        let array = Array(array);
        build_2d_with(hl, |index| array[index], level)
    }

    fn make_hashlife<'a>(delay: bool) -> HL<'a> {
        HL::new(
            LeafData { delay },
            RayonParallel::default(),
            12,
            StdWaitWake,
        )
    }

    #[test]
    fn test0() {
        let hl = make_hashlife(false);
        hl.get_empty_node(0).unwrap();
    }

    #[test]
    fn test1() {
        let hl = make_hashlife(false);
        let hl = &hl;
        let empty0 = build_2d(hl, [[0, 0], [0, 0]]);
        dump_2d(&hl, empty0.clone(), "empty0");
        assert_eq!(
            hl.intern_leaf_node(Array([[0, 0], [0, 0]])).unwrap(),
            empty0
        );
        assert_eq!(hl.get_empty_node(0).unwrap(), empty0);
        let node0 = build_2d(&hl, [[1, 1], [1, 1]]);
        dump_2d(&hl, node0.clone(), "node0");
        let node1 = hl
            .intern_non_leaf_node(NodeAndLevel {
                node: Array([
                    [empty0.node.clone(), empty0.node.clone()],
                    [empty0.node.clone(), node0.node.clone()],
                ]),
                level: 0,
            })
            .unwrap();
        dump_2d(&hl, node1.clone(), "node1");
        assert_eq!(
            node1,
            build_2d(
                &hl,
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]
            )
        );
        let node1_next = hl.recursive_hashlife_compute_node_next(node1, 0).unwrap();
        dump_2d(&hl, node1_next.clone(), "node1_next");
        assert_eq!(
            hl.get_node_key(node1_next.clone()),
            NodeOrLeaf::Leaf(Array([[0, 0], [0, 1]]))
        );
    }

    fn make_step0<'a>(hl: &'a HL<'a>) -> NodeAndLevel<NodeId<'a>> {
        build_2d(
            hl,
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        )
    }

    fn make_step80<'a>(hl: &'a HL<'a>) -> NodeAndLevel<NodeId<'a>> {
        build_2d(
            hl,
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        )
    }

    fn test_with_step_size(log2_step_size: usize) {
        for &delay in &[false, true] {
            let hl = make_hashlife(delay);
            let mut root = make_step0(&hl);
            dump_2d(&hl, root.clone(), "root");
            let mut step = 0u128;
            while step < 80 {
                root = hl.expand_root(root).unwrap();
                root = hl
                    .recursive_hashlife_compute_node_next(root, log2_step_size)
                    .unwrap();
                step += 1 << log2_step_size;
                dbg!(step);
                dump_2d(&hl, root.clone(), "root");
            }
            let expected = make_step80(&hl);
            dump_2d(&hl, expected.clone(), "expected");
            assert_eq!(root, expected);
        }
    }

    #[test]
    fn test_step_8() {
        test_with_step_size(3);
    }

    #[test]
    fn test_step_4() {
        test_with_step_size(2);
    }

    #[test]
    fn test_step_2() {
        test_with_step_size(1);
    }

    #[test]
    fn test_step_1() {
        test_with_step_size(0);
    }
}
