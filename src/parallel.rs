use crate::{
    array::{Array, ArrayRepr},
    hash_table::{HashTable, HashTableImpl, IndexCell, NotEnoughSpace},
    index_vec::{IndexVec, IndexVecExt, IndexVecForEach},
    traits::{
        parallel::ParallelBuildArray, HasErrorType, HasLeafType, HasNodeType, HashlifeData,
        LeafStep,
    },
    NodeAndLevel, NodeOrLeaf,
};
use alloc::{boxed::Box, vec::Vec};
use core::{
    convert::{TryFrom, TryInto},
    fmt,
    hash::{BuildHasher, Hash, Hasher},
    num::{NonZeroU16, NonZeroUsize},
};
use hashbrown::hash_map::DefaultHashBuilder;

type LevelType = u16;
type NonZeroLevelType = NonZeroU16;
const LEVEL_COUNT: usize = LevelType::MAX as usize + 1;

type Node<Leaf, IndexCell, const DIMENSION: usize> =
    NodeOrLeaf<NodeData<IndexCell, DIMENSION>, Array<Leaf, 2, DIMENSION>>;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct NodeId {
    index_plus_1: NonZeroUsize,
}

impl NodeId {
    #[inline]
    fn index(self) -> usize {
        self.index_plus_1.get() - 1
    }
    fn from_index(index: usize) -> Self {
        Self {
            index_plus_1: NonZeroUsize::new(index.wrapping_add(1)).expect("index too big"),
        }
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeId")
            .field("index", &self.index())
            .finish()
    }
}

#[repr(transparent)]
struct AtomicOptionNodeId<IndexCellT: IndexCell>(IndexCellT);

impl<IndexCellT: IndexCell> AtomicOptionNodeId<IndexCellT> {
    const NONE: Self = Self(IndexCellT::ZERO);
    fn get(&self) -> Option<NodeId> {
        Some(NodeId {
            index_plus_1: NonZeroUsize::new(self.0.get())?,
        })
    }
    fn replace(&self, v: Option<NodeId>) -> Option<NodeId> {
        let retval = self.0.replace(v.map(|v| v.index_plus_1.get()).unwrap_or(0));
        Some(NodeId {
            index_plus_1: NonZeroUsize::new(retval)?,
        })
    }
}

impl<IndexCellT: IndexCell> fmt::Debug for AtomicOptionNodeId<IndexCellT> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("AtomicOptionNodeId")
            .field(&self.get())
            .finish()
    }
}

struct NodeData<IndexCellT, const DIMENSION: usize>
where
    NodeId: ArrayRepr<2, DIMENSION>,
    IndexCellT: IndexCell,
{
    key: Array<NodeId, 2, DIMENSION>,
    level: NonZeroLevelType,
    next: AtomicOptionNodeId<IndexCellT>,
}

impl<IndexCellT, const DIMENSION: usize> fmt::Debug for NodeData<IndexCellT, DIMENSION>
where
    NodeId: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    IndexCellT: IndexCell,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeData")
            .field("key", &self.key)
            .field("next", &self.next)
            .finish()
    }
}

impl<IndexCellT, const DIMENSION: usize> PartialEq for NodeData<IndexCellT, DIMENSION>
where
    NodeId: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    IndexCellT: IndexCell,
{
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<IndexCellT, const DIMENSION: usize> Eq for NodeData<IndexCellT, DIMENSION>
where
    NodeId: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    IndexCellT: IndexCell,
{
}

impl<IndexCellT, const DIMENSION: usize> Hash for NodeData<IndexCellT, DIMENSION>
where
    NodeId: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    IndexCellT: IndexCell,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

pub struct Parallel<LeafStepT, ArrayBuilder, HTI, const DIMENSION: usize>
where
    LeafStepT: HasLeafType<DIMENSION>,
    NodeId: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    HTI: HashTableImpl,
{
    hash_table: HashTable<HTI, Node<LeafStepT::Leaf, HTI::IndexCell, DIMENSION>>,
    hasher: DefaultHashBuilder,
    empty_nodes: Box<[AtomicOptionNodeId<HTI::IndexCell>; LEVEL_COUNT]>,
    leaf_step: LeafStepT,
    array_builder: ArrayBuilder,
}

impl<LeafStepT, ArrayBuilder, HTI, const DIMENSION: usize>
    Parallel<LeafStepT, ArrayBuilder, HTI, DIMENSION>
where
    LeafStepT: HasLeafType<DIMENSION>,
    NodeId: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    HTI: HashTableImpl,
{
    fn from_hash_table(
        leaf_step: LeafStepT,
        array_builder: ArrayBuilder,
        hash_table: HashTable<HTI, Node<LeafStepT::Leaf, HTI::IndexCell, DIMENSION>>,
    ) -> Self {
        let mut empty_nodes = Vec::with_capacity(LEVEL_COUNT);
        empty_nodes.resize_with(LEVEL_COUNT, || AtomicOptionNodeId::NONE);
        Self {
            hash_table,
            hasher: DefaultHashBuilder::new(),
            empty_nodes: empty_nodes.into_boxed_slice().try_into().ok().unwrap(),
            leaf_step,
            array_builder,
        }
    }
    pub fn with_hash_table_impl(
        leaf_step: LeafStepT,
        array_builder: ArrayBuilder,
        log2_capacity: u32,
        hash_table_impl: HTI,
    ) -> Self {
        Self::from_hash_table(
            leaf_step,
            array_builder,
            HashTable::with_impl(log2_capacity, hash_table_impl),
        )
    }
    pub fn with_hash_table_impl_and_probe_distance(
        leaf_step: LeafStepT,
        array_builder: ArrayBuilder,
        log2_capacity: u32,
        hash_table_impl: HTI,
        probe_distance: usize,
    ) -> Self {
        Self::from_hash_table(
            leaf_step,
            array_builder,
            HashTable::with_impl_and_probe_distance(log2_capacity, hash_table_impl, probe_distance),
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

impl<LeafStepT, ArrayBuilder, HTI, const DIMENSION: usize> HasNodeType<DIMENSION>
    for Parallel<LeafStepT, ArrayBuilder, HTI, DIMENSION>
where
    LeafStepT: HasLeafType<DIMENSION>,
    NodeId: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    HTI: HashTableImpl,
{
    type NodeId = NodeId;
}

impl<LeafStepT, ArrayBuilder, HTI, const DIMENSION: usize> HasLeafType<DIMENSION>
    for Parallel<LeafStepT, ArrayBuilder, HTI, DIMENSION>
where
    LeafStepT: HasLeafType<DIMENSION>,
    NodeId: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    HTI: HashTableImpl,
{
    type Leaf = LeafStepT::Leaf;
}

impl<LeafStepT, ArrayBuilder, HTI, const DIMENSION: usize> HasErrorType
    for Parallel<LeafStepT, ArrayBuilder, HTI, DIMENSION>
where
    LeafStepT: HasLeafType<DIMENSION> + HasErrorType,
    NodeId: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    HTI: HashTableImpl,
{
    type Error = LeafStepT::Error;
}

impl<LeafStepT, ArrayBuilder, HTI, const DIMENSION: usize> LeafStep<DIMENSION>
    for Parallel<LeafStepT, ArrayBuilder, HTI, DIMENSION>
where
    LeafStepT: LeafStep<DIMENSION>,
    NodeId: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    HTI: HashTableImpl,
{
    fn leaf_step(
        &self,
        neighborhood: Array<Self::Leaf, 3, DIMENSION>,
    ) -> Result<Self::Leaf, Self::Error> {
        self.leaf_step.leaf_step(neighborhood)
    }
}

impl<T, LeafStepT, ArrayBuilder, HTI, const LENGTH: usize, const DIMENSION: usize>
    ParallelBuildArray<T, LeafStepT::Error, LENGTH, DIMENSION>
    for Parallel<LeafStepT, ArrayBuilder, HTI, DIMENSION>
where
    LeafStepT: HasLeafType<DIMENSION> + HasErrorType,
    ArrayBuilder: ParallelBuildArray<T, LeafStepT::Error, LENGTH, DIMENSION>,
    NodeId: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    T: ArrayRepr<LENGTH, DIMENSION> + Send,
    IndexVec<DIMENSION>: IndexVecExt,
    LeafStepT::Error: Send,
    HTI: HashTableImpl,
{
    fn parallel_build_array<F: Fn(IndexVec<DIMENSION>) -> Result<T, LeafStepT::Error> + Sync>(
        &self,
        f: F,
    ) -> Result<Array<T, LENGTH, DIMENSION>, LeafStepT::Error> {
        self.array_builder.parallel_build_array(f)
    }
}

impl<LeafStepT, ArrayBuilder, HTI, const DIMENSION: usize>
    Parallel<LeafStepT, ArrayBuilder, HTI, DIMENSION>
where
    LeafStepT: LeafStep<DIMENSION>,
    LeafStepT::Leaf: Hash + Eq,
    NodeId: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    LeafStepT::Error: From<NotEnoughSpace>,
    HTI: HashTableImpl,
{
    #[inline(always)]
    fn get_node_level(&self, node_id: NodeId) -> usize {
        match self.hash_table.index(node_id.index()).unwrap() {
            NodeOrLeaf::Node(v) => v.level.get().into(),
            NodeOrLeaf::Leaf(_) => 0,
        }
    }
    #[inline(always)]
    fn intern_node(
        &self,
        node: Node<LeafStepT::Leaf, HTI::IndexCell, DIMENSION>,
    ) -> Result<NodeId, LeafStepT::Error> {
        let mut hasher = self.hasher.build_hasher();
        node.hash(&mut hasher);
        let get_or_insert_output = self.hash_table.get_or_insert(
            hasher.finish(),
            #[inline(always)]
            |_index, v, node| v == node,
            node,
        )?;
        Ok(NodeId::from_index(get_or_insert_output.index))
    }
    #[cold]
    fn fill_empty_nodes(
        &self,
        target_level: usize,
    ) -> Result<NodeAndLevel<NodeId>, LeafStepT::Error> {
        let mut start_node = None;
        for (i, node) in self.empty_nodes[0..target_level].iter().enumerate().rev() {
            if let Some(node) = node.get() {
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
            if let Some(old_node) = self.empty_nodes[node.level].replace(Some(node.node)) {
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

impl<LeafStepT, ArrayBuilder, HTI, const DIMENSION: usize> HashlifeData<DIMENSION>
    for Parallel<LeafStepT, ArrayBuilder, HTI, DIMENSION>
where
    LeafStepT: LeafStep<DIMENSION>,
    LeafStepT::Leaf: Hash + Eq,
    NodeId: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    LeafStepT::Error: From<NotEnoughSpace>,
    HTI: HashTableImpl,
{
    #[inline(always)]
    fn intern_non_leaf_node(
        &self,
        key: NodeAndLevel<Array<Self::NodeId, 2, DIMENSION>>,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        let level: NonZeroLevelType = LevelType::try_from(key.level + 1)
            .unwrap()
            .try_into()
            .unwrap();
        IndexVec::<DIMENSION>::for_each_index(
            #[inline(always)]
            |index| assert_eq!(self.get_node_level(key.node[index]), key.level),
            2,
            ..,
        );
        Ok(NodeAndLevel {
            node: self.intern_node(NodeOrLeaf::Node(NodeData {
                key: key.node,
                level,
                next: AtomicOptionNodeId::NONE,
            }))?,
            level: level.get().into(),
        })
    }
    #[inline(always)]
    fn intern_leaf_node(
        &self,
        key: Array<Self::Leaf, 2, DIMENSION>,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        Ok(NodeAndLevel {
            node: self.intern_node(NodeOrLeaf::Leaf(key))?,
            level: 0,
        })
    }
    #[inline(always)]
    fn get_node_key(
        &self,
        node: NodeAndLevel<Self::NodeId>,
    ) -> NodeOrLeaf<NodeAndLevel<Array<Self::NodeId, 2, DIMENSION>>, Array<Self::Leaf, 2, DIMENSION>>
    {
        assert_eq!(self.get_node_level(node.node), node.level);
        match self.hash_table.index(node.node.index()).unwrap() {
            NodeOrLeaf::Node(node_data) => NodeOrLeaf::Node(NodeAndLevel {
                node: node_data.key.clone(),
                level: node.level - 1,
            }),
            NodeOrLeaf::Leaf(key) => NodeOrLeaf::Leaf(key.clone()),
        }
    }
    #[inline(always)]
    fn get_non_leaf_node_next(
        &self,
        node: NodeAndLevel<Self::NodeId>,
    ) -> Option<NodeAndLevel<Self::NodeId>> {
        assert_eq!(self.get_node_level(node.node), node.level);
        match self.hash_table.index(node.node.index()).unwrap() {
            NodeOrLeaf::Leaf(_) => panic!("leaf nodes don't have a next field"),
            NodeOrLeaf::Node(node_data) => node_data.next.get().map(|next| NodeAndLevel {
                node: next,
                level: node.level - 1,
            }),
        }
    }
    #[inline(always)]
    fn fill_non_leaf_node_next(
        &self,
        node: NodeAndLevel<Self::NodeId>,
        new_next: NodeAndLevel<Self::NodeId>,
    ) {
        assert_eq!(self.get_node_level(node.node), node.level);
        assert_eq!(self.get_node_level(new_next.node), new_next.level);
        match self.hash_table.index(node.node.index()).unwrap() {
            NodeOrLeaf::Leaf(_) => panic!("leaf nodes don't have a next field"),
            NodeOrLeaf::Node(node_data) => {
                assert_eq!(node.level - 1, new_next.level);
                if let Some(old_next) = node_data.next.replace(Some(new_next.node)) {
                    assert_eq!(old_next, new_next.node);
                }
            }
        }
    }
    #[inline(always)]
    fn get_empty_node(&self, level: usize) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        if let Some(node) = self.empty_nodes[level].get() {
            Ok(NodeAndLevel { node, level })
        } else {
            self.fill_empty_nodes(level)
        }
    }
}

#[cfg(test)]
mod test {
    use super::{NodeId, Parallel};
    use crate::{
        array::Array,
        hash_table::{sync::SyncHashTableImpl, NotEnoughSpace},
        index_vec::{IndexVec, IndexVecForEach},
        std_support::{RayonParallel, StdWaitWake},
        traits::{parallel::Hashlife, HasErrorType, HasLeafType, HashlifeData, LeafStep},
        NodeAndLevel, NodeOrLeaf,
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

    type HL = Parallel<LeafData, RayonParallel, SyncHashTableImpl<StdWaitWake>, DIMENSION>;

    fn get_leaf(hl: &HL, mut node: NodeAndLevel<NodeId>, mut location: IndexVec<DIMENSION>) -> u8 {
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

    fn dump_2d(hl: &HL, node: NodeAndLevel<NodeId>, title: &str) {
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

    fn build_2d_with_helper(
        hl: &HL,
        f: &mut impl FnMut(IndexVec<DIMENSION>) -> u8,
        outer_location: IndexVec<DIMENSION>,
        level: usize,
    ) -> NodeAndLevel<NodeId> {
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

    fn build_2d_with(
        hl: &HL,
        mut f: impl FnMut(IndexVec<DIMENSION>) -> u8,
        level: usize,
    ) -> NodeAndLevel<NodeId> {
        build_2d_with_helper(hl, &mut f, 0usize.into(), level)
    }

    fn build_2d<const SIZE: usize>(hl: &HL, array: [[u8; SIZE]; SIZE]) -> NodeAndLevel<NodeId> {
        assert!(SIZE.is_power_of_two());
        assert_ne!(SIZE, 1);
        let log2_size = SIZE.trailing_zeros();
        let level = log2_size as usize - 1;
        let array = Array(array);
        build_2d_with(hl, |index| array[index], level)
    }

    fn make_hashlife(delay: bool) -> HL {
        HL::with_hash_table_impl(
            LeafData { delay },
            RayonParallel::default(),
            12,
            SyncHashTableImpl::default(),
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

    fn make_step0(hl: &HL) -> NodeAndLevel<NodeId> {
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

    fn make_step80(hl: &HL) -> NodeAndLevel<NodeId> {
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
