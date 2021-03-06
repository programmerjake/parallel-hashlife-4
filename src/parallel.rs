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
    debug_assert, fmt,
    hash::{BuildHasher, Hash, Hasher},
    hint::unreachable_unchecked,
    marker::PhantomData,
    num::{NonZeroU32, TryFromIntError},
};
use hashbrown::hash_map::DefaultHashBuilder;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
struct LevelType(u32);

impl LevelType {
    const MAX: Self = LevelType(0xFFF);
    const fn value(self) -> usize {
        self.0 as usize
    }
}

impl TryFrom<usize> for LevelType {
    type Error = TryFromIntError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if value > Self::MAX.value() {
            u32::try_from(!0u128)?;
        }
        Ok(Self(value as u32))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
struct NonZeroLevelType(NonZeroU32);

impl NonZeroLevelType {
    const fn get(self) -> LevelType {
        LevelType(self.0.get())
    }
}

impl TryFrom<LevelType> for NonZeroLevelType {
    type Error = TryFromIntError;

    fn try_from(value: LevelType) -> Result<Self, Self::Error> {
        Ok(NonZeroLevelType(value.0.try_into()?))
    }
}

const LEVEL_COUNT: usize = 0x1000;

type Node<'instance_tag, Leaf, IndexCell, const DIMENSION: usize> =
    NodeOrLeaf<NodeData<'instance_tag, IndexCell, DIMENSION>, Array<Leaf, 2, DIMENSION>>;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(C)]
struct InstanceTag<'instance_tag>(PhantomData<fn(&'instance_tag ()) -> &'instance_tag ()>);

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct NodeId<'instance_tag> {
    index: NodeIdUnderlying,
    instance_tag: InstanceTag<'instance_tag>,
}

pub type NodeIdUnderlying = u32;

const _: () = {
    // assert NodeIdUnderlying is not bigger than usize
    let too_big = NodeIdUnderlying::MAX as usize as NodeIdUnderlying != NodeIdUnderlying::MAX;
    [0][too_big as usize];
};

impl<'instance_tag> NodeId<'instance_tag> {
    const MAX_NODE_COUNT: usize = NodeIdUnderlying::MAX as usize;
    const INVALID_INDEX: usize = Self::MAX_NODE_COUNT;
    const MAX_INDEX: usize = Self::INVALID_INDEX - 1;
    #[inline]
    fn index(self) -> usize {
        self.index as usize
    }
    /// # Safety
    /// `index` must be less than `Parallel::capacity` for the `Parallel` associated with `instance_tag`.
    /// The entry at `index` in the `Parallel` associated with `instance_tag` must be occupied.
    unsafe fn from_index(index: usize, instance_tag: InstanceTag<'instance_tag>) -> Self {
        debug_assert!(index <= Self::MAX_INDEX);
        Self {
            index: index as NodeIdUnderlying,
            instance_tag: instance_tag,
        }
    }
}

impl<'instance_tag> fmt::Debug for NodeId<'instance_tag> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeId")
            .field("index", &self.index())
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct OptionNodeId<'instance_tag> {
    index: u32,
    instance_tag: InstanceTag<'instance_tag>,
}

impl<'instance_tag> OptionNodeId<'instance_tag> {
    pub const NONE: Self = OptionNodeId {
        index: NodeId::INVALID_INDEX as u32,
        instance_tag: InstanceTag(PhantomData),
    };
    pub const fn some(v: NodeId<'instance_tag>) -> Self {
        Self {
            index: v.index,
            instance_tag: v.instance_tag,
        }
    }
    pub const fn is_none(self) -> bool {
        self.index == Self::NONE.index
    }
    pub const fn is_some(self) -> bool {
        self.index != Self::NONE.index
    }
    pub const fn from(v: Option<NodeId<'instance_tag>>) -> Self {
        match v {
            None => Self::NONE,
            Some(v) => Self::some(v),
        }
    }
    pub const fn into(self) -> Option<NodeId<'instance_tag>> {
        if self.is_some() {
            Some(NodeId {
                index: self.index,
                instance_tag: self.instance_tag,
            })
        } else {
            None
        }
    }
}

impl<'instance_tag> From<Option<NodeId<'instance_tag>>> for OptionNodeId<'instance_tag> {
    fn from(v: Option<NodeId<'instance_tag>>) -> Self {
        OptionNodeId::from(v)
    }
}

impl<'instance_tag> From<OptionNodeId<'instance_tag>> for Option<NodeId<'instance_tag>> {
    fn from(v: OptionNodeId<'instance_tag>) -> Self {
        v.into()
    }
}

impl<'instance_tag> fmt::Debug for OptionNodeId<'instance_tag> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&Option::<NodeId<'instance_tag>>::from(*self), f)
    }
}

#[repr(transparent)]
struct AtomicOptionNodeId<'instance_tag, IndexCellT: IndexCell<Underlying = NodeIdUnderlying>> {
    index_cell: IndexCellT,
    instance_tag: InstanceTag<'instance_tag>,
}

impl<'instance_tag, IndexCellT: IndexCell<Underlying = NodeIdUnderlying>>
    AtomicOptionNodeId<'instance_tag, IndexCellT>
{
    #[inline(always)]
    fn none(instance_tag: InstanceTag<'instance_tag>) -> Self {
        Self {
            index_cell: IndexCellT::new(OptionNodeId::NONE.index),
            instance_tag,
        }
    }
    fn get(&self) -> OptionNodeId<'instance_tag> {
        OptionNodeId {
            index: self.index_cell.get(),
            instance_tag: self.instance_tag,
        }
    }
    fn replace(&self, v: OptionNodeId<'instance_tag>) -> OptionNodeId<'instance_tag> {
        let index = self.index_cell.replace(v.index);
        OptionNodeId {
            index,
            instance_tag: self.instance_tag,
        }
    }
}

impl<'instance_tag, IndexCellT: IndexCell<Underlying = NodeIdUnderlying>> fmt::Debug
    for AtomicOptionNodeId<'instance_tag, IndexCellT>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("AtomicOptionNodeId")
            .field(&self.get())
            .finish()
    }
}

struct NodeData<'instance_tag, IndexCellT, const DIMENSION: usize>
where
    NodeId<'instance_tag>: ArrayRepr<2, DIMENSION>,
    IndexCellT: IndexCell<Underlying = NodeIdUnderlying>,
{
    key: Array<NodeId<'instance_tag>, 2, DIMENSION>,
    level: NonZeroLevelType,
    next: AtomicOptionNodeId<'instance_tag, IndexCellT>,
}

impl<'instance_tag, IndexCellT, const DIMENSION: usize> fmt::Debug
    for NodeData<'instance_tag, IndexCellT, DIMENSION>
where
    NodeId<'instance_tag>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    IndexCellT: IndexCell<Underlying = NodeIdUnderlying>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeData")
            .field("key", &self.key)
            .field("next", &self.next)
            .finish()
    }
}

impl<'instance_tag, IndexCellT, const DIMENSION: usize> PartialEq
    for NodeData<'instance_tag, IndexCellT, DIMENSION>
where
    NodeId<'instance_tag>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    IndexCellT: IndexCell<Underlying = NodeIdUnderlying>,
{
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<'instance_tag, IndexCellT, const DIMENSION: usize> Eq
    for NodeData<'instance_tag, IndexCellT, DIMENSION>
where
    NodeId<'instance_tag>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    IndexCellT: IndexCell<Underlying = NodeIdUnderlying>,
{
}

impl<'instance_tag, IndexCellT, const DIMENSION: usize> Hash
    for NodeData<'instance_tag, IndexCellT, DIMENSION>
where
    NodeId<'instance_tag>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    IndexCellT: IndexCell<Underlying = NodeIdUnderlying>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

/// # Safety: relies on only having one instance of Parallel for every `'instance_tag`
pub struct Builder<'instance_tag>(InstanceTag<'instance_tag>);

#[inline]
pub fn make_builder<R, F: for<'instance_tag> FnOnce(Builder<'instance_tag>) -> R>(f: F) -> R {
    let instance_tag = ();
    #[inline]
    fn make_builder<'instance_tag>(_: &'instance_tag ()) -> Builder<'instance_tag> {
        Builder(InstanceTag(PhantomData))
    }
    let builder = make_builder(&instance_tag);
    f(builder)
}

pub struct Parallel<'instance_tag, LeafStepT, ArrayBuilder, HTI, const DIMENSION: usize>
where
    LeafStepT: HasLeafType<DIMENSION>,
    NodeId<'instance_tag>: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    HTI: HashTableImpl,
{
    hash_table: HashTable<HTI, Node<'instance_tag, LeafStepT::Leaf, HTI::IndexCellU32, DIMENSION>>,
    hasher: DefaultHashBuilder,
    empty_nodes: Box<[AtomicOptionNodeId<'instance_tag, HTI::IndexCellU32>; LEVEL_COUNT]>,
    leaf_step: LeafStepT,
    array_builder: ArrayBuilder,
}

impl<'instance_tag, LeafStepT, ArrayBuilder, HTI, const DIMENSION: usize>
    Parallel<'instance_tag, LeafStepT, ArrayBuilder, HTI, DIMENSION>
where
    LeafStepT: HasLeafType<DIMENSION>,
    NodeId<'instance_tag>: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    HTI: HashTableImpl,
{
    fn from_hash_table(
        leaf_step: LeafStepT,
        array_builder: ArrayBuilder,
        hash_table: HashTable<
            HTI,
            Node<'instance_tag, LeafStepT::Leaf, HTI::IndexCellU32, DIMENSION>,
        >,
        instance_tag: InstanceTag<'instance_tag>,
    ) -> Self {
        assert!(hash_table.capacity() <= NodeId::MAX_NODE_COUNT);
        let mut empty_nodes = Vec::with_capacity(LEVEL_COUNT);
        empty_nodes.resize_with(
            LEVEL_COUNT,
            #[inline(always)]
            || AtomicOptionNodeId::none(instance_tag),
        );
        Self {
            hash_table,
            hasher: DefaultHashBuilder::new(),
            empty_nodes: empty_nodes.into_boxed_slice().try_into().ok().unwrap(),
            leaf_step,
            array_builder,
        }
    }
    pub fn with_hash_table_impl(
        builder: Builder<'instance_tag>,
        leaf_step: LeafStepT,
        array_builder: ArrayBuilder,
        log2_capacity: u32,
        hash_table_impl: HTI,
    ) -> Self {
        Self::from_hash_table(
            leaf_step,
            array_builder,
            HashTable::with_impl(log2_capacity, hash_table_impl),
            builder.0,
        )
    }
    pub fn with_hash_table_impl_and_probe_distance(
        builder: Builder<'instance_tag>,
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
            builder.0,
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

impl<'instance_tag, LeafStepT, ArrayBuilder, HTI, const DIMENSION: usize> HasNodeType<DIMENSION>
    for Parallel<'instance_tag, LeafStepT, ArrayBuilder, HTI, DIMENSION>
where
    LeafStepT: HasLeafType<DIMENSION>,
    NodeId<'instance_tag>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<NodeId<'instance_tag>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    HTI: HashTableImpl,
{
    type NodeId = NodeId<'instance_tag>;
}

impl<'instance_tag, LeafStepT, ArrayBuilder, HTI, const DIMENSION: usize> HasLeafType<DIMENSION>
    for Parallel<'instance_tag, LeafStepT, ArrayBuilder, HTI, DIMENSION>
where
    LeafStepT: HasLeafType<DIMENSION>,
    NodeId<'instance_tag>: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    HTI: HashTableImpl,
{
    type Leaf = LeafStepT::Leaf;
}

impl<'instance_tag, LeafStepT, ArrayBuilder, HTI, const DIMENSION: usize> HasErrorType
    for Parallel<'instance_tag, LeafStepT, ArrayBuilder, HTI, DIMENSION>
where
    LeafStepT: HasLeafType<DIMENSION> + HasErrorType,
    NodeId<'instance_tag>: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    HTI: HashTableImpl,
{
    type Error = LeafStepT::Error;
}

impl<'instance_tag, LeafStepT, ArrayBuilder, HTI, const DIMENSION: usize> LeafStep<DIMENSION>
    for Parallel<'instance_tag, LeafStepT, ArrayBuilder, HTI, DIMENSION>
where
    LeafStepT: LeafStep<DIMENSION>,
    NodeId<'instance_tag>: ArrayRepr<2, DIMENSION>,
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

unsafe impl<
        'instance_tag,
        T,
        LeafStepT,
        ArrayBuilder,
        HTI,
        const LENGTH: usize,
        const DIMENSION: usize,
    > ParallelBuildArray<T, LeafStepT::Error, LENGTH, DIMENSION>
    for Parallel<'instance_tag, LeafStepT, ArrayBuilder, HTI, DIMENSION>
where
    LeafStepT: HasLeafType<DIMENSION> + HasErrorType,
    ArrayBuilder: ParallelBuildArray<T, LeafStepT::Error, LENGTH, DIMENSION>,
    NodeId<'instance_tag>: ArrayRepr<2, DIMENSION>,
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

impl<'instance_tag, LeafStepT, ArrayBuilder, HTI, const DIMENSION: usize>
    Parallel<'instance_tag, LeafStepT, ArrayBuilder, HTI, DIMENSION>
where
    LeafStepT: LeafStep<DIMENSION>,
    LeafStepT::Leaf: Hash + Eq,
    NodeId<'instance_tag>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<NodeId<'instance_tag>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafStepT::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    LeafStepT::Error: From<NotEnoughSpace>,
    HTI: HashTableImpl,
{
    #[inline(always)]
    fn get_node(
        &self,
        node_id: NodeId<'instance_tag>,
    ) -> &Node<'instance_tag, LeafStepT::Leaf, HTI::IndexCellU32, DIMENSION> {
        unsafe {
            let retval = self.hash_table.index_unchecked(node_id.index());
            debug_assert!(retval.is_some());
            if let Some(retval) = retval {
                retval
            } else {
                unreachable_unchecked()
            }
        }
    }
    #[inline(always)]
    fn get_node_level(&self, node_id: NodeId<'instance_tag>) -> usize {
        match self.get_node(node_id) {
            NodeOrLeaf::Node(v) => v.level.get().value(),
            NodeOrLeaf::Leaf(_) => 0,
        }
    }
    #[inline(always)]
    fn intern_node(
        &self,
        node: Node<'instance_tag, LeafStepT::Leaf, HTI::IndexCellU32, DIMENSION>,
    ) -> Result<NodeId<'instance_tag>, LeafStepT::Error> {
        let mut hasher = self.hasher.build_hasher();
        node.hash(&mut hasher);
        let get_or_insert_output = self.hash_table.get_or_insert(
            hasher.finish(),
            #[inline(always)]
            |_index, v, node| v == node,
            node,
        )?;
        debug_assert!(get_or_insert_output.index < self.capacity());
        unsafe {
            Ok(NodeId::from_index(
                get_or_insert_output.index,
                InstanceTag::<'instance_tag>(PhantomData),
            ))
        }
    }
    #[cold]
    fn fill_empty_nodes(
        &self,
        target_level: usize,
    ) -> Result<NodeAndLevel<NodeId<'instance_tag>>, LeafStepT::Error> {
        let mut start_node = None;
        for (i, node) in self.empty_nodes[0..target_level].iter().enumerate().rev() {
            if let Some(node) = node.get().into() {
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
            if let Some(old_node) = self.empty_nodes[node.level]
                .replace(OptionNodeId::some(node.node))
                .into()
            {
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

impl<'instance_tag, LeafStepT, ArrayBuilder, HTI, const DIMENSION: usize> HashlifeData<DIMENSION>
    for Parallel<'instance_tag, LeafStepT, ArrayBuilder, HTI, DIMENSION>
where
    LeafStepT: LeafStep<DIMENSION>,
    LeafStepT::Leaf: Hash + Eq,
    NodeId<'instance_tag>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<NodeId<'instance_tag>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
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
        assert!(key.level < LevelType::MAX.value());
        let level: NonZeroLevelType = LevelType::try_from(key.level + 1)
            .unwrap()
            .try_into()
            .unwrap();
        IndexVec::<DIMENSION>::for_each_index(
            #[inline(always)]
            |index| debug_assert_eq!(self.get_node_level(key.node[index]), key.level),
            2,
            ..,
        );
        Ok(NodeAndLevel {
            node: self.intern_node(NodeOrLeaf::Node(NodeData {
                key: key.node,
                level,
                next: AtomicOptionNodeId::none(InstanceTag::<'instance_tag>(PhantomData)),
            }))?,
            level: level.get().value(),
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
        debug_assert_eq!(self.get_node_level(node.node), node.level);
        match self.get_node(node.node) {
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
        debug_assert_eq!(self.get_node_level(node.node), node.level);
        match self.get_node(node.node) {
            NodeOrLeaf::Leaf(_) => panic!("leaf nodes don't have a next field"),
            NodeOrLeaf::Node(node_data) => node_data.next.get().into().map(|next| NodeAndLevel {
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
        debug_assert_eq!(self.get_node_level(node.node), node.level);
        debug_assert_eq!(self.get_node_level(new_next.node), new_next.level);
        match self.get_node(node.node) {
            NodeOrLeaf::Leaf(_) => panic!("leaf nodes don't have a next field"),
            NodeOrLeaf::Node(node_data) => {
                assert_eq!(node.level - 1, new_next.level);
                if let Some(old_next) = node_data
                    .next
                    .replace(OptionNodeId::some(new_next.node))
                    .into()
                {
                    assert_eq!(old_next, new_next.node);
                }
            }
        }
    }
    #[inline(always)]
    fn get_empty_node(&self, level: usize) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        assert!(level < LEVEL_COUNT);
        if let Some(node) = self.empty_nodes[level].get().into() {
            Ok(NodeAndLevel { node, level })
        } else {
            self.fill_empty_nodes(level)
        }
    }
}

#[cfg(test)]
mod test {
    use super::{make_builder, Builder, NodeId, Parallel};
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

    type HL<'instance_tag> =
        Parallel<'instance_tag, LeafData, RayonParallel, SyncHashTableImpl<StdWaitWake>, DIMENSION>;

    fn get_leaf<'instance_tag>(
        hl: &HL<'instance_tag>,
        mut node: NodeAndLevel<NodeId<'instance_tag>>,
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

    fn dump_2d<'instance_tag>(
        hl: &HL<'instance_tag>,
        node: NodeAndLevel<NodeId<'instance_tag>>,
        title: &str,
    ) {
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

    fn build_2d_with_helper<'instance_tag>(
        hl: &HL<'instance_tag>,
        f: &mut impl FnMut(IndexVec<DIMENSION>) -> u8,
        outer_location: IndexVec<DIMENSION>,
        level: usize,
    ) -> NodeAndLevel<NodeId<'instance_tag>> {
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

    fn build_2d_with<'instance_tag>(
        hl: &HL<'instance_tag>,
        mut f: impl FnMut(IndexVec<DIMENSION>) -> u8,
        level: usize,
    ) -> NodeAndLevel<NodeId<'instance_tag>> {
        build_2d_with_helper(hl, &mut f, 0usize.into(), level)
    }

    fn build_2d<'instance_tag, const SIZE: usize>(
        hl: &HL<'instance_tag>,
        array: [[u8; SIZE]; SIZE],
    ) -> NodeAndLevel<NodeId<'instance_tag>> {
        assert!(SIZE.is_power_of_two());
        assert_ne!(SIZE, 1);
        let log2_size = SIZE.trailing_zeros();
        let level = log2_size as usize - 1;
        let array = Array(array);
        build_2d_with(hl, |index| array[index], level)
    }

    fn make_hashlife<'instance_tag>(
        delay: bool,
        builder: Builder<'instance_tag>,
    ) -> HL<'instance_tag> {
        HL::with_hash_table_impl(
            builder,
            LeafData { delay },
            RayonParallel::default(),
            12,
            SyncHashTableImpl::default(),
        )
    }

    #[test]
    fn test0() {
        make_builder(|builder| {
            let hl = make_hashlife(false, builder);
            hl.get_empty_node(0).unwrap();
        })
    }

    #[test]
    fn test1() {
        make_builder(|builder| {
            let hl = make_hashlife(false, builder);
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
        })
    }

    fn make_step0<'instance_tag>(hl: &HL<'instance_tag>) -> NodeAndLevel<NodeId<'instance_tag>> {
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

    fn make_step80<'instance_tag>(hl: &HL<'instance_tag>) -> NodeAndLevel<NodeId<'instance_tag>> {
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
            make_builder(|builder| {
                let hl = make_hashlife(delay, builder);
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
            })
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
