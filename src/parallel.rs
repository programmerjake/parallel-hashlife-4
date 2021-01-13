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

pub struct Parallel<'a, Base, W, const DIMENSION: usize>
where
    Base: HasLeafType<DIMENSION>,
    NodeId<'a, Base::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Base::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    hash_table: ParallelHashTable<Node<'a, Base::Leaf, DIMENSION>, W>,
    hasher: DefaultHashBuilder,
    empty_nodes: Box<[AtomicCell<Option<NodeId<'a, Base::Leaf, DIMENSION>>>; LEVEL_COUNT]>,
    base: Base,
}

impl<'a, Base, W, const DIMENSION: usize> Parallel<'a, Base, W, DIMENSION>
where
    Base: HasLeafType<DIMENSION>,
    NodeId<'a, Base::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Base::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn from_hash_table(
        base: Base,
        hash_table: ParallelHashTable<Node<'a, Base::Leaf, DIMENSION>, W>,
    ) -> Self {
        let mut empty_nodes = Vec::with_capacity(LEVEL_COUNT);
        empty_nodes.resize_with(LEVEL_COUNT, || AtomicCell::new(None));
        Self {
            hash_table,
            hasher: DefaultHashBuilder::new(),
            empty_nodes: empty_nodes.into_boxed_slice().try_into().ok().unwrap(),
            base,
        }
    }
    pub fn new(base: Base, log2_capacity: u32, wait_waker: W) -> Self {
        Self::from_hash_table(base, ParallelHashTable::new(log2_capacity, wait_waker))
    }
    pub fn with_probe_distance(
        base: Base,
        log2_capacity: u32,
        wait_waker: W,
        probe_distance: usize,
    ) -> Self {
        Self::from_hash_table(
            base,
            ParallelHashTable::with_probe_distance(log2_capacity, wait_waker, probe_distance),
        )
    }
    pub fn capacity(&self) -> usize {
        self.hash_table.capacity()
    }
    pub fn probe_distance(&self) -> usize {
        self.hash_table.probe_distance()
    }
    pub fn base(&self) -> &Base {
        &self.base
    }
    pub fn base_mut(&mut self) -> &mut Base {
        &mut self.base
    }
}

impl<'a, Base, W, const DIMENSION: usize> HasNodeType<DIMENSION>
    for Parallel<'a, Base, W, DIMENSION>
where
    Base: HasLeafType<DIMENSION>,
    NodeId<'a, Base::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<NodeId<'a, Base::Leaf, DIMENSION>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Base::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    type NodeId = NodeId<'a, Base::Leaf, DIMENSION>;
}

impl<'a, Base, W, const DIMENSION: usize> HasLeafType<DIMENSION>
    for Parallel<'a, Base, W, DIMENSION>
where
    Base: HasLeafType<DIMENSION>,
    NodeId<'a, Base::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Base::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    type Leaf = Base::Leaf;
}

impl<'a, Base, W, const DIMENSION: usize> HasErrorType for Parallel<'a, Base, W, DIMENSION>
where
    Base: HasLeafType<DIMENSION> + HasErrorType,
    NodeId<'a, Base::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Base::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    type Error = Base::Error;
}

impl<'a, Base, W, const DIMENSION: usize> LeafStep<DIMENSION> for Parallel<'a, Base, W, DIMENSION>
where
    Base: LeafStep<DIMENSION>,
    NodeId<'a, Base::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Base::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn leaf_step(
        &self,
        neighborhood: Array<Self::Leaf, 3, DIMENSION>,
    ) -> Result<Self::Leaf, Self::Error> {
        self.base.leaf_step(neighborhood)
    }
}

impl<'a, T, Base, W, const LENGTH: usize, const DIMENSION: usize>
    ParallelBuildArray<T, LENGTH, DIMENSION> for Parallel<'a, Base, W, DIMENSION>
where
    Base: HasLeafType<DIMENSION> + ParallelBuildArray<T, LENGTH, DIMENSION>,
    NodeId<'a, Base::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Base::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    T: ArrayRepr<LENGTH, DIMENSION> + Send,
    IndexVec<DIMENSION>: IndexVecExt,
    Base::Error: Send,
{
    fn parallel_build_array<F: Fn(IndexVec<DIMENSION>) -> Result<T, Self::Error> + Sync>(
        &self,
        f: F,
    ) -> Result<Array<T, LENGTH, DIMENSION>, Self::Error> {
        self.base.parallel_build_array(f)
    }
}

impl<'a, Base, W, const DIMENSION: usize> Parallel<'a, Base, W, DIMENSION>
where
    Base: LeafStep<DIMENSION>,
    Base::Leaf: Hash + Eq,
    NodeId<'a, Base::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<NodeId<'a, Base::Leaf, DIMENSION>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Base::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    Base::Error: From<parallel_hash_table::NotEnoughSpace>,
    W: WaitWake,
{
    fn intern_node(
        &'a self,
        node: Node<'a, Base::Leaf, DIMENSION>,
    ) -> Result<NodeId<'a, Base::Leaf, DIMENSION>, Base::Error> {
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
    ) -> Result<NodeAndLevel<NodeId<'a, Base::Leaf, DIMENSION>>, Base::Error> {
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

impl<'a, Base, W, const DIMENSION: usize> HashlifeData<'a, DIMENSION>
    for Parallel<'a, Base, W, DIMENSION>
where
    Base: LeafStep<DIMENSION>,
    Base::Leaf: Hash + Eq,
    NodeId<'a, Base::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<NodeId<'a, Base::Leaf, DIMENSION>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<Base::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    Base::Error: From<parallel_hash_table::NotEnoughSpace>,
    W: WaitWake,
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
