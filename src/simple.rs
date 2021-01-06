use crate::{
    array::{Array, ArrayRepr},
    index_vec::{IndexVec, IndexVecExt},
    HasErrorType, HasLeafType, HasNodeType, HashlifeData, LeafStep, NodeOrLeaf,
};
use alloc::{rc::Rc, vec::Vec};
use core::{
    cell::RefCell,
    fmt,
    hash::{Hash, Hasher},
    mem,
    num::NonZeroUsize,
};
use hashbrown::{hash_map::RawEntryMut, HashMap};

pub struct NodeId<Leaf: ArrayRepr<2, DIMENSION>, const DIMENSION: usize>(Rc<Node<Leaf, DIMENSION>>)
where
    NodeId<Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>;

impl<Leaf: ArrayRepr<2, DIMENSION>, const DIMENSION: usize> Clone for NodeId<Leaf, DIMENSION>
where
    NodeId<Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn clone(&self) -> Self {
        NodeId(self.0.clone())
    }
    fn clone_from(&mut self, source: &Self) {
        self.0.clone_from(&source.0)
    }
}

impl<Leaf: ArrayRepr<2, DIMENSION>, const DIMENSION: usize> PartialEq for NodeId<Leaf, DIMENSION>
where
    NodeId<Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl<Leaf: ArrayRepr<2, DIMENSION>, const DIMENSION: usize> Hash for NodeId<Leaf, DIMENSION>
where
    NodeId<Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

impl<Leaf: ArrayRepr<2, DIMENSION>, const DIMENSION: usize> Eq for NodeId<Leaf, DIMENSION> where
    NodeId<Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>
{
}

impl<Leaf: ArrayRepr<2, DIMENSION>, const DIMENSION: usize> NodeId<Leaf, DIMENSION>
where
    NodeId<Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn level(&self) -> usize {
        match &*self.0 {
            NodeOrLeaf::Node(node) => node.level.get(),
            NodeOrLeaf::Leaf(_) => 0,
        }
    }
}

impl<Leaf: ArrayRepr<2, DIMENSION> + fmt::Debug, const DIMENSION: usize> fmt::Debug
    for NodeId<Leaf, DIMENSION>
where
    NodeId<Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Node<Leaf, DIMENSION> as fmt::Debug>::fmt(&self.0, f)
    }
}

struct NodeData<Leaf: ArrayRepr<2, DIMENSION>, const DIMENSION: usize>
where
    NodeId<Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    level: NonZeroUsize,
    key: Array<NodeId<Leaf, DIMENSION>, 2, DIMENSION>,
    next: RefCell<Option<NodeId<Leaf, DIMENSION>>>,
}

impl<Leaf: ArrayRepr<2, DIMENSION> + fmt::Debug, const DIMENSION: usize> fmt::Debug
    for NodeData<Leaf, DIMENSION>
where
    NodeId<Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("level", &self.level)
            .field("key", &self.key)
            .field("next", &self.next)
            .finish()
    }
}

impl<Leaf: ArrayRepr<2, DIMENSION>, const DIMENSION: usize> PartialEq for NodeData<Leaf, DIMENSION>
where
    NodeId<Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<Leaf: ArrayRepr<2, DIMENSION>, const DIMENSION: usize> Eq for NodeData<Leaf, DIMENSION>
where
    NodeId<Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
}

impl<Leaf: ArrayRepr<2, DIMENSION>, const DIMENSION: usize> Hash for NodeData<Leaf, DIMENSION>
where
    NodeId<Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state)
    }
}

type Node<Leaf, const DIMENSION: usize> =
    NodeOrLeaf<NodeData<Leaf, DIMENSION>, Array<Leaf, 2, DIMENSION>>;

pub struct Simple<LeafData, const DIMENSION: usize>
where
    LeafData: LeafStep<DIMENSION>,
    NodeId<LeafData::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafData::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    leaf_data: LeafData,
    nodes: RefCell<HashMap<Rc<Node<LeafData::Leaf, DIMENSION>>, ()>>,
    empty_nodes: RefCell<Vec<NodeId<LeafData::Leaf, DIMENSION>>>,
}

impl<LeafData, const DIMENSION: usize> Simple<LeafData, DIMENSION>
where
    LeafData: LeafStep<DIMENSION>,
    NodeId<LeafData::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafData::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    pub fn new(leaf_data: LeafData) -> Self {
        Self {
            leaf_data,
            nodes: RefCell::new(HashMap::new()),
            empty_nodes: RefCell::new(Vec::new()),
        }
    }
    pub fn leaf_data(&self) -> &LeafData {
        &self.leaf_data
    }
    pub fn leaf_data_mut(&mut self) -> &mut LeafData {
        &mut self.leaf_data
    }
    pub fn into_leaf_data(self) -> LeafData {
        self.leaf_data
    }
}

impl<LeafData, const DIMENSION: usize> HasErrorType for Simple<LeafData, DIMENSION>
where
    LeafData: LeafStep<DIMENSION>,
    NodeId<LeafData::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafData::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    type Error = LeafData::Error;
}

impl<LeafData, const DIMENSION: usize> HasLeafType<DIMENSION> for Simple<LeafData, DIMENSION>
where
    LeafData: LeafStep<DIMENSION>,
    NodeId<LeafData::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafData::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    type Leaf = LeafData::Leaf;
}

impl<LeafData, const DIMENSION: usize> HasNodeType<DIMENSION> for Simple<LeafData, DIMENSION>
where
    LeafData: LeafStep<DIMENSION>,
    NodeId<LeafData::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<LeafData::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<NodeId<LeafData::Leaf, DIMENSION>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    type NodeId = NodeId<LeafData::Leaf, DIMENSION>;
}

impl<LeafData, const DIMENSION: usize> LeafStep<DIMENSION> for Simple<LeafData, DIMENSION>
where
    LeafData: LeafStep<DIMENSION>,
    NodeId<LeafData::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<LeafData::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn leaf_step(
        &self,
        neighborhood: Array<Self::Leaf, 3, DIMENSION>,
    ) -> Result<Self::Leaf, Self::Error> {
        self.leaf_data.leaf_step(neighborhood)
    }
}

impl<LeafData, const DIMENSION: usize> HashlifeData<DIMENSION> for Simple<LeafData, DIMENSION>
where
    LeafData: LeafStep<DIMENSION>,
    NodeId<LeafData::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<LeafData::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    Array<NodeId<LeafData::Leaf, DIMENSION>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    LeafData::Leaf: Hash + Eq,
{
    fn intern_node(
        &self,
        key: NodeOrLeaf<Array<Self::NodeId, 2, DIMENSION>, Array<Self::Leaf, 2, DIMENSION>>,
        level: usize,
    ) -> Result<Self::NodeId, Self::Error> {
        let node = match key {
            NodeOrLeaf::Node(key) => {
                let level =
                    NonZeroUsize::new(level).expect("non-leaf node must have non-zero level");
                IndexVec::for_each_index(
                    |index| assert_eq!(key[index].level(), level.get() - 1),
                    2,
                );
                NodeOrLeaf::Node(NodeData {
                    level,
                    key,
                    next: RefCell::new(None),
                })
            }
            NodeOrLeaf::Leaf(key) => {
                assert_eq!(level, 0, "leaf node must have level of 0");
                NodeOrLeaf::Leaf(key)
            }
        };
        let mut nodes = self.nodes.borrow_mut();
        Ok(NodeId(match nodes.raw_entry_mut().from_key(&node) {
            RawEntryMut::Occupied(entry) => entry.key().clone(),
            RawEntryMut::Vacant(entry) => {
                let node = Rc::new(node);
                entry.insert(node.clone(), ());
                node
            }
        }))
    }
    fn get_node_key(
        &self,
        node: Self::NodeId,
        level: usize,
    ) -> NodeOrLeaf<Array<Self::NodeId, 2, DIMENSION>, Array<Self::Leaf, 2, DIMENSION>> {
        assert_eq!(node.level(), level);
        match &*node.0 {
            NodeOrLeaf::Node(node) => NodeOrLeaf::Node(node.key.clone()),
            NodeOrLeaf::Leaf(key) => NodeOrLeaf::Leaf(key.clone()),
        }
    }
    fn get_node_next(&self, node: Self::NodeId, level: usize) -> Option<Self::NodeId> {
        assert_eq!(node.level(), level);
        match &*node.0 {
            NodeOrLeaf::Leaf(_) => panic!("leaf nodes don't have a next field"),
            NodeOrLeaf::Node(node) => node.next.borrow().clone(),
        }
    }
    fn fill_node_next(&self, node: Self::NodeId, level: usize, new_next: Self::NodeId) {
        assert_eq!(node.level(), level);
        match &*node.0 {
            NodeOrLeaf::Leaf(_) => panic!("leaf nodes don't have a next field"),
            NodeOrLeaf::Node(node) => {
                assert_eq!(node.level.get() - 1, new_next.level());
                let mut next = node.next.borrow_mut();
                if let Some(next) = &*next {
                    assert_eq!(*next, new_next);
                } else {
                    *next = Some(new_next);
                }
            }
        }
    }
    fn get_empty_node(&self, level: usize) -> Result<Self::NodeId, Self::Error> {
        let mut empty_nodes = self.empty_nodes.borrow_mut();
        if let Some(node) = empty_nodes.get(level) {
            return Ok(node.clone());
        }
        let additional = level - empty_nodes.len() + 1;
        empty_nodes.reserve(additional);
        mem::drop(empty_nodes);
        loop {
            let empty_nodes = self.empty_nodes.borrow();
            if let Some(node) = empty_nodes.get(level) {
                return Ok(node.clone());
            }
            let key = if let Some(node) = empty_nodes.last() {
                NodeOrLeaf::Node(Array::build_array(|_| node.clone()))
            } else {
                NodeOrLeaf::Leaf(Array::default())
            };
            let level = empty_nodes.len();
            let node = self.intern_node(key, level)?;
            mem::drop(empty_nodes);
            self.empty_nodes.borrow_mut().push(node);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        array::Array, index_vec::IndexVec, serial::Hashlife, HasErrorType, HasLeafType,
        HashlifeData, LeafStep, NodeOrLeaf,
    };

    use super::Simple;

    const DIMENSION: usize = 2;

    struct LeafData;

    impl HasErrorType for LeafData {
        type Error = ();
    }

    impl HasLeafType<DIMENSION> for LeafData {
        type Leaf = usize;
    }

    impl LeafStep<DIMENSION> for LeafData {
        fn leaf_step(
            &self,
            neighborhood: crate::array::Array<Self::Leaf, 3, DIMENSION>,
        ) -> Result<Self::Leaf, Self::Error> {
            Ok(match neighborhood[IndexVec([1, 1])] {
                0 => neighborhood[IndexVec([0, 1])],
                v => match neighborhood[IndexVec([1, 2])] {
                    0 => v + 1,
                    v2 => v + v2 + 2,
                },
            })
        }
    }

    #[test]
    fn test1() {
        let simple = Simple::new(LeafData);
        let empty0 = simple
            .intern_node(NodeOrLeaf::Leaf(Array([[0, 0], [0, 0]])), 0)
            .unwrap();
        assert_eq!(
            simple
                .intern_node(NodeOrLeaf::Leaf(Array([[0, 0], [0, 0]])), 0)
                .unwrap(),
            empty0
        );
        assert_eq!(simple.get_empty_node(0).unwrap(), empty0);
        let node0 = simple
            .intern_node(NodeOrLeaf::Leaf(Array([[1, 0], [0, 0]])), 0)
            .unwrap();
        let node1 = simple
            .intern_node(
                NodeOrLeaf::Node(Array([
                    [empty0.clone(), empty0.clone()],
                    [empty0.clone(), node0.clone()],
                ])),
                1,
            )
            .unwrap();
        let node2 = simple
            .intern_node(NodeOrLeaf::Leaf(Array([[0, 0], [0, 1]])), 0)
            .unwrap();
        assert_eq!(simple.expand_root(node2, 0).unwrap(), node1);
        let node1_next = simple
            .recursive_hashlife_compute_node_next(node1, 1, 0)
            .unwrap();
        assert_eq!(
            simple.get_node_key(node1_next.clone(), 0),
            NodeOrLeaf::Leaf(Array([[0, 0], [0, 2]]))
        );
    }
}
