use crate::{
    array::{Array, ArrayRepr},
    index_vec::{IndexVec, IndexVecExt},
    HasErrorType, HasLeafType, HasNodeType, HashlifeData, LeafStep, NodeAndLevel, NodeOrLeaf,
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

impl<LeafData, const DIMENSION: usize> Simple<LeafData, DIMENSION>
where
    LeafData: LeafStep<DIMENSION>,
    NodeId<LeafData::Leaf, DIMENSION>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
    Array<LeafData::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
    Array<NodeId<LeafData::Leaf, DIMENSION>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    LeafData::Leaf: Hash + Eq,
{
    fn intern_node_helper(
        &self,
        node: Node<LeafData::Leaf, DIMENSION>,
    ) -> NodeId<LeafData::Leaf, DIMENSION> {
        let mut nodes = self.nodes.borrow_mut();
        NodeId(match nodes.raw_entry_mut().from_key(&node) {
            RawEntryMut::Occupied(entry) => entry.key().clone(),
            RawEntryMut::Vacant(entry) => {
                let node = Rc::new(node);
                entry.insert(node.clone(), ());
                node
            }
        })
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
    fn intern_nonleaf_node(
        &self,
        key: NodeAndLevel<Array<Self::NodeId, 2, DIMENSION>>,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        let level = NonZeroUsize::new(key.level + 1).unwrap();
        IndexVec::for_each_index(|index| assert_eq!(key.node[index].level(), key.level), 2);
        Ok(NodeAndLevel {
            node: self.intern_node_helper(NodeOrLeaf::Node(NodeData {
                level,
                key: key.node,
                next: RefCell::new(None),
            })),
            level: level.get(),
        })
    }
    fn intern_leaf_node(
        &self,
        key: Array<Self::Leaf, 2, DIMENSION>,
    ) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        Ok(NodeAndLevel {
            node: self.intern_node_helper(NodeOrLeaf::Leaf(key)),
            level: 0,
        })
    }
    fn get_node_key(
        &self,
        node: NodeAndLevel<Self::NodeId>,
    ) -> NodeOrLeaf<NodeAndLevel<Array<Self::NodeId, 2, DIMENSION>>, Array<Self::Leaf, 2, DIMENSION>>
    {
        assert_eq!(node.node.level(), node.level);
        match &*node.node.0 {
            NodeOrLeaf::Node(node_data) => NodeOrLeaf::Node(NodeAndLevel {
                node: node_data.key.clone(),
                level: node.level - 1,
            }),
            NodeOrLeaf::Leaf(key) => NodeOrLeaf::Leaf(key.clone()),
        }
    }
    fn get_nonleaf_node_next(
        &self,
        node: NodeAndLevel<Self::NodeId>,
    ) -> Option<NodeAndLevel<Self::NodeId>> {
        assert_eq!(node.node.level(), node.level);
        match &*node.node.0 {
            NodeOrLeaf::Leaf(_) => panic!("leaf nodes don't have a next field"),
            NodeOrLeaf::Node(node_data) => {
                node_data.next.borrow().clone().map(|next| NodeAndLevel {
                    node: next,
                    level: node.level - 1,
                })
            }
        }
    }
    fn fill_nonleaf_node_next(
        &self,
        node: NodeAndLevel<Self::NodeId>,
        new_next: NodeAndLevel<Self::NodeId>,
    ) {
        assert_eq!(node.node.level(), node.level);
        assert_eq!(new_next.node.level(), new_next.level);
        match &*node.node.0 {
            NodeOrLeaf::Leaf(_) => panic!("leaf nodes don't have a next field"),
            NodeOrLeaf::Node(node_data) => {
                assert_eq!(node.level - 1, new_next.level);
                let mut next = node_data.next.borrow_mut();
                if let Some(next) = &*next {
                    assert_eq!(*next, new_next.node);
                } else {
                    *next = Some(new_next.node);
                }
            }
        }
    }
    fn get_empty_node(&self, level: usize) -> Result<NodeAndLevel<Self::NodeId>, Self::Error> {
        let mut empty_nodes = self.empty_nodes.borrow_mut();
        if let Some(node) = empty_nodes.get(level).cloned() {
            return Ok(NodeAndLevel { node, level });
        }
        let additional = level - empty_nodes.len() + 1;
        empty_nodes.reserve(additional);
        mem::drop(empty_nodes);
        loop {
            let empty_nodes = self.empty_nodes.borrow();
            if let Some(node) = empty_nodes.get(level).cloned() {
                return Ok(NodeAndLevel { node, level });
            }
            let level = empty_nodes.len();
            let node = if let Some(node) = { empty_nodes }.last().cloned() {
                self.intern_nonleaf_node(NodeAndLevel {
                    node: Array::build_array(|_| node.clone()),
                    level: level - 1,
                })?
                .node
            } else {
                self.intern_leaf_node(Array::default())?.node
            };
            self.empty_nodes.borrow_mut().push(node);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        array::Array,
        index_vec::{IndexVec, IndexVecExt},
        serial::Hashlife,
        HasErrorType, HasLeafType, HasNodeType, HashlifeData, LeafStep, NodeAndLevel, NodeOrLeaf,
    };

    use super::Simple;

    const DIMENSION: usize = 2;

    struct LeafData;

    impl HasErrorType for LeafData {
        type Error = ();
    }

    impl HasLeafType<DIMENSION> for LeafData {
        type Leaf = u8;
    }

    impl LeafStep<DIMENSION> for LeafData {
        fn leaf_step(
            &self,
            neighborhood: crate::array::Array<Self::Leaf, 3, DIMENSION>,
        ) -> Result<Self::Leaf, Self::Error> {
            let mut sum = 0;
            IndexVec::<DIMENSION>::for_each_index(|index| sum += neighborhood[index], 3);
            Ok(match sum {
                3 => 1,
                4 if neighborhood[IndexVec([1, 1])] != 0 => 1,
                _ => 0,
            })
        }
    }

    type NodeId = <Simple<LeafData, DIMENSION> as HasNodeType<DIMENSION>>::NodeId;

    fn get_leaf(
        hl: &Simple<LeafData, DIMENSION>,
        mut node: NodeAndLevel<NodeId>,
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

    fn dump_2d(hl: &Simple<LeafData, DIMENSION>, node: NodeAndLevel<NodeId>, title: &str) {
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
        hl: &Simple<LeafData, DIMENSION>,
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
            hl.intern_nonleaf_node(NodeAndLevel {
                node: key,
                level: level - 1,
            })
            .unwrap()
        }
    }

    fn build_2d_with(
        hl: &Simple<LeafData, DIMENSION>,
        mut f: impl FnMut(IndexVec<DIMENSION>) -> u8,
        level: usize,
    ) -> NodeAndLevel<NodeId> {
        build_2d_with_helper(hl, &mut f, 0usize.into(), level)
    }

    fn build_2d<const SIZE: usize>(
        hl: &Simple<LeafData, DIMENSION>,
        array: [[u8; SIZE]; SIZE],
    ) -> NodeAndLevel<NodeId> {
        assert!(SIZE.is_power_of_two());
        assert_ne!(SIZE, 1);
        let log2_size = SIZE.trailing_zeros();
        let level = log2_size as usize - 1;
        let array = Array(array);
        build_2d_with(hl, |index| array[index], level)
    }

    #[test]
    fn test1() {
        let hl = Simple::new(LeafData);
        let empty0 = build_2d(&hl, [[0, 0], [0, 0]]);
        dump_2d(&hl, empty0.clone(), "empty0");
        assert_eq!(
            hl.intern_leaf_node(Array([[0, 0], [0, 0]])).unwrap(),
            empty0
        );
        assert_eq!(hl.get_empty_node(0).unwrap(), empty0);
        let node0 = build_2d(&hl, [[1, 1], [1, 1]]);
        dump_2d(&hl, node0.clone(), "node0");
        let node1 = hl
            .intern_nonleaf_node(NodeAndLevel {
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

    fn make_step0(hl: &Simple<LeafData, DIMENSION>) -> NodeAndLevel<NodeId> {
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

    fn make_step80(hl: &Simple<LeafData, DIMENSION>) -> NodeAndLevel<NodeId> {
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
        let hl = Simple::new(LeafData);
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
