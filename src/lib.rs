use std::{
    fmt::{self, Debug},
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ops::{Index, IndexMut, RangeTo},
};

mod arrayvec;
use arrayvec::ArrayVec;
pub mod array;
pub mod index_vec;

/*
pub trait NodeId: Copy + Debug + Send + Sync {}

pub trait Array<IV: IndexVec, T: Copy + Debug + Send + Sync + Default>:
    Copy + Debug + Send + Sync + Index<IV, Output = T> + IndexMut<IV> + Default
{
    fn try_build_array<E>(f: impl FnMut(IV) -> Result<T, E>) -> Result<Self, E>;
    fn build_array(mut f: impl FnMut(IV) -> T) -> Self {
        Self::try_build_array(|index_vec| Ok::<T, ()>(f(index_vec))).unwrap()
    }
    fn try_for_each_indexed<E>(self, f: impl FnMut(IV, T) -> Result<(), E>) -> Result<(), E>;
    fn try_for_each<E>(self, mut f: impl FnMut(T) -> Result<(), E>) -> Result<(), E> {
        self.try_for_each_indexed(|_, v| f(v))
    }
    fn for_each_indexed(self, mut f: impl FnMut(IV, T)) {
        self.try_for_each_indexed(|index_vec, v| Ok::<(), ()>(f(index_vec, v)))
            .unwrap()
    }
    fn for_each(self, mut f: impl FnMut(T)) {
        self.try_for_each_indexed(|_, v| Ok::<(), ()>(f(v)))
            .unwrap()
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(transparent)]
pub struct Array0D<T>(T);

impl<T> Index<IndexVec0D> for Array0D<T> {
    type Output = T;
    fn index(&self, _: IndexVec0D) -> &Self::Output {
        &self.0
    }
}

impl<T> IndexMut<IndexVec0D> for Array0D<T> {
    fn index_mut(&mut self, _: IndexVec0D) -> &mut Self::Output {
        &mut self.0
    }
}

impl<T: Copy + Debug + Send + Sync + Default> Array<IndexVec0D, T> for Array0D<T> {
    fn try_build_array<E>(mut f: impl FnMut(IndexVec0D) -> Result<T, E>) -> Result<Self, E> {
        Ok(Array0D(f(0.into())?))
    }
    fn try_for_each_indexed<E>(
        self,
        mut f: impl FnMut(IndexVec0D, T) -> Result<(), E>,
    ) -> Result<(), E> {
        f(0.into(), self.0)
    }
}

macro_rules! impl_array {
    ($array:ident, $n:literal, $index_vec:ident, $prev_array:ident, [$($indexes:literal),+]) => {
        #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Default)]
        #[repr(transparent)]
        pub struct $array<T>([$prev_array<T>; $n]);

        impl<T> Index<$index_vec> for $array<T> {
            type Output = T;
            fn index(&self, index: $index_vec) -> &Self::Output {
                &self.0[index.first()][index.rest()]
            }
        }

        impl<T> IndexMut<$index_vec> for $array<T> {
            fn index_mut(&mut self, index: $index_vec) -> &mut Self::Output {
                &mut self.0[index.first()][index.rest()]
            }
        }

        impl<T: Copy + Debug + Send + Sync + Default> Array<$index_vec, T> for $array<T> {
            fn try_build_array<E>(
                mut f: impl FnMut($index_vec) -> Result<T, E>,
            ) -> Result<Self, E> {
                Ok(Self([
                    $(
                        $prev_array::<T>::try_build_array(|i| f($index_vec::combine($indexes, i)))?,
                    )+
                ]))
            }
            fn try_for_each_indexed<E>(self, mut f: impl FnMut($index_vec, T) -> Result<(), E>) -> Result<(), E> {
                for i in 0..$n {
                    self.0[i].try_for_each_indexed(|index_vec, v| f($index_vec::combine(i, index_vec), v))?;
                }
                Ok(())
            }
        }
    };
}

impl_array!(Array2, 2, IndexVec1D, Array0D, [0, 1]);
impl_array!(Array3, 3, IndexVec1D, Array0D, [0, 1, 2]);
impl_array!(Array2x2, 2, IndexVec2D, Array2, [0, 1]);
impl_array!(Array3x3, 3, IndexVec2D, Array3, [0, 1, 2]);
impl_array!(Array2x2x2, 2, IndexVec3D, Array2x2, [0, 1]);
impl_array!(Array3x3x3, 3, IndexVec3D, Array3x3, [0, 1, 2]);
impl_array!(Array2x2x2x2, 2, IndexVec4D, Array2x2x2, [0, 1]);
impl_array!(Array3x3x3x3, 3, IndexVec4D, Array3x3x3, [0, 1, 2]);

pub trait ParArray<
    IV: IndexVec,
    T: Copy + Debug + Send + Sync + Default,
    ParallelBuilder: Send + Sync + ?Sized,
>: Array<IV, T>
{
    type Error: Debug + Send;
    fn parallel_build_array(
        parallel_builder: &ParallelBuilder,
        f: impl Fn(IV) -> Result<T, Self::Error> + Send,
    ) -> Result<Self, Self::Error>;
}

pub trait HashlifeData: Send + Sync {
    type NodeId: NodeId;
    type LeafData: Copy + Send + Sync + Debug;
    type IndexVec: IndexVec;
    type Error: Debug + Send;
    type Array2: ParArray<Self::IndexVec, Self::NodeId, Self, Error = Self::Error>;
    type Array3: ParArray<Self::IndexVec, Self::NodeId, Self, Error = Self::Error>;
    type Array2OfArray2: ParArray<Self::IndexVec, Self::Array2, Self, Error = Self::Error>;
    fn intern_node(&self, key: Self::Array2, level: usize) -> Result<Self::NodeId, Self::Error>;
    fn get_node_key(&self, node: Self::NodeId, level: usize) -> Self::Array2;
    fn get_or_compute_node_next(
        &self,
        node: Self::NodeId,
        level: usize,
        compute_next_nonleaf: impl FnOnce() -> Result<Self::NodeId, Self::Error>,
    ) -> Result<Self::NodeId, Self::Error>;
}

pub trait HashlifeDataExt: HashlifeData {
    fn recursive_hashlife_compute_node_next(
        &self,
        node: Self::NodeId,
        level: usize,
        double_step_levels: RangeTo<usize>,
    ) -> Result<Self::NodeId, Self::Error> {
        self.get_or_compute_node_next(node, level, || -> Result<Self::NodeId, Self::Error> {
            if double_step_levels.contains(&level) {
                let node_key = self.get_node_key(node, level);
                let node_key_keys = Self::Array2OfArray2::build_array(|index_vec| {
                    self.get_node_key(node_key[index_vec], level)
                });
                let step1 = Self::Array3::parallel_build_array(
                    self,
                    |index_vec3| -> Result<Self::NodeId, Self::Error> {
                        let key = Self::Array2::build_array(|index_vec2| {
                            let sum_vec: Self::IndexVec = index_vec3 + index_vec2;
                            let sum_div2 = sum_vec.map(|v| v / 2);
                            let sum_mod2 = sum_vec.map(|v| v % 2);
                            node_key_keys[sum_div2][sum_mod2]
                        });
                        let temp = self.intern_node(key, level - 1)?;
                        self.recursive_hashlife_compute_node_next(
                            temp,
                            level - 1,
                            double_step_levels,
                        )
                    },
                )?;
                let final_key = Self::Array2::parallel_build_array(
                    self,
                    |outer_index| -> Result<Self::NodeId, Self::Error> {
                        let key = Self::Array2::build_array(|inner_index| {
                            step1[outer_index + inner_index]
                        });
                        let temp = self.intern_node(key, level - 1)?;
                        self.recursive_hashlife_compute_node_next(
                            temp,
                            level - 1,
                            double_step_levels,
                        )
                    },
                )?;
                self.intern_node(final_key, level - 1)
            } else {
                todo!()
            }
        })
    }
}

impl<T: HashlifeData + ?Sized> HashlifeDataExt for T {}

#[cfg(test)]
mod test {
    use super::*;
    use send_wrapper::SendWrapper;
    use std::{
        cell::{Cell, RefCell},
        collections::{hash_map::Entry, HashMap, HashSet},
        hash::{Hash, Hasher},
    };
    use typed_arena::Arena;

    #[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
    enum Leaf {
        Empty,
        Value(u32),
    }

    impl Leaf {
        fn step(neighborhood: Array3x3<Self>) -> Self {
            match neighborhood[IndexVec2D([1, 1])] {
                Leaf::Empty => neighborhood[IndexVec2D([0, 1])],
                Leaf::Value(v) => match neighborhood[IndexVec2D([1, 2])] {
                    Leaf::Empty => Leaf::Value(v + 1),
                    Leaf::Value(v2) => Leaf::Value(v + v2 + 1),
                },
            }
        }
    }

    #[derive(Clone, Debug)]
    struct NodeData<'a> {
        key: Array2x2<NodeRef<'a>>,
        next: SendWrapper<Cell<Option<NodeRef<'a>>>>,
        level: u8,
        id: u32,
    }

    impl PartialEq for NodeData<'_> {
        fn eq(&self, rhs: &Self) -> bool {
            self.key == rhs.key
        }
    }

    impl Eq for NodeData<'_> {}

    impl Hash for NodeData<'_> {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.key.hash(state);
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    enum Node<'a> {
        Leaf(Leaf),
        Node(NodeData<'a>),
    }

    impl Node<'_> {
        fn next_level(&self) -> usize {
            match self {
                Node::Leaf(_) => 0,
                Node::Node(node) => node.level as usize + 1,
            }
        }
    }

    #[derive(Copy, Clone, Debug, Eq)]
    struct NodeRef<'a>(&'a Node<'a>);

    impl Default for NodeRef<'_> {
        fn default() -> Self {
            NodeRef(&Node::Leaf(Leaf::Empty))
        }
    }

    impl PartialEq for NodeRef<'_> {
        fn eq(&self, rhs: &Self) -> bool {
            match (&self.0, &rhs.0) {
                (Node::Leaf(lhs), Node::Leaf(rhs)) => lhs == rhs,
                (Node::Node(lhs), Node::Node(rhs)) => lhs.id == rhs.id,
                _ => false,
            }
        }
    }

    impl Hash for NodeRef<'_> {
        fn hash<H: Hasher>(&self, state: &mut H) {
            match &self.0 {
                Node::Leaf(v) => {
                    0u8.hash(state);
                    v.hash(state);
                }
                Node::Node(v) => {
                    1u8.hash(state);
                    v.id.hash(state);
                }
            }
        }
    }

    impl NodeId for NodeRef<'_> {}

    struct TestDataImpl<'a> {
        node_arena: &'a Arena<Node<'a>>,
        next_node: Option<&'a mut Node<'a>>,
        next_id: u32,
        hash_set: HashSet<&'a Node<'a>>,
    }

    type TestData<'a> = SendWrapper<RefCell<TestDataImpl<'a>>>;

    impl<'a> ParArray<IndexVec2D, NodeRef<'a>, TestData<'a>> for Array2x2<NodeRef<'a>> {
        type Error = ();
        fn parallel_build_array(
            _parallel_builder: &TestData<'a>,
            f: impl Fn(IndexVec2D) -> Result<NodeRef<'a>, Self::Error> + Send,
        ) -> Result<Self, Self::Error> {
            Self::try_build_array(f)
        }
    }

    impl<'a> ParArray<IndexVec2D, NodeRef<'a>, TestData<'a>> for Array3x3<NodeRef<'a>> {
        type Error = ();
        fn parallel_build_array(
            _parallel_builder: &TestData<'a>,
            f: impl Fn(IndexVec2D) -> Result<NodeRef<'a>, Self::Error> + Send,
        ) -> Result<Self, Self::Error> {
            Self::try_build_array(f)
        }
    }

    impl<'a> ParArray<IndexVec2D, Array2x2<NodeRef<'a>>, TestData<'a>>
        for Array2x2<Array2x2<NodeRef<'a>>>
    {
        type Error = ();
        fn parallel_build_array(
            _parallel_builder: &TestData<'a>,
            f: impl Fn(IndexVec2D) -> Result<Array2x2<NodeRef<'a>>, Self::Error> + Send,
        ) -> Result<Self, Self::Error> {
            Self::try_build_array(f)
        }
    }

    impl<'a> HashlifeData for TestData<'a> {
        type NodeId = NodeRef<'a>;
        type IndexVec = IndexVec2D;
        type Error = ();
        type Array2 = Array2x2<NodeRef<'a>>;
        type Array3 = Array3x3<NodeRef<'a>>;
        type Array2OfArray2 = Array2x2<Self::Array2>;
        fn intern_node(
            &self,
            key: Self::Array2,
            level: usize,
        ) -> Result<Self::NodeId, Self::Error> {
            assert!(level <= u8::MAX as usize);
            key.for_each(|v| assert_eq!(v.0.next_level(), level));
            let mut this = self.borrow_mut();
            let next_node = Node::Node(NodeData {
                key,
                level: level as u8,
                next: SendWrapper::new(Cell::new(None)),
                id: this.next_id,
            });
            match this.hash_set.get(&next_node) {
                Some(&node) => Ok(NodeRef(node)),
                None => {
                    this.next_id += 1;
                    let node = this.node_arena.alloc(next_node);
                    assert!(this.hash_set.insert(node));
                    Ok(NodeRef(node))
                }
            }
        }
        fn get_node_key(&self, node: Self::NodeId, level: usize) -> Self::Array2 {
            match node.0 {
                Node::Leaf(_) => panic!("leaf node has no key"),
                Node::Node(node) => {
                    assert_eq!(level, node.level as usize);
                    node.key
                }
            }
        }
        fn get_or_compute_node_next(
            &self,
            node: Self::NodeId,
            level: usize,
            compute_next_nonleaf: impl FnOnce() -> Result<Self::NodeId, Self::Error>,
        ) -> Result<Self::NodeId, Self::Error> {
            if let Node::Node(node) = node.0 {
                assert_eq!(node.level as usize, level);
                if let Some(next) = node.next.get() {
                    return Ok(next);
                }
                let next = match level {
                    0 => unreachable!(),
                    1 => {
                        let key = Self::Array2::try_build_array(|index_vec| todo!())?;
                        todo!()
                    }
                    _ => compute_next_nonleaf()?,
                };
                assert_eq!(next.0.next_level(), level);
                let old = node.next.replace(Some(next));
                if let Some(old) = old {
                    assert_eq!(old, next);
                }
                Ok(next)
            } else {
                panic!()
            }
        }
    }
}
*/
