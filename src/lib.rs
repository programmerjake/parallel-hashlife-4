#![no_std]
#![deny(elided_lifetimes_in_paths)]

extern crate alloc;
#[cfg(any(test, feature = "std"))]
extern crate std;

pub mod array;
mod array_vec;
pub mod index_vec;
#[cfg(any(test, feature = "io"))]
pub mod io;
pub mod parallel;
pub mod parallel_hash_table;
pub mod serial;
pub mod serial_hash_table;
pub mod simple;
#[cfg(any(test, feature = "std"))]
pub mod std_support;
#[cfg(test)]
pub mod testing;
pub mod traits;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct NodeAndLevel<Node> {
    pub node: Node,
    pub level: usize,
}

impl<T> NodeAndLevel<T> {
    pub fn map_node<R, F: FnOnce(T) -> R>(self, f: F) -> NodeAndLevel<R> {
        NodeAndLevel {
            node: f(self.node),
            level: self.level,
        }
    }
    pub fn try_map_node<R, E, F: FnOnce(T) -> Result<R, E>>(
        self,
        f: F,
    ) -> Result<NodeAndLevel<R>, E> {
        Ok(NodeAndLevel {
            node: f(self.node)?,
            level: self.level,
        })
    }
    pub const fn as_ref(&self) -> NodeAndLevel<&T> {
        let NodeAndLevel { ref node, level } = *self;
        NodeAndLevel { node, level }
    }
    pub fn as_mut(&mut self) -> NodeAndLevel<&mut T> {
        let NodeAndLevel {
            ref mut node,
            level,
        } = *self;
        NodeAndLevel { node, level }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum NodeOrLeaf<Node, Leaf> {
    Node(Node),
    Leaf(Leaf),
}

impl<Node, Leaf> NodeOrLeaf<Node, Leaf> {
    pub const fn as_ref(&self) -> NodeOrLeaf<&Node, &Leaf> {
        match self {
            NodeOrLeaf::Node(v) => NodeOrLeaf::Node(v),
            NodeOrLeaf::Leaf(v) => NodeOrLeaf::Leaf(v),
        }
    }
    pub fn as_mut(&mut self) -> NodeOrLeaf<&mut Node, &mut Leaf> {
        match self {
            NodeOrLeaf::Node(v) => NodeOrLeaf::Node(v),
            NodeOrLeaf::Leaf(v) => NodeOrLeaf::Leaf(v),
        }
    }
    pub const fn is_node(&self) -> bool {
        matches!(self, NodeOrLeaf::Node(_))
    }
    pub const fn is_leaf(&self) -> bool {
        matches!(self, NodeOrLeaf::Leaf(_))
    }
    pub fn node(self) -> Option<Node> {
        match self {
            NodeOrLeaf::Node(v) => Some(v),
            NodeOrLeaf::Leaf(_) => None,
        }
    }
    pub fn leaf(self) -> Option<Leaf> {
        match self {
            NodeOrLeaf::Leaf(v) => Some(v),
            NodeOrLeaf::Node(_) => None,
        }
    }
    pub fn map_node<T, F: FnOnce(Node) -> T>(self, f: F) -> NodeOrLeaf<T, Leaf> {
        match self {
            NodeOrLeaf::Node(v) => NodeOrLeaf::Node(f(v)),
            NodeOrLeaf::Leaf(v) => NodeOrLeaf::Leaf(v),
        }
    }
    pub fn map_leaf<T, F: FnOnce(Leaf) -> T>(self, f: F) -> NodeOrLeaf<Node, T> {
        match self {
            NodeOrLeaf::Node(v) => NodeOrLeaf::Node(v),
            NodeOrLeaf::Leaf(v) => NodeOrLeaf::Leaf(f(v)),
        }
    }
}
