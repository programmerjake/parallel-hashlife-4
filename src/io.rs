use crate::{
    array::{Array, ArrayRepr},
    index_vec::{IndexVec, IndexVecExt, IndexVecForEach},
    traits::HashlifeData,
    NodeAndLevel, NodeOrLeaf,
};
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::num::NonZeroUsize;
use hashbrown::{hash_map::Entry, HashMap};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Deserializer;
use std::{
    hash::Hash,
    io::{self, BufRead, Write},
};

#[derive(Debug)]
struct LineReader<R> {
    buf: String,
    buf_valid: bool,
    hit_eof: bool,
    reader: R,
}

impl<R: BufRead> LineReader<R> {
    fn new(reader: R) -> Self {
        Self {
            buf: String::new(),
            buf_valid: false,
            hit_eof: false,
            reader,
        }
    }
    fn fill_buf(&mut self) -> io::Result<()> {
        if !self.buf_valid {
            self.buf.clear();
            self.reader.read_line(&mut self.buf)?;
            self.buf_valid = true;
            self.hit_eof = self.buf.is_empty();
            if self.buf.ends_with('\n') {
                self.buf.pop();
                if self.buf.ends_with('\r') {
                    self.buf.pop();
                }
            }
        }
        Ok(())
    }
    fn get_opt(&mut self) -> io::Result<Option<&str>> {
        self.fill_buf()?;
        if self.hit_eof {
            Ok(None)
        } else {
            self.buf_valid = false;
            Ok(Some(&self.buf))
        }
    }
    fn get_if<'a: 'b, 'b, T, F: FnOnce(&'b str) -> Option<T>>(
        &'a mut self,
        f: F,
    ) -> io::Result<Option<T>> {
        self.fill_buf()?;
        if self.hit_eof {
            Ok(None)
        } else {
            let retval = f(&self.buf);
            if retval.is_some() {
                self.buf_valid = false;
            }
            Ok(retval)
        }
    }
    fn get_or_err(&mut self) -> io::Result<&str> {
        self.get_opt()?
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected eof"))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum MacrocellVersion {
    M2,
    M3,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct MacrocellHeader {
    pub version: MacrocellVersion,
    pub dimension: usize,
    pub comment_lines: Option<String>,
    pub generation: Option<String>,
    pub rule: Option<String>,
    pub node_count: Option<NonZeroUsize>,
}

impl MacrocellHeader {
    pub fn calculate_version(&self) -> MacrocellVersion {
        if let Self {
            version,
            dimension: 2,
            comment_lines: _,
            generation: _,
            rule: _,
            node_count: None,
        } = *self
        {
            version
        } else {
            MacrocellVersion::M3
        }
    }
}

impl Default for MacrocellHeader {
    fn default() -> Self {
        Self {
            version: MacrocellVersion::M2,
            dimension: 2,
            comment_lines: None,
            generation: None,
            rule: None,
            node_count: None,
        }
    }
}

struct LineAccumulator(Option<String>);

impl LineAccumulator {
    fn append(&mut self, line: &str) {
        if let Some(lines) = &mut self.0 {
            lines.push('\n');
            lines.push_str(line);
        } else {
            self.0 = Some(line.into());
        }
    }
}

#[derive(Debug)]
pub struct MacrocellReader<R> {
    line_reader: LineReader<R>,
    header: MacrocellHeader,
}

impl<R> MacrocellReader<R> {
    pub fn header(&self) -> &MacrocellHeader {
        &self.header
    }
}

impl<R: BufRead> MacrocellReader<R> {
    fn parse_format_line(mut format_line: &str) -> Option<&str> {
        format_line = format_line.strip_prefix("[M")?;
        let (version, rest) =
            format_line.split_at(format_line.find(|c: char| !c.is_ascii_digit())?);
        if version.is_empty() || version.len() > 100 {
            None
        } else if rest.starts_with(']') {
            Some(version)
        } else {
            None
        }
    }
    pub fn new(reader: R) -> io::Result<Self> {
        let mut line_reader = LineReader::new(reader);
        let version = match Self::parse_format_line(line_reader.get_or_err()?) {
            Some("2") => MacrocellVersion::M2,
            Some("3") => MacrocellVersion::M3,
            Some(version) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported macrocell file version {}", version),
                ));
            }
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid macrocell format line",
                ));
            }
        };
        let mut rule = None;
        let mut generation = None;
        let mut dimension = None;
        let mut node_count = None;
        let mut comment_lines = LineAccumulator(None);
        while let Some(comment) = line_reader.get_if(|line| line.strip_prefix("#"))? {
            match (comment.as_bytes().first(), version) {
                (Some(b'R'), _) => {
                    if rule.is_some() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "duplicate rule line",
                        ));
                    }
                    rule = Some(String::from(comment[1..].trim_start()));
                }
                (Some(b'G'), _) => {
                    if generation.is_some() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "duplicate generation line",
                        ));
                    }
                    generation = Some(String::from(comment[1..].trim_start()));
                }
                (Some(b'D'), MacrocellVersion::M3) => {
                    if dimension.is_some() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "duplicate dimension line",
                        ));
                    }
                    dimension = Some(
                        comment[1..]
                            .trim_start()
                            .parse::<usize>()
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
                    );
                }
                (Some(b'N'), MacrocellVersion::M3) => {
                    if node_count.is_some() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "duplicate node-count line",
                        ));
                    }
                    node_count = Some(
                        comment[1..]
                            .trim_start()
                            .parse::<NonZeroUsize>()
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
                    );
                }
                (Some(b' '), _) => comment_lines.append(&comment[1..]),
                (Some(_), _) if version != MacrocellVersion::M2 => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "unknown special comment kind `#{:?}`",
                            comment.chars().next().unwrap()
                        ),
                    ));
                }
                _ => comment_lines.append(comment),
            }
        }
        Ok(Self {
            line_reader,
            header: MacrocellHeader {
                version,
                dimension: dimension.unwrap_or(2),
                comment_lines: comment_lines.0,
                generation,
                rule,
                node_count,
            },
        })
    }
    pub fn read_body<'a, HL, const DIMENSION: usize>(
        self,
        hash_life: &'a HL,
    ) -> Result<NodeAndLevel<HL::NodeId>, HL::Error>
    where
        HL: HashlifeData<'a, DIMENSION>,
        HL::Error: From<io::Error>,
        IndexVec<DIMENSION>: IndexVecExt,
        Array<HL::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
        Array<HL::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
        HL::Leaf: DeserializeOwned,
    {
        let Self {
            mut line_reader,
            header,
        } = self;
        if header.dimension != DIMENSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported dimension {} (only supported dimension is {})",
                    header.dimension, DIMENSION
                ),
            )
            .into());
        }
        let mut biggest_seen_level = 0;
        let mut multiple_biggest = false;
        let mut nodes = Vec::new();
        nodes.push(hash_life.get_empty_node(biggest_seen_level)?);
        while let Some(line) = line_reader.get_opt()? {
            let node = match line.bytes().next() {
                Some(b'$') | Some(b'.') | Some(b'*')
                    if DIMENSION == 2 && header.version == MacrocellVersion::M2 =>
                {
                    const LEVEL2_SIZE: usize = 8;
                    let mut cells = Array([[false; LEVEL2_SIZE]; LEVEL2_SIZE]);
                    let mut x = 0;
                    let mut y = 0;
                    for ch in line.bytes() {
                        if x > LEVEL2_SIZE || y > LEVEL2_SIZE {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "moved out of range parsing 2-state leaf node's children",
                            )
                            .into());
                        }
                        match ch {
                            b'$' => {
                                x = 0;
                                y += 1;
                            }
                            b'.' => {
                                if x >= LEVEL2_SIZE || y >= LEVEL2_SIZE {
                                    return Err(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "moved out of range parsing 2-state leaf node's children",
                                    )
                                    .into());
                                }
                                x += 1;
                            }
                            b'*' => {
                                if x >= LEVEL2_SIZE || y >= LEVEL2_SIZE {
                                    return Err(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "moved out of range parsing 2-state leaf node's children",
                                    )
                                    .into());
                                }
                                cells[IndexVec([y, x])] = true;
                                x += 1;
                            }
                            _ => {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "invalid character",
                                )
                                .into());
                            }
                        }
                    }
                    let key =
                        Array::try_build_array(|level2_index| -> Result<HL::NodeId, HL::Error> {
                            let key = Array::try_build_array(
                                |level1_index| -> Result<HL::NodeId, HL::Error> {
                                    let key = Array::try_build_array(
                                        |level0_index| -> Result<HL::Leaf, HL::Error> {
                                            let index = level0_index
                                                + level1_index.map(|v| v * 2)
                                                + level2_index.map(|v| v * 4);
                                            let index = IndexVec([index.0[0], index.0[1]]);
                                            if cells[index] {
                                                Ok(serde_json::from_str("1")
                                                    .map_err(io::Error::from)?)
                                            } else {
                                                Ok(HL::Leaf::default())
                                            }
                                        },
                                    )?;
                                    Ok(hash_life.intern_leaf_node(key)?.node)
                                },
                            )?;
                            Ok(hash_life
                                .intern_non_leaf_node(NodeAndLevel {
                                    node: key,
                                    level: 0,
                                })?
                                .node)
                        })?;
                    hash_life.intern_non_leaf_node(NodeAndLevel {
                        node: key,
                        level: 1,
                    })?
                }
                Some(b'0') => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "node size must not start with 0",
                    )
                    .into());
                }
                Some(ch) if ch.is_ascii_digit() => {
                    let mut parts = line.splitn(2, ' ');
                    let log2_size: NonZeroUsize = parts
                        .next()
                        .expect("must have first part since first part has a digit")
                        .parse()
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                    let parts = parts.next().unwrap_or("");
                    if log2_size.get() == 1 {
                        let mut parts = Deserializer::from_str(parts).into_iter();
                        let key = Array::try_build_array(|_| match parts.next() {
                            Some(Err(e)) => Err(e.into()),
                            Some(Ok(v)) => Ok(v),
                            None => Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "expected more cells",
                            )),
                        })?;
                        if parts.next().transpose().map_err(io::Error::from)?.is_some() {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "node has too many children",
                            )
                            .into());
                        }
                        hash_life.intern_leaf_node(key)?
                    } else {
                        let mut parts = parts.split(' ');
                        let key = Array::try_build_array(|_| -> Result<_, HL::Error> {
                            let node_id: usize = parts
                                .next()
                                .ok_or_else(|| {
                                    io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "expected more children",
                                    )
                                })?
                                .parse()
                                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                            if node_id == 0 {
                                Ok(hash_life.get_empty_node(log2_size.get() - 2)?.node)
                            } else {
                                let NodeAndLevel { node, level } = nodes
                                    .get(node_id)
                                    .ok_or_else(|| {
                                        io::Error::new(
                                            io::ErrorKind::InvalidData,
                                            "node id out of range",
                                        )
                                    })?
                                    .clone();
                                if level == log2_size.get() - 2 {
                                    Ok(node)
                                } else {
                                    Err(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        format!(
                                            "node {} is wrong size to be used as a child of node {}",
                                            node_id,
                                            nodes.len()
                                        ),
                                    )
                                    .into())
                                }
                            }
                        })?;
                        if parts.next().is_some() {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "node has too many children",
                            )
                            .into());
                        }
                        hash_life.intern_non_leaf_node(NodeAndLevel {
                            node: key,
                            level: log2_size.get() - 2,
                        })?
                    }
                }
                Some(_) => {
                    return Err(
                        io::Error::new(io::ErrorKind::InvalidData, "invalid character").into(),
                    );
                }
                None => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "blank lines not allowed in Macrocell file",
                    )
                    .into());
                }
            };
            if nodes.len() > 1 && biggest_seen_level == node.level {
                multiple_biggest = true;
            } else if biggest_seen_level < node.level {
                biggest_seen_level = node.level;
                multiple_biggest = false;
            }
            nodes.push(node);
        }
        if let Some(header_node_count) = header.node_count {
            let actual_node_count = nodes.len() - 1; // subtract 1 since node #0 isn't included in file
            if header_node_count.get() < actual_node_count {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "too many nodes in macrocell file",
                )
                .into());
            }
            if header_node_count.get() > actual_node_count {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "too few nodes in macrocell file",
                )
                .into());
            }
        }
        let node = nodes.pop().unwrap();
        if node.level != biggest_seen_level || multiple_biggest {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "last node must be bigger than all other nodes",
            )
            .into())
        } else {
            Ok(node)
        }
    }
}

#[derive(Debug)]
pub struct MacrocellWriter<W> {
    writer: W,
    header: MacrocellHeader,
}

impl<W: Write> MacrocellWriter<W> {
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            header: MacrocellHeader::default(),
        }
    }
    pub fn with_header(writer: W, header: MacrocellHeader) -> Self {
        Self { writer, header }
    }
    pub fn write<'a, HL, const DIMENSION: usize>(
        self,
        hash_life: &'a HL,
        root_node: NodeAndLevel<HL::NodeId>,
    ) -> Result<(), HL::Error>
    where
        HL: HashlifeData<'a, DIMENSION>,
        HL::Error: From<io::Error>,
        IndexVec<DIMENSION>: IndexVecExt,
        Array<HL::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
        Array<HL::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
        HL::NodeId: Hash + Eq,
        HL::Leaf: Serialize,
    {
        let Self {
            mut writer,
            mut header,
        } = self;
        let mut nodes = Vec::new();
        nodes.push(hash_life.get_empty_node(0)?);
        let mut nodes_map = HashMap::new();
        for level in 0..=root_node.level {
            nodes_map.insert(hash_life.get_empty_node(level)?.node, 0);
        }
        fn get_nodes<'a, HL, const DIMENSION: usize>(
            hash_life: &'a HL,
            node: NodeAndLevel<HL::NodeId>,
            nodes: &mut Vec<NodeAndLevel<HL::NodeId>>,
            nodes_map: &mut HashMap<HL::NodeId, usize>,
        ) where
            HL: HashlifeData<'a, DIMENSION>,
            IndexVec<DIMENSION>: IndexVecExt,
            Array<HL::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
            Array<HL::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
            HL::NodeId: Hash + Eq,
        {
            if nodes_map.contains_key(&node.node) {
                return;
            }
            if let NodeOrLeaf::Node(key) = hash_life.get_node_key(node.clone()) {
                IndexVec::<DIMENSION>::for_each_index(
                    |index| {
                        get_nodes(
                            hash_life,
                            key.as_ref().map_node(|node| node[index].clone()),
                            nodes,
                            nodes_map,
                        );
                    },
                    2,
                    ..,
                );
            }
            if let Entry::Vacant(entry) = nodes_map.entry(node.node.clone()) {
                let index = nodes.len();
                nodes.push(node);
                entry.insert(index);
            }
        }
        get_nodes(hash_life, root_node.clone(), &mut nodes, &mut nodes_map);
        if nodes.len() == 1 {
            nodes.push(root_node);
        } else {
            assert_eq!(nodes.last().unwrap().node, root_node.node);
        }
        header.dimension = DIMENSION;
        if header.calculate_version() != MacrocellVersion::M2 {
            header.node_count = Some(NonZeroUsize::new(nodes.len() - 1).unwrap());
        }
        header.version = header.calculate_version();
        let MacrocellHeader {
            version,
            dimension,
            comment_lines,
            generation,
            rule,
            node_count,
        } = header;
        fn write_lines(
            writer: &mut impl Write,
            line_prefix: &str,
            text: Option<impl ToString>,
        ) -> io::Result<()> {
            if let Some(text) = text {
                let text = text.to_string();
                for i in text.lines() {
                    if i.is_empty() {
                        writeln!(writer, "{}", line_prefix)?;
                    } else {
                        writeln!(writer, "{} {}", line_prefix, i)?;
                    }
                }
                if text.is_empty() {
                    writeln!(writer, "{}", line_prefix)?;
                }
            }
            Ok(())
        }
        match version {
            MacrocellVersion::M2 => {
                writeln!(writer, "[M2] (parallel-hashlife)")?;
                assert_eq!(dimension, 2);
            }
            MacrocellVersion::M3 => {
                writeln!(writer, "[M3] (parallel-hashlife)")?;
                write_lines(&mut writer, "#D", Some(dimension))?;
            }
        }
        write_lines(&mut writer, "#G", generation)?;
        write_lines(&mut writer, "#R", rule)?;
        write_lines(&mut writer, "#N", node_count)?;
        write_lines(&mut writer, "#", comment_lines)?;
        for node in nodes.drain(1..) {
            match hash_life.get_node_key(node) {
                NodeOrLeaf::Node(key) => {
                    write!(writer, "{}", 2 + key.level)?;
                    IndexVec::<DIMENSION>::try_for_each_index(
                        |index| write!(writer, " {}", nodes_map[&key.node[index]]),
                        2,
                        ..,
                    )?;
                    writeln!(writer)?;
                }
                NodeOrLeaf::Leaf(key) => {
                    write!(writer, "1")?;
                    IndexVec::<DIMENSION>::try_for_each_index(
                        |index| -> Result<(), io::Error> {
                            write!(writer, " ")?;
                            serde_json::to_writer(&mut writer, &key[index]).map_err(io::Error::from)
                        },
                        2,
                        ..,
                    )?;
                    writeln!(writer)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::{
        array::{Array, ArrayRepr},
        index_vec::{IndexVec, IndexVecExt},
        simple::Simple,
        traits::{HasErrorType, HasLeafType, HashlifeData},
        NodeAndLevel, NodeOrLeaf,
    };
    use alloc::{string::String, vec::Vec};
    use core::num::NonZeroUsize;
    use std::{io, print, println};

    use super::{MacrocellHeader, MacrocellReader, MacrocellVersion, MacrocellWriter};

    struct LeafData;

    impl HasErrorType for LeafData {
        type Error = io::Error;
    }

    type Leaf = Option<u8>;

    impl<const DIMENSION: usize> HasLeafType<DIMENSION> for LeafData
    where
        Leaf: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
        Array<Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    {
        type Leaf = Leaf;
    }

    type NodeId<const DIMENSION: usize> = crate::simple::NodeId<Leaf, DIMENSION>;

    fn get_leaf<const DIMENSION: usize>(
        hl: &Simple<LeafData, DIMENSION>,
        mut node: NodeAndLevel<NodeId<DIMENSION>>,
        mut location: IndexVec<DIMENSION>,
    ) -> Leaf
    where
        Leaf: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
        Array<Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
        NodeId<DIMENSION>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
        Array<NodeId<DIMENSION>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
        IndexVec<DIMENSION>: IndexVecExt,
    {
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

    fn dump_2d(hl: &Simple<LeafData, 2>, node: NodeAndLevel<NodeId<2>>, title: &str) {
        println!("{}:", title);
        let size = 2usize << node.level;
        for y in 0..size {
            for x in 0..size {
                match get_leaf(hl, node.clone(), IndexVec([y, x])) {
                    None => print!("_ "),
                    Some(leaf) => print!("{} ", leaf),
                }
            }
            println!();
        }
    }

    fn dump_1d(hl: &Simple<LeafData, 1>, node: NodeAndLevel<NodeId<1>>, title: &str) {
        println!("{}:", title);
        let size = 2usize << node.level;
        for x in 0..size {
            match get_leaf(hl, node.clone(), IndexVec([x])) {
                None => print!("_ "),
                Some(leaf) => print!("{} ", leaf),
            }
        }
        println!();
    }

    fn build_with_helper<const DIMENSION: usize>(
        hl: &Simple<LeafData, DIMENSION>,
        f: &mut impl FnMut(IndexVec<DIMENSION>) -> Leaf,
        outer_location: IndexVec<DIMENSION>,
        level: usize,
    ) -> NodeAndLevel<NodeId<DIMENSION>>
    where
        Leaf: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
        Array<Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
        NodeId<DIMENSION>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
        Array<NodeId<DIMENSION>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
        IndexVec<DIMENSION>: IndexVecExt,
    {
        if level == 0 {
            hl.intern_leaf_node(Array::build_array(|index| {
                f(index + outer_location.map(|v| v * 2))
            }))
            .unwrap()
        } else {
            let key = Array::build_array(|index| {
                build_with_helper(hl, f, index + outer_location.map(|v| v * 2), level - 1).node
            });
            hl.intern_non_leaf_node(NodeAndLevel {
                node: key,
                level: level - 1,
            })
            .unwrap()
        }
    }

    fn build_with<const DIMENSION: usize>(
        hl: &Simple<LeafData, DIMENSION>,
        mut f: impl FnMut(IndexVec<DIMENSION>) -> Leaf,
        level: usize,
    ) -> NodeAndLevel<NodeId<DIMENSION>>
    where
        Leaf: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
        Array<Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
        NodeId<DIMENSION>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
        Array<NodeId<DIMENSION>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
        IndexVec<DIMENSION>: IndexVecExt,
    {
        build_with_helper(hl, &mut f, 0usize.into(), level)
    }

    fn build_2d<const SIZE: usize>(
        hl: &Simple<LeafData, 2>,
        array: [[Leaf; SIZE]; SIZE],
    ) -> NodeAndLevel<NodeId<2>> {
        assert!(SIZE.is_power_of_two());
        assert_ne!(SIZE, 1);
        let log2_size = SIZE.trailing_zeros();
        let level = log2_size as usize - 1;
        let array = Array(array);
        build_with(hl, |index| array[index], level)
    }

    fn build_1d<const SIZE: usize>(
        hl: &Simple<LeafData, 1>,
        array: [Leaf; SIZE],
    ) -> NodeAndLevel<NodeId<1>> {
        assert!(SIZE.is_power_of_two());
        assert_ne!(SIZE, 1);
        let log2_size = SIZE.trailing_zeros();
        let level = log2_size as usize - 1;
        let array = Array(array);
        build_with(hl, |index| array[index], level)
    }

    macro_rules! leaf {
        (_) => {
            None
        };
        ($literal:literal) => {
            Some($literal)
        };
    }

    macro_rules! array_1d {
        ($($cell:tt),+) => {
            [$(leaf!($cell),)+]
        };
    }

    macro_rules! array_2d {
        ($([$($cell:tt),+]),+) => {
            [$([$(leaf!($cell),)+],)+]
        };
    }

    #[track_caller]
    fn test_write<BuildIn, Build, Dump, const DIMENSION: usize, const LINES: usize>(
        header: MacrocellHeader,
        array: BuildIn,
        expected_lines: [&str; LINES],
        expected_header: MacrocellHeader,
        build: Build,
        dump: Dump,
    ) where
        Leaf: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
        Array<Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
        NodeId<DIMENSION>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
        Array<NodeId<DIMENSION>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
        IndexVec<DIMENSION>: IndexVecExt,
        Build: FnOnce(&Simple<LeafData, DIMENSION>, BuildIn) -> NodeAndLevel<NodeId<DIMENSION>>,
        Dump: Fn(&Simple<LeafData, DIMENSION>, NodeAndLevel<NodeId<DIMENSION>>, &str),
    {
        let hl = Simple::new(LeafData);
        let node = build(&hl, array);
        let mut writer = Vec::<u8>::new();
        MacrocellWriter::with_header(&mut writer, header)
            .write(&hl, node.clone())
            .unwrap();
        let text = String::from_utf8(writer).unwrap();
        println!("text:");
        println!("{}", text);
        let expected_text = expected_lines.join("\n") + "\n";
        assert_eq!(text, expected_text);
        let reader = MacrocellReader::new(expected_text.as_bytes()).unwrap();
        assert_eq!(*reader.header(), expected_header);
        let read_node = reader.read_body(&hl).unwrap();
        dump(&hl, read_node.clone(), "read_node");
        assert_eq!(read_node, node);
    }

    #[track_caller]
    fn test_read<BuildIn, Build, Dump, const DIMENSION: usize, const LINES: usize>(
        lines: [&str; LINES],
        expected_array: BuildIn,
        expected_header: MacrocellHeader,
        build: Build,
        dump: Dump,
    ) where
        Leaf: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
        Array<Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
        NodeId<DIMENSION>: ArrayRepr<2, DIMENSION> + ArrayRepr<3, DIMENSION>,
        Array<NodeId<DIMENSION>, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
        IndexVec<DIMENSION>: IndexVecExt,
        Build: FnOnce(&Simple<LeafData, DIMENSION>, BuildIn) -> NodeAndLevel<NodeId<DIMENSION>>,
        Dump: Fn(&Simple<LeafData, DIMENSION>, NodeAndLevel<NodeId<DIMENSION>>, &str),
    {
        let hl = Simple::new(LeafData);
        let text = lines.join("\n") + "\n";
        let reader = MacrocellReader::new(text.as_bytes()).unwrap();
        assert_eq!(*reader.header(), expected_header);
        let read_node = reader.read_body(&hl).unwrap();
        dump(&hl, read_node.clone(), "read_node");
        let expected_node = build(&hl, expected_array);
        dump(&hl, expected_node.clone(), "expected_node");
        assert_eq!(read_node, expected_node);
    }

    #[test]
    fn test1() {
        test_write(
            MacrocellHeader {
                version: MacrocellVersion::M2,
                comment_lines: None,
                generation: None,
                rule: None,
                ..MacrocellHeader::default()
            },
            array_2d![
                [_, _], //
                [_, _]
            ],
            [
                "[M2] (parallel-hashlife)", //
                "1 null null null null",
            ],
            MacrocellHeader {
                version: MacrocellVersion::M2,
                comment_lines: None,
                generation: None,
                rule: None,
                dimension: 2,
                node_count: None,
            },
            build_2d,
            dump_2d,
        );
    }

    #[test]
    fn test2() {
        test_write(
            MacrocellHeader {
                version: MacrocellVersion::M2,
                comment_lines: Some("".into()),
                generation: None,
                rule: None,
                ..MacrocellHeader::default()
            },
            array_2d![
                [_, _], //
                [_, _]
            ],
            [
                "[M2] (parallel-hashlife)", //
                "#",
                "1 null null null null",
            ],
            MacrocellHeader {
                version: MacrocellVersion::M2,
                comment_lines: Some("".into()),
                generation: None,
                rule: None,
                dimension: 2,
                node_count: None,
            },
            build_2d,
            dump_2d,
        );
    }

    #[test]
    fn test3() {
        test_write(
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: Some("".into()),
                generation: None,
                rule: None,
                ..MacrocellHeader::default()
            },
            array_2d![
                [_, _], //
                [_, _]
            ],
            [
                "[M3] (parallel-hashlife)",
                "#D 2",
                "#N 1",
                "#",
                "1 null null null null",
            ],
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: Some("".into()),
                generation: None,
                rule: None,
                dimension: 2,
                node_count: NonZeroUsize::new(1),
            },
            build_2d,
            dump_2d,
        );
    }

    #[test]
    fn test4() {
        test_write(
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: Some("".into()),
                generation: Some("23456".into()),
                rule: Some("MyRule".into()),
                ..MacrocellHeader::default()
            },
            array_2d![
                [0, 1], //
                [_, _]
            ],
            [
                "[M3] (parallel-hashlife)",
                "#D 2",
                "#G 23456",
                "#R MyRule",
                "#N 1",
                "#",
                "1 0 1 null null",
            ],
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: Some("".into()),
                generation: Some("23456".into()),
                rule: Some("MyRule".into()),
                dimension: 2,
                node_count: NonZeroUsize::new(1),
            },
            build_2d,
            dump_2d,
        );
    }

    #[test]
    fn test5() {
        test_write(
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: Some("".into()),
                generation: Some("23456".into()),
                rule: Some("MyRule".into()),
                ..MacrocellHeader::default()
            },
            array_2d![
                [_, _, _, _], //
                [_, _, _, _],
                [_, _, _, _],
                [_, _, _, _]
            ],
            [
                "[M3] (parallel-hashlife)",
                "#D 2",
                "#G 23456",
                "#R MyRule",
                "#N 1",
                "#",
                "2 0 0 0 0",
            ],
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: Some("".into()),
                generation: Some("23456".into()),
                rule: Some("MyRule".into()),
                dimension: 2,
                node_count: NonZeroUsize::new(1),
            },
            build_2d,
            dump_2d,
        );
    }

    #[test]
    fn test6() {
        test_write(
            MacrocellHeader {
                ..MacrocellHeader::default()
            },
            array_2d![
                [0, _, _, _], //
                [_, _, _, _],
                [_, _, _, _],
                [_, _, _, _]
            ],
            [
                "[M2] (parallel-hashlife)",
                "1 0 null null null",
                "2 1 0 0 0",
            ],
            MacrocellHeader {
                version: MacrocellVersion::M2,
                comment_lines: None,
                generation: None,
                rule: None,
                dimension: 2,
                node_count: None,
            },
            build_2d,
            dump_2d,
        );
    }

    #[test]
    fn test7() {
        test_write(
            MacrocellHeader {
                ..MacrocellHeader::default()
            },
            array_2d![
                [0, _, _, _], //
                [_, 1, _, _],
                [_, _, 1, _],
                [_, _, _, 1]
            ],
            [
                "[M2] (parallel-hashlife)",
                "1 0 null null 1",
                "1 1 null null 1",
                "2 1 0 0 2",
            ],
            MacrocellHeader {
                version: MacrocellVersion::M2,
                comment_lines: None,
                generation: None,
                rule: None,
                dimension: 2,
                node_count: None,
            },
            build_2d,
            dump_2d,
        );
    }

    #[test]
    fn test8() {
        test_write(
            MacrocellHeader {
                version: MacrocellVersion::M3,
                ..MacrocellHeader::default()
            },
            array_2d![
                [_, _, _, _, _, _, _, _],
                [_, _, _, _, _, _, _, _],
                [_, _, _, _, _, _, _, _],
                [_, _, _, _, _, _, _, _],
                [_, _, _, _, _, _, _, _],
                [_, _, _, _, _, _, _, _],
                [_, _, _, _, _, _, _, _],
                [_, _, _, _, _, _, _, _]
            ],
            [
                "[M3] (parallel-hashlife)", //
                "#D 2",
                "#N 1",
                "3 0 0 0 0",
            ],
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: None,
                generation: None,
                rule: None,
                dimension: 2,
                node_count: NonZeroUsize::new(1),
            },
            build_2d,
            dump_2d,
        );
    }

    #[test]
    fn test9() {
        test_write(
            MacrocellHeader {
                ..MacrocellHeader::default()
            },
            array_2d![
                [_, _, _, _, _, _, _, _],
                [_, _, 1, 1, 1, _, _, _],
                [_, 1, _, _, _, 1, _, _],
                [1, _, _, _, _, _, 1, _],
                [1, _, _, _, _, _, 1, _],
                [1, _, _, _, _, _, 1, _],
                [_, 1, _, _, _, 1, _, _],
                [_, _, 1, 1, 1, _, _, _]
            ],
            [
                "[M2] (parallel-hashlife)",
                "1 null null 1 1",
                "1 null 1 1 null",
                "2 0 1 2 0",
                "1 null null 1 null",
                "1 null 1 null null",
                "2 4 0 5 4",
                "1 1 null 1 null",
                "2 7 0 5 1",
                "2 0 7 2 0",
                "3 3 6 8 9",
            ],
            MacrocellHeader {
                version: MacrocellVersion::M2,
                comment_lines: None,
                generation: None,
                rule: None,
                dimension: 2,
                node_count: None,
            },
            build_2d,
            dump_2d,
        );
    }

    #[test]
    fn test10() {
        test_write(
            MacrocellHeader {
                version: MacrocellVersion::M3,
                ..MacrocellHeader::default()
            },
            array_2d![
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]
            ],
            [
                "[M3] (parallel-hashlife)",
                "#D 2",
                "#N 3",
                "1 1 1 1 1",
                "2 1 1 1 1",
                "3 2 2 2 2",
            ],
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: None,
                generation: None,
                rule: None,
                dimension: 2,
                node_count: NonZeroUsize::new(3),
            },
            build_2d,
            dump_2d,
        );
    }

    #[test]
    fn test11() {
        test_write(
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: Some("a\n".into()),
                ..MacrocellHeader::default()
            },
            array_2d![
                [_, _], //
                [_, _]
            ],
            [
                "[M3] (parallel-hashlife)",
                "#D 2",
                "#N 1",
                "# a",
                "1 null null null null",
            ],
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: Some("a".into()),
                generation: None,
                rule: None,
                dimension: 2,
                node_count: NonZeroUsize::new(1),
            },
            build_2d,
            dump_2d,
        );
    }

    #[test]
    fn test12() {
        test_write(
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: Some("a\n\n\n\nb".into()),
                ..MacrocellHeader::default()
            },
            array_2d![
                [_, _], //
                [_, _]
            ],
            [
                "[M3] (parallel-hashlife)",
                "#D 2",
                "#N 1",
                "# a",
                "#",
                "#",
                "#",
                "# b",
                "1 null null null null",
            ],
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: Some("a\n\n\n\nb".into()),
                generation: None,
                rule: None,
                dimension: 2,
                node_count: NonZeroUsize::new(1),
            },
            build_2d,
            dump_2d,
        );
    }

    #[test]
    fn test13() {
        test_write(
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: Some(" a".into()),
                ..MacrocellHeader::default()
            },
            array_2d![
                [_, _], //
                [_, _]
            ],
            [
                "[M3] (parallel-hashlife)",
                "#D 2",
                "#N 1",
                "#  a",
                "1 null null null null",
            ],
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: Some(" a".into()),
                generation: None,
                rule: None,
                dimension: 2,
                node_count: NonZeroUsize::new(1),
            },
            build_2d,
            dump_2d,
        );
    }

    #[test]
    fn test14() {
        test_read(
            [
                "[M2] (parallel-hashlife)",
                "..****$.*....*$*.*..*.*$*......*$*.*..*.*$*..**..*$.*....*$..****$",
            ],
            array_2d![
                [_, _, 1, 1, 1, 1, _, _],
                [_, 1, _, _, _, _, 1, _],
                [1, _, 1, _, _, 1, _, 1],
                [1, _, _, _, _, _, _, 1],
                [1, _, 1, _, _, 1, _, 1],
                [1, _, _, 1, 1, _, _, 1],
                [_, 1, _, _, _, _, 1, _],
                [_, _, 1, 1, 1, 1, _, _]
            ],
            MacrocellHeader {
                version: MacrocellVersion::M2,
                comment_lines: None,
                generation: None,
                rule: None,
                dimension: 2,
                node_count: None,
            },
            build_2d,
            dump_2d,
        );
    }

    #[test]
    fn test15() {
        #[rustfmt::skip]
        let expected_array = array_2d![
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, 1, 1, 1, 1, 1, 1, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, 1, _, _, _, _, _, 1, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, 1, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, 1, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 1, 1, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 1, 1, 1, 1, _, 1, 1, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 1, 1, 1, 1, 1, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 1, 1, 1, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 1, 1, 1, 1, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 1, _, _, _, 1, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, 1, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, 1, 1, _, _, _, _, _, _, 1, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, 1, 1, 1, 1, _, 1, 1, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, 1, 1, 1, 1, 1, 1, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, 1, 1, 1, 1, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _]
        ];
        test_read(
            [
                "[M2] (golly 2.8)",
                "#R B3/S23",
                "$$$$$$.******$*.....*$",
                "......*$.....*$",
                "4 0 1 0 2",
                "$$$$......**$..****.*$...*****$....***$",
                "$$$$$*$",
                "4 0 0 4 5",
                "$$$$$......**$..****.*$..******$",
                "...****$",
                "4 0 7 0 8",
                "$$....****$...*...*$.......*$......*$*$",
                "4 10 0 0 0",
                "5 3 6 9 11",
            ],
            expected_array,
            MacrocellHeader {
                version: MacrocellVersion::M2,
                comment_lines: None,
                generation: None,
                rule: Some("B3/S23".into()),
                dimension: 2,
                node_count: None,
            },
            build_2d,
            dump_2d,
        );
    }

    #[test]
    fn test1d_1() {
        test_write(
            MacrocellHeader {
                ..MacrocellHeader::default()
            },
            array_1d![_, _],
            [
                "[M3] (parallel-hashlife)", //
                "#D 1",
                "#N 1",
                "1 null null",
            ],
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: None,
                generation: None,
                rule: None,
                dimension: 1,
                node_count: NonZeroUsize::new(1),
            },
            build_1d,
            dump_1d,
        );
    }

    #[test]
    fn test1d_2() {
        test_write(
            MacrocellHeader {
                ..MacrocellHeader::default()
            },
            array_1d![0, _],
            [
                "[M3] (parallel-hashlife)", //
                "#D 1",
                "#N 1",
                "1 0 null",
            ],
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: None,
                generation: None,
                rule: None,
                dimension: 1,
                node_count: NonZeroUsize::new(1),
            },
            build_1d,
            dump_1d,
        );
    }

    #[test]
    fn test1d_3() {
        test_write(
            MacrocellHeader {
                ..MacrocellHeader::default()
            },
            array_1d![0, 0, 0, 0],
            [
                "[M3] (parallel-hashlife)", //
                "#D 1",
                "#N 2",
                "1 0 0",
                "2 1 1",
            ],
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: None,
                generation: None,
                rule: None,
                dimension: 1,
                node_count: NonZeroUsize::new(2),
            },
            build_1d,
            dump_1d,
        );
    }

    #[test]
    fn test1d_4() {
        test_write(
            MacrocellHeader {
                ..MacrocellHeader::default()
            },
            array_1d![0, 0, 0, 0, 0, 0, 0, 0],
            [
                "[M3] (parallel-hashlife)", //
                "#D 1",
                "#N 3",
                "1 0 0",
                "2 1 1",
                "3 2 2",
            ],
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: None,
                generation: None,
                rule: None,
                dimension: 1,
                node_count: NonZeroUsize::new(3),
            },
            build_1d,
            dump_1d,
        );
    }

    #[test]
    fn test1d_5() {
        test_write(
            MacrocellHeader {
                ..MacrocellHeader::default()
            },
            array_1d![0, _, _, 0, _, 0, 0, _],
            [
                "[M3] (parallel-hashlife)",
                "#D 1",
                "#N 5",
                "1 0 null",
                "1 null 0",
                "2 1 2",
                "2 2 1",
                "3 3 4",
            ],
            MacrocellHeader {
                version: MacrocellVersion::M3,
                comment_lines: None,
                generation: None,
                rule: None,
                dimension: 1,
                node_count: NonZeroUsize::new(5),
            },
            build_1d,
            dump_1d,
        );
    }
}
