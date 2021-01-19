use crate::{
    array::{Array, ArrayRepr},
    index_vec::{IndexVec, IndexVecExt},
    traits::HashlifeData,
    NodeAndLevel,
};
use alloc::{format, string::String, vec::Vec};
use core::num::NonZeroUsize;
use serde::de::DeserializeOwned;
use serde_json::Deserializer;
use std::io::{self, BufRead};

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
    pub comment_lines: String,
    pub generation: Option<String>,
    pub rule: Option<String>,
}

impl Default for MacrocellHeader {
    fn default() -> Self {
        Self {
            version: MacrocellVersion::M2,
            dimension: 2,
            comment_lines: String::new(),
            generation: None,
            rule: None,
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

impl From<LineAccumulator> for String {
    fn from(v: LineAccumulator) -> Self {
        v.0.unwrap_or_default()
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
                comment_lines: comment_lines.into(),
                generation,
                rule,
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
                        if x >= LEVEL2_SIZE || y >= LEVEL2_SIZE {
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
                                x += 1;
                            }
                            b'*' => {
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
