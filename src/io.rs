use alloc::{boxed::Box, format, string::String, vec::Vec};
use core::{
    convert::TryInto,
    num::{NonZeroU128, NonZeroUsize},
};
use std::{
    error,
    io::{self, BufRead},
};

use crate::{
    array::{Array, ArrayRepr},
    index_vec::{IndexVec, IndexVecExt},
    traits::{HasErrorType, HasLeafType, HashlifeData},
    NodeAndLevel,
};

pub trait ReadComment<'a>: HasErrorType {
    fn read_comment(&'a self, text: &str) -> Result<(), Self::Error> {
        let _ = text;
        Ok(())
    }
}

pub trait ReadRule<'a>: HasErrorType {
    fn read_rule(&'a self, text: &str) -> Result<(), Self::Error> {
        let _ = text;
        Ok(())
    }
}

pub trait ReadGenerationCount<'a>: HasErrorType {
    fn read_generation_count(&'a self, text: &str) -> Result<(), Self::Error> {
        let _ = text;
        Ok(())
    }
}

pub trait ReadLeaf<'a, const DIMENSION: usize>: HasLeafType<DIMENSION> + HasErrorType
where
    Array<Self::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn read_leaf(&'a self, text: &str) -> Result<Self::Leaf, Self::Error>;
}

impl<'a, T, const DIMENSION: usize> ReadLeaf<'a, DIMENSION> for T
where
    T: HasLeafType<DIMENSION> + HasErrorType,
    NonZeroU128: TryInto<T::Leaf>,
    <NonZeroU128 as TryInto<T::Leaf>>::Error: Into<Box<dyn error::Error + Send + Sync>>,
    T::Error: From<io::Error>,
    Array<Self::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
{
    fn read_leaf(&'a self, text: &str) -> Result<Self::Leaf, Self::Error> {
        match NonZeroU128::new(
            text.parse()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
        ) {
            Some(value) => Ok(value
                .try_into()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?),
            None => Ok(Self::Leaf::default()),
        }
    }
}

struct LineReader<R> {
    buf: String,
    reader: R,
}

impl<R: BufRead> LineReader<R> {
    fn new(reader: R) -> Self {
        Self {
            buf: String::new(),
            reader,
        }
    }
    fn next(&mut self) -> Result<Option<&str>, io::Error> {
        if self.reader.read_line(&mut self.buf)? == 0 {
            Ok(None)
        } else {
            Ok(Some(self.buf.trim_end_matches(&['\n', '\r'] as &[char])))
        }
    }
}

pub fn read_macrocell<'a, HL, R, const DIMENSION: usize>(
    hash_life: &'a HL,
    reader: R,
) -> Result<NodeAndLevel<HL::NodeId>, HL::Error>
where
    HL: HashlifeData<'a, DIMENSION>
        + ReadLeaf<'a, DIMENSION>
        + ReadComment<'a>
        + ReadRule<'a>
        + ReadGenerationCount<'a>,
    HL::Error: From<io::Error>,
    IndexVec<DIMENSION>: IndexVecExt,
    Array<HL::NodeId, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    Array<HL::Leaf, 2, DIMENSION>: ArrayRepr<2, DIMENSION>,
    R: BufRead,
{
    let mut line_reader = LineReader::new(reader);
    let mut line = line_reader.next()?;
    match line {
        // TODO: check for correct format line for DIMENSION != 2 and/or non-integer
        // cells -- `[M2]` is only correct for 2 dimensions and for integer cells
        Some(line) if line.starts_with("[M2]") => {}
        Some(_) => {
            return Err(
                io::Error::new(io::ErrorKind::InvalidData, "invalid MacroCell format line").into(),
            )
        }
        None => {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "missing MacroCell format line",
            )
            .into())
        }
    }
    line = line_reader.next()?;
    while let Some(comment_line) = line.filter(|line| line.starts_with("#")) {
        if comment_line.starts_with("#R") {
            hash_life.read_rule(&comment_line[2..])?;
        } else if comment_line.starts_with("#G") {
            hash_life.read_generation_count(&comment_line[2..])?;
        } else {
            hash_life.read_comment(&comment_line[1..])?;
        }
        line = line_reader.next()?;
    }
    let mut biggest_seen_level = 0;
    let mut multiple_biggest = false;
    let mut nodes = Vec::new();
    nodes.push(hash_life.get_empty_node(biggest_seen_level)?);
    while line.is_some() {
        let node = match line.unwrap().chars().next() {
            Some('$') | Some('.') | Some('*') if DIMENSION == 2 => {
                const LEVEL2_SIZE: usize = 8;
                let mut cells = Array([[false; LEVEL2_SIZE]; LEVEL2_SIZE]);
                let mut x = 0;
                let mut y = 0;
                for ch in line.unwrap().chars() {
                    if x >= LEVEL2_SIZE || y >= LEVEL2_SIZE {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "moved out of range parsing 2-state leaf node's children",
                        )
                        .into());
                    }
                    match ch {
                        '$' => {
                            x = 0;
                            y += 1;
                        }
                        '.' => {
                            x += 1;
                        }
                        '*' => {
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
                                            hash_life.read_leaf("1")
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
            Some('0') => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "node size must not start with 0",
                )
                .into());
            }
            Some(ch) if ch.is_ascii_digit() => {
                let mut parts = line.unwrap().split(' ');
                let log2_size: NonZeroUsize = parts
                    .next()
                    .expect("must have first part since first part has a digit")
                    .parse()
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                let node = if log2_size.get() == 1 {
                    let key = Array::try_build_array(|_| {
                        hash_life.read_leaf(parts.next().ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidData, "expected more cells")
                        })?)
                    })?;
                    hash_life.intern_leaf_node(key)?
                } else {
                    let key = Array::try_build_array(|_| -> Result<_, HL::Error> {
                        let node_id: usize = parts
                            .next()
                            .ok_or_else(|| {
                                io::Error::new(io::ErrorKind::InvalidData, "expected more children")
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
                    hash_life.intern_non_leaf_node(NodeAndLevel {
                        node: key,
                        level: log2_size.get() - 2,
                    })?
                };
                if parts.next().is_some() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "node has too many children",
                    )
                    .into());
                }
                node
            }
            Some(_) => {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid character").into());
            }
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "blank lines not allowed in MacroCell file",
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
        line = line_reader.next()?;
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
