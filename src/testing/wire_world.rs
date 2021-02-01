use crate::{
    array::Array,
    index_vec::{IndexVec, IndexVecForEach},
    traits::{HasErrorType, HasLeafType, LeafStep},
};

pub const DIMENSION: usize = 2;
pub struct LeafData;

impl HasLeafType<DIMENSION> for LeafData {
    type Leaf = u8;
}

impl HasErrorType for LeafData {
    type Error = std::io::Error;
}

pub const EMPTY: u8 = 0;
pub const ELECTRON_HEAD: u8 = 1;
pub const ELECTRON_TAIL: u8 = 2;
pub const CONDUCTOR: u8 = 3;

impl LeafStep<DIMENSION> for LeafData {
    fn leaf_step(
        &self,
        neighborhood: Array<Self::Leaf, 3, DIMENSION>,
    ) -> Result<Self::Leaf, Self::Error> {
        let mut neighboring_electron_head_count = 0;
        IndexVec::<DIMENSION>::for_each_index(
            |index| {
                if neighborhood[index] == ELECTRON_HEAD {
                    neighboring_electron_head_count += 1;
                }
            },
            3,
            ..,
        );
        Ok(match neighborhood[IndexVec([1, 1])] {
            ELECTRON_HEAD => ELECTRON_TAIL,
            ELECTRON_TAIL => CONDUCTOR,
            CONDUCTOR => match neighboring_electron_head_count {
                1 | 2 => ELECTRON_HEAD,
                _ => CONDUCTOR,
            },
            _ => EMPTY,
        })
    }
}
