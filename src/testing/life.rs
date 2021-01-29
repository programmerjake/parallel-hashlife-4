use crate::{
    array::Array,
    index_vec::{IndexVec, IndexVecForEach},
    traits::{HasErrorType, HasLeafType, LeafStep},
};

const DIMENSION: usize = 2;
struct LeafData;

impl HasLeafType<'_, DIMENSION> for LeafData {
    type Leaf = u8;
}

impl HasErrorType for LeafData {
    type Error = std::io::Error;
}

impl LeafStep<'_, DIMENSION> for LeafData {
    fn leaf_step(
        &self,
        neighborhood: Array<Self::Leaf, 3, DIMENSION>,
    ) -> Result<Self::Leaf, Self::Error> {
        let mut sum = 0;
        IndexVec::<DIMENSION>::for_each_index(|index| sum += neighborhood[index], 3, ..);
        Ok(match sum {
            3 => 1,
            4 if neighborhood[IndexVec([1, 1])] != 0 => 1,
            _ => 0,
        })
    }
}
