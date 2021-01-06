use crate::arrayvec::ArrayVec;
use std::{
    fmt,
    hash::{Hash, Hasher},
};

pub unsafe trait IndexVecExt: From<usize> {
    fn as_slice(&self) -> &[usize];
    fn as_mut_slice(&mut self) -> &mut [usize];
    fn try_for_each_index<E, F: FnMut(Self) -> Result<(), E>>(f: F, length: usize)
        -> Result<(), E>;
    fn for_each_index<F: FnMut(Self)>(f: F, length: usize);
    fn try_for_each_index_before<E, F: FnMut(Self) -> Result<(), E>>(
        f: F,
        length: usize,
        one_past_end: Self,
    ) -> Result<(), E>;
    fn for_each_index_before<F: FnMut(Self)>(f: F, length: usize, one_past_end: Self);
}

pub unsafe trait IndexVecNonzeroDimension: IndexVecExt {
    type PrevDimension: IndexVecExt;
    fn first(self) -> usize;
    fn rest(self) -> Self::PrevDimension;
    fn combine(first: usize, rest: Self::PrevDimension) -> Self;
}

unsafe impl IndexVecExt for IndexVec<0> {
    fn as_slice(&self) -> &[usize] {
        &self.0
    }
    fn as_mut_slice(&mut self) -> &mut [usize] {
        &mut self.0
    }
    fn try_for_each_index<E, F: FnMut(Self) -> Result<(), E>>(
        mut f: F,
        _length: usize,
    ) -> Result<(), E> {
        f(IndexVec([]))
    }
    fn for_each_index<F: FnMut(Self)>(mut f: F, _length: usize) {
        f(IndexVec([]))
    }
    fn try_for_each_index_before<E, F: FnMut(Self) -> Result<(), E>>(
        _f: F,
        _length: usize,
        _one_past_end: Self,
    ) -> Result<(), E> {
        Ok(())
    }
    fn for_each_index_before<F: FnMut(Self)>(_f: F, _length: usize, _one_past_end: Self) {}
}

macro_rules! impl_nonzero_dimension {
    ($Dimension:literal) => {
        unsafe impl IndexVecExt for IndexVec<$Dimension> {
            fn as_slice(&self) -> &[usize] {
                &self.0
            }
            fn as_mut_slice(&mut self) -> &mut [usize] {
                &mut self.0
            }
            fn try_for_each_index<E, F: FnMut(Self) -> Result<(), E>>(
                mut f: F,
                length: usize,
            ) -> Result<(), E> {
                for i in 0..length {
                    IndexVec::<{ $Dimension - 1 }>::try_for_each_index(
                        |rest| f(Self::combine(i, rest)),
                        length,
                    )?;
                }
                Ok(())
            }
            fn for_each_index<F: FnMut(Self)>(mut f: F, length: usize) {
                for i in 0..length {
                    IndexVec::<{ $Dimension - 1 }>::for_each_index(
                        |rest| f(Self::combine(i, rest)),
                        length,
                    );
                }
            }
            fn try_for_each_index_before<E, F: FnMut(Self) -> Result<(), E>>(
                mut f: F,
                length: usize,
                one_past_end: Self,
            ) -> Result<(), E> {
                for i in 0..one_past_end.first() {
                    IndexVec::<{ $Dimension - 1 }>::try_for_each_index(
                        |rest| f(Self::combine(i, rest)),
                        length,
                    )?;
                }
                IndexVec::<{ $Dimension - 1 }>::try_for_each_index_before(
                    |rest| f(Self::combine(one_past_end.first(), rest)),
                    length,
                    one_past_end.rest(),
                )
            }
            fn for_each_index_before<F: FnMut(Self)>(mut f: F, length: usize, one_past_end: Self) {
                for i in 0..one_past_end.first() {
                    IndexVec::<{ $Dimension - 1 }>::for_each_index(
                        |rest| f(Self::combine(i, rest)),
                        length,
                    );
                }
                IndexVec::<{ $Dimension - 1 }>::for_each_index_before(
                    |rest| f(Self::combine(one_past_end.first(), rest)),
                    length,
                    one_past_end.rest(),
                )
            }
        }

        unsafe impl IndexVecNonzeroDimension for IndexVec<$Dimension> {
            type PrevDimension = IndexVec<{ $Dimension - 1 }>;
            fn first(self) -> usize {
                self.0[0]
            }
            fn rest(self) -> Self::PrevDimension {
                let mut retval = [0; $Dimension - 1];
                for i in 1..$Dimension {
                    retval[i - 1] = self.0[i];
                }
                IndexVec(retval)
            }
            fn combine(first: usize, rest: Self::PrevDimension) -> Self {
                let mut retval = [0; $Dimension];
                retval[0] = first;
                for i in 1..$Dimension {
                    retval[i] = rest.0[i - 1];
                }
                Self(retval)
            }
        }
    };
}

macro_rules! impl_nonzero_dimensions {
    ([$($Dimension:literal),*]) => {
        $(
            impl_nonzero_dimension!($Dimension);
        )*
    };
}

impl_nonzero_dimensions!([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);

#[repr(transparent)]
pub struct IndexVec<const D: usize>(pub [usize; D]);

impl<const D: usize> Clone for IndexVec<D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<const D: usize> Copy for IndexVec<D> {}

impl<const D: usize> fmt::Debug for IndexVec<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(&self.0).finish()
    }
}

impl<const D: usize> Hash for IndexVec<D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<const D: usize> IndexVec<D> {
    pub fn try_map<E>(self, mut f: impl FnMut(usize) -> Result<usize, E>) -> Result<Self, E> {
        let mut retval = ArrayVec::<usize, D>::new();
        for &v in &self.0 {
            retval.push(f(v)?);
        }
        Ok(IndexVec(retval.into_inner().unwrap()))
    }
    pub fn map(self, mut f: impl FnMut(usize) -> usize) -> Self {
        self.try_map(|v| Ok::<_, ()>(f(v))).unwrap()
    }
}

impl<const D: usize> From<usize> for IndexVec<D> {
    fn from(v: usize) -> Self {
        IndexVec([v; D])
    }
}
