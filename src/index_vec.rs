use core::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    iter::FusedIterator,
    ops::{
        Add, Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo,
        RangeToInclusive,
    },
};

pub unsafe trait IndexVecForEach<RangeT: RangeBounds<Self>>: Sized {
    fn try_for_each_index<E, F: FnMut(Self) -> Result<(), E>>(
        f: F,
        length: usize,
        range: RangeT,
    ) -> Result<(), E>;
    fn for_each_index<F: FnMut(Self)>(f: F, length: usize, range: RangeT);
}

pub unsafe trait IndexVecExt:
    From<usize>
    + Ord
    + IndexVecForEach<RangeFull>
    + IndexVecForEach<RangeFrom<Self>>
    + IndexVecForEach<RangeTo<Self>>
    + IndexVecForEach<Range<Self>>
    + IndexVecForEach<RangeToInclusive<Self>>
    + IndexVecForEach<RangeInclusive<Self>>
    + IndexVecForEach<(Bound<Self>, Bound<Self>)>
    + for<'a> IndexVecForEach<(Bound<&'a Self>, Bound<&'a Self>)>
{
}

pub unsafe trait IndexVecNonzeroDimension: IndexVecExt {
    type PrevDimension: IndexVecExt;
    fn first(self) -> usize;
    fn rest(self) -> Self::PrevDimension;
    fn combine(first: usize, rest: Self::PrevDimension) -> Self;
}

unsafe impl<RangeT: RangeBounds<Self>> IndexVecForEach<RangeT> for IndexVec<0> {
    fn try_for_each_index<E, F: FnMut(Self) -> Result<(), E>>(
        mut f: F,
        _length: usize,
        range: RangeT,
    ) -> Result<(), E> {
        match (range.start_bound(), range.end_bound()) {
            (_, Bound::Excluded(_)) | (Bound::Excluded(_), _) => Ok(()),
            _ => f(IndexVec([])),
        }
    }
    fn for_each_index<F: FnMut(Self)>(mut f: F, _length: usize, range: RangeT) {
        match (range.start_bound(), range.end_bound()) {
            (_, Bound::Excluded(_)) | (Bound::Excluded(_), _) => {}
            _ => f(IndexVec([])),
        }
    }
}

impl<const D: usize> PartialEq for IndexVec<D> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const D: usize> Eq for IndexVec<D> {}

impl<const D: usize> PartialOrd for IndexVec<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const D: usize> Ord for IndexVec<D> {
    fn cmp(&self, other: &Self) -> Ordering {
        for i in 0..D {
            match self.0[i].cmp(&other.0[i]) {
                Ordering::Equal => {}
                retval => return retval,
            }
        }
        Ordering::Equal
    }
}

unsafe impl IndexVecExt for IndexVec<0> {}

enum ForEachIndexHelper<const D: usize>
where
    IndexVec<D>: IndexVecNonzeroDimension,
{
    Multi {
        length: usize,
        first: Option<(
            usize,
            Bound<<IndexVec<D> as IndexVecNonzeroDimension>::PrevDimension>,
        )>,
        middle: Range<usize>,
        last: Option<(
            usize,
            Bound<<IndexVec<D> as IndexVecNonzeroDimension>::PrevDimension>,
        )>,
    },
    Single {
        length: usize,
        first: usize,
        range: (
            Bound<<IndexVec<D> as IndexVecNonzeroDimension>::PrevDimension>,
            Bound<<IndexVec<D> as IndexVecNonzeroDimension>::PrevDimension>,
        ),
    },
    None,
}

impl<const D: usize> ForEachIndexHelper<D>
where
    IndexVec<D>: IndexVecNonzeroDimension,
{
    #[inline(always)]
    fn new<RangeT: RangeBounds<IndexVec<D>>>(length: usize, range: RangeT) -> Self {
        match (range.start_bound(), range.end_bound()) {
            (Bound::Included(&start), Bound::Included(&end)) => {
                if start.first() >= length {
                    Self::None
                } else if start.first() > end.first() {
                    Self::None
                } else if start.first() == end.first() {
                    Self::Single {
                        length,
                        first: start.first(),
                        range: (Bound::Included(start.rest()), Bound::Included(end.rest())),
                    }
                } else if end.first() >= length {
                    Self::Multi {
                        length,
                        first: Some((start.first(), Bound::Included(start.rest()))),
                        middle: start.first() + 1..length,
                        last: None,
                    }
                } else {
                    Self::Multi {
                        length,
                        first: Some((start.first(), Bound::Included(start.rest()))),
                        middle: start.first() + 1..end.first(),
                        last: Some((end.first(), Bound::Included(end.rest()))),
                    }
                }
            }
            (Bound::Included(&start), Bound::Excluded(&end)) => {
                if start.first() >= length {
                    Self::None
                } else if start.first() > end.first() {
                    Self::None
                } else if start.first() == end.first() {
                    Self::Single {
                        length,
                        first: start.first(),
                        range: (Bound::Included(start.rest()), Bound::Excluded(end.rest())),
                    }
                } else if end.first() >= length {
                    Self::Multi {
                        length,
                        first: Some((start.first(), Bound::Included(start.rest()))),
                        middle: start.first() + 1..length,
                        last: None,
                    }
                } else {
                    Self::Multi {
                        length,
                        first: Some((start.first(), Bound::Included(start.rest()))),
                        middle: start.first() + 1..end.first(),
                        last: Some((end.first(), Bound::Excluded(end.rest()))),
                    }
                }
            }
            (Bound::Included(&start), Bound::Unbounded) => {
                if start.first() >= length {
                    Self::None
                } else {
                    Self::Multi {
                        length,
                        first: Some((start.first(), Bound::Included(start.rest()))),
                        middle: start.first() + 1..length,
                        last: None,
                    }
                }
            }
            (Bound::Excluded(&start), Bound::Included(&end)) => {
                if start.first() >= length {
                    Self::None
                } else if start.first() > end.first() {
                    Self::None
                } else if start.first() == end.first() {
                    Self::Single {
                        length,
                        first: start.first(),
                        range: (Bound::Excluded(start.rest()), Bound::Included(end.rest())),
                    }
                } else if end.first() >= length {
                    Self::Multi {
                        length,
                        first: Some((start.first(), Bound::Excluded(start.rest()))),
                        middle: start.first() + 1..length,
                        last: None,
                    }
                } else {
                    Self::Multi {
                        length,
                        first: Some((start.first(), Bound::Excluded(start.rest()))),
                        middle: start.first() + 1..end.first(),
                        last: Some((end.first(), Bound::Included(end.rest()))),
                    }
                }
            }
            (Bound::Excluded(&start), Bound::Excluded(&end)) => {
                if start.first() >= length {
                    Self::None
                } else if start.first() > end.first() {
                    Self::None
                } else if start.first() == end.first() {
                    Self::Single {
                        length,
                        first: start.first(),
                        range: (Bound::Excluded(start.rest()), Bound::Excluded(end.rest())),
                    }
                } else if end.first() >= length {
                    Self::Multi {
                        length,
                        first: Some((start.first(), Bound::Excluded(start.rest()))),
                        middle: start.first() + 1..length,
                        last: None,
                    }
                } else {
                    Self::Multi {
                        length,
                        first: Some((start.first(), Bound::Excluded(start.rest()))),
                        middle: start.first() + 1..end.first(),
                        last: Some((end.first(), Bound::Excluded(end.rest()))),
                    }
                }
            }
            (Bound::Excluded(&start), Bound::Unbounded) => {
                if start.first() >= length {
                    Self::None
                } else {
                    Self::Multi {
                        length,
                        first: Some((start.first(), Bound::Excluded(start.rest()))),
                        middle: start.first() + 1..length,
                        last: None,
                    }
                }
            }
            (Bound::Unbounded, Bound::Included(&end)) => {
                if end.first() >= length {
                    Self::Multi {
                        length,
                        first: None,
                        middle: 0..length,
                        last: None,
                    }
                } else {
                    Self::Multi {
                        length,
                        first: None,
                        middle: 0..end.first(),
                        last: Some((end.first(), Bound::Included(end.rest()))),
                    }
                }
            }
            (Bound::Unbounded, Bound::Excluded(&end)) => {
                if end.first() >= length {
                    Self::Multi {
                        length,
                        first: None,
                        middle: 0..length,
                        last: None,
                    }
                } else {
                    Self::Multi {
                        length,
                        first: None,
                        middle: 0..end.first(),
                        last: Some((end.first(), Bound::Excluded(end.rest()))),
                    }
                }
            }
            (Bound::Unbounded, Bound::Unbounded) => Self::Multi {
                length,
                first: None,
                middle: 0..length,
                last: None,
            },
        }
    }
    #[inline(always)]
    fn for_each<F: FnMut(IndexVec<D>)>(self, mut f: F) {
        match self {
            ForEachIndexHelper::Multi {
                length,
                first,
                middle,
                last,
            } => {
                if let Some((first_index, first_bound)) = first {
                    <IndexVec<D> as IndexVecNonzeroDimension>::PrevDimension::for_each_index(
                        #[inline(always)]
                        |rest| f(IndexVec::<D>::combine(first_index, rest)),
                        length,
                        (first_bound, Bound::Unbounded),
                    );
                }
                for i in middle {
                    <IndexVec<D> as IndexVecNonzeroDimension>::PrevDimension::for_each_index(
                        #[inline(always)]
                        |rest| f(IndexVec::<D>::combine(i, rest)),
                        length,
                        ..,
                    );
                }
                if let Some((last_index, last_bound)) = last {
                    <IndexVec<D> as IndexVecNonzeroDimension>::PrevDimension::for_each_index(
                        #[inline(always)]
                        |rest| f(IndexVec::<D>::combine(last_index, rest)),
                        length,
                        (Bound::Unbounded, last_bound),
                    );
                }
            }
            ForEachIndexHelper::Single {
                length,
                first,
                range,
            } => {
                <IndexVec<D> as IndexVecNonzeroDimension>::PrevDimension::for_each_index(
                    #[inline(always)]
                    |rest| f(IndexVec::<D>::combine(first, rest)),
                    length,
                    range,
                );
            }
            ForEachIndexHelper::None => {}
        }
    }
    #[inline(always)]
    fn try_for_each<E, F: FnMut(IndexVec<D>) -> Result<(), E>>(self, mut f: F) -> Result<(), E> {
        match self {
            ForEachIndexHelper::Multi {
                length,
                first,
                middle,
                last,
            } => {
                if let Some((first_index, first_bound)) = first {
                    <IndexVec<D> as IndexVecNonzeroDimension>::PrevDimension::try_for_each_index(
                        #[inline(always)]
                        |rest| f(IndexVec::<D>::combine(first_index, rest)),
                        length,
                        (first_bound, Bound::Unbounded),
                    )?;
                }
                for i in middle {
                    <IndexVec<D> as IndexVecNonzeroDimension>::PrevDimension::try_for_each_index(
                        #[inline(always)]
                        |rest| f(IndexVec::<D>::combine(i, rest)),
                        length,
                        ..,
                    )?;
                }
                if let Some((last_index, last_bound)) = last {
                    <IndexVec<D> as IndexVecNonzeroDimension>::PrevDimension::try_for_each_index(
                        #[inline(always)]
                        |rest| f(IndexVec::<D>::combine(last_index, rest)),
                        length,
                        (Bound::Unbounded, last_bound),
                    )?;
                }
                Ok(())
            }
            ForEachIndexHelper::Single {
                length,
                first,
                range,
            } => <IndexVec<D> as IndexVecNonzeroDimension>::PrevDimension::try_for_each_index(
                #[inline(always)]
                |rest| f(IndexVec::<D>::combine(first, rest)),
                length,
                range,
            ),
            ForEachIndexHelper::None => Ok(()),
        }
    }
}

impl IndexVec<1> {
    #[inline(always)]
    fn for_each_index_helper_1<RangeT: RangeBounds<Self>>(
        length: usize,
        range: RangeT,
    ) -> Range<usize> {
        let start = match range.start_bound() {
            Bound::Included(&IndexVec([start])) => {
                if start < length {
                    start
                } else {
                    length
                }
            }
            Bound::Excluded(&IndexVec([start])) => {
                if start < length {
                    start + 1
                } else {
                    length
                }
            }
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&IndexVec([end])) => {
                if end < length {
                    end + 1
                } else {
                    length
                }
            }
            Bound::Excluded(&IndexVec([end])) => {
                if end < length {
                    end
                } else {
                    length
                }
            }
            Bound::Unbounded => length,
        };
        start..end
    }
}

unsafe impl<RangeT: RangeBounds<Self>> IndexVecForEach<RangeT> for IndexVec<1> {
    #[inline(always)]
    fn try_for_each_index<E, F: FnMut(Self) -> Result<(), E>>(
        mut f: F,
        length: usize,
        range: RangeT,
    ) -> Result<(), E> {
        for i in Self::for_each_index_helper_1(length, range) {
            f(IndexVec([i]))?;
        }
        Ok(())
    }
    #[inline(always)]
    fn for_each_index<F: FnMut(Self)>(mut f: F, length: usize, range: RangeT) {
        for i in Self::for_each_index_helper_1(length, range) {
            f(IndexVec([i]));
        }
    }
}

unsafe impl IndexVecExt for IndexVec<1> {}

unsafe impl IndexVecNonzeroDimension for IndexVec<1> {
    type PrevDimension = IndexVec<0>;
    #[inline(always)]
    fn first(self) -> usize {
        self.0[0]
    }
    #[inline(always)]
    fn rest(self) -> Self::PrevDimension {
        IndexVec([])
    }
    #[inline(always)]
    fn combine(first: usize, _rest: Self::PrevDimension) -> Self {
        Self([first])
    }
}

macro_rules! impl_dimension_more_than_1 {
    ($Dimension:literal) => {
        unsafe impl<RangeT: RangeBounds<Self>> IndexVecForEach<RangeT> for IndexVec<$Dimension> {
            #[inline(always)]
            fn try_for_each_index<E, F: FnMut(Self) -> Result<(), E>>(
                f: F,
                length: usize,
                range: RangeT,
            ) -> Result<(), E> {
                ForEachIndexHelper::<$Dimension>::new(length, range).try_for_each(f)
            }
            #[inline(always)]
            fn for_each_index<F: FnMut(Self)>(f: F, length: usize, range: RangeT) {
                ForEachIndexHelper::<$Dimension>::new(length, range).for_each(f)
            }
        }

        unsafe impl IndexVecExt for IndexVec<$Dimension> {}

        unsafe impl IndexVecNonzeroDimension for IndexVec<$Dimension> {
            type PrevDimension = IndexVec<{ $Dimension - 1 }>;
            #[inline(always)]
            fn first(self) -> usize {
                self.0[0]
            }
            #[inline(always)]
            fn rest(self) -> Self::PrevDimension {
                let mut retval = [0; $Dimension - 1];
                for i in 1..$Dimension {
                    retval[i - 1] = self.0[i];
                }
                IndexVec(retval)
            }
            #[inline(always)]
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

macro_rules! impl_dimensions_more_than_1 {
    ([$($Dimension:literal),*]) => {
        $(
            impl_dimension_more_than_1!($Dimension);
        )*
    };
}

impl_dimensions_more_than_1!([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);

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
    #[inline(always)]
    pub fn try_map<E>(mut self, mut f: impl FnMut(usize) -> Result<usize, E>) -> Result<Self, E> {
        for v in &mut self.0 {
            *v = f(*v)?;
        }
        Ok(self)
    }
    #[inline(always)]
    pub fn map(mut self, mut f: impl FnMut(usize) -> usize) -> Self {
        for v in &mut self.0 {
            *v = f(*v);
        }
        self
    }
    const fn linear_index_factors_helper(length: usize) -> (Self, usize) {
        let mut retval = [0; D];
        let mut factor = 1usize;
        let mut i = D;
        while i > 0 {
            i -= 1;
            retval[i] = factor;
            factor *= length;
        }
        (IndexVec(retval), factor)
    }
    pub const fn linear_index_factors(length: usize) -> Self {
        Self::linear_index_factors_helper(length).0
    }
    pub const fn linear_index_len(length: usize) -> usize {
        Self::linear_index_factors_helper(length).1
    }
}

#[derive(Clone, Debug)]
pub struct Indexes<const ARRAY_LENGTH: usize, const D: usize>(Range<usize>);

impl<const ARRAY_LENGTH: usize, const D: usize> Indexes<ARRAY_LENGTH, D> {
    pub const LEN: usize = IndexVec::<D>::linear_index_len(ARRAY_LENGTH);
    pub const LINEAR_INDEX_FACTORS: IndexVec<D> = IndexVec::<D>::linear_index_factors(ARRAY_LENGTH);
    pub const fn from_linear_index_range(mut linear_index_range: Range<usize>) -> Self {
        if linear_index_range.end > Self::LEN {
            linear_index_range.end = Self::LEN;
        }
        Self(linear_index_range)
    }
    pub const fn new() -> Self {
        Self(0..Self::LEN)
    }
    #[inline(always)]
    pub const fn item_for_linear_index(linear_index: usize) -> IndexVec<D> {
        let mut retval = Self::LINEAR_INDEX_FACTORS;
        let mut i = 0;
        while i < D {
            retval.0[i] = (linear_index / retval.0[i]) % ARRAY_LENGTH;
            i += 1;
        }
        retval
    }
    pub const fn linear_index_range(&self) -> Range<usize> {
        Range { ..self.0 }
    }
}

impl<const ARRAY_LENGTH: usize, const D: usize> Default for Indexes<ARRAY_LENGTH, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const ARRAY_LENGTH: usize, const D: usize> Iterator for Indexes<ARRAY_LENGTH, D>
where
    IndexVec<D>: IndexVecExt,
{
    type Item = IndexVec<D>;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        Some(Self::item_for_linear_index(self.0.next()?))
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
    #[inline(always)]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        Some(Self::item_for_linear_index(self.0.nth(n)?))
    }
    #[inline(always)]
    fn for_each<F>(self, f: F)
    where
        Self: Sized,
        F: FnMut(Self::Item),
    {
        if self.0.start == 0 {
            if self.0.end >= Self::LEN {
                IndexVec::<D>::for_each_index(f, ARRAY_LENGTH, ..);
            } else {
                IndexVec::<D>::for_each_index(
                    f,
                    ARRAY_LENGTH,
                    ..Self::item_for_linear_index(self.0.end),
                );
            }
        } else {
            if self.0.end >= Self::LEN {
                IndexVec::<D>::for_each_index(
                    f,
                    ARRAY_LENGTH,
                    Self::item_for_linear_index(self.0.start)..,
                );
            } else {
                IndexVec::<D>::for_each_index(
                    f,
                    ARRAY_LENGTH,
                    Self::item_for_linear_index(self.0.start)
                        ..Self::item_for_linear_index(self.0.end),
                );
            }
        }
    }
}

impl<const ARRAY_LENGTH: usize, const D: usize> ExactSizeIterator for Indexes<ARRAY_LENGTH, D>
where
    IndexVec<D>: IndexVecExt,
{
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<const ARRAY_LENGTH: usize, const D: usize> FusedIterator for Indexes<ARRAY_LENGTH, D> where
    IndexVec<D>: IndexVecExt
{
}

impl<const ARRAY_LENGTH: usize, const D: usize> DoubleEndedIterator for Indexes<ARRAY_LENGTH, D>
where
    IndexVec<D>: IndexVecExt,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(Self::item_for_linear_index(self.0.next_back()?))
    }
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        Some(Self::item_for_linear_index(self.0.nth_back(n)?))
    }
}

impl<const D: usize> Add for IndexVec<D> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        for (l, r) in self.0.iter_mut().zip(&rhs.0) {
            *l += *r;
        }
        self
    }
}

impl<const D: usize> From<usize> for IndexVec<D> {
    fn from(v: usize) -> Self {
        IndexVec([v; D])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::{dbg, vec::Vec};

    fn do_for_each_index_test<const ARRAY_LENGTH: usize, const D: usize>()
    where
        IndexVec<D>: IndexVecExt,
    {
        let mut expected = Vec::new();
        for v in Indexes::<ARRAY_LENGTH, D>::new() {
            // don't use collect to make sure we get the right results even if for_each is broken
            expected.push(v);
        }
        let mut produced = Vec::new();
        for start in 0..expected.len() {
            for end in 0..expected.len() {
                for &(start_bound, start_index) in &[
                    (Bound::Excluded(expected[start]), start + 1),
                    (Bound::Included(expected[start]), start),
                    (Bound::Unbounded, 0),
                ] {
                    for &(end_bound, end_index) in &[
                        (Bound::Excluded(expected[end]), end),
                        (Bound::Included(expected[end]), end + 1),
                        (Bound::Unbounded, expected.len()),
                    ] {
                        let expected = expected.get(start_index..end_index).unwrap_or(&[]);
                        produced.clear();
                        IndexVec::<D>::for_each_index(
                            |v| produced.push(v),
                            ARRAY_LENGTH,
                            dbg!((start_bound, end_bound)),
                        );
                        assert_eq!(produced, expected);
                        produced.clear();
                        IndexVec::<D>::try_for_each_index::<(), _>(
                            |v| {
                                produced.push(v);
                                Ok(())
                            },
                            ARRAY_LENGTH,
                            dbg!((start_bound, end_bound)),
                        )
                        .unwrap();
                        assert_eq!(produced, expected);
                        for stop_at in 0..expected.len() {
                            produced.clear();
                            let mut stopped = false;
                            IndexVec::<D>::try_for_each_index::<(), _>(
                                |v| {
                                    assert!(!stopped);
                                    if produced.len() == stop_at {
                                        stopped = true;
                                        return Err(());
                                    }
                                    produced.push(v);
                                    Ok(())
                                },
                                ARRAY_LENGTH,
                                dbg!((start_bound, end_bound)),
                            )
                            .unwrap_err();
                            assert_eq!(produced, &expected[..stop_at]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn for_each_index_test_0d() {
        do_for_each_index_test::<0, 0>();
        do_for_each_index_test::<0, 1>();
        do_for_each_index_test::<0, 2>();
        do_for_each_index_test::<0, 3>();
        do_for_each_index_test::<0, 4>();
    }

    #[test]
    fn for_each_index_test_1d() {
        do_for_each_index_test::<1, 0>();
        do_for_each_index_test::<1, 1>();
        do_for_each_index_test::<1, 2>();
        do_for_each_index_test::<1, 3>();
        do_for_each_index_test::<1, 4>();
    }

    #[test]
    fn for_each_index_test_2d() {
        do_for_each_index_test::<2, 0>();
        do_for_each_index_test::<2, 1>();
        do_for_each_index_test::<2, 2>();
        do_for_each_index_test::<2, 3>();
        do_for_each_index_test::<2, 4>();
    }

    #[test]
    fn for_each_index_test_3d_0() {
        do_for_each_index_test::<3, 0>();
    }

    #[test]
    fn for_each_index_test_3d_1() {
        do_for_each_index_test::<3, 1>();
    }

    #[test]
    fn for_each_index_test_3d_2() {
        do_for_each_index_test::<3, 2>();
    }

    #[test]
    fn for_each_index_test_3d_3() {
        do_for_each_index_test::<3, 3>();
    }
}
