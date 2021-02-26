use crate::index_vec::{IndexVec, IndexVecExt, IndexVecForEach, IndexVecNonzeroDimension};
use core::{
    fmt,
    hash::{Hash, Hasher},
    mem::{self, MaybeUninit},
    ops::{Index, IndexMut},
    ptr,
};

/// # Safety
/// `Self::Repr` must be an array of `Self` or just `Self`.
pub unsafe trait ArrayRepr<const LENGTH: usize, const DIMENSION: usize>: Sized {
    type Repr;
    /// # Safety
    /// ptr must point to a valid instance of `Self::Repr`, it doesn't need to point to a mutable value
    #[inline(always)]
    unsafe fn index_ptr_checked(ptr: *mut Self::Repr, index: IndexVec<DIMENSION>) -> *mut Self {
        for &i in &index.0 {
            assert!(i < LENGTH);
        }
        Self::index_ptr_unchecked(ptr, index)
    }
    /// # Safety
    /// ptr must point to a valid instance of `Self::Repr` and index must be in-bounds, ptr doesn't need to point to a mutable value
    unsafe fn index_ptr_unchecked(ptr: *mut Self::Repr, index: IndexVec<DIMENSION>) -> *mut Self;
    fn debug_helper<DebugFn: Fn(&Self, &mut fmt::Formatter<'_>) -> fmt::Result>(
        value: &Self::Repr,
        debug_fn: &DebugFn,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result;
}

unsafe impl<T, const LENGTH: usize> ArrayRepr<LENGTH, 0> for T {
    type Repr = T;
    #[inline(always)]
    unsafe fn index_ptr_unchecked(ptr: *mut Self::Repr, _index: IndexVec<0>) -> *mut Self {
        ptr
    }
    fn debug_helper<DebugFn: Fn(&Self, &mut fmt::Formatter<'_>) -> fmt::Result>(
        value: &Self::Repr,
        debug_fn: &DebugFn,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        debug_fn(value, f)
    }
}

struct DebugWrapper<T>(T);

impl<T: Fn(&mut fmt::Formatter<'_>) -> fmt::Result> fmt::Debug for DebugWrapper<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (self.0)(f)
    }
}

macro_rules! impl_nonzero_dimension {
    ($DIMENSION:literal) => {
        unsafe impl<T, const LENGTH: usize> ArrayRepr<LENGTH, $DIMENSION> for T {
            type Repr = [<T as ArrayRepr<LENGTH, { $DIMENSION - 1 }>>::Repr; LENGTH];
            #[inline(always)]
            unsafe fn index_ptr_unchecked(
                ptr: *mut Self::Repr,
                index: IndexVec<$DIMENSION>,
            ) -> *mut Self {
                let ptr = ptr as *mut <T as ArrayRepr<LENGTH, { $DIMENSION - 1 }>>::Repr;
                <T as ArrayRepr<LENGTH, { $DIMENSION - 1 }>>::index_ptr_unchecked(
                    ptr.offset(index.first() as isize),
                    index.rest(),
                )
            }
            fn debug_helper<DebugFn: Fn(&Self, &mut fmt::Formatter<'_>) -> fmt::Result>(
                value: &Self::Repr,
                debug_fn: &DebugFn,
                f: &mut fmt::Formatter<'_>,
            ) -> fmt::Result {
                f.debug_list()
                    .entries(value.iter().map(|value| {
                        DebugWrapper(move |f: &mut fmt::Formatter<'_>| {
                            <T as ArrayRepr<LENGTH, { $DIMENSION - 1 }>>::debug_helper(
                                value, debug_fn, f,
                            )
                        })
                    }))
                    .finish()
            }
        }
    };
}

macro_rules! impl_nonzero_dimensions {
    ([$($DIMENSION:literal),*]) => {
        $(
            impl_nonzero_dimension!($DIMENSION);
        )*
    };
}

impl_nonzero_dimensions!([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);

#[repr(transparent)]
pub struct Array<T, const LENGTH: usize, const DIMENSION: usize>(pub T::Repr)
where
    T: ArrayRepr<LENGTH, DIMENSION>;

impl<T, const LENGTH: usize, const DIMENSION: usize> Copy for Array<T, LENGTH, DIMENSION>
where
    T: ArrayRepr<LENGTH, DIMENSION>,
    T: Copy,
    T::Repr: Copy,
    IndexVec<DIMENSION>: IndexVecExt,
{
}

impl<T, const LENGTH: usize, const DIMENSION: usize> Clone for Array<T, LENGTH, DIMENSION>
where
    T: ArrayRepr<LENGTH, DIMENSION>,
    T: Clone,
    IndexVec<DIMENSION>: IndexVecExt,
{
    #[inline(always)]
    fn clone(&self) -> Self {
        Array::build_array(
            #[inline(always)]
            |index| self[index].clone(),
        )
    }
    #[inline(always)]
    fn clone_from(&mut self, source: &Self) {
        IndexVec::for_each_index(
            #[inline(always)]
            |index| self[index].clone_from(&source[index]),
            LENGTH,
            ..,
        );
    }
}

impl<T, const LENGTH: usize, const DIMENSION: usize> fmt::Debug for Array<T, LENGTH, DIMENSION>
where
    T: ArrayRepr<LENGTH, DIMENSION>,
    T: fmt::Debug,
    IndexVec<DIMENSION>: IndexVecExt,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Array")
            .field(&DebugWrapper(|f: &mut fmt::Formatter<'_>| {
                T::debug_helper(&self.0, &|v, f| fmt::Debug::fmt(v, f), f)
            }))
            .finish()
    }
}

impl<T, const LENGTH: usize, const DIMENSION: usize> Default for Array<T, LENGTH, DIMENSION>
where
    T: ArrayRepr<LENGTH, DIMENSION>,
    T: Default,
    IndexVec<DIMENSION>: IndexVecExt,
{
    #[inline]
    fn default() -> Self {
        Array::build_array(
            #[inline(always)]
            |_| T::default(),
        )
    }
}

impl<T, Other, const LENGTH: usize, const DIMENSION: usize>
    PartialEq<Array<Other, LENGTH, DIMENSION>> for Array<T, LENGTH, DIMENSION>
where
    T: ArrayRepr<LENGTH, DIMENSION>,
    Other: ArrayRepr<LENGTH, DIMENSION>,
    T: PartialEq<Other>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    #[inline(always)]
    fn eq(&self, other: &Array<Other, LENGTH, DIMENSION>) -> bool {
        IndexVec::try_for_each_index(
            #[inline(always)]
            |index| {
                if self[index] == other[index] {
                    Ok(())
                } else {
                    Err(())
                }
            },
            LENGTH,
            ..,
        )
        .is_ok()
    }
}

impl<T, const LENGTH: usize, const DIMENSION: usize> Hash for Array<T, LENGTH, DIMENSION>
where
    T: ArrayRepr<LENGTH, DIMENSION>,
    T: Hash,
    IndexVec<DIMENSION>: IndexVecExt,
{
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        IndexVec::for_each_index(
            #[inline(always)]
            |index| self[index].hash(state),
            LENGTH,
            ..,
        )
    }
}

impl<T, const LENGTH: usize, const DIMENSION: usize> Eq for Array<T, LENGTH, DIMENSION>
where
    T: ArrayRepr<LENGTH, DIMENSION>,
    T: Eq,
    IndexVec<DIMENSION>: IndexVecExt,
{
}

impl<T, const LENGTH: usize, const DIMENSION: usize> Index<IndexVec<DIMENSION>>
    for Array<T, LENGTH, DIMENSION>
where
    T: ArrayRepr<LENGTH, DIMENSION>,
{
    type Output = T;
    #[inline(always)]
    fn index(&self, index: IndexVec<DIMENSION>) -> &Self::Output {
        unsafe {
            &*<T as ArrayRepr<LENGTH, DIMENSION>>::index_ptr_checked(
                &self.0 as *const T::Repr as *mut T::Repr,
                index,
            )
        }
    }
}

impl<T, const LENGTH: usize, const DIMENSION: usize> IndexMut<IndexVec<DIMENSION>>
    for Array<T, LENGTH, DIMENSION>
where
    T: ArrayRepr<LENGTH, DIMENSION>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: IndexVec<DIMENSION>) -> &mut Self::Output {
        unsafe { &mut *<T as ArrayRepr<LENGTH, DIMENSION>>::index_ptr_checked(&mut self.0, index) }
    }
}

unsafe impl<T: Send, const LENGTH: usize, const DIMENSION: usize> Send
    for Array<T, LENGTH, DIMENSION>
where
    T: ArrayRepr<LENGTH, DIMENSION>,
{
    // redundant implementation just to reduce constraints required in generic contexts
}

unsafe impl<T: Sync, const LENGTH: usize, const DIMENSION: usize> Sync
    for Array<T, LENGTH, DIMENSION>
where
    T: ArrayRepr<LENGTH, DIMENSION>,
{
    // redundant implementation just to reduce constraints required in generic contexts
}

struct CallOnDrop<T: FnMut()>(T);

impl<T: FnMut()> CallOnDrop<T> {
    fn cancel(self) -> T {
        unsafe {
            let retval = ptr::read(&self.0);
            mem::forget(self);
            retval
        }
    }
}

impl<T: FnMut()> Drop for CallOnDrop<T> {
    fn drop(&mut self) {
        self.0();
    }
}

impl<T, const LENGTH: usize, const DIMENSION: usize> Array<T, LENGTH, DIMENSION>
where
    T: ArrayRepr<LENGTH, DIMENSION>,
    IndexVec<DIMENSION>: IndexVecExt,
{
    #[inline(always)]
    pub fn try_build_array<E, F: FnMut(IndexVec<DIMENSION>) -> Result<T, E>>(
        mut f: F,
    ) -> Result<Self, E> {
        unsafe {
            let mut array = MaybeUninit::<T::Repr>::uninit();
            let array_ptr = array.as_mut_ptr();
            IndexVec::<DIMENSION>::try_for_each_index(
                #[inline(always)]
                |index| -> Result<(), E> {
                    let handle_failure = CallOnDrop(
                        #[cold]
                        || {
                            IndexVec::<DIMENSION>::for_each_index(
                                |index| {
                                    ptr::drop_in_place(T::index_ptr_unchecked(array_ptr, index))
                                },
                                LENGTH,
                                ..index,
                            )
                        },
                    );
                    let value = f(index)?;
                    let _ = handle_failure.cancel();
                    T::index_ptr_unchecked(array_ptr, index).write(value);
                    Ok(())
                },
                LENGTH,
                ..,
            )?;
            Ok(Self(array.assume_init()))
        }
    }
    #[inline(always)]
    pub fn build_array<F: FnMut(IndexVec<DIMENSION>) -> T>(mut f: F) -> Self {
        Self::try_build_array::<(), _>(
            #[inline(always)]
            |index| Ok(f(index)),
        )
        .unwrap()
    }
}
