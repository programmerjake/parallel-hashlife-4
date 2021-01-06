use crate::index_vec::{IndexVec, IndexVecExt, IndexVecNonzeroDimension};
use mem::MaybeUninit;
use std::{
    fmt, mem,
    ops::{Index, IndexMut},
    ptr,
};

pub unsafe trait ArrayRepr<const LENGTH: usize, const DIMENSION: usize>: Sized {
    type Repr;
    unsafe fn index_ptr_checked(ptr: *mut Self::Repr, index: IndexVec<DIMENSION>) -> *mut Self {
        for &i in &index.0 {
            assert!(i < LENGTH);
        }
        Self::index_ptr_unchecked(ptr, index)
    }
    unsafe fn index_ptr_unchecked(ptr: *mut Self::Repr, index: IndexVec<DIMENSION>) -> *mut Self;
}

unsafe impl<T, const LENGTH: usize> ArrayRepr<LENGTH, 0> for T {
    type Repr = T;
    unsafe fn index_ptr_unchecked(ptr: *mut Self::Repr, _index: IndexVec<0>) -> *mut Self {
        ptr
    }
}

macro_rules! impl_nonzero_dimension {
    ($DIMENSION:literal) => {
        unsafe impl<T, const LENGTH: usize> ArrayRepr<LENGTH, $DIMENSION> for T {
            type Repr = [<T as ArrayRepr<LENGTH, { $DIMENSION - 1 }>>::Repr; LENGTH];
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

pub struct Array<T, const LENGTH: usize, const DIMENSION: usize>(T::Repr)
where
    T: ArrayRepr<LENGTH, DIMENSION>;

impl<T, const LENGTH: usize, const DIMENSION: usize> Copy for Array<T, LENGTH, DIMENSION>
where
    T: ArrayRepr<LENGTH, DIMENSION>,
    T: Copy,
    T::Repr: Copy,
{
}

impl<T, const LENGTH: usize, const DIMENSION: usize> Clone for Array<T, LENGTH, DIMENSION>
where
    T: ArrayRepr<LENGTH, DIMENSION>,
    T: Clone,
    T::Repr: Clone,
{
    fn clone(&self) -> Self {
        Array(self.0.clone())
    }
    fn clone_from(&mut self, source: &Self) {
        self.0.clone_from(&source.0);
    }
}

impl<T, const LENGTH: usize, const DIMENSION: usize> fmt::Debug for Array<T, LENGTH, DIMENSION>
where
    T: ArrayRepr<LENGTH, DIMENSION>,
    T: fmt::Debug,
    T::Repr: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Array").field(&self.0).finish()
    }
}

impl<T, const LENGTH: usize, const DIMENSION: usize> Default for Array<T, LENGTH, DIMENSION>
where
    T: ArrayRepr<LENGTH, DIMENSION>,
    T: Default,
    T::Repr: Default,
{
    fn default() -> Self {
        Array(T::Repr::default())
    }
}

impl<T, const LENGTH: usize, const DIMENSION: usize> Index<IndexVec<DIMENSION>>
    for Array<T, LENGTH, DIMENSION>
where
    T: ArrayRepr<LENGTH, DIMENSION>,
{
    type Output = T;
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
    fn index_mut(&mut self, index: IndexVec<DIMENSION>) -> &mut Self::Output {
        unsafe { &mut *<T as ArrayRepr<LENGTH, DIMENSION>>::index_ptr_checked(&mut self.0, index) }
    }
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
    pub fn try_build_array<E, F: FnMut(IndexVec<DIMENSION>) -> Result<T, E>>(
        mut f: F,
    ) -> Result<Self, E> {
        unsafe {
            let mut array = MaybeUninit::<T::Repr>::uninit();
            let array_ptr = array.as_mut_ptr();
            IndexVec::<DIMENSION>::try_for_each_index(
                |index| -> Result<(), E> {
                    let handle_failure = CallOnDrop(|| {
                        IndexVec::<DIMENSION>::for_each_index_before(
                            |index| ptr::drop_in_place(T::index_ptr_unchecked(array_ptr, index)),
                            LENGTH,
                            index,
                        )
                    });
                    let value = f(index)?;
                    let _ = handle_failure.cancel();
                    T::index_ptr_unchecked(array_ptr, index).write(value);
                    Ok(())
                },
                LENGTH,
            )?;
            Ok(Self(array.assume_init()))
        }
    }
    pub fn build_array<F: FnMut(IndexVec<DIMENSION>) -> T>(mut f: F) -> Self {
        Self::try_build_array::<(), _>(|index| Ok(f(index))).unwrap()
    }
}
