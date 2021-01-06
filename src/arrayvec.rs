use std::{
    fmt,
    iter::FusedIterator,
    mem::{self, MaybeUninit},
    ops::{Deref, DerefMut, Range},
    ptr, slice,
};

use mem::ManuallyDrop;

pub struct ArrayVec<T, const N: usize> {
    data: MaybeUninit<[T; N]>,
    len: usize,
}

impl<T, const N: usize> Default for ArrayVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> ArrayVec<T, N> {
    pub const fn new() -> Self {
        Self {
            data: MaybeUninit::uninit(),
            len: 0,
        }
    }
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr() as *mut T
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr() as *const T
    }
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }
    pub const fn len(&self) -> usize {
        self.len
    }
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub const fn is_full(&self) -> bool {
        self.len == N
    }
    pub const fn capacity(&self) -> usize {
        N
    }
    pub fn truncate(&mut self, new_len: usize) {
        unsafe {
            if new_len < self.len {
                let old_len = self.len;
                let ptr: *mut [_] = &mut self[new_len..old_len];
                self.len = new_len;
                ptr::drop_in_place(ptr);
            }
        }
    }
    pub fn clear(&mut self) {
        self.truncate(0)
    }
    pub unsafe fn push_unchecked(&mut self, v: T) {
        debug_assert!(!self.is_full());
        self.as_mut_ptr().offset(self.len as isize).write(v);
        self.len += 1;
    }
    pub fn try_push(&mut self, v: T) -> Result<(), T> {
        if self.is_full() {
            Err(v)
        } else {
            unsafe {
                self.push_unchecked(v);
                Ok(())
            }
        }
    }
    pub fn push(&mut self, v: T) {
        unsafe {
            assert!(!self.is_full());
            self.push_unchecked(v);
        }
    }
    pub fn into_inner(self) -> Result<[T; N], Self> {
        if self.is_full() {
            let mut this = ManuallyDrop::new(self);
            Ok(unsafe { this.data.as_mut_ptr().read() })
        } else {
            Err(self)
        }
    }
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= N);
        self.len = new_len;
    }
}

impl<T, const N: usize> From<[T; N]> for ArrayVec<T, N> {
    fn from(data: [T; N]) -> Self {
        ArrayVec {
            data: MaybeUninit::new(data),
            len: N,
        }
    }
}

impl<T, const N: usize> Deref for ArrayVec<T, N> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, const N: usize> DerefMut for ArrayVec<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T, const N: usize> Drop for ArrayVec<T, N> {
    fn drop(&mut self) {
        self.clear()
    }
}

impl<T: Clone, const N: usize> Clone for ArrayVec<T, N> {
    fn clone(&self) -> Self {
        let mut retval = Self::new();
        for v in self {
            unsafe { retval.push_unchecked(v.clone()) }
        }
        retval
    }
    fn clone_from(&mut self, source: &Self) {
        self.clear();
        for v in source {
            unsafe { self.push_unchecked(v.clone()) }
        }
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a ArrayVec<T, N> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut ArrayVec<T, N> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, const N: usize> IntoIterator for ArrayVec<T, N> {
    type Item = T;
    type IntoIter = IntoIter<T, N>;
    fn into_iter(self) -> Self::IntoIter {
        let mut this = ManuallyDrop::new(self);
        unsafe {
            IntoIter {
                data: (this.data.as_mut_ptr() as *mut [MaybeUninit<T>; N]).read(),
                valid_range: 0..this.len,
            }
        }
    }
}

pub struct IntoIter<T, const N: usize> {
    data: [MaybeUninit<T>; N],
    valid_range: Range<usize>,
}

impl<T, const N: usize> IntoIter<T, N> {
    pub fn as_slice(&self) -> &[T] {
        unsafe { mem::transmute(&self.data[self.valid_range.clone()]) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { mem::transmute(&mut self.data[self.valid_range.clone()]) }
    }
    pub fn into_array_vec(self) -> ArrayVec<T, N> {
        let mut retval = ArrayVec::<T, N>::new();
        let mut this = ManuallyDrop::new(self);
        let this = &mut *this;
        unsafe {
            ptr::copy_nonoverlapping(
                this.data[this.valid_range.clone()].as_mut_ptr() as *mut T,
                retval.as_mut_ptr(),
                this.valid_range.len(),
            );
            retval.set_len(this.valid_range.len());
        }
        retval
    }
}

impl<T, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        unsafe { Some(self.data[self.valid_range.next()?].as_mut_ptr().read()) }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.valid_range.size_hint()
    }
}

impl<T, const N: usize> DoubleEndedIterator for IntoIter<T, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        unsafe { Some(self.data[self.valid_range.next_back()?].as_mut_ptr().read()) }
    }
}

impl<T, const N: usize> FusedIterator for IntoIter<T, N> {}

impl<T, const N: usize> ExactSizeIterator for IntoIter<T, N> {}

impl<T, const N: usize> Drop for IntoIter<T, N> {
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(self.as_mut_slice());
        }
    }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for ArrayVec<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self).finish()
    }
}
