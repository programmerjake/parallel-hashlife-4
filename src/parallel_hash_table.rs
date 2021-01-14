use alloc::{boxed::Box, vec, vec::Vec};
use core::{
    cell::UnsafeCell,
    fmt,
    hint::spin_loop,
    iter,
    iter::FusedIterator,
    marker::PhantomData,
    mem,
    mem::MaybeUninit,
    num::NonZeroUsize,
    ops::{Deref, Range},
    ptr::NonNull,
    slice,
    sync::atomic::{AtomicU32, Ordering},
};

pub trait WaitWake {
    /// Does the following steps:
    /// 1. Lock the mutex associated with key `key`.
    /// 2. If `should_cancel()` returns `true`, unlock the mutex and return without blocking.
    /// 3. Atomically unlock the mutex and wait for wake-ups associated with key `key`.
    /// It is valid for `wait` to stop waiting even without any associated wake-ups.
    /// # Safety
    /// `key` must be a memory address controlled by the caller.
    /// `should_cancel` must not call `wait` or `wake_all` and must not panic.
    unsafe fn wait<SC: FnOnce() -> bool>(&self, key: NonZeroUsize, should_cancel: SC);
    /// wake all waiting threads that have the key `key`
    /// # Safety
    /// `key` must be a memory address controlled by the caller
    unsafe fn wake_all(&self, key: NonZeroUsize);
}

impl<T: WaitWake> WaitWake for &'_ T {
    unsafe fn wait<SC: FnOnce() -> bool>(&self, key: NonZeroUsize, should_cancel: SC) {
        (**self).wait(key, should_cancel);
    }
    unsafe fn wake_all(&self, key: NonZeroUsize) {
        (**self).wake_all(key);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
struct State(u32);

impl State {
    const EMPTY: Self = Self(0);
    const LOCKED: Self = Self(1);
    const LOCKED_WAITERS: Self = Self(2);
    const FIRST_FULL: Self = Self(3);
    const fn is_empty(self) -> bool {
        self.0 == Self::EMPTY.0
    }
    const fn is_full(self) -> bool {
        self.0 >= Self::FIRST_FULL.0
    }
    const fn make_full(hash: u64) -> Self {
        let retval = Self((hash >> 32) as u32);
        if retval.is_full() {
            return retval;
        }
        let retval = Self(hash as u32);
        if retval.is_full() {
            return retval;
        }
        return Self::FIRST_FULL;
    }
}

#[derive(Debug)]
#[repr(transparent)]
struct AtomicState(AtomicU32);

impl AtomicState {
    const fn new(s: State) -> Self {
        AtomicState(AtomicU32::new(s.0))
    }
    fn load(&self, order: Ordering) -> State {
        State(self.0.load(order))
    }
    fn into_inner(self) -> State {
        State(self.0.into_inner())
    }
    fn get_mut(&mut self) -> &mut State {
        unsafe { mem::transmute(self.0.get_mut()) }
    }
    fn compare_exchange_weak(
        &self,
        current: State,
        new: State,
        success: Ordering,
        failure: Ordering,
    ) -> Result<State, State> {
        match self
            .0
            .compare_exchange_weak(current.0, new.0, success, failure)
        {
            Ok(v) => Ok(State(v)),
            Err(v) => Err(State(v)),
        }
    }
    fn swap(&self, new: State, order: Ordering) -> State {
        State(self.0.swap(new.0, order))
    }
    unsafe fn write_start<W: WaitWake>(&self, wait_waker: &W) -> Result<(), State> {
        let mut spin_count = 0;
        let mut state = self.load(Ordering::Acquire);
        loop {
            if state.is_full() {
                return Err(state);
            }
            if !state.is_empty() && spin_count < 32 {
                spin_count += 1;
                state = self.load(Ordering::Acquire);
                spin_loop();
                continue;
            }
            if let Err(v) = self.compare_exchange_weak(
                state,
                if state.is_empty() {
                    State::LOCKED
                } else {
                    State::LOCKED_WAITERS
                },
                Ordering::Acquire,
                Ordering::Acquire,
            ) {
                state = v;
                spin_loop();
                continue;
            }
            if state.is_empty() {
                return Ok(());
            }
            wait_waker.wait(
                NonZeroUsize::new_unchecked(self as *const _ as usize),
                || self.load(Ordering::Acquire) != State::LOCKED_WAITERS,
            );
        }
    }
    unsafe fn write_cancel<W: WaitWake>(&self, wait_waker: &W) {
        let state = self.swap(State::EMPTY, Ordering::Release);
        if state == State::LOCKED_WAITERS {
            wait_waker.wake_all(NonZeroUsize::new_unchecked(self as *const _ as usize));
        } else {
            debug_assert_eq!(state, State::LOCKED);
        }
    }
    unsafe fn write_finish<W: WaitWake>(&self, wait_waker: &W, hash: u64) {
        let state = self.swap(State::make_full(hash), Ordering::Release);
        if state == State::LOCKED_WAITERS {
            wait_waker.wake_all(NonZeroUsize::new_unchecked(self as *const _ as usize));
        } else {
            debug_assert_eq!(state, State::LOCKED);
        }
    }
}

/// use separate non-generic struct to work around drop check
struct HashTableStorage {
    states: Box<[AtomicState]>,
    /// really a `Box<[UnsafeCell<MaybeUninit<T>>]>` with the same length as `self.states`
    values: NonNull<()>,
    drop_fn: unsafe fn(&mut Self),
}

impl Drop for HashTableStorage {
    fn drop(&mut self) {
        unsafe {
            (self.drop_fn)(self);
        }
    }
}

impl HashTableStorage {
    unsafe fn drop_fn<T>(&mut self) {
        let (states, values) = self.take::<T>();
        // safety: must work correctly when self.states and self.values are left empty by `ParallelHashTable::into_iter()`
        mem::drop(IntoIter {
            states: states.into_vec().into_iter().enumerate(),
            values,
        });
    }
    fn new<T>(states: Box<[AtomicState]>, values: Box<[UnsafeCell<MaybeUninit<T>>]>) -> Self {
        assert_eq!(states.len(), values.len());
        unsafe {
            Self {
                states,
                values: NonNull::new_unchecked(Box::into_raw(values) as *mut ()),
                drop_fn: Self::drop_fn::<T>,
            }
        }
    }
    unsafe fn take<T>(&mut self) -> (Box<[AtomicState]>, Box<[UnsafeCell<MaybeUninit<T>>]>) {
        let states = mem::replace(&mut self.states, Box::new([]));
        let replacement_values = NonNull::new_unchecked(
            Box::<[UnsafeCell<MaybeUninit<T>>]>::into_raw(Box::new([])) as *mut (),
        );
        let values = Box::from_raw(slice::from_raw_parts_mut(
            mem::replace(&mut self.values, replacement_values).as_ptr()
                as *mut UnsafeCell<MaybeUninit<T>>,
            states.len(),
        ));
        (states, values)
    }
    unsafe fn values<T>(&self) -> &[UnsafeCell<MaybeUninit<T>>] {
        slice::from_raw_parts(
            self.values.as_ptr() as *const UnsafeCell<MaybeUninit<T>>,
            self.states.len(),
        )
    }
    unsafe fn states_values_mut<T>(
        &mut self,
    ) -> (&mut [AtomicState], &mut [UnsafeCell<MaybeUninit<T>>]) {
        let values = slice::from_raw_parts_mut(
            self.values.as_ptr() as *mut UnsafeCell<MaybeUninit<T>>,
            self.states.len(),
        );
        (&mut self.states, values)
    }
}

pub struct ParallelHashTable<T, W> {
    storage: HashTableStorage,
    _phantom: PhantomData<T>,
    wait_waker: W,
    probe_distance: usize,
}

unsafe impl<T: Sync + Send, W: Sync + Send> Sync for ParallelHashTable<T, W> {}
unsafe impl<T: Sync + Send, W: Sync + Send> Send for ParallelHashTable<T, W> {}

impl<T, W> ParallelHashTable<T, W> {
    pub fn new(log2_capacity: u32, wait_waker: W) -> Self {
        let capacity = 1isize
            .checked_shl(log2_capacity)
            .expect("capacity is too big") as usize;
        Self::with_probe_distance(log2_capacity, wait_waker, capacity.min(16))
    }
    pub fn with_probe_distance(log2_capacity: u32, wait_waker: W, probe_distance: usize) -> Self {
        let capacity = 1isize
            .checked_shl(log2_capacity)
            .expect("capacity is too big") as usize;
        assert!(probe_distance <= capacity);
        let mut values = Vec::with_capacity(capacity);
        values.resize_with(capacity, || UnsafeCell::new(MaybeUninit::uninit()));
        let mut states = Vec::with_capacity(capacity);
        states.resize_with(capacity, || AtomicState::new(State::EMPTY));
        Self {
            storage: HashTableStorage::new::<T>(states.into(), values.into()),
            _phantom: PhantomData,
            wait_waker,
            probe_distance,
        }
    }
    pub fn capacity(&self) -> usize {
        self.storage.states.len()
    }
    pub fn probe_distance(&self) -> usize {
        self.probe_distance
    }
    pub fn iter(&self) -> Iter<T> {
        self.into_iter()
    }
    pub fn iter_mut(&mut self) -> IterMut<T> {
        self.into_iter()
    }
    pub fn wait_waker(&self) -> &W {
        &self.wait_waker
    }
    pub fn wait_waker_mut(&mut self) -> &mut W {
        &mut self.wait_waker
    }
}

impl<T: fmt::Debug, W> fmt::Debug for ParallelHashTable<T, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

#[derive(Clone, Debug)]
pub struct NotEnoughSpace;

impl fmt::Display for NotEnoughSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "not enough space left in hash_table")
    }
}

pub struct LockedEntry<'a, T, W: WaitWake> {
    state: &'a AtomicState,
    value: &'a UnsafeCell<MaybeUninit<T>>,
    hash: u64,
    wait_waker: &'a W,
}

impl<'a, T, W: WaitWake> LockedEntry<'a, T, W> {
    pub fn fill(self, value: T) -> &'a T {
        let state = self.state;
        let value_cell = self.value;
        let hash = self.hash;
        let wait_waker = self.wait_waker;
        mem::forget(self);
        unsafe {
            value_cell.get().write(MaybeUninit::new(value));
            state.write_finish(wait_waker, hash);
            &*(value_cell.get() as *const T)
        }
    }
}

impl<T, W: WaitWake> Drop for LockedEntry<'_, T, W> {
    fn drop(&mut self) {
        unsafe { self.state.write_cancel(self.wait_waker) }
    }
}

pub enum LockResult<'a, T, W: WaitWake> {
    Vacant(LockedEntry<'a, T, W>),
    Full(&'a T),
}

struct ProbeSequence<'a, T> {
    index: usize,
    mask: usize,
    states: &'a [AtomicState],
    values: &'a [UnsafeCell<MaybeUninit<T>>],
}

impl<'a, T> Iterator for ProbeSequence<'a, T> {
    type Item = (&'a AtomicState, &'a UnsafeCell<MaybeUninit<T>>);
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index & self.mask;
        self.index += 1;
        Some((&self.states[index], &self.values[index]))
    }
}

impl<T, W> ParallelHashTable<T, W> {
    fn probe_sequence(&self, hash: u64) -> iter::Take<ProbeSequence<T>> {
        debug_assert!(self.capacity().is_power_of_two());
        let mask = self.capacity() - 1;
        ProbeSequence {
            index: hash as usize,
            mask,
            states: &self.storage.states,
            values: unsafe { self.storage.values() },
        }
        .take(self.probe_distance)
    }
}

impl<T, W: WaitWake> ParallelHashTable<T, W> {
    pub fn find<F: FnMut(&T) -> bool>(&self, hash: u64, mut entry_eq: F) -> Option<&T> {
        let expected = State::make_full(hash);
        for (state, value) in self.probe_sequence(hash) {
            unsafe {
                let state = state.load(Ordering::Acquire);
                if state == expected {
                    let value = &*(value.get() as *const T);
                    if entry_eq(value) {
                        return Some(value);
                    }
                } else if state.is_empty() {
                    break;
                }
            }
        }
        None
    }
    pub fn lock_entry<F: FnMut(&T) -> bool>(
        &self,
        hash: u64,
        mut entry_eq: F,
    ) -> Result<LockResult<T, W>, NotEnoughSpace> {
        let expected = State::make_full(hash);
        for (state, value) in self.probe_sequence(hash) {
            unsafe {
                match state.write_start(&self.wait_waker) {
                    Ok(()) => {
                        return Ok(LockResult::Vacant(LockedEntry {
                            state,
                            value,
                            hash,
                            wait_waker: &self.wait_waker,
                        }));
                    }
                    Err(state) => {
                        if state == expected {
                            let value = &*(value.get() as *const T);
                            if entry_eq(value) {
                                return Ok(LockResult::Full(value));
                            }
                        }
                    }
                }
            }
        }
        Err(NotEnoughSpace)
    }
}

pub struct IntoIter<T> {
    states: iter::Enumerate<vec::IntoIter<AtomicState>>,
    values: Box<[UnsafeCell<MaybeUninit<T>>]>,
}

unsafe impl<T: Sync + Send> Sync for IntoIter<T> {}
unsafe impl<T: Sync + Send> Send for IntoIter<T> {}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        self.for_each(mem::drop);
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((index, state)) = self.states.next() {
            if state.into_inner().is_full() {
                unsafe {
                    return Some(self.values[index].get().read().assume_init());
                }
            }
        }
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.states.len()))
    }
}

impl<T> FusedIterator for IntoIter<T> {}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while let Some((index, state)) = self.states.next_back() {
            if state.into_inner().is_full() {
                unsafe {
                    return Some(self.values[index].get().read().assume_init());
                }
            }
        }
        None
    }
}

impl<T, H> IntoIterator for ParallelHashTable<T, H> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    fn into_iter(mut self) -> Self::IntoIter {
        let (states, values) = unsafe { self.storage.take::<T>() };
        IntoIter {
            states: states.into_vec().into_iter().enumerate(),
            values,
        }
        // `HashTableStorage::drop()` works after calling `HashTableStorage::take()`
    }
}

pub struct Iter<'a, T> {
    states: iter::Enumerate<slice::Iter<'a, AtomicState>>,
    values: &'a [UnsafeCell<MaybeUninit<T>>],
}

fn split_iter<T: Clone + DoubleEndedIterator + ExactSizeIterator>(iter: T) -> (T, Option<T>) {
    if iter.len() <= 1 {
        return (iter, None);
    }
    let mut first_half = iter.clone();
    let mut last_half = iter;
    let mid = first_half.len() / 2;
    first_half.nth_back(first_half.len() - mid - 1);
    last_half.nth(mid - 1);
    (first_half, Some(last_half))
}

#[cfg(test)]
#[test]
fn test_split_iter() {
    assert_eq!(split_iter(1..1), (1..1, None));
    assert_eq!(split_iter(1..2), (1..2, None));
    assert_eq!(split_iter(1..3), (1..2, Some(2..3)));
    assert_eq!(split_iter(1..4), (1..2, Some(2..4)));
    assert_eq!(split_iter(1..5), (1..3, Some(3..5)));
    assert_eq!(split_iter(1..6), (1..3, Some(3..6)));
}

impl<'a, T> Iter<'a, T> {
    pub fn split(&self) -> (Self, Option<Self>) {
        let (first_half, last_half) = split_iter(self.states.clone());
        (
            Self {
                states: first_half,
                values: self.values,
            },
            last_half.map(|states| Self {
                states,
                values: self.values,
            }),
        )
    }
}

impl<'a, T> Clone for Iter<'a, T> {
    fn clone(&self) -> Self {
        Self {
            states: self.states.clone(),
            values: self.values,
        }
    }
}

unsafe impl<T: Sync + Send> Sync for Iter<'_, T> {}
unsafe impl<T: Sync + Send> Send for Iter<'_, T> {}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((index, state)) = self.states.next() {
            unsafe {
                if state.load(Ordering::Acquire).is_full() {
                    return Some(&*(self.values[index].get() as *const T));
                }
            }
        }
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.states.len()))
    }
}

impl<'a, T> FusedIterator for Iter<'a, T> {}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while let Some((index, state)) = self.states.next_back() {
            unsafe {
                if state.load(Ordering::Acquire).is_full() {
                    return Some(&*(self.values[index].get() as *const T));
                }
            }
        }
        None
    }
}

impl<'a, T, H> IntoIterator for &'a ParallelHashTable<T, H> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        Iter {
            states: self.storage.states.iter().enumerate(),
            values: unsafe { self.storage.values::<T>() },
        }
    }
}

pub struct EntryMut<'a, T> {
    state: &'a mut AtomicState,
    value: &'a UnsafeCell<MaybeUninit<T>>,
}

unsafe impl<T: Sync + Send> Sync for EntryMut<'_, T> {}
unsafe impl<T: Sync + Send> Send for EntryMut<'_, T> {}

impl<'a, T> EntryMut<'a, T> {
    pub fn remove(self) -> T {
        *self.state = AtomicState::new(State::EMPTY);
        unsafe { self.value.get().read().assume_init() }
    }
}

impl<'a, T> Deref for EntryMut<'a, T> {
    type Target = T;
    fn deref<'b>(&'b self) -> &'b Self::Target {
        unsafe { &*(self.value.get() as *const T) }
    }
}

pub struct IterMut<'a, T> {
    range: Range<usize>,
    states: *mut AtomicState,
    values: &'a [UnsafeCell<MaybeUninit<T>>],
}

impl<'a, T> IterMut<'a, T> {
    pub fn split(self) -> (Self, Option<Self>) {
        let (first_half, last_half) = split_iter(self.range.clone());
        (
            Self {
                range: first_half,
                ..self
            },
            last_half.map(|range| Self { range, ..self }),
        )
    }
}

unsafe impl<T: Sync + Send> Sync for IterMut<'_, T> {}
unsafe impl<T: Sync + Send> Send for IterMut<'_, T> {}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = EntryMut<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(index) = self.range.next() {
            let state = unsafe { &mut *self.states.offset(index as isize) };
            if state.get_mut().is_full() {
                return Some(EntryMut {
                    state,
                    value: &self.values[index],
                });
            }
        }
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.range.len()))
    }
}

impl<'a, T> FusedIterator for IterMut<'a, T> {}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while let Some(index) = self.range.next_back() {
            let state = unsafe { &mut *self.states.offset(index as isize) };
            if state.get_mut().is_full() {
                return Some(EntryMut {
                    state,
                    value: &self.values[index],
                });
            }
        }
        None
    }
}

impl<'a, T, H> IntoIterator for &'a mut ParallelHashTable<T, H> {
    type Item = EntryMut<'a, T>;
    type IntoIter = IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        let (states, values) = unsafe { self.storage.states_values_mut::<T>() };
        IterMut {
            range: 0..states.len(),
            states: states.as_mut_ptr(),
            values,
        }
    }
}

#[cfg(test)]
mod test {
    use super::{LockResult, ParallelHashTable, WaitWake};
    use crate::std_support::StdWaitWake;
    use alloc::{sync::Arc, vec::Vec};
    use core::{
        cell::{Cell, RefCell},
        hash::{BuildHasher, Hash, Hasher},
        marker::PhantomData,
        mem,
        num::NonZeroUsize,
        sync::atomic::{AtomicU8, AtomicUsize, Ordering},
    };
    use hashbrown::{hash_map::DefaultHashBuilder, HashMap, HashSet};
    use std::{
        sync::{
            mpsc::{sync_channel, Receiver},
            Condvar, Mutex,
        },
        thread,
    };

    #[derive(Clone, Copy, Default, Debug)]
    struct SingleThreadedWaitWake(PhantomData<*mut ()>);

    impl WaitWake for SingleThreadedWaitWake {
        unsafe fn wait<SC: FnOnce() -> bool>(&self, _key: NonZeroUsize, should_cancel: SC) {
            should_cancel();
        }
        unsafe fn wake_all(&self, _key: NonZeroUsize) {}
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Entry {
        hash: u8,
        non_hash: u8,
    }

    impl Hash for Entry {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.hash.hash(state);
        }
    }

    impl Entry {
        fn new() -> Self {
            static NEXT_HASH: AtomicU8 = AtomicU8::new(1);
            static NEXT_NON_HASH: AtomicU8 = AtomicU8::new(1);
            let hash = NEXT_HASH.fetch_add(1, Ordering::Relaxed);
            Self {
                hash,
                non_hash: NEXT_NON_HASH.fetch_add(hash, Ordering::Relaxed),
            }
        }
    }

    #[derive(Clone, Debug)]
    struct EntryWithLifetime<'a>(Entry, RefCell<Option<&'a EntryWithLifetime<'a>>>);

    impl Hash for EntryWithLifetime<'_> {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.0.hash(state);
        }
    }

    impl PartialEq for EntryWithLifetime<'_> {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    #[test]
    fn arena_test() {
        let ht: ParallelHashTable<EntryWithLifetime, SingleThreadedWaitWake> =
            ParallelHashTable::new(2, Default::default());
        let hasher = DefaultHashBuilder::new();
        let hasher = |entry: &EntryWithLifetime| -> u64 {
            let mut hasher = hasher.build_hasher();
            entry.hash(&mut hasher);
            hasher.finish()
        };
        let entry1 = EntryWithLifetime(Entry::new(), RefCell::new(None));
        let entry_ref1 = match ht.lock_entry(hasher(&entry1), |v| *v == entry1) {
            Ok(LockResult::Vacant(locked_entry)) => {
                let v = locked_entry.fill(entry1.clone());
                assert_eq!(*v, entry1);
                v
            }
            Ok(LockResult::Full(_)) => unreachable!(),
            Err(super::NotEnoughSpace) => unreachable!(),
        };
        let entry2 = EntryWithLifetime(Entry::new(), RefCell::new(None));
        let entry_ref2 = match ht.lock_entry(hasher(&entry2), |v| *v == entry2) {
            Ok(LockResult::Vacant(locked_entry)) => {
                let v = locked_entry.fill(entry2.clone());
                assert_eq!(*v, entry2);
                v
            }
            Ok(LockResult::Full(_)) => unreachable!(),
            Err(super::NotEnoughSpace) => unreachable!(),
        };
        *entry_ref1.1.borrow_mut() = Some(entry_ref2);
    }

    struct DropTestHelper<'a>(&'a AtomicUsize);

    impl Drop for DropTestHelper<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn drop_test() {
        let item1_count = AtomicUsize::new(0);
        let item2_count = AtomicUsize::new(0);
        let ht: ParallelHashTable<(Entry, DropTestHelper), SingleThreadedWaitWake> =
            ParallelHashTable::new(6, Default::default());
        let hasher = DefaultHashBuilder::new();
        let hasher = |entry: &Entry| -> u64 {
            let mut hasher = hasher.build_hasher();
            entry.hash(&mut hasher);
            hasher.finish()
        };
        let entry1 = Entry::new();
        match ht.lock_entry(hasher(&entry1), |v| v.0 == entry1) {
            Ok(LockResult::Vacant(locked_entry)) => {
                locked_entry.fill((entry1.clone(), DropTestHelper(&item1_count)))
            }
            Ok(LockResult::Full(_)) => unreachable!(),
            Err(super::NotEnoughSpace) => unreachable!(),
        };
        let entry2 = Entry::new();
        match ht.lock_entry(hasher(&entry2), |v| v.0 == entry2) {
            Ok(LockResult::Vacant(locked_entry)) => {
                locked_entry.fill((entry2.clone(), DropTestHelper(&item2_count)))
            }
            Ok(LockResult::Full(_)) => unreachable!(),
            Err(super::NotEnoughSpace) => unreachable!(),
        };
        assert_eq!(item1_count.load(Ordering::Relaxed), 0);
        assert_eq!(item2_count.load(Ordering::Relaxed), 0);
        mem::drop(ht);
        assert_eq!(item1_count.load(Ordering::Relaxed), 1);
        assert_eq!(item2_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test1() {
        let mut ht: ParallelHashTable<Entry, SingleThreadedWaitWake> =
            ParallelHashTable::new(6, Default::default());
        let hasher = DefaultHashBuilder::new();
        let hasher = |entry: &Entry| -> u64 {
            let mut hasher = hasher.build_hasher();
            entry.hash(&mut hasher);
            hasher.finish()
        };
        let mut reference_ht = HashSet::<Entry>::new();
        for i in 0..100000u64 {
            if i.wrapping_mul(0x40FA_5283_D1A3_292B_u64) >> 56 == 0 {
                let entries: Vec<Entry> = ht.iter().cloned().collect();
                let entries2: Vec<Entry> = ht.iter_mut().map(|v| v.remove()).collect();
                assert_eq!(entries, entries2);
                for entry in entries {
                    assert_eq!(reference_ht.take(&entry), Some(entry));
                }
                assert!(reference_ht.is_empty());
            } else if i.wrapping_mul(0xA331_ABB2_E016_BC0A_u64) >> 63 != 0 {
                let entry = Entry::new();
                match ht.lock_entry(hasher(&entry), |v| *v == entry) {
                    Ok(LockResult::Vacant(locked_entry)) => {
                        assert_eq!(*locked_entry.fill(entry.clone()), entry);
                        assert!(
                            reference_ht.insert(entry.clone()),
                            "failed to insert {:?}",
                            entry
                        );
                    }
                    Ok(LockResult::Full(old_entry)) => {
                        assert_eq!(reference_ht.get(&entry), Some(old_entry));
                    }
                    Err(super::NotEnoughSpace) => {
                        assert!(reference_ht.len() >= ht.probe_distance());
                    }
                }
            } else {
                for _ in 0..10 {
                    let entry = Entry::new();
                    assert_eq!(
                        ht.find(hasher(&entry), |v| *v == entry),
                        reference_ht.get(&entry)
                    );
                }
            }
        }
        let entries: Vec<Entry> = ht.iter().cloned().collect();
        let entries2: Vec<Entry> = ht.into_iter().collect();
        assert_eq!(entries, entries2);
    }

    #[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
    struct ThreadIndex(u32);

    std::thread_local! {
        static THREAD_INDEX: Cell<Option<ThreadIndex>> = Cell::new(None);
    }

    struct WaitWakeTracker<W> {
        wait_set: Mutex<HashMap<ThreadIndex, NonZeroUsize>>,
        wait_set_changed: Condvar,
        inner_wait_wake: W,
    }

    impl<W> WaitWakeTracker<W> {
        fn new(inner_wait_wake: W) -> Self {
            Self {
                wait_set: Mutex::new(HashMap::new()),
                wait_set_changed: Condvar::new(),
                inner_wait_wake,
            }
        }
    }

    impl<W: WaitWake> WaitWake for WaitWakeTracker<W> {
        unsafe fn wait<SC: FnOnce() -> bool>(&self, key: NonZeroUsize, should_cancel: SC) {
            THREAD_INDEX.with(|thread_index| {
                let mut lock = self.wait_set.lock().unwrap();
                lock.insert(thread_index.get().unwrap(), key);
                self.wait_set_changed.notify_all();
            });
            self.inner_wait_wake.wait(key, should_cancel);
            THREAD_INDEX.with(|thread_index| {
                let mut lock = self.wait_set.lock().unwrap();
                lock.remove(&thread_index.get().unwrap());
                self.wait_set_changed.notify_all();
            });
        }

        unsafe fn wake_all(&self, key: NonZeroUsize) {
            self.inner_wait_wake.wake_all(key);
        }
    }

    #[test]
    fn lock_test() {
        enum Op {
            InsertIntoEmpty(Entry),
            AttemptInsertIntoFull(Entry),
            LockEntry(Entry),
            CancelLock,
            FillLock,
            Finish,
            Sync,
        }
        let ht = Arc::new(ParallelHashTable::new(8, WaitWakeTracker::new(StdWaitWake)));
        let hasher = DefaultHashBuilder::new();
        let hasher = move |entry: &Entry| -> u64 {
            let mut hasher = hasher.build_hasher();
            entry.hash(&mut hasher);
            hasher.finish()
        };
        let thread_fn = {
            let ht = ht.clone();
            let hasher = hasher.clone();
            move |thread_index, op_channel: Receiver<Op>| {
                THREAD_INDEX.with(|v| v.set(Some(thread_index)));
                loop {
                    match op_channel.recv().unwrap() {
                        Op::Sync => {}
                        Op::InsertIntoEmpty(entry) => {
                            match ht.lock_entry(hasher(&entry), |v| *v == entry).unwrap() {
                                LockResult::Vacant(lock) => lock.fill(entry),
                                LockResult::Full(_) => {
                                    panic!("failed to lock already full entry: {:?}", entry);
                                }
                            };
                        }
                        Op::AttemptInsertIntoFull(entry) => {
                            match ht.lock_entry(hasher(&entry), |v| *v == entry).unwrap() {
                                LockResult::Vacant(_) => panic!("expected full entry: {:?}", entry),
                                LockResult::Full(v) => assert_eq!(v, &entry),
                            };
                        }
                        Op::LockEntry(entry) => {
                            let lock = match ht.lock_entry(hasher(&entry), |v| *v == entry).unwrap()
                            {
                                LockResult::Vacant(lock) => lock,
                                LockResult::Full(_) => {
                                    panic!("failed to lock already full entry: {:?}", entry);
                                }
                            };
                            loop {
                                match op_channel.recv().unwrap() {
                                    Op::Sync => {}
                                    Op::CancelLock => break,
                                    Op::FillLock => {
                                        lock.fill(entry);
                                        break;
                                    }
                                    Op::InsertIntoEmpty(_)
                                    | Op::AttemptInsertIntoFull(_)
                                    | Op::LockEntry(_)
                                    | Op::Finish => panic!("locked"),
                                }
                            }
                        }
                        Op::CancelLock | Op::FillLock => panic!("not locked"),
                        Op::Finish => break,
                    }
                }
            }
        };
        const THREAD_COUNT: u32 = 3;
        let threads: Vec<_> = (0..THREAD_COUNT)
            .map(|i| {
                let f = thread_fn.clone();
                let (sender, receiver) = sync_channel(0);
                (thread::spawn(move || f(ThreadIndex(i), receiver)), sender)
            })
            .collect();
        let send_op = |thread_index: usize, op| threads[thread_index].1.send(op).unwrap();
        let assert_contents = |expected: &[&Entry]| {
            let expected: HashSet<_> = expected.iter().copied().collect();
            let actual: HashSet<_> = ht.iter().collect();
            assert_eq!(expected, actual);
        };
        let wait_for_thread_to_wait = |thread_index: u32| {
            let wait_waker = ht.wait_waker();
            let _ = ht
                .wait_waker()
                .wait_set_changed
                .wait_while(wait_waker.wait_set.lock().unwrap(), |v| {
                    !v.contains_key(&ThreadIndex(thread_index))
                })
                .unwrap();
        };
        let assert_find =
            |entry, expected| assert_eq!(ht.find(hasher(entry), |v| v == entry), expected);
        let entry1 = Entry::new();
        send_op(0, Op::InsertIntoEmpty(entry1.clone()));
        send_op(0, Op::Sync);
        assert_contents(&[&entry1]);
        assert_find(&entry1, Some(&entry1));
        let entry2 = Entry::new();
        assert_find(&entry2, None);
        send_op(0, Op::LockEntry(entry2.clone()));
        send_op(0, Op::Sync);
        assert_find(&entry2, None);
        send_op(1, Op::InsertIntoEmpty(entry2.clone()));
        wait_for_thread_to_wait(1);
        assert_contents(&[&entry1]);
        assert_find(&entry2, None);
        send_op(0, Op::CancelLock);
        send_op(1, Op::Sync);
        assert_contents(&[&entry1, &entry2]);
        assert_find(&entry2, Some(&entry2));
        let entry3 = Entry::new();
        send_op(0, Op::LockEntry(entry3.clone()));
        send_op(0, Op::Sync);
        assert_find(&entry3, None);
        send_op(1, Op::AttemptInsertIntoFull(entry3.clone()));
        send_op(2, Op::AttemptInsertIntoFull(entry3.clone()));
        wait_for_thread_to_wait(1);
        wait_for_thread_to_wait(2);
        assert_find(&entry3, None);
        assert_contents(&[&entry1, &entry2]);
        send_op(0, Op::FillLock);
        send_op(1, Op::Sync);
        send_op(2, Op::Sync);
        assert_find(&entry3, Some(&entry3));
        assert_contents(&[&entry1, &entry2, &entry3]);
        threads.into_iter().for_each(|(join_handle, sender)| {
            sender.send(Op::Finish).unwrap();
            mem::drop(sender);
            join_handle.join().unwrap();
        });
    }
}
