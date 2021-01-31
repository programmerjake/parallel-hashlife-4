use alloc::{boxed::Box, vec::Vec};
use core::{
    cell::{Cell, UnsafeCell},
    fmt::{self, Debug},
    hint::spin_loop,
    marker::PhantomData,
    mem::{self, MaybeUninit},
    num::NonZeroUsize,
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

unsafe trait HashTableImpl {
    type FullState: 'static + Debug + Eq;
    type StateCell: 'static + Debug;
    const STATE_CELL_EMPTY: Self::StateCell;
    fn make_full_state(hash: u64) -> Self::FullState;
    fn read_state(state_cell: &Self::StateCell) -> Option<Self::FullState>;
    unsafe fn lock_state(&self, state_cell: &Self::StateCell) -> Result<(), Self::FullState>;
    unsafe fn unlock_and_fill_state(
        &self,
        state_cell: &Self::StateCell,
        full_state: Self::FullState,
    );
    unsafe fn unlock_and_empty_state(&self, state_cell: &Self::StateCell);
}

const STATE_EMPTY: u32 = 0;
const STATE_LOCKED: u32 = STATE_EMPTY + 1;
const STATE_LOCKED_WAITERS: u32 = STATE_LOCKED + 1;
const UNSYNC_STATE_FIRST_FULL: u32 = STATE_LOCKED + 1;
const SYNC_STATE_FIRST_FULL: u32 = STATE_LOCKED_WAITERS + 1;

fn make_full_state<const STATE_FIRST_FULL: u32>(hash: u64) -> u32 {
    let retval = (hash >> 32) as u32;
    if retval >= STATE_FIRST_FULL {
        return retval;
    }
    let retval = hash as u32;
    if retval >= STATE_FIRST_FULL {
        return retval;
    }
    return STATE_FIRST_FULL;
}

#[derive(Clone, Copy, Default, Debug)]
struct UnsyncHashTableImpl;

unsafe impl HashTableImpl for UnsyncHashTableImpl {
    type FullState = u32;

    type StateCell = Cell<u32>;

    const STATE_CELL_EMPTY: Self::StateCell = Cell::new(STATE_EMPTY);

    fn make_full_state(hash: u64) -> Self::FullState {
        make_full_state::<UNSYNC_STATE_FIRST_FULL>(hash)
    }

    fn read_state(state_cell: &Self::StateCell) -> Option<Self::FullState> {
        let retval = state_cell.get();
        if retval >= UNSYNC_STATE_FIRST_FULL {
            Some(retval)
        } else {
            None
        }
    }

    unsafe fn lock_state(&self, state_cell: &Self::StateCell) -> Result<(), Self::FullState> {
        let state = state_cell.get();
        if state >= UNSYNC_STATE_FIRST_FULL {
            Err(state)
        } else {
            assert_eq!(
                state_cell.replace(STATE_LOCKED),
                STATE_EMPTY,
                "attempt to lock already locked UnsyncHashTable entry"
            );
            Ok(())
        }
    }

    unsafe fn unlock_and_fill_state(
        &self,
        state_cell: &Self::StateCell,
        full_state: Self::FullState,
    ) {
        debug_assert!(full_state >= UNSYNC_STATE_FIRST_FULL);
        let state = state_cell.replace(full_state);
        debug_assert_eq!(state, STATE_LOCKED);
    }

    unsafe fn unlock_and_empty_state(&self, state_cell: &Self::StateCell) {
        let state = state_cell.replace(STATE_EMPTY);
        debug_assert_eq!(state, STATE_LOCKED);
    }
}

#[derive(Clone, Copy, Default, Debug)]
struct SyncHashTableImpl<W: WaitWake> {
    wait_waker: W,
}

unsafe impl<W: WaitWake> HashTableImpl for SyncHashTableImpl<W> {
    type FullState = u32;

    type StateCell = AtomicU32;

    const STATE_CELL_EMPTY: Self::StateCell = AtomicU32::new(STATE_EMPTY);

    fn make_full_state(hash: u64) -> Self::FullState {
        make_full_state::<SYNC_STATE_FIRST_FULL>(hash)
    }

    fn read_state(state_cell: &Self::StateCell) -> Option<Self::FullState> {
        let retval = state_cell.load(Ordering::Acquire);
        if retval >= SYNC_STATE_FIRST_FULL {
            Some(retval)
        } else {
            None
        }
    }

    unsafe fn lock_state(&self, state_cell: &Self::StateCell) -> Result<(), Self::FullState> {
        let mut spin_count = 0;
        let mut state = state_cell.load(Ordering::Acquire);
        loop {
            if state >= SYNC_STATE_FIRST_FULL {
                return Err(state);
            }
            if state != STATE_EMPTY && spin_count < 32 {
                spin_count += 1;
                state = state_cell.load(Ordering::Acquire);
                spin_loop();
                continue;
            }
            if let Err(v) = state_cell.compare_exchange_weak(
                state,
                if state == STATE_EMPTY {
                    STATE_LOCKED
                } else {
                    STATE_LOCKED_WAITERS
                },
                Ordering::Acquire,
                Ordering::Acquire,
            ) {
                state = v;
                spin_loop();
                continue;
            }
            if state == STATE_EMPTY {
                return Ok(());
            }
            self.wait_waker.wait(
                NonZeroUsize::new_unchecked(state_cell as *const _ as usize),
                || state_cell.load(Ordering::Acquire) != STATE_LOCKED_WAITERS,
            );
        }
    }

    unsafe fn unlock_and_fill_state(
        &self,
        state_cell: &Self::StateCell,
        full_state: Self::FullState,
    ) {
        debug_assert!(full_state >= SYNC_STATE_FIRST_FULL);
        let state = state_cell.swap(full_state, Ordering::Release);
        if state == STATE_LOCKED_WAITERS {
            self.wait_waker
                .wake_all(NonZeroUsize::new_unchecked(state_cell as *const _ as usize));
        } else {
            debug_assert_eq!(state, STATE_LOCKED);
        }
    }

    unsafe fn unlock_and_empty_state(&self, state_cell: &Self::StateCell) {
        let state = state_cell.swap(STATE_EMPTY, Ordering::Release);
        if state == STATE_LOCKED_WAITERS {
            self.wait_waker
                .wake_all(NonZeroUsize::new_unchecked(state_cell as *const _ as usize));
        } else {
            debug_assert_eq!(state, STATE_LOCKED);
        }
    }
}

/// use separate non-generic struct to work around drop check
struct HashTableStorageInternal {
    /// really `Box<[StateCell]>` with length of `self.capacity`
    state_cells_ptr: NonNull<()>,
    /// really `Box<[Value]>` with length of `self.capacity`
    values_ptr: NonNull<()>,
    capacity: usize,
}

impl HashTableStorageInternal {
    fn from_boxes<StateCell, Value>(state_cells: Box<[StateCell]>, values: Box<[Value]>) -> Self {
        assert_eq!(state_cells.len(), values.len());
        let capacity = state_cells.len();
        unsafe {
            Self {
                state_cells_ptr: NonNull::new_unchecked(Box::into_raw(state_cells) as *mut ()),
                values_ptr: NonNull::new_unchecked(Box::into_raw(values) as *mut ()),
                capacity,
            }
        }
    }
    unsafe fn as_ref<StateCell, Value>(&self) -> (&[StateCell], &[Value]) {
        (
            slice::from_raw_parts(self.state_cells_ptr as *const StateCell, self.capacity),
            slice::from_raw_parts(self.values_ptr as *const Value, self.capacity),
        )
    }
    unsafe fn as_mut<StateCell, Value>(&mut self) -> (&mut [StateCell], &mut [Value]) {
        (
            slice::from_raw_parts_mut(self.state_cells_ptr as *mut StateCell, self.capacity),
            slice::from_raw_parts_mut(self.values_ptr as *mut Value, self.capacity),
        )
    }
    unsafe fn into_boxes<StateCell, Value>(self) -> (Box<[StateCell]>, Box<[Value]>) {
        (
            Box::from_raw(slice::from_raw_parts_mut(
                self.state_cells_ptr as *mut StateCell,
                self.capacity,
            )),
            Box::from_raw(slice::from_raw_parts_mut(
                self.values_ptr as *mut Value,
                self.capacity,
            )),
        )
    }
}

/// use separate non-generic struct to work around drop check
struct HashTableStorage {
    internal: Option<HashTableStorageInternal>,
    drop_fn: unsafe fn(HashTableStorageInternal),
}

impl Drop for HashTableStorage {
    fn drop(&mut self) {
        if let Some(internal) = self.internal.take() {
            unsafe {
                (self.drop_fn)(self);
            }
        }
    }
}

impl HashTableStorage {
    fn capacity(&self) -> usize {
        self.internal.unwrap().capacity
    }
    unsafe fn drop_fn<HTI: HashTableImpl, Value>(internal: HashTableStorageInternal) {
        let (state_cells, values) =
            internal.into_boxes::<HTI::StateCell, UnsafeCell<MaybeUninit<Value>>>();
        mem::drop(IntoIter::<HTI, Value> {
            state_cells,
            values,
        });
    }
    unsafe fn new<HTI: HashTableImpl, Value>(
        state_cells: Box<[HTI::StateCell]>,
        values: Box<[UnsafeCell<MaybeUninit<Value>>]>,
    ) -> Self {
        unsafe {
            Self {
                internal: HashTableStorageInternal::from_boxes::<
                    HTI::StateCell,
                    UnsafeCell<MaybeUninit<Value>>,
                >(state_cells, values),
                drop_fn: Self::drop_fn::<HTI, Value>,
            }
        }
    }
    unsafe fn take<HTI: HashTableImpl, Value>(
        &mut self,
    ) -> (Box<[HTI::StateCell]>, Box<[UnsafeCell<MaybeUninit<Value>>]>) {
        self.internal
            .unwrap()
            .into_boxes::<HTI::StateCell, UnsafeCell<MaybeUninit<Value>>>()
    }
    unsafe fn states_values<HTI: HashTableImpl, Value>(
        &self,
    ) -> (&[HTI::StateCell], &[UnsafeCell<MaybeUninit<Value>>]) {
        self.internal
            .unwrap()
            .as_ref::<HTI::StateCell, UnsafeCell<MaybeUninit<Value>>>()
    }
    unsafe fn states_values_mut<HTI: HashTableImpl, Value>(
        &mut self,
    ) -> (&mut [HTI::StateCell], &mut [UnsafeCell<MaybeUninit<Value>>]) {
        self.internal
            .unwrap()
            .as_mut::<HTI::StateCell, UnsafeCell<MaybeUninit<Value>>>()
    }
}

struct HashTable<HTI: HashTableImpl, Value> {
    storage: HashTableStorage,
    _phantom: PhantomData<Value>,
    hash_table_impl: HTI,
    probe_distance: usize,
}

unsafe impl<HTI: HashTableImpl, Value> Sync for HashTable<HTI, Value>
where
    HTI::StateCell: Send + Sync,
    HTI: Send + Sync,
    Value: Send + Sync,
{
}

unsafe impl<HTI: HashTableImpl, Value> Send for HashTable<HTI, Value>
where
    HTI::StateCell: Send + Sync,
    HTI: Send + Sync,
    Value: Send + Sync,
{
}

impl<HTI: HashTableImpl, Value> HashTable<HTI, Value> {
    fn new(log2_capacity: u32, hash_table_impl: HTI) -> Self {
        let capacity = 1isize
            .checked_shl(log2_capacity)
            .expect("capacity is too big") as usize;
        Self::with_probe_distance(log2_capacity, hash_table_impl, capacity.min(16))
    }
    fn with_probe_distance(
        log2_capacity: u32,
        hash_table_impl: HTI,
        probe_distance: usize,
    ) -> Self {
        let capacity = 1isize
            .checked_shl(log2_capacity)
            .expect("capacity is too big") as usize;
        assert!(probe_distance <= capacity);
        let mut values = Vec::with_capacity(capacity);
        values.resize_with(capacity, || UnsafeCell::new(MaybeUninit::uninit()));
        let mut states = Vec::with_capacity(capacity);
        states.resize_with(capacity, || HTI::STATE_CELL_EMPTY);
        unsafe {
            Self {
                storage: HashTableStorage::new::<HTI, Value>(states.into(), values.into()),
                _phantom: PhantomData,
                hash_table_impl,
                probe_distance,
            }
        }
    }
    fn capacity(&self) -> usize {
        self.storage.capacity()
    }
    fn probe_distance(&self) -> usize {
        self.probe_distance
    }
    fn iter<'a>(&'a self) -> Iter<'a, HTI, Value> {
        self.into_iter()
    }
    fn iter_mut<'a>(&'a mut self) -> IterMut<'a, HTI, Value> {
        self.into_iter()
    }
    fn hash_table_impl(&self) -> &HTI {
        &self.hash_table_impl
    }
    fn hash_table_impl_mut(&mut self) -> &mut HTI {
        &mut self.hash_table_impl
    }
}

impl<HTI: HashTableImpl, Value: fmt::Debug> fmt::Debug for HashTable<HTI, Value> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

struct LockedEntry<'a, HTI: HashTableImpl, Value> {
    state_cell: &'a HTI::StateCell,
    value: &'a UnsafeCell<MaybeUninit<Value>>,
    full_state: HTI::FullState,
    hash_table_impl: &'a HTI,
}

impl<'a, HTI: HashTableImpl, Value> LockedEntry<'a, HTI, Value> {
    fn fill(self, value: Value) -> &'a Value {
        let state_cell = self.state_cell;
        let value_cell = self.value;
        let full_state = self.full_state;
        let hash_table_impl = self.hash_table_impl;
        mem::forget(self);
        unsafe {
            value_cell.get().write(MaybeUninit::new(value));
            hash_table_impl.unlock_and_fill_state(state_cell, full_state);
            &*(value_cell.get() as *const Value)
        }
    }
}

impl<HTI: HashTableImpl, Value> Drop for LockedEntry<'_, HTI, Value> {
    fn drop(&mut self) {
        unsafe { self.state.write_cancel(self.wait_waker) }
    }
}

enum LockResult<'a, HTI: HashTableImpl, Value> {
    Vacant(LockedEntry<'a, HTI, Value>),
    Full(&'a Value),
}

compile_error!("FIXME: finish from here and add table access by index");

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
    fn probe_sequence<'a>(&'a self, hash: u64) -> iter::Take<ProbeSequence<'a, T>> {
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

impl<HTI: HashTableImpl, Value> ParallelHashTable<T, W> {
    fn find<'a, F: FnMut(&'a T) -> bool>(&'a self, hash: u64, mut entry_eq: F) -> Option<&'a T> {
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
    fn lock_entry<'a, F: FnMut(&'a T) -> bool>(
        &'a self,
        hash: u64,
        mut entry_eq: F,
    ) -> Result<LockResult<'a, T, W>, NotEnoughSpace> {
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

struct IntoIter<T> {
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

struct Iter<'a, T> {
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
    fn split(&self) -> (Self, Option<Self>) {
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

struct EntryMut<'a, T> {
    state: &'a mut AtomicState,
    value: &'a UnsafeCell<MaybeUninit<T>>,
}

unsafe impl<T: Sync + Send> Sync for EntryMut<'_, T> {}
unsafe impl<T: Sync + Send> Send for EntryMut<'_, T> {}

impl<'a, T> EntryMut<'a, T> {
    fn remove(self) -> T {
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

struct IterMut<'a, T> {
    range: Range<usize>,
    states: *mut AtomicState,
    values: &'a [UnsafeCell<MaybeUninit<T>>],
}

impl<'a, T> IterMut<'a, T> {
    fn split(self) -> (Self, Option<Self>) {
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
        let ht: ParallelHashTable<EntryWithLifetime<'_>, SingleThreadedWaitWake> =
            ParallelHashTable::new(2, Default::default());
        let hasher = DefaultHashBuilder::new();
        let hasher = |entry: &EntryWithLifetime<'_>| -> u64 {
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
        let ht: ParallelHashTable<(Entry, DropTestHelper<'_>), SingleThreadedWaitWake> =
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
