use alloc::{boxed::Box, vec, vec::Vec};
use core::{
    cell::UnsafeCell,
    fmt,
    hash::BuildHasher,
    hint::spin_loop,
    iter,
    iter::FusedIterator,
    marker::PhantomData,
    mem,
    mem::{ManuallyDrop, MaybeUninit},
    num::NonZeroUsize,
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
enum StateKind {
    Empty = 0,
    Locked = 1,
    LockedWaiters = 2,
    Full = 3,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
struct State(u32);

impl State {
    const EMPTY: Self = Self(0);
    const LOCKED: Self = Self(1);
    const LOCKED_WAITERS: Self = Self(2);
    const FIRST_FULL: Self = Self(3);
    const fn kind(self) -> StateKind {
        match self {
            Self::EMPTY => StateKind::Empty,
            Self::LOCKED => StateKind::Locked,
            Self::LOCKED_WAITERS => StateKind::LockedWaiters,
            _ => StateKind::Full,
        }
    }
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
    unsafe fn locked_load<W: WaitWake>(&self, wait_waker: &W) -> State {
        let mut spin_count = 0;
        let mut state = self.load(Ordering::Acquire);
        loop {
            match state.kind() {
                StateKind::Empty | StateKind::Full => return state,
                StateKind::Locked | StateKind::LockedWaiters => {
                    spin_loop();
                    if spin_count < 32 {
                        spin_count += 1;
                        state = self.load(Ordering::Acquire);
                        continue;
                    }
                    if let Err(v) = self.compare_exchange_weak(
                        state,
                        State::LOCKED_WAITERS,
                        Ordering::Acquire,
                        Ordering::Acquire,
                    ) {
                        state = v;
                        continue;
                    }
                    wait_waker.wait(
                        NonZeroUsize::new_unchecked(self as *const _ as usize),
                        || self.load(Ordering::Acquire) != State::LOCKED_WAITERS,
                    );
                }
            }
        }
    }
    unsafe fn write_start<W: WaitWake>(&self, wait_waker: &W) -> Result<(), State> {
        todo!()
    }
    unsafe fn write_stop<W: WaitWake>(&self, wait_waker: &W, hash: Option<u64>) {
        todo!()
    }
}

pub struct ParallelHashtable<T, W> {
    // states and values are always the same length, which is a power of 2
    states: Box<[AtomicState]>,
    values: Box<[UnsafeCell<MaybeUninit<T>>]>,
    wait_waker: W,
    probe_distance: usize,
}

unsafe impl<T: Sync + Send, W: Sync + Send> Sync for ParallelHashtable<T, W> {}
unsafe impl<T: Sync + Send, W: Sync + Send> Send for ParallelHashtable<T, W> {}

impl<T, W> ParallelHashtable<T, W> {
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
            states: states.into(),
            values: values.into(),
            wait_waker,
            probe_distance,
        }
    }
    pub fn capacity(&self) -> usize {
        self.states.len()
    }
    pub fn probe_distance(&self) -> usize {
        self.probe_distance
    }
    pub fn iter(&self) -> Iter<T> {
        self.into_iter()
    }
}

impl<T: fmt::Debug, W> fmt::Debug for ParallelHashtable<T, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

#[derive(Clone, Debug)]
pub struct NotEnoughSpace;

impl fmt::Display for NotEnoughSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "not enough space left in hashtable")
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
            state.write_stop(wait_waker, Some(hash));
            &*(value_cell.get() as *const T)
        }
    }
}

impl<T, W: WaitWake> Drop for LockedEntry<'_, T, W> {
    fn drop(&mut self) {
        unsafe { self.state.write_stop(self.wait_waker, None) }
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

impl<T, W> ParallelHashtable<T, W> {
    fn probe_sequence(&self, hash: u64) -> iter::Take<ProbeSequence<T>> {
        debug_assert!(self.capacity().is_power_of_two());
        let mask = self.capacity() - 1;
        ProbeSequence {
            index: hash as usize,
            mask,
            states: &self.states,
            values: &self.values,
        }
        .take(self.probe_distance)
    }
}

impl<T, W: WaitWake> ParallelHashtable<T, W> {
    pub fn find<F: FnMut(&T) -> bool>(&self, hash: u64, mut entry_eq: F) -> Option<&T> {
        let expected = State::make_full(hash);
        for (state, value) in self.probe_sequence(hash) {
            unsafe {
                let state = state.locked_load(&self.wait_waker);
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

impl<T, H> IntoIterator for ParallelHashtable<T, H> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            states: self.states.into_vec().into_iter().enumerate(),
            values: self.values,
        }
    }
}

pub struct Iter<'a, T> {
    states: iter::Enumerate<slice::Iter<'a, AtomicState>>,
    values: &'a [UnsafeCell<MaybeUninit<T>>],
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

impl<'a, T, H> IntoIterator for &'a ParallelHashtable<T, H> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        Iter {
            states: self.states.iter().enumerate(),
            values: &self.values,
        }
    }
}
