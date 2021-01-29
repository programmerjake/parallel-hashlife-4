use alloc::{boxed::Box, vec, vec::Vec};
use core::{
    cell::{Cell, UnsafeCell},
    fmt, iter,
    iter::FusedIterator,
    marker::PhantomData,
    mem,
    mem::MaybeUninit,
    ops::{Deref, Range},
    ptr::NonNull,
    slice,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
struct State(u32);

impl State {
    const EMPTY: Self = Self(0);
    const LOCKED: Self = Self(1);
    const FIRST_FULL: Self = Self(2);
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
struct CellState(Cell<State>);

impl CellState {
    const fn new(s: State) -> Self {
        CellState(Cell::new(s))
    }
    fn load(&self) -> State {
        self.0.get()
    }
    fn into_inner(self) -> State {
        self.0.into_inner()
    }
    fn get_mut(&mut self) -> &mut State {
        self.0.get_mut()
    }
    fn swap(&self, new: State) -> State {
        self.0.replace(new)
    }
    fn write_start(&self) -> Result<(), State> {
        let state = self.load();
        if state.is_full() {
            return Err(state);
        } else {
            assert!(
                self.swap(State::LOCKED).is_empty(),
                "attempt to lock already locked SerialHashTable entry"
            );
            Ok(())
        }
    }
    fn write_cancel(&self) {
        let state = self.swap(State::EMPTY);
        debug_assert_eq!(state, State::LOCKED);
    }
    fn write_finish(&self, hash: u64) {
        let state = self.swap(State::make_full(hash));
        debug_assert_eq!(state, State::LOCKED);
    }
}

/// use separate non-generic struct to work around drop check
struct HashTableStorage {
    states: Box<[CellState]>,
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
        // safety: must work correctly when self.states and self.values are left empty by `SerialHashTable::into_iter()`
        mem::drop(IntoIter {
            states: states.into_vec().into_iter().enumerate(),
            values,
        });
    }
    fn new<T>(states: Box<[CellState]>, values: Box<[UnsafeCell<MaybeUninit<T>>]>) -> Self {
        assert_eq!(states.len(), values.len());
        unsafe {
            Self {
                states,
                values: NonNull::new_unchecked(Box::into_raw(values) as *mut ()),
                drop_fn: Self::drop_fn::<T>,
            }
        }
    }
    unsafe fn take<T>(&mut self) -> (Box<[CellState]>, Box<[UnsafeCell<MaybeUninit<T>>]>) {
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
    ) -> (&mut [CellState], &mut [UnsafeCell<MaybeUninit<T>>]) {
        let values = slice::from_raw_parts_mut(
            self.values.as_ptr() as *mut UnsafeCell<MaybeUninit<T>>,
            self.states.len(),
        );
        (&mut self.states, values)
    }
}

pub struct SerialHashTable<T> {
    storage: HashTableStorage,
    _phantom: PhantomData<T>,
    probe_distance: usize,
}

impl<T> SerialHashTable<T> {
    pub fn new(log2_capacity: u32) -> Self {
        let capacity = 1isize
            .checked_shl(log2_capacity)
            .expect("capacity is too big") as usize;
        Self::with_probe_distance(log2_capacity, capacity.min(16))
    }
    pub fn with_probe_distance(log2_capacity: u32, probe_distance: usize) -> Self {
        let capacity = 1isize
            .checked_shl(log2_capacity)
            .expect("capacity is too big") as usize;
        assert!(probe_distance <= capacity);
        let mut values = Vec::with_capacity(capacity);
        values.resize_with(capacity, || UnsafeCell::new(MaybeUninit::uninit()));
        let mut states = Vec::with_capacity(capacity);
        states.resize_with(capacity, || CellState::new(State::EMPTY));
        Self {
            storage: HashTableStorage::new::<T>(states.into(), values.into()),
            _phantom: PhantomData,
            probe_distance,
        }
    }
    pub fn capacity(&self) -> usize {
        self.storage.states.len()
    }
    pub fn probe_distance(&self) -> usize {
        self.probe_distance
    }
    pub fn iter<'a>(&'a self) -> Iter<'a, T> {
        self.into_iter()
    }
    pub fn iter_mut<'a>(&'a mut self) -> IterMut<'a, T> {
        self.into_iter()
    }
}

impl<T: fmt::Debug> fmt::Debug for SerialHashTable<T> {
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

pub struct LockedEntry<'a, T> {
    state: &'a CellState,
    value: &'a UnsafeCell<MaybeUninit<T>>,
    hash: u64,
}

impl<'a, T> LockedEntry<'a, T> {
    pub fn fill(self, value: T) -> &'a T {
        let state = self.state;
        let value_cell = self.value;
        let hash = self.hash;
        mem::forget(self);
        unsafe {
            value_cell.get().write(MaybeUninit::new(value));
            state.write_finish(hash);
            &*(value_cell.get() as *const T)
        }
    }
}

impl<T> Drop for LockedEntry<'_, T> {
    fn drop(&mut self) {
        self.state.write_cancel();
    }
}

pub enum LockResult<'a, T> {
    Vacant(LockedEntry<'a, T>),
    Full(&'a T),
}

struct ProbeSequence<'a, T> {
    index: usize,
    mask: usize,
    states: &'a [CellState],
    values: &'a [UnsafeCell<MaybeUninit<T>>],
}

impl<'a, T> Iterator for ProbeSequence<'a, T> {
    type Item = (&'a CellState, &'a UnsafeCell<MaybeUninit<T>>);
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index & self.mask;
        self.index += 1;
        Some((&self.states[index], &self.values[index]))
    }
}

impl<T> SerialHashTable<T> {
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

impl<T> SerialHashTable<T> {
    pub fn find<'a, F: FnMut(&'a T) -> bool>(
        &'a self,
        hash: u64,
        mut entry_eq: F,
    ) -> Option<&'a T> {
        let expected = State::make_full(hash);
        for (state, value) in self.probe_sequence(hash) {
            unsafe {
                let state = state.load();
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
    pub fn lock_entry<'a, F: FnMut(&'a T) -> bool>(
        &'a self,
        hash: u64,
        mut entry_eq: F,
    ) -> Result<LockResult<'a, T>, NotEnoughSpace> {
        let expected = State::make_full(hash);
        for (state, value) in self.probe_sequence(hash) {
            unsafe {
                match state.write_start() {
                    Ok(()) => {
                        return Ok(LockResult::Vacant(LockedEntry { state, value, hash }));
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
    states: iter::Enumerate<vec::IntoIter<CellState>>,
    values: Box<[UnsafeCell<MaybeUninit<T>>]>,
}

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

impl<T> IntoIterator for SerialHashTable<T> {
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
    states: iter::Enumerate<slice::Iter<'a, CellState>>,
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

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((index, state)) = self.states.next() {
            unsafe {
                if state.load().is_full() {
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
                if state.load().is_full() {
                    return Some(&*(self.values[index].get() as *const T));
                }
            }
        }
        None
    }
}

impl<'a, T> IntoIterator for &'a SerialHashTable<T> {
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
    state: &'a mut CellState,
    value: &'a UnsafeCell<MaybeUninit<T>>,
}

impl<'a, T> EntryMut<'a, T> {
    pub fn remove(self) -> T {
        *self.state = CellState::new(State::EMPTY);
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
    states: *mut CellState,
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

impl<'a, T> IntoIterator for &'a mut SerialHashTable<T> {
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
    use super::{LockResult, SerialHashTable};
    use alloc::vec::Vec;
    use core::{
        cell::RefCell,
        hash::{BuildHasher, Hash, Hasher},
        mem,
        sync::atomic::{AtomicU8, AtomicUsize, Ordering},
    };
    use hashbrown::{hash_map::DefaultHashBuilder, HashSet};

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
        let ht: SerialHashTable<EntryWithLifetime<'_>> = SerialHashTable::new(2);
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
        let ht: SerialHashTable<(Entry, DropTestHelper<'_>)> = SerialHashTable::new(6);
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
        let mut ht: SerialHashTable<Entry> = SerialHashTable::new(6);
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
}
