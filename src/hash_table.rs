use alloc::{
    boxed::Box,
    vec::{self, Vec},
};
use core::{
    cell::UnsafeCell,
    debug_assert,
    fmt::{self, Debug},
    hint::unreachable_unchecked,
    iter::{self, FusedIterator},
    marker::PhantomData,
    mem::{self, MaybeUninit},
    num::NonZeroU32,
    ops::Range,
    ptr::NonNull,
    slice,
};
use sync::WaitWake;

pub mod sync;
pub mod unsync;

mod sealed {
    pub trait Sealed {}
}

impl sealed::Sealed for unsync::UnsyncHashTableImpl {}
impl<W: sync::WaitWake> sealed::Sealed for sync::SyncHashTableImpl<W> {}

pub trait IndexCell: 'static + Debug + Sized {
    const ZERO: Self;
    fn get_mut(&mut self) -> &mut usize;
    fn new(v: usize) -> Self;
    fn into_inner(self) -> usize;
    fn get(&self) -> usize;
    fn replace(&self, v: usize) -> usize;
    fn set(&self, v: usize);
}

#[derive(Debug)]
pub struct TryFillStateFailed<FullState, T> {
    pub read_state: FullState,
    pub write_value: T,
}

pub unsafe trait HashTableImpl: sealed::Sealed {
    type FullState: 'static + Debug + Eq + Copy;
    type StateCell: 'static + Debug;
    type IndexCell: IndexCell;
    const STATE_CELL_EMPTY: Self::StateCell;
    fn make_full_state(hash: u64) -> Self::FullState;
    fn read_state(state_cell: &Self::StateCell) -> Option<Self::FullState>;
    fn read_state_nonatomic(state_cell: &mut Self::StateCell) -> Option<Self::FullState> {
        Self::read_state(state_cell)
    }
    unsafe fn try_fill_state<T>(
        &self,
        state_cell: &Self::StateCell,
        new_full_state: Self::FullState,
        write_target: NonNull<T>,
        write_value: T,
    ) -> Result<(), TryFillStateFailed<Self::FullState, T>>;
}

pub unsafe trait LockableHashTableImpl: HashTableImpl {
    unsafe fn lock_state(&self, state_cell: &Self::StateCell) -> Result<(), Self::FullState>;
    unsafe fn unlock_and_fill_state(
        &self,
        state_cell: &Self::StateCell,
        full_state: Self::FullState,
    );
    unsafe fn unlock_and_empty_state(&self, state_cell: &Self::StateCell);
}

#[inline(always)]
fn make_full_state<const STATE_FIRST_FULL: u32>(hash: u64) -> NonZeroU32 {
    let retval = (hash >> 32) as u32;
    if retval >= STATE_FIRST_FULL {
        return NonZeroU32::new(retval).unwrap();
    }
    let retval = hash as u32;
    if retval >= STATE_FIRST_FULL {
        return NonZeroU32::new(retval).unwrap();
    }
    NonZeroU32::new(STATE_FIRST_FULL).unwrap()
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
            slice::from_raw_parts(
                self.state_cells_ptr.as_ptr() as *const StateCell,
                self.capacity,
            ),
            slice::from_raw_parts(self.values_ptr.as_ptr() as *const Value, self.capacity),
        )
    }
    unsafe fn as_mut<StateCell, Value>(&mut self) -> (&mut [StateCell], &mut [Value]) {
        (
            slice::from_raw_parts_mut(
                self.state_cells_ptr.as_ptr() as *mut StateCell,
                self.capacity,
            ),
            slice::from_raw_parts_mut(self.values_ptr.as_ptr() as *mut Value, self.capacity),
        )
    }
    unsafe fn into_boxes<StateCell, Value>(self) -> (Box<[StateCell]>, Box<[Value]>) {
        (
            Box::from_raw(slice::from_raw_parts_mut(
                self.state_cells_ptr.as_ptr() as *mut StateCell,
                self.capacity,
            )),
            Box::from_raw(slice::from_raw_parts_mut(
                self.values_ptr.as_ptr() as *mut Value,
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
                (self.drop_fn)(internal);
            }
        }
    }
}

impl HashTableStorage {
    unsafe fn capacity(&self) -> usize {
        debug_assert!(self.internal.is_some());
        match &self.internal {
            Some(v) => v.capacity,
            None => unreachable_unchecked(),
        }
    }
    unsafe fn drop_fn<HTI: HashTableImpl, Value>(internal: HashTableStorageInternal) {
        let (state_cells, values) =
            internal.into_boxes::<HTI::StateCell, UnsafeCell<MaybeUninit<Value>>>();
        mem::drop(IntoIter::<HTI, Value> {
            state_cells: Vec::from(state_cells).into_iter().enumerate(),
            values,
        });
    }
    unsafe fn new<HTI: HashTableImpl, Value>(
        state_cells: Box<[HTI::StateCell]>,
        values: Box<[UnsafeCell<MaybeUninit<Value>>]>,
    ) -> Self {
        Self {
            internal: Some(HashTableStorageInternal::from_boxes::<
                HTI::StateCell,
                UnsafeCell<MaybeUninit<Value>>,
            >(state_cells, values)),
            drop_fn: Self::drop_fn::<HTI, Value>,
        }
    }
    unsafe fn take<HTI: HashTableImpl, Value>(
        &mut self,
    ) -> (Box<[HTI::StateCell]>, Box<[UnsafeCell<MaybeUninit<Value>>]>) {
        self.internal
            .take()
            .unwrap()
            .into_boxes::<HTI::StateCell, UnsafeCell<MaybeUninit<Value>>>()
    }
    unsafe fn state_cells_values<HTI: HashTableImpl, Value>(
        &self,
    ) -> (&[HTI::StateCell], &[UnsafeCell<MaybeUninit<Value>>]) {
        debug_assert!(self.internal.is_some());
        match &self.internal {
            Some(v) => v.as_ref::<HTI::StateCell, UnsafeCell<MaybeUninit<Value>>>(),
            None => unreachable_unchecked(),
        }
    }
    unsafe fn state_cells_values_mut<HTI: HashTableImpl, Value>(
        &mut self,
    ) -> (&mut [HTI::StateCell], &mut [UnsafeCell<MaybeUninit<Value>>]) {
        self.internal
            .as_mut()
            .unwrap()
            .as_mut::<HTI::StateCell, UnsafeCell<MaybeUninit<Value>>>()
    }
}

pub struct HashTable<HTI: HashTableImpl, Value> {
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

impl<Value, W: WaitWake> sync::HashTable<Value, W> {
    pub fn with_wait_waker(log2_capacity: u32, wait_waker: W) -> Self {
        Self::with_impl(log2_capacity, sync::SyncHashTableImpl { wait_waker })
    }
    pub fn with_wait_waker_and_probe_distance(
        log2_capacity: u32,
        wait_waker: W,
        probe_distance: usize,
    ) -> Self {
        Self::with_impl_and_probe_distance(
            log2_capacity,
            sync::SyncHashTableImpl { wait_waker },
            probe_distance,
        )
    }
}

impl<HTI: HashTableImpl, Value> HashTable<HTI, Value> {
    pub fn with_impl(log2_capacity: u32, hash_table_impl: HTI) -> Self {
        let capacity = 1isize
            .checked_shl(log2_capacity)
            .expect("capacity is too big") as usize;
        Self::with_impl_and_probe_distance(log2_capacity, hash_table_impl, capacity.min(16))
    }
    pub fn new(log2_capacity: u32) -> Self
    where
        HTI: Default,
    {
        Self::with_impl(log2_capacity, HTI::default())
    }
    pub fn with_probe_distance(log2_capacity: u32, probe_distance: usize) -> Self
    where
        HTI: Default,
    {
        Self::with_impl_and_probe_distance(log2_capacity, HTI::default(), probe_distance)
    }
    pub fn with_impl_and_probe_distance(
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
        let mut state_cells = Vec::with_capacity(capacity);
        state_cells.resize_with(capacity, || HTI::STATE_CELL_EMPTY);
        unsafe {
            Self {
                storage: HashTableStorage::new::<HTI, Value>(state_cells.into(), values.into()),
                _phantom: PhantomData,
                hash_table_impl,
                probe_distance,
            }
        }
    }
    pub fn capacity(&self) -> usize {
        unsafe { self.storage.capacity() }
    }
    pub fn probe_distance(&self) -> usize {
        self.probe_distance
    }
    pub fn iter<'a>(&'a self) -> Iter<'a, HTI, Value> {
        self.into_iter()
    }
    pub fn iter_mut<'a>(&'a mut self) -> IterMut<'a, HTI, Value> {
        self.into_iter()
    }
    pub fn hash_table_impl(&self) -> &HTI {
        &self.hash_table_impl
    }
    pub fn hash_table_impl_mut(&mut self) -> &mut HTI {
        &mut self.hash_table_impl
    }
    pub fn index<'a>(&'a self, index: usize) -> Option<&'a Value> {
        unsafe {
            assert!(index < self.capacity());
            self.index_unchecked(index)
        }
    }
    /// # Safety
    /// `index` must be less than `capacity`
    pub unsafe fn index_unchecked<'a>(&'a self, index: usize) -> Option<&'a Value> {
        debug_assert!(index < self.capacity());
        let (state_cells, values) = self.storage.state_cells_values::<HTI, Value>();
        let state_cell = state_cells.get_unchecked(index);
        if HTI::read_state(state_cell).is_some() {
            Some(&*(values[index].get() as *const Value))
        } else {
            None
        }
    }
    pub fn index_mut<'a>(&'a mut self, index: usize) -> Option<&'a mut Value> {
        unsafe {
            let (state_cells, values) = self.storage.state_cells_values_mut::<HTI, Value>();
            let state_cell = state_cells.get_mut(index)?;
            if HTI::read_state_nonatomic(state_cell).is_some() {
                Some(&mut *(values[index].get() as *mut Value))
            } else {
                None
            }
        }
    }
}

impl<HTI: HashTableImpl, Value: fmt::Debug> fmt::Debug for HashTable<HTI, Value> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

pub struct LockedEntry<'a, HTI: LockableHashTableImpl, Value> {
    state_cell: &'a HTI::StateCell,
    value: &'a UnsafeCell<MaybeUninit<Value>>,
    full_state: HTI::FullState,
    hash_table_impl: &'a HTI,
}

impl<'a, HTI: LockableHashTableImpl, Value> LockedEntry<'a, HTI, Value> {
    pub fn fill(self, value: Value) -> &'a Value {
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

impl<HTI: LockableHashTableImpl, Value> Drop for LockedEntry<'_, HTI, Value> {
    fn drop(&mut self) {
        unsafe { self.hash_table_impl.unlock_and_empty_state(self.state_cell) }
    }
}

pub enum LockResult<'a, HTI: LockableHashTableImpl, Value> {
    Vacant(LockedEntry<'a, HTI, Value>),
    Full(&'a Value),
}

struct ProbeSequence<'a, HTI: HashTableImpl, Value> {
    index: usize,
    mask: usize,
    state_cells: &'a [HTI::StateCell],
    values: &'a [UnsafeCell<MaybeUninit<Value>>],
}

impl<'a, HTI: HashTableImpl, Value> Iterator for ProbeSequence<'a, HTI, Value> {
    type Item = (
        usize,
        &'a HTI::StateCell,
        &'a UnsafeCell<MaybeUninit<Value>>,
    );
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index & self.mask;
        self.index += 1;
        Some((index, &self.state_cells[index], &self.values[index]))
    }
}

impl<HTI: HashTableImpl, Value> HashTable<HTI, Value> {
    fn probe_sequence<'a>(&'a self, hash: u64) -> iter::Take<ProbeSequence<'a, HTI, Value>> {
        debug_assert!(self.capacity().is_power_of_two());
        let mask = self.capacity() - 1;
        unsafe {
            let (state_cells, values) = self.storage.state_cells_values::<HTI, Value>();
            ProbeSequence {
                index: hash as usize,
                mask,
                state_cells,
                values,
            }
            .take(self.probe_distance)
        }
    }
}

#[derive(Clone, Debug)]
pub struct NotEnoughSpace;

impl fmt::Display for NotEnoughSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "not enough space left in hash table")
    }
}

#[derive(Debug)]
pub struct GetOrInsertOutput<'a, Value> {
    pub index: usize,
    pub value: &'a Value,
    pub uninserted_value: Option<Value>,
}

impl<HTI: HashTableImpl, Value> HashTable<HTI, Value> {
    pub fn find<'a, F: FnMut(usize, &'a Value) -> bool>(
        &'a self,
        hash: u64,
        mut entry_eq: F,
    ) -> Option<(usize, &'a Value)> {
        let full_state = HTI::make_full_state(hash);
        for (index, state_cell, value) in self.probe_sequence(hash) {
            unsafe {
                let state = HTI::read_state(state_cell);
                if state == Some(full_state) {
                    let value = &*(value.get() as *const Value);
                    if entry_eq(index, value) {
                        return Some((index, value));
                    }
                } else if state.is_none() {
                    break;
                }
            }
        }
        None
    }
    #[inline]
    pub fn get_or_insert<'a, F: FnMut(usize, &'a Value, &Value) -> bool>(
        &'a self,
        hash: u64,
        mut entry_eq: F,
        mut new_value: Value,
    ) -> Result<GetOrInsertOutput<'a, Value>, NotEnoughSpace> {
        let full_state = HTI::make_full_state(hash);
        for (index, state_cell, value) in self.probe_sequence(hash) {
            unsafe {
                match self.hash_table_impl.try_fill_state(
                    state_cell,
                    full_state,
                    NonNull::new_unchecked(value.get() as *mut Value),
                    new_value,
                ) {
                    Ok(()) => {
                        return Ok(GetOrInsertOutput {
                            index,
                            value: &*(value.get() as *const Value),
                            uninserted_value: None,
                        });
                    }
                    Err(TryFillStateFailed {
                        read_state,
                        write_value,
                    }) => {
                        new_value = write_value;
                        if read_state == full_state {
                            let value = &*(value.get() as *const Value);
                            if entry_eq(index, value, &new_value) {
                                return Ok(GetOrInsertOutput {
                                    index,
                                    value,
                                    uninserted_value: Some(new_value),
                                });
                            }
                        }
                    }
                }
            }
        }
        Err(NotEnoughSpace)
    }
    pub fn lock_entry<'a, F: FnMut(usize, &'a Value) -> bool>(
        &'a self,
        hash: u64,
        mut entry_eq: F,
    ) -> Result<(usize, LockResult<'a, HTI, Value>), NotEnoughSpace>
    where
        HTI: LockableHashTableImpl,
    {
        let full_state = HTI::make_full_state(hash);
        for (index, state_cell, value) in self.probe_sequence(hash) {
            unsafe {
                match self.hash_table_impl.lock_state(state_cell) {
                    Ok(()) => {
                        return Ok((
                            index,
                            LockResult::Vacant(LockedEntry {
                                state_cell,
                                value,
                                full_state,
                                hash_table_impl: &self.hash_table_impl,
                            }),
                        ));
                    }
                    Err(state) => {
                        if state == full_state {
                            let value = &*(value.get() as *const Value);
                            if entry_eq(index, value) {
                                return Ok((index, LockResult::Full(value)));
                            }
                        }
                    }
                }
            }
        }
        Err(NotEnoughSpace)
    }
}

pub struct IntoIter<HTI: HashTableImpl, Value> {
    state_cells: iter::Enumerate<vec::IntoIter<HTI::StateCell>>,
    values: Box<[UnsafeCell<MaybeUninit<Value>>]>,
}

unsafe impl<HTI: HashTableImpl, Value> Sync for IntoIter<HTI, Value>
where
    HTI::StateCell: Send + Sync,
    Value: Send + Sync,
{
}

unsafe impl<HTI: HashTableImpl, Value> Send for IntoIter<HTI, Value>
where
    HTI::StateCell: Send + Sync,
    Value: Send + Sync,
{
}

impl<HTI: HashTableImpl, Value> Drop for IntoIter<HTI, Value> {
    fn drop(&mut self) {
        self.for_each(mem::drop);
    }
}

impl<HTI: HashTableImpl, Value> Iterator for IntoIter<HTI, Value> {
    type Item = Value;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((index, mut state_cell)) = self.state_cells.next() {
            if HTI::read_state_nonatomic(&mut state_cell).is_some() {
                unsafe {
                    return Some(self.values[index].get().read().assume_init());
                }
            }
        }
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.state_cells.len()))
    }
}

impl<HTI: HashTableImpl, Value> FusedIterator for IntoIter<HTI, Value> {}

impl<HTI: HashTableImpl, Value> DoubleEndedIterator for IntoIter<HTI, Value> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while let Some((index, mut state_cell)) = self.state_cells.next_back() {
            if HTI::read_state_nonatomic(&mut state_cell).is_some() {
                unsafe {
                    return Some(self.values[index].get().read().assume_init());
                }
            }
        }
        None
    }
}

impl<HTI: HashTableImpl, Value> IntoIterator for HashTable<HTI, Value> {
    type Item = Value;
    type IntoIter = IntoIter<HTI, Value>;
    fn into_iter(mut self) -> Self::IntoIter {
        let (state_cells, values) = unsafe { self.storage.take::<HTI, Value>() };
        IntoIter {
            state_cells: state_cells.into_vec().into_iter().enumerate(),
            values,
        }
    }
}

pub struct Iter<'a, HTI: HashTableImpl, Value> {
    state_cells: iter::Enumerate<slice::Iter<'a, HTI::StateCell>>,
    values: &'a [UnsafeCell<MaybeUninit<Value>>],
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

impl<'a, HTI: HashTableImpl, Value> Iter<'a, HTI, Value> {
    pub fn split(&self) -> (Self, Option<Self>) {
        let (first_half, last_half) = split_iter(self.state_cells.clone());
        (
            Self {
                state_cells: first_half,
                values: self.values,
            },
            last_half.map(|state_cells| Self {
                state_cells,
                values: self.values,
            }),
        )
    }
}

impl<'a, HTI: HashTableImpl, Value> Clone for Iter<'a, HTI, Value> {
    fn clone(&self) -> Self {
        Self {
            state_cells: self.state_cells.clone(),
            values: self.values,
        }
    }
}

unsafe impl<HTI: HashTableImpl, Value> Sync for Iter<'_, HTI, Value>
where
    HTI::StateCell: Send + Sync,
    Value: Send + Sync,
{
}

unsafe impl<HTI: HashTableImpl, Value> Send for Iter<'_, HTI, Value>
where
    HTI::StateCell: Send + Sync,
    Value: Send + Sync,
{
}

impl<'a, HTI: HashTableImpl, Value> Iterator for Iter<'a, HTI, Value> {
    type Item = (usize, &'a Value);
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((index, state_cell)) = self.state_cells.next() {
            unsafe {
                if HTI::read_state(state_cell).is_some() {
                    return Some((index, &*(self.values[index].get() as *const Value)));
                }
            }
        }
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.state_cells.len()))
    }
}

impl<'a, HTI: HashTableImpl, Value> FusedIterator for Iter<'a, HTI, Value> {}

impl<'a, HTI: HashTableImpl, Value> DoubleEndedIterator for Iter<'a, HTI, Value> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while let Some((index, state_cell)) = self.state_cells.next_back() {
            unsafe {
                if HTI::read_state(state_cell).is_some() {
                    return Some((index, &*(self.values[index].get() as *const Value)));
                }
            }
        }
        None
    }
}

impl<'a, HTI: HashTableImpl, Value> IntoIterator for &'a HashTable<HTI, Value> {
    type Item = (usize, &'a Value);
    type IntoIter = Iter<'a, HTI, Value>;
    fn into_iter(self) -> Self::IntoIter {
        unsafe {
            let (state_cells, values) = self.storage.state_cells_values::<HTI, Value>();
            Iter {
                state_cells: state_cells.iter().enumerate(),
                values,
            }
        }
    }
}

pub struct IterMut<'a, HTI: HashTableImpl, Value> {
    range: Range<usize>,
    state_cells: *mut HTI::StateCell,
    values: &'a [UnsafeCell<MaybeUninit<Value>>],
}

impl<'a, HTI: HashTableImpl, Value> IterMut<'a, HTI, Value> {
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

unsafe impl<HTI: HashTableImpl, Value> Sync for IterMut<'_, HTI, Value>
where
    HTI::StateCell: Send + Sync,
    Value: Send + Sync,
{
}

unsafe impl<HTI: HashTableImpl, Value> Send for IterMut<'_, HTI, Value>
where
    HTI::StateCell: Send + Sync,
    Value: Send + Sync,
{
}

impl<'a, HTI: HashTableImpl, Value> Iterator for IterMut<'a, HTI, Value> {
    type Item = (usize, &'a mut Value);
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(index) = self.range.next() {
            let state_cell = unsafe { &mut *self.state_cells.offset(index as isize) };
            if HTI::read_state_nonatomic(state_cell).is_some() {
                unsafe {
                    return Some((index, &mut *(self.values[index].get() as *mut Value)));
                }
            }
        }
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.range.len()))
    }
}

impl<'a, HTI: HashTableImpl, Value> FusedIterator for IterMut<'a, HTI, Value> {}

impl<'a, HTI: HashTableImpl, Value> DoubleEndedIterator for IterMut<'a, HTI, Value> {
    fn next_back(&mut self) -> Option<Self::Item> {
        while let Some(index) = self.range.next_back() {
            let state_cell = unsafe { &mut *self.state_cells.offset(index as isize) };
            if HTI::read_state_nonatomic(state_cell).is_some() {
                unsafe {
                    return Some((index, &mut *(self.values[index].get() as *mut Value)));
                }
            }
        }
        None
    }
}

impl<'a, HTI: HashTableImpl, Value> IntoIterator for &'a mut HashTable<HTI, Value> {
    type Item = (usize, &'a mut Value);
    type IntoIter = IterMut<'a, HTI, Value>;
    fn into_iter(self) -> Self::IntoIter {
        let (state_cells, values) = unsafe { self.storage.state_cells_values_mut::<HTI, Value>() };
        IterMut {
            range: 0..state_cells.len(),
            state_cells: state_cells.as_mut_ptr(),
            values,
        }
    }
}

#[cfg(test)]
mod test {
    use super::{
        sync::{self, WaitWake},
        unsync, GetOrInsertOutput, LockResult,
    };
    use crate::std_support::StdWaitWake;
    use alloc::{sync::Arc, vec::Vec};
    use core::{
        cell::{Cell, RefCell},
        hash::{BuildHasher, Hash, Hasher},
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
        let ht: sync::HashTable<EntryWithLifetime<'_>> = sync::HashTable::new(2);
        let hasher = DefaultHashBuilder::new();
        let hasher = |entry: &EntryWithLifetime<'_>| -> u64 {
            let mut hasher = hasher.build_hasher();
            entry.hash(&mut hasher);
            hasher.finish()
        };
        let entry1 = EntryWithLifetime(Entry::new(), RefCell::new(None));
        let entry_ref1 = match ht.lock_entry(hasher(&entry1), |_index, v| *v == entry1) {
            Ok((_index, LockResult::Vacant(locked_entry))) => {
                let v = locked_entry.fill(entry1.clone());
                assert_eq!(*v, entry1);
                v
            }
            Ok((_index, LockResult::Full(_))) => unreachable!(),
            Err(super::NotEnoughSpace) => unreachable!(),
        };
        let entry2 = EntryWithLifetime(Entry::new(), RefCell::new(None));
        let entry_ref2 = match ht.lock_entry(hasher(&entry2), |_index, v| *v == entry2) {
            Ok((_index, LockResult::Vacant(locked_entry))) => {
                let v = locked_entry.fill(entry2.clone());
                assert_eq!(*v, entry2);
                v
            }
            Ok((_index, LockResult::Full(_))) => unreachable!(),
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
        let ht: sync::HashTable<(Entry, DropTestHelper<'_>)> = sync::HashTable::new(6);
        let hasher = DefaultHashBuilder::new();
        let hasher = |entry: &Entry| -> u64 {
            let mut hasher = hasher.build_hasher();
            entry.hash(&mut hasher);
            hasher.finish()
        };
        let entry1 = Entry::new();
        match ht.lock_entry(hasher(&entry1), |_index, v| v.0 == entry1) {
            Ok((_index, LockResult::Vacant(locked_entry))) => {
                locked_entry.fill((entry1.clone(), DropTestHelper(&item1_count)))
            }
            Ok((_index, LockResult::Full(_))) => unreachable!(),
            Err(super::NotEnoughSpace) => unreachable!(),
        };
        let entry2 = Entry::new();
        match ht.lock_entry(hasher(&entry2), |_index, v| v.0 == entry2) {
            Ok((_index, LockResult::Vacant(locked_entry))) => {
                locked_entry.fill((entry2.clone(), DropTestHelper(&item2_count)))
            }
            Ok((_index, LockResult::Full(_))) => unreachable!(),
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
        let ht: sync::HashTable<Entry> = sync::HashTable::new(6);
        let hasher = DefaultHashBuilder::new();
        let hasher = |entry: &Entry| -> u64 {
            let mut hasher = hasher.build_hasher();
            entry.hash(&mut hasher);
            hasher.finish()
        };
        let mut reference_ht = HashSet::<Entry>::new();
        for i in 0..100000u64 {
            match i.wrapping_mul(0xA331_ABB2_E016_BC0A_u64) >> 62 {
                1 => {
                    let entry = Entry::new();
                    match ht.lock_entry(hasher(&entry), |_index, v| *v == entry) {
                        Ok((_index, LockResult::Vacant(locked_entry))) => {
                            assert_eq!(*locked_entry.fill(entry.clone()), entry);
                            assert!(
                                reference_ht.insert(entry.clone()),
                                "failed to insert {:?}",
                                entry
                            );
                        }
                        Ok((_index, LockResult::Full(old_entry))) => {
                            assert_eq!(reference_ht.get(&entry), Some(old_entry));
                        }
                        Err(super::NotEnoughSpace) => {
                            assert!(reference_ht.len() >= ht.probe_distance());
                        }
                    }
                }
                2 => {
                    let entry = Entry::new();
                    match ht.get_or_insert(
                        hasher(&entry),
                        |_index, v, entry| v == entry,
                        entry.clone(),
                    ) {
                        Ok(GetOrInsertOutput {
                            index: _,
                            value: filled_entry,
                            uninserted_value: None,
                        }) => {
                            assert_eq!(*filled_entry, entry);
                            assert!(
                                reference_ht.insert(entry.clone()),
                                "failed to insert {:?}",
                                entry
                            );
                        }
                        Ok(GetOrInsertOutput {
                            index: _,
                            value: old_entry,
                            uninserted_value: Some(new_entry),
                        }) => {
                            assert_eq!(reference_ht.get(&entry), Some(old_entry));
                            assert_eq!(entry, new_entry);
                        }
                        Err(super::NotEnoughSpace) => {
                            assert!(reference_ht.len() >= ht.probe_distance());
                        }
                    }
                }
                _ => {
                    for _ in 0..10 {
                        let entry = Entry::new();
                        assert_eq!(
                            ht.find(hasher(&entry), |_index, v| *v == entry)
                                .map(|v| v.1),
                            reference_ht.get(&entry)
                        );
                    }
                }
            }
        }
        let entries: Vec<Entry> = ht.iter().map(|v| v.1).cloned().collect();
        let entries2: Vec<Entry> = ht.into_iter().collect();
        assert_eq!(entries, entries2);
    }

    #[test]
    fn test2() {
        let ht: unsync::HashTable<Entry> = unsync::HashTable::new(6);
        let hasher = DefaultHashBuilder::new();
        let hasher = |entry: &Entry| -> u64 {
            let mut hasher = hasher.build_hasher();
            entry.hash(&mut hasher);
            hasher.finish()
        };
        let mut reference_ht = HashSet::<Entry>::new();
        for i in 0..100000u64 {
            match i.wrapping_mul(0xA331_ABB2_E016_BC0A_u64) >> 62 {
                1 => {
                    let entry = Entry::new();
                    match ht.get_or_insert(
                        hasher(&entry),
                        |_index, v, entry| v == entry,
                        entry.clone(),
                    ) {
                        Ok(GetOrInsertOutput {
                            index: _,
                            value: filled_entry,
                            uninserted_value: None,
                        }) => {
                            assert_eq!(*filled_entry, entry);
                            assert!(
                                reference_ht.insert(entry.clone()),
                                "failed to insert {:?}",
                                entry
                            );
                        }
                        Ok(GetOrInsertOutput {
                            index: _,
                            value: old_entry,
                            uninserted_value: Some(new_entry),
                        }) => {
                            assert_eq!(reference_ht.get(&entry), Some(old_entry));
                            assert_eq!(entry, new_entry);
                        }
                        Err(super::NotEnoughSpace) => {
                            assert!(reference_ht.len() >= ht.probe_distance());
                        }
                    }
                }
                _ => {
                    for _ in 0..10 {
                        let entry = Entry::new();
                        assert_eq!(
                            ht.find(hasher(&entry), |_index, v| *v == entry)
                                .map(|v| v.1),
                            reference_ht.get(&entry)
                        );
                    }
                }
            }
        }
        let entries: Vec<Entry> = ht.iter().map(|v| v.1).cloned().collect();
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
        let ht = Arc::new(sync::HashTable::with_wait_waker(
            8,
            WaitWakeTracker::new(StdWaitWake),
        ));
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
                            match ht
                                .lock_entry(hasher(&entry), |_index, v| *v == entry)
                                .unwrap()
                                .1
                            {
                                LockResult::Vacant(lock) => lock.fill(entry),
                                LockResult::Full(_) => {
                                    panic!("failed to lock already full entry: {:?}", entry);
                                }
                            };
                        }
                        Op::AttemptInsertIntoFull(entry) => {
                            match ht
                                .lock_entry(hasher(&entry), |_index, v| *v == entry)
                                .unwrap()
                                .1
                            {
                                LockResult::Vacant(_) => panic!("expected full entry: {:?}", entry),
                                LockResult::Full(v) => assert_eq!(v, &entry),
                            };
                        }
                        Op::LockEntry(entry) => {
                            let lock = match ht
                                .lock_entry(hasher(&entry), |_index, v| *v == entry)
                                .unwrap()
                                .1
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
            let actual: HashSet<_> = ht.iter().map(|v| v.1).collect();
            assert_eq!(expected, actual);
        };
        let wait_for_thread_to_wait = |thread_index: u32| {
            let wait_waker = &ht.hash_table_impl().wait_waker;
            let _ = wait_waker
                .wait_set_changed
                .wait_while(wait_waker.wait_set.lock().unwrap(), |v| {
                    !v.contains_key(&ThreadIndex(thread_index))
                })
                .unwrap();
        };
        let assert_find = |entry, expected| {
            assert_eq!(
                ht.find(hasher(entry), |_index, v| v == entry).map(|v| v.1),
                expected
            )
        };
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
