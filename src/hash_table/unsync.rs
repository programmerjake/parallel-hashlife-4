use super::{HashTableImpl, IndexCell, TryFillStateFailed, STATE_EMPTY, STATE_LOCKED};
use core::{cell::Cell, num::NonZeroU32, ptr::NonNull};

const STATE_FIRST_FULL: u32 = STATE_LOCKED + 1;

impl IndexCell for Cell<usize> {
    const ZERO: Self = Cell::new(0);

    #[inline(always)]
    fn get_mut(&mut self) -> &mut usize {
        self.get_mut()
    }

    #[inline(always)]
    fn get(&self) -> usize {
        self.get()
    }

    #[inline(always)]
    fn replace(&self, v: usize) -> usize {
        self.replace(v)
    }

    #[inline(always)]
    fn new(v: usize) -> Self {
        Cell::new(v)
    }

    #[inline(always)]
    fn into_inner(self) -> usize {
        self.into_inner()
    }

    #[inline(always)]
    fn set(&self, v: usize) {
        self.set(v)
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct UnsyncHashTableImpl;

unsafe impl HashTableImpl for UnsyncHashTableImpl {
    type FullState = NonZeroU32;

    type StateCell = Cell<u32>;

    type IndexCell = Cell<usize>;

    const STATE_CELL_EMPTY: Self::StateCell = Cell::new(STATE_EMPTY);

    #[inline(always)]
    fn make_full_state(hash: u64) -> Self::FullState {
        super::make_full_state::<STATE_FIRST_FULL>(hash)
    }

    #[inline(always)]
    fn read_state(state_cell: &Self::StateCell) -> Option<Self::FullState> {
        let retval = state_cell.get();
        if retval >= STATE_FIRST_FULL {
            Some(NonZeroU32::new(retval).unwrap())
        } else {
            None
        }
    }

    unsafe fn lock_state(&self, state_cell: &Self::StateCell) -> Result<(), Self::FullState> {
        let state = state_cell.get();
        if state >= STATE_FIRST_FULL {
            Err(NonZeroU32::new(state).unwrap())
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
        debug_assert!(full_state.get() >= STATE_FIRST_FULL);
        let state = state_cell.replace(full_state.get());
        debug_assert_eq!(state, STATE_LOCKED);
    }

    unsafe fn unlock_and_empty_state(&self, state_cell: &Self::StateCell) {
        let state = state_cell.replace(STATE_EMPTY);
        debug_assert_eq!(state, STATE_LOCKED);
    }

    unsafe fn try_fill_state<T>(
        &self,
        state_cell: &Self::StateCell,
        new_full_state: Self::FullState,
        write_target: NonNull<T>,
        write_value: T,
    ) -> Result<(), TryFillStateFailed<Self::FullState, T>> {
        let state = state_cell.get();
        if state >= STATE_FIRST_FULL {
            Err(TryFillStateFailed {
                read_state: NonZeroU32::new(state).unwrap(),
                write_value,
            })
        } else {
            assert_eq!(
                state, STATE_EMPTY,
                "attempt to lock already locked UnsyncHashTable entry"
            );
            write_target.as_ptr().write(write_value);
            state_cell.set(new_full_state.get());
            Ok(())
        }
    }
}

pub type HashTable<Value> = super::HashTable<UnsyncHashTableImpl, Value>;
