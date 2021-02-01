use super::{HashTableImpl, IndexCell, STATE_EMPTY, STATE_LOCKED};
use core::{
    hint::spin_loop,
    num::{NonZeroU32, NonZeroUsize},
    sync::atomic::{AtomicU32, AtomicUsize, Ordering},
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

const STATE_LOCKED_WAITERS: u32 = STATE_LOCKED + 1;
const STATE_FIRST_FULL: u32 = STATE_LOCKED_WAITERS + 1;

#[cfg(any(test, feature = "std"))]
#[derive(Clone, Copy, Default, Debug)]
pub struct SyncHashTableImpl<W: WaitWake = crate::std_support::StdWaitWake> {
    pub wait_waker: W,
}

#[cfg(not(any(test, feature = "std")))]
#[derive(Clone, Copy, Default, Debug)]
pub struct SyncHashTableImpl<W: WaitWake> {
    pub wait_waker: W,
}

impl IndexCell for AtomicUsize {
    const ZERO: Self = AtomicUsize::new(0);

    fn get_mut(&mut self) -> &mut usize {
        self.get_mut()
    }

    fn get(&self) -> usize {
        self.load(Ordering::Acquire)
    }

    fn replace(&self, v: usize) -> usize {
        self.swap(v, Ordering::AcqRel)
    }

    fn new(v: usize) -> Self {
        AtomicUsize::new(v)
    }

    fn into_inner(self) -> usize {
        self.into_inner()
    }

    fn set(&self, v: usize) {
        self.store(v, Ordering::Release)
    }
}

unsafe impl<W: WaitWake> HashTableImpl for SyncHashTableImpl<W> {
    type FullState = NonZeroU32;

    type StateCell = AtomicU32;

    type IndexCell = AtomicUsize;

    const STATE_CELL_EMPTY: Self::StateCell = AtomicU32::new(STATE_EMPTY);

    fn make_full_state(hash: u64) -> Self::FullState {
        super::make_full_state::<STATE_FIRST_FULL>(hash)
    }

    fn read_state(state_cell: &Self::StateCell) -> Option<Self::FullState> {
        let retval = state_cell.load(Ordering::Acquire);
        if retval >= STATE_FIRST_FULL {
            Some(NonZeroU32::new(retval).unwrap())
        } else {
            None
        }
    }

    fn read_state_nonatomic(state_cell: &mut Self::StateCell) -> Option<Self::FullState> {
        let retval = *state_cell.get_mut();
        if retval >= STATE_FIRST_FULL {
            Some(NonZeroU32::new(retval).unwrap())
        } else {
            None
        }
    }

    unsafe fn lock_state(&self, state_cell: &Self::StateCell) -> Result<(), Self::FullState> {
        let mut spin_count = 0;
        let mut state = state_cell.load(Ordering::Acquire);
        loop {
            if state >= STATE_FIRST_FULL {
                return Err(NonZeroU32::new(state).unwrap());
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
        debug_assert!(full_state.get() >= STATE_FIRST_FULL);
        let state = state_cell.swap(full_state.get(), Ordering::Release);
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

#[cfg(any(test, feature = "std"))]
pub type HashTable<Value, W = crate::std_support::StdWaitWake> =
    super::HashTable<SyncHashTableImpl<W>, Value>;
#[cfg(not(any(test, feature = "std")))]
pub type HashTable<Value, W> = super::HashTable<SyncHashTableImpl<W>, Value>;
