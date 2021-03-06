use super::{HashTableImpl, IndexCell, LockableHashTableImpl, TryFillStateFailed};
use core::{
    hint::spin_loop,
    num::{NonZeroU32, NonZeroUsize},
    ptr::NonNull,
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

const STATE_EMPTY: u32 = 0;
const STATE_LOCKED: u32 = STATE_EMPTY + 1;
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

macro_rules! impl_index_cell {
    ($underlying:ty, $atomic_underlying:ty) => {
        impl IndexCell for $atomic_underlying {
            const ZERO: Self = <$atomic_underlying>::new(0);

            type Underlying = $underlying;

            #[inline(always)]
            fn get_mut(&mut self) -> &mut $underlying {
                self.get_mut()
            }

            #[inline(always)]
            fn get(&self) -> $underlying {
                self.load(Ordering::Acquire)
            }

            #[inline(always)]
            fn replace(&self, v: $underlying) -> $underlying {
                self.swap(v, Ordering::AcqRel)
            }

            #[inline(always)]
            fn new(v: $underlying) -> Self {
                <$atomic_underlying>::new(v)
            }

            #[inline(always)]
            fn into_inner(self) -> $underlying {
                self.into_inner()
            }

            #[inline(always)]
            fn set(&self, v: $underlying) {
                self.store(v, Ordering::Release)
            }
        }
    };
}

impl_index_cell!(u32, AtomicU32);
impl_index_cell!(usize, AtomicUsize);

impl<W: WaitWake> SyncHashTableImpl<W> {
    #[cold]
    unsafe fn lock_state_slow(
        &self,
        state_cell: &<Self as HashTableImpl>::StateCell,
        mut state: u32,
    ) -> Result<(), <Self as HashTableImpl>::FullState> {
        let mut spin_count = 0;
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
    #[cold]
    unsafe fn unlock_state_slow(&self, state_cell: &<Self as HashTableImpl>::StateCell) {
        self.wait_waker
            .wake_all(NonZeroUsize::new_unchecked(state_cell as *const _ as usize));
    }
}

unsafe impl<W: WaitWake> HashTableImpl for SyncHashTableImpl<W> {
    type FullState = NonZeroU32;

    type StateCell = AtomicU32;

    type IndexCellUsize = AtomicUsize;

    type IndexCellU32 = AtomicU32;

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

    fn read_state_non_atomic(state_cell: &mut Self::StateCell) -> Option<Self::FullState> {
        let retval = *state_cell.get_mut();
        if retval >= STATE_FIRST_FULL {
            Some(NonZeroU32::new(retval).unwrap())
        } else {
            None
        }
    }

    unsafe fn try_fill_state<T>(
        &self,
        state_cell: &Self::StateCell,
        new_full_state: Self::FullState,
        write_target: NonNull<T>,
        write_value: T,
    ) -> Result<(), TryFillStateFailed<Self::FullState, T>> {
        match self.lock_state(state_cell) {
            Ok(()) => {
                write_target.as_ptr().write(write_value);
                self.unlock_and_fill_state(state_cell, new_full_state);
                Ok(())
            }
            Err(read_state) => Err(TryFillStateFailed {
                read_state,
                write_value,
            }),
        }
    }
}

unsafe impl<W: WaitWake> LockableHashTableImpl for SyncHashTableImpl<W> {
    unsafe fn lock_state(&self, state_cell: &Self::StateCell) -> Result<(), Self::FullState> {
        let state = state_cell.load(Ordering::Acquire);
        if state >= STATE_FIRST_FULL {
            Err(NonZeroU32::new(state).unwrap())
        } else if state != STATE_EMPTY {
            self.lock_state_slow(state_cell, state)
        } else if let Err(state) = state_cell.compare_exchange_weak(
            state,
            STATE_LOCKED,
            Ordering::Acquire,
            Ordering::Acquire,
        ) {
            self.lock_state_slow(state_cell, state)
        } else {
            Ok(())
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
            self.unlock_state_slow(state_cell);
        } else {
            debug_assert_eq!(state, STATE_LOCKED);
        }
    }

    unsafe fn unlock_and_empty_state(&self, state_cell: &Self::StateCell) {
        let state = state_cell.swap(STATE_EMPTY, Ordering::Release);
        if state == STATE_LOCKED_WAITERS {
            self.unlock_state_slow(state_cell);
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
