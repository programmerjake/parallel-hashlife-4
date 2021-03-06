use super::{HashTableImpl, IndexCell, TryFillStateFailed};
use core::{cell::Cell, num::NonZeroU32, ptr::NonNull};

macro_rules! impl_index_cell {
    ($underlying:ty) => {
        impl IndexCell for Cell<$underlying> {
            const ZERO: Self = Cell::new(0);

            type Underlying = $underlying;

            #[inline(always)]
            fn get_mut(&mut self) -> &mut $underlying {
                self.get_mut()
            }

            #[inline(always)]
            fn get(&self) -> $underlying {
                self.get()
            }

            #[inline(always)]
            fn replace(&self, v: $underlying) -> $underlying {
                self.replace(v)
            }

            #[inline(always)]
            fn new(v: $underlying) -> Self {
                Cell::new(v)
            }

            #[inline(always)]
            fn into_inner(self) -> $underlying {
                self.into_inner()
            }

            #[inline(always)]
            fn set(&self, v: $underlying) {
                self.set(v)
            }
        }
    };
}

impl_index_cell!(u32);
impl_index_cell!(usize);

#[derive(Clone, Copy, Default, Debug)]
pub struct UnsyncHashTableImpl;

const STATE_FIRST_FULL: u32 = 1;

unsafe impl HashTableImpl for UnsyncHashTableImpl {
    type FullState = NonZeroU32;

    type StateCell = Cell<Option<NonZeroU32>>;

    type IndexCellUsize = Cell<usize>;

    type IndexCellU32 = Cell<u32>;

    const STATE_CELL_EMPTY: Self::StateCell = Cell::new(None);

    #[inline(always)]
    fn make_full_state(hash: u64) -> Self::FullState {
        super::make_full_state::<STATE_FIRST_FULL>(hash)
    }

    #[inline(always)]
    fn read_state(state_cell: &Self::StateCell) -> Option<Self::FullState> {
        state_cell.get()
    }

    unsafe fn try_fill_state<T>(
        &self,
        state_cell: &Self::StateCell,
        new_full_state: Self::FullState,
        write_target: NonNull<T>,
        write_value: T,
    ) -> Result<(), TryFillStateFailed<Self::FullState, T>> {
        let state = state_cell.get();
        if let Some(state) = state {
            Err(TryFillStateFailed {
                read_state: state,
                write_value,
            })
        } else {
            write_target.as_ptr().write(write_value);
            state_cell.set(Some(new_full_state));
            Ok(())
        }
    }
}

pub type HashTable<Value> = super::HashTable<UnsyncHashTableImpl, Value>;
