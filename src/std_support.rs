use crate::{
    array::{Array, ArrayRepr},
    index_vec::{IndexVec, IndexVecExt, IndexVecNonzeroDimension, Indexes},
    parallel::ParallelBuildArray,
    parallel_hash_table::WaitWake,
    HasErrorType,
};
use ahash::RandomState;
use core::{
    convert::TryInto,
    hash::{BuildHasher, Hash, Hasher},
    num::NonZeroUsize,
    ops::Range,
    ptr,
    sync::atomic::{AtomicPtr, Ordering},
};
use plumbing::{bridge_producer_consumer, Producer};
use rayon::iter::{plumbing, IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::{
    boxed::Box,
    sync::{Condvar, Mutex},
    vec::Vec,
};

#[derive(Default)]
struct LockShard {
    mutex: Mutex<()>,
    condition_variable: Condvar,
}

struct LockShards {
    locks: Box<[LockShard; Self::SHARD_COUNT]>,
    hasher: RandomState,
}

impl LockShards {
    const SHARD_COUNT: usize = 1 << 12;
    fn get() -> &'static Self
    where
        Self: Send + Sync,
    {
        static VALUE: AtomicPtr<LockShards> = AtomicPtr::new(ptr::null_mut());
        #[cold]
        fn fill() -> *const LockShards {
            let mut locks = Vec::with_capacity(LockShards::SHARD_COUNT);
            locks.resize_with(LockShards::SHARD_COUNT, LockShard::default);
            let lock_shards = Box::into_raw(
                LockShards {
                    locks: locks.into_boxed_slice().try_into().ok().unwrap(),
                    hasher: RandomState::default(),
                }
                .into(),
            );
            match VALUE.compare_exchange(
                ptr::null_mut(),
                lock_shards,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => lock_shards,
                Err(retval) => unsafe {
                    Box::from_raw(lock_shards);
                    retval
                },
            }
        }
        let mut retval: *const LockShards = VALUE.load(Ordering::Acquire);
        if retval.is_null() {
            retval = fill();
        }
        unsafe { &*retval }
    }
    fn get_shard(&self, key: NonZeroUsize) -> &LockShard {
        let mut hasher = self.hasher.build_hasher();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        &self.locks[(hash % Self::SHARD_COUNT as u64) as usize]
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub struct StdWaitWake;

impl WaitWake for StdWaitWake {
    unsafe fn wait<SC: FnOnce() -> bool>(&self, key: NonZeroUsize, should_cancel: SC) {
        let lock_shard = LockShards::get().get_shard(key);
        let lock = lock_shard.mutex.lock().unwrap();
        if should_cancel() {
            return;
        }
        let _ = lock_shard.condition_variable.wait(lock).unwrap();
    }
    unsafe fn wake_all(&self, key: NonZeroUsize) {
        let lock_shard = LockShards::get().get_shard(key);
        let _lock = lock_shard.mutex.lock().unwrap();
        lock_shard.condition_variable.notify_all();
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub struct RayonParallel<Base: ?Sized>(Base);

impl<Base: HasErrorType + ?Sized> HasErrorType for RayonParallel<Base> {
    type Error = Base::Error;
}

impl<T, Base, const LENGTH: usize> ParallelBuildArray<T, LENGTH, 0> for RayonParallel<Base>
where
    Base: ?Sized + HasErrorType,
    T: ArrayRepr<LENGTH, 0, Repr = T> + Send,
    Self::Error: Send,
{
    fn parallel_build_array<F: Fn(IndexVec<0>) -> Result<T, Self::Error> + Sync>(
        &self,
        f: F,
    ) -> Result<Array<T, LENGTH, 0>, Self::Error> {
        f(IndexVec([])).map(Array)
    }
}

#[derive(Clone, Debug)]
pub struct ParallelIndexes<const ARRAY_LENGTH: usize, const DIMENSION: usize>(
    pub Indexes<ARRAY_LENGTH, DIMENSION>,
);

impl<const ARRAY_LENGTH: usize, const DIMENSION: usize> ParallelIterator
    for ParallelIndexes<ARRAY_LENGTH, DIMENSION>
where
    IndexVec<DIMENSION>: IndexVecExt,
{
    type Item = IndexVec<DIMENSION>;
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge_producer_consumer(self.len(), self, consumer)
    }
    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl<const ARRAY_LENGTH: usize, const DIMENSION: usize> IndexedParallelIterator
    for ParallelIndexes<ARRAY_LENGTH, DIMENSION>
where
    IndexVec<DIMENSION>: IndexVecExt,
{
    fn len(&self) -> usize {
        self.0.len()
    }
    fn drive<C: plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge_producer_consumer(self.len(), self, consumer)
    }
    fn with_producer<CB: plumbing::ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(self)
    }
}

impl<const ARRAY_LENGTH: usize, const DIMENSION: usize> Producer
    for ParallelIndexes<ARRAY_LENGTH, DIMENSION>
where
    IndexVec<DIMENSION>: IndexVecExt,
{
    type Item = IndexVec<DIMENSION>;
    type IntoIter = Indexes<ARRAY_LENGTH, DIMENSION>;
    fn into_iter(self) -> Self::IntoIter {
        self.0
    }
    fn split_at(self, index: usize) -> (Self, Self) {
        assert!(index < self.len());
        let Range { start, end } = self.0.linear_index_range();
        let mid = index + start;
        (
            Self(Indexes::from_linear_index_range(start..mid)),
            Self(Indexes::from_linear_index_range(mid..end)),
        )
    }
}

impl<const ARRAY_LENGTH: usize, const DIMENSION: usize> IntoParallelIterator
    for Indexes<ARRAY_LENGTH, DIMENSION>
where
    IndexVec<DIMENSION>: IndexVecExt,
{
    type Item = IndexVec<DIMENSION>;
    type Iter = ParallelIndexes<ARRAY_LENGTH, DIMENSION>;
    fn into_par_iter(self) -> Self::Iter {
        ParallelIndexes(self)
    }
}

impl<T, Base, const LENGTH: usize, const DIMENSION: usize, const PREV_DIMENSION: usize>
    ParallelBuildArray<T, LENGTH, DIMENSION> for RayonParallel<Base>
where
    Base: ?Sized + HasErrorType,
    T: ArrayRepr<LENGTH, DIMENSION> + Send,
    Option<T>: ArrayRepr<LENGTH, DIMENSION> + Send,
    IndexVec<DIMENSION>: IndexVecNonzeroDimension<PrevDimension = IndexVec<PREV_DIMENSION>>,
    Self::Error: Send,
{
    fn parallel_build_array<F: Fn(IndexVec<DIMENSION>) -> Result<T, Self::Error> + Sync>(
        &self,
        f: F,
    ) -> Result<Array<T, LENGTH, DIMENSION>, Self::Error> {
        let mut retval = Array::<Option<T>, LENGTH, DIMENSION>::build_array(|_| None);
        struct SyncWrapper<T>(T);
        unsafe impl<T> Sync for SyncWrapper<T> {}
        let ptr: SyncWrapper<*mut _> = SyncWrapper(&mut retval.0);
        Indexes::<LENGTH, DIMENSION>::new()
            .into_par_iter()
            .try_for_each(|index| {
                let v = unsafe {
                    &mut *<Option<T> as ArrayRepr<LENGTH, DIMENSION>>::index_ptr_checked(
                        ptr.0, index,
                    )
                };
                assert!(v.is_none());
                *v = Some(f(index)?);
                Ok(())
            })?;
        Ok(Array::build_array(|index| retval[index].take().unwrap()))
    }
}
