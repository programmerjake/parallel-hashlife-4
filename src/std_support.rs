use crate::{
    array::{Array, ArrayRepr},
    hash_table::{self, sync::WaitWake, HashTableImpl, NotEnoughSpace},
    index_vec::{IndexVec, IndexVecExt, Indexes},
    traits::parallel::ParallelBuildArray,
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
use rayon::iter::{
    plumbing, plumbing::Producer, IndexedParallelIterator, IntoParallelIterator, ParallelIterator,
};
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

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Default)]
pub struct RayonParallel;

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
        plumbing::bridge_producer_consumer(self.len(), self, consumer)
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
        plumbing::bridge_producer_consumer(self.len(), self, consumer)
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

unsafe impl<T, Error, const LENGTH: usize, const DIMENSION: usize>
    ParallelBuildArray<T, Error, LENGTH, DIMENSION> for RayonParallel
where
    T: ArrayRepr<LENGTH, DIMENSION> + Send,
    Option<T>: ArrayRepr<LENGTH, DIMENSION> + Send,
    IndexVec<DIMENSION>: IndexVecExt,
    Error: Send,
{
    fn parallel_build_array<F: Fn(IndexVec<DIMENSION>) -> Result<T, Error> + Sync>(
        &self,
        f: F,
    ) -> Result<Array<T, LENGTH, DIMENSION>, Error> {
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

pub struct HashTableParIter<'a, HTI: HashTableImpl, Value>(pub hash_table::Iter<'a, HTI, Value>);

impl<'a, HTI: HashTableImpl, Value> Clone for HashTableParIter<'a, HTI, Value> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<'a, HTI: HashTableImpl, Value> ParallelIterator for HashTableParIter<'a, HTI, Value>
where
    hash_table::Iter<'a, HTI, Value>: Send,
    &'a Value: Send,
{
    type Item = (usize, &'a Value);
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: plumbing::UnindexedConsumer<Self::Item>,
    {
        plumbing::bridge_unindexed(self, consumer)
    }
}

impl<'a, HTI: HashTableImpl, Value> plumbing::UnindexedProducer for HashTableParIter<'a, HTI, Value>
where
    hash_table::Iter<'a, HTI, Value>: Send,
{
    type Item = (usize, &'a Value);
    fn split(self) -> (Self, Option<Self>) {
        let (first_half, last_half) = self.0.split();
        (Self(first_half), last_half.map(Self))
    }
    fn fold_with<F>(self, folder: F) -> F
    where
        F: plumbing::Folder<Self::Item>,
    {
        folder.consume_iter(self.0)
    }
}

impl<'a, HTI: HashTableImpl, Value> IntoParallelIterator for &'a hash_table::HashTable<HTI, Value>
where
    hash_table::Iter<'a, HTI, Value>: Send,
    &'a Value: Send,
{
    type Iter = HashTableParIter<'a, HTI, Value>;
    type Item = (usize, &'a Value);
    fn into_par_iter(self) -> Self::Iter {
        HashTableParIter(self.iter())
    }
}

impl<'a, HTI: HashTableImpl, Value> IntoParallelIterator for hash_table::Iter<'a, HTI, Value>
where
    Self: Send,
    &'a Value: Send,
{
    type Iter = HashTableParIter<'a, HTI, Value>;
    type Item = (usize, &'a Value);
    fn into_par_iter(self) -> Self::Iter {
        HashTableParIter(self)
    }
}

pub struct HashTableParIterMut<'a, HTI: HashTableImpl, Value>(
    pub hash_table::IterMut<'a, HTI, Value>,
);

impl<'a, HTI: HashTableImpl, Value> ParallelIterator for HashTableParIterMut<'a, HTI, Value>
where
    hash_table::IterMut<'a, HTI, Value>: Send,
    Value: Send,
{
    type Item = (usize, &'a mut Value);
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: plumbing::UnindexedConsumer<Self::Item>,
    {
        plumbing::bridge_unindexed(self, consumer)
    }
}

impl<'a, HTI: HashTableImpl, Value> plumbing::UnindexedProducer
    for HashTableParIterMut<'a, HTI, Value>
where
    hash_table::IterMut<'a, HTI, Value>: Send,
{
    type Item = (usize, &'a mut Value);
    fn split(self) -> (Self, Option<Self>) {
        let (first_half, last_half) = self.0.split();
        (Self(first_half), last_half.map(Self))
    }
    fn fold_with<F>(self, folder: F) -> F
    where
        F: plumbing::Folder<Self::Item>,
    {
        folder.consume_iter(self.0)
    }
}

impl<'a, HTI: HashTableImpl, Value> IntoParallelIterator
    for &'a mut hash_table::HashTable<HTI, Value>
where
    hash_table::IterMut<'a, HTI, Value>: Send,
    Value: Send,
{
    type Iter = HashTableParIterMut<'a, HTI, Value>;
    type Item = (usize, &'a mut Value);
    fn into_par_iter(self) -> Self::Iter {
        HashTableParIterMut(self.iter_mut())
    }
}

impl<'a, HTI: HashTableImpl, Value> IntoParallelIterator for hash_table::IterMut<'a, HTI, Value>
where
    hash_table::IterMut<'a, HTI, Value>: Send,
    Value: Send,
{
    type Iter = HashTableParIterMut<'a, HTI, Value>;
    type Item = (usize, &'a mut Value);
    fn into_par_iter(self) -> Self::Iter {
        HashTableParIterMut(self)
    }
}

impl std::error::Error for NotEnoughSpace {}

impl From<NotEnoughSpace> for std::io::Error {
    fn from(v: NotEnoughSpace) -> Self {
        std::io::Error::new(std::io::ErrorKind::Other, v)
    }
}
