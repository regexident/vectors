use core::fmt;

use arrayvec::ArrayVec;
use num_traits::Zero;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{FromIteratorLossy, Index, TryFromIterator};

#[cfg(feature = "alloc")]
use super::SparseVecIntoIter;
use super::util::canonicalize_entries;

use super::{SparseArrayVecIntoIter, SparseSliceIter};

/// Construct storage from already-canonical `(index, value)` pairs
/// without re-sorting, re-deduplicating, or filtering.
///
/// # Invariants
///
/// The caller MUST guarantee that the iterator yields pairs satisfying
/// all of the following invariants. Violating any of them will produce
/// a silently corrupted storage:
///
/// 1. **Strictly increasing indices**, each index must be strictly
///    greater than the previous one. Equality or inversion both
///    corrupt the ordering.
/// 2. **No duplicate indices**, each index must appear at most once.
///    A duplicate effectively merges two entries with the same key
///    into parallel slices, violating the 1:1 correspondence.
///
/// The call SHOULD also guarantee that the iterator yields pairs
/// satisfying the following invariants:
///
/// 1. **No zero values**, every value should be non-zero.
///
/// This is a low-level building block. Higher-level operations
/// (`outer_join`, in-place `Mul`, etc.) produce canonical output and
/// route through this constructor as a fast path.
pub trait FromCanonicalPairs<Idx, T>: Sized {
    /// Construct `Self` from guaranteed-canonical `(index, value)` pairs.
    fn from_canonical_pairs(iter: impl IntoIterator<Item = (Idx, T)>) -> Self;
}

/// Backing storage for vector components.
///
/// Implemented by `Vec<T>` and `ArrayVec<T, N>`.
pub trait SparseStorage<Idx, T> {
    /// The iterator type returned by [`iter`](SparseStorage::iter).
    type Iter<'a>: Iterator<Item = (Idx, T)>
    where
        Self: 'a;
    /// The iterator type returned by [`into_iter`](SparseStorage::into_iter).
    type IntoIter: Iterator<Item = (Idx, T)>;

    /// Returns the number of elements in the storage.
    fn len(&self) -> usize;

    /// Returns true if the storage has a length of 0.
    fn is_empty(&self) -> bool;

    /// Returns a slice containing the entire index storage.
    fn indices(&self) -> &[Idx];

    /// Returns a mutable slice containing the entire index storage.
    fn indices_mut(&mut self) -> &mut [Idx];

    /// Returns a slice containing the entire value storage.
    fn values(&self) -> &[T];

    /// Returns a mutable slice containing the entire value storage.
    fn values_mut(&mut self) -> &mut [T];

    /// Returns a borrowing iterator over `(index, value)` pairs.
    fn iter<'a>(&'a self) -> Self::Iter<'a>;

    /// Returns an owning iterator over `(index, value)` pairs.
    fn into_iter(self) -> Self::IntoIter;
}

// MARK: Vec Storage

/// Errors that can occur when constructing a sparse vector from unsorted input.
#[derive(Clone, Eq, PartialEq, Debug)]
pub enum SparseVecStorageError<Idx> {
    /// Indices are not in strictly increasing order; the value at this index was out of order.
    UnsortedIndices {
        /// The index in the input where the unsorted entry was found.
        index: Idx,
    },
    /// The same index appears more than once; this is the second occurrence.
    DuplicateIndices {
        /// The index that appears more than once.
        index: Idx,
    },
}

impl<Idx: fmt::Display> fmt::Display for SparseVecStorageError<Idx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsortedIndices { index } => {
                write!(f, "unsorted sparse-vector keys at index {index}")
            }
            Self::DuplicateIndices { index } => {
                write!(f, "duplicate sparse-vector key {index}")
            }
        }
    }
}

/// Heap-allocated storage for a sparse vector.
///
/// Stores parallel `Vec<Idx>` and `Vec<T>` buffers in a
/// struct-of-arrays layout. Construction from an iterator of
/// `(index, value)` pairs is fallible; see
/// [`SparseVecStorageError`].
#[cfg(feature = "alloc")]
#[derive(Clone, Eq, PartialEq)]
pub struct SparseVecStorage<Idx, T> {
    indices: Vec<Idx>,
    values: Vec<T>,
}

#[cfg(feature = "alloc")]
impl<Idx, T> SparseStorage<Idx, T> for SparseVecStorage<Idx, T>
where
    Idx: Index,
    T: Copy,
{
    type Iter<'a>
        = SparseSliceIter<'a, Idx, T>
    where
        Self: 'a;

    type IntoIter = SparseVecIntoIter<Idx, T>;

    fn len(&self) -> usize {
        self.values.len()
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn indices(&self) -> &[Idx] {
        self.indices.as_slice()
    }

    fn indices_mut(&mut self) -> &mut [Idx] {
        self.indices.as_mut_slice()
    }

    fn values(&self) -> &[T] {
        self.values.as_slice()
    }

    fn values_mut(&mut self) -> &mut [T] {
        self.values.as_mut_slice()
    }

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
        SparseSliceIter::new(self.indices.iter(), self.values.iter())
    }

    fn into_iter(self) -> Self::IntoIter {
        SparseVecIntoIter::new(self.indices.into_iter(), self.values.into_iter())
    }
}

/// Constructs a [`SparseVecStorage`] from a sorted, unique iterator.
///
/// Validates that entries are sorted and non-duplicate.
/// Zero-valued entries are silently dropped.
#[cfg(feature = "alloc")]
impl<Idx, T> TryFromIterator<(Idx, T)> for SparseVecStorage<Idx, T>
where
    Idx: Ord + Copy,
    T: Zero,
{
    type Error = SparseVecStorageError<Idx>;

    fn try_from_iter<I>(iter: I) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = (Idx, T)>,
    {
        let iter = iter.into_iter();
        let (lo, hi) = iter.size_hint();
        let cap = hi.unwrap_or(lo);
        let mut indices = Vec::with_capacity(cap);
        let mut values = Vec::with_capacity(cap);
        let mut prev_index: Option<Idx> = None;

        for (index, value) in iter {
            if value.is_zero() {
                continue;
            }

            if let Some(prev) = prev_index {
                if index <= prev {
                    return if index == prev {
                        Err(SparseVecStorageError::DuplicateIndices { index })
                    } else {
                        Err(SparseVecStorageError::UnsortedIndices { index })
                    };
                }
            }
            indices.push(index);
            values.push(value);
            prev_index = Some(index);
        }

        Ok(Self { indices, values })
    }
}

#[cfg(feature = "alloc")]
impl<Idx, T> FromIteratorLossy<(Idx, T)> for SparseVecStorage<Idx, T>
where
    Idx: Ord + Copy,
    T: Zero + PartialEq + Clone,
{
    fn from_iter_lossy<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (Idx, T)>,
    {
        let (mut indices, mut values): (Vec<Idx>, Vec<T>) = iter.into_iter().unzip();
        let len = canonicalize_entries(&mut indices, &mut values, true);
        indices.truncate(len);
        values.truncate(len);
        indices.shrink_to_fit();
        values.shrink_to_fit();
        Self { indices, values }
    }
}

#[cfg(feature = "alloc")]
impl<Idx, T> FromCanonicalPairs<Idx, T> for SparseVecStorage<Idx, T>
where
    Idx: Ord + Copy,
{
    fn from_canonical_pairs(iter: impl IntoIterator<Item = (Idx, T)>) -> Self {
        let (indices, values): (Vec<Idx>, Vec<T>) = iter.into_iter().unzip();

        if cfg!(debug_assertions) {
            let mut prev_index: Option<Idx> = None;

            for (index, _) in indices.iter().zip(values.iter()) {
                if let Some(prev) = prev_index {
                    debug_assert!(
                        *index > prev,
                        "FromCanonicalPairs requires strictly increasing indices"
                    );
                }
                prev_index = Some(*index);
            }
        }

        Self { indices, values }
    }
}

#[cfg(feature = "alloc")]
impl<Idx, T> fmt::Debug for SparseVecStorage<Idx, T>
where
    Idx: fmt::Debug + Copy + Index,
    T: fmt::Debug + Copy,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

// MARK: ArrayVec Storage

/// Errors that can occur when constructing a sparse vector from unsorted input.
#[derive(Clone, Eq, PartialEq, Debug)]
pub enum SparseArrayVecStorageError<Idx> {
    /// Indices are not in strictly increasing order; the value at this index was out of order.
    UnsortedIndices {
        /// The index in the input where the unsorted entry was found.
        index: Idx,
    },
    /// The same index appears more than once; this is the second occurrence.
    DuplicateIndices {
        /// The index that appears more than once.
        index: Idx,
    },
    /// The iterator yielded more items than the storage's fixed capacity.
    StorageExceeded {
        /// The number of items that could not be stored.
        excess: usize,
    },
}

impl<Idx: fmt::Display> fmt::Display for SparseArrayVecStorageError<Idx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsortedIndices { index } => {
                write!(f, "unsorted sparse-vector keys at index {index}")
            }
            Self::DuplicateIndices { index } => {
                write!(f, "duplicate sparse-vector key {}", index)
            }
            Self::StorageExceeded { excess } => {
                write!(f, "storage capacity exceeded by {}", excess)
            }
        }
    }
}

/// Stack-allocated sparse storage with fixed capacity `N`.
///
/// Stores parallel [`ArrayVec<Idx, N>`] and [`ArrayVec<T, N>`]
/// buffers in a struct-of-arrays layout. Construction from an
/// iterator of `(index, value)` pairs is fallible; see
/// [`SparseArrayVecStorageError`].
#[derive(Clone, Eq, PartialEq)]
pub struct SparseArrayVecStorage<Idx, T, const N: usize> {
    indices: ArrayVec<Idx, N>,
    values: ArrayVec<T, N>,
}

impl<Idx, T, const N: usize> SparseStorage<Idx, T> for SparseArrayVecStorage<Idx, T, N>
where
    Idx: Index,
    T: Copy,
{
    type Iter<'a>
        = SparseSliceIter<'a, Idx, T>
    where
        Self: 'a;

    type IntoIter = SparseArrayVecIntoIter<Idx, T, N>;

    fn len(&self) -> usize {
        self.values.len()
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn indices(&self) -> &[Idx] {
        self.indices.as_slice()
    }

    fn indices_mut(&mut self) -> &mut [Idx] {
        self.indices.as_mut_slice()
    }

    fn values(&self) -> &[T] {
        self.values.as_slice()
    }

    fn values_mut(&mut self) -> &mut [T] {
        self.values.as_mut_slice()
    }

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
        SparseSliceIter::new(self.indices.iter(), self.values.iter())
    }

    fn into_iter(self) -> Self::IntoIter {
        SparseArrayVecIntoIter::new(self.indices.into_iter(), self.values.into_iter())
    }
}

/// Constructs a [`SparseArrayVecStorage`] from a sorted, unique iterator.
///
/// Validates that entries are sorted and non-duplicate (and fit within capacity `N`).
/// Zero-valued entries are silently dropped.
impl<Idx, T, const N: usize> TryFromIterator<(Idx, T)> for SparseArrayVecStorage<Idx, T, N>
where
    Idx: Ord + Copy,
    T: Zero,
{
    type Error = SparseArrayVecStorageError<Idx>;

    fn try_from_iter<I>(iter: I) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = (Idx, T)>,
    {
        let mut iter = iter.into_iter().filter(|(_, v)| !v.is_zero());

        let mut indices = ArrayVec::<Idx, N>::new();
        let mut values = ArrayVec::<T, N>::new();
        let mut prev_index: Option<Idx> = None;

        for _ in 0..N {
            match iter.next() {
                Some((index, value)) => {
                    if let Some(prev) = prev_index {
                        if index <= prev {
                            return if index == prev {
                                Err(SparseArrayVecStorageError::DuplicateIndices { index })
                            } else {
                                Err(SparseArrayVecStorageError::UnsortedIndices { index })
                            };
                        }
                    }
                    indices.push(index);
                    values.push(value);
                    prev_index = Some(index);
                }
                None => break,
            }
        }

        let excess = iter.count();

        if excess > 0 {
            return Err(SparseArrayVecStorageError::StorageExceeded { excess });
        }

        Ok(Self { indices, values })
    }
}

// #[cfg(feature = "alloc")]
impl<Idx, T, const N: usize> FromIteratorLossy<(Idx, T)> for SparseArrayVecStorage<Idx, T, N>
where
    Idx: Ord + Copy,
    T: Zero + PartialEq + Clone,
{
    fn from_iter_lossy<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (Idx, T)>,
    {
        let mut indices = ArrayVec::<Idx, N>::new();
        let mut values = ArrayVec::<T, N>::new();

        for (index, value) in iter.into_iter().take(N) {
            indices.push(index);
            values.push(value);
        }

        let len = canonicalize_entries(indices.as_mut_slice(), values.as_mut_slice(), true);
        indices.truncate(len);
        values.truncate(len);

        Self { indices, values }
    }
}

impl<Idx, T, const N: usize> FromCanonicalPairs<Idx, T> for SparseArrayVecStorage<Idx, T, N>
where
    Idx: Ord + Copy,
{
    fn from_canonical_pairs(iter: impl IntoIterator<Item = (Idx, T)>) -> Self {
        let mut indices = ArrayVec::<Idx, N>::new();
        let mut values = ArrayVec::<T, N>::new();
        for (index, value) in iter.into_iter().take(N) {
            indices.push(index);
            values.push(value);
        }

        if cfg!(debug_assertions) {
            let mut prev_index: Option<Idx> = None;

            for (index, _) in indices.iter().zip(values.iter()) {
                if let Some(prev) = prev_index {
                    debug_assert!(
                        *index > prev,
                        "FromCanonicalPairs requires strictly increasing indices"
                    );
                }
                prev_index = Some(*index);
            }
        }

        Self { indices, values }
    }
}

impl<Idx, T, const N: usize> fmt::Debug for SparseArrayVecStorage<Idx, T, N>
where
    Idx: fmt::Debug + Copy + Index,
    T: fmt::Debug + Copy,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::TryFromIterator;

    #[test]
    fn vec_unsorted_indices() {
        let err = SparseVecStorage::<u32, f64>::try_from_iter([(2, 1.0), (1, 2.0)]).unwrap_err();
        assert_eq!(err, SparseVecStorageError::UnsortedIndices { index: 1 });
    }

    #[test]
    fn vec_duplicate_indices() {
        let err = SparseVecStorage::<u32, f64>::try_from_iter([(1, 1.0), (1, 2.0)]).unwrap_err();
        assert_eq!(err, SparseVecStorageError::DuplicateIndices { index: 1 });
    }

    #[test]
    fn array_vec_capacity_exceeded() {
        let err =
            SparseArrayVecStorage::<u32, f64, 2>::try_from_iter([(0, 1.0), (1, 2.0), (2, 3.0)])
                .unwrap_err();
        assert_eq!(
            err,
            SparseArrayVecStorageError::StorageExceeded { excess: 1 }
        );
    }

    #[test]
    fn array_vec_unsorted_indices() {
        let err =
            SparseArrayVecStorage::<u32, f64, 3>::try_from_iter([(2, 1.0), (1, 2.0)]).unwrap_err();
        assert_eq!(
            err,
            SparseArrayVecStorageError::UnsortedIndices { index: 1 }
        );
    }

    #[test]
    fn array_vec_duplicate_indices() {
        let err =
            SparseArrayVecStorage::<u32, f64, 3>::try_from_iter([(1, 1.0), (1, 2.0)]).unwrap_err();
        assert_eq!(
            err,
            SparseArrayVecStorageError::DuplicateIndices { index: 1 }
        );
    }
}
