use core::fmt;

use arrayvec::ArrayVec;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::{FromIteratorLossy, Index, TryFromIterator};

#[cfg(feature = "alloc")]
use super::DenseVecIntoIter;
use super::{DenseArrayVecIntoIter, DenseSliceIter};

/// Backing storage for dense vectors.
///
/// Implemented by `Vec<T>` and `ArrayVec<T, N>`.
pub trait DenseStorage<T>: Sized {
    /// The iterator type returned by [`iter`](DenseStorage::iter).
    type Iter<'a, Idx>: Iterator<Item = (Idx, T)>
    where
        Idx: Index + 'a,
        Self: 'a;
    /// The iterator type returned by [`into_iter`](DenseStorage::into_iter).
    type IntoIter<Idx>: Iterator<Item = (Idx, T)>
    where
        Idx: Index;

    /// Returns the number of elements in the storage.
    fn len(&self) -> usize;

    /// Returns true if the storage has a length of 0.
    fn is_empty(&self) -> bool;

    /// Returns a slice containing the entire value storage.
    fn values(&self) -> &[T];

    /// Returns a mutable slice containing the entire value storage.
    fn values_mut(&mut self) -> &mut [T];

    /// Returns a borrowing iterator over `(index, value)` pairs.
    fn iter<'a, Idx>(&'a self) -> Self::Iter<'a, Idx>
    where
        Idx: Index + 'a,
        Self: 'a;

    /// Returns an owning iterator over `(index, value)` pairs.
    fn into_iter<Idx>(self) -> Self::IntoIter<Idx>
    where
        Idx: Index;
}

// MARK: Vec Storage

/// Errors that can arise from a [`DenseVecStorage`].
///
/// This enum is empty because construction from a [`Vec`] is always
/// infallible (unlike fixed-capacity storage). It exists so that
/// [`DenseVecStorage`] satisfies [`TryFromIterator`].
#[derive(Debug)]
pub enum DenseVecStorageError {}

/// Heap-allocated storage for a dense vector.
///
/// Wraps a [`Vec<T>`] where every element is a component. Construction
/// from an iterator is always infallible.
#[cfg(feature = "alloc")]
#[derive(Clone, Eq, PartialEq)]
pub struct DenseVecStorage<T> {
    values: Vec<T>,
}

#[cfg(feature = "alloc")]
impl<T> From<Vec<T>> for DenseVecStorage<T> {
    fn from(values: Vec<T>) -> Self {
        Self { values }
    }
}

#[cfg(feature = "alloc")]
impl<T> DenseStorage<T> for DenseVecStorage<T>
where
    T: Copy,
{
    type Iter<'a, Idx>
        = DenseSliceIter<'a, Idx, T>
    where
        Idx: Index + 'a,
        Self: 'a;
    type IntoIter<Idx>
        = DenseVecIntoIter<Idx, T>
    where
        Idx: Index;

    fn len(&self) -> usize {
        self.values.len()
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn values(&self) -> &[T] {
        self.values.as_slice()
    }

    fn values_mut(&mut self) -> &mut [T] {
        self.values.as_mut_slice()
    }

    fn iter<'a, Idx>(&'a self) -> Self::Iter<'a, Idx>
    where
        Idx: Index + 'a,
        Self: 'a,
    {
        DenseSliceIter::new(self.values.iter())
    }

    fn into_iter<Idx>(self) -> Self::IntoIter<Idx>
    where
        Idx: Index,
    {
        DenseVecIntoIter::new(self.values.into_iter())
    }
}

#[cfg(feature = "alloc")]
impl<T> TryFromIterator<T> for DenseVecStorage<T> {
    type Error = DenseVecStorageError;

    fn try_from_iter<I>(iter: I) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = T>,
    {
        let values = Vec::from_iter(iter);
        Ok(Self { values })
    }
}

#[cfg(feature = "alloc")]
impl<T> FromIteratorLossy<T> for DenseVecStorage<T> {
    fn from_iter_lossy<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let values = Vec::from_iter(iter);
        Self { values }
    }
}

#[cfg(feature = "alloc")]
impl<T> fmt::Debug for DenseVecStorage<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.values)
    }
}

// MARK: ArrayVec Storage

/// Errors that can arise from a [`DenseArrayVecStorage`].
#[derive(Debug)]
pub enum DenseArrayVecStorageError {
    /// The iterator yielded more items than the fixed capacity `N`.
    StorageExceeded {
        /// The number of items that could not be stored.
        excess: usize,
    },
}

impl fmt::Display for DenseArrayVecStorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StorageExceeded { excess } => {
                write!(f, "storage capacity exceeded by {}", excess)
            }
        }
    }
}

/// Stack-allocated dense storage backed by an [`ArrayVec`] with fixed
/// capacity `N`.
///
/// Construction from an iterator fails ([`DenseArrayVecStorageError::StorageExceeded`]) when the
/// number of elements exceeds `N`.
#[derive(Clone, Eq, PartialEq)]
pub struct DenseArrayVecStorage<T, const N: usize> {
    values: ArrayVec<T, N>,
}

impl<T, const N: usize> From<ArrayVec<T, N>> for DenseArrayVecStorage<T, N> {
    fn from(values: ArrayVec<T, N>) -> Self {
        Self { values }
    }
}

impl<T, const N: usize> DenseStorage<T> for DenseArrayVecStorage<T, N>
where
    T: Copy,
{
    type Iter<'a, Idx>
        = DenseSliceIter<'a, Idx, T>
    where
        Idx: Index + 'a,
        Self: 'a;
    type IntoIter<Idx>
        = DenseArrayVecIntoIter<Idx, T, N>
    where
        Idx: Index;

    fn len(&self) -> usize {
        self.values.len()
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn values(&self) -> &[T] {
        self.values.as_slice()
    }

    fn values_mut(&mut self) -> &mut [T] {
        self.values.as_mut_slice()
    }

    fn iter<'a, Idx>(&'a self) -> Self::Iter<'a, Idx>
    where
        Idx: Index + 'a,
        Self: 'a,
    {
        DenseSliceIter::new(self.values.iter())
    }

    fn into_iter<Idx>(self) -> Self::IntoIter<Idx>
    where
        Idx: Index,
    {
        DenseArrayVecIntoIter::new(self.values.into_iter())
    }
}

impl<T, const N: usize> TryFromIterator<T> for DenseArrayVecStorage<T, N> {
    type Error = DenseArrayVecStorageError;

    fn try_from_iter<I>(iter: I) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter = iter.into_iter();

        let mut values = ArrayVec::<T, N>::new();

        for _ in 0..N {
            match iter.next() {
                Some(item) => values.push(item),
                None => break,
            }
        }

        let excess = iter.count();

        if excess > 0 {
            return Err(DenseArrayVecStorageError::StorageExceeded { excess });
        }

        Ok(Self { values })
    }
}

impl<T, const N: usize> FromIteratorLossy<T> for DenseArrayVecStorage<T, N> {
    fn from_iter_lossy<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let iter = iter.into_iter().take(N);
        let values = ArrayVec::from_iter(iter);
        Self { values }
    }
}

impl<T, const N: usize> fmt::Debug for DenseArrayVecStorage<T, N>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.values)
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {

    use super::*;

    #[test]
    fn vec_debug() {
        let storage = DenseVecStorage::from(Vec::from([0.0, 0.25, 0.5, 0.75, 1.0]));
        assert_eq!(format!("{:?}", storage), "[0.0, 0.25, 0.5, 0.75, 1.0]");
    }

    #[test]
    fn array_vec_debug() {
        let storage = DenseArrayVecStorage::from(ArrayVec::from([0.0, 0.25, 0.5, 0.75, 1.0]));
        assert_eq!(format!("{:?}", storage), "[0.0, 0.25, 0.5, 0.75, 1.0]");
    }
}
