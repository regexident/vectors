use std::{iter::Copied, ops::AddAssign};

use num_traits::{One, Zero};

#[cfg(feature = "alloc")]
use alloc::vec::IntoIter as VecIntoIter;

use crate::Index;

// MARK: Indexed Iterator

/// An iterator that pairs each element from `inner` with a
/// monotonically increasing index starting at `0`.
///
/// This is the building block for all dense iterators in the crate.
pub struct Indexed<Idx, I> {
    index: Idx,
    inner: I,
}

impl<Idx, I> From<I> for Indexed<Idx, I>
where
    Idx: Zero,
{
    fn from(inner: I) -> Self {
        Self {
            index: Idx::zero(),
            inner,
        }
    }
}

impl<Idx, I> Iterator for Indexed<Idx, I>
where
    Idx: One + AddAssign + Copy,
    I: Iterator,
{
    type Item = (Idx, <I as Iterator>::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.inner.next()?;
        let index = self.index;
        self.index += Idx::one();
        Some((index, value))
    }
}

impl<Idx, I> ExactSizeIterator for Indexed<Idx, I>
where
    I: ExactSizeIterator,
    Self: Iterator,
{
    fn len(&self) -> usize {
        self.inner.len()
    }
}

// MARK: Slice Storage

/// A borrowing iterator over a dense vector's components and their
/// implicit indices, backed by a slice.
pub struct DenseSliceIter<'a, Idx, T> {
    inner: Indexed<Idx, Copied<core::slice::Iter<'a, T>>>,
}

impl<'a, Idx, T> DenseSliceIter<'a, Idx, T>
where
    Idx: Index,
    T: Copy,
{
    /// Creates a new `DenseSliceIter` from an iterator over slice
    /// references.
    pub fn new(values: core::slice::Iter<'a, T>) -> Self {
        Self {
            inner: Indexed::from(values.copied()),
        }
    }
}

impl<'a, Idx, T> Iterator for DenseSliceIter<'a, Idx, T>
where
    Idx: Index,
    T: Copy,
{
    type Item = (Idx, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<'a, Idx, T> ExactSizeIterator for DenseSliceIter<'a, Idx, T>
where
    Idx: Index,
    T: Copy,
{
    fn len(&self) -> usize {
        self.inner.len()
    }
}

// MARK: Vec Storage

/// An owning iterator over a dense vector's components and their
/// implicit indices, backed by a [`Vec`].
#[cfg(feature = "alloc")]
pub struct DenseVecIntoIter<Idx, T> {
    inner: Indexed<Idx, VecIntoIter<T>>,
}

#[cfg(feature = "alloc")]
impl<Idx, T> DenseVecIntoIter<Idx, T>
where
    Idx: Index,
{
    /// Creates a new `DenseVecIntoIter` from a [`Vec`] `VecIntoIter`.
    pub fn new(inner: VecIntoIter<T>) -> Self {
        Self {
            inner: Indexed::from(inner),
        }
    }
}

#[cfg(feature = "alloc")]
impl<Idx, T> Iterator for DenseVecIntoIter<Idx, T>
where
    Indexed<Idx, VecIntoIter<T>>: Iterator<Item = (Idx, T)>,
{
    type Item = (Idx, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

#[cfg(feature = "alloc")]
impl<Idx, T> ExactSizeIterator for DenseVecIntoIter<Idx, T>
where
    Indexed<Idx, VecIntoIter<T>>: Iterator<Item = (Idx, T)> + ExactSizeIterator,
{
    fn len(&self) -> usize {
        self.inner.len()
    }
}

// MARK: ArrayVec Storage

/// An owning iterator over a dense vector's components and their
/// implicit indices, backed by an [`ArrayVec`](arrayvec::ArrayVec)
/// with capacity `N`.
pub struct DenseArrayVecIntoIter<Idx, T, const N: usize> {
    inner: Indexed<Idx, arrayvec::IntoIter<T, N>>,
}

impl<Idx, T, const N: usize> DenseArrayVecIntoIter<Idx, T, N>
where
    Idx: Index,
{
    /// Creates a new `DenseArrayVecIntoIter` from an
    /// [`arrayvec::IntoIter`].
    pub fn new(inner: arrayvec::IntoIter<T, N>) -> Self {
        Self {
            inner: Indexed::from(inner),
        }
    }
}

impl<Idx, T, const N: usize> Iterator for DenseArrayVecIntoIter<Idx, T, N>
where
    Idx: Index,
{
    type Item = (Idx, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<Idx, T, const N: usize> ExactSizeIterator for DenseArrayVecIntoIter<Idx, T, N>
where
    Idx: Index,
{
    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use arrayvec::ArrayVec;

    use super::*;

    #[test]
    fn indexed_iter() {
        let iter = [0.1, 0.2, 0.3, 0.4, 0.5].into_iter();
        let iter = Indexed::from(iter);
        let actual: Vec<(u32, f32)> = iter.collect();
        let expected = vec![(0, 0.1), (1, 0.2), (2, 0.3), (3, 0.4), (4, 0.5)];
        assert_eq!(actual, expected);
    }

    #[test]
    fn slice_iter() {
        let values = [0.1, 0.2, 0.3, 0.4, 0.5];
        let iter = DenseSliceIter::new(values.iter());
        let actual: Vec<(u32, f32)> = iter.collect();
        let expected = vec![(0, 0.1), (1, 0.2), (2, 0.3), (3, 0.4), (4, 0.5)];
        assert_eq!(actual, expected);
    }

    #[test]
    fn vec_into_iter() {
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let iter = DenseVecIntoIter::new(values.into_iter());
        let actual: Vec<(u32, f32)> = iter.collect();
        let expected = vec![(0, 0.1), (1, 0.2), (2, 0.3), (3, 0.4), (4, 0.5)];
        assert_eq!(actual, expected);
    }

    #[test]
    fn array_vec_into_iter() {
        let values = ArrayVec::from([0.1, 0.2, 0.3, 0.4, 0.5]);
        let iter = DenseArrayVecIntoIter::new(values.into_iter());
        let actual: Vec<(u32, f32)> = iter.collect();
        let expected = vec![(0, 0.1), (1, 0.2), (2, 0.3), (3, 0.4), (4, 0.5)];
        assert_eq!(actual, expected);
    }
}
