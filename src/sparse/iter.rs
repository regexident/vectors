use std::iter::{Copied, Zip};

#[cfg(feature = "alloc")]
use alloc::vec::IntoIter as VecIntoIter;

// MARK: Slice Storage

/// A borrowing iterator over a sparse vector's `(index, value)` pairs,
/// backed by parallel slices.
pub struct SparseSliceIter<'a, Idx, T> {
    inner: Zip<Copied<core::slice::Iter<'a, Idx>>, Copied<core::slice::Iter<'a, T>>>,
}

impl<'a, Idx, T> SparseSliceIter<'a, Idx, T>
where
    Idx: Copy,
    T: Copy,
{
    pub(crate) fn new(
        indices: core::slice::Iter<'a, Idx>,
        values: core::slice::Iter<'a, T>,
    ) -> Self {
        Self {
            inner: indices.copied().zip(values.copied()),
        }
    }
}

impl<'a, Idx, T> Iterator for SparseSliceIter<'a, Idx, T>
where
    Idx: Copy,
    T: Copy,
{
    type Item = (Idx, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, Idx, T> ExactSizeIterator for SparseSliceIter<'a, Idx, T>
where
    Idx: Copy,
    T: Copy,
{
}

// MARK: Vec Storage

/// An owning iterator over a sparse vector's `(index, value)` pairs,
/// backed by parallel [`Vec`] into-iterators.
#[cfg(feature = "alloc")]
pub struct SparseVecIntoIter<Idx, T> {
    inner: Zip<VecIntoIter<Idx>, VecIntoIter<T>>,
}

#[cfg(feature = "alloc")]
impl<Idx, T> SparseVecIntoIter<Idx, T> {
    /// Creates an `IntoIter` from storage.
    pub fn new(indices: VecIntoIter<Idx>, values: VecIntoIter<T>) -> Self {
        SparseVecIntoIter {
            inner: indices.zip(values),
        }
    }
}

#[cfg(feature = "alloc")]
impl<Idx, T> Iterator for SparseVecIntoIter<Idx, T> {
    type Item = (Idx, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

#[cfg(feature = "alloc")]
impl<Idx, T> ExactSizeIterator for SparseVecIntoIter<Idx, T> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

// MARK: ArrayVec Storage

/// An owning iterator over a sparse vector's `(index, value)` pairs,
/// backed by parallel [`ArrayVec`](arrayvec::ArrayVec) into-iterators
/// with capacity `N`.
pub struct SparseArrayVecIntoIter<Idx, T, const N: usize> {
    inner: Zip<arrayvec::IntoIter<Idx, N>, arrayvec::IntoIter<T, N>>,
}

impl<Idx, T, const N: usize> SparseArrayVecIntoIter<Idx, T, N> {
    /// Creates a new `SparseArrayVecIntoIter` from parallel
    /// [`arrayvec::IntoIter`] streams for indices and values.
    pub fn new(indices: arrayvec::IntoIter<Idx, N>, values: arrayvec::IntoIter<T, N>) -> Self {
        SparseArrayVecIntoIter {
            inner: indices.zip(values),
        }
    }
}

impl<Idx, T, const N: usize> Iterator for SparseArrayVecIntoIter<Idx, T, N> {
    type Item = (Idx, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<Idx, T, const N: usize> ExactSizeIterator for SparseArrayVecIntoIter<Idx, T, N> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(all(test, feature = "std"))]
mod test {
    use arrayvec::ArrayVec;

    use super::*;

    #[test]
    fn slice_iter() {
        let indices = [0, 1, 2, 3, 4];
        let values = [0.1, 0.2, 0.3, 0.4, 0.5];
        let iter = SparseSliceIter::new(indices.iter(), values.iter());
        let actual: Vec<(u32, f32)> = iter.collect();
        let expected = vec![(0, 0.1), (1, 0.2), (2, 0.3), (3, 0.4), (4, 0.5)];
        assert_eq!(actual, expected);
    }

    #[test]
    fn vec_into_iter() {
        let indices = vec![0, 1, 2, 3, 4];
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let iter = SparseVecIntoIter::new(indices.into_iter(), values.into_iter());
        let actual: Vec<(u32, f32)> = iter.collect();
        let expected = vec![(0, 0.1), (1, 0.2), (2, 0.3), (3, 0.4), (4, 0.5)];
        assert_eq!(actual, expected);
    }

    #[test]
    fn array_vec_into_iter() {
        let indices = ArrayVec::from([0, 1, 2, 3, 4]);
        let values = ArrayVec::from([0.1, 0.2, 0.3, 0.4, 0.5]);
        let iter = SparseArrayVecIntoIter::new(indices.into_iter(), values.into_iter());
        let actual: Vec<(u32, f32)> = iter.collect();
        let expected = vec![(0, 0.1), (1, 0.2), (2, 0.3), (3, 0.4), (4, 0.5)];
        assert_eq!(actual, expected);
    }
}
