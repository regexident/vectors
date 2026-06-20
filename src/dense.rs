//! Dense vector representations.
//!
//! Dense vectors store every component from index `0` to `len()-1`.
//! The generic `DenseVector<T, S>` is parameterized by the backing storage `S`.
//!
//! ## Type aliases
//!
//! | Alias | Storage | Description |
//! |-------|---------|-------------|
//! | `HeapDenseVector<T>` | `Vec<T>` | Heap-allocated |
//! | `StackDenseVector<T, const N>` | `ArrayVec<T, N>` | Stack-allocated, fixed capacity |

use arrayvec::ArrayVec;
use std::iter::FromIterator;
use std::marker::PhantomData;

use crate::Vector;
use crate::storage::Storage;

mod debug;
mod distance;
mod dot;
mod iter;
mod ops;

pub use self::iter::{IntoIter, Iter};

/// A dense vector backed by storage `S`.
pub struct DenseVector<T, S: Storage<T>> {
    pub(crate) components: S,
    _phantom: PhantomData<T>,
}

impl<T, S: Storage<T>> Clone for DenseVector<T, S>
where
    S: Clone,
{
    fn clone(&self) -> Self {
        Self {
            components: self.components.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T, S: Storage<T>> PartialEq for DenseVector<T, S>
where
    S: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.components == other.components
    }
}

/// Heap-allocated dense vector.
pub type HeapDenseVector<T> = DenseVector<T, Vec<T>>;

/// Stack-allocated dense vector with capacity `N`.
pub type StackDenseVector<T, const N: usize> = DenseVector<T, ArrayVec<T, N>>;

impl<T, S: Storage<T>> DenseVector<T, S> {
    /// The number of components in `self`.
    #[inline]
    pub fn len(&self) -> usize {
        self.components.as_ref().len()
    }

    /// `true` if `self.len() == 0`, otherwise `false`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.components.as_ref().is_empty()
    }

    /// A borrowing iterator over `self`.
    #[inline]
    pub fn iter<'a>(&'a self) -> Iter<'a, T>
    where
        T: 'a,
    {
        Iter::new(self.components.as_ref())
    }

    /// The underlying components as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.components.as_ref()
    }
}

// MARK: From

impl<T> From<Vec<T>> for DenseVector<T, Vec<T>> {
    #[inline]
    fn from(items: Vec<T>) -> Self {
        Self {
            components: items,
            _phantom: PhantomData,
        }
    }
}

impl<T, const N: usize> From<[T; N]> for DenseVector<T, ArrayVec<T, N>> {
    #[inline]
    fn from(items: [T; N]) -> Self {
        Self {
            components: ArrayVec::from(items),
            _phantom: PhantomData,
        }
    }
}

impl<T, const N: usize> From<ArrayVec<T, N>> for DenseVector<T, ArrayVec<T, N>> {
    #[inline]
    fn from(items: ArrayVec<T, N>) -> Self {
        Self {
            components: items,
            _phantom: PhantomData,
        }
    }
}

// MARK: FromIterator

impl<T, S: Storage<T>> FromIterator<T> for DenseVector<T, S> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            components: S::from_iter_in_place(iter.into_iter()),
            _phantom: PhantomData,
        }
    }
}

// MARK: IntoIterator

impl<T, S: Storage<T>> IntoIterator for DenseVector<T, S> {
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = IntoIter<S>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self.components)
    }
}

impl<'a, T, S: Storage<T>> IntoIterator for &'a DenseVector<T, S>
where
    T: 'a + Copy,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self.components.as_ref())
    }
}

// MARK: Vector trait

impl<T, S: Storage<T>> Vector for DenseVector<T, S>
where
    T: Copy,
{
    type Scalar = T;
}
