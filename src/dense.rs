//! Dense vector representations.
//!
//! Dense vectors store every component from index `0` to `len()-1`.
//! The generic `DenseVector<T, S>` is parameterized by the backing storage `S`.
//!
//! ## Type aliases
//!
//! | Alias | Storage | Description |
//! |-------|---------|-------------|
//! | `DenseVec<T>` | `Vec<T>` | Heap-allocated |
//! | `DenseArrayVec<T, const N>` | `ArrayVec<T, N>` | Stack-allocated, fixed capacity |

use core::fmt;
use std::marker::PhantomData;
use std::ops::Range;

use crate::FromIteratorLossy;
use crate::Index;
use crate::TryFromIterator;
use crate::Vector;

mod distance;
mod dot;
mod iter;
mod ops;
mod storage;

pub use self::iter::*;
pub use self::storage::*;

/// A vector whose components are implicitly indexed `0..len`.
///
/// Every position in `0..len` is stored, making dense vectors suitable
/// when most components are non-zero.
pub trait DenseVector: Vector {
    /// Returns the implicit index space as a half-open range `0..len`.
    fn indices<Idx>(&self) -> Range<Idx>
    where
        Idx: Index;
    /// Returns a slice of all vector components in index order.
    fn values(&self) -> &[Self::Value];

    /// A borrowing iterator over `self`.
    fn iter<'a, Idx>(&'a self) -> impl Iterator<Item = (Idx, Self::Value)>
    where
        Idx: Index + 'a;

    /// An owning iterator over `self`.
    fn into_iter<Idx>(self) -> impl Iterator<Item = (Idx, Self::Value)>
    where
        Idx: Index;
}

/// A dense vector backed by storage `S`.
#[derive(Clone, Eq, PartialEq)]
pub struct GenericDenseVec<T, S> {
    pub(crate) storage: S,
    _phantom: PhantomData<T>,
}

impl<T, S> From<S> for GenericDenseVec<T, S> {
    fn from(storage: S) -> Self {
        Self {
            storage,
            _phantom: PhantomData,
        }
    }
}

/// Heap-allocated dense vector.
#[cfg(feature = "alloc")]
pub type DenseVec<T> = GenericDenseVec<T, DenseVecStorage<T>>;

/// Stack-allocated dense vector with capacity `N`.
pub type DenseArrayVec<T, const N: usize> = GenericDenseVec<T, DenseArrayVecStorage<T, N>>;

impl<T, S> GenericDenseVec<T, S>
where
    S: DenseStorage<T>,
{
    /// The number of components in `self`.
    #[inline]
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// `true` if `self.len() == 0`, otherwise `false`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }
}

// MARK: TryFromIterator

impl<T, S> TryFromIterator<T> for GenericDenseVec<T, S>
where
    S: DenseStorage<T> + TryFromIterator<T>,
{
    type Error = <S as TryFromIterator<T>>::Error;

    fn try_from_iter<I>(iter: I) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = T>,
    {
        S::try_from_iter(iter).map(Self::from)
    }
}

// MARK: FromIteratorLossy

impl<T, S> FromIteratorLossy<T> for GenericDenseVec<T, S>
where
    S: DenseStorage<T> + FromIteratorLossy<T>,
{
    fn from_iter_lossy<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        S::from_iter_lossy(iter).into()
    }
}

// MARK: Debug

impl<T, S> fmt::Debug for GenericDenseVec<T, S>
where
    T: fmt::Debug,
    S: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.storage)
    }
}

// MARK: Vector trait

impl<T, S: DenseStorage<T>> Vector for GenericDenseVec<T, S>
where
    T: Copy,
{
    type Value = T;

    fn len(&self) -> usize {
        self.storage.len()
    }

    fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }
}

impl<T, S> DenseVector for GenericDenseVec<T, S>
where
    T: Copy,
    S: DenseStorage<T>,
{
    /// The implicit indices as a range.
    fn indices<Idx>(&self) -> Range<Idx>
    where
        Idx: Index,
    {
        Idx::zero()..Idx::from_usize(self.storage.len())
    }

    /// The underlying components as a slice.
    fn values(&self) -> &[Self::Value] {
        self.storage.values()
    }

    fn iter<'a, Idx>(&'a self) -> impl Iterator<Item = (Idx, Self::Value)>
    where
        Idx: Index + 'a,
        S: 'a,
    {
        self.storage.iter()
    }

    fn into_iter<Idx>(self) -> impl Iterator<Item = (Idx, Self::Value)>
    where
        Idx: Index,
    {
        self.storage.into_iter()
    }
}
