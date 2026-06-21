//! Sparse vector representations.
//!
//! Sparse vectors store only non-zero `(index, value)` pairs and enforce
//! sorted, unique keys. The storage uses a SoA (Struct-of-Arrays) layout:
//! `indices: SI` and `values: SV`. The generic `SparseVector<Idx, T, S>`
//! is parameterized by the index type `Idx`, the value type `T`, and the
//! backing storages `SI` (indices) and `SV` (values).
//!
//! ## Type aliases
//!
//! | Alias | Index storage | Value storage | Description |
//! |-------|---------------|---------------|-------------|
//! | `SparseVec<Idx, T>` | `Vec<Idx>` | `Vec<T>` | Heap-allocated |
//! | `SparseArrayVec<Idx, T, const N>` | `ArrayVec<Idx, N>` | `ArrayVec<T, N>` | Stack-allocated, fixed capacity |

use std::fmt;
use std::marker::PhantomData;

use crate::FromIteratorLossy;
use crate::Index;
use crate::TryFromIterator;
use crate::Vector;

/// Utility functions for sparse vector construction (e.g., splitting
/// sorted pairs into SoA storage).
pub mod util;

mod common;
mod distance;
mod dot;
mod iter;
mod join;
mod ops;
mod storage;

pub(crate) use self::common::*;

pub use self::dot::dot_gallop;
pub use self::iter::*;
pub use self::join::*;
pub use self::storage::*;

/// A vector whose components are stored as explicit `(index, value)`
/// pairs, sorted by index.
///
/// Unlike [`DenseVector`](crate::dense::DenseVector), sparse vectors
/// only store non-zero entries. The index type is generic, allowing
/// compact representations (e.g. `u32` for low-dimensional spaces).
pub trait SparseVector: Vector {
    /// The unsigned integer type used for component indices.
    type Index: Index;

    /// A slice of all stored indices, in increasing order.
    fn indices(&self) -> &[Self::Index];
    /// A slice of all stored values, parallel to [`indices`](SparseVector::indices).
    fn values(&self) -> &[Self::Value];

    /// A borrowing iterator over stored `(index, value)` pairs.
    fn iter<'a>(&'a self) -> impl Iterator<Item = (Self::Index, Self::Value)>
    where
        Self::Index: 'a,
        Self: 'a;

    /// An owning iterator over stored `(index, value)` pairs.
    fn into_iter(self) -> impl Iterator<Item = (Self::Index, Self::Value)>;
}

/// A sparse vector backed by SoA storage: `indices: SI` and `values: SV`.
#[derive(Clone, Eq, PartialEq)]
pub struct GenericSparseVec<Idx, T, S> {
    pub(crate) storage: S,
    _phantom: PhantomData<(Idx, T)>,
}

impl<Idx, T, S> From<S> for GenericSparseVec<Idx, T, S> {
    fn from(storage: S) -> Self {
        Self {
            storage,
            _phantom: PhantomData,
        }
    }
}

/// Heap-allocated sparse vector.
#[cfg(feature = "alloc")]
pub type SparseVec<Idx, T> = GenericSparseVec<Idx, T, SparseVecStorage<Idx, T>>;

/// Stack-allocated sparse vector with capacity `N`.
pub type SparseArrayVec<Idx, T, const N: usize> =
    GenericSparseVec<Idx, T, SparseArrayVecStorage<Idx, T, N>>;

impl<Idx, T, S> GenericSparseVec<Idx, T, S>
where
    S: SparseStorage<Idx, T>,
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

impl<Idx, T, S> TryFromIterator<(Idx, T)> for GenericSparseVec<Idx, T, S>
where
    S: SparseStorage<Idx, T> + TryFromIterator<(Idx, T)>,
{
    type Error = <S as TryFromIterator<(Idx, T)>>::Error;

    fn try_from_iter<I>(iter: I) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = (Idx, T)>,
    {
        S::try_from_iter(iter).map(Self::from)
    }
}

// MARK: FromIteratorLossy

impl<Idx, T, S> FromIteratorLossy<(Idx, T)> for GenericSparseVec<Idx, T, S>
where
    S: SparseStorage<Idx, T> + FromIteratorLossy<(Idx, T)>,
{
    fn from_iter_lossy<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (Idx, T)>,
    {
        S::from_iter_lossy(iter).into()
    }
}

// MARK: Debug

impl<Idx, T, S> fmt::Debug for GenericSparseVec<Idx, T, S>
where
    Idx: fmt::Debug,
    T: fmt::Debug,
    S: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.storage)
    }
}

// MARK: Vector

impl<Idx, T, S: SparseStorage<Idx, T>> Vector for GenericSparseVec<Idx, T, S>
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

// MARK: SparseVector

impl<Idx, T, S> SparseVector for GenericSparseVec<Idx, T, S>
where
    Idx: Index,
    T: Copy,
    S: SparseStorage<Idx, T>,
{
    type Index = Idx;

    /// The underlying indices as a slice.
    fn indices(&self) -> &[Self::Index] {
        self.storage.indices()
    }

    /// The underlying values as a slice.
    fn values(&self) -> &[Self::Value] {
        self.storage.values()
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = (Self::Index, Self::Value)>
    where
        Self::Index: 'a,
        Self: 'a,
    {
        self.storage.iter()
    }

    fn into_iter(self) -> impl Iterator<Item = (Self::Index, Self::Value)> {
        self.storage.into_iter()
    }
}
