//! Sparse vector representations.
//!
//! Sparse vectors store only non-zero `(index, value)` pairs and enforce
//! sorted, unique keys. The generic `SparseVector<T, S>` is parameterized
//! by the backing storage `S`.
//!
//! ## Type aliases
//!
//! | Alias | Storage | Description |
//! |-------|---------|-------------|
//! | `HeapSparseVector<T>` | `Vec<(usize, T)>` | Heap-allocated |
//! | `StackSparseVector<T, const N>` | `ArrayVec<(usize, T), N>` | Stack-allocated, fixed capacity |

use std::fmt;
use std::iter::FromIterator;
use std::marker::PhantomData;

use arrayvec::ArrayVec;
use num_traits::Zero;

use crate::Vector;
use crate::storage::Storage;

mod iter;
mod join;

mod debug;
mod distance;
mod dot;
mod ops;

pub use self::iter::{IntoIter, Iter};
pub use self::join::{Join, inner_join, outer_join};

/// Errors that can occur when constructing a sparse vector from unsorted input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SparseVectorError {
    /// Indexes are not in strictly increasing order; the value at this index was out of order.
    UnsortedIndexes {
        /// The index in the input where the unsorted entry was found.
        index: usize,
    },
    /// The same index appears more than once; this is the second occurrence.
    DuplicateIndexes {
        /// The index that appears more than once.
        index: usize,
    },
    /// The value for this index is zero (only produced on the strict-validate path).
    ZeroValue {
        /// The index whose value is zero.
        index: usize,
    },
}

impl fmt::Display for SparseVectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SparseVectorError::UnsortedIndexes { index: at } => {
                write!(f, "unsorted sparse-vector keys at index {}", at)
            }
            SparseVectorError::DuplicateIndexes { index } => {
                write!(f, "duplicate sparse-vector key {}", index)
            }
            SparseVectorError::ZeroValue { index: at } => {
                write!(f, "zero value at sparse-vector key {}", at)
            }
        }
    }
}

/// Validates that entries have strictly increasing keys and no zero values.
pub fn validate_entries<T: Zero + PartialEq>(
    entries: &[(usize, T)],
    check_zeros: bool,
) -> Result<(), SparseVectorError> {
    if entries.is_empty() {
        return Ok(());
    }

    let mut prev_key = entries[0].0;

    if check_zeros && entries[0].1.is_zero() {
        return Err(SparseVectorError::ZeroValue {
            index: entries[0].0,
        });
    }

    for (_i, (key, val)) in entries.iter().enumerate().skip(1) {
        if *key <= prev_key {
            if *key == prev_key {
                return Err(SparseVectorError::DuplicateIndexes { index: *key });
            }

            return Err(SparseVectorError::UnsortedIndexes { index: *key });
        }

        if check_zeros && val.is_zero() {
            return Err(SparseVectorError::ZeroValue { index: *key });
        }

        prev_key = *key;
    }

    Ok(())
}

/// Sorts entries by key, deduplicates (keeping the last value), drops zeros.
/// Returns the new length.
pub fn canonicalize_entries<T: Zero + PartialEq + Copy>(entries: &mut [(usize, T)]) -> usize {
    entries.sort_by_key(|(k, _)| *k);

    let mut write = 0;

    for read in 0..entries.len() {
        if read + 1 < entries.len() && entries[read].0 == entries[read + 1].0 {
            continue;
        }

        if entries[read].1.is_zero() {
            continue;
        }

        if write != read {
            entries[write] = entries[read];
        }

        write += 1;
    }

    write
}

/// A sparse vector backed by storage `S`.
pub struct SparseVector<T, S: Storage<(usize, T)>> {
    pub(crate) components: S,
    _phantom: PhantomData<T>,
}

impl<T, S: Storage<(usize, T)>> Clone for SparseVector<T, S>
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

impl<T, S: Storage<(usize, T)>> PartialEq for SparseVector<T, S>
where
    S: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.components == other.components
    }
}

/// Heap-allocated sparse vector.
pub type HeapSparseVector<T> = SparseVector<T, Vec<(usize, T)>>;

/// Stack-allocated sparse vector with capacity `N`.
pub type StackSparseVector<T, const N: usize> = SparseVector<T, ArrayVec<(usize, T), N>>;

impl<T, S: Storage<(usize, T)>> SparseVector<T, S> {
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

    /// The underlying components as a slice of `(index, value)` pairs.
    #[inline]
    pub fn as_slice(&self) -> &[(usize, T)] {
        self.components.as_ref()
    }
}

// MARK: Constructors

impl<T> SparseVector<T, Vec<(usize, T)>> {
    /// Creates a `SparseVector` from a `Vec` that the caller guarantees is valid.
    #[inline]
    pub fn try_from_sorted(v: Vec<(usize, T)>) -> Result<Self>
    where
        T: Zero + PartialEq,
    {
        match validate_entries(&v, false).is_ok() {
            true => todo!(),
            false => todo!(),
        }
        Self {
            components: v,
            _phantom: PhantomData,
        }
    }

    /// Creates a `SparseVector` from a `Vec` that the caller guarantees is valid.
    #[inline]
    pub fn from_sorted_unchecked(v: Vec<(usize, T)>) -> Self
    where
        T: Zero + PartialEq,
    {
        debug_assert!(validate_entries(&v, false).is_ok());
        Self {
            components: v,
            _phantom: PhantomData,
        }
    }

    /// Creates a `SparseVector` from unsorted pairs, normalizing in the process.
    #[inline]
    pub fn from_unsorted(mut v: Vec<(usize, T)>) -> Self
    where
        T: Copy + Zero + PartialEq,
    {
        let new_len = canonicalize_entries(&mut v);
        v.truncate(new_len);
        Self {
            components: v,
            _phantom: PhantomData,
        }
    }

    /// Normalizes `self` in-place.
    #[inline]
    pub fn canonicalize(&mut self)
    where
        T: Copy + Zero + PartialEq,
    {
        let new_len = canonicalize_entries(self.components.as_mut());
        Storage::truncate(&mut self.components, new_len);
    }
}

// MARK: From

impl<T> From<Vec<(usize, T)>> for SparseVector<T, Vec<(usize, T)>>
where
    T: Zero + PartialEq,
{
    /// # Panics
    ///
    /// Panics if the input has unsorted or duplicate indices, or contains zero values.
    #[inline]
    fn from(items: Vec<(usize, T)>) -> Self {
        if let Err(e) = validate_entries(&items, true) {
            panic!("{}", e);
        }

        Self {
            components: items,
            _phantom: PhantomData,
        }
    }
}

impl<T, const N: usize> From<[(usize, T); N]> for SparseVector<T, ArrayVec<(usize, T), N>>
where
    T: Zero + PartialEq,
{
    /// # Panics
    ///
    /// Panics if the input has unsorted or duplicate indices, or contains zero values.
    #[inline]
    fn from(items: [(usize, T); N]) -> Self {
        if let Err(e) = validate_entries(&items, true) {
            panic!("{}", e);
        }

        Self {
            components: ArrayVec::from(items),
            _phantom: PhantomData,
        }
    }
}

// MARK: FromIterator

impl<T, S: Storage<(usize, T)>> FromIterator<(usize, T)> for SparseVector<T, S>
where
    T: Zero + PartialEq,
{
    #[inline]
    fn from_iter<I: IntoIterator<Item = (usize, T)>>(iter: I) -> Self {
        let components = S::from_iter_in_place(iter.into_iter());
        if let Err(e) = validate_entries(components.as_ref(), true) {
            panic!("{}", e);
        }

        Self {
            components,
            _phantom: PhantomData,
        }
    }
}

// MARK: IntoIterator

impl<T, S: Storage<(usize, T)>> IntoIterator for SparseVector<T, S> {
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = IntoIter<S>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self.components)
    }
}

impl<'a, T, S: Storage<(usize, T)>> IntoIterator for &'a SparseVector<T, S>
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

impl<T, S: Storage<(usize, T)>> Vector for SparseVector<T, S>
where
    T: Copy,
{
    type Scalar = T;
}
