//! Sparse vector representations.
//!
//! Sparse vectors store only non-zero `(index, value)` pairs and enforce
//! sorted, unique keys. The generic `SparseVector<Idx, T, S>` is parameterized
//! by the value type `T`, the index type `Idx`, and the backing storage `S`.
//!
//! ## Type aliases
//!
//! | Alias | Storage | Description |
//! |-------|---------|-------------|
//! | `HeapSparseVector<Idx, T>` | `Vec<(Idx, T)>` | Heap-allocated |
//! | `StackSparseVector<Idx, T, const N>` | `ArrayVec<(Idx, T), N>` | Stack-allocated, fixed capacity |

use std::fmt;
use std::iter::FromIterator;
use std::marker::PhantomData;

use arrayvec::ArrayVec;
use num_traits::Zero;

use crate::Vector;
use crate::storage::Storage;

pub use self::iter::{IntoIter, Iter};
pub use self::join::{Join, inner_join, outer_join};

mod iter;
mod join;

mod debug;
mod distance;
mod dot;
mod ops;

/// Errors that can occur when constructing a sparse vector from unsorted input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SparseVectorError<Idx> {
    /// Indexes are not in strictly increasing order; the value at this index was out of order.
    UnsortedIndexes {
        /// The index in the input where the unsorted entry was found.
        index: Idx,
    },
    /// The same index appears more than once; this is the second occurrence.
    DuplicateIndexes {
        /// The index that appears more than once.
        index: Idx,
    },
    /// The value for this index is zero (only produced on the strict-validate path).
    ZeroValue {
        /// The index whose value is zero.
        index: Idx,
    },
}

impl<Idx: fmt::Display> fmt::Display for SparseVectorError<Idx> {
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

/// A sparse vector backed by storage `S`, using index type `Idx`.
pub struct SparseVector<Idx, T, S: Storage<(Idx, T)>> {
    pub(crate) components: S,
    _phantom: PhantomData<(Idx, T)>,
}

impl<Idx, T, S: Storage<(Idx, T)>> Clone for SparseVector<Idx, T, S>
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

impl<Idx, T, S: Storage<(Idx, T)>> PartialEq for SparseVector<Idx, T, S>
where
    S: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.components == other.components
    }
}

/// Heap-allocated sparse vector.
pub type HeapSparseVector<Idx, T> = SparseVector<Idx, T, Vec<(Idx, T)>>;

/// Stack-allocated sparse vector with capacity `N`.
pub type StackSparseVector<Idx, T, const N: usize> = SparseVector<Idx, T, ArrayVec<(Idx, T), N>>;

impl<Idx, T, S: Storage<(Idx, T)>> SparseVector<Idx, T, S> {
    /// The number of components in `self`.
    #[inline]
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// `true` if `self.len() == 0`, otherwise `false`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// A borrowing iterator over `self`.
    #[inline]
    pub fn iter<'a>(&'a self) -> Iter<'a, Idx, T>
    where
        Idx: 'a,
        T: 'a,
    {
        Iter::new(self.components.as_slice())
    }

    /// The underlying components as a slice of `(index, value)` pairs.
    #[inline]
    pub fn as_slice(&self) -> &[(Idx, T)] {
        self.components.as_slice()
    }
}

// MARK: Constructors

impl<Idx, T> SparseVector<Idx, T, Vec<(Idx, T)>> {
    /// Creates a sparse vector from a sorted, deduplicated, non-zero list of entries without validation.
    #[inline]
    pub fn from_sorted_unchecked(items: Vec<(Idx, T)>) -> Self {
        Self {
            components: items,
            _phantom: PhantomData,
        }
    }

    /// Creates a sparse vector from an unsorted, possibly duplicated list of entries.
    #[inline]
    pub fn from_unsorted(items: Vec<(Idx, T)>) -> Self
    where
        Idx: Ord + Copy,
        T: Copy + Zero + PartialEq,
    {
        let mut components = items;
        let new_len = canonicalize_entries(components.as_mut_slice(), true);
        components.truncate(new_len);
        Self {
            components,
            _phantom: PhantomData,
        }
    }

    /// Normalizes `self` in-place.
    #[inline]
    pub fn canonicalize(&mut self)
    where
        Idx: Ord + Copy,
        T: Copy + Zero + PartialEq,
    {
        let new_len = canonicalize_entries(self.components.as_mut(), true);
        Storage::truncate(&mut self.components, new_len);
    }
}

// MARK: From

impl<Idx, T> From<Vec<(Idx, T)>> for SparseVector<Idx, T, Vec<(Idx, T)>>
where
    Idx: Ord + Copy,
    T: Zero + PartialEq + Copy,
{
    #[inline]
    fn from(mut items: Vec<(Idx, T)>) -> Self {
        canonicalize_entries(items.as_mut_slice(), true);

        Self {
            components: items,
            _phantom: PhantomData,
        }
    }
}

impl<Idx, T, const N: usize> From<ArrayVec<(Idx, T), N>>
    for SparseVector<Idx, T, ArrayVec<(Idx, T), N>>
where
    Idx: Ord + Copy,
    T: Zero + PartialEq + Copy,
{
    #[inline]
    fn from(mut items: ArrayVec<(Idx, T), N>) -> Self {
        canonicalize_entries(items.as_mut_slice(), true);

        Self {
            components: ArrayVec::from(items),
            _phantom: PhantomData,
        }
    }
}

// MARK: FromIterator

impl<Idx, T, S: Storage<(Idx, T)>> FromIterator<(Idx, T)> for SparseVector<Idx, T, S>
where
    Idx: Ord + Copy,
    T: Zero + PartialEq + Copy,
{
    #[inline]
    fn from_iter<I: IntoIterator<Item = (Idx, T)>>(iter: I) -> Self {
        let mut components = S::from_iter(iter.into_iter());

        let new_len = canonicalize_entries(components.as_mut_slice(), true);
        Storage::truncate(&mut components, new_len);

        Self {
            components,
            _phantom: PhantomData,
        }
    }
}

// MARK: IntoIterator

impl<Idx, T, S: Storage<(Idx, T)>> IntoIterator for SparseVector<Idx, T, S> {
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = IntoIter<S>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self.components)
    }
}

impl<'a, Idx, T, S> IntoIterator for &'a SparseVector<Idx, T, S>
where
    Idx: 'a + Copy,
    T: 'a + Copy,
    S: Storage<(Idx, T)>,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = Iter<'a, Idx, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self.components.as_slice())
    }
}

// MARK: Vector trait

impl<Idx, T, S: Storage<(Idx, T)>> Vector for SparseVector<Idx, T, S>
where
    T: Copy,
{
    type Scalar = T;
}

// MARK: Helper functions

/// Validates that entries have strictly increasing keys and no zero values.
#[allow(dead_code)]
fn validate_entries<Idx, T>(
    entries: &[(Idx, T)],
    check_zeros: bool,
) -> Result<(), SparseVectorError<Idx>>
where
    Idx: Ord + Copy,
    T: Zero + PartialEq,
{
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

/// Sorts entries by key, deduplicates (keeping the last value), drops zeros (optional).
/// Returns the new length.
fn canonicalize_entries<Idx, T>(entries: &mut [(Idx, T)], drop_zeros: bool) -> usize
where
    Idx: Ord + Copy,
    T: Zero + PartialEq + Copy,
{
    if !entries.is_sorted_by(|lhs, rhs| lhs.0 < rhs.0) {
        entries.sort_by_key(|(k, _)| *k);
    }

    let mut write = 0;

    for read in 0..entries.len() {
        if read + 1 < entries.len() && entries[read].0 == entries[read + 1].0 {
            continue;
        }

        if drop_zeros && entries[read].1.is_zero() {
            continue;
        }

        if write != read {
            entries[write] = entries[read].clone();
        }

        write += 1;
    }

    write
}
