// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Vector representations for use in high dimensional vector spaces.
//!
//! ## Feature flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std`   | yes     | Enables the standard library. Without this the crate is `no_std`. |
//! | `alloc` | yes (via `std`) | Enables `Vec`-backed types (`DenseVec`, `SparseVec`). Disable for pure `no_std`. |
//! | `libm`  | no      | Uses `libm` to provide `Real::sqrt` under `no_std` (for `Distance::distance`). |

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

#[cfg(not(feature = "std"))]
extern crate core as std;

#[cfg(feature = "alloc")]
extern crate alloc;

use std::ops::AddAssign;

#[cfg(any(feature = "std", feature = "libm"))]
use num_traits::real::Real;
use num_traits::{Num, One, Zero};

pub mod dense;
pub mod sparse;

mod common;
mod iter;

pub use self::iter::*;

/// The crate's prelude
pub mod prelude {
    pub use super::{Distance, Dot, Vector};
}

/// Numeric index types that can be used to identify vector components.
///
/// This trait is implemented for standard unsigned integer types:
/// `u8`, `u16`, `u32`, `u64`, and `usize`. Any type implementing
/// `Index` must satisfy the constraints required for position
/// arithmetic and comparison (addition, ordering, zero, and one).
pub trait Index: Zero + One + Ord + AddAssign + Copy + FromUsize {}

impl Index for u8 {}
impl Index for u16 {}
impl Index for u32 {}
impl Index for u64 {}
impl Index for usize {}

/// Conversion from `usize` to an index type.
///
/// Implemented for `u8`, `u16`, `u32`, `u64`, and `usize`.
pub trait FromUsize: Sized {
    /// Convert from `usize` to `Self`.
    fn from_usize(value: usize) -> Self;
}

impl FromUsize for u8 {
    fn from_usize(value: usize) -> Self {
        value as u8
    }
}
impl FromUsize for u16 {
    fn from_usize(value: usize) -> Self {
        value as u16
    }
}
impl FromUsize for u32 {
    fn from_usize(value: usize) -> Self {
        value as u32
    }
}
impl FromUsize for u64 {
    fn from_usize(value: usize) -> Self {
        value as u64
    }
}
impl FromUsize for usize {
    fn from_usize(value: usize) -> Self {
        value
    }
}

/// Scalar value types that can be stored as vector components.
///
/// This trait is implemented for `f32` and `f64`. Implementors must
/// be [`Num`] (supporting basic arithmetic) and
/// [`Copy`].
pub trait Value: Num + Copy {}

impl Value for f32 {}
impl Value for f64 {}

/// The base trait for vector types, covering comparisons,
/// basic numeric operations, and the dot product.
pub trait Vector {
    /// The type of the `Vector`'s scalar components.
    type Value;

    /// The number of components in `self`.
    #[must_use]
    fn len(&self) -> usize;

    /// `true` if `self.len() == 0`, otherwise `false`.
    #[must_use]
    fn is_empty(&self) -> bool;
}

/// The trait for types supporting the calculation of the dot product
pub trait Dot<Rhs = Self>: Sized {
    /// The scalar type returned by `self`'s dot product
    type Output;

    /// Calculates the dot-product between `self` and `rhs`.
    #[must_use]
    fn dot(&self, rhs: &Rhs) -> Self::Output;
}

/// The trait for types supporting the calculation of distance
pub trait Distance<Rhs = Self>: Sized {
    /// The scalar type returned by `self`'s distance
    type Output;

    /// Calculates the squared euclidian distance between `self` and `rhs`.
    #[must_use]
    fn squared_distance(&self, rhs: &Rhs) -> Self::Output;

    /// Calculates the euclidian distance between `self` and `rhs`.
    #[cfg(any(feature = "std", feature = "libm"))]
    #[must_use]
    fn distance(&self, rhs: &Rhs) -> Self::Output
    where
        Self::Output: Real,
    {
        self.squared_distance(rhs).sqrt()
    }
}
