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
//! | `libm`  | no      | Uses `libm` to provide `Real::sqrt` under `no_std` (for `Distance::distance`). |

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

#[cfg(not(feature = "std"))]
extern crate core as std;

pub mod dense;
pub mod sparse;

mod storage;

pub use storage::Storage;

use num_traits::real::Real;

/// The crate's prelude
pub mod prelude {
    pub use super::{Distance, Dot, Vector};
}

/// The base trait for vector types, covering comparisons,
/// basic numeric operations, and the dot product.
pub trait Vector {
    /// The type of the `Vector`'s scalar components.
    type Scalar;
}

/// The trait for types supporting the calculation of the dot product
pub trait Dot<Rhs = Self>: Sized {
    /// The scalar type returned by `self`'s dot product
    type Output;

    /// Calculates the dot-product between `self` and `rhs`.
    fn dot(&self, rhs: &Rhs) -> Self::Output;
}

/// The trait for types supporting the calculation of distance
pub trait Distance<Rhs = Self>: Sized {
    /// The scalar type returned by `self`'s distance
    type Output;

    /// Calculates the squared euclidian distance between `self` and `rhs`.
    fn squared_distance(&self, rhs: &Rhs) -> Self::Output;

    /// Calculates the euclidian distance between `self` and `rhs`.
    fn distance(&self, rhs: &Rhs) -> Self::Output
    where
        Self::Output: Real,
    {
        self.squared_distance(rhs).sqrt()
    }
}
