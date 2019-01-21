// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Vector representations for use in high dimensional vector spaces.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "missing_mpl", feature(plugin))]
#![cfg_attr(feature = "missing_mpl", plugin(missing_mpl))]
#![cfg_attr(feature = "missing_mpl", deny(missing_mpl))]
#![warn(missing_docs)]

#[cfg(not(feature = "std"))]
extern crate core as std;

#[cfg(not(feature = "std"))]
#[macro_use]
extern crate std;

#[cfg(test)]
#[macro_use(expect)]
extern crate expectest;

extern crate arrayvec;
extern crate num_traits;
extern crate ordered_iter;

pub mod dense;
pub mod sparse;

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use num_traits::{real::Real, MulAdd, MulAddAssign};

/// The crate's prelude
pub mod prelude {
    pub use super::{
        Distance, Dot, Vector, VectorAssign, VectorAssignOps, VectorAssignRef, VectorOps, VectorRef,
    };
}

/// The base trait for vector types, covering comparisons,
/// basic numeric operations, and the dot product.
pub trait Vector {
    /// The type of the `Vector`'s scalar components.
    type Scalar;
}

/// The trait for vector types implementing basic numeric operations.
pub trait VectorOps<Scalar, Rhs = Self>:
    Sized
    + Vector
    + Add<Rhs, Output = Self>
    + Sub<Rhs, Output = Self>
    + Mul<Scalar, Output = Self>
    + Div<Scalar, Output = Self>
    + MulAdd<Scalar, Rhs, Output = Self>
{
}

/// The trait for vector types implementing numeric assignment operators (like `+ = `).
pub trait VectorAssignOps<Scalar, Rhs = Self>:
    Sized
    + AddAssign<Rhs>
    + SubAssign<Rhs>
    + MulAssign<Scalar>
    + DivAssign<Scalar>
    + MulAddAssign<Scalar, Rhs>
{
}

/// The trait for `Vector` types which also implement numeric operations
// taking the second operand by reference.
pub trait VectorRef<Scalar>: Vector + for<'a> VectorOps<Scalar, &'a Self> {}

impl<T, S> VectorRef<S> for T where T: Vector + for<'a> VectorOps<S, &'a T> {}

/// The trait for `Vector` types which also implement assignment operators.
pub trait VectorAssign<Scalar>: Vector + VectorAssignOps<Self, Scalar> {}

impl<T, S> VectorAssign<S> for T where T: Vector + VectorAssignOps<Self, S> {}

/// The trait for `VectorAssign` types which also implement
/// assignment operations taking the second operand by reference.
pub trait VectorAssignRef<Scalar>:
    VectorAssign<Scalar> + for<'a> VectorAssignOps<&'a Self, Scalar>
{
}

impl<T, S> VectorAssignRef<S> for T where T: VectorAssign<S> + for<'a> VectorAssignOps<&'a T, S> {}

/// The trait for types supporting the calculation of the dot product
pub trait Dot: Sized {
    /// The scalar type returned by `self`'s dot product
    type Scalar;

    /// Calculates the dot-product between `self` and `rhs`.
    fn dot(&self, rhs: &Self) -> Self::Scalar;
}

/// The trait for types supporting the calculation of distance
pub trait Distance: Sized {
    /// The scalar type returned by `self`'s distance
    type Scalar;

    /// Calculates the squared euclidian distance between `self` and `rhs`.
    fn squared_distance(&self, rhs: &Self) -> Self::Scalar;

    /// Calculates the euclidian distance between `self` and `rhs`.
    fn distance(&self, rhs: &Self) -> Self::Scalar
    where
        Self::Scalar: Real,
    {
        self.squared_distance(rhs).sqrt()
    }
}
