// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Sparse stack-allocated vector representation.

use std::ops::{Add, Div, Mul, Sub};
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

use arrayvec::ArrayVec;
use num_traits::{MulAdd, MulAddAssign, NumAssign};

use {Vector, VectorAssignOps, VectorOps};

mod add;
mod div;
mod mul;
mod mul_add;
mod sub;

mod distance;
mod dot;

mod debug;
mod iter;

pub use self::iter::{IntoIter, Iter};

/// A sparse stack-allocated multi-dimensional vector.
pub struct SparseVector<T, const N: usize> {
    components: ArrayVec<(usize, T), N>,
}

impl<T, const N: usize> SparseVector<T, N> {
    /// The number of components in `self`
    #[inline]
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// `true` if `self.len() == 0`, otherwise `false`
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// A borrowing iterator over `self`
    #[inline]
    pub fn iter<'a>(&'a self) -> Iter<'a, T> {
        Iter::new(&self.components[..])
    }
}

impl<T, const N: usize> Clone for SparseVector<T, N>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        let components = self.components.clone();
        Self { components }
    }
}

impl<T, const N: usize> PartialEq for SparseVector<T, N>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.components.eq(&other.components)
    }
}

impl<T, const N: usize> From<[(usize, T); N]> for SparseVector<T, N> {
    #[inline]
    fn from(items: [(usize, T); N]) -> Self {
        Self {
            components: ArrayVec::from(items),
        }
    }
}

impl<T, const N: usize> Vector for SparseVector<T, N>
where
    T: Copy + NumAssign + MulAdd<T, T, Output = T>,
{
    type Scalar = T;
}

impl<T, V, const N: usize> VectorOps<T, V> for SparseVector<T, N>
where
    Self: Add<V, Output = Self>
        + Sub<V, Output = Self>
        + Mul<T, Output = Self>
        + Div<T, Output = Self>
        + MulAdd<T, V, Output = Self>,
    T: Copy + NumAssign + MulAdd<T, T, Output = T>,
    V: Vector<Scalar = T>,
{
}

impl<T, V, const N: usize> VectorAssignOps<T, V> for SparseVector<T, N>
where
    Self: AddAssign<V> + SubAssign<V> + MulAssign<T> + DivAssign<T> + MulAddAssign<T, V>,
    T: Copy + NumAssign + MulAddAssign<T, T>,
    V: Vector<Scalar = T>,
{
}

#[cfg(test)]
mod test {
    use super::*;

    use expectest::prelude::*;

    #[test]
    fn from() {
        const VALUES: [(usize, f32); 5] = [(0, 0.0), (1, 1.0), (2, 0.5), (4, 0.25), (8, 0.125)];
        let subject = SparseVector::from(VALUES);
        let expected = ArrayVec::from(VALUES);
        expect!(subject.components).to(be_equal_to(expected));
    }
}
