// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Dense heap-allocated vector representation.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

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

/// A dense heap-allocated multi-dimensional vector.
#[derive(Clone, PartialEq)]
pub struct DenseVector<T> {
    components: Vec<T>,
}

impl<T> DenseVector<T> {
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

impl<T> From<Vec<T>> for DenseVector<T> {
    #[inline]
    fn from(items: Vec<T>) -> Self {
        Self { components: items }
    }
}

impl<V, T> VectorOps<V, T> for DenseVector<T>
where
    Self: Add<V, Output = Self>
        + Sub<V, Output = Self>
        + Mul<T, Output = Self>
        + Div<T, Output = Self>
        + MulAdd<T, V, Output = Self>,
    T: Copy + NumAssign + MulAdd<T, T, Output = T>,
{
}

impl<V, T> VectorAssignOps<V, T> for DenseVector<T>
where
    Self: AddAssign<V> + SubAssign<V> + MulAssign<T> + DivAssign<T> + MulAddAssign<T, V>,
    T: Copy + NumAssign + MulAddAssign,
{
}

impl<T> Vector<T> for DenseVector<T>
where
    Self: VectorOps<Self, T>,
    T: Copy + NumAssign + MulAdd<T, T, Output = T>,
{
    type Scalar = T;
}

#[cfg(test)]
mod test {
    use super::*;

    use expectest::prelude::*;

    #[test]
    fn from() {
        let values: Vec<_> = vec![0.0; 5];
        let subject = DenseVector::from(values.clone());
        expect!(subject.components).to(be_equal_to(values));
    }
}
