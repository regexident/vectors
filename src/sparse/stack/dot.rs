// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::ops::{Add, Mul};


use num_traits::Zero;
use ordered_iter::OrderedMapIterator;

use super::SparseVector;
use Dot;

impl<T, const N: usize> Dot for SparseVector<T, N>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Zero,
    
{
    type Output = T;

    fn dot(&self, rhs: &Self) -> Self::Output {
        let lhs_iter = self.iter();
        let rhs_iter = rhs.iter();
        lhs_iter
            .inner_join_map(rhs_iter)
            .fold(T::zero(), |sum, (_, (lhs, rhs))| sum + (lhs * rhs))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn dot() {
        let subject = SparseVector::from([(0, 0.2), (1, 0.5), (2, 1.0), (4, 2.0), (5, 4.0)]);
        let other = SparseVector::from([(1, 0.1), (2, 0.2), (3, 0.3), (5, 0.4), (6, 0.5)]);

        let dot = subject.dot(&other);
        assert_relative_eq!(dot, 1.85, epsilon = 0.001);
    }
}
