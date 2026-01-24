// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


use num_traits::Signed;
use ordered_iter::OrderedMapIterator;

use super::SparseVector;
use Distance;

impl<T, const N: usize> Distance for SparseVector<T, N>
where
    T: Copy + Signed,
    
{
    type Output = T;

    fn squared_distance(&self, rhs: &Self) -> Self::Output {
        let lhs_iter = self.iter();
        let rhs_iter = rhs.iter();
        lhs_iter
            .inner_join_map(rhs_iter)
            .fold(T::zero(), |sum, (_, (lhs, rhs))| {
                let delta = lhs - rhs;
                sum + (delta * delta)
            })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn squared_distance() {
        let subject = SparseVector::from([(0, 0.2), (1, 0.5), (2, 1.0), (4, 2.0), (5, 4.0)]);
        let other = SparseVector::from([(1, 0.1), (2, 0.2), (3, 0.3), (5, 0.4), (6, 0.5)]);
        let squared_distance = subject.squared_distance(&other);
        assert_relative_eq!(squared_distance, 13.76, epsilon = 0.001);
    }

    #[test]
    fn distance() {
        let subject = SparseVector::from([(0, 0.2), (1, 0.5), (2, 1.0), (4, 2.0), (5, 4.0)]);
        let other = SparseVector::from([(1, 0.1), (2, 0.2), (3, 0.3), (5, 0.4), (6, 0.5)]);
        let distance = subject.distance(&other);
        assert_relative_eq!(distance, 3.71, epsilon = 0.001);
    }
}
