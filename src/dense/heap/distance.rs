// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use num_traits::Signed;

use super::DenseVector;
use Distance;

impl<T> Distance for DenseVector<T>
where
    T: Copy + Signed,
{
    type Output = T;

    fn squared_distance(&self, rhs: &Self) -> Self::Output {
        let lhs_iter = self.iter();
        let rhs_iter = rhs.iter();
        debug_assert_eq!(lhs_iter.len(), rhs_iter.len());
        lhs_iter
            .zip(rhs_iter)
            .fold(T::zero(), |sum, ((_, lhs), (_, rhs))| {
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
        let subject = DenseVector::from(vec![0.0, 0.5, 1.0, 2.0, 4.0]);
        let other = DenseVector::from(vec![0.1, 0.2, 0.3, 0.4, 0.0]);
        let squared_distance = subject.squared_distance(&other);
        assert_relative_eq!(squared_distance, 19.15, epsilon = 0.001);
    }

    #[test]
    fn distance() {
        let subject = DenseVector::from(vec![0.0, 0.5, 1.0, 2.0, 4.0]);
        let other = DenseVector::from(vec![0.1, 0.2, 0.3, 0.4, 0.0]);
        let distance = subject.distance(&other);
        assert_relative_eq!(distance, 4.376, epsilon = 0.001);
    }
}
