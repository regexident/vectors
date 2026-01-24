// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use num_traits::Num;

use super::DenseVector;
use Dot;

impl<'a, T> Dot for DenseVector<T>
where
    T: Copy + Num,
{
    type Output = T;

    fn dot(&self, rhs: &Self) -> Self::Output {
        debug_assert_eq!(self.len(), rhs.len());
        let lhs_iter = self.components.iter();
        let rhs_iter = rhs.components.iter();
        lhs_iter
            .zip(rhs_iter)
            .fold(T::zero(), |sum, (lhs, rhs)| sum + ((*lhs) * (*rhs)))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn dot() {
        let subject = DenseVector::from(vec![0.0, 0.5, 1.0, 2.0, 4.0]);
        let other = DenseVector::from(vec![0.1, 0.2, 0.3, 0.4, 0.0]);

        let dot = subject.dot(&other);
        assert_relative_eq!(dot, 1.2, epsilon = 0.001);
    }
}
