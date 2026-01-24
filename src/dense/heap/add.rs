// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::ops::{Add, AddAssign};

use super::DenseVector;

impl<'a, T> Add<&'a Self> for DenseVector<T>
where
    T: AddAssign<&'a T>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: &'a Self) -> Self::Output {
        self.add_assign(rhs);
        self
    }
}

impl<'a, T> AddAssign<&'a Self> for DenseVector<T>
where
    T: AddAssign<&'a T>,
{
    #[inline]
    fn add_assign(&mut self, rhs: &'a Self) {
        let iter = rhs.components.iter();
        debug_assert_eq!(self.len(), iter.len());
        for (lhs, rhs) in self.components.iter_mut().zip(iter) {
            lhs.add_assign(rhs);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use expectest::prelude::*;

    #[test]
    fn add() {
        let subject = DenseVector::from(vec![0.0, 0.5, 1.0, 2.0, 3.0]);
        let other = DenseVector::from(vec![2.0, 1.0, 0.0, -1.0, -2.0]);
        let expected = DenseVector::from(vec![2.0, 1.5, 1.0, 1.0, 1.0]);
        let result = subject + &other;
        expect!(result).to(be_equal_to(expected));
    }

    #[test]
    fn add_assign() {
        let subject = DenseVector::from(vec![0.0, 0.5, 1.0, 2.0, 3.0]);
        let other = DenseVector::from(vec![2.0, 1.0, 0.0, -1.0, -2.0]);
        let expected = DenseVector::from(vec![2.0, 1.5, 1.0, 1.0, 1.0]);
        let mut result = subject;
        result += &other;
        expect!(result).to(be_equal_to(expected));
    }
}
