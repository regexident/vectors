use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use num_traits::{MulAdd, MulAddAssign, Zero};

use super::SparseVector;
use crate::sparse::join::{Join, outer_join};
use crate::storage::Storage;

// MARK: Add

impl<Idx, T, S> Add<&Self> for SparseVector<Idx, T, S>
where
    Idx: Ord + Copy,
    T: Copy + Zero + Add<T, Output = T>,
    S: Storage<(Idx, T)>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: &Self) -> Self::Output {
        self.add_assign(rhs);
        self
    }
}

impl<Idx, T, S> AddAssign<&Self> for SparseVector<Idx, T, S>
where
    Idx: Ord + Copy,
    T: Copy + Zero + Add<T, Output = T>,
    S: Storage<(Idx, T)>,
{
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        self.components = {
            let joined = outer_join(self.as_slice(), rhs.as_slice());
            S::from_iter(joined.filter_map(|(index, join)| {
                let value = match join {
                    Join::Both(l, r) => l.add(r),
                    Join::Left(l) => l,
                    Join::Right(r) => r,
                };
                if value.is_zero() {
                    None
                } else {
                    Some((index, value))
                }
            }))
        };
    }
}

// MARK: Sub

impl<Idx, T, S> Sub<&Self> for SparseVector<Idx, T, S>
where
    Idx: Ord + Copy,
    T: Copy + Zero + Sub<T, Output = T>,
    S: Storage<(Idx, T)>,
{
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        self.sub_assign(rhs);
        self
    }
}

impl<Idx, T, S> SubAssign<&Self> for SparseVector<Idx, T, S>
where
    Idx: Ord + Copy,
    T: Copy + Zero + Sub<T, Output = T>,
    S: Storage<(Idx, T)>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        self.components = {
            let joined = outer_join(self.as_slice(), rhs.as_slice());
            S::from_iter(joined.filter_map(|(index, join)| {
                let value = match join {
                    Join::Both(l, r) => l.sub(r),
                    Join::Left(l) => l,
                    Join::Right(r) => T::zero().sub(r),
                };
                if value.is_zero() {
                    None
                } else {
                    Some((index, value))
                }
            }))
        };
    }
}

// MARK: Mul

impl<Idx, T, S> Mul<T> for SparseVector<Idx, T, S>
where
    Idx: Ord + Copy,
    T: Copy + Zero + Mul<T, Output = T>,
    S: Storage<(Idx, T)>,
{
    type Output = Self;

    #[inline]
    fn mul(mut self, rhs: T) -> Self::Output {
        self.mul_assign(rhs);
        self
    }
}

impl<Idx, T, S> MulAssign<T> for SparseVector<Idx, T, S>
where
    Idx: Ord + Copy,
    T: Copy + Zero + Mul<T, Output = T>,
    S: Storage<(Idx, T)>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        self.components = {
            let joined = self.iter().filter_map(|(index, lhs)| {
                let value = lhs * rhs;
                if value.is_zero() {
                    None
                } else {
                    Some((index, value))
                }
            });
            S::from_iter(joined)
        };
    }
}

// MARK: Div

impl<Idx, T, S> Div<T> for SparseVector<Idx, T, S>
where
    Idx: Ord + Copy,
    T: Copy + Zero + Div<T, Output = T>,
    S: Storage<(Idx, T)>,
{
    type Output = Self;

    #[inline]
    fn div(mut self, rhs: T) -> Self::Output {
        self.div_assign(rhs);
        self
    }
}

impl<Idx, T, S> DivAssign<T> for SparseVector<Idx, T, S>
where
    Idx: Ord + Copy,
    T: Copy + Zero + Div<T, Output = T>,
    S: Storage<(Idx, T)>,
{
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.components = {
            let joined = self.iter().filter_map(|(index, lhs)| {
                let value = lhs / rhs;
                if value.is_zero() {
                    None
                } else {
                    Some((index, value))
                }
            });
            S::from_iter(joined)
        };
    }
}

// MARK: MulAdd

impl<Idx, T, S> MulAdd<T, &Self> for SparseVector<Idx, T, S>
where
    Idx: Ord + Copy,
    T: Copy + Zero + MulAdd<T, T, Output = T>,
    S: Storage<(Idx, T)>,
{
    type Output = Self;

    #[inline]
    fn mul_add(mut self, a: T, b: &Self) -> Self::Output {
        self.mul_add_assign(a, b);
        self
    }
}

impl<Idx, T, S> MulAddAssign<T, &Self> for SparseVector<Idx, T, S>
where
    Idx: Ord + Copy,
    T: Copy + Zero + MulAdd<T, T, Output = T>,
    S: Storage<(Idx, T)>,
{
    #[inline]
    fn mul_add_assign(&mut self, a: T, b: &Self) {
        self.components = {
            let joined = outer_join(self.as_slice(), b.as_slice());
            S::from_iter(joined.filter_map(|(index, join)| {
                let value = match join {
                    Join::Both(l, r) => l.mul_add(a, r),
                    Join::Left(l) => l.mul_add(a, T::zero()),
                    Join::Right(r) => T::zero().mul_add(a, r),
                };
                if value.is_zero() {
                    None
                } else {
                    Some((index, value))
                }
            }))
        };
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn add() {
        let subject =
            SparseVector::try_from(vec![(0, 0.2), (1, 0.5), (2, 1.0), (4, 2.0), (5, 4.0)]).unwrap();
        let other = SparseVector::try_from(vec![(1, 0.1), (2, 0.2), (3, 0.3), (5, 0.4)]).unwrap();
        let expected = SparseVector::try_from(vec![
            (0, 0.2),
            (1, 0.6),
            (2, 1.2),
            (3, 0.3),
            (4, 2.0),
            (5, 4.4),
        ])
        .unwrap();
        assert_eq!(subject + &other, expected);
    }

    #[test]
    fn sub() {
        let subject =
            SparseVector::try_from(vec![(0, 0.2), (1, 0.6), (2, 1.2), (4, 2.0), (5, 0.4)]).unwrap();
        let other = SparseVector::try_from(vec![(1, 0.1), (2, 0.2), (3, 0.3), (5, 0.4)]).unwrap();
        let expected =
            SparseVector::try_from(vec![(0, 0.2), (1, 0.5), (2, 1.0), (3, -0.3), (4, 2.0)])
                .unwrap();
        assert_eq!(subject - &other, expected);
    }

    #[test]
    fn mul() {
        let subject =
            SparseVector::try_from(vec![(0, 0.2), (1, 0.5), (2, 1.0), (4, 2.0), (5, 4.0)]).unwrap();
        assert_eq!(
            subject * 2.0,
            SparseVector::try_from(vec![(0, 0.4), (1, 1.0), (2, 2.0), (4, 4.0), (5, 8.0)]).unwrap()
        );
    }

    #[test]
    fn div() {
        let subject =
            SparseVector::try_from(vec![(0, 0.4), (1, 1.0), (2, 2.0), (4, 4.0), (5, 8.0)]).unwrap();
        assert_eq!(
            subject / 2.0,
            SparseVector::try_from(vec![(0, 0.2), (1, 0.5), (2, 1.0), (4, 2.0), (5, 4.0)]).unwrap()
        );
    }

    #[test]
    fn mul_add() {
        let subject = SparseVector::try_from(vec![(1, 0.1), (2, 0.2), (3, 0.3), (5, 0.4)]).unwrap();
        let other =
            SparseVector::try_from(vec![(0, 0.2), (1, 0.5), (2, 1.0), (4, 2.0), (5, 4.0)]).unwrap();
        let expected = SparseVector::try_from(vec![
            (0, 0.2),
            (1, 0.7),
            (2, 1.4),
            (3, 0.6),
            (4, 2.0),
            (5, 4.8),
        ])
        .unwrap();
        assert_eq!(subject.mul_add(2.0, &other), expected);
    }

    #[test]
    fn stack_add() {
        use crate::sparse::StackSparseVector;
        use std::iter::FromIterator;
        type SV = StackSparseVector<usize, f32, 6>;
        let subject = SV::from_iter(vec![(0, 0.2), (1, 0.5), (2, 1.0), (4, 2.0), (5, 4.0)]);
        let other = SV::from_iter(vec![(1, 0.1), (2, 0.2), (3, 0.3), (5, 0.4)]);
        let _result = subject + &other;
    }
}
