use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use num_traits::{MulAdd, MulAddAssign};

use super::DenseVector;
use crate::storage::Storage;

// MARK: Add

impl<T, S> Add<&Self> for DenseVector<T, S>
where
    T: for<'a> AddAssign<&'a T>,
    S: Storage<T>,
{
    type Output = Self;

    /// # Panics
    ///
    /// Panics if `self` and `rhs` have different lengths.
    #[inline]
    fn add(mut self, rhs: &Self) -> Self::Output {
        self.add_assign(rhs);
        self
    }
}

impl<T, S> AddAssign<&Self> for DenseVector<T, S>
where
    T: for<'a> AddAssign<&'a T>,
    S: Storage<T>,
{
    /// # Panics
    ///
    /// Panics if `self` and `rhs` have different lengths.
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        let lhs_slice = self.components.as_mut();
        let rhs_slice = rhs.components.as_ref();
        assert_eq!(lhs_slice.len(), rhs_slice.len(), "dimension mismatch");
        for (lhs, rhs) in lhs_slice.iter_mut().zip(rhs_slice.iter()) {
            lhs.add_assign(rhs);
        }
    }
}

// MARK: Sub

impl<T, S> Sub<&Self> for DenseVector<T, S>
where
    T: for<'a> SubAssign<&'a T>,
    S: Storage<T>,
{
    type Output = Self;

    /// # Panics
    ///
    /// Panics if `self` and `rhs` have different lengths.
    #[inline]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        self.sub_assign(rhs);
        self
    }
}

impl<T, S> SubAssign<&Self> for DenseVector<T, S>
where
    T: for<'a> SubAssign<&'a T>,
    S: Storage<T>,
{
    /// # Panics
    ///
    /// Panics if `self` and `rhs` have different lengths.
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        let lhs_slice = self.components.as_mut();
        let rhs_slice = rhs.components.as_ref();
        assert_eq!(lhs_slice.len(), rhs_slice.len(), "dimension mismatch");
        for (lhs, rhs) in lhs_slice.iter_mut().zip(rhs_slice.iter()) {
            lhs.sub_assign(rhs);
        }
    }
}

// MARK: Mul

impl<T, S> Mul<T> for DenseVector<T, S>
where
    T: Copy + MulAssign<T>,
    S: Storage<T>,
{
    type Output = Self;

    #[inline]
    fn mul(mut self, rhs: T) -> Self::Output {
        self.mul_assign(rhs);
        self
    }
}

impl<T, S> MulAssign<T> for DenseVector<T, S>
where
    T: Copy + MulAssign<T>,
    S: Storage<T>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        for lhs in self.components.as_mut() {
            *lhs *= rhs;
        }
    }
}

// MARK: Div

impl<T, S> Div<T> for DenseVector<T, S>
where
    T: Copy + DivAssign<T>,
    S: Storage<T>,
{
    type Output = Self;

    #[inline]
    fn div(mut self, rhs: T) -> Self::Output {
        self.div_assign(rhs);
        self
    }
}

impl<T, S> DivAssign<T> for DenseVector<T, S>
where
    T: Copy + DivAssign<T>,
    S: Storage<T>,
{
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        for lhs in self.components.as_mut() {
            *lhs /= rhs;
        }
    }
}

// MARK: MulAdd

impl<T, S> MulAdd<T, &Self> for DenseVector<T, S>
where
    T: Copy + MulAddAssign<T, T>,
    S: Storage<T>,
{
    type Output = Self;

    /// # Panics
    ///
    /// Panics if `self` and `b` have different lengths.
    #[inline]
    fn mul_add(mut self, a: T, b: &Self) -> Self::Output {
        self.mul_add_assign(a, b);
        self
    }
}

impl<T, S> MulAddAssign<T, &Self> for DenseVector<T, S>
where
    T: Copy + MulAddAssign<T, T>,
    S: Storage<T>,
{
    /// # Panics
    ///
    /// Panics if `self` and `b` have different lengths.
    #[inline]
    fn mul_add_assign(&mut self, a: T, b: &Self) {
        let lhs_slice = self.components.as_mut();
        let rhs_slice = b.components.as_ref();
        assert_eq!(lhs_slice.len(), rhs_slice.len(), "dimension mismatch");
        for (lhs, rhs) in lhs_slice.iter_mut().zip(rhs_slice.iter()) {
            lhs.mul_add_assign(a, *rhs);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::dense::StackDenseVector;

    type DV = DenseVector<f64, Vec<f64>>;

    #[test]
    fn add() {
        let subject: DV = DenseVector::from(vec![0.0, 0.5, 1.0, 2.0, 3.0]);
        let other: DV = DenseVector::from(vec![2.0, 1.0, 0.0, -1.0, -2.0]);
        let expected: DV = DenseVector::from(vec![2.0, 1.5, 1.0, 1.0, 1.0]);
        assert_eq!(subject + &other, expected);
    }

    #[test]
    fn add_assign() {
        let mut subject: DV = DenseVector::from(vec![0.0, 0.5, 1.0, 2.0, 3.0]);
        let other: DV = DenseVector::from(vec![2.0, 1.0, 0.0, -1.0, -2.0]);
        subject += &other;
        assert_eq!(subject, DenseVector::from(vec![2.0, 1.5, 1.0, 1.0, 1.0]));
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn add_panics_on_dimension_mismatch() {
        let subject: DV = DenseVector::from(vec![0.0, 0.5]);
        let other: DV = DenseVector::from(vec![2.0, 1.0, 0.0]);
        let _ = subject + &other;
    }

    #[test]
    fn sub() {
        let subject: DV = DenseVector::from(vec![2.0, 1.5, 1.0, 1.0, 1.0]);
        let other: DV = DenseVector::from(vec![2.0, 1.0, 0.0, -1.0, -2.0]);
        assert_eq!(
            subject - &other,
            DenseVector::from(vec![0.0, 0.5, 1.0, 2.0, 3.0])
        );
    }

    #[test]
    fn mul() {
        let subject: DV = DenseVector::from(vec![0.0, 0.5, 1.25, 2.0, 3.0]);
        assert_eq!(
            subject * 2.0,
            DenseVector::from(vec![0.0, 1.0, 2.5, 4.0, 6.0])
        );
    }

    #[test]
    fn div() {
        let subject: DV = DenseVector::from(vec![0.0, 1.0, 2.0, 4.0, 6.0]);
        assert_eq!(
            subject / 2.0,
            DenseVector::from(vec![0.0, 0.5, 1.0, 2.0, 3.0])
        );
    }

    #[test]
    fn mul_add() {
        let subject: DV = DenseVector::from(vec![0.0, 0.5, 1.0, 2.0, 3.0]);
        let other: DV = DenseVector::from(vec![2.0, 1.0, 0.0, -1.0, -2.0]);
        assert_eq!(
            other.mul_add(2.0, &subject),
            DenseVector::from(vec![4.0, 2.5, 1.0, 0.0, -1.0])
        );
    }

    #[test]
    fn stack_add() {
        let subject = StackDenseVector::from([0.0, 0.5, 1.0, 2.0, 3.0]);
        let other = StackDenseVector::from([2.0, 1.0, 0.0, -1.0, -2.0]);
        let expected = StackDenseVector::from([2.0, 1.5, 1.0, 1.0, 1.0]);
        assert_eq!(subject + &other, expected);
    }
}
