use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use num_traits::{MulAdd, MulAddAssign};

use super::{DenseStorage, DenseVector, GenericDenseVec};

// MARK: Add

impl<T, S> Add<&Self> for GenericDenseVec<T, S>
where
    T: for<'a> AddAssign<&'a T>,
    S: DenseStorage<T>,
    Self: DenseVector<Value = T>,
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

impl<T, S> AddAssign<&Self> for GenericDenseVec<T, S>
where
    T: for<'a> AddAssign<&'a T>,
    S: DenseStorage<T>,
    Self: DenseVector<Value = T>,
{
    /// # Panics
    ///
    /// Panics if `self` and `rhs` have different lengths.
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        let lhs_slice = self.storage.values_mut();
        let rhs_slice = rhs.storage.values();
        assert_eq!(lhs_slice.len(), rhs_slice.len(), "dimension mismatch");
        for (lhs, rhs) in lhs_slice.iter_mut().zip(rhs_slice.iter()) {
            lhs.add_assign(rhs);
        }
    }
}

// MARK: Sub

impl<T, S> Sub<&Self> for GenericDenseVec<T, S>
where
    T: for<'a> SubAssign<&'a T>,
    S: DenseStorage<T>,
    Self: DenseVector<Value = T>,
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

impl<T, S> SubAssign<&Self> for GenericDenseVec<T, S>
where
    T: for<'a> SubAssign<&'a T>,
    S: DenseStorage<T>,
    Self: DenseVector<Value = T>,
{
    /// # Panics
    ///
    /// Panics if `self` and `rhs` have different lengths.
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        let lhs_slice = self.storage.values_mut();
        let rhs_slice = rhs.storage.values();
        assert_eq!(lhs_slice.len(), rhs_slice.len(), "dimension mismatch");
        for (lhs, rhs) in lhs_slice.iter_mut().zip(rhs_slice.iter()) {
            lhs.sub_assign(rhs);
        }
    }
}

// MARK: Mul

impl<T, S> Mul<T> for GenericDenseVec<T, S>
where
    T: Copy + MulAssign<T>,
    S: DenseStorage<T>,
    Self: DenseVector<Value = T>,
{
    type Output = Self;

    #[inline]
    fn mul(mut self, rhs: T) -> Self::Output {
        self.mul_assign(rhs);
        self
    }
}

impl<T, S> MulAssign<T> for GenericDenseVec<T, S>
where
    T: Copy + MulAssign<T>,
    S: DenseStorage<T>,
    Self: DenseVector<Value = T>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        for lhs in self.storage.values_mut() {
            *lhs *= rhs;
        }
    }
}

// MARK: Div

impl<T, S> Div<T> for GenericDenseVec<T, S>
where
    T: Copy + DivAssign<T>,
    S: DenseStorage<T>,
    Self: DenseVector<Value = T>,
{
    type Output = Self;

    #[inline]
    fn div(mut self, rhs: T) -> Self::Output {
        self.div_assign(rhs);
        self
    }
}

impl<T, S> DivAssign<T> for GenericDenseVec<T, S>
where
    T: Copy + DivAssign<T>,
    S: DenseStorage<T>,
    Self: DenseVector<Value = T>,
{
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        for lhs in self.storage.values_mut() {
            *lhs /= rhs;
        }
    }
}

// MARK: MulAdd

impl<T, S> MulAdd<T, &Self> for GenericDenseVec<T, S>
where
    T: Copy + MulAddAssign<T, T>,
    S: DenseStorage<T>,
    Self: DenseVector<Value = T>,
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

impl<T, S> MulAddAssign<T, &Self> for GenericDenseVec<T, S>
where
    T: Copy + MulAddAssign<T, T>,
    S: DenseStorage<T>,
    Self: DenseVector<Value = T>,
{
    /// # Panics
    ///
    /// Panics if `self` and `b` have different lengths.
    #[inline]
    fn mul_add_assign(&mut self, a: T, b: &Self) {
        let lhs_slice = self.storage.values_mut();
        let rhs_slice = b.storage.values();
        assert_eq!(lhs_slice.len(), rhs_slice.len(), "dimension mismatch");
        for (lhs, rhs) in lhs_slice.iter_mut().zip(rhs_slice.iter()) {
            lhs.mul_add_assign(a, *rhs);
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    use crate::TryFromIterator;
    use crate::dense::DenseVec;

    type Vector = DenseVec<f64>;

    #[test]
    fn add() {
        let subject: Vector = GenericDenseVec::try_from_iter([0.0, 0.5, 1.0, 2.0, 3.0]).unwrap();
        let other: Vector = GenericDenseVec::try_from_iter([2.0, 1.0, 0.0, -1.0, -2.0]).unwrap();
        let expected: Vector = GenericDenseVec::try_from_iter([2.0, 1.5, 1.0, 1.0, 1.0]).unwrap();
        assert_eq!(subject + &other, expected);
    }

    #[test]
    fn add_assign() {
        let mut subject: Vector =
            GenericDenseVec::try_from_iter([0.0, 0.5, 1.0, 2.0, 3.0]).unwrap();
        let other: Vector = GenericDenseVec::try_from_iter([2.0, 1.0, 0.0, -1.0, -2.0]).unwrap();
        subject += &other;
        assert_eq!(
            subject,
            GenericDenseVec::try_from_iter([2.0, 1.5, 1.0, 1.0, 1.0]).unwrap()
        );
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn add_panics_on_dimension_mismatch() {
        let subject: Vector = GenericDenseVec::try_from_iter([0.0, 0.5]).unwrap();
        let other: Vector = GenericDenseVec::try_from_iter([2.0, 1.0, 0.0]).unwrap();
        let _ = subject + &other;
    }

    #[test]
    fn sub() {
        let subject: Vector = GenericDenseVec::try_from_iter([2.0, 1.5, 1.0, 1.0, 1.0]).unwrap();
        let other: Vector = GenericDenseVec::try_from_iter([2.0, 1.0, 0.0, -1.0, -2.0]).unwrap();
        assert_eq!(
            subject - &other,
            GenericDenseVec::try_from_iter([0.0, 0.5, 1.0, 2.0, 3.0]).unwrap()
        );
    }

    #[test]
    fn mul() {
        let subject: Vector = GenericDenseVec::try_from_iter([0.0, 0.5, 1.25, 2.0, 3.0]).unwrap();
        assert_eq!(
            subject * 2.0,
            GenericDenseVec::try_from_iter([0.0, 1.0, 2.5, 4.0, 6.0]).unwrap()
        );
    }

    #[test]
    fn div() {
        let subject: Vector = GenericDenseVec::try_from_iter([0.0, 1.0, 2.0, 4.0, 6.0]).unwrap();
        assert_eq!(
            subject / 2.0,
            GenericDenseVec::try_from_iter([0.0, 0.5, 1.0, 2.0, 3.0]).unwrap()
        );
    }

    #[test]
    fn mul_add() {
        let subject: Vector = GenericDenseVec::try_from_iter([0.0, 0.5, 1.0, 2.0, 3.0]).unwrap();
        let other: Vector = GenericDenseVec::try_from_iter([2.0, 1.0, 0.0, -1.0, -2.0]).unwrap();
        assert_eq!(
            other.mul_add(2.0, &subject),
            GenericDenseVec::try_from_iter([4.0, 2.5, 1.0, 0.0, -1.0]).unwrap()
        );
    }
}
