use num_traits::{Num, Zero};

use crate::sparse::SparseVector;
use crate::storage::Storage;

use super::DenseVector;
use crate::Dot;

impl<T, S> Dot for DenseVector<T, S>
where
    T: Copy + Num,
    S: Storage<T>,
{
    type Output = T;

    /// # Panics
    ///
    /// Panics if `self` and `rhs` have different lengths.
    fn dot(&self, rhs: &Self) -> <Self as Dot>::Output {
        let lhs_slice = self.components.as_ref();
        let rhs_slice = rhs.components.as_ref();
        assert_eq!(lhs_slice.len(), rhs_slice.len(), "dimension mismatch");
        lhs_slice
            .iter()
            .zip(rhs_slice.iter())
            .fold(T::zero(), |sum, (lhs, rhs)| sum + ((*lhs) * (*rhs)))
    }
}

impl<Idx, T, S, S2> Dot<SparseVector<Idx, T, S2>> for DenseVector<T, S>
where
    Idx: Into<usize> + Copy,
    T: Copy + Num + Zero,
    S: Storage<T>,
    S2: Storage<(Idx, T)>,
{
    type Output = T;

    fn dot(
        &self,
        rhs: &SparseVector<Idx, T, S2>,
    ) -> <Self as Dot<SparseVector<Idx, T, S2>>>::Output {
        let dense_slice = self.components.as_ref();
        rhs.as_slice().iter().fold(T::zero(), |sum, (idx, val)| {
            let idx: usize = (*idx).into();
            if idx < dense_slice.len() {
                sum + (dense_slice[idx] * (*val))
            } else {
                sum
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;

    type DV = DenseVector<f64, Vec<f64>>;

    #[test]
    fn dot() {
        let subject: DV = DenseVector::from(vec![0.0, 0.5, 1.0, 2.0, 4.0]);
        let other: DV = DenseVector::from(vec![0.1, 0.2, 0.3, 0.4, 0.0]);
        assert_relative_eq!(subject.dot(&other), 1.2, epsilon = 0.001);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn dot_panics_on_dimension_mismatch() {
        let a: DV = DenseVector::from(vec![0.0, 0.5]);
        let b: DV = DenseVector::from(vec![0.1, 0.2, 0.3]);
        let _ = a.dot(&b);
    }

    #[test]
    fn dot_sparse() {
        let dense: DV = DenseVector::from(vec![0.0_f64, 0.5, 1.0, 2.0, 4.0, 0.0, 0.0]);
        let sparse = crate::sparse::SparseVector::try_from(vec![
            (1usize, 0.1_f64),
            (2usize, 0.2),
            (3usize, 0.3),
            (5usize, 0.4),
            (6usize, 0.5),
        ])
        .unwrap();
        let dot: f64 = dense.dot(&sparse);
        let expected: f64 =
            0.0 * 0.0 + 0.5 * 0.1 + 1.0 * 0.2 + 2.0 * 0.3 + 4.0 * 0.0 + 0.0 * 0.4 + 0.0 * 0.5;
        assert_relative_eq!(dot, expected, epsilon = 1e-9);
    }
}
