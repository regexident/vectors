use std::ops::{Mul, Sub};

use num_traits::{Signed, Zero};

use crate::sparse::SparseVector;
use crate::storage::Storage;

use super::DenseVector;
use crate::Distance;

impl<T, S> Distance for DenseVector<T, S>
where
    T: Copy + Signed,
    S: Storage<T>,
{
    type Output = T;

    /// # Panics
    ///
    /// Panics if `self` and `rhs` have different lengths.
    fn squared_distance(&self, rhs: &Self) -> <Self as Distance>::Output {
        let lhs_slice = self.components.as_ref();
        let rhs_slice = rhs.components.as_ref();
        assert_eq!(lhs_slice.len(), rhs_slice.len(), "dimension mismatch");
        lhs_slice
            .iter()
            .zip(rhs_slice.iter())
            .fold(T::zero(), |sum, (lhs, rhs)| {
                let delta = *lhs - *rhs;
                sum + (delta * delta)
            })
    }
}

impl<T, S, S2> Distance<SparseVector<T, S2>> for DenseVector<T, S>
where
    T: Copy + Signed + Zero + Sub<T, Output = T> + Mul<T, Output = T>,
    S: Storage<T>,
    S2: Storage<(usize, T)>,
{
    type Output = T;

    fn squared_distance(&self, rhs: &SparseVector<T, S2>) -> <Self as Distance<SparseVector<T, S2>>>::Output {
        let dense_slice = self.components.as_ref();
        let sparse_slice = rhs.as_slice();
        let max_rhs_idx = sparse_slice.iter().map(|(i, _)| *i).max().unwrap_or(0);
        let capacity = dense_slice.len().max(max_rhs_idx + 1);

        let mut sum = T::zero();

        for idx in 0..capacity {
            let l_val = if idx < dense_slice.len() {
                dense_slice[idx]
            } else {
                T::zero()
            };

            let r_val = sparse_slice
                .iter()
                .find(|(i, _)| *i == idx)
                .map(|(_, v)| *v)
                .unwrap_or(T::zero());

            let delta = l_val - r_val;
            sum = sum + (delta * delta);
        }

        sum
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use approx::assert_relative_eq;

    type DV = DenseVector<f64, Vec<f64>>;

    #[test]
    fn squared_distance() {
        let subject: DV = DenseVector::from(vec![0.0, 0.5, 1.0, 2.0, 4.0]);
        let other: DV = DenseVector::from(vec![0.1, 0.2, 0.3, 0.4, 0.0]);
        assert_relative_eq!(subject.squared_distance(&other), 19.15, epsilon = 0.001);
    }

    #[test]
    fn distance() {
        let subject: DV = DenseVector::from(vec![0.0, 0.5, 1.0, 2.0, 4.0]);
        let other: DV = DenseVector::from(vec![0.1, 0.2, 0.3, 0.4, 0.0]);
        assert_relative_eq!(subject.distance(&other), 4.376, epsilon = 0.001);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn squared_distance_panics() {
        let a: DV = DenseVector::from(vec![0.0, 0.5]);
        let b: DV = DenseVector::from(vec![0.1, 0.2, 0.3]);
        let _ = a.squared_distance(&b);
    }
}
