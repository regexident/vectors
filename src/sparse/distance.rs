use std::ops::{Mul, Sub};

use num_traits::Zero;

use crate::dense::DenseVector;
use crate::sparse::join::{outer_join, Join};
use crate::storage::Storage;

use super::SparseVector;
use crate::Distance;

impl<T, S> Distance for SparseVector<T, S>
where
    T: Copy + Zero + Sub<T, Output = T> + Mul<T, Output = T>,
    S: Storage<(usize, T)>,
{
    type Output = T;

    fn squared_distance(&self, rhs: &Self) -> <Self as Distance>::Output {
        outer_join(self.as_slice(), rhs.as_slice()).fold(T::zero(), |sum, (_, join)| match join {
            Join::Both(l, r) => {
                let delta = l - r;
                sum + (delta * delta)
            }
            Join::Left(l) => sum + (l * l),
            Join::Right(r) => sum + (r * r),
        })
    }
}

impl<T, S, S2> Distance<DenseVector<T, S2>> for SparseVector<T, S>
where
    T: Copy + Zero + Sub<T, Output = T> + Mul<T, Output = T>,
    S: Storage<(usize, T)>,
    S2: Storage<T>,
{
    type Output = T;

    fn squared_distance(&self, rhs: &DenseVector<T, S2>) -> <Self as Distance<DenseVector<T, S2>>>::Output {
        let sparse_slice = self.as_slice();
        let dense_slice = rhs.as_slice();
        let max_self_idx = sparse_slice.iter().map(|(i, _)| *i).max().unwrap_or(0);
        let capacity = dense_slice.len().max(max_self_idx + 1);

        let mut sum = T::zero();

        for idx in 0..capacity {
            let l_val = sparse_slice
                .iter()
                .find(|(i, _)| *i == idx)
                .map(|(_, v)| *v)
                .unwrap_or(T::zero());

            let r_val = if idx < dense_slice.len() {
                dense_slice[idx]
            } else {
                T::zero()
            };

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
    use crate::dense::DenseVector;

    #[test]
    fn squared_distance() {
        let subject = SparseVector::from(vec![(0, 0.2), (1, 0.5), (2, 1.0), (4, 2.0), (5, 4.0)]);
        let other = SparseVector::from(vec![(1, 0.1), (2, 0.2), (3, 0.3), (5, 0.4), (6, 0.5)]);
        let dense_l: DenseVector<_, Vec<_>> = DenseVector::from(vec![0.2, 0.5, 1.0, 0.0, 2.0, 4.0, 0.0]);
        let dense_r: DenseVector<_, Vec<_>> = DenseVector::from(vec![0.0, 0.1, 0.2, 0.3, 0.0, 0.4, 0.5]);
        let expected: f64 = dense_l.squared_distance(&dense_r);
        assert_relative_eq!(subject.squared_distance(&other), expected, epsilon = 1e-9);
    }

    #[test]
    fn distance() {
        let subject = SparseVector::from(vec![(0, 0.2), (1, 0.5), (2, 1.0), (4, 2.0), (5, 4.0)]);
        let other = SparseVector::from(vec![(1, 0.1), (2, 0.2), (3, 0.3), (5, 0.4), (6, 0.5)]);
        let dense_l: DenseVector<_, Vec<_>> = DenseVector::from(vec![0.2, 0.5, 1.0, 0.0, 2.0, 4.0, 0.0]);
        let dense_r: DenseVector<_, Vec<_>> = DenseVector::from(vec![0.0, 0.1, 0.2, 0.3, 0.0, 0.4, 0.5]);
        let expected: f64 = dense_l.distance(&dense_r);
        assert_relative_eq!(subject.distance(&other), expected, epsilon = 1e-9);
    }

    #[test]
    fn squared_distance_dense() {
        let sparse = SparseVector::from(vec![(0, 0.2_f64), (1, 0.5), (2, 1.0), (4, 2.0), (5, 4.0)]);
        let dense: DenseVector<_, Vec<_>> = DenseVector::from(vec![0.0_f64, 0.1, 0.2, 0.3, 0.0, 0.4, 0.5]);
        let sq_dist: f64 = sparse.squared_distance(&dense);
        let expected: f64 = 0.2_f64.powi(2)
            + (0.5_f64 - 0.1_f64).powi(2)
            + (1.0_f64 - 0.2_f64).powi(2)
            + (-0.3_f64).powi(2)
            + 2.0_f64.powi(2)
            + (4.0_f64 - 0.4_f64).powi(2)
            + (-0.5_f64).powi(2);
        assert_relative_eq!(sq_dist, expected, epsilon = 1e-9);
    }
}
