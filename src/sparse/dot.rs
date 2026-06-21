use std::ops::{Add, Mul};

use num_traits::Zero;

use crate::dense::DenseVector;
use crate::sparse::join::inner_join;
use crate::storage::Storage;

use super::SparseVector;
use crate::Dot;

impl<Idx, T, S> Dot for SparseVector<Idx, T, S>
where
    Idx: Ord + Copy,
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Zero,
    S: Storage<(Idx, T)>,
{
    type Output = T;

    fn dot(&self, rhs: &Self) -> <Self as Dot>::Output {
        inner_join(self.as_slice(), rhs.as_slice())
            .fold(T::zero(), |sum, (_, (lhs, rhs))| sum + (lhs * rhs))
    }
}

impl<Idx, T, S, S2> Dot<DenseVector<T, S2>> for SparseVector<Idx, T, S>
where
    Idx: Ord + Copy + Into<usize>,
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Zero,
    S: Storage<(Idx, T)>,
    S2: Storage<T>,
{
    type Output = T;

    fn dot(&self, rhs: &DenseVector<T, S2>) -> <Self as Dot<DenseVector<T, S2>>>::Output {
        let dense_slice = rhs.as_slice();
        self.as_slice().iter().fold(T::zero(), |sum, (idx, val)| {
            let idx: usize = (*idx).into();
            if idx < dense_slice.len() {
                sum + (*val * dense_slice[idx])
            } else {
                sum
            }
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::dense::DenseVector;
    use approx::assert_relative_eq;

    #[test]
    fn dot() {
        let subject =
            SparseVector::try_from(vec![(0, 0.2), (1, 0.5), (2, 1.0), (4, 2.0), (5, 4.0)]).unwrap();
        let other =
            SparseVector::try_from(vec![(1, 0.1), (2, 0.2), (3, 0.3), (5, 0.4), (6, 0.5)]).unwrap();
        assert_relative_eq!(subject.dot(&other), 1.85, epsilon = 0.001);
    }

    #[test]
    fn dot_dense() {
        let sparse = SparseVector::try_from(vec![
            (0usize, 0.2_f64),
            (1usize, 0.5),
            (2usize, 1.0),
            (4usize, 2.0),
            (5usize, 4.0),
        ])
        .unwrap();
        let dense: DenseVector<_, Vec<_>> =
            DenseVector::from(vec![0.0_f64, 0.1, 0.2, 0.3, 0.0, 0.4, 0.5]);
        let dot: f64 = sparse.dot(&dense);
        let expected: f64 =
            0.2 * 0.0 + 0.5 * 0.1 + 1.0 * 0.2 + 0.0 * 0.3 + 2.0 * 0.0 + 4.0 * 0.4 + 0.0 * 0.5;
        assert_relative_eq!(dot, expected, epsilon = 1e-9);
    }
}
