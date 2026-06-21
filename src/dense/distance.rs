use std::ops::{Mul, Sub};

use num_traits::{Signed, Zero};

use crate::Distance;
use crate::sparse::{GenericSparseVec, SparseStorage};

use super::{DenseStorage, GenericDenseVec};

impl<T, S> Distance for GenericDenseVec<T, S>
where
    T: Copy + Signed,
    S: DenseStorage<T>,
{
    type Output = T;

    /// # Panics
    ///
    /// Panics if `self` and `rhs` have different lengths.
    fn squared_distance(&self, rhs: &Self) -> <Self as Distance>::Output {
        let lhs_slice = self.storage.values();
        let rhs_slice = rhs.storage.values();
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

impl<Idx, T, S, S2> Distance<GenericSparseVec<Idx, T, S2>> for GenericDenseVec<T, S>
where
    Idx: Ord + Copy + Into<usize>,
    T: Copy + Signed + Zero + Sub<T, Output = T> + Mul<T, Output = T>,
    S: DenseStorage<T>,
    S2: SparseStorage<Idx, T>,
{
    type Output = T;

    fn squared_distance(
        &self,
        rhs: &GenericSparseVec<Idx, T, S2>,
    ) -> <Self as Distance<GenericSparseVec<Idx, T, S2>>>::Output {
        let dense_slice = self.storage.values();
        let (indices, values) = (rhs.storage.indices(), rhs.storage.values());

        let mut sum = T::zero();
        let mut dense_idx: usize = 0;
        let mut sparse_pos: usize = 0;

        while dense_idx < dense_slice.len() || sparse_pos < indices.len() {
            if sparse_pos >= indices.len()
                || (dense_idx < dense_slice.len() && dense_idx < indices[sparse_pos].into())
            {
                let l_val = dense_slice[dense_idx];
                sum = sum + (l_val * l_val);
                dense_idx += 1;
            } else if dense_idx >= dense_slice.len() || dense_idx > indices[sparse_pos].into() {
                let r_val = values[sparse_pos];
                sum = sum + (r_val * r_val);
                sparse_pos += 1;
            } else {
                let delta = dense_slice[dense_idx] - values[sparse_pos];
                sum = sum + (delta * delta);
                dense_idx += 1;
                sparse_pos += 1;
            }
        }

        sum
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use crate::TryFromIterator;
    use crate::dense::DenseVec;
    use crate::sparse::SparseVec;

    use super::*;

    use approx::assert_relative_eq;

    type Vector = DenseVec<f32>;

    #[test]
    fn squared_distance() {
        let subject: Vector = GenericDenseVec::try_from_iter([0.0, 0.5, 1.0, 2.0, 4.0]).unwrap();
        let other: Vector = GenericDenseVec::try_from_iter([0.1, 0.2, 0.3, 0.4, 0.0]).unwrap();
        assert_relative_eq!(subject.squared_distance(&other), 19.15, epsilon = 0.001);
    }

    #[test]
    fn distance() {
        let subject: Vector = GenericDenseVec::try_from_iter([0.0, 0.5, 1.0, 2.0, 4.0]).unwrap();
        let other: Vector = GenericDenseVec::try_from_iter([0.1, 0.2, 0.3, 0.4, 0.0]).unwrap();
        assert_relative_eq!(subject.distance(&other), 4.376, epsilon = 0.001);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn squared_distance_panics() {
        let a: Vector = GenericDenseVec::try_from_iter([0.0, 0.5]).unwrap();
        let b: Vector = GenericDenseVec::try_from_iter([0.1, 0.2, 0.3]).unwrap();
        let _ = a.squared_distance(&b);
    }

    #[test]
    fn squared_distance_sparse_dense_overlapping() {
        let dense: Vector = GenericDenseVec::try_from_iter([0.0, 0.5, 1.0, 2.0, 4.0]).unwrap();
        let sparse: SparseVec<usize, f32> =
            SparseVec::try_from_iter(vec![(0usize, 0.1), (2usize, 0.5), (4usize, 3.0)]).unwrap();
        assert_relative_eq!(dense.squared_distance(&sparse), 5.51, epsilon = 0.001);
    }

    #[test]
    fn squared_distance_sparse_dense_disjoint() {
        let dense: Vector = GenericDenseVec::try_from_iter([1.0, 2.0, 3.0]).unwrap();
        let sparse: SparseVec<usize, f32> =
            SparseVec::try_from_iter(vec![(10usize, 1.0), (11usize, 2.0)]).unwrap();
        assert_relative_eq!(dense.squared_distance(&sparse), 19.0, epsilon = 0.001);
    }

    #[test]
    fn squared_distance_sparse_dense_identical() {
        let dense: Vector = GenericDenseVec::try_from_iter([1.0, 2.0, 3.0]).unwrap();
        let sparse: SparseVec<usize, f32> =
            SparseVec::try_from_iter(vec![(0usize, 1.0), (1usize, 2.0), (2usize, 3.0)]).unwrap();
        assert_relative_eq!(dense.squared_distance(&sparse), 0.0, epsilon = 0.001);
    }
}
