use num_traits::Num;

use crate::dense::DenseVector as _;
use crate::sparse::{GenericSparseVec, SparseStorage, SparseVector as _};
use crate::{Dot, Index, Value};

use super::{DenseStorage, GenericDenseVec};

impl<T, S> Dot for GenericDenseVec<T, S>
where
    T: Copy + Num,
    S: DenseStorage<T>,
{
    type Output = T;

    /// # Panics
    ///
    /// Panics if `self` and `rhs` have different lengths.
    fn dot(&self, rhs: &Self) -> <Self as Dot>::Output {
        let lhs_slice = self.storage.values();
        let rhs_slice = rhs.storage.values();
        assert_eq!(lhs_slice.len(), rhs_slice.len(), "dimension mismatch");
        lhs_slice
            .iter()
            .zip(rhs_slice.iter())
            .fold(T::zero(), |sum, (lhs, rhs)| sum + ((*lhs) * (*rhs)))
    }
}

impl<Idx, T, S, S2> Dot<GenericSparseVec<Idx, T, S2>> for GenericDenseVec<T, S>
where
    Idx: Index,
    T: Value,
    S: DenseStorage<T>,
    S2: SparseStorage<Idx, T>,
{
    type Output = T;

    fn dot(
        &self,
        rhs: &GenericSparseVec<Idx, T, S2>,
    ) -> <Self as Dot<GenericSparseVec<Idx, T, S2>>>::Output {
        let (sparse_indices, sparse_values): (&[Idx], &[T]) = (rhs.indices(), rhs.values());
        let dense_values: &[T] = self.values();

        let mut sum = T::zero();

        let mut sparse_pos = 0;
        let mut dense_pos = Idx::zero();

        for dense_val in dense_values.iter() {
            if sparse_pos < sparse_indices.len() && sparse_indices[sparse_pos] == dense_pos {
                sum = sum + sparse_values[sparse_pos] * *dense_val;
                sparse_pos += 1;
            }
            dense_pos += Idx::one();
        }

        sum
    }
}

#[cfg(all(test, feature = "std"))]
mod test2 {
    use approx::assert_relative_eq;

    use crate::TryFromIterator;
    use crate::dense::DenseVec;

    use super::*;

    type Vector = DenseVec<f32>;

    #[test]
    fn dot() {
        let subject: Vector = GenericDenseVec::try_from_iter([0.0, 0.5, 1.0, 2.0, 4.0]).unwrap();
        let other: Vector = GenericDenseVec::try_from_iter([0.1, 0.2, 0.3, 0.4, 0.0]).unwrap();
        assert_relative_eq!(subject.dot(&other), 1.2, epsilon = 0.001);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn dot_panics_on_dimension_mismatch() {
        let a: Vector = GenericDenseVec::try_from_iter([0.0, 0.5]).unwrap();
        let b: Vector = GenericDenseVec::try_from_iter([0.1, 0.2, 0.3]).unwrap();
        let _ = a.dot(&b);
    }

    #[test]
    fn dot_sparse() {
        let dense: Vector =
            GenericDenseVec::try_from_iter([0.0, 0.5, 1.0, 2.0, 4.0, 0.0, 0.0]).unwrap();
        let sparse: crate::sparse::SparseVec<usize, f32> =
            crate::sparse::SparseVec::try_from_iter(vec![
                (1usize, 0.1_f32),
                (2usize, 0.2),
                (3usize, 0.3),
                (5usize, 0.4),
                (6usize, 0.5),
            ])
            .unwrap();
        let dot: f32 = dense.dot(&sparse);
        let expected: f32 =
            0.0 * 0.0 + 0.5 * 0.1 + 1.0 * 0.2 + 2.0 * 0.3 + 4.0 * 0.0 + 0.0 * 0.4 + 0.0 * 0.5;
        assert_relative_eq!(dot, expected, epsilon = 1e-9);
    }
}
