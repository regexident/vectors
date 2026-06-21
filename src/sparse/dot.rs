use std::ops::Mul;

use num_traits::Zero;

use crate::{
    Dot, Index, Value,
    dense::{DenseStorage, DenseVector, GenericDenseVec},
};

use super::{GenericSparseVec, SparseStorage, SparseVector};

impl<Idx, T, S> Dot for GenericSparseVec<Idx, T, S>
where
    Idx: Ord + Copy,
    T: Copy + Mul<Output = T> + Zero,
    S: SparseStorage<Idx, T>,
{
    type Output = T;

    fn dot(&self, rhs: &Self) -> Self::Output {
        let (left_i, left_v) = (self.storage.indices(), self.storage.values());
        let (right_i, right_v) = (rhs.storage.indices(), rhs.storage.values());

        let mut result = T::zero();
        let mut left_pos = 0;
        let mut right_pos = 0;

        while left_pos < left_i.len() && right_pos < right_i.len() {
            let cmp = left_i[left_pos].cmp(&right_i[right_pos]);
            if cmp == std::cmp::Ordering::Less {
                left_pos += 1;
            } else if cmp == std::cmp::Ordering::Greater {
                right_pos += 1;
            } else {
                result = result + left_v[left_pos] * right_v[right_pos];
                left_pos += 1;
                right_pos += 1;
            }
        }

        result
    }
}

impl<Idx, T, S, S2> Dot<GenericDenseVec<T, S2>> for GenericSparseVec<Idx, T, S>
where
    Idx: Index,
    T: Value,
    S: SparseStorage<Idx, T>,
    S2: DenseStorage<T>,
{
    type Output = T;

    fn dot(&self, rhs: &GenericDenseVec<T, S2>) -> Self::Output {
        let (sparse_indices, sparse_values): (&[Idx], &[T]) = (self.indices(), self.values());
        let dense_values: &[T] = rhs.values();

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
