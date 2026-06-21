use std::ops::{Add, Mul};

use num_traits::Zero;

use crate::Index;

/// Compute the dot product of a dense slice with a sparse vector's
/// index/value slices.
#[must_use]
pub(crate) fn dot_mixed<T, Idx>(dense: &[T], sparse_i: &[Idx], sparse_v: &[T]) -> T
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Zero,
    Idx: Index,
{
    let mut sum = T::zero();
    let mut sparse_pos = 0;
    let mut dense_pos = Idx::zero();

    for dense_val in dense.iter() {
        if sparse_pos < sparse_i.len() && sparse_i[sparse_pos] == dense_pos {
            sum = sum + sparse_v[sparse_pos] * *dense_val;
            sparse_pos += 1;
        }
        dense_pos += Idx::one();
    }

    sum
}
