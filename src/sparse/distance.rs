use std::ops::{Add, Mul, Sub};

use num_traits::Zero;

use crate::Distance;

use super::{GenericSparseVec, SparseStorage};

impl<Idx, T, S> Distance for GenericSparseVec<Idx, T, S>
where
    Idx: Ord + Copy,
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Zero,
    S: SparseStorage<Idx, T>,
{
    type Output = T;

    fn squared_distance(&self, rhs: &Self) -> Self::Output {
        let (left_i, left_v) = (self.storage.indices(), self.storage.values());
        let (right_i, right_v) = (rhs.storage.indices(), rhs.storage.values());

        let mut result = T::zero();
        let mut left_pos = 0;
        let mut right_pos = 0;

        while left_pos < left_i.len() || right_pos < right_i.len() {
            if right_pos >= right_i.len()
                || (left_pos < left_i.len() && left_i[left_pos] < right_i[right_pos])
            {
                result = result + left_v[left_pos] * left_v[left_pos];
                left_pos += 1;
            } else if left_pos >= left_i.len() || right_i[right_pos] < left_i[left_pos] {
                result = result + right_v[right_pos] * right_v[right_pos];
                right_pos += 1;
            } else {
                let d = left_v[left_pos] - right_v[right_pos];
                result = result + d * d;
                left_pos += 1;
                right_pos += 1;
            }
        }

        result
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use crate::TryFromIterator;
    use crate::sparse::SparseVec;

    use super::*;

    use approx::assert_relative_eq;

    type Vector = SparseVec<usize, f32>;

    #[test]
    fn squared_distance_overlapping() {
        let a: Vector = SparseVec::try_from_iter(vec![(0, 1.0), (1, 2.0), (2, 3.0)]).unwrap();
        let b: Vector = SparseVec::try_from_iter(vec![(0, 3.0), (1, 2.0), (2, 1.0)]).unwrap();
        assert_relative_eq!(a.squared_distance(&b), 8.0, epsilon = 0.001);
    }

    #[test]
    fn squared_distance_disjoint() {
        let a: Vector = SparseVec::try_from_iter(vec![(0, 1.0), (1, 2.0)]).unwrap();
        let b: Vector = SparseVec::try_from_iter(vec![(10, 3.0), (11, 4.0)]).unwrap();
        assert_relative_eq!(a.squared_distance(&b), 30.0, epsilon = 0.001);
    }

    #[test]
    fn squared_distance_empty() {
        let a: Vector = SparseVec::try_from_iter(vec![]).unwrap();
        let b: Vector = SparseVec::try_from_iter(vec![(0, 5.0)]).unwrap();
        assert_relative_eq!(a.squared_distance(&b), 25.0, epsilon = 0.001);
    }

    #[test]
    fn squared_distance_both_empty() {
        let a: Vector = SparseVec::try_from_iter(vec![]).unwrap();
        let b: Vector = SparseVec::try_from_iter(vec![]).unwrap();
        assert_relative_eq!(a.squared_distance(&b), 0.0, epsilon = 0.001);
    }
}
