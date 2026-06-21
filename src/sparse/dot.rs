use std::ops::{Add, Mul};

use num_traits::Zero;

use crate::{
    Dot, Index, Value,
    common::dot_mixed,
    dense::{DenseStorage, DenseVector, GenericDenseVec},
};

use super::{GenericSparseVec, SparseStorage, SparseVector, galloping_seek};

impl<Idx, T, S> Dot for GenericSparseVec<Idx, T, S>
where
    Idx: Ord + Copy,
    T: Copy + Mul<Output = T> + Zero + Add<Output = T>,
    S: SparseStorage<Idx, T>,
{
    type Output = T;

    fn dot(&self, rhs: &Self) -> Self::Output {
        let (lhs_i, lhs_v) = (self.storage.indices(), self.storage.values());
        let (rhs_i, rhs_v) = (rhs.storage.indices(), rhs.storage.values());
        if lhs_i.is_empty() || rhs_i.is_empty() {
            return T::zero();
        }
        if lhs_i.len() <= rhs_i.len() {
            dot_gallop(lhs_i, lhs_v, rhs_i, rhs_v)
        } else {
            dot_gallop(rhs_i, rhs_v, lhs_i, lhs_v)
        }
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
        dot_mixed(rhs.values(), self.indices(), self.values())
    }
}

/// Dot product using galloping (exponential) search on the larger array.
///
/// Iterates over the smaller side element-by-element, using exponential
/// probes to skip ahead in the larger side. O(small * log large) in the
/// worst case.
///
/// The caller must pass the *smaller* side as `small_i`/`small_v` and the
/// *larger* side as `large_i`/`large_v` for optimal performance.
#[inline]
pub fn dot_gallop<Idx, T>(small_i: &[Idx], small_v: &[T], large_i: &[Idx], large_v: &[T]) -> T
where
    Idx: Ord + Copy,
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Zero,
{
    assert_eq!(
        small_i.len(),
        small_v.len(),
        "parallel slice length mismatch"
    );
    assert_eq!(
        large_i.len(),
        large_v.len(),
        "parallel slice length mismatch"
    );

    let mut sum = T::zero();
    let mut cursor = 0;

    for k in 0..small_i.len() {
        // SAFETY: `k` is bounded by `k < small_i.len()` via the loop
        // `for k in 0..small_i.len()`, so this access is in-bounds.
        let target = unsafe { *small_i.get_unchecked(k) };

        if cursor >= large_i.len() {
            break;
        }

        // SAFETY: `cursor < large_i.len()` is guarded by the check above,
        // so this access is in-bounds.
        let current = unsafe { *large_i.get_unchecked(cursor) };
        if current == target {
            assert!(k < small_v.len());
            assert!(cursor < large_v.len());
            // SAFETY: Both `k` and `cursor` are in-bounds — `k` is bounded by the
            // loop, `cursor` by the prior `cursor < n` check. `small_v` and `large_v`
            // are parallel slices to `small_i`/`large_i`, so lengths match.
            sum = sum + (unsafe { *small_v.get_unchecked(k) * *large_v.get_unchecked(cursor) });
            cursor += 1;
            continue;
        }
        if current > target {
            continue;
        }

        // INVARIANT: large_i[cursor] < target.
        if let Some(pos) = galloping_seek(large_i, &mut cursor, target) {
            assert!(k < small_v.len());
            assert!(pos < large_v.len());
            // SAFETY: Both `k` and `pos` are in-bounds — `k` is bounded by the
            // loop and `pos` was returned by `galloping_seek` which guarantees
            // `pos < large.len()`.
            sum = sum + (unsafe { *small_v.get_unchecked(k) * *large_v.get_unchecked(pos) });
        }
    }

    sum
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::TryFromIterator;
    use crate::sparse::SparseVec;
    use approx::assert_relative_eq;

    type Vector = SparseVec<usize, f64>;

    fn make_v(pairs: &[(usize, f64)]) -> Vector {
        Vector::try_from_iter(pairs.iter().copied()).unwrap()
    }

    /// Run `dot_gallop` on both orderings and verify they agree.
    fn check_all(l_i: &[usize], l_v: &[f64], r_i: &[usize], r_v: &[f64]) {
        if l_i.is_empty() || r_i.is_empty() {
            let forward = if l_i.len() <= r_i.len() {
                dot_gallop(l_i, l_v, r_i, r_v)
            } else {
                dot_gallop(r_i, r_v, l_i, l_v)
            };
            assert_relative_eq!(forward, 0.0, epsilon = 1e-12);
            return;
        }
        let forward = dot_gallop(l_i, l_v, r_i, r_v);
        let reverse = dot_gallop(r_i, r_v, l_i, l_v);
        assert_relative_eq!(forward, reverse, epsilon = 1e-12);
    }

    #[test]
    fn all_inputs_agree() {
        check_all(
            &[0, 1, 2, 4, 5],
            &[0.2, 0.5, 1.0, 2.0, 4.0],
            &[1, 2, 3, 5, 6],
            &[0.1, 0.2, 0.3, 0.4, 0.5],
        );

        check_all(
            &[0, 10, 100],
            &[1.0, 2.0, 3.0],
            &(0..500).map(|i| i * 2).collect::<Vec<_>>(),
            &(0..500).map(|i| (i as f64) * 0.1).collect::<Vec<_>>(),
        );

        check_all(
            &(0..500).map(|i| i * 2).collect::<Vec<_>>(),
            &(0..500).map(|i| (i as f64) * 0.1).collect::<Vec<_>>(),
            &[0, 10, 100],
            &[1.0, 2.0, 3.0],
        );

        check_all(
            &[0, 1, 2],
            &[1.0, 2.0, 3.0],
            &[100, 200, 300],
            &[4.0, 5.0, 6.0],
        );

        let all: Vec<usize> = (0..50).collect();
        let all_v: Vec<f64> = (0..50).map(|i| i as f64).collect();
        check_all(&all, &all_v, &all, &all_v);

        check_all(&[], &[], &[0, 1, 2], &[1.0, 2.0, 3.0]);
        check_all(&[], &[], &[], &[]);

        check_all(&[5], &[3.0], &[5], &[4.0]);
        check_all(&[5], &[3.0], &[7], &[4.0]);
    }

    #[test]
    fn dot_sparse_sparse() {
        let a = make_v(&[(0, 0.2), (1, 0.5), (2, 1.0), (4, 2.0), (5, 4.0)]);
        let b = make_v(&[(1, 0.1), (2, 0.2), (3, 0.3), (5, 0.4), (6, 0.5)]);
        assert_relative_eq!(a.dot(&b), 1.85, epsilon = 0.001);
    }

    #[test]
    fn gallop_single_element_near_end() {
        let a_i: Vec<usize> = (0..10).map(|i| i * 2).collect(); // [0,2,4,6,8,10,12,14,16,18]
        let a_v: Vec<f64> = vec![1.0; 10];
        let b_i = [18usize];
        let b_v = [2.5f64];
        assert_eq!(dot_gallop(&b_i, &b_v, &a_i, &a_v), 2.5);
    }

    #[test]
    fn gallop_single_element_at_first() {
        let a_i: Vec<usize> = (1..=9).collect(); // [1,2,3,4,5,6,7,8,9]
        let a_v: Vec<f64> = vec![1.0; 9];
        let b_i = [1usize];
        let b_v = [3.0f64];
        assert_eq!(dot_gallop(&b_i, &b_v, &a_i, &a_v), 3.0);
    }

    #[test]
    fn gallop_single_in_middle() {
        let a_i: Vec<usize> = (1..=9).collect(); // [1,2,3,4,5,6,7,8,9]
        let a_v: Vec<f64> = (0..9).map(|i| (i + 1) as f64).collect();
        let b_i = [5usize];
        let b_v = [10.0f64];
        assert_eq!(dot_gallop(&b_i, &b_v, &a_i, &a_v), 50.0);
    }

    #[test]
    fn gallop_no_match_single() {
        let a_i: Vec<usize> = (1..=9).step_by(2).collect(); // [1,3,5,7,9]
        let a_v: Vec<f64> = vec![1.0; 5];
        let b_i = [2usize];
        let b_v = [10.0f64];
        assert_eq!(dot_gallop(&b_i, &b_v, &a_i, &a_v), 0.0);
    }
}
