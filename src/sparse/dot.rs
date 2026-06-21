use std::ops::{Add, Mul};

use num_traits::{One, Zero};

use crate::{
    Dot, Index, Value,
    dense::{DenseStorage, DenseVector, GenericDenseVec},
};

use super::join::ADAPTIVE_THRESHOLD;
use super::{GenericSparseVec, SparseStorage, SparseVector};

/// Dot product using the classic merge-join strategy.
///
/// Compares indices at the current position of each side and advances
/// the smaller side until a match is found. O(n + m) comparisons.
#[inline]
pub fn dot_merge<Idx, T>(lhs_i: &[Idx], lhs_v: &[T], rhs_i: &[Idx], rhs_v: &[T]) -> T
where
    Idx: Ord + Copy,
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Zero,
{
    let mut i = 0;
    let mut j = 0;
    let mut sum = T::zero();

    while i < lhs_i.len() && j < rhs_i.len() {
        let ai = lhs_i[i];
        let bi = rhs_i[j];

        if ai == bi {
            sum = sum + (lhs_v[i] * rhs_v[j]);
            i += 1;
            j += 1;
        } else if ai < bi {
            i += 1;
        } else {
            j += 1;
        }
    }

    sum
}

/// Dot product using branchless comparison masks.
///
/// Computes an equality mask (1.0 when indices match, 0.0 otherwise)
/// and uses it to unconditionally accumulate the product — eliminating
/// the equality branch at the cost of a predictable multiply-by-zero on
/// non-matching positions. Pointer advancement also uses boolean-to-usize
/// casts to avoid branches.
#[inline]
pub fn dot_branchless<Idx, T>(lhs_i: &[Idx], lhs_v: &[T], rhs_i: &[Idx], rhs_v: &[T]) -> T
where
    Idx: Ord + Copy,
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Zero + One,
{
    let (na, nb) = (lhs_i.len(), rhs_i.len());
    let (mut i, mut j) = (0, 0);
    let mut acc = T::zero();

    while i < na && j < nb {
        let ai = lhs_i[i];
        let bj = rhs_i[j];

        let eq = ai == bj;
        let lt = ai < bj;

        let mask = if eq { T::one() } else { T::zero() };
        acc = acc + mask * (lhs_v[i] * rhs_v[j]);

        i += (eq || lt) as usize;
        j += (eq || !lt) as usize;
    }

    acc
}

/// Dot product using galloping (exponential) search on the larger array.
///
/// Iterates over the smaller side element-by-element, using exponential
/// probes to skip ahead in the larger side. O(small * log large) in the
/// worst case. Most effective when one side is at least
/// [`ADAPTIVE_THRESHOLD`] times the size of the other.
#[inline]
pub fn dot_gallop<Idx, T>(small_i: &[Idx], small_v: &[T], large_i: &[Idx], large_v: &[T]) -> T
where
    Idx: Ord + Copy,
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Zero,
{
    let mut sum = T::zero();
    let mut cursor = 0;
    let n = large_i.len();

    for k in 0..small_i.len() {
        let target = unsafe { *small_i.get_unchecked(k) };

        if cursor >= n {
            break;
        }

        let current = unsafe { *large_i.get_unchecked(cursor) };
        if current == target {
            sum = sum + (unsafe { *small_v.get_unchecked(k) * *large_v.get_unchecked(cursor) });
            cursor += 1;
            continue;
        }
        if current > target {
            continue;
        }

        // INVARIANT: large_i[cursor] < target.
        //
        // Phase 1: exponential search to find the first probe with
        // large_i[probe] >= target.
        let (last_lt, final_step) = {
            let mut step: usize = 1;
            let mut last_lt = cursor;
            loop {
                let probe = cursor.saturating_add(step);
                if probe >= n {
                    break (last_lt, 0);
                }
                if unsafe { *large_i.get_unchecked(probe) } >= target {
                    break (last_lt, step);
                }
                last_lt = probe;
                step = match step.checked_shl(1) {
                    Some(s) => s,
                    None => break (last_lt, 0),
                };
            }
        };

        if final_step == 0 {
            cursor = n;
            continue;
        }

        // Phase 2: one-sided galloping — only advance the lower bound
        // forward with successively halved step sizes.
        let mut step = final_step >> 1;
        let mut pos = last_lt;
        while step > 0 {
            let probe = pos + step;
            if probe < n && unsafe { *large_i.get_unchecked(probe) } < target {
                pos = probe;
            }
            step >>= 1;
        }

        let insertion = pos + 1;

        if insertion < n && unsafe { *large_i.get_unchecked(insertion) } == target {
            sum = sum + (unsafe { *small_v.get_unchecked(k) * *large_v.get_unchecked(insertion) });
            cursor = insertion + 1;
        } else {
            cursor = insertion;
        }
    }

    sum
}

/// Dot product with adaptive dispatch between merge and galloping strategies.
///
/// Delegates to [`dot_merge`] for balanced inputs and [`dot_gallop`] when one
/// side is at least `threshold` times the size of the other.
#[inline]
pub fn dot_adaptive<Idx, T>(
    lhs_i: &[Idx],
    lhs_v: &[T],
    rhs_i: &[Idx],
    rhs_v: &[T],
    threshold: usize,
) -> T
where
    Idx: Ord + Copy,
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Zero,
{
    let (n, m) = (lhs_i.len(), rhs_i.len());

    if n == 0 || m == 0 {
        return T::zero();
    }

    if n <= m {
        if m >= threshold * n.max(1) {
            dot_gallop(lhs_i, lhs_v, rhs_i, rhs_v)
        } else {
            dot_merge(lhs_i, lhs_v, rhs_i, rhs_v)
        }
    } else if n >= threshold * m.max(1) {
        dot_gallop(rhs_i, rhs_v, lhs_i, lhs_v)
    } else {
        dot_merge(lhs_i, lhs_v, rhs_i, rhs_v)
    }
}

impl<Idx, T, S> Dot for GenericSparseVec<Idx, T, S>
where
    Idx: Ord + Copy,
    T: Copy + Mul<Output = T> + Zero + Add<Output = T>,
    S: SparseStorage<Idx, T>,
{
    type Output = T;

    fn dot(&self, rhs: &Self) -> Self::Output {
        dot_adaptive(
            self.storage.indices(),
            self.storage.values(),
            rhs.storage.indices(),
            rhs.storage.values(),
            ADAPTIVE_THRESHOLD,
        )
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

    /// Verify all four dot variants produce the same result across diverse inputs.
    fn check_all(l_i: &[usize], l_v: &[f64], r_i: &[usize], r_v: &[f64]) {
        let m = dot_merge(l_i, l_v, r_i, r_v);
        let g = if l_i.len() <= r_i.len() {
            dot_gallop(l_i, l_v, r_i, r_v)
        } else {
            dot_gallop(r_i, r_v, l_i, l_v)
        };
        let a = dot_adaptive(l_i, l_v, r_i, r_v, ADAPTIVE_THRESHOLD);
        let b = dot_branchless(l_i, l_v, r_i, r_v);

        assert_relative_eq!(m, g, epsilon = 1e-12);
        assert_relative_eq!(m, a, epsilon = 1e-12);
        assert_relative_eq!(m, b, epsilon = 1e-12);
    }

    #[test]
    fn all_variants_agree() {
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
}
