use num_traits::{MulAdd, Zero};
use proptest::prelude::*;

use vectors::dense::{DenseVec, DenseVecStorage, DenseVector};
use vectors::sparse::{SparseVec, SparseVector};
use vectors::{Distance, Dot, TryFromIterator};

/// Convert a sparse vector to a dense equivalent by zero-filling missing indices.
fn to_dense<T: Copy + Zero>(sparse: &SparseVec<usize, T>) -> DenseVec<T> {
    let max_index = sparse.indices().iter().copied().max().unwrap_or(0);
    let capacity = max_index + 1;
    let mut dense_components = vec![T::zero(); capacity];
    for i in 0..sparse.len() {
        dense_components[sparse.indices()[i]] = sparse.values()[i];
    }
    DenseVec::from(DenseVecStorage::from(dense_components))
}

fn pad_dense(v: &DenseVec<f64>, target_len: usize) -> DenseVec<f64> {
    let mut components: Vec<f64> = v.iter::<usize>().map(|(_, val)| val).collect();
    while components.len() < target_len {
        components.push(0.0);
    }
    DenseVec::from(DenseVecStorage::from(components))
}

fn assert_dense_eq(a: &DenseVec<f64>, b: &DenseVec<f64>, epsilon: f64) {
    let a_vals: Vec<_> = a.iter::<usize>().map(|(_, v)| v).collect();
    let b_vals: Vec<_> = b.iter::<usize>().map(|(_, v)| v).collect();
    assert_eq!(a_vals.len(), b_vals.len(), "length mismatch");
    for (i, (av, bv)) in a_vals.iter().zip(b_vals.iter()).enumerate() {
        assert!((av - bv).abs() < epsilon, "index {}: {} != {}", i, av, bv);
    }
}

// MARK: Strategies

fn sparse_strategy() -> impl Strategy<Value = SparseVec<usize, f64>> {
    proptest::collection::vec((0_usize..20, -10.0_f64..10.0), 0..15).prop_map(|mut pairs| {
        pairs.sort_by_key(|(k, _)| *k);
        pairs.dedup_by_key(|(k, _)| *k);
        pairs.retain(|(_, v)| !v.is_zero());
        SparseVec::try_from_iter(pairs).unwrap()
    })
}

fn sparse_pair_strategy() -> impl Strategy<Value = (SparseVec<usize, f64>, SparseVec<usize, f64>)> {
    (sparse_strategy(), sparse_strategy())
}

// MARK: Parity properties

proptest! {
    #[test]
    fn sparse_add_equals_dense_add((a, b) in sparse_pair_strategy()) {
        let sparse_result = a.clone() + b.clone();
        let dense_a = to_dense(&a);
        let dense_b = to_dense(&b);
        let max_len = dense_a.len().max(dense_b.len());
        let dense_a = pad_dense(&dense_a, max_len);
        let dense_b = pad_dense(&dense_b, max_len);
        let dense_expected = dense_a + &dense_b;
        let sparse_as_dense = to_dense(&sparse_result);
        let padded = pad_dense(&sparse_as_dense, max_len);
        assert_dense_eq(&padded, &dense_expected, 1e-9);
    }

    #[test]
    fn sparse_sub_equals_dense_sub((a, b) in sparse_pair_strategy()) {
        let sparse_result = a.clone() - b.clone();
        let dense_a = to_dense(&a);
        let dense_b = to_dense(&b);
        let max_len = dense_a.len().max(dense_b.len());
        let dense_a = pad_dense(&dense_a, max_len);
        let dense_b = pad_dense(&dense_b, max_len);
        let dense_expected = dense_a - &dense_b;
        let sparse_as_dense = to_dense(&sparse_result);
        let padded = pad_dense(&sparse_as_dense, max_len);
        assert_dense_eq(&padded, &dense_expected, 1e-9);
    }

    #[test]
    fn sparse_mul_add_equals_dense_mul_add((a, b) in sparse_pair_strategy()) {
        let scalar = 2.0_f64;
        let sparse_result = a.clone().mul_add(scalar, &b);
        let dense_a = to_dense(&a);
        let dense_b = to_dense(&b);
        let max_len = dense_a.len().max(dense_b.len());
        let dense_a = pad_dense(&dense_a, max_len);
        let dense_b = pad_dense(&dense_b, max_len);
        let dense_expected = dense_a.mul_add(scalar, &dense_b);
        let sparse_as_dense = to_dense(&sparse_result);
        let padded = pad_dense(&sparse_as_dense, max_len);
        assert_dense_eq(&padded, &dense_expected, 1e-9);
    }

    #[test]
    fn sparse_dot_equals_dense_dot((a, b) in sparse_pair_strategy()) {
        let sparse_result = a.dot(&b);
        let dense_a = to_dense(&a);
        let dense_b = to_dense(&b);
        let max_len = dense_a.len().max(dense_b.len());
        let dense_a = pad_dense(&dense_a, max_len);
        let dense_b = pad_dense(&dense_b, max_len);
        let dense_expected = dense_a.dot(&dense_b);
        assert!((sparse_result - dense_expected).abs() < 1e-9,
            "sparse dot {} != dense dot {}", sparse_result, dense_expected);
    }

    #[test]
    fn dense_sparse_squared_distance_equals_dense_dense_squared_distance((a, b) in sparse_pair_strategy()) {
        let dense_a = to_dense(&a);
        let dense_b = to_dense(&b);
        let max_len = dense_a.len().max(dense_b.len());
        let dense_a = pad_dense(&dense_a, max_len);
        let dense_b_converted = to_dense(&b);
        let dense_b_padded = pad_dense(&dense_b_converted, max_len);
        let dense_dist = dense_a.squared_distance(&dense_b_padded);
        let cross_dist = dense_a.squared_distance(&b);
        assert!((dense_dist - cross_dist).abs() < 1e-9,
            "dense×sparse distance {} != dense×dense distance {}", cross_dist, dense_dist);
    }

    #[test]
    fn sparse_squared_distance_equals_dense_squared_distance((a, b) in sparse_pair_strategy()) {
        let sparse_result = a.squared_distance(&b);
        let dense_a = to_dense(&a);
        let dense_b = to_dense(&b);
        let max_len = dense_a.len().max(dense_b.len());
        let dense_a = pad_dense(&dense_a, max_len);
        let dense_b = pad_dense(&dense_b, max_len);
        let dense_expected = dense_a.squared_distance(&dense_b);
        assert!((sparse_result - dense_expected).abs() < 1e-9,
            "sparse squared_distance {} != dense squared_distance {}", sparse_result, dense_expected);
    }
}

// MARK: Edge-case unit tests

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn empty_sparse_vectors() {
        let a: SparseVec<usize, f64> = SparseVec::try_from_iter(vec![]).unwrap();
        let b: SparseVec<usize, f64> = SparseVec::try_from_iter(vec![]).unwrap();
        let dot: f64 = a.dot(&b);
        let dist: f64 = a.squared_distance(&b);
        assert_eq!(dot, 0.0);
        assert_eq!(dist, 0.0);
        let sum = a + b;
        assert!(sum.is_empty());
    }

    #[test]
    fn single_element_vectors() {
        let a: SparseVec<usize, f64> = SparseVec::try_from_iter(vec![(5, 3.0_f64)]).unwrap();
        let b: SparseVec<usize, f64> = SparseVec::try_from_iter(vec![(5, 4.0_f64)]).unwrap();
        let dot: f64 = a.dot(&b);
        let dist: f64 = a.squared_distance(&b);
        assert!((dot - 12.0).abs() < 1e-9);
        assert!((dist - 1.0).abs() < 1e-9);
    }

    #[test]
    fn single_element_different_keys() {
        let a: SparseVec<usize, f64> = SparseVec::try_from_iter(vec![(1, 3.0_f64)]).unwrap();
        let b: SparseVec<usize, f64> = SparseVec::try_from_iter(vec![(5, 4.0_f64)]).unwrap();
        let dot: f64 = a.dot(&b);
        let dist: f64 = a.squared_distance(&b);
        assert!((dot - 0.0).abs() < 1e-9);
        assert!((dist - 25.0).abs() < 1e-9);
    }

    #[test]
    fn large_indices() {
        let a: SparseVec<usize, f64> =
            SparseVec::try_from_iter(vec![(0, 1.0_f64), (1_000_000, 2.0_f64)]).unwrap();
        let b: SparseVec<usize, f64> =
            SparseVec::try_from_iter(vec![(1_000_000, 3.0_f64)]).unwrap();
        let dot: f64 = a.dot(&b);
        let dist: f64 = a.squared_distance(&b);
        assert!((dot - 6.0).abs() < 1e-9);
        assert!((dist - 2.0).abs() < 1e-9);
    }

    #[test]
    fn zero_valued_entries_are_filtered() {
        let a: SparseVec<usize, f64> = SparseVec::try_from_iter(vec![(0, 0.0_f64)]).unwrap();
        assert_eq!(a.len(), 0);
    }

    #[test]
    fn nan_propagation_f64() {
        let a: SparseVec<usize, f64> = SparseVec::try_from_iter(vec![(0, f64::NAN)]).unwrap();
        let b: SparseVec<usize, f64> = SparseVec::try_from_iter(vec![(0, 1.0_f64)]).unwrap();
        let dot: f64 = a.dot(&b);
        assert!(dot.is_nan());
        let dist: f64 = a.squared_distance(&b);
        assert!(dist.is_nan());
    }
}
