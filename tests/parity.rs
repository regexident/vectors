use num_traits::{MulAdd, Zero};
use proptest::prelude::*;
use vectors::dense::DenseVector;
use vectors::sparse::SparseVector;
use vectors::{Distance, Dot};

/// Convert a sparse vector to a dense equivalent by zero-filling missing indices.
fn to_dense<T: Copy + Zero>(sparse: &SparseVector<T, Vec<(usize, T)>>) -> DenseVector<T, Vec<T>> {
    let max_index = sparse.iter().map(|(i, _)| i).max().unwrap_or(0);
    let capacity = max_index + 1;
    let mut dense_components = vec![T::zero(); capacity];
    for (idx, val) in sparse.iter() {
        dense_components[idx] = val;
    }
    DenseVector::from(dense_components)
}

fn pad_dense(v: &DenseVector<f64, Vec<f64>>, target_len: usize) -> DenseVector<f64, Vec<f64>> {
    let mut components: Vec<f64> = v.iter().map(|(_, val)| val).collect();
    while components.len() < target_len {
        components.push(0.0);
    }
    DenseVector::from(components)
}

fn assert_dense_eq(a: &DenseVector<f64, Vec<f64>>, b: &DenseVector<f64, Vec<f64>>, epsilon: f64) {
    let a_vals: Vec<_> = a.iter().map(|(_, v)| v).collect();
    let b_vals: Vec<_> = b.iter().map(|(_, v)| v).collect();
    assert_eq!(a_vals.len(), b_vals.len(), "length mismatch");
    for (i, (av, bv)) in a_vals.iter().zip(b_vals.iter()).enumerate() {
        assert!((av - bv).abs() < epsilon, "index {}: {} != {}", i, av, bv);
    }
}

// MARK: Strategies

fn sparse_strategy() -> impl Strategy<Value = SparseVector<f64, Vec<(usize, f64)>>> {
    proptest::collection::vec((0_usize..20, -10.0_f64..10.0), 0..15)
        .prop_map(|mut pairs| {
            pairs.sort_by_key(|(k, _)| *k);
            pairs.dedup_by_key(|(k, _)| *k);
            pairs.retain(|(_, v)| !v.is_zero());
            SparseVector::from_sorted_unchecked(pairs)
        })
}

fn sparse_pair_strategy() -> impl Strategy<Value = (SparseVector<f64, Vec<(usize, f64)>>, SparseVector<f64, Vec<(usize, f64)>>)> {
    (sparse_strategy(), sparse_strategy())
}

// MARK: Parity properties

proptest! {
    #[test]
    fn sparse_add_equals_dense_add((a, b) in sparse_pair_strategy()) {
        let sparse_result = a.clone() + &b;
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
        let sparse_result = a.clone() - &b;
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
        let a: SparseVector<f64, _> = SparseVector::from(vec![]);
        let b: SparseVector<f64, _> = SparseVector::from(vec![]);
        let dot: f64 = a.dot(&b);
        let dist: f64 = a.squared_distance(&b);
        assert_eq!(dot, 0.0);
        assert_eq!(dist, 0.0);
        let sum = a + &b;
        assert!(sum.is_empty());
    }

    #[test]
    fn single_element_vectors() {
        let a = SparseVector::from(vec![(5, 3.0_f64)]);
        let b = SparseVector::from(vec![(5, 4.0_f64)]);
        let dot: f64 = a.dot(&b);
        let dist: f64 = a.squared_distance(&b);
        assert!((dot - 12.0).abs() < 1e-9);
        assert!((dist - 1.0).abs() < 1e-9);
    }

    #[test]
    fn single_element_different_keys() {
        let a = SparseVector::from(vec![(1, 3.0_f64)]);
        let b = SparseVector::from(vec![(5, 4.0_f64)]);
        let dot: f64 = a.dot(&b);
        let dist: f64 = a.squared_distance(&b);
        assert!((dot - 0.0).abs() < 1e-9);
        assert!((dist - 25.0).abs() < 1e-9);
    }

    #[test]
    fn large_indices() {
        let a = SparseVector::from(vec![(0, 1.0_f64), (1_000_000, 2.0_f64)]);
        let b = SparseVector::from(vec![(1_000_000, 3.0_f64)]);
        let dot: f64 = a.dot(&b);
        let dist: f64 = a.squared_distance(&b);
        assert!((dot - 6.0).abs() < 1e-9);
        assert!((dist - 2.0).abs() < 1e-9);
    }

    #[test]
    fn zero_dimensional_after_canonicalization() {
        let a = SparseVector::try_from_unsorted(vec![(0, 0.0_f64)]);
        assert!(a.is_empty());
    }

    #[test]
    fn nan_propagation_f64() {
        let a = SparseVector::from(vec![(0, f64::NAN)]);
        let b = SparseVector::from(vec![(0, 1.0_f64)]);
        let dot: f64 = a.dot(&b);
        assert!(dot.is_nan());
        let dist: f64 = a.squared_distance(&b);
        assert!(dist.is_nan());
    }
}
