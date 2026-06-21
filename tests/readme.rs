use vectors::dense::{DenseVector, StackDenseVector};
use vectors::sparse::SparseVector;
use vectors::{Distance, Dot};

#[test]
fn readme_basic_example() {
    let dense_1 = DenseVector::from(vec![0.0, 0.5, 1.0, 2.0, 4.0]);
    let dense_2 = DenseVector::from(vec![0.1, 0.2, 0.3, 0.4, 0.0]);
    let _dot = dense_1.dot(&dense_2);
    let _dist = dense_1.distance(&dense_2);

    let sparse_1 = SparseVector::try_from(vec![
        (0usize, 0.2),
        (2usize, 0.2),
        (4usize, 0.3),
        (6usize, 0.4),
    ])
    .unwrap();
    let sparse_2 = SparseVector::try_from(vec![
        (0usize, 0.2),
        (3usize, 0.4),
        (5usize, 0.2),
        (6usize, 0.6),
    ])
    .unwrap();
    let _dot = sparse_1.dot(&sparse_2);
    let _dist = sparse_1.distance(&sparse_2);

    let _dot = dense_1.dot(&sparse_1);
}

#[test]
fn readme_sparse_construction() {
    let _v = SparseVector::try_from(vec![(0usize, 1.0), (2usize, 3.0), (5usize, 4.0)]).unwrap();
    let _v = SparseVector::from_sorted_unchecked(vec![(0usize, 1.0), (2usize, 3.0)]);
    let v = SparseVector::from_unsorted(vec![(5usize, 4.0), (0usize, 1.0), (1usize, 0.0)]);
    assert_eq!(v.iter().collect::<Vec<_>>(), vec![(0, 1.0), (5, 4.0)]);
}

#[test]
fn readme_stack_variant() {
    let _v: StackDenseVector<f64, 4> = StackDenseVector::from([1.0, 2.0, 3.0, 4.0]);
}
