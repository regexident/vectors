use vectors::dense::{DenseArrayVec, DenseVec, DenseVecStorage};
use vectors::sparse::{SparseArrayVec, SparseVec, SparseVector};
use vectors::{Distance, Dot, TryFromIterator};

#[test]
fn readme_basic_example() {
    let dense_1 = DenseVec::from(DenseVecStorage::from(vec![0.0, 0.5, 1.0, 2.0, 4.0]));
    let dense_2 = DenseVec::from(DenseVecStorage::from(vec![0.1, 0.2, 0.3, 0.4, 0.0]));
    let _dot = dense_1.dot(&dense_2);
    let _dist = dense_1.distance(&dense_2);

    let sparse_1: SparseVec<usize, f64> = SparseVec::try_from_iter(vec![
        (0usize, 0.2),
        (2usize, 0.2),
        (4usize, 0.3),
        (6usize, 0.4),
    ])
    .unwrap();
    let sparse_2: SparseVec<usize, f64> = SparseVec::try_from_iter(vec![
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
    let _v: SparseVec<usize, f64> =
        SparseVec::try_from_iter(vec![(0usize, 1.0), (2usize, 3.0), (5usize, 4.0)]).unwrap();
    let _v: SparseVec<usize, f64> =
        SparseVec::try_from_iter(vec![(0usize, 1.0), (2usize, 3.0)]).unwrap();
    // try_from_iter validates sorted+unique and drops zero-valued entries.
    let v: SparseVec<usize, f64> =
        SparseVec::try_from_iter(vec![(0usize, 1.0), (1usize, 0.0), (5usize, 4.0)]).unwrap();
    assert_eq!(
        v.iter().collect::<Vec<_>>(),
        vec![(0usize, 1.0), (5usize, 4.0)]
    );
}

#[test]
fn readme_stack_variant() {
    let _v: DenseArrayVec<f64, 4> = DenseArrayVec::try_from_iter([1.0, 2.0, 3.0, 4.0]).unwrap();

    // Stack-allocated sparse vector with capacity 6
    let _v: SparseArrayVec<usize, f64, 6> =
        SparseArrayVec::try_from_iter(vec![(0usize, 1.0), (2usize, 3.0)]).unwrap();
}
