# vectors

[![CI](https://github.com/Regexident/vectors/actions/workflows/ci.yml/badge.svg)](https://github.com/Regexident/vectors/actions/workflows/ci.yml)
[![Version](https://img.shields.io/crates/v/vectors.svg?style=flat-square)](https://crates.io/crates/vectors/)
[![License](https://img.shields.io/crates/l/vectors.svg?style=flat-square)](https://crates.io/crates/vectors/)

Sparse & dense vectors for use in high dimensional vector spaces.

## Usage

```rust
use vectors::dense::DenseVector;
use vectors::sparse::SparseVector;
use vectors::{Distance, Dot};

fn main() {
    // Dense vectors (heap-allocated)
    let dense_1 = DenseVector::from(vec![0.0, 0.5, 1.0, 2.0, 4.0]);
    let dense_2 = DenseVector::from(vec![0.1, 0.2, 0.3, 0.4, 0.0]);

    let dot = dense_1.dot(&dense_2);
    let dist = dense_1.distance(&dense_2);

    println!("Dense dot: {:?}, distance: {:?}", dot, dist);

    // Sparse vectors (heap-allocated, Idx = usize via type inference)
    let sparse_1 = SparseVector::try_from(vec![(0usize, 0.2), (2usize, 0.2), (4usize, 0.3), (6usize, 0.4)]).unwrap();
    let sparse_2 = SparseVector::try_from(vec![(0usize, 0.2), (3usize, 0.4), (5usize, 0.2), (6usize, 0.6)]).unwrap();

    let dot = sparse_1.dot(&sparse_2);
    let dist = sparse_1.distance(&sparse_2);

    println!("Sparse dot: {:?}, distance: {:?}", dot, dist);

    // Cross-type operations
    let dot = dense_1.dot(&sparse_1);
    println!("Cross-type dot: {:?}", dot);
}
```

## Constructing sparse vectors

Sparse vectors enforce sorted, unique, non-zero invariants:

```rust
use vectors::sparse::SparseVector;

// Returns `Err` on invalid input (unsorted, duplicates, or zeros)
let v = SparseVector::try_from(vec![(0usize, 1.0), (2usize, 3.0), (5usize, 4.0)]).unwrap();

// Trusted fast path (caller guarantees sorted, unique, non-zero)
let v = SparseVector::from_sorted_unchecked(vec![(0usize, 1.0), (2usize, 3.0)]);

// Automatically sorts, deduplicates, and drops zeros
let v = SparseVector::from_unsorted(vec![(5usize, 4.0), (0usize, 1.0), (1usize, 0.0)]);
assert_eq!(v.iter().collect::<Vec<_>>(), vec![(0usize, 1.0), (5usize, 4.0)]);
```

## Custom index types

`SparseVector` is generic over the index type `Idx`. Any `Ord + Copy` type works:

```rust
use vectors::sparse::{SparseVector, HeapSparseVector};

// Using u32 indices
let v: SparseVector<u32, f64, Vec<(u32, f64)>> = SparseVector::try_from(vec![(0u32, 1.0), (2u32, 3.0)]).unwrap();

// Using the HeapSparseVector type alias (Idx first)
let v: HeapSparseVector<usize, f64> = SparseVector::from_sorted_unchecked(vec![(0usize, 1.0), (2usize, 3.0)]);
```

## Stack-allocated variants

For fixed-size vectors, use the stack variants:

```rust
use vectors::dense::StackDenseVector;
use vectors::sparse::StackSparseVector;

let v: StackDenseVector<f64, 4> = StackDenseVector::from([1.0, 2.0, 3.0, 4.0]);

// Stack-allocated sparse vector with capacity 6
let v: StackSparseVector<usize, f64, 6> = StackSparseVector::from_iter(vec![(0usize, 1.0), (2usize, 3.0)]);
```

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std`   | yes     | Enables the standard library. Without this the crate is `no_std`. |
| `libm`  | no      | Uses `libm` for `Real::sqrt` under `no_std`. |

## License

Licensed under the [**MPL-2.0**](https://www.tldrlegal.com/l/mpl-2.0).
