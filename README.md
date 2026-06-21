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

    // Sparse vectors (heap-allocated)
    let sparse_1 = SparseVector::from(vec![(0, 0.2), (2, 0.2), (4, 0.3), (6, 0.4)]);
    let sparse_2 = SparseVector::from(vec![(0, 0.2), (3, 0.4), (5, 0.2), (6, 0.6)]);

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

// Panics on invalid input (unsorted, duplicates, or zeros)
let v = SparseVector::from(vec![(0, 1.0), (2, 3.0), (5, 4.0)]);

// Trusted fast path (caller guarantees sorted, unique, non-zero)
let v = SparseVector::from_sorted_unchecked(vec![(0, 1.0), (2, 3.0)]);

// Automatically sorts, deduplicates, and drops zeros
let v = SparseVector::try_from_unsorted(vec![(5, 4.0), (0, 1.0), (1, 0.0)]);
assert_eq!(v.iter().collect::<Vec<_>>(), vec![(0, 1.0), (5, 4.0)]);
```

## Stack-allocated variants

For fixed-size vectors, use the stack variants:

```rust
use vectors::dense::StackDenseVector;

let v: StackDenseVector<f64, 4> = StackDenseVector::from([1.0, 2.0, 3.0, 4.0]);
```

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std`   | yes     | Enables the standard library. Without this the crate is `no_std`. |
| `libm`  | no      | Uses `libm` for `Real::sqrt` under `no_std`. |

## License

Licensed under the [**MPL-2.0**](https://www.tldrlegal.com/l/mpl-2.0).
