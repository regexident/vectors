# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please make sure to add your changes to the appropriate categories:

- `Added`: for new functionality
- `Changed`: for changes in existing functionality
- `Deprecated`: for soon-to-be removed functionality
- `Removed`: for removed functionality
- `Fixed`: for fixed bugs
- `Performance`: for performance-relevant changes
- `Security`: for security-relevant changes
- `Other`: for everything else

## [Unreleased]

### Added

- n/a

### Changed

- n/a

### Deprecated

- n/a

### Removed

- n/a

### Fixed

- n/a

### Performance

- n/a

### Security

- n/a

### Other

- n/a

## [0.4.0] - 2026-06-22

### Added

- `InnerJoin` struct and `inner_join` function for galloping-based inner join of two sorted sparse-vector entry slices.
- `dot_gallop` function exposed as public API for direct use of the galloping dot product algorithm.
- `TryFromIterator` and `FromIteratorLossy` traits in crate root for fallible and truncating collection construction.
- `FromCanonicalPairs` trait for constructing storage from already-canonical pairs.
- `sparse::util` module with `canonicalize_entries` helper to sort, deduplicate, and drop zero entries from mutable index/value slices.
- `Storage` trait as foundation for generic vector types.
- `libm` feature for `no_std` support of `Distance::distance` (via `num-traits/libm`).

### Changed

- `SparseVector` now takes `Idx` as its first type parameter: `SparseVector<Idx, T, S>`.
- Cross-type operations (`Dot`, `Distance`) with `DenseVector` require `Idx: Into<usize>`.
- `SparseVecStorage` and `SparseArrayVecStorage` storage layout changed from AoS to SoA.
- Construction now uses `TryFromIterator`, `FromIteratorLossy`, and `FromCanonicalPairs`.
- `Dot<SparseVector<T>>` and `Distance<SparseVector<T>>` now implemented for `DenseVector<T>` (and vice versa).
- All binary ops (`Add`, `Sub`, `MulAdd`) now consistently take `&Self` for the right-hand side.
- Replace naïve linear probing dot product algorithm with a galloping one.
- Updated dependencies:
  - `arrayvec` from `0.4` to `0.7`
- Bumped MSRV to `1.85.1`
- Bumped edition from `2015` to `2024`

### Deprecated

- n/a

### Removed

- `ordered_iter` dependency.
- `.travis.yml`.
- `VectorOps`, `VectorAssignOps`, `VectorRef`, `VectorAssign`, `VectorAssignRef` traits.

### Fixed

- Sparse `squared_distance` now uses outer join: previously used inner join, silently producing wrong results for non-matching indices.
- Sparse `From` validates invariants: previously accepted unsorted, duplicate, or zero-valued entries without error.

### Performance

- n/a

### Security

- n/a

### Other

- n/a

## [0.3.0] - 2018-05-12

### Added

- Stack-allocated variants `dense::stack::DenseVector<A>` and `sparse::stack::SparseVector<A>` backed by `ArrayVec`.
- `Distance` trait with `squared_distance` and `distance` methods.
- `Vector` trait as base trait for all vector types with associated `Scalar` type.
- `VectorOps`, `VectorAssignOps`, `VectorRef`, `VectorAssign`, `VectorAssignRef` traits.
- `Prelude` module re-exporting key traits.
- `no_std` support behind `std` feature flag (enabled by default).
- `nightly` feature flag for optional `missing_mpl` lint plugin.
- `missing_docs` lint.
- Support for `MulAdd`/`MulAddAssign` operations (replacing `AddScaled`/`AddAssignScaled`).

### Changed

- Complete module restructuring into `dense::heap`, `dense::stack`, `sparse::heap`, `sparse::stack` hierarchy.
- `Dot` trait now returns associated `Scalar` type instead of hard-coded `f64`.
- `DenseVector` and `SparseVector` no longer wrap components in `Item<T>` — they store `Vec<T>` and `Vec<(usize, T)>` directly.
- `IntoIterator` for `DenseVector` now yields `(usize, T)` instead of just `T`.
- Iteration now uses `ordered_iter` for sparse vector inner/outer joins.
- Operators now support both owned (`Self`) and reference (`&Self`) right-hand sides.
- Updated dependencies:
  - `ordered_iter` from `0.1` to `0.1`
  - `expectest` from `0.6` to `0.9`
- Added dependencies:
  - `arrayvec` at `0.4`
  - `num-traits` at `0.2`

### Removed

- `Item<T>` wrapper type.
- `AddScaled` and `AddAssignScaled` traits (replaced by `MulAdd`/`MulAddAssign` from `num-traits`).
- `dense_vec!` and `sparse_vec!` macros (migrate to `From`/`FromIterator`).

## [0.1.0] - 2017-02-18

Initial release.
