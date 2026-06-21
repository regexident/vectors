use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

use vectors::Distance;
use vectors::Dot;
use vectors::TryFromIterator;
use vectors::dense::DenseVec;
use vectors::sparse::SparseVec;

type SparsePair = (Vec<(usize, f64)>, Vec<(usize, f64)>);

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

fn generate_sorted_indices(
    rng: &mut StdRng,
    nnz: usize,
    range_start: usize,
    range_end: usize,
) -> Vec<usize> {
    let mut set: HashSet<usize> = HashSet::with_capacity(nnz);
    while set.len() < nnz {
        let idx = rng.gen_range(range_start..range_end);
        set.insert(idx);
    }
    let mut result: Vec<usize> = set.into_iter().collect();
    result.sort_unstable();
    result
}

fn zip_with_values(rng: &mut StdRng, indices: Vec<usize>) -> Vec<(usize, f64)> {
    indices
        .into_iter()
        .map(|i| (i, rng.gen_range(0.1..1.0)))
        .collect()
}

fn make_sparse(data: &[(usize, f64)]) -> SparseVec<usize, f64> {
    SparseVec::try_from_iter(data.iter().copied()).unwrap()
}

fn max_index(data: &[(usize, f64)]) -> usize {
    data.iter().map(|(i, _)| *i).max().unwrap_or(0)
}

fn make_dense_from_sparse(data: &[(usize, f64)], dim: usize) -> DenseVec<f64> {
    let mut values = vec![0.0; dim];
    for &(i, v) in data.iter() {
        values[i] = v;
    }
    DenseVec::try_from_iter(values).unwrap()
}

pub fn generate_balanced_high_overlap(nnz: usize) -> SparsePair {
    let mut rng = rng();
    let overlap = (nnz as f64 * 0.80) as usize;
    let non_overlap = nnz - overlap;
    let range_size = nnz * 2;

    let a_indices = generate_sorted_indices(&mut rng, nnz, 0, range_size);
    let mut b_indices = a_indices[..overlap].to_vec();
    let b_extra = generate_sorted_indices(
        &mut rng,
        non_overlap,
        range_size,
        range_size + non_overlap * 2,
    );
    b_indices.extend(b_extra);
    b_indices.sort_unstable();

    let a = zip_with_values(&mut rng, a_indices);
    let b = zip_with_values(&mut rng, b_indices);
    (a, b)
}

pub fn generate_balanced_low_overlap(nnz: usize) -> SparsePair {
    let mut rng = rng();
    let overlap = ((nnz as f64) * 0.01).max(1.0) as usize;
    let non_overlap = nnz - overlap;
    let range_size = nnz * 100;

    let a_indices = generate_sorted_indices(&mut rng, nnz, 0, range_size);
    let mut b_indices = a_indices[..overlap].to_vec();
    let b_extra = generate_sorted_indices(
        &mut rng,
        non_overlap,
        range_size,
        range_size + non_overlap * 2,
    );
    b_indices.extend(b_extra);
    b_indices.sort_unstable();

    let a = zip_with_values(&mut rng, a_indices);
    let b = zip_with_values(&mut rng, b_indices);
    (a, b)
}

pub fn generate_skewed_low_overlap(nnz_small: usize, nnz_large: usize) -> SparsePair {
    let mut rng = rng();
    let overlap = ((nnz_small as f64) * 0.01).max(1.0) as usize;
    let non_overlap_small = nnz_small - overlap;
    let non_overlap_large = nnz_large - overlap;
    let range_small = nnz_small * 100;

    let a_indices = generate_sorted_indices(&mut rng, nnz_small, 0, range_small);
    let overlap_indices = a_indices[..overlap].to_vec();
    let a_rest = generate_sorted_indices(
        &mut rng,
        non_overlap_small,
        range_small,
        range_small + non_overlap_small * 100,
    );

    let mut a_all = overlap_indices.clone();
    a_all.extend(a_rest);
    a_all.sort_unstable();

    let mut b_indices = overlap_indices;
    let b_extra = generate_sorted_indices(
        &mut rng,
        non_overlap_large,
        range_small + non_overlap_small * 100,
        range_small + non_overlap_small * 100 + non_overlap_large * 10,
    );
    b_indices.extend(b_extra);
    b_indices.sort_unstable();

    let a = zip_with_values(&mut rng, a_all);
    let b = zip_with_values(&mut rng, b_indices);
    (a, b)
}

fn add_mixed_benches(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    a: &[(usize, f64)],
    b: &[(usize, f64)],
) {
    let dim = max_index(a).max(max_index(b)) + 1;
    let dense = make_dense_from_sparse(a, dim);
    let sparse = make_sparse(b);

    group.bench_function("dense_dot_sparse", |bencher| {
        bencher.iter(|| black_box(dense.dot(black_box(&sparse))))
    });

    group.bench_function("sparse_dot_dense", |bencher| {
        bencher.iter(|| black_box(sparse.dot(black_box(&dense))))
    });

    group.bench_function("squared_distance", |bencher| {
        bencher.iter(|| black_box(dense.squared_distance(black_box(&sparse))))
    });

    group.bench_function("distance", |bencher| {
        bencher.iter(|| black_box(dense.distance(black_box(&sparse))))
    });
}

fn bench_mixed(c: &mut Criterion) {
    let (a, b) = generate_balanced_high_overlap(1000);
    add_mixed_benches(
        &mut c.benchmark_group("mixed/balanced_high_overlap"),
        &a,
        &b,
    );

    let (a, b) = generate_balanced_low_overlap(1000);
    add_mixed_benches(&mut c.benchmark_group("mixed/balanced_low_overlap"), &a, &b);

    let (a, b) = generate_skewed_low_overlap(100, 1000);
    add_mixed_benches(&mut c.benchmark_group("mixed/skewed_low_overlap"), &a, &b);
}

criterion_group!(benches, bench_mixed);
criterion_main!(benches);
