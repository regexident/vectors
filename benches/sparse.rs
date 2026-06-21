use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

use num_traits::MulAdd;
use vectors::Distance;
use vectors::Dot;
use vectors::TryFromIterator;
use vectors::sparse::{
    Join, SparseVec, dot_adaptive, dot_branchless, dot_gallop, dot_merge, inner_join,
    inner_join_adaptive, inner_join_galloping, outer_join,
};

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

pub fn generate_skewed_no_overlap(nnz_small: usize, nnz_large: usize) -> SparsePair {
    let mut rng = rng();
    let range_small = nnz_small * 100;

    let a_indices = generate_sorted_indices(&mut rng, nnz_small, 0, range_small);
    let b_indices = generate_sorted_indices(
        &mut rng,
        nnz_large,
        range_small,
        range_small + nnz_large * 10,
    );

    let a = zip_with_values(&mut rng, a_indices);
    let b = zip_with_values(&mut rng, b_indices);
    (a, b)
}

pub fn generate_extreme_skew(nnz_small: usize, nnz_large: usize) -> SparsePair {
    let mut rng = rng();
    let range_small = nnz_small * 100;

    let a_indices = generate_sorted_indices(&mut rng, nnz_small, 0, range_small);
    let b_indices = generate_sorted_indices(&mut rng, nnz_large, 0, nnz_large * 10);

    let a = zip_with_values(&mut rng, a_indices);
    let b = zip_with_values(&mut rng, b_indices);
    (a, b)
}

fn add_sparse_benches(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    a: &[(usize, f64)],
    b: &[(usize, f64)],
) {
    // Split AoS pairs into SoA slices for join/dot benchmarks
    let a_i: Vec<usize> = a.iter().map(|(i, _)| *i).collect();
    let a_v: Vec<f64> = a.iter().map(|(_, v)| *v).collect();
    let b_i: Vec<usize> = b.iter().map(|(i, _)| *i).collect();
    let b_v: Vec<f64> = b.iter().map(|(_, v)| *v).collect();

    // MARK: Join benchmarks

    group.bench_function("inner_join", |bencher| {
        bencher.iter(|| {
            let result: Vec<(usize, f64)> = inner_join(
                black_box(&a_i),
                black_box(&a_v),
                black_box(&b_i),
                black_box(&b_v),
                |_key: usize, l: &f64, r: &f64| *l + *r,
            )
            .collect();
            black_box(result);
        })
    });

    group.bench_function("outer_join", |bencher| {
        bencher.iter(|| {
            let result: Vec<(usize, f64)> = outer_join(
                black_box(&a_i),
                black_box(&a_v),
                black_box(&b_i),
                black_box(&b_v),
                |_key: usize, join| match join {
                    Join::Left(v) => *v,
                    Join::Right(v) => *v,
                    Join::Both(a, b) => *a + *b,
                },
            )
            .collect();
            black_box(result);
        })
    });

    // MARK: Dot benchmarks (slice-based)

    group.bench_function("dot_merge", |bencher| {
        bencher.iter(|| {
            black_box(dot_merge(
                black_box(&a_i),
                black_box(&a_v),
                black_box(&b_i),
                black_box(&b_v),
            ))
        })
    });

    group.bench_function("dot_branchless", |bencher| {
        bencher.iter(|| {
            black_box(dot_branchless(
                black_box(&a_i),
                black_box(&a_v),
                black_box(&b_i),
                black_box(&b_v),
            ))
        })
    });

    group.bench_function("dot_gallop", |bencher| {
        bencher.iter(|| {
            black_box(dot_gallop(
                black_box(&a_i),
                black_box(&a_v),
                black_box(&b_i),
                black_box(&b_v),
            ))
        })
    });

    group.bench_function("dot_adaptive", |bencher| {
        bencher.iter(|| {
            black_box(dot_adaptive(
                black_box(&a_i),
                black_box(&a_v),
                black_box(&b_i),
                black_box(&b_v),
                vectors::sparse::ADAPTIVE_THRESHOLD,
            ))
        })
    });

    let sv_a = make_sparse(a);
    let sv_b = make_sparse(b);

    group.bench_function("dot", |bencher| {
        bencher.iter(|| black_box(sv_a.dot(black_box(&sv_b))))
    });

    group.bench_function("squared_distance", |bencher| {
        bencher.iter(|| black_box(sv_a.squared_distance(black_box(&sv_b))))
    });

    group.bench_function("distance", |bencher| {
        bencher.iter(|| black_box(sv_a.distance(black_box(&sv_b))))
    });

    group.bench_function("add", |bencher| {
        bencher.iter(|| black_box(sv_a.clone() + sv_b.clone()))
    });

    group.bench_function("sub", |bencher| {
        bencher.iter(|| black_box(sv_a.clone() - sv_b.clone()))
    });

    group.bench_function("mul", |bencher| {
        bencher.iter(|| black_box(sv_a.clone() * 2.0))
    });

    group.bench_function("mul_add", |bencher| {
        bencher.iter(|| black_box(sv_a.clone().mul_add(2.0, &sv_b)))
    });
}

/// Sweep `dot_adaptive` and `inner_join_adaptive` across thresholds.
/// Uncomment the call sites in `bench_sparse` to enable.
#[allow(dead_code)]
fn add_threshold_benches(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    a: &[(usize, f64)],
    b: &[(usize, f64)],
) {
    let a_i: Vec<usize> = a.iter().map(|(i, _)| *i).collect();
    let a_v: Vec<f64> = a.iter().map(|(_, v)| *v).collect();
    let b_i: Vec<usize> = b.iter().map(|(i, _)| *i).collect();
    let b_v: Vec<f64> = b.iter().map(|(_, v)| *v).collect();
    let thresholds = [2, 4, 8, 16, 32, 64, 128];

    for &t in &thresholds {
        group.bench_function(format!("inner_join_adaptive/t={t}"), |bencher| {
            bencher.iter(|| {
                let result: Vec<(usize, f64)> = inner_join_adaptive(
                    black_box(&a_i),
                    black_box(&a_v),
                    black_box(&b_i),
                    black_box(&b_v),
                    |_key: usize, l: &f64, r: &f64| *l + *r,
                );
                black_box(result);
            })
        });
    }

    for &t in &thresholds {
        group.bench_function(format!("dot_adaptive/t={t}"), |bencher| {
            bencher.iter(|| {
                black_box(dot_adaptive(
                    black_box(&a_i),
                    black_box(&a_v),
                    black_box(&b_i),
                    black_box(&b_v),
                    t,
                ))
            })
        });
    }
}

fn bench_sparse(c: &mut Criterion) {
    let (a, b) = generate_balanced_high_overlap(1000);
    add_sparse_benches(
        &mut c.benchmark_group("sparse/balanced_high_overlap"),
        &a,
        &b,
    );

    let (a, b) = generate_balanced_low_overlap(1000);
    add_sparse_benches(
        &mut c.benchmark_group("sparse/balanced_low_overlap"),
        &a,
        &b,
    );

    let (a, b) = generate_skewed_low_overlap(100, 100_000);
    add_sparse_benches(&mut c.benchmark_group("sparse/skewed_low_overlap"), &a, &b);
    // Re-enable to sweep `dot_adaptive` and `inner_join_adaptive` thresholds:
    // add_threshold_benches(
    //     &mut c.benchmark_group("sparse/threshold_skewed_low_overlap"),
    //     &a,
    //     &b,
    // );

    let (a, b) = generate_skewed_no_overlap(100, 100_000);
    add_sparse_benches(&mut c.benchmark_group("sparse/skewed_no_overlap"), &a, &b);
    // add_threshold_benches(
    //     &mut c.benchmark_group("sparse/threshold_skewed_no_overlap"),
    //     &a,
    //     &b,
    // );

    let (a, b) = generate_extreme_skew(10, 1_000_000);
    add_sparse_benches(&mut c.benchmark_group("sparse/extreme_skew"), &a, &b);
    // add_threshold_benches(
    //     &mut c.benchmark_group("sparse/threshold_extreme_skew"),
    //     &a,
    //     &b,
    // );
}

criterion_group!(benches, bench_sparse);
criterion_main!(benches);
