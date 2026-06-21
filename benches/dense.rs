use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use num_traits::MulAdd;
use vectors::Distance;
use vectors::Dot;
use vectors::TryFromIterator;
use vectors::dense::DenseVec;

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

fn make_dense(data: &[f64]) -> DenseVec<f64> {
    DenseVec::try_from_iter(data.iter().copied()).unwrap()
}

fn generate_dense_values(rng: &mut StdRng, dim: usize) -> Vec<f64> {
    (0..dim).map(|_| rng.gen_range(0.0..1.0)).collect()
}

fn add_dense_benches(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    a: &[f64],
    b: &[f64],
) {
    let dv_a = make_dense(a);
    let dv_b = make_dense(b);

    group.bench_function("dot", |bencher| {
        bencher.iter(|| black_box(dv_a.dot(black_box(&dv_b))))
    });

    group.bench_function("squared_distance", |bencher| {
        bencher.iter(|| black_box(dv_a.squared_distance(black_box(&dv_b))))
    });

    group.bench_function("distance", |bencher| {
        bencher.iter(|| black_box(dv_a.distance(black_box(&dv_b))))
    });

    group.bench_function("add", |bencher| {
        bencher.iter(|| black_box(dv_a.clone() + black_box(&dv_b)))
    });

    group.bench_function("sub", |bencher| {
        bencher.iter(|| black_box(dv_a.clone() - black_box(&dv_b)))
    });

    group.bench_function("mul", |bencher| {
        bencher.iter(|| black_box(dv_a.clone() * 2.0))
    });

    group.bench_function("div", |bencher| {
        bencher.iter(|| black_box(dv_a.clone() / 2.0))
    });

    group.bench_function("mul_add", |bencher| {
        bencher.iter(|| black_box(dv_a.clone().mul_add(2.0, &dv_b)))
    });
}

fn bench_dense(c: &mut Criterion) {
    let mut rng = rng();

    for &dim in &[10, 100, 1_000, 10_000, 100_000] {
        let a = generate_dense_values(&mut rng, dim);
        let b = generate_dense_values(&mut rng, dim);
        let mut group = c.benchmark_group(format!("dense/size_{dim}"));
        add_dense_benches(&mut group, &a, &b);
    }
}

criterion_group!(benches, bench_dense);
criterion_main!(benches);
