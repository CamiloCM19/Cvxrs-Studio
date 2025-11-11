use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use cvxrs_api::{Method, QpBuilder, Solver};
use cvxrs_core::math::Scalar;
use cvxrs_core::options::SolveOptions;
use cvxrs_core::problem::{Bounds, CscMatrix};
use rand::{rngs::SmallRng, Rng, SeedableRng};

fn random_spd_matrix(n: usize, rng: &mut SmallRng) -> CscMatrix<Scalar> {
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::with_capacity(n * n);
    let mut data = Vec::with_capacity(n * n);
    indptr.push(0);
    for col in 0..n {
        indices.push(col);
        data.push(1.0 + rng.gen::<Scalar>() * 0.1);
        indptr.push(indices.len());
    }
    CscMatrix {
        nrows: n,
        ncols: n,
        indptr,
        indices,
        data,
    }
}

fn random_constraints(m: usize, n: usize, rng: &mut SmallRng) -> CscMatrix<Scalar> {
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);
    for col in 0..n {
        for row in 0..m {
            indices.push(row);
            data.push(rng.gen::<Scalar>() * 0.5 - 0.25);
        }
        indptr.push(indices.len());
    }
    CscMatrix {
        nrows: m,
        ncols: n,
        indptr,
        indices,
        data,
    }
}

fn build_problem(n: usize, m: usize, rng: &mut SmallRng) -> QpBuilder<Scalar> {
    let p = random_spd_matrix(n, rng);
    let q = (0..n)
        .map(|_| rng.gen::<Scalar>() - 0.5)
        .collect::<Vec<_>>();
    let a = random_constraints(m, n, rng);
    let b = (0..m)
        .map(|_| rng.gen::<Scalar>() + 0.5)
        .collect::<Vec<_>>();
    let lower = vec![-1.0; n];
    let upper = vec![1.0; n];
    QpBuilder::new()
        .p(p)
        .q(q)
        .a(a, b)
        .bounds(Bounds { lower, upper })
}

fn solve_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("admm_qp_solve");
    let mut rng = SmallRng::seed_from_u64(42);
    group.bench_function("n=50_m=75", |b| {
        b.iter_batched(
            || build_problem(50, 75, &mut rng).build().unwrap(),
            |problem| {
                let mut solver = Solver::<Scalar>::new()
                    .method(Method::Admm)
                    .options(SolveOptions::default());
                let _ = solver.solve_qp(problem).unwrap();
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, solve_benchmark);
criterion_main!(benches);
