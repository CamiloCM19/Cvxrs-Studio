use cvxrs_algos::admm::AdmmSolver;
use cvxrs_core::math::Scalar;
use cvxrs_core::options::SolveOptions;
use cvxrs_core::problem::{Bounds, CscMatrix, ProblemQP};
use cvxrs_core::scaling::RuizScaler;

fn diagonal(n: usize, value: Scalar) -> CscMatrix<Scalar> {
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::with_capacity(n);
    let mut data = Vec::with_capacity(n);
    indptr.push(0);
    for i in 0..n {
        indices.push(i);
        data.push(value);
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

#[test]
fn solves_box_qp() {
    let problem = ProblemQP {
        quadratic: diagonal(2, 4.0),
        linear: vec![-1.0, -1.0],
        inequalities: None,
        equalities: None,
        bounds: Some(Bounds {
            lower: vec![0.0, 0.0],
            upper: vec![1.0, 1.0],
        }),
    };
    let options = SolveOptions::<Scalar>::default();
    let mut solver = AdmmSolver::new(options);
    let mut scaler = RuizScaler::default();
    let solution = solver.solve_qp(problem, &mut scaler).expect("solve");
    assert_eq!(solution.status, cvxrs_core::solution::Status::Optimal);
    for &x in &solution.primal {
        assert!(x >= -1e-6 && x <= 1.0 + 1e-6);
    }
}
