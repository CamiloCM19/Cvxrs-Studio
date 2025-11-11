
use anyhow::Result;
use cvxrs_api::{Method, QpBuilder, Solver};
use cvxrs_core::math::Scalar;
use cvxrs_core::options::SolveOptions;
use cvxrs_core::problem::{Bounds, CscMatrix};

fn main() -> Result<()> {
    let p = diagonal(vec![2.0, 4.0, 6.0]);
    let q = vec![-2.0, -5.0, -3.0];
    let bounds = Bounds {
        lower: vec![0.0, -1.0, 0.0],
        upper: vec![1.0, 2.0, 4.0],
    };
    let problem = QpBuilder::new().p(p).q(q).bounds(bounds).build()?;
    let mut solver = Solver::<Scalar>::new()
        .method(Method::Admm)
        .options(SolveOptions::default());
    let solution = solver.solve_qp(problem)?;

    println!("status: {:?}", solution.status);
    println!("x: {:?}", solution.primal);
    println!("objective: {:.6}", solution.objective_value);
    Ok(())
}

fn diagonal(diag: Vec<Scalar>) -> CscMatrix<Scalar> {
    let n = diag.len();
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::with_capacity(n);
    let mut data = Vec::with_capacity(n);
    indptr.push(0);
    for (idx, value) in diag.into_iter().enumerate() {
        indices.push(idx);
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
