
use anyhow::Result;
use cvxrs_api::{Method, QpBuilder, Solver};
use cvxrs_core::math::Scalar;
use cvxrs_core::options::SolveOptions;
use cvxrs_core::problem::{Bounds, CscMatrix};

fn main() -> Result<()> {
    let returns = vec![0.12, 0.10, 0.07, 0.03];
    let cov_diag = vec![0.05, 0.02, 0.01, 0.005];
    let target_return = 0.08;

    let p = diagonal_csc(&cov_diag);
    let q = vec![0.0; returns.len()];
    let mut builder = QpBuilder::new().p(p).q(q);

    let mut c_indptr = vec![0];
    let mut c_indices = Vec::new();
    let mut c_data = Vec::new();
    let mut rhs = vec![1.0, target_return];
    for col in 0..returns.len() {
        c_indices.push(0);
        c_data.push(1.0);
        c_indices.push(1);
        c_data.push(returns[col]);
        c_indptr.push(c_data.len());
    }
    let equality = cvxrs_core::problem::CscMatrix {
        nrows: 2,
        ncols: returns.len(),
        indptr: c_indptr,
        indices: c_indices,
        data: c_data,
    };
    builder = builder.c(equality, rhs);

    let bounds = Bounds {
        lower: vec![0.0; returns.len()],
        upper: vec![1.0; returns.len()],
    };
    builder = builder.bounds(bounds);

    let problem = builder.build()?;
    let mut solver = Solver::<Scalar>::new()
        .method(Method::Admm)
        .options(SolveOptions::default());
    let solution = solver.solve_qp(problem)?;

    println!("status: {:?}", solution.status);
    println!("weights: {:?}", solution.primal);
    println!("objective: {:.6}", solution.objective_value);
    Ok(())
}

fn diagonal_csc(diag: &[Scalar]) -> CscMatrix<Scalar> {
    let mut indptr = Vec::with_capacity(diag.len() + 1);
    let mut indices = Vec::with_capacity(diag.len());
    let mut data = Vec::with_capacity(diag.len());
    indptr.push(0);
    for (idx, &value) in diag.iter().enumerate() {
        indices.push(idx);
        data.push(value);
        indptr.push(indices.len());
    }
    CscMatrix {
        nrows: diag.len(),
        ncols: diag.len(),
        indptr,
        indices,
        data,
    }
}
