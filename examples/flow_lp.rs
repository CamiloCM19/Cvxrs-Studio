
use anyhow::Result;
use cvxrs_api::{LpBuilder, Method, Solver};
use cvxrs_core::math::Scalar;
use cvxrs_core::options::SolveOptions;
use cvxrs_core::problem::{Bounds, CscMatrix};

fn main() -> Result<()> {
    let cost = vec![2.0, 3.0, 1.5, 2.5, 4.0, 3.5, 3.0, 2.0, 1.0];
    let mut builder = LpBuilder::new().c(cost);

    let a = supply_demand_constraints();
    let b = vec![80.0, 65.0, 75.0, 70.0, 60.0, 90.0];
    builder = builder.a(a, b);

    let bounds = Bounds {
        lower: vec![0.0; 9],
        upper: vec![Scalar::infinity(); 9],
    };
    builder = builder.bounds(bounds);

    let problem = builder.build()?;
    let mut solver = Solver::<Scalar>::new()
        .method(Method::Admm)
        .options(SolveOptions::default());
    let solution = solver.solve_lp(problem)?;

    println!("status: {:?}", solution.status);
    println!("flows: {:?}", solution.primal);
    println!("objective: {:.6}", solution.objective_value);
    Ok(())
}

fn supply_demand_constraints() -> CscMatrix<Scalar> {
    let mut indptr = vec![0];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for source in 0..3 {
        for sink in 0..3 {
            indices.push(source);
            data.push(1.0);
        }
        indptr.push(indices.len());
    }
    for sink in 0..3 {
        for source in 0..3 {
            indices.push(3 + sink);
            data.push(1.0);
        }
        indptr.push(indices.len());
    }
    CscMatrix {
        nrows: 6,
        ncols: 9,
        indptr,
        indices,
        data,
    }
}
