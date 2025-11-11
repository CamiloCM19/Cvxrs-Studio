use anyhow::{bail, Result};
use cvxrs_core::math::RealNumber;
use cvxrs_core::options::SolveOptions;
use cvxrs_core::problem::{ProblemLP, ProblemQP};
use cvxrs_core::solution::Solution;

pub struct IpmSolver;

impl IpmSolver {
    pub fn new() -> Self {
        Self
    }

    pub fn solve_qp<T: RealNumber>(
        &self,
        _problem: &ProblemQP<T>,
        _options: &SolveOptions<T>,
    ) -> Result<Solution<T>> {
        bail!("The IPM backend is not yet implemented. Enable feature `ipm` once available.");
    }

    pub fn solve_lp<T: RealNumber>(
        &self,
        _problem: &ProblemLP<T>,
        _options: &SolveOptions<T>,
    ) -> Result<Solution<T>> {
        bail!("The IPM backend is not yet implemented.");
    }
}
