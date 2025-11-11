use crate::math::RealNumber;
use crate::options::SolveOptions;
use crate::problem::{ProblemLP, ProblemQP, ProblemResult};
use crate::stats::{IterationRecord, SolveStats};
use anyhow::Result;

pub trait LinearOperator<T: RealNumber>: Send + Sync {
    fn dim(&self) -> (usize, usize);

    fn apply(&self, x: &[T], y: &mut [T]);

    fn apply_transpose(&self, x: &[T], y: &mut [T]) {
        let _ = (x, y);
        panic!("transpose not implemented for this operator");
    }
}

pub trait KktSolver<T: RealNumber>: Send {
    type Pattern;
    type Matrix;

    fn analyze_pattern(&mut self, pattern: &Self::Pattern) -> Result<()>;

    fn factor(&mut self, matrix: &Self::Matrix) -> Result<()>;

    fn solve(&self, rhs: &mut [T]) -> Result<()>;
}

pub trait StoppingCriterion<T: RealNumber> {
    fn is_converged(&self, record: &IterationRecord<T>, options: &SolveOptions<T>) -> bool;
}

pub trait Scaler<T: RealNumber> {
    fn scale_lp(&mut self, problem: &mut ProblemLP<T>) -> ProblemResult<()>;

    fn scale_qp(&mut self, problem: &mut ProblemQP<T>) -> ProblemResult<()>;

    fn unscale_primal(&self, _primal: &mut [T]) {}

    fn unscale_dual(&self, _equality: &mut [T], _inequality: &mut [T]) {}

    fn unscale_stats(&self, stats: &mut SolveStats<T>);
}
