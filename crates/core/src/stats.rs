use crate::math::RealNumber;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationRecord<T: RealNumber> {
    pub iteration: usize,
    pub primal_residual: T,
    pub dual_residual: T,
    pub relative_gap: T,
    pub rho: T,
    pub relaxation: T,
    pub primal_objective: T,
    pub dual_objective: T,
    pub elapsed: Duration,
}

impl<T> IterationRecord<T>
where
    T: RealNumber,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        iteration: usize,
        primal_residual: T,
        dual_residual: T,
        relative_gap: T,
        rho: T,
        relaxation: T,
        primal_objective: T,
        dual_objective: T,
        elapsed: Duration,
    ) -> Self {
        Self {
            iteration,
            primal_residual,
            dual_residual,
            relative_gap,
            rho,
            relaxation,
            primal_objective,
            dual_objective,
            elapsed,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveStats<T: RealNumber> {
    pub history: Vec<IterationRecord<T>>,
    pub solve_time: Duration,
    pub factorizations: usize,
    pub linear_solves: usize,
}

impl<T> SolveStats<T>
where
    T: RealNumber,
{
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            solve_time: Duration::ZERO,
            factorizations: 0,
            linear_solves: 0,
        }
    }

    pub fn push(&mut self, record: IterationRecord<T>) {
        self.history.push(record);
    }
}
