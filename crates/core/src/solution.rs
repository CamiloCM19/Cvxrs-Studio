use crate::math::RealNumber;
use crate::stats::SolveStats;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Status {
    Optimal,
    PrimalInfeasible,
    DualInfeasible,
    MaxIterations,
    MaxTime,
    NumericalFailure,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution<T: RealNumber> {
    pub primal: Vec<T>,
    pub equality_dual: Vec<T>,
    pub inequality_dual: Vec<T>,
    pub status: Status,
    pub objective_value: T,
    pub iterations: usize,
    pub stats: SolveStats<T>,
}

impl<T> Solution<T>
where
    T: RealNumber,
{
    pub fn with_capacity(n: usize, meq: usize, mineq: usize) -> Self {
        Self {
            primal: vec![T::zero(); n],
            equality_dual: vec![T::zero(); meq],
            inequality_dual: vec![T::zero(); mineq],
            status: Status::NumericalFailure,
            objective_value: T::zero(),
            iterations: 0,
            stats: SolveStats::new(),
        }
    }
}
