use crate::math::RealNumber;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Method {
    Admm,
    Ipm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveOptions<T: RealNumber> {
    pub tolerance: T,
    pub max_iterations: usize,
    pub max_time: Option<Duration>,
    pub admm_rho: T,
    pub admm_relaxation: T,
    pub admm_adaptive_rho: bool,
    pub check_every: usize,
    pub seed: u64,
}

impl<T> SolveOptions<T>
where
    T: RealNumber,
{
    pub fn with_tolerance(tolerance: T) -> Self {
        Self {
            tolerance,
            ..Self::default()
        }
    }
}

impl<T> Default for SolveOptions<T>
where
    T: RealNumber,
{
    fn default() -> Self {
        Self {
            tolerance: T::from(1e-6).unwrap(),
            max_iterations: 10_000,
            max_time: None,
            admm_rho: T::from(1.0).unwrap(),
            admm_relaxation: T::from(1.5).unwrap(),
            admm_adaptive_rho: true,
            check_every: 1,
            seed: 42,
        }
    }
}
