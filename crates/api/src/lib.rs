#![forbid(unsafe_code)]

use anyhow::Result;
use cvxrs_algos::AdmmSolver;
use cvxrs_core::math::RealNumber;
use cvxrs_core::options::SolveOptions;
use cvxrs_core::problem::{
    Bounds, CscMatrix, EqualityConstraints, InequalityConstraints, ProblemLP, ProblemQP,
};
use cvxrs_core::traits::Scaler;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use cvxrs_core::options::Method;
pub use cvxrs_core::solution::{Solution, Status};
pub use cvxrs_core::stats::SolveStats;
pub use cvxrs_core::{problem::WarmStart, scaling::RuizScaler};

#[derive(Debug, Error)]
pub enum SolverError {
    #[error("problem validation failed: {0}")]
    InvalidProblem(String),
    #[error("unsupported method: {0:?}")]
    Unsupported(Method),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QpBuilder<T: RealNumber> {
    p: Option<CscMatrix<T>>,
    q: Option<Vec<T>>,
    equality: Option<EqualityConstraints<T>>,
    inequality: Option<InequalityConstraints<T>>,
    bounds: Option<Bounds<T>>,
}

impl<T> Default for QpBuilder<T>
where
    T: RealNumber,
{
    fn default() -> Self {
        Self {
            p: None,
            q: None,
            equality: None,
            inequality: None,
            bounds: None,
        }
    }
}

impl<T> QpBuilder<T>
where
    T: RealNumber,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn p(mut self, matrix: CscMatrix<T>) -> Self {
        self.p = Some(matrix);
        self
    }

    pub fn q(mut self, vector: Vec<T>) -> Self {
        self.q = Some(vector);
        self
    }

    pub fn c(mut self, matrix: CscMatrix<T>, rhs: Vec<T>) -> Self {
        self.equality = Some(EqualityConstraints { matrix, rhs });
        self
    }

    pub fn a(mut self, matrix: CscMatrix<T>, rhs: Vec<T>) -> Self {
        self.inequality = Some(InequalityConstraints { matrix, rhs });
        self
    }

    pub fn bounds(mut self, bounds: Bounds<T>) -> Self {
        self.bounds = Some(bounds);
        self
    }

    pub fn build(self) -> Result<ProblemQP<T>, SolverError> {
        let quadratic = self
            .p
            .ok_or_else(|| SolverError::InvalidProblem("quadratic matrix missing".into()))?;
        let linear = self
            .q
            .ok_or_else(|| SolverError::InvalidProblem("linear term missing".into()))?;
        let mut problem = ProblemQP {
            quadratic,
            linear,
            inequalities: self.inequality,
            equalities: self.equality,
            bounds: self.bounds,
        };
        problem
            .validate()
            .map_err(|err| SolverError::InvalidProblem(err.to_string()))?;
        Ok(problem)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpBuilder<T: RealNumber> {
    cost: Option<Vec<T>>,
    equality: Option<EqualityConstraints<T>>,
    inequality: Option<InequalityConstraints<T>>,
    bounds: Option<Bounds<T>>,
}

impl<T> Default for LpBuilder<T>
where
    T: RealNumber,
{
    fn default() -> Self {
        Self {
            cost: None,
            equality: None,
            inequality: None,
            bounds: None,
        }
    }
}

impl<T> LpBuilder<T>
where
    T: RealNumber,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn c(mut self, cost: Vec<T>) -> Self {
        self.cost = Some(cost);
        self
    }

    pub fn c_eq(mut self, matrix: CscMatrix<T>, rhs: Vec<T>) -> Self {
        self.equality = Some(EqualityConstraints { matrix, rhs });
        self
    }

    pub fn a(mut self, matrix: CscMatrix<T>, rhs: Vec<T>) -> Self {
        self.inequality = Some(InequalityConstraints { matrix, rhs });
        self
    }

    pub fn bounds(mut self, bounds: Bounds<T>) -> Self {
        self.bounds = Some(bounds);
        self
    }

    pub fn build(self) -> Result<ProblemLP<T>, SolverError> {
        let cost = self
            .cost
            .ok_or_else(|| SolverError::InvalidProblem("objective vector missing".into()))?;
        let mut problem = ProblemLP {
            cost,
            inequalities: self.inequality,
            equalities: self.equality,
            bounds: self.bounds,
        };
        problem
            .validate()
            .map_err(|err| SolverError::InvalidProblem(err.to_string()))?;
        Ok(problem)
    }
}

pub struct Solver<T: RealNumber> {
    method: Method,
    options: SolveOptions<T>,
    scaler: RuizScaler<T>,
    warm_start: Option<WarmStart<T>>,
}

impl<T> Solver<T>
where
    T: RealNumber,
{
    pub fn new() -> Self {
        Self {
            method: Method::Admm,
            options: SolveOptions::default(),
            scaler: RuizScaler::default(),
            warm_start: None,
        }
    }

    pub fn method(mut self, method: Method) -> Self {
        self.method = method;
        self
    }

    pub fn options(mut self, options: SolveOptions<T>) -> Self {
        self.options = options;
        self
    }

    pub fn warm_start(mut self, warm: WarmStart<T>) -> Self {
        self.warm_start = Some(warm);
        self
    }

    pub fn solve_qp(&mut self, problem: ProblemQP<T>) -> Result<Solution<T>, SolverError> {
        match self.method {
            Method::Admm => {
                let options = self.options.clone();
                let mut admm = AdmmSolver::new(options);
                if let Some(warm) = self.warm_start.clone() {
                    admm = admm.with_warm_start(warm);
                }
                admm.solve_qp(problem, &mut self.scaler)
                    .map_err(|err| SolverError::InvalidProblem(err.to_string()))
            }
            Method::Ipm => Err(SolverError::Unsupported(Method::Ipm)),
        }
    }

    pub fn solve_lp(&mut self, problem: ProblemLP<T>) -> Result<Solution<T>, SolverError> {
        match self.method {
            Method::Admm => {
                let options = self.options.clone();
                let mut admm = AdmmSolver::new(options);
                if let Some(warm) = self.warm_start.clone() {
                    admm = admm.with_warm_start(warm);
                }
                admm.solve_lp(problem, &mut self.scaler)
                    .map_err(|err| SolverError::InvalidProblem(err.to_string()))
            }
            Method::Ipm => Err(SolverError::Unsupported(Method::Ipm)),
        }
    }
}

impl<T> Default for Solver<T>
where
    T: RealNumber,
{
    fn default() -> Self {
        Self::new()
    }
}

pub fn solve_qp<T: RealNumber>(
    problem: ProblemQP<T>,
    options: SolveOptions<T>,
) -> Result<Solution<T>, SolverError> {
    Solver::new().options(options).solve_qp(problem)
}

pub fn solve_lp<T: RealNumber>(
    problem: ProblemLP<T>,
    options: SolveOptions<T>,
) -> Result<Solution<T>, SolverError> {
    Solver::new().options(options).solve_lp(problem)
}
