use crate::math::RealNumber;
use serde::{Deserialize, Serialize};
use sprs::CsMat;
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProblemError {
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("invalid structure: {0}")]
    InvalidStructure(String),
}

pub type ProblemResult<T> = Result<T, ProblemError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CscMatrix<T> {
    pub nrows: usize,
    pub ncols: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<T>,
}

impl<T> CscMatrix<T>
where
    T: RealNumber,
{
    pub fn empty() -> Self {
        Self {
            nrows: 0,
            ncols: 0,
            indptr: vec![0],
            indices: Vec::new(),
            data: Vec::new(),
        }
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn to_csmat(&self) -> ProblemResult<CsMat<T>> {
        if self.indptr.len() != self.ncols + 1 {
            return Err(ProblemError::DimensionMismatch(format!(
                "indptr length {} != ncols + 1 ({})",
                self.indptr.len(),
                self.ncols + 1
            )));
        }
        if self.indices.len() != self.data.len() {
            return Err(ProblemError::DimensionMismatch(format!(
                "indices length {} != data length {}",
                self.indices.len(),
                self.data.len()
            )));
        }
        Ok(CsmatBuilder::build(self))
    }

    pub fn validate(&self) -> ProblemResult<()> {
        if self.indptr.len() != self.ncols + 1 {
            return Err(ProblemError::DimensionMismatch(format!(
                "indptr length {} != ncols + 1 ({})",
                self.indptr.len(),
                self.ncols + 1
            )));
        }
        if self.indices.len() != self.data.len() {
            return Err(ProblemError::DimensionMismatch(format!(
                "indices length {} != data length {}",
                self.indices.len(),
                self.data.len()
            )));
        }
        Ok(())
    }
}

struct CsmatBuilder;

impl CsmatBuilder {
    fn build<T>(matrix: &CscMatrix<T>) -> CsMat<T>
    where
        T: RealNumber,
    {
        CsMat::new_csc(
            (matrix.nrows, matrix.ncols),
            matrix.indptr.clone(),
            matrix.indices.clone(),
            matrix.data.clone(),
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bounds<T> {
    pub lower: Vec<T>,
    pub upper: Vec<T>,
}

impl<T> Bounds<T>
where
    T: RealNumber,
{
    pub fn unbounded(dim: usize) -> Self {
        Self {
            lower: vec![T::neg_infinity(); dim],
            upper: vec![T::infinity(); dim],
        }
    }

    pub fn validate(&self) -> ProblemResult<()> {
        if self.lower.len() != self.upper.len() {
            return Err(ProblemError::DimensionMismatch(format!(
                "lower len {} != upper len {}",
                self.lower.len(),
                self.upper.len()
            )));
        }
        for (i, (lo, hi)) in self.lower.iter().zip(self.upper.iter()).enumerate() {
            if lo > hi {
                return Err(ProblemError::InvalidStructure(format!(
                    "lower bound exceeds upper bound at index {i}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EqualityConstraints<T> {
    pub matrix: CscMatrix<T>,
    pub rhs: Vec<T>,
}

impl<T> EqualityConstraints<T>
where
    T: RealNumber,
{
    fn validate(&self, nvars: usize) -> ProblemResult<()> {
        self.matrix.validate()?;
        if self.matrix.ncols != nvars {
            return Err(ProblemError::DimensionMismatch(format!(
                "constraint matrix columns {} != nvars {}",
                self.matrix.ncols, nvars
            )));
        }
        if self.matrix.nrows != self.rhs.len() {
            return Err(ProblemError::DimensionMismatch(format!(
                "constraint rows {} != rhs len {}",
                self.matrix.nrows,
                self.rhs.len()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InequalityConstraints<T> {
    pub matrix: CscMatrix<T>,
    pub rhs: Vec<T>,
}

impl<T> InequalityConstraints<T>
where
    T: RealNumber,
{
    fn validate(&self, nvars: usize) -> ProblemResult<()> {
        self.matrix.validate()?;
        if self.matrix.ncols != nvars {
            return Err(ProblemError::DimensionMismatch(format!(
                "constraint matrix columns {} != nvars {}",
                self.matrix.ncols, nvars
            )));
        }
        if self.matrix.nrows != self.rhs.len() {
            return Err(ProblemError::DimensionMismatch(format!(
                "constraint rows {} != rhs len {}",
                self.matrix.nrows,
                self.rhs.len()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemLP<T> {
    pub cost: Vec<T>,
    pub inequalities: Option<InequalityConstraints<T>>,
    pub equalities: Option<EqualityConstraints<T>>,
    pub bounds: Option<Bounds<T>>,
}

impl<T> ProblemLP<T>
where
    T: RealNumber,
{
    pub fn nvars(&self) -> usize {
        self.cost.len()
    }

    pub fn validate(&self) -> ProblemResult<()> {
        let n = self.nvars();
        if let Some(bounds) = &self.bounds {
            if bounds.lower.len() != n {
                return Err(ProblemError::DimensionMismatch(format!(
                    "bounds size {} != nvars {n}",
                    bounds.lower.len()
                )));
            }
            bounds.validate()?;
        }
        if let Some(eq) = &self.equalities {
            eq.validate(n)?;
        }
        if let Some(ineq) = &self.inequalities {
            ineq.validate(n)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemQP<T> {
    pub quadratic: CscMatrix<T>,
    pub linear: Vec<T>,
    pub inequalities: Option<InequalityConstraints<T>>,
    pub equalities: Option<EqualityConstraints<T>>,
    pub bounds: Option<Bounds<T>>,
}

impl<T> ProblemQP<T>
where
    T: RealNumber,
{
    pub fn nvars(&self) -> usize {
        self.linear.len()
    }

    pub fn validate(&self) -> ProblemResult<()> {
        let n = self.nvars();
        self.quadratic.validate()?;
        if self.quadratic.ncols != n || self.quadratic.nrows != n {
            return Err(ProblemError::DimensionMismatch(format!(
                "quadratic matrix must be square and match variable dimension {n}"
            )));
        }
        if let Some(bounds) = &self.bounds {
            if bounds.lower.len() != n {
                return Err(ProblemError::DimensionMismatch(format!(
                    "bounds size {} != nvars {n}",
                    bounds.lower.len()
                )));
            }
            bounds.validate()?;
        }
        if let Some(eq) = &self.equalities {
            eq.validate(n)?;
        }
        if let Some(ineq) = &self.inequalities {
            ineq.validate(n)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Cone {
    Zero(usize),
    NonNegative(usize),
    SecondOrder(usize),
}

impl fmt::Display for Cone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cone::Zero(n) => write!(f, "Zero({n})"),
            Cone::NonNegative(n) => write!(f, "NonNegative({n})"),
            Cone::SecondOrder(n) => write!(f, "SecondOrder({n})"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmStart<T> {
    pub primal: Vec<T>,
    pub equality_dual: Vec<T>,
    pub inequality_dual: Vec<T>,
}

impl<T> WarmStart<T>
where
    T: RealNumber,
{
    pub fn empty() -> Self {
        Self {
            primal: Vec::new(),
            equality_dual: Vec::new(),
            inequality_dual: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagonal(n: usize) -> CscMatrix<f64> {
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::with_capacity(n);
        let mut data = Vec::with_capacity(n);
        indptr.push(0);
        for i in 0..n {
            indices.push(i);
            data.push(1.0);
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
    fn qp_validation_passes() {
        let n = 3;
        let qp = ProblemQP {
            quadratic: diagonal(n),
            linear: vec![1.0; n],
            inequalities: None,
            equalities: None,
            bounds: Some(Bounds {
                lower: vec![0.0; n],
                upper: vec![1.0; n],
            }),
        };
        assert!(qp.validate().is_ok());
    }

    #[test]
    fn lp_detects_mismatch() {
        let lp = ProblemLP {
            cost: vec![1.0, 2.0],
            inequalities: None,
            equalities: None,
            bounds: Some(Bounds {
                lower: vec![0.0],
                upper: vec![1.0],
            }),
        };
        assert!(lp.validate().is_err());
    }
}
