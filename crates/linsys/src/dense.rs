use anyhow::{anyhow, Result};
use cvxrs_core::math::RealNumber;
use cvxrs_core::traits::KktSolver;
use num_traits::{FromPrimitive, One};

#[derive(Debug, Clone)]
pub struct DensePattern {
    dimension: usize,
}

impl DensePattern {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

#[derive(Debug, Clone)]
pub struct DenseKktMatrix<T: RealNumber> {
    pub dimension: usize,
    pub data: Vec<T>,
}

impl<T> DenseKktMatrix<T>
where
    T: RealNumber,
{
    pub fn new(dimension: usize, data: Vec<T>) -> Self {
        assert_eq!(dimension * dimension, data.len());
        Self { dimension, data }
    }

    fn entry(&self, row: usize, col: usize) -> T {
        self.data[row * self.dimension + col]
    }
}

pub struct DenseKktSolver<T: RealNumber> {
    dimension: usize,
    l: Vec<T>,
    d: Vec<T>,
    analyzed: bool,
    last_factor: usize,
}

impl<T> DenseKktSolver<T>
where
    T: RealNumber + FromPrimitive + One,
{
    pub fn new() -> Self {
        Self {
            dimension: 0,
            l: Vec::new(),
            d: Vec::new(),
            analyzed: false,
            last_factor: 0,
        }
    }

    fn epsilon() -> T {
        T::from_f64(1e-12).unwrap()
    }

    fn l(&self, row: usize, col: usize) -> T {
        let idx = row * self.dimension + col;
        self.l[idx]
    }

    fn l_mut(&mut self, row: usize, col: usize) -> &mut T {
        let idx = row * self.dimension + col;
        &mut self.l[idx]
    }
}

impl<T> Default for DenseKktSolver<T>
where
    T: RealNumber + FromPrimitive + One,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> KktSolver<T> for DenseKktSolver<T>
where
    T: RealNumber + FromPrimitive + One,
{
    type Pattern = DensePattern;
    type Matrix = DenseKktMatrix<T>;

    fn analyze_pattern(&mut self, pattern: &Self::Pattern) -> Result<()> {
        self.dimension = pattern.dimension();
        self.l = vec![T::zero(); self.dimension * self.dimension];
        self.d = vec![T::zero(); self.dimension];
        for i in 0..self.dimension {
            *self.l_mut(i, i) = T::one();
        }
        self.analyzed = true;
        Ok(())
    }

    fn factor(&mut self, matrix: &Self::Matrix) -> Result<()> {
        if !self.analyzed {
            self.analyze_pattern(&DensePattern::new(matrix.dimension))?;
        }
        if matrix.dimension != self.dimension {
            return Err(anyhow!(
                "matrix dimension {} does not match analysed dimension {}",
                matrix.dimension,
                self.dimension
            ));
        }
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                *self.l_mut(i, j) = if i == j { T::one() } else { T::zero() };
            }
        }

        for j in 0..self.dimension {
            let mut d_j = matrix.entry(j, j);
            for k in 0..j {
                let l_jk = self.l(j, k);
                d_j -= l_jk * l_jk * self.d[k];
            }
            if d_j.abs() <= Self::epsilon() {
                let magnitude = d_j.abs().to_f64().unwrap_or(f64::NAN);
                return Err(anyhow!(
                    "near-singular pivot encountered at column {} (|d_j| = {:.3e})",
                    j,
                    magnitude
                ));
            }
            self.d[j] = d_j;

            for i in (j + 1)..self.dimension {
                let mut lij = matrix.entry(i, j);
                for k in 0..j {
                    lij -= self.l(i, k) * self.l(j, k) * self.d[k];
                }
                lij = lij / self.d[j];
                *self.l_mut(i, j) = lij;
            }
        }
        self.last_factor += 1;
        Ok(())
    }

    fn solve(&self, rhs: &mut [T]) -> Result<()> {
        if rhs.len() != self.dimension {
            return Err(anyhow!(
                "rhs length {} does not match dimension {}",
                rhs.len(),
                self.dimension
            ));
        }
        for i in 0..self.dimension {
            for j in 0..i {
                rhs[i] -= self.l(i, j) * rhs[j];
            }
        }
        for i in 0..self.dimension {
            if self.d[i].abs() <= Self::epsilon() {
                return Err(anyhow!("singular diagonal entry encountered at {}", i));
            }
            rhs[i] = rhs[i] / self.d[i];
        }
        for i in (0..self.dimension).rev() {
            for j in (i + 1)..self.dimension {
                rhs[i] -= self.l(j, i) * rhs[j];
            }
        }
        Ok(())
    }
}
