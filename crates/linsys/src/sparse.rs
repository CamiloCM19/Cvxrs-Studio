use crate::dense::{DenseKktMatrix, DenseKktSolver, DensePattern};
use anyhow::Result;
use cvxrs_core::math::RealNumber;
use cvxrs_core::traits::KktSolver;
use num_traits::{FromPrimitive, One};
use sprs::CsMat;

#[derive(Debug, Clone)]
pub struct SparsePattern {
    dimension: usize,
}

impl SparsePattern {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

#[derive(Debug, Clone)]
pub struct SparseKktMatrix<T: RealNumber> {
    pub matrix: CsMat<T>,
}

impl<T> SparseKktMatrix<T>
where
    T: RealNumber,
{
    pub fn new(matrix: CsMat<T>) -> Self {
        Self { matrix }
    }

    fn to_dense(&self) -> DenseKktMatrix<T> {
        let (rows, cols) = self.matrix.shape();
        assert_eq!(rows, cols, "sparse KKT matrices must be square");
        let dimension = rows;
        let mut data = vec![T::zero(); dimension * dimension];
        for (col, column) in self.matrix.outer_iterator().enumerate() {
            for (row, value) in column.iter() {
                data[row * dimension + col] = *value;
                data[col * dimension + row] = *value;
            }
        }
        DenseKktMatrix::new(dimension, data)
    }
}

pub struct SparseKktSolver<T: RealNumber> {
    dense: DenseKktSolver<T>,
    pattern: Option<SparsePattern>,
}

impl<T> SparseKktSolver<T>
where
    T: RealNumber + FromPrimitive + One,
{
    pub fn new() -> Self {
        Self {
            dense: DenseKktSolver::new(),
            pattern: None,
        }
    }
}

impl<T> Default for SparseKktSolver<T>
where
    T: RealNumber + FromPrimitive + One,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> KktSolver<T> for SparseKktSolver<T>
where
    T: RealNumber + FromPrimitive + One,
{
    type Pattern = SparsePattern;
    type Matrix = SparseKktMatrix<T>;

    fn analyze_pattern(&mut self, pattern: &Self::Pattern) -> Result<()> {
        self.pattern = Some(pattern.clone());
        self.dense
            .analyze_pattern(&DensePattern::new(pattern.dimension()))
    }

    fn factor(&mut self, matrix: &Self::Matrix) -> Result<()> {
        if self.pattern.is_none() {
            let (rows, _) = matrix.matrix.shape();
            self.analyze_pattern(&SparsePattern::new(rows))?;
        }
        let dense = matrix.to_dense();
        self.dense.factor(&dense)
    }

    fn solve(&self, rhs: &mut [T]) -> Result<()> {
        self.dense.solve(rhs)
    }
}
