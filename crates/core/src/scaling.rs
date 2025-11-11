use crate::math::RealNumber;
use crate::problem::{Bounds, CscMatrix, ProblemLP, ProblemQP, ProblemResult};
use crate::stats::SolveStats;
use crate::traits::Scaler;
use num_traits::One;

fn equilibrate_columns<T: RealNumber>(matrix: &CscMatrix<T>, scaling: &mut [T]) {
    for col in 0..matrix.ncols {
        let start = matrix.indptr[col];
        let end = matrix.indptr[col + 1];
        let mut max_val = T::zero();
        for idx in start..end {
            let value = matrix.data[idx].abs();
            if value > max_val {
                max_val = value;
            }
        }
        if max_val > T::zero() {
            let factor = max_val.sqrt();
            if factor > T::zero() {
                scaling[col] = scaling[col] / factor;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuizScaler<T: RealNumber> {
    column_scaling: Vec<T>,
    iterations: usize,
}

impl<T> RuizScaler<T>
where
    T: RealNumber,
{
    pub fn new(iterations: usize) -> Self {
        Self {
            column_scaling: Vec::new(),
            iterations,
        }
    }

    fn apply_column_scaling(&self, matrix: &mut CscMatrix<T>, scaling: &[T]) {
        for col in 0..matrix.ncols {
            let start = matrix.indptr[col];
            let end = matrix.indptr[col + 1];
            let col_scale = scaling[col];
            if col_scale == T::zero() {
                continue;
            }
            let inv_col = T::one() / col_scale;
            for idx in start..end {
                let row = matrix.indices[idx];
                let inv_row = if row < scaling.len() {
                    T::one() / scaling[row]
                } else {
                    T::one()
                };
                matrix.data[idx] = matrix.data[idx] * inv_row * inv_col;
            }
        }
    }

    fn apply_vector_scaling(&self, vector: &mut [T], scaling: &[T]) {
        for (value, &scale) in vector.iter_mut().zip(scaling.iter()) {
            if scale != T::zero() {
                *value = *value / scale;
            }
        }
    }

    fn scale_bounds(&self, bounds: &mut Bounds<T>, scaling: &[T]) {
        for ((lower, upper), &scale) in bounds
            .lower
            .iter_mut()
            .zip(bounds.upper.iter_mut())
            .zip(scaling.iter())
        {
            if scale != T::zero() {
                *lower = *lower * scale;
                *upper = *upper * scale;
            }
        }
    }
}

impl<T> Default for RuizScaler<T>
where
    T: RealNumber,
{
    fn default() -> Self {
        Self::new(5)
    }
}

impl<T> Scaler<T> for RuizScaler<T>
where
    T: RealNumber + One,
{
    fn scale_lp(&mut self, problem: &mut ProblemLP<T>) -> ProblemResult<()> {
        let n = problem.nvars();
        if self.column_scaling.len() != n {
            self.column_scaling = vec![T::one(); n];
        }
        for _ in 0..self.iterations {
            if let Some(ineq) = &problem.inequalities {
                equilibrate_columns(&ineq.matrix, &mut self.column_scaling);
            }
            if let Some(eq) = &problem.equalities {
                equilibrate_columns(&eq.matrix, &mut self.column_scaling);
            }
        }
        if let Some(ineq) = problem.inequalities.as_mut() {
            self.apply_column_scaling(&mut ineq.matrix, &self.column_scaling);
        }
        if let Some(eq) = problem.equalities.as_mut() {
            self.apply_column_scaling(&mut eq.matrix, &self.column_scaling);
        }
        self.apply_vector_scaling(&mut problem.cost, &self.column_scaling);
        if let Some(bounds) = problem.bounds.as_mut() {
            self.scale_bounds(bounds, &self.column_scaling);
        }
        Ok(())
    }

    fn scale_qp(&mut self, problem: &mut ProblemQP<T>) -> ProblemResult<()> {
        let n = problem.nvars();
        if self.column_scaling.len() != n {
            self.column_scaling = vec![T::one(); n];
        }
        for _ in 0..self.iterations {
            equilibrate_columns(&problem.quadratic, &mut self.column_scaling);
            if let Some(ineq) = &problem.inequalities {
                equilibrate_columns(&ineq.matrix, &mut self.column_scaling);
            }
            if let Some(eq) = &problem.equalities {
                equilibrate_columns(&eq.matrix, &mut self.column_scaling);
            }
        }
        self.apply_column_scaling(&mut problem.quadratic, &self.column_scaling);
        self.apply_vector_scaling(&mut problem.linear, &self.column_scaling);
        if let Some(ineq) = problem.inequalities.as_mut() {
            self.apply_column_scaling(&mut ineq.matrix, &self.column_scaling);
        }
        if let Some(eq) = problem.equalities.as_mut() {
            self.apply_column_scaling(&mut eq.matrix, &self.column_scaling);
        }
        if let Some(bounds) = problem.bounds.as_mut() {
            self.scale_bounds(bounds, &self.column_scaling);
        }
        Ok(())
    }

    fn unscale_primal(&self, primal: &mut [T]) {
        if primal.len() == self.column_scaling.len() {
            for (x, &scale) in primal.iter_mut().zip(self.column_scaling.iter()) {
                if scale != T::zero() {
                    *x = *x / scale;
                }
            }
        }
    }

    fn unscale_stats(&self, _stats: &mut SolveStats<T>) {}
}
