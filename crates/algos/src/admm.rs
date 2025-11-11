use anyhow::Result;
use cvxrs_core::math::{dot, project_box, relative_gap, residuals_inf, RealNumber, Timer};
use cvxrs_core::options::SolveOptions;
use cvxrs_core::problem::{CscMatrix, ProblemLP, ProblemQP, ProblemResult, WarmStart};
use cvxrs_core::solution::{Solution, Status};
use cvxrs_core::stats::{IterationRecord, SolveStats};
use cvxrs_core::traits::{KktSolver, Scaler};
use cvxrs_linsys::dense::{DenseKktMatrix, DenseKktSolver, DensePattern};
use num_traits::FromPrimitive;

pub type AdmmResult<T> = Solution<T>;

struct AdmmWorkspace<T: RealNumber> {
    n: usize,
    m: usize,
    p_base: Vec<T>,
    ata: Vec<T>,
    a_dense: Vec<T>,
    lower: Vec<T>,
    upper: Vec<T>,
}

impl<T> AdmmWorkspace<T>
where
    T: RealNumber + FromPrimitive,
{
    fn new(problem: &ProblemQP<T>) -> ProblemResult<Self> {
        let n = problem.nvars();
        let mut m = 0;
        let mut has_bounds = false;
        if let Some(eq) = &problem.equalities {
            m += eq.matrix.nrows;
        }
        if let Some(ineq) = &problem.inequalities {
            m += ineq.matrix.nrows;
        }
        if let Some(bounds) = &problem.bounds {
            has_bounds = true;
            m += bounds.lower.len();
        }
        let mut a_dense = vec![T::zero(); m * n];
        let mut lower = vec![T::neg_infinity(); m];
        let mut upper = vec![T::infinity(); m];
        let mut row_offset = 0;
        if let Some(eq) = &problem.equalities {
            scatter_csc(&eq.matrix, n, row_offset, &mut a_dense);
            for (idx, value) in eq.rhs.iter().enumerate() {
                lower[row_offset + idx] = *value;
                upper[row_offset + idx] = *value;
            }
            row_offset += eq.matrix.nrows;
        }
        if let Some(ineq) = &problem.inequalities {
            scatter_csc(&ineq.matrix, n, row_offset, &mut a_dense);
            for (idx, value) in ineq.rhs.iter().enumerate() {
                upper[row_offset + idx] = *value;
            }
            row_offset += ineq.matrix.nrows;
        }
        if has_bounds {
            if let Some(bounds) = &problem.bounds {
                for var in 0..n {
                    let row = row_offset + var;
                    let col = var;
                    a_dense[row * n + col] = T::one();
                    lower[row] = bounds.lower[var];
                    upper[row] = bounds.upper[var];
                }
            }
        }
        let p_base = csc_to_dense(&problem.quadratic);
        let ata = compute_ata(&a_dense, m, n);
        Ok(Self {
            n,
            m,
            p_base,
            ata,
            a_dense,
            lower,
            upper,
        })
    }

    fn multiply_a(&self, x: &[T], out: &mut [T]) {
        assert_eq!(x.len(), self.n);
        assert_eq!(out.len(), self.m);
        for row in 0..self.m {
            let mut acc = T::zero();
            for col in 0..self.n {
                acc += self.a_dense[row * self.n + col] * x[col];
            }
            out[row] = acc;
        }
    }

    fn multiply_at(&self, dual: &[T], out: &mut [T]) {
        assert_eq!(dual.len(), self.m);
        assert_eq!(out.len(), self.n);
        for col in 0..self.n {
            let mut acc = T::zero();
            for row in 0..self.m {
                acc += self.a_dense[row * self.n + col] * dual[row];
            }
            out[col] = acc;
        }
    }
}

struct LinearSystem<T: RealNumber> {
    n: usize,
    base: Vec<T>,
    ata: Vec<T>,
    buffer: Vec<T>,
    solver: DenseKktSolver<T>,
    current_rho: Option<T>,
}

impl<T> LinearSystem<T>
where
    T: RealNumber + FromPrimitive,
{
    fn new(base: Vec<T>, ata: Vec<T>, n: usize) -> Result<Self> {
        let mut solver = DenseKktSolver::new();
        solver.analyze_pattern(&DensePattern::new(n))?;
        Ok(Self {
            n,
            buffer: base.clone(),
            base,
            ata,
            solver,
            current_rho: None,
        })
    }

    fn factor(&mut self, rho: T) -> Result<()> {
        if self
            .current_rho
            .map(|prev| (prev - rho).abs() <= T::from_f64(1e-12).unwrap() * (T::one() + rho.abs()))
            .unwrap_or(false)
        {
            return Ok(());
        }
        self.buffer.clone_from(&self.base);
        for i in 0..self.n * self.n {
            self.buffer[i] = self.buffer[i] + rho * self.ata[i];
        }
        let matrix = DenseKktMatrix::new(self.n, self.buffer.clone());
        self.solver.factor(&matrix)?;
        self.current_rho = Some(rho);
        Ok(())
    }

    fn solve(&self, rhs: &mut [T]) -> Result<()> {
        self.solver.solve(rhs)
    }
}

pub struct AdmmSolver<T: RealNumber> {
    options: SolveOptions<T>,
    warm_start: Option<WarmStart<T>>,
}

impl<T> AdmmSolver<T>
where
    T: RealNumber + FromPrimitive,
{
    pub fn new(options: SolveOptions<T>) -> Self {
        Self {
            options,
            warm_start: None,
        }
    }

    pub fn with_warm_start(mut self, warm: WarmStart<T>) -> Self {
        self.warm_start = Some(warm);
        self
    }

    pub fn solve_qp<S: Scaler<T>>(
        self,
        mut problem: ProblemQP<T>,
        scaler: &mut S,
    ) -> Result<AdmmResult<T>> {
        problem.validate()?;
        scaler.scale_qp(&mut problem)?;
        let workspace = AdmmWorkspace::new(&problem)?;
        let mut lin_sys =
            LinearSystem::new(workspace.p_base.clone(), workspace.ata.clone(), workspace.n)?;
        let mut stats = SolveStats::new();
        let timer = Timer::start();

        let mut x = if let Some(w) = &self.warm_start {
            if w.primal.len() == workspace.n {
                w.primal.clone()
            } else {
                vec![T::zero(); workspace.n]
            }
        } else {
            vec![T::zero(); workspace.n]
        };
        let mut ax = vec![T::zero(); workspace.m];
        workspace.multiply_a(&x, &mut ax);
        let mut z = ax.clone();
        project_box(&mut z, &workspace.lower, &workspace.upper);
        let mut y = vec![T::zero(); workspace.m];
        let mut tmp_dual = vec![T::zero(); workspace.m];
        let mut rhs = vec![T::zero(); workspace.n];
        let mut dual_residual_vec = vec![T::zero(); workspace.n];

        let tol = self.options.tolerance;
        let mut rho = self.options.admm_rho;
        let mut status = Status::MaxIterations;
        let mut last_objective = compute_objective(&problem, &workspace.p_base, &x);
        let mut dual_objective = T::zero();

        for iter in 0..self.options.max_iterations {
            lin_sys.factor(rho)?;
            stats.factorizations += 1;

            for i in 0..workspace.m {
                tmp_dual[i] = z[i] - y[i] / rho;
            }
            workspace.multiply_at(&tmp_dual, &mut rhs);
            for i in 0..workspace.n {
                rhs[i] = rho * rhs[i] - problem.linear[i];
            }
            lin_sys.solve(&mut rhs)?;
            x.copy_from_slice(&rhs);
            stats.linear_solves += 1;

            workspace.multiply_a(&x, &mut ax);
            let z_old = z.clone();
            for i in 0..workspace.m {
                z[i] = ax[i] + y[i] / rho;
            }
            project_box(&mut z, &workspace.lower, &workspace.upper);
            for i in 0..workspace.m {
                y[i] += rho * (ax[i] - z[i]);
            }

            let primal_residual: Vec<T> = ax.iter().zip(z.iter()).map(|(a, b)| *a - *b).collect();
            for i in 0..workspace.m {
                tmp_dual[i] = z_old[i] - z[i];
                tmp_dual[i] *= rho;
            }
            workspace.multiply_at(&tmp_dual, &mut dual_residual_vec);

            let objective = compute_objective(&problem, &workspace.p_base, &x);
            dual_objective = objective - dot(&y, &primal_residual);
            let (pr_norm, du_norm) = residuals_inf(&primal_residual, &dual_residual_vec);
            let gap = relative_gap(objective, dual_objective);
            stats.push(IterationRecord::new(
                iter,
                pr_norm,
                du_norm,
                gap,
                rho,
                self.options.admm_relaxation,
                objective,
                dual_objective,
                timer.elapsed(),
            ));
            last_objective = objective;

            if pr_norm <= tol && du_norm <= tol && gap <= tol {
                status = Status::Optimal;
                break;
            }

            if let Some(limit) = self.options.max_time {
                if timer.elapsed() > limit {
                    status = Status::MaxTime;
                    break;
                }
            }

            if self.options.admm_adaptive_rho {
                let ten = T::from_f64(10.0).unwrap();
                let two = T::from_f64(2.0).unwrap();
                if pr_norm > ten * du_norm {
                    rho *= two;
                } else if du_norm > ten * pr_norm {
                    rho = rho / two;
                }
            }
        }

        stats.solve_time = timer.elapsed();
        let mut solution = Solution {
            primal: x,
            equality_dual: Vec::new(),
            inequality_dual: y,
            status,
            objective_value: last_objective,
            iterations: stats.history.len(),
            stats,
        };
        scaler.unscale_primal(&mut solution.primal);
        scaler.unscale_stats(&mut solution.stats);
        Ok(solution)
    }

    pub fn solve_lp<S: Scaler<T>>(
        self,
        problem: ProblemLP<T>,
        scaler: &mut S,
    ) -> Result<AdmmResult<T>> {
        let n = problem.nvars();
        let mut qp = ProblemQP {
            quadratic: CscMatrix::empty(),
            linear: problem.cost.clone(),
            inequalities: problem.inequalities.clone(),
            equalities: problem.equalities.clone(),
            bounds: problem.bounds.clone(),
        };
        qp.quadratic = identity_csc(n, T::zero());
        self.solve_qp(qp, scaler)
    }
}

fn compute_objective<T: RealNumber + FromPrimitive>(
    problem: &ProblemQP<T>,
    p_dense: &[T],
    x: &[T],
) -> T {
    let mut obj = dot(&problem.linear, x);
    let mut px = vec![T::zero(); problem.nvars()];
    multiply_dense(p_dense, problem.nvars(), problem.nvars(), x, &mut px);
    obj += T::from_f64(0.5).unwrap() * dot(x, &px);
    obj
}

fn scatter_csc<T: RealNumber>(
    matrix: &CscMatrix<T>,
    ncols: usize,
    offset: usize,
    target: &mut [T],
) {
    for col in 0..matrix.ncols {
        let start = matrix.indptr[col];
        let end = matrix.indptr[col + 1];
        for idx in start..end {
            let row = matrix.indices[idx];
            let value = matrix.data[idx];
            let dest_row = offset + row;
            target[dest_row * ncols + col] = value;
        }
    }
}

fn csc_to_dense<T: RealNumber>(matrix: &CscMatrix<T>) -> Vec<T> {
    let mut dense = vec![T::zero(); matrix.nrows * matrix.ncols];
    scatter_csc(matrix, matrix.ncols, 0, &mut dense);
    dense
}

fn compute_ata<T: RealNumber>(a: &[T], m: usize, n: usize) -> Vec<T> {
    let mut ata = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = T::zero();
            for row in 0..m {
                acc += a[row * n + i] * a[row * n + j];
            }
            ata[i * n + j] = acc;
        }
    }
    ata
}

fn multiply_dense<T: RealNumber>(matrix: &[T], rows: usize, cols: usize, x: &[T], out: &mut [T]) {
    for row in 0..rows {
        let mut acc = T::zero();
        for col in 0..cols {
            acc += matrix[row * cols + col] * x[col];
        }
        out[row] = acc;
    }
}

fn identity_csc<T: RealNumber>(n: usize, value: T) -> CscMatrix<T> {
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::with_capacity(n);
    let mut data = Vec::with_capacity(n);
    indptr.push(0);
    for idx in 0..n {
        indices.push(idx);
        data.push(value);
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
