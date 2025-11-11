use num_traits::{Float as NumFloat, FromPrimitive};
use std::ops::{AddAssign, MulAssign, SubAssign};
use std::time::{Duration, Instant};

pub trait RealNumber:
    NumFloat + FromPrimitive + Send + Sync + AddAssign + SubAssign + MulAssign + 'static
{
}

impl<T> RealNumber for T where
    T: NumFloat + FromPrimitive + Send + Sync + AddAssign + SubAssign + MulAssign + 'static
{
}

#[cfg(not(feature = "f32"))]
pub type Scalar = f64;

#[cfg(feature = "f32")]
pub type Scalar = f32;

pub fn dot<T: RealNumber>(lhs: &[T], rhs: &[T]) -> T {
    assert_eq!(lhs.len(), rhs.len(), "dot product dimension mismatch");
    lhs.iter()
        .zip(rhs.iter())
        .fold(T::zero(), |acc, (a, b)| acc + (*a) * (*b))
}

pub fn norm2<T: RealNumber>(data: &[T]) -> T {
    dot(data, data).sqrt()
}

pub fn norm_inf<T: RealNumber>(data: &[T]) -> T {
    data.iter()
        .copied()
        .map(|v| v.abs())
        .fold(T::zero(), |acc, value| acc.max(value))
}

pub fn axpy<T: RealNumber>(alpha: T, x: &[T], y: &mut [T]) {
    assert_eq!(x.len(), y.len(), "axpy dimension mismatch");
    for (xi, yi) in x.iter().zip(y.iter_mut()) {
        *yi += alpha * (*xi);
    }
}

pub fn project_box<T: RealNumber>(x: &mut [T], lower: &[T], upper: &[T]) {
    assert_eq!(x.len(), lower.len());
    assert_eq!(x.len(), upper.len());
    for ((xi, lo), hi) in x.iter_mut().zip(lower.iter()).zip(upper.iter()) {
        *xi = xi.max(*lo).min(*hi);
    }
}

pub fn residuals_inf<T: RealNumber>(primal: &[T], dual: &[T]) -> (T, T) {
    (norm_inf(primal), norm_inf(dual))
}

pub fn relative_gap<T: RealNumber>(primal_obj: T, dual_obj: T) -> T {
    let gap = (primal_obj - dual_obj).abs();
    let denom = T::from_f64(1.0).unwrap() + primal_obj.abs().max(dual_obj.abs());
    gap / denom
}

#[derive(Debug, Clone)]
pub struct Timer {
    start: Instant,
    elapsed: Duration,
    running: bool,
}

impl Timer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
            elapsed: Duration::ZERO,
            running: true,
        }
    }

    pub fn stop(&mut self) {
        if self.running {
            self.elapsed += self.start.elapsed();
            self.running = false;
        }
    }

    pub fn resume(&mut self) {
        if !self.running {
            self.start = Instant::now();
            self.running = true;
        }
    }

    pub fn elapsed(&self) -> Duration {
        if self.running {
            self.elapsed + self.start.elapsed()
        } else {
            self.elapsed
        }
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::start()
    }
}

#[cfg(test)]
mod tests {
    use super::{dot, norm2, norm_inf, project_box, Scalar};

    #[test]
    fn test_dot_norms() {
        let v = [3.0 as Scalar, 4.0];
        assert!((dot(&v, &v) - 25.0).abs() < 1e-9);
        assert!((norm2(&v) - 5.0).abs() < 1e-9);
        assert!((norm_inf(&v) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_project_box() {
        let mut x = [5.0 as Scalar, -1.0];
        let lower = [0.0, 0.0];
        let upper = [3.0, 2.0];
        project_box(&mut x, &lower, &upper);
        assert!((x[0] - 3.0).abs() < 1e-9);
        assert!((x[1] - 0.0).abs() < 1e-9);
    }
}
