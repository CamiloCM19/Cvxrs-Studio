#![forbid(unsafe_code)]

pub mod dense;
pub mod sparse;

pub use dense::{DenseKktMatrix, DenseKktSolver, DensePattern};
pub use sparse::{SparseKktMatrix, SparseKktSolver, SparsePattern};
