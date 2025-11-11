#![forbid(unsafe_code)]

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use cvxrs_api::{Method, Solver};
use cvxrs_core::math::Scalar;
use cvxrs_core::options::SolveOptions;
use cvxrs_core::solution::Solution;
use cvxrs_io::{read_json_problem, write_solution, JsonProblem};
use serde_json;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "cvxrs")]
#[command(version, about = "Pure Rust convex optimisation solver")]
struct Cli {
    #[arg(long)]
    log_json: bool,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Solve {
        #[arg(long)]
        problem: PathBuf,
        #[arg(long, default_value = "admm")]
        method: MethodArg,
        #[arg(long)]
        tol: Option<f64>,
        #[arg(long)]
        max_iters: Option<usize>,
        #[arg(long)]
        time_limit: Option<u64>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        log_json: bool,
    },
    Check {
        #[arg(long)]
        problem: PathBuf,
    },
    Bench {},
}

#[derive(Clone, Copy, ValueEnum)]
enum MethodArg {
    Admm,
    Ipm,
}

impl From<MethodArg> for Method {
    fn from(arg: MethodArg) -> Method {
        match arg {
            MethodArg::Admm => Method::Admm,
            MethodArg::Ipm => Method::Ipm,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    initialize_tracing(cli.log_json)?;
    match cli.command {
        Commands::Solve {
            problem,
            method,
            tol,
            max_iters,
            time_limit,
            output,
            log_json,
        } => solve_command(
            problem,
            method.into(),
            tol,
            max_iters,
            time_limit,
            output,
            log_json,
        ),
        Commands::Check { problem } => check_command(problem),
        Commands::Bench {} => {
            println!("Benchmarks are available via `cargo bench -p cvxrs-benches`.");
            Ok(())
        }
    }
}

fn initialize_tracing(log_json: bool) -> Result<()> {
    if log_json {
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .json()
            .try_init()
            .ok();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .try_init()
            .ok();
    }
    Ok(())
}

fn solve_command(
    path: PathBuf,
    method: Method,
    tol: Option<f64>,
    max_iters: Option<usize>,
    time_limit: Option<u64>,
    output: Option<PathBuf>,
    output_json: bool,
) -> Result<()> {
    let mut options = SolveOptions::<Scalar>::default();
    if let Some(tolerance) = tol {
        options.tolerance = tolerance as Scalar;
    }
    if let Some(iters) = max_iters {
        options.max_iterations = iters;
    }
    if let Some(limit) = time_limit {
        options.max_time = Some(Duration::from_secs(limit));
    }

    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    let mut solver = Solver::<Scalar>::new().method(method).options(options);
    match extension.as_str() {
        "json" => match read_json_problem::<Scalar, _>(&path)? {
            JsonProblem::Qp { problem } => {
                let solution = solver.solve_qp(problem)?;
                emit_solution(solution, output, output_json)?;
            }
            JsonProblem::Lp { problem } => {
                let solution = solver.solve_lp(problem)?;
                emit_solution(solution, output, output_json)?;
            }
        },
        "mps" => {
            anyhow::bail!("MPS parsing is not implemented yet.");
        }
        _ => {
            anyhow::bail!("Unsupported file extension: {}", extension);
        }
    }
    Ok(())
}

fn emit_solution(
    solution: Solution<Scalar>,
    output: Option<PathBuf>,
    output_json: bool,
) -> Result<()> {
    if output_json {
        let stdout = std::io::stdout();
        let mut handle = stdout.lock();
        serde_json::to_writer_pretty(&mut handle, &solution)?;
        handle.write_all(b"\n")?;
        handle.flush()?;
    } else {
        println!(
            "status: {:?}\nobjective: {:.6}\niters: {}",
            solution.status, solution.objective_value, solution.iterations
        );
    }
    if let Some(path) = output {
        write_solution(path, &solution)?;
    }
    Ok(())
}

fn check_command(path: PathBuf) -> Result<()> {
    match read_json_problem::<Scalar, _>(&path)? {
        JsonProblem::Qp { mut problem } => {
            problem.validate().context("QP validation failed")?;
            println!("QP validation succeeded.");
        }
        JsonProblem::Lp { mut problem } => {
            problem.validate().context("LP validation failed")?;
            println!("LP validation succeeded.");
        }
    }
    Ok(())
}
