#![forbid(unsafe_code)]

use anyhow::{anyhow, Context, Result};
use cvxrs_core::math::Scalar;
use cvxrs_core::problem::{ProblemLP, ProblemQP};
use cvxrs_core::solution::Solution;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum JsonProblem {
    Qp { problem: ProblemQP<Scalar> },
    Lp { problem: ProblemLP<Scalar> },
}

pub fn read_json_problem<P: AsRef<Path>>(path: P) -> Result<JsonProblem> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("failed to open {:?}", path))?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader
        .read_to_string(&mut contents)
        .with_context(|| format!("failed to read {:?}", path))?;

    match serde_json::from_str::<JsonProblem>(&contents) {
        Ok(problem) => Ok(problem),
        Err(parse_err) => {
            if serde_json::from_str::<Solution<Scalar>>(&contents).is_ok() {
                Err(anyhow!(
                    "JSON file contains a solver solution, but the GUI expects a cvxrs problem (with a 'kind' field)."
                ))
            } else {
                Err(parse_err).context("failed to parse JSON problem")
            }
        }
    }
}

pub fn write_json_problem<P: AsRef<Path>>(path: P, problem: &JsonProblem) -> Result<()> {
    let file = File::create(path.as_ref())
        .with_context(|| format!("failed to create {:?}", path.as_ref()))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, problem).context("failed to serialise problem")?;
    Ok(())
}

pub fn write_solution<P: AsRef<Path>>(path: P, solution: &Solution<Scalar>) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create parent directory {:?}", parent))?;
        }
    }

    let file = File::create(path).with_context(|| format!("failed to create {:?}", path))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, solution).context("failed to serialise solution")?;
    writer
        .flush()
        .with_context(|| format!("failed to write solution into {:?}", path))?;
    Ok(())
}

pub fn read_mps_problem<P: AsRef<Path>>(_path: P) -> Result<()> {
    anyhow::bail!("MPS parsing is not yet implemented.");
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip() {
        let input = r#"{"kind":"lp","problem":{"cost":[1.0,2.0],"inequalities":null,"equalities":null,"bounds":null}}"#;
        let parsed: JsonProblem = serde_json::from_str(input).unwrap();
        let mut buffer = Vec::new();
        serde_json::to_writer(&mut buffer, &parsed).unwrap();
        assert!(!buffer.is_empty());
    }
}
