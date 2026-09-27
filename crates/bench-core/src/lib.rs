//! Scoring core + golden schema + the engine conformance kit.
//!
//! The ONE home for constants triplicated today across MLXFastConstants, benchmark.yml,
//! and overlay-paired-timing.sh: score = decode^0.75 * prefill^0.25, 0.95 speedup floors,
//! acceptance bands (prefill +/-5%, decode +2%/-5%), sealing. Golden anchor/free-run
//! tolerances define cross-backend consistency. See docs/architecture.md §2 and docs/track-release-branches.md.
//!
//! This crate deliberately does NOT depend on `bench-protocol`; it defines the small
//! local types it needs (see `conformance::EngineHandle`) so the two crates can be
//! built in parallel. Values and rules are ported from the Swift `MLXFastCore`
//! sources; non-obvious ports cite the Swift file + symbol inline.

pub mod cohort_tolerance;
pub mod conformance;
pub mod constants;
pub mod contract;
pub mod free_run;
pub mod golden;
pub mod harness_hash;
pub mod hash;
pub mod near_tie;
pub mod per_stream_attestation;
pub mod prefill_window;
pub mod runner_manifest;
pub mod score;
pub mod stats;
pub mod tape;
pub mod timed_replay;

use std::fmt;

/// Shared crate error. Mirrors Swift `MLXFastError.invalidInput(String)`; the
/// message strings are kept close to the Swift originals so failures read the same.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchError {
    InvalidInput(String),
}

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BenchError::InvalidInput(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for BenchError {}

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, BenchError>;
