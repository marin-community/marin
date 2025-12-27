use pyo3::prelude::*;

pub mod coordinator;
pub mod types;
pub mod worker;

mod python;

pub use types::{ActorId, ObjectId, Task, TaskId, TaskResult, WorkerId};

use python::{RustyActorHandle, RustyActorMethod, RustyContext, RustyFuture};

/// fray_rusty: High-performance distributed computing runtime for Fray
///
/// This module provides Rust-based implementations of Fray's core primitives:
/// - Distributed object storage
/// - Remote task execution
/// - Actor model support
/// - Future-based result handling
#[pymodule]
fn fray_rusty(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add classes
    m.add_class::<RustyContext>()?;
    m.add_class::<RustyFuture>()?;
    m.add_class::<RustyActorHandle>()?;
    m.add_class::<RustyActorMethod>()?;

    Ok(())
}
