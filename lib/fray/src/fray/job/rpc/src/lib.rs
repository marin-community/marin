use pyo3::prelude::*;
use tokio::runtime::Runtime;

pub mod coordinator;
pub mod types;
pub mod worker;

mod python;

pub use types::{ActorId, ObjectId, Task, TaskId, TaskResult, WorkerId};

use python::{RustyActorHandle, RustyActorMethod, RustyContext, RustyFuture};

/// Start a coordinator server in a background thread.
///
/// Args:
///     addr: Address to bind to (e.g., "127.0.0.1:50051")
///
/// Returns:
///     None. The server runs in a daemon thread.
#[pyfunction]
fn start_coordinator_server(py: Python, addr: String) -> PyResult<()> {
    use coordinator::server::run_coordinator_server;

    // Parse the address
    let socket_addr: std::net::SocketAddr = addr.parse().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid address: {}", e))
    })?;

    // Create runtime for the server
    let runtime = Runtime::new().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e))
    })?;

    // Spawn server in a thread
    std::thread::spawn(move || {
        runtime.block_on(async {
            if let Err(e) = run_coordinator_server(socket_addr).await {
                eprintln!("Coordinator server error: {}", e);
            }
        });
    });

    // Give server time to start
    py.allow_threads(|| std::thread::sleep(std::time::Duration::from_millis(500)));

    Ok(())
}

/// fray_rpc: High-performance distributed computing runtime for Fray
///
/// This module provides Rust-based implementations of Fray's core primitives:
/// - Distributed object storage
/// - Remote task execution
/// - Actor model support
/// - Future-based result handling
#[pymodule]
fn fray_rpc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add classes
    m.add_class::<RustyContext>()?;
    m.add_class::<RustyFuture>()?;
    m.add_class::<RustyActorHandle>()?;
    m.add_class::<RustyActorMethod>()?;

    // Add functions
    m.add_function(wrap_pyfunction!(start_coordinator_server, m)?)?;

    Ok(())
}
