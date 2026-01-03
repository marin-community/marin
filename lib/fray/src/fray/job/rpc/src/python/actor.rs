use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule, PyTuple};
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::runtime::Runtime;

use super::future::RustyFuture;
use crate::types::ActorId;
use crate::worker::CoordinatorClient;

/// A handle to a remote actor instance.
///
/// Provides access to actor methods via attribute access, which returns
/// RustyActorMethod instances that can be invoked with .remote().
#[pyclass(name = "RustyActorHandle")]
pub struct RustyActorHandle {
    actor_id: ActorId,
    actor_name: String,
    coordinator_addr: String,
    runtime: Arc<Runtime>,
    client: Arc<Mutex<CoordinatorClient>>,
}

impl RustyActorHandle {
    pub fn new_from_actor_id(
        actor_id: ActorId,
        actor_name: String,
        coordinator_addr: String,
        runtime: Arc<Runtime>,
        client: Arc<Mutex<CoordinatorClient>>,
    ) -> Self {
        Self {
            actor_id,
            actor_name,
            coordinator_addr,
            runtime,
            client,
        }
    }

    pub fn get_actor_id(&self) -> ActorId {
        self.actor_id
    }
}

#[pymethods]
impl RustyActorHandle {
    #[new]
    fn new(
        actor_id_bytes: Vec<u8>,
        actor_name: String,
        coordinator_addr: String,
    ) -> PyResult<Self> {
        if actor_id_bytes.len() < 16 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ActorId must be at least 16 bytes",
            ));
        }

        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&actor_id_bytes[..16]);

        // Create runtime and client
        let runtime = Runtime::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create runtime: {}",
                e
            ))
        })?;

        let addr: std::net::SocketAddr = coordinator_addr.parse().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid coordinator address: {}",
                e
            ))
        })?;

        let client = runtime
            .block_on(async { CoordinatorClient::connect(addr).await })
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyConnectionError, _>(format!(
                    "Failed to connect to coordinator: {}",
                    e
                ))
            })?;

        Ok(RustyActorHandle {
            actor_id: ActorId::from_bytes(bytes),
            actor_name,
            coordinator_addr,
            runtime: Arc::new(runtime),
            client: Arc::new(Mutex::new(client)),
        })
    }

    fn __repr__(&self) -> String {
        format!("RustyActorHandle(name={})", self.actor_name)
    }

    /// Get an actor method by name.
    ///
    /// Args:
    ///     name: The method name.
    ///
    /// Returns:
    ///     A RustyActorMethod instance that can be invoked with .remote().
    fn __getattr__(&self, name: String) -> PyResult<RustyActorMethod> {
        Ok(RustyActorMethod {
            actor_id: self.actor_id,
            actor_name: self.actor_name.clone(),
            method_name: name,
            coordinator_addr: self.coordinator_addr.clone(),
            runtime: self.runtime.clone(),
            client: self.client.clone(),
        })
    }

    /// Get the actor ID as a string.
    #[getter]
    fn actor_id_str(&self) -> String {
        self.actor_id.to_string()
    }

    /// Get the actor name.
    #[getter]
    fn actor_name(&self) -> String {
        self.actor_name.clone()
    }
}

/// A reference to a specific method on a remote actor.
///
/// Created by attribute access on RustyActorHandle. Call .remote() to
/// invoke the method asynchronously.
#[pyclass(name = "RustyActorMethod")]
pub struct RustyActorMethod {
    actor_id: ActorId,
    actor_name: String,
    method_name: String,
    coordinator_addr: String,
    runtime: Arc<Runtime>,
    client: Arc<Mutex<CoordinatorClient>>,
}

#[pymethods]
impl RustyActorMethod {
    fn __repr__(&self) -> String {
        format!(
            "RustyActorMethod(actor={}, method={})",
            self.actor_name, self.method_name
        )
    }

    /// Invoke this actor method remotely.
    ///
    /// Args:
    ///     *args: Positional arguments to pass to the method.
    ///     **kwargs: Keyword arguments to pass to the method.
    ///
    /// Returns:
    ///     A RustyFuture representing the pending result.
    #[pyo3(signature = (*args, **_kwargs))]
    fn remote(
        &self,
        py: Python,
        args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<RustyFuture> {
        // Import cloudpickle
        let cloudpickle = PyModule::import(py, "cloudpickle")?;
        let dumps = cloudpickle.getattr("dumps")?;

        // Serialize each argument
        let mut arg_bytes = Vec::new();
        for arg in args.iter() {
            let pickled = dumps.call1((arg,))?;
            let bytes = pickled.downcast::<PyBytes>()?;
            arg_bytes.push(bytes.as_bytes().to_vec());
        }

        // Call coordinator to execute actor method
        let client = self.client.clone();
        let actor_id = self.actor_id;
        let method_name = self.method_name.clone();

        let task_id = py
            .allow_threads(|| {
                self.runtime.block_on(async move {
                    client
                        .lock()
                        .call_actor_method(actor_id, method_name, arg_bytes)
                        .await
                        .map_err(|e| e.to_string())
                })
            })
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "call_actor_method failed: {}",
                    e
                ))
            })?;

        Ok(RustyFuture::new_from_task_id(task_id))
    }

    /// Get the actor ID as a string.
    #[getter]
    fn actor_id_str(&self) -> String {
        self.actor_id.to_string()
    }

    /// Get the method name.
    #[getter]
    fn method_name(&self) -> String {
        self.method_name.clone()
    }
}
