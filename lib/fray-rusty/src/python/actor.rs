use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::future::RustyFuture;
use crate::types::ActorId;

/// A handle to a remote actor instance.
///
/// Provides access to actor methods via attribute access, which returns
/// RustyActorMethod instances that can be invoked with .remote().
#[pyclass(name = "RustyActorHandle")]
pub struct RustyActorHandle {
    actor_id: ActorId,
    actor_name: String,
}

impl RustyActorHandle {
    pub fn new_from_actor_id(actor_id: ActorId, actor_name: String) -> Self {
        Self {
            actor_id,
            actor_name,
        }
    }

    pub fn get_actor_id(&self) -> ActorId {
        self.actor_id
    }
}

#[pymethods]
impl RustyActorHandle {
    #[new]
    fn new(actor_id_bytes: Vec<u8>, actor_name: String) -> PyResult<Self> {
        if actor_id_bytes.len() < 16 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ActorId must be at least 16 bytes",
            ));
        }

        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&actor_id_bytes[..16]);

        Ok(RustyActorHandle {
            actor_id: ActorId::from_bytes(bytes),
            actor_name,
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
    ///
    /// Note: This is a placeholder. Actual RPC call needs context.
    #[pyo3(signature = (*args, **kwargs))]
    fn remote(
        &self,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<RustyFuture> {
        // TODO: Implement actor method invocation via coordinator RPC
        // For now, return error indicating not implemented
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Actor method invocation not yet implemented in this version. \
             Actors will be fully implemented in the next iteration.",
        ))
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
