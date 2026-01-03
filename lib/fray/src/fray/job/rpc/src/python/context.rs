use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList, PyModule, PyTuple};
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::runtime::Runtime;

use super::actor::RustyActorHandle;
use super::future::RustyFuture;
use crate::types::{ObjectId, TaskId};
use crate::worker::CoordinatorClient;

/// The main entry point for interacting with the Fray distributed runtime.
///
/// RustyContext manages connections to the coordinator and provides methods
/// for storing objects, running tasks, creating actors, and waiting on futures.
#[pyclass(name = "RustyContext")]
pub struct RustyContext {
    coordinator_addr: String,
    client: Arc<Mutex<CoordinatorClient>>,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl RustyContext {
    #[new]
    fn new(py: Python, coordinator_addr: String) -> PyResult<Self> {
        // Create tokio runtime
        let runtime = Runtime::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create runtime: {}",
                e
            ))
        })?;

        // Parse coordinator address
        let addr: std::net::SocketAddr = coordinator_addr.parse().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid coordinator address: {}",
                e
            ))
        })?;

        // Connect to coordinator (allow Python to handle potential blocking)
        py.detach(|| {
            let client = runtime.block_on(async {
                CoordinatorClient::connect(addr).await
                    .map_err(|e| e.to_string())
            });

            client.map(|c| Self {
                coordinator_addr: coordinator_addr.clone(),
                client: Arc::new(Mutex::new(c)),
                runtime: Arc::new(runtime),
            })
        })
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyConnectionError, _>(format!(
                "Failed to connect to coordinator: {}",
                e
            ))
        })
    }

    fn __repr__(&self) -> String {
        format!("RustyContext(coordinator={})", self.coordinator_addr)
    }

    /// Store an object in the distributed object store.
    ///
    /// Args:
    ///     obj: The Python object to store.
    ///
    /// Returns:
    ///     An ObjectRef that can be used to retrieve the object later.
    fn put(&self, py: Python, obj: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // Import cloudpickle and serialize object
        let cloudpickle = PyModule::import(py, "cloudpickle")?;
        let dumps = cloudpickle.getattr("dumps")?;
        let pickled = dumps.call1((obj,))?;
        let pickled_bytes = pickled.cast::<PyBytes>()?;
        let data = pickled_bytes.as_bytes().to_vec();

        // Send to coordinator
        let client = self.client.clone();
        let object_id = py
            .detach(|| {
                self.runtime
                    .block_on(async move {
                        client.lock().put(data).await
                            .map_err(|e| e.to_string())
                    })
            })
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("put failed: {}", e))
            })?;

        // Return a simple dict-like object ref for now
        let dict = PyDict::new(py);
        dict.set_item("_type", "ObjectRef")?;
        dict.set_item("_id", object_id.to_string())?;
        Ok(dict.into())
    }

    /// Retrieve an object from the distributed object store.
    ///
    /// Args:
    ///     object_ref: The ObjectRef returned by put(), or a RustyFuture.
    ///
    /// Returns:
    ///     The deserialized Python object.
    fn get(&self, py: Python, object_ref: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // Check if it's a RustyFuture
        if let Ok(future) = object_ref.extract::<PyRef<RustyFuture>>() {
            return self.get_future_result(py, &future);
        }

        // Check if it's an ObjectRef (dict with _type == "ObjectRef")
        if let Ok(dict) = object_ref.cast::<PyDict>() {
            if let Ok(Some(ref_type)) = dict.get_item("_type") {
                let type_str: String = ref_type.extract()?;
                if type_str == "ObjectRef" {
                    let id_str: String = dict.get_item("_id")?.ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing _id in ObjectRef")
                    })?.extract()?;

                    let object_id = ObjectId::from_str(&id_str).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Invalid ObjectId: {}",
                            e
                        ))
                    })?;

                    let client = self.client.clone();
                    let data = py
                        .detach(|| {
                            self.runtime
                                .block_on(async move {
                                    client.lock().get(object_id).await
                                        .map_err(|e| e.to_string())
                                })
                        })
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                "get failed: {}",
                                e
                            ))
                        })?;

                    // Unpickle the result
                    let cloudpickle = PyModule::import(py, "cloudpickle")?;
                    let loads = cloudpickle.getattr("loads")?;
                    let bytes = PyBytes::new(py, &data);
                    return Ok(loads.call1((bytes,))?.unbind());
                }
            }
        }

        // If it's not a reference, return as-is
        Ok(object_ref.clone().into())
    }

    /// Get the result of a future (helper method)
    fn get_future_result(&self, py: Python, future: &RustyFuture) -> PyResult<Py<PyAny>> {
        let task_id = future.get_task_id();
        let client = self.client.clone();

        let result_data = py
            .detach(|| {
                self.runtime
                    .block_on(async move {
                        client.lock().get_task_result(task_id).await
                            .map_err(|e| e.to_string())
                    })
            })
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "get_task_result failed: {}",
                    e
                ))
            })?;

        match result_data {
            Some(data) => {
                // Unpickle the result
                let cloudpickle = PyModule::import(py, "cloudpickle")?;
                let loads = cloudpickle.getattr("loads")?;
                let bytes = PyBytes::new(py, &data);
                Ok(loads.call1((bytes,))?.unbind())
            }
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Task result not available",
            )),
        }
    }

    /// Run a function remotely.
    ///
    /// Args:
    ///     func: The function to execute remotely.
    ///     *args: Positional arguments to pass to the function.
    ///
    /// Returns:
    ///     A RustyFuture representing the pending result.
    #[pyo3(signature = (func, *args))]
    fn run(&self, py: Python, func: &Bound<'_, PyAny>, args: &Bound<'_, PyTuple>) -> PyResult<RustyFuture> {
        // Import cloudpickle
        let cloudpickle = PyModule::import(py, "cloudpickle")?;
        let dumps = cloudpickle.getattr("dumps")?;

        // Pickle function and arguments into a single payload
        let payload_list = PyList::empty(py);
        payload_list.append(func)?;
        for arg in args.iter() {
            payload_list.append(arg)?;
        }

        let pickled = dumps.call1((payload_list,))?;
        let pickled_bytes = pickled.cast::<PyBytes>()?;
        let payload = pickled_bytes.as_bytes().to_vec();

        // Submit task
        let task_id = TaskId::new();
        let client = self.client.clone();

        py.detach(|| {
            self.runtime
                .block_on(async move {
                    client.lock().submit_task(task_id, payload).await
                        .map_err(|e| e.to_string())
                })
        })
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "submit_task failed: {}",
                e
            ))
        })?;

        Ok(RustyFuture::new_from_task_id(task_id))
    }

    /// Wait for futures to complete.
    ///
    /// Args:
    ///     futures: A list of RustyFuture instances.
    ///     num_returns: Number of futures to wait for (default: 1).
    ///     timeout: Optional timeout in seconds.
    ///
    /// Returns:
    ///     A tuple of (ready, pending) future lists.
    #[pyo3(signature = (futures, num_returns=1, _timeout=None))]
    fn wait(
        &self,
        py: Python,
        futures: &Bound<'_, PyList>,
        num_returns: usize,
        _timeout: Option<f64>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        // Extract TaskIds from futures
        let mut task_ids = Vec::new();
        for future_obj in futures.iter() {
            let future: PyRef<RustyFuture> = future_obj.extract()?;
            task_ids.push(future.get_task_id());
        }

        // Call coordinator wait
        let client = self.client.clone();
        let task_ids_clone = task_ids.clone();
        let ready_ids = py
            .detach(|| {
                self.runtime.block_on(async move {
                    client.lock().wait_tasks(task_ids_clone, num_returns as u32).await
                        .map_err(|e| e.to_string())
                })
            })
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "wait_tasks failed: {}",
                    e
                ))
            })?;

        // Build ready and pending lists
        let ready_list = PyList::empty(py);
        let pending_list = PyList::empty(py);

        for (i, task_id) in task_ids.iter().enumerate() {
            let future_obj = futures.get_item(i)?;
            if ready_ids.contains(task_id) {
                ready_list.append(future_obj)?;
            } else {
                pending_list.append(future_obj)?;
            }
        }

        Ok((ready_list.into(), pending_list.into()))
    }

    /// Create a new actor instance.
    ///
    /// Args:
    ///     actor_class: The actor class to instantiate.
    ///     *args: Positional arguments for the actor constructor.
    ///     name: Optional actor name.
    ///     get_if_exists: If True, return existing actor with same name.
    ///     lifetime: Actor lifetime ('detached' or 'non_detached').
    ///     **kwargs: Keyword arguments for the actor constructor.
    ///
    /// Returns:
    ///     A RustyActorHandle for interacting with the actor.
    #[pyo3(signature = (actor_class, *args, name=None, get_if_exists=false, _lifetime=None, _preemptible=true, **_kwargs))]
    fn create_actor(
        &self,
        py: Python,
        actor_class: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        name: Option<String>,
        get_if_exists: bool,
        _lifetime: Option<String>,
        _preemptible: bool,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<RustyActorHandle> {
        let actor_name = name.unwrap_or_else(|| format!("actor_{}", uuid::Uuid::new_v4()));

        // Check if actor with this name already exists
        let client = self.client.clone();
        let name_clone = actor_name.clone();
        let existing = py
            .detach(|| {
                self.runtime
                    .block_on(async move {
                        client.lock().get_actor_by_name(&name_clone).await
                            .map_err(|e| e.to_string())
                    })
            })
            .ok()
            .flatten();

        if let Some(actor_id) = existing {
            if get_if_exists {
                // Return existing actor
                return Ok(RustyActorHandle::new_from_actor_id(
                    actor_id,
                    actor_name,
                    self.coordinator_addr.clone(),
                    self.runtime.clone(),
                    self.client.clone(),
                ));
            } else {
                // get_if_exists=False but actor exists - raise error
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Actor with name '{}' already exists", actor_name)
                ));
            }
        }

        // Import cloudpickle
        let cloudpickle = PyModule::import(py, "cloudpickle")?;
        let dumps = cloudpickle.getattr("dumps")?;

        // Pickle the class definition
        let pickled_class = dumps.call1((actor_class,))?;
        let class_bytes = pickled_class.cast::<PyBytes>()?;
        let class_data = class_bytes.as_bytes().to_vec();

        // Pickle the arguments
        let mut arg_bytes = Vec::new();
        for arg in args.iter() {
            let pickled = dumps.call1((arg,))?;
            let bytes = pickled.cast::<PyBytes>()?;
            arg_bytes.push(bytes.as_bytes().to_vec());
        }

        // Create actor on coordinator
        let client = self.client.clone();
        let name_clone = actor_name.clone();
        let actor_id = py
            .detach(|| {
                self.runtime.block_on(async move {
                    client.lock().create_actor(class_data, arg_bytes, name_clone).await
                        .map_err(|e| e.to_string())
                })
            })
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "create_actor failed: {}",
                    e
                ))
            })?;

        Ok(RustyActorHandle::new_from_actor_id(
            actor_id,
            actor_name,
            self.coordinator_addr.clone(),
            self.runtime.clone(),
            self.client.clone(),
        ))
    }

    /// Get the coordinator address.
    #[getter]
    fn coordinator_addr(&self) -> String {
        self.coordinator_addr.clone()
    }
}
