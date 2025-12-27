use pyo3::prelude::*;
use crate::types::TaskId;

/// A future representing a pending computation result.
///
/// This wraps a TaskRef from the protocol and provides a Python API
/// for checking status and retrieving results.
#[pyclass(name = "RustyFuture")]
pub struct RustyFuture {
    task_id: TaskId,
}

impl RustyFuture {
    pub fn new_from_task_id(task_id: TaskId) -> Self {
        Self { task_id }
    }

    pub fn get_task_id(&self) -> TaskId {
        self.task_id
    }
}

#[pymethods]
impl RustyFuture {
    #[new]
    fn new(task_id_bytes: Vec<u8>) -> PyResult<Self> {
        // Convert bytes to TaskId
        if task_id_bytes.len() < 16 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "TaskId must be at least 16 bytes",
            ));
        }

        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&task_id_bytes[..16]);
        Ok(RustyFuture {
            task_id: TaskId::from_bytes(bytes),
        })
    }

    fn __repr__(&self) -> String {
        format!("RustyFuture(task_id={})", self.task_id)
    }

    /// Get the task ID as a string.
    #[getter]
    fn task_id_str(&self) -> String {
        self.task_id.to_string()
    }
}
