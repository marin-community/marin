use log::{debug, error, info};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule, PyTuple};

/// Pickled Python data for serialization/deserialization.
#[derive(Clone, Debug)]
pub struct PickledData {
    pub data: Vec<u8>,
}

impl PickledData {
    pub fn new(data: Vec<u8>) -> Self {
        PickledData { data }
    }
}

/// Task specification containing function and arguments.
#[derive(Clone, Debug)]
pub struct Task {
    pub id: crate::TaskId,
    pub func: PickledData,
    pub args: Vec<PickledData>,
}

/// Result of task execution.
#[derive(Clone, Debug)]
pub enum TaskResult {
    Success { data: Vec<u8> },
    Error { traceback: String },
}

/// Execute a Python task by unpickling the function and arguments, calling the function,
/// and pickling the result.
///
/// This function:
/// 1. Imports cloudpickle module
/// 2. Unpickles the function from task.func.data
/// 3. Unpickles each argument from task.args
/// 4. Calls the function with the arguments
/// 5. Pickles the result
/// 6. Returns TaskResult::Success with pickled bytes
/// 7. On error: returns TaskResult::Error with traceback string
pub fn execute_task_impl(task: Task) -> TaskResult {
    debug!(target: "fray::rpc::worker::executor",
           "Executing task {}", task.id);

    Python::attach(|py| {
        // Import cloudpickle module
        let cloudpickle = match PyModule::import(py, "cloudpickle") {
            Ok(module) => module,
            Err(err) => {
                let err_msg = format!("Failed to import cloudpickle: {}", err);
                error!(target: "fray::rpc::worker::executor",
                       "Task {}: {}", task.id, err_msg);
                return TaskResult::Error {
                    traceback: err_msg,
                };
            }
        };

        // Get the loads function
        let loads = match cloudpickle.getattr("loads") {
            Ok(func) => func,
            Err(err) => {
                return TaskResult::Error {
                    traceback: format!("Failed to get cloudpickle.loads: {}", err),
                };
            }
        };

        // Get the dumps function
        let dumps = match cloudpickle.getattr("dumps") {
            Ok(func) => func,
            Err(err) => {
                return TaskResult::Error {
                    traceback: format!("Failed to get cloudpickle.dumps: {}", err),
                };
            }
        };

        // Unpickle the function
        let func_bytes = PyBytes::new(py, &task.func.data);
        let func = match loads.call1((func_bytes,)) {
            Ok(f) => f,
            Err(err) => {
                return TaskResult::Error {
                    traceback: format!("Failed to unpickle function: {}", format_py_error(py, err)),
                };
            }
        };

        // Unpickle each argument
        let mut unpickled_args = Vec::new();
        for arg_data in &task.args {
            let arg_bytes = PyBytes::new(py, &arg_data.data);
            match loads.call1((arg_bytes,)) {
                Ok(arg) => unpickled_args.push(arg),
                Err(err) => {
                    return TaskResult::Error {
                        traceback: format!("Failed to unpickle argument: {}", format_py_error(py, err)),
                    };
                }
            }
        }

        // Create a tuple of arguments
        let args_tuple = match PyTuple::new(py, &unpickled_args) {
            Ok(tuple) => tuple,
            Err(err) => {
                return TaskResult::Error {
                    traceback: format!("Failed to create args tuple: {}", format_py_error(py, err)),
                };
            }
        };

        // Call the function with the arguments
        let result = match func.call1(&args_tuple) {
            Ok(res) => res,
            Err(err) => {
                let traceback = format_py_error(py, err);
                error!(target: "fray::rpc::worker::executor",
                       "Task {} function call failed: {}", task.id, traceback);
                return TaskResult::Error {
                    traceback,
                };
            }
        };

        // Pickle the result
        let pickled_result = match dumps.call1((result,)) {
            Ok(pickled) => pickled,
            Err(err) => {
                return TaskResult::Error {
                    traceback: format!("Failed to pickle result: {}", format_py_error(py, err)),
                };
            }
        };

        // Extract bytes from the pickled result
        let result_bytes = match pickled_result.cast::<PyBytes>() {
            Ok(bytes) => bytes,
            Err(err) => {
                return TaskResult::Error {
                    traceback: format!("Failed to convert pickled result to bytes: {}", err),
                };
            }
        };

        let data = result_bytes.as_bytes().to_vec();
        info!(target: "fray::rpc::worker::executor",
              "Task {} completed successfully, result size: {} bytes", task.id, data.len());

        TaskResult::Success { data }
    })
}

/// Format a Python error with traceback information.
fn format_py_error(py: Python, err: PyErr) -> String {
    let traceback_module = match PyModule::import(py, "traceback") {
        Ok(module) => module,
        Err(_) => return err.to_string(),
    };

    let format_exception = match traceback_module.getattr("format_exception") {
        Ok(func) => func,
        Err(_) => return err.to_string(),
    };

    let formatted = match format_exception.call1((err.get_type(py), err.value(py), err.traceback(py))) {
        Ok(result) => result,
        Err(_) => return err.to_string(),
    };

    // Join the formatted exception lines
    match formatted.str() {
        Ok(s) => s.to_string(),
        Err(_) => err.to_string(),
    }
}
