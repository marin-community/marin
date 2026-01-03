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
    Python::with_gil(|py| {
        // Import cloudpickle module
        let cloudpickle = match PyModule::import(py, "cloudpickle") {
            Ok(module) => module,
            Err(err) => {
                return TaskResult::Error {
                    traceback: format!("Failed to import cloudpickle: {}", err),
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
                return TaskResult::Error {
                    traceback: format_py_error(py, err),
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
        let result_bytes = match pickled_result.downcast::<PyBytes>() {
            Ok(bytes) => bytes,
            Err(err) => {
                return TaskResult::Error {
                    traceback: format!("Failed to convert pickled result to bytes: {}", err),
                };
            }
        };

        TaskResult::Success {
            data: result_bytes.as_bytes().to_vec(),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execute_simple_task() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let cloudpickle = PyModule::import(py, "cloudpickle").unwrap();
            let dumps = cloudpickle.getattr("dumps").unwrap();

            // Create a simple function: lambda x: x + 1
            let code = "lambda x: x + 1";
            let func = py.eval(code, None, None).unwrap();
            let pickled_func = dumps.call1((func,)).unwrap();
            let func_bytes: &PyBytes = pickled_func.downcast().unwrap();

            // Create argument: 42
            let arg = 42i32.to_object(py);
            let pickled_arg = dumps.call1((arg,)).unwrap();
            let arg_bytes: &PyBytes = pickled_arg.downcast().unwrap();

            let task = Task {
                id: crate::TaskId::new(),
                func: PickledData::new(func_bytes.as_bytes().to_vec()),
                args: vec![PickledData::new(arg_bytes.as_bytes().to_vec())],
            };

            let result = execute_task_impl(task);

            match result {
                TaskResult::Success { data } => {
                    // Unpickle and check the result
                    let loads = cloudpickle.getattr("loads").unwrap();
                    let result_bytes = PyBytes::new(py, &data);
                    let unpickled_result = loads.call1((result_bytes,)).unwrap();
                    let result_value: i32 = unpickled_result.extract().unwrap();
                    assert_eq!(result_value, 43);
                }
                TaskResult::Error { traceback } => {
                    panic!("Task execution failed: {}", traceback);
                }
            }
        });
    }

    #[test]
    fn test_execute_task_with_error() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let cloudpickle = PyModule::import(py, "cloudpickle").unwrap();
            let dumps = cloudpickle.getattr("dumps").unwrap();

            // Create a function that raises an error
            let code = "lambda: 1 / 0";
            let func = py.eval(code, None, None).unwrap();
            let pickled_func = dumps.call1((func,)).unwrap();
            let func_bytes: &PyBytes = pickled_func.downcast().unwrap();

            let task = Task {
                id: crate::TaskId::new(),
                func: PickledData::new(func_bytes.as_bytes().to_vec()),
                args: vec![],
            };

            let result = execute_task_impl(task);

            match result {
                TaskResult::Success { .. } => {
                    panic!("Expected error, got success");
                }
                TaskResult::Error { traceback } => {
                    assert!(traceback.contains("ZeroDivisionError") || traceback.contains("division"));
                }
            }
        });
    }
}
