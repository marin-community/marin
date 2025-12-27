pub mod actor_host;
pub mod client;
pub mod executor;

pub use client::CoordinatorClient;

use crate::WorkerId;
use actor_host::{ActorHost, ActorMethodCall, ActorSpec};
use executor::{Task, TaskResult};
use std::sync::{Arc, Mutex};

/// Main worker struct that manages task execution and actor lifecycle.
///
/// The Worker is the primary entry point for executing Python tasks and managing
/// Python actors. It coordinates between the executor (for stateless tasks) and
/// the actor host (for stateful actors).
pub struct Worker {
    /// Unique identifier for this worker instance.
    pub id: WorkerId,

    /// Actor host manages all actors on this worker.
    actors: Arc<Mutex<ActorHost>>,
}

impl Worker {
    /// Create a new worker with the given ID.
    pub fn new(id: WorkerId) -> Self {
        Worker {
            id,
            actors: Arc::new(Mutex::new(ActorHost::new())),
        }
    }

    /// Execute a stateless task.
    ///
    /// This delegates to the executor module which handles Python function execution
    /// via cloudpickle serialization.
    pub fn execute_task(&self, task: Task) -> TaskResult {
        executor::execute_task_impl(task)
    }

    /// Create a new actor on this worker.
    ///
    /// This delegates to the actor host which manages actor lifecycle and state.
    pub fn create_actor(&self, spec: ActorSpec) -> Result<crate::ActorId, String> {
        let actors = self
            .actors
            .lock()
            .map_err(|e| format!("Failed to lock actors: {}", e))?;

        actors.create_actor(spec)
    }

    /// Call a method on an existing actor.
    ///
    /// This delegates to the actor host which handles method dispatch and
    /// state management for the actor.
    pub fn call_actor_method(&self, call: ActorMethodCall) -> TaskResult {
        let actors = match self.actors.lock() {
            Ok(a) => a,
            Err(e) => {
                return TaskResult::Error {
                    traceback: format!("Failed to lock actors: {}", e),
                };
            }
        };

        actors.call_method(call)
    }

    /// Get the worker ID.
    pub fn worker_id(&self) -> WorkerId {
        self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::types::{PyBytes, PyModule};
    use executor::PickledData;

    #[test]
    fn test_worker_new() {
        let worker_id = WorkerId::new();
        let worker = Worker::new(worker_id);
        assert_eq!(worker.worker_id(), worker_id);
    }

    #[test]
    fn test_worker_execute_task() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let cloudpickle = PyModule::import(py, "cloudpickle").unwrap();
            let dumps = cloudpickle.getattr("dumps").unwrap();

            // Create a simple function: lambda x: x * 2
            let code = "lambda x: x * 2";
            let func = py.eval(code, None, None).unwrap();
            let pickled_func = dumps.call1((func,)).unwrap();
            let func_bytes: &PyBytes = pickled_func.downcast().unwrap();

            // Create argument: 21
            let arg = 21i32.to_object(py);
            let pickled_arg = dumps.call1((arg,)).unwrap();
            let arg_bytes: &PyBytes = pickled_arg.downcast().unwrap();

            let task = Task {
                id: crate::TaskId::new(),
                func: PickledData::new(func_bytes.as_bytes().to_vec()),
                args: vec![PickledData::new(arg_bytes.as_bytes().to_vec())],
            };

            let worker = Worker::new(WorkerId::new());
            let result = worker.execute_task(task);

            match result {
                TaskResult::Success { data } => {
                    let loads = cloudpickle.getattr("loads").unwrap();
                    let result_bytes = PyBytes::new(py, &data);
                    let unpickled_result = loads.call1((result_bytes,)).unwrap();
                    let result_value: i32 = unpickled_result.extract().unwrap();
                    assert_eq!(result_value, 42);
                }
                TaskResult::Error { traceback } => {
                    panic!("Task execution failed: {}", traceback);
                }
            }
        });
    }

    #[test]
    fn test_worker_create_and_call_actor() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let cloudpickle = PyModule::import(py, "cloudpickle").unwrap();
            let dumps = cloudpickle.getattr("dumps").unwrap();
            let loads = cloudpickle.getattr("loads").unwrap();

            // Create a simple class
            let code = r#"
class Accumulator:
    def __init__(self, initial=0):
        self.value = initial

    def add(self, x):
        self.value += x
        return self.value

    def get(self):
        return self.value
"#;
            py.run(code, None, None).unwrap();
            let accum_class = py.eval("Accumulator", None, None).unwrap();
            let pickled_class = dumps.call1((accum_class,)).unwrap();
            let class_bytes: &PyBytes = pickled_class.downcast().unwrap();

            // Create init argument: 100
            let arg = 100i32.to_object(py);
            let pickled_arg = dumps.call1((arg,)).unwrap();
            let arg_bytes: &PyBytes = pickled_arg.downcast().unwrap();

            let spec = ActorSpec {
                id: crate::ActorId::new(),
                class_def: PickledData::new(class_bytes.as_bytes().to_vec()),
                init_args: vec![PickledData::new(arg_bytes.as_bytes().to_vec())],
            };

            let worker = Worker::new(WorkerId::new());
            let actor_id = worker.create_actor(spec).unwrap();

            // Call add method with argument 23
            let add_arg = 23i32.to_object(py);
            let pickled_add_arg = dumps.call1((add_arg,)).unwrap();
            let add_arg_bytes: &PyBytes = pickled_add_arg.downcast().unwrap();

            let call = ActorMethodCall {
                actor_id,
                method_name: "add".to_string(),
                args: vec![PickledData::new(add_arg_bytes.as_bytes().to_vec())],
            };

            let result = worker.call_actor_method(call);

            match result {
                TaskResult::Success { data } => {
                    let result_bytes = PyBytes::new(py, &data);
                    let unpickled_result = loads.call1((result_bytes,)).unwrap();
                    let result_value: i32 = unpickled_result.extract().unwrap();
                    assert_eq!(result_value, 123);
                }
                TaskResult::Error { traceback } => {
                    panic!("Method call failed: {}", traceback);
                }
            }

            // Call get method to verify state
            let call = ActorMethodCall {
                actor_id,
                method_name: "get".to_string(),
                args: vec![],
            };

            let result = worker.call_actor_method(call);

            match result {
                TaskResult::Success { data } => {
                    let result_bytes = PyBytes::new(py, &data);
                    let unpickled_result = loads.call1((result_bytes,)).unwrap();
                    let result_value: i32 = unpickled_result.extract().unwrap();
                    assert_eq!(result_value, 123);
                }
                TaskResult::Error { traceback } => {
                    panic!("Method call failed: {}", traceback);
                }
            }
        });
    }

    #[test]
    fn test_worker_multiple_tasks() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let cloudpickle = PyModule::import(py, "cloudpickle").unwrap();
            let dumps = cloudpickle.getattr("dumps").unwrap();
            let loads = cloudpickle.getattr("loads").unwrap();

            let worker = Worker::new(WorkerId::new());

            // Execute multiple tasks
            for i in 1..=5 {
                let code = "lambda x: x + 10";
                let func = py.eval(code, None, None).unwrap();
                let pickled_func = dumps.call1((func,)).unwrap();
                let func_bytes: &PyBytes = pickled_func.downcast().unwrap();

                let arg = i.to_object(py);
                let pickled_arg = dumps.call1((arg,)).unwrap();
                let arg_bytes: &PyBytes = pickled_arg.downcast().unwrap();

                let task = Task {
                    id: crate::TaskId::new(),
                    func: PickledData::new(func_bytes.as_bytes().to_vec()),
                    args: vec![PickledData::new(arg_bytes.as_bytes().to_vec())],
                };

                let result = worker.execute_task(task);

                match result {
                    TaskResult::Success { data } => {
                        let result_bytes = PyBytes::new(py, &data);
                        let unpickled_result = loads.call1((result_bytes,)).unwrap();
                        let result_value: i32 = unpickled_result.extract().unwrap();
                        assert_eq!(result_value, i + 10);
                    }
                    TaskResult::Error { traceback } => {
                        panic!("Task execution failed: {}", traceback);
                    }
                }
            }
        });
    }
}
