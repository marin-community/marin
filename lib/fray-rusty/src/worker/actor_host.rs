use crate::ActorId;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule, PyTuple};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::executor::{PickledData, TaskResult};

/// Actor specification containing class and initialization arguments.
#[derive(Clone, Debug)]
pub struct ActorSpec {
    pub id: ActorId,
    pub class_def: PickledData,
    pub init_args: Vec<PickledData>,
}

/// Actor method call specification.
#[derive(Clone, Debug)]
pub struct ActorMethodCall {
    pub actor_id: ActorId,
    pub method_name: String,
    pub args: Vec<PickledData>,
}

/// A Python actor instance stored in pickled form.
/// The lock ensures only one method call executes at a time on the actor.
pub struct PyActorInstance {
    pub id: ActorId,
    pub pickled_instance: Vec<u8>,
    pub lock: Mutex<()>,
}

/// Manages actor instances and provides methods for creating actors and calling their methods.
pub struct ActorHost {
    actors: Mutex<HashMap<ActorId, Arc<PyActorInstance>>>,
}

impl ActorHost {
    pub fn new() -> Self {
        ActorHost {
            actors: Mutex::new(HashMap::new()),
        }
    }

    /// Create a new actor by unpickling the class definition, instantiating it,
    /// and storing the pickled instance.
    ///
    /// Steps:
    /// 1. Import cloudpickle
    /// 2. Unpickle the class definition
    /// 3. Unpickle the initialization arguments
    /// 4. Instantiate the class with the arguments
    /// 5. Pickle the instance
    /// 6. Store in the actors map
    pub fn create_actor(&self, spec: ActorSpec) -> Result<ActorId, String> {
        Python::with_gil(|py| {
            // Import cloudpickle module
            let cloudpickle = PyModule::import(py, "cloudpickle")
                .map_err(|e| format!("Failed to import cloudpickle: {}", e))?;

            let loads = cloudpickle
                .getattr("loads")
                .map_err(|e| format!("Failed to get cloudpickle.loads: {}", e))?;

            let dumps = cloudpickle
                .getattr("dumps")
                .map_err(|e| format!("Failed to get cloudpickle.dumps: {}", e))?;

            // Unpickle the class definition
            let class_bytes = PyBytes::new(py, &spec.class_def.data);
            let class = loads
                .call1((class_bytes,))
                .map_err(|e| format!("Failed to unpickle class: {}", e))?;

            // Unpickle each initialization argument
            let mut unpickled_args = Vec::new();
            for arg_data in &spec.init_args {
                let arg_bytes = PyBytes::new(py, &arg_data.data);
                let arg = loads
                    .call1((arg_bytes,))
                    .map_err(|e| format!("Failed to unpickle init argument: {}", e))?;
                unpickled_args.push(arg);
            }

            // Create a tuple of arguments
            let args_tuple = PyTuple::new(py, &unpickled_args);

            // Instantiate the class
            let instance = class
                .call1(args_tuple)
                .map_err(|e| format!("Failed to instantiate class: {}", e))?;

            // Pickle the instance
            let pickled_instance = dumps
                .call1((instance,))
                .map_err(|e| format!("Failed to pickle instance: {}", e))?;

            let instance_bytes: &PyBytes = pickled_instance
                .downcast()
                .map_err(|e| format!("Failed to convert pickled instance to bytes: {}", e))?;

            // Create the actor instance
            let actor = Arc::new(PyActorInstance {
                id: spec.id,
                pickled_instance: instance_bytes.as_bytes().to_vec(),
                lock: Mutex::new(()),
            });

            // Store in the actors map
            let mut actors = self
                .actors
                .lock()
                .map_err(|e| format!("Failed to lock actors map: {}", e))?;

            actors.insert(spec.id, actor);

            Ok(spec.id)
        })
    }

    /// Call a method on an actor.
    ///
    /// Steps:
    /// 1. Acquire the actor lock
    /// 2. Import cloudpickle
    /// 3. Unpickle the actor instance
    /// 4. Unpickle the method arguments
    /// 5. Call the method
    /// 6. Pickle the updated instance (in case it was mutated)
    /// 7. Pickle the result
    /// 8. Update the stored pickled instance
    /// 9. Release the lock
    pub fn call_method(&self, call: ActorMethodCall) -> TaskResult {
        // Get the actor
        let actor = {
            let actors = match self.actors.lock() {
                Ok(a) => a,
                Err(e) => {
                    return TaskResult::Error {
                        traceback: format!("Failed to lock actors map: {}", e),
                    };
                }
            };

            match actors.get(&call.actor_id) {
                Some(actor) => Arc::clone(actor),
                None => {
                    return TaskResult::Error {
                        traceback: format!("Actor not found: {}", call.actor_id),
                    };
                }
            }
        };

        // Acquire the actor lock to ensure single-threaded access
        let _lock = match actor.lock.lock() {
            Ok(l) => l,
            Err(e) => {
                return TaskResult::Error {
                    traceback: format!("Failed to lock actor: {}", e),
                };
            }
        };

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

            let loads = match cloudpickle.getattr("loads") {
                Ok(func) => func,
                Err(err) => {
                    return TaskResult::Error {
                        traceback: format!("Failed to get cloudpickle.loads: {}", err),
                    };
                }
            };

            let dumps = match cloudpickle.getattr("dumps") {
                Ok(func) => func,
                Err(err) => {
                    return TaskResult::Error {
                        traceback: format!("Failed to get cloudpickle.dumps: {}", err),
                    };
                }
            };

            // Unpickle the instance
            let instance_bytes = PyBytes::new(py, &actor.pickled_instance);
            let instance = match loads.call1((instance_bytes,)) {
                Ok(inst) => inst,
                Err(err) => {
                    return TaskResult::Error {
                        traceback: format!("Failed to unpickle actor instance: {}", err),
                    };
                }
            };

            // Unpickle each argument
            let mut unpickled_args = Vec::new();
            for arg_data in &call.args {
                let arg_bytes = PyBytes::new(py, &arg_data.data);
                match loads.call1((arg_bytes,)) {
                    Ok(arg) => unpickled_args.push(arg),
                    Err(err) => {
                        return TaskResult::Error {
                            traceback: format!("Failed to unpickle argument: {}", err),
                        };
                    }
                }
            }

            // Get the method
            let method = match instance.getattr(call.method_name.as_str()) {
                Ok(m) => m,
                Err(err) => {
                    return TaskResult::Error {
                        traceback: format!(
                            "Failed to get method '{}': {}",
                            call.method_name, err
                        ),
                    };
                }
            };

            // Create a tuple of arguments
            let args_tuple = PyTuple::new(py, &unpickled_args);

            // Call the method
            let result = match method.call1(args_tuple) {
                Ok(res) => res,
                Err(err) => {
                    return TaskResult::Error {
                        traceback: format!("Method call failed: {}", err),
                    };
                }
            };

            // Pickle the result
            let pickled_result = match dumps.call1((result,)) {
                Ok(pickled) => pickled,
                Err(err) => {
                    return TaskResult::Error {
                        traceback: format!("Failed to pickle result: {}", err),
                    };
                }
            };

            let result_bytes: &PyBytes = match pickled_result.downcast() {
                Ok(bytes) => bytes,
                Err(err) => {
                    return TaskResult::Error {
                        traceback: format!("Failed to convert pickled result to bytes: {}", err),
                    };
                }
            };

            // Pickle the updated instance (it may have been mutated)
            let pickled_instance = match dumps.call1((instance,)) {
                Ok(pickled) => pickled,
                Err(err) => {
                    return TaskResult::Error {
                        traceback: format!("Failed to pickle updated instance: {}", err),
                    };
                }
            };

            let instance_bytes: &PyBytes = match pickled_instance.downcast() {
                Ok(bytes) => bytes,
                Err(err) => {
                    return TaskResult::Error {
                        traceback: format!("Failed to convert pickled instance to bytes: {}", err),
                    };
                }
            };

            // Update the stored pickled instance
            // We need to do this carefully because we're inside the actor lock
            // but we can't modify the Arc directly
            // Instead, we'll need to create a new PyActorInstance
            let new_pickled_instance = instance_bytes.as_bytes().to_vec();

            // Drop the GIL lock before acquiring the actors map lock to avoid deadlocks
            drop(loads);
            drop(dumps);
            drop(cloudpickle);

            // Update the stored instance
            // We need to be careful here - we have the actor lock but need to update the map
            // This is safe because we're only updating the pickled_instance field
            // The actor itself is behind an Arc and the lock prevents concurrent access
            let new_actor = Arc::new(PyActorInstance {
                id: actor.id,
                pickled_instance: new_pickled_instance,
                lock: Mutex::new(()),
            });

            let mut actors = match self.actors.lock() {
                Ok(a) => a,
                Err(e) => {
                    return TaskResult::Error {
                        traceback: format!("Failed to lock actors map for update: {}", e),
                    };
                }
            };

            actors.insert(call.actor_id, new_actor);

            TaskResult::Success {
                data: result_bytes.as_bytes().to_vec(),
            }
        })
    }

    /// Get an actor by ID.
    pub fn get_actor(&self, id: &ActorId) -> Option<Arc<PyActorInstance>> {
        let actors = self.actors.lock().ok()?;
        actors.get(id).map(Arc::clone)
    }
}

impl Default for ActorHost {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_actor() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let cloudpickle = PyModule::import(py, "cloudpickle").unwrap();
            let dumps = cloudpickle.getattr("dumps").unwrap();

            // Create a simple class
            let code = r#"
class Counter:
    def __init__(self, start=0):
        self.count = start

    def increment(self):
        self.count += 1
        return self.count

    def get(self):
        return self.count
"#;
            py.run(code, None, None).unwrap();
            let counter_class = py.eval("Counter", None, None).unwrap();
            let pickled_class = dumps.call1((counter_class,)).unwrap();
            let class_bytes: &PyBytes = pickled_class.downcast().unwrap();

            // Create init argument: 10
            let arg = 10i32.to_object(py);
            let pickled_arg = dumps.call1((arg,)).unwrap();
            let arg_bytes: &PyBytes = pickled_arg.downcast().unwrap();

            let spec = ActorSpec {
                id: ActorId::new(),
                class_def: PickledData::new(class_bytes.as_bytes().to_vec()),
                init_args: vec![PickledData::new(arg_bytes.as_bytes().to_vec())],
            };

            let host = ActorHost::new();
            let actor_id = host.create_actor(spec).unwrap();

            // Verify the actor exists
            assert!(host.get_actor(&actor_id).is_some());
        });
    }

    #[test]
    fn test_call_actor_method() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let cloudpickle = PyModule::import(py, "cloudpickle").unwrap();
            let dumps = cloudpickle.getattr("dumps").unwrap();
            let loads = cloudpickle.getattr("loads").unwrap();

            // Create a simple class
            let code = r#"
class Counter:
    def __init__(self, start=0):
        self.count = start

    def increment(self):
        self.count += 1
        return self.count

    def get(self):
        return self.count
"#;
            py.run(code, None, None).unwrap();
            let counter_class = py.eval("Counter", None, None).unwrap();
            let pickled_class = dumps.call1((counter_class,)).unwrap();
            let class_bytes: &PyBytes = pickled_class.downcast().unwrap();

            // Create init argument: 10
            let arg = 10i32.to_object(py);
            let pickled_arg = dumps.call1((arg,)).unwrap();
            let arg_bytes: &PyBytes = pickled_arg.downcast().unwrap();

            let spec = ActorSpec {
                id: ActorId::new(),
                class_def: PickledData::new(class_bytes.as_bytes().to_vec()),
                init_args: vec![PickledData::new(arg_bytes.as_bytes().to_vec())],
            };

            let host = ActorHost::new();
            let actor_id = host.create_actor(spec).unwrap();

            // Call increment method
            let call = ActorMethodCall {
                actor_id,
                method_name: "increment".to_string(),
                args: vec![],
            };

            let result = host.call_method(call);

            match result {
                TaskResult::Success { data } => {
                    // Unpickle and check the result
                    let result_bytes = PyBytes::new(py, &data);
                    let unpickled_result = loads.call1((result_bytes,)).unwrap();
                    let result_value: i32 = unpickled_result.extract().unwrap();
                    assert_eq!(result_value, 11);
                }
                TaskResult::Error { traceback } => {
                    panic!("Method call failed: {}", traceback);
                }
            }

            // Call get method to verify state was preserved
            let call = ActorMethodCall {
                actor_id,
                method_name: "get".to_string(),
                args: vec![],
            };

            let result = host.call_method(call);

            match result {
                TaskResult::Success { data } => {
                    let result_bytes = PyBytes::new(py, &data);
                    let unpickled_result = loads.call1((result_bytes,)).unwrap();
                    let result_value: i32 = unpickled_result.extract().unwrap();
                    assert_eq!(result_value, 11);
                }
                TaskResult::Error { traceback } => {
                    panic!("Method call failed: {}", traceback);
                }
            }
        });
    }

    #[test]
    fn test_actor_not_found() {
        let host = ActorHost::new();
        let call = ActorMethodCall {
            actor_id: ActorId::new(),
            method_name: "foo".to_string(),
            args: vec![],
        };

        let result = host.call_method(call);

        match result {
            TaskResult::Error { traceback } => {
                assert!(traceback.contains("Actor not found"));
            }
            TaskResult::Success { .. } => {
                panic!("Expected error for non-existent actor");
            }
        }
    }

    #[test]
    fn test_method_not_found() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let cloudpickle = PyModule::import(py, "cloudpickle").unwrap();
            let dumps = cloudpickle.getattr("dumps").unwrap();

            // Create a simple class
            let code = r#"
class Counter:
    def __init__(self):
        self.count = 0
"#;
            py.run(code, None, None).unwrap();
            let counter_class = py.eval("Counter", None, None).unwrap();
            let pickled_class = dumps.call1((counter_class,)).unwrap();
            let class_bytes: &PyBytes = pickled_class.downcast().unwrap();

            let spec = ActorSpec {
                id: ActorId::new(),
                class_def: PickledData::new(class_bytes.as_bytes().to_vec()),
                init_args: vec![],
            };

            let host = ActorHost::new();
            let actor_id = host.create_actor(spec).unwrap();

            // Call non-existent method
            let call = ActorMethodCall {
                actor_id,
                method_name: "nonexistent".to_string(),
                args: vec![],
            };

            let result = host.call_method(call);

            match result {
                TaskResult::Error { traceback } => {
                    assert!(traceback.contains("nonexistent"));
                }
                TaskResult::Success { .. } => {
                    panic!("Expected error for non-existent method");
                }
            }
        });
    }
}
