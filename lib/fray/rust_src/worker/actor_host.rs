use crate::ActorId;
use log::{debug, error, info, warn};
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
        info!(target: "fray::rpc::worker::actors",
              "Creating actor {}", spec.id);

        Python::attach(|py| {
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
            let args_tuple = PyTuple::new(py, &unpickled_args)
                .map_err(|e| format!("Failed to create args tuple: {}", e))?;

            // Instantiate the class
            let instance = class
                .call1(&args_tuple)
                .map_err(|e| {
                    let err_msg = format!("Failed to instantiate class: {}", e);
                    error!(target: "fray::rpc::worker::actors",
                           "Actor {}: {}", spec.id, err_msg);
                    err_msg
                })?;

            // Pickle the instance
            let pickled_instance = dumps
                .call1((instance,))
                .map_err(|e| format!("Failed to pickle instance: {}", e))?;

            let instance_bytes = pickled_instance
                .cast::<PyBytes>()
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

            let instance_size = instance_bytes.as_bytes().len();
            actors.insert(spec.id, actor);

            info!(target: "fray::rpc::worker::actors",
                  "Actor {} created successfully, instance size: {} bytes",
                  spec.id, instance_size);

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
        debug!(target: "fray::rpc::worker::actors",
               "Calling method {} on actor {}", call.method_name, call.actor_id);

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
                    warn!(target: "fray::rpc::worker::actors",
                          "Actor {} not found", call.actor_id);
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

        Python::attach(|py| {
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
            let args_tuple = match PyTuple::new(py, &unpickled_args) {
                Ok(tuple) => tuple,
                Err(err) => {
                    return TaskResult::Error {
                        traceback: format!("Failed to create args tuple: {}", err),
                    };
                }
            };

            // Call the method
            let result = match method.call1(&args_tuple) {
                Ok(res) => res,
                Err(err) => {
                    let err_msg = format!("Method call failed: {}", err);
                    error!(target: "fray::rpc::worker::actors",
                           "Actor {} method {} failed: {}",
                           call.actor_id, call.method_name, err_msg);
                    return TaskResult::Error {
                        traceback: err_msg,
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

            let result_bytes = match pickled_result.cast::<PyBytes>() {
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

            let instance_bytes = match pickled_instance.cast::<PyBytes>() {
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

            let data = result_bytes.as_bytes().to_vec();
            debug!(target: "fray::rpc::worker::actors",
                   "Actor {} method {} completed, result size: {} bytes",
                   call.actor_id, call.method_name, data.len());

            TaskResult::Success { data }
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
