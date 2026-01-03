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

