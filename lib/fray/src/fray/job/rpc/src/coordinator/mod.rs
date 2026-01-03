pub mod actor_registry;
pub mod object_store;
pub mod server;
pub mod task_scheduler;
pub mod worker_pool;

pub use actor_registry::{ActorLifetime, ActorMetadata, ActorRegistry};
pub use object_store::ObjectStore;
pub use server::{run_coordinator_server, CoordinatorImpl};
pub use task_scheduler::{TaskScheduler, TaskState};
pub use worker_pool::{WorkerPool, WorkerState};

use crate::worker::actor_host::ActorHost;
use std::sync::Arc;

/// Main coordinator struct aggregating all component subsystems
#[derive(Clone)]
pub struct Coordinator {
    pub object_store: ObjectStore,
    pub task_scheduler: TaskScheduler,
    pub actor_registry: ActorRegistry,
    pub worker_pool: WorkerPool,
    pub actor_host: Arc<ActorHost>,
}

impl Coordinator {
    /// Create a new coordinator with all components initialized
    pub fn new() -> Self {
        Self {
            object_store: ObjectStore::new(),
            task_scheduler: TaskScheduler::new(),
            actor_registry: ActorRegistry::new(),
            worker_pool: WorkerPool::new(),
            actor_host: Arc::new(ActorHost::new()),
        }
    }

    /// Create a new coordinator wrapped in an Arc for shared ownership
    pub fn new_shared() -> Arc<Self> {
        Arc::new(Self::new())
    }
}

impl Default for Coordinator {
    fn default() -> Self {
        Self::new()
    }
}

