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

use std::sync::Arc;

/// Main coordinator struct aggregating all component subsystems
#[derive(Clone)]
pub struct Coordinator {
    pub object_store: ObjectStore,
    pub task_scheduler: TaskScheduler,
    pub actor_registry: ActorRegistry,
    pub worker_pool: WorkerPool,
}

impl Coordinator {
    /// Create a new coordinator with all components initialized
    pub fn new() -> Self {
        Self {
            object_store: ObjectStore::new(),
            task_scheduler: TaskScheduler::new(),
            actor_registry: ActorRegistry::new(),
            worker_pool: WorkerPool::new(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ActorId, ObjectId, Task, TaskId, TaskResult, WorkerId};
    use std::time::SystemTime;

    #[test]
    fn test_coordinator_creation() {
        let coordinator = Coordinator::new();

        assert_eq!(coordinator.worker_pool.worker_count(), 0);
        assert_eq!(coordinator.actor_registry.count(), 0);
        assert_eq!(coordinator.task_scheduler.pending_count(), 0);
    }

    #[test]
    fn test_coordinator_integration() {
        let coordinator = Coordinator::new();

        // Register a worker
        let worker_id = WorkerId(1);
        coordinator.worker_pool.register_worker(worker_id, 10);

        // Register an actor
        let actor_id = ActorId(1);
        let metadata = ActorMetadata {
            id: actor_id,
            name: Some("test_actor".to_string()),
            worker_id,
            lifetime: ActorLifetime::Persistent,
            created_at: SystemTime::now(),
        };
        coordinator.actor_registry.register(actor_id, metadata);

        // Add an object
        let object_id = ObjectId(1);
        coordinator
            .object_store
            .insert(object_id, vec![1, 2, 3, 4]);

        // Schedule a task
        let task = Task {
            id: TaskId(1),
            payload: vec![5, 6, 7],
        };
        coordinator.task_scheduler.enqueue(task.clone());

        // Verify all components are working
        assert_eq!(coordinator.worker_pool.worker_count(), 1);
        assert_eq!(coordinator.actor_registry.count(), 1);
        assert_eq!(
            coordinator.actor_registry.get_by_name("test_actor"),
            Some(actor_id)
        );
        assert_eq!(
            coordinator.object_store.get(&object_id),
            Some(vec![1, 2, 3, 4])
        );
        assert_eq!(coordinator.task_scheduler.pending_count(), 1);

        // Dequeue and run the task
        let dequeued = coordinator.task_scheduler.dequeue().unwrap();
        assert_eq!(dequeued.id, task.id);

        coordinator
            .task_scheduler
            .mark_running(dequeued.id, worker_id);
        coordinator.worker_pool.increment_tasks(&worker_id);

        // Complete the task
        let result = TaskResult {
            task_id: dequeued.id,
            result: vec![8, 9, 10],
        };
        coordinator.task_scheduler.complete_task(dequeued.id, result);
        coordinator.worker_pool.decrement_tasks(&worker_id);

        assert_eq!(coordinator.task_scheduler.running_count(), 0);
        assert_eq!(
            coordinator
                .task_scheduler
                .get_result(&dequeued.id)
                .unwrap()
                .result,
            vec![8, 9, 10]
        );
    }
}
