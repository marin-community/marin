use crate::types::WorkerId;
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::SystemTime;

/// State of a worker in the pool
#[derive(Debug, Clone)]
pub struct WorkerState {
    pub id: WorkerId,
    pub capacity: usize,
    pub active_tasks: usize,
    pub last_heartbeat: SystemTime,
}

/// Thread-safe worker pool for managing worker resources
#[derive(Clone)]
pub struct WorkerPool {
    inner: Arc<Mutex<WorkerPoolInner>>,
}

struct WorkerPoolInner {
    /// Worker states by ID
    workers: HashMap<WorkerId, WorkerState>,
    /// Round-robin queue for worker selection
    round_robin: VecDeque<WorkerId>,
}

impl WorkerPool {
    /// Create a new worker pool
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(WorkerPoolInner {
                workers: HashMap::new(),
                round_robin: VecDeque::new(),
            })),
        }
    }

    /// Register a new worker with the given capacity
    pub fn register_worker(&self, id: WorkerId, capacity: usize) {
        let mut inner = self.inner.lock();
        let state = WorkerState {
            id,
            capacity,
            active_tasks: 0,
            last_heartbeat: SystemTime::now(),
        };
        inner.workers.insert(id, state);
        inner.round_robin.push_back(id);
    }

    /// Get an available worker (one with capacity remaining)
    /// Uses round-robin selection among available workers
    pub fn get_available_worker(&self) -> Option<WorkerId> {
        let mut inner = self.inner.lock();
        let queue_len = inner.round_robin.len();

        for _ in 0..queue_len {
            if let Some(worker_id) = inner.round_robin.pop_front() {
                if let Some(state) = inner.workers.get(&worker_id) {
                    if state.active_tasks < state.capacity {
                        inner.round_robin.push_back(worker_id);
                        return Some(worker_id);
                    }
                }
                inner.round_robin.push_back(worker_id);
            }
        }

        None
    }

    /// Update the heartbeat timestamp for a worker
    pub fn update_heartbeat(&self, id: &WorkerId) -> bool {
        let mut inner = self.inner.lock();
        if let Some(state) = inner.workers.get_mut(id) {
            state.last_heartbeat = SystemTime::now();
            true
        } else {
            false
        }
    }

    /// Increment the active task count for a worker
    pub fn increment_tasks(&self, id: &WorkerId) -> Option<usize> {
        let mut inner = self.inner.lock();
        inner.workers.get_mut(id).map(|state| {
            state.active_tasks += 1;
            state.active_tasks
        })
    }

    /// Decrement the active task count for a worker
    pub fn decrement_tasks(&self, id: &WorkerId) -> Option<usize> {
        let mut inner = self.inner.lock();
        inner.workers.get_mut(id).map(|state| {
            if state.active_tasks > 0 {
                state.active_tasks -= 1;
            }
            state.active_tasks
        })
    }

    /// Get the state of a specific worker
    pub fn get_worker(&self, id: &WorkerId) -> Option<WorkerState> {
        let inner = self.inner.lock();
        inner.workers.get(id).cloned()
    }

    /// Remove a worker from the pool
    pub fn remove_worker(&self, id: &WorkerId) -> Option<WorkerState> {
        let mut inner = self.inner.lock();
        inner.round_robin.retain(|&wid| wid != *id);
        inner.workers.remove(id)
    }

    /// Get all workers in the pool
    pub fn get_all_workers(&self) -> Vec<WorkerState> {
        let inner = self.inner.lock();
        inner.workers.values().cloned().collect()
    }

    /// Get the total number of workers
    pub fn worker_count(&self) -> usize {
        let inner = self.inner.lock();
        inner.workers.len()
    }

    /// Get the total capacity across all workers
    pub fn total_capacity(&self) -> usize {
        let inner = self.inner.lock();
        inner.workers.values().map(|state| state.capacity).sum()
    }

    /// Get the total number of active tasks across all workers
    pub fn total_active_tasks(&self) -> usize {
        let inner = self.inner.lock();
        inner.workers.values().map(|state| state.active_tasks).sum()
    }
}

impl Default for WorkerPool {
    fn default() -> Self {
        Self::new()
    }
}

