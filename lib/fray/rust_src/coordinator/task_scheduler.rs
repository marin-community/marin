use crate::types::{Task, TaskId, TaskResult, WorkerId};
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::SystemTime;

/// State of a running task
#[derive(Debug, Clone)]
pub struct TaskState {
    pub task_id: TaskId,
    pub worker_id: WorkerId,
    pub submitted_at: SystemTime,
}

/// Thread-safe task scheduler with queue and execution tracking
#[derive(Clone)]
pub struct TaskScheduler {
    inner: Arc<Mutex<TaskSchedulerInner>>,
}

struct TaskSchedulerInner {
    /// Queue of pending tasks
    pending: VecDeque<Task>,
    /// Currently running tasks
    running: HashMap<TaskId, TaskState>,
    /// Completed tasks with their results
    completed: HashMap<TaskId, TaskResult>,
}

impl TaskScheduler {
    /// Create a new task scheduler
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(TaskSchedulerInner {
                pending: VecDeque::new(),
                running: HashMap::new(),
                completed: HashMap::new(),
            })),
        }
    }

    /// Enqueue a task for execution
    pub fn enqueue(&self, task: Task) {
        let mut inner = self.inner.lock();
        inner.pending.push_back(task);
    }

    /// Dequeue the next pending task
    pub fn dequeue(&self) -> Option<Task> {
        let mut inner = self.inner.lock();
        inner.pending.pop_front()
    }

    /// Mark a task as running on a specific worker
    pub fn mark_running(&self, task_id: TaskId, worker_id: WorkerId) {
        let mut inner = self.inner.lock();
        let state = TaskState {
            task_id,
            worker_id,
            submitted_at: SystemTime::now(),
        };
        inner.running.insert(task_id, state);
    }

    /// Complete a task with its result
    /// Removes the task from the running set and stores the result
    pub fn complete_task(&self, task_id: TaskId, result: TaskResult) {
        let mut inner = self.inner.lock();
        inner.running.remove(&task_id);
        inner.completed.insert(task_id, result);
    }

    /// Get the result of a completed task
    pub fn get_result(&self, task_id: &TaskId) -> Option<TaskResult> {
        let inner = self.inner.lock();
        inner.completed.get(task_id).cloned()
    }

    /// Get the state of a running task
    pub fn get_running(&self, task_id: &TaskId) -> Option<TaskState> {
        let inner = self.inner.lock();
        inner.running.get(task_id).cloned()
    }

    /// Get the number of pending tasks
    pub fn pending_count(&self) -> usize {
        let inner = self.inner.lock();
        inner.pending.len()
    }

    /// Get the number of running tasks
    pub fn running_count(&self) -> usize {
        let inner = self.inner.lock();
        inner.running.len()
    }

    /// Get all running tasks
    pub fn get_all_running(&self) -> Vec<TaskState> {
        let inner = self.inner.lock();
        inner.running.values().cloned().collect()
    }
}

impl Default for TaskScheduler {
    fn default() -> Self {
        Self::new()
    }
}

