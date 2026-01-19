# Task State Tracking Refactoring Design

This document describes the design for refactoring Iris task/job state tracking to properly separate concerns between Jobs, Tasks, and Attempts.

## Background

### Problem Statement

The Iris controller dashboard was showing "No worker assigned" for jobs because worker assignments were being lost on retry. The root cause was architectural: both `Job.worker_id` and `Task.worker_id` were mutable fields that got reset to `None` on retry, when in fact:

1. **Jobs don't execute on workers** - tasks do
2. **Worker assignments belong to attempts**, not tasks directly

### Initial Fix

The initial fix addressed the immediate symptom:

1. Removed `Job.worker_id` since jobs don't execute directly on workers
2. Converted `Task.worker_id` to a property that reads from `attempts[-1].worker_id`
3. Updated `mark_dispatched()` to create attempts immediately

However, this revealed deeper architectural issues that this document proposes to address.

## Current State Analysis

### Problems with Current Implementation

#### 1. Task State Duplicates Attempt Fields

The `Task` dataclass currently has fields that conceptually belong to individual attempts:

```python
@dataclass
class Task:
    # ...
    started_at_ms: int | None = None      # Should be on attempt
    finished_at_ms: int | None = None     # Should be on attempt
    error: str | None = None              # Should be on attempt
    exit_code: int | None = None          # Should be on attempt
```

These are redundant with `TaskAttempt` data and require manual synchronization during transitions.

#### 2. State Transitions Happen on Tasks, Not Attempts

Currently `task.transition()` updates task state directly, with attempt tracking as a secondary concern:

```python
def transition(self, new_state, now_ms, ...):
    # Updates task.state directly
    # Attempt tracking is an afterthought
    if new_state == cluster_pb2.TASK_STATE_SUCCEEDED:
        self.state = new_state
        self.finished_at_ms = now_ms  # Task field, not attempt
        self.exit_code = exit_code or 0
```

This inverts the correct relationship: attempts should own state, and task state should be derived.

#### 3. Job State Handling Confusion

Jobs call `mark_dispatched(now_ms)` as if they dispatch themselves:

```python
# Current: Jobs mark themselves as dispatched
job.mark_dispatched(now_ms=1000)
```

But jobs don't execute - their tasks do. Job state should be purely derived from task state aggregation.

#### 4. Internal State vs External APIs Conflation

The controller, state manager, and scheduler still think about tasks having direct `worker_id` fields:

```python
# Scheduler assigns tasks to workers
transaction.tentatively_assign(task, worker)

# State tracks task -> worker
worker.running_tasks.add(task_id)
```

Internally, we should think about attempts. External APIs can expose `worker_id` for convenience.

## Proposed Architecture

### Design Principles

1. **Attempts own execution state** - All execution-related data (worker_id, timestamps, exit_code, error) belongs to attempts
2. **Task state is derived** - Task's `state` field summarizes the current attempt's state
3. **Job state is aggregated** - Job state is computed from task state counts
4. **State flows upward** - Attempt transitions percolate up to Task, then to Job

### Entity Hierarchy

```
Job
 |
 +-- num_tasks: int
 +-- task_state_counts: Counter[TaskState]
 +-- state: JobState (derived from counts)
 |
 +-- Task[]
      |
      +-- task_id, job_id, task_index
      +-- state: TaskState (derived from current attempt)
      +-- attempts: list[TaskAttempt]
      +-- current_attempt_id: int
      |
      +-- TaskAttempt[]
           |
           +-- attempt_id
           +-- worker_id
           +-- state: TaskState
           +-- started_at_ms, finished_at_ms
           +-- exit_code, error
           +-- is_worker_failure
```

### State Flow

```
                    External Report
                          |
                          v
+--------------------+   transition()   +--------------------+
|   TaskAttempt      | <--------------- |   Worker Report    |
|                    |                  |   (attempt_id,     |
|  - state           |                  |    new_state)      |
|  - worker_id       |                  +--------------------+
|  - exit_code       |
|  - error           |
+--------------------+
          |
          | percolate
          v
+--------------------+
|   Task             |
|                    |
|  - state (property)|  <-- reads from current attempt
|  - worker_id (prop)|  <-- reads from current attempt
+--------------------+
          |
          | aggregate
          v
+--------------------+
|   Job              |
|                    |
|  - task_state_counts
|  - state (derived) |  <-- computed from counts
+--------------------+
```

## Data Model Changes

### TaskAttempt (expanded)

The `TaskAttempt` dataclass should own all execution state:

```python
@dataclass
class TaskAttempt:
    """Record of a single task execution attempt.

    An attempt represents one try at executing a task on a specific worker.
    All execution-related state (timestamps, exit codes, errors) lives here.
    """
    attempt_id: int
    worker_id: WorkerId | None = None
    state: int = cluster_pb2.TASK_STATE_PENDING

    # Timing
    created_at_ms: int = 0           # When attempt was created
    started_at_ms: int | None = None # When execution actually started
    finished_at_ms: int | None = None

    # Result
    exit_code: int | None = None
    error: str | None = None
    is_worker_failure: bool = False

    def transition(
        self,
        new_state: int,
        now_ms: int,
        *,
        exit_code: int | None = None,
        error: str | None = None,
        is_worker_failure: bool = False,
    ) -> None:
        """Transition this attempt to a new state."""
        self.state = new_state

        if new_state == cluster_pb2.TASK_STATE_RUNNING:
            self.started_at_ms = now_ms

        if new_state in TERMINAL_TASK_STATES:
            self.finished_at_ms = now_ms
            self.exit_code = exit_code
            self.error = error
            self.is_worker_failure = is_worker_failure

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_TASK_STATES
```

### Task (simplified)

The `Task` dataclass should delegate execution state to attempts:

```python
@dataclass
class Task:
    """Controller's representation of a task within a job.

    Tasks track identity and retry policy. Execution state (worker assignments,
    timestamps, results) is stored in TaskAttempt objects.
    """
    task_id: TaskId
    job_id: JobId
    task_index: int

    # Retry policy (immutable after creation)
    max_retries_failure: int = 0
    max_retries_preemption: int = 100

    # Retry counters
    failure_count: int = 0
    preemption_count: int = 0

    # Attempt tracking
    attempts: list[TaskAttempt] = field(default_factory=list)

    # Submission timestamp (distinct from attempt start times)
    submitted_at_ms: int = 0

    # --- Properties that delegate to current attempt ---

    @property
    def current_attempt(self) -> TaskAttempt | None:
        """The most recent attempt, or None if no attempts yet."""
        return self.attempts[-1] if self.attempts else None

    @property
    def current_attempt_id(self) -> int:
        """ID of current attempt (0-indexed)."""
        return len(self.attempts) - 1 if self.attempts else -1

    @property
    def state(self) -> int:
        """Task state is the state of the current attempt, or PENDING if none."""
        if not self.attempts:
            return cluster_pb2.TASK_STATE_PENDING
        return self.attempts[-1].state

    @property
    def worker_id(self) -> WorkerId | None:
        """Worker from current attempt, if any."""
        if not self.attempts:
            return None
        return self.attempts[-1].worker_id

    @property
    def started_at_ms(self) -> int | None:
        """Start time of current attempt."""
        if not self.attempts:
            return None
        return self.attempts[-1].started_at_ms

    @property
    def finished_at_ms(self) -> int | None:
        """Finish time of current attempt."""
        if not self.attempts:
            return None
        return self.attempts[-1].finished_at_ms

    @property
    def exit_code(self) -> int | None:
        """Exit code from current attempt."""
        if not self.attempts:
            return None
        return self.attempts[-1].exit_code

    @property
    def error(self) -> str | None:
        """Error from current attempt."""
        if not self.attempts:
            return None
        return self.attempts[-1].error
```

### Task Methods (refactored)

```python
class Task:
    # ... (properties from above)

    def create_attempt(self, worker_id: WorkerId, now_ms: int) -> TaskAttempt:
        """Create a new attempt for this task.

        Called when the scheduler assigns this task to a worker.
        Returns the new attempt so caller can track it.
        """
        attempt = TaskAttempt(
            attempt_id=len(self.attempts),
            worker_id=worker_id,
            state=cluster_pb2.TASK_STATE_RUNNING,
            created_at_ms=now_ms,
            started_at_ms=now_ms,
        )
        self.attempts.append(attempt)
        return attempt

    def revert_attempt(self) -> None:
        """Remove the current attempt if dispatch RPC fails.

        Called when we created an attempt but the RPC to dispatch
        to the worker failed, so we need to undo.
        """
        if self.attempts:
            self.attempts.pop()

    def handle_attempt_result(
        self,
        new_state: int,
        now_ms: int,
        *,
        is_worker_failure: bool = False,
        error: str | None = None,
        exit_code: int | None = None,
    ) -> TaskTransitionResult:
        """Handle a state report for the current attempt.

        Transitions the current attempt's state and handles retry logic:
        - If retriable failure: returns SHOULD_RETRY (task stays in list, ready for new attempt)
        - If terminal success: returns COMPLETE
        - If exhausted retries: returns EXCEEDED_RETRY_LIMIT

        Does NOT create new attempts - that's the scheduler's job.
        """
        if not self.attempts:
            raise ValueError("Cannot handle result without an attempt")

        attempt = self.attempts[-1]
        attempt.transition(
            new_state,
            now_ms,
            exit_code=exit_code,
            error=error,
            is_worker_failure=is_worker_failure,
        )

        # Handle retry logic for failure states
        if new_state in (
            cluster_pb2.TASK_STATE_FAILED,
            cluster_pb2.TASK_STATE_WORKER_FAILED,
        ):
            return self._handle_failure(is_worker_failure)

        return TaskTransitionResult.COMPLETE

    def _handle_failure(self, is_worker_failure: bool) -> TaskTransitionResult:
        """Determine if task should retry after a failure."""
        if is_worker_failure:
            self.preemption_count += 1
            can_retry = self.preemption_count <= self.max_retries_preemption
        else:
            self.failure_count += 1
            can_retry = self.failure_count <= self.max_retries_failure

        if can_retry:
            return TaskTransitionResult.SHOULD_RETRY
        else:
            return TaskTransitionResult.EXCEEDED_RETRY_LIMIT

    def is_finished(self) -> bool:
        """Check if task is in a terminal state (no more retries possible)."""
        if not self.attempts:
            return False
        return self.state in TERMINAL_TASK_STATES

    def can_be_scheduled(self) -> bool:
        """Check if task is ready to be scheduled.

        A task can be scheduled if:
        - It has no attempts yet (fresh task), or
        - Its current attempt is terminal AND it should retry
        """
        if not self.attempts:
            return True
        return self.attempts[-1].is_terminal() and not self.is_finished()
```

### Job State (simplified)

Jobs should not track worker assignments or have mark_dispatched for workers:

```python
@dataclass
class Job:
    """Job with task-aggregated state.

    Job state is derived from task state counts. Jobs do not track
    workers directly - that information lives in task attempts.
    """
    job_id: JobId
    request: cluster_pb2.Controller.LaunchJobRequest

    # Task state aggregation
    num_tasks: int = 0
    task_state_counts: Counter[int] = field(default_factory=Counter)

    # Derived state (set by on_task_transition)
    state: int = cluster_pb2.JOB_STATE_PENDING
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    error: str | None = None
    exit_code: int | None = None

    # Gang scheduling
    gang_id: str | None = None
    parent_job_id: JobId | None = None

    # Timestamps
    submitted_at_ms: int = 0

    # NOTE: No worker_id field - jobs don't execute on workers
    # NOTE: No mark_dispatched(worker_id) - that's for tasks
    # NOTE: No attempts list - job "retries" are really task retries

    def on_task_transition(
        self,
        old_state: int | None,
        new_state: int,
        now_ms: int,
    ) -> int | None:
        """Update task state counts and compute new job state.

        Called by ControllerState when a task transitions.
        Returns new job state if changed, None otherwise.
        """
        if old_state is not None:
            self.task_state_counts[old_state] -= 1
        self.task_state_counts[new_state] += 1

        return self._compute_job_state(now_ms)

    def _compute_job_state(self, now_ms: int) -> int | None:
        """Derive job state from task counts. O(1) - no iteration."""
        counts = self.task_state_counts

        # Job succeeds when all tasks succeed
        if counts[cluster_pb2.TASK_STATE_SUCCEEDED] == self.num_tasks:
            self.finished_at_ms = now_ms
            self.exit_code = 0
            return cluster_pb2.JOB_STATE_SUCCEEDED

        # Job fails when failures exceed threshold
        max_failures = self.request.max_task_failures
        if counts[cluster_pb2.TASK_STATE_FAILED] > max_failures:
            self.finished_at_ms = now_ms
            return cluster_pb2.JOB_STATE_FAILED

        # Job unschedulable if any task is unschedulable
        if counts[cluster_pb2.TASK_STATE_UNSCHEDULABLE] > 0:
            self.finished_at_ms = now_ms
            return cluster_pb2.JOB_STATE_UNSCHEDULABLE

        # Job killed if any task was killed
        if counts[cluster_pb2.TASK_STATE_KILLED] > 0:
            if not self.is_finished():
                self.finished_at_ms = now_ms
                return cluster_pb2.JOB_STATE_KILLED

        # Job is RUNNING if any task is running
        if counts[cluster_pb2.TASK_STATE_RUNNING] > 0:
            if self.state != cluster_pb2.JOB_STATE_RUNNING:
                self.started_at_ms = now_ms
                return cluster_pb2.JOB_STATE_RUNNING

        return None

    def is_finished(self) -> bool:
        return self.state in (
            cluster_pb2.JOB_STATE_SUCCEEDED,
            cluster_pb2.JOB_STATE_FAILED,
            cluster_pb2.JOB_STATE_KILLED,
            cluster_pb2.JOB_STATE_WORKER_FAILED,
            cluster_pb2.JOB_STATE_UNSCHEDULABLE,
        )
```

### JobAttempt Removal

The current `JobAttempt` class should be removed. Jobs don't have attempts in the same sense tasks do:

```python
# REMOVE THIS:
@dataclass
class JobAttempt:
    """Record of a single job execution attempt."""
    attempt_id: int
    worker_id: WorkerId | None = None  # Jobs don't have workers!
    state: int = cluster_pb2.JOB_STATE_PENDING
    # ...
```

Job "retry" semantics are really task retry semantics. When a task fails and retries, the job stays in RUNNING state. Job-level retry (resubmitting all tasks) is a separate operation that could be explicit if needed.

## State Transition Flow

### Dispatch Flow (new)

```python
# In Scheduler/Dispatcher:
def dispatch_task(task: Task, worker: ControllerWorker, now_ms: int):
    """Dispatch a task to a worker."""

    # 1. Create attempt BEFORE RPC (optimistic)
    attempt = task.create_attempt(worker.worker_id, now_ms)

    # 2. Update state tracking
    state.assign_task_to_worker(worker.worker_id, task.task_id)

    # 3. Notify job of task state change
    old_state = cluster_pb2.TASK_STATE_PENDING  # Task was pending
    new_state = task.state  # Now RUNNING (from attempt)
    job = state.get_job(task.job_id)
    if job:
        new_job_state = job.on_task_transition(old_state, new_state, now_ms)
        if new_job_state:
            job.state = new_job_state

    # 4. Attempt RPC to worker
    try:
        worker_stub.run_task(RunTaskRequest(
            task_id=task.task_id,
            attempt_id=attempt.attempt_id,
            # ...
        ))
    except RpcError:
        # 5. Rollback on failure
        task.revert_attempt()
        state.rollback_task_assignment(worker.worker_id, task)
        # Revert job state counts
        if job:
            job.task_state_counts[new_state] -= 1
            job.task_state_counts[old_state] += 1
        raise
```

### Status Report Flow (new)

```python
# In ControllerService.report_task_state:
def report_task_state(request: ReportTaskStateRequest):
    task = state.get_task(request.task_id)
    if not task:
        return  # Unknown task

    # Validate attempt ID
    if request.attempt_id != task.current_attempt_id:
        logger.warning("Stale report: expected attempt %d, got %d",
                       task.current_attempt_id, request.attempt_id)
        return

    # Get old state for job aggregation
    old_state = task.state

    # Transition the attempt (which updates task.state via property)
    result = task.handle_attempt_result(
        request.state,
        now_ms,
        is_worker_failure=(request.state == TASK_STATE_WORKER_FAILED),
        error=request.error,
        exit_code=request.exit_code,
    )

    # Update job state counts
    job = state.get_job(task.job_id)
    if job:
        new_job_state = job.on_task_transition(old_state, task.state, now_ms)
        if new_job_state:
            job.state = new_job_state

    # Handle retry
    if result == TaskTransitionResult.SHOULD_RETRY:
        # Task is ready to be scheduled again
        # Scheduler will create a new attempt when it assigns the task
        state.requeue_task(task.task_id)
        state.unassign_task_from_worker(task.worker_id, task.task_id)
```

### Retry Flow (clarified)

The key insight is that **retry does not create a new attempt immediately**. Instead:

1. Task reports failure with `handle_attempt_result()`
2. If retriable, method returns `SHOULD_RETRY`
3. Task is re-queued for scheduling
4. Scheduler later assigns task to a (possibly different) worker
5. Assignment creates a new attempt via `create_attempt()`

This means a task can be in a state where:
- Its current attempt is in a TERMINAL state (FAILED, WORKER_FAILED)
- But the task itself is not "finished" because it can still retry
- The task is waiting in the queue for a new attempt

```python
# Task with retry available:
task.attempts[-1].state == TASK_STATE_FAILED  # Last attempt failed
task.can_be_scheduled() == True               # Ready for new attempt
task.is_finished() == False                   # Not terminal

# Task with no retries left:
task.attempts[-1].state == TASK_STATE_FAILED  # Last attempt failed
task.can_be_scheduled() == False              # No more retries
task.is_finished() == True                    # Terminal
```

## Proto Changes

### Remove Stale Fields from JobStatus

```protobuf
message JobStatus {
  string job_id = 1;
  JobState state = 2;
  int32 exit_code = 3;
  string error = 4;
  int64 started_at_ms = 5;
  int64 finished_at_ms = 6;

  // REMOVE: Jobs don't execute on workers
  // string worker_id = 11;
  // string worker_address = 12;

  // Keep port allocations (aggregated from tasks)
  map<string, int32> ports = 7;

  // Keep resource usage (aggregated)
  ResourceUsage resource_usage = 8;
  string status_message = 9;
  BuildMetrics build_metrics = 10;
  bytes serialized_result = 13;
  string parent_job_id = 14;

  // REMOVE: Job attempts don't make sense
  // int32 current_attempt_id = 15;
  // repeated JobAttempt attempts = 16;

  // Keep retry counts (aggregated from task retries)
  int32 failure_count = 17;
  int32 preemption_count = 18;

  // Task aggregation
  int32 num_tasks = 19;
  int32 tasks_pending = 20;
  int32 tasks_running = 21;
  int32 tasks_succeeded = 22;
  int32 tasks_failed = 23;
  repeated TaskStatus tasks = 24;
}
```

### Remove worker_id from JobAttempt (or remove JobAttempt entirely)

```protobuf
// REMOVE this message entirely - jobs don't have attempts
// message JobAttempt {
//   int32 attempt_id = 1;
//   string worker_id = 2;  // Jobs don't have workers!
//   ...
// }
```

### Keep TaskStatus for External API

```protobuf
message TaskStatus {
  string task_id = 1;
  string job_id = 2;
  int32 task_index = 3;
  TaskState state = 4;

  // Keep for external API convenience
  // Internally this comes from task.attempts[-1].worker_id
  string worker_id = 5;
  string worker_address = 6;

  int32 exit_code = 7;
  string error = 8;
  int64 started_at_ms = 9;
  int64 finished_at_ms = 10;

  map<string, int32> ports = 11;
  ResourceUsage resource_usage = 12;
  BuildMetrics build_metrics = 13;

  int32 current_attempt_id = 14;
  repeated TaskAttempt attempts = 15;
}
```

## Controller/Scheduler/State Changes

### ControllerState Changes

```python
class ControllerState:
    def mark_task_dispatched(self, task: Task, worker_id: WorkerId, now_ms: int) -> int | None:
        """Mark a task as dispatched by creating a new attempt.

        Returns new job state if changed.
        """
        with self._lock:
            job = self._jobs.get(task.job_id)
            old_state = task.state  # Before creating attempt

            # Create attempt (changes task.state to RUNNING)
            task.create_attempt(worker_id, now_ms)

            if job:
                new_job_state = job.on_task_transition(old_state, task.state, now_ms)
                if new_job_state:
                    job.state = new_job_state
                return new_job_state
            return None

    def transition_task(
        self,
        task_id: TaskId,
        new_state: int,
        now_ms: int,
        **kwargs,
    ) -> tuple[TaskTransitionResult, list[ControllerEndpoint]]:
        """Transition a task via its current attempt."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return TaskTransitionResult.COMPLETE, []

            job = self._jobs.get(task.job_id)
            old_state = task.state

            # Transition via attempt
            result = task.handle_attempt_result(new_state, now_ms, **kwargs)

            # Update job state
            if job:
                new_job_state = job.on_task_transition(old_state, task.state, now_ms)
                if new_job_state:
                    job.state = new_job_state
                    self._handle_job_state_change(job, new_job_state, now_ms)

            # Handle side effects
            removed_endpoints = []
            if result == TaskTransitionResult.SHOULD_RETRY:
                self._requeue_task_for_retry(task)
            elif task.is_finished():
                removed_endpoints = self._cleanup_finished_task(task)

            return result, removed_endpoints
```

### Scheduler Changes

The scheduler already thinks in terms of tasks, which is correct. The main change is that it should be aware that:

1. A task might have previous attempts
2. Creating an attempt is part of the dispatch process

```python
class Scheduler:
    def find_assignments(
        self,
        pending_tasks: list[Task],
        workers: list[ControllerWorker],
        now_ms: int,
    ) -> SchedulingTransaction:
        """Match pending tasks to available workers.

        Tasks in the pending list may have previous attempts (retries).
        We filter by can_be_scheduled() to ensure we only schedule
        tasks that are ready for a new attempt.
        """
        transaction = SchedulingTransaction(self._state)
        capacities = build_capacity_map(self._state, workers)

        for task in pending_tasks:
            # Skip tasks that can't be scheduled
            if not task.can_be_scheduled():
                continue

            job = self._state.get_job(task.job_id)
            if not job:
                continue

            if self._is_task_timed_out(task, job, now_ms):
                transaction.timed_out_tasks.append(task)
                continue

            for worker in workers:
                if not worker.healthy:
                    continue
                capacity = capacities[worker.worker_id]
                if worker_can_fit_task(capacity, job):
                    deduct_task_from_capacity(capacity, job)
                    transaction.tentatively_assign(task, worker)
                    break

        return transaction
```

## Dashboard/UI Changes

### Job Detail Page

The job detail page should:

1. **Remove** job-level worker display (jobs don't have workers)
2. **Add** task list table showing all tasks with their current workers
3. Show task-level attempt history in expandable rows

```
Job: my-training-job
Status: RUNNING
Started: 2025-01-15 10:30:00
Tasks: 3 total, 2 running, 1 pending

+--------+----------+--------+-----------------+----------+
| Task   | State    | Worker | Started         | Attempts |
+--------+----------+--------+-----------------+----------+
| task-0 | RUNNING  | w1     | 10:30:05        | 1        |
| task-1 | RUNNING  | w2     | 10:30:06        | 2        |  <- Had a retry
| task-2 | PENDING  | -      | -               | 0        |
+--------+----------+--------+-----------------+----------+

[Click task-1 for attempt history]
```

### Task Detail Page

When clicking on a task, show attempt history:

```
Task: my-training-job/task-1
Current State: RUNNING
Current Worker: w2

Attempt History:
+----------+--------+----------+-------------+---------------+
| Attempt  | Worker | State    | Started     | Finished      |
+----------+--------+----------+-------------+---------------+
| 0        | w3     | FAILED   | 10:30:02    | 10:30:04      |
| 1 (curr) | w2     | RUNNING  | 10:30:06    | -             |
+----------+--------+----------+-------------+---------------+

Attempt 0 Error: "Worker w3 timed out"
```

### API Endpoints

Add or modify endpoints:

```
GET /api/jobs/{job_id}/tasks
  Returns: list of TaskStatus for all tasks in job

GET /api/tasks/{task_id}
  Returns: TaskStatus with full attempt history

GET /api/tasks/{task_id}/attempts
  Returns: list of TaskAttempt for task
```

## Migration Strategy

Since Iris is new code without production deployments, we can make breaking changes:

1. **Remove stale proto fields** - Update `cluster.proto` to remove `worker_id` from `JobStatus` and `JobAttempt`
2. **Run `scripts/generate_protos.py`** - Regenerate Python bindings
3. **Update Task dataclass** - Convert fields to properties
4. **Remove Job.mark_dispatched(worker_id)** - Already done, verify no callers
5. **Remove JobAttempt** - Replace with task-level retry tracking
6. **Update tests** - Check attempt state instead of task fields for worker info

### Code Changes Required

Files to modify:

1. `lib/iris/src/iris/cluster/controller/task.py` - Refactor Task/TaskAttempt
2. `lib/iris/src/iris/cluster/controller/job.py` - Remove JobAttempt, simplify Job
3. `lib/iris/src/iris/cluster/controller/state.py` - Update dispatch/transition methods
4. `lib/iris/src/iris/cluster/controller/service.py` - Update RPC handlers
5. `lib/iris/src/iris/rpc/cluster.proto` - Remove stale fields
6. `lib/iris/tests/cluster/controller/test_*.py` - Update tests

## Testing Strategy

### Tests to Update

1. **Remove assertions about `job.worker_id`** - Jobs don't have workers
2. **Check attempt state instead of task fields** for worker info:

```python
# OLD:
assert task.worker_id == "w1"
assert task.started_at_ms == 1000

# NEW:
assert task.current_attempt.worker_id == "w1"
assert task.current_attempt.started_at_ms == 1000
# Or via properties:
assert task.worker_id == "w1"  # Property delegates to attempt
```

### New Tests to Add

```python
def test_task_state_is_property_of_current_attempt():
    """Task.state reflects current attempt state."""
    task = Task(task_id=TaskId("t1"), job_id=JobId("j1"), task_index=0)

    # No attempts yet
    assert task.state == TASK_STATE_PENDING
    assert task.worker_id is None

    # Create attempt
    attempt = task.create_attempt(WorkerId("w1"), now_ms=1000)
    assert task.state == TASK_STATE_RUNNING
    assert task.worker_id == "w1"

    # Transition attempt
    task.handle_attempt_result(TASK_STATE_SUCCEEDED, now_ms=2000)
    assert task.state == TASK_STATE_SUCCEEDED
    assert task.finished_at_ms == 2000


def test_task_retry_creates_new_attempt():
    """Retrying a task should result in a new attempt."""
    task = Task(
        task_id=TaskId("t1"),
        job_id=JobId("j1"),
        task_index=0,
        max_retries_failure=1,
    )

    # First attempt fails
    task.create_attempt(WorkerId("w1"), now_ms=1000)
    result = task.handle_attempt_result(TASK_STATE_FAILED, now_ms=2000, error="oops")

    assert result == TaskTransitionResult.SHOULD_RETRY
    assert len(task.attempts) == 1
    assert task.attempts[0].state == TASK_STATE_FAILED
    assert task.can_be_scheduled() == True

    # Second attempt (would be created by scheduler)
    task.create_attempt(WorkerId("w2"), now_ms=3000)

    assert len(task.attempts) == 2
    assert task.current_attempt.worker_id == "w2"
    assert task.state == TASK_STATE_RUNNING


def test_job_state_aggregates_from_tasks():
    """Job state should be derived from task state counts."""
    state = ControllerState()
    job = Job(job_id=JobId("j1"), request=make_request(replicas=3))
    tasks = state.add_job(job)

    assert job.state == JOB_STATE_PENDING

    # Start first task
    state.mark_task_dispatched(tasks[0], WorkerId("w1"), now_ms=1000)
    assert job.state == JOB_STATE_RUNNING

    # All tasks succeed
    for task in tasks:
        if task.state == TASK_STATE_PENDING:
            state.mark_task_dispatched(task, WorkerId("w1"), now_ms=2000)
        state.transition_task(task.task_id, TASK_STATE_SUCCEEDED, now_ms=3000)

    assert job.state == JOB_STATE_SUCCEEDED


def test_attempt_id_validation_rejects_stale_reports():
    """Reports for old attempts should be rejected."""
    task = Task(task_id=TaskId("t1"), job_id=JobId("j1"), task_index=0)

    # Create first attempt
    task.create_attempt(WorkerId("w1"), now_ms=1000)
    assert task.current_attempt_id == 0

    # Fail and retry
    task.handle_attempt_result(TASK_STATE_FAILED, now_ms=2000)
    task.create_attempt(WorkerId("w2"), now_ms=3000)
    assert task.current_attempt_id == 1

    # Simulate stale report for attempt 0
    # Controller should validate and reject (tested at controller level)
```

### Integration Tests

```python
def test_full_lifecycle_with_retry():
    """Test complete task lifecycle including retry."""
    state = ControllerState()
    job = Job(job_id=JobId("j1"), request=make_request())
    job.max_retries_failure = 1
    tasks = state.add_job(job)
    task = tasks[0]

    # Dispatch to worker 1
    state.mark_task_dispatched(task, WorkerId("w1"), now_ms=1000)
    assert task.worker_id == "w1"
    assert job.state == JOB_STATE_RUNNING

    # Worker 1 reports failure
    result, _ = state.transition_task(
        task.task_id, TASK_STATE_FAILED, now_ms=2000, error="crash"
    )
    assert result == TaskTransitionResult.SHOULD_RETRY
    assert task.state == TASK_STATE_FAILED  # Current attempt failed
    assert len(task.attempts) == 1

    # Task should be back in queue
    pending = state.peek_pending_tasks()
    assert task in pending

    # Dispatch to worker 2 (new attempt)
    state.mark_task_dispatched(task, WorkerId("w2"), now_ms=3000)
    assert task.worker_id == "w2"
    assert len(task.attempts) == 2

    # Worker 2 succeeds
    result, _ = state.transition_task(
        task.task_id, TASK_STATE_SUCCEEDED, now_ms=4000
    )
    assert result == TaskTransitionResult.COMPLETE
    assert task.state == TASK_STATE_SUCCEEDED
    assert job.state == JOB_STATE_SUCCEEDED
```

## Summary

This refactoring achieves:

1. **Clear ownership**: Attempts own execution state, tasks aggregate attempts, jobs aggregate tasks
2. **Correct abstractions**: Jobs don't have workers, tasks delegate to attempts
3. **Simpler mental model**: State flows upward (attempt -> task -> job)
4. **Better retry handling**: Retries are explicit (new attempt), not implicit (field reset)
5. **Cleaner external API**: Properties provide convenient access without exposing internals

The key insight is that **attempts are the unit of execution**, while tasks and jobs are organizational containers that aggregate attempt state.
