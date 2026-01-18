# Implementation Recipe: Stage 5 - Failure Domains & Co-scheduling

This stage focuses on verifying and enhancing the failure domain semantics in Iris. When `max_task_failures` is exceeded, all remaining tasks in the job are killed.

## Purpose

Ensure that the failure domain logic is properly implemented with comprehensive logging and verification. This stage validates that task failures propagate correctly to job-level state and that co-scheduled tasks share failure domains.

## Prerequisites

- Stage 1 (Proto Updates) must be complete
- Stage 2 (Worker Updates) must be complete
- Stage 3 (Controller Updates) must be complete
- Stage 4 (Client Updates) must be complete
- Core failure domain logic in `_update_job_from_tasks()` and `_kill_remaining_tasks()` must be implemented (from Stage 3)

## Stage-Specific Instructions

### Stage: Failure Domains & Co-scheduling

**Objective**: Verify that failure domain semantics work correctly - when `max_task_failures` is exceeded, all remaining tasks are killed. Add enhanced logging for task-level scheduling events.

**Note**: The core failure domain logic is already implemented in Phase 3's `_update_job_from_tasks()` method in `state.py`. This stage adds enhanced logging, verification, and optional dashboard integration.

**Files to Modify**:
- `src/iris/cluster/controller/scheduler.py` - Add task-level logging for timeout and unschedulable events
- `src/iris/cluster/controller/state.py` - Verify ControllerWorker uses `running_tasks`
- `src/iris/cluster/controller/dashboard.py` (optional) - Add task-level UI details
- `src/iris/cluster/worker/dashboard.py` (optional) - Update UI terminology

**Files to Verify**:
- `src/iris/cluster/controller/state.py` - Confirm `_update_job_from_tasks()` and `_kill_remaining_tasks()` are implemented correctly

### Detailed Implementation Steps

#### 5.1 Verify ControllerWorker uses running_tasks

Ensure the `ControllerWorker` dataclass has been renamed from `running_jobs` to `running_tasks`:

```python
# src/iris/cluster/controller/state.py

@dataclass
class ControllerWorker:
    """Controller's view of a worker."""
    worker_id: WorkerId
    address: str
    metadata: cluster_pb2.WorkerMetadata

    healthy: bool = True
    consecutive_failures: int = 0
    last_heartbeat_ms: int = 0

    # Renamed from running_jobs
    running_tasks: set[TaskId] = field(default_factory=set)
```

#### 5.2 Add scheduler logging for task events

Update the scheduler to log when tasks time out or cannot be scheduled:

```python
# src/iris/cluster/controller/scheduler.py

def find_assignments(
    self,
    pending_tasks: list[Task],
    workers: list[ControllerWorker],
    now_ms: int,
) -> SchedulingTransaction:
    """Match pending tasks to available workers."""
    transaction = SchedulingTransaction(self._state)
    capacities = build_capacity_map(self._state, workers)

    for task in pending_tasks:
        job = self._state.get_job(task.job_id)
        if not job:
            continue

        if self._is_task_timed_out(task, job, now_ms):
            transaction.timed_out_tasks.append(task)
            self._state.log_action(
                "task_timeout",
                job_id=task.job_id,
                details=f"task={task.task_id} attempt={task.current_attempt_id}",
            )
            continue

        assigned = False
        for worker in workers:
            if not worker.healthy:
                continue
            capacity = capacities[worker.worker_id]
            if worker_can_fit_task(capacity, job):
                deduct_task_from_capacity(capacity, job)
                transaction.tentatively_assign(task, worker)
                assigned = True
                break

        if not assigned:
            self._state.log_action(
                "task_unschedulable",
                job_id=task.job_id,
                details=f"task={task.task_id} no_worker_has_capacity",
            )

    return transaction
```

#### 5.3 Verify failure domain logic

Confirm that `_update_job_from_tasks()` in `state.py` implements the correct failure domain semantics:

**Expected behavior**:
- Job fails when `failed_permanently > max_task_failures`
- Only `TASK_STATE_FAILED` counts toward `max_task_failures` (not preemptions)
- When threshold is exceeded, `_kill_remaining_tasks()` is called
- All non-finished tasks are marked as `TASK_STATE_KILLED`

**Key code section to verify**:

```python
# src/iris/cluster/controller/state.py

def _update_job_from_tasks(self, job_id: JobId, now_ms: int) -> None:
    """Update job state based on aggregate task states.

    Failure policy: Job fails when task failures exceed max_task_failures.
    - Only counts tasks that exhausted their per-task retries (TASK_STATE_FAILED)
    - Preemptions (TASK_STATE_WORKER_FAILED) do NOT count toward max_task_failures
    - max_task_failures=0 means fail on first task failure (default)
    """
    job = self._jobs.get(job_id)
    if not job:
        return

    tasks = self.get_job_tasks(job_id)
    if not tasks:
        return

    # Count task states
    succeeded = sum(1 for t in tasks if t.state == cluster_pb2.TASK_STATE_SUCCEEDED)
    # Only count actual failures (not preemptions) toward max_task_failures
    failed_permanently = sum(1 for t in tasks if t.state == cluster_pb2.TASK_STATE_FAILED)
    killed_or_unschedulable = sum(1 for t in tasks if t.state in (
        cluster_pb2.TASK_STATE_KILLED,
        cluster_pb2.TASK_STATE_UNSCHEDULABLE,
    ))

    max_task_failures = job.request.max_task_failures  # Default 0

    # Job succeeds when all tasks succeed
    if succeeded == len(tasks):
        job.state = cluster_pb2.JOB_STATE_SUCCEEDED
        job.finished_at_ms = now_ms
    # Job fails when task failures exceed threshold
    elif failed_permanently > max_task_failures:
        job.state = cluster_pb2.JOB_STATE_FAILED
        job.finished_at_ms = now_ms
        for t in tasks:
            if t.error and t.state == cluster_pb2.TASK_STATE_FAILED:
                job.error = t.error
                break
        # Kill remaining running/pending tasks (failure domain)
        self._kill_remaining_tasks(job_id, now_ms, "Job exceeded max_task_failures")
    # Job killed if any task was killed/unschedulable
    elif killed_or_unschedulable > 0:
        job.state = cluster_pb2.JOB_STATE_KILLED
        job.finished_at_ms = now_ms
        for t in tasks:
            if t.error:
                job.error = t.error
                break

def _kill_remaining_tasks(self, job_id: JobId, now_ms: int, error: str) -> None:
    """Kill all non-finished tasks in a job."""
    for task_id in self._tasks_by_job.get(job_id, []):
        task = self._tasks.get(task_id)
        if not task or task.is_finished():
            continue

        task.state = cluster_pb2.TASK_STATE_KILLED
        task.finished_at_ms = now_ms
        task.error = error

        self._task_queue = deque(tid for tid in self._task_queue if tid != task_id)
        if task.worker_id:
            self._unassign_from_worker(task.worker_id, task_id)
        self._remove_endpoints_for_task(task_id)
```

#### 5.4 Dashboard Integration (Optional)

Update dashboards to show task-level details:

**Controller Dashboard** (`src/iris/cluster/controller/dashboard.py`):
1. Job list view: Add column showing task count (e.g., "3/5 tasks running")
2. Job detail view: Show all tasks with their states, worker assignments, and attempt counts
3. Task selector: Allow viewing logs for individual tasks
4. Task attempt history: Show all attempts for a task with timestamps and errors

**Worker Dashboard** (`src/iris/cluster/worker/dashboard.py`):
1. Update terminology from "jobs" to "tasks"
2. Show task ID, job ID, and task index for each running task
3. Display task-level resource usage

### Verification Commands

```bash
# Run all controller tests
uv run pytest tests/cluster/controller/ -v

# Run state tests specifically
uv run pytest tests/cluster/controller/test_state.py -v

# Run scheduler tests
uv run pytest tests/cluster/controller/test_scheduler.py -v

# Run gang and failure domain tests
uv run pytest tests/cluster/controller/test_state.py -v -k "gang or failure"

# Run E2E tests with multi-task jobs
uv run pytest tests/cluster/test_e2e.py -v

# Full test suite
uv run pytest tests/cluster/ -v
```

### Acceptance Criteria

This stage is complete when:

1. **Failure Domain Logic is Verified**:
   - `_update_job_from_tasks()` correctly counts only `TASK_STATE_FAILED` toward `max_task_failures`
   - `_kill_remaining_tasks()` is called when threshold is exceeded
   - All non-finished tasks are marked as `TASK_STATE_KILLED` when failure domain is triggered

2. **Enhanced Logging Works**:
   - Scheduler logs `task_timeout` action when scheduling timeout is exceeded
   - Scheduler logs `task_unschedulable` action when no worker has capacity
   - Logs include task ID and attempt ID for debugging

3. **Tests Pass**:
   - All controller tests pass, including gang and failure domain tests
   - E2E tests with multi-task jobs pass
   - Tests verify that when one task fails beyond retries, remaining tasks are killed

4. **Optional Dashboard Updates** (if implemented):
   - Job list shows task counts and states
   - Job detail view shows individual task status
   - Task logs are accessible via task selector
   - Task attempt history is visible

## Quality Checklist

Before committing, verify all items:

- [ ] All tests pass (`uv run pytest tests/cluster/ -v`)
- [ ] `_update_job_from_tasks()` correctly implements failure domain semantics
- [ ] `_kill_remaining_tasks()` is called when `max_task_failures` is exceeded
- [ ] Scheduler logs `task_timeout` and `task_unschedulable` actions
- [ ] ControllerWorker uses `running_tasks` (not `running_jobs`)
- [ ] Gang failure tests pass
- [ ] No regressions introduced in existing functionality
- [ ] Code follows project conventions (see `AGENTS.md`)
- [ ] Changes are minimal and focused on this stage only

## Troubleshooting

**Failure domain logic doesn't trigger**:
- Check that `_update_job_from_tasks()` is called after task state transitions
- Verify that `max_task_failures` is being read from `job.request.max_task_failures`
- Confirm that only `TASK_STATE_FAILED` counts toward the threshold (not `TASK_STATE_WORKER_FAILED`)

**Tasks aren't killed when threshold is exceeded**:
- Verify `_kill_remaining_tasks()` is called in `_update_job_from_tasks()`
- Check that tasks are properly transitioned to `TASK_STATE_KILLED`
- Confirm that tasks are removed from `_task_queue` and unassigned from workers

**Scheduler logging doesn't appear**:
- Verify `self._state.log_action()` is called in the scheduler
- Check that action log entries are being stored in `_actions` deque
- Confirm the log action names match: `task_timeout` and `task_unschedulable`

**Tests fail with "no such attribute running_tasks"**:
- Ensure all references to `running_jobs` have been renamed to `running_tasks`
- Check both the dataclass definition and all usage sites
- Search codebase for any remaining `running_jobs` references

## E2E Verification

Create an E2E test that validates the complete failure domain behavior:

```python
# tests/cluster/test_e2e.py

def test_failure_domain_kills_remaining_tasks():
    """When one task fails beyond retries, remaining tasks should be killed."""
    # Submit a job with replicas=3, max_task_failures=0
    job_id = client.submit_job(
        entrypoint=lambda: fail_if_task_index(0),  # Only task-0 fails
        resources=ResourceSpec(replicas=3),
        max_task_failures=0,
        max_retries_failure=0,
    )

    # Wait for job to complete
    status = client.wait(job_id, timeout=30)

    # Job should fail
    assert status.state == cluster_pb2.JOB_STATE_FAILED

    # Get all tasks
    tasks = client.list_tasks(job_id)
    assert len(tasks) == 3

    # Task-0 should be FAILED
    task_0 = next(t for t in tasks if t.task_index == 0)
    assert task_0.state == cluster_pb2.TASK_STATE_FAILED

    # Task-1 and Task-2 should be KILLED (failure domain)
    for task in tasks:
        if task.task_index != 0:
            assert task.state == cluster_pb2.TASK_STATE_KILLED
            assert "max_task_failures" in task.error
```

Run this test to confirm the complete failure domain implementation:

```bash
uv run pytest tests/cluster/test_e2e.py::test_failure_domain_kills_remaining_tasks -v
```
