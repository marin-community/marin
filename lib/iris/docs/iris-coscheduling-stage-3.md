# Stage 3: Controller Updates

This stage implements task-level tracking in the Iris controller. The controller will create tasks from jobs, track task state independently, and aggregate task states into job state.

## Purpose

The controller currently operates at a job granularity. This stage extends the controller to:
1. Expand jobs into tasks (1 job with replicas=N creates N tasks)
2. Track each task's state, attempts, and worker assignment independently
3. Aggregate task states to compute job-level state
4. Implement job failure policies based on `max_task_failures`
5. Support task-level retry logic with separate failure and preemption counters

## Prerequisites

Before starting this stage:
- **Stage 1** (Proto Updates) must be complete - TaskState, TaskStatus messages exist
- **Stage 2** (Worker Updates) must be complete - workers can run tasks and report task state
- Proto generation has been run: `uv run python scripts/generate_protos.py`
- Worker tests pass: `uv run pytest tests/cluster/worker/ -v`

## Objective

Controller creates tasks from jobs, tracks task state, aggregates to job state. The controller becomes task-aware while maintaining job-level APIs for clients.

## Design Decisions

Key decisions from the design document:

1. **Task expansion**: Jobs with `replicas=N` expand into N tasks at submission time
2. **Task ID format**: `"{job_id}/task-{index}"` where index is 0-based
3. **Failure policy**: `max_task_failures` (default 0) controls how many task failures cause job failure
   - Preemptions (TASK_STATE_WORKER_FAILED) do NOT count toward `max_task_failures`
   - Only actual failures (TASK_STATE_FAILED) after per-task retries are exhausted count
4. **Failure domains**: When `max_task_failures` is exceeded, all remaining tasks are killed
5. **Endpoint scope**: Endpoints are tracked at task level, cleaned up when tasks finish
6. **Naming**: `ControllerWorker.running_jobs` → `running_tasks`

## Files to Modify

### Core Task Implementation
- `src/iris/cluster/controller/task.py` (new file) - Task and TaskAttempt dataclasses
- `src/iris/cluster/controller/job.py` - Add `expand_job_to_tasks` function

### State Management
- `src/iris/cluster/controller/state.py` - Add task tracking to ControllerState
  - Add `_tasks`, `_tasks_by_job`, `_task_queue`, `_endpoints_by_task`
  - Update `ControllerWorker.running_jobs` → `running_tasks`
  - Add `add_job`, `get_task`, `get_job_tasks`, `peek_pending_tasks`
  - Add `transition_task` with `_update_job_from_tasks`
  - Add `_kill_remaining_tasks` for failure domains
  - Update `get_committed_resources` to use `running_tasks`
  - Update endpoint methods to use `task_id`

### Scheduling
- `src/iris/cluster/controller/scheduler.py` - Schedule tasks instead of jobs
  - Update `find_assignments` to operate on tasks
  - Update `SchedulingTransaction` to track task assignments

### Orchestration
- `src/iris/cluster/controller/controller.py` - Dispatch tasks to workers
  - Update `_run_scheduling` to use `peek_pending_tasks`
  - Add `_dispatch_task` method
  - Update `_apply_schedule_result`

### RPC Service
- `src/iris/cluster/controller/service.py` - Handle task state reports
  - Update `launch_job` to expand job into tasks
  - Add `report_task_state` handler
  - Update `get_job_status` to aggregate task counts
  - Add `get_task_status` and `list_tasks` handlers

## Detailed Implementation Steps

### Step 1: Create task.py with Task and TaskAttempt

Create the controller's representation of tasks with retry logic.

```python
# src/iris/cluster/controller/task.py

from dataclasses import dataclass, field
from enum import Enum

from iris.cluster.types import JobId, TaskId, WorkerId
from iris.rpc import cluster_pb2


class TaskTransitionResult(Enum):
    COMPLETE = "complete"
    SHOULD_RETRY = "should_retry"
    EXCEEDED_RETRY_LIMIT = "exceeded_retry_limit"


@dataclass
class TaskAttempt:
    """Record of a single task execution attempt."""
    attempt_id: int
    worker_id: WorkerId | None = None
    state: int = cluster_pb2.TASK_STATE_PENDING
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    exit_code: int | None = None
    error: str | None = None
    is_worker_failure: bool = False


@dataclass
class Task:
    """Controller's representation of a task within a job.

    Tasks are created when a job is submitted:
    - Job with replicas=1 creates 1 task
    - Job with replicas=N creates N tasks
    """
    task_id: TaskId                    # "{job_id}/task-{index}"
    job_id: JobId                      # Parent job
    task_index: int                    # 0-indexed

    state: int = cluster_pb2.TASK_STATE_PENDING
    worker_id: WorkerId | None = None

    # Retry tracking (per-task)
    failure_count: int = 0
    preemption_count: int = 0
    max_retries_failure: int = 0
    max_retries_preemption: int = 100

    # Attempt tracking
    current_attempt_id: int = 0
    attempts: list[TaskAttempt] = field(default_factory=list)

    # Timestamps
    submitted_at_ms: int = 0
    started_at_ms: int | None = None
    finished_at_ms: int | None = None

    error: str | None = None
    exit_code: int | None = None

    def mark_dispatched(self, worker_id: WorkerId, now_ms: int) -> None:
        self.state = cluster_pb2.TASK_STATE_RUNNING
        self.worker_id = worker_id
        self.started_at_ms = now_ms

    def revert_dispatch(self) -> None:
        self.state = cluster_pb2.TASK_STATE_PENDING
        self.worker_id = None
        self.started_at_ms = None

    def transition(
        self,
        new_state: int,
        now_ms: int,
        *,
        is_worker_failure: bool = False,
        error: str | None = None,
        exit_code: int | None = None,
    ) -> TaskTransitionResult:
        """Transition task state with retry logic."""
        if new_state in (cluster_pb2.TASK_STATE_FAILED, cluster_pb2.TASK_STATE_WORKER_FAILED):
            return self._handle_failure(now_ms, is_worker_failure, error, exit_code)

        if new_state == cluster_pb2.TASK_STATE_SUCCEEDED:
            self.state = new_state
            self.finished_at_ms = now_ms
            self.exit_code = exit_code or 0
            return TaskTransitionResult.COMPLETE

        if new_state == cluster_pb2.TASK_STATE_KILLED:
            self.state = new_state
            self.finished_at_ms = now_ms
            self.error = error
            return TaskTransitionResult.COMPLETE

        if new_state == cluster_pb2.TASK_STATE_UNSCHEDULABLE:
            self.state = new_state
            self.finished_at_ms = now_ms
            self.error = error or "Scheduling timeout exceeded"
            return TaskTransitionResult.COMPLETE

        self.state = new_state
        return TaskTransitionResult.COMPLETE

    def _handle_failure(
        self,
        now_ms: int,
        is_worker_failure: bool,
        error: str | None,
        exit_code: int | None,
    ) -> TaskTransitionResult:
        if is_worker_failure:
            self.preemption_count += 1
            can_retry = self.preemption_count <= self.max_retries_preemption
        else:
            self.failure_count += 1
            can_retry = self.failure_count <= self.max_retries_failure

        if can_retry:
            self._reset_for_retry(is_worker_failure=is_worker_failure)
            return TaskTransitionResult.SHOULD_RETRY
        else:
            self.state = (
                cluster_pb2.TASK_STATE_WORKER_FAILED
                if is_worker_failure
                else cluster_pb2.TASK_STATE_FAILED
            )
            self.finished_at_ms = now_ms
            self.error = error
            self.exit_code = exit_code
            return TaskTransitionResult.EXCEEDED_RETRY_LIMIT

    def _reset_for_retry(self, *, is_worker_failure: bool) -> None:
        if self.started_at_ms is not None:
            self.attempts.append(
                TaskAttempt(
                    attempt_id=self.current_attempt_id,
                    worker_id=self.worker_id,
                    state=self.state,
                    started_at_ms=self.started_at_ms,
                    finished_at_ms=self.finished_at_ms,
                    exit_code=self.exit_code,
                    error=self.error,
                    is_worker_failure=is_worker_failure,
                )
            )

        self.current_attempt_id += 1
        self.state = cluster_pb2.TASK_STATE_PENDING
        self.worker_id = None
        self.started_at_ms = None
        self.finished_at_ms = None
        self.error = None
        self.exit_code = None

    def is_finished(self) -> bool:
        return self.state in (
            cluster_pb2.TASK_STATE_SUCCEEDED,
            cluster_pb2.TASK_STATE_FAILED,
            cluster_pb2.TASK_STATE_KILLED,
            cluster_pb2.TASK_STATE_WORKER_FAILED,
            cluster_pb2.TASK_STATE_UNSCHEDULABLE,
        )
```

### Step 2: Add expand_job_to_tasks to job.py

Add a function to create tasks from a job's replica count.

```python
# src/iris/cluster/controller/job.py

from iris.cluster.controller.task import Task
from iris.cluster.types import TaskId


def expand_job_to_tasks(job: Job, now_ms: int) -> list[Task]:
    """Expand a job into its constituent tasks based on replicas."""
    num_replicas = job.request.resources.replicas or 1
    tasks = []

    for i in range(num_replicas):
        task_id = TaskId(f"{job.job_id}/task-{i}")
        task = Task(
            task_id=task_id,
            job_id=job.job_id,
            task_index=i,
            max_retries_failure=job.max_retries_failure,
            max_retries_preemption=job.max_retries_preemption,
            submitted_at_ms=now_ms,
        )
        tasks.append(task)

    return tasks
```

### Step 3: Update state.py - Add task tracking

First, update `ControllerWorker`:

```python
# src/iris/cluster/controller/state.py

@dataclass
class ControllerWorker:
    worker_id: WorkerId
    address: str
    metadata: cluster_pb2.WorkerMetadata
    healthy: bool = True
    consecutive_failures: int = 0
    last_heartbeat_ms: int = 0
    running_tasks: set[TaskId] = field(default_factory=set)  # Renamed from running_jobs
```

Then update `ControllerState` to add task tracking:

```python
# src/iris/cluster/controller/state.py

from iris.cluster.controller.task import Task, TaskTransitionResult


class ControllerState:
    def __init__(self):
        self._lock = RLock()
        self._jobs: dict[JobId, Job] = {}
        self._tasks: dict[TaskId, Task] = {}              # NEW
        self._tasks_by_job: dict[JobId, list[TaskId]] = {}  # NEW
        self._workers: dict[WorkerId, ControllerWorker] = {}
        self._task_queue: deque[TaskId] = deque()         # Renamed from _queue
        self._gangs: dict[str, set[JobId]] = {}
        self._actions: deque[ActionLogEntry] = deque(maxlen=100)
        self._endpoints: dict[str, ControllerEndpoint] = {}
        self._endpoints_by_task: dict[TaskId, set[str]] = {}  # Changed from job

    def add_job(self, job: Job, tasks: list[Task]) -> None:
        """Add a job and its tasks to state."""
        with self._lock:
            self._jobs[job.job_id] = job
            self._tasks_by_job[job.job_id] = []

            for task in tasks:
                self._tasks[task.task_id] = task
                self._tasks_by_job[job.job_id].append(task.task_id)
                self._task_queue.append(task.task_id)

            if job.gang_id:
                self._gangs.setdefault(job.gang_id, set()).add(job.job_id)

    def get_task(self, task_id: TaskId) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def get_job_tasks(self, job_id: JobId) -> list[Task]:
        with self._lock:
            task_ids = self._tasks_by_job.get(job_id, [])
            return [self._tasks[tid] for tid in task_ids if tid in self._tasks]

    def peek_pending_tasks(self) -> list[Task]:
        """Return all PENDING tasks in queue order."""
        with self._lock:
            pending = []
            for task_id in self._task_queue:
                task = self._tasks.get(task_id)
                if task and task.state == cluster_pb2.TASK_STATE_PENDING:
                    pending.append(task)
            return pending

    def assign_task_to_worker(self, worker_id: WorkerId, task_id: TaskId) -> bool:
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False
            worker.running_tasks.add(task_id)
            self._task_queue = deque(tid for tid in self._task_queue if tid != task_id)
            return True

    def _unassign_from_worker(self, worker_id: WorkerId, task_id: TaskId) -> None:
        """Remove task from worker's running set."""
        worker = self._workers.get(worker_id)
        if worker:
            worker.running_tasks.discard(task_id)

    def transition_task(
        self,
        task_id: TaskId,
        new_state: int,
        now_ms: int,
        *,
        is_worker_failure: bool = False,
        error: str | None = None,
        exit_code: int | None = None,
    ) -> tuple[TaskTransitionResult, list[ControllerEndpoint]]:
        """Transition a task and handle side effects."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return TaskTransitionResult.COMPLETE, []

            worker_id = task.worker_id
            result = task.transition(
                new_state, now_ms,
                is_worker_failure=is_worker_failure,
                error=error,
                exit_code=exit_code,
            )

            removed_endpoints: list[ControllerEndpoint] = []
            if result == TaskTransitionResult.SHOULD_RETRY:
                if task.task_id not in self._task_queue:
                    self._task_queue.append(task.task_id)
                if worker_id:
                    worker = self._workers.get(worker_id)
                    if worker:
                        worker.running_tasks.discard(task_id)
            elif task.is_finished():
                self._task_queue = deque(tid for tid in self._task_queue if tid != task_id)
                removed_endpoints = self._remove_endpoints_for_task(task_id)
                if worker_id:
                    worker = self._workers.get(worker_id)
                    if worker:
                        worker.running_tasks.discard(task_id)

                # Update job state based on task states
                self._update_job_from_tasks(task.job_id, now_ms)

            return result, removed_endpoints

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

    # --- Resource tracking (update to use running_tasks) ---

    def get_committed_resources(self, worker: ControllerWorker) -> tuple[int, int, int]:
        """Compute resources committed to running tasks on this worker."""
        from iris.cluster.controller.scheduler import get_gpu_count

        with self._lock:
            cpu = 0
            memory = 0
            gpu = 0

            for task_id in worker.running_tasks:  # Changed from running_jobs
                task = self._tasks.get(task_id)
                if task:
                    job = self._jobs.get(task.job_id)
                    if job:
                        resources = job.request.resources
                        cpu += resources.cpu
                        memory += resources.memory_bytes
                        gpu += get_gpu_count(resources.device)

            return cpu, memory, gpu

    # Endpoint methods updated to use task_id
    def add_endpoint(self, endpoint: ControllerEndpoint, task_id: TaskId) -> None:
        with self._lock:
            self._endpoints[endpoint.endpoint_id] = endpoint
            self._endpoints_by_task.setdefault(task_id, set()).add(endpoint.endpoint_id)

    def _remove_endpoints_for_task(self, task_id: TaskId) -> list[ControllerEndpoint]:
        endpoint_ids = list(self._endpoints_by_task.get(task_id, []))
        removed = []
        for eid in endpoint_ids:
            endpoint = self._endpoints.pop(eid, None)
            if endpoint:
                removed.append(endpoint)
        self._endpoints_by_task.pop(task_id, None)
        return removed
```

### Step 4: Update scheduler.py - Schedule tasks

Update the scheduler to operate on tasks instead of jobs:

```python
# src/iris/cluster/controller/scheduler.py

class Scheduler:
    def find_assignments(
        self,
        pending_tasks: list[Task],          # Changed from pending_jobs
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

    def _is_task_timed_out(self, task: Task, job: Job, now_ms: int) -> bool:
        timeout_seconds = job.request.scheduling_timeout_seconds
        if timeout_seconds <= 0:
            return False
        pending_duration_ms = now_ms - task.submitted_at_ms
        return pending_duration_ms > timeout_seconds * 1000


@dataclass
class SchedulingTransaction:
    state: ControllerState
    assignments: list[tuple[Task, ControllerWorker]] = field(default_factory=list)
    timed_out_tasks: list[Task] = field(default_factory=list)

    def tentatively_assign(self, task: Task, worker: ControllerWorker) -> None:
        worker.running_tasks.add(task.task_id)
        self.assignments.append((task, worker))

    def rollback_assignment(self, task: Task, worker: ControllerWorker) -> None:
        worker.running_tasks.discard(task.task_id)
```

### Step 5: Update controller.py - Dispatch tasks

Update the controller to dispatch tasks to workers:

```python
# src/iris/cluster/controller/controller.py

class Controller:
    def _run_scheduling(self) -> None:
        pending_tasks = self._state.peek_pending_tasks()
        workers = self._state.get_available_workers()
        now_ms = int(time.time() * 1000)

        result = self._scheduler.find_assignments(pending_tasks, workers, now_ms)
        self._apply_schedule_result(result, now_ms)

    def _apply_schedule_result(
        self,
        result: SchedulingTransaction,
        now_ms: int,
    ) -> None:
        # Handle timed out tasks
        for task in result.timed_out_tasks:
            self._state.transition_task(
                task.task_id,
                cluster_pb2.TASK_STATE_UNSCHEDULABLE,
                now_ms,
                error=f"Scheduling timeout exceeded",
            )

        # Dispatch assigned tasks
        for task, worker in result.assignments:
            self._dispatch_task(result, task, worker, now_ms)

    def _dispatch_task(
        self,
        transaction: SchedulingTransaction,
        task: Task,
        worker: ControllerWorker,
        now_ms: int,
    ) -> None:
        job = self._state.get_job(task.job_id)
        if not job:
            return

        task.mark_dispatched(worker.worker_id, now_ms)

        try:
            stub = self._stub_factory.get_stub(worker.address)
            request = cluster_pb2.Worker.RunTaskRequest(
                job_id=str(task.job_id),
                task_id=str(task.task_id),
                task_index=task.task_index,
                num_tasks=len(self._state.get_job_tasks(task.job_id)),
                serialized_entrypoint=job.request.serialized_entrypoint,
                environment=job.request.environment,
                bundle_gcs_path=job.request.bundle_gcs_path,
                resources=job.request.resources,
                ports=list(job.request.ports),
                attempt_id=task.current_attempt_id,
            )
            stub.run_task(request)
            self._state.assign_task_to_worker(worker.worker_id, task.task_id)
            self._state.log_action(
                "task_dispatched",
                job_id=task.job_id,
                worker_id=worker.worker_id,
                details=f"task={task.task_id}",
            )
        except Exception as e:
            task.revert_dispatch()
            transaction.rollback_assignment(task, worker)
            logger.warning(f"Failed to dispatch task {task.task_id}: {e}")
```

### Step 6: Update service.py - Handle task reports

Update the RPC service to expand jobs into tasks and handle task state reports:

```python
# src/iris/cluster/controller/service.py

from iris.cluster.controller.job import expand_job_to_tasks


class ControllerServiceImpl:
    def launch_job(
        self,
        request: cluster_pb2.Controller.LaunchJobRequest,
        ctx: grpc.ServicerContext,
    ) -> cluster_pb2.Controller.LaunchJobResponse:
        now_ms = int(time.time() * 1000)
        job_id = JobId(request.name)

        # Create job
        job = Job(
            job_id=job_id,
            request=request,
            submitted_at_ms=now_ms,
            parent_job_id=JobId(request.parent_job_id) if request.parent_job_id else None,
        )

        # Expand into tasks
        tasks = expand_job_to_tasks(job, now_ms)

        # Add to state
        self._state.add_job(job, tasks)
        self._scheduler.wake()

        return cluster_pb2.Controller.LaunchJobResponse(job_id=str(job_id))

    def report_task_state(
        self,
        request: cluster_pb2.Controller.ReportTaskStateRequest,
        ctx: grpc.ServicerContext,
    ) -> cluster_pb2.Controller.ReportTaskStateResponse:
        task_id = TaskId(request.task_id)
        task = self._state.get_task(task_id)

        if not task:
            return cluster_pb2.Controller.ReportTaskStateResponse()

        # Only process if this is the current attempt
        if request.attempt_id != task.current_attempt_id:
            return cluster_pb2.Controller.ReportTaskStateResponse()

        now_ms = int(time.time() * 1000)
        is_worker_failure = request.state == cluster_pb2.TASK_STATE_WORKER_FAILED

        self._state.transition_task(
            task_id,
            request.state,
            now_ms,
            is_worker_failure=is_worker_failure,
            error=request.error or None,
            exit_code=request.exit_code,
        )

        return cluster_pb2.Controller.ReportTaskStateResponse()

    def get_job_status(
        self,
        request: cluster_pb2.Controller.GetJobStatusRequest,
        ctx: grpc.ServicerContext,
    ) -> cluster_pb2.Controller.GetJobStatusResponse:
        job_id = JobId(request.job_id)
        job = self._state.get_job(job_id)

        if not job:
            ctx.abort(grpc.StatusCode.NOT_FOUND, f"Job not found: {job_id}")

        tasks = self._state.get_job_tasks(job_id)

        # Build task status list if requested
        task_statuses = []
        if request.include_tasks:
            for task in tasks:
                task_statuses.append(self._task_to_proto(task))

        # Aggregate task counts
        pending = sum(1 for t in tasks if t.state == cluster_pb2.TASK_STATE_PENDING)
        running = sum(1 for t in tasks if t.state == cluster_pb2.TASK_STATE_RUNNING)
        succeeded = sum(1 for t in tasks if t.state == cluster_pb2.TASK_STATE_SUCCEEDED)
        failed = sum(1 for t in tasks if t.state in (
            cluster_pb2.TASK_STATE_FAILED,
            cluster_pb2.TASK_STATE_WORKER_FAILED,
        ))

        return cluster_pb2.Controller.GetJobStatusResponse(
            job=cluster_pb2.JobStatus(
                job_id=str(job.job_id),
                state=job.state,
                # ... other existing fields ...
                num_tasks=len(tasks),
                tasks_pending=pending,
                tasks_running=running,
                tasks_succeeded=succeeded,
                tasks_failed=failed,
                tasks=task_statuses if request.include_tasks else [],
            )
        )

    def get_task_status(
        self,
        request: cluster_pb2.Controller.GetTaskStatusRequest,
        ctx: grpc.ServicerContext,
    ) -> cluster_pb2.Controller.GetTaskStatusResponse:
        job_id = JobId(request.job_id)
        tasks = self._state.get_job_tasks(job_id)

        for task in tasks:
            if task.task_index == request.task_index:
                return cluster_pb2.Controller.GetTaskStatusResponse(
                    task=self._task_to_proto(task)
                )

        ctx.abort(grpc.StatusCode.NOT_FOUND, f"Task not found: {job_id}/task-{request.task_index}")

    def _task_to_proto(self, task: Task) -> cluster_pb2.TaskStatus:
        """Convert Task to TaskStatus proto."""
        return cluster_pb2.TaskStatus(
            task_id=str(task.task_id),
            job_id=str(task.job_id),
            task_index=task.task_index,
            state=task.state,
            worker_id=str(task.worker_id) if task.worker_id else "",
            exit_code=task.exit_code or 0,
            error=task.error or "",
            started_at_ms=task.started_at_ms or 0,
            finished_at_ms=task.finished_at_ms or 0,
            current_attempt_id=task.current_attempt_id,
            attempts=[
                cluster_pb2.TaskAttempt(
                    attempt_id=a.attempt_id,
                    worker_id=str(a.worker_id) if a.worker_id else "",
                    state=a.state,
                    exit_code=a.exit_code or 0,
                    error=a.error or "",
                    started_at_ms=a.started_at_ms or 0,
                    finished_at_ms=a.finished_at_ms or 0,
                    is_worker_failure=a.is_worker_failure,
                )
                for a in task.attempts
            ],
        )
```

## Verification Commands

Run tests to verify the implementation:

```bash
# Run controller state tests
uv run pytest tests/cluster/controller/test_state.py -v

# Run controller scheduler tests
uv run pytest tests/cluster/controller/test_scheduler.py -v

# Run controller service tests
uv run pytest tests/cluster/controller/test_service.py -v

# Run job expansion tests
uv run pytest tests/cluster/controller/test_job.py -v

# Run all controller tests
uv run pytest tests/cluster/controller/ -v

# Run E2E tests
uv run pytest tests/cluster/test_e2e.py -v
```

## Acceptance Criteria

This stage is complete when:

1. **Task creation**: Jobs expand into tasks correctly based on `replicas` field
2. **Task tracking**: Controller tracks each task's state independently
3. **Task scheduling**: Scheduler assigns tasks to workers (not jobs)
4. **Task dispatch**: Controller dispatches tasks to workers via `RunTaskRequest`
5. **Task state reporting**: Workers report task state, controller updates task
6. **Job aggregation**: Job state is correctly computed from task states
7. **Failure policy**: `max_task_failures` correctly triggers job failure
8. **Failure domains**: When job fails, remaining tasks are killed
9. **Retry logic**: Tasks retry on failure/preemption according to per-task limits
10. **Endpoint cleanup**: Task endpoints are removed when tasks finish
11. **Resource tracking**: `get_committed_resources` uses `running_tasks`
12. **All tests pass**: Controller test suite passes without regressions

## Dependencies

This stage depends on:
- **Stage 1 (Proto Updates)**: TaskState, TaskStatus, RunTaskRequest, ReportTaskStateRequest
- **Stage 2 (Worker Updates)**: Workers can run tasks and report task state

## Quality Checklist

Before committing, verify:

- [ ] All tests pass: `uv run pytest tests/cluster/controller/ -v`
- [ ] Task expansion works for replicas=1 and replicas>1
- [ ] Task state transitions are correct (PENDING → RUNNING → SUCCEEDED/FAILED)
- [ ] Task retry logic works (failures vs preemptions counted separately)
- [ ] Job state aggregation works (all tasks succeed → job succeeds)
- [ ] Failure policy works (max_task_failures triggers job failure)
- [ ] Failure domains work (job failure kills remaining tasks)
- [ ] Endpoint cleanup happens when tasks finish (not when job finishes)
- [ ] No regressions in existing controller functionality
- [ ] Code follows project conventions (see `AGENTS.md`)
- [ ] Type hints are correct and pass `uv run mypy`
- [ ] Pre-commit hooks pass: `uv run python infra/pre-commit.py --all-files`

## Troubleshooting

**Tests fail with "Task not found"**:
- Verify `add_job` is adding tasks to `_tasks` and `_tasks_by_job`
- Check that task IDs are formatted correctly: `"{job_id}/task-{index}"`

**Job never completes**:
- Verify `_update_job_from_tasks` is called after task transitions
- Check that task state counts are correct (succeeded, failed, etc.)
- Ensure all tasks are finishing (check for stuck PENDING or RUNNING tasks)

**Failure domains not working**:
- Verify `_kill_remaining_tasks` is called when `max_task_failures` is exceeded
- Check that killed tasks are removed from `_task_queue` and worker `running_tasks`

**Resource tracking incorrect**:
- Verify `get_committed_resources` iterates over `running_tasks` not `running_jobs`
- Check that task assignments update worker `running_tasks` set

**Worker can't report task state**:
- Verify Stage 2 (Worker Updates) is complete
- Check that workers are calling `report_task_state` with correct task_id
- Verify attempt_id matching logic in `report_task_state`
