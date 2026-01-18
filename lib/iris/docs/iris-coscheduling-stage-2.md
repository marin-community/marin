# Stage 2: Worker Updates

## Purpose

This stage updates the Iris worker implementation to operate on tasks instead of jobs. Workers will track individual tasks (which are units of execution within a job), manage task lifecycles, and report task state to the controller.

## Prerequisites

Before implementing this stage, ensure:

- **Stage 1 (Proto Updates) is complete**: The proto definitions must include `TaskState`, `TaskStatus`, `TaskAttempt`, and all Worker RPCs renamed from Job to Task terminology
- Protos have been regenerated: `uv run python scripts/generate_protos.py`
- Proto imports work: `uv run python -c "from iris.rpc import cluster_pb2; print(cluster_pb2.TaskState.Name(1))"`

## Objective

Transform the worker layer to:
- Accept and execute individual tasks (not jobs)
- Track task state with task IDs like `{job_id}/task-{index}`
- Provide environment variables for task identification: `IRIS_TASK_ID`, `IRIS_TASK_INDEX`, `IRIS_NUM_TASKS`
- Report task state to the controller via `ReportTaskStateRequest`

## Files to Modify

1. **`src/iris/cluster/worker/worker_types.py`**
   - Rename `Job` class to `Task`
   - Update all field names and types to use task terminology
   - Add `TaskLogs` class
   - Add `is_task_finished` helper function

2. **`src/iris/cluster/worker/worker.py`**
   - Rename `_jobs` to `_tasks` (dict mapping task_id to Task)
   - Update method names: `submit_job` → `submit_task`, etc.
   - Update `_build_iris_env` to set task-specific environment variables
   - Rename `_execute_job` → `_execute_task`
   - Update `_report_task_state` to use new proto message

3. **`src/iris/cluster/worker/service.py`**
   - Update `TaskProvider` protocol with task methods
   - Rename RPC handlers: `run_job` → `run_task`, etc.
   - Update all request/response types

4. **`src/iris/cluster/worker/dashboard.py`**
   - Update UI to display tasks instead of jobs
   - Update templates to use task terminology

5. **`src/iris/cluster/types.py`**
   - Add `is_task_finished` helper
   - Add `TaskId` type alias

## Detailed Implementation Steps

### Step 1: Update worker_types.py

Rename the `Job` class to `Task` and update all fields to use task terminology.

```python
# src/iris/cluster/worker/worker_types.py

from dataclasses import dataclass, field
from pathlib import Path
import threading
import time

from iris.rpc import cluster_pb2


@dataclass(kw_only=True)
class Task:
    """Internal worker representation of a running task."""
    task_id: str                      # Full task ID: "{job_id}/task-{index}"
    job_id: str                       # Parent job ID
    task_index: int = 0               # 0-indexed
    num_tasks: int = 1                # Total tasks in job
    attempt_id: int = 0
    request: cluster_pb2.Worker.RunTaskRequest
    status: cluster_pb2.TaskState = cluster_pb2.TASK_STATE_PENDING
    exit_code: int | None = None
    error: str | None = None
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    ports: dict[str, int] = field(default_factory=dict)
    status_message: str = ""

    # Resource tracking (unchanged)
    current_memory_mb: int = 0
    peak_memory_mb: int = 0
    current_cpu_percent: int = 0
    process_count: int = 0
    disk_mb: int = 0

    # Build tracking (unchanged)
    build_started_ms: int | None = None
    build_finished_ms: int | None = None
    build_from_cache: bool = False
    image_tag: str = ""

    # Internals (unchanged)
    container_id: str | None = None
    workdir: Path | None = None
    thread: threading.Thread | None = None
    cleanup_done: bool = False
    should_stop: bool = False
    logs: "TaskLogs" = field(default_factory=lambda: TaskLogs())
    result: bytes | None = None

    def transition_to(
        self,
        state: cluster_pb2.TaskState,
        *,
        message: str = "",
        error: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        self.status = state
        self.status_message = message
        if is_task_finished(state):
            self.finished_at_ms = int(time.time() * 1000)
            if error:
                self.error = error
            if exit_code is not None:
                self.exit_code = exit_code

    def to_proto(self) -> cluster_pb2.TaskStatus:
        return cluster_pb2.TaskStatus(
            task_id=self.task_id,
            job_id=self.job_id,
            task_index=self.task_index,
            state=self.status,
            exit_code=self.exit_code or 0,
            error=self.error or "",
            started_at_ms=self.started_at_ms or 0,
            finished_at_ms=self.finished_at_ms or 0,
            ports=self.ports,
            current_attempt_id=self.attempt_id,
            resource_usage=cluster_pb2.ResourceUsage(
                current_memory_mb=self.current_memory_mb,
                peak_memory_mb=self.peak_memory_mb,
                current_cpu_percent=self.current_cpu_percent,
                process_count=self.process_count,
                disk_mb=self.disk_mb,
            ),
            build_metrics=cluster_pb2.BuildMetrics(
                build_started_ms=self.build_started_ms or 0,
                build_finished_ms=self.build_finished_ms or 0,
                build_from_cache=self.build_from_cache,
                image_tag=self.image_tag,
            ),
        )


def is_task_finished(state: cluster_pb2.TaskState) -> bool:
    """Check if a task state is terminal."""
    return state in (
        cluster_pb2.TASK_STATE_SUCCEEDED,
        cluster_pb2.TASK_STATE_FAILED,
        cluster_pb2.TASK_STATE_KILLED,
        cluster_pb2.TASK_STATE_WORKER_FAILED,
        cluster_pb2.TASK_STATE_UNSCHEDULABLE,
    )


class TaskLogs:
    """Log storage for a task."""
    def __init__(self):
        self.lines: list[tuple[str, str, int]] = []  # (source, data, timestamp_ms)

    def add(self, source: str, data: str) -> None:
        timestamp_ms = int(time.time() * 1000)
        self.lines.append((source, data, timestamp_ms))

    def get_lines(self, start_line: int = 0) -> list[cluster_pb2.Worker.LogEntry]:
        entries = []
        for i, (source, data, timestamp_ms) in enumerate(self.lines[start_line:]):
            entries.append(
                cluster_pb2.Worker.LogEntry(
                    line_number=start_line + i,
                    source=source,
                    data=data,
                    timestamp_ms=timestamp_ms,
                )
            )
        return entries
```

### Step 2: Update worker.py

Update the `Worker` class to use task terminology throughout.

```python
# src/iris/cluster/worker/worker.py

class Worker:
    def __init__(self, config: WorkerConfig):
        # Renamed from _jobs
        self._tasks: dict[str, Task] = {}  # task_id -> Task
        self._lock = threading.RLock()
        # ... rest unchanged

    def submit_task(self, request: cluster_pb2.Worker.RunTaskRequest) -> str:
        """Submit a new task for execution."""
        task_id = request.task_id

        with self._lock:
            if task_id in self._tasks:
                raise ValueError(f"Task {task_id} already exists")

            task = Task(
                task_id=task_id,
                job_id=request.job_id,
                task_index=request.task_index,
                num_tasks=request.num_tasks,
                attempt_id=request.attempt_id,
                request=request,
            )
            self._tasks[task_id] = task

        # Start execution thread
        thread = threading.Thread(
            target=self._execute_task,
            args=(task,),
            daemon=True,
        )
        task.thread = thread
        thread.start()

        return task_id

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> list[Task]:
        """List all tasks."""
        with self._lock:
            return list(self._tasks.values())

    def kill_task(self, task_id: str, term_timeout_ms: int = 5000) -> bool:
        """Kill a running task."""
        task = self.get_task(task_id)
        if not task:
            return False
        task.should_stop = True
        # ... container kill logic unchanged
        return True

    def get_logs(self, task_id: str, start_line: int = 0) -> list[cluster_pb2.Worker.LogEntry]:
        """Get logs for a task."""
        task = self.get_task(task_id)
        if not task:
            return []
        return task.logs.get_lines(start_line)

    def _execute_task(self, task: Task) -> None:
        """Execute a task (renamed from _execute_job)."""
        try:
            task.transition_to(cluster_pb2.TASK_STATE_BUILDING, message="starting")
            task.started_at_ms = int(time.time() * 1000)
            # ... execution logic unchanged, just use Task instead of Job
        except Exception as e:
            task.transition_to(
                cluster_pb2.TASK_STATE_FAILED,
                error=format_exception_with_traceback(e),
            )
        finally:
            self._report_task_state(task)

    def _build_iris_env(self, task: Task) -> dict[str, str]:
        """Build Iris environment variables for the task."""
        env = {}

        # Task identification
        env["IRIS_JOB_ID"] = task.job_id
        env["IRIS_TASK_ID"] = task.task_id
        env["IRIS_TASK_INDEX"] = str(task.task_index)
        env["IRIS_NUM_TASKS"] = str(task.num_tasks)
        env["IRIS_ATTEMPT_ID"] = str(task.attempt_id)

        if self._config.worker_id:
            env["IRIS_WORKER_ID"] = self._config.worker_id

        if self._config.controller_address:
            env["IRIS_CONTROLLER_ADDRESS"] = _rewrite_address_for_container(
                self._config.controller_address
            )

        if task.request.bundle_gcs_path:
            env["IRIS_BUNDLE_GCS_PATH"] = task.request.bundle_gcs_path

        env["IRIS_BIND_HOST"] = "0.0.0.0" if isinstance(self._runtime, DockerRuntime) else "127.0.0.1"

        for name, port in task.ports.items():
            env[f"IRIS_PORT_{name.upper()}"] = str(port)

        return env

    def _report_task_state(self, task: Task) -> None:
        """Report task state to controller."""
        if not self._controller_client:
            return

        request = cluster_pb2.Controller.ReportTaskStateRequest(
            worker_id=self._worker_id,
            task_id=task.task_id,
            job_id=task.job_id,
            task_index=task.task_index,
            state=task.status,
            exit_code=task.exit_code or 0,
            error=task.error or "",
            finished_at_ms=task.finished_at_ms or 0,
            attempt_id=task.attempt_id,
        )
        try:
            self._controller_client.report_task_state(request)
        except Exception as e:
            logger.warning(f"Failed to report task state: {e}")
```

### Step 3: Update service.py

Update the RPC service layer to use task terminology.

```python
# src/iris/cluster/worker/service.py

from typing import Protocol
import grpc

from iris.rpc import cluster_pb2
from iris.cluster.worker.worker_types import Task


class TaskProvider(Protocol):
    """Protocol that Worker implements for the RPC service."""

    def submit_task(self, request: cluster_pb2.Worker.RunTaskRequest) -> str: ...
    def get_task(self, task_id: str) -> Task | None: ...
    def list_tasks(self) -> list[Task]: ...
    def kill_task(self, task_id: str, term_timeout_ms: int = 5000) -> bool: ...
    def get_logs(self, task_id: str, start_line: int = 0) -> list[cluster_pb2.Worker.LogEntry]: ...


class WorkerServiceImpl:
    """gRPC service implementation for Worker."""

    def __init__(self, provider: TaskProvider):
        self._provider = provider

    def run_task(
        self,
        request: cluster_pb2.Worker.RunTaskRequest,
        ctx: grpc.ServicerContext,
    ) -> cluster_pb2.Worker.RunTaskResponse:
        """Start execution of a task."""
        task_id = self._provider.submit_task(request)
        return cluster_pb2.Worker.RunTaskResponse(
            task_id=task_id,
            state=cluster_pb2.TASK_STATE_PENDING,
        )

    def get_task_status(
        self,
        request: cluster_pb2.Worker.GetTaskStatusRequest,
        ctx: grpc.ServicerContext,
    ) -> cluster_pb2.TaskStatus:
        """Get status of a task."""
        task = self._provider.get_task(request.task_id)
        if not task:
            ctx.abort(grpc.StatusCode.NOT_FOUND, f"Task not found: {request.task_id}")
        return task.to_proto()

    def list_tasks(
        self,
        request: cluster_pb2.Worker.ListTasksRequest,
        ctx: grpc.ServicerContext,
    ) -> cluster_pb2.Worker.ListTasksResponse:
        """List all tasks on this worker."""
        tasks = self._provider.list_tasks()
        return cluster_pb2.Worker.ListTasksResponse(
            tasks=[t.to_proto() for t in tasks]
        )

    def fetch_task_logs(
        self,
        request: cluster_pb2.Worker.FetchTaskLogsRequest,
        ctx: grpc.ServicerContext,
    ) -> cluster_pb2.Worker.FetchTaskLogsResponse:
        """Fetch logs for a task."""
        logs = self._provider.get_logs(
            request.task_id,
            start_line=int(request.filter.start_line) if request.filter else 0,
        )
        return cluster_pb2.Worker.FetchTaskLogsResponse(logs=logs)

    def kill_task(
        self,
        request: cluster_pb2.Worker.KillTaskRequest,
        ctx: grpc.ServicerContext,
    ) -> cluster_pb2.Empty:
        """Kill a running task."""
        success = self._provider.kill_task(request.task_id, request.term_timeout_ms)
        if not success:
            ctx.abort(grpc.StatusCode.NOT_FOUND, f"Task not found: {request.task_id}")
        return cluster_pb2.Empty()

    def health_check(
        self,
        request: cluster_pb2.Empty,
        ctx: grpc.ServicerContext,
    ) -> cluster_pb2.Worker.HealthResponse:
        """Report worker health."""
        tasks = self._provider.list_tasks()
        running_count = sum(1 for t in tasks if t.status == cluster_pb2.TASK_STATE_RUNNING)
        return cluster_pb2.Worker.HealthResponse(
            healthy=True,
            uptime_ms=0,  # TODO: track uptime
            running_tasks=running_count,
        )
```

### Step 4: Update types.py

Add task state helper functions.

```python
# src/iris/cluster/types.py

from typing import NewType
from iris.rpc import cluster_pb2


# Type aliases
TaskId = NewType("TaskId", str)


def is_task_finished(state: int) -> bool:
    """Check if a task state is terminal."""
    return state in (
        cluster_pb2.TASK_STATE_SUCCEEDED,
        cluster_pb2.TASK_STATE_FAILED,
        cluster_pb2.TASK_STATE_KILLED,
        cluster_pb2.TASK_STATE_WORKER_FAILED,
        cluster_pb2.TASK_STATE_UNSCHEDULABLE,
    )
```

### Step 5: Update dashboard.py (worker)

Update the worker dashboard to display tasks instead of jobs.

```python
# src/iris/cluster/worker/dashboard.py

# Update all references from "job" to "task"
# Update templates to show task_id, task_index, num_tasks
# Update log fetching to use task_id
# This is primarily UI work - update HTML templates and route handlers
```

## Verification Commands

After implementing each step, run the following commands to verify correctness:

```bash
# Run worker unit tests
uv run pytest tests/cluster/worker/test_worker.py -v

# Run worker service tests
uv run pytest tests/cluster/worker/test_service.py -v

# Run all worker tests
uv run pytest tests/cluster/worker/ -v

# Verify worker can start and accept tasks
uv run python -c "
from iris.cluster.worker.worker import Worker, WorkerConfig
w = Worker(WorkerConfig())
w.start()
print('Worker started successfully')
w.stop()
"
```

## Acceptance Criteria

This stage is complete when:

1. All worker types use task terminology (Task, TaskLogs, is_task_finished)
2. Worker class methods are renamed: `submit_task`, `get_task`, `list_tasks`, `kill_task`
3. Environment variables are set correctly: `IRIS_TASK_ID`, `IRIS_TASK_INDEX`, `IRIS_NUM_TASKS`
4. Worker reports task state using `ReportTaskStateRequest`
5. RPC service implements task-based handlers: `run_task`, `get_task_status`, `list_tasks`, `fetch_task_logs`, `kill_task`
6. All worker tests pass
7. No references to "job" remain in worker layer (except for `job_id` field which references parent job)

## Quality Checklist

Before marking this stage complete:

- [ ] All tests pass: `uv run pytest tests/cluster/worker/ -v`
- [ ] No regressions in existing worker functionality
- [ ] Code follows project conventions (see `AGENTS.md`)
- [ ] Changes are minimal and focused on this stage
- [ ] All task methods have appropriate test coverage
- [ ] Environment variable setting is tested
- [ ] Task state reporting is tested
- [ ] Commit message clearly describes the change

## Dependencies

This stage depends on:
- **Stage 1 (Proto Updates)**: All proto changes must be complete and generated

## Next Stage

After completing this stage, proceed to:
- **Stage 3 (Controller Updates)**: Update controller to create tasks from jobs and track task state

## Troubleshooting

**Import errors for TaskState or TaskStatus**:
- Verify Stage 1 is complete: `uv run python -c "from iris.rpc import cluster_pb2; print(cluster_pb2.TaskState)"`
- Re-run proto generation: `uv run python scripts/generate_protos.py`

**Tests fail with missing attributes**:
- Check that all references to `Job` are renamed to `Task` in worker_types.py
- Verify `_jobs` dict is renamed to `_tasks` in worker.py
- Update test fixtures to use task terminology

**RPC handlers fail**:
- Verify TaskProvider protocol matches Worker implementation
- Check that all RPC methods use new proto message types (RunTaskRequest, etc.)
- Ensure service registration uses correct method names
