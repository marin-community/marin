# Stage 4: Client Updates - Co-scheduling Implementation

This document describes Stage 4 of the Iris co-scheduling feature implementation: updating the cluster client to support replicas and expose task-level information.

## Purpose

Stage 4 extends the Iris client APIs to expose task-level information to users. After the controller has been updated to create and track tasks (Stage 3), clients need to be able to query individual task status, fetch task-specific logs, and understand their task context when running inside a job.

## Prerequisites

The following stages must be completed before starting Stage 4:

- **Stage 1**: Proto Updates - TaskState, TaskStatus messages defined
- **Stage 2**: Worker Updates - Workers operate on tasks instead of jobs
- **Stage 3**: Controller Updates - Controller creates tasks from jobs, tracks task state

Verify these are complete by checking:
```bash
# Verify protos have task support
uv run python -c "from iris.rpc import cluster_pb2; print(cluster_pb2.TaskState.Name(1))"

# Verify controller has task methods
uv run python -c "from iris.cluster.controller.task import Task; print(Task)"
```

## Objective

Update cluster client interfaces to:
1. Expose task identification fields in `JobInfo` (task_id, task_index, num_tasks)
2. Add task-specific query methods to `RemoteClusterClient`
3. Extend `IrisClient` with task-level APIs
4. Support task-specific log streaming

## Files to Modify

- `src/iris/cluster/types.py` - Update JobInfo dataclass
- `src/iris/cluster/client/remote_client.py` - Add task RPC methods
- `src/iris/cluster/client/local_client.py` - Add task methods for local mode
- `src/iris/client/client.py` - Expose task APIs in IrisClient

## Implementation Tasks

### Task 1: Update JobInfo Dataclass

**File**: `src/iris/cluster/types.py`

Add task-level fields to the `JobInfo` dataclass so that code running inside a task can access its context:

```python
@dataclass
class JobInfo:
    """Runtime context available within a running job/task."""
    job_id: str
    task_id: str | None = None        # NEW: Full task ID ("{job_id}/task-{index}")
    task_index: int = 0               # NEW: 0-indexed task number
    num_tasks: int = 1                # NEW: Total tasks in job
    attempt_id: int = 0
    worker_id: str | None = None
    controller_address: str | None = None
    ports: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "JobInfo":
        """Load job info from environment variables set by the worker."""
        return cls(
            job_id=os.environ.get("IRIS_JOB_ID", ""),
            task_id=os.environ.get("IRIS_TASK_ID"),                    # NEW
            task_index=int(os.environ.get("IRIS_TASK_INDEX", "0")),    # NEW
            num_tasks=int(os.environ.get("IRIS_NUM_TASKS", "1")),      # NEW
            attempt_id=int(os.environ.get("IRIS_ATTEMPT_ID", "0")),
            worker_id=os.environ.get("IRIS_WORKER_ID"),
            controller_address=os.environ.get("IRIS_CONTROLLER_ADDRESS"),
            ports=cls._load_ports_from_env(),
        )

    @classmethod
    def _load_ports_from_env(cls) -> dict[str, int]:
        """Extract IRIS_PORT_* environment variables into ports dict."""
        ports = {}
        for key, value in os.environ.items():
            if key.startswith("IRIS_PORT_"):
                port_name = key.replace("IRIS_PORT_", "").lower()
                ports[port_name] = int(value)
        return ports
```

**Verification**:
```bash
# Test that JobInfo can parse task environment variables
uv run python -c "
import os
os.environ['IRIS_JOB_ID'] = 'test-job'
os.environ['IRIS_TASK_ID'] = 'test-job/task-0'
os.environ['IRIS_TASK_INDEX'] = '0'
os.environ['IRIS_NUM_TASKS'] = '4'
from iris.cluster.types import JobInfo
info = JobInfo.from_env()
assert info.task_id == 'test-job/task-0'
assert info.task_index == 0
assert info.num_tasks == 4
print('JobInfo.from_env() works correctly')
"
```

### Task 2: Add Task Methods to RemoteClusterClient

**File**: `src/iris/cluster/client/remote_client.py`

Add methods to query individual task status, list all tasks in a job, and fetch task-specific logs:

```python
class RemoteClusterClient:
    """Client for communicating with a remote Iris controller."""

    def __init__(self, controller_address: str):
        self._controller_address = controller_address
        self._stub = self._create_controller_stub()
        self._worker_stubs: dict[str, Any] = {}  # Cache worker stubs by address

    # Existing methods remain unchanged...

    def get_task_status(self, job_id: str, task_index: int) -> cluster_pb2.TaskStatus:
        """Get status of a specific task within a job.

        Args:
            job_id: Parent job ID
            task_index: 0-indexed task number

        Returns:
            TaskStatus proto for the requested task

        Raises:
            grpc.RpcError: If task not found or RPC fails
        """
        request = cluster_pb2.Controller.GetTaskStatusRequest(
            job_id=job_id,
            task_index=task_index,
        )
        response = self._stub.get_task_status(request)
        return response.task

    def list_tasks(self, job_id: str) -> list[cluster_pb2.TaskStatus]:
        """List all tasks for a job.

        Args:
            job_id: Job ID to query tasks for

        Returns:
            List of TaskStatus protos, one per task in the job
        """
        request = cluster_pb2.Controller.ListTasksRequest(job_id=job_id)
        response = self._stub.list_tasks(request)
        return list(response.tasks)

    def fetch_task_logs(
        self,
        task_id: str,
        start_ms: int = 0,
        max_lines: int = 0,
    ) -> list[cluster_pb2.Worker.LogEntry]:
        """Fetch logs for a specific task.

        This method queries the controller to find which worker is running the task,
        then fetches logs directly from that worker.

        Args:
            task_id: Full task ID in format "{job_id}/task-{index}"
            start_ms: Only return logs after this timestamp (milliseconds since epoch)
            max_lines: Maximum number of log lines to return (0 = unlimited)

        Returns:
            List of LogEntry protos from the worker

        Raises:
            ValueError: If task_id format is invalid
            grpc.RpcError: If task not found or worker unreachable
        """
        # Parse task_id to extract job_id and task_index
        if "/task-" not in task_id:
            raise ValueError(f"Invalid task_id format: {task_id}. Expected 'job_id/task-index'")

        job_id, task_suffix = task_id.rsplit("/", 1)
        task_index = int(task_suffix.split("-")[1])

        # Get task status to find which worker is running it
        task_status = self.get_task_status(job_id, task_index)
        if not task_status.worker_address:
            # Task not yet scheduled or has finished
            return []

        # Fetch logs from the worker
        worker_stub = self._get_worker_stub(task_status.worker_address)
        request = cluster_pb2.Worker.FetchTaskLogsRequest(
            task_id=task_id,
            filter=cluster_pb2.Worker.FetchLogsFilter(
                start_ms=start_ms,
                max_lines=max_lines,
            ),
        )
        response = worker_stub.fetch_task_logs(request)
        return list(response.logs)

    def _get_worker_stub(self, worker_address: str) -> Any:
        """Get or create a gRPC stub for a worker."""
        if worker_address not in self._worker_stubs:
            channel = grpc.insecure_channel(worker_address)
            self._worker_stubs[worker_address] = cluster_pb2_grpc.WorkerServiceStub(channel)
        return self._worker_stubs[worker_address]
```

**Verification**:
```bash
# This will be tested in integration tests when controller is running
# Unit test can verify the request construction
uv run pytest tests/cluster/client/test_remote_client.py::test_get_task_status -v
uv run pytest tests/cluster/client/test_remote_client.py::test_list_tasks -v
uv run pytest tests/cluster/client/test_remote_client.py::test_fetch_task_logs -v
```

### Task 3: Add Task Methods to LocalClusterClient

**File**: `src/iris/cluster/client/local_client.py`

Mirror the task methods in `LocalClusterClient` for consistency (local mode may have limited task support):

```python
class LocalClusterClient:
    """Client for local/in-process Iris execution."""

    # Existing methods remain unchanged...

    def get_task_status(self, job_id: str, task_index: int) -> cluster_pb2.TaskStatus:
        """Get status of a specific task within a job."""
        # For local mode, jobs typically have only 1 task
        if task_index != 0:
            raise ValueError(f"Local mode only supports task_index=0, got {task_index}")

        # Return a synthetic TaskStatus based on job status
        job_status = self.get_job_status(job_id)
        return cluster_pb2.TaskStatus(
            task_id=f"{job_id}/task-0",
            job_id=job_id,
            task_index=0,
            state=self._job_state_to_task_state(job_status.state),
            exit_code=job_status.exit_code,
            error=job_status.error,
            started_at_ms=job_status.started_at_ms,
            finished_at_ms=job_status.finished_at_ms,
        )

    def list_tasks(self, job_id: str) -> list[cluster_pb2.TaskStatus]:
        """List all tasks for a job (local mode always returns 1 task)."""
        return [self.get_task_status(job_id, 0)]

    def fetch_task_logs(
        self,
        task_id: str,
        start_ms: int = 0,
        max_lines: int = 0,
    ) -> list[cluster_pb2.Worker.LogEntry]:
        """Fetch logs for a specific task."""
        # Extract job_id from task_id
        job_id = task_id.split("/task-")[0]
        # Delegate to existing log fetching
        return self.fetch_logs(job_id, start_ms=start_ms, max_lines=max_lines)

    def _job_state_to_task_state(self, job_state: int) -> int:
        """Convert JobState to TaskState (enums are parallel)."""
        # JobState and TaskState have identical values, just different types
        state_mapping = {
            cluster_pb2.JOB_STATE_PENDING: cluster_pb2.TASK_STATE_PENDING,
            cluster_pb2.JOB_STATE_BUILDING: cluster_pb2.TASK_STATE_BUILDING,
            cluster_pb2.JOB_STATE_RUNNING: cluster_pb2.TASK_STATE_RUNNING,
            cluster_pb2.JOB_STATE_SUCCEEDED: cluster_pb2.TASK_STATE_SUCCEEDED,
            cluster_pb2.JOB_STATE_FAILED: cluster_pb2.TASK_STATE_FAILED,
            cluster_pb2.JOB_STATE_KILLED: cluster_pb2.TASK_STATE_KILLED,
            cluster_pb2.JOB_STATE_WORKER_FAILED: cluster_pb2.TASK_STATE_WORKER_FAILED,
            cluster_pb2.JOB_STATE_UNSCHEDULABLE: cluster_pb2.TASK_STATE_UNSCHEDULABLE,
        }
        return state_mapping.get(job_state, cluster_pb2.TASK_STATE_PENDING)
```

**Verification**:
```bash
uv run pytest tests/cluster/client/test_local_client.py::test_task_methods -v
```

### Task 4: Extend IrisClient with Task APIs

**File**: `src/iris/client/client.py`

Add user-facing task methods to the high-level `IrisClient` class:

```python
class IrisClient:
    """High-level client for interacting with Iris clusters."""

    # Existing methods remain unchanged...

    def task_status(self, job_id: JobId, task_index: int) -> cluster_pb2.TaskStatus:
        """Get status of a specific task within a job.

        Args:
            job_id: Job identifier
            task_index: 0-indexed task number

        Returns:
            TaskStatus proto containing state, worker assignment, and metrics

        Example:
            ```python
            client = IrisClient.remote("localhost:50051")
            job = client.run(my_function, replicas=4)

            # Check status of task 2
            task = client.task_status(job.job_id, task_index=2)
            print(f"Task 2 state: {task.state}")
            print(f"Task 2 worker: {task.worker_id}")
            ```
        """
        return self._client.get_task_status(str(job_id), task_index)

    def list_tasks(self, job_id: JobId) -> list[cluster_pb2.TaskStatus]:
        """List all tasks for a job.

        Args:
            job_id: Job identifier

        Returns:
            List of TaskStatus protos, one per task

        Example:
            ```python
            client = IrisClient.remote("localhost:50051")
            job = client.run(my_function, replicas=4)

            # Get all tasks
            tasks = client.list_tasks(job.job_id)
            for task in tasks:
                print(f"Task {task.task_index}: {task.state}")
            ```
        """
        return self._client.list_tasks(str(job_id))

    def fetch_task_logs(
        self,
        job_id: JobId,
        task_index: int,
        start_ms: int = 0,
        max_lines: int = 0,
    ) -> list[LogEntry]:
        """Fetch logs for a specific task.

        Args:
            job_id: Job identifier
            task_index: 0-indexed task number
            start_ms: Only return logs after this timestamp (milliseconds since epoch)
            max_lines: Maximum number of log lines to return (0 = unlimited)

        Returns:
            List of LogEntry objects from the task

        Example:
            ```python
            client = IrisClient.remote("localhost:50051")
            job = client.run(my_function, replicas=4)

            # Fetch logs from task 0
            logs = client.fetch_task_logs(job.job_id, task_index=0)
            for log in logs:
                print(f"[{log.source}] {log.message}")
            ```
        """
        task_id = f"{job_id}/task-{task_index}"
        entries = self._client.fetch_task_logs(task_id, start_ms, max_lines)
        return [LogEntry.from_proto(e) for e in entries]

    def wait(
        self,
        job_id: JobId,
        timeout: float | None = None,
        poll_interval: float = 1.0,
        stream_logs: bool = False,
        stream_task_index: int | None = None,  # NEW parameter
    ) -> cluster_pb2.JobStatus:
        """Wait for job completion with optional log streaming.

        Args:
            job_id: Job identifier
            timeout: Maximum seconds to wait (None = wait forever)
            poll_interval: Seconds between status polls
            stream_logs: If True, print logs while waiting
            stream_task_index: Which task to stream logs from (default: task 0)
                              Only used if stream_logs=True

        Returns:
            Final JobStatus when job reaches terminal state

        Raises:
            TimeoutError: If timeout is exceeded

        Example:
            ```python
            # Stream logs from task 2 of a multi-task job
            client = IrisClient.remote("localhost:50051")
            job = client.run(my_function, replicas=4)
            status = client.wait(
                job.job_id,
                stream_logs=True,
                stream_task_index=2  # Watch task 2 specifically
            )
            ```
        """
        start_time = time.time()
        last_log_ms = 0

        # Determine which task to stream logs from
        task_idx = stream_task_index if stream_task_index is not None else 0

        while True:
            status = self.status(job_id)

            # Stream logs if requested
            if stream_logs and status.num_tasks > 0:
                task_id = f"{job_id}/task-{task_idx}"
                new_logs = self._client.fetch_task_logs(
                    task_id,
                    start_ms=last_log_ms,
                    max_lines=1000,
                )
                for log_proto in new_logs:
                    log = LogEntry.from_proto(log_proto)
                    print(f"[task-{task_idx}:{log.source}] {log.message}")
                    last_log_ms = max(last_log_ms, log.timestamp_ms)

            # Check if job is finished
            if is_job_finished(status.state):
                return status

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

            time.sleep(poll_interval)
```

**Verification**:
```bash
# Test task methods
uv run pytest tests/client/test_iris_client.py::test_task_status -v
uv run pytest tests/client/test_iris_client.py::test_list_tasks -v
uv run pytest tests/client/test_iris_client.py::test_fetch_task_logs -v
uv run pytest tests/client/test_iris_client.py::test_wait_with_task_logs -v
```

## Code Snippets Summary

### Environment Variables Available to Tasks

After Stage 4, tasks will have access to these environment variables:

```python
# In your task code:
import os

job_id = os.environ["IRIS_JOB_ID"]           # e.g., "training-job-123"
task_id = os.environ["IRIS_TASK_ID"]         # e.g., "training-job-123/task-2"
task_index = int(os.environ["IRIS_TASK_INDEX"])  # e.g., 2
num_tasks = int(os.environ["IRIS_NUM_TASKS"])    # e.g., 4
attempt_id = int(os.environ["IRIS_ATTEMPT_ID"])  # e.g., 0 (first attempt)

# Or use the helper:
from iris.cluster.types import JobInfo
info = JobInfo.from_env()
print(f"I am task {info.task_index} of {info.num_tasks}")
```

### Querying Task Status from Client

```python
from iris.client import IrisClient

client = IrisClient.remote("localhost:50051")

# Launch multi-task job
job = client.run(my_training_function, replicas=4)

# Check individual task
task = client.task_status(job.job_id, task_index=2)
print(f"Task 2: {task.state}, Worker: {task.worker_id}")

# List all tasks
for task in client.list_tasks(job.job_id):
    print(f"Task {task.task_index}: {task.state}")

# Fetch logs from specific task
logs = client.fetch_task_logs(job.job_id, task_index=0)
for log in logs:
    print(f"[{log.source}] {log.message}")
```

## Verification Commands

After implementing all tasks, run the following commands to verify correctness:

```bash
# Unit tests for client modules
uv run pytest tests/cluster/client/test_remote_client.py -v
uv run pytest tests/cluster/client/test_local_client.py -v
uv run pytest tests/client/test_iris_client.py -v

# Integration tests that exercise task APIs end-to-end
uv run pytest tests/cluster/test_e2e.py::test_multi_task_job -v
uv run pytest tests/cluster/test_e2e.py::test_task_log_streaming -v

# Full client test suite
uv run pytest tests/client/ -v
uv run pytest tests/cluster/test_client.py -v
```

## Acceptance Criteria

Stage 4 is complete when:

1. **JobInfo includes task fields**: `task_id`, `task_index`, `num_tasks` are populated from environment
2. **RemoteClusterClient supports task queries**:
   - `get_task_status(job_id, task_index)` returns TaskStatus
   - `list_tasks(job_id)` returns all tasks
   - `fetch_task_logs(task_id, ...)` retrieves task-specific logs
3. **LocalClusterClient mirrors task APIs**: All task methods work in local mode
4. **IrisClient exposes task APIs**: User-facing methods for task status and logs
5. **Log streaming supports task selection**: `wait()` accepts `stream_task_index` parameter
6. **All tests pass**: Client tests validate task methods work correctly
7. **Integration tests pass**: End-to-end tests confirm task APIs work with real controller/workers

## Dependencies

This stage depends on:

- **Stage 1 (Proto Updates)**: `TaskStatus`, `TaskState`, and task RPC messages must be defined
- **Stage 2 (Worker Updates)**: Workers must populate `IRIS_TASK_ID`, `IRIS_TASK_INDEX`, `IRIS_NUM_TASKS` environment variables
- **Stage 3 (Controller Updates)**: Controller must implement `GetTaskStatus`, `ListTasks` RPCs

## Quality Checklist

Before committing Stage 4, verify:

- [ ] All client tests pass (`uv run pytest tests/client/ tests/cluster/client/ -v`)
- [ ] `JobInfo.from_env()` correctly parses task environment variables
- [ ] Task methods work for both remote and local clients
- [ ] Log streaming can target specific tasks in multi-task jobs
- [ ] No regressions in existing job-level APIs
- [ ] Code follows project conventions (see `AGENTS.md`)
- [ ] Changes are minimal and focused on client updates only
- [ ] Integration tests verify end-to-end task workflows

## Troubleshooting

**Tests fail with "Task not found"**:
- Verify Stage 3 is complete and controller implements task RPCs
- Check that `GetTaskStatus` and `ListTasks` RPCs are registered in ControllerService
- Ensure test jobs have `replicas > 1` if testing multi-task scenarios

**Environment variables not populated**:
- Verify Stage 2 (Worker) sets `IRIS_TASK_ID`, `IRIS_TASK_INDEX`, `IRIS_NUM_TASKS`
- Check that `_build_iris_env()` in worker.py includes the new variables
- Confirm worker receives `task_index` and `num_tasks` in `RunTaskRequest`

**Log streaming shows no logs**:
- Ensure task is running and has worker assignment
- Check `task.worker_address` is populated before fetching logs
- Verify worker implements `FetchTaskLogs` RPC (Stage 2)

## Next Steps

After completing Stage 4, proceed to **Stage 5: Failure Domains & Co-scheduling**, which adds:
- Failure domain enforcement (when one task fails, kill related tasks)
- Enhanced logging for task retry attempts
- Gang scheduling support for multi-task jobs
