# Iris Co-scheduling Design

## Overview

Co-scheduling enables Iris to run distributed workloads that require multiple coordinated processes (e.g., multi-host TPU training on v4-4x4x4 topologies). This feature transforms Iris from a job-per-worker model to a task-based model where a single job can spawn multiple tasks across different workers.

### Before: Single-Task Jobs

Previously, each job ran as a single unit of work on one worker:

```python
# User submits a job
client.run(train_model, resources=ResourceSpec(cpu=8, memory_gb=32))

# Controller creates 1 job → runs on 1 worker
```

### After: Multi-Task Jobs with Failure Domains

Now jobs expand into coordinated tasks based on the `replicas` field:

```python
# User submits a job with replicas
client.run(train_model, resources=ResourceSpec(cpu=8, memory_gb=32, replicas=4))

# Controller creates:
#   - 1 Job with job_id="abc123"
#   - 4 Tasks: "abc123/task-0", "abc123/task-1", "abc123/task-2", "abc123/task-3"
#   - Each task runs on a different worker
#   - All tasks share a failure domain
```

Each task has its own lifecycle, worker assignment, and retry logic. Tasks within a job share a **failure domain**: when task failures exceed `max_task_failures` (default 0), all remaining tasks are killed. This prevents partially-completed distributed jobs from wasting resources.

### Task Identity

Code running inside a task can access its context:

```python
from iris.cluster.client import get_job_info

info = get_job_info()
print(f"I am task {info.task_index} of {info.num_tasks}")  # "I am task 2 of 4"
print(f"Task ID: {info.task_id}")  # "abc123/task-2"
```

## Key Design Decisions

### 1. Task as the Unit of Execution

Jobs with `replicas=N` expand into N tasks at submission time. Each task runs independently on a worker:

```python
# Controller: expand_job_to_tasks (job.py)
def expand_job_to_tasks(job: Job, now_ms: int) -> list[Task]:
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

### 2. Separate Task and Job State Enums

Tasks use `TaskState` instead of reusing `JobState` for type safety:

```protobuf
enum TaskState {
  TASK_STATE_PENDING = 1;
  TASK_STATE_RUNNING = 3;
  TASK_STATE_SUCCEEDED = 4;
  TASK_STATE_FAILED = 5;
  TASK_STATE_WORKER_FAILED = 7;  // Preemption - doesn't count toward max_task_failures
  ...
}
```

### 3. Failure Domains with max_task_failures

Jobs fail when task failures exceed `max_task_failures` (default 0). Remaining tasks are killed:

```python
# Controller job.py: Job._compute_job_state()
# Derives job state from task_state_counts. O(1) - no task iteration.
def _compute_job_state(self, now_ms: int) -> int | None:
    counts = self.task_state_counts

    # Job succeeds when all tasks succeed
    if counts[cluster_pb2.TASK_STATE_SUCCEEDED] == self.num_tasks:
        return cluster_pb2.JOB_STATE_SUCCEEDED

    # Only actual failures count (not preemptions/worker failures)
    max_task_failures = self.request.max_task_failures
    if counts[cluster_pb2.TASK_STATE_FAILED] > max_task_failures:
        return cluster_pb2.JOB_STATE_FAILED
    ...

# Controller state.py: transition_task() triggers failure domain
if new_job_state == cluster_pb2.JOB_STATE_FAILED:
    job.finished_at_ms = now_ms
    job.error = self._get_first_task_error(task.job_id)
    self._kill_remaining_tasks(task.job_id, now_ms, "Job exceeded max_task_failures")
```

Preemptions (`TASK_STATE_WORKER_FAILED`) do **not** count toward `max_task_failures` - they retry automatically.

### 4. Task-Level Retry Logic

Each task tracks failures and preemptions separately:

```python
# Controller task.py: Task dataclass
@dataclass
class Task:
    failure_count: int = 0           # Actual task failures
    preemption_count: int = 0        # Worker died/preempted
    max_retries_failure: int = 0     # Retry limit for failures
    max_retries_preemption: int = 100  # Retry limit for preemptions
```

### 5. Task-Level Endpoints and Resource Tracking

Workers now track `running_tasks` instead of `running_jobs`:

```python
@dataclass
class ControllerWorker:
    running_tasks: set[TaskId] = field(default_factory=set)  # Was: running_jobs
```

Endpoints are scoped to tasks and cleaned up when tasks finish, not when the job finishes.

## Implementation Stages

### Stage 1: Proto Updates

**Scope**: Add task-level protobuf messages and rename Worker RPCs from job to task terminology.

**Key Changes**:
- Add `TaskState`, `TaskStatus`, and `TaskAttempt` messages
- Rename `Worker.RunJobRequest` → `Worker.RunTaskRequest`
- Add `Controller.ReportTaskStateRequest` for workers to report task completion
- Add `max_task_failures` field to `LaunchJobRequest`

```protobuf
enum TaskState {
  TASK_STATE_PENDING = 1;
  TASK_STATE_RUNNING = 3;
  TASK_STATE_SUCCEEDED = 4;
  TASK_STATE_FAILED = 5;
  ...
}

message Worker {
  message RunTaskRequest {
    string task_id = 2;          // "{job_id}/task-{index}"
    int32 task_index = 3;        // 0-indexed task number
    int32 num_tasks = 4;         // Total tasks in job
    ...
  }
}
```

- **Document**: [Stage 1: Proto Updates](iris-coscheduling-stage-1.md)
- **Verification**: `uv run python scripts/generate_protos.py`

---

### Stage 2: Worker Updates

**Scope**: Transform workers to execute and manage tasks instead of jobs.

**Key Changes**:
- Rename internal `Job` class to `Task` in `worker_types.py`
- Update all methods: `submit_job` → `submit_task`, etc.
- Set task environment variables: `IRIS_TASK_ID`, `IRIS_TASK_INDEX`, `IRIS_NUM_TASKS`
- Report task state to controller via `ReportTaskStateRequest`

```python
# Worker worker.py: _build_iris_env
def _build_iris_env(self, task: Task) -> dict[str, str]:
    env = {}
    env["IRIS_JOB_ID"] = task.job_id
    env["IRIS_TASK_ID"] = task.task_id         # NEW
    env["IRIS_TASK_INDEX"] = str(task.task_index)  # NEW
    env["IRIS_NUM_TASKS"] = str(task.num_tasks)    # NEW
    ...
    return env
```

- **Document**: [Stage 2: Worker Updates](iris-coscheduling-stage-2.md)
- **Verification**: `uv run pytest tests/cluster/worker/ -v`

---

### Stage 3: Controller Updates

**Scope**: Implement task-level tracking, scheduling, and failure domain logic in the controller.

**Key Changes**:
- Create `Task` and `TaskAttempt` dataclasses in `task.py`
- Add `expand_job_to_tasks()` to create tasks from job replicas
- Track tasks in `ControllerState`: `_tasks`, `_tasks_by_job`, `_task_queue`
- Implement failure domain logic in `Job._compute_job_state()` and `ControllerState.transition_task()`
- Scheduler now assigns tasks (not jobs) to workers

```python
# Controller state.py
class ControllerState:
    def __init__(self):
        self._tasks: dict[TaskId, Task] = {}
        self._tasks_by_job: dict[JobId, list[TaskId]] = {}
        self._task_queue: deque[TaskId] = deque()
        ...

    def add_job(self, job: Job, tasks: list[Task] | None = None):
        if tasks is None:
            tasks = expand_job_to_tasks(job, now_ms())
        for task in tasks:
            self._tasks[task.task_id] = task
            self._task_queue.append(task.task_id)
        ...
```

This is the largest stage, touching state management, scheduling, and RPC handling.

- **Document**: [Stage 3: Controller Updates](iris-coscheduling-stage-3.md)
- **Verification**: `uv run pytest tests/cluster/controller/ -v`

---

### Stage 4: Client Updates

**Scope**: Expose task-level APIs to users for querying and debugging individual tasks.

**Key Changes**:
- Add `task_id`, `task_index`, `num_tasks` to `JobInfo` dataclass
- Add `get_task_status()`, `list_tasks()`, `fetch_task_logs()` to clients
- Support `stream_task_index` parameter in `wait()` for task-specific log streaming

```python
# Client client.py
class IrisClient:
    def task_status(self, job_id: JobId, task_index: int) -> TaskStatus:
        """Get status of a specific task within a job."""
        return self._cluster.get_task_status(str(job_id), task_index)

    def list_tasks(self, job_id: JobId) -> list[TaskStatus]:
        """List all tasks for a job."""
        return self._cluster.list_tasks(str(job_id))
```

- **Document**: [Stage 4: Client Updates](iris-coscheduling-stage-4.md)
- **Verification**: `uv run pytest tests/cluster/test_client.py tests/client/ -v`

---

### Stage 5: Failure Domains & Verification

**Scope**: Verify failure domain logic and add enhanced logging for task-level events.

**Key Changes**:
- Verify `Job._compute_job_state()` and `ControllerState.transition_task()` correctly implement failure domain semantics
- Add scheduler logging for `task_timeout` and `task_unschedulable` events
- Confirm `ControllerWorker.running_tasks` rename is complete
- End-to-end verification that failure domains work correctly

```python
# Scheduler scheduler.py: find_assignments()
for task in pending_tasks:
    job = self._state.get_job(task.job_id)
    if not job:
        continue

    if self._is_task_timed_out(task, job, now_ms):
        transaction.timed_out_tasks.append(task)
        continue
```

- **Document**: [Stage 5: Failure Domains & Co-scheduling](iris-coscheduling-stage-5.md)
- **Verification**: `uv run pytest tests/cluster/ -v -k "gang or failure"`

## Architecture Overview

The co-scheduling implementation follows a clear data flow from user submission to task execution:

```
User submits job with replicas=4
         ↓
Controller.launch_job()
         ↓
Job created + expand_job_to_tasks() → 4 Task objects
         ↓
Tasks added to _task_queue
         ↓
Scheduler.find_assignments() → match tasks to workers
         ↓
Controller._dispatch_task() → send RunTaskRequest to worker
         ↓
Worker executes task, sets IRIS_TASK_ID/INDEX/NUM_TASKS env vars
         ↓
Worker reports completion via ReportTaskStateRequest
         ↓
ControllerState.transition_task() → update task state
         ↓
Job.on_task_transition() → update task counts, Job._compute_job_state() → derive job state
         ↓
If failures exceed max_task_failures → _kill_remaining_tasks()
```

## Usage Examples

### Submit a Multi-Task Job

```python
from iris.client import IrisClient
from iris.cluster.types import ResourceSpec

client = IrisClient.remote("http://localhost:8080", workspace=Path("."))

# Submit job with 4 replicas
job = client.submit(
    train_model,
    "my-training-job",
    resources=ResourceSpec(cpu=8, memory_gb=32, replicas=4),
)

# Wait and stream logs from all tasks
status = job.wait(stream_logs=True)
```

Note: `max_task_failures` defaults to 0 (fail job if any task fails) and is configured in the proto `LaunchJobRequest.max_task_failures` field.

### Query Task Status

```python
# Get individual task status
task = client.task_status(job.job_id, task_index=2)
print(f"Task 2: {task.state}, Worker: {task.worker_id}")

# List all tasks
for task in client.list_tasks(job.job_id):
    print(f"Task {task.task_index}: {task.state}")

# Fetch logs from specific task
logs = client.fetch_task_logs(job.job_id, task_index=1)
```

### Access Task Context in Code

```python
# Inside your task code
from iris.cluster.client import get_job_info

info = get_job_info()

if info.task_index == 0:
    # Only task 0 does setup
    initialize_shared_state()

# Each task processes its own shard
shard = get_data_shard(info.task_index, info.num_tasks)
process(shard)
```
