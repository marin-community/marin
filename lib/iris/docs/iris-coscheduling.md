# Iris Co-scheduling Design

## Overview

Co-scheduling enables Iris to schedule related units of work together, which is essential for workloads that span multiple accelerator slices (e.g., TPU v4-4x4x4 topologies). In a co-scheduled system, a single job can request multiple replicas that are scheduled and executed as a coordinated group. This design document describes the architecture for adding native co-scheduling support to Iris.

The core concept is that **jobs have one or more tasks**. A task is the unit of execution: it runs on a single worker, has its own lifecycle, and reports its own state. When a user submits a job with `replicas=N`, the controller expands that job into N tasks, each identified by a task ID of the form `{job_id}/task-{index}`. Tasks are scheduled independently but share a common failure domain.

**Failure domains** ensure that related tasks fail together when appropriate. If a job's task failures exceed `max_task_failures`, all remaining tasks in that job are killed. This prevents partially-completed jobs from wasting resources and ensures clean failure semantics for distributed workloads. Preemptions (worker failures) do not count toward the failure limit, only actual task failures after per-task retries are exhausted.

## Key Design Decisions

- **Separate TaskState enum for type safety**: Tasks use their own `TaskState` enum rather than reusing `JobState`, providing clear separation between job-level and task-level state machines.

- **max_task_failures configuration (default 0)**: Jobs fail when task failures exceed this threshold. The default of 0 means the job fails immediately on the first task failure after retries are exhausted. Setting this higher allows partial task failure tolerance.

- **Task-level endpoint tracking**: Endpoints are associated with individual tasks, not jobs. When a task finishes, only its endpoints are cleaned up, allowing other tasks in the same job to continue exposing their services.

- **running_jobs renamed to running_tasks**: The `ControllerWorker.running_jobs` field is renamed to `running_tasks` to reflect that workers track task assignments, not job assignments.

- **Task ID format**: Task IDs follow the pattern `{job_id}/task-{index}` where index is 0-based. This makes it easy to identify the parent job and task position.

- **Preemptions don't count toward failure limits**: `TASK_STATE_WORKER_FAILED` indicates a preemption or infrastructure failure, which gets retried automatically without counting against `max_task_failures`.

## Implementation Stages

### Stage 1: Proto Updates

This stage establishes the foundational data structures for task-level tracking by updating the protobuf definitions. New messages include `TaskState`, `TaskStatus`, and `TaskAttempt`. The `Worker` message block is renamed from job to task terminology (`RunTaskRequest`, `GetTaskStatusRequest`, etc.), and the `Controller` gains task-specific RPCs and the `max_task_failures` field.

After this stage, the proto API supports tasks as first-class citizens. Python code will not compile until Stage 2 is complete, as the proto changes are breaking.

- **Document**: [Stage 1: Proto Updates](iris-coscheduling-stage-1.md)
- **Verification**: `uv run python scripts/generate_protos.py`

### Stage 2: Worker Updates

This stage transforms the worker layer to operate on tasks instead of jobs. The internal `Job` class is renamed to `Task`, method names are updated (`submit_job` becomes `submit_task`), and environment variables are extended to include `IRIS_TASK_ID`, `IRIS_TASK_INDEX`, and `IRIS_NUM_TASKS`. Workers report task state to the controller via `ReportTaskStateRequest`.

After this stage, workers can execute tasks and report their completion status back to the controller.

- **Document**: [Stage 2: Worker Updates](iris-coscheduling-stage-2.md)
- **Verification**: `uv run pytest tests/cluster/worker/ -v`

### Stage 3: Controller Updates

This stage implements task-level tracking in the controller. Jobs are expanded into tasks at submission time based on the `replicas` field. The controller tracks each task's state independently, handles task state reports from workers, and aggregates task states to compute job-level state. The failure domain logic is implemented: when `max_task_failures` is exceeded, `_kill_remaining_tasks()` terminates all non-finished tasks in the job.

This is the largest stage, touching state management, scheduling, dispatch, and RPC handling.

- **Document**: [Stage 3: Controller Updates](iris-coscheduling-stage-3.md)
- **Verification**: `uv run pytest tests/cluster/controller/ -v`

### Stage 4: Client Updates

This stage extends the client APIs to expose task-level information. The `JobInfo` dataclass gains `task_id`, `task_index`, and `num_tasks` fields populated from environment variables. Both `RemoteClusterClient` and `IrisClient` gain methods for querying individual task status, listing all tasks in a job, and fetching task-specific logs. The `wait()` method supports streaming logs from a specific task.

After this stage, users can inspect and debug individual tasks within a multi-task job.

- **Document**: [Stage 4: Client Updates](iris-coscheduling-stage-4.md)
- **Verification**: `uv run pytest tests/cluster/test_client.py tests/client/ -v`

### Stage 5: Failure Domains & Co-scheduling

This stage verifies and enhances the failure domain semantics. The core logic is implemented in Stage 3, but this stage adds enhanced logging (task timeout and unschedulable events), confirms the `running_jobs` to `running_tasks` rename is complete, and provides end-to-end verification that failure domains work correctly.

After this stage, the complete co-scheduling implementation is verified and ready for use.

- **Document**: [Stage 5: Failure Domains & Co-scheduling](iris-coscheduling-stage-5.md)
- **Verification**: `uv run pytest tests/cluster/ -v -k "gang or failure"`

## Implementation Recipe

For detailed guidance on implementing each stage, including the workflow for exploration, task breakdown, review, and commit, see the [Implementation Recipe](impl-recipe.md). This recipe provides a structured approach for completing each stage as a focused, reviewable unit of work.

## Verification Summary

| Phase | Verification Command | Expected Outcome |
|-------|---------------------|------------------|
| 1. Proto | `uv run python scripts/generate_protos.py` | Protos regenerate without errors |
| 2. Worker | `uv run pytest tests/cluster/worker/ -v` | All worker tests pass |
| 3. Controller | `uv run pytest tests/cluster/controller/ -v` | All controller tests pass |
| 4. Client | `uv run pytest tests/cluster/test_client.py tests/client/ -v` | All client tests pass |
| 5. Failure domains | `uv run pytest tests/cluster/ -v` | Gang failure tests pass |
| Full | `uv run pytest tests/ -v` | All tests pass |
