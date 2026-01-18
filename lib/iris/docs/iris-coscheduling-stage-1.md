# Stage 1: Proto Updates for Co-scheduling

## Objective

Update the proto definitions to support tasks as first-class citizens. The controller will expand `replicas` into tasks, and workers will operate on tasks instead of jobs. This establishes the foundational data structures needed for co-scheduling support.

## Prerequisites

None - this is the first stage of the co-scheduling implementation.

## Files to Modify

- `src/iris/rpc/cluster.proto` - Add task messages and update existing messages
- `scripts/generate_protos.py` - Run to regenerate Python bindings

## Detailed Implementation Steps

### 1.1 Add TaskState and TaskStatus messages

Tasks share the same state machine as jobs but are tracked separately. Add these new messages after the `JobState` enum in `cluster.proto`:

```protobuf
// In cluster.proto, after JobState enum

// Task state mirrors job state - a task is a single unit of execution within a job
// TaskState is identical to JobState but distinct for type safety and clarity
enum TaskState {
  TASK_STATE_UNSPECIFIED = 0;
  TASK_STATE_PENDING = 1;
  TASK_STATE_BUILDING = 2;
  TASK_STATE_RUNNING = 3;
  TASK_STATE_SUCCEEDED = 4;
  TASK_STATE_FAILED = 5;
  TASK_STATE_KILLED = 6;
  TASK_STATE_WORKER_FAILED = 7;
  TASK_STATE_UNSCHEDULABLE = 8;
}

// Status of a single task within a job
message TaskStatus {
  string task_id = 1;                 // "{job_id}/task-{index}"
  string job_id = 2;
  int32 task_index = 3;               // 0-indexed task number
  TaskState state = 4;

  string worker_id = 5;
  string worker_address = 6;

  int32 exit_code = 7;
  string error = 8;

  int64 started_at_ms = 9;
  int64 finished_at_ms = 10;

  map<string, int32> ports = 11;      // Per-task port allocations
  ResourceUsage resource_usage = 12;
  BuildMetrics build_metrics = 13;

  int32 current_attempt_id = 14;
  repeated TaskAttempt attempts = 15;
}

// Record of a single task execution attempt
message TaskAttempt {
  int32 attempt_id = 1;
  string worker_id = 2;
  TaskState state = 3;
  int32 exit_code = 4;
  string error = 5;
  int64 started_at_ms = 6;
  int64 finished_at_ms = 7;
  bool is_worker_failure = 8;
}
```

### 1.2 Update JobStatus to include task information

Extend `JobStatus` to report aggregate task state by adding these fields:

```protobuf
message JobStatus {
  // ... existing fields 1-18 ...

  // Task information (populated for jobs with replicas > 1)
  int32 num_tasks = 19;
  int32 tasks_pending = 20;
  int32 tasks_running = 21;
  int32 tasks_succeeded = 22;
  int32 tasks_failed = 23;
  repeated TaskStatus tasks = 24;     // Per-task status (optional, controlled by include_tasks)
}
```

### 1.3 Update Controller messages for task operations

Add `max_task_failures` to `LaunchJobRequest` and add task-specific RPC messages to the `Controller` message block:

```protobuf
message Controller {
  message LaunchJobRequest {
    // ... existing fields 1-10 ...

    // NEW: Job-level failure tolerance
    // A job fails when this many tasks have exhausted their retries (default 0 = fail on first task failure)
    // Preemptions do not count toward this limit
    int32 max_task_failures = 11;
  }

  // --- Task Operations ---
  message GetTaskStatusRequest {
    string job_id = 1;
    int32 task_index = 2;
  }

  message GetTaskStatusResponse {
    TaskStatus task = 1;
  }

  message ListTasksRequest {
    string job_id = 1;
  }

  message ListTasksResponse {
    repeated TaskStatus tasks = 1;
  }

  // Update GetJobStatusRequest to optionally include task details
  message GetJobStatusRequest {
    string job_id = 1;
    bool include_result = 2;
    bool include_tasks = 3;         // NEW: include per-task status in response
  }
}
```

### 1.4 Rename Worker messages from Job to Task

The worker operates on tasks, not jobs. Rename all `Job` references to `Task` in the `Worker` message block:

```protobuf
message Worker {
  // Renamed: RunJobRequest -> RunTaskRequest
  message RunTaskRequest {
    string job_id = 1;              // Parent job ID
    string task_id = 2;             // Full task ID: "{job_id}/task-{index}"
    int32 task_index = 3;           // 0-indexed task number
    int32 num_tasks = 4;            // Total tasks in job (for IRIS_NUM_TASKS)

    bytes serialized_entrypoint = 5;
    EnvironmentConfig environment = 6;
    string bundle_gcs_path = 7;
    ResourceSpecProto resources = 8;
    int32 timeout_seconds = 9;
    repeated string ports = 10;
    int32 attempt_id = 11;
  }

  message RunTaskResponse {
    string task_id = 1;
    TaskState state = 2;
  }

  // Renamed: GetJobStatusRequest -> GetTaskStatusRequest
  message GetTaskStatusRequest {
    string task_id = 1;
    bool include_result = 2;
  }

  // Renamed: ListJobsRequest/Response -> ListTasksRequest/Response
  message ListTasksRequest {}
  message ListTasksResponse {
    repeated TaskStatus tasks = 1;
  }

  // Renamed: FetchLogsRequest -> FetchTaskLogsRequest
  message FetchTaskLogsRequest {
    string task_id = 1;
    FetchLogsFilter filter = 2;
  }

  message FetchTaskLogsResponse {
    repeated LogEntry logs = 1;
  }

  // Renamed: KillJobRequest -> KillTaskRequest
  message KillTaskRequest {
    string task_id = 1;
    int32 term_timeout_ms = 2;
  }

  message HealthResponse {
    bool healthy = 1;
    int64 uptime_ms = 2;
    int32 running_tasks = 3;        // Renamed from running_jobs
  }
}
```

### 1.5 Update WorkerService

Update the service definition to use the renamed task methods:

```protobuf
service WorkerService {
  rpc RunTask(Worker.RunTaskRequest) returns (Worker.RunTaskResponse);
  rpc GetTaskStatus(Worker.GetTaskStatusRequest) returns (TaskStatus);
  rpc ListTasks(Worker.ListTasksRequest) returns (Worker.ListTasksResponse);
  rpc FetchTaskLogs(Worker.FetchTaskLogsRequest) returns (Worker.FetchTaskLogsResponse);
  rpc KillTask(Worker.KillTaskRequest) returns (Empty);
  rpc HealthCheck(Empty) returns (Worker.HealthResponse);
}
```

### 1.6 Update Controller state reporting

Add task-specific state reporting messages to the `Controller` block and update the service definition:

```protobuf
message Controller {
  // Renamed: ReportJobStateRequest -> ReportTaskStateRequest
  message ReportTaskStateRequest {
    string worker_id = 1;
    string task_id = 2;             // Full task ID
    string job_id = 3;              // Parent job ID (for job-level state aggregation)
    int32 task_index = 4;
    TaskState state = 5;
    int32 exit_code = 6;
    string error = 7;
    int64 finished_at_ms = 8;
    int32 attempt_id = 9;
  }

  message ReportTaskStateResponse {}
}

service ControllerService {
  // ... existing RPCs ...

  // Renamed
  rpc ReportTaskState(Controller.ReportTaskStateRequest) returns (Controller.ReportTaskStateResponse);

  // New task-specific RPCs
  rpc GetTaskStatus(Controller.GetTaskStatusRequest) returns (Controller.GetTaskStatusResponse);
  rpc ListTasks(Controller.ListTasksRequest) returns (Controller.ListTasksResponse);
}
```

## Code Snippets

All code snippets are provided inline in the implementation steps above. These are complete protobuf definitions that should be added or modified in `src/iris/rpc/cluster.proto`.

## Verification Commands

After completing the proto updates, regenerate the Python bindings and verify they work:

```bash
# Regenerate protos
uv run python scripts/generate_protos.py

# Verify proto imports work (this will fail initially until Python code is updated in Stage 2)
uv run python -c "from iris.rpc import cluster_pb2; print(cluster_pb2.TaskState.Name(1))"

# Verify the proto file has no syntax errors
uv run python -c "from iris.rpc import cluster_pb2; print('Proto loaded successfully')"
```

## Acceptance Criteria

This stage is complete when:

- [ ] All proto changes have been made to `cluster.proto`
- [ ] `TaskState` enum exists with all 8 states (UNSPECIFIED through UNSCHEDULABLE)
- [ ] `TaskStatus` message exists with all 15 fields
- [ ] `TaskAttempt` message exists with all 8 fields
- [ ] `JobStatus` has new fields 19-24 for task information
- [ ] `LaunchJobRequest` has `max_task_failures` field (field 11)
- [ ] `Controller.GetJobStatusRequest` has `include_tasks` field (field 3)
- [ ] All `Worker` message names have been renamed from `Job` to `Task`
- [ ] `Worker.RunTaskRequest` includes task-specific fields (`task_id`, `task_index`, `num_tasks`)
- [ ] `Worker.HealthResponse.running_tasks` renamed from `running_jobs`
- [ ] `WorkerService` uses renamed RPC methods (`RunTask`, `GetTaskStatus`, etc.)
- [ ] `Controller.ReportTaskStateRequest` message exists with task fields
- [ ] `ControllerService` has `ReportTaskState`, `GetTaskStatus`, and `ListTasks` RPCs
- [ ] `uv run python scripts/generate_protos.py` runs successfully without errors
- [ ] Python bindings can be imported: `from iris.rpc import cluster_pb2`

## Notes

- This stage introduces breaking changes to the proto API. Subsequent stages will update the Python code to use these new protos.
- The Python code will not compile after this stage until Stage 2 (Worker Updates) is complete. This is expected.
- Do not attempt to run integration tests yet - they will fail due to the proto changes.
- The design decision to use a separate `TaskState` enum (instead of reusing `JobState`) provides type safety and clarity in the codebase.
- The `max_task_failures` field defaults to 0, meaning jobs fail on the first task failure unless explicitly configured otherwise.

## Next Stage

After completing this stage, proceed to Stage 2: Worker Updates, which will update the worker implementation to use the new task-based protos.
