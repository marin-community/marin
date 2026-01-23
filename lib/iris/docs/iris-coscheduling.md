# Iris Co-scheduling Design

## Overview

Co-scheduling transforms Iris from a job-per-worker model to a task-based model where a single job can spawn multiple coordinated tasks across different workers (e.g., multi-host TPU training).

**Before**: `client.run(fn, replicas=1)` → 1 job on 1 worker

**After**: `client.run(fn, replicas=4)` → 1 job with 4 tasks on 4 workers

Tasks share a **failure domain**: when failures exceed `max_task_failures` (default 0), all remaining tasks are killed.

## Key Concepts

### Task Identity

Each task knows its position in the job:

```python
info = get_job_info()
# info.task_id = "abc123/task-2"
# info.task_index = 2
# info.num_tasks = 4
```

Environment variables: `IRIS_TASK_ID`, `IRIS_TASK_INDEX`, `IRIS_NUM_TASKS`

### Job → Task Expansion

Jobs with `replicas=N` expand into N tasks at submission:

```
Job(replicas=4) → [Task-0, Task-1, Task-2, Task-3]
```

Each task has independent lifecycle, worker assignment, and retry tracking.

### Failure Domains

Job state is derived from task states:

- **SUCCESS**: All tasks succeeded
- **FAILED**: Task failures > `max_task_failures`
- **RUNNING**: Otherwise

When job fails, remaining running tasks are killed immediately.

### Retry Logic

Tasks track failures and preemptions separately:

- `failure_count`: Actual task failures (counts toward `max_task_failures`)
- `preemption_count`: Worker died/preempted (auto-retry, doesn't count toward failure limit)

## Data Flow

```
User: client.run(fn, replicas=4)
         ↓
Controller: create Job + expand to 4 Tasks
         ↓
Scheduler: match tasks to workers
         ↓
Controller: dispatch RunTaskRequest to each worker
         ↓
Workers: execute tasks, set IRIS_* env vars
         ↓
Workers: report completion via ReportTaskStateRequest
         ↓
Controller: update task states, derive job state
         ↓
If failures > max_task_failures → kill remaining tasks
```

## Usage

### Submit Multi-Task Job

```python
job = client.submit(train_model, resources=ResourceSpec(replicas=4))
status = job.wait(stream_logs=True)
```

### Query Tasks

```python
tasks = client.list_tasks(job.job_id)
task = client.task_status(job.job_id, task_index=2)
logs = client.fetch_task_logs(job.job_id, task_index=1)
```

### Task-Aware Code

```python
info = get_job_info()
if info.task_index == 0:
    initialize_shared_state()
shard = get_data_shard(info.task_index, info.num_tasks)
process(shard)
```
