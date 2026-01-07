# Cluster Resource Scheduling Tuneup

This document describes the implementation plan for resource-aware scheduling in the fluster controller.

## Goals

1. Track worker resource availability and consumption (request-based, not actual utilization)
2. Implement FIFO queue that skips unfittable jobs (don't block smaller jobs behind large ones)
3. Add scheduling timeout with `JOB_STATE_UNSCHEDULABLE` state
4. Match jobs to workers based on CPU, memory, device type, and device variant

## Current Problem

The current scheduler (`scheduler.py:129-141`) has this behavior:
```python
while True:
    job = self._state.pop_next_pending()
    if not job:
        break
    worker = find_worker_for_job(self._state, job)
    if not worker:
        self._state.add_job(job)  # re-queue
        break  # STOP TRYING - blocks everything!
```

This means a large job that can't fit blocks all smaller jobs behind it.

## Implementation Plan

### 1. Proto Changes (`lib/fluster/src/fluster/proto/cluster.proto`)

Add new job state and scheduling timeout field:

```protobuf
enum JobState {
  // ... existing states ...
  JOB_STATE_UNSCHEDULABLE = 8;  // NEW: Couldn't be scheduled within timeout
}

message LaunchJobRequest {
  // ... existing fields ...
  int32 scheduling_timeout_seconds = 8;  // NEW: 0 = no timeout (wait forever)
}
```

Run `uv run buf generate` after changes.

### 2. New Resource Utilities (`lib/fluster/src/fluster/cluster/controller/resources.py`)

Create new module for resource parsing:

```python
def parse_memory_string(memory_str: str) -> int:
    """Parse '8g', '16gb', '512m' to bytes."""

def get_device_type(device: DeviceConfig) -> str:
    """Return 'cpu', 'gpu', or 'tpu'."""

def get_device_variant(device: DeviceConfig) -> str | None:
    """Return variant like 'A100', 'v5litepod-16', or None."""

def get_gpu_count(device: DeviceConfig) -> int:
    """Return GPU count from device config."""
```

### 3. State Changes (`lib/fluster/src/fluster/cluster/controller/state.py`)

Add new methods to `ControllerState` for queue management:

```python
def peek_pending_jobs(self) -> list[ControllerJob]:
    """Return all PENDING jobs in queue order without removing them."""

def remove_from_queue(self, job_id: JobId) -> None:
    """Remove a specific job from the queue."""
```

**Note:** We do NOT track committed resources incrementally. Instead, we compute
available headroom dynamically by summing resources of jobs in `worker.running_jobs`.
This avoids sync issues and is simpler to reason about.

### 4. Worker Matching (`lib/fluster/src/fluster/cluster/controller/workers.py`)

Replace first-fit with resource-aware matching:

```python
def get_committed_resources(state: ControllerState, worker: ControllerWorker) -> tuple[int, int, int]:
    """Compute resources committed to running jobs on this worker.

    Dynamically sums resources from all jobs in worker.running_jobs.
    Returns (cpu, memory_bytes, gpu_count).
    """
    cpu, memory, gpu = 0, 0, 0
    for job_id in worker.running_jobs:
        job = state.get_job(job_id)
        if job:
            cpu += job.request.resources.cpu
            memory += parse_memory_string(job.request.resources.memory)
            gpu += get_gpu_count(job.request.resources.device)
    return cpu, memory, gpu

def worker_can_fit_job(state: ControllerState, worker: ControllerWorker, job: ControllerJob) -> bool:
    """Check if worker has sufficient capacity.

    Computes available headroom dynamically from running_jobs:
    1. CPU: job.cpu <= worker.total_cpu - committed_cpu
    2. Memory: job.memory <= worker.total_memory - committed_memory
    3. Device type: exact match (GPU job only on GPU worker)
    4. Device variant: if specified (not "auto"), must match worker
    5. GPU count: job.gpu_count <= available_gpus
    """

def find_worker_for_job(state, job) -> ControllerWorker | None:
    """Find first healthy worker that can fit the job."""
    for worker in state.get_available_workers():
        if worker_can_fit_job(state, worker, job):
            return worker
    return None
```

### 5. Scheduler Loop (`lib/fluster/src/fluster/cluster/controller/scheduler.py`)

New scheduling algorithm:

```python
def _schedule_pending_jobs(self) -> None:
    """Schedule pending jobs with resource-aware matching.

    New algorithm:
    1. Peek all pending jobs (don't pop)
    2. For each job in FIFO order:
       a. Check scheduling timeout - if expired, mark UNSCHEDULABLE
       b. Find a worker that can fit the job (headroom computed dynamically)
       c. If found: dispatch, remove from queue, add to worker.running_jobs
       d. If not found: skip to next job (DON'T block queue)
    """
    now_ms = int(time.time() * 1000)
    pending_jobs = self._state.peek_pending_jobs()

    for job in pending_jobs:
        if self._is_job_timed_out(job, now_ms):
            self._mark_unschedulable(job, now_ms)
            continue

        worker = find_worker_for_job(self._state, job)
        if not worker:
            continue  # Skip, don't block!

        success = self._dispatch_fn(job, worker)
        if success:
            self._handle_successful_dispatch(job, worker, now_ms)
        else:
            self._handle_failed_dispatch(job, worker)
```

Key helper methods:
- `_is_job_timed_out(job, now_ms)` - check if scheduling timeout exceeded
- `_mark_unschedulable(job, now_ms)` - set state, set error, remove from queue
- `_handle_successful_dispatch(job, worker, now_ms)` - update state, add to running_jobs, remove from queue

**Note:** No explicit resource commit/release needed - headroom is computed dynamically
from `worker.running_jobs` each time we check if a job fits.

### 6. Heartbeat Updates (`lib/fluster/src/fluster/cluster/controller/heartbeat.py`)

**No changes needed for resource tracking.** When jobs complete:
1. Heartbeat syncs terminal state from worker
2. Job is removed from `worker.running_jobs` (existing behavior)
3. Next scheduling pass automatically sees increased headroom

The dynamic computation approach means resource release is automatic when
`running_jobs.discard(job_id)` is called.

### 7. Types Update (`lib/fluster/src/fluster/cluster/types.py`)

Add UNSCHEDULABLE to terminal states:

```python
def is_job_finished(state: int) -> bool:
    return state in (
        JOB_STATE_SUCCEEDED, JOB_STATE_FAILED, JOB_STATE_KILLED,
        JOB_STATE_WORKER_FAILED, JOB_STATE_UNSCHEDULABLE,  # NEW
    )
```

## Test Plan

### New Test File: `tests/cluster/controller/test_resources.py`

- `test_parse_memory_string` - parameterized for '1g', '8g', '512m', etc.
- `test_parse_memory_string_invalid` - ValueError for bad input
- `test_get_device_type_cpu/gpu/tpu` - extract device type
- `test_get_device_variant` - extract variant or None
- `test_get_gpu_count` - extract count from DeviceConfig

### Updates to `tests/cluster/controller/test_workers.py`

Add resource matching tests:
- `test_worker_can_fit_job_cpu_constraint` - job.cpu > available
- `test_worker_can_fit_job_memory_constraint` - job.memory > available
- `test_worker_can_fit_job_device_type_mismatch` - GPU job on CPU worker
- `test_worker_can_fit_job_gpu_variant_match` - exact variant match
- `test_worker_can_fit_job_gpu_variant_auto` - "auto" matches any

### Updates to `tests/cluster/controller/test_scheduler.py`

- `test_scheduler_skips_jobs_that_dont_fit` - big job doesn't block small job
- `test_scheduler_marks_job_unschedulable_on_timeout` - timeout handling
- `test_scheduler_commits_resources_on_dispatch` - committed updated
- `test_scheduler_fifo_ordering_preserved` - jobs dispatched in order when possible

### Updates to `tests/cluster/controller/test_heartbeat.py`

- `test_heartbeat_releases_resources_on_completion` - committed decremented

### Tests to Remove (Obvious/Redundant)

From `test_state.py`:
- `test_controller_job_defaults` - validates default values
- `test_controller_worker_defaults` - validates default values

From `test_workers.py`:
- `test_load_workers_from_config_empty_list` - trivial

From `test_service.py`:
- `test_launch_job_returns_job_id` - obvious RPC behavior
- `test_list_jobs_empty` - trivial

## Example Updates (`lib/fluster/examples/cluster_example.py`)

### New Example: Resource Serialization

```python
def example_resource_scheduling(cluster: ClusterContext):
    """Demonstrate resource-aware scheduling with queuing."""
    def cpu_job(n):
        time.sleep(2)
        return n

    # Worker has 4 CPUs. Submit 4 jobs each requiring 2 CPUs.
    # Only 2 can run at a time, so jobs serialize in pairs.
    job_ids = [
        cluster.submit(cpu_job, i, name=f"job-{i}", resources={"cpu": 2})
        for i in range(4)
    ]

    for jid in job_ids:
        cluster.wait(jid)
```

### New Example: Scheduling Timeout

```python
def example_scheduling_timeout(cluster: ClusterContext):
    """Demonstrate scheduling timeout."""
    def impossible_job():
        pass

    job_id = cluster.submit(
        impossible_job,
        resources={"cpu": 100},  # More than any worker has
        scheduling_timeout_seconds=5,
    )
    status = cluster.wait(job_id)
    assert status["state"] == "JOB_STATE_UNSCHEDULABLE"
```

### Update `submit()` Method

Add `resources` and `scheduling_timeout_seconds` parameters to the `submit()` method.

## Implementation Order

1. Proto changes + regenerate bindings
2. `resources.py` - memory parsing, device helpers
3. `state.py` - CommittedResources, new methods
4. `workers.py` - worker_can_fit_job
5. `scheduler.py` - new scheduling loop
6. `heartbeat.py` - release resources on completion
7. `types.py` - add UNSCHEDULABLE to is_job_finished
8. Tests - add new, remove obvious
9. Examples - add resource scheduling demos

## Files Modified

- `lib/fluster/src/fluster/proto/cluster.proto`
- `lib/fluster/src/fluster/cluster/controller/state.py`
- `lib/fluster/src/fluster/cluster/controller/workers.py`
- `lib/fluster/src/fluster/cluster/controller/scheduler.py`
- `lib/fluster/src/fluster/cluster/controller/heartbeat.py`
- `lib/fluster/src/fluster/cluster/types.py`
- `lib/fluster/examples/cluster_example.py`
- `lib/fluster/tests/cluster/controller/test_workers.py`
- `lib/fluster/tests/cluster/controller/test_scheduler.py`
- `lib/fluster/tests/cluster/controller/test_heartbeat.py`
- `lib/fluster/tests/cluster/controller/test_state.py` (removals)
- `lib/fluster/tests/cluster/controller/test_service.py` (removals)

## New Files

- `lib/fluster/src/fluster/cluster/controller/resources.py`
- `lib/fluster/tests/cluster/controller/test_resources.py`

## Verification

1. Run existing tests: `uv run pytest lib/fluster/tests/cluster/controller/ -v`
2. Run the cluster example: `cd lib/fluster && uv run python examples/cluster_example.py`
3. Verify new examples demonstrate serialized scheduling
4. Verify scheduling timeout produces UNSCHEDULABLE state
