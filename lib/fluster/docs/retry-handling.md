# Retry Handling Design

This document describes how job retry handling works in fluster and proposes changes to expose user-configurable retries.

## Current State Analysis

### What Happens Today When a Job Fails

Fluster has two types of failures with different retry behaviors:

1. **Worker failures (preemption)**: When a worker dies or times out, the controller automatically retries jobs up to `max_retries_preemption` (default: 100). This handles preemption gracefully in cloud environments.

2. **Job failures**: When a job exits with non-zero code, it is marked FAILED with no retry by default (`max_retries_failure` = 0).

### How Job State Transitions Work

The `Job` class in `src/fluster/cluster/controller/job.py` owns its state machine:

```python
@dataclass
class Job:
    state: int = cluster_pb2.JOB_STATE_PENDING
    failure_count: int = 0
    preemption_count: int = 0
    max_retries_failure: int = 0      # No failure retries by default
    max_retries_preemption: int = 100  # Many preemption retries
    current_attempt_id: int = 0
    attempts: list[JobAttempt] = field(default_factory=list)
```

State transitions flow through `Job.transition()`:

```
PENDING → RUNNING → SUCCEEDED (terminal)
                  → FAILED → PENDING (if retry) or FAILED (terminal)
                  → WORKER_FAILED → PENDING (if retry) or WORKER_FAILED (terminal)
                  → KILLED (terminal)
```

When `transition()` is called with a failure state, it:
1. Increments the appropriate counter (`failure_count` or `preemption_count`)
2. Checks if retries are available
3. Returns `TransitionResult.SHOULD_RETRY` or `TransitionResult.EXCEEDED_RETRY_LIMIT`
4. If retrying, saves current attempt to history and resets state to PENDING

### What Information Is Tracked About Failures

Each job tracks:
- `failure_count`: Number of job failures (non-zero exit code)
- `preemption_count`: Number of worker failures (worker died/timeout)
- `current_attempt_id`: 0-indexed attempt number
- `attempts`: List of `JobAttempt` records for all previous attempts

Each `JobAttempt` contains:
- `attempt_id`, `worker_id`
- `state`, `exit_code`, `error`
- `started_at_ms`, `finished_at_ms`
- `is_worker_failure`: True if this was a preemption

The proto `JobStatus` exposes all this information to clients via `GetJobStatus`.

### Current Gap

The retry infrastructure is fully implemented internally, but there is **no way for users to configure `max_retries_failure`** when submitting jobs. The `LaunchJobRequest` proto has no retry fields, and the client `submit()` method has no retry parameter.

## Proposed Retry Design

### Proto Changes

Add retry configuration to `LaunchJobRequest`:

```protobuf
message LaunchJobRequest {
    // ... existing fields ...

    // Retry configuration
    RetryConfig retry = 11;
}

message RetryConfig {
    // Maximum retries for job failures (exit code != 0)
    // 0 means no retries (one attempt only). Default: 0
    int32 max_retries_failure = 1;

    // Maximum retries for worker failures (preemption, timeout)
    // Default: 100
    int32 max_retries_preemption = 2;

    // Initial delay between retries in milliseconds
    // Default: 1000 (1 second)
    int64 initial_retry_delay_ms = 3;

    // Maximum delay between retries in milliseconds (for exponential backoff)
    // Default: 60000 (60 seconds)
    int64 max_retry_delay_ms = 4;

    // Multiplier for exponential backoff. Default: 2.0
    double backoff_multiplier = 5;

    // Only retry on these exit codes. Empty means retry on any non-zero exit.
    repeated int32 retry_on_exit_codes = 6;

    // Don't retry on these exit codes (takes precedence over retry_on_exit_codes)
    repeated int32 no_retry_on_exit_codes = 7;
}
```

### Client Interface Changes

Add retry configuration to `RpcClusterClient.submit()`:

```python
@dataclass
class RetryConfig:
    """Job retry configuration.

    Args:
        max_retries_failure: Max retries for job failures (default: 0, no retry)
        max_retries_preemption: Max retries for worker failures (default: 100)
        initial_delay: Initial delay between retries (default: 1.0 seconds)
        max_delay: Maximum delay for exponential backoff (default: 60.0 seconds)
        backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
        retry_on_exit_codes: Only retry on these exit codes (empty = any non-zero)
        no_retry_on_exit_codes: Never retry on these exit codes
    """
    max_retries_failure: int = 0
    max_retries_preemption: int = 100
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    retry_on_exit_codes: list[int] = field(default_factory=list)
    no_retry_on_exit_codes: list[int] = field(default_factory=list)


def submit(
    self,
    entrypoint: Entrypoint,
    name: str,
    resources: cluster_pb2.ResourceSpec,
    environment: cluster_pb2.EnvironmentConfig | None = None,
    ports: list[str] | None = None,
    scheduling_timeout_seconds: int = 0,
    retry: RetryConfig | None = None,  # NEW
) -> JobId:
```

### Controller/Scheduler Changes

The controller already handles retry logic correctly. Changes needed:

1. **Service layer** (`service.py`): Parse `RetryConfig` from request and set fields on `Job`:

```python
def launch_job(self, request, ctx):
    retry = request.retry
    job = Job(
        job_id=JobId(job_id),
        request=request,
        max_retries_failure=retry.max_retries_failure if retry else 0,
        max_retries_preemption=retry.max_retries_preemption if retry else 100,
        # ... other fields ...
    )
```

2. **Job class** (`job.py`): Add delay tracking for exponential backoff:

```python
@dataclass
class Job:
    # Existing fields...

    # Retry delay configuration
    initial_retry_delay_ms: int = 1000
    max_retry_delay_ms: int = 60000
    backoff_multiplier: float = 2.0
    retry_on_exit_codes: list[int] = field(default_factory=list)
    no_retry_on_exit_codes: list[int] = field(default_factory=list)

    # Computed delay state
    next_retry_delay_ms: int = 0
    retry_available_at_ms: int = 0  # When job can be retried
```

3. **Scheduler** (`scheduler.py`): Respect retry delays:

```python
def find_assignments(self, pending_jobs, workers, now_ms):
    for job in pending_jobs:
        # Skip jobs waiting for retry delay
        if job.retry_available_at_ms > now_ms:
            continue
        # ... existing scheduling logic ...
```

4. **Exit code filtering** in `Job._handle_failure()`:

```python
def _should_retry_exit_code(self, exit_code: int) -> bool:
    if exit_code in self.no_retry_on_exit_codes:
        return False
    if self.retry_on_exit_codes:
        return exit_code in self.retry_on_exit_codes
    return True  # Retry any non-zero by default
```

### Worker Changes

No changes needed. The worker already:
- Reports job state via `ReportJobState` RPC with exit code
- Includes `attempt_id` in requests and reports
- Creates isolated workdirs per attempt (`{job_id}_attempt_{attempt_id}`)

### State Tracking

The existing state tracking is sufficient:
- `JobStatus.current_attempt_id` shows current attempt
- `JobStatus.attempts` contains full history
- `JobStatus.failure_count` and `preemption_count` track totals

Add to `JobStatus` for visibility:
```protobuf
// Retry configuration (echoed back for visibility)
int32 max_retries_failure = 19;
int32 max_retries_preemption = 20;
int64 retry_available_at_ms = 21;  // When next retry can be scheduled (0 if ready)
```

## Configuration Options

### Per-Job Configuration

```python
# Basic: retry up to 3 times on any failure
job_id = client.submit(
    ...,
    retry=RetryConfig(max_retries_failure=3)
)

# With exponential backoff: 1s, 2s, 4s, 8s delays
job_id = client.submit(
    ...,
    retry=RetryConfig(
        max_retries_failure=5,
        initial_delay=1.0,
        max_delay=30.0,
        backoff_multiplier=2.0,
    )
)

# Only retry on specific exit codes (e.g., OOM = 137)
job_id = client.submit(
    ...,
    retry=RetryConfig(
        max_retries_failure=3,
        retry_on_exit_codes=[137, 139],  # SIGKILL (OOM), SIGSEGV
    )
)

# Don't retry on assertion errors (user bugs)
job_id = client.submit(
    ...,
    retry=RetryConfig(
        max_retries_failure=3,
        no_retry_on_exit_codes=[1],  # Generic error, likely a bug
    )
)
```

### Global Defaults

Could add to `ControllerConfig` for cluster-wide defaults:
```python
@dataclass
class ControllerConfig:
    # ... existing ...
    default_max_retries_failure: int = 0
    default_max_retries_preemption: int = 100
    default_retry_delay_ms: int = 1000
```

## Edge Cases

### Job Killed vs Job Failed

- **KILLED** (via `TerminateJob`): No retry. User explicitly requested termination.
- **FAILED** (exit code != 0): Retry based on `max_retries_failure` and exit code filters.
- **WORKER_FAILED** (preemption): Retry based on `max_retries_preemption`.

### Worker Failures vs Job Failures

Worker failures and job failures use separate counters:
- A job with `max_retries_failure=3` and `max_retries_preemption=100` can fail 3 times AND be preempted 100 times before final failure.
- This matches cloud provider behavior where preemption is "free" from a retry budget perspective.

### Resource Cleanup Between Retries

Current cleanup is already correct:
- Worker creates isolated workdir per attempt: `{job_id}_attempt_{attempt_id}`
- Workdir is cleaned up when job reaches terminal state
- Ports are released and reallocated on each attempt
- Container is removed after each attempt

### Idempotency Considerations

Jobs should be idempotent to benefit from retries. The `FLUSTER_ATTEMPT_ID` environment variable is already injected, allowing jobs to:
- Skip already-completed work
- Resume from checkpoints
- Use attempt-specific output paths

Example:
```python
def my_job():
    attempt = int(os.environ.get("FLUSTER_ATTEMPT_ID", "0"))
    checkpoint_path = f"gs://bucket/job-{os.environ['FLUSTER_JOB_ID']}/checkpoint_{attempt}.pkl"

    # Resume from previous checkpoint if exists
    if attempt > 0:
        previous_checkpoint = f"gs://bucket/job-{os.environ['FLUSTER_JOB_ID']}/checkpoint_{attempt-1}.pkl"
        if exists(previous_checkpoint):
            state = load(previous_checkpoint)
```

## Implementation Plan

### Phase 1: Proto and Basic Retry (1-2 days)

1. Add `RetryConfig` message to `cluster.proto`
2. Add `retry` field to `LaunchJobRequest`
3. Update `ControllerServiceImpl.launch_job()` to parse retry config
4. Add `RetryConfig` dataclass to client
5. Update `RpcClusterClient.submit()` signature

### Phase 2: Exponential Backoff (1 day)

1. Add delay fields to `Job` dataclass
2. Add `retry_available_at_ms` tracking
3. Update `Scheduler.find_assignments()` to respect delays
4. Add `_compute_next_retry_delay()` to `Job`

### Phase 3: Exit Code Filtering (0.5 days)

1. Add `retry_on_exit_codes` and `no_retry_on_exit_codes` to proto and Job
2. Update `Job._handle_failure()` to check exit code filters
3. Worker already reports exit codes correctly

### Phase 4: Testing and Documentation (1 day)

1. Add integration test: job that fails N-1 times then succeeds
2. Add unit tests for exit code filtering
3. Add unit tests for exponential backoff delays
4. Update demo notebook with retry example
5. Update API documentation

### Testing Strategy

**Unit tests** (in `tests/cluster/controller/test_job.py`):
- Retry config propagation from request to Job
- Exit code filtering logic
- Backoff delay calculation
- `retry_available_at_ms` computation

**Integration tests** (in `tests/cluster/test_e2e.py`):
- Job that fails then succeeds after retry
- Job that exceeds retry limit
- Preemption retry (worker timeout)
- Exit code filtering (retry only on specific codes)

**Demo notebook** example:
```python
import random

def flaky_job():
    """Job that fails ~50% of the time."""
    if random.random() < 0.5:
        raise RuntimeError("Random failure")
    return "success"

job_id = client.submit(
    entrypoint=Entrypoint.from_callable(flaky_job),
    name="flaky-job",
    resources=cluster_pb2.ResourceSpec(cpu=1, memory="512m"),
    retry=RetryConfig(max_retries_failure=5),
)

status = client.wait(job_id, timeout=60.0)
assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
print(f"Succeeded after {status.failure_count + 1} attempts")
```

## Summary

The retry infrastructure already exists in fluster's controller. This design exposes it to users via:
1. New proto `RetryConfig` message
2. New `retry` parameter on `RpcClusterClient.submit()`
3. Exponential backoff delay support in scheduler
4. Exit code filtering for fine-grained retry control

Total estimated effort: 3-4 days for full implementation.
