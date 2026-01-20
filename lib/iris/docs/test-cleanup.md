# Iris Test Cleanup Guidelines

This document captures principles for cleaning up and improving tests in the Iris codebase.

## Core Principles

### 1. Tests Should Exercise Public APIs, Not Internal State

**Bad Pattern:**
```python
def test_killing_job_with_retrying_task(service, state, job_request, worker_metadata):
    service.launch_job(job_request("test-job"), None)
    service.register_worker(...)

    # BAD: Directly manipulating internal state
    task = state.get_task(state.get_job_tasks(JobId("test-job"))[0].task_id)
    task.max_retries_preemption = 2  # Tinkering with internals
```

**Why This Is Wrong:**
- Tests become coupled to implementation details
- If we refactor internals, tests break even though behavior is unchanged
- The test doesn't verify that the real system can reach this state
- Masks bugs in the actual code paths that set up retry state

**Good Pattern:**
```python
def test_killing_job_with_retrying_task(service, job_request, worker_metadata):
    # Submit job with retry configuration through public API
    request = job_request("test-job", max_retries=2)
    service.launch_job(request, None)
    service.register_worker(...)

    # Cause a preemption through normal operations
    service.report_task_failure(task_id, preemption=True)

    # Now kill and verify
    service.terminate_job(...)
```

### 2. Follow the Same Path as Production Code

If a test needs a job in a specific state, it should reach that state through the same sequence of operations that would occur in production:

1. Submit job through `launch_job`
2. Register workers through `register_worker`
3. Let scheduler assign tasks through `tick()` or equivalent
4. Report completions/failures through normal APIs
5. Then test the behavior you care about

### 3. Prefer Integration-Style Tests Over Unit Tests with Mocked State

**Prefer:**
```python
def test_job_retry_on_preemption(local_client):
    """Test that jobs retry when preempted."""
    job = local_client.submit(entrypoint, "test-job", resources)
    # ... let it run, cause preemption, verify retry behavior
```

**Avoid:**
```python
def test_job_retry_on_preemption(state):
    """Test retry logic by setting up fake state."""
    job = create_fake_job()
    state.add_job(job)
    task = create_fake_task(state=RETRYING)
    state.add_task(task)
    # ... this doesn't test real behavior
```

### 4. Don't Create Tests That Validate Obvious Features

Tests should verify behavior, not structure:

**Don't test:**
- That a type exists
- That a constant has a specific value
- That a field is present on a dataclass

**Do test:**
- That submitting a job results in it being scheduled
- That killing a job terminates all its tasks
- That retry logic actually retries on the right failures

### 5. Use Fixtures for Common Setup, Not Shortcuts Around the API

Fixtures should encapsulate common API call sequences, not bypass them:

**Good fixture:**
```python
@pytest.fixture
def running_job(service, job_request, worker_metadata):
    """Create a job that has been scheduled and is running."""
    service.launch_job(job_request("test-job"), None)
    service.register_worker(worker_metadata())
    service.tick()  # Schedule the task
    return JobId("test-job")
```

**Bad fixture:**
```python
@pytest.fixture
def running_job(state):
    """Create a running job by manipulating state directly."""
    job = Job(job_id="test-job", state=RUNNING)
    state._jobs["test-job"] = job  # BAD: bypassing APIs
    return job
```

## Specific Anti-Patterns to Eliminate

### Direct State Manipulation
```python
# BAD
state.add_job(job)
task.max_retries_preemption = 2
state._workers["w1"] = worker

# GOOD
service.launch_job(request, None)
# Use job_request fixture with retry config
service.register_worker(request, None)
```

### Testing Implementation Details
```python
# BAD - tests internal data structure
assert state._pending_tasks["task-0"].retry_count == 1

# GOOD - tests observable behavior
status = service.get_job_status(job_id)
assert status.state == JOB_STATE_RUNNING  # Job survived the failure
```

### Overly Granular Unit Tests
```python
# BAD - testing that scheduler has a method
def test_scheduler_has_tick_method():
    scheduler = Scheduler()
    assert hasattr(scheduler, 'tick')

# GOOD - testing scheduler behavior
def test_scheduler_assigns_tasks_to_workers():
    # Submit job, register worker, tick, verify assignment
```

## Migration Strategy

When cleaning up existing tests:

1. **Identify tests that manipulate `state` directly** - these are candidates for rewriting
2. **Determine what behavior the test is actually verifying** - often the test name tells you
3. **Rewrite to use public APIs** to reach the same state
4. **If impossible via public API**, that's a signal the API may be incomplete - consider adding the missing capability
5. **Delete tests that only validate structure** - they add maintenance burden without catching bugs

## When Internal Access Is Acceptable

There are limited cases where accessing internals is reasonable:

1. **Verification after the fact** - reading internal state to verify an invariant holds (but prefer public status APIs)
2. **Test-only hooks** - explicit test interfaces like `_test_get_internal_state()` that are clearly marked
3. **Performance/stress tests** - where you need to set up large state quickly
4. **Unit tests of internal classes** - e.g., `test_job.py` testing `ControllerJob.on_task_transition()` directly is valid because it's testing that class's public API

Even in these cases, the *setup* should go through public APIs when possible.

---

## Concrete Findings (January 2026)

### Problematic Patterns Found in `lib/iris/tests/`

#### 1. Direct `task.max_retries_*` Mutation

**Files:**
- `lib/iris/tests/cluster/controller/test_service.py:531, 576, 629, 675, 712`
- `lib/iris/tests/cluster/controller/test_state.py:217, 286, 321`

```python
# BAD: Direct state manipulation
task = state.get_task(state.get_job_tasks(JobId("retry-job"))[0].task_id)
task.max_retries_preemption = 2
```

#### 2. Direct `state.add_job()` with Pre-configured State

**Files:**
- `lib/iris/tests/cluster/controller/test_dashboard.py:99-101, 118, 154-171, 191-203`
- `lib/iris/tests/cluster/controller/test_service.py:347`

```python
# BAD: Bypassing normal submission flow
state.add_job(ControllerJob(job_id=JobId("running"), request=job_request, state=cluster_pb2.JOB_STATE_RUNNING))
```

#### 3. Direct `job.state =` Assignment

**Files:**
- `lib/iris/tests/cluster/controller/test_state.py:532, 536`
- `lib/iris/tests/cluster/controller/test_job.py:246-247, 278-279, 323-325, 354-355, 368-369, 409, 425, 523, 546-547`

```python
# BAD: Should use job.mark_dispatched() or task transitions
job.state = cluster_pb2.JOB_STATE_RUNNING
```

#### 4. Direct `job.task_state_counts[]` Manipulation

**Files:**
- `lib/iris/tests/cluster/controller/test_job.py:389, 408, 424, 442, 465, 510, 522`

```python
# BAD: Manipulating internal counters
job.task_state_counts[cluster_pb2.TASK_STATE_PENDING] = 2
```

#### 5. Direct `worker.healthy = False`

**File:** `lib/iris/tests/cluster/controller/test_scheduler.py:289`

```python
# BAD: Should use WORKER_FAILED event
unhealthy_worker.healthy = False
```

### API Gaps Discovered

| Gap | Description | Suggested Fix |
|-----|-------------|---------------|
| **Task retry limits** | `LaunchJobRequest` lacks `max_retries_failure` and `max_retries_preemption` | Add fields to proto, propagate to tasks |
| **BUILDING state** | Cannot reach `JOB_STATE_BUILDING` through RPC | Accept direct setup for dashboard tests |
| **Worker health** | No RPC to mark worker unhealthy | Use `WORKER_FAILED` event (already exists) |

### Recommended Proto Changes

```protobuf
message LaunchJobRequest {
  // ... existing fields ...
  int32 max_retries_failure = 12;      // Per-task failure retry limit (default: 0)
  int32 max_retries_preemption = 13;   // Per-task preemption retry limit (default: 100)
}
```

### Rewrite Examples

**Before (test_scheduler.py):**
```python
unhealthy_worker.healthy = False
state.add_worker(unhealthy_worker)
```

**After:**
```python
state.add_worker(ControllerWorker(WorkerId("w2"), "addr2", worker_metadata()))
state.handle_event(Event(
    EventType.WORKER_FAILED,
    worker_id=WorkerId("w2"),
    error="Lost connection",
))
```

### Notes on `test_job.py`

The tests in `test_job.py` that manipulate `task_state_counts` and call `on_task_transition()` are **valid unit tests** of the `ControllerJob` state machine. They test the class's public method directly. These should be kept but clearly documented as unit tests of internal classes.

---

## Complete Review by Test Directory

### `lib/iris/tests/cluster/controller/` - NEEDS WORK

| File | Test Name | Issue | Suggested Fix | API Gap? |
|------|-----------|-------|---------------|----------|
| test_job.py | `test_gang_all_or_nothing_retry` | Direct `job.state = JOB_STATE_RUNNING` | Use `job.mark_dispatched()` | No |
| test_job.py | `test_gang_retry_success` | Direct `job.state = JOB_STATE_RUNNING` | Use `job.mark_dispatched()` | No |
| test_job.py | `test_gang_only_running_jobs_killed` | Direct `job.state =` assignments | Use `job.mark_dispatched()` for RUNNING jobs | No |
| test_job.py | `test_gang_tracks_correct_failure_type` | Direct `job.state = JOB_STATE_RUNNING` | Use `job.mark_dispatched()` | No |
| test_job.py | `test_gang_failure_respects_minimum_retry_budget` | Direct `job.state = JOB_STATE_RUNNING` | Use `job.mark_dispatched()` | No |
| test_job.py | `test_job_compute_job_state_*` (6 tests) | Direct `task_state_counts[]` manipulation | **Valid unit tests** of `on_task_transition()` | N/A |
| test_service.py | `test_terminate_job_skips_already_finished_children` | `state.add_job()` with state=SUCCEEDED | Document as exception | Yes - no way to mock SUCCEEDED child |
| test_service.py | `test_task_retry_preserves_attempt_history` | `task.max_retries_preemption = 2` | Pass via `LaunchJobRequest` | Yes - proto lacks field |
| test_service.py | `test_stale_worker_report_ignored_after_retry` | `task.max_retries_preemption = 2` | Pass via `LaunchJobRequest` | Yes - proto lacks field |
| test_service.py | `test_job_running_while_tasks_retry` | `task.max_retries_preemption = 2` | Pass via `LaunchJobRequest` | Yes - proto lacks field |
| test_service.py | `test_killing_job_with_retrying_task` | `task.max_retries_preemption = 2` | Pass via `LaunchJobRequest` | Yes - proto lacks field |
| test_service.py | `test_full_lifecycle_submit_fail_retry_succeed` | `task.max_retries_preemption = 2` | Pass via `LaunchJobRequest` | Yes - proto lacks field |
| test_state.py | `test_task_failure_with_retry_requeues` | `task.max_retries_failure = 1` | Pass via `ControllerJob` constructor | Partial |
| test_state.py | `test_worker_failure_cascades_to_running_tasks` | `task.max_retries_preemption = 1` | Pass via `ControllerJob` constructor | Partial |
| test_state.py | `test_dispatch_failure_marks_worker_failed_and_requeues_task` | `task.max_retries_preemption = 1` | Pass via `ControllerJob` constructor | Partial |
| test_state.py | `test_preemption_does_not_count_toward_max_task_failures` | `task.max_retries_preemption = 1` | Pass via `ControllerJob` constructor | Partial |
| test_state.py | `test_endpoint_visibility_by_job_state` | Direct `job.state =` assignment | Use event-based transitions | No |
| test_state.py | `test_namespace_isolation` | `state.add_job()` with state=RUNNING | Use event-based flow | Borderline |
| test_scheduler.py | `test_scheduler_skips_unhealthy_workers` | `worker.healthy = False` | Use `WORKER_FAILED` event | No |
| test_scheduler.py | `test_scheduler_considers_running_tasks_for_capacity` | Direct `job.state =` and `create_attempt()` | Use event-based flow | No |
| test_dashboard.py | 10+ tests | `state.add_job()` with preset states | **Accept for dashboard tests** | Partial - BUILDING unreachable |

### `lib/iris/tests/cluster/worker/` - MIXED

| File | Test Name | Issue | Suggested Fix | API Gap? |
|------|-----------|-------|---------------|----------|
| test_dashboard.py | `test_get_task_success` | `service._provider.get_task()`, `task.thread.join()` | Poll public API until status changes | No |
| test_dashboard.py | `test_get_logs_with_tail_parameter` | `task.logs.add()` directly | Accept - log injection for testing | Yes - no log injection API |
| test_dashboard.py | `test_get_logs_with_source_filter` | `task.logs.add()`, `task.should_stop = True` | Accept - log filtering test | Yes - no log injection API |
| test_dashboard.py | `test_fetch_task_logs_*` (3 tests) | `task.logs.add()` directly | Accept - log filtering tests | Yes - no log injection API |
| test_dashboard.py | `test_run_task_with_ports` | `service._provider.get_task()` | Use `GetTaskStatus` RPC | No |
| test_dashboard.py | `test_get_task_status_completed_task` | `task.thread.join()` | Poll `GetTaskStatus` | No |
| test_dashboard.py | `test_kill_task_*` (2 tests) | `task.thread.join()`, direct state access | Poll public API | Partial |
| test_worker.py | 10+ tests | `task.thread.join()` | Poll `worker.get_task()` | Borderline - impl detail |
| test_builder.py | `test_deps_hash_change_triggers_rebuild` | `builder._docker.exists()` | Add `ImageCache.exists()` public method | Yes |
| test_builder.py | `test_buildkit_cache_mounts` | `builder._docker.exists()` | Add `ImageCache.exists()` | Yes |
| test_builder.py | `test_lru_eviction_of_images` | `builder._docker.exists()` | Add `ImageCache.exists()` | Yes |
| test_bundle_cache.py | All tests | **GOOD** - uses public API | N/A | No |
| test_main.py | All tests | **GOOD** - uses CLI runner | N/A | No |
| test_runtime.py | All tests | **GOOD** - uses public API | N/A | No |

### `lib/iris/tests/cluster/client/` - GOOD

| File | Test Name | Issue | Suggested Fix | API Gap? |
|------|-----------|-------|---------------|----------|
| test_bundle.py | All 4 tests | Tests private `_get_git_non_ignored_files` | **Acceptable** - unit tests of internal helper | No |

### `lib/iris/tests/rpc/` - EXCELLENT

| File | Test Name | Issue | Suggested Fix | API Gap? |
|------|-----------|-------|---------------|----------|
| test_errors.py | All 3 tests | **GOOD** - uses public API only | N/A | No |

### `lib/iris/tests/actor/` - EXCELLENT

| File | Test Name | Issue | Suggested Fix | API Gap? |
|------|-----------|-------|---------------|----------|
| test_actor_pool.py | All tests | **GOOD** - uses public API | N/A | No |
| test_resolver.py | All tests | **GOOD** - uses public API + MockGcsApi | N/A | No |
| test_actor_e2e.py | Most tests | **GOOD** - uses public API | N/A | No |
| test_actor_e2e.py | `test_list_actors`, `test_list_methods*` | Calls server methods directly | Could add client methods | Minor |

### `lib/iris/tests/client/` - MIXED

| File | Test Name | Issue | Suggested Fix | API Gap? |
|------|-----------|-------|---------------|----------|
| test_worker_pool.py | `test_execute_*` (3 tests) | **GOOD** - unit tests of `TaskExecutorActor` | N/A | No |
| test_worker_pool.py | `test_dispatch_discovers_worker_endpoint` | Creates `WorkerState` directly | Use E2E test | Yes - no per-worker status API |
| test_worker_pool.py | `test_dispatch_executes_task_on_worker` | Creates `WorkerState`, `PendingTask` directly | Use E2E test | Partial |
| test_worker_pool.py | `test_dispatch_propagates_user_exceptions` | Creates `WorkerState` directly | Use E2E test | Partial |
| test_worker_pool.py | `test_dispatch_retries_on_infrastructure_failure` | Creates two `WorkerState` objects | Use E2E test | Yes - no failure injection |
| test_worker_pool.py | `TestWorkerPoolE2E::*` (10 tests) | **EXCELLENT** - uses public API only | N/A | No |

---

## Summary of API Gaps

| Gap | Description | Affected Tests | Priority |
|-----|-------------|----------------|----------|
| `LaunchJobRequest` retry fields | No `max_retries_failure`/`max_retries_preemption` | 5 tests in test_service.py | **High** |
| Log injection API | No way to inject logs for filter testing | 5 tests in worker/test_dashboard.py | Medium |
| `ImageCache.exists()` | No public method to check cached images | 3 tests in test_builder.py | Low |
| Per-worker status API | No way to observe individual worker discovery | 4 tests in test_worker_pool.py | Low |
| BUILDING state | Cannot reach via RPC | 1 test in controller/test_dashboard.py | Accept |

## Verdict by Directory

| Directory | Status | Action |
|-----------|--------|--------|
| `tests/rpc/` | ✅ Excellent | None needed |
| `tests/actor/` | ✅ Excellent | None needed |
| `tests/cluster/client/` | ✅ Good | None needed |
| `tests/client/` | ⚠️ Mixed | E2E tests excellent; unit tests acceptable |
| `tests/cluster/worker/` | ⚠️ Mixed | Some API gaps; `task.thread.join()` pattern |
| `tests/cluster/controller/` | ❌ Needs Work | Many direct state manipulations; proto gaps |

---

## Implementation Plan: Add Retry Fields to Proto

### Overview

This plan addresses the high-priority API gap: `LaunchJobRequest` lacks `max_retries_failure` and `max_retries_preemption` fields, forcing 5+ tests in `test_service.py` to directly manipulate `task.max_retries_preemption`.

### Step 1: Proto Changes

**File:** `lib/iris/src/iris/rpc/cluster.proto`

Add two new fields to `LaunchJobRequest` (lines 246-273):

```protobuf
message LaunchJobRequest {
  string name = 1;
  bytes serialized_entrypoint = 2;
  ResourceSpecProto resources = 3;
  EnvironmentConfig environment = 4;

  string bundle_gcs_path = 5;
  string bundle_hash = 6;
  bytes bundle_blob = 7;

  int32 scheduling_timeout_seconds = 8;
  repeated string ports = 9;
  string parent_job_id = 10;
  int32 max_task_failures = 11;

  // NEW: Per-task retry limits
  // When a task fails, it retries up to this many times before being marked as permanently failed
  int32 max_retries_failure = 12;      // Default: 0 (no retries on task failure)

  // When a task is preempted (worker dies), it retries up to this many times
  int32 max_retries_preemption = 13;   // Default: 100 (generous preemption tolerance)
}
```

After modifying the proto, regenerate the Python bindings:

```bash
cd lib/iris && uv run python -m grpc_tools.protoc \
  -I src/iris/rpc \
  --python_out=src/iris/rpc \
  --pyi_out=src/iris/rpc \
  src/iris/rpc/cluster.proto
```

### Step 2: Controller State Changes

**File:** `lib/iris/src/iris/cluster/controller/state.py`

#### 2a. Update `_on_job_submitted` handler (lines 1004-1029)

The job submission handler creates a `ControllerJob` but currently doesn't read retry fields from the request. Update to pass through the proto fields:

```python
def _on_job_submitted(self, txn: TransactionLog, event: Event) -> None:
    assert event.job_id and event.request
    parent_job_id = JobId(event.request.parent_job_id) if event.request.parent_job_id else None

    # Read retry limits from request, using defaults if not set
    max_retries_failure = event.request.max_retries_failure  # proto default: 0
    max_retries_preemption = event.request.max_retries_preemption or 100  # default: 100

    job = ControllerJob(
        job_id=event.job_id,
        request=event.request,
        submitted_at_ms=event.timestamp_ms or now_ms(),
        parent_job_id=parent_job_id,
        max_retries_failure=max_retries_failure,
        max_retries_preemption=max_retries_preemption,
    )
    # ... rest unchanged
```

**Note:** The `expand_job_to_tasks()` function (lines 704-731) already copies `max_retries_failure` and `max_retries_preemption` from the job to each task:

```python
task = ControllerTask(
    task_id=task_id,
    job_id=job.job_id,
    task_index=i,
    max_retries_failure=job.max_retries_failure,      # Already propagated
    max_retries_preemption=job.max_retries_preemption,  # Already propagated
    submitted_at_ms=job.submitted_at_ms,
)
```

### Step 3: Test Fixture Update

**File:** `lib/iris/tests/cluster/controller/test_service.py`

Update the `job_request` fixture to accept retry parameters:

```python
@pytest.fixture
def job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(
        name: str = "test-job",
        replicas: int = 1,
        parent_job_id: str | None = None,
        max_retries_failure: int = 0,
        max_retries_preemption: int = 0,  # Default to 0 for tests (no implicit retries)
    ) -> cluster_pb2.Controller.LaunchJobRequest:
        return cluster_pb2.Controller.LaunchJobRequest(
            name=name,
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3, replicas=replicas),
            environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
            parent_job_id=parent_job_id or "",
            max_retries_failure=max_retries_failure,
            max_retries_preemption=max_retries_preemption,
        )

    return _make
```

### Step 4: Test Rewrites

#### 4a. `test_task_retry_preserves_attempt_history` (line 508)

**Before:**
```python
def test_task_retry_preserves_attempt_history(service, state, job_request, worker_metadata):
    """Verify that when a task fails and retries, attempt history is preserved."""
    request = cluster_pb2.Controller.LaunchJobRequest(
        name="retry-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    service.launch_job(request, None)
    # ...
    task = state.get_task(state.get_job_tasks(JobId("retry-job"))[0].task_id)
    task.max_retries_preemption = 2  # BAD: Direct state manipulation
```

**After:**
```python
def test_task_retry_preserves_attempt_history(service, state, job_request, worker_metadata):
    """Verify that when a task fails and retries, attempt history is preserved."""
    # Launch job with retry enabled via proto field
    service.launch_job(job_request("retry-job", max_retries_preemption=2), None)

    # Register worker
    service.register_worker(
        cluster_pb2.Controller.RegisterWorkerRequest(
            worker_id="w1",
            address="host:8080",
            metadata=worker_metadata(),
        ),
        None,
    )

    # Get task - no manipulation needed, retry config came from proto
    task = state.get_task(state.get_job_tasks(JobId("retry-job"))[0].task_id)

    # First attempt: dispatch and fail
    dispatch_task(state, task, WorkerId("w1"))
    # ... rest unchanged
```

#### 4b. `test_stale_worker_report_ignored_after_retry` (line 559)

**Before:**
```python
def test_stale_worker_report_ignored_after_retry(service, state, job_request, worker_metadata):
    service.launch_job(job_request("test-job"), None)
    # ...
    task = state.get_task(state.get_job_tasks(JobId("test-job"))[0].task_id)
    task.max_retries_preemption = 2  # BAD
```

**After:**
```python
def test_stale_worker_report_ignored_after_retry(service, state, job_request, worker_metadata):
    # Launch with retry config via proto
    service.launch_job(job_request("test-job", max_retries_preemption=2), None)
    # ...
    task = state.get_task(state.get_job_tasks(JobId("test-job"))[0].task_id)
    # No manipulation needed - retry config already set
```

#### 4c. `test_job_running_while_tasks_retry` (line 612)

**Before:**
```python
def test_job_running_while_tasks_retry(service, state, job_request, worker_metadata):
    service.launch_job(job_request("test-job"), None)
    # ...
    task = state.get_task(state.get_job_tasks(JobId("test-job"))[0].task_id)
    task.max_retries_preemption = 2  # BAD
```

**After:**
```python
def test_job_running_while_tasks_retry(service, state, job_request, worker_metadata):
    service.launch_job(job_request("test-job", max_retries_preemption=2), None)
    # ...
    task = state.get_task(state.get_job_tasks(JobId("test-job"))[0].task_id)
    # No manipulation needed
```

#### 4d. `test_killing_job_with_retrying_task` (line 658)

**Before:**
```python
def test_killing_job_with_retrying_task(service, state, job_request, worker_metadata):
    service.launch_job(job_request("test-job"), None)
    # ...
    task = state.get_task(state.get_job_tasks(JobId("test-job"))[0].task_id)
    task.max_retries_preemption = 2  # BAD
```

**After:**
```python
def test_killing_job_with_retrying_task(service, state, job_request, worker_metadata):
    service.launch_job(job_request("test-job", max_retries_preemption=2), None)
    # ...
    task = state.get_task(state.get_job_tasks(JobId("test-job"))[0].task_id)
    # No manipulation needed
```

#### 4e. `test_full_lifecycle_submit_fail_retry_succeed` (line 695)

**Before:**
```python
def test_full_lifecycle_submit_fail_retry_succeed(service, state, job_request, worker_metadata):
    service.launch_job(job_request("lifecycle-job"), None)
    # ...
    task = state.get_task(state.get_job_tasks(JobId("lifecycle-job"))[0].task_id)
    task.max_retries_preemption = 2  # BAD
```

**After:**
```python
def test_full_lifecycle_submit_fail_retry_succeed(service, state, job_request, worker_metadata):
    service.launch_job(job_request("lifecycle-job", max_retries_preemption=2), None)
    # ...
    task = state.get_task(state.get_job_tasks(JobId("lifecycle-job"))[0].task_id)
    # No manipulation needed
```

### Step 5: Additional Test Updates (test_state.py)

**File:** `lib/iris/tests/cluster/controller/test_state.py`

These tests use `state.add_job()` directly, so they can pass retry config to `ControllerJob`:

#### 5a. `test_task_failure_with_retry_requeues` (line 217)

**Before:**
```python
task.max_retries_failure = 1
```

**After:** Pass `max_retries_failure=1` to `ControllerJob` constructor when creating the job.

#### 5b. `test_worker_failure_cascades_to_running_tasks` (line 286)

**Before:**
```python
task.max_retries_preemption = 1
```

**After:** Pass `max_retries_preemption=1` to `ControllerJob` constructor.

#### 5c. `test_dispatch_failure_marks_worker_failed_and_requeues_task` (line 321)

**Before:**
```python
task.max_retries_preemption = 1
```

**After:** Pass `max_retries_preemption=1` to `ControllerJob` constructor.

#### 5d. `test_preemption_does_not_count_toward_max_task_failures` (line 458)

**Before:**
```python
tasks[0].max_retries_preemption = 1
```

**After:** Pass `max_retries_preemption=1` to `ControllerJob` constructor.

### Summary of Changes

| File | Change |
|------|--------|
| `cluster.proto` | Add `max_retries_failure` (field 12) and `max_retries_preemption` (field 13) to `LaunchJobRequest` |
| `state.py` | Update `_on_job_submitted` to read retry fields from request |
| `test_service.py` | Update `job_request` fixture; remove 5 instances of `task.max_retries_*` manipulation |
| `test_state.py` | Update 4 tests to pass retry config via `ControllerJob` constructor |

### Verification

After making changes, run:

```bash
cd lib/iris && uv run pytest tests/cluster/controller/test_service.py tests/cluster/controller/test_state.py -v
```

All retry-related tests should pass without any direct `task.max_retries_*` manipulation.
