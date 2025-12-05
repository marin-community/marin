# Executor Cleanup: Heartbeat-Based Locking

## Summary

Replaced the Ray-based `StatusActor` with a filesystem-based heartbeat locking system that works with any storage backend.

## Changes Made

### 1. `lib/marin/src/marin/execution/executor_step_status.py`
- Added `HEARTBEAT_INTERVAL = 30` and `HEARTBEAT_TIMEOUT = 90` constants
- Added `is_heartbeat_stale(event: ExecutorStepEvent) -> bool` function

### 2. `lib/marin/src/marin/execution/step_runner.py` (NEW)
- Created `StepRunner` class that wraps Fray job execution with automatic heartbeat
- Heartbeat thread periodically appends `RUNNING` events to `.executor_status`
- Provides `launch()`, `poll()`, `wait()` interface

### 3. `lib/marin/src/marin/execution/executor.py`
- Removed `StatusActor` import and usage
- Removed `self.job_ctx` (no longer needed)
- Changed `self.jobs` to `self.step_runners`
- Updated `_run_steps()` to use `StepRunner` instead of raw `JobId`
- Updated `_launch_step()` to return `StepRunner | None`
- Replaced old `_get_task_status()`, `get_status()`, `should_run()` with new heartbeat-based `should_run()`
- Added `PreviousTaskFailedError` class (was in status_actor.py)

### 4. `lib/marin/src/marin/execution/status_actor.py`
- Deleted entirely

### 5. `lib/fray/src/fray/cluster/ray/cluster.py`
- Fixed to use `JobStatus` enum values instead of string literals (separate fix)

## New Locking Protocol

1. Read `.executor_status` events
2. If last status is `SUCCESS` → skip (already done)
3. If last status is `FAILED` → raise error (unless `force_run_failed`)
4. If last status is `RUNNING`:
   - Check if heartbeat is stale (timestamp > 90s old)
   - If not stale → wait (another worker is active)
   - If stale → owner died, take over
5. Write `RUNNING` with our `worker_id`
6. Verify we won the race (our `worker_id` is last)
7. Start executing with heartbeat thread

## Remaining Work

- [ ] Run tests to verify changes work
- [ ] May need to update test fixtures (currently use `ray_tpu_cluster`)

## Test Command

```bash
uv run pytest tests/test_executor.py::test_force_run_failed -v
```
