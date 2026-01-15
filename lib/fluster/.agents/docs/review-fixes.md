# Review Fixes: JobAttempt Implementation

Following senior code review of the JobAttempt feature, these fixes address code quality issues.

## Fixes Applied

### 1. Remove `getattr()` hack in Job class

**Problem:** `_last_failure_was_worker` was a hidden instance variable accessed via `getattr()`, violating AGENTS.md guidelines against hasattr-style hacks.

**Fix:** Pass `is_worker_failure` as parameter to `_reset_for_retry()`.

**Files:** `lib/fluster/src/fluster/cluster/controller/job.py`

```python
# Before (bad):
self._last_failure_was_worker = is_worker_failure
# ...later...
is_worker_failure=getattr(self, "_last_failure_was_worker", False)

# After (good):
if can_retry:
    self._reset_for_retry(is_worker_failure=is_worker_failure)
```

### 2. Remove `hasattr()` in WorkerDashboard

**Problem:** `hasattr(self, "_server")` used instead of proper initialization.

**Fix:** Initialize `self._server = None` in `__init__`.

**Files:** `lib/fluster/src/fluster/cluster/worker/dashboard.py`

### 3. Remove `namespace` from endpoints API

**Problem:** Dashboard HTML referenced `e.namespace` but the endpoint registry no longer uses explicit namespaces (namespacing is implicit via job_id prefixes).

**Fix:** Remove namespace column from dashboard HTML.

**Files:** `lib/fluster/src/fluster/cluster/controller/dashboard.py`

### 4. Fix Worker `Job.to_proto()` to include `current_attempt_id`

**Problem:** Worker's `Job.to_proto()` didn't set the `current_attempt_id` field.

**Fix:** Add `current_attempt_id=self.attempt_id` to proto conversion.

**Files:** `lib/fluster/src/fluster/cluster/worker/worker_types.py`

**Tests:** Added round-trip test in `lib/fluster/tests/cluster/controller/test_job.py`

### 5. Rename `TransitionResult` enum values for clarity

**Problem:** Generic names `OK`, `RETRY`, `NO_RETRY` were not self-documenting.

**Fix:** Renamed to `COMPLETE`, `SHOULD_RETRY`, `EXCEEDED_RETRY_LIMIT`.

**Files:**
- `lib/fluster/src/fluster/cluster/controller/job.py`
- All callers updated

## Not Changed

- **Log level for stale reports:** Kept at WARNING per user preference
