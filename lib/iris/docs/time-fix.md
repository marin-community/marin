# Time Migration Fix Plan

The branch introduced `Timestamp`, `Duration`, `Deadline`, `Timer` in `time_utils.py` and partially migrated
the codebase. This plan fixes the incomplete/inconsistent migration.

## Core Design Principles

- **`Duration`**: time intervals, configuration values, proto fields representing "how long"
- **`Deadline`**: monotonic-clock-based "point in the future", for timeout checking. Created from `Deadline.from_now(duration)`.
  A `Duration` in a proto becomes a `Deadline` when we start tracking it.
- **`Timestamp`**: wall-clock event times (when something happened), stored as epoch_ms
- **`Timer`**: measuring elapsed time (replaces `time.time() - start` and `time.monotonic() - start` patterns)
- Internal state uses `Timestamp`/`Deadline` directly — NOT `common_pb2.Timestamp`. Proto messages are a serialization boundary.
- `__repr__` on all time classes so logging "just works" without extracting raw values.

## Conversion Patterns Reference

These patterns appear repeatedly. Each stage applies them to its files.

### Pattern A: `__repr__` / `__str__` for logging
Add to `Timestamp`, `Duration`, `Deadline`, `Timer`:
```python
class Timestamp:
    def __repr__(self) -> str: return f"Timestamp({self.as_formatted_date()})"
class Duration:
    def __repr__(self) -> str: return f"Duration({self.to_seconds():.3f}s)"
class Deadline:
    def __repr__(self) -> str: return f"Deadline(remaining={self.remaining_seconds():.3f}s)"
```
Eliminates all `f"... ({timeout_seconds}s)"` manual formatting throughout.

### Pattern B: `Timestamp.from_proto` / `to_proto` operate on proto messages
```python
# Before (double-wrapping everywhere):
common_pb2.Timestamp(epoch_ms=Timestamp.now().to_proto())
Timestamp.from_proto(task.submitted_at.epoch_ms)

# After:
Timestamp.now().to_proto()   # returns common_pb2.Timestamp
Timestamp.from_proto(task.submitted_at)  # accepts common_pb2.Timestamp
```

### Pattern C: `Deadline.from_now(duration)` replaces manual milliseconds extraction
```python
# Before:
timeout_seconds = self.request.timeout.milliseconds / 1000
deadline = Deadline.from_seconds(timeout_seconds)

# After:
deadline = Deadline.from_now(Duration.from_proto(self.request.timeout))
```

### Pattern D: Internal state stores `Timestamp`, not `common_pb2.Timestamp`
```python
# Before (state.py):
started_at: common_pb2.Timestamp | None = None
self.started_at = common_pb2.Timestamp(epoch_ms=Timestamp.now().to_proto())

# After:
started_at: Timestamp | None = None
self.started_at = Timestamp.now()
```
Proto conversion happens at the serialization boundary (service.py).

### Pattern E: `CopyFrom` for proto assignment instead of field-by-field
```python
# Before:
proto_attempt.started_at.epoch_ms = attempt.started_at.epoch_ms

# After (with Pattern D, started_at is now Timestamp):
proto_attempt.started_at.CopyFrom(attempt.started_at.to_proto())
```

### Pattern F: `Timer` replaces `time.time() - start` / `time.monotonic() - start`
```python
# Before:
start = time.time()
...
build_time_ms = int((time.time() - start) * 1000)

# After:
timer = Timer()
...
build_time_ms = timer.elapsed_ms()
```

### Pattern G: Scheduling timeout as `Deadline` on ControllerJob
```python
# Before (scheduler.py):
timeout = Duration.from_ms(job.request.scheduling_timeout.milliseconds)
pending_duration_ms = Timestamp.from_proto(task.submitted_at.epoch_ms).age_ms()
return pending_duration_ms > timeout.to_ms()

# After: ControllerJob stores a Deadline, computed at submission time
class ControllerJob:
    scheduling_deadline: Deadline | None = None  # set from Duration in request at submit time

# scheduler.py:
def _is_task_timed_out(self, task, job) -> bool:
    return job.scheduling_deadline is not None and job.scheduling_deadline.expired()
```

### Pattern H: `_format_timestamp` in cli.py uses `Timestamp`
```python
# Before:
def _format_timestamp(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc)
    return dt.strftime(...)

# After:
def _format_timestamp(ms: int) -> str:
    if ms == 0: return "-"
    return Timestamp.from_ms(ms).as_formatted_date()
```

### Pattern I: Client APIs accept `Duration | None`, not raw int seconds
```python
# Before:
def submit(..., scheduling_timeout_seconds: int = 0, timeout_seconds: int = 0):
    request.scheduling_timeout.milliseconds = scheduling_timeout_seconds * 1000

# After:
def submit(..., scheduling_timeout: Duration | None = None, timeout: Duration | None = None):
    if scheduling_timeout:
        request.scheduling_timeout.CopyFrom(scheduling_timeout.to_proto())
```

### Pattern J: `WorkerConfig` uses `Duration`, `_serve()` uses `Deadline`
```python
# Before:
heartbeat_timeout_seconds: float = 60.0
...
elapsed = time.monotonic() - self._last_heartbeat_time
if elapsed > heartbeat_timeout: ...

# After:
heartbeat_timeout: Duration = Duration.from_seconds(60)
...
# Reset deadline on each heartbeat
self._heartbeat_deadline = Deadline.from_now(self._config.heartbeat_timeout)
...
if self._heartbeat_deadline.expired(): ...
```

### Pattern K: `client.py` log formatting uses `Timestamp`
```python
# Before:
ts = datetime.fromtimestamp(entry.timestamp.epoch_seconds(), tz=timezone.utc)
ts_str = ts.strftime("%H:%M:%S")

# After (Timestamp already has as_formatted_date; add a short format):
ts_str = Timestamp.from_proto(entry.timestamp).as_short_time()  # "HH:MM:SS"
```

---

## Execution Stages

### Stage 1: Fix `time_utils.py` core

**Agent: senior-engineer**

**File: `src/iris/time_utils.py`**

1. Move `from iris.rpc import common_pb2` to top-level (currently local import at line 219)
2. Trim docstrings: single-line methods get 1-line docstring max. Remove Args/Returns boilerplate from trivial methods like `to_ms()`, `from_ms()`, `expired()`, etc.
3. Add `__repr__` to `Timestamp`, `Duration`, `Deadline`, `Timer` (Pattern A)
4. Add `__hash__` to `Timestamp` and `Duration` (based on `_epoch_ms` / `_ms`)
5. `Timestamp.from_proto(proto: common_pb2.Timestamp) -> Timestamp` — accept proto message (Pattern B)
6. `Timestamp.to_proto() -> common_pb2.Timestamp` — return proto message (Pattern B)
7. Add `Timestamp.as_short_time() -> str` — returns "HH:MM:SS" format for log lines (Pattern K)
8. `Deadline.from_now(d: Duration) -> Deadline` — canonical factory (Pattern C)
9. `Duration.to_proto()` — already returns proto message, verify and keep
10. `Duration.from_proto()` — already accepts proto message, verify and keep

No other files change. Tests will break — fixed in Stage 5.

### Stage 2: Rename `common.proto` → `time.proto`, fix `config.proto`

**Agent: senior-engineer**

**Files changed:**
- `src/iris/rpc/common.proto` → rename to `src/iris/rpc/time.proto`
- `src/iris/rpc/cluster.proto` — update `import "common.proto"` → `import "time.proto"`
- `src/iris/rpc/config.proto` — update import; change `float evaluation_interval_seconds` and `float requesting_timeout_seconds` to `iris.common.Duration` messages
- `src/iris/rpc/vm.proto` — update import
- `src/iris/rpc/errors.proto` — update import if applicable
- `scripts/generate_protos.py` — update if it references common.proto
- Run proto generation: `uv run python lib/iris/scripts/generate_protos.py`
- All Python files importing `common_pb2` → update to `time_pb2` (or whatever the generated name is)

**Pattern applied:** file rename + proto field type change for config.proto

### Stage 3: Internal state uses `Timestamp` natively (state.py + service.py)

**Agent: senior-engineer**

**File: `src/iris/cluster/controller/state.py`**
- Pattern D throughout: Change all `common_pb2.Timestamp` fields to `Timestamp`:
  - `ControllerTaskAttempt.created_at`, `.started_at`, `.finished_at` → `Timestamp`
  - `ControllerTask.submitted_at`, `.started_at`, `.finished_at` → `Timestamp`
  - `ControllerJob.submitted_at`, `.started_at`, `.finished_at` → `Timestamp`
  - `ControllerWorker.last_heartbeat` (if exists) → `Timestamp`
- All `common_pb2.Timestamp(epoch_ms=Timestamp.now().to_proto())` → `Timestamp.now()` (lines ~199, 202, 205, 349, 354, 394, 419, 434, 451, 575, 610, 615, 621, 627)
- Pattern G: Add `scheduling_deadline: Deadline | None = None` to `ControllerJob`, set at submission time from `Duration.from_proto(request.scheduling_timeout)` → `Deadline.from_now(...)`
- `state.py:628-632` — use `Duration.from_proto()` for error message formatting via `__repr__`

**File: `src/iris/cluster/controller/service.py`**
- Pattern E: All field-by-field timestamp copies become `CopyFrom`:
  - Lines ~188-190, 207-210, 224-227, and all similar locations
  - `proto_attempt.started_at.CopyFrom(attempt.started_at.to_proto())`
  - Null checks remain: `if attempt.started_at is not None:`

**File: `src/iris/cluster/controller/controller.py`**
- `_mark_task_unschedulable` (line ~491): Use `Duration.from_proto()` for logging, use `repr()` instead of manual formatting

**File: `src/iris/cluster/controller/scheduler.py`**
- Pattern G: `_is_task_timed_out` → just check `job.scheduling_deadline.expired()`

**File: `src/iris/cluster/controller/events.py`**
- If events carry timestamps as `common_pb2.Timestamp`, change to `Timestamp`

### Stage 4: Fix client API signatures + worker internals

**Agent: senior-engineer**

**File: `src/iris/client/client.py`**
- Pattern I: `scheduling_timeout_seconds: int` → `scheduling_timeout: Duration | None = None`, same for `timeout_seconds`
- Pattern K (line ~955): Replace `datetime.fromtimestamp(...).strftime(...)` with `Timestamp.from_proto(entry.timestamp).as_short_time()`

**File: `src/iris/cluster/client/remote_client.py`**
- Pattern I: Same signature change
- Remove `* 1000` conversions (lines 101, 103, 121, 123) → `CopyFrom(duration.to_proto())`
- Deduplicate the two branches (with/without bundle_gcs_path) — the timeout setup is identical

**File: `src/iris/cluster/client/local_client.py`**
- Pattern I: Same signature passthrough

**File: `src/iris/iris_run.py`**
- Convert CLI `--timeout` (int seconds) → `Duration.from_seconds(timeout)` before passing to client

**File: `src/iris/cluster/worker/worker.py`**
- Pattern J: `WorkerConfig.heartbeat_timeout_seconds` → `heartbeat_timeout: Duration`, `poll_interval_seconds` → `poll_interval: Duration`
- `_serve()`: Use `Deadline.from_now(self._config.heartbeat_timeout)` instead of manual monotonic math. Reset deadline on heartbeat.
- Line ~470: Pattern B — `Timestamp.now().to_proto()` instead of `common_pb2.Timestamp(epoch_ms=int(time.time() * 1000))`

**File: `src/iris/cluster/worker/service.py`**
- Pattern F: Line ~131 — use `Timer` for uptime. `self._timer = Timer()` in `__init__`, `response.uptime.CopyFrom(Duration.from_ms(self._timer.elapsed_ms()).to_proto())`

**File: `src/iris/cluster/worker/builder.py`**
- Pattern F: Line ~211 — use `Timer` for build time

**File: `src/iris/cluster/worker/task_attempt.py`**
- Pattern C: Lines ~425-426, 496-498 — `Deadline.from_now(Duration.from_proto(self.request.timeout))`
- The `ContainerConfig.timeout_seconds` field should become `timeout: Duration | None`

**File: `src/iris/cluster/worker/worker_types.py`**
- Line ~40: `Timestamp.from_seconds(self.timestamp.timestamp()).to_proto()` using Pattern B

**File: `src/iris/cluster/worker/docker.py`**
- `ContainerConfig.timeout_seconds: int | None` → `timeout: Duration | None`

### Stage 5: Fix remaining peripheral files

**Agent: ml-engineer**

**File: `src/iris/cli.py`**
- Pattern H: `_format_timestamp(ms: int)` → use `Timestamp.from_ms(ms).as_formatted_date()`

**File: `src/iris/cluster/vm/config.py`**
- Lines ~500-502: Pattern E — `CopyFrom(Duration.from_seconds(300).to_proto())` instead of `.milliseconds = Duration.from_seconds(300).to_ms()`

**File: `src/iris/cluster/vm/autoscaler.py`**
- Lines ~189-190, 402: After Stage 2 proto change, use `Duration.from_proto(config.requesting_timeout)` instead of raw float

**File: `src/iris/cluster/vm/controller.py`**
- Line ~402: `timeout_ms=int(timeout * 1000)` → use Duration

**File: `src/iris/client/resolver.py`**
- Line ~67: `timeout_ms=int(timeout * 1000)` → consider taking Duration

**File: `src/iris/actor/client.py`, `src/iris/actor/pool.py`**
- `timeout_ms=int(self._timeout * 1000)` → consider taking Duration

### Stage 6: Fix tests

**Agent: ml-engineer**

**File: `tests/test_time_utils.py`**
- Delete the entire file. Replace with a minimal file containing ONLY:
  - `test_deadline_expires` (behavioral: sleep and check expiry)
  - `test_deadline_raise_if_expired` (behavioral: verify exception)
  - `test_deadline_from_now_with_duration` (new method)
  - `test_timestamp_proto_roundtrip` (integration: through proto message)
  - `test_duration_proto_roundtrip` (integration: through proto message)
  - `test_repr_formats` (verify __repr__ output is usable in logs)
- All other tests are tautological per AGENTS.md: "No tests for create object(x,y,z) and attributes are x,y,z"

**File: `tests/cluster/test_time_integration.py`**
- Delete: `test_timestamp_comparison_operators`, `test_duration_comparison_operators` (tautological)
- Delete: `test_duration_multi_unit_conversions`, `test_timestamp_conversions` (tautological)
- Delete: `test_timestamp_advances_with_time` (obvious behavior)
- Delete: `test_deadline_not_affected_by_timestamp` (not testing real behavior)
- Delete: `test_optional_timestamp_field_with_hasfield` (tests proto3 behavior, not our code)
- Delete: `test_duration_proto_roundtrip_manual` (redundant with non-manual version)
- Fix remaining tests to use `Timestamp.from_proto(proto_msg)` not `Timestamp.from_proto(proto_msg.epoch_ms)`
- Keep: proto round-trip tests, deadline timeout integration, deadline remaining prevents overshoot

**All other test files:**
- Fix any calls using `scheduling_timeout_seconds=` → `scheduling_timeout=Duration.from_seconds(...)`
- Fix any calls using `timeout_seconds=` → `timeout=Duration.from_seconds(...)`
- Fix any `common_pb2` → `time_pb2` imports
- Fix any direct `.epoch_ms` access where `Timestamp.from_proto()` should be used
- `test_dashboard.py:342`: Leave `int(job_status["startedAt"]["epochMs"])` — this is testing JSON wire format

### Stage 7: Validate

**Agent: senior-engineer**

- `uv run pytest lib/iris/tests/ -x`
- `./infra/pre-commit.py --all-files`
- Audit: `grep -r "time\.time()" lib/iris/src/` — should only be in `time_utils.py`
- Audit: `grep -r "\.milliseconds" lib/iris/src/` — should only be in `time_utils.py` and proto-generated files
- Audit: `grep -r "timeout_seconds" lib/iris/src/` — should be zero hits
- Audit: `grep -r "common_pb2" lib/iris/src/` — should be zero hits (renamed to time_pb2)
