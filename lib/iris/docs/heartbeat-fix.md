# Heartbeat Protocol Fix Plan

## Context

Branch `rjpower/20260130-iris-heartbeat` switched from a worker-initiated
heartbeat/registration model to a controller-initiated heartbeat. The controller
now calls `WorkerService.Heartbeat` on each worker, delivering task assignments
and kill requests, and receiving running/completed task reports in the response.

This is directionally correct — a single controller-driven heartbeat eliminates
race conditions between dual RPC paths (`RegisterWorker` + `ReportTaskState`)
and simplifies the protocol. However, the current implementation has several
correctness gaps and stale documentation.

## Issues

### 1. `as_completed` timeout crashes the scheduler loop

**File:** `controller.py:505`

```python
for future in as_completed(futures, timeout=10):
```

If any heartbeat takes longer than 10s total wall-clock time, `as_completed`
raises `TimeoutError`. This exception is uncaught and will unwind the entire
scheduling loop.

**Fix:** Wrap the `as_completed` loop in a `try/except TimeoutError`. Any
futures still pending after the timeout should be treated as failures: re-queue
their outbox entries and call `_handle_heartbeat_failure`.

```python
try:
    for future in as_completed(futures, timeout=10):
        worker, tasks_to_run, tasks_to_kill = futures[future]
        try:
            response = future.result()
            if response is None:
                self._requeue_and_fail(worker, tasks_to_run, tasks_to_kill)
                continue
            self._process_heartbeat_response(worker, response)
        except Exception as e:
            logger.warning(f"Heartbeat error for {worker.worker_id}: {e}")
            self._requeue_and_fail(worker, tasks_to_run, tasks_to_kill)
except TimeoutError:
    # Any futures not yet completed are treated as failures
    for future, (worker, tasks_to_run, tasks_to_kill) in futures.items():
        if not future.done():
            logger.warning(f"Heartbeat timed out for {worker.worker_id}")
            self._requeue_and_fail(worker, tasks_to_run, tasks_to_kill)
            future.cancel()
```

Extract the re-queue + failure handling into a small helper to avoid repeating
it three times:

```python
def _requeue_and_fail(self, worker, tasks_to_run, tasks_to_kill):
    if tasks_to_run:
        self._dispatch_outbox[worker.worker_id].extend(tasks_to_run)
    if tasks_to_kill:
        self._kill_outbox[worker.worker_id].extend(tasks_to_kill)
    self._handle_heartbeat_failure(worker)
```

### 2. `consecutive_failures` mutated on controller object, also reset by state event — conflicting ownership

**File:** `controller.py:605`, `state.py:1081`

The controller directly increments `worker.consecutive_failures` in
`_handle_heartbeat_failure`, but the state layer resets it to 0 in
`_on_worker_heartbeat`. This dual-ownership is fragile: the controller reads a
field it doesn't own, and the behavior is correct only by coincidence.

**Fix:** Move all failure-count logic into the state layer. Add a
`WorkerHeartbeatFailedEvent` that increments the counter and checks the
threshold. The controller calls `self._state.handle_event(...)` instead of
mutating the worker directly.

```python
@dataclass
class WorkerHeartbeatFailedEvent:
    worker_id: WorkerId
    error: str
```

In `_on_worker_heartbeat_failed`:
```python
def _on_worker_heartbeat_failed(self, txn, event):
    worker = self._workers[event.worker_id]
    worker.consecutive_failures += 1
    if worker.consecutive_failures >= 3:
        self._on_worker_failed(txn, WorkerFailedEvent(
            worker_id=event.worker_id,
            error=event.error,
        ))
```

The success path (`_on_worker_heartbeat`) already resets to 0, so the
"consecutive" semantics are preserved naturally.

### 3. Worker `_last_heartbeat_time` not reset before `_serve()`

**File:** `worker.py:125, 293-308`

`_last_heartbeat_time` is set to `time.monotonic()` at construction time
(line 125). If registration takes 30+ seconds, the clock has already been
ticking, and `_serve()` may immediately time out (60s timeout minus
registration time).

**Fix:** Reset `_last_heartbeat_time` at the start of `_serve()`:

```python
def _serve(self) -> None:
    self._last_heartbeat_time = time.monotonic()
    heartbeat_timeout = self._config.heartbeat_timeout_seconds
    logger.info("Serving (waiting for controller heartbeats)")
    ...
```

### 4. Missing unknown-task reconciliation (controller restart case)

**File:** `controller.py:586-601`

The current reconciliation only detects tasks that the controller expects but
the worker doesn't report (missing tasks → `WORKER_FAILED`). It does NOT detect
the reverse: a worker reports running tasks that the controller doesn't know
about (e.g. after controller restart).

In the old protocol, this was handled by `should_reset=True` in the registration
response. With the new protocol, the controller should detect unknown task IDs
in the heartbeat response and send kill requests for them.

**Fix:** After processing completions and missing tasks, add an unknown-task
check:

```python
# Detect tasks the worker is running that the controller doesn't know about
known_task_ids = {str(tid) for tid in worker.running_tasks}
unknown = reported_ids - known_task_ids
for tid_str in unknown:
    task = self._state.get_task(TaskId(tid_str))
    if task is None or task.is_finished():
        # Task doesn't exist or is already done — ask worker to kill it
        self._kill_outbox[worker.worker_id].append(tid_str)
        logger.warning(f"Unknown task {tid_str} on worker {worker.worker_id}, sending kill")
```

This covers the controller-restart case: the restarted controller has empty
state, so all worker-reported tasks are "unknown" and get killed. The worker
will then have no running tasks, and the tasks' original jobs will eventually
time out or be resubmitted.

### 5. Stale documentation

#### README.md (lines 129-144)

The "Worker Lifecycle > Registration and Reconciliation" section still describes
the old protocol:
- "Workers register with the controller via heartbeat (every 10 seconds)"
- "`running_tasks` - the list of tasks the worker believes it's running"
- "`should_reset=True`"

**Fix:** Rewrite this section to describe the new protocol:

```markdown
## Worker Lifecycle

### Registration and Heartbeat

Workers register with the controller once at startup via the `Register` RPC.
After registration, the worker enters a serve loop and waits for controller-
initiated heartbeats.

The controller sends `Heartbeat` RPCs to all registered workers on each
scheduler tick (~5s). The heartbeat request carries:
- `tasks_to_run`: new task assignments for this worker
- `tasks_to_kill`: task IDs to terminate

The worker responds with:
- `running_tasks`: tasks currently executing (task_id + attempt_id)
- `completed_tasks`: tasks that finished since the last heartbeat

The controller reconciles the response:

1. **Worker missing expected tasks** (e.g., worker restarted mid-task):
   - Controller marks missing tasks as `WORKER_FAILED`
   - Tasks are retried on another worker

2. **Worker reports unknown tasks** (e.g., controller restarted):
   - Controller sends kill requests for unknown tasks on next heartbeat
   - Worker terminates orphaned containers
```

#### cluster.proto (lines 443-454)

The `CompletedTaskEntry` comment references `report_task_state RPC` which no
longer exists.

**Fix:** Update the comment:

```protobuf
// A task completion reported by the worker via heartbeat response.
// Workers buffer completions and deliver them in the next heartbeat.
message CompletedTaskEntry {
```

#### state.py `WORKER_REPORTED_TERMINAL_STATES` comment (lines 128-130)

References "both report_task_state RPC and heartbeat". Remove the RPC reference.

```python
# Terminal states that originate from worker reports via heartbeat (as opposed
# to controller decisions like KILLED or UNSCHEDULABLE). Used to detect
# duplicate completions across multiple heartbeats.
```

### 6. Swallowed exception in worker

**File:** `worker.py:339-348` (or wherever `_report_task_state` or equivalent
notify logic lives)

Per AGENTS.md: "NEVER EVER SWALLOW EXCEPTIONS." If there is a bare `except
Exception: pass` anywhere in the heartbeat/notify path, it needs to either log
the exception or let it propagate.

**Fix:** Add `logger.debug(...)` at minimum, or restructure so the caller
handles the exception.

## Summary of Changes

| File | Change |
|------|--------|
| `controller.py` | Catch `TimeoutError` from `as_completed`, extract `_requeue_and_fail` helper |
| `controller.py` | Move `consecutive_failures` mutation into state layer via new event |
| `controller.py` | Add unknown-task reconciliation in `_process_heartbeat_response` |
| `state.py` | Add `WorkerHeartbeatFailedEvent` handler |
| `state.py` | Fix stale comment on `WORKER_REPORTED_TERMINAL_STATES` |
| `worker.py` | Reset `_last_heartbeat_time` at start of `_serve()` |
| `worker.py` | Fix swallowed exceptions (add logging) |
| `cluster.proto` | Update `CompletedTaskEntry` comment |
| `README.md` | Rewrite Worker Lifecycle section |
| `events.py` | Add `WorkerHeartbeatFailedEvent` dataclass |

## End State

After these changes, the heartbeat protocol will have:

1. **Single owner for worker health state**: all failure counting and health
   transitions go through the state layer's event system, not direct mutation.

2. **Crash-safe scheduler loop**: a slow worker heartbeat cannot crash the
   scheduling loop. Timed-out workers are treated as failures and their outbox
   entries are re-queued.

3. **Correct heartbeat timing on the worker**: the timeout clock starts when
   the worker enters its serve loop, not at construction time. This eliminates
   the immediate-timeout-after-slow-registration bug.

4. **Full bidirectional reconciliation**: the controller detects both missing
   tasks (worker lost them) and unknown tasks (controller doesn't know about
   them). This covers the controller-restart case that was lost in the protocol
   switch.

5. **Accurate documentation**: README, proto comments, and state-layer comments
   all describe the actual controller-initiated heartbeat protocol.

## Reflection

The core protocol change — from worker-initiated heartbeats to controller-
initiated — is a good simplification. It removes the dual-path problem where
`RegisterWorker` and `ReportTaskState` could race, and it puts the controller
in charge of timing (which is where scheduling decisions live anyway).

The issues found are all consequences of an incomplete migration:
- The `as_completed` timeout was probably copied from a pattern that assumed
  individual RPC timeouts rather than wall-clock aggregate timeout.
- The `consecutive_failures` dual-ownership is a classic symptom of incremental
  refactoring where one layer was updated but the other wasn't fully migrated.
- The missing unknown-task reconciliation is the old `should_reset` logic that
  didn't get a new equivalent in the heartbeat path.
- The stale docs are just the tail end of the migration.

None of these are architectural problems. They're all fixable with targeted
changes that don't alter the protocol design.
