# Event-Driven Dispatch vs Direct Methods: Design Analysis

## Executive Summary

**Recommendation: Keep the current event-driven dispatch architecture.**

The current architecture uses events as the control flow mechanism with centralized dispatch. The alternative would use direct methods with events created only for audit logging. After analyzing the real complexity in the codebase—particularly cascading effects in `_on_worker_failed()` and `_on_job_cancelled()`—the event-driven dispatch provides cleaner transaction boundaries and simpler cascading semantics.

## The Two Architectures

### Architecture 1: Event-Driven Dispatch (Current)

Events are the control flow mechanism. A central `handle_event()` method dispatches to handlers based on event type.

```python
# External caller (service.py)
state.handle_event(
    Event(
        EventType.WORKER_REGISTERED,
        worker_id=worker_id,
        address=address,
        metadata=metadata,
    )
)

# Implementation (state.py)
class ControllerState:
    def handle_event(self, event: Event) -> TransactionLog:
        with self._lock:
            txn = TransactionLog(event=event)
            match event.event_type:
                case EventType.WORKER_REGISTERED:
                    self._on_worker_registered(txn, event)
                case EventType.WORKER_FAILED:
                    self._on_worker_failed(txn, event)
                # ...
            self._transactions.append(txn)
            return txn

    def _on_worker_registered(self, txn: TransactionLog, event: Event) -> None:
        worker = self._workers.get(event.worker_id)
        if worker:
            worker.last_heartbeat_ms = event.timestamp_ms
            # ...
        else:
            worker = ControllerWorker(...)
            self._workers[event.worker_id] = worker
        txn.log("worker_registered", ...)
```

### Architecture 2: Direct Methods with Event Logging (Alternative)

Methods contain the logic directly. Events are created purely for audit logging after the fact.

```python
# External caller
state.register_worker(
    worker_id=worker_id,
    address=address,
    metadata=metadata,
)

# Implementation
class ControllerState:
    def register_worker(
        self,
        worker_id: WorkerId,
        address: str,
        metadata: cluster_pb2.WorkerMetadata,
        timestamp_ms: int | None = None,
    ) -> TransactionLog:
        with self._lock:
            txn = TransactionLog()

            worker = self._workers.get(worker_id)
            if worker:
                worker.last_heartbeat_ms = timestamp_ms or now_ms()
                txn.log("worker_heartbeat", worker_id)
            else:
                worker = ControllerWorker(...)
                self._workers[worker_id] = worker
                txn.log("worker_registered", worker_id)

            # Event created for audit trail only
            txn.event = Event(
                EventType.WORKER_REGISTERED,
                worker_id=worker_id,
                address=address,
                metadata=metadata,
            )
            return txn
```

## Critical Analysis: Cascading Effects

The key differentiator is how cascading effects work. The current codebase has two major cascading operations:

### Worker Failure Cascade (Current Implementation)

When a worker fails, all tasks on that worker must transition to `TASK_STATE_WORKER_FAILED`:

```python
def _on_worker_failed(self, txn: TransactionLog, event: Event) -> None:
    worker = self._workers[event.worker_id]
    worker.healthy = False
    txn.log("worker_failed", event.worker_id, error=event.error)

    # Cascade to all tasks on this worker
    for task_id in list(worker.running_tasks):
        task = self._tasks[task_id]
        if task.state in TERMINAL_TASK_STATES:
            continue

        # Create synthetic event and call handler directly
        cascade_event = Event(
            EventType.TASK_STATE_CHANGED,
            task_id=task_id,
            worker_id=event.worker_id,
            new_state=cluster_pb2.TASK_STATE_WORKER_FAILED,
            error=f"Worker {event.worker_id} failed: {event.error or 'unknown'}",
        )
        self._on_task_state_changed(txn, cascade_event)  # Same txn!
```

The current architecture cascades by calling handlers directly with the **same transaction log**. This produces a unified audit trail:

```
TransactionLog:
  event: WORKER_FAILED(worker_id="w1")
  actions:
    - worker_failed: w1
    - task_state_changed: j1/task-0 (RUNNING -> WORKER_FAILED)
    - task_unassigned: j1/task-0
    - task_requeued: j1/task-0
    - task_state_changed: j1/task-1 (RUNNING -> WORKER_FAILED)
    - task_unassigned: j1/task-1
    - task_requeued: j1/task-1
```

### How Direct Methods Would Handle Cascading

With direct methods, `mark_worker_failed()` needs to call `transition_task()` in a loop:

```python
def mark_worker_failed(self, worker_id: WorkerId, error: str | None = None) -> TransactionLog:
    with self._lock:
        txn = TransactionLog()

        worker = self._workers[worker_id]
        worker.healthy = False
        txn.log("worker_failed", worker_id)

        for task_id in list(worker.running_tasks):
            task = self._tasks[task_id]
            if task.state in TERMINAL_TASK_STATES:
                continue

            # Problem: transition_task() returns its own TransactionLog
            cascade_txn = self.transition_task(
                task_id=task_id,
                new_state=TASK_STATE_WORKER_FAILED,
                error=f"Worker {worker_id} failed",
            )
            # How do we merge cascade_txn into txn?
            # Option A: txn.actions.extend(cascade_txn.actions)
            # Option B: Return list of TransactionLogs
            # Option C: Pass txn into transition_task() somehow

        txn.event = Event(EventType.WORKER_FAILED, ...)
        return txn
```

**The transaction boundary problem emerges:**

| Approach | Code | Issues |
|----------|------|--------|
| Merge actions | `txn.actions.extend(cascade_txn.actions)` | Loses the cascade's triggering event; nested events become orphaned |
| Return list | `-> list[TransactionLog]` | Changes return type; callers must handle multiple logs |
| Pass txn in | `transition_task(..., txn=txn)` | Defeats purpose of "direct methods"; txn becomes threading parameter |
| Private helper | `_transition_task_internal(txn, ...)` | Duplicates logic between public/private methods |

The current architecture sidesteps this entirely: handlers receive a transaction log and pass it to sub-handlers. The transaction boundary is established once at `handle_event()`.

### Job Cancellation Cascade (Current Implementation)

```python
def _on_job_cancelled(self, txn: TransactionLog, event: Event) -> None:
    job = self._jobs[event.job_id]

    for task_id in self._tasks_by_job.get(event.job_id, []):
        task = self._tasks[task_id]
        if task.state in TERMINAL_TASK_STATES:
            continue

        cascade_event = Event(
            EventType.TASK_STATE_CHANGED,
            task_id=task_id,
            new_state=cluster_pb2.TASK_STATE_KILLED,
            error=event.reason,
        )
        self._on_task_state_changed(txn, cascade_event)

    job.state = cluster_pb2.JOB_STATE_KILLED
    job.error = event.reason
    job.finished_at_ms = now_ms()
    txn.log("job_cancelled", event.job_id, reason=event.reason)
```

Same pattern: cascade through handler calls, unified transaction.

## Trade-off Analysis

### Call Site Ergonomics

**Current (Event-Driven)**:
```python
state.handle_event(
    Event(
        EventType.TASK_STATE_CHANGED,
        task_id=task_id,
        new_state=cluster_pb2.TASK_STATE_SUCCEEDED,
        exit_code=0,
    )
)
```

**Direct Methods**:
```python
state.transition_task(
    task_id=task_id,
    new_state=cluster_pb2.TASK_STATE_SUCCEEDED,
    exit_code=0,
)
```

**Verdict**: Direct methods are slightly more concise. However, the Event construction makes the event-driven nature explicit, which aids understanding.

### Type Safety

**Current**: Event fields are optional; wrong combinations caught at runtime via asserts in handlers.

**Direct Methods**: Method signatures enforce required parameters statically.

**Verdict**: Marginal win for direct methods. The codebase has ~10 call sites and runtime asserts catch misuse immediately.

### Code Organization

**Current**:
- Single entry point (`handle_event`)
- All dispatch logic visible in one match statement
- Handlers are private methods with uniform signatures

**Direct Methods**:
- Multiple public entry points
- Dispatch is implicit (method name = operation)
- Must maintain parallel private helpers for cascading

**Verdict**: Event-driven has better discoverability. The match statement shows all possible state transitions at a glance.

### Testability

**Current test code**:
```python
def dispatch_task(state: ControllerState, task: ControllerTask, worker_id: WorkerId) -> None:
    """Dispatch a task to a worker: assign + mark running."""
    state.handle_event(
        Event(EventType.TASK_ASSIGNED, task_id=task.task_id, worker_id=worker_id)
    )
    state.handle_event(
        Event(EventType.TASK_STATE_CHANGED, task_id=task.task_id,
              new_state=cluster_pb2.TASK_STATE_RUNNING)
    )
```

Tests already wrap Event construction in helpers. Direct methods would simplify helpers slightly but the test code complexity is equivalent.

**Verdict**: No meaningful difference.

### Transaction Logging

**Current**: Events are first-class. `TransactionLog.event` captures the triggering event directly. Cascading effects share the parent transaction naturally.

**Direct Methods**: Must either:
- Create events retroactively (loses causality)
- Thread transaction logs through method calls (awkward)
- Accept fragmented logs for cascading operations

**Verdict**: Event-driven wins clearly. The transaction log design assumes events are the source of truth.

## Where Complexity Lives

| Aspect | Event-Driven | Direct Methods |
|--------|--------------|----------------|
| Entry point | Single (`handle_event`) | Multiple (one per operation) |
| Dispatch | Explicit match statement | Implicit (method names) |
| Cascading | Call handler with same txn | Merge transactions somehow |
| Transaction boundaries | Established at entry | Must be threaded or merged |
| Handler signatures | `(txn, event) -> None` | `(...) -> TransactionLog` |

The current architecture concentrates complexity in one place (the match statement and handler signatures). The alternative distributes it across methods and requires solving the transaction merge problem.

## Recommendation

**Keep the event-driven dispatch architecture.**

The direct methods approach appears simpler for leaf operations but creates genuine complexity for cascading effects. The current codebase already solves this:

1. **Unified transaction boundaries**: `handle_event()` creates the transaction; handlers extend it.
2. **Clean cascading**: Handlers call other handlers with the same transaction log.
3. **Explicit event model**: The Event type documents all possible state transitions.
4. **Audit trail integrity**: Each external trigger produces exactly one TransactionLog with full cascade visibility.

### If Type Safety Is Desired

Add optional factory functions without changing the core architecture:

```python
# events.py
def worker_registered_event(
    worker_id: WorkerId,
    address: str,
    metadata: cluster_pb2.WorkerMetadata,
    timestamp_ms: int | None = None,
) -> Event:
    return Event(
        EventType.WORKER_REGISTERED,
        worker_id=worker_id,
        address=address,
        metadata=metadata,
        timestamp_ms=timestamp_ms or now_ms(),
    )

# Usage - still event-driven, but type-checked
state.handle_event(worker_registered_event(
    worker_id=WorkerId(request.worker_id),
    address=request.address,
    metadata=request.metadata,
))
```

This preserves all benefits of the current architecture while adding static type checking for callers who want it.
