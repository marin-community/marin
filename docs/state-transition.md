# Controller State Transition Design

## Overview

Event-driven state management for the Iris controller. All state changes flow through `ControllerState.handle_event()`, which dispatches to handlers and logs actions to a transaction log for debugging.

## Design Principles

1. **Explicit event types** - Enum discriminator, not boolean flags
2. **Single dispatch point** - All changes through `handle_event(event)`
3. **Transaction logging** - Record actions for debugging/replay
4. **Assume valid state** - Use `[]` and `assert`, not defensive checks
5. **Simple entities** - Core logic in `ControllerState`, entities are data
6. **State carries semantics** - Task state determines retry behavior (no separate flags)

## Event Types

Simplified event types where task state changes use a single `TASK_STATE_CHANGED` event:

```python
class EventType(Enum):
    # Worker lifecycle
    WORKER_REGISTERED = auto()
    WORKER_HEARTBEAT = auto()
    WORKER_FAILED = auto()

    # Job lifecycle
    JOB_SUBMITTED = auto()
    JOB_CANCELLED = auto()

    # Task lifecycle
    TASK_ASSIGNED = auto()       # Creates attempt, assigns to worker
    TASK_STATE_CHANGED = auto()  # All task state transitions (new_state field carries target)


@dataclass(frozen=True)
class Event:
    """All state change events. Fields are optional based on event_type."""
    event_type: EventType

    # Entity IDs (use whichever are relevant)
    task_id: TaskId | None = None
    worker_id: WorkerId | None = None
    job_id: JobId | None = None

    # For TASK_STATE_CHANGED - the target task state
    new_state: int | None = None

    # Event data
    error: str | None = None
    exit_code: int | None = None
    reason: str | None = None
    timestamp_ms: int | None = None

    # For WORKER_REGISTERED
    address: str | None = None
    metadata: cluster_pb2.WorkerMetadata | None = None

    # For JOB_SUBMITTED
    request: cluster_pb2.Controller.LaunchJobRequest | None = None
```

Usage:
```python
Event(EventType.WORKER_FAILED, worker_id=worker_id, error="Connection lost")
Event(EventType.TASK_STATE_CHANGED, task_id=task_id,
      new_state=cluster_pb2.TASK_STATE_SUCCEEDED, exit_code=0)
Event(EventType.TASK_STATE_CHANGED, task_id=task_id,
      new_state=cluster_pb2.TASK_STATE_WORKER_FAILED, error="Worker died")
```

## Key Design Decision: State Carries Semantics

The task state already encodes all semantic information:
- `TASK_STATE_FAILED` -> uses failure retry budget
- `TASK_STATE_WORKER_FAILED` -> uses preemption retry budget
- `TASK_STATE_KILLED` -> no retry
- `TASK_STATE_SUCCEEDED` -> no retry
- `TASK_STATE_RUNNING` -> task is running

This means we don't need separate event types for each state - one event type with a `new_state` field is sufficient. The canonical state transition logic lives in `ControllerTask.handle_attempt_result()`.

## Transaction Log

Simple action log for debugging:

```python
@dataclass
class Action:
    """Single action taken during event handling."""
    timestamp_ms: int
    action: str
    entity_id: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransactionLog:
    """Records actions from handling one event."""
    event: Event
    timestamp_ms: int = field(default_factory=now_ms)
    actions: list[Action] = field(default_factory=list)

    def log(self, action: str, entity_id: str, **details) -> None:
        self.actions.append(Action(
            timestamp_ms=now_ms(),
            action=action,
            entity_id=str(entity_id),
            details=details,
        ))
```

## Event Dispatch

```python
class ControllerState:
    def handle_event(self, event: Event) -> TransactionLog:
        """Main entry point for all state changes."""
        with self._lock:
            txn = TransactionLog(event=event)

            match event.event_type:
                case EventType.WORKER_REGISTERED:
                    self._on_worker_registered(txn, event)
                case EventType.WORKER_HEARTBEAT:
                    self._on_worker_heartbeat(txn, event)
                case EventType.WORKER_FAILED:
                    self._on_worker_failed(txn, event)
                case EventType.JOB_SUBMITTED:
                    self._on_job_submitted(txn, event)
                case EventType.JOB_CANCELLED:
                    self._on_job_cancelled(txn, event)
                case EventType.TASK_ASSIGNED:
                    self._on_task_assigned(txn, event)
                case EventType.TASK_STATE_CHANGED:
                    self._on_task_state_changed(txn, event)

            self._transactions.append(txn)
            return txn
```

## Task State Handler

A single handler for all task state changes that delegates to the canonical state transition logic:

```python
def _on_task_state_changed(self, txn: TransactionLog, event: Event) -> None:
    """Handle all task state transitions.

    Delegates to task.handle_attempt_result() which contains the canonical
    state transition logic including retry budget management.
    """
    assert event.task_id and event.new_state is not None
    task = self._tasks[event.task_id]
    job = self._jobs[task.job_id]
    old_state = task.state

    # Delegate to the canonical state transition logic
    result = task.handle_attempt_result(
        event.new_state,
        error=event.error,
        exit_code=event.exit_code,
    )

    # Handle side effects based on result
    self._handle_task_side_effects(task, job, result, txn)

    # Update job state counters and finalize if needed
    new_job_state = job.on_task_transition(old_state, task.state)
    if new_job_state is not None:
        self._finalize_job_state(job, new_job_state)

    txn.log(
        "task_state_changed",
        event.task_id,
        old_state=old_state,
        new_state=task.state,
        result=result.name,
    )
```

## Cascading Events

Worker failure cascades to running tasks using the same transaction:

```python
def _on_worker_failed(self, txn: TransactionLog, event: Event) -> None:
    assert event.worker_id
    worker = self._workers[event.worker_id]
    worker.healthy = False
    txn.log("worker_failed", event.worker_id, error=event.error)

    # Cascade to running tasks - call handler directly, same transaction
    for task_id in list(worker.running_tasks):
        task = self._tasks[task_id]
        if task.state != cluster_pb2.TASK_STATE_RUNNING:
            continue

        cascade_event = Event(
            EventType.TASK_STATE_CHANGED,
            task_id=task_id,
            worker_id=event.worker_id,
            new_state=cluster_pb2.TASK_STATE_WORKER_FAILED,
            error=f"Worker {event.worker_id} failed: {event.error or 'unknown'}",
        )
        self._on_task_state_changed(txn, cascade_event)
```

## Terminal States

`TERMINAL_TASK_STATES` means "this attempt is done" - the task may still retry:
- `SUCCEEDED` - done, no retry
- `FAILED` - may retry (failure budget)
- `KILLED` - done, no retry
- `WORKER_FAILED` - may retry (preemption budget)
- `UNSCHEDULABLE` - done, no retry

## Files

- `lib/iris/src/iris/cluster/controller/events.py` - EventType, Event, TransactionLog
- `lib/iris/src/iris/cluster/controller/state.py` - ControllerState with handle_event()
