# Controller State Transition Design

## Overview

Event-driven state management for the Iris controller. All state changes flow through `ControllerState.handle_event()`, which dispatches to handlers and logs actions to a transaction log for debugging.

## Design Principles

1. **Explicit event types** - Enum discriminator, not boolean flags
2. **Single dispatch point** - All changes through `handle_event(event)`
3. **Transaction logging** - Record actions for debugging/replay
4. **Assume valid state** - Use `[]` and `assert`, not defensive checks
5. **Simple entities** - Core logic in `ControllerState`, entities are data

## Event Type

Single event class with enum discriminator:

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
    TASK_ASSIGNED = auto()
    TASK_RUNNING = auto()
    TASK_SUCCEEDED = auto()
    TASK_FAILED = auto()
    TASK_KILLED = auto()
    TASK_WORKER_FAILED = auto()


@dataclass(frozen=True)
class Event:
    """All state change events. Fields are optional based on event_type."""
    event_type: EventType

    # Entity IDs (use whichever are relevant)
    task_id: TaskId | None = None
    worker_id: WorkerId | None = None
    job_id: JobId | None = None

    # Event data
    error: str | None = None
    exit_code: int | None = None
    reason: str | None = None
    timestamp_ms: int | None = None

    # For WORKER_REGISTERED
    address: str | None = None
    metadata: cluster_pb2.WorkerMetadata | None = None

    # For JOB_SUBMITTED
    request: cluster_pb2.SubmitJobRequest | None = None
```

Usage:
```python
Event(EventType.WORKER_FAILED, worker_id=worker_id, error="Connection lost")
Event(EventType.TASK_SUCCEEDED, task_id=task_id, exit_code=0)
Event(EventType.TASK_WORKER_FAILED, task_id=task_id, worker_id=worker_id, error="Worker died")
```

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
    def __init__(self):
        self._lock = RLock()
        self._jobs: dict[JobId, ControllerJob] = {}
        self._tasks: dict[TaskId, ControllerTask] = {}
        self._workers: dict[WorkerId, ControllerWorker] = {}
        self._transactions: deque[TransactionLog] = deque(maxlen=1000)

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
                case EventType.TASK_RUNNING:
                    self._on_task_running(txn, event)
                case EventType.TASK_SUCCEEDED:
                    self._on_task_succeeded(txn, event)
                case EventType.TASK_FAILED:
                    self._on_task_failed(txn, event)
                case EventType.TASK_KILLED:
                    self._on_task_killed(txn, event)
                case EventType.TASK_WORKER_FAILED:
                    self._on_task_worker_failed(txn, event)

            self._transactions.append(txn)
            return txn
```

## Event Handlers

### Worker Events

```python
def _on_worker_registered(self, txn: TransactionLog, event: Event) -> None:
    assert event.worker_id and event.address and event.metadata
    worker = ControllerWorker(
        worker_id=event.worker_id,
        address=event.address,
        metadata=event.metadata,
    )
    self._workers[event.worker_id] = worker
    txn.log("worker_registered", event.worker_id, address=event.address)


def _on_worker_heartbeat(self, txn: TransactionLog, event: Event) -> None:
    assert event.worker_id and event.timestamp_ms is not None
    worker = self._workers[event.worker_id]
    worker.last_heartbeat_ms = event.timestamp_ms
    worker.consecutive_failures = 0
    txn.log("heartbeat", event.worker_id)


def _on_worker_failed(self, txn: TransactionLog, event: Event) -> None:
    assert event.worker_id
    worker = self._workers[event.worker_id]
    worker.healthy = False
    txn.log("worker_failed", event.worker_id, error=event.error)

    # Cascade to running tasks - call handler directly, same transaction
    for task_id in list(worker.running_tasks):
        task = self._tasks[task_id]
        assert task.worker_id == event.worker_id
        if task.state != cluster_pb2.TASK_STATE_RUNNING:
            continue

        cascade_event = Event(
            EventType.TASK_WORKER_FAILED,
            task_id=task_id,
            worker_id=event.worker_id,
            error=f"Worker {event.worker_id} failed",
        )
        self._on_task_worker_failed(txn, cascade_event)
```

### Job Events

```python
def _on_job_submitted(self, txn: TransactionLog, event: Event) -> None:
    assert event.job_id and event.request
    job = ControllerJob(job_id=event.job_id, request=event.request)
    self._jobs[event.job_id] = job

    for i in range(event.request.num_tasks):
        task_id = TaskId(f"{event.job_id}:{i}")
        task = ControllerTask(task_id=task_id, job_id=event.job_id)
        self._tasks[task_id] = task
        self._task_queue.append(task_id)
        txn.log("task_created", task_id, job_id=str(event.job_id))

    txn.log("job_submitted", event.job_id, num_tasks=event.request.num_tasks)


def _on_job_cancelled(self, txn: TransactionLog, event: Event) -> None:
    assert event.job_id and event.reason
    job = self._jobs[event.job_id]

    for task_id, task in self._tasks.items():
        if task.job_id != event.job_id:
            continue
        if task.state in TERMINAL_TASK_STATES:
            continue

        cascade_event = Event(EventType.TASK_KILLED, task_id=task_id, reason=event.reason)
        self._on_task_killed(txn, cascade_event)

    job.state = cluster_pb2.JOB_STATE_CANCELLED
    txn.log("job_cancelled", event.job_id, reason=event.reason)
```

### Task Events

```python
def _on_task_assigned(self, txn: TransactionLog, event: Event) -> None:
    assert event.task_id and event.worker_id
    task = self._tasks[event.task_id]
    worker = self._workers[event.worker_id]
    job = self._jobs[task.job_id]

    attempt = ControllerTaskAttempt(
        attempt_id=len(task.attempts),
        worker_id=event.worker_id,
    )
    task.attempts.append(attempt)
    task.state = cluster_pb2.TASK_STATE_PENDING
    worker.assign_task(event.task_id, job.request.resources)

    txn.log("task_assigned", event.task_id, worker_id=str(event.worker_id))


def _on_task_running(self, txn: TransactionLog, event: Event) -> None:
    assert event.task_id
    task = self._tasks[event.task_id]
    old_state = task.state

    task.state = cluster_pb2.TASK_STATE_RUNNING
    task.attempts[-1].state = cluster_pb2.TASK_STATE_RUNNING
    task.attempts[-1].started_at_ms = now_ms()

    self._update_job_counters(task.job_id, old_state, task.state)
    txn.log("task_running", event.task_id)


def _on_task_succeeded(self, txn: TransactionLog, event: Event) -> None:
    assert event.task_id
    task = self._tasks[event.task_id]
    old_state = task.state

    task.state = cluster_pb2.TASK_STATE_SUCCEEDED
    task.exit_code = event.exit_code or 0
    task.finished_at_ms = now_ms()
    task.attempts[-1].state = cluster_pb2.TASK_STATE_SUCCEEDED
    task.attempts[-1].exit_code = event.exit_code
    task.attempts[-1].finished_at_ms = now_ms()

    self._finalize_task(task, txn)
    self._update_job_counters(task.job_id, old_state, task.state)
    txn.log("task_succeeded", event.task_id, exit_code=event.exit_code)


def _on_task_failed(self, txn: TransactionLog, event: Event) -> None:
    """Task failed due to task error. Uses failure retry budget."""
    assert event.task_id
    task = self._tasks[event.task_id]
    old_state = task.state

    task.failure_count += 1
    can_retry = task.failure_count <= task.max_retries_failure

    self._apply_task_failure(
        task, txn,
        new_state=cluster_pb2.TASK_STATE_FAILED,
        error=event.error,
        exit_code=event.exit_code,
        can_retry=can_retry,
    )

    self._update_job_counters(task.job_id, old_state, task.state)
    txn.log("task_failed", event.task_id, error=event.error, can_retry=can_retry)


def _on_task_killed(self, txn: TransactionLog, event: Event) -> None:
    """Task killed by user/scheduler. No retry."""
    assert event.task_id
    task = self._tasks[event.task_id]
    old_state = task.state

    self._apply_task_failure(
        task, txn,
        new_state=cluster_pb2.TASK_STATE_KILLED,
        error=event.reason,
        can_retry=False,
    )

    self._update_job_counters(task.job_id, old_state, task.state)
    txn.log("task_killed", event.task_id, reason=event.reason)


def _on_task_worker_failed(self, txn: TransactionLog, event: Event) -> None:
    """Task failed because worker died. Uses preemption retry budget."""
    assert event.task_id
    task = self._tasks[event.task_id]
    old_state = task.state

    task.preemption_count += 1
    can_retry = task.preemption_count <= task.max_retries_preemption

    self._apply_task_failure(
        task, txn,
        new_state=cluster_pb2.TASK_STATE_WORKER_FAILED,
        error=event.error,
        can_retry=can_retry,
    )

    self._update_job_counters(task.job_id, old_state, task.state)
    txn.log("task_worker_failed", event.task_id,
            worker_id=str(event.worker_id), can_retry=can_retry)
```

### Shared Helpers

```python
def _apply_task_failure(
    self,
    task: ControllerTask,
    txn: TransactionLog,
    *,
    new_state: int,
    error: str | None,
    exit_code: int | None = None,
    can_retry: bool,
) -> None:
    """Common logic for task failure states."""
    task.state = new_state
    task.error = error
    task.exit_code = exit_code
    task.attempts[-1].state = new_state
    task.attempts[-1].error = error
    task.attempts[-1].exit_code = exit_code
    task.attempts[-1].finished_at_ms = now_ms()

    if can_retry:
        self._requeue_task(task, txn)
    else:
        task.finished_at_ms = now_ms()

    self._finalize_task(task, txn)


def _finalize_task(self, task: ControllerTask, txn: TransactionLog) -> None:
    """Unassign from worker and clean up endpoints."""
    worker = self._workers[task.worker_id]
    job = self._jobs[task.job_id]
    worker.unassign_task(task.task_id, job.request.resources)
    txn.log("task_unassigned", task.task_id, worker_id=str(task.worker_id))

    self._remove_endpoints_for_task(task.task_id)
    self._maybe_finalize_job(task.job_id)


def _requeue_task(self, task: ControllerTask, txn: TransactionLog) -> None:
    """Put task back on scheduling queue for retry."""
    self._task_queue.append(task.task_id)
    txn.log("task_requeued", task.task_id)


def _update_job_counters(self, job_id: JobId, old_state: int, new_state: int) -> None:
    job = self._jobs[job_id]
    job.task_state_counts[old_state] -= 1
    job.task_state_counts[new_state] += 1


def _maybe_finalize_job(self, job_id: JobId) -> None:
    job = self._jobs[job_id]
    new_state = job.compute_job_state()
    if new_state is not None:
        self._finalize_job_state(job, new_state)


def get_transactions(self, limit: int = 100) -> list[TransactionLog]:
    """Return recent transactions for debugging."""
    with self._lock:
        return list(self._transactions)[-limit:]
```

## Files to Create/Modify

### Create: `lib/iris/src/iris/cluster/controller/events.py`
- `EventType` enum
- `Event` dataclass
- `Action` and `TransactionLog` dataclasses

### Modify: `lib/iris/src/iris/cluster/controller/state.py`
- Add `handle_event()` dispatch
- Add `_on_*` handlers
- Add `_transactions` buffer
- Replace existing `ActionLogEntry` with `TransactionLog`

### Update callers
```python
# Before
state.transition_task(task_id, TASK_STATE_FAILED, error=error)

# After
state.handle_event(Event(EventType.TASK_FAILED, task_id=task_id, error=error))
```

## Terminal States

`TERMINAL_TASK_STATES` means "this attempt is done" - the task may still retry:
- `SUCCEEDED` - done, no retry
- `FAILED` - may retry (failure budget)
- `KILLED` - done, no retry
- `WORKER_FAILED` - may retry (preemption budget)
- `UNSCHEDULABLE` - done, no retry
