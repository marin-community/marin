# Sub-doc: Splitting `transitions.py` into Pure-Compute + Apply

Companion to `spec.md` §4.4-4.5. Draft 3.

## Changes from Draft 2

- **No `_polling_wake` event.** The reconcile loop is a fixed-interval timer; the wake-event references in the v2 audit table become "stays; no longer fires wake event."
- **No worker-initiated apply path.** Apply layer is only invoked from the controller's tick. Removes the conceptual question "what if the apply runs from a worker push too?"

## Changes from Draft 1

- **No per-worker asyncio tasks.** The reviewer caught the lock-convoy bug: per-worker reconciles all hit `apply_heartbeats_batch`'s single write lock plus cross-worker cascades, recreating the contention we're trying to avoid. v2 kept `_reconcile_worker_batch` single-batch as the orchestrator.
- **The pure function `reconcile_worker(inputs) → outputs` runs *inside* the batched apply.** Per-worker computation, single-transaction commit. Same batching properties as today.
- **Apply layer responsibility tightened.** `apply_reconcile_response` consumes `TransitionDelta`s plus the response's observations; cross-worker cascades fire in the same transaction as today.

## Target shape

```
                  one polling tick = one outer pass
                                |
                                v
   ┌──────────────────────────────────────────────────────────┐
   │  Phase 1: read snapshot                                  │
   │    db.read_snapshot → reconcile_rows_for_workers         │
   └──────────────────────────────────────────────────────────┘
                                |
                                v  (no DB lock)
   ┌──────────────────────────────────────────────────────────┐
   │  Phase 2: per-worker pure compute                         │
   │    for wid in worker_ids:                                 │
   │        outputs[wid] = reconcile_worker(inputs[wid])       │
   │                                                           │
   │  outputs = { wid: WorkerReconcilePlan + deltas + events }│
   └──────────────────────────────────────────────────────────┘
                                |
                                v  (no DB lock)
   ┌──────────────────────────────────────────────────────────┐
   │  Phase 3: fan out RPCs concurrently                      │
   │    asyncio.gather(send_reconcile(wid, plan))             │
   │    results = { wid: ReconcileResponse | error }          │
   └──────────────────────────────────────────────────────────┘
                                |
                                v  (single write txn — TODAY'S SHAPE)
   ┌──────────────────────────────────────────────────────────┐
   │  Phase 4: batched apply                                   │
   │    with db.transaction() as cur:                         │
   │      for wid, plan, response in zip(outputs, results):   │
   │        apply_reconcile_response(cur, plan, response)     │
   │      # Cross-worker cascades fire inside this txn         │
   └──────────────────────────────────────────────────────────┘
```

The change relative to today: Phase 2 is explicit (today it's interleaved into Phase 1). The pure function is a new code path; the lock and transaction structure are preserved.

## What stays in transitions.py

`transitions.py` shrinks but does not disappear:

- **Cross-worker cascades** stay here. Job-state recompute (`_recompute_job_state`), coscheduled-sibling termination (`_terminate_coscheduled_siblings`), failure-count tracking (`_resolve_task_failure_state`). These need cross-row reasoning that the per-worker pure function doesn't have access to.
- **Whole-job lifecycle** stays here: `submit_job`, `cancel_job`, `remove_finished_job`. These are not in the reconcile hot path.
- **Worker lifecycle**: `register_or_refresh_worker`, `remove_worker`, `fail_workers`. Not in the hot path.
- **Apply layer**: new function `apply_reconcile_response` lives in `transitions.py` and runs inside the existing write transaction.

What moves out:
- **Per-attempt state transitions** computed during reconcile move to `reconcile.py` as `TransitionDelta` types.
- **Read-only helpers** (`run_request_template`, `_resolve_task_failure_state` — already pure) move to `query.py`.

## The new module: `controller/reconcile.py`

```python
# lib/iris/src/iris/cluster/controller/reconcile.py

@dataclass(frozen=True)
class WorkerReconcileInputs:
    """All state needed to decide one worker's next desired set. Pure inputs."""
    worker: WorkerRow
    rows: list[ReconcileRow]            # tasks+attempts owned by this worker
    job_specs: dict[JobName, RunTaskRequest | None]   # pre-built specs per job
    now: Timestamp

@dataclass(frozen=True)
class WorkerReconcilePlan:
    """The reconcile decision. RPC payload + DB writes if RPC succeeds."""
    request: ReconcileMessage          # the wire payload
    db_writes: list[TransitionDelta]   # writes to apply on RPC success
    events: list[SchedulingEvent]      # audit log entries

def reconcile_worker(inputs: WorkerReconcileInputs) -> WorkerReconcilePlan:
    """Pure function. No DB, no RPC, no time.time() — inputs.now is the clock.

    Spec dispatch invariant: AttemptSpec.request is set exactly when the DB
    attempt state is ASSIGNED. Every other dispatched state sends an empty
    AttemptSpec; the worker is expected to have it cached.
    """
    desired = []
    db_writes = []
    events = []

    for row in inputs.rows:
        match row.task_state:
            case TASK_STATE_ASSIGNED:
                spec = inputs.job_specs.get(row.job_id)
                if spec is None:
                    # reservation holder or job disappeared mid-tick
                    continue
                desired.append(DesiredAttempt(
                    attempt_uid=row.attempt_uid,
                    intent_run=AttemptSpec(request=_stamp_uid(spec, row)),  # inline
                ))
            case TASK_STATE_BUILDING | TASK_STATE_RUNNING:
                desired.append(DesiredAttempt(
                    attempt_uid=row.attempt_uid,
                    intent_run=AttemptSpec(),  # spec omitted; worker has it cached
                ))
            case TASK_STATE_CANCELLED | TASK_STATE_PREEMPTED:
                desired.append(DesiredAttempt(
                    attempt_uid=row.attempt_uid,
                    intent_stop=_stop_reason_from_state(row.task_state),
                ))
            # Terminal states: don't include in desired at all.

    return WorkerReconcilePlan(
        request=ReconcileRequest(
            worker_id=inputs.worker.worker_id,
            desired=desired,
        ),
        db_writes=db_writes,
        events=events,
    )
```

The function is mechanical: it reads rows, classifies by state, builds a desired entry. No DB, no I/O, no asyncio.

## TransitionDelta types

```python
class TransitionDelta(Protocol):
    """A single DB-mutating effect of a reconcile decision."""

@dataclass(frozen=True)
class AttemptObserved(TransitionDelta):
    """Recorded when a worker reports observing this attempt."""
    attempt_uid: AttemptUid
    state: TaskState
    container_id: str | None
    finished_at: Timestamp | None
    exit_code: int | None
    error: str | None

@dataclass(frozen=True)
class AttemptMissingOnWorker(TransitionDelta):
    """Worker reported MISSING — spec cache lost mid-attempt.

    Apply: transition attempt to FAILED("worker_lost_spec"). Cascades fire.
    Scheduler reissues under a new uid on a subsequent tick.
    """
    attempt_uid: AttemptUid
```

Most `TransitionDelta`s actually come from the *response* (worker's observations), not the request (controller's intent). The pure function emits very few writes; most are computed in the apply step from the RPC response.

## Apply layer

```python
# transitions.py — NEW method

def apply_reconcile_response(
    cur: TransactionCursor,
    plan: WorkerReconcilePlan,
    response: ReconcileMessage | None,         # None on RPC failure
    error: str | None,                          # set on RPC failure
    now: Timestamp,
) -> None:
    """Apply the consequences of one worker's reconcile.

    On RPC success: write through each AttemptObservation; update controller's
    per-worker observed-uid cache; fire cross-worker cascades (coscheduling
    failures, job-state recompute).

    On RPC failure: log; mark worker degraded if repeated; do not write any
    state transitions (we don't know if the worker received the request).
    """
    if error is not None:
        # Soft-fail: record but don't transition state.
        _record_worker_rpc_failure(cur, plan.request.worker_id, error)
        return

    # Apply pre-computed writes from the pure layer.
    for delta in plan.db_writes:
        _apply_delta(cur, delta)

    # Apply observations from the response. MISSING is handled here as
    # attempt -> FAILED("worker_lost_spec"); every other observation is a
    # normal state transition.
    for obs in response.observed:
        _apply_observation(cur, obs, now)

    # Cross-worker cascades. These need to see the freshly-applied state.
    # Run after observation apply so cascades reason on the new state.
    # `is_task_finished` returns True for MISSING (treated as terminal failure)
    # along with the usual terminal states.
    _cascade_for_terminal_attempts(cur, [o for o in response.observed if is_task_finished(o.state)])
```

The cascade methods (`_cascade_terminal_job`, `_cascade_children`, `_terminate_coscheduled_siblings`) stay where they are. They're called from `apply_reconcile_response` exactly the way they're called from `apply_heartbeats_batch` today.

## Method audit of `transitions.py`

Every method in `ControllerTransitions`, classified by what happens to it:

| Method | Lines | After refactor |
|---|---|---|
| `run_request_template` | 836 | Moves to `query.py` (already pure-ish, just takes snapshot + job_id). |
| `run_request_for_attempt` | 813 | Moves to `query.py`. |
| `_resolve_task_failure_state` | 722 | Moves to `reconcile.py` as a helper used by the pure function. Already pure. |
| `submit_job` | 950 | Stays. Not in reconcile hot path. |
| `cancel_job` | 1213 | Stays. v2 fired `_polling_wake.set()`; v3 just writes CANCELLED rows and the next tick picks them up. |
| `register_or_refresh_worker` | 1254 | Stays. |
| `queue_assignments` | 1360 | Stays. Wakes the poll loop. |
| `_apply_task_transitions` | 1431 | **Split.** The per-attempt state logic moves to `reconcile.py`. The cascade-firing (preemption-cascade, job-state recompute) stays in `transitions.py` as `_apply_observation` + cascade calls. |
| `apply_task_updates`, `apply_heartbeats_batch` | 1669, 1689 | **Becomes `apply_reconcile_response`.** Same shape, calls the new pure function, fires the same cascades. |
| `preempt_task`, `cancel_tasks_for_timeout` | 1980, 2079 | Stay. Initiator of `STOP_REASON_PREEMPTED` / `STOP_REASON_TASK_TIMEOUT`. Write `task.state=PREEMPTED`/etc and fire wake. |
| `fail_workers`, `_remove_failed_worker` | 1904, 1776 | Stay. |
| `prune_old_data`, `remove_finished_job`, `remove_worker` | 2215, 2185, 2206 | Stay. |
| `get_running_tasks_for_poll` | 2303 | Moves to `query.py`. Read-only, misplaced. |
| `update_worker_pings`, `set_worker_*_for_test` | misc | Stay. |
| `drain_for_direct_provider` | 2441 | Stays. K8sTaskProvider path; orthogonal. |
| `_build_run_request` | 2519 | Moves to `query.py`. |
| `apply_direct_provider_updates` | 2548 | Stays. K8s path. |

After the refactor `transitions.py` is ~2000 lines — same role, less interleaving.

## Testing strategy

### Pure function unit tests

`tests/cluster/controller/test_reconcile_pure.py`. Each test constructs a `WorkerReconcileInputs` by hand, calls `reconcile_worker`, asserts on the output. The matrix from `spec.md` §5.3:

```python
def test_running_no_spec_dispatched():
    inputs = WorkerReconcileInputs(
        worker=WORKER_W1,
        rows=[reconcile_row(uid="a1", state=RUNNING, worker=W1)],
        job_specs={JOB1: SPEC1},
        now=T0,
    )
    plan = reconcile_worker(inputs)
    assert plan.request.desired == [
        DesiredAttempt(attempt_uid="a1", intent_run=AttemptSpec()),  # spec omitted (not ASSIGNED)
    ]
    assert plan.db_writes == []

def test_assigned_spec_inline():
    inputs = WorkerReconcileInputs(
        worker=WORKER_W1,
        rows=[reconcile_row(uid="new1", state=ASSIGNED, worker=W1, job_id=JOB1)],
        job_specs={JOB1: SPEC1},
        now=T0,
    )
    plan = reconcile_worker(inputs)
    assert plan.request.desired[0].intent_run.request == SPEC1
```

Twenty-five-ish tests cover the decision space. None need a DB.

### Apply layer integration tests

`tests/cluster/controller/test_reconcile_apply.py`. Run the apply function against a real SQLite DB, verify state transitions and cascades.

### End-to-end

The existing `_reconcile_worker_batch` test (if any) extends to cover both wire layers: legacy and new. CI matrix has the flag on and off.

## PR sequence

1. **Extract types + stub.** `reconcile.py` with the dataclasses and an empty `reconcile_worker` returning a no-op plan. `transitions.py` adds `apply_reconcile_response` that just calls into existing methods. Wire it up so the current heartbeat path goes through the new function. Zero behavior change. ~300 LOC.

2. **Move read-only helpers** (`run_request_template`, etc.) to `query.py`. Mechanical refactor. ~200 LOC.

3. **Fill in `reconcile_worker`.** Implement the decision matrix. Unit tests. ~400 LOC + tests.

4. **Switch the hot path.** `_reconcile_worker_batch` calls `reconcile_worker` per worker, uses the output. Today's `apply_heartbeats_batch` becomes `apply_reconcile_response` (renamed, slightly tightened). One behavior-preserving PR. ~150 LOC.

5. **Phase B: wire shape.** Reconcile RPC + capability negotiation. The pure layer doesn't change; only the wire serializer is new. See `sub/protocol.md` and `sub/rollout.md`.

Steps 1–4 are Phase A in `spec.md` §7. They ship before any protocol change.
