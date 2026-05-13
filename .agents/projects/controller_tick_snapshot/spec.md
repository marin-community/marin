# Spec — controller_tick_snapshot

Concrete contracts for the design in [design.md](./design.md). Internal refactor; no new protos, no schema changes, no new persisted data. References pin to `origin/main` at `0d440a1b20d23f038fce7c5e0d6ab6d9833fb268`.

This spec is the contract surface. Where it disagrees with the design, the design is wrong; please raise an issue.

---

## File layout

| Path | Phase | Purpose |
|---|---|---|
| `lib/iris/src/iris/cluster/controller/tick.py` | 1 | New module. Hosts `ControlTick`, `TickDiagnostics`, `TickDriver`. |
| `lib/iris/src/iris/cluster/controller/intents.py` | 3 | New module. Intent taxonomy + `IntentBatch` + `ApplyTickResult` types. |
| `lib/iris/src/iris/cluster/controller/controller.py` | 1, 2, 3 | Delete `_run_scheduling_loop`, `_run_autoscaler_loop` (P1); `_run_polling_loop` (P2). `Controller.__init__` spawns `TickDriver` instead. `_cache_scheduling_diagnostics`, `get_job_scheduling_diagnostics`, `_scheduling_diagnostics` move to `TickDriver` (P1). |
| `lib/iris/src/iris/cluster/controller/transitions.py` | 3 | Add `apply_tick_intents(cur, batch) -> ApplyTickResult`. Existing per-section methods (`queue_assignments`, `apply_heartbeats_batch`, ...) stay public — used by ping and by tests; their bodies are reused by `apply_tick_intents`. |
| `lib/iris/src/iris/cluster/controller/reads.py` | 1, 2 | Add `build_control_tick(tx, *, health, attrs) -> ControlTick`. Add `bulk_run_request_templates(tx, transitions, job_ids) -> dict[JobName, RunTaskRequest \| None]` (Phase 2). |
| `lib/iris/tests/cluster/controller/test_control_tick.py` | 1 | Integration test; parity checks against per-loop reads on a checkpoint DB. |
| `lib/iris/tests/cluster/controller/test_tick_driver.py` | 1, 2 | Lifecycle test: start/stop/wake/exception isolation. |
| `lib/iris/tests/cluster/controller/test_apply_tick_intents.py` | 3 | Property test + per-intent CAS contract tests. |

Files **not** touched (intentional): `db.py`, `schema.py`, `projections/*`, `service.py`, `worker_health.py`. Ping path (`_run_ping_loop` and its writes) is unchanged.

---

## Phase 1 contracts

### `ControlTick` (Phase 1 shape)

In `tick.py`:

```python
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from iris.cluster.types import JobName, WorkerId
from iris.cluster.controller.controller import PendingTask
from iris.cluster.controller.reads import (
    SchedulableWorker,
    WorkerResourceUsage,
    TaskDetailRow,
)

@dataclass(frozen=True, slots=True)
class ControlTick:
    """Immutable snapshot of the controller's view of the world for one tick.

    All fields are built inside one ``read_snapshot()`` so they are mutually
    consistent. Collections are tuple/frozen/MappingProxyType so accidental
    mutation in a consumer raises rather than corrupts the next tick.

    ``state_read_ms`` is the wall-clock milliseconds-since-epoch the snapshot
    transaction began (NOT the time the snapshot finished building). Use it
    to compute staleness against a later tick.
    """

    pending_tasks: tuple[PendingTask, ...]
    healthy_workers: tuple[SchedulableWorker, ...]
    resource_usage: Mapping[WorkerId, WorkerResourceUsage]   # MappingProxyType
    tasks_index: Mapping[JobName, TaskDetailRow]             # MappingProxyType; keyed by task_id
    reserved_jobs: frozenset[JobName]
    state_read_ms: int
```

**Snapshot-build order** (inside `build_control_tick`, all under the same `Tx`):

1. `state_read_ms = int(Timestamp.now().epoch_ms())` *before* the first DB read, so consumers see the tx-open wall time.
2. `pending_tasks = tuple(_pending_tasks_with_jobs_tx(tx))` — moves the body of `controller._pending_tasks_with_jobs` to accept a `Tx` instead of opening its own. The current function ([controller.py:448](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/controller.py#L448)) is kept as a thin wrapper that opens a snapshot — used only by tests; production code uses the `_tx` variant.
3. `healthy_workers = tuple(reads.healthy_active_workers_with_attributes(tx, health, attrs))` — unchanged signature.
4. `resource_usage = MappingProxyType(reads.resource_usage_by_worker(tx))`.
5. `reserved_jobs = frozenset(controller._reserved_job_ids_tx(tx))` — also gets a `_tx` variant.
6. `tasks_index = MappingProxyType(reads.bulk_get_task_detail(tx, _task_ids_referenced(pending_tasks, resource_usage)))` — `bulk_get_task_detail` ([reads.py:829](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/reads.py#L829)) keys by task_id (`JobName`). Covers only the tasks referenced in this tick to bound memory.

`MappingProxyType` is the standard-library read-only view; mutation raises `TypeError`. Mandatory — relying on convention failed in `20260223_iris_resultectomy.md` cleanup, so we use the type system here.

### `reads.build_control_tick`

```python
def build_control_tick(
    tx: Tx,
    *,
    health: WorkerLivenessSource,
    attrs: WorkerAttrsSource,
) -> ControlTick:
    """Build one ``ControlTick`` from a single read_snapshot Tx.

    Caller owns the ``read_snapshot()`` context manager. This function only
    issues reads via ``tx`` so the returned ``ControlTick`` is a consistent
    SQLite WAL snapshot. The first read inside ``tx`` is what pins the
    snapshot; subsequent writers cannot change what this Tx sees.

    Cost is dominated by the four queries listed in "Snapshot-build order"
    above. Roughly equal to the union of today's per-loop reads on the
    same DB state; the win is that this happens once per tick instead of
    2–3 times.

    Raises any error the underlying reads raise (lock-timeout, decode);
    caller is expected to handle by aborting the tick (see TickDriver._tick).
    """
```

### `TickDriver`

```python
class TickDriver:
    """Owner of the scheduler + autoscaler tick loop.

    Replaces ``_run_scheduling_loop`` and ``_run_autoscaler_loop`` in Phase 1.
    Polling and ping continue on their own threads in Phase 1; polling joins
    this driver in Phase 2; ping remains separate permanently.

    Threading: one ``ManagedThread`` runs ``_tick`` in a loop. The thread
    *does not hold ``ControllerDB._lock``* during snapshot reads — those
    use ``read_snapshot()``. It acquires the lock once per inline write
    call in Phase 1, once per ``apply_tick_intents`` call in Phase 3.
    """

    def __init__(
        self,
        *,
        db: ControllerDB,
        transitions: ControllerTransitions,
        scheduler: Scheduler,
        autoscaler: AutoscalerRuntime | None,
        health: WorkerHealthTracker,
        worker_attrs: WorkerAttrsProjection,
        tick_interval_s: float = 1.0,
        scheduler_idle_max_s: float = 10.0,
        autoscaler_interval_s: float = 10.0,
        rpc_executor_workers: int = 8,        # Phase 2; ignored in Phase 1
        heartbeat_subbatch_cap: int = 200,    # Phase 3; ignored in Phase 1/2
    ) -> None: ...

    def start(self) -> None: ...
    def stop(self, *, join_timeout_s: float = 30.0) -> None: ...
    def join(self, timeout: float | None = None) -> None: ...

    @property
    def wake(self) -> threading.Event:
        """Set this to interrupt the current sleep and run the next tick now."""

    @property
    def diagnostics(self) -> "TickDiagnostics":
        """Atomic snapshot of last-tick timings and counters."""

    # RPC-compat surface (Phase 1):
    def get_job_scheduling_diagnostics(self, job_wire_id: str) -> str | None:
        """Same contract as ``Controller.get_job_scheduling_diagnostics`` today."""
```

### Section runners

`_run_scheduler`, `_run_autoscaler`, and (Phase 2) `_run_polling_section` are **methods on `TickDriver`** — not free functions, not methods on `Controller`. Each receives the immutable `ControlTick` and has read-only access to driver state (backoff counters, last-run timestamps, diagnostics). They call `self._transitions.*` to write inline in Phase 1/2; in Phase 3 they return intent tuples and write nothing themselves.

Migration of existing scheduler body: the Phase-1 PR refactors `_run_scheduling_iteration` ([controller.py:1809–1908](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/controller.py#L1809-L1908)) to take a `ControlTick` parameter instead of reading scheduling state itself; the body moves verbatim to `TickDriver._run_scheduler`. Same exercise for `_run_autoscaler_once` ([controller.py:2596–2618](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/controller.py#L2596-L2618)).

### `_tick()` body — complete lifecycle

```python
def _tick(self) -> None:
    """One iteration. Never raises. All exceptions are caught and logged.

    Lifecycle (strict order):
      A. Build snapshot.
      B. Run scheduler section (if due).
      C. Run autoscaler section (if due).
      D. Update diagnostics.

    Each step is wrapped so an exception in step N does not abort N+1.
    Section-level failure semantics differ per section (see below).
    """
    tick_started_ms = int(Timestamp.now().epoch_ms())

    # A. Snapshot
    try:
        snapshot = self._build_control_tick()      # one read_snapshot
    except Exception:
        logger.exception("Tick aborted: snapshot build failed")
        self._diag.record_snapshot_failure(tick_started_ms)
        return                                     # entire tick aborted

    # B. Scheduler
    scheduler_ran = False
    if self._scheduler_due(snapshot):
        try:
            self._run_scheduler(snapshot)          # Phase 1: writes inline
            scheduler_ran = True
            self._scheduler_backoff_reset()
        except Exception:
            logger.exception("Scheduler section failed; will retry next tick")
            self._scheduler_backoff_grow()         # on error, treat like idle
    else:
        self._scheduler_backoff_grow()

    # C. Autoscaler
    autoscaler_ran = False
    if self._autoscaler is not None and self._autoscaler_due(tick_started_ms):
        try:
            self._run_autoscaler(snapshot)         # Phase 1: writes inline
            autoscaler_ran = True
            self._last_autoscaler_run_ms = tick_started_ms
        except Exception:
            logger.exception("Autoscaler section failed; will retry next tick")

    # D. Diagnostics — always runs, even if previous sections failed.
    self._diag.record_tick(
        snapshot=snapshot,
        tick_started_ms=tick_started_ms,
        scheduler_ran=scheduler_ran,
        autoscaler_ran=autoscaler_ran,
    )
```

**Section failure isolation**: each section's `try/except` is independent. A poisoned scheduler does not block the autoscaler. The *snapshot itself* failing is fatal for the tick (no work runs that iteration); the next tick fires on cadence.

**`stop()` semantics**: sets a stop event watched by the outer driver loop, then `join`s the thread for up to `join_timeout_s` (default 30 s). If a tick is in flight, `stop()` waits for it to finish (no mid-tick cancellation — every section runs to completion or its own exception path). Expected tick-body wall time: ~50 ms idle; ~700–800 ms when autoscaler runs (Phase 1 inline HTTP); ~1.2–1.5 s once polling RPC fan-out joins (Phase 2). The 30 s `join_timeout_s` is the budget, not the expected wait.

### `_scheduler_due` — preserves existing hysteresis

```python
def _scheduler_due(self, snapshot: ControlTick) -> bool:
    """True when the scheduler section should run this tick.

    Hysteresis matches today's ExponentialBackoff(initial=1.0, max=10.0):
    run if (a) there is pending work, OR (b) the current backoff window
    has elapsed since the last scheduler run.
    """
    if snapshot.pending_tasks:
        return True
    return (Timestamp.now().epoch_ms() - self._last_scheduler_run_ms
            >= int(self._scheduler_backoff_s * 1000))

def _scheduler_backoff_reset(self) -> None:
    self._scheduler_backoff_s = self._scheduler_backoff_initial_s
    self._last_scheduler_run_ms = int(Timestamp.now().epoch_ms())

def _scheduler_backoff_grow(self) -> None:
    """Geometric growth toward scheduler_idle_max_s. Matches ExponentialBackoff.

    Called on both idle-skip and exception paths; the only diff is whether
    last_run_ms gets updated (idle: no, error: no — neither did work).
    """
    self._scheduler_backoff_s = min(
        self._scheduler_backoff_s * 2.0,
        self._scheduler_idle_max_s,
    )
```

This replaces the use of `rigging.timing.ExponentialBackoff` ([controller.py:1567](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/controller.py#L1567)). Behavior is identical.

### `TickDiagnostics`

```python
@dataclass(frozen=True, slots=True)
class TickDiagnostics:
    """In-process telemetry for the most recent tick.

    Reset every tick; readers grab a snapshot reference (immutable).
    Not persisted; not RPC-exposed in this phase (see "Out of scope").
    """

    # Last completed tick:
    last_tick_started_at_ms: int
    last_tick_duration_ms: int

    # Phase breakdown:
    last_snapshot_build_ms: int
    last_scheduler_section_ms: int    # 0 if not run
    last_autoscaler_section_ms: int   # 0 if not run
    last_polling_section_ms: int      # Phase 2; 0 in Phase 1
    last_apply_intents_ms: int        # Phase 3; 0 in Phase 1/2

    # Section "did it run":
    scheduler_ran: bool
    autoscaler_ran: bool
    polling_ran: bool                 # Phase 2

    # Counts:
    pending_tasks_in_snapshot: int
    healthy_workers_in_snapshot: int
    task_attempts_in_snapshot: int    # Phase 2; 0 in Phase 1
    queue_assignments_emitted: int    # Phase 3
    preemptions_emitted: int          # Phase 3
    heartbeat_updates_emitted: int    # Phase 3
    intents_dropped_stale: int        # Phase 3

    # Failure counters (lifetime, not per-tick — diff across snapshots):
    snapshot_failures: int
    scheduler_section_failures: int
    autoscaler_section_failures: int
    polling_section_failures: int     # Phase 2
    apply_intents_failures: int       # Phase 3
```

The driver maintains a `_diag` mutable builder internally and publishes a frozen `TickDiagnostics` to a single `threading.RLock`-guarded slot at the end of each tick. Readers (RPC handlers, tests) call `tick_driver.diagnostics` to read a consistent point-in-time view.

### `_scheduling_diagnostics` migration (Phase 1, RPC-visible)

The existing `Controller._scheduling_diagnostics: dict[str, str]` ([controller.py:1320](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/controller.py#L1320)) is populated by `_cache_scheduling_diagnostics` ([controller.py:2180](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/controller.py#L2180)) and read by `get_job_scheduling_diagnostics` ([controller.py:2218](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/controller.py#L2218)). The RPC handler (service.py — wired through `controller.get_job_scheduling_diagnostics`) is unchanged.

Migration:
- `_scheduling_diagnostics` dict moves to `TickDriver._scheduling_diagnostics`, guarded by a small `RLock` to make read/write atomic across threads (the RPC handler thread vs. the tick thread).
- `_cache_scheduling_diagnostics` becomes `TickDriver._cache_scheduling_diagnostics`; called from inside `_run_scheduler` at the same point in the cycle.
- `Controller.get_job_scheduling_diagnostics(job_wire_id)` delegates to `self._tick_driver.get_job_scheduling_diagnostics(job_wire_id)`.

No RPC-surface change. No persisted-state change. Test: existing `test_controller_diagnostics.py` (if present) or a new test asserting RPC parity before/after.

### Controller `__init__` delta (Phase 1)

Attributes **deleted** from `Controller.__init__`:
- `self._scheduling_wake: threading.Event` (deleted in Phase 2; in Phase 1 still set by external producers, see below)
- `self._scheduling_round: int`
- `self._scheduling_diagnostics: dict[str, str]` (moved to `TickDriver`)

Threads **deleted** in Phase 1: `self._scheduling_thread`, `self._autoscaler_thread` (their `start`/`stop`/`join` lines too — full list in `controller.py:1513–1545`).

Attributes **kept**: `self._polling_thread` (Phase 2 deletes it), `self._ping_thread`, `self._direct_provider_thread`, `self._prune_thread`, `self._polling_wake` (Phase 2 folds into `TickDriver.wake`), `self._threads: ManagedThreadGroup`.

Attributes **added**:
- `self._tick_driver: TickDriver`
- `Controller.get_job_scheduling_diagnostics(...)` body becomes a one-line delegate.

Migration plan:
1. Add `TickDriver` and its tests behind no flag.
2. Switch `Controller.__init__` to spawn `TickDriver`; delete the two threads.
3. Switch the RPC delegate.
4. Land in one PR with parity tests green.

### Wake-event migration

Today's producers of `_scheduling_wake` (set the event to re-run the scheduler):

- `controller.py:1350` — `submit_job` after writing.
- `controller.py:1504` — `apply_heartbeats_batch` when capacity freed.
- (Any others discovered during implementation must be enumerated in the PR description.)

Today's producers of `_polling_wake`:

- `controller.py:1351`, `1505`, `2143` — same paths plus `submit_job` / task state changes.

**Phase 1**: `TickDriver.wake` is a new Event. Producers of `_scheduling_wake` *additionally* set `TickDriver.wake`. `_scheduling_wake` itself is deleted because nothing reads it (the scheduler thread is gone). Polling thread still reads `_polling_wake` unchanged.

**Phase 2**: producers of `_polling_wake` migrate to setting `TickDriver.wake` instead. `_polling_wake` is deleted.

The migration is a grep-and-replace exercise. The PR description must list every producer found and confirm it was migrated.

---

## Phase 2 contracts

### `ControlTick` additions

```python
# Added to ControlTick dataclass (still frozen):
task_attempts: tuple[ActiveTaskRow, ...]                 # unfinished worker-bound
run_request_templates: Mapping[JobName, "job_pb2.RunTaskRequest | None"]  # MappingProxyType
```

`ActiveTaskRow` is the existing type at [reads.py:50](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/reads.py#L50). The filter ("unfinished worker-bound" — `worker_id IS NOT NULL AND finished_at_ms IS NULL`) matches the current polling-loop query at controller.py:2339–2366.

`run_request_templates` covers only the job IDs the polling fan-out will hit. Built via `reads.bulk_run_request_templates(tx, transitions, job_ids_in_tick)` which iterates over `transitions.run_request_template(snap, job_id)` (existing method at [transitions.py:966](https://github.com/marin-community/marin/blob/0d440a1b20d23f038fce7c5e0d6ab6d9833fb268/lib/iris/src/iris/cluster/controller/transitions.py#L966)); the template cache inside `ControllerTransitions` is unchanged.

### `_run_polling_section`

```python
def _run_polling_section(
    self,
    tick: ControlTick,
) -> "PollingSectionResult":
    """Run the polling RPC fan-out + heartbeat-apply for this tick.

    Calls into self._rpc_executor (ThreadPoolExecutor with rpc_executor_workers
    workers). Each PollTasksRequest has a per-RPC timeout (5 s, matching
    today's polling loop). RPCs whose futures don't complete within
    (tick_interval_s - 200ms) are timed out and counted; their workers
    are NOT marked failed by this path (ping owns failure).

    Returns a PollingSectionResult with the heartbeat-apply intents/results.
    Phase 2: calls transitions.apply_heartbeats_batch inline.
    Phase 3: returns intents (no write).
    """
```

### `PollingSectionResult`

```python
@dataclass(frozen=True, slots=True)
class PollingSectionResult:
    """Outcome of one polling section.

    In Phase 2, ``hb_apply_result`` is populated (the section called
    apply_heartbeats_batch inline). In Phase 3, ``hb_intents`` is populated
    and ``hb_apply_result`` is None — the tick driver will call
    apply_tick_intents.
    """

    workers_polled: int
    rpcs_timed_out: int
    rpcs_failed: int
    hb_apply_result: "HeartbeatApplyResult | None"            # Phase 2 path
    hb_intents: tuple["HeartbeatApplyIntent", ...]            # Phase 3 path
    fail_worker_intents: tuple["WorkerFailureIntent", ...]    # only when ping threshold exceeded
```

### RPC executor lifecycle

- Owned by `TickDriver`. Constructed in `__init__` with `max_workers=rpc_executor_workers` (default 8).
- `stop()` calls `executor.shutdown(wait=True, cancel_futures=True)` after the current tick body returns.
- The pool size 8 is intentionally small. At 340 workers and 1 s ticks, fanout uses futures faster than the pool can synchronously execute them — but the worker RPCs are network-bound, so blocked workers don't consume CPU. If saturation becomes a problem (`diagnostics.rpcs_timed_out` rising), increase the pool size; do not change it without a measurement.
- The pool is NOT shared with anything else. Scheduler and autoscaler do no RPCs from inside the tick.

### Memory bound

After Phase 2, `ControlTick.task_attempts` carries every unfinished worker-bound attempt — production scale ~200k rows × ~200 B each ≈ 40 MB. The tuple is held only during the tick body (~1 s) and dropped at `_tick` return. We rely on Python GC; no explicit `del`. If memory growth becomes a concern at 10× scale, follow-up work materializes the tuple lazily per consumer (Open Question 5).

---

## Phase 3 contracts

### Intent taxonomy

In `intents.py`:

```python
from dataclasses import dataclass
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2
from iris.cluster.controller.transitions import TaskUpdate
from iris.cluster.controller.reads import ReservationClaim

@dataclass(frozen=True, slots=True)
class QueueAssignmentIntent:
    task_id: JobName
    attempt_id: int             # CAS target: live tasks.current_attempt_id
    worker_id: WorkerId
    # CAS check: tasks.current_attempt_id == attempt_id
    #            AND tasks.state == PENDING
    # On mismatch: dropped(reason="stale_attempt" | "task_not_pending")

@dataclass(frozen=True, slots=True)
class PreemptionIntent:
    task_id: JobName
    attempt_id: int             # CAS target: live tasks.current_attempt_id
    new_state: int              # one of {TASK_STATE_PREEMPTED, TASK_STATE_KILLED}
    reason: str
    # CAS check: tasks.current_attempt_id == attempt_id
    #            AND tasks.state IN {ASSIGNED, BUILDING, RUNNING}
    # On mismatch: dropped(reason="stale_attempt" | "task_terminal")

@dataclass(frozen=True, slots=True)
class HeartbeatApplyIntent:
    worker_id: WorkerId
    updates: tuple[TaskUpdate, ...]
    # CAS is per-update (each TaskUpdate already carries an attempt_id).
    # The intent itself is applied if the worker still exists; missing workers
    # are dropped(reason="worker_unknown"); per-update CAS failures are
    # accumulated in ApplyTickResult.dropped_stale, not in the intent itself.

@dataclass(frozen=True, slots=True)
class WorkerFailureIntent:
    worker_id: WorkerId
    address: str | None
    reason: str
    # CAS check: workers.state != FAILED (idempotent — repeated failures no-op).
    # On mismatch: dropped(reason="already_failed").

@dataclass(frozen=True, slots=True)
class TaskTimeoutCancelIntent:
    task_id: JobName
    attempt_id: int
    reason: str
    # CAS check: tasks.current_attempt_id == attempt_id
    #            AND tasks.state NOT IN terminal_states
    # On mismatch: dropped(reason="stale_attempt" | "task_terminal").

@dataclass(frozen=True, slots=True)
class TaskUnschedulableIntent:
    task_id: JobName
    reason: str
    # No CAS — state transition to UNSCHEDULABLE is monotonic.
    # Apply checks tasks.state == PENDING; non-PENDING task is dropped(reason="task_not_pending").

@dataclass(frozen=True, slots=True)
class ReservationClaimsIntent:
    """Bulk replacement of reservation_claims for the current tick.

    No CAS — by construction this is a full-table replacement built from a
    consistent snapshot. The intent is applied unconditionally as the very
    first step of apply_tick_intents so that subsequent intents see the
    post-replacement claim set.
    """
    claims: tuple[ReservationClaim, ...]
```

`EndpointPruneIntent` is **not** in this taxonomy. Endpoint pruning cascades from worker-failure via `transitions.remove_endpoints_for_workers`, called from inside `apply_tick_intents` whenever a `WorkerFailureIntent` succeeds. Documented as a side-effect of `WorkerFailureIntent`, not a separate intent — keeping the taxonomy minimal.

### `IntentBatch`

```python
@dataclass(frozen=True, slots=True)
class IntentBatch:
    queue_assignments: tuple[QueueAssignmentIntent, ...] = ()
    preemptions: tuple[PreemptionIntent, ...] = ()
    heartbeat_applies: tuple[HeartbeatApplyIntent, ...] = ()
    worker_failures: tuple[WorkerFailureIntent, ...] = ()
    task_timeouts: tuple[TaskTimeoutCancelIntent, ...] = ()
    task_unschedulable: tuple[TaskUnschedulableIntent, ...] = ()
    reservation_claims: ReservationClaimsIntent | None = None

    @property
    def is_empty(self) -> bool: ...
    def total_count(self) -> int: ...
```

**Construction pattern**: each section returns a typed result (`SchedulerSectionResult`, `AutoscalerSectionResult`, `PollingSectionResult`) carrying its intents as tuples. The tick driver concatenates these directly into a single `IntentBatch(...)` constructor call at the end of the tick — no Builder type. Sections do NOT see each other's intents within the same tick. `IntentBatch` itself is always immutable.

### `apply_tick_intents`

```python
class ControllerTransitions:
    def apply_tick_intents(
        self,
        cur: Tx,
        batch: IntentBatch,
    ) -> ApplyTickResult:
        """Apply every intent in batch inside one write transaction.

        Empty batch (``batch.is_empty``): returns ``ApplyTickResult.empty()``
        without acquiring the write lock or opening a transaction.

        Non-empty batch: caller wraps in ``db.transaction()``; this method
        runs entirely inside that transaction. On any uncaught exception,
        the caller's transaction rolls back; ``ApplyTickResult`` is NOT
        returned. CAS-failed intents do NOT raise — they are recorded in
        ``result.dropped_stale`` and skipped.

        Application order (justified below):
          1. reservation_claims (full-table replace)
          2. task_unschedulable
          3. task_timeouts
          4. preemptions
          5. queue_assignments
          6. heartbeat_applies
          7. worker_failures

        Hooks (in-memory projection updates) fire after COMMIT under the
        write lock. Order of hook execution matches application order.
        If a hook raises, the COMMIT has already succeeded; remaining hooks
        are run inside a try/except so a single buggy hook does not poison
        the rest. Hook exceptions are logged and counted in
        ``result.hook_failures``.
        """

@dataclass(frozen=True, slots=True)
class ApplyTickResult:
    applied: int
    dropped_stale: tuple[DroppedIntent, ...]
    hb_apply_result: "HeartbeatApplyResult | None"
    queued_task_ids: tuple[JobName, ...]
    preempted_task_ids: tuple[JobName, ...]
    failed_worker_ids: tuple[WorkerId, ...]
    hook_failures: int

    @classmethod
    def empty(cls) -> "ApplyTickResult": ...

@dataclass(frozen=True, slots=True)
class DroppedIntent:
    kind: str                       # "queue_assignment" | "preemption" | ...
    task_id: JobName | None
    worker_id: WorkerId | None
    reason: str                     # see per-intent CAS table above
```

### Application order rationale

Each adjacent pair preserves an invariant that the bug-prevention prose makes explicit. The order is **not** historical convention; each pairing is justified.

| Order | Reason adjacent step happens *before* the next |
|---|---|
| 1. `reservation_claims` → 2. `task_unschedulable` | Reservation claims replaced first so that an unschedulable check that runs next does not race against a stale claim set. The claim set is the input the scheduler used to decide unschedulability. |
| 2. `task_unschedulable` → 3. `task_timeouts` | Unschedulable is a stronger termination than timeout; if both fire on the same task in the same tick, the unschedulable transition wins (matches today's `_apply_terminal_transition` ordering). |
| 3. `task_timeouts` → 4. `preemptions` | A timeout-cancelled task should not also be preempted (would double-write the same `task_attempts` row). |
| 4. `preemptions` → 5. `queue_assignments` | Preemptions free worker capacity. Assignments are based on the snapshot (which already accounted for preemption decisions made in this tick by the scheduler), but the post-preemption state must commit *before* assignments so that the worker reads downstream see consistent state. **Pure ordering hygiene** — the scheduler already produced consistent intents from the snapshot. |
| 5. `queue_assignments` → 6. `heartbeat_applies` | Heartbeat applies may transition newly-assigned tasks (a worker can respond fast). Doing assignments first ensures the heartbeat finds the assigned row. |
| 6. `heartbeat_applies` → 7. `worker_failures` | Heartbeats process per-worker updates first; if a worker is *also* being failed this tick (ping threshold), the failure applies last, ensuring no in-flight heartbeat update is dropped against a worker already marked FAILED. |

If any of these orderings prove wrong in production, the spec should be updated and the test in `test_apply_tick_intents.py` strengthened to catch the regression.

### Stale-intent flow

CAS-failed intents are dropped with a reason. The driver:

1. Logs the drop count via `diagnostics.intents_dropped_stale`.
2. Sets `self.wake` so the next tick fires immediately.
3. Does **not** re-enqueue the dropped intent. The section that emitted it regenerates from the fresh snapshot on the next tick.

This is intentional — keeping the section the source of truth for "should this intent exist" prevents stale-intent thrashing. Open Question 3 in the design proposes per-intent backoff if regeneration tight-loops; not added pre-emptively.

### Lock-hold bound for ping coexistence

`apply_tick_intents` is the single longest write-lock-hold per tick in Phase 3. To bound ping-write latency:

- `HeartbeatApplyIntent` accumulating > 200 `TaskUpdate` entries causes `apply_tick_intents` to split into multiple intra-tick transactions of ≤ 200 updates each. Each sub-tx releases and re-acquires the write lock, giving ping (and any other contender) a chance to interleave.
- Default cap (200) chosen against the bench measurements (Phase 1 `apply_heartbeats_batch` p95 at 200 updates ≈ 50 ms). Configurable via `TickDriver(heartbeat_subbatch_cap=200)`.

Bench measurements (SA Core PR): `apply_heartbeats_batch` p95 at the production heartbeat shape (~340 workers × ~3 updates/worker ≈ 900 updates) is ~10 ms p95. At a 200-update cap, expected lock-hold is well under 5 ms p95; the spec budgets 50 ms as conservative headroom. Ping's 5 s cadence has ample margin.

### Ping write path — explicit exception

Ping (`_run_ping_loop`) continues to call `transitions.update_worker_pings` and `transitions.fail_workers_batch` directly. These are NOT part of `IntentBatch`. The rationale is documented in design.md (sub-second responsiveness; writes are tiny). Tests must verify ping writes still succeed when a Phase 3 bulk apply is concurrently running.

---

## Test contract

### Phase 1: `test_control_tick.py`

Fixtures:

```python
@pytest.fixture
def checkpoint_db(tmp_path: Path) -> ControllerDB:
    """Returns a ControllerDB seeded from the production checkpoint.

    Resolution order:
      1. ``IRIS_BENCH_CHECKPOINT`` env var (path to local checkpoint dir).
      2. Local cache at ``~/.cache/iris/benchmark_checkpoint/`` if present.
      3. Download from gs://marin-us-central2/iris/marin/state (via
         download_checkpoint_to_local) — only if --download-checkpoint
         pytest flag is set. CI sets this flag; local devs default to
         the cache and the test is skipped if absent.

    The checkpoint is read-only at the source; the fixture clones into
    tmp_path so tests can mutate.
    """
```

Parity assertions:

```python
def test_control_tick_parity(checkpoint_db: ControllerDB) -> None:
    health = WorkerHealthTracker()
    _seed_health(checkpoint_db, health)  # mark all persisted workers live

    with checkpoint_db.read_snapshot() as tx:
        tick = build_control_tick(tx, health=health, attrs=worker_attrs)

    # Pending tasks parity
    expected_pending = list(_pending_tasks_with_jobs(checkpoint_db))
    assert tuple(expected_pending) == tick.pending_tasks

    # Healthy workers parity (set-equality on worker_id since order may differ)
    with checkpoint_db.read_snapshot() as tx:
        expected_workers = reads.healthy_active_workers_with_attributes(tx, health, attrs)
    assert {w.worker_id for w in expected_workers} == {w.worker_id for w in tick.healthy_workers}
    # And content parity per worker:
    by_id = {w.worker_id: w for w in tick.healthy_workers}
    for w in expected_workers:
        assert by_id[w.worker_id] == w

    # Resource usage parity
    with checkpoint_db.read_snapshot() as tx:
        expected_usage = reads.resource_usage_by_worker(tx)
    assert dict(tick.resource_usage) == expected_usage

    # Reserved jobs parity
    assert tick.reserved_jobs == frozenset(_reserved_job_ids(checkpoint_db))
```

Functional test:

```python
def test_scheduler_runs_against_control_tick(checkpoint_db: ControllerDB) -> None:
    """Scheduler called with ControlTick produces the same assignments
    as scheduler called via the old path."""
    expected = _run_scheduler_via_old_path(checkpoint_db)   # pre-refactor path retained as test fixture
    with checkpoint_db.read_snapshot() as tx:
        tick = build_control_tick(tx, health=health, attrs=attrs)
    actual = _run_scheduler_via_tick(tick, scheduler=Scheduler())
    assert _assignments_equal(expected, actual)
```

`_assignments_equal` compares ordered assignment tuples; ordering matters because today's scheduler is order-sensitive (priority band tie-breaks).

### Phase 1: `test_tick_driver.py`

```python
def test_tick_driver_lifecycle(): ...      # start, run a tick, stop within timeout
def test_tick_driver_wake(): ...           # wake interrupts sleep, runs tick promptly
def test_tick_driver_section_isolation():  # scheduler exception does not block autoscaler
def test_tick_driver_snapshot_failure():   # snapshot build raise → tick aborted but driver alive
def test_tick_driver_diagnostics(): ...    # diagnostics reflect last tick
def test_scheduler_diagnostics_rpc(): ...  # parity of get_job_scheduling_diagnostics
```

### Phase 3: `test_apply_tick_intents.py`

```python
# Per-intent CAS contract:
def test_queue_assignment_stale_attempt_dropped(): ...
def test_preemption_terminal_dropped(): ...
def test_worker_failure_idempotent_drop(): ...
# ...one per intent kind.

# Application-order invariants:
def test_preemption_before_assignment_for_same_worker(): ...
def test_heartbeat_before_worker_failure(): ...

# Property: per-intent equivalence to per-loop method
@hypothesis.given(intents_strategy)
def test_apply_tick_intents_equivalent_to_per_loop(intents: list[Intent]): ...

# Empty batch:
def test_empty_batch_no_lock(monkeypatch): ...  # asserts no write lock acquired

# Lock-hold bound:
def test_heartbeat_subbatch_cap_releases_lock(monkeypatch): ...
```

---

## Rollback

### Per-phase revert plan

Each phase ships as **one PR** with a single revert-able commit. There is no feature flag. Revert = `git revert <commit>` and ship. Because each phase deletes the older code path, revert *re-introduces* the old code from the prior commit; no orphaned dead code.

### Production rollout thresholds

After Phase 1 deploys:

| Metric | Source | Threshold | Action |
|---|---|---|---|
| Tick duration p95 | `TickDiagnostics.last_tick_duration_ms` (export via Prometheus follow-up; until then, log scrape) | > 3 s sustained > 5 min | revert |
| Snapshot build p95 | `last_snapshot_build_ms` | > 500 ms sustained > 5 min | investigate, possibly revert |
| Snapshot failures / min | `snapshot_failures` diff | > 5 / min | revert |
| Scheduler-section failures / min | `scheduler_section_failures` diff | > 1 / min | investigate |
| `get_job_scheduling_diagnostics` returns None for known job | RPC | sustained | revert (diagnostics migration broken) |

After Phase 3 deploys, additionally:

| Metric | Threshold | Action |
|---|---|---|
| Ping write p99 | > 200 ms | investigate cap |
| `intents_dropped_stale / total_intents` | > 5% sustained | investigate; possible regeneration thrash |
| `apply_intents_failures / min` | > 1 / min | revert |

These thresholds are starting points; the implementation PR is expected to tune them post-canary.

### Kill switch

There is no in-band kill switch. The two operational levers are:
1. `git revert` of the phase's commit and redeploy (1–2 hours including CI).
2. A controller restart, which falls back to whatever code is deployed (no special "safe mode").

For Phase 1, this is judged acceptable because revert is a small diff (the TickDriver replaces two well-defined loops). Phase 3's revert is larger but isolated to `transitions.py` and the intent dispatch in `TickDriver._tick`.

---

## Errors

No new exception types. Behavioral changes:

- `apply_tick_intents` never raises for CAS-failed intents (recorded in `result.dropped_stale`). It raises on:
  - SQLite transactional errors (lock timeout, schema violation) — same surface as today's `transitions.*` methods.
  - Programmer error (an intent with a `task_id` not in the DB and not in the snapshot — indicates a bug in the section that emitted it).
- `TickDriver._tick` catches every section's exception independently. The driver thread never dies from a section exception; only from `stop()`.
- `build_control_tick` raises on any read failure. `TickDriver._tick` catches and records; no tick body runs that iteration.
- Snapshot field access (`tick.pending_tasks[i]`) — frozen tuples, raises `IndexError` like any tuple. Mutation attempts on `MappingProxyType` raise `TypeError`. These should never fire in correct code.

---

## Out of scope

- **Autoscaler-inline-HTTP fix.** Phase 1 runs `autoscaler.refresh()` inline including `tpu_describe` calls. A sibling design proposes a background `TpuMonitor` cache; not a hard dependency. Phase 1 caps `autoscaler_interval_s ≥ 10.0` to bound the impact.
- **Service-layer (gRPC) refactor.** `list_jobs` and other gRPC paths under `service.py` are the single largest CPU sink in the profile but are touched by this design only at the diagnostics-RPC delegate. A separate follow-up should address the gRPC path.
- **`_direct_provider_thread` and `_prune_thread`.** Continue to run on their own threads. Could fold into the tick driver later; not now.
- **Diagnostics RPC.** `TickDiagnostics` is in-process; surfacing it via a new RPC handler is a follow-up.
- **Backwards-compatibility flag.** Explicit non-goal. Each phase rolls forward; revert is the rollback.
- **Sub-second autoscaler / streaming `ControlTick`.** Phase 2 holds ~40 MB per tick at current scale. A streaming/lazy materialization (Open Question 5) is a follow-up if growth materializes.
- **`replace_reservation_claims` call-site consolidation.** Open Question 4 in design; spec treats Phase 3 as "one intent per tick" but acknowledges reviewers may push for a per-call-site split.
- **`MappingProxyType` for the two `dict` fields after Phase 2 additions.** This spec mandates `MappingProxyType` from Phase 1 onward (a reversal from a prior draft that left it for "later"). Mandatory because relying on convention has failed before.
