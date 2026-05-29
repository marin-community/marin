# Reconcile package — design revision v4 (rev 2, post-Codex)

Author: russell + claude — 2026-05-28
Branch: `weaver/iris-reconcile-performance`
Supersedes: `reconcile-package-split.md` v3 (already landed in stages A → C3).

## Changes vs v4 rev 1

Codex revalidated v4 rev 1 with verdict APPROVE WITH FIXES. Five must-fix
items addressed here:

1. **Resolve the working_state ↔ aggregate cycle.** `WorkingState` is now a
   *narrow record bag* — overlay reads + flat `record_*` methods. It does
   NOT own transition semantics. Transition helpers
   (`mark_task_terminating`, `finalize_attempt`, `finalize_terminal_job`,
   `_apply_transitions`) live in their aggregate files.
2. **Bring back a thin orchestrator.** `batches.py` (the rename of v3's
   `sweep.py`) imports only PUBLIC aggregate functions. It composes
   primitives: task → peer cascade → job recompute. This is the cleanest
   way to keep aggregate files internally pure without cycles.
3. **Specify non-current-attempt loading.** Scoped loaders accept
   `extra_attempt_keys` and `observation_uids` so direct-provider stale-
   attempt validation and reconcile-observation UID resolution work
   inside the closure contract.
4. **Define creation/refresh loaders.** `load_creation_context_for_job`
   and `load_refresh_context_for_worker` exist; their return types live
   beside them in `loader.py`. No placeholder dataclasses anywhere.
5. **Move `apply_reconcile_observations` to `ops/worker.py`.** The kernel
   sweep `worker.apply_reconcile_batch` is in `reconcile/worker`; the RPC
   adapter belongs in `ops/worker`. `ops/task.py` is task-RPC-only.

## What v4 fixes

This revision responds directly to a Codex review of the landed code
(`/tmp/codex-slop.md`). The landed code carried v3 forward correctly but
v3 had its own gaps that surfaced once the modules were inhabited. v4 is
about closing those gaps with **real boundaries**, not more re-export
sprawl.

The ten Codex findings, addressed in order:

| Codex finding | v4 fix |
|---|---|
| 1. Loader exposes only generic snapshot; no scoped loaders | Replace `load_transition_snapshot()` with `load_full_snapshot` + three scoped loaders. Each owns its closure. |
| 2. `TransitionSnapshot` not actually closed (peers + descendant tasks missing) | Closure contract enforced in the scoped loaders + a loader closure test. |
| 3. Direct-provider transitions duplicate heartbeat transitions | Route both through one `task._apply_transitions` core called from `batches.apply_heartbeats_batch` and `batches.apply_direct_provider_updates_batch`. |
| 4. `_mark_task_terminating` lives in `state.py`, leaks transition logic into state | Move into `task.py` as the public `task.mark_task_terminating(state, …)`. peers/job/worker call `task.mark_task_terminating`; no module-level helper in state. (rev 2: NOT promoted onto WorkingState — that would create the working_state↔aggregate cycle Codex flagged.) |
| 5. `JobCreationContext` / `WorkerRefreshContext` are unfinished placeholders | Move into `loader.py` next to `load_creation_context_for_job` / `load_refresh_context_for_worker`. Fields limited to what `ops.job.submit` / `ops.worker.register_or_refresh` actually consume. |
| 6. `WorkerAttributeParams` lives in `reconcile/worker.py` but is scheduler/ops data | Move to `ops/worker.py`. |
| 7. `sweep.py` mixes cross-aggregate batch orchestrators with cancel + direct-provider | Delete `sweep.py`. A thin `batches.py` owns all `apply_*_batch` entry points, importing PUBLIC primitives only from `task`, `job`, `worker`, `peers`. Aggregate files contain only primitives. |
| 8. Private aggregate helpers are de-facto API (`_preempt_one`, `_apply_task_transitions`, …) | Aggregate files expose a small set of PUBLIC primitives. `batches.py` is the only cross-aggregate caller; aggregate files do not import each other except for the single `peers → task` edge. |
| 9. `has_reservation_flag` is in `reconcile/task.py` even though it's an RPC adapter | Move to `ops/job.py` (it's used by `ops.job.submit`). |
| 10. `ops/reservation.py` is empty TODO slop | Delete it. Document the reservation loops live on `controller.py` until they're worth extracting. |

Plus Codex's "centralized mutations" cross-cutting concern: v4 makes
`Mutation` a Protocol and unifies `ControllerEffects.mutations: list[Mutation]`.
Each aggregate file owns its own mutation classes. `mutations.py`
disappears.

Plus Codex's misnavigation traps: every module's name should be true to
its contents. v4 renames where needed.

---

## North star — unchanged

Functional Core, Imperative Shell. Pure kernel parameterized by scope.
Ad-hoc and bulk share the same kernel; they differ only in the slice the
loader hands them. What v4 fixes is the leaks in that boundary.

---

## Final layout (v4 rev 2)

```
lib/iris/src/iris/cluster/controller/
  reconcile/
    __init__.py          # empty docstring only (kept from C3c)

    # ─── leaf data shapes ───
    snapshot.py          # TransitionSnapshot + closure-helper row dataclasses
    policy.py            # constants: limits, thresholds, well-known names, predicate sets
    effects.py           # Mutation Protocol, ControllerEffects, cross-aggregate effect
                         # categories (EndpointDeletion, WorkerHealthEffect, LoggerEvent,
                         # LogEvent), and apply_effects(cur, effects, ...)

    # ─── working state (NARROW record bag) ───
    working_state.py     # WorkingState class. Overlay reads + flat record_* methods
                         # ONLY. No transition semantics. record(m: Mutation),
                         # record_endpoint_deletion, record_log_event, record_logger_event,
                         # record_worker_heartbeat, record_worker_build_failed,
                         # record_worker_make_unhealthy. Reads: job_config, job_basis,
                         # task_state, task_state_histogram, first_task_error,
                         # active_tasks_for_job.

    # ─── pure I/O ───
    loader.py            # load_full_snapshot, load_workers_slice, load_jobs_slice,
                         # load_tasks_slice (with extra_attempt_keys / observation_uids
                         # parameters), load_creation_context_for_job,
                         # load_refresh_context_for_worker. JobCreationContext and
                         # WorkerRefreshContext live here. NO generic load_transition_snapshot.

    # ─── per-aggregate rules + mutations. PUBLIC primitives only — no  ───
    # ─── cross-aggregate imports between these four files.             ───
    task.py              # TaskMutation, AttemptMutation, TaskUpdate, TerminalDecision,
                         # TerminalKind, HeartbeatApplyRequest. PUBLIC:
                         # mark_task_terminating(state, ...), finalize_attempt(...),
                         # apply_one_transition(state, worker_id, update, ...),
                         # active_row_from_snapshot, task_is_finished.
                         # No imports from job/worker/peers/batches.
    job.py               # JobStateMutation, CascadeKillJobMutation. PUBLIC:
                         # recompute_state(state, job_id, ...), kill_non_terminal_tasks(...),
                         # cascade_children(...), finalize_terminal(...).
                         # No imports from task/worker/peers/batches.
    worker.py            # ReconcileRow, ReconcileInputs, WorkerReconcilePlan,
                         # ReconcileResult, _resolve_task_failure_state, plan_all,
                         # plan_one, filter_observations_to_plan, observations_to_updates,
                         # assigned_updates_from_plan, apply_worker_failure_one(state, ...).
                         # No imports from task/job/peers/batches.
    peers.py             # PUBLIC: find_coscheduled_siblings,
                         # terminate_coscheduled_siblings(state, ...),
                         # requeue_coscheduled_siblings(state, ...). Imports
                         # `task.mark_task_terminating` since peer cascade is a
                         # task-state primitive. No imports from job/worker/batches.

    # ─── thin orchestration. PUBLIC imports only. ───
    batches.py           # apply_heartbeats_batch, apply_observations_batch (one core
                         # for heartbeat AND direct-provider), apply_terminal_decisions_batch,
                         # apply_reconcile_batch, apply_worker_failures_batch,
                         # apply_cancel_job_batch. Each composes task/job/worker/peers
                         # PUBLIC primitives. NO module imports batches (it's a leaf for
                         # callers).

  ops/
    __init__.py          # empty
    job.py               # submit, cancel, remove_finished, has_reservation_flag,
                         # _submit_reservation_holder (private helper for the embedded
                         # reservation-holder path Codex flagged).
                         # cancel uses load_jobs_slice + batches.apply_cancel_job_batch.
    worker.py            # register_or_refresh, fail, apply_reconcile_observations
                         # (MOVED from ops/task.py — kernel sweep + types both live in
                         # reconcile/worker, so the RPC adapter belongs here).
                         # WorkerAttributeParams, WorkerFailureBatchResult.
    task.py              # queue_assignments, Assignment, apply_heartbeats,
                         # apply_terminal_decisions, apply_provider_updates.
                         # Task-RPC adapters only.
    # NO reservation.py — deleted.

  # unchanged
  reads.py, writes.py, task_state.py, direct_provider.py, codec.py,
  schema.py, db.py, projections/, …
```

What's gone:
- `reconcile/sweep.py` — replaced by `batches.py` whose imports are PUBLIC only.
- `reconcile/mutations.py` — `Mutation` Protocol lives in `effects.py`; concrete mutation classes live in their aggregate.
- `reconcile/state.py` — splits into `snapshot.py` + `policy.py` + `working_state.py`. The misc-bucket goes away.
- `ops/reservation.py` — deleted until reservation ops are real.

What's new:
- `reconcile/snapshot.py`, `reconcile/policy.py`, `reconcile/working_state.py`, `reconcile/batches.py`.

---

## Module-by-module contracts

### `reconcile/snapshot.py`

Pure data shapes. Leaf module — depends only on `iris.cluster.types`,
`iris.cluster.controller.task_state`, `rigging.timing`.

```python
@dataclass(frozen=True, slots=True)
class JobConfigRow: ...

@dataclass(frozen=True, slots=True)
class JobStateBasis: ...

@dataclass(frozen=True, slots=True)
class JobDescendants: ...

@dataclass(frozen=True, slots=True)
class TaskHistogramRow: ...

@dataclass(frozen=True)
class TransitionSnapshot:
    """Pre-loaded inputs for one pure-function call into the state machine.

    A snapshot is CLOSED under the kernel's read patterns. The closure
    contract per relation is documented in loader.py and asserted by the
    loader-closure test.
    """
    now: Timestamp
    tasks: dict[JobName, TaskDetailRow]
    attempts: dict[tuple[JobName, int], AttemptRow]
    attempt_uid_index: dict[AttemptUid, tuple[JobName, int]]
    job_configs: dict[JobName, JobConfigRow]
    job_state_basis: dict[JobName, JobStateBasis]
    job_descendants: dict[JobName, JobDescendants]
    all_tasks_by_job: dict[JobName, tuple[TaskHistogramRow, ...]]
    active_tasks_by_job: dict[JobName, tuple[ActiveTaskRow, ...]]
    active_workers: frozenset[WorkerId]
```

No methods. No logic. Just dataclasses. ~120 LOC.

### `reconcile/policy.py`

Constants and small predicate sets. Leaf — only stdlib and types/protos.

```python
MAX_REPLICAS_PER_JOB: int
DEFAULT_MAX_RETRIES_PREEMPTION: int
RESERVATION_HOLDER_JOB_NAME: str
HEARTBEAT_STALENESS_THRESHOLD: Duration

FAILURE_TASK_STATES: frozenset[int]
NON_TERMINAL_TASK_STATES: frozenset[int]
CANCEL_GUARD_STATES: frozenset[int]   # TERMINAL_JOB_STATES - {WORKER_FAILED}
_ERROR_STATES: frozenset[int]
_TERMINAL_STATE_REASONS: dict[int, str]
```

~50 LOC. Replaces the misc-constants part of `state.py`.

### `reconcile/effects.py`

The effect contract.

```python
class Mutation(Protocol):
    """Row-mutating effects. apply(cur) issues SQL on the active Tx."""
    def apply(self, cur: Tx) -> None: ...

@dataclass(frozen=True, slots=True)
class EndpointDeletion:
    task_id: JobName

@dataclass(frozen=True, slots=True)
class WorkerHealthEffect:
    heartbeat: tuple[WorkerId, ...] = ()
    build_failed: tuple[WorkerId, ...] = ()
    make_unhealthy: tuple[WorkerId, ...] = ()

@dataclass(frozen=True, slots=True)
class LoggerEvent: ...
@dataclass(frozen=True, slots=True)
class LogEvent: ...

@dataclass
class ControllerEffects:
    """Pure output of one state-machine call. Caller persists with apply_effects."""

    # Row-mutating effects. Flat list. Each implements Mutation.
    mutations: list[Mutation] = field(default_factory=list)

    # Post-commit cross-aggregate effect categories. NOT Mutation — these
    # do not issue SQL during the active Tx; they fire after commit.
    endpoint_deletions: list[EndpointDeletion] = field(default_factory=list)
    worker_health: WorkerHealthEffect = field(default_factory=WorkerHealthEffect)
    log_events: list[LogEvent] = field(default_factory=list)
    logger_events: list[LoggerEvent] = field(default_factory=list)


def apply_effects(
    cur: Tx,
    effects: ControllerEffects,
    *,
    health: WorkerHealthTracker,
    endpoints: EndpointsProjection,
    now: Timestamp,
) -> None:
    for m in effects.mutations:
        m.apply(cur)
    for d in effects.endpoint_deletions:
        endpoints.remove_by_task(cur, d.task_id)
    # … post-commit hooks for worker_health, log_events, logger_events
```

Key wins:
- Flat `mutations: list[Mutation]` replaces the four typed lists
  (`attempt_mutations`, `task_mutations`, `job_state_mutations`,
  `cascade_kill_mutations`). No `isinstance` dispatch in `WorkingState.record`.
- Concrete mutation classes (`TaskMutation`, `AttemptMutation`,
  `JobStateMutation`, `CascadeKillJobMutation`) live in their aggregate
  file, not centrally. effects.py only knows the Protocol.
- Cross-aggregate effect categories stay here because their post-commit
  application semantics live with `apply_effects`.

~150 LOC.

### `reconcile/working_state.py`

The kernel's mutable scratch space. **Narrow record bag — no transition
semantics.** Per Codex must-fix #1 and #3: WorkingState aggregates effects
but does not own decisions about what those effects should be.

```python
@dataclass
class WorkingStateOverlay:
    """The three dicts the overlay tracks. Passed to mutation overlay_apply()."""
    task_states: dict[JobName, int]
    task_errors: dict[JobName, str | None]
    job_states: dict[JobName, int]


class WorkingState:
    def __init__(self, snapshot: TransitionSnapshot) -> None: ...

    # Read accessors (overlay-aware). Snapshot is the source of truth;
    # overlay layers prospective state on top so iteration N+1 sees N's
    # changes.
    @property
    def now(self) -> Timestamp: ...
    @property
    def effects(self) -> ControllerEffects: ...

    def job_config(self, job_id) -> JobConfigRow | None: ...
    def job_basis(self, job_id) -> JobStateBasis | None: ...
    def task_state(self, task_id) -> int | None: ...
    def task_state_histogram(self, job_id) -> dict[int, int]: ...
    def first_task_error(self, job_id) -> str | None: ...
    def active_tasks_for_job(self, job_id, *, exclude, states) -> list[ActiveTaskRow]: ...

    # Record API. ONE polymorphic record() for any Mutation; named
    # methods for the cross-aggregate effect categories that aren't
    # row mutations.
    def record(self, mutation: Mutation) -> None:
        """Append a row mutation. If the mutation also implements
        OverlayUpdater, invoke overlay_apply() so subsequent reads see
        the prospective state."""

    def record_endpoint_deletion(self, task_id: JobName) -> None
    def record_log_event(self, event: LogEvent) -> None
    def record_logger_event(self, event: LoggerEvent) -> None
    def record_worker_heartbeat(self, worker_ids: Iterable[WorkerId]) -> None
    def record_worker_build_failed(self, worker_id: WorkerId) -> None
    def record_worker_make_unhealthy(self, worker_id: WorkerId) -> None
```

**What's intentionally NOT here:** `mark_task_terminating`,
`finalize_attempt`, `finalize_terminal_job`, `_apply_transitions`. Those
are transition rules — they belong with the aggregate that owns them.
v4 rev 1 put them on WorkingState; Codex flagged that as the same
centralization slop with public methods. Rev 2: they stay in the
aggregate files as public free functions that take `state` as a parameter.

**Resolving the import cycle.** v4 rev 1's working_state imported
`TaskMutation` etc. from aggregate files at module scope. Rev 2 doesn't.
`record()` takes the `Mutation` Protocol — WorkingState has no
type-level knowledge of concrete mutation classes. If a mutation needs
to update the overlay (e.g., `TaskMutation` changes `task_state`), it
implements an optional `OverlayUpdater` protocol:

```python
# effects.py
class OverlayUpdater(Protocol):
    def overlay_apply(self, overlay: WorkingStateOverlay) -> None: ...
```

`WorkingState.record(m)` does:
```python
self._effects.mutations.append(m)
if hasattr(m, "overlay_apply"):
    m.overlay_apply(self._overlay)
```

So `working_state.py` imports only `effects.py` + `snapshot.py` — both
leaves. No import of `task.py`, `job.py`, `worker.py`. Cycle gone.

Import direction:
- `working_state.py` imports `snapshot.py`, `effects.py`. NOTHING ELSE.
- Aggregate files receive `state: WorkingState` as a parameter. They do
  NOT import `working_state.py` at module scope either; their `state`
  parameter is annotated against the `WorkingState` Protocol exposed
  from `effects.py`, OR they import `working_state.py` ONE-WAY (since
  working_state.py doesn't import them, no cycle).

~200 LOC.

### `reconcile/loader.py`

Snapshot loaders + creation/refresh context loaders. No generic
`load_transition_snapshot`.

```python
# Context return types live beside their loaders.
@dataclass(frozen=True, slots=True)
class JobCreationContext:
    parent_exists: bool
    existing_replacement_job_id: JobName | None
    submitted_at_ms_watermark: int
    # any other fields ops.job.submit actually needs

@dataclass(frozen=True, slots=True)
class WorkerRefreshContext:
    existing_worker_row: WorkerRow | None
    existing_attributes: dict[str, AttributeValue]
    active_task_ids: list[JobName]


def load_full_snapshot(cur, *, now) -> TransitionSnapshot:
    """No seed. Every active row. Used by the bulk reconcile tick."""

def load_workers_slice(
    cur, worker_ids, *,
    now,
    observation_uids: Iterable[AttemptUid] = (),
) -> TransitionSnapshot:
    """Seed=worker_ids (plus tasks resolved from observation_uids).
    Closure: tasks WHERE current_worker_id IN ids OR (task_id, attempt_id)
    matches an observation; jobs of those tasks + ALL their tasks;
    coscheduled peer jobs + peer tasks; cascade-children; latest
    task_attempts; job_config; ATTEMPTS resolved from observation_uids
    (which may be non-current attempts)."""

def load_jobs_slice(cur, job_ids, *, now) -> TransitionSnapshot:
    """Seed=job_ids. Closure: full descendant subtree; ALL tasks of every
    job in subtree; coscheduled peer jobs + peer tasks; latest
    task_attempts; job_config."""

def load_tasks_slice(
    cur, task_ids, *,
    now,
    extra_attempt_keys: Iterable[tuple[JobName, int]] = (),
) -> TransitionSnapshot:
    """Seed=task_ids. Closure: jobs of those tasks + ALL their tasks;
    coscheduled peer jobs + peer tasks; cascade-children; latest
    task_attempts; job_config; plus any (task_id, attempt_id) in
    extra_attempt_keys (non-current attempts for stale-attempt
    validation in direct-provider updates)."""

def load_creation_context_for_job(
    cur, parent_id: JobName | None, replacing_name: str | None,
) -> JobCreationContext:
    """Tiny read used by ops.job.submit. Parent existence guard +
    same-name replacement row lookup + submitted_at watermark."""

def load_refresh_context_for_worker(
    cur, worker_id: WorkerId,
) -> WorkerRefreshContext:
    """Tiny read used by ops.worker.register_or_refresh."""
```

Codex flagged two real loader gaps (must-fix #2 and #4):

1. **Non-current attempts.** Direct-provider stale-attempt validation
   currently passes `(task_id, attempt_id)` after the snapshot loads
   (`ops/task.py:237`). v4 rev 2 folds those keys into the slice via
   `extra_attempt_keys`.
2. **Observation UID resolution.** Reconcile observation application
   currently resolves UIDs outside the snapshot (`ops/task.py:172`).
   v4 rev 2 folds them into `load_workers_slice` via `observation_uids`;
   the loader does the UID→(task,attempt) resolution and pulls those
   attempts into the snapshot.

Closure contract per relation:
- **descendant subtree**: transitive
- **coscheduled peers**: one hop only
- **cascade-children**: transitive descendant subtree of any seeded job
- **latest task_attempts**: per task (current_attempt_id only)
- **extra_attempt_keys**: pulled verbatim as requested (non-current)
- **observation_uids**: resolved to (task_id, attempt_id), then pulled
- **job_config**: per job in the slice
- **job_state_basis**: per job in the slice (NOT just root)
- **all_tasks_by_job**: per job in the slice (NOT just root)

The previous design specified this but the implementation didn't actually
honor it. v4 adds a `tests/.../test_loader_closure.py` that fixtures a
small DB with a known graph and asserts every kernel-touched entity has
its closure peers present.

Lower-level builders (CTE statement, descendant walker, etc.) stay private:
`_build_descendants_stmt`, `_load_descendants_multi`,
`_bulk_load_job_state_basis`, `_load_all_tasks_for_jobs`, `_load_job_num_tasks`.

~400 LOC.

### `reconcile/peers.py`

Cross-aggregate rules for coscheduled task peers. PUBLIC primitives.

```python
def find_coscheduled_siblings(state, job_id, exclude_task_id, has_cosched)
    -> list[ActiveTaskRow]

def terminate_coscheduled_siblings(state, task_id, reason, now_ms) -> None
    # calls task.mark_task_terminating(state, sibling_id, ...)

def requeue_coscheduled_siblings(state, task_id, now_ms) -> None
    # calls task.mark_task_terminating(state, sibling_id, ...) with PENDING
```

Import direction: `peers.py` imports `working_state.py`, `snapshot.py`,
`task.py` (for `mark_task_terminating`). NEVER imports job/worker/batches.
Single-direction: peers → task. ~100 LOC.

### `reconcile/job.py`

Job-aggregate primitives + mutations. **No batches here** — batches live
in `batches.py`.

```python
# Mutations owned here.
@dataclass(frozen=True, slots=True)
class JobStateMutation:
    job_id: JobName
    state: int
    set_started_at: Timestamp | None = None
    set_finished_at: Timestamp | None = None
    error: str | None = None
    def apply(self, cur: Tx) -> None: ...
    def overlay_apply(self, overlay: WorkingStateOverlay) -> None: ...

@dataclass(frozen=True, slots=True)
class CascadeKillJobMutation:
    job_id: JobName
    error: str
    finished_at: Timestamp
    allow_overwrite_worker_failed: bool = False
    def apply(self, cur: Tx) -> None: ...
    def overlay_apply(self, overlay: WorkingStateOverlay) -> None: ...

# PUBLIC primitives — operate on WorkingState. Called from batches.py.
def recompute_state(state, job_id, now_ms) -> int | None
def kill_non_terminal_tasks(state, job_id, reason, now_ms)
def cascade_children(state, parent_job_id, reason, now_ms)
def finalize_terminal(state, job_id, terminal_state, now_ms)
```

Import direction: `job.py` imports `working_state.py`, `snapshot.py`,
`policy.py`, `effects.py`. NEVER imports task/worker/peers/batches. ~250 LOC.

### `reconcile/task.py`

Task-aggregate primitives + mutations. **No batches here** — batches live
in `batches.py`.

```python
# Mutations owned here. Implement Mutation Protocol + OverlayUpdater.
class TaskMutation: ...
class AttemptMutation: ...

# Inputs
class TaskUpdate: ...
class TerminalDecision: ...
class TerminalKind: ...
class HeartbeatApplyRequest: ...

# PUBLIC primitives — called from batches.py and peers.py.
def mark_task_terminating(state, task_id, attempt_id, task_state, error, now_ms, *, ...)
def finalize_attempt(state, task_id, attempt_id, task_state, error, now_ms, *, ...)
def apply_one_transition(state, worker_id, update, now_ms)
    # The 230-LOC core, scoped to ONE update. batches.py drives the loop.
def active_row_from_snapshot(snap, task_id) -> ActiveTaskRow | None
def task_is_finished(task) -> bool
def preempt_one(state, task_id, ...)
def unschedulable_one(state, task_id, ...)
def timeout_helper(state, decisions, now_ms)
```

`has_reservation_flag` MOVES to `ops/job.py`. The `_apply_transitions`
dispatcher splits: the per-update transition logic stays here as
`apply_one_transition`; the loop and cross-aggregate fan-out
(peer cascade, job recompute) move to `batches.py` where the orchestration
belongs.

Import direction: `task.py` imports `working_state.py`, `snapshot.py`,
`policy.py`, `effects.py`. NEVER imports job/worker/peers/batches. ~500 LOC.

### `reconcile/worker.py`

Worker-aggregate primitives. **No batches here** — batches live in
`batches.py`.

```python
# Inputs/outputs
class ReconcileRow: ...
class ReconcileInputs: ...
class WorkerReconcilePlan: ...
class ReconcileResult: ...

# PUBLIC API
def plan_all(snapshot, healthy_worker_ids, now_ms) -> list[WorkerReconcilePlan]
def plan_one(snapshot, worker_id, now_ms) -> WorkerReconcilePlan

# PUBLIC primitives — called from batches.py.
def resolve_task_failure_state(prior_state, preemption_count, max_pre, terminal)
    -> tuple[int, int]
def filter_observations_to_plan(plan, observations, worker_id, state)
    -> list[AttemptObservation]
def observations_to_updates(state, plan, observations) -> list[TaskUpdate]
def assigned_updates_from_plan(snapshot, candidates, error) -> list[TaskUpdate]
def apply_worker_failure_one(state, worker_id, reason, now_ms)
```

`WorkerAttributeParams` MOVES to `ops/worker.py` (it's scheduler-facing
data, not a kernel concern).

Import direction: `worker.py` imports `working_state.py`, `snapshot.py`,
`policy.py`, `effects.py`. NEVER imports task/job/peers/batches. ~350 LOC.

### `reconcile/batches.py`

The thin orchestration layer. Each batch composes PUBLIC primitives from
task/job/worker/peers. **No private imports** — that was Codex's main
objection to v3's `sweep.py`. No private helpers from other modules; only
the functions documented above.

```python
def _apply_task_update_with_cascades(state, worker_id, update, now_ms):
    """Shared inner loop: heartbeats, direct-provider, and reconcile plan
    results all produce a TaskUpdate; this is the kernel that applies one.

    Heartbeats and direct-provider updates produce `TaskUpdate` directly via
    their own adapters (NOT through worker reconcile-plan semantics).
    Reconcile plan results go through `worker.observations_to_updates`,
    which translates plan observations into the same `TaskUpdate` shape.
    """
    task.apply_one_transition(state, worker_id, update, now_ms)
    if task.task_state(state, update.task_id) in TERMINAL:
        peers.terminate_coscheduled_siblings(state, update.task_id, ...)
    job.recompute_state(state, job_of(update.task_id), now_ms)
    if job_state(job) in TERMINAL:
        job.finalize_terminal(state, job, ...)
        job.cascade_children(state, job, ...)

def apply_heartbeats_batch(snap, requests, now) -> ControllerEffects
    # heartbeat → TaskUpdate directly, then _apply_task_update_with_cascades
def apply_direct_provider_updates_batch(snap, updates, now) -> ControllerEffects
    # provider update → TaskUpdate directly, then _apply_task_update_with_cascades
def apply_reconcile_batch(snap, plan_results, now) -> ControllerEffects
    # plan result → worker.observations_to_updates → TaskUpdate, then kernel
def apply_terminal_decisions_batch(snap, decisions, now) -> ControllerEffects
def apply_worker_failures_batch(snap, worker_ids, reason, now) -> ControllerEffects
def apply_cancel_job_batch(snap, job_id, reason, now) -> ControllerEffects
```

Per Codex N1: do NOT force heartbeats or direct-provider updates through
worker reconcile-plan concepts. Each adapter produces a neutral
`TaskUpdate`; the shared kernel is unified at the `TaskUpdate` level, not
at the plan-observation level.

Import direction: `batches.py` imports `task`, `job`, `worker`, `peers`,
`working_state`, `snapshot`. ALL imports are PUBLIC functions (no
underscore) per Codex finding 8. ~400 LOC.

Why this works where v3's `sweep.py` didn't:
- Aggregate files contain primitives. They have no batch logic and no
  peer-cascade fan-out — those are orchestration concerns.
- `batches.py` contains orchestration. Each batch is a clear sequence of
  primitive calls. A new reader can read the batch and see the rule
  composition.
- Private helpers are private to their own file. They never get imported
  cross-file. The aggregate's PUBLIC surface is what cross-file code
  consumes.

### `ops/job.py`

```python
def submit(db, request, *, run_template_cache, audit) -> JobName
    # → loader.load_creation_context_for_job
    # → writes.insert_job + writes.insert_job_config + writes.bulk_insert_tasks
def _submit_reservation_holder(...) -> JobName  # private; the embedded
                                                # path Codex flagged
def cancel(db, job_id, reason, *, endpoints, health, audit) -> None
    # → loader.load_jobs_slice([job_id])
    # → batches.apply_cancel_job_batch(snap, job_id, reason, now)
    # → effects.apply_effects(cur, eff, ...)
def remove_finished(db, job_id) -> bool
def has_reservation_flag(request) -> int     # moved from reconcile/task.py
```

cancel passes no `subtree` arg — the loader closes the snapshot.

~400 LOC.

### `ops/worker.py`

```python
class WorkerAttributeParams: ...     # moved from reconcile/worker.py
class WorkerFailureBatchResult: ...

def register_or_refresh(db, worker_id, ..., *, health, worker_attrs, ...) -> None
    # → loader.load_refresh_context_for_worker
    # → writes.upsert_worker + attribute writes
def fail(db, worker_ids, reason, *, health, endpoints, ...) -> WorkerFailureBatchResult
    # → loader.load_workers_slice(worker_ids)
    # → batches.apply_worker_failures_batch(snap, worker_ids, reason, now)
    # → effects.apply_effects(cur, eff, ...)
    # → writes.remove_worker
def apply_reconcile_observations(db, plan_results, *, health, endpoints, audit) -> None
    # MOVED from ops/task.py. The kernel sweep (batches.apply_reconcile_batch)
    # and the types (ReconcileRow/Plan/Result) both live in reconcile/worker;
    # the RPC adapter belongs here too. Codex must-fix #5.
    # → loader.load_workers_slice(worker_ids,
    #       observation_uids=uids_from_plan_results)
    # → batches.apply_reconcile_batch(snap, plan_results, now)
    # → effects.apply_effects(cur, eff, ...)
```

~300 LOC.

### `ops/task.py`

Task-RPC adapters only.

```python
class Assignment: ...

def queue_assignments(db, assignments, *, health, audit) -> None
def apply_heartbeats(db, requests, *, health, endpoints, audit) -> None
    # → loader.load_tasks_slice(task_ids)
    # → batches.apply_heartbeats_batch
def apply_terminal_decisions(db, decisions, *, health, endpoints, audit) -> None
    # → loader.load_tasks_slice(task_ids)
    # → batches.apply_terminal_decisions_batch
def apply_provider_updates(db, updates, *, health, endpoints, audit) -> None
    # → loader.load_tasks_slice(task_ids,
    #       extra_attempt_keys=[(t, aid) for t, aid in updates])
    # → batches.apply_observations_batch
```

`apply_reconcile_observations` is NOT here — moved to `ops/worker.py`.
~250 LOC.

---

## Import direction (the contract)

Strictly downward, no cycles, no shims. Per Codex must-fix #1: no module
above the leaves imports from `working_state` while `working_state`
imports back.

```
Leaves:
  policy.py        — constants
  snapshot.py      — dataclasses
  effects.py       — Mutation Protocol + ControllerEffects + apply_effects +
                     cross-aggregate effect classes + OverlayUpdater Protocol +
                     WorkingStateOverlay dataclass

Level 1 (depends on leaves only):
  working_state.py — imports snapshot + effects ONLY. The record() method
                     uses the Mutation Protocol; it has no type-level
                     dependency on any concrete mutation class.

Level 2 — aggregate primitives. EACH depends on:
  task.py    — snapshot, policy, effects, working_state
  job.py     — snapshot, policy, effects, working_state
  worker.py  — snapshot, policy, effects, working_state
  No cross-aggregate imports between task/job/worker.

Level 3:
  peers.py   — snapshot, working_state, task. Single edge: peers → task
               (peers calls task.mark_task_terminating). task does NOT
               import peers. No cycle.

Level 4 — orchestration:
  batches.py — task, job, worker, peers (PUBLIC functions only).
               No imports back into batches from any module.

Level 5 — I/O:
  loader.py  — snapshot, policy, schema, db, reads. NO dependency on any
               aggregate rule file. The loader is pure I/O; it doesn't
               know about transitions.

Level 6 — RPC adapters:
  ops/*.py   — loader, batches, effects, writes, reads. ops files DO NOT
               import aggregate rule files directly; they call batches.X.
               Exception: ops/job.py and ops/worker.py import their own
               aggregate ONLY for mutation type annotations on local
               helpers (no cycle since aggregate files don't import ops).
```

The cycle that broke v4 rev 1 was: working_state → aggregate (for
mutation class imports) AND aggregate → working_state (for the
`state: WorkingState` annotation). Rev 2 closes this with the
`OverlayUpdater` Protocol: WorkingState invokes `m.overlay_apply(overlay)`
through the Protocol without importing any concrete mutation type.

Aggregate files freely import `working_state` because working_state does
not import them. One-way edge.

---

## Closure contract enforcement

New test: `lib/iris/tests/cluster/controller/test_loader_closure.py`.

For each scoped loader, fixtures a small in-memory DB with a known graph:
two coscheduled jobs each with two tasks, plus a child job, plus a
worker holding some tasks. Calls the loader. Asserts:

- Every task referenced by a job in the slice is in `snapshot.tasks`.
- Every job in the slice has its `JobConfigRow` in `job_configs`.
- Every job in the slice has its `JobStateBasis` in `job_state_basis`
  (not just root jobs).
- Every job in the slice has its `all_tasks_by_job` entry (not just root).
- Coscheduled peer jobs of any seeded task are in the slice.
- Cascade-children (full descendant subtree of any seeded job) are in
  the slice.
- Latest task_attempts (current_attempt_id only) are in `attempts` for
  every task in the slice.

This test is the primary defense against Codex finding 2.

---

## Mutation Protocol enforcement

`mutations: list[Mutation]` plus per-aggregate-owned classes means we lose
the typed-list per-mutation-kind sanity check. To compensate:

- Each mutation class implements `Mutation` Protocol (`apply(self, cur)`).
- `apply_effects` calls `m.apply(cur)` on each. No `isinstance`.
- A purity guard test asserts that `effects.py` does not name any
  concrete mutation class — only the Protocol.
- An overlay-correctness test asserts that every mutation class which
  changes task or job state implements `OverlayUpdater.overlay_apply`.
  Without this, `WorkingState.record` would queue the SQL mutation but
  later reads of the overlay would silently return stale state — SQL
  ends up correct, but subsequent in-kernel decisions are wrong.
  Concretely: iterate all dataclasses in `task.py` / `job.py` whose
  `apply(cur)` writes to `tasks` or `jobs`, and assert each has
  `overlay_apply`.

---

## Migration plan

The landed code is in a working state with 960 tests passing. v4 lands
as a sequence of focused commits, each green:

**V4-1 — Mutation Protocol + effects.py consolidation** (1 sonnet, ~200 LOC)
- Add `apply(cur)` method to each existing mutation dataclass.
- Define `Mutation` Protocol and `OverlayUpdater` Protocol in `effects.py`.
- Move cross-aggregate effect dataclasses (`EndpointDeletion`,
  `WorkerHealthEffect`, `LoggerEvent`, `LogEvent`) from `mutations.py`
  into `effects.py`.
- Move `ControllerEffects` into `effects.py`. Collapse its per-kind lists
  into a single flat `mutations: list[Mutation]`.
- Move `TaskMutation`, `AttemptMutation` from `mutations.py` into `task.py`.
- Move `JobStateMutation`, `CascadeKillJobMutation` into `job.py`.
- Implement `overlay_apply(overlay)` on each mutation that needs to
  influence subsequent reads (per Mutation Protocol enforcement section).
- Delete `mutations.py`.
- Tests green.

**V4-2 — Narrow WorkingState + split state.py** (1 sonnet, ~200 LOC)
- Split `state.py` into:
  - `snapshot.py` (TransitionSnapshot + dataclasses)
  - `policy.py` (constants)
  - `working_state.py` (the narrow record-bag class)
- `working_state.py` imports ONLY `snapshot` and `effects`. It exposes:
  - `record(mutation: Mutation)` — appends to effects, calls
    `mutation.overlay_apply(overlay)` if the mutation implements
    `OverlayUpdater`.
  - read methods that consult `overlay` then `snapshot`.
  - Named cross-aggregate effect emitters that are NOT per-aggregate row
    writes (e.g. `record_endpoint_deletion`, `record_worker_health`,
    `log`).
  - It does NOT have `mark_task_terminating` / `finalize_attempt` /
    `finalize_terminal_job`. Those are functions in the aggregate files.
- Move `_mark_task_terminating` from `state.py` into `task.py` as the
  public `mark_task_terminating(state, …)`.
- Update peers.py to call `task.mark_task_terminating(state, …)` —
  peers → task is the single allowed cross-aggregate edge.
- Delete `state.py`.
- Tests green.

**V4-3 — Aggregate primitives + batches.py orchestrator** (1 sonnet, ~150 LOC)
- Keep aggregate files (`task.py`, `job.py`, `worker.py`, `peers.py`)
  PUBLIC-primitive only. No batch entries live in aggregate files.
- Create `batches.py` as a thin orchestrator. It imports only PUBLIC
  primitives from `task`, `job`, `worker`, `peers`. It owns:
  - `apply_reconcile_batch`
  - `apply_worker_failures_batch`
  - `apply_heartbeats_batch`
  - `apply_terminal_decisions_batch`
  - `apply_direct_provider_updates_batch`
  - `apply_cancel_job_batch`
- Route `apply_direct_provider_updates_batch` through the same
  `_apply_transitions` core path as heartbeats (Codex finding 3).
- Delete `sweep.py`.
- Tests green.

**V4-4 — Scoped loaders + closed snapshot** (1 opus, ~300 LOC)
- Add to `loader.py`:
  - `load_full_snapshot(...)`
  - `load_workers_slice(..., observation_uids=())` — observation_uids
    resolve to (task_id, attempt_id) for non-current-attempt closure
  - `load_jobs_slice(...)` — closure already covers descendants + peers
  - `load_tasks_slice(..., extra_attempt_keys=())` — extra_attempt_keys
    pulled verbatim for direct-provider stale-attempt validation
  - `load_creation_context_for_job(...)` returning `JobCreationContext`
  - `load_refresh_context_for_worker(...)` returning `WorkerRefreshContext`
- Move `JobCreationContext` and `WorkerRefreshContext` dataclasses into
  `loader.py` (NOT placeholders in `snapshot.py`).
- Fix closure: `job_state_basis` and `all_tasks_by_job` cover every job
  in slice. Coscheduled peer + descendant expansion happens in the
  loader, parameterized by `extra_attempt_keys` / `observation_uids` so
  callers with non-current-attempt observations get the right rows.
- Update `ops/*` to call the scoped loaders. Drop `subtree` side inputs.
- Remove generic `load_transition_snapshot`.
- Add `test_loader_closure.py` asserting every relation referenced by
  any aggregate is closed under the scoped loaders.
- Tests green.

**V4-5 — Cleanup ops boundaries** (1 sonnet, ~50 LOC)
- Move `WorkerAttributeParams` from `reconcile/worker.py` to `ops/worker.py`.
- Move `has_reservation_flag` from `reconcile/task.py` to `ops/job.py`.
- Move `apply_reconcile_observations` orchestration from `ops/task.py`
  to `ops/worker.py` (Codex finding 5 — reconcile observations are a
  worker-side concept).
- Delete `ops/reservation.py` placeholder.
- Ensure ops files import `batches` rather than aggregate rule files
  for batch entry points.
- Tests green.

Total: ~900 LOC churn across 5 commits. Each is independently green.
After V4-5, every Codex finding plus the 5 revalidation must-fixes are
closed.

---

## What this revision deliberately does NOT change

- Pure kernel + thin I/O shell. The shape is right; v4 only fixes leaks.
- `peers.py` stays its own file because the cross-aggregate rule
  is genuinely shared — `task.py` and `worker.py` both call into it.
- `ops/*` aggregate split stays. `reservation.py` was the only fake one.
- `direct_provider.py` stays (already renamed from `dispatch.py` in Stage A).
- `reads.py` / `writes.py` stay flat. Splitting is a separate refactor.
- `controller.py` stays the wiring layer. Codex didn't review it; the
  ~1200 LOC overage vs design doc is a known separate cleanup.

---

## Demonstration: a new reader's path

A reader investigating "what happens when a heartbeat lands?":

1. `ops/task.py:apply_heartbeats` — clearly the RPC adapter.
2. It loads a snapshot with `loader.load_tasks_slice(...)`. The loader
   file owns the closure contract; no surprises.
3. It calls `batches.apply_heartbeats_batch(snap, requests, now)`.
4. `batches.py` is the orchestrator. `apply_heartbeats_batch` composes
   `task.apply_one_transition` per request over a `WorkingState`.
5. `task._apply_transitions` writes via `task.mark_task_terminating(state, …)`
   (which calls `state.record(TaskMutation(...))` and
   `state.record(AttemptMutation(...))`). `WorkingState.record` runs
   `mutation.overlay_apply(overlay)` so the next read sees the update.
6. For coscheduled cascade: `task._apply_transitions` calls into
   `peers.terminate_coscheduled_siblings`, which calls back into
   `task.mark_task_terminating` (single allowed cross-aggregate edge).
7. Returns `ControllerEffects`. Back in ops, `apply_effects(cur, eff, ...)`
   persists.

No private helper imports across files. No re-export sprawl. Each module's
name describes its contents truthfully.

A reader investigating "cancel a job":

1. `ops/job.py:cancel`.
2. `loader.load_jobs_slice([job_id])` — closure includes subtree + peers.
3. `batches.apply_cancel_job_batch(snap, job_id, reason, now)` —
   orchestrator function. Composes `job.cascade_kill_descendants` (which
   emits `CascadeKillJobMutation` from `job.py`).
4. Effects → `apply_effects`. Done.

A reader investigating "what's `WorkerHealthEffect`?":

1. `effects.py`. It's a cross-aggregate post-commit category.
2. `apply_effects` schedules it for the post-commit hook.
3. Producers: `WorkingState.record_worker_heartbeat(...)` etc, called
   from the worker aggregate's batch entries.

---

## Out of scope for v4

- Reservation ops extraction (still inline in controller.py).
- `controller.py` slim past current state.
- `reads.py`/`writes.py` split by aggregate.
- A long-lived informer-style cache.
- ORM. Stays Core, stays Tx-based.
