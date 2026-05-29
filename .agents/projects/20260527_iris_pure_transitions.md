# Pure-Functional Reconcile & Transitions

**Goal**: Refactor `lib/iris/src/iris/cluster/controller/transitions.py` so that every
function in it is a pure transformation from an in-memory snapshot to an
explicit set of mutations. The controller assembles the snapshot once per
operation (e.g. one reconcile tick), calls the pure transition functions,
and then bulk-applies the returned mutation set in a single write
transaction. **Acceptance criterion: `transitions.py` contains zero
references to `Tx`, `cur`, `db`, `reads.*`, `writes.*`, or SQLAlchemy
statements.**

## Why

Today the reconcile path mixes decision logic with per-row DB I/O. Each
`apply_reconcile_result` call:

1. Re-reads the task and attempt rows it needs (`reads.bulk_get_task_detail`,
   `reads.bulk_get_attempts`, `reads.filter_existing_workers`,
   `reads.resolve_attempt_uids`, `reads.get_job_config`).
2. Runs the state machine in `_apply_task_transitions`, which interleaves
   reads of `job_config` and ad-hoc helpers (`_find_coscheduled_siblings`,
   `_resolve_preemption_policy`, `_recompute_job_state`,
   `_finalize_terminal_job`, `_cascade_children`) with `sa_update`/`writes.*`
   calls.
3. Writes attempt + task + endpoint rows inline, then recomputes the job
   state with another query.

Per reconcile tick the controller fans this out over every healthy worker;
each per-worker call opens a fresh transaction (`with self._db.transaction()`),
re-reads largely the same `job_config` / `tasks` rows, and contends with the
single writer `RLock`. This dominates the reconcile loop wall time and
contributes to the dashboard-RPC latency budget identified in
`.agents/projects/iris-perf/design.md`.

If reconcile + transitions are pure:

- The controller takes **one** read snapshot per tick covering everything the
  state machine needs across all workers (tasks, attempts, job_configs, the
  reservation/coscheduling lookup tables).
- The pure transitions decide what to do and emit a `MutationBatch`.
- The controller commits the batch in **one** write transaction at the end of
  the tick.

Beyond performance, this gives us a much smaller surface to test: state
machine behaviour becomes a function from dataclass-in to dataclass-out and
can be exercised without spinning up SQLite. It also realises the discipline
captured in the project memory: *core code uses agnostic Protocols /
dataclasses; protos and DB rows are constructed at the boundary, not inside
the core.*

## Current state

### `reconcile.py` — already pure

`reconcile.py` is the model we want everywhere. It defines `ReconcileInputs`
(a frozen dataclass holding `job_specs`, `worker_ids`, `rows_by_worker`) and
`reconcile_workers(inputs) -> list[WorkerReconcilePlan]`. No DB, no `Tx`, no
SQLAlchemy. The controller builds `ReconcileInputs` inside one
`read_snapshot()` (`controller.py:2323`) and passes it through.

### `transitions.py` — mixed

`transitions.py` (3261 lines) is the inverse: every public method except the
heartbeat translation helpers takes a `Tx` (or `self._db`) and freely mixes
reads and writes. Categorising the methods by their role on the reconcile
hot path:

**Hot-path (called every reconcile tick):**

- `apply_reconcile_result(cur, plan, result, now)` — applies one worker's
  reconcile outcome. Internally reads `bulk_get_task_detail`,
  `bulk_get_attempts`, `filter_existing_workers`, `resolve_attempt_uids`,
  `get_job_config`, then calls `_apply_task_transitions`.
- `_apply_task_transitions(cur, req, now_ms, task_map, attempt_map)` — the
  per-update state machine. Decides retry vs. terminal, then writes the
  attempt and task rows via `writes.apply_attempt_update`/
  `writes.apply_task_state_update`, deletes endpoints, finds coscheduled
  siblings (another read), bounces or terminates them, and finally calls
  `_recompute_job_state` + `_finalize_terminal_job`.
- `_recompute_job_state(cur, job_id)` — reads task-state counts for a job
  and writes a new `jobs.state`.
- `_finalize_terminal_job` / `_cascade_children` / `_kill_non_terminal_tasks`
  / `_finalize_attempt` / `_mark_task_producing_transition` — module-level
  helpers that take a `Tx` and emit `UPDATE`/`DELETE` statements directly.
- `_observations_to_updates(cur, observations)` — resolves
  `attempt_uid → (task_id, attempt_id)` via `reads.resolve_attempt_uids`.
- `_assigned_updates_from_plan(plan, error, cur)` — re-reads task detail to
  decide which ASSIGNED rows in the plan should be bounced.

**Hot-path adjacent (called once per tick, not per worker):**

- `apply_heartbeats_batch(cur, requests)` — same shape as
  `apply_reconcile_result` but batched across workers; reuses
  `_apply_task_transitions`.

**Command-API (not on the reconcile tick — invoked by RPC handlers and
auxiliary loops):**

- `submit_job`, `cancel_job`, `register_or_refresh_worker`, `register_worker`,
  `queue_assignments`, `preempt_task`, `cancel_tasks_for_timeout`,
  `mark_task_unschedulable`, `replace_reservation_claims`,
  `remove_finished_job`, `remove_worker`, `prune_old_data`,
  `update_worker_pings`, `get_running_tasks_for_poll`, `fail_workers`,
  `fail_workers_batch`, `load_workers_from_config`,
  `add_endpoint`/`remove_endpoint`, `drain_for_direct_provider`,
  `apply_direct_provider_updates`, `run_request_template`.

The acceptance criterion ("no DB references in `transitions.py`") covers
both groups. The reconcile hot path is the highest-value chunk and a clean
seam to land first.

### Why apply-vs-decide split is feasible

Every mutation already funnels through `writes.*` helpers (`writes.py` has a
small, enumerable surface: `insert_job`, `insert_job_config`,
`bulk_insert_tasks`, `assign_to_worker`, `apply_attempt_update`,
`apply_task_state_update`, `remove_worker`, `delete_job`,
`promote_to_direct_provider`, plus a handful of user helpers). Every read
funnels through `reads.*` returning typed dataclasses
(`TaskDetailRow`, `ActiveTaskRow`, `PendingDispatchRow`, …). That means we
can already round-trip the inputs through dataclasses without touching SQL;
the DB-binding lives outside the state machine. The work is mechanical, not
exploratory.

The two places where transitions interleave reads inside the inner loop
(`reads.get_job_config` in `_apply_task_transitions`, sibling and policy
look-ups in the cascade helpers) need their inputs pre-loaded into the
snapshot. They are bounded: at most one job_config per distinct
`task.job_id`, one sibling list per coscheduled job touched, and one
preemption-policy per job recomputed. The controller already knows the set
of task updates before calling the state machine, so it can preload these.

## Target shape

### Inputs: `TransitionSnapshot`

A single frozen dataclass passed into every pure transition function. For
the reconcile tick, the controller populates it from one `read_snapshot()`
covering all healthy workers. Sketch:

```python
@dataclass(frozen=True, slots=True)
class TransitionSnapshot:
    now: Timestamp
    tasks: dict[JobName, TaskDetailRow]                 # bulk_get_task_detail
    attempts: dict[tuple[JobName, int], AttemptRow]     # bulk_get_attempts
    job_configs: dict[JobName, JobConfigRow]            # get_job_config x N
    active_workers: set[WorkerId]                       # filter_existing_workers
    attempt_uid_index: dict[AttemptUid, tuple[JobName, int]]  # resolve_attempt_uids
    active_siblings_by_job: dict[JobName, list[ActiveTaskRow]]  # coscheduled lookup
    job_task_states: dict[JobName, dict[int, int]]      # state -> count, for _recompute_job_state
    # ... only what the state machine needs; not the whole DB.
```

The exact shape is finalised in Task 1 by tracing every read currently
performed by `_apply_task_transitions` and friends. The key design
constraint is: **the snapshot is built outside `transitions.py`, lives only
in memory, and never contains a `Tx`.**

### Outputs: `MutationBatch`

A frozen dataclass enumerating every write the controller will apply.
Sketch:

```python
@dataclass(frozen=True, slots=True)
class AttemptMutation:
    task_id: JobName
    attempt_id: int
    state: int
    started_at_ms: int | None
    finished_at_ms: int | None
    exit_code: int | None
    error: str | None

@dataclass(frozen=True, slots=True)
class TaskMutation:
    task_id: JobName
    state: int
    error: str | None
    exit_code: int | None
    started_at_ms: int | None
    finished_at_ms: int | None
    failure_count: int | None
    preemption_count: int | None
    clear_worker: bool = False

@dataclass(frozen=True, slots=True)
class JobMutation:
    job_id: JobName
    state: int | None = None
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    error: str | None = None

@dataclass(frozen=True, slots=True)
class EndpointDeletion:
    task_id: JobName

@dataclass(frozen=True, slots=True)
class WorkerHealthEffect:
    """In-memory health updates. Applied in the same commit hook so a
    rolled-back batch doesn't leave the tracker ahead of the DB."""
    heartbeat: list[WorkerId] = ()
    build_failed: list[WorkerId] = ()

@dataclass(frozen=True, slots=True)
class MutationBatch:
    attempts: list[AttemptMutation]
    tasks: list[TaskMutation]
    jobs: list[JobMutation]
    endpoint_deletions: list[EndpointDeletion]
    health: WorkerHealthEffect
    tx_result: TxResult                                 # tasks_to_kill / task_kill_workers
    log_events: list[LogEvent]                          # event=… entity=… kv=…
```

Two design notes:

- **Order matters for some writes** (attempts before tasks, endpoint
  deletions after the task is moved to a non-active state, job updates after
  task counts settle). The `apply_mutations` function below codifies that
  order in one place. The state machine emits dataclasses without caring.
- **In-memory health updates ride along.** `WorkerHealthTracker` is mutated
  today inside transitions; we keep its lifecycle tied to the same commit so
  the in-memory tracker never disagrees with the DB.

### Apply layer: `mutations.py`

A new module — pure I/O, no decisions. Single entry point:

```python
def apply_mutations(
    cur: Tx,
    batch: MutationBatch,
    *,
    health: WorkerHealthTracker,
    endpoints: EndpointsProjection,
) -> TxResult:
    ...
```

This is the **only** place that knows how to translate a `MutationBatch` to
SQL. It calls the existing `writes.apply_attempt_update`,
`writes.apply_task_state_update`, etc., in the correct order. It also fires
the audit `log_event(...)` lines.

### Transitions: pure module

After the refactor, `transitions.py` has one allowed import surface:

- Standard library
- `iris.cluster.types`, `iris.cluster.constraints`, `iris.cluster.controller.task_state`
- `iris.cluster.controller.reads` *for its dataclasses only* (`TaskDetailRow`,
  `ActiveTaskRow`, `PendingDispatchRow`, …). The `reads` module's *functions*
  are forbidden. A small follow-up may move the dataclasses into a
  `controller/rows.py` so the type-only dependency is honest.
- `iris.cluster.controller.reconcile` (already pure)
- `iris.cluster.controller.mutations` *for the output dataclass types only*
- `iris.rpc.*_pb2`, `rigging.timing`

Disallowed: `iris.cluster.controller.db`, `iris.cluster.controller.writes`,
`iris.cluster.controller.schema`, `sqlalchemy`, `sqlite3`.

This is enforceable as a one-line `grep` test in CI.

## Migration plan

The refactor is staged so each step keeps the cluster bootable, the test
suite green, and the controller correct. The order is:

1. **Carve the apply layer in place.** Move the `writes.*` / `delete()` /
   `sa_update()` calls inside `_apply_task_transitions` and its helpers
   behind small "emit-then-apply" wrappers that still execute inline. No
   behaviour change; this is mechanical refactoring that makes the next step
   straightforward.

2. **Introduce `MutationBatch` and `apply_mutations`.** Build the dataclass
   types, write `mutations.apply_mutations`, and convert the wrappers from
   Step 1 to emit dataclasses into a batch instead of executing. At the
   bottom of each public method that mutates, call
   `apply_mutations(cur, batch, …)`. Still no controller-level change.

3. **Introduce `TransitionSnapshot` for the reconcile hot path.** Define the
   dataclass; populate it inside the controller's
   `_reconcile_worker_batch`, replacing per-worker `read_snapshot()` /
   `bulk_get_*` calls. Change `apply_reconcile_result` /
   `apply_heartbeats_batch` / `_apply_task_transitions` to take the snapshot
   instead of a `Tx` and `task_map`/`attempt_map`. The methods no longer
   read; they consume.

4. **Lift the apply call out of transitions.** With Step 2 and Step 3 in
   place, the public hot-path methods can return `MutationBatch` instead of
   calling `apply_mutations` themselves. The controller does the single
   end-of-tick `with self._db.transaction(): apply_mutations(cur, batch, …)`.

5. **Repeat Steps 1–4 for the command-API methods.** They are decoupled, so
   each method can migrate independently: `cancel_job`, `preempt_task`,
   `cancel_tasks_for_timeout`, `submit_job`, `queue_assignments`,
   `register_or_refresh_worker`, `mark_task_unschedulable`,
   `apply_direct_provider_updates`, `drain_for_direct_provider`, …

6. **Delete the last `Tx` reference.** Move `replace_reservation_claims`,
   `remove_finished_job`, `prune_old_data`, `remove_worker`,
   `fail_workers*`, `load_workers_from_config`, `add_endpoint`/
   `remove_endpoint`, `record_task_status_text` (in-memory only — trivial)
   to the same shape. Status text and endpoint helpers may end up moving out
   of `transitions.py` entirely; they are not state-machine code. Once the
   file is purely decision logic plus dataclass returns, add a CI grep:

   ```bash
   ! grep -nE 'sqlalchemy|sqlite3|self\._db|: Tx\b|writes\.|reads\.[a-z_]+\(' \
         lib/iris/src/iris/cluster/controller/transitions.py
   ```

Each step ships as one PR with the existing controller test suite
(`lib/iris/tests/cluster/controller/`) passing. Steps 3 and 4 are the
performance-relevant ones: after Step 4 the reconcile tick performs one
write transaction instead of one-per-worker, and one read snapshot instead
of one-per-worker.

## Risks and trade-offs

1. **Snapshot completeness.** The pure transitions can't ask for more data
   mid-flight, so the controller has to over-fetch in the snapshot. For the
   reconcile tick we already know the worker set, the candidate task IDs
   come from the reconcile observations, and the coscheduled siblings can be
   pre-resolved from the same join. Worst case the snapshot reads ~2x the
   rows it needs; net win is still large because we're saving N transactions
   per tick. We will measure.

2. **`replace_reservation_claims` and other "full-table replace" writes.**
   These don't fit the row-level `MutationBatch` shape cleanly. They can
   ride as a separate `replace_reservation_claims: dict[WorkerId,
   ReservationClaim] | None` field on the batch; the apply layer handles the
   `DELETE … INSERT …` atomically. Same approach for any future
   bulk-replace.

3. **`prune_old_data` is incremental and self-paced.** It currently sleeps
   between row deletions. Pure-decision-then-apply doesn't fit the
   long-running loop shape. The prune loop stays a `while`-loop in the
   controller; each iteration is a tiny `MutationBatch` of one job/worker
   deletion. The decision (*which job to prune next*) becomes a pure
   function over a small snapshot (`oldest terminal job before cutoff`).

4. **In-memory projections (`EndpointsProjection`, `WorkerAttrsProjection`,
   `WorkerHealthTracker`).** These mutate alongside DB writes today, with
   `cur.register(...)` / `cur.on_commit(...)` hooks to keep them consistent.
   They stay on the apply side; the batch describes the intended in-memory
   delta and `apply_mutations` enqueues the same `on_commit` hooks. The
   pure transitions never touch them directly.

5. **Diff-based regressions.** The migration changes shape, not semantics.
   The state-machine tests in `lib/iris/tests/cluster/controller/` already
   cover transitions end-to-end against a real SQLite. We extend them with
   a fixture that calls the pure transition functions on a hand-built
   `TransitionSnapshot` and asserts the emitted `MutationBatch`, which gives
   us a fast, side-effect-free regression net for future state-machine
   changes.

6. **Doing too much at once.** The "no DB references in `transitions.py`"
   bar is the *final* state. Steps 1–4 already produce the reconcile-hot-path
   win and are mergeable independently. Step 5 is repeated work across
   methods that don't all need to land together. If we run out of cycles or
   discover a snag, the file remains partially-migrated but each migrated
   method is strictly cleaner than before.

## Out of scope

- Schema changes (covered in `.agents/projects/iris-perf/design.md`).
- Dashboard / RPC query rewrites (also covered separately).
- The autoscaler / polling thread restructuring proposed in
  `.agents/projects/controller_tick_snapshot/design.md`. That work is
  complementary — once `transitions.py` is pure, the TickDriver phase 4 of
  that design becomes a straightforward consumer of `MutationBatch`.

## Task breakdown

The detailed task list is tracked through the harness `TaskCreate` tool;
see `weaver note` entries for status. The high-level shape is:

| ID | Task | Depends on |
|----|------|------------|
| T1 | Inventory all DB reads/writes inside `_apply_task_transitions` and the cascade helpers; finalise `TransitionSnapshot` and `MutationBatch` dataclasses. | — |
| T2 | Create `mutations.py` with `MutationBatch`, `AttemptMutation`, `TaskMutation`, `JobMutation`, `EndpointDeletion`, `WorkerHealthEffect`, and `apply_mutations(cur, batch, …)`. Cover only the row types Step 3 needs. | T1 |
| T3 | Refactor `_apply_task_transitions` + `_finalize_attempt` + `_mark_task_producing_transition` + `_kill_non_terminal_tasks` + `_cascade_children` + `_terminate/_requeue_coscheduled_siblings` + `_finalize_terminal_job` + `_recompute_job_state` to emit `MutationBatch` rows instead of executing writes. Inline apply at the end of each public method (no controller change yet). | T2 |
| T4 | Build `TransitionSnapshot` in the controller (`_reconcile_worker_batch`); refactor `apply_reconcile_result` / `apply_heartbeats_batch` to consume the snapshot and return `MutationBatch`. The controller commits the batch in one transaction. | T3 |
| T5 | Add a state-machine test that feeds a hand-built `TransitionSnapshot` to the pure transitions and asserts the emitted `MutationBatch`. Use it to lock in semantics before further migrations. | T4 |
| T6 | Migrate command APIs that are *adjacent* to the hot path: `apply_direct_provider_updates`, `cancel_tasks_for_timeout`, `preempt_task`, `cancel_job`, `mark_task_unschedulable`, `queue_assignments`. Each is its own PR. | T2 |
| T7 | Migrate creation-side APIs: `submit_job`, `register_or_refresh_worker`, `register_worker`, `replace_reservation_claims`, `load_workers_from_config`. Introduce bulk-replace effect for reservation claims. | T2 |
| T8 | Migrate the long-running and incidental APIs: `prune_old_data` (incremental: per-iteration batch), `remove_finished_job`, `remove_worker`, `fail_workers`, `fail_workers_batch`, `drain_for_direct_provider`, `run_request_template`, `get_running_tasks_for_poll`. Move `add_endpoint`/`remove_endpoint`/`record_task_status_text` out of `transitions.py` if they no longer belong. | T6, T7 |
| T9 | Land the CI grep guard. Strip the now-unused imports (`from iris.cluster.controller.db import …`, `from iris.cluster.controller import writes, reads`, `sqlalchemy`). | T8 |
| T10 | Benchmark the reconcile-loop wall time before (HEAD~N) and after on a controller checkpoint with 200+ workers using the harness from `.agents/projects/iris-perf/design.md`. Document the result in this file's appendix. | T4 |

T1 → T4 lands the hot-path win and is the minimum to call the feature done.
T5 — T9 finishes the file-level acceptance criterion. T10 records the
performance result and closes the loop with the iris-perf project.

### Revision after starting implementation

After landing T2 (the `mutations.py` apply layer) I started T3 and walked
through `_apply_task_transitions` carefully. The original phrasing of T3
("emit `MutationBatch` rows instead of executing writes; inline apply at
the end of each public method, no controller change yet") *cannot
preserve semantics*. The state machine's cascade helpers read DB state
that earlier iterations of the inner loop have written:

- `_find_coscheduled_siblings(cur, job_id, exclude=update_task_id)`
  enumerates `tasks` rows in `ACTIVE_TASK_STATES` for the job. If a
  previous iteration moved a sibling to FAILED, the original code sees
  that sibling as no-longer-active; a deferred-apply version would still
  see it as active and re-cascade.
- `_recompute_job_state` does a `GROUP BY state` on `tasks` for the job.
  Deferred-apply means it sees the pre-batch counts and decides the job
  hasn't moved.
- `_finalize_terminal_job` → `_kill_non_terminal_tasks` enumerates
  active tasks under a job and its descendants. Same story.

So T3 has to merge with T4: the cascade helpers must be rewritten to
consume a *prospective* in-memory `WorkingState` (the snapshot **plus**
the pending mutations in the batch), not to query the DB. Once that's
done, the entire `_apply_task_transitions` call can produce a single
`MutationBatch` applied once.

#### Revised T3 plan

Rename T3 to **"WorkingState refactor of the state machine"** and split
into substages, each landed as its own PR with the existing test suite
passing:

| Substage | Description |
|---|---|
| T3a | Introduce `WorkingState` (in `transitions.py` or a new `working_state.py`) that wraps a `TransitionSnapshot` plus a mutable view of the prospective row state. Has methods like `task_state(task_id) -> int`, `mark_task_state(task_id, new_state)`, `active_tasks_in_job(job_id, exclude) -> list[ActiveTaskRow]`, `task_state_counts(job_id) -> dict[int, int]`. All reads route through `WorkingState`; writes go to the underlying `MutationBatch`. |
| T3b | Migrate `_recompute_job_state` to consume `WorkingState` for the count read and emit `JobStateUpdate` into the batch. Simplest cascade helper — clean seam. |
| T3c | Migrate `_find_coscheduled_siblings` to read sibling rows from `WorkingState.active_tasks_in_job(...)`. |
| T3d | Migrate `_finalize_terminal_job` and `_cascade_children` to consume `WorkingState` for descendant enumeration. Requires preloading the descendant subtree into the snapshot (already noted in Appendix A). |
| T3e | Migrate the `_apply_task_transitions` inner loop and `_finalize_attempt` / `_mark_task_producing_transition` to route writes through the batch. With T3a–T3d in place this is mechanical. |
| T3f | Lift `apply_mutations` out of `_apply_task_transitions`; the public methods (`apply_reconcile_result`, `apply_heartbeats_batch`) return a `MutationBatch` and the controller applies it. |

T4 (controller-side snapshot building + bulk apply across all workers) is
now a thin step on top of T3f: the controller builds one snapshot for
all healthy workers, runs the pure pipeline, gets one batch, applies
once. The "one bulk update for our whole reconcile workflow" stated goal
is reached at the end of T3f + T4.

#### Why this is the right shape

The `WorkingState` is the right abstraction because the state machine
*already* assumes a coherent view of the world that evolves as it
processes updates. Today that view is maintained by the DB (each write
makes the next read consistent); after T3, it's maintained explicitly in
memory. Once it's explicit:

- The state machine is easy to test: hand-build a `WorkingState`, run a
  transition, inspect the emitted `MutationBatch`. No SQLite. This is T5.
- The same abstraction generalises to other transitions: `cancel_job`,
  `preempt_task`, `cancel_tasks_for_timeout` all do "read-some-tasks,
  emit-some-writes, possibly-cascade" with the same kind of DB
  interleaving today. They migrate to `WorkingState` via T6.
- Pure-functional `transitions.py` becomes natural: every method takes a
  `WorkingState` (or a `TransitionSnapshot` it wraps internally) and
  returns a `MutationBatch`. No `Tx` parameter survives.

Current status: T2 is landed (mutations.py infrastructure). T3 is the
next step but has been re-scoped per the substages above. T4 collapses
into "build the snapshot for all workers and call apply_mutations once
on the controller side" — small step on top of T3f.

## Tightened public-interface direction (from usage map)

A sub-agent mapped every external caller of `ControllerTransitions` and
its module-level exports. Findings (verified file:line in the agent
report; do not re-litigate without re-running):

### What every external caller already looks like

The hot path is already single-caller per method: `apply_reconcile_result`
is called once (`controller.py:2406`), `apply_heartbeats_batch` is
called once (`controller.py:2470`). Both already take a caller-owned
`Tx`. Flipping them to "snapshot in, `MutationBatch` out" is a localised
change in two places. Every scheduler-side and RPC-handler method is
also single-caller and already wraps the call in `with
self._db.transaction() as cur:` in the caller. So lifting the `Tx` out
of `transitions.py` does not need a fan-in change anywhere — the
caller is already framing the transaction.

### Methods that don't belong in `transitions.py`

These are *not* state-machine code. Move them out as standalone work
(see T12 and friends in the task list):

- `prune_old_data(job_retention, worker_retention)` — incremental
  background loop that picks one prunable job/worker at a time. It is
  controller-loop logic that happens to delete rows. Move to
  `controller.py` (or a small `pruner.py`); the actual `DELETE` stays
  in `writes.delete_job` / `writes.remove_worker`.
- `record_task_status_text(task_id, detail_md, summary_md)`,
  `get_status_text_detail`, `get_status_text_summary` — pure
  in-memory string dicts. No state-machine relationship. Move to a
  controller-owned `TaskStatusText` helper (or directly onto the
  service/controller that hosts the dict).
- `add_endpoint(cur, endpoint, expected_attempt_id)`,
  `remove_endpoint(cur, endpoint_id)` — single-line wrappers around
  `EndpointsProjection.add` / `.remove`. The caller (service.py) can
  call the projection directly; the methods on `ControllerTransitions`
  are dead intermediaries.
- `update_worker_pings(worker_ids)` — one-line wrapper around
  `WorkerHealthTracker.bump_heartbeat`. Inline into the caller in
  `controller.py`.
- `load_workers_from_config(configs)` — no production caller. Tests
  that need worker fixtures should use a dedicated test helper.
- `fail_workers(failures, chunk_size)` — only called from test
  scenarios. The production path uses `fail_workers_batch`. Make
  `fail_workers` a test-only helper or absorb its body into
  `fail_workers_batch`.
- `log_event(action, entity_id, **details)` — module-level audit
  helper, used ~20 sites across `controller.py` and `service.py`. It
  has no `Tx`/DB dependency. Move to a small `audit.py` so
  `transitions.py` no longer hosts it as a side gate.

### Tightened state-machine surface

After the above relocations and the WorkingState refactor, the public
interface of `transitions.py` collapses to a handful of pure functions
of the shape `snapshot → MutationBatch`:

| Function | Inputs (snapshot fields) | Output |
|---|---|---|
| `apply_reconcile(snapshot, plans, results, now)` | tasks, attempts, attempt_uid_index, job_configs, active_workers, coscheduled_siblings, job_state_basis, job_descendants | `MutationBatch` |
| `apply_heartbeats(snapshot, requests, now)` | same as above | `MutationBatch` |
| `apply_direct_provider_updates(snapshot, updates, now)` | same | `MutationBatch` |
| `queue_assignments(snapshot, assignments, now)` | tasks, workers (+ liveness), job_configs | `(AssignmentResult, MutationBatch)` |
| `submit_job(snapshot, job_id, request, ts)` | meta (last_submission_ms), parent job basis | `(SubmitJobResult, MutationBatch)` |
| `cancel_job(snapshot, job_id, reason, now)` | subtree active tasks, jobs | `MutationBatch` |
| `cancel_tasks_for_timeout(snapshot, task_ids, reason, now)` | active task rows + sibling join | `MutationBatch` |
| `preempt_task(snapshot, task_id, reason, now)` | active task row + sibling/policy basis | `MutationBatch` |
| `mark_task_unschedulable(snapshot, task_id, reason, now)` | task row | `MutationBatch` |
| `register_or_refresh_worker(worker_id, address, metadata, ts, slice_id, scale_group)` | (no snapshot — full upsert) | `MutationBatch` |
| `replace_reservation_claims(claims)` | n/a | `MutationBatch` (whole-table replace) |
| `remove_finished_job(snapshot, job_id)` | job state | `(bool, MutationBatch)` |
| `remove_worker(snapshot, worker_id)` | worker detail | `(WorkerDetail | None, MutationBatch)` |
| `fail_workers_batch(snapshot, worker_ids, reason, now)` | active task rows for failing workers | `WorkerFailureBatchResult` (already aggregates `tasks_to_kill` / `task_kill_workers`) |
| `drain_for_direct_provider(snapshot, max_promotions, now)` | pending dispatch rows + active null-worker rows | `(DirectProviderBatch, MutationBatch)` |

`run_request_template(snapshot, job_id)` becomes a pure builder over the
job-config slice of the snapshot; the LRU cache that currently lives on
`ControllerTransitions` moves out to controller.py (a per-controller
cache, not a per-state-machine cache).

### Implication for `mutations.py`

Per user feedback (memory: [[feedback-iris-transitions-shape]]), the
ownership inverts: `transitions.py` defines `MutationBatch`,
`TransitionSnapshot`, and all the mutation dataclasses. The apply
function (currently `mutations.apply_mutations`) lives in a sibling
file but imports types from `transitions`. The file may be renamed to
`apply.py` to reflect that ownership. **T11 captures this move.**

### Implication for `ControllerTransitions` as a class

Once the relocations are done, the class itself has nothing to be
stateful about: `_health`, `_endpoints`, `_worker_attrs`, and
`_run_template_cache` are all controller-owned dependencies that don't
belong to the pure state machine. The remaining methods are
unambiguously functions over a snapshot. Drop the class — expose
free-function module API. The controller (which still owns the
projections and health tracker) feeds the snapshot in and consumes the
batch on the way out.

This is the "no DB references in transitions.py" end state, and it is
narrower than the original 10-task plan implied: most of the file
shrinks dramatically, and several methods leave the file entirely.

## Appendix A: read/write inventory for the reconcile hot path (T1)

Every DB call reachable from `apply_reconcile_result` / `apply_heartbeats_batch`.

### Reads

| Caller | Function | What it returns | Snapshot field |
|---|---|---|---|
| `apply_reconcile_result` | `reads.filter_existing_workers` | set of present worker IDs | `active_workers: set[WorkerId]` |
| `apply_reconcile_result` (error path) | `reads.bulk_get_task_detail` | `{task_id: TaskDetailRow}` for ASSIGNED candidates in the plan | `tasks: dict[JobName, TaskDetailRow]` |
| `apply_reconcile_result` (success path) | `reads.resolve_attempt_uids` | `{attempt_uid: (task_id, attempt_id)}` | `attempt_uid_index: dict[AttemptUid, tuple[JobName, int]]` |
| `apply_reconcile_result` (success path) | `reads.bulk_get_task_detail` | tasks referenced by all updates | `tasks` |
| `apply_reconcile_result` (success path) | `reads.bulk_get_attempts` | attempts referenced by all updates, plus current_attempt_id when stale | `attempts: dict[tuple[JobName, int], AttemptRow]` |
| `_apply_task_transitions` | `reads.get_task_detail` (fallback) | one task — only fires when bulk fetch missed | served from `tasks`; defensive fallback dropped |
| `_apply_task_transitions` | `reads.get_job_config` (memoized per job) | dict with `has_coscheduling`, `max_task_failures`, `preemption_policy`, `num_tasks`, `coscheduling_group_by` | `job_configs: dict[JobName, JobConfigRow]` |
| `_find_coscheduled_siblings` (inside `_apply_task_transitions` and `_remove_failed_worker`/`preempt_task`/`cancel_tasks_for_timeout`) | `reads.list_active_tasks(TaskScope(job_id=…), states=ACTIVE_TASK_STATES, exclude_task_id=…)` | `list[ActiveTaskRow]` per coscheduled parent job | `coscheduled_siblings: dict[JobName, list[ActiveTaskRow]]` keyed by parent job (consumer filters out the trigger task) |
| `_recompute_job_state` | inline SELECT `jobs.state, started_at_ms, max_task_failures` for one job | row | `job_state_basis: dict[JobName, JobStateBasis]` |
| `_recompute_job_state` | inline SELECT `tasks.state, COUNT(*)` GROUP BY for one job | dict of state → count | `JobStateBasis.task_state_counts` |
| `_recompute_job_state` | inline SELECT `tasks.error` WHERE non-null, ORDER BY `task_index` LIMIT 1 | first task error for terminal attribution | `JobStateBasis.first_task_error` |
| `_finalize_terminal_job` → `_kill_non_terminal_tasks` | `reads.list_active_tasks(TaskScope(job_id=…), states=NON_TERMINAL_TASK_STATES)` | tasks to kill | `job_descendants[job_id].active_tasks_by_job[job_id]` |
| `_finalize_terminal_job` → `_cascade_children` | inline recursive CTE: descendants of a parent job (with/without holder exclusion) | list of descendant job IDs | `job_descendants: dict[JobName, JobDescendants]` (precomputed for any job that could go terminal this tick) |
| `_finalize_terminal_job` → `_cascade_children` → `_kill_non_terminal_tasks` | `reads.list_active_tasks(TaskScope(job_id=child), states=NON_TERMINAL_TASK_STATES)` | per-descendant active tasks | `JobDescendants.active_tasks_by_job` |
| `_finalize_terminal_job` → `_resolve_preemption_policy` | SELECT `job_config.preemption_policy, jobs.num_tasks` for one job | int policy | reuse `job_configs` |

### Writes

| Caller | What gets written |
|---|---|
| `_finalize_attempt` | UPDATE `task_attempts` (state, finished_at_ms, error) + UPDATE `tasks` (state, error, finished_at_ms, optional failure_count/preemption_count, current_worker_id/address) + endpoint deletion |
| `_mark_task_producing_transition` | identical to `_finalize_attempt` but COALESCE-leaves `attempts.finished_at_ms` (capacity stays held) |
| `_apply_task_transitions` inner loop | `writes.apply_attempt_update` + `writes.apply_task_state_update` (one each per update); endpoint deletion when terminal |
| `_apply_task_transitions` stranded-attempt finalize | UPDATE `task_attempts.finished_at_ms` only |
| `_recompute_job_state` | UPDATE `jobs` (state, started_at_ms, finished_at_ms, error) when state changes |
| `_cascade_children` | UPDATE `jobs` (state=KILLED, error, finished_at_ms) for each descendant + (via `_kill_non_terminal_tasks`) attempt/task/endpoint updates per descendant's active tasks |
| `apply_reconcile_result` / `apply_heartbeats_batch` | in-memory `self._health.heartbeat([…], now_ms)` |
| `_apply_task_transitions` (BUILDING→FAILED, ASSIGNED→WORKER_FAILED) | in-memory `self._health.build_failed(WorkerId)` |

### Resulting snapshot/effect shapes

```python
@dataclass(frozen=True, slots=True)
class JobConfigRow:
    job_id: JobName
    has_coscheduling: bool
    max_task_failures: int
    preemption_policy: int      # JOB_PREEMPTION_POLICY_*
    num_tasks: int

@dataclass(frozen=True, slots=True)
class JobStateBasis:
    job_id: JobName
    state: int
    started_at_ms: int | None
    max_task_failures: int
    task_state_counts: dict[int, int]   # state → count
    first_task_error: str | None

@dataclass(frozen=True, slots=True)
class JobDescendants:
    job_id: JobName
    descendants_full: tuple[JobName, ...]                 # exclude_holders=False
    descendants_no_holders: tuple[JobName, ...]           # exclude_holders=True
    active_tasks_by_job: dict[JobName, tuple[ActiveTaskRow, ...]]
    # includes the root job_id entry (its own active tasks for _kill_non_terminal_tasks)

@dataclass(frozen=True, slots=True)
class TransitionSnapshot:
    now: Timestamp
    tasks: dict[JobName, TaskDetailRow]
    attempts: dict[tuple[JobName, int], AttemptDetailRow]
    attempt_uid_index: dict[AttemptUid, tuple[JobName, int]]
    job_configs: dict[JobName, JobConfigRow]
    job_state_basis: dict[JobName, JobStateBasis]
    job_descendants: dict[JobName, JobDescendants]
    coscheduled_siblings: dict[JobName, tuple[ActiveTaskRow, ...]]
    active_workers: frozenset[WorkerId]
```

```python
@dataclass(frozen=True, slots=True)
class AttemptMutation:
    """Update one task_attempts row. None means 'leave unchanged'; the
    apply layer translates None → no column in the SET."""
    task_id: JobName
    attempt_id: int
    state: int | None = None
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    coalesce_finished_at: bool = False  # COALESCE(finished_at_ms, :finished_at_ms)
    exit_code: int | None = None
    error: str | None = None
    coalesce_error: bool = False        # COALESCE(error, :error)

@dataclass(frozen=True, slots=True)
class TaskMutation:
    task_id: JobName
    state: int | None = None
    error: str | None = None
    exit_code: int | None = None
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    coalesce_finished_at: bool = False
    failure_count: int | None = None
    preemption_count: int | None = None
    clear_worker: bool = False          # set current_worker_id/address to NULL

@dataclass(frozen=True, slots=True)
class JobMutation:
    job_id: JobName
    state: int | None = None
    started_at_ms: int | None = None
    coalesce_started_at: bool = False
    finished_at_ms: int | None = None
    coalesce_finished_at: bool = False
    error: str | None = None

@dataclass(frozen=True, slots=True)
class EndpointDeletion:
    task_id: JobName

@dataclass(frozen=True, slots=True)
class WorkerHealthEffect:
    heartbeat: tuple[WorkerId, ...] = ()
    build_failed: tuple[WorkerId, ...] = ()

@dataclass(frozen=True, slots=True)
class LogEvent:
    action: str
    entity_id: str
    trigger: str | None = None
    details: tuple[tuple[str, object], ...] = ()

@dataclass(frozen=True, slots=True)
class MutationBatch:
    attempts: tuple[AttemptMutation, ...] = ()
    tasks: tuple[TaskMutation, ...] = ()
    jobs: tuple[JobMutation, ...] = ()
    endpoint_deletions: tuple[EndpointDeletion, ...] = ()
    health: WorkerHealthEffect = WorkerHealthEffect()
    log_events: tuple[LogEvent, ...] = ()
    tx_result: TxResult = TxResult()
```

Apply order in `apply_mutations`:

1. `health.heartbeat` — in-memory; OK to be eager.
2. `attempts` — UPDATE `task_attempts` (state, finished_at_ms, …); attempt
   release is the capacity-return signal so it goes first.
3. `tasks` — UPDATE `tasks` (state, counts, finished_at_ms, current_worker_*).
4. `endpoint_deletions` — DELETE endpoint rows for newly-terminal tasks.
5. `jobs` — UPDATE `jobs` (state, started/finished, error).
6. `health.build_failed` — in-memory; deferred so a failed write doesn't
   bump the tracker.
7. `log_events` — `logger.info` lines via `log_event()`.

Step 6 stays last among in-memory updates because today
`build_failed` only fires after we've decided the row really did go to
the failure-attributable state.

### Bulk-replace, deletion, and creation effects (T6–T8)

The hot-path migration above covers updates. Later migrations need
additional row types in `MutationBatch`:

- `ReservationClaimsReplace(claims: dict[WorkerId, ReservationClaim])` —
  whole-table swap used by `replace_reservation_claims`.
- `JobInsert`, `JobConfigInsert`, `JobWorkdirFilesInsert`,
  `TaskInsertBatch` — for `submit_job`.
- `WorkerUpsert`, `WorkerAttributesReplace` — for
  `register_or_refresh_worker`.
- `WorkerDeletion` — for `remove_worker`/`fail_workers*`.
- `TaskAssignment(task_id, worker_id, address, attempt_id, …)` — for
  `queue_assignments` (UPDATE `tasks` + INSERT `task_attempts`).
- `JobDeletion` — for `remove_finished_job` and `prune_old_data`.
- `MetaSet(key, value)` — for `submit_job`'s `last_submission_ms`.
- `UserUpsert(user_id, role, created_at_ms)` — for `submit_job`.

These don't need to land in T2; they show up as we migrate each method.
