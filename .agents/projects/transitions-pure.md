# Pure reconcile logic + thin commands

Status: **design v2 — review before implementation**

This is a follow-up to the `weaver/iris-reconcile-performance` branch. The
branch already split reconcile/heartbeat/terminal-decisions/worker-failures
into a `ControllerEffects` returning state machine + a single `apply_effects`
glue site. This document captures what is still wrong with `transitions.py`
and proposes the target structure.

v2 changes from v1: drop "wrapper" compromise, delete `ControllerTransitions`
outright, recognize that commands don't need the effects model at all.

## 1. High-level goal

There are **two kinds of writes** to the controller DB, and they should be
structured differently:

1. **Reconcile-style writes** are bulk: many workers' observations
   collide on shared state (job-state recomputation, coscheduled sibling
   cascade, cross-job descendant kills). They benefit from the
   snapshot-in → effects-out functional model so cascades from worker N
   are visible to worker N+1 inside one transaction.
2. **Command-style writes** are individual RPC handlers: `submit_job`,
   `cancel_job`, `register_worker`, `queue_assignments`,
   `remove_finished_job`, `replace_reservation_claims`. Each handles one
   client request and mostly populates SQL tables. There is no
   cross-record cascade; the "effects" model adds no value.

The branch refactored (1) but left (2) sitting on the same
`ControllerTransitions` class, with each command awkwardly building a
near-empty `ControllerEffects` just to emit one audit log line. That is the
"vestigial garbage" the user called out.

### Target structure

```
lib/iris/src/iris/cluster/controller/
├── reconcile.py          # PURE layer (snapshot in → ControllerEffects out).
│                         #   Absorbs the existing planner (rows → ReconcileRequest),
│                         #   the lifted state machine (was transitions.py), and the
│                         #   ControllerEffects / WorkingState / TransitionSnapshot
│                         #   dataclasses. No DB I/O.
├── reconcile_io.py       # I/O layer. Merges today's reconcile_state.py
│                         #   (snapshot loader), reconcile_writer.py (apply_effects),
│                         #   and reconcile_apply.py (per-tick glue entry points
│                         #   apply_reconcile / apply_heartbeats /
│                         #   apply_terminal_decisions / apply_worker_failures_batch /
│                         #   apply_direct_provider_updates).
├── commands.py           # NEW: direct-SQL RPC handlers + cur.register post-commit.
├── dispatch.py           # NEW: drain_for_direct_provider, build_run_request,
│                         #   RunTemplateCache.
├── reads.py              # bulk SELECTs (existing)
└── writes.py             # mutation helpers (existing)
                          #   `replace_reservation_claims` lives here — see §3.
```

Two reconcile-prefixed files instead of four: the pure/IO split that this
whole refactor is about becomes the file boundary. `controller.py` ends up
with two imports — `reconcile` for the dataclasses and pure layer,
`reconcile_io` for the run-a-tick entry points.

`ControllerTransitions` class is **deleted**. Its current methods either
move to module functions in `reconcile.py` (pure ones), to
`commands.py` / `dispatch.py` (impure ones), or are deleted.

The renaming of `transitions.py` → absorbed into `reconcile.py` is part of
Stage 7 (the lift). If `reconcile_io.py` later proves too large (~1100
lines projected), the snapshot loader is the natural break-out point; don't
preemptively split.

## 2. Invariants

After this work:

### `reconcile.py` (the pure module) must satisfy

1. No `sqlalchemy` imports.
2. No `iris.cluster.controller.reads` / `writes` / `schema` / `db` imports.
3. No `Tx` parameters anywhere.
4. No `cur.execute(...)` / `cur.register(...)` calls.
5. No `with db.transaction()` / `with db.read_snapshot()` blocks.
6. No `apply_effects` calls — pure functions return `ControllerEffects`; the
   caller (`reconcile_io`) applies.
7. Every public function takes a `TransitionSnapshot` (or a subset) plus a
   request shape, returns `ControllerEffects`. No class methods.

### `commands.py` (the new direct-SQL module) is

- A collection of free functions. The default shape is `cur: Tx` plus the
  RPC's inputs and any required projection references (`health`,
  `endpoints`, `worker_attrs`, `run_template_cache`). Commands take a
  projection only when they actually touch it; `submit_job` /
  `remove_finished_job` take none beyond the cache. Don't carry
  parameters through "for symmetry."
- **Hybrid exception**: `fail_workers` takes `db: ControllerDB` instead of
  `cur: Tx` because it opens its own chunked write transactions
  (see §4). It is the only command that owns transaction boundaries.
- Allowed dependencies: `sqlalchemy`, `Tx`, `reads`, `writes`, `schema`,
  `audit.log_event`, `projections.*`, `reconcile_io.*`. (Free to do
  whatever a normal RPC handler does — purity rules apply only to
  `reconcile.py`.)
- Each function does its SQL directly via `reads`/`writes`/`cur.execute`.
- Post-commit work (audit log lines, in-memory projection bumps) is
  scheduled via `cur.register(...)`. No `ControllerEffects` envelope.
- No state-machine / cascade logic. If a command needs cascade reasoning,
  it calls into `reconcile_io.apply_*` (e.g. `fail_workers` calls
  `apply_worker_failures_batch`).

### Projection ownership (unchanged)

`Controller.__init__` already constructs and owns `WorkerHealthTracker`,
`EndpointsProjection`, `WorkerAttrsProjection` (controller.py:1424–1428
passes them into `ControllerTransitions`). After the refactor, these
references stay on the `Controller` instance and its `ControllerServiceImpl`
(which already accepts them as kwargs). Command call sites pass their
existing references through. No new ownership / DI surface needs to be
introduced — the projections were always controller-owned, just laundered
through the transitions object. `RunTemplateCache` migrates from
`ControllerTransitions._run_template_cache` to a peer field on
`Controller` (constructed once in `__init__`).

### `dispatch.py` (the new dispatch module) is

- Free functions: `drain_for_direct_provider(cur, ...)`,
  `build_run_request(cur, row, attempt_id)`.
- One small class `RunTemplateCache` wrapping the `LRUCache`.
- `run_request_template(cache, snap, job_id)` takes the cache explicitly.

## 3. Audit: what each item becomes

Verdicts taken from the canvas run on the current branch HEAD
(`8a46821ad2`); two corrections vs. the agent's verdict are flagged with
**(*)**.

### Pure batched reconcile methods → lift into `reconcile.py`

| Method | Verdict | Notes |
|---|---|---|
| `apply_reconcile_batch` | keep | production reconcile path |
| `apply_heartbeats_batch` | keep | production heartbeat path |
| `apply_worker_failures_batch` (the pure one on the class) | keep | called from `reconcile_io.apply_worker_failures_batch` glue |
| `apply_terminal_decisions_batch` | keep | production timeouts/preempts/unschedulables |
| `apply_direct_provider_updates_batch` | keep | production k8s updates |
| `_recompute_job_state` + all `_apply_*_one` / `_*_batch` / `_assigned_updates_from_plan` / `_observations_to_updates` / `_filter_observations_to_plan` | keep | internal helpers; lift to module functions |

### Single-record variants → **DELETE** (and fix tests)

The full chain is test-only. Production never touches these.

| Method / function | Where | Plan |
|---|---|---|
| `transitions.ControllerTransitions.apply_reconcile_result` | transitions.py:1749 | delete |
| `transitions.ControllerTransitions.apply_task_updates_batch` | transitions.py:1731 | delete |
| `reconcile_apply.reconcile_apply` (single-result glue) | reconcile_apply.py:31 | delete (target file: `reconcile_io.py`) |
| `reconcile_apply.apply_task_updates` (single-worker glue) | reconcile_apply.py:191 | delete (target file: `reconcile_io.py`) |

**Fix the tests**: ~80 test sites and the replay framework use
`apply_task_updates(cur, ..., req)`. They must rewrite to
`apply_heartbeats(cur, ..., [req])`. The two `reconcile_apply` (single-result)
sites in `test_reconcile.py` rewrite to `apply_reconcile(cur, ...,
plans_by_worker, [result])`. Dispatch agents to do this mechanically;
no wrappers.

Test files affected (from the canvas + my grep):

- `tests/cluster/controller/conftest.py`
- `tests/cluster/conftest.py`
- `tests/cluster/controller/replay/events.py`
- `tests/cluster/controller/test_reconcile.py`
- `tests/cluster/controller/test_transitions.py`
- `tests/cluster/controller/test_dashboard.py`
- `tests/cluster/controller/test_scheduler.py`
- `tests/cluster/controller/test_service.py`
- `tests/cluster/controller/test_preemption.py`
- `tests/cluster/controller/test_reservation.py`
- `tests/cluster/controller/test_5470_preemption_reassignment.py`
- `tests/test_budget.py`

### Impure command methods → move to `commands.py` (direct SQL, no effects envelope)

Signatures show only the parameters each command actually uses. Don't pass
projection references "for symmetry" — the post-`ControllerEffects` shape
varies per command.

| Method | New name | Strategy |
|---|---|---|
| `submit_job` | `commands.submit_job(cur, *, job_id, request, ts, run_template_cache)` | inline SQL; `cur.register` for audit log. No `health`/`endpoints` — submission emits only one log line. |
| `cancel_job` | `commands.cancel_job(cur, *, job_id, reason, endpoints)` | direct SA UPDATE on subtree; `cur.register` for audit log; `endpoints.remove_by_job_ids(cur, subtree)` stays the same. No `health` — cancel doesn't touch liveness. |
| `register_or_refresh_worker` | `commands.register_or_refresh_worker(cur, *, worker_id, ..., health, worker_attrs)` | already uses `cur.register` for health bump; drop the `ControllerEffects` envelope and call `log_event` via `cur.register`. No `endpoints` — registration doesn't touch them. |
| `register_worker` | **DELETE** (bench-only) — update `benchmark_controller.py:1725` to call `commands.register_or_refresh_worker` directly |
| `queue_assignments` | `commands.queue_assignments(cur, assignments, *, health)` | already bulk-reads liveness via `health.all()`; replace `apply_effects` with `cur.register` log lines. No `endpoints`. |
| `remove_finished_job` | `commands.remove_finished_job(cur, job_id)` | trivial — `reads.get_job_state` + `writes.delete_job` + `cur.register(log)`. Takes no projections. |
| `remove_worker` | **DELETE** (test-only) — tests rewrite to `writes.remove_worker(cur, ..., health=..., worker_attrs=...)` directly (which is what production uses) |
| `replace_reservation_claims` | **MOVE TO `writes.py`** — not a command, just a `DELETE + INSERT` sync helper for the `reservation_claims` table. Its callers are `controller.py:1917,1965,2074` (scheduler/autoscaler loops), not any RPC handler in `service.py`. Lives next to `writes.bulk_insert_tasks` / `writes.delete_job` / etc. Skip `commands.py` entirely. |
| `fail_workers_batch` | `commands.fail_workers(db, *, worker_ids, reason, health, endpoints, worker_attrs)` | **Hybrid**: takes `db: ControllerDB` (opens its own chunked txns), then calls into the pure `reconcile.apply_worker_failures_batch` via `reconcile_io.apply_worker_failures_batch` per chunk. |

### Dispatch surface → move to `dispatch.py`

| Method | New name |
|---|---|
| `run_request_template` | `dispatch.run_request_template(cache, snap, job_id)` |
| `drain_for_direct_provider` | `dispatch.drain_for_direct_provider(cur, *, cache, max_promotions)` |
| `_build_run_request` | `dispatch.build_run_request(cur, row, attempt_id)` |
| `_PENDING_DISPATCH_COLS`, `_pending_dispatch_row` | move to `reads.py` (where the other dispatch row decoders live) |
| `RUN_REQUEST_TEMPLATE_CACHE_SIZE`, `LRUCache` wrapper | new `dispatch.RunTemplateCache` |
| `endpoints` property on `ControllerTransitions` | **DELETE** (test-only) — tests rewrite to pass the `EndpointsProjection` they already construct |

### Module-level dead code → **DELETE**

| Item | Why |
|---|---|
| `_worker_row_exists` (transitions.py:1493) | zero callers, even internally |
| `WorkerConfig` dataclass (transitions.py:218) | zero external imports; the production `WorkerConfig` is `iris.cluster.worker.WorkerConfig` (a different class) |

### Constants / dataclasses → keep, may move

`WorkerAttributeParams`, `TaskUpdate`, `HeartbeatApplyRequest`,
`TerminalKind`, `TerminalDecision`, `Assignment`, `SchedulingEvent`,
`ClusterCapacity`, `DirectProviderBatch`, `DirectProviderSyncResult`,
`WorkerFailureBatchResult`, plus constants `MAX_REPLICAS_PER_JOB`,
`DEFAULT_MAX_RETRIES_PREEMPTION`, `RESERVATION_HOLDER_JOB_NAME`,
`HEARTBEAT_STALENESS_THRESHOLD`, `DIRECT_PROVIDER_PROMOTION_RATE`,
`_LAST_SUBMISSION_KEY` → keep. Most stay in `reconcile.py`;
`Assignment`, `SchedulingEvent`, `ClusterCapacity`, `DirectProviderBatch`,
`DirectProviderSyncResult` migrate alongside their callers — see
§7 open question 2 for the per-shape split.

## 4. The command pattern

Concrete shape for `commands.py` entries. The point: no `ControllerEffects`,
no planner. Just SQL + `cur.register` for post-commit side effects.

### Example: `submit_job` (after)

```python
# commands.py
def submit_job(
    cur: Tx,
    *,
    job_id: JobName,
    request: controller_pb2.Controller.LaunchJobRequest,
    ts: Timestamp,
    run_template_cache: RunTemplateCache,
) -> None:
    """Insert the job row and expand its tasks. Caller owns the transaction."""
    run_template_cache.pop(job_id.to_wire())
    submitted_ms = ts.epoch_ms()

    # ... all the existing SQL ...
    writes.insert_job(cur, ...)
    writes.insert_job_config(cur, ...)
    writes.bulk_insert_tasks(cur, ...)

    cur.register(
        lambda: log_event("job_submitted", job_id.to_wire(),
                          num_tasks=replicas, error=validation_error)
    )
```

Before: built a `ControllerEffects` with one `LogEvent`, called
`apply_effects(cur, effects, health=..., endpoints=..., now=ts)` — which
just iterated zero mutations and ran one post-commit hook.

After: directly schedules the post-commit hook. No envelope, no projection
references it doesn't need.

### Example: `register_or_refresh_worker` (after)

```python
# commands.py
def register_or_refresh_worker(
    cur: Tx,
    *,
    worker_id: WorkerId,
    address: str,
    metadata: job_pb2.WorkerMetadata,
    ts: Timestamp,
    health: WorkerHealthTracker,
    worker_attrs: WorkerAttrsProjection,
    slice_id: str = "",
    scale_group: str = "",
) -> None:
    # ... existing SQL ...
    writes.upsert_worker_row(cur, {...})
    cur.execute(delete(worker_attributes_table).where(...))
    if attrs:
        cur.execute(insert(worker_attributes_table), [...])

    worker_attrs.set(cur, worker_id, attr_dict)  # registers its own commit hook
    cur.register(lambda: health.register(worker_id, now_ms=ts.epoch_ms()))
    cur.register(
        lambda: log_event("worker_registered", str(worker_id), address=address)
    )
```

### Hybrid: `fail_workers` (after)

```python
# commands.py
def fail_workers(
    db: ControllerDB,
    *,
    worker_ids: list[str],
    reason: str,
    health: WorkerHealthTracker,
    endpoints: EndpointsProjection,
    worker_attrs: WorkerAttrsProjection,
) -> WorkerFailureBatchResult:
    """Open chunked write txns and apply pure worker-failure cascade per chunk."""
    if not worker_ids:
        return WorkerFailureBatchResult(removed_workers=[])

    # snapshot read for active workers
    with db.read_snapshot() as snap:
        ...

    removed: list[tuple[WorkerId, str | None]] = []
    for chunk in chunks(failures, FAIL_WORKERS_CHUNK_SIZE):
        with db.transaction() as cur:
            ...
            reconcile_io.apply_worker_failures_batch(
                cur, live_chunk,
                health=health, endpoints=endpoints, worker_attrs=worker_attrs,
                now=Timestamp.now(),
            )
            ...
    return WorkerFailureBatchResult(removed_workers=removed)
```

The only difference from today: it lives in `commands.py` instead of being a
method on the now-deleted `ControllerTransitions`, and the
`reconcile_io.apply_worker_failures_batch` function takes `worker_attrs`
directly instead of fishing it out of `transitions._worker_attrs`.

## 5. Migration stages

Stages are numbered 1–8 within this document. (The branch this work sits
on has already shipped earlier numbered stages for unrelated performance
work; this document restarts the count to stay self-contained.)

Each stage commits independently. Tests pass after every stage.

### Stage 1 — Delete dead code

- Delete `ControllerTransitions._worker_row_exists`.
- Delete `transitions.WorkerConfig` (the unused dataclass — not the
  `iris.cluster.worker.WorkerConfig` that's actually used).

### Stage 2 — Delete single-record reconcile variants + fix tests

- Delete `transitions.apply_reconcile_result`,
  `transitions.apply_task_updates_batch`.
- Delete `reconcile_apply.reconcile_apply`, `reconcile_apply.apply_task_updates`.
- Refactor every test/replay-framework callsite to use the batched glue.

Dispatch sub-agents per test file (or per cluster of test files) — the
rewrite is mechanical: `apply_task_updates(cur, txns, req, ...)` becomes
`apply_heartbeats(cur, txns, [req], ...)`. The replay framework needs
slightly more care because it dispatches by event type.

Update `benchmark_controller.py:2675` to use `apply_reconcile` instead of
`apply_reconcile_result`. Note: the current bench call
`state.txns.apply_reconcile_result(cur, plan, result, now)` is already
wrong — it passes `cur` where `apply_reconcile_result` expects a
`TransitionSnapshot` and ignores the returned `ControllerEffects`. The
migration is not a rename: rewrite the loop body to drop into the new
batched glue with `plans_by_worker={plan.worker_id: plan}` and a single
`results=[result]`, matching the production call shape.

### Stage 3 — Delete bench-only and test-only impure surface

- Delete `transitions.register_worker`; update
  `benchmark_controller.py:1725` to call `register_or_refresh_worker`
  directly. (`register_or_refresh_worker` itself is still real; only the
  parameter-less wrapper goes away.)
- Delete `transitions.remove_worker`; refactor its test callers in
  `test_transitions.py` and the replay framework to call
  `writes.remove_worker(cur, ..., health=..., worker_attrs=...)` directly,
  matching what production already does in today's
  `reconcile_apply.apply_worker_failures_batch` (→ `reconcile_io.*` after
  the file merge in Stage 7) and `pruner.py`.
- Delete the `endpoints` property; refactor tests to take the
  `EndpointsProjection` they already construct.

### Stage 4 — Introduce `commands.py` + move `replace_reservation_claims` to `writes.py`

Create `commands.py`. Move the impure command methods listed in §3 as free
functions, **dropping the `ControllerEffects` envelope** for each. Each
function gets `cur.register(...)` calls for its audit logs and any
in-memory projection bumps it needs.

Methods moved into `commands.py`:
- `submit_job`, `cancel_job`, `register_or_refresh_worker`,
  `queue_assignments`, `remove_finished_job`.

Method moved into `writes.py` (separate from the commands list — it's a
table-sync helper, not an RPC handler):
- `replace_reservation_claims` → `writes.replace_reservation_claims(cur,
  claims)`. Update `controller.py:1917, 1965, 2074` to call
  `writes.replace_reservation_claims(cur, claims)` directly.

Update `service.py`, `controller.py`, and any test callers to import from
`commands` instead of `ControllerTransitions`. The signatures change only
in that they take explicit `health` / `endpoints` / `worker_attrs` /
`run_template_cache` arguments instead of finding them on a transitions
instance.

**Drop `_LAST_SUBMISSION_KEY` in the same commit as `submit_job` moves.**
The meta-key is a parallel bookkeeping value for monotone submission
timestamps; it is derivable from the jobs table itself:

```python
# replaces the meta SELECT + UPSERT pair in submit_job
last_ms = cur.execute(
    select(func.coalesce(func.max(jobs_table.c.submitted_at_ms), 0))
).scalar_one()
effective_submission_ms = max(submitted_ms, last_ms + 1)
```

The MAX query is O(log n) with the existing index on `submitted_at_ms`
(used by the priority queue). Robust to pruning: re-issuing a deleted
job's timestamp doesn't break ordering of anything that still exists.
Removes 12 LOC, one meta-table key, and the read-modify-write pattern.

The sibling `_PRIORITY_INSERTION_KEY` (a true monotone counter, never
compared to wall clock) stays in `writes.py` — it's not derivable from a
column max.

### Stage 5 — Introduce `dispatch.py`

Create `dispatch.py`. Move `run_request_template`,
`drain_for_direct_provider`, `_build_run_request`, `_PENDING_DISPATCH_COLS`,
`_pending_dispatch_row`. Wrap the `LRUCache` as `RunTemplateCache`.

Update `controller.py:2418` and `benchmark_controller.py:1383..2641` to
take the cache explicitly. `controller.py:1874` updates to
`dispatch.drain_for_direct_provider(cur, cache=..., max_promotions=...)`.

### Stage 6 — Move `fail_workers_batch`

- `fail_workers_batch` → `commands.fail_workers` (per §4).
  `controller.py:2556,2567` updates.
- Change today's `reconcile_apply.apply_worker_failures_batch` (the glue
  function, becoming `reconcile_io.apply_worker_failures_batch` after
  Stage 7) to take an explicit `worker_attrs: WorkerAttrsProjection`
  kwarg instead of reaching into `transitions._worker_attrs`
  (private-attr access at `reconcile_apply.py:299`). Update the one call
  site inside `commands.fail_workers` to forward its `worker_attrs` arg.

After this stage, only pure methods (`apply_*_batch`, `_recompute_job_state`,
the per-task helpers) remain on `ControllerTransitions`. `replace_reservation_claims`
already moved to `writes.py` in Stage 4.

### Stage 7 — Lift pure methods + collapse `reconcile_*` files into two

This is the biggest single commit. Three concurrent moves, all in one
stage because they share a wholesale import-surface rewrite and there's
no value in shipping any subset half-done.

**(a) Lift methods, delete class.**
- Lift every pure method (`apply_*_batch`, `_recompute_job_state`,
  `_apply_*_one`, etc.) out of `ControllerTransitions` into module
  functions.
- Delete the `ControllerTransitions` class.
- Drop the now-pointless string forward refs on `snapshot:
  "TransitionSnapshot"` — `TransitionSnapshot` is imported unconditionally
  at the top of the file.

**(b) Merge five reconcile-related files into two.** Today: `reconcile.py`
(existing planner) + `reconcile_state.py` + `reconcile_writer.py` +
`reconcile_apply.py` + the post-(a) lifted pure state machine in
`transitions.py`. Target:
- **`reconcile.py`** — the entire pure layer. Pulls in the existing
  planner content, the lifted state-machine functions (from the renamed
  `transitions.py`), the `ControllerEffects` shape (from
  `reconcile_writer.py`), the `WorkingState` and `TransitionSnapshot`
  dataclasses (from `reconcile_state.py`), and the various
  `*Mutation` / `LogEvent` / `LoggerEvent` / `WorkerHealthEffect` shapes.
  No DB I/O. `transitions.py` is deleted as part of this merge.
- **`reconcile_io.py`** — the I/O layer. Pulls in the snapshot loader
  (`load_transition_snapshot` and its helpers from `reconcile_state.py`),
  the effects applier (`apply_effects` from `reconcile_writer.py`), and
  the per-tick glue entry points (`apply_reconcile`, `apply_heartbeats`,
  `apply_terminal_decisions`, `apply_worker_failures_batch`,
  `apply_direct_provider_updates`, all from `reconcile_apply.py`).
  `reconcile_state.py`, `reconcile_writer.py`, and `reconcile_apply.py`
  are deleted as part of this merge.

**(c) Rewire callers.** `controller.py`, `commands.py`, `service.py`, all
tests, and `benchmark_controller.py` switch their imports from
`transitions` / `reconcile_state` / `reconcile_writer` / `reconcile_apply`
to `reconcile` (pure stuff) and `reconcile_io` (run-a-tick entry points).
The `transitions: ControllerTransitions` kwarg on every glue function
disappears at the same time (callers stop threading it through).

Top-of-file docstrings: `reconcile.py` says "Pure state-machine layer:
snapshot in, `ControllerEffects` out. No DB I/O." `reconcile_io.py` says
"Snapshot loader + effects applier + per-tick glue entry points. The DB-
facing side of reconcile."

### Stage 8 — Enforce invariants + final cleanup

- Delete now-unused imports / dead constants.

## 6. Decisions taken (no longer open)

From the v1 doc, the user has decided:

1. **Delete `ControllerTransitions` class.** Pure methods become module
   functions; impure ones move to `commands.py` / `dispatch.py`.
2. **No test-helper wrappers** for the deleted single-record variants.
   Tests get rewritten by sub-agents.
3. **No "effects" model for commands.** They don't need it — they're just
   SQL. `cur.register` handles the post-commit log-event need.

## 7. Open questions

1. ~~**Rename `transitions.py` → `reconcile_logic.py`?**~~ **Superseded.**
   The decision is no longer "rename to `reconcile_logic.py`" but
   "collapse `transitions.py` plus the three `reconcile_*` siblings into
   `reconcile.py` (pure) + `reconcile_io.py` (DB-facing)." See Stage 7(b)
   for the merge plan. Rationale: four `reconcile`-prefixed files
   distinguished only by suffix is itself a smell; the pure/IO split that
   this whole refactor is about should be the file boundary.
2. **Where do `Assignment` / `SchedulingEvent` / `ClusterCapacity` /
   `DirectProviderBatch` / `DirectProviderSyncResult` live?** Split by
   consumer, don't bundle:
   - `Assignment` → live next to `commands.queue_assignments` (its sole
     consumer); export from `commands.py`.
   - `SchedulingEvent`, `ClusterCapacity`, `DirectProviderBatch`,
     `DirectProviderSyncResult` → `dispatch.py`. These are the
     direct-provider sync protocol; they belong with
     `drain_for_direct_provider` and `build_run_request`.
   Done during Stage 4 (`Assignment`) and Stage 5 (the rest).
3. ~~**`_LAST_SUBMISSION_KEY`** is a string constant for the `meta` table.
   Recommendation: move to `writes.py` next to the other meta-table
   helpers during Stage 4.~~ **Resolved:** delete the key. The value is
   derivable from `MAX(jobs.submitted_at_ms)`; the meta row is duplicated
   bookkeeping. See Stage 4 for the swap.

## 8. Out of scope

- `WorkingState` / `TransitionSnapshot` shape changes.
- `ControllerEffects` shape changes (it stays exactly as-is, used only by
  the reconcile path).
- `apply_effects` ordering changes.
- Performance work (covered by prior stages 10–13).
