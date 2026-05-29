# Reconcile package — final design (as landed)

Author: russell + claude
Branch: `weaver/iris-reconcile-performance`
Status: **landed** (V4-1…V4-5 + Codex-fix + consolidation). Supersedes the
planning doc `reconcile-revision-v4.md`; where the two disagree, this file
describes what is actually in the tree and why.

This is a report of how the controller's `reconcile/` + `ops/` modules fit
together and a justification for every non-trivial function boundary. It is
meant to stand alone: a new reader should be able to trace any controller
action end-to-end from here.

---

## North star: functional core, imperative shell

The controller's state machine is a **pure kernel** parameterized by a
pre-loaded snapshot. Every mutation the kernel decides on is *returned as a
description* (`ControllerEffects`), never executed inline. The thin
**imperative shell** (`ops/`) loads a snapshot, calls the kernel, and drains
the effects into the write transaction.

```
ops verb  →  loader.load_*           (I/O: read a closed snapshot)
          →  batches.apply_*_batch   (pure: snapshot → ControllerEffects)
          →  effects.apply_effects   (I/O: drain effects into the Tx)
```

The one deliberate exception to "kernel does no I/O" is **diagnostic
logging** — see *Logging* below. It is observability, not state, and is
emitted inline; everything that touches state goes through effects.

---

## Module layout and responsibilities

Ten reconcile modules (plus a re-export-free `__init__`) and three `ops/`
modules, in dependency order (leaves first). Line counts approximate.

### Leaves — pure data, no reconcile imports

| Module | Responsibility |
|---|---|
| `snapshot.py` | The closed input bundle `TransitionSnapshot` + its row dataclasses (`JobConfigRow`, `JobStateBasis`, `JobDescendants`, `TaskHistogramRow`) and the neutral kernel inputs `TaskUpdate` / `HeartbeatApplyRequest`. No logic. |
| `policy.py` | Constants and predicate sets: replica/retry limits, well-known names, `FAILURE_TASK_STATES`, `NON_TERMINAL_TASK_STATES`, `CANCEL_GUARD_STATES`, `ERROR_STATES`, `TERMINAL_STATE_REASONS`. |
| `effects.py` | The effect contract: the `Mutation` Protocol, the `OverlayUpdater` Protocol + `WorkingStateOverlay`, the cross-aggregate effect categories (`EndpointDeletion`, `WorkerHealthEffect`, `LogEvent`), the `ControllerEffects` bundle, and the `apply_effects` sink. Knows the *Protocol*, never a concrete mutation class. |

### Level 1 — working state

| Module | Responsibility |
|---|---|
| `working_state.py` | `WorkingState`: the kernel's mutable scratchpad. Overlay-aware **reads** (`task_state`, `job_basis`, `task_state_histogram`, `active_tasks_for_job`, `job_descendants`, …) layered over the snapshot, and flat **record** methods (`record(mutation)`, `record_endpoint_deletion`, `record_log_event`, `record_worker_*`). Imports `snapshot` + `effects` only. |

### Level 2 — aggregate primitives (no cross-aggregate imports among the three)

| Module | Responsibility |
|---|---|
| `task.py` | Task-aggregate rules + its mutations (`TaskMutation`, `AttemptMutation`). Public primitives: `apply_one_transition` (the per-update core), `mark_task_terminating`, `finalize_attempt`, `preempt_one`, `unschedulable_one`, `timeout_one`, `resolve_task_failure_state`, `active_row_from_snapshot`, `task_is_finished_row`. |
| `job.py` | Job-aggregate rules + its mutations (`JobStateMutation`, `CascadeKillJobMutation`). Public: `recompute_state`. |
| `worker.py` | Worker reconcile-plan construction (`reconcile_workers`) and observation→`TaskUpdate` translation (`filter_observations_to_plan`, `observations_to_updates`, `assigned_updates_from_plan`). Imports `snapshot` only. |

### Level 3 — the single cross-aggregate edge

| Module | Responsibility |
|---|---|
| `peers.py` | Coscheduled-sibling rules: `find_coscheduled_siblings`, `terminate_coscheduled_siblings`, `requeue_coscheduled_siblings`. Imports `task` (for `mark_task_terminating`) — the **only** edge between aggregates. |

### Level 4 — orchestration

| Module | Responsibility |
|---|---|
| `batches.py` | The kernel's outer layer. Six `apply_*_batch` entry points (reconcile, heartbeats, direct-provider, worker-failures, terminal-decisions, cancel) that compose aggregate primitives + peer cascades + job recompute over one `WorkingState`. Imports only **public** names from `task`/`job`/`worker`/`peers`. |

### Level 5 — pure I/O

| Module | Responsibility |
|---|---|
| `loader.py` | Builds **closed** `TransitionSnapshot`s. One private closure core `_load_closed_snapshot` + four named public loaders. Imports leaves (`policy`, `snapshot`) only — knows nothing about transitions. |

### Level 6 — imperative shell

| Module | Responsibility |
|---|---|
| `ops/job.py` | `submit`, `cancel`, `remove_finished`, `has_reservation_flag`. |
| `ops/task.py` | `queue_assignments`, `apply_heartbeats`, `apply_provider_updates`, `apply_terminal_decisions`, `Assignment`. |
| `ops/worker.py` | `register_or_refresh`, `fail`, `apply_reconcile_observations`, `WorkerAttributeParams`, `WorkerFailureBatchResult`. |

---

## Import DAG (the contract)

```
        snapshot  policy  effects            ← leaves (no reconcile imports)
            │        │       │
            └────────┴───┬───┘
                         ▼
                  working_state                ← snapshot + effects only
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
       task             job            worker   ← no edges among the three
         ▲
         │ (single allowed cross-aggregate edge)
       peers
         │
         ▼
      batches            ← public symbols of task/job/worker/peers only
         ▲
         │
       ops/*             ← batches entry points + data-type classes only
         ▲
         │
      loader  ← policy + snapshot only (pure I/O; depends on no rule file)
```

Strictly downward, no cycles, no re-export shims. The cycle that sank an
earlier revision (`working_state` ↔ aggregate, because the overlay needed to
import concrete mutation types) is broken by the `OverlayUpdater` Protocol:
`WorkingState.record(m)` calls `m.overlay_apply(overlay)` through the Protocol
without importing any concrete mutation class. Aggregates import
`working_state` one-way.

**The contract.** The intended edges are: `loader` imports no rule file; the
three aggregates don't import each other; `peers` imports `task` only;
`batches` imports only public names; `ops/*` import only *data-type classes*
(never rule `def`s) from aggregate files; and the deleted modules (`state.py`,
`sweep.py`, `mutations.py`, `ops/reservation.py`) stay deleted. This is a
convention maintained by review, not a machine-checked invariant.

---

## Function-boundary justifications

The package was deliberately scrubbed of thin forwarding wrappers. The
helpers that remain each earn their boundary on one of three grounds:
**reuse** (≥2 call sites that would otherwise duplicate logic), **a named
public API surface**, or **hiding a correctness-critical detail**. The
judgment calls worth recording:

**The four loaders (`load_tasks_slice`, `load_jobs_slice`, `load_workers_slice`,
`load_reconcile_snapshot`) over one private `_load_closed_snapshot`.** The
closure logic — expand seeds to jobs, walk descendant subtrees, close
`all_tasks_by_job` / `job_state_basis` / `job_descendants` over *every* job in
the slice (not just roots) — lives in exactly one place. The public loaders
exist so the shell names its intent (`load_jobs_slice([job_id])` for cancel)
and never touches the internal `seed_*` vocabulary; exposing
`_load_closed_snapshot` directly would force every caller to understand seeds
and would leak the closure contract. `load_reconcile_snapshot` is separate
(not a `load_workers_slice` parameter) precisely because the reconcile-RPC
error path needs `observation_uids` / `extra_task_ids` / `extra_attempt_keys`
that no other caller wants — keeping them off `load_workers_slice` is what
makes that the clean two-argument loader it should be.

> `load_tasks_slice` retains a `worker_ids` kwarg (used only by
> `apply_heartbeats`). It is **load-bearing**: a preempted task is `PENDING`
> with no `current_worker_id`, so deriving `active_workers` from the loaded
> task rows would drop the worker's deferred terminal heartbeat — the one that
> stamps the attempt's `finished_at_ms` to release capacity. Documented in the
> loader docstring.

**`mark_task_terminating` vs `finalize_attempt` vs `_record_task_termination`.**
`_record_task_termination` is the single body (task mutation + attempt mutation
+ endpoint deletion). The two public wrappers differ by *one* flag,
`stamp_attempt_finished`: `finalize_attempt` (True) stamps the attempt finished
now (worker-failed / unschedulable — the attempt is truly over);
`mark_task_terminating` (False) marks the task terminal while the attempt row
stays held until the worker's next poll diffs it out. That boolean is a
capacity-accounting correctness detail; the two names keep it out of all nine
call sites. This is the textbook case for a wrapper-over-a-flag — the
abstraction hides something subtle, it isn't indirection for its own sake.

**`_drive_cascade` and `_apply_task_updates_with_cascades` (kept) vs
`apply_direct_provider_one` (removed).** The post-transition fan-out (peer
cascade → job recompute → finalize-if-terminal) is shared by the
heartbeat/reconcile and direct-provider batches, so it lives once in
`_drive_cascade`. The direct-provider path used to have its own near-duplicate
transition function; it is gone — the direct batch now reuses
`_apply_task_updates_with_cascades` with `track_worker_build_failures=False`
(direct providers manage their own hosts, so they don't reap build-failing
workers). One update-application core, one cascade core, two batch wrappers
that differ only in how updates are sourced.

**`_apply_timeout_batch` (kept, single caller).** A ~50-line two-phase
sibling-dedup algorithm. A cohesive sub-routine extracted for readability is a
*small concrete helper*, not indirection — inlining it would bloat
`apply_terminal_decisions_batch` and bury the timeout logic among the
preempt/unschedulable branches.

**`batches.py` owns the job-orchestration helpers (`_kill_non_terminal_tasks`,
`_cascade_children`, `_finalize_terminal_job`).** These call
`task.mark_task_terminating`, and `job.py` is forbidden from importing `task`.
Putting them in the orchestrator (which may import both) is what lets the
aggregate files stay non-cross-importing.

**`peers.py` is its own module, not folded into `task.py`.** The coscheduled
cascade is genuinely cross-aggregate (it is triggered from task transitions,
worker failures, timeouts, and cancel) and it calls back into
`task.mark_task_terminating`. Isolating it as the single `peers → task` edge
keeps that dependency visible and prevents `task.py` from growing peer logic.

---

## Effects and the Mutation Protocol

`ControllerEffects` carries one flat `mutations: list[Mutation]` plus
post-commit categories (`endpoint_deletions`, `health`, `log_events`). Each
concrete mutation (`TaskMutation`, `AttemptMutation`, `JobStateMutation`,
`CascadeKillJobMutation`) lives in its aggregate file and implements
`Mutation.apply(cur)`; those that change task/job state also implement
`OverlayUpdater.overlay_apply` so subsequent in-kernel reads see the
prospective state. `apply_effects` loops `m.apply(cur)` with **no `isinstance`
dispatch**.

The contract: every `apply`-bearing dataclass is a `Mutation`; every
state-changing mutation also implements `overlay_apply`; `effects.py` names no
concrete mutation class.

---

## Logging

`audit.log_event` is itself a `logger.info` line (consumed by the Iris log
server), not a durable store. There were two logging effect types; that
duality was over-built. The landed design keeps **one**:

- **Structured audit** (`LogEvent`, ~19 producers: `job_terminated`,
  `task_preempted`, `worker_failed`, `job_cancelled`, …) stays a *captured
  effect*, drained post-commit by `apply_effects`. It is the semantic audit
  trail, it benefits from rollback-safety (a rolled-back Tx records nothing),
  and it stays trivially testable via `effects.log_events`.
- **Free-form diagnostics** (the ~6 "worker vanished; dropping observations",
  "stale attempt", "unresolved uid" warnings) are **logged inline** at their
  produce sites via a module logger. They describe dropped/ignored inputs, not
  state changes; deferring them bought nothing. This is the single, conscious
  place the pure kernel emits I/O — justified because diagnostics are
  observability.

The overlay-staleness class of bug (a same-batch cascade being invisible to a
later update in the same batch) is closed: `apply_one_transition` and the
worker-failures loop read `state.task_state(...)` (overlay-aware), not the raw
snapshot row.

---

## Control-flow walkthroughs

**Heartbeat lands.** `ops/task.apply_heartbeats` → `loader.load_tasks_slice` →
`batches.apply_heartbeats_batch` (per update: `task.apply_one_transition` →
`_drive_cascade` → peer cascade via `peers` + `job.recompute_state` →
`_finalize_terminal_job` if the job went terminal) → `effects.apply_effects`.

**Cancel a job.** `ops/job.cancel` → `loader.load_jobs_slice([job_id])` (closes
the descendant subtree + coscheduled peers) → `batches.apply_cancel_job_batch`
(derives the subtree from `snapshot.job_descendants`, kills each job's tasks,
records `CascadeKillJobMutation`, cascades coscheduled peers) →
`effects.apply_effects` → endpoint sweep.

**Worker fails.** `ops/worker.fail` (chunked) → `loader.load_workers_slice` →
`batches.apply_worker_failures_batch` (per held task: `task.finalize_attempt`
→ `job.recompute_state` → cascade children / coscheduled siblings) →
`effects.apply_effects` → `writes.remove_worker`.

---

## Deliberate deviations from the v4 plan

Three items the planning doc specified were dropped because, once the modules
were inhabited, they would have been **dead code** (AGENTS.md: no dead code):

- **No `load_full_snapshot`.** The bulk reconcile tick builds plans from
  `ReconcileInputs` / `worker.reconcile_workers`, never a full
  `TransitionSnapshot`. Nothing calls it.
- **No `JobCreationContext` / `WorkerRefreshContext` + context loaders.**
  `ops.job.submit` reads its watermark/parent inline; `register_or_refresh` is
  a blind upsert that reads no prior state. The placeholders had zero
  consumers and were deleted.
- **`ops/__init__.py` keeps its submodule imports.** The plan said "empty," but
  the codebase reaches verbs via `from … import ops` then `ops.job` / `ops.task`
  / `ops.worker`; the redundant `__all__` was removed (AGENTS.md) but the
  load-bearing imports stay.

---

## Architecture conventions (maintained by review)

The package's structure rests on three contracts that are documented here
rather than machine-checked:

- **Import DAG** — the strictly-downward edges in the diagram above.
- **Loader closure** — every job in a loaded slice is closed: its configs,
  state basis, full task set, and descendant graph are all present, not just
  the roots.
- **Mutation purity** — every mutation implements `Mutation.apply`;
  state-changers also implement `OverlayUpdater.overlay_apply`; `effects.py`
  names no concrete mutation class.
