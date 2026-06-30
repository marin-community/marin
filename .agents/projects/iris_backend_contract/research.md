# Research: the right backend contract for multi-backend Iris

Grounding for `design.md`. In-repo findings, the code trace done this session, and
the codex peer-review of the first-pass analysis.

## The current contract (what exists today)

`TaskBackend` Protocol — `lib/iris/src/iris/cluster/controller/backend.py`. Two
impls: `RpcTaskBackend` (in-process GCP/TPU/CW workers,
`backends/rpc/backend.py`) and `K8sTaskProvider` (`backends/k8s/tasks.py`).
Phase methods called uniformly each tick: `schedule` / `reconcile` /
`run_teardown` / `autoscale` / `status`, plus on-demand
`get_process_status`/`profile_task`/`exec_in_container`, and lifecycle
`bind_runtime`/`seed_liveness`/`configure_routing`.

Recent stack (PR #6730, all landed on the branch, **held** pending this design):
- effects-only `ReconcileResult` (backend returns task-state `effects`; controller
  commits).
- each backend owns its own `WorkerHealthTracker` (no shared controller tracker).
- deps threaded via ctor args + a `BackendRuntime` dataclass (commit 790989cb7d),
  which still carries the raw `db: ControllerDB`.

## The DB write/read trace (established this session, verified by codex)

| Phase | Reads DB | Writes DB | Who commits |
|---|---|---|---|
| `schedule` | scoped reads | no | returns `ScheduleResult` |
| `reconcile` | scoped reads | **no** | returns `effects` → controller commits (controller.py:1142/1147) |
| `autoscale` (normal) | scoped reads | **no** | returns `autoscaler_state` → controller persists (controller.py:1160) |
| `status` / `seed_liveness` | scoped reads | no | returns data |
| **`teardown`** | fresh reads | **YES** | **backend**, directly |

- `reconcile` (`rpc/backend.py` ~402-424): `apply_reconcile` authors effects only
  (`controller/ops/worker.py:235/241/281`); no write. Controller commits.
- `autoscale` normal path (`rpc/backend.py:475/479`): returns
  `autoscaler_state=persistable_state()`; controller persists
  (`controller.py:1129/1160`).
- `teardown` / `teardown_dead_workers` (`rpc/backend.py:119/137/139/145/153`): the
  ONLY direct backend write — `fail_workers` (`ops/worker.py`, which mutates the
  `WorkerHealthTracker` via `Tx` hooks through `writes.remove_worker`,
  `writes.py:552/556`) + `db.transaction()` → `persist_autoscaler_state`, then
  `health.forget_many`. The DB write and the backend-owned tracker mutation are
  coupled in one sequence.
- `DbWorkerSource` (`controller/worker_source.py`) query methods are all read-only
  (`control_read_snapshot`): `transition_snapshot` (58), `owned_worker_ids` (132),
  `scheduling_inputs` (136), `reconcile_snapshot` (151), `worker_status` (165).

## The smell: recycled-IP eviction leaks worker identity through the controller

Two recycled-IP mechanisms, asymmetric homes:
- **Already backend-internal:** misrouted-reconcile detection — a different live
  worker answers at a dead worker's recycled address; `RpcTaskBackend._observe_fleet`
  marks it UNREACHABLE inside the backend's own liveness fold. Clean.
- **Leaks through the controller:** at `RegisterWorker`, `service.py`
  `_request_recycled_address_eviction` reads the **shared DB**
  (`reads.worker_ids_at_address`) to find stale rows at the colliding address, then
  `controller.py` `request_worker_eviction` → `_pending_evictions` →
  `_drain_pending_evictions` maps workers→backends (`_worker_to_backend_map`) and
  calls `backend.teardown([worker_ids], reason=…)` (controller.py:1438/1444).

So the controller does worker-level address-collision detection and routes
`WorkerId`s into the backend. `teardown(dead_workers)` is the interface scar.

## codex peer-review of the first-pass analysis (`scratchpad/backend_db_interface.md`)

- Verified all four DB-trace claims against source (refs above).
- "backend-owned trackers ⇒ backend-owned teardown writes" is a **design
  consequence of the current transaction-hook model, not a logical necessity**;
  you could move it but only by passing backend-owned tracker hooks through the
  controller or a less-atomic two-phase commit.
- A read-only view is wrong (teardown writes). A thin `{read_snapshot, transaction}`
  protocol is too narrow (teardown also reads) and too weak (exposing
  `transaction()` is barely a boundary). **The meaningful boundary is
  operation-scoped:** a `BackendWorkerStore` with `transition_snapshot` /
  `worker_status` / `fail_workers` / `persist_autoscaler_state` etc., not raw
  transactions.
- **Doc bug:** `multi_backend.md:119` "No worker identity passes through the
  controller at any step" is FALSE — recycled-IP routes `WorkerId`s. Restate:
  reconcile never *returns* worker IDs; controller-originated maintenance paths may
  route them back to the owning backend (and under this design, shouldn't).

## Root-cause reading

The `TaskBackend` Protocol is a clean interface drawn around a monolith whose
**state was never partitioned**. Workers, placement, the liveness tracker, the
autoscaler, and slices all live in the one controller DB; the backend reaches in
to read (scoped) and write (teardown). The Protocol is a fiction over shared
mutable state — so the boundary holds only by convention, and concerns leak
across it (recycled-IP). The fix is to make the partition *structural*: each
backend owns its own store, even in-process.

## State partition inventory (Explore agent `a3fbd6d`)

Source: `controller/schema.py`. Two `MetaData` objects: `metadata` (main DB) +
`auth_metadata` (separate `auth.sqlite3`).

### Schema partition

**CONTROLLER-OWNED:** `meta`, `users`, `jobs` (incl. `backend_id` routing pin),
`job_config`, `job_workdir_files`, `tasks`, `task_attempts`, `endpoints` (scoped
to job/task), `user_budgets`, `api_keys`, `controller_secrets`.

**BACKEND-OWNED:** `workers` (incl. `slice_id`, `scale_group`),
`worker_attributes`, `scaling_groups` (autoscaler per-group state), `slices`.

**DEAD:** `backends` table (schema.py:414) is written only by migration 0032; no
live read/write — routing runs from in-memory `_scale_group_to_backend`.

**The structural seam (the hard part):** `tasks.current_worker_id` and
`task_attempts.worker_id` are FKs from **controller-owned** rows INTO the
**backend-owned** `workers` table (`ON DELETE SET NULL`). `workers.slice_id` /
`scale_group` link worker rows to the backend-owned autoscaler tables. The
scheduler/reconcile reads JOIN across this seam: `running_tasks_by_worker`,
`resource_usage_by_worker`, `load_reconcile_rows`, `build_scheduling_context`.
**This is why a physical store split is expensive in-process — the joins are
in-SQL today.** `task_attempts.worker_id` is *placement* (which worker runs an
attempt) — conceptually backend-owned, physically in a controller table.

### Backend read surface
`DbWorkerSource` (worker_source.py), every method via `control_read_snapshot()`:
`owned_worker_ids`→`worker_scale_groups`; `scheduling_inputs`→
`build_scheduling_context` (pending_tasks_with_jobs, healthy_active_workers_with_attributes,
resource_usage_by_worker, user budgets/bands, building_counts);
`reconcile_snapshot`→`load_control_snapshot` (list_active_healthy_workers,
load_reconcile_rows, scan_execution_timeout_rows) + run-templates; `worker_status`→
`running_tasks_by_worker`+`worker_scale_groups`+`health`. Slices/scaling_groups
are NOT read through DbWorkerSource — only via the autoscaler.

### Backend write surface
Only `teardown` writes directly: `fail_workers` (ops/worker.py:132 →
`_apply_worker_failures_chunk` → `commit_effects` + `writes.remove_worker`) writes
`workers` (delete), nulls `tasks`/`task_attempts` back-refs, invalidates
`worker_attributes`, and registers `tx.register(health.forget)` hooks
(writes.py:556); plus `persist_autoscaler_state` inside `db.transaction()`
(backend.py:139) writes `slices`+`scaling_groups`. `reconcile`/`autoscale` write
nothing (return effects/state). **`ops.worker.register` writes `workers`+
`worker_attributes` but is called from the controller service RPC handler, NOT
the backend** (see below).

### The recycled-IP / register flow (the smell)
1. `service.register` RPC (service.py:1807) resolves owning backend by scale group
   (1829), grabs that backend's `health` (1830), and **directly opens
   `self._db.transaction()` + `ops.worker.register(...)` (1832-43)** — the
   controller service writes backend-owned worker tables itself.
2. `_request_recycled_address_eviction` (1851) reads `workers` via
   `reads.worker_ids_at_address` (1858) to find stale rows at the recycled IP.
3. `controller.request_worker_eviction` → `_pending_evictions` →
   `_drain_pending_evictions` (controller.py:1424) reads `_worker_to_backend_map`
   (workers.scale_group) and calls `backend.teardown(group, reason)` (1444).

So worker creation, recycled-IP detection, and worker→backend routing all live in
controller/service code reading worker tables that should be backend-private.

### UP-facing projection (worker state surfacing to dashboard/CLI)
Almost all read `workers`/`worker_attributes` **DIRECTLY in service.py**, bypassing
the backend: `_worker_roster` (744), `_read_worker_detail` (447), `list_workers`
(1870), `get_autoscaler_status` (2000), `list_backends` (2810/2848). Liveness goes
through `controller.all_liveness()` (534, union of backends' trackers). Only
`get_kubernetes_cluster_status` (2074) routes *through* a backend. There is no
worker-projection `status()` method (the `BackendStatus` enum is just a column).

### Other controller-side toucher: the pruner
`controller/pruner.py` (a controller background thread): `_prune_dead_workers`
(87) writes `workers`+cascade via `writes.remove_worker`; `_prune_orphan_slices`
(115) reads `slices`+`workers` (`find_prunable_slice`) and writes `slices`
(`delete_slice`); scans every backend's `WorkerHealthTracker.all()`.

### Autoscaler persist/restore (dual-writer)
`persist_autoscaler_state` (autoscaler/persistence.py:24) writes
`scaling_groups`+`slices` and is called from **BOTH** backend teardown
(backend.py:140) AND the controller main-tick commit (controller.py:1161).
`load_autoscaler_checkpoint` (recovery.py:37) reads `scaling_groups`+`slices`+
`workers` at startup to rebuild tracked workers.

### Hardest cross-boundary couplings (ranked)
1. **`service.register` writing `workers`+`worker_attributes` + recycled-IP
   detection + worker→backend routing** (service.py:1832/1858, controller.py:1424).
   The section-4 smell — worker creation/eviction live controller-side.
2. **The pruner mutating `workers`+`slices` directly** (pruner.py:108/131-135) and
   scanning every tracker — a controller thread touching backend tables globally.
3. **Autoscaler restore reading `workers`** (recovery.py:66) + `persist_autoscaler_state`
   called from two sites/transactions (backend + controller main tick) — split
   ownership of slice/group writes.
4. **The whole UP-facing projection in service.py** reading `workers`/
   `worker_attributes` directly (every dashboard/CLI worker view bypasses the
   backend).
5. **The FK seam** `tasks.current_worker_id` / `task_attempts.worker_id` → `workers`
   and `workers.slice_id`/`scale_group`: controller-owned task rows physically
   point at backend-owned worker rows in one SQLite file; a physical store split
   forces the scheduler/reconcile joins to cross the interface or denormalize.

## Open / unclear (to resolve in Interrogate)

- Physical store split (separate tables/DB the backend exclusively owns) vs a
  typed `BackendWorkerStore` boundary over the *shared* tables (physical split
  only for remote).
- Whether `RegisterWorker` routes wholesale to the backend, or the controller
  keeps the RPC endpoint + auth and delegates only collision-detection.
- Whether all worker reads (dashboard/list_workers) route through the backend's
  projection, or the controller keeps a read-projection the backend publishes into.
