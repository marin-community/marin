# Backend contract: encapsulated sub-controllers

_Why are we doing this? What's the benefit?_

The `TaskBackend` Protocol (`controller/backend.py`) looks like a clean seam, but
it is an interface drawn around a monolith whose **state was never partitioned**:
workers, placement, liveness, the autoscaler, and slices all live in the one
controller DB, and the in-process backend reaches into that DB to read (scoped)
and write (teardown). The boundary holds only by convention, so concerns leak
across it. We want it to be **structural**: each backend owns its worker state
behind a typed store, the control interface carries no worker identities, and the
controller reasons about a backend's workers only through an opaque published
status. Then an in-process backend and a remote (possibly unreachable) Iris are
the same thing modulo transport — the multi-backend north star.

## Background

See [`research.md`](research.md) for the full trace. Established this session and
verified by codex: `reconcile`/`autoscale` are write-free (return effects/state,
controller commits at `controller.py:1142`/`1160`); `teardown` is the sole direct
backend DB write (`fail_workers` + `persist_autoscaler_state`, coupled to the
backend-owned `WorkerHealthTracker`); `DbWorkerSource` reads are scale-group
scoped. The schema splits cleanly into controller-owned (jobs/tasks/attempts/
budget) and backend-owned (`workers`/`worker_attributes`/`slices`/`scaling_groups`)
— **except** `tasks.current_worker_id` / `task_attempts.worker_id` are FKs from
controller rows into `workers`, and the scheduler joins across them in SQL.

## Challenges

- **The FK join seam.** The scheduler/reconcile reads (`build_scheduling_context`,
  `running_tasks_by_worker`, `resource_usage_by_worker`, `load_reconcile_rows`)
  join controller-owned tasks against backend-owned `workers` in one SQLite. A
  *physical* store split breaks those joins in-process. → Resolve with a **typed
  boundary over shared tables**: a logical partition enforced by an interface, not
  a second database. Physical split falls out only for remote (where the join is
  replaced by the request payload + the backend's own store).
- **Placement lives in a controller table.** `task_attempts.worker_id` is *which
  worker runs an attempt* — backend-authoritative, but physically a controller
  column. → Treat it as a **backend-authored projection**: the backend owns the
  authoritative assignment in its store and authors the controller's copy through
  reconcile `effects`; the controller never writes it independently.
- **Four controller-side touchers of worker tables** must move (research §6): the
  `register` RPC writes `workers` directly; the recycled-IP eviction detects
  collisions and routes `WorkerId`s back via `teardown(dead_workers)`; the pruner
  mutates `workers`/`slices`; `persist_autoscaler_state` is a dual-writer.
- **Every worker view reads `workers` directly** in `service.py`.

## Costs / Risks

- Large surface across several PRs (register RPC, pruner, autoscaler persist, every
  dashboard worker view). Mitigated by an N=1 byte-identical gate per phase.
- The typed store is a logical boundary over one DB — it doesn't physically stop a
  determined import of the tables; enforcement is the interface + review. That is
  the deliberate trade for keeping the in-SQL joins (acceptable: structural at the
  type level, which is what convention lacked).
- A published-status projection serves worker views from last-tick data, not live
  DB reads. Acceptable for the dashboard; we must confirm no *control* path in the
  controller depends on a live worker read.

## Design

The controller↔backend control interface carries **no worker identities**:

| Method | In | Out |
|---|---|---|
| `schedule` | tasks routed here + budget | placements |
| `reconcile` | request | task-state `effects` only |
| `autoscale` | residual demand | capacity changes + autoscaler state |
| `register_worker` | a worker registration | ack — **collision detection + recycled-IP eviction internal** |
| `status` | — | a generic **cached `BackendStatus`** (the projection) |

`teardown(dead_workers)` is **removed** — the backend reaps its own dead workers
(it already detects them in its liveness fold) and evicts its own address
collisions at registration. No `WorkerId` crosses for control.

**1. `BackendWorkerStore` — the typed boundary.** Replace the raw
`db: ControllerDB` in `BackendRuntime` with a `BackendWorkerStore` exposing exactly
the worker/placement/slice/autoscaler operations the backend uses today
(`scheduling_inputs`, `reconcile_snapshot`, `worker_status`, `owned_worker_ids`,
`register_worker`, `reap_workers`, `persist_autoscaler_state`,
`find_address_conflicts`). The in-process impl backs it with the shared
`workers`/`worker_attributes`/`slices`/`scaling_groups` tables; the backend
structurally cannot touch job tables, and the controller cannot touch worker tables
except through the store. The remote impl is an RPC client to the remote's own
store. This is codex's operation-scoped store, not a raw transaction handle.

**2. Registration internalized.** `RegisterWorker` routes to the owning backend's
`register_worker` (by scale group); the backend writes its workers, runs
`find_address_conflicts`, and evicts the stale prior owner — all internal. Deletes
`service.register`'s direct table writes, `_request_recycled_address_eviction`,
`request_worker_eviction`, `_pending_evictions`, `_drain_pending_evictions`,
`_worker_to_backend_map`, and the `teardown` interface method.

**3. Projection via published `BackendStatus` (generalizes #6773).** Each backend
publishes a generic cached status object each tick; the controller caches it per
backend and serves **all** worker views (`list_workers`, worker detail, capacity,
autoscaler status) from the cached object — never reading worker tables for views.
The controller reasons about a backend's workers only as this opaque status. #6773
already introduced the `BackendStatus`/`status()` mechanism and the always-on
Backends tab; this design promotes it from a dashboard surface to the canonical
projection contract.

**4. Single-writer autoscaler + per-backend pruning.** The backend owns
`slices`/`scaling_groups` writes through the store; the controller main-tick
`persist_autoscaler_state` dual-write is removed. The pruner's worker/slice GC
moves behind `store.prune()` (or into each backend's own maintenance) rather than a
controller thread mutating tables globally.

**Phased migration** (each phase byte-identical, suite green):
1. Introduce `BackendWorkerStore`; swap `BackendRuntime.db` for it; route current
   backend reads/writes through it. Establishes the structural boundary.
2. Internalize `register_worker` + recycled-IP eviction; delete the controller
   eviction plumbing + `teardown(dead_workers)`.
3. Move worker views to the published cached `BackendStatus`; stop reading worker
   tables in `service.py` views.
4. Single-writer autoscaler persist + per-backend pruner.
5. (Later, falls out of remote) physical store split for remote backends.

## Testing

- **N=1 byte-identical per phase** — the existing `lib/iris/tests/cluster` suite
  green with no expected-value changes (the gate this whole stack has used).
- **2-backend recycled-IP test** — a worker registers with an internal IP recycled
  from a worker the *owning* backend already holds; eviction happens inside that
  backend, with no controller routing and no cross-backend leakage.
- **Dashboard e2e** — worker views render from the published status; during the
  transition, assert parity against the DB-read baseline.

## Open Questions

- **BackendStatus sufficiency.** Does #6773's worker-detail object carry enough
  (per-worker attributes, running tasks, address) to serve the worker-detail page,
  or does that page need a separate on-demand backend call alongside the cached
  projection?
- **Placement projection longevity.** Is keeping `task_attempts.worker_id` as a
  backend-authored projection the permanent shape, or does it physically split for
  remote (the backend reports placement up, controller stores only the projection)?
- **Pruner home.** Fully into each backend, or a thin controller loop calling
  `store.prune()` per backend?
- **Any surviving control-path worker read?** After this, does the meta-scheduler
  (or any non-view controller path) still need a live worker read — and if so, does
  it come from the published status rather than the DB?
