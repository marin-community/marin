# Backend contract: local execution substrates (Model D)

> **DECIDED 2026-07-01 — Model D (see the `delegation-model` artifact + `spec.md`).** A backend is a
> **local execution substrate** (GCP, k8s) that shares the controller's one job DAG. A **remote Iris
> cluster is NOT a backend** — it is a *federation peer* (owns its own DAG + backends; whole jobs are
> handed off to it), designed as a separate project (Track 2). This **rejects** the earlier
> "in-process and remote are the same modulo transport" north star below: unifying them was the
> source of the DAG-ownership confusion. Consequence: **the backend contract does not have to be
> remote-safe** — this doc's store/projection work (Track 1) is pure local hygiene; the DB-file split
> and any `Remote*Store` are dropped.

_Why are we doing this? What's the benefit?_

The `TaskBackend` Protocol (`controller/backend.py`) looks like a clean seam, but
it is an interface drawn around a monolith whose **state was never partitioned**:
workers, placement, liveness, the autoscaler, and slices all live in the one
controller DB, and the in-process backend reaches into that DB to read (scoped)
and write (teardown). The boundary holds only by convention, so concerns leak
across it. We want it to be **structural**: each backend owns its worker state
behind a typed store, the controller never independently reads worker state, and
worker views are served from an opaque published projection — clean ownership of *local*
worker state, so the controller's in-process multi-backend layering (GCP + k8s) stops leaking.
(The original goal here was "in-process and remote are the same modulo transport"; Model D
retires that — remote is federation, not a backend — see the banner above.)

## Background

See [`research.md`](research.md) for the full code trace (verified by codex).
Key facts: `reconcile`/`autoscale` are write-free (return effects/state, controller
commits); `teardown` is the sole direct backend DB write, coupled to the
backend-owned `WorkerHealthTracker`; `DbWorkerSource` reads are scale-group scoped.
The schema splits cleanly into controller-owned (jobs/tasks/attempts/budget) and
backend-owned (`workers`/`worker_attributes`/`slices`/`scaling_groups`) — except
`tasks.current_worker_id` / `task_attempts.worker_id` are FKs from controller rows
into `workers` (`schema.py:348/393`), and the scheduler joins across them in SQL.

## The corrected invariant

Worker **identities** legitimately cross the seam — a placement (`ScheduleResult`,
`backend.py:180`) *is* a task→worker binding, and the controller records it. What
must **not** cross is the controller independently **reading worker state**
(address, liveness, attributes). Today it does: assignment-commit re-reads
`workers.address` (`ops/task.py:80`), and `profile`/`process`/`exec` RPCs route
through `_read_worker` (`service.py:2170/2348/2559`). The invariant this design
enforces: **the backend authors everything the controller records or routes on;
the controller never opens the worker tables for control.**

## Challenges

- **The FK join seam.** Scheduler/reconcile reads (`build_scheduling_context`,
  `running_tasks_by_worker`, `resource_usage_by_worker`, `load_reconcile_rows`)
  join controller tasks against backend `workers` in one SQLite. So the boundary
  is a **typed store over shared tables**, not a second DB; physical split only for
  remote (where the join is replaced by the request payload + the remote's store).
- **The store is a thread-affinity boundary, not a DB wrapper** (codex's biggest
  risk). It spans the control loop, RPC-handler threads (register, profile/exec),
  the autoscaler thread, and liveness `Tx` hooks. Slice teardown is
  *control-thread-only*, which is why recycled-IP eviction is *deferred* today
  (`_pending_evictions`, drained on the control tick, `controller.py:487/1424`).
  The store must encapsulate that thread discipline, not just SQL.
- **Placement lives in a controller table.** `task_attempts.worker_id` is
  backend-authoritative but a controller column. Keep it as a **backend-authored
  projection** the controller never writes independently.

## Costs / Risks

- Large surface across several PRs (register, assignment-commit, on-demand worker
  RPCs, dashboard views, autoscaler persist, pruner). Mitigated by an N=1
  byte-identical gate per phase.
- **Thread affinity is the trap.** Moving register/eviction behind the store must
  preserve the control-thread deferral; a synchronous slice teardown in the
  Register RPC handler is a regression. This is where byte-identity is easiest to
  break.
- The typed store is a logical boundary over one DB in-process; enforcement is the
  interface + review (deliberate, to keep the in-SQL joins).
- Published-status views are last-tick, not live. Acceptable for the dashboard;
  on-demand worker RPCs (exec/profile) need live routing through the backend, not
  the cached projection.

## Design

**Control interface** (controller ↔ backend), no method opens worker tables:

| Method | In | Out |
|---|---|---|
| `schedule` | tasks + budget | placements — each carries the backend-authored `worker_id` **+ address/lease** |
| `reconcile` | request | task-state `effects` only |
| `autoscale` | demand | capacity + autoscaler state |
| `register_worker` | a validated, routed registration | ack — storage + collision decision + eviction internal |
| `status` | — | cached `BackendStatus` projection |
| `profile`/`process`/`exec` | task/worker target | routed *through* the backend (it resolves the address) |

`teardown(dead_workers)` is **deleted** from the control interface; eviction is
backend-internal.

**1. `BackendWorkerStore` — the typed, thread-aware boundary.** Replace the raw
`db: ControllerDB` in `BackendRuntime` (and stop `WorkerSource` exposing raw `db`,
`backend.py:450`) with a `BackendWorkerStore` exposing the operation-scoped surface
the backend uses — `scheduling_inputs`, `reconcile_snapshot`, `worker_status`,
`owned_worker_ids`, `register_worker`, `find_address_conflicts`,
`drain_pending_evictions` (control-thread), `reap_workers`,
`persist_autoscaler_state`, `worker_address`. In-process it's backed by the shared
worker tables; remote it's an RPC client to the remote's store. It encapsulates
both ownership (backend can't touch job tables) and thread affinity (which calls
are control-thread-only).

**2. Registration — controller-routed, backend-authored.** The `register` RPC
keeps controller auth + scale-group→backend routing (`service.py:1816/1827`), then
calls `store.register_worker(...)`: the backend writes its `workers`, runs
`find_address_conflicts`, and **queues** the stale prior owner into a backend-local
control-thread eviction drain (replacing `_pending_evictions`, preserving the
deferral). Deletes the controller's `_request_recycled_address_eviction`,
`request_worker_eviction`, `_drain_pending_evictions`, `_worker_to_backend_map`,
and `teardown`.

**3. No controller worker-state reads for control.** Placements carry the
backend-authored address/lease, so assignment-commit records it without reading
`workers`. `profile`/`process`/`exec` route through the backend, which resolves the
worker address. After this, the controller opens worker tables only never.

**4. Projection — published cached `BackendStatus` (the #6773 generalization).**
#6773's `BackendStatus` is absent on this branch today (it's a DB enum;
`BackendSummary` is aggregate-only). Define a dedicated cached
`BackendStatus(workers_by_id, autoscaler_status, counts)` carrying the worker-detail
fields the views need (metadata/attrs/running-task ids/address). Recent *attempts*
stay controller-owned task history (they're `task_attempts`), joined with the cached
worker projection at render. The controller serves `list_workers`, worker detail,
capacity, and autoscaler status from the cache — preserving current
filter/sort/page/count semantics — and never reads worker tables for views.

**5. Autoscaler — single writer, defined ordering.** The backend owns
`slices`/`scaling_groups` writes through the store. Today normal autoscale state
commits in the main-tick transaction while teardown persists separately
(`controller.py:1160`, `rpc/backend.py:137`); the store must pin that ordering, not
just collapse the two writers.

**6. Pruner — split, not relocated.** A thin controller loop calls
`store.prune_dead_workers()` per **live** backend; the controller **retains a global
orphan-slice GC** for abandoned/retired groups that have no owning backend
(`pruner.py:115`, `recovery.py:122`). Fully backend-owned pruning would leak
abandoned-group slices after a config removal.

**Phased migration** (each phase byte-identical, suite green):
1. **Faithful store wrap.** `BackendWorkerStore` over shared tables preserving the
   exact current read surface **plus `reap_workers` carrying the full teardown
   algorithm** (removing the raw `db` breaks `RpcTaskBackend.teardown`, which needs
   `db`/`endpoints`/`worker_attrs`). `WorkerSource` deleted; `BackendRuntime.db`→`store`.
   `teardown(dead_workers)` stays until P2. No registration/assignment/view/autoscaler
   changes. The store is operation-scoped (reads task/job tables in-process for the
   scheduler join), not a worker-table partition — see `spec.md`.
2. Registration backend-authored + backend-local control-thread eviction drain;
   delete controller eviction routing + `teardown(dead_workers)`. (Byte-identical
   only if the drain preserves control-thread timing.)
3. Backend-authored placement address/lease; route assignment-commit + on-demand
   worker RPCs through the backend. Eliminates the controller's worker-state reads.
4. Worker views off the published cached `BackendStatus` (preserving list/detail
   semantics).
5. Autoscaler persistence single-writer through the store (defined tx ordering).
6. Pruning split: per-backend dead-worker prune + retained controller global
   orphan-slice GC.
7. (Later) remote: explicit placement-projection contract (opaque
   `backend_id + worker_id + address/lease`, no SQLite FK); physical worker store.

## Testing

- **N=1 byte-identical per phase** — existing `lib/iris/tests/cluster` green, no
  expected-value changes (the gate this stack has used).
- **Eviction-drain timing** — recycled-IP eviction still defers to the control
  thread; no synchronous slice teardown in the Register RPC path.
- **Projection parity** — worker views from the cached status match the DB-read
  baseline (filter/sort/page/counts/detail) during the transition.
- **2-backend recycled-IP** — collision in the owning backend evicts internally,
  no controller routing, no cross-backend leakage.

## Open Questions

- **Worker-detail page staleness.** Codex's call is "no separate on-demand backend
  call unless cache staleness is unacceptable." Is last-tick worker detail
  acceptable for the detail page, or does it need a live fetch?
- **Address/lease authoring (Phase 3 fork).** Does the backend stamp an
  address/lease token into each placement (controller just records it), or does
  assignment-commit itself become a `store` operation? Both remove the worker-table
  read; they differ in where the commit transaction lives.
- **Remote placement-projection contract.** What exactly does a remote backend
  report up for placement (id + address + lease + liveness), and at what freshness,
  so on-demand routing (exec/profile to a remote worker) stays correct?
