# Spec: BackendWorkerStore contract

Pins the public surface of the encapsulated-sub-controller backend contract
([`design.md`](design.md)). Signatures are the **target**; phases (design §Phased
migration) land them incrementally, each byte-identical. Types referenced without
definition are existing (`WorkerId`, `WorkerStatusMap`, `ControlSnapshot`,
`BackendSchedulingInputs`, `AutoscalerState`, `TransitionSnapshot`,
`ControllerEffects`, `TaskTarget`, `Assignment`).

## 1. `BackendWorkerStore` — the typed, thread-aware boundary

New Protocol, `lib/iris/src/iris/cluster/controller/backend_store.py`. Replaces the
raw `ControllerDB` handle a worker-daemon backend holds today. Two impls: an
in-process `DbBackendWorkerStore` over the shared worker tables, and (later) a
`RemoteBackendWorkerStore` RPC client. **Thread affinity is part of the contract**
— each method documents which thread may call it.

```python
class BackendWorkerStore(Protocol):
    """A worker-daemon backend's owned worker/placement/capacity state.

    Encapsulates the `workers`, `worker_attributes`, `slices`, and `scaling_groups`
    state behind operation-scoped methods. The backend cannot reach job/task tables
    through it; the controller cannot reach worker tables except through it. In-process
    it is backed by the shared SQLite (so scheduler task<->worker joins survive);
    remote it is an RPC client to the backend's own store.
    """

    # --- Reads (any thread; each opens its own scoped read snapshot) ---
    def scheduling_inputs(self) -> BackendSchedulingInputs: ...
    def reconcile_snapshot(self) -> ControlSnapshot: ...
    def worker_status(self) -> WorkerStatusMap: ...
    def owned_worker_ids(self) -> set[WorkerId]: ...
    def transition_snapshot(self, *, now: Timestamp, **seeds) -> TransitionSnapshot: ...
    def worker_address(self, worker_id: WorkerId) -> str | None:
        """Resolve a worker's address for on-demand RPC routing (profile/exec/process)."""

    # --- Control-thread-only writes (asserted) ---
    def register_worker(self, registration: WorkerRegistration) -> RegisterOutcome:
        """Persist a worker (workers + worker_attributes), seed liveness, and detect
        a recycled-IP address conflict. If a stale prior owner holds the address, it is
        QUEUED for eviction (NOT torn down synchronously — slice teardown is
        control-thread-only). Returns the assigned worker id + any queued conflict."""
    def drain_pending_evictions(self) -> list[WorkerId]:
        """Tear down workers queued by register_worker's conflict detection: fail them,
        terminate their slices + healthy siblings, forget them from the tracker.
        Called once per control tick. Returns the workers actually removed."""
    def reap_workers(self, worker_ids: list[WorkerId], *, reason: str) -> list[WorkerId]:
        """Fail `worker_ids`, terminate their slices + siblings, persist autoscaler
        state, and forget them from the tracker (the reconcile-fold teardown path).
        Returns the full removed set (dead + siblings)."""
    def prune_dead_workers(self) -> list[WorkerId]:
        """GC workers this backend's tracker has held past the prune threshold.
        Per-backend; the controller retains a separate global orphan-slice sweep."""
    def persist_autoscaler_state(self, state: AutoscalerState) -> None:
        """Persist slices + scaling_groups. Single writer for this backend's capacity
        state; the controller no longer persists autoscaler state in its main tick."""

    def publish_status(self) -> BackendStatus:
        """Author this backend's cached worker projection (see §2)."""
```

`WorkerSource` (`controller/backend.py:433`) is **removed**; its read methods move
onto the store, and its raw `db`/`endpoints`/`worker_attrs` properties
(`backend.py:451/457/463`) are deleted — they were the leak.

## 2. `BackendStatus` — the cached worker projection

New, `lib/iris/src/iris/cluster/controller/backend_status.py`. Generalizes #6773's
`BackendSummary` (currently aggregate-only) into the per-worker projection the
controller serves all worker views from. The controller caches the last
`publish_status()` per backend and **never reads worker tables for views**.

```python
@dataclass(frozen=True)
class WorkerView:
    worker_id: WorkerId
    address: str
    scale_group: str
    device_type: str
    device_variant: str
    hardware: dict[str, str]              # md_* metadata
    attributes: dict[str, str]            # worker_attributes
    running_task_ids: frozenset[str]
    usability: WorkerUsability            # from the backend's tracker

@dataclass(frozen=True)
class BackendStatus:
    workers_by_id: dict[WorkerId, WorkerView]
    autoscaler_status: AutoscalerStatus   # existing autoscaler status shape
    counts: BackendCounts                 # workers/healthy/slices/tasks rollups
```

Recent **attempts** stay controller-owned task history (`task_attempts`, joined at
render) — they are not worker state. Worker-detail pages render from the cached
`WorkerView` + controller-owned attempt history; no on-demand backend call on the
normal path (see §11).

## 3. `BackendRuntime` change

`controller/backend.py:485`. The raw `db` field is replaced by the store; the other
DB-derived ingredients move into the store's construction (composer-side).

```python
@dataclass(frozen=True)
class BackendRuntime:
    store: BackendWorkerStore
    budget_defaults: UserBudgetDefaults
```

`bind_runtime(runtime)` (unchanged signature) now binds the store. The composer's
`make_backend` builds the in-process `DbBackendWorkerStore(db, endpoints,
run_template_cache, worker_attrs, owns_scale_group, budget_defaults, health=...)`
— the same ingredients, now behind the store.

## 4. `TaskBackend` control-interface delta

`controller/backend.py:512`.

- **ADD** `register_worker(self, registration: WorkerRegistration) -> RegisterAck`
  — the backend persists + collision-detects (via `store.register_worker`); the
  controller's `register` RPC keeps auth + scale-group→backend routing and calls
  this. Deletes `ops.worker.register` from the service path.
- **ADD** `status(self) -> BackendStatus` — publishes the cached projection (§2).
- **REMOVE** `teardown(self, dead_workers, *, reason)` (`backend.py:605`) — the
  recycled-IP path is internal (`store.drain_pending_evictions`). `run_teardown()`
  (no-arg, drains the backend's own `_pending_dead`) **stays**.
- **CHANGE** `get_process_status` / `profile_task` / `exec_in_container`: signatures
  unchanged (still take `TaskTarget`), but the controller stops filling
  `TaskTarget.address` from the DB at the RPC boundary — the backend resolves it via
  `store.worker_address`. The `TaskTarget.address` field becomes backend-filled.

Controller-side deletions (research §6): `_request_recycled_address_eviction`,
`request_worker_eviction`, `_pending_evictions`, `_drain_pending_evictions`,
`_worker_to_backend_map` (`service.py`/`controller.py`); the direct `workers`/
`worker_attributes` reads in `_worker_roster`, `_read_worker_detail`, `list_workers`,
`get_autoscaler_status`, `list_backends` move to the cached `BackendStatus`.

## 5. `Assignment` change — kill the commit-time worker read

`controller/ops/task.py:42`. Today `ops.task.assign` re-reads worker liveness
(`health.all()`, `task.py:80`) and addresses (`bulk_get_worker_addresses`,
`task.py:84`). The backend already schedules only over healthy workers it owns, so
it authors the address into the placement:

```python
@dataclass(frozen=True)
class Assignment:
    task_id: JobName
    worker_id: WorkerId
    address: str                          # NEW — backend-authored at schedule time
    lease_deadline_ms: int | None = None  # NEW — staleness bound for on-demand routing
    priority_band: int | None = None
```

`assign(cur, assignments, *)` drops the `health` parameter and the
`bulk_get_worker_addresses`/`health.all()` reads, using `assignment.address`. After
this the controller opens worker tables for **no** control path.

## 6. Registration types + flow

New, alongside `register_worker`:

```python
@dataclass(frozen=True)
class WorkerRegistration:
    worker_id: WorkerId
    address: str
    scale_group: str
    hardware: dict[str, str]
    resources: ResourceSpec
    attributes: dict[str, str]

@dataclass(frozen=True)
class RegisterOutcome:
    worker_id: WorkerId
    queued_eviction: list[WorkerId]       # stale prior owners of a recycled address

RegisterAck = RegisterOutcome             # returned up through the RPC
```

Flow: `service.register` (keeps auth + `backend_id_for_scale_group` routing) →
`backend.register_worker(registration)` → `store.register_worker` writes the row,
runs `find_address_conflicts`, queues conflicts. The control tick calls
`store.drain_pending_evictions()` (replacing `_drain_pending_evictions`),
preserving the control-thread teardown deferral.

## 7. Persisted shapes

- **In-process: no new tables.** The store is over existing `workers`,
  `worker_attributes`, `slices`, `scaling_groups` (`schema.py:426/458/508/494`).
  `BackendStatus` is in-memory, republished each tick (not persisted).
- **Remote (P7): placement projection.** The controller-owned task projection
  stores opaque `backend_id, worker_id, address, lease_deadline_ms` on
  `tasks`/`task_attempts` instead of a SQLite FK into a (nonexistent local) worker
  row. The remote backend's published status carries per-worker
  `WorkerView` + a freshness stamp; on-demand routing uses the published
  `address`/`lease_deadline_ms`. The wire contract is specced with the remote
  transport (out of scope here).

## 8. Errors

- `WorkerRegistrationConflict` is **not** raised — a recycled-IP collision returns
  via `RegisterOutcome.queued_eviction` (registration succeeds; eviction defers).
- `ProviderUnsupportedError` (existing, `backend.py:83`) still signals on-demand RPCs
  a backend can't serve.
- Store misuse (a backend reaching for job tables) is prevented structurally — the
  job tables are absent from the store surface — not by a runtime error.

## 9. File paths

| Piece | Path |
|---|---|
| `BackendWorkerStore` Protocol + `DbBackendWorkerStore` | `controller/backend_store.py` |
| `WorkerView` / `BackendStatus` / `BackendCounts` | `controller/backend_status.py` |
| `BackendRuntime` (store field), `TaskBackend` delta | `controller/backend.py` |
| `Assignment` (+address/lease), `assign` (drop reads) | `controller/ops/task.py` |
| `WorkerRegistration` / `RegisterOutcome` | `controller/backend_store.py` |
| Controller deletions (eviction plumbing, direct worker reads) | `controller/controller.py`, `controller/service.py` |
| Per-backend prune call + retained global orphan-slice GC | `controller/pruner.py` |

## 10. Out of scope

- The **remote transport wire** (RPC for `RemoteBackendWorkerStore` / `RemoteAgent`,
  auth) — the #6731 stack.
- The **meta-scheduler constraint language** (routing tasks to backends by
  availability constraints) — separate design.
- **Physical store split in-process** — never; the shared tables stay (the typed
  boundary is the partition).
- **k8s/`CLUSTER_VIEW`** changes beyond conforming to the (worker-less) store
  interface — it holds no worker store.

## 11. Open-question resolutions (locked)

- **Worker-detail staleness:** served from the cached `WorkerView` (last-tick) +
  controller-owned attempt history. No on-demand backend call on the normal render
  path; if a future need for live detail appears, add `store.worker_detail(id)` as
  an explicit on-demand call.
- **Address/lease authoring:** the backend stamps `Assignment.address`
  (+`lease_deadline_ms`); assignment-commit stays a controller task-state write that
  merely records them. (Chosen over making commit a store operation — keeps the
  commit transaction with the controller's task effects.)
- **Pruner home:** thin controller loop calls `store.prune_dead_workers()` per live
  backend; the controller keeps the global orphan-slice GC for abandoned/retired
  groups (no owning backend).
