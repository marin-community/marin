# Spec: BackendWorkerStore contract

Pins the public surface of the encapsulated-sub-controller backend contract
([`design.md`](design.md)). Signatures are the **target**; phases (§Phasing) land
them incrementally, each byte-identical. Types referenced without definition are
existing (`WorkerId`, `WorkerStatusMap`, `ControlSnapshot`, `BackendSchedulingInputs`,
`AutoscalerState`, `ControllerEffects`, `TaskTarget`, `Assignment`, `EndpointsProjection`,
`WorkerAttrsProjection`, `RunTemplateCache`, `WorkerHealthTracker`).

> **What the store actually is (codex-corrected).** It is **not** a worker-table-only
> partition. In-process the backend's `scheduling_inputs`/`reconcile_snapshot` read
> controller task/job/budget tables (the scheduler joins pending tasks + budgets +
> run-templates against `workers` in one SQLite — `policy.py:743-750`,
> `worker_source.py:187-195`), and teardown mutates controller task/endpoint
> projections (`backend.py:96-153`). So the store is an **operation-scoped snapshot +
> effects provider** over the shared DB: a fixed set of named operations the backend
> calls instead of holding a raw `ControllerDB`. The structural win is that the remote
> impl swaps those named operations for RPCs; the in-process impl keeps the in-SQL
> joins. The boundary is "the backend never holds a raw `db`," not "the backend can't
> read task tables."

## 1. `BackendWorkerStore` — the operation-scoped, thread-aware surface

New Protocol, `lib/iris/src/iris/cluster/controller/backend_store.py`. Replaces the raw
`ControllerDB` a worker-daemon backend holds today (via `BackendRuntime` →
`DbWorkerSource`). Two impls: in-process `DbBackendWorkerStore` over the shared DB, and
(P7) a `RemoteBackendWorkerStore` RPC client. Each method documents which thread may
call it.

```python
class BackendWorkerStore(Protocol):
    """A worker-daemon backend's owned worker state + the cross-seam snapshots/effects
    it needs to schedule, reconcile, and tear down its workers.

    In-process backed by the shared DB (so the scheduler's task<->worker joins survive);
    remote, an RPC client to the backend's own store. The backend writes worker rows only
    through this surface; the controller reads worker state only through a backend's
    published status (see §2).
    """

    # --- Snapshot reads (any thread; each opens its own scoped read snapshot) ---
    def owned_worker_ids(self) -> set[WorkerId]: ...
    def scheduling_inputs(self) -> BackendSchedulingInputs:
        """The backend's live workers + building counts + preemption rows joined with
        the controller's pending tasks/budgets into a SchedulingContext. Reads task/job
        tables in-process (the scheduler join); remote, the request payload carries them."""
    def reconcile_snapshot(self) -> ControlSnapshot:
        """The assigned-row + run-template snapshot reconcile folds against."""
    def worker_status(self) -> WorkerStatusMap: ...
    def transition_snapshot(
        self,
        *,
        now: Timestamp,
        seed_worker_ids: set[WorkerId],
        observation_uids: dict[WorkerId, str],
        seed_task_ids: set[str],
        extra_attempt_keys: set[tuple[str, int]],
    ) -> TransitionSnapshot:
        """The k8s/transition-reader snapshot (exact current seeds — `worker_source.py:114`)."""
    def worker_address(self, worker_id: WorkerId) -> str | None:
        """Resolve a worker's address for on-demand RPC routing (profile/exec/process/status)."""

    # --- Writes ---
    def register_worker(self, registration: WorkerRegistration) -> RegisterOutcome:
        """ANY THREAD (runs in the Register RPC handler). Persist a worker (workers +
        worker_attributes), seed liveness, detect a recycled-IP address conflict, and
        QUEUE the stale prior owner for eviction (do NOT tear down here — slice teardown
        is control-thread-only)."""
    def drain_pending_evictions(self) -> list[WorkerId]:
        """CONTROL THREAD ONLY. Tear down the workers register_worker queued: fail them,
        terminate slices + healthy siblings, forget them. Called once per control tick."""
    def reap_workers(self, worker_ids: list[WorkerId], *, reason: str) -> list[WorkerId]:
        """CONTROL THREAD ONLY. The full existing teardown algorithm (`backend.py:96-153`):
        fail_workers + commit_effects (endpoint projection drain) + worker-attrs eviction
        + autoscaler slice termination + persist_autoscaler_state + health.forget_many.
        Returns the removed set (dead + siblings)."""
    def prune_dead_workers(self) -> list[WorkerId]:
        """CONTROL THREAD ONLY. Per-backend dead-worker GC (P6)."""
    def persist_autoscaler_state(self, state: AutoscalerState) -> None:
        """CONTROL THREAD ONLY. Persist slices + scaling_groups (P5)."""

    def publish_status(self) -> BackendStatus:
        """CONTROL THREAD ONLY. Author the cached worker projection (§2) from an immutable
        snapshot so RPC-thread reads never race the control tick."""
```

`WorkerSource` (`controller/backend.py:433`) is **removed**; its read methods move onto
the store, and its raw `db`/`endpoints`/`worker_attrs` properties are deleted.

**Thread safety (codex #13).** DB writes are serialized by `ControllerDB`'s write lock
and `WorkerHealthTracker` has its own lock, but the autoscaler has no object-level
locking — control-thread `refresh`/`terminate_slices_for_workers` mutate group state
while RPC threads read status today (`autoscaler/runtime.py:625`, `service.py:2058`).
So `publish_status` runs on the control thread and returns an immutable snapshot; the
controller serves all views from the last published snapshot, never by calling
`autoscaler.get_status()` from an RPC thread.

## 2. `BackendStatus` — the cached worker projection

New, `controller/backend_status.py`. Generalizes #6773's aggregate `BackendSummary` into
the per-worker projection the controller serves all worker views from. Field set is the
union of what `_worker_roster`/`_read_worker_detail`/`list_workers`/`get_autoscaler_status`
read today (codex #9-#11):

```python
@dataclass(frozen=True)
class WorkerView:
    worker_id: WorkerId
    address: str
    backend_id: str
    scale_group: str
    metadata: worker_pb2.WorkerMetadata   # full typed metadata (device/provenance/attrs), not a lossy dict
    attributes: dict[str, AttrValue]      # typed worker_attributes
    running_task_ids: frozenset[str]
    healthy: bool
    usability: WorkerUsability
    consecutive_failures: int
    last_heartbeat_ms: int
    status_message: str

@dataclass(frozen=True)
class BackendStatus:
    workers_by_id: dict[WorkerId, WorkerView]
    autoscaler_vm_overlay: dict[WorkerId, int]   # running-task counts for VMs in autoscaler
                                                 # status but absent from the roster (codex #10)
    autoscaler_status: AutoscalerStatus
    counts: BackendCounts
```

Worker-detail rendering still performs a **controller-owned** task/job read for recent
attempts (`_attempts_for_worker` reads job-config resources — `service.py:843`); that
join stays controller-side and is not worker state (codex #11).

## 3. `BackendRuntime` change

`controller/backend.py:485`. The raw `db` and its DB-derived siblings move into the
store's construction; `run_template_cache` is honestly a reconcile-snapshot input the
store provides (not "worker state" — codex #8), which is consistent with the store being
a snapshot provider rather than a worker-table partition.

```python
@dataclass(frozen=True)
class BackendRuntime:
    store: BackendWorkerStore
    budget_defaults: UserBudgetDefaults
```

The composer builds the in-process store from today's ingredients:
`DbBackendWorkerStore(db, endpoints, run_template_cache, worker_attrs, owns_scale_group,
health, budget_defaults)`.

## 4. `TaskBackend` control-interface delta

`controller/backend.py:512`.

- **ADD** `register_worker(registration) -> RegisterAck` — the `register` RPC keeps auth
  + scale-group→backend routing and calls this (`store.register_worker`).
- **ADD** `status() -> BackendStatus` — publishes the cached projection (§2).
- **`teardown(dead_workers, *, reason)` stays through P1-P2** and is deleted only in the
  phase that internalizes eviction; `run_teardown()` (drains the backend's own
  `_pending_dead`) stays.
- **`get_process_status`/`profile_task`/`exec_in_container`** keep their `TaskTarget`
  signature; the controller stops filling `TaskTarget.address` from the DB — the backend
  resolves it via `store.worker_address`.
- **`GetTaskStatus` (codex #12)** — currently reads the live worker address from `workers`
  (`service.py:1673`). Switch it to the **denormalized task address** that `ListTasks`
  already uses (`service.py:1714`), closing the last view-path worker read.

## 5. `Assignment` change — the commit-time worker read

`controller/ops/task.py:42`. Today `assign` re-reads `health.all()` AND
`bulk_get_worker_addresses` (`task.py:80-84`). That recheck guards a **real intra-tick
race** (codex #5-#6): schedule runs before reconcile, reconcile can fold a worker bad,
and only then does commit run — so blindly trusting a schedule-time address can assign to
a just-reaped/missing worker (and would hit the `workers` FK, `schema.py:348`).

So the backend authors the address into the placement, but commit still validates:

```python
@dataclass(frozen=True)
class Assignment:
    task_id: JobName
    worker_id: WorkerId
    address: str              # NEW — backend-authored at schedule time
    priority_band: int | None = None
```

**Open fork (P3):** either (a) `assign` keeps a cheap commit-time liveness/existence
recheck against the backend's tracker (drops only the address *read*, not the validation),
or (b) the store exposes `validate_assignments(assignments) -> list[Assignment]` that
returns the surviving placements and commit trusts that. (a) is the smaller P3; (b) is the
cleaner seam for remote. `lease_deadline_ms` is **not** added here — there is no task/attempt
lease column today and nothing in P1-P3 reads one; it is deferred to the remote
placement-projection contract (P7), where it gets persistence + an expiry reader.

## 6. Registration types + flow

```python
@dataclass(frozen=True)
class WorkerRegistration:
    worker_id: WorkerId
    address: str
    scale_group: str
    metadata: worker_pb2.WorkerMetadata
    resources: ResourceSpec
    attributes: dict[str, AttrValue]

@dataclass(frozen=True)
class RegisterOutcome:
    worker_id: WorkerId
    queued_eviction: list[WorkerId]   # stale prior owners of a recycled address

RegisterAck = RegisterOutcome
```

`service.register` (auth + routing) → `backend.register_worker` → `store.register_worker`
(any-thread DB write + conflict queue). The control tick calls
`store.drain_pending_evictions()` (control-thread), preserving the current deferral
(`service.py:1869`, `controller.py:487`).

## 7. Phasing (codex "minimal P1")

Each phase byte-identical (N=1 suite green, no expected-value changes).

- **P1 — faithful store wrap.** Introduce `BackendWorkerStore`/`DbBackendWorkerStore`
  preserving the **exact** current surface (`owned_worker_ids`, `scheduling_inputs`,
  `reconcile_snapshot`, `worker_status`, exact `transition_snapshot`, `worker_address`)
  **plus `reap_workers` carrying the full existing teardown algorithm** — because removing
  the raw `db` breaks `RpcTaskBackend.teardown` (`backend.py:438`), which needs
  `db`/`endpoints`/`worker_attrs`. `BackendRuntime.db`→`store`; `WorkerSource` deleted.
  No registration, assignment-address, view, autoscaler-writer, or pruner changes. The
  one structural change: the backend holds a typed store, not a raw `db`.
- **P2** — backend-authored registration + control-thread eviction drain; delete the
  controller eviction plumbing and `teardown(dead_workers)` (now `store.reap_workers`).
- **P3** — `Assignment.address` + the commit-validation fork (§5); route on-demand worker
  RPCs + `GetTaskStatus` through the backend/denormalized address.
- **P4** — published cached `BackendStatus` (§2); views read the cache (incl. the
  autoscaler VM overlay), never the worker tables.
- **P5** — autoscaler persistence single-writer through the store.
- **P6** — per-backend `prune_dead_workers` + retained controller global orphan-slice GC.
- **P7 (remote)** — placement-projection contract: opaque `backend_id + worker_id +
  address (+ lease)`, no SQLite FK; the remote store + published-status freshness.

## 8. Persisted shapes

- **In-process: no new tables.** The store spans the existing `workers`,
  `worker_attributes`, `slices`, `scaling_groups` plus the read-only task/job/budget joins.
  `BackendStatus` is in-memory, republished each control tick.
- **Remote (P7):** the controller-owned task projection stores opaque `backend_id,
  worker_id, address (+ lease)` instead of a SQLite FK into a (nonexistent local) worker
  row; the wire contract is specced with the remote transport (out of scope here).

## 9. Errors

- Recycled-IP collision returns via `RegisterOutcome.queued_eviction` (no exception).
- Commit-time validation (P3) **skips** an assignment whose worker no longer validates
  (matching today's silent skip — `task.py:92`); it does not raise.
- `ProviderUnsupportedError` (existing) still signals on-demand RPCs a backend can't serve.

## 10. File paths

| Piece | Path |
|---|---|
| `BackendWorkerStore` + `DbBackendWorkerStore` + `WorkerRegistration`/`RegisterOutcome` | `controller/backend_store.py` |
| `WorkerView` / `BackendStatus` / `BackendCounts` | `controller/backend_status.py` |
| `BackendRuntime` (store field), `TaskBackend` delta | `controller/backend.py` |
| `Assignment` (+address), `assign` (validate not read) | `controller/ops/task.py` |
| Controller deletions (eviction plumbing, direct worker reads) | `controller/controller.py`, `controller/service.py` |
| Per-backend prune + retained global orphan-slice GC | `controller/pruner.py` |

## 11. Out of scope

- Remote transport wire (the #6731 stack); meta-scheduler constraint language; physical
  in-process store split (never — shared tables stay); k8s/`CLUSTER_VIEW` changes beyond
  conforming to the (worker-less) store interface.

## 12. Open questions (for review)

- **P3 commit validation fork** (§5): keep a cheap liveness/existence recheck in `assign`,
  or move to `store.validate_assignments`? The recheck guards a real intra-tick race, so
  one of the two is mandatory — which seam?
- **Worker-detail staleness:** served from the last-tick cached `WorkerView` + the
  controller-owned attempt/job read. Acceptable, or does the detail page need a live fetch?
- **Remote placement-projection (P7):** the per-worker view + freshness a remote backend
  reports so on-demand routing (exec/profile to a remote worker) stays correct.
