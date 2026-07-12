# Spec: Iris multi-backend contract

Pins the controller ↔ backend surface for multi-backend Iris (`design.md`). Reads as
**today → end state**. The architecture is **decided: Model D** (§1) — *backends are local
execution substrates that share the controller's one DAG; remote Iris clusters are separate
federation peers, not backends*. §2 the control contract, §3 the WorkerJobService split, §4 the
store, §5 the status projection are the local backend work (Track 1); remote is a separate
federation project (Track 2). Decision + rejected alternatives (A/B/C): the `delegation-model`
weaver artifact.

## 0. Today vs end state

**Today.** `TaskBackend` (`controller/backend.py`) is an interface drawn around a monolith
whose *state was never partitioned*. Workers, placement, liveness, the autoscaler, and slices
all live in the one `ControllerDB`; the in-process backend reaches into that DB to read
(scale-group scoped) and write (teardown). The same `ControllerService` is the single network
endpoint for **users** (LaunchJob, GetJobStatus…), **workers** (Register, RegisterEndpoint),
and **on-demand worker RPCs** (ProfileTask, GetProcessStatus, ExecInContainer, GetWorkerStatus,
BeginCheckpoint) — the controller resolves `task→worker→address` and forwards each of the last
group into the backend method-by-method (`_read_worker`, `service.py:452/2184/2233/2362/2573`).

**End state.** The controller keeps only cross-backend concerns; each backend owns its worker
state + execution behind a typed seam (a *self-contained sub-controller* only in Model B — under
Models A/D a local backend shares the controller's DAG). What the controller keeps depends on the
§1 fork — the *settled* part is below the line, the *forked* part (DAG ownership) is called out:

| Controller keeps (settled) | Each backend owns |
|---|---|
| routing (job → backend), meta-scheduling | workers, worker_attributes, endpoints, liveness, slices, autoscaler |
| a **service router** (task/worker/scale-group → backend) | task **execution** state |
| budget enforcement, a reported per-backend status projection | its **WorkerJobService** (§3) |
| *the job/task **DAG*** — **only in Model A** (§1); in Model B the backend owns it | *the DAG* — **only in Model B** (§1) |

Two structural moves get us to the settled part: **(a)** the backend owns its worker *state*
behind a typed store (P1–P5), and **(b)** the backend owns the worker/execution-facing *service*
while the controller becomes a router (§3). After both, the controller never opens a worker table
— not for control, not for forwarding. Whether it *also* stops owning the DAG is §1.

## 1. The model — federation ≠ backend (Model D, decided)

`TaskBackend` splits into two method families. Family 1 — **the control fold** — is the real
multi-backend seam (§2). Family 2 — **worker/execution forwards** — leaves the control interface
and becomes the backend's own service (§3). The decided architecture:

- **Backends are local execution substrates.** GCP workers, k8s. Tasks on a backend live in *this*
  Iris's one DAG; the backend authors effects, the controller folds them. This is the classic
  effects-up model, and it is correct and cheap for the local case.
- **Remote Iris clusters are federation peers, not backends.** A remote cluster is a full Iris that
  owns its own DAG + its own backends. Whole root jobs are *handed off* to it; it runs the entire
  tree and reports status + spend back. The federating Iris tracks a *federated-job handle*, never
  remote task rows. There is no `RemoteTaskBackend`; **the backend contract does not have to be
  remote-safe** — the single biggest simplification (no per-backend DB-file split, no
  `RemoteBackendWorkerStore`).
- **Job trees are locked to the parent's peer.** A federated root and its whole subtree run on the
  one peer. This is largely self-enforcing: a running task's Iris client targets
  `job_info.controller_address` — the controller that launched it (`client/client.py:1190`) — so a
  child spawned on a peer submits back to the *peer*, materialized in the peer's DAG. Federation
  only handles the root handoff and refuses the two escapes (re-pointing a child at the federating
  Iris; a cross-cluster constraint on a non-root job).

Rejected alternatives — **A** (controller owns every backend's DAG, remote ships effects up; made
#353 the capstone) and **B/C** (every backend a mini-iris) — and the full rationale are in the
`delegation-model` artifact (2× codex-reviewed). Consequence for **#353**: it becomes lower-priority
*local* cross-backend-tree fold hardening (not a remote blocker) and is **stood down** — PR #6805
closed, branch preserved for salvage.

## 2. The control contract — `TaskBackend` is the meta-scheduler fold

`controller/backend.py`. End-state control interface, four methods, none opening a worker table:

| Method | In | Out |
|---|---|---|
| `schedule` | routed pending tasks + budgets | placements — each carries the backend-authored `worker_id` + `address` |
| `reconcile` | request | task-state `effects` (Model A) / a status delta (Model B) |
| `autoscale` | demand | capacity + autoscaler state |
| `status` | — | cached `BackendWorkerReport` projection (§5) |

`register_worker`, `teardown(dead_workers)`, and the on-demand `profile`/`process`/`exec`
methods **leave** this interface: registration/teardown become store-internal (P2, landed),
and the on-demand RPCs move to the WorkerJobService (§3). *(Under Models A/B a `RemoteTaskBackend`
would implement these four over connect RPC; under the recommended Model D there is no
`RemoteTaskBackend` — remote is a federation peer, §1, not a backend, so this interface stays
in-process.)*

## 3. The WorkerJobService — backend-owned, controller-routed

The worker/execution-facing slice of `ControllerService` becomes a service the **backend
defines and serves**; the controller stops forwarding it method-by-method and instead
**routes** at the service edge.

- **Surface:** `Register` (worker→controller) and the on-demand `ProfileTask` /
  `GetProcessStatus` / `ExecInContainer` / `GetWorkerStatus` — the RPCs that need
  `task/worker → address` resolution, which is backend-owned state. The endpoint RPCs
  (`RegisterEndpoint`/`UnregisterEndpoint`/`ListEndpoints`) already live on a **separate**
  `EndpointService` (`controller.proto:689`); P5 makes its backing projection backend-owned and
  turns the service into a router. **Not included:** `BeginCheckpoint` — it snapshots the
  *controller's* DB (`controller.py:1427`), not worker state, and stays controller-side.
- **In-process:** the backend's service impl runs in the controller's uvicorn server; the
  controller dispatches at the *service* level (register by scale-group → backend; on-demand by
  `task → backend`) and the backend resolves the address internally. This deletes `_read_worker`
  / `_read_worker_detail` / `bulk_get_worker_addresses` from the controller.
- **The routing lookup itself is fork-sensitive.** Dispatching an on-demand RPC needs a
  `task_id → backend` (and `root_id → backend`) map. In Model A the controller owns the task rows,
  so this is a local read of `task.backend_id`. In Model B the meta owns no task rows, so it must
  keep its own routing index (root → backend, extended to any task in that tree) — the spec for B
  must define where that index lives and how it stays consistent with sub-Iris job trees.
- **Remote (Model D):** a remote Iris peer serves its *own* WorkerJobService internally; the
  federating Iris does **not** route worker RPCs into it as a backend. On-demand exec/profile/logs
  against a *federated* job reach the peer at the **federation** layer (proxy vs. redirect — §7),
  keyed by the federated-job handle, not by `task.backend_id`. (Under Models A/B this would instead
  be a `RemoteTaskBackend` serving the same WorkerJobService and P9 a transport swap; D drops that.)

This supersedes the earlier plan that kept the on-demand RPCs as `TaskBackend` methods routed via
`store.worker_address`; naming the whole family as one backend-owned service is the cleaner seam.

## 4. `BackendWorkerStore` — the operation-scoped, thread-aware boundary

`controller/backend_store.py`. Replaces the raw `ControllerDB` a worker-daemon backend held via
`BackendRuntime`. **Not** a worker-table-only partition: in-process its snapshot reads join the
controller's task/job/budget tables (the scheduler join, `policy.py:743`), so it is an
*operation-scoped snapshot + effects provider* over the shared DB. The structural win is that the
remote impl swaps these named operations for RPCs; the boundary is "the backend never holds a raw
`db`," not "the backend can't read task tables." Each method documents its calling thread.

```python
class BackendWorkerStore(Protocol):
    # Snapshot reads (any thread; each opens its own scoped read snapshot)
    def owned_worker_ids(self) -> set[WorkerId]: ...
    def scheduling_inputs(self) -> BackendSchedulingInputs: ...     # live workers ⋈ pending tasks/budgets
    def reconcile_snapshot(self) -> ControlSnapshot: ...            # assigned rows + run-templates
    def transition_snapshot(self, *, now, seed_worker_ids, observation_uids,
                            seed_task_ids, extra_attempt_keys) -> TransitionSnapshot: ...
    def worker_status(self) -> WorkerStatusMap: ...
    def worker_address(self, worker_id: WorkerId) -> str | None: ...  # on-demand RPC routing

    # Writes
    def register_worker(self, registration: WorkerRegistration) -> RegisterOutcome: ...  # ANY THREAD; queues eviction
    def drain_pending_evictions(self) -> list[WorkerId]: ...        # CONTROL THREAD; per tick
    def reap_workers(self, worker_ids, *, reason) -> list[WorkerId]: ...  # CONTROL THREAD; full teardown algo
    def prune_dead_workers(self) -> list[WorkerId]: ...             # CONTROL THREAD (P3, landed)
    def persist_autoscaler_state(self, state: AutoscalerState) -> None: ...  # CONTROL THREAD (P8)
    def publish_status(self) -> BackendWorkerReport: ...                  # CONTROL THREAD; immutable snapshot
```

**Thread safety.** The autoscaler has no object-level lock; control-thread `refresh` mutates
group state while RPC threads read. So `publish_status` runs on the control thread and returns
an immutable snapshot, and all views are served from the last published snapshot — never by an
RPC thread calling `autoscaler.get_status()`.

`BackendRuntime` collapses to `store` + `budget_defaults`; each backend builds its store from the
shared `ControllerDB` plus **its own** endpoints / worker_attrs projections + liveness tracker.
Bootstrap constructs these so the store is non-Optional (no two-phase `bind_runtime`).

## 5. `BackendWorkerReport` — the cached worker projection

`controller/backend_status.py`. Generalizes the aggregate `BackendSummary` into the per-worker
projection the controller serves *all* worker views from — the union of what
`_worker_roster`/`_read_worker_detail`/`list_workers`/`get_autoscaler_status` read today.

```python
@dataclass(frozen=True)
class WorkerView:
    worker_id: WorkerId; address: str; backend_id: str; scale_group: str
    metadata: worker_pb2.WorkerMetadata; attributes: dict[str, AttrValue]
    running_task_ids: frozenset[str]; healthy: bool; usability: WorkerUsability
    consecutive_failures: int; last_heartbeat_ms: int; status_message: str

@dataclass(frozen=True)
class BackendWorkerReport:
    workers_by_id: dict[WorkerId, WorkerView]
    autoscaler_vm_overlay: dict[WorkerId, int]   # task counts for VMs in autoscaler status, absent from roster
    autoscaler_status: AutoscalerStatus
    counts: BackendCounts
```

Worker-detail rendering still does a **controller-owned** task/job read for recent attempts
(`task_attempts` joined with job-config resources) — that is task history, not worker state.

## 6. PR sequence

**Landed:** P1 store wrap (#6788) · P2 backend-authored registration + eviction drain (#6792) ·
P3 per-backend prune (#6795) · P4 backend owns `WorkerAttrsProjection` (#6799).

**Track 1 — local backend hygiene (continue now):**

- **P5 — WorkerJobService: backend owns endpoints + the on-demand RPC surface; `EndpointService`
  and the on-demand handlers become controller routers (§3).** Supersedes the old "P5 endpoints
  only" + "P6 on-demand via store.worker_address" split — move the whole worker/execution-facing
  family as one unit. First concrete controller-as-router move.
- **P7 published `BackendWorkerReport` · P8 autoscaler single-writer** — local hygiene.
- **P6 / #353 — stood down.** The controller-side DAG fold over local backends is *local* robustness
  (no subtree pinning today, so a GCP+k8s job *tree* can span backends); it no longer gates anything.
  PR #6805 closed; branch `weaver/iris-mb-6-commit-pivot` + commits preserved to salvage the
  reconcile-kernel cleanup opportunistically. Resume only if local cross-backend-tree correctness
  becomes a priority.
- **~~P9 remote-as-a-backend~~ — deleted.** Remote is not a backend: no per-backend DB-file split,
  no `RemoteBackendWorkerStore`.

**Track 2 — federation (new, greenfield, later):** a `remote Iris peer` concept — root-job handoff
(exactly-once), federated-job handle + status/spend sync, cross-peer cancel, budget admission, peer
auth. Whole-subtree lock is largely free via `controller_address` inheritance (§1). Designed on its
own when a real remote cluster is on the table.

## 7. Out of scope / open questions

**Out of scope (this spec is Track 1 — local backend hygiene):** the entire federation project
(Track 2 — remote Iris peers, handoff, federated-job handle, cross-peer cancel/budget/auth);
meta-scheduler constraint language; k8s/`CLUSTER_VIEW` changes beyond conforming to the worker-less
store. There is **no** per-backend DB-file split and **no** `RemoteBackendWorkerStore` under Model D.

**Open (Track 1):**
1. **Worker-detail staleness** — served from the last-tick `WorkerView` + a controller-owned attempt
   read. Acceptable, or does the detail page need a live fetch?

**Open (Track 2 — federation, when designed):**
2. **On-demand RPC to a federated job: proxy or redirect?** Proxy through the federating Iris
   (uniform auth, one endpoint, on the data path) or redirect the client to the peer (off the hot
   path, client needs reach + auth). Shapes the federation router.
3. **Budget admission across peers** — grant-per-root with a reservation vs report-and-throttle;
   overspend bound while a spend report is in flight.
4. **Federated-job handle durability + re-attach** — the handle survives a federating-Iris restart
   and re-syncs to still-running peers (fields in the `delegation-model` artifact).
