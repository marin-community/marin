# Iris TaskBackend contract (archived design record)

Implemented design record for the TaskBackend control-plane contract
(marin#6178), landed as one PR with per-stage commits T1–T6 on
`weaver/iris-multi-backend-support`. This distills the design narrative in
`docs/plans/iris-taskprovider-contract.md`; the code is the source of truth.

Ground truth:

- Contract: `lib/iris/src/iris/cluster/controller/backend.py`
- IRIS_CONTROLLER backend: `lib/iris/src/iris/cluster/backends/rpc/backend.py`
- TASK_BACKEND backend: `lib/iris/src/iris/cluster/backends/k8s/tasks.py`
- Dispatch loops: `lib/iris/src/iris/cluster/controller/controller.py`
- Apply paths: `ops/worker.py` (`apply_reconcile`), `ops/task.py`
  (`apply_dispatch_updates`)
- Autoscaler DB-less persistence:
  `controller/autoscaler/persistence.py` (`persist_autoscaler_state`),
  `controller/autoscaler/state.py` (`AutoscalerState`, `persistable_state()`)

## Problem

The controller hard-coded two execution models and branched between them with
`isinstance`:

- The worker-daemon model (GCP/TPU, CoreWeave bare-metal, manual, local): the
  Iris scheduler assigns task→worker and the controller fans a per-worker
  Reconcile RPC to worker daemons.
- The Kubernetes model: no worker daemon — the controller launches one Pod per
  task attempt directly via `kubectl`, and Kueue / the cluster autoscaler own
  placement and capacity.

The K8s provider bypassed the existing `TaskProvider` Protocol entirely, and
`controller.py` carried a `TaskProvider | K8sTaskProvider` union with
`isinstance` branches to pick behavior. Scheduling and autoscaling lived inside
the controller itself (DB-writing `Scheduler` and `Autoscaler`), so there was no
clean, DB-less seam at which a third execution model (a Slurm backend) could be
added without extending the `isinstance` ladder and the controller's DB
coupling. The word "provider" meant three different things (machine lifecycle,
task execution, the UI's provider-kind), which made the boundary hard to reason
about.

## The contract

`controller/backend.py` defines `TaskBackend`, the single Protocol every
execution backend implements. The controller owns the database and the loop
cadences; the backend owns the backend-specific logic and I/O. The interface is
**plain data in, plain data out — backends never read or write the controller
DB.**

Two declared capabilities select the controller's behavior, and the controller
branches on these, never on the concrete type:

- `placement: PlacementOwner` — `IRIS_CONTROLLER` (the Iris `Scheduler` assigns task→worker, then
  the backend fans the per-worker reconcile RPC to daemons) vs `TASK_BACKEND` (the
  backend places tasks itself: Kueue today, slurmctld later).
- `manages_capacity: bool` — `True` when the backend provisions its own nodes
  (k8s cluster autoscaler), so Iris runs no autoscaler loop for it; `False` when
  the Iris `Autoscaler` provisions capacity.

### Method set, by concern

- **Reconcile.** `reconcile(BackendReconcileInput) -> BackendReconcileResult` —
  converge the backend toward the desired state and report observed state.
- **Schedule.** `schedule(ScheduleInput) -> ScheduleResult` — placement decisions
  from a DB-less snapshot. IRIS_CONTROLLER runs the full Iris pipeline (gates → order →
  reservation taints → preference pass → `find_assignments` → preemption, via
  `run_scheduling_decision`); TASK_BACKEND returns an empty `ScheduleResult`.
- **Capacity.** `manage_capacity(CapacityInput) -> CapacityResult` and
  `on_workers_failed(worker_ids) -> WorkersFailedResult`. IRIS_CONTROLLER drives the in-memory
  `Autoscaler` and returns its `AutoscalerState` for the controller to persist;
  TASK_BACKEND returns empty results. `attach_autoscaler(autoscaler)` is called once
  after construction on `manages_capacity=False` backends only (it raises on
  `K8sTaskProvider`). `capacity() -> ClusterCapacity | None` reports aggregate
  capacity (k8s computes it from node allocatable minus pod requests; rpc returns
  None).
- **On-demand ops** (request/response, not loop-driven), each addressed by a
  `TaskTarget`: `get_process_status`, `profile_task`, `exec_in_container`. Each
  raises `ProviderUnsupportedError` where N/A (e.g. K8s has no per-process
  status). IRIS_CONTROLLER routes by worker address; TASK_BACKEND routes by task_id/attempt_id.
- **Lifecycle / wiring.** `ping_workers` (IRIS_CONTROLLER liveness probe; K8s no-op),
  `on_worker_failed` (evict cached connection state), `set_log_sink` (inject
  finelog handles for daemonless backends that write rows themselves), `close`.

The result/input dataclasses (`BackendReconcileInput/Result`, `ScheduleInput/
Result`, `CapacityInput/Result`, `WorkersFailedResult`, `BackendDescriptor`) are
all frozen plain-data carriers. Each backend reads only the fields matching its
placement; the controller leaves the others empty.

### The DB-less rule

Every method takes a snapshot the controller assembled from its own DB reads and
returns plain data the controller commits. The backend does backend I/O
(`kubectl apply`, worker-daemon RPC fan-out) but holds no DB handle. This is what
keeps the controller the sole DB owner and lets a new backend be a pure function
of its snapshot.

## The two reconcile apply paths

The reconcile results are *not* applied through one merged path — they emit
different effects, and the controller selects the path on `placement`:

- **IRIS_CONTROLLER** → `ops.worker.apply_reconcile`. Input carries pre-built, worker-bound
  `plans` (the scheduler already chose the worker) plus `worker_addresses`. The
  result carries raw `worker_results` (`ReconcileResult` per worker); the
  controller resolves attempt UIDs and interprets worker loss against its own DB
  snapshot at apply time. This path emits **worker heartbeats** and runs the
  `WORKER_RECONCILE` transition source.
- **TASK_BACKEND** → `ops.task.apply_dispatch_updates`. Input carries the
  desired `tasks_to_run` (no worker_id — the backend chooses the node) plus the
  `running_tasks` snapshot. The backend converges its own resources (applies new
  pods, deletes strays, polls running pods) and returns pre-computed
  `TaskUpdate`s. This path runs the `DISPATCH` transition source and emits
  **no heartbeats** (there is no worker daemon).

Merging them would conflate the heartbeat/no-heartbeat and worker-loss
interpretation differences, so they stay distinct and capability-selected.

## The controller as dispatcher

Each control activity is a controller loop on its own cadence that reads a
snapshot, calls one backend method, and commits/persists the result. The
controller keeps the threads (it owns timing); the backend owns the logic.

- **Scheduling** (`_run_scheduling`): refresh reservation claims, build the
  scheduling context + running-task band/value for preemption, hand a
  `ScheduleInput` to `backend.schedule`, then commit assignments (`ops.task`),
  preemptions and unschedulable marks. TASK_BACKEND returns empty, so the loop is a
  no-op there.
- **Reconcile** (`_reconcile_tick` for IRIS_CONTROLLER, `_sync_dispatch` for
  TASK_BACKEND): snapshot desired/observed, call `backend.reconcile`, apply through the
  placement-appropriate path above. The execution-timeout deadline scan rides the
  IRIS_CONTROLLER tick.
- **Capacity** (`_run_autoscaler_once`): read the worker status map + demand
  entries, call `backend.manage_capacity`, then `persist_autoscaler_state`. Gated
  off entirely when `manages_capacity` is True.
- **Worker failure** (`_terminate_workers`): fail the workers, call
  `backend.on_workers_failed` to tear down their slices and learn the sibling
  worker ids to fail too, then `persist_autoscaler_state` with the post-teardown
  state.
- **Ping** (`_run_ping_loop`): `backend.ping_workers` for liveness; a K8s no-op.

The fast/slow split is load-bearing: fast backend I/O (kubectl apply, worker RPC
with hard timeouts) runs in-tick; slow infra I/O (VM provisioning, tens of
seconds) stays on the autoscaler cadence and its effects are observed on a later
tick. That is why the loops keep separate cadences rather than collapsing into
one.

### Making the autoscaler DB-less

`ScalingGroup`/`Autoscaler` now hold their slice/scaling-group state in memory as
authoritative and expose it via `persistable_state() -> AutoscalerState`. After
each capacity cycle (or worker-failure teardown), the controller calls
`persist_autoscaler_state(db, state)`, which **wholesale-syncs** the `slices` and
`scaling_groups` tables to match the returned state (upsert present rows, delete
absent ones). The backend owns the autoscaling logic and the cloud I/O without
ever owning a DB handle.

## What moved where (T1–T6)

| Stage | Change | Lands at |
|---|---|---|
| T1 | `TaskBackend` contract + neutral reconcile types | `controller/backend.py` |
| T2 | Adopt reconcile across backends + controller + service; K8s `sync`→`reconcile`; `WorkerProvider`→`RpcTaskBackend`; drive both via `reconcile`, no `isinstance`; delete `controller/provider.py` | `backends/rpc/backend.py`, `backends/k8s/tasks.py`, `controller.py`, `service.py` |
| T3 | Move the scheduling decision into the backend (`backend.schedule`); `RpcTaskBackend` owns the stateless `Scheduler` | `controller/backend.py` (`run_scheduling_decision`), `backends/rpc/backend.py` |
| T4 | Move autoscaling into the backend, DB-less (`manage_capacity` + `on_workers_failed`; `persistable_state()`; `persist_autoscaler_state`); `attach_autoscaler` wiring | `backends/rpc/backend.py`, `autoscaler/{state,persistence}.py`, `controller.py`, `main.py` |
| T5 | Group the shared scheduling layer | `controller/scheduling/{scheduler.py,policy.py}` (was `controller/scheduler.py`, `controller/scheduling_policy.py`) |
| T6 | Capability-driven dashboard: `/auth/config` serves a `BackendDescriptor`; `App.vue` filters one tab list by capabilities | `controller/dashboard.py`, `dashboard/src/App.vue` |

The dashboard descriptor (`backend_descriptor(backend)`) derives capability
strings from the live backend: `workers` (IRIS_CONTROLLER placement → Workers/Fleet tab),
`autoscaler` (`manages_capacity` False → Autoscaler tab), `cluster` (TASK_BACKEND
placement → Cluster tab). `App.vue` shows a tab only when its required capability
is present, so the tab list is data-driven rather than keyed on a provider-kind
binary.

## Resolved decisions

- **Name: `TaskBackend`** for the new contract. "Provider" stays for the
  machine-lifecycle protocols (`ControllerProvider`, `WorkerInfraProvider`); the
  bare `TaskProvider` name is retired. (`K8sTaskProvider` keeps its historical
  class name but is a `TaskBackend`.)
- **One PR for all of Stage 1** (T1–T6), validated end-to-end; the commits are
  the spiral within it, each building and passing tests.
- **Contract lives in `controller/backend.py`.** The plan floated a neutral
  `cluster/backend_types.py`; the implementation kept the contract in the
  controller layer and lets `backends/` import it upward, because the contract
  (and the `Scheduler`/`Autoscaler`/reconcile types it names) is conceptually
  controller logic. This is the one accepted upward edge.
- **Dashboard descriptor via `/auth/config`**, not a new `GetBackendInfo` RPC
  (the frontend already fetches `/auth/config` on load).
- **Liveness ping thread kept**, gated by placement — least behavioral risk than
  folding the heartbeat into `reconcile`'s return for Stage 1.
- **Autoscaler made DB-less via wholesale state sync** rather than incremental
  deltas — in-memory state is authoritative and the controller mirrors it.

## Follow-ups

- **Slurm backend** (the motivating next step): `placement=TASK_BACKEND`,
  `manages_capacity=True`, `sbatch`/`squeue`/`sacct`, reusing this contract. Open
  question carried from the issue: run a worker daemon inside the allocation
  (reuse `RpcTaskBackend`) vs. direct sbatch launch (closer to k8s).
- **Multi-backend**: `Controller` accepting `list[TaskBackend]` with a
  meta-scheduler routing pending tasks by constraint/selector, and capacity +
  reconcile fanning out per backend. Out of scope for this PR (single backend
  per controller).
- **Single-tick collapse** (optional): folding the per-activity loops into one
  driving tick. Deferred — separate cadences and slow provisioning I/O are
  load-bearing today.
- **Pre-existing frontend `build:check` break**: `npm run build:check` is red on
  `main` independently of this PR (a TypeScript `^6.0.3` pin from #6173); the
  `App.vue` capability change is itself type-clean.
