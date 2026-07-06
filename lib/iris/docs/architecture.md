# Source layout: `lib/iris/src/iris`

How the Iris source tree is organized and the rule that keeps it navigable.
For the controller's reconcile kernel specifically, see
[`reconcile_rpc.md`](reconcile_rpc.md) and the design notes in
`.agents/projects/reconcile-final-design.md`.

## The one rule: five layers, imports go down

Iris is organized as five layers. **A module may only import from layers below
it.** Reading top to bottom answers a chain of questions:

```
┌─ ENTRY POINTS ──────────────────────────────────────────────────────┐
│ cli/              command line (job/cluster/build/auth/rpc)          │
│ client/           high-level user SDK: IrisClient, IrisContext        │
│ cluster/client/   low-level RPC client: RemoteClusterClient, bundle   │
└──────────────────────────────┬───────────────────────────────────────┘
┌─ CONTROLLER  (cluster/controller/) — the brain ──▼────────────────────┐
│  transport/loops  controller.py · service.py · dashboard.py · main.py │
│  imperative shell  ops/{job,task,worker} · reconcile/dispatch · pruner│
│  decision kernels  reconcile/ · scheduling/ · autoscaler/             │
│  state predicates  task_state.py · worker_health.py · audit.py        │
│  persistence spine schema→codec→db→reads/writes · projections/        │
└──────────────────────────────┬───────────────────────────────────────┘
┌─ EXECUTION SUBSTRATE  (cluster/) ────────────────▼────────────────────┐
│  worker/    the agent daemon that runs on each machine                │
│  runtime/   container execution (Docker / subprocess)                 │
│  platforms/ machine lifecycle: gcp · k8s · local · manual             │
│  backends/  TaskBackend implementations: rpc · k8s                    │
└──────────────────────────────┬───────────────────────────────────────┘
┌─ CLUSTER VOCABULARY  (cluster/ top-level) ───────▼────────────────────┐
│  types · constraints · config · config_serde · endpoints · bundle …   │
└──────────────────────────────┬───────────────────────────────────────┘
┌─ FOUNDATION ─────────────────────────────────────▼────────────────────┐
│  rpc/ (proto wire types + auth/errors/interceptors/stats/compression)  │
│  actor/ (generic RPC actor framework)  runtime/ (JAX init)             │
│  top-level: chaos · managed_thread · time_proto · env_resources · …    │
└────────────────────────────────────────────────────────────────────────┘
```

| Layer | Owns | The question it answers |
|---|---|---|
| **Foundation** | `rpc/`, `actor/`, top-level utils | What vocabulary does everything speak? (protos, RPC middleware, threads, time) |
| **Cluster vocabulary** | `cluster/types,constraints,config,bundle,endpoints` | What *is* a job / constraint / resource? |
| **Execution substrate** | `cluster/platforms,backends,runtime,worker` | How do we get a machine, and run a task on it? |
| **Controller** | `cluster/controller/**` | What is the desired state, and how do we drive toward it? |
| **Entry points** | `cli/`, `client/`, `cluster/client/` | How does a human/program submit and observe? |

## What goes where

**Foundation.** `rpc/` holds protobuf-generated wire types plus hand-written
middleware (auth, errors, interceptors, stats, compression, JSON codecs) — the
language every process speaks. `actor/` is a transport-agnostic RPC actor
framework (client/server/pool/resolver). The loose top-level files
(`chaos`, `managed_thread`, `time_proto`, `env_resources`, …) are
process-level utilities with no cluster knowledge.

**Cluster vocabulary** (`cluster/` top level). Domain types (`types.py`),
the placement-constraint system (`constraints.py`), config loading/validation
+ provider/autoscaler factories (`config.py`), the pure proto→dict serializer
(`config_serde.py`, a leaf both `config` and provider bootstrap depend on),
content-addressed bundles (`bundle.py`), endpoint URI resolution, and small
shared concerns (`redaction`, `service_mode`, `log_keys`,
`process_status`, `dashboard_common`).

**Execution substrate.** Two distinct abstractions, one package each:

- *Machine lifecycle* lives in `platforms/`: two Protocols
  (`ControllerProvider`, `WorkerInfraProvider` in `platforms/protocols.py`),
  the shared handle/status/error types (`platforms/types.py`), and four
  implementations (`gcp`, `k8s`, `local`, `manual`); `vm_lifecycle.py`
  (controller VM start/stop/restart) lives here because it is provider code.
- *The task control-plane contract* (`TaskBackend`, defined in
  `controller/backend.py`) lives in `backends/`: `backends/rpc/backend.py`
  (`RpcTaskBackend`) and `backends/k8s/tasks.py` (`K8sTaskProvider`) each
  implement it. This is a different axis from machine lifecycle — a
  `TaskBackend` drives task execution and capacity for one cluster, while the
  lifecycle Protocols get/stop machines.

`runtime/` abstracts *task execution* behind `ContainerRuntime` (Docker /
subprocess). `worker/` is the agent daemon that runs on each machine.

The `TaskBackend` contract type lives in the controller layer
(`controller/backend.py`), and the two implementations in `backends/` import it
upward — an intended exception to "imports go down" (see
[Known boundary debt](#known-boundary-debt)). It is the seam by which the
controller stays a thin, backend-agnostic dispatcher; see
[The TaskBackend contract](#the-taskbackend-contract).

**Controller** (`cluster/controller/`) — the coordination engine, itself
sub-layered:

| Sub-layer | Modules | Role |
|---|---|---|
| Persistence spine | `schema` → `codec` → `db` → `reads`/`writes` · `projections/` | State at rest. `reads`/`writes` are the **only** sanctioned query/mutation surface; `projections/` are write-through caches. |
| State predicates | `task_state` · `worker_health` · `audit` | What the rows *mean*. |
| Decision kernels | `reconcile/` (lifecycle) · `scheduling/scheduler.py` (matching) · `scheduling/policy.py` (preemption/gating) · `autoscaler/` (capacity) | Compute what *should* change. `reconcile/` and `scheduling/` are parameterized with no live I/O; `autoscaler/` also actuates its plan through `WorkerInfraProvider` (live cloud create/describe calls and worker health probes). |
| Imperative shell | `ops/{job,task,worker}` · `reconcile/dispatch` · `pruner` | Load a snapshot, call a kernel, apply effects. |
| Transport / loops | `controller.py` (loops) · `service.py` (RPC) · `dashboard.py` · `main.py` | Drive it / expose it. |

The `reconcile/` package is the lifecycle kernel: leaves
(`snapshot`/`policy`/`effects`) → `working_state` → aggregate primitives
(`task`/`job`/`worker`, no cross-imports) → `peers` (the lone cross-aggregate
edge) → `batches` (orchestrator) → `loader` (I/O) → `ops/` shell. `reads`/`writes`
are the canonical data layer; **one-off queries may stay in `service.py`** —
`reads.py` is reserved for load-bearing, multiply-used queries.

### The TaskBackend contract

`controller/backend.py` defines `TaskBackend`: the single uniform Protocol that
drives task execution and capacity for one cluster. The controller owns the
database and the loop cadences; a backend takes a plain-data snapshot in and
returns a plain-data, method-specific result out. A backend **authors a
projection** — task-state `effects` plus a `status` snapshot — that the
controller commits/stores; the controller never learns which worker ran a task.
Every backend implements the same phase methods (plus on-demand
`get_process_status`/`profile_task`/`exec_in_container`):

- `schedule(ScheduleRequest) -> ScheduleResult` — a placement decision.
- `reconcile(ReconcileRequest) -> ReconcileResult` — converge the substrate and
  author task-state `effects` for the controller to commit. A worker-daemon
  backend also folds the liveness it observed into its **own**
  `WorkerHealthTracker` and stashes the workers its fold reaped.
- `run_teardown()` — fail the reaped workers, tear down their slices and healthy
  siblings, and forget them from the backend's own tracker. Called once per
  backend after the tick's effects are committed; no worker identity is passed in.
- `autoscale(AutoscaleRequest) -> AutoscaleResult` — provision capacity.
- `status() -> BackendStatus` — author this backend's dashboard status.

Each method returns its own frozen result type (`ScheduleResult` /
`ReconcileResult` / `AutoscaleResult`). `reconcile` returns `effects` only —
uniform across backend kinds — so the controller commits the projection without
branching on the concrete backend type; there are no `isinstance` branches. How
authority is split between the controller and its backends (and how a remote
backend becomes a controller of its own) is described in
[`multi_backend.md`](multi_backend.md).

A backend declares `capabilities: frozenset[BackendCapability]`, metadata the
dashboard and on-demand RPC routing key on. The controller calls every phase
uniformly regardless, with one per-tick exception: `CLUSTER_VIEW` makes the
controller drain the dispatch queue (a DB write it owns) into that backend's
reconcile snapshot.

| Capability | Meaning |
|---|---|
| `WORKER_DAEMON` (`"workers"`) | Iris tracks worker daemons; the backend fans the per-worker Reconcile RPC out (`RpcTaskBackend`, `backends/rpc/backend.py` — GCP/TPU, CoreWeave bare-metal, manual, local). |
| `IRIS_AUTOSCALER` (`"autoscaler"`) | the Iris `Autoscaler` provisions capacity for this backend. |
| `CLUSTER_VIEW` (`"cluster"`) | the backend places tasks on its own cluster (`K8sTaskProvider`, `backends/k8s/tasks.py` — Kueue schedules and provisions; `schedule`/`autoscale` are effectively no-ops). |

Worker health is **authored by each worker-daemon backend**, not the controller.
A `WORKER_DAEMON` backend constructs and owns its own `WorkerHealthTracker`,
folds the liveness it observed during `reconcile` (REACHED / UNREACHABLE, plus
BUILD_FAILED synthesized from its effects), and reaps over-threshold workers in
its own `run_teardown`. There is no ping loop and no separate liveness channel —
the reconcile RPC outcome is the only liveness signal. The controller reaches
per-worker liveness only through the backends (`liveness_for_worker`,
`all_liveness`), and worker registration routes to the owning backend's tracker
by scale group. Cluster-view (e.g. Kubernetes) backends have **no Iris workers**:
they hold no tracker, and pod status flows back as task `effects`. A new backend
(e.g. Slurm) slots in by implementing the phase methods and declaring its
capabilities.

**Entry points.** `cluster/client/` is the low-level RPC client
(`RemoteClusterClient`); `client/` is the high-level user SDK (`IrisClient`,
`IrisContext`); `cli/` is the command line. Nothing imports *into* these.

## Known boundary debt

Honest exceptions to the layering, as of this writing:

- **`backends` → `controller/backend.py` upward import.** Both `TaskBackend`
  implementations (`backends/rpc/backend.py`, `backends/k8s/tasks.py`) import
  the contract type — and the `Scheduler`/`Autoscaler`/reconcile types it
  references — up from the controller layer. This is a deliberate, narrowed edge:
  the controller depends only on the `TaskBackend` Protocol and dispatches on
  result-field content (no `isinstance`), so the old runtime
  `TaskProvider | K8sTaskProvider` union and its `isinstance` ladder are gone
  (the dead `controller/provider.py` was deleted). The residual coupling is now a
  static import of one contract type rather than a behavioral branch. Fully
  removing it would mean hoisting the contract (and the scheduler/autoscaler it
  names) into `cluster/`; deferred, because the contract is conceptually
  controller logic.
- **Device introspection is split** across `types.py` (counts/devices) and
  `constraints.py` (type/variant). Consolidation is blocked because the type
  reader returns `constraints.DeviceType`, which is pinned to `constraints.py`
  by `PlacementRequirements`; moving it would only invert the coupling.
- **`service.py` is large** (~2.5k lines) but deliberately wide-and-flat
  (RPC dispatch + one-off queries). Only proto *encoding* belongs in `codec.py`.

Layering is a convention maintained by review, not a machine-checked invariant.
