# Source layout: `lib/iris/src/iris`

How the Iris source tree is organized and the rule that keeps it navigable.
For the controller's reconcile kernel specifically, see
[`reconcile_rpc.md`](reconcile_rpc.md).

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
│  request operations  jobs · tasks · attempts · workers · endpoints    │
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
subprocess). `worker/` is the task agent daemon. The Iris node-agent entrypoint
runs beside it on GCP and as a Kubernetes DaemonSet, publishing physical host
and accelerator evidence independently of task and application telemetry.

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
| Persistence spine | `schema` → `codec` → `db` → `reads`/`writes` · `projections/` | State at rest. `reads`/`writes` hold shared queries and mutations; `projections/` are write-through caches. |
| State predicates | `task_state` · `worker_health` · `audit` | What the rows *mean*. |
| Decision kernels | `reconcile/` (lifecycle) · `scheduling/scheduler.py` (matching) · `scheduling/policy.py` (preemption/gating) · `autoscaler/` (capacity) | Compute what *should* change. `reconcile/` and `scheduling/` are parameterized with no live I/O; `autoscaler/` also actuates its plan through `WorkerInfraProvider` (live cloud create/describe calls and worker health probes). |
| Imperative shell | `ops/{job,task,worker}` · `reconcile/dispatch` · `pruner` | Load a snapshot, call a kernel, apply effects. |
| Request operations | `jobs` · `tasks` · `attempts` · `workers` · `endpoints` · `accounts` · `backend_status` · `federation_service` · `artifacts` · `diagnostics` · `checkpoint` | Authorize one request, call the data and runtime boundaries, and build its response. |
| Transport / loops | `controller.py` (loops) · `service.py` (thin Connect adapter) · `dashboard.py` · `main.py` | Drive the controller and expose its APIs. |

The `reconcile/` package is the lifecycle kernel: leaves
(`snapshot`/`policy`/`effects`) → `working_state` → aggregate primitives
(`task`/`job`/`worker`, no cross-imports) → `peers` (the lone cross-aggregate
edge) → `batches` (orchestrator) → `loader` (I/O) → `ops/` shell. `reads`/`writes`
are the canonical data layer. A query used by one request may stay in its
operation module. `reads.py` is reserved for load-bearing, multiply-used
queries.

`service.py` and `endpoint_service.py` wire each generated Connect method to one
operation function. Operation modules use frozen dependency records and small
runtime Protocols where live collaborators are needed. This keeps
authorization, exact target resolution, database work, and response
construction together under the owning controller concern. The operation
modules use the existing protobuf request and response messages; Iris does not
maintain parallel native copies solely to isolate the wire type.

### The TaskBackend contract

`controller/backend.py` defines `TaskBackend`: the Protocol that drives task
execution and capacity for one cluster. Composition registers exactly one
self-described backend with `Controller.register_backend` before `start()`.
`BackendDescriptor` declares its ID, kind, advertised attributes, and scale
groups. Federation connects controllers when work can execute on another
cluster.

The controller owns the database and loop cadences. It builds a complete,
single-use request for every phase. These include scheduling facts, exact
desired attempts and worker addresses, status/capacity facts, residual demand,
and recovery checkpoints. The controller also owns worker liveness and all Iris
persistence. A backend receives no database, transaction, transition reader, or
liveness tracker. It owns provider-specific decisions, observation, and
actuation; the controller folds returned observations into Iris state.
Both backend implementations expose the same phase methods (plus on-demand
`get_process_status`/`profile_task`/`exec_in_container`):

- `initialize(BackendRecoveryRequest) -> BackendRecoveryResult` — reconcile a
  persisted checkpoint with the provider before loops start.
- `schedule(ScheduleRequest) -> ScheduleResult` — a placement decision.
- `reconcile(ReconcileRequest) -> ReconcileObservation` — converge the external
  substrate and return exact task updates, optional worker reachability, and
  confirmed absence of terminal runtimes.
- `observe(BackendObservationRequest) -> BackendObservation` — publish provider
  status, capacity, and pending hints from controller-owned facts.
- `autoscale(AutoscaleRequest) -> AutoscaleResult` — provision capacity.
- `remove_capacity(RemoveCapacityRequest) -> RemoveCapacityResult` — remove
  controller-fenced capacity and report affected siblings.

Each phase returns a frozen result record. Every reconcile result has the same
shape: exact task-attempt updates, optional worker-health events, and exact
Attempt UIDs whose runtimes were observed stopped or absent. The controller
loads a fresh post-I/O snapshot, validates Attempt UIDs, applies one state-machine
path, accounts for worker health, and persists release observations with the
other effects. It only then requests physical capacity removal. It never asks a
backend to mutate Iris state.

`BackendDescriptor.kind` is presentation metadata. Its capabilities declare one
reconciliation mechanism: `WORKER_FLEET` or `DIRECT_DISPATCH`; `AUTOSCALER` is
optional for worker fleets. Kubernetes reconcile receives the controller-owned
dispatch drain. Worker reconcile receives a complete list pairing every desired
worker plan with its address. Dashboard capability strings are derived from the
descriptor.

The controller owns `WorkerHealthTracker`, applies REACHED / UNREACHABLE plus
kernel-derived BUILD_FAILED events, and collects workers that cross the reap
threshold. There is no ping loop: reconcile RPC outcomes are the only liveness
signal. Kubernetes reports no worker events; exact Pod observations are resolved
entirely in the controller.

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
Layering is a convention maintained by review, not a machine-checked invariant.
