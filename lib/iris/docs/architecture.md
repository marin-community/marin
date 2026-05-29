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
│  imperative shell  ops/{job,task,worker} · direct_provider · pruner   │
│  decision kernels  reconcile/ · scheduler.py · scheduling_policy.py · │
│                    autoscaler/                                        │
│  state predicates  task_state.py · worker_health.py · audit.py        │
│  persistence spine schema→codec→db→reads/writes · projections/        │
└──────────────────────────────┬───────────────────────────────────────┘
┌─ EXECUTION SUBSTRATE  (cluster/) ────────────────▼────────────────────┐
│  worker/    the agent daemon that runs on each machine                │
│  runtime/   container execution (Docker / subprocess)                 │
│  providers/ machine lifecycle: gcp · k8s · local · manual             │
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
| **Execution substrate** | `cluster/providers,runtime,worker` | How do we get a machine, and run a task on it? |
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
shared concerns (`redaction`, `service_mode`, `log_keys`, `token_store`,
`process_status`, `dashboard_common`).

**Execution substrate.** `providers/` abstracts *machine lifecycle* behind two
Protocols (`ControllerProvider`, `WorkerInfraProvider`) with four backends
(`gcp`, `k8s`, `local`, `manual`); `vm_lifecycle.py` (controller VM
start/stop/restart) lives here because it is provider code. `runtime/`
abstracts *task execution* behind `ContainerRuntime` (Docker / subprocess).
`worker/` is the agent daemon that runs on each machine.

**Controller** (`cluster/controller/`) — the coordination engine, itself
sub-layered:

| Sub-layer | Modules | Role |
|---|---|---|
| Persistence spine | `schema` → `codec` → `db` → `reads`/`writes` · `projections/` | State at rest. `reads`/`writes` are the **only** sanctioned query/mutation surface; `projections/` are write-through caches. |
| State predicates | `task_state` · `worker_health` · `audit` | What the rows *mean*. |
| Decision kernels | `reconcile/` (lifecycle) · `scheduler.py` (matching) · `scheduling_policy.py` (preemption/reservation/gating) · `autoscaler/` (capacity) | Compute what *should* change. Parameterized; no live I/O. |
| Imperative shell | `ops/{job,task,worker}` · `direct_provider` · `pruner` | Load a snapshot, call a kernel, apply effects. |
| Transport / loops | `controller.py` (loops) · `service.py` (RPC) · `dashboard.py` · `main.py` | Drive it / expose it. |

The `reconcile/` package is the lifecycle kernel: leaves
(`snapshot`/`policy`/`effects`) → `working_state` → aggregate primitives
(`task`/`job`/`worker`, no cross-imports) → `peers` (the lone cross-aggregate
edge) → `batches` (orchestrator) → `loader` (I/O) → `ops/` shell. `reads`/`writes`
are the canonical data layer; **one-off queries may stay in `service.py`** —
`reads.py` is reserved for load-bearing, multiply-used queries.

**Entry points.** `cluster/client/` is the low-level RPC client
(`RemoteClusterClient`); `client/` is the high-level user SDK (`IrisClient`,
`IrisContext`); `cli/` is the command line. Nothing imports *into* these.

## Known boundary debt

Honest exceptions to the layering, as of this writing:

- **`controller` ↔ `providers/k8s` cycle.** `providers/k8s/tasks.py` imports up
  into the controller, and `controller.py` carries a `TaskProvider |
  K8sTaskProvider` union with `isinstance` branches. The `TaskProvider` Protocol
  exists to prevent this but K8s bypasses it. (Fix: make `K8sTaskProvider`
  satisfy the Protocol — deferred; the same move `service.py`'s
  `ControllerProtocol` already demonstrates.)
- **Device introspection is split** across `types.py` (counts/devices) and
  `constraints.py` (type/variant). Consolidation is blocked because the type
  reader returns `constraints.DeviceType`, which is pinned to `constraints.py`
  by `PlacementRequirements`; moving it would only invert the coupling.
- **`service.py` is large** (~2.5k lines) but deliberately wide-and-flat
  (RPC dispatch + one-off queries). Only proto *encoding* belongs in `codec.py`.

Layering is a convention maintained by review, not a machine-checked invariant.
