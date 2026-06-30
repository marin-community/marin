# IrisDaemon — spec

Concrete contracts for the `IrisDaemon` primitive. Pins the operator-facing config, the
controller reconciler surface, the wire/status RPCs, discovery convention, and errors. Route A
(reconciler over pinned tasks) is the committed v1; Route B (bootstrap sidecar) is out of scope
here (see §7).

## 1. Config surface (`DaemonSpec`)

New dataclass in `lib/iris/src/iris/cluster/config.py`, declared as a top-level `daemons:` list
in the cluster YAML (sibling to `scale_groups:`).

```python
@dataclass(frozen=True)
class DaemonSelector:
    """Which workers a daemon runs on. All predicates AND together; empty matches none
    (a daemon must target at least a scale group — fail-fast, no accidental cluster-wide)."""
    scale_group: str                          # required: the operator-facing pool unit
    constraints: tuple[Constraint, ...] = ()  # optional narrowing, reuses cluster/constraints.py

@dataclass(frozen=True)
class DaemonSpec:
    name: str                                 # cluster-unique; used in endpoint + task naming
    selector: DaemonSelector
    image: str                                # container image for the daemon process
    command: tuple[str, ...]                  # argv (no implicit shell)
    resources: ResourceSpec                   # cluster/types.py; see §4 for the recommended shape
    priority_band: PriorityBand = PriorityBand.BATCH
    endpoint_name: str | None = None          # if set, instances register <endpoint_name>/<worker_id>
    environment: Mapping[str, str] = field(default_factory=dict)
    ports: tuple[PortSpec, ...] = ()
    max_restarts_per_worker: int = 20         # per-worker crash ceiling before degraded (§6)
    reserve_memory: bool = True               # §4 accounting policy, explicit per daemon
```

YAML:

```yaml
daemons:
  - name: smallquery-worker
    selector:
      scale_group: cpu-spot
      constraints:
        - { key: region, op: EQ, value: us-east5 }
    image: us-docker.pkg.dev/.../smallquery-worker:<digest>
    command: ["python", "-m", "smallquery.worker", "serve"]
    resources: { cpu_millicores: 0, memory_bytes: 32Gi, disk_bytes: 5Gi }
    priority_band: BATCH
    endpoint_name: smallquery/worker
    reserve_memory: true
```

`DaemonSpec.to_proto()` / `from_proto()` mirror `ResourceSpec.to_proto()`
(`cluster/types.py:574-605`).

## 2. Reconciler (`DaemonReconciler`)

New controller component, `lib/iris/src/iris/controller/daemon_reconciler.py`. Pure-ish: takes a
membership snapshot + desired specs, emits create/cancel intents.

```python
class DaemonReconciler:
    def __init__(self, specs: Sequence[DaemonSpec], task_backend: TaskBackend,
                 endpoints: EndpointService) -> None: ...

    def reconcile(self, members: Sequence[WorkerSnapshot]) -> ReconcileResult:
        """Maintain the invariant: for every worker in `members` matching a spec's selector,
        exactly one running daemon task of that spec is pinned to that worker; and **no daemon
        task pinned to a worker absent from `members`, nor duplicated on a worker**.

        Idempotent. Called on every membership change (worker join/leave) and on spec change. It:
        - **creates** a task for a matching worker that has none;
        - **cancels (GC)** any daemon task pinned to a departed/non-matching worker, or any
          duplicate beyond the first on a worker — WORKER_FAILED is the common case, but a task
          stuck PENDING against a worker-id that will never return must be **explicitly cancelled**,
          not left to retry forever;
        - **re-creates** an instance that died while its worker is still live, up to
          `max_restarts_per_worker`, then marks it `DAEMON_DEGRADED` and stops.

        **State durability**: the `(spec.name, worker.id) → task-attempt` mapping is *derived*, not
        stored — on controller restart the reconciler lists daemon tasks from the task backend by a
        `daemon=<spec.name>` label and rebuilds the mapping, so it survives restarts without its own
        persistence.
        """
```

`ReconcileResult` = `{created: list[DaemonInstanceId], cancelled: list[DaemonInstanceId],
degraded: list[DaemonInstanceId]}`.

**Placement contract**: each instance is submitted as a normal task labelled `daemon=<spec.name>`
with a HARD constraint `Constraint(key=WORKER_ID_ATTR, op=EQ, value=worker.id)` (worker-id attribute
per `cluster/types.py:58-72`) and `priority_band` from the spec. **The reconciler is the single
restart authority**: daemon tasks set `max_retries_failure=0` and `max_retries_preemption=0` so
Iris's own task-retry never races the reconciler — a dead instance terminates, and only the
reconciler recreates it (incrementing the per-`(spec,worker)` restart count). This removes the
ambiguity of task-retry, preemption-retry, and reconciler-recreate all minting attempts.
`DaemonInstanceId = (spec_name: str, worker_id: str)`.

## 3. Status RPCs (`DaemonService`)

New proto `lib/iris/src/iris/rpc/daemon.proto`, served by the controller. Specs come from cluster
config (not RPC-created in v1), so the service is **read/status only**.

```proto
service DaemonService {
  rpc ListDaemons(ListDaemonsRequest) returns (ListDaemonsResponse);
  rpc GetDaemon(GetDaemonRequest) returns (GetDaemonResponse);
}

message DaemonSpecProto {
  string name = 1;
  DaemonSelectorProto selector = 2;
  string image = 3;
  repeated string command = 4;
  ResourceSpecProto resources = 5;            // reuse job.proto ResourceSpecProto
  PriorityBand priority_band = 6;             // reuse job.proto PriorityBand
  optional string endpoint_name = 7;
  map<string, string> environment = 8;
  int32 max_restarts_per_worker = 9;
  bool reserve_memory = 10;
  repeated PortSpecProto ports = 11;          // reuse job.proto PortSpecProto (matches DaemonSpec.ports)
}

message DaemonSelectorProto { string scale_group = 1; repeated ConstraintProto constraints = 2; }

message DaemonInstanceProto {
  string spec_name = 1;
  string worker_id = 2;
  string task_id = 3;
  string attempt_id = 4;
  DaemonInstanceState state = 5;              // PENDING / RUNNING / WORKER_FAILED / DEGRADED
  optional string endpoint_address = 6;       // resolved from EndpointService, if registered
  int32 restarts = 7;
}

message ListDaemonsResponse { repeated DaemonStatusProto daemons = 1; }
message DaemonStatusProto {
  DaemonSpecProto spec = 1;
  int32 desired = 2;                          // matching live workers
  int32 running = 3;
  repeated DaemonInstanceProto instances = 4;
}
```

`DaemonInstanceState` enum: `DAEMON_PENDING=0`, `DAEMON_RUNNING=1`, `DAEMON_WORKER_FAILED=2`,
`DAEMON_DEGRADED=3`.

## 4. Resource accounting

The recommended opportunistic shape (smallquery): `cpu_millicores=0` (bypasses CPU fit,
`scheduler.py:271-276`), real `memory_bytes` (hard cgroup cap, `runtime/docker.py:684-686`),
`priority_band=BATCH`. `reserve_memory` selects the policy:

- `reserve_memory=True` (default): the instance's `memory_bytes` is added to the worker's
  `committed_memory_bytes` (`scheduler.py:240-248`), so co-located jobs see less free RAM —
  daemon RAM is guaranteed.
- `reserve_memory=False`: instance is placed without crediting committed memory (best-effort
  overflow); still hard-capped by its own cgroup, but co-located jobs are scheduled as if that RAM
  were free — so under load the **node** can exhaust memory and the kernel OOM-kills some process
  (a cgroup cap bounds one container, not node-level pressure). **Only safe for a small daemon
  footprint**; default is `True`.

This is the one global-impact knob; it is explicit per `DaemonSpec`, never implicit. **The default
policy — and whether `reserve_memory=False` should exist at all — is the load-bearing decision
pending Iris-owner sign-off** (`design.md` Open Questions); this spec pins the *mechanism*, not a
settled policy. (smallquery itself commits to `reserve_memory=True` — its workers take a real
bounded RAM reservation, not free overflow.)

## 5. Discovery convention

If `endpoint_name` is set, the daemon **self-registers** `<endpoint_name>/<worker_id>` via
`EndpointClient.register(...)` (`client/endpoint_client.py:83-154`). Registration is **not**
automatic from config alone — Iris injects the instance identity into the container env
(`IRIS_WORKER_ID`, the controller endpoint address, the task-attempt for the lease), and the daemon
process calls `EndpointClient` itself (the lease auto-renews and unregisters on exit). Consumers
(e.g. the smallquery coordinator) call `list_endpoints(prefix=endpoint_name)` to get the live fleet;
dead instances auto-expire on task termination (`controller/endpoint_service.py:90-94`). No
daemon-specific discovery API is added — EndpointService is the contract.

## 6. Errors / degraded handling

- A daemon container that crashes is restarted (task restart) up to `max_restarts_per_worker`
  per `(spec, worker)`; beyond that the instance is marked `DAEMON_DEGRADED` and the reconciler
  stops recreating it on that worker until the worker is replaced or the spec changes. Modeled on
  zephyr's `MAX_SHARD_INFRA_FAILURES` ceiling (`lib/zephyr/src/zephyr/execution.py:73-83`).
- `DaemonSpec` referencing an unknown `scale_group` → config validation error at load
  (fail-fast), not a silent no-op.
- Empty/absent `scale_group` → validation error (no accidental cluster-wide daemon).
- Worker preemption is **not** an error: the instance goes `DAEMON_WORKER_FAILED`, the
  reconciler drops it, and a replacement worker (autoscaler) gets a fresh instance.

## 7. Out of scope (v1)

- **Route B bootstrap sidecar** (`WorkerConfig.sidecar_containers` + GCP bootstrap loop, k8s pod
  sidecar). v1 is Route A only; revisit if a consumer needs to escape the task cgroup.
- **Dynamic RPC-created daemons** (`CreateDaemon`/`DeleteDaemon`). v1 daemons are declared in
  cluster config; the service is read-only.
- **Cross-scale-group / cluster-wide selectors.** v1 selector requires exactly one scale group.
- **Rolling updates** on image/spec change beyond create/cancel (no surge/maxUnavailable
  semantics in v1; a spec change recreates instances).
- **Per-daemon autoscaling** (scaling the scale group based on daemon load) — orthogonal.

## 8. File map

| Piece | Path |
|---|---|
| `DaemonSpec` / `DaemonSelector` config | `lib/iris/src/iris/cluster/config.py` |
| `DaemonReconciler` | `lib/iris/src/iris/controller/daemon_reconciler.py` |
| Status proto | `lib/iris/src/iris/rpc/daemon.proto` |
| `DaemonService` handler | `lib/iris/src/iris/controller/daemon_service.py` |
| CLI `iris daemon list` / `iris daemon status <name>` | `lib/iris/src/iris/cli/daemon.py` |
