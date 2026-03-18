# Iris Architecture Findings: Workers, Controller, Dashboard

Research for Provider abstraction design. All line refs are relative to
`lib/iris/src/iris/`.

---

## 1. The Worker Class

**File:** `cluster/worker/worker.py`

`Worker` is the unified daemon managing all components on a single machine.
It owns:

- **HTTP server** (uvicorn + Starlette): serves the `WorkerDashboard`
  (Connect RPC + web UI). Lines 209–225.
- **Controller client** (`ControllerServiceClientSync`): used exclusively in
  the `_run_lifecycle` thread to register and maintain liveness. Lines 228–236.
- **Task map** `dict[(task_id_wire, attempt_id), TaskAttempt]`: the only
  mutable execution state. Lines 167–168.
- **LogStore** (SQLite): captures worker-process logs accessible via FetchLogs
  RPC. Lines 172–176.
- **BundleStore**, **ContainerRuntime** (Docker or Kubernetes), **PortAllocator**.

### Worker Lifecycle

```
start()
  ├─ cleanup_all_iris_containers()     # orphan cleanup
  ├─ spawn HTTP server thread
  └─ spawn worker-lifecycle thread
       └─ _run_lifecycle()
            loop:
              _reset_worker_state()    # kill all containers, clear task map
              _register()              # POST RegisterRequest to controller, retry
              _serve()                 # block until heartbeat_deadline expires
```

The lifecycle thread retries registration indefinitely until accepted
(`line 291-328`). Once registered, it blocks in `_serve()` checking the
heartbeat deadline every second. If the deadline expires (default 600 s with
no heartbeat from the controller), it resets and re-registers.

### Worker Configuration — `WorkerConfig` dataclass (line 47)

```python
@dataclass
class WorkerConfig:
    host: str                       # bind address
    port: int                       # HTTP port
    cache_dir: Path | None          # required; bundles + work dirs
    port_range: tuple[int, int]     # task-port allocation range
    controller_address: str | None
    worker_id: str | None           # overrides auto-detection
    slice_id: str | None            # TPU slice membership
    worker_attributes: dict[str, str]
    default_task_env: dict[str, str]
    default_task_image: str | None
    resolve_image: Callable[[str], str]
    poll_interval: Duration         # default 5 s
    heartbeat_timeout: Duration     # default 600 s
    accelerator_type: int
    accelerator_variant: str
    gpu_count: int
    preemptible: bool
    storage_prefix: str
    auth_token: str
```

Loaded from `config_pb2.WorkerConfig` proto via `worker_config_from_proto()`
(line 71). Default values centralized in `cluster/config.py:DEFAULT_CONFIG`
(worker section, line 51).

### Task Dispatch to Worker

Tasks arrive via the `heartbeat` RPC (`cluster/worker/service.py:114`), which
calls `Worker.handle_heartbeat()` (line 538). Each heartbeat carries
`tasks_to_run`, `tasks_to_kill`, and `expected_tasks`. The worker:

1. Calls `submit_task()` for each `tasks_to_run` entry — creates a
   `TaskAttempt` and spawns a managed thread (`task-{id}`).
2. Kills tasks in `tasks_to_kill` asynchronously (daemon thread).
3. Reconciles `expected_tasks`: reports states or marks missing tasks as
   `TASK_STATE_WORKER_FAILED`.
4. Kills any task **not** in `expected_tasks ∪ tasks_to_run`.
5. Returns resource snapshot + health status.

The heartbeat is the **only** channel for task delivery; there is no separate
"submit" RPC from the controller to the worker.

### Worker RPC Protocol

Workers expose **Connect RPC** (`WorkerService`) over HTTP/1.1 via
`WorkerServiceWSGIApplication`. The controller uses
`WorkerServiceClientSync` (cached by `RpcWorkerStubFactory`). The proto is
at `rpc/cluster.proto`. Key RPCs:

| RPC | Caller | Purpose |
|-----|--------|---------|
| `Heartbeat` | controller | deliver tasks_to_run/kill, get state |
| `GetTaskStatus` | CLI/dashboard | task state query |
| `ListTasks` | dashboard | list all attempts |
| `KillTask` | service layer | on-demand kill |
| `FetchLogs` | CLI/dashboard | log streaming |
| `GetProcessStatus` | CLI/dashboard | worker process info |
| `ProfileTask` | controller loop | CPU profile capture |
| `HealthCheck` | bootstrap | liveness |

---

## 2. The Controller

**File:** `cluster/controller/controller.py`

### `ControllerConfig` dataclass (line 648)

```python
@dataclass
class ControllerConfig:
    host: str = "127.0.0.1"
    port: int = 0
    remote_state_dir: str = ""     # REQUIRED: GCS path for checkpoints
    scheduler_interval: Duration   # default 0.5 s
    heartbeat_interval: Duration   # default 5 s
    max_dispatch_parallelism: int  # default 32 (concurrent RPC workers)
    max_tasks_per_job_per_cycle: int  # default 4
    heartbeat_failure_threshold: int
    autoscaler_enabled: bool = False
    worker_access_address: str = ""
    checkpoint_interval: Duration | None = None
    profile_interval: Duration     # default 600 s
    profile_duration: int = 10
    profile_concurrency: int = 8
    local_state_dir: Path          # SQLite DB, logs, bundle cache
    auth_verifier: TokenVerifier | None
    auth_provider: str | None
    auth: ControllerAuth | None
```

### Controller Threads

The controller runs four background threads:

1. **scheduling-loop** (`_run_scheduling_loop`, line 945): runs at 0.5 s
   intervals (woken immediately on new submissions). Calls `_run_scheduling()`
   to assign pending tasks to healthy workers.
2. **heartbeat-loop** (`_run_heartbeat_loop`, line 981): runs at 5 s intervals.
   Calls `_heartbeat_all_workers()` to deliver dispatches and collect task state.
3. **autoscaler-loop** (`_run_autoscaler_loop`, line 961): runs at configured
   evaluation interval (default 10 s). Calls autoscaler `refresh()` + `update()`.
4. **profile-loop** (`_run_profile_loop`, line 997): runs at 600 s intervals.
   Captures CPU profiles for all running tasks.

### Controller Assumptions About Workers

- Workers are **pull-based via heartbeat**: controller pushes tasks on the
  next heartbeat after scheduling.
- Workers are identified by `WorkerId` (string) and `address` (host:port).
- Workers register themselves (`RegisterRequest`) — the controller does not
  initiate worker creation (that's the autoscaler/platform layer).
- Workers are assumed to be reachable via HTTP from the controller host.
- Each worker self-reports metadata (`WorkerMetadata` proto): CPU, memory,
  device (GPU/TPU), IP, preemptible flag, attributes dict.
- Worker health is tracked by consecutive heartbeat failures. After
  `heartbeat_failure_threshold` (default 3) consecutive failures, the worker
  is marked dead and its tasks are reset to PENDING.

### Controller `Worker` DB Model (line 667)

```python
@db_row_model
class Worker:
    worker_id: WorkerId
    address: str           # host:port — the ONLY addressing info controller stores
    metadata: WorkerMetadata   # proto: cpu_count, memory_bytes, device, ip_address, ...
    healthy: bool
    consecutive_failures: int
    last_heartbeat: Timestamp
    committed_cpu_millicores: int
    committed_mem: int
    committed_gpu: int
    committed_tpu: int
    active: bool
    attributes: dict[str, AttributeValue]  # WellKnownAttribute keys + extras
```

---

## 3. Autoscaler Logic

**Files:** `cluster/controller/autoscaler.py`, `cluster/controller/scaling_group.py`

### Overview

The autoscaler coordinates demand-driven scaling across **scale groups**. Each
scale group wraps a `Platform` and owns a set of `SliceHandle`s.

### `run_once()` flow (line 1213)

```
run_once(demand_entries, worker_status_map)
  ├─ refresh(worker_status_map)   # state-read phase
  │    for each non-READY slice:
  │      poll handle.describe() (cloud API call)
  │      transition BOOTSTRAPPING → READY/FAILED based on describe() result
  │      if slice is READY, register workers with autoscaler
  │    for each group:
  │      scale_down_if_idle(): terminate slices with no running tasks + past idle timeout
  └─ update(demand_entries)       # CPU phase
       evaluate(demand_entries):
         route demand entries to groups via constraint waterfall
         compute_required_slices per group (bin-packing + coscheduling)
         compare required vs current (ready + requesting)
         emit ScalingDecision(SCALE_UP) if gap > 0
       execute(decisions):
         for SCALE_UP: group.scale_up() → platform.create_slice()
```

### Scale-up Decision (autoscaler/evaluate)

For each demand entry (one per unabsorbed pending task):
1. Route to the first `ScalingGroup` that matches the task's constraints
   and is in `GroupAvailability.AVAILABLE/COOLDOWN/REQUESTING` state.
2. Compute `required_slices = ceil(vms_needed / workers_per_slice)` using
   first-fit-decreasing bin packing for CPU/memory, 1-VM-per-task for
   accelerator jobs.
3. If `required > (ready_slices + requesting_slices)`, emit a SCALE_UP.

### Scale-down Decision (ScalingGroup.scale_down_if_idle)

A slice is idle if all its tracked workers have no running tasks in the
`worker_status_map` for `>= scale_down_delay` (default 600 s). Idle slices
are terminated. Scale-down rate is limited (default 5/min).

### Backoff / Quota

- `QuotaExhaustedError` from `create_slice()` → group enters QUOTA_EXCEEDED.
- Other errors → exponential backoff in BACKOFF state.
- Both states cause demand to fall through to lower-priority groups.

---

## 4. Dashboard — Worker-Specific Fields and Endpoints

### Controller Dashboard (`cluster/controller/dashboard.py`)

Routes (line 91):
- `GET /worker/{worker_id:path}` → HTML shell for worker detail page
- All data fetched client-side via Connect RPC to `ControllerService`

The controller's `ControllerService` provides worker-related RPCs consumed by
the dashboard:

| RPC | Worker fields exposed |
|-----|----------------------|
| `ListWorkers` | worker_id, address, healthy, metadata (device/cpu/mem), attributes, last_heartbeat, committed resources |
| `GetWorker` | full `Worker` row |
| `GetWorkerLogs` | proxied via `/system/worker/<worker_id>` target to worker's FetchLogs |
| `GetProcessStatus` target `/system/worker/<id>` | worker process info (via proxy RPC) |
| `ProfileTask` target `/system/worker/<id>` | worker process profile (via proxy RPC) |

### Worker Dashboard (`cluster/worker/dashboard.py`)

Routes (line 43):
- `GET /` → HTML shell (worker overview)
- `GET /task/{task_id:path}` → HTML shell (task detail)
- `GET /status` → HTML shell (worker status)
- `GET /health` → JSON `{"status": "healthy"}`
- `WorkerService` Connect RPC at `/iris.cluster.WorkerService/*`

Worker-side RPCs: `ListTasks`, `GetTaskStatus`, `FetchLogs`, `GetProcessStatus`,
`ProfileTask`, `Heartbeat`, `HealthCheck`.

---

## 5. Task Dispatch Protocol

Tasks flow controller → worker exclusively via the **heartbeat RPC**
(`Heartbeat` in `cluster.proto`). There is no separate dispatch RPC.

```
Controller._do_heartbeat_rpc(snapshot):
    stub = RpcWorkerStubFactory.get_stub(worker.address)  # cached httpx client
    request = HeartbeatRequest(
        tasks_to_run=[RunTaskRequest(task_id, attempt_id, image, command, ...)],
        tasks_to_kill=[task_id, ...],
        expected_tasks=[(task_id, attempt_id), ...],  # reconciliation set
    )
    response = stub.heartbeat(request, timeout_ms=5000)
    # response: tasks=[], resource_snapshot, worker_healthy
```

The controller uses `drain_dispatch(worker_id)` (in `transitions.py`) to
snapshot the pending dispatch queue before the RPC, so retries are safe.
`complete_heartbeat()` / `fail_heartbeat()` apply the response atomically
(line 1436–1537).

**RpcWorkerStubFactory** (line 614): caches one `WorkerServiceClientSync`
per worker address. Uses `httpx` under the hood (Connect RPC over HTTP/1.1).
The factory implements `WorkerStubFactory` Protocol (line 606), making it
swappable.

---

## 6. "WorkerProxy" Mode

**Not present in the current codebase.** The grep finds no `WorkerProxy`,
`worker_proxy`, or similar. The concept may be a proposed design.

What **does** exist is a **controller-as-proxy** pattern:
- `GET /worker/{id}` on the controller dashboard forwards the UI shell.
- `FetchLogs(source="/system/worker/<id>")` is proxied by
  `ControllerService.fetch_logs()` (line ~1461–1480) to
  `WorkerService.fetch_logs(source="/system/process")`.
- `ProfileTask(target="/system/worker/<id>")` is proxied to
  `WorkerService.profile_task(target="/system/process")` (line ~1368–1382).
- `GetProcessStatus(target="/system/worker/<id>")` is proxied (line 1574–1608).

The `_WORKER_TARGET_PREFIX = "/system/worker/"` sentinel and
`_parse_worker_target()` helper (line 171–180 in service.py) are the
current proxy convention.

There is also a `ProxyControllerDashboard` (dashboard.py line 181) that
proxies an entire remote controller's Connect RPC over HTTP, for local
dashboard access to a remote controller.

---

## 7. Log Fetching / Offline Sync

**LogStore** (`cluster/log_store.py`): SQLite-backed, WAL mode. Keyed by
arbitrary string. Keys used:

- `PROCESS_LOG_KEY = "/system/process"` — worker or controller process logs
- `task_log_key(attempt)` — per-task attempt logs (e.g. `/alice/job/0:attempt_1`)

**On the worker:** `LogStore` instance shared between `WorkerServiceImpl`
(FetchLogs RPC) and `WorkerServiceImpl.heartbeat()` which drains logs into
`HeartbeatResponse.log_entries` for the controller to store in its own
LogStore (see `transitions.py` line 26 import).

**On the controller:** `ControllerTransitions` writes heartbeat log entries
into `controller._log_store` tagged by task attempt key
(`cluster/controller/transitions.py`, `_apply_log_entries()` called from
`complete_heartbeat()`).

**Log proxy RPC flow** (controller service.py line ~1461):

```python
# source = "/system/worker/<worker_id>"
stub = self._resolve_worker_stub(worker_id)
forwarded_req = FetchLogsRequest(source="/system/process", ...)
return stub.fetch_logs(forwarded_req)   # directly to worker's LogStore
```

No offline buffering: if the worker is dead, the proxy RPC fails. Historical
task logs are stored in the controller's SQLite LogStore (written during
heartbeats while the task was alive).

**Checkpoint**: The controller's SQLite DB (including the LogStore) is
periodically snapshotted to `remote_state_dir` (GCS) via
`write_checkpoint()` (`cluster/controller/checkpoint.py`). This is the only
persistence path for task logs after a controller restart.

---

## 8. Kubernetes Integration

**Two distinct usages:**

### 8a. KubernetesRuntime — worker-side task execution

**File:** `cluster/runtime/kubernetes.py`

When a worker's config has `runtime = "kubernetes"` (set at worker startup in
`main.py` line 71), tasks run as **Pods** instead of Docker containers.
`KubernetesRuntime` implements the `ContainerRuntime` protocol:

- `run()` → `kubectl apply` a generated Pod spec
- `logs()` → `kubectl logs --timestamps --follow`
- `wait()` → polls `kubectl get pod` for phase
- `kill()` → `kubectl delete pod`
- `profile()` → `kubectl exec` pyspy into the running pod

GPU and RDMA resources are requested via Pod resource limits. On CoreWeave, the
worker Pod itself must **not** claim GPU/RDMA (those go to task Pods).
Uses `Kubectl` wrapper (`cluster/k8s/kubectl.py`) which shells out to
`kubectl`.

### 8b. CoreweavePlatform — controller + worker slice lifecycle

**File:** `cluster/platform/coreweave.py`

Implements the `Platform` protocol for CoreWeave CKS. Node lifecycle is
managed by CoreWeave autoscaler (not Iris). Iris manages **Pods** only:

- `create_slice()` → creates a Pod per worker using NodePool selectors
- `start_controller()` → `kubectl apply` ConfigMap + Deployment + Service
- `stop_controller()` → `kubectl delete` those resources
- `stop_all()` → `kubectl delete` all NodePools + controller resources
- Uses `Kubectl` wrapper (`cluster/k8s/kubectl.py`)

NodePool names: `{label_prefix}-{scale_group_name}`. Bootstrap polling
timeout: 2400 s (CoreWeave autoscaler may need to provision bare-metal).

### 8c. No GKE/standard Kubernetes cluster support

There is no `KubernetesPlatform` — CoreWeave is the only K8s-based Platform.
The `KubernetesRuntime` is used for task execution on existing workers, not
for provisioning workers.

---

## 9. Files Referencing 'worker' (top-count files)

From grep count output (files with most `worker` references):

| File | Count | Role |
|------|-------|------|
| `cluster/controller/service.py` | 161 | RPC service, worker proxy logic |
| `cluster/controller/controller.py` | 222 | heartbeat, scheduling, worker tracking |
| `cluster/controller/transitions.py` | 222 | DB state machine for workers |
| `cluster/controller/scheduler.py` | 147 | WorkerSnapshot, assignment |
| `cluster/controller/autoscaler.py` | 106 | TrackedWorker, scale decisions |
| `cluster/controller/db.py` | 50 | Worker DB model |
| `cluster/controller/scaling_group.py` | 57 | slice → worker mapping |
| `cluster/worker/worker.py` | 59 | Worker class |
| `cluster/worker/task_attempt.py` | 24 | TaskAttempt (runs per-worker) |
| `cluster/platform/coreweave.py` | 80 | Pod lifecycle |
| `cluster/platform/gcp.py` | 88 | TPU worker SSH |
| `cluster/platform/local.py` | 64 | In-process LocalWorkerHandle |
| `cluster/platform/bootstrap.py` | 45 | Worker bootstrap scripts |

**All files that touch the Worker abstraction significantly:**

```
cluster/worker/worker.py           # Worker class (daemon)
cluster/worker/service.py          # WorkerServiceImpl + TaskProvider protocol
cluster/worker/dashboard.py        # HTTP dashboard
cluster/worker/main.py             # CLI entrypoint
cluster/worker/task_attempt.py     # TaskAttempt execution
cluster/worker/worker_types.py     # TaskInfo protocol, LogLine
cluster/worker/env_probe.py        # hardware detection
cluster/worker/port_allocator.py   # port management
cluster/controller/controller.py   # WorkerStubFactory, heartbeat dispatch
cluster/controller/service.py      # /system/worker/* proxy RPCs
cluster/controller/transitions.py  # register/fail/timeout DB state
cluster/controller/scheduler.py    # WorkerSnapshot for scheduling
cluster/controller/autoscaler.py   # TrackedWorker, bootstrap monitoring
cluster/controller/scaling_group.py# SliceHandle → workers
cluster/platform/base.py           # RemoteWorkerHandle, Platform protocol
cluster/platform/gcp.py            # GcpWorkerHandle (SSH)
cluster/platform/manual.py         # ManualWorkerHandle (SSH)
cluster/platform/local.py          # _LocalWorkerHandle (in-process)
cluster/platform/coreweave.py      # Pod-based worker
cluster/platform/bootstrap.py      # bootstrap scripts for workers
cluster/platform/_worker_base.py   # RemoteExecWorkerBase shared base
cluster/runtime/kubernetes.py      # KubernetesRuntime (task execution)
cluster/log_store.py               # per-worker/task log storage
```

---

## 10. Config Structures

### Worker-side

**`WorkerConfig`** (`cluster/worker/worker.py:47`) — dataclass, see §1.

**`config_pb2.WorkerConfig`** (`rpc/config.proto`) — proto; loaded from JSON
file passed to `iris worker serve --worker-config`.

### Controller-side

**`ControllerConfig`** (`cluster/controller/controller.py:648`) — dataclass,
see §2.

**`config_pb2.IrisClusterConfig`** — top-level proto, loaded from YAML config
file (`cluster/config.py`). Contains:
- `PlatformConfig` (oneof: gcp, manual, local, coreweave)
- `SshConfig`
- `AutoscalerConfig` (evaluation_interval, scale_up_delay, scale_down_delay)
- `DefaultsConfig` → `WorkerConfig` defaults
- `ScaleGroupConfig[]`

**`DefaultsConfig`** (`cluster/config.py:41`):
```python
DEFAULT_CONFIG = config_pb2.DefaultsConfig(
    ssh=SshConfig(user="root", connect_timeout=30s),
    autoscaler=AutoscalerConfig(evaluation_interval=10s, scale_up_delay=60s, scale_down_delay=600s),
    worker=WorkerConfig(port=10001, cache_dir="/dev/shm/iris", host="0.0.0.0", port_range="30000-40000"),
)
```

**`IrisConfig`** wrapper (`cluster/config.py`) — high-level Python wrapper
with factory methods for creating `Platform`, `ControllerConfig`, etc. from
the loaded proto.

---

## Summary for Provider Abstraction Design

**What's already abstracted:**
- `Platform` protocol (`platform/base.py`) cleanly separates infrastructure
  provisioning from lifecycle. Four implementations: GCP, Manual, Local,
  CoreWeave.
- `RemoteWorkerHandle` / `SliceHandle` protocols for infrastructure access.
- `WorkerStubFactory` Protocol (controller.py:606) for worker RPC connection,
  making it mockable in tests.
- `ContainerRuntime` protocol (two implementations: Docker, Kubernetes).
- `TaskProvider` Protocol (worker/service.py:26) decouples RPC service from
  `Worker` execution internals.
- `TaskInfo` Protocol (worker/worker_types.py:53) — read-only task view.

**What's NOT abstracted (tightly coupled to address/RPC model):**
- Controller always heartbeats workers over HTTP using `worker.address`
  (host:port). There is no protocol indirection.
- Log proxy RPCs in `service.py` hardcode the pattern of forwarding to
  `WorkerServiceClientSync` by address.
- `RpcWorkerStubFactory` is the concrete factory used in production; the
  `WorkerStubFactory` Protocol exists but isn't widely leveraged for
  alternative transports.
- Worker registration is always network-based (controller HTTP endpoint);
  no in-process registration path except `LocalPlatform`'s thread-based workers
  which still go through the HTTP stack.

**Open questions:**
1. What is the concrete design goal of "Provider abstraction"? Is it to
   abstract the worker-side scheduling/provisioning, the RPC transport, or
   both?
2. "WorkerProxy mode" referenced in the issue is absent from the codebase —
   is it a proposed new mode where the controller proxies task execution to
   an external provider (e.g., a cloud batch system) rather than heartbeating
   to a running worker daemon?
3. The `worker_access_address` field in `ControllerConfig` (line 680) is set
   but never read in the controller source — its semantics are unclear.
