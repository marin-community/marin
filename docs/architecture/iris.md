# Iris: Distributed Job Orchestration

Iris is Marin's job orchestration system. It replaced Ray for most production workloads because it offers simpler operational primitives, a central SQLite audit trail, and native support for multi-slice TPU jobs and preemptible VMs.

The core contract: you submit a job (Docker image + entrypoint + resource requirements), and Iris runs it on a worker VM that matches those requirements. Everything else — worker lifecycle, autoscaling, preemption, retries, heartbeating — is Iris's problem.

**Source**: `lib/iris/src/iris/`  
**AGENTS.md**: `lib/iris/AGENTS.md`  
**OPS.md**: `lib/iris/OPS.md`

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Controller](#controller)
  - [Scheduling](#scheduling)
  - [Preemption](#preemption)
  - [Heartbeat Model](#heartbeat-model)
3. [Database Schema](#database-schema)
4. [Job and Task State Machines](#job-and-task-state-machines)
5. [Worker](#worker)
6. [Autoscaler](#autoscaler)
  - [Scale Groups](#scale-groups)
  - [Demand Routing](#demand-routing)
  - [Reservations](#reservations)
7. [Actor System](#actor-system)
8. [Platform Adapters](#platform-adapters)
9. [Client API](#client-api)
10. [Dashboard](#dashboard)
11. [Configuration Reference](#configuration-reference)
12. [Design Decisions](#design-decisions)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Iris Controller                            │
│                 (single GCE VM or K8s Pod)                      │
│                                                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │  gRPC / HTTP│   │  Scheduler   │   │    Autoscaler        │  │
│  │  service    │──▶│  (FIFO +     │   │  (demand routing,    │  │
│  │             │   │   priority)  │   │   bin packing)       │  │
│  └─────────────┘   └──────┬───────┘   └──────────────────────┘  │
│                            │                                    │
│  ┌─────────────────────────▼─────────────────────────────────┐  │
│  │           SQLite (WAL mode, 3 DB files)                   │  │
│  │  jobs · tasks · workers · endpoints · scaling_groups …    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Background threads: scheduling · provider/heartbeat ·          │
│                      prune · profile · autoscaler               │
└──────────────────────────────┬──────────────────────────────────┘
                               │ gRPC (controller-initiated heartbeat)
           ┌───────────────────┴───────────────────┐
           ▼                                       ▼
┌─────────────────────┐                 ┌─────────────────────┐
│    Worker VM A      │                 │    Worker VM B      │
│                     │                 │                     │
│  Task executor ──▶  │                 │  Task executor ──▶  │
│  Docker containers  │                 │  Docker containers  │
│  (iris.managed=true)│                 │  (iris.managed=true)│
│                     │                 │                     │
│ Heartbeat responder │                 │ Heartbeat responder │
└─────────────────────┘                 └─────────────────────┘
```

The controller is a single process (Python + uvicorn). It owns all state in SQLite and drives workers via outbound heartbeat RPCs. Workers are passive: they execute what the controller tells them and report back.

---

## Controller

**Source**: `lib/iris/src/iris/cluster/controller/controller.py`

`Controller` (line 970) runs five background threads:


| Thread                  | Purpose                                               |
| ----------------------- | ----------------------------------------------------- |
| Scheduling loop         | Finds task↔worker assignments, checks worker timeouts |
| Provider/heartbeat loop | Sends heartbeat RPCs to all workers, reconciles state |
| Autoscaler loop         | Evaluates demand and drives VM scale-up/scale-down    |
| Prune thread            | Cleans up old job/worker records per retention policy |
| Profile thread          | Captures periodic CPU/memory snapshots                |


The scheduling and heartbeat threads share no state except the database. Each uses its own connection pool. The scheduling loop uses exponential backoff (1s → 10s) when there's nothing to do, so an idle cluster costs almost no CPU.

Key `ControllerConfig` defaults:

```python
scheduler_min_interval = 1.0      # seconds between scheduling passes when active
scheduler_max_interval = 10.0     # seconds when idle (backoff ceiling)
heartbeat_interval    = 5.0       # seconds between provider loop ticks
max_tasks_per_job_per_cycle = 4   # limits GIL pressure on scheduling passes
job_retention         = 7 days
worker_retention      = 24 hours
```

### Scheduling

**Source**: `lib/iris/src/iris/cluster/controller/scheduler.py`

The scheduler is a **pure function** — it takes a `SchedulingContext` snapshot and returns assignments. It has no imports from the controller module, making it independently testable.

```python
@dataclass
class SchedulingContext:
    index: ConstraintIndex           # posting-list index for O(1) attribute matching
    capacities: dict[WorkerId, WorkerCapacity]
    assignment_counts: dict[WorkerId, int]  # per-cycle limit → round-robin effect
    pending_tasks: list[JobName]     # priority-ordered
    jobs: dict[JobName, JobRequirements]
```

**Priority order** (controller.py:577) — tasks are sorted by:

1. `priority_band ASC` — PRODUCTION(1) > INTERACTIVE(2) > BATCH(3)
2. `priority_neg_depth ASC` — deeper dependency chains run first
3. `priority_root_submitted_ms ASC` — FIFO among root jobs
4. `submitted_at_ms ASC` — FIFO among siblings
5. `priority_insertion ASC` — tiebreaker

Within a scheduling pass, each worker is assigned at most one task by default (`assignment_counts` limits this). This implements approximate round-robin without explicit tracking.

**Constraint matching** uses a posting-list index (`ConstraintIndex`): for each attribute value, it keeps a set of workers that have that attribute. Evaluating a constraint is an O(1) set lookup. Supported constraint types: `EQ`, `NOT_EQ`, `EXISTS`, `NOT_EXISTS`, `LT`, `GT`, `LTE`, `GTE`.

Constraints are decoded from protobuf to native `Constraint` objects at load time (not per-scheduling-pass), so the hot path involves no protobuf parsing.

### Preemption

**Source**: `controller.py`, `_run_preemption_pass` (line 508)

Preemption rules:

- PRODUCTION preempts INTERACTIVE and BATCH
- INTERACTIVE preempts BATCH only
- BATCH never preempts anything
- Coscheduled tasks are never preemptible

When the scheduler finds a high-priority task that can't be scheduled due to resource contention, it selects victim tasks from lower-priority work currently running on suitable workers. Victims are sorted by lowest priority first, then cheapest resource value.

**Budget-based demotion**: Users who have exceeded their compute budget have their tasks' *effective* priority band downgraded to BATCH for preemption purposes, even if submitted as PRODUCTION. This prevents budget overruns from starving other users.

### Heartbeat Model

The controller **initiates** heartbeats to workers (not the other way around). On each provider loop tick (~5s):

```
Controller → Worker: Heartbeat(tasks_to_run=[...], tasks_to_kill=[...])
Worker     → Controller: HeartbeatResp(running_tasks=[...], completed_tasks=[...])
```

The controller reconciles the response:

- Task expected on worker but absent from `running_tasks` and not in `completed_tasks` → mark `WORKER_FAILED`, retry
- Worker reports unknown task → send kill request next heartbeat
- `HEARTBEAT_FAILURE_THRESHOLD = 10` consecutive failures → worker marked dead, all its tasks re-queued
- `HEARTBEAT_STALENESS_THRESHOLD = 900s` — grace period for workers restored from checkpoint

This outbound model means workers don't need network access to a central registry; the controller just needs to reach each worker's IP.

---

## Database Schema

**Source**: `lib/iris/src/iris/cluster/controller/db.py`, `schema.py`

Three WAL-mode SQLite files: `controller.sqlite3`, `auth.sqlite3`, `profiles.sqlite3`.

Read operations use a pool of 8 connections (`_READ_POOL_SIZE = 8`) via `read_snapshot()` — no write lock needed. Write operations use a single connection with `BEGIN IMMEDIATE` and an `RLock`.

Core tables in `controller.sqlite3`:


| Table                       | Key columns                                                                                                |
| --------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `jobs`                      | `id`, `name`, `state`, `priority_band`, `submitted_at_ms`, `max_retries_failure`, `max_retries_preemption` |
| `job_config`                | `job_id`, `docker_image`, `entrypoint`, `resources_json`, `env_json`                                       |
| `tasks`                     | `id`, `job_id`, `state`, `worker_id`, `failure_count`, `preemption_count`                                  |
| `task_attempts`             | `task_id`, `attempt`, `worker_id`, `started_at`, `finished_at`, `exit_code`, `error`                       |
| `workers`                   | `id`, `host`, `port`, `committed_cpu_millicores`, `committed_mem`, `committed_gpus`, `committed_tpus`      |
| `worker_attributes`         | `worker_id`, `key`, `value` — used for constraint matching                                                 |
| `endpoints`                 | `name`, `host`, `port`, `actor_id` — actor endpoint registry                                               |
| `scaling_groups` / `slices` | Autoscaler state                                                                                           |
| `dispatch_queue`            | Tasks ready to be sent to workers                                                                          |
| `txn_log` / `txn_actions`   | Full transaction audit trail                                                                               |
| `user_budgets`              | Per-user compute budget limits                                                                             |


---

## Job and Task State Machines

```
Job States:
                     ┌──────────┐
               ┌────▶│ BUILDING │
               │     └────┬─────┘
  submit()     │          │
┌─────────┐    │          ▼
│ PENDING │────┴─────▶ RUNNING ──▶ SUCCEEDED
└─────────┘                │
                           ├──▶ FAILED        (max retries exceeded)
                           ├──▶ KILLED        (user cancelled)
                           ├──▶ WORKER_FAILED (worker died)
                           └──▶ UNSCHEDULABLE (timeout)

Task States (superset of Job States):
  PENDING → ASSIGNED → BUILDING → RUNNING → SUCCEEDED
                │           │         │
                └───────────┴─────────┤
                                      ├──▶ FAILED
                                      ├──▶ KILLED
                                      ├──▶ WORKER_FAILED
                                      ├──▶ UNSCHEDULABLE
                                      └──▶ PREEMPTED
```

**Retry logic** (`db.py`, line 110):

- `SUCCEEDED`, `KILLED`, `UNSCHEDULABLE`: always terminal — no retry
- `FAILED`: terminal when `failure_count > max_retries_failure`
- `WORKER_FAILED` or `PREEMPTED`: terminal when `preemption_count > max_retries_preemption` (default: 100)

This asymmetry is intentional: preempted or dead-worker tasks are nearly always safe to retry (the job didn't fail, the hardware did). Application failures (non-zero exit code) are more likely a bug and have a lower retry budget.

---

## Worker

**Source**: `lib/iris/src/iris/cluster/worker/worker.py` (line 123)

Workers register with the controller via `Register` RPC on startup. At startup they also **kill all `iris.managed=true` containers** — this is crash recovery: if the worker process died, its containers might still be running and consuming resources.

**Task container lifecycle**:

```
Controller sends task_to_run via heartbeat
    │
    ▼
Worker pulls Docker image (if not cached)
    │
    ▼
Worker runs container with labels:
  iris.managed=true
  iris.task_id=<id>
  iris.job_id=<id>
    │
    ▼
Container exits (success or failure)
    │
    ▼
Worker reports exit code in next heartbeat response
```

**Port allocation**: Workers use `PortAllocator` with range 30000–40000. Actor servers and inference servers pick from this pool.

**TPU-specific Docker flags**:

```
--device /dev/vfio:/dev/vfio
--shm-size=100g
--cap-add=SYS_RESOURCE
```

Plus environment variables for JAX distributed training: `JAX_COORDINATOR_ADDRESS`, `JAX_PROCESS_COUNT`, `JAX_PROCESS_INDEX`.

---

## Autoscaler

**Source**: `lib/iris/src/iris/cluster/autoscaler/`

The autoscaler runs as a background thread in the controller. Its `run_once()` is split into two phases to decouple state-reading from state-mutation:

1. `**refresh()`** — read phase: sync slice states from the platform API, scale down idle slices
2. `**update()**` — compute phase: evaluate demand, execute scale-up decisions

### Scale Groups

A Scale Group defines a type of hardware the autoscaler can provision. Example config:

```yaml
scale_groups:
  - name: tpu_v5e_256
    resources:
      device_type: tpu
      device_variant: v5litepod-256
      cpu_millicores: 960_000
      memory_mib: 1_966_080
    zones: [us-central2-b]
    priority: 1
    buffer_slices: 1         # keep 1 warm slice always ready
    max_slices: 8            # hard cap
    preemptible: true

  - name: cpu_workers
    resources:
      cpu_millicores: 32_000
      memory_mib: 131_072
    zones: [us-central1-a, us-central1-b]
    priority: 2
    buffer_slices: 0
    max_slices: 50
    preemptible: false
```

Slice lifecycle:

```
REQUESTING → BOOTING → INITIALIZING → READY
                │
                └──▶ FAILED (terminal, triggers backoff)
```

Scale Group availability states: `AVAILABLE`, `COOLDOWN`, `REQUESTING`, `AT_MAX_SLICES`, `BACKOFF`, `QUOTA_EXCEEDED`. The autoscaler skips groups that aren't `AVAILABLE`.

Rate limits: `DEFAULT_SCALE_UP_RATE_LIMIT = 16/min`, `DEFAULT_SCALE_DOWN_RATE_LIMIT = 32/min`. Idle threshold before scale-down: 10 minutes.

### Demand Routing

**Source**: `lib/iris/src/iris/cluster/autoscaler/routing.py`

The autoscaler computes demand (how many of each hardware type are needed) and routes it to Scale Groups:

- **CPU demand** is fungible — routes to any available group
- **GPU/TPU demand** must match `device_type` and `device_variant`
- Groups are tried in priority order (waterfall routing)

**Bin packing** (`first_fit_decreasing`): Non-coscheduled tasks are bin-packed into VMs within a group. Coscheduled tasks (e.g., multi-slice TPU runs that must start simultaneously) require an exact VM count match and are not bin-packed.

**Planning** (`planning.py`, `build_group_scale_plan`, line 93):

```python
target_slices = min(required_slices + group.buffer_slices, group.max_slices)
demand_gap    = max(0, required_slices - counts.pending)
buffer_gap    = max(0, target_slices - counts.total)
slices_to_add = min(max(demand_gap, buffer_gap), group.max_slices - counts.total)
```

### Reservations

Reservations allow pre-provisioning capacity for a specific upcoming job. When a job declares reservations:

1. Synthetic `:reservation:` child jobs are created to hold slices
2. Reserved workers get a `reservation-job` taint attribute
3. Non-reservation jobs get a `NOT_EXISTS(reservation-job)` constraint — blocked from reserved workers
4. The reservation job itself gets an `EQ(reservation-job, <job-id>)` constraint — pinned to its reserved workers

This lets a user guarantee their large training run will start promptly without competing with ongoing work.

---

## Actor System

**Source**: `lib/iris/src/iris/actor/`

The actor system is for **long-running service-style workloads** that need endpoint discovery and RPC, as opposed to one-shot job submission. Primary use case: Zephyr coordinators and workers, inference servers.

```
ActorServer (runs inside container on worker)
    │  registers actors by name
    │  exposes gRPC endpoint
    │
    ◀── ActorClient (caller side)
         │  transparent method proxying via __getattr__
         │  retry with exponential backoff (10 attempts, 0.5s→10s)

ActorPool
    │  round-robin across multiple actor endpoints
    │  broadcast to all endpoints
    │  TTL-based endpoint cache

ClusterResolver (client/resolver.py)
    │  queries controller ListEndpoints RPC to find actors by name
```

**ActorServer** (`actor/server.py`, line 84): Actors are registered by name. Methods are discovered via reflection (`dir(actor)`). Calls serialize arguments and return values with `cloudpickle`. Long-running operations use `start_operation` / `get_operation` / `cancel_operation` for async method dispatch.

**ActorClient** (`actor/client.py`, line 50):

```python
class ActorClient:
    def __getattr__(self, method_name: str) -> "_RpcMethod":
        return _RpcMethod(self, method_name)

# Usage:
client = ActorClient(endpoint)
result = client.my_method(arg1, arg2)  # transparent RPC
```

**Endpoint registry**: Actors register their host:port with the controller's `RegisterEndpoint` RPC. `ClusterResolver` queries `ListEndpoints` to find all actors with a given name prefix. This enables dynamic service discovery without a separate registry.

---

## Platform Adapters

**Source**: `lib/iris/src/iris/cluster/platform/`

Each platform adapter implements slice (VM) provisioning and deprovisioning. Iris supports:


| Platform    | Use case                                          |
| ----------- | ------------------------------------------------- |
| `gcp`       | Google Cloud Platform (GCE VMs, TPU VMs via CAAS) |
| `coreweave` | CoreWeave Kubernetes (GPU workloads)              |
| `manual`    | Pre-provisioned VMs with SSH access               |
| `local`     | Local process execution (testing, development)    |


Platform selection is in the cluster YAML config (`platform: gcp`). The controller instantiates the right adapter and calls `provision_slice(group, zone)` / `deprovision_slice(slice_id)`.

---

## Client API

**Source**: `lib/iris/src/iris/client/client.py`

```python
ctx = IrisContext.from_env()  # reads IRIS_CONTROLLER_URL from environment

# Submit a one-shot job
handle = ctx.client.submit(JobRequest(
    name="my-job",
    docker_image="gcr.io/marin/training:latest",
    entrypoint=["python", "-m", "levanter.main.train_lm", "--config", "cfg.yaml"],
    resources=ResourceSpec(tpu_type="v5litepod-256", tpu_count=1),
    env={"WANDB_API_KEY": "..."},
))
result = handle.wait()

# Register/discover an actor
ctx.client.register_endpoint(name="my-coordinator", host="10.0.0.1", port=50051)
endpoints = ctx.client.list_endpoints(name_prefix="my-coordinator")
```

`get_iris_ctx()` is a process-level singleton. In production Iris jobs, the controller URL and auth token are injected as environment variables.

---

## gRPC Service Methods

The `ControllerService` exposes 30+ RPCs, grouped by concern:


| Group             | Methods                                                                                                                          |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Job lifecycle     | `LaunchJob`, `GetJobStatus`, `GetJobState`, `TerminateJob`, `ListJobs`                                                           |
| Task operations   | `GetTaskStatus`, `ListTasks`                                                                                                     |
| Worker management | `Register`, `ListWorkers`, `GetWorkerStatus`, `RestartWorker`                                                                    |
| Endpoint registry | `RegisterEndpoint`, `UnregisterEndpoint`, `ListEndpoints`                                                                        |
| Autoscaler        | `GetAutoscalerStatus`                                                                                                            |
| Debugging         | `GetTransactions`, `GetTaskLogs`, `ProfileTask`, `ExecInContainer`, `GetProcessStatus`, `GetProviderStatus`, `GetSchedulerState` |
| Auth              | `GetAuthInfo`, `Login`, `CreateApiKey`, `RevokeApiKey`, `ListApiKeys`, `GetCurrentUser`                                          |
| Budgets           | `SetUserBudget`, `GetUserBudget`, `ListUserBudgets`                                                                              |
| Admin             | `BeginCheckpoint`, `ExecuteRawQuery`, `ListUsers`                                                                                |


---

## Dashboard

The Iris controller serves a Vue-based web dashboard alongside its gRPC endpoint. The same port handles both. There are three ways to access it depending on where you're running.

### Local cluster (development)

```bash
MARIN_PREFIX=/tmp/marin uv run iris --config=lib/iris/examples/local.yaml cluster start --local
```

The controller prints the URL with an auto-login token on startup:

```
Controller started at http://127.0.0.1:<PORT>
Dashboard: http://127.0.0.1:<PORT>?session_token=<TOKEN>
```

Open that URL directly. The token is a one-time bootstrap that exchanges for a session cookie — subsequent page loads don't need it.

### Remote cluster (production)

```bash
iris --config=infra/marin-us-central2.yaml cluster dashboard
```

`cluster dashboard` establishes an SSH tunnel to the controller VM and prints the local URL. It blocks until you press Ctrl+C.

```
Dashboard:      http://127.0.0.1:<PORT>
Controller RPC: http://127.0.0.1:<PORT>

Press Ctrl+C to close tunnel.
```

Open the printed URL in your browser. The tunnel is kept alive for the duration — close it when done.

The `iris` group command handles tunnel setup lazily: any subcommand that calls `require_controller_url()` (which includes `cluster dashboard`, `cluster status`, `job run`, etc.) will establish the tunnel automatically from the `--config` file.

### Development proxy (frontend hot-reload)

If you're working on the dashboard UI itself and want hot module replacement:

```bash
iris --config=infra/marin-us-central2.yaml cluster dashboard-proxy [--port 8080]
```

This starts two local processes:

- A Python RPC proxy at `http://localhost:8080` that forwards Connect RPC calls to the remote controller
- An rsbuild dev server (usually at `http://localhost:3000`) that serves the Vue frontend with HMR

Open the rsbuild URL printed on startup. The `--port` flag controls the proxy port; the rsbuild port is chosen by rsbuild.

### Authentication

The dashboard uses session-based auth. The bootstrap flow:

1. Navigate to `/?session_token=<TOKEN>` (or the URL printed by `cluster start`)
2. The server exchanges the token for an `iris_session` httpOnly cookie
3. Subsequent requests use the cookie — the token is not needed again

In production, auth tokens come from your cluster config (`iris login`). On local clusters, the auto-login token is printed at startup and skips the login step entirely. Public routes (`/`, `/job/{id}`, `/health`) serve the HTML shell without auth; the RPC calls they make require a valid session.

---

## Configuration Reference

Cluster configuration is a YAML file passed to `iris cluster start`. The full structure:

```yaml
platform: gcp                          # gcp | coreweave | manual | local
controller:
  host: 0.0.0.0
  port: 50051
  scheduler_min_interval: 1.0
  heartbeat_interval: 5.0
  checkpoint_interval: 300.0
  job_retention_days: 7
  worker_retention_hours: 24

storage:
  remote_state_dir: gs://my-bucket/iris-state/

defaults:
  docker_image: gcr.io/my-project/marin:latest
  autoscaler:
    buffer_slices: 1
    idle_threshold_minutes: 10
    scale_up_rate_limit: 16       # slices/minute
    scale_down_rate_limit: 32

scale_groups:
  - name: v5litepod-256
    resources:
      device_type: tpu
      device_variant: v5litepod-256
    zones: [us-central2-b]
    priority: 1
    buffer_slices: 1
    max_slices: 8
    preemptible: true
```

Production cluster configs live in `infra/marin-us-central2.yaml`, `infra/marin-us-central1.yaml`, etc.

---

## Design Decisions

**Why SQLite instead of an external database?**  
The controller is a single process on a single VM. SQLite's WAL mode gives concurrent readers at near-zero operational overhead. There's no network hop, no connection pooling failure mode, and the entire cluster state is a single file that can be inspected with standard tools. The read pool (8 connections) handles concurrent API calls; the single write connection with `BEGIN IMMEDIATE` prevents write conflicts.

**Why does the controller initiate heartbeats instead of workers?**  
Workers don't need to know the controller's address dynamically — only the controller needs to track workers. More importantly, this simplifies the controller's failure model: it always knows which workers it should be hearing from, and any worker that stops responding to controller-initiated heartbeats is dead. A worker-initiated model requires the controller to distinguish "no heartbeat yet" from "heartbeat late" from "worker dead".

**Why cap tasks per job per scheduling cycle (`max_tasks_per_job_per_cycle = 4`)?**  
The scheduling loop runs in the same process as the heartbeat loop. Assigning thousands of tasks in one pass can hold the GIL long enough to delay heartbeat responses, causing workers to appear stale. The cap bounds worst-case scheduling latency and produces round-robin-like distribution across jobs.

**Why separate job vs task retries for preemption vs failure?**  
Application failures (non-zero exit code) are counted against `max_retries_failure` (default: low). Hardware failures like preemption or dead workers are counted against `max_retries_preemption` (default: 100). A training run that gets preempted 50 times should keep going; a training run that crashes with an OOM on every attempt should not.

**Why the actor system in addition to job submission?**  
Jobs are one-shot: they run to completion and exit. Actors are persistent services that accept RPC calls. Zephyr's coordinator pattern requires the coordinator to receive calls from N workers throughout the pipeline's lifetime. Using jobs for this would require re-launching the coordinator on every worker interaction. Actors give a stable endpoint with automatic retry on the client side.