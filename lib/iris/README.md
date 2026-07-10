# Iris

Distributed job orchestration for Marin.

## Quick Start

### Cluster Basics

```bash
# Start controller VM (runs autoscaler internally)
uv run iris --cluster=marin cluster start

# Start a local cluster for testing (mimics the config without GCP)
# Dashboard is available at the printed URL; press Ctrl+C to stop.
uv run iris --cluster=marin cluster start --local

# Check cluster status
uv run iris --cluster=marin cluster status

# Validate cluster with test jobs (establishes SSH tunnel automatically)
uv run iris --cluster=marin cluster debug validate

# Stop cluster (controller + all worker slices, terminated in parallel; 60s timeout)
uv run iris --cluster=marin cluster stop
```

### Submit a Job

```python
from iris.client import IrisClient
from iris.cluster.types import Entrypoint, ResourceSpec

def my_task():
    print("Hello from Iris!")

client = IrisClient.remote("http://controller:10000", workspace=Path("."))
job = client.submit(
    name="my-job",
    entrypoint=Entrypoint.from_callable(my_task),
    resources=ResourceSpec(cpu=1, memory="2GB"),
)
job.wait()
```

For accelerator jobs, request the accelerator on the task itself with `--tpu ...` or `--gpu ...`.
`--reserve <accel>` is a hard constraint that confines the job to a zone where `<accel>` has actually
been obtained (a live, non-erroring slice in the region), and the job waits otherwise; it does not
attach accelerator devices (use `--tpu`/`--gpu` for that) and does not hold capacity.

## Architecture

```
Controller Process (in Docker container):
├── gRPC service (job dispatch, worker registration)
├── HTTP dashboard (monitoring, status)
├── Scheduler thread (task→worker matching)
├── Autoscaler thread (VM lifecycle management)
└── WorkerVm threads (per-VM state machines)

Worker Process (on each VM):
├── Task executor (runs jobs in containers)
└── Heartbeat reporter (health monitoring)
```

The controller drives task execution through a single `TaskBackend` contract with
two placements: `Placement.IRIS` (Iris schedules task→worker and fans reconcile
RPCs to worker daemons — the diagram above) and `Placement.BACKEND` (the backend,
e.g. Kubernetes via Kueue, places tasks itself). See
[`docs/architecture.md`](docs/architecture.md#the-taskbackend-contract) for the
contract and how the controller dispatches by placement.

## Actor System

Iris includes a lightweight actor RPC system for service-style workloads. Actor
servers run inside worker containers (or standalone VMs), and clients resolve
actor endpoints via a resolver implementation:

```
Actor Client
  │
  │ resolve(actor_name)
  v
Resolver (ClusterResolver / FixedResolver)
  │
  │ endpoints (url + actor_id)
  v
Worker VM
  └─ Job Container (iris-managed)
       └─ Actor Server
            └─ Actor instance (registered methods)
```

Resolver options:
- **ClusterResolver** (in `iris.client.resolver`): query the controller for
  namespace-aware actor endpoints (best for Iris clusters).
- **FixedResolver**: static endpoint mapping (tests or fixed deployments).

The actor system also provides `ActorPool` for round-robin calls and broadcast
RPCs across all resolved endpoints.

Example:

```python
from iris.actor import ActorClient
from iris.client.resolver import ClusterResolver

resolver = ClusterResolver("http://controller:10000", namespace="default")
client = ActorClient(resolver, "inference")
result = client.predict({"text": "hello"})
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Controller** | Central coordinator: job scheduling, worker registry, autoscaling |
| **Worker** | Execution agent running jobs in isolated containers |
| **Scale Group** | Configuration for a type of accelerator (TPU, GPU) with min/max slices |
| **Slice** | Atomic scaling unit - a complete TPU pod that succeeds or fails as a whole |
| **Platform** | Protocol for VM/slice lifecycle (GCP, Manual, Local, or Fake for testing) |

### Network Architecture

#### Controller Addresses

| Client Type | Address Type | Notes |
|-------------|--------------|-------|
| Workers | Internal IP | Workers on VPC connect via internal IP (automatic with autoscaler) |
| External Clients | SSH Tunnel | Use `gcloud compute ssh` with port forwarding |

Workers communicate with the controller using internal VPC IPs. External clients (your laptop, CI) should use SSH tunneling to access the controller.

## Worker Lifecycle

### Registration and Reconcile

Workers register with the controller once at startup via the `Register` RPC.
After registration, the worker serves the `WorkerService` gRPC and waits for
controller-initiated `Reconcile` calls.

The controller issues one `Reconcile` RPC per worker on each scheduler tick
(~5s). The request carries the controller's complete **desired attempt set**
for that worker: each `DesiredAttempt` is keyed by `attempt_uid` and contains
either an `AttemptSpec.run` intent (with the `RunTaskRequest` payload on the
dispatch tick) or a `StopReason` intent.

The worker responds with the complete **observed set** plus a `WorkerHealth`
block:
- `AttemptObservation` per attempt the controller asked about (`attempt_uid`,
  current `TaskState`, `exit_code`, `error`, `finished_at`, `container_id`,
  `resource_usage`).
- `WorkerHealth` carries `healthy`/`health_error` and a
  `WorkerResourceSnapshot`.

The controller reconciles the response:

1. **Worker missing expected tasks** (e.g., worker restarted mid-task):
   - The worker emits `TASK_STATE_MISSING` observations for run intents that
     resolved to nothing locally.
   - Controller marks those tasks as `WORKER_FAILED` and retries them on
     another worker.

2. **Worker has unknown tasks** (e.g., controller restarted):
   - The worker kills any local attempt not in the desired set ("zombie") and
     returns an observation for the kill on this tick.

See [`docs/reconcile_rpc.md`](docs/reconcile_rpc.md) for the protocol details.

## Jobs, Tasks, Replicas, and Sub-jobs

A **job** is the user-facing workload submitted to Iris. A **task** is the
independently scheduled unit of execution created for that job. Job state is
derived from the states of its tasks.

By default a job creates a single task, but setting `replicas=N` creates N sibling
tasks under the same job, with task IDs like `/user/job/0`, `/user/job/1`, and so on.
Use replicas when one logical workload needs multiple peer containers, such as
multi-host training ranks or gang-scheduled TPU/GPU jobs.

Scheduling and failure happen at task granularity: the scheduler places
each task on eligible capacity; failures retry a given task.
See [`docs/task-states.md`](docs/task-states.md) for the task state machine,
retry rules, and job-state rollup.

A **sub-job** is a separate job submitted by code already running inside an Iris
job, usually via `iris_ctx().client.submit(...)`. Sub-jobs get hierarchical IDs
like `/user/parent/child`, can have their own entrypoint and resources, and may
themselves create multiple replicas. Use sub-jobs when a launcher or
orchestrator job needs to admit independent follow-on work over time.

### Job State Transitions

Jobs progress through the following states. For task states, retry budgets, and
job-state rollup rules, see [`docs/task-states.md`](docs/task-states.md).

| State | Description |
|-------|-------------|
| **PENDING** | Job submitted, waiting for worker assignment |
| **BUILDING** | Job bundle being built/transferred (future use) |
| **RUNNING** | At least one task is actively executing |
| **SUCCEEDED** | All tasks completed successfully |
| **FAILED** | Job failed (exceeded max task failures or retry limit) |
| **KILLED** | Job was cancelled by user |
| **UNSCHEDULABLE** | Job could not be scheduled (constraint mismatch or timeout) |

### Endpoint Visibility

Job endpoints (registered via `RegisterEndpoint` RPC) are visible for all non-terminal states:
- **PENDING**: Endpoint visible (tasks may be executing before job state updates)
- **BUILDING**: Endpoint visible
- **RUNNING**: Endpoint visible
- **Terminal states** (SUCCEEDED, FAILED, KILLED): Endpoints **not visible**

This behavior accounts for controller-worker communication delay: a task may start
executing and register an endpoint before the controller updates the job state to RUNNING.

### Startup Cleanup

Workers wipe ALL `iris.managed=true` containers at startup. This simple approach:
- Handles crash recovery without complex tracking
- Cleans orphaned containers from previous runs
- Ensures fresh state on every worker start

### Container Labels

Task containers are labeled for discoverability:
- `iris.managed=true` - All iris-managed containers
- `iris.task_id=<id>` - Task identifier
- `iris.job_id=<id>` - Job identifier

### TPU Container Configuration

When a job requests TPU resources (`device=tpu_device("v5litepod-16")`), workers automatically configure Docker containers with the necessary flags and environment variables for TPU access:

**Docker flags:**
- `--device /dev/vfio:/dev/vfio` - VFIO device for TPU passthrough
- `--shm-size=100g` - Large shared memory for TPU operations
- `--cap-add=SYS_RESOURCE` - Resource management capabilities
- `--ulimit memlock=68719476736:68719476736` - Unlocked memory limits

**Environment variables:**
- `JAX_PLATFORMS=tpu,cpu` - JAX platform configuration
- `PJRT_DEVICE=TPU` - PJRT runtime device
- `TPU_SKIP_MDS_QUERY=1` - force JAX to use explicit TPU worker metadata in containers
- `TPU_ACCELERATOR_TYPE`, `TPU_TYPE` - TPU accelerator variant (for libtpu/JAX topology init)
- `TPU_NAME`, `TPU_WORKER_ID`, `TPU_WORKER_HOSTNAMES`, `TPU_CHIPS_PER_HOST_BOUNDS` - TPU metadata from host
- `JAX_COORDINATOR_ADDRESS`, `JAX_NUM_PROCESSES`, `JAX_PROCESS_ID` - explicit JAX distributed coordination

This enables JAX and other TPU-aware frameworks to initialize correctly inside job containers.

## Bundle Storage (Required)

Jobs can include a `bundle_blob` containing workspace files. The controller stores these in a shared location accessible to all workers.

**Configuration** (required):

```yaml
storage:
  remote_state_dir: gs://my-bucket/iris/state  # remote storage for checkpoints and worker profiles
```

The controller will **fail at startup** if `storage.remote_state_dir` is not configured.

### Multi-Region Bundle Storage

Bundles are stored in one centralized GCS bucket and fetched by workers in all
regions. Bundles are small enough that cross-region transfer is cheaper than
operating regional caches, and centralized storage keeps worker bootstrap
simple.

## CLI Reference

**Cluster selection:** Prefer `--cluster=<name>` for known clusters; it resolves named configs such as
`marin` from the configured search paths. Use `--config=<path>` when you need to pin an exact YAML file,
such as a custom config or local experiment. Both flags are global and must appear before the subcommand:
`iris --cluster=marin cluster start`.

### Cluster Commands

```bash
# Start/stop/restart controller VM
iris --cluster=marin cluster start
iris --cluster=marin cluster start --local   # Local cluster for testing
iris --cluster=marin cluster stop
iris --cluster=marin cluster restart
iris --cluster=marin cluster status
```

### Controller Subcommands

```bash
# Controller-specific operations
iris --cluster=marin cluster controller start       # Boot controller GCE VM
iris --cluster=marin cluster controller status      # Controller status
```

### VM Operations (via controller RPC)

```bash
# VM status and logs (always via controller)
iris --controller-url=http://localhost:10000 cluster vm status
iris --controller-url=http://localhost:10000 cluster vm logs VM_ID
```

### Image Builds

```bash
# Build and push Docker images
iris build worker-image -t iris-worker:v1 --push --region us-central1
iris build controller-image -t iris-controller:v1 --push --region us-central1
```

### Dashboard & Debugging

```bash
# Remote clusters: opens SSH tunnel to controller dashboard
iris --cluster=marin cluster dashboard
iris --cluster=marin cluster dashboard --port 8080

# Local clusters: dashboard is at the URL printed by `cluster start --local`
```

### Job Management

```bash
# Submit a command to the cluster
iris --config cluster.yaml job run -- python train.py
iris --config cluster.yaml job run --tpu v5litepod-16 -e WANDB_API_KEY $WANDB_API_KEY -- python train.py
iris --config cluster.yaml job run --no-wait -- python long_job.py
# Pin a zone when you need to colocate with data or target a specific pool.
iris --config cluster.yaml job run --zone us-central2-b -- python train.py

# Stream logs for a job and child jobs (batch-fetches matching tasks in one RPC)
iris --config cluster.yaml job logs /my-job
iris --config cluster.yaml job logs /my-job --follow
iris --config cluster.yaml job logs /my-job --since-seconds 300

# Stop one or more jobs
iris --config cluster.yaml job stop /my-job
iris --config cluster.yaml job stop /my-job --no-include-children
```

## Smoke Test

The smoke test validates end-to-end cluster functionality including scheduling,
dashboard rendering, log levels, profiling, and constraint routing.

```bash
# Local mode (in-process cluster, default)
uv run pytest lib/iris/tests/e2e/test_smoke.py -m requires_cluster -o "addopts=" -v

# Cloud mode: start cluster via CLI, then run tests against it
# iris --cluster=ci-gcp-smoke cluster start-smoke --label-prefix my-test --url-file /tmp/url --wait-for-workers 1
uv run pytest lib/iris/tests/e2e/test_smoke.py -m requires_cluster --iris-controller-url "$(cat /tmp/url)" -o "addopts="

# Cloud mode: connect to existing cluster
uv run pytest lib/iris/tests/e2e/test_smoke.py -m requires_cluster --iris-controller-url http://localhost:8080 -o "addopts="

# Screenshots saved to custom directory
IRIS_SCREENSHOT_DIR=/tmp/shots uv run pytest lib/iris/tests/e2e/test_smoke.py -m requires_cluster -o "addopts="
```

## Configuration

Configuration uses platform-first settings with typed defaults:

```yaml
platform:
  label_prefix: iris
  gcp:
    project_id: my-project

defaults:
  autoscaler:
    evaluation_interval: { milliseconds: 10000 }
    scale_up_delay: { milliseconds: 60000 }
    scale_down_delay: { milliseconds: 300000 }
  ssh:
    user: ubuntu
    key_file: ~/.ssh/cluster_key
    connect_timeout: { milliseconds: 30000 }
  bootstrap:
    docker_image: us-central1-docker.pkg.dev/my-project/marin/iris-worker:latest
    worker_port: 10001
    controller_address: "10.0.0.1:10000"  # Or use env var: "${IRIS_CONTROLLER_ADDRESS}"

storage:
  remote_state_dir: gs://my-bucket/iris/state  # remote storage for checkpoints and worker profiles

controller:
  image: us-central1-docker.pkg.dev/my-project/marin/iris-controller:latest
  gcp:
    zone: us-central1-a
    machine_type: n2-standard-4
    port: 10000

scale_groups:
  tpu_v5e_4:
    zones: [us-central1-a, us-central1-b]
    num_vms: 1
    priority: 10
    resources:
      cpu: 64
      ram: 64GB
      disk: 500GB
      device_type: tpu
      device_variant: v5litepod-4
      device_count: 4
      preemptible: true
    buffer_slices: 0
    max_slices: 10
    slice_template:
      gcp:
        zone: us-central1-a
        runtime_version: v2-alpha-tpuv5-lite

  manual_hosts:
    num_vms: 1
    resources:
      cpu: 16
      ram: 32GB
      disk: 100GB
      device_type: cpu
      preemptible: false
    buffer_slices: 0
    max_slices: 2
    slice_template:
      manual:
        hosts: [10.0.0.1, 10.0.0.2]
        ssh_user: ubuntu
        ssh_key_file: ~/.ssh/manual_key
```

## References

- [Architecture](docs/architecture.md) - source layout, import layers, and the `TaskBackend` contract
- [Task States](docs/task-states.md) - Task state machine and retry semantics
- [Priority Bands](docs/priority-bands.md) - production, interactive, and batch scheduling priority
- [CoreWeave](docs/coreweave.md) - CoreWeave GPU cluster quickstart and operator guide
- [Federation](docs/federation.md) - how a job is routed to a peer cluster, and what travels with it
