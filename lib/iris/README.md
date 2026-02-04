# Iris

Distributed job orchestration replacing Ray with simpler primitives.

## Quick Start

### Production: GCP Cluster

```bash
# Start controller VM (runs autoscaler internally)
uv run iris cluster --config=examples/eu-west4.yaml start

# Start a local cluster for testing (mimics the config without GCP)
uv run iris cluster --config=examples/eu-west4.yaml start --local

# Check cluster status
uv run iris cluster --config=examples/eu-west4.yaml status

# Validate cluster with test jobs (establishes SSH tunnel automatically)
uv run iris cluster --config=examples/eu-west4.yaml debug validate

# Stop cluster (controller + all worker slices)
uv run iris cluster --config=examples/eu-west4.yaml stop
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

## Architecture

```
Controller Process (in Docker container):
├── gRPC service (job dispatch, worker registration)
├── HTTP dashboard (monitoring, status)
├── Scheduler thread (task→worker matching)
├── Autoscaler thread (VM lifecycle management)
└── ManagedVm threads (per-VM state machines)

Worker Process (on each VM):
├── Task executor (runs jobs in containers)
└── Heartbeat reporter (health monitoring)
```

## Actor System

Iris includes a lightweight actor RPC system for service-style workloads. Actor
servers run inside worker containers (or standalone VMs), and clients resolve
actor endpoints via a resolver implementation:

```
Actor Client
  │
  │ resolve(actor_name)
  v
Resolver (ClusterResolver / GcsResolver / FixedResolver)
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
- **GcsResolver**: discover endpoints via GCP VM metadata tags
  (`iris_actor_<name>`).
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
| **VmManager** | Abstraction for VM lifecycle (GCP, Manual, or Fake for testing) |

### Network Architecture

#### Controller Addresses

| Client Type | Address Type | Notes |
|-------------|--------------|-------|
| Workers | Internal IP | Workers on VPC connect via internal IP (automatic with autoscaler) |
| External Clients | SSH Tunnel | Use `gcloud compute ssh` with port forwarding |

Workers communicate with the controller using internal VPC IPs. External clients (your laptop, CI) should use SSH tunneling to access the controller.

## Worker Lifecycle

### Registration and Heartbeat

Workers register with the controller once at startup via the `Register` RPC.
After registration, the worker enters a serve loop and waits for controller-
initiated heartbeats.

The controller sends `Heartbeat` RPCs to all registered workers on each
scheduler tick (~5s). The heartbeat request carries:
- `tasks_to_run`: new task assignments for this worker
- `tasks_to_kill`: task IDs to terminate

The worker responds with:
- `running_tasks`: tasks currently executing (task_id + attempt_id)
- `completed_tasks`: tasks that finished since the last heartbeat

The controller reconciles the response:

1. **Worker missing expected tasks** (e.g., worker restarted mid-task):
   - Controller marks missing tasks as `WORKER_FAILED`
   - Tasks are retried on another worker

2. **Worker reports unknown tasks** (e.g., controller restarted):
   - Controller sends kill requests for unknown tasks on next heartbeat
   - Worker terminates orphaned containers

## Job State Transitions

Jobs progress through the following states:

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
- `TPU_NAME`, `TPU_WORKER_ID`, `TPU_WORKER_HOSTNAMES`, `TPU_CHIPS_PER_HOST_BOUNDS` - TPU metadata from host

This enables JAX and other TPU-aware frameworks to initialize correctly inside job containers.

## Bundle Storage (Required)

Jobs can include a `bundle_blob` containing workspace files. The controller stores these in a shared location accessible to all workers.

**Configuration** (required):

```yaml
controller:
  bundle_prefix: gs://my-bucket/iris/bundles  # GCS for distributed workers
```

The controller will **fail at startup** if `bundle_prefix` is not configured.

## CLI Reference

**Note:** The `--config` option is a global option on the top-level `iris` command group. It must be placed after `iris` but before the subcommand (e.g., `iris --config cluster.yaml run ...` or `iris cluster --config cluster.yaml start`).

### Cluster Commands

```bash
# Start/stop/restart controller VM (--config on cluster group)
iris cluster --config=cluster.yaml start
iris cluster --config=cluster.yaml start --local   # Local cluster for testing
iris cluster --config=cluster.yaml stop
iris cluster --config=cluster.yaml restart
iris cluster --config=cluster.yaml reload           # Rebuild images + redeploy on existing VMs
iris cluster --config=cluster.yaml status
```

### Controller Subcommands

```bash
# Controller-specific operations
iris cluster --config=... controller start          # Boot controller GCE VM
iris cluster --config=... controller status          # Controller status
```

### VM Operations (via controller RPC)

```bash
# VM status and logs (always via controller)
iris cluster vm --controller-url=http://localhost:10000 status
iris cluster vm --controller-url=http://localhost:10000 logs VM_ID
```

### Image Builds

```bash
# Build and push Docker images
iris build worker-image -t iris-worker:v1 --push --region us-central1
iris build controller-image -t iris-controller:v1 --push --region us-central1
```

### Dashboard & Debugging

```bash
# Open SSH tunnel to controller and print dashboard URL
iris cluster --config=... dashboard
iris cluster --config=... dashboard --port 8080

# Debug commands (auto-establish SSH tunnel)
iris cluster --config=... debug discover         # Find controller VM
iris cluster --config=... debug health           # Health check
iris cluster --config=... debug autoscaler-status
iris cluster --config=... debug list-workers
iris cluster --config=... debug list-jobs
iris cluster --config=... debug logs --follow    # Controller docker logs
iris cluster --config=... debug bootstrap-logs   # VM startup logs
iris cluster --config=... debug show-task-logs JOB_ID
iris cluster --config=... debug validate         # Run test TPU jobs
iris cluster --config=... debug cleanup          # Dry-run by default
iris cluster --config=... debug cleanup --no-dry-run
```

### Job Submission

```bash
# Submit a command to the cluster
iris --config cluster.yaml run -- python train.py
iris --config cluster.yaml run --tpu v5litepod-16 -e WANDB_API_KEY $WANDB_API_KEY -- python train.py
iris --config cluster.yaml run --no-wait -- python long_job.py
```

## Smoke Test

The smoke test validates end-to-end cluster functionality including autoscaling.

```bash
# Full smoke test (builds images, starts cluster, runs TPU jobs)
uv run python lib/iris/scripts/smoke-test.py --config lib/iris/examples/eu-west4.yaml

# Skip image builds (use existing images)
uv run python lib/iris/scripts/smoke-test.py --config ... --no-build-images

# Keep cluster on failure for debugging
uv run python lib/iris/scripts/smoke-test.py --config ... --no-cleanup-on-failure

# Custom job timeout
uv run python lib/iris/scripts/smoke-test.py --config ... --job-timeout 900

# Save logs to a custom directory
uv run python lib/iris/scripts/smoke-test.py --config ... --log-dir /path/to/logs

# Use a unique prefix (isolates resources from other smoke tests)
uv run python lib/iris/scripts/smoke-test.py --config ... --prefix my-test
```

The smoke test:
1. Builds and pushes controller + worker images
2. Starts controller VM with autoscaler
3. Submits 4 TPU jobs to exercise autoscaling:
   - Simple TPU job (basic execution)
   - Concurrent TPU jobs (parallel provisioning)
   - Coscheduled multi-task job (distributed work)
   - JAX TPU job (validates TPU initialization and computation)
4. Collects logs on failure for debugging
5. Cleans up all resources

## Configuration

Configuration uses platform-first settings with typed defaults:

```yaml
platform:
  label_prefix: iris
  gcp:
    project_id: my-project
    region: us-central1
    zone: us-central1-a
    default_zones: [us-central1-a, us-central1-b]

defaults:
  timeouts:
    boot_timeout: { milliseconds: 300000 }
    init_timeout: { milliseconds: 600000 }
    ssh_poll_interval: { milliseconds: 5000 }
  autoscaler:
    evaluation_interval: { milliseconds: 10000 }
    requesting_timeout: { milliseconds: 120000 }
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

controller:
  image: us-central1-docker.pkg.dev/my-project/marin/iris-controller:latest
  bundle_prefix: gs://my-bucket/iris/bundles
  gcp:
    machine_type: n2-standard-4
    port: 10000

scale_groups:
  tpu_v5e_4:
    vm_type: tpu_vm
    accelerator_type: tpu
    accelerator_variant: v5litepod-4
    runtime_version: v2-alpha-tpuv5-lite
    slice_size: 4
    resources:
      cpu: 64
      ram: 64GB
      disk: 500GB
      tpu_count: 4
      gpu_count: 0
    min_slices: 0
    max_slices: 10
    preemptible: true
    zones: [us-central1-a, us-central1-b]

  manual_hosts:
    vm_type: manual_vm
    accelerator_type: cpu
    slice_size: 1
    resources:
      cpu: 16
      ram: 32GB
      disk: 100GB
      tpu_count: 0
      gpu_count: 0
    min_slices: 0
    max_slices: 2
    manual:
      hosts: [10.0.0.1, 10.0.0.2]
      ssh_user: ubuntu
      ssh_key_file: ~/.ssh/manual_key
```

## Directory Structure

```
src/iris/
├── actor/                    # Actor RPC system
│   ├── client.py            # Actor method invocation
│   ├── pool.py              # Multi-endpoint management
│   ├── resolver.py          # Endpoint discovery
│   └── server.py            # Actor hosting
├── client/                   # High-level client layer
│   ├── client.py            # IrisClient and IrisContext
│   ├── resolver.py          # ClusterResolver
│   └── worker_pool.py       # Task dispatch
├── cluster/                  # Cluster orchestration
│   ├── controller/          # Controller service + autoscaler
│   ├── worker/              # Worker service
│   └── vm/                  # VM management + autoscaling
├── rpc/                      # Protocol definitions + generated code
└── cli/                      # CLI package
    ├── main.py               # Top-level iris group
    ├── cluster.py            # Cluster lifecycle, controller, VM ops, dashboard
    ├── build.py              # Image build commands
    ├── run.py                # Job submission (command passthrough)
    ├── rpc.py                # Dynamic RPC CLI
    └── debug.py              # Debugging & validation
```

## References

- [Original Design](docs/fray-zero.md) - Design rationale and architectural decisions
- [Autoscaler Design](docs/autoscaler-v0-design.md) - Technical specification for VM autoscaling
