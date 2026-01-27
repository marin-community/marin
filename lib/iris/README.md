# Iris

Distributed job orchestration replacing Ray with simpler primitives.

## Quick Start

### Production: GCP Cluster

```bash
# Start controller VM (runs autoscaler internally)
uv run iris cluster --config=examples/eu-west4.yaml start

# Check cluster status
uv run iris cluster --config=examples/eu-west4.yaml status

# Validate cluster with test jobs (establishes SSH tunnel automatically)
uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models validate

# Stop cluster (controller + all worker slices)
uv run iris cluster --config=examples/eu-west4.yaml stop
```

### Development: Local Controller

```bash
# Run controller locally with integrated autoscaler
uv run iris cluster --config=cluster.yaml controller run-local

# Or use the controller daemon directly
uv run python -m iris.cluster.controller.main serve --config=cluster.yaml
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

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Controller** | Central coordinator: job scheduling, worker registry, autoscaling |
| **Worker** | Execution agent running jobs in isolated containers |
| **Scale Group** | Configuration for a type of accelerator (TPU, GPU) with min/max slices |
| **Slice** | Atomic scaling unit - a complete TPU pod that succeeds or fails as a whole |
| **VmManager** | Abstraction for VM lifecycle (GCP, Manual, or Fake for testing) |

## Worker Lifecycle

### Registration and Reconciliation

Workers register with the controller via heartbeat (every 10 seconds). The registration
includes `running_task_ids` - the list of tasks the worker believes it's running.

The controller performs bidirectional reconciliation:

1. **Worker claims unknown tasks** (e.g., controller restarted):
   - Controller sends `should_reset=True`
   - Worker wipes all containers and re-registers

2. **Worker missing expected tasks** (e.g., worker restarted):
   - Controller marks missing tasks as `WORKER_FAILED`
   - Tasks will be retried on another worker

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

## CLI Reference

### Cluster Commands

```bash
# Start/stop/restart controller VM (--config on cluster group)
iris cluster --config=cluster.yaml start
iris cluster --config=cluster.yaml stop
iris cluster --config=cluster.yaml restart
iris cluster --config=cluster.yaml status

# Controller subcommands (for GCE-managed controller)
iris cluster --config=... controller start
iris cluster --config=... controller stop
iris cluster --config=... controller restart
iris cluster --config=... controller status
iris cluster --config=... controller run-local  # Development mode
```

### Slice Management

```bash
# Create/list/terminate slices
iris cluster --config=... slice create --scale-group tpu_v5e_4
iris cluster --config=... slice list
iris cluster --config=... slice get SLICE_ID
iris cluster --config=... slice terminate SLICE_ID
iris cluster --config=... slice terminate --all
```

### VM Operations

```bash
# VM status and logs (via config or controller URL)
iris cluster --config=... vm status
iris cluster vm --controller-url=http://localhost:10000 status
iris cluster --config=... vm logs VM_ID
iris cluster --config=... vm get VM_ID
```

### Image Builds

```bash
# Build and push Docker images
iris build worker-image -t iris-worker:v1 --push --region us-central1
iris build controller-image -t iris-controller:v1 --push --region us-central1
```

### Cluster Tools (Debugging & Validation)

The `scripts/cluster-tools.py` script provides debugging and validation commands:

```bash
uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models --help

# Discover and show controller VM status
discover
autoscaler-status
list-workers
# controller logs
logs {--follow}
bootstrap-logs
validate
# Cleanup all iris resources (dry-run by default)
cleanup {--no-dry-run}
```

## Configuration

Configuration uses a nested structure with `bootstrap`, `timeouts`, and `ssh` sub-configs:

```yaml
# Cluster-level settings
project_id: my-project
region: us-central1
zone: us-central1-a

# Bootstrap config for worker VMs
bootstrap:
  docker_image: us-central1-docker.pkg.dev/my-project/marin/iris-worker:latest
  worker_port: 10001
  controller_address: "10.0.0.1:10000"  # Or use env var: "${IRIS_CONTROLLER_ADDRESS}"

# Timeout settings (VM lifecycle)
timeouts:
  boot_timeout_seconds: 300        # Time for VM to become SSH-reachable
  init_timeout_seconds: 600        # Time for worker to register with controller
  ssh_poll_interval_seconds: 5     # Interval for health checks

# SSH config (for manual provider)
ssh:
  user: ubuntu
  key_file: ~/.ssh/cluster_key
  port: 22
  connect_timeout: 30

# Controller VM (GCP-managed)
controller_vm:
  gcp:
    image: us-central1-docker.pkg.dev/my-project/marin/iris-controller:latest
    machine_type: n2-standard-4
    port: 10000

# Scale groups define VM pools with autoscaling
scale_groups:
  tpu_v5e_4:
    provider:
      tpu:
        project_id: my-project
    accelerator_type: tpu
    accelerator_variant: v5litepod-4
    runtime_version: v2-alpha-tpuv5-lite
    min_slices: 0
    max_slices: 10
    preemptible: true
    zones: [us-central1-a, us-central1-b]

  # Manual hosts example (no cloud provisioning)
  manual_hosts:
    provider:
      manual:
        hosts: [10.0.0.1, 10.0.0.2]
        ssh_user: ubuntu        # Per-group SSH override
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
└── cli.py                    # Main CLI entry point
```

## References

- [Original Design](docs/fray-zero.md) - Design rationale and architectural decisions
- [Autoscaler Design](docs/autoscaler-v0-design.md) - Technical specification for VM autoscaling
