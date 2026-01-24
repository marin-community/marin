# Iris

Distributed job orchestration replacing Ray with simpler primitives.

## Quick Start

### Production: GCP Cluster

```bash
# Start controller VM (runs autoscaler internally)
uv run iris cluster --config=cluster.yaml start

# Check cluster status
uv run iris cluster --config=cluster.yaml status

# Stop cluster (controller + all worker slices)
uv run iris cluster --config=cluster.yaml stop
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

## Configuration

```yaml
provider:
  type: gcp
  project_id: my-project
  region: us-central1
  zone: us-central1-a

docker:
  image: us-central1-docker.pkg.dev/my-project/marin/iris-worker:latest
  worker_port: 10001

controller:
  vm:
    enabled: true
    image: us-central1-docker.pkg.dev/my-project/marin/iris-controller:latest
    machine_type: n2-standard-4
    port: 10000

scale_groups:
  tpu_v5e_4:
    accelerator_type: v5litepod-4
    runtime_version: v2-alpha-tpuv5-lite
    min_slices: 0
    max_slices: 10
    preemptible: true
    zones: [us-central1-a, us-central1-b]
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
