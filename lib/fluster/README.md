# Fluster

Fluster is a distributed job orchestration and RPC framework designed to replace Ray with simpler, more focused primitives. It provides job lifecycle management, actor-based RPC communication, and task dispatch capabilities for distributed Python workloads.

## Architecture Overview

Fluster consists of four main components:

| Component | Description |
|-----------|-------------|
| **Controller** | Central coordinator managing job scheduling, worker registration, and service discovery |
| **Worker** | Execution agent that runs jobs in isolated containers with resource management |
| **Actor System** | RPC framework enabling Python object method invocation across processes |
| **WorkerPool** | High-level task dispatch abstraction for stateless parallel workloads |

```
┌─────────────────────────────────────────────────────────────────┐
│                         Controller                               │
│                                                                  │
│     Job Scheduling    │    Worker Registry    │  Endpoint Registry│
└─────────────────────────────────────────────────────────────────┘
        │                       │                       ▲
        │ dispatch              │ health                │ register
        ▼                       ▼                       │
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Worker      │     │     Worker      │     │   ActorServer   │
│                 │     │                 │     │   (in job)      │
│  runs jobs in   │     │  runs jobs in   │     │                 │
│  containers     │     │  containers     │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Directory Structure

```
src/fluster/
├── actor/                    # Actor RPC system
│   ├── client.py            # Actor method invocation
│   ├── pool.py              # Multi-endpoint management
│   ├── resolver.py          # Endpoint discovery
│   ├── server.py            # Actor hosting
│   └── types.py             # Core types
├── cluster/                  # Cluster orchestration
│   ├── controller/          # Controller service
│   ├── worker/              # Worker service
│   ├── client.py            # Client interface
│   └── types.py             # Shared types
├── proto/                    # Protocol definitions
├── worker_pool.py           # Task dispatch
└── *_pb2.py, *_connect.py   # Generated RPC code
```

## Component Documentation

- [Controller Overview](docs/controller.md) - Job scheduling and coordination
- [Worker Overview](docs/worker.md) - Job execution and container management
- [Actor System Overview](docs/actor.md) - RPC and service discovery

## Quick Start

### Submitting a Job

```python
from fluster.cluster import RpcClusterClient, Entrypoint, create_environment
from fluster.cluster_pb2 import ResourceSpec

def my_task():
    print("Hello from fluster!")

client = RpcClusterClient("http://controller:8080")
job_id = client.submit(
    name="my-job",
    entrypoint=Entrypoint(callable=my_task),
    resources=ResourceSpec(cpu=1, memory="2GB"),
    environment=create_environment(),
)
client.wait(job_id)
```

### Running an Actor Server

```python
from fluster.actor import ActorServer, ActorContext

class InferenceActor:
    def predict(self, ctx: ActorContext, data: list) -> list:
        return [x * 2 for x in data]

server = ActorServer(controller_address="http://controller:8080")
server.register("inference", InferenceActor())
server.serve()
```

### Calling Actors

```python
from fluster.actor import ActorPool, ClusterResolver

resolver = ClusterResolver("http://controller:8080")
pool: ActorPool = resolver.lookup("inference")
pool.wait_for_size(1)

result = pool.call().predict([1, 2, 3])
```

### Using WorkerPool for Task Dispatch

```python
from fluster.worker_pool import WorkerPool, WorkerPoolConfig
from fluster.cluster import RpcClusterClient

client = RpcClusterClient("http://controller:8080")
config = WorkerPoolConfig(num_workers=10, resources=ResourceSpec(cpu=2))
pool = WorkerPool(client, config)

futures = [pool.submit(process_shard, shard) for shard in shards]
results = [f.result() for f in futures]
pool.shutdown()
```

## Design Principles

1. **Shallow interfaces**: Components expose minimal APIs with clear responsibilities
2. **Explicit over implicit**: No magic discovery or hidden state synchronization
3. **Stateless workers**: Task retry and load balancing work because workers maintain no shared state
4. **Arbitrary callables**: Jobs and actor methods accept any picklable Python callable

## Related Documentation

- [Fray-Zero Design](docs/fray-zero.md) - Original design document and rationale
