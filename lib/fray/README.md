# Fray

Ray-like distributed execution abstraction layer for Marin.

## Overview

Fray provides a context-based API for distributed task execution that can be backed by different implementations. It was created to address issues with Ray's global state and enable easier testing and migration to alternative backends.

## Goals

- **Ray-compatible API** that can be swapped without major code changes
- **Testing without Ray cluster** via in-memory backend
- **Support for multiple backends** (in-memory, Ray, future: Dask, Kubernetes)
- **Explicit context management** instead of implicit global state
- **Separation of concerns** between job scheduling (cluster-level) and task execution (job-level)

## Quick Start

### Basic Usage

```python
from fray import get_job_context

# Auto-initializes with local backend for testing
ctx = get_job_context()

def my_task(x):
    return x * 2

ref = ctx.create_task(my_task, 5)
result = ctx.get(ref)  # Returns 10
```

### Actor Example

```python
from fray import get_job_context, ActorOptions, Lifetime

ctx = get_job_context()

class Counter:
    def __init__(self, start=0):
        self.count = start

    def increment(self):
        self.count += 1
        return self.count

# Simple actor
actor = ctx.create_actor(Counter, kwargs={"start": 5})
ref = actor.increment()
value = ctx.get(ref)  # Returns 6

# Actor with options (detached lifetime, specific resources)
options = ActorOptions(
    name="status_actor",
    get_if_exists=True,  # Reuse if exists
    lifetime=Lifetime.DETACHED,  # Survives beyond job
    resources={"CPU": 0, "head_node": 0.0001}  # Schedule on head node
)
status_actor = ctx.create_actor(StatusActor, options=options)
```

### Cluster Management

```python
from fray import LocalClusterContext
from fray.types import RuntimeEnv

cluster = LocalClusterContext()

def my_job(ctx):
    refs = [ctx.create_task(lambda x: x**2, i) for i in range(10)]
    results = ctx.get(refs)
    print(f"Results: {results}")

job_id = cluster.create_job(
    my_job,
    RuntimeEnv(package_requirements=["numpy"])
)

# List all jobs
jobs = cluster.list_jobs()
```

## Architecture

Fray separates two concerns that Ray conflates:

1. **Cluster Context** - Managing jobs and resources
2. **Job Context** - Executing tasks within a job

This separation enables:
- Clearer resource allocation boundaries
- Easier testing (jobs don't need cluster context)
- Better isolation between jobs
- Support for different scheduling backends

See `/docs/design/ray-abstraction.md` in the main Marin repo for detailed design rationale.

## API Comparison

### Ray (global state)

```python
import ray

ray.init()

@ray.remote
def task(x):
    return x * 2

ref = task.remote(5)
result = ray.get(ref)
```

### Fray (context-based)

```python
from fray import get_job_context

ctx = get_job_context()

def task(x):
    return x * 2

ref = ctx.create_task(task, 5)
result = ctx.get(ref)
```

### Resource-Based Scheduling

Fray uses resources to control scheduling instead of explicit strategies:

```python
from fray import ActorOptions, TaskOptions

# Schedule actor on head node (non-preemptible in Marin clusters)
actor_options = ActorOptions(
    resources={"CPU": 0, "head_node": 0.0001}  # Claims head_node resource
)
actor = ctx.create_actor(MyActor, options=actor_options)

# Task with specific resources
task_options = TaskOptions(
    resources={"CPU": 4, "GPU": 1, "memory": 8 * 1024**3}
)
ref = ctx.create_task(my_task, args, options=task_options)
```

**How it works:**
- Ray clusters are initialized with custom resources (e.g., `resources={"head_node": 1}`)
- Actors/tasks request these resources to be scheduled on specific nodes
- Simpler and more portable than Ray's NodeAffinitySchedulingStrategy

## Backends

### In-Memory (Testing)

The in-memory backend uses Python's `ThreadPoolExecutor` for local execution. It's automatically used when you call `get_job_context()` without setting a context.

**Characteristics:**
- Thread-based parallelism
- Shared memory (no serialization)
- No cluster required
- Perfect for unit tests

### Ray (Future)

The Ray backend will delegate to Ray's distributed execution engine for production workloads.

**Characteristics:**
- Distributed execution across machines
- Proper serialization and object store
- TPU/GPU resource management
- Production-grade fault tolerance

## Development

### Running Tests

```bash
# From the fray directory
uv run pytest tests/ -v

# From the main marin directory
uv run pytest lib/fray/tests/ -v
```

### Project Structure

```
lib/fray/
├── src/fray/
│   ├── __init__.py          # Public API
│   ├── types.py             # Core types (Resource, RuntimeEnv)
│   ├── job.py               # JobContext interface
│   ├── cluster.py           # ClusterContext interface
│   ├── context.py           # Context management (ContextVar)
│   └── backend/
│       ├── in_memory.py     # Local testing backend
│       └── ray_backend/     # Ray implementation (future)
└── tests/
    └── test_in_memory_backend.py
```

## Design Decisions

### Why Context Objects Instead of Decorators?

Ray uses `@ray.remote` decorators which create implicit global dependencies. Fray uses explicit context passing to enable:

1. **Testability** - Can swap backends easily
2. **Explicit dependencies** - Clear where distributed execution happens
3. **Thread safety** - Each context is independent
4. **Migration flexibility** - Can run multiple backends simultaneously

### Why Separate Job and Cluster Contexts?

Ray conflates job scheduling and task execution. Fray separates them to:

1. **Resource isolation** - Jobs have bounded resource allocation
2. **Better autoscaling** - Can scale per-job, not just per-cluster
3. **Testing** - Can test tasks without cluster context
4. **Multiple backends** - Different schedulers for different workload types

## Future Work

- [ ] Implement Ray backend in `ray_backend/`
- [ ] Add Dask backend for batch processing
- [ ] Support Kubernetes Jobs for long-running services
- [ ] Add resource specification translation per backend
- [ ] Implement placement groups / affinity scheduling
- [ ] Add metrics and observability hooks

## Contributing

See the main Marin repository for contribution guidelines.

## License

Apache 2.0 - See LICENSE file in the main Marin repository.
