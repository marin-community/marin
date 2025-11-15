# Fray

Execution contexts and cluster abstraction for distributed and parallel computing.

## Overview

Fray provides two main abstractions:

### Execution Contexts

A common interface (`ExecutionContext`) for different execution strategies:

- **SyncContext**: Synchronous execution in the current thread
- **ThreadContext**: Parallel execution using ThreadPoolExecutor
- **RayContext**: Distributed execution using Ray

All contexts implement the same protocol with `put`, `get`, `run`, and `wait` primitives, allowing code to be execution-agnostic.

### Cluster API

A unified interface for job scheduling across different cluster backends:

- **LocalCluster**: Local subprocess-based execution for development and testing
- **RayCluster**: Distributed job execution on Ray clusters

The Cluster API provides a clean abstraction for launching, monitoring, and managing jobs with support for CPU, GPU, and TPU resources.

## Installation

```bash
# Base installation (sync and threadpool contexts)
pip install fray

# With Ray support
pip install fray[ray]
```

## Usage

### Execution Contexts

```python
from fray import create_context

# Create a synchronous context
ctx = create_context("sync")

# Create a thread pool context
ctx = create_context("threadpool", max_workers=4)

# Create a Ray context (requires ray to be installed)
ctx = create_context("ray", memory=2*1024**3, num_cpus=2)

# Use the context
ref = ctx.put({"data": [1, 2, 3]})
future = ctx.run(lambda x: sum(x["data"]), ref)
result = ctx.get(future)  # Returns 6
```

### Cluster API

```python
from fray.cluster import (
    LocalCluster,
    RayCluster,
    JobRequest,
    ResourceConfig,
    TpuConfig,
    create_environment,
)

# Launch a job on LocalCluster
cluster = LocalCluster()
request = JobRequest(
    name="my-job",
    entrypoint="my_module.main",
    entrypoint_args=["--config", "path/to/config.yaml"],
    environment=create_environment(),
)
job_id = cluster.launch(request)

# Monitor job logs
for line in cluster.monitor(job_id):
    print(line)

# Check job status
info = cluster.poll(job_id)
print(f"Status: {info.status}")

# Launch a TPU job on Ray
ray_cluster = RayCluster()
tpu_request = JobRequest(
    name="tpu-training",
    entrypoint="train",
    resources=ResourceConfig(
        cpu=96,
        ram="512g",
        device=TpuConfig(type="v5e-16", count=8),
    ),
    environment=create_environment(
        extra_dependency_groups=["tpu"],
        env_vars={"WANDB_API_KEY": "your-key"},
    ),
)
job_id = ray_cluster.launch(tpu_request)

# Use cluster API for ray.remote configuration
import ray

runtime_env = ray_cluster.get_runtime_env(tpu_request)
resources = ray_cluster.get_ray_resources(tpu_request)

@ray.remote(runtime_env=runtime_env, resources=resources)
def train_model():
    # Your training code here
    pass
```

## Context Protocol

All contexts implement the `ExecutionContext` protocol:

```python
class ExecutionContext(Protocol):
    def put(self, obj: Any) -> Any:
        """Store an object and return a reference."""
        ...

    def get(self, ref: Any) -> Any:
        """Retrieve an object from its reference."""
        ...

    def run(self, fn: Callable, *args) -> Any:
        """Execute a function with arguments and return a future."""
        ...

    def wait(self, futures: list, num_returns: int = 1) -> tuple[list, list]:
        """Wait for futures to complete."""
        ...
```

## License

Apache License 2.0
