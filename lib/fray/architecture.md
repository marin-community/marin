# Fray Architecture

## Overview

Fray is a Ray-like distributed execution abstraction layer designed to address Ray's global state issues and enable easier testing, migration, and multi-backend support. The core design principle is **explicit context management** instead of implicit global state.

## Design Philosophy

### Problems with Ray

1. **Global State**: Ray uses global state via `ray.init()` and decorators like `@ray.remote`, making testing difficult and preventing multiple Ray "sessions" in one process
2. **Tight Coupling**: Code using Ray cannot easily switch backends or run locally for testing
3. **Conflated Concerns**: Ray combines job scheduling (cluster-level) and task execution (job-level) into a single API

### Fray's Solutions

1. **Context-Based API**: All distributed operations go through explicit `JobContext` or `ClusterContext` objects
2. **Pluggable Backends**: Same API works with in-memory (testing) or Ray (production) backends
3. **Separation of Concerns**: Clear distinction between cluster management and task execution

## Core Architecture

### Layer 1: Type System

**File**: `src/fray/types.py`

Defines the core types used across all backends:

- **`ObjectRef`**: Type alias for backend-specific future/reference types
- **`ActorRef`**: Type alias for backend-specific actor handles
- **`EntryPoint`**: String representing shell commands for cluster job submission
- **`Resource`**: Named resource specification (CPU, GPU, TPU, memory)
- **`RuntimeEnv`**: Execution environment (packages, resources, env vars)
- **`ActorOptions`**: Actor creation options (name, resources, scheduling)
- **`JobInfo`**: Metadata about cluster jobs
- **`TpuRunConfig`**: TPU-specific execution configuration

### Layer 2: Abstract Interfaces

#### JobContext (`src/fray/job.py`)

Interface for task execution within a job. Provides:

```python
class JobContext(ABC):
    @abstractmethod
    def create_task(self, fn: Callable, *args, **kwargs) -> Any

    @abstractmethod
    def get(self, ref: Any) -> Any

    @abstractmethod
    def wait(self, refs: list[Any], num_returns: int = 1,
             timeout: float | None = None) -> tuple[list[Any], list[Any]]

    @abstractmethod
    def put(self, obj: Any) -> Any

    @abstractmethod
    def create_actor(self, klass: type, args: tuple = (),
                    kwargs: dict | None = None,
                    options: Any | None = None) -> Any
```

**Key difference from Ray**: Instead of decorators, functions are passed directly to `create_task()`, making the execution context explicit.

#### ClusterContext (`src/fray/cluster.py`)

Interface for managing jobs on a cluster. Provides:

```python
class ClusterContext(ABC):
    @abstractmethod
    def create_job(self, entrypoint: EntryPoint, env: RuntimeEnv) -> str

    @abstractmethod
    def list_jobs(self) -> list[JobInfo]

    @abstractmethod
    def delete_job(self, job_id: str) -> None

    @abstractmethod
    def run_on_tpu(self, fn: Callable[[JobContext], Any],
                  config: TpuRunConfig,
                  runtime_env: RuntimeEnv | None = None) -> list[Any]
```

**Key difference from Ray**: Jobs are submitted as shell commands (`EntryPoint`), not Python functions, supporting script-based execution and Docker containers.

### Layer 3: Context Management

**File**: `src/fray/context.py`

Uses Python's `contextvars` for thread-safe context management:

```python
_job_context: ContextVar[JobContext | None] = ContextVar("job_context", default=None)

def get_job_context() -> JobContext
def set_job_context(ctx: JobContext) -> None
def clear_job_context() -> None
```

**Auto-initialization**: If no context is set, `get_job_context()` automatically creates a `LocalJobContext` for testing.

### Layer 4: Backend Implementations

#### In-Memory Backend (`src/fray/backend/in_memory.py`)

For local testing without a Ray cluster:

**LocalJobContext**:
- Uses `ThreadPoolExecutor` for parallel execution
- `LocalObjectRef`: Wraps `concurrent.futures.Future`
- `LocalActorRef`: Actor instances with dedicated single-threaded executor
- Context propagation via `contextvars.copy_context()`

**LocalClusterContext**:
- Runs jobs as `subprocess.Popen` for complete testing environment
- Background threads monitor process completion
- Mocks TPU execution by running functions in parallel (one per simulated VM)

#### Ray Backend (`src/fray/backend/ray/`)

Production implementation using Ray:

**RayJobContext** (`ray_job.py`):
- `create_task()`: Wraps functions as `ray.remote()` tasks
- Context propagation: Injects `set_job_context()` calls
- `RayActorHandle`: Wraps Ray actors to provide uniform API (auto-adds `.remote()`)

**RayClusterContext** (`ray_cluster.py`):
- Uses Ray's `JobSubmissionClient` for job management
- `create_job()`: Submits shell commands with `runtime_env`
- `run_on_tpu()`: Delegates to sophisticated TPU management system

**TPU Management** (`ray_tpu.py`):
- **SlicePoolManager**: Manages pool of TPU slices
- **SliceActor**: Represents single TPU slice, manages host actors
- **TPUHostActor**: Represents single TPU host/VM
- **Features**:
  - Automatic retry on preemption (configurable max retries)
  - Health checking and actor replacement
  - Multislice coordination via `MEGASCALE_*` environment variables
  - Lockfile cleanup for libtpu
  - Flex multislice support (scale up/down as capacity available)

**Utilities** (`ray_utils.py`):
- Exception serialization with `tblib` for cross-process error handling
- `RayResources` dataclass for resource specifications
- `SnitchRecipient`: Pattern for actors to report child failures

### Layer 5: Public API

**File**: `src/fray/__init__.py`

Exports the core interfaces and implementations:

```python
__all__ = [
    "ActorOptions",
    "ClusterContext",
    "JobContext",
    "LocalClusterContext",
    "LocalJobContext",
    "Resource",
    "RuntimeEnv",
    "TpuRunConfig",
    "clear_job_context",
    "get_job_context",
    "set_job_context",
]
```

## Key Design Patterns

### 1. Context Propagation

Tasks and actors need access to the `JobContext`. Fray handles this via:

**In-Memory Backend**: Uses `contextvars.copy_context()` to capture and restore context in worker threads.

**Ray Backend**: Wraps user functions to inject `set_job_context()` before execution:

```python
def task_with_context(*task_args, **task_kwargs):
    set_job_context(current_ctx)
    return fn(*task_args, **task_kwargs)
```

### 2. Actor Handle Wrapping

To provide uniform API, both backends wrap method calls:

**In-Memory**: `_MethodWrapper` schedules methods on actor's single-threaded executor.

**Ray**: `_RayMethodWrapper` automatically appends `.remote()` to method calls.

This allows uniform syntax:
```python
actor = ctx.create_actor(Counter)
ref = actor.increment()  # No .remote() needed
result = ctx.get(ref)
```

### 3. Resource Translation

`ActorOptions.resources` uses generic names (CPU, GPU) that are translated to backend-specific formats:

- **In-Memory**: Ignored (resources meaningless in local testing)
- **Ray**: Translated to `num_cpus`, `num_gpus`, and `resources` dict

### 4. TPU Pod Scheduling

TPU management uses a hierarchical actor structure:

```
SlicePoolManager (manages N slices)
  └─ SliceActor (per slice, owns "TPU-{type}-head" resource)
      └─ TPUHostActor (per VM, owns "tpu-{name}" resource)
          └─ Remote Function (scheduled on specific node + TPU resources)
```

Scheduling flow:
1. Pool manager creates slice actors (bound to head nodes)
2. Each slice actor creates host actors (bound to specific VMs)
3. Host actors schedule remote functions using `NodeAffinitySchedulingStrategy`

## Testing Strategy

Fray enables comprehensive testing without Ray infrastructure:

1. **Unit Tests**: Use `LocalJobContext` directly
2. **Integration Tests**: Can test full pipeline logic locally
3. **Production Tests**: Switch to `RayJobContext` via environment/config

Example:
```python
# Test code
ctx = LocalJobContext()
set_job_context(ctx)

# Production code (unchanged)
ctx = get_job_context()
ref = ctx.create_task(my_function, arg)
```

## Comparison to Ray

| Feature | Ray | Fray |
|---------|-----|------|
| API Style | Decorators + global state | Explicit context objects |
| Testing | Requires cluster | In-memory backend |
| Context Access | `ray.init()` sets global | `get_job_context()` from ContextVar |
| Job Submission | Python functions | Shell commands (EntryPoint) |
| Actor Methods | Require `.remote()` | Wrapped to look like regular calls |
| Multiple Backends | No | Yes (in-memory, Ray, future: Dask, K8s) |

## Directory Structure

```
lib/fray/
├── src/fray/
│   ├── __init__.py          # Public API
│   ├── types.py             # Core types
│   ├── job.py               # JobContext interface
│   ├── cluster.py           # ClusterContext interface
│   ├── context.py           # ContextVar management
│   └── backend/
│       ├── __init__.py
│       ├── in_memory.py     # Local testing backend
│       └── ray/
│           ├── ray_cluster.py   # Ray ClusterContext
│           ├── ray_job.py       # Ray JobContext
│           ├── ray_tpu.py       # TPU management
│           └── ray_utils.py     # Ray utilities
└── tests/
    ├── conftest.py
    ├── test_backend.py
    └── test_tpu_interface.py
```

## Future Enhancements

Based on README.md, planned improvements include:

1. **Additional Backends**: Dask (batch processing), Kubernetes Jobs (long-running services)
2. **Resource Management**: Better placement groups and affinity scheduling
3. **Observability**: Metrics and tracing hooks
4. **Flex Multislice**: Dynamic scaling of TPU slice count based on availability

## References

- See `/docs/design/ray-abstraction.md` in main Marin repo for design rationale
- Ray TPU management ported from Levanter with enhancements
- Multislice coordination uses JAX's MEGASCALE environment variables
