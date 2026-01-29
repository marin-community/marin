# FrayV2 Implementation Plan

## Overview

Implement FrayV2 as an **Iris-shaped API** in `lib/fray/src/fray/v2/` with three backends:
1. **Local** - In-process execution for CI/development (no external dependencies)
2. **Iris** - Delegate to `IrisClient` (local and cluster modes)
3. **Ray** - Adapter that makes Ray behave like Iris

**Design Principle**: FrayV2 is Iris-first. When there are API conflicts between Iris and Ray, FrayV2 follows Iris. The Ray backend adapts to match Iris semantics.

**Lifecycle**: Once all consumers migrate from v1 to v2, v1 will be deleted and the `v2` namespace will be removed (v2 becomes the main API).

---

## Module Structure

```
lib/fray/src/fray/v2/
├── __init__.py              # Public exports
├── types.py                 # Fresh types: JobId, JobStatus, ResourceSpec, etc.
├── cluster.py               # Cluster protocol + current_cluster() factory
├── job.py                   # Job handle class
├── actor/
│   ├── __init__.py
│   ├── server.py            # ActorServer
│   ├── pool.py              # ActorPool, BroadcastResult
│   └── resolver.py          # Resolver protocol
├── worker_pool.py           # WorkerPool context manager
└── backends/
    ├── __init__.py
    ├── local.py             # LocalCluster (thread-based, no deps)
    ├── iris.py              # IrisCluster (wraps IrisClient)
    └── ray.py               # RayCluster (adapts Ray to Iris semantics)
```

---

## Core Types (`types.py`)

Fresh types aligned with Iris. No v1 imports.

```python
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, NewType, Sequence

# Type aliases
JobId = NewType("JobId", str)
Namespace = NewType("Namespace", str)

class JobStatus(StrEnum):
    """Job lifecycle states."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    KILLED = "killed"

@dataclass
class ResourceSpec:
    """Resource specification (Iris-aligned).

    Memory/disk accept human-readable strings ("8g", "512m") or bytes.
    Device uses Iris's DeviceConfig proto pattern but simplified for Python.
    """
    cpu: int = 0
    memory: str | int = 0
    disk: str | int = 0
    replicas: int = 1
    preemptible: bool = False
    regions: Sequence[str] | None = None

    # Device config (simplified from Iris proto)
    device_type: str | None = None      # "tpu" or "gpu"
    device_variant: str | None = None   # "v5litepod-16", "H100", etc.
    device_count: int = 0

    @classmethod
    def with_tpu(cls, variant: str, replicas: int = 1, **kw) -> "ResourceSpec":
        return cls(device_type="tpu", device_variant=variant, replicas=replicas, **kw)

    @classmethod
    def with_gpu(cls, variant: str = "auto", count: int = 1, **kw) -> "ResourceSpec":
        return cls(device_type="gpu", device_variant=variant, device_count=count, **kw)

@dataclass
class EnvironmentSpec:
    """Environment configuration (Iris-aligned)."""
    workspace: str | None = None
    pip_packages: Sequence[str] | None = None
    env_vars: dict[str, str] | None = None
    extras: Sequence[str] | None = None

@dataclass
class Entrypoint:
    """Job entrypoint (Iris-aligned)."""
    callable: Callable[..., Any]
    args: tuple = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_callable(cls, fn: Callable, *args, **kwargs) -> "Entrypoint":
        return cls(callable=fn, args=args, kwargs=kwargs)
```

---

## Cluster Protocol (`cluster.py`)

Iris-shaped cluster interface.

```python
from typing import Protocol, runtime_checkable
from concurrent.futures import Future

@runtime_checkable
class Job(Protocol):
    """Job handle (Iris-aligned)."""

    @property
    def job_id(self) -> JobId: ...

    def status(self) -> JobStatus: ...

    def wait(
        self,
        timeout: float = 300.0,
        *,
        stream_logs: bool = False,
        raise_on_failure: bool = True,
    ) -> JobStatus: ...

    def terminate(self) -> None: ...

@runtime_checkable
class Cluster(Protocol):
    """Cluster protocol (Iris-aligned).

    Key differences from v1:
    - submit() returns Job handle (not JobId)
    - resolver() for actor discovery
    - worker_pool() for task dispatch
    """

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: ResourceSpec,
        environment: EnvironmentSpec | None = None,
    ) -> Job: ...

    def terminate(self, job_id: JobId) -> None: ...

    def list_jobs(self) -> list[Job]: ...

    def resolver(self) -> "Resolver": ...

    def worker_pool(
        self,
        num_workers: int,
        resources: ResourceSpec,
        environment: EnvironmentSpec | None = None,
    ) -> "WorkerPool": ...

@runtime_checkable
class Resolver(Protocol):
    """Actor discovery (Iris-aligned)."""

    def lookup(self, name: str) -> "ActorPool": ...

@runtime_checkable
class ActorPool(Protocol):
    """Actor pool for RPC (Iris-aligned)."""

    @property
    def size(self) -> int: ...

    def wait_for_size(self, min_size: int, timeout: float = 60.0) -> None: ...

    def call(self) -> Any:
        """Round-robin call to one actor."""
        ...

    def broadcast(self) -> "BroadcastResult":
        """Fan-out call to all actors."""
        ...

@runtime_checkable
class WorkerPool(Protocol):
    """Task dispatch pool (Iris-aligned)."""

    def submit(self, fn: Callable, *args, **kwargs) -> Future: ...

    def shutdown(self, wait: bool = True) -> None: ...

    def __enter__(self) -> "WorkerPool": ...
    def __exit__(self, *args) -> None: ...

def current_cluster() -> Cluster:
    """Get cluster from FRAY_CLUSTER_SPEC env var.

    Spec formats:
    - "" or "local" -> LocalCluster
    - "iris" or "iris?address=http://..." -> IrisCluster
    - "ray" or "ray?namespace=xyz" -> RayCluster
    """
    ...

def create_cluster(spec: str = "local") -> Cluster:
    """Create cluster from spec string."""
    ...
```

---

## Backend Implementations

### Local Backend (`backends/local.py`)

Thread-based execution with no external dependencies. Perfect for CI.

**Key Features:**
- Jobs run as threads
- ActorServer uses in-process calls with cloudpickle serialization (catches bugs)
- Resolver uses thread-safe in-memory registry
- WorkerPool uses `ThreadPoolExecutor`

```python
class LocalCluster:
    """In-process cluster for testing."""

    def __init__(self):
        self._jobs: dict[JobId, LocalJob] = {}
        self._registry: dict[str, list[LocalEndpoint]] = {}
        self._executor = ThreadPoolExecutor()

    def submit(self, entrypoint, name, resources, environment=None) -> Job:
        job = LocalJob(entrypoint, name)
        job.start()  # Runs in thread
        self._jobs[job.job_id] = job
        return job

    def resolver(self) -> Resolver:
        return LocalResolver(self._registry)

    def worker_pool(self, num_workers, resources, environment=None) -> WorkerPool:
        return LocalWorkerPool(num_workers, self._executor)
```

### Iris Backend (`backends/iris.py`)

Thin wrapper around `IrisClient`.

**Key Features:**
- Re-exports Iris's ActorServer, ActorPool, Resolver directly
- Type conversion functions: `fray_to_iris_resources()`, etc.
- Supports both `IrisClient.local()` and `IrisClient.remote()`

```python
from iris.client import IrisClient, Job as IrisJob
from iris.actor import ActorServer as IrisActorServer
from iris.cluster.types import ResourceSpec as IrisResourceSpec

class IrisCluster:
    """Iris-backed cluster."""

    def __init__(self, client: IrisClient):
        self._client = client

    @classmethod
    def local(cls) -> "IrisCluster":
        return cls(IrisClient.local())

    @classmethod
    def remote(cls, address: str, workspace: Path | None = None) -> "IrisCluster":
        return cls(IrisClient.remote(address, workspace=workspace))

    def submit(self, entrypoint, name, resources, environment=None) -> Job:
        iris_entrypoint = _to_iris_entrypoint(entrypoint)
        iris_resources = _to_iris_resources(resources)
        iris_env = _to_iris_environment(environment) if environment else None

        iris_job = self._client.submit(iris_entrypoint, name, iris_resources, iris_env)
        return IrisJobAdapter(iris_job)

    def resolver(self) -> Resolver:
        return self._client.resolver()  # Direct pass-through

    def worker_pool(self, num_workers, resources, environment=None) -> WorkerPool:
        from iris.client import WorkerPool as IrisWorkerPool, WorkerPoolConfig
        config = WorkerPoolConfig(num_workers=num_workers, resources=_to_iris_resources(resources))
        return IrisWorkerPool(self._client, config)
```

### Ray Backend (`backends/ray.py`)

Adapts Ray to behave like Iris. Uses Ray's object store internally for efficiency, but does NOT expose `ObjectRef` types in the public API.

**Key Features:**
- Jobs: Creates proper Ray jobs via JobSubmissionClient
- ActorServer: Ray named actors with `actor.method.remote()` + `ray.get()`
- Resolver: Uses `ray.get_actor(name)` for discovery
- WorkerPool: Ray actors as explicit workers, wraps `ObjectRef` in `Future`

**Internal Ray usage** (not exposed in public API):
- `ray.put()` for large task arguments
- `ray.get()` to retrieve results from `ObjectRef`
- `actor.method.remote()` for actor calls

```python
class RayCluster:
    """Ray cluster with Iris-compatible semantics."""

    def __init__(self, namespace: str | None = None):
        self._namespace = namespace or "fray"
        ray.init(ignore_reinit_error=True)

    def submit(self, entrypoint, name, resources, environment=None) -> Job:
        # Submit via Ray Jobs API (not ray.remote)
        from ray.job_submission import JobSubmissionClient
        client = JobSubmissionClient()
        job_id = client.submit_job(...)
        return RayJob(job_id, client)

    def resolver(self) -> Resolver:
        return RayResolver(self._namespace)

    def worker_pool(self, num_workers, resources, environment=None) -> WorkerPool:
        return RayWorkerPool(num_workers, resources)

class RayActorServer:
    """Ray actor server using named actors."""

    def register(self, name: str, actor: Any):
        # Create Ray actor with name
        ActorClass = ray.remote(type(actor))
        self._actors[name] = ActorClass.options(
            name=f"{self._namespace}/{name}",
            get_if_exists=False,
        ).remote()

class RayResolver:
    """Ray resolver using ray.get_actor()."""

    def lookup(self, name: str) -> ActorPool:
        prefixed = f"{self._namespace}/{name}"
        actor = ray.get_actor(prefixed)
        return RayActorPool([actor])
```

---

## ActorServer (`actor/server.py`)

Unified interface that delegates to backend implementations.

```python
class ActorServer:
    """Actor server for hosting RPC services.

    Example:
        server = ActorServer()
        server.register("counter", Counter())
        server.serve()  # Blocks

        # Or background:
        server.serve_background()
    """

    def __init__(self, cluster: Cluster | None = None, host: str = "0.0.0.0", port: int = 0):
        self._cluster = cluster or current_cluster()
        self._impl = self._cluster._create_actor_server(host, port)

    def register(self, name: str, actor: Any) -> None:
        """Register an actor instance."""
        self._impl.register(name, actor)

    def serve(self) -> None:
        """Start serving (blocks)."""
        self._impl.serve()

    def serve_background(self) -> int:
        """Start serving in background thread, return port."""
        return self._impl.serve_background()
```

---

## Key Files to Create

| File | Description |
|------|-------------|
| `lib/fray/src/fray/v2/__init__.py` | Public exports |
| `lib/fray/src/fray/v2/types.py` | Fresh types (no v1 imports) |
| `lib/fray/src/fray/v2/cluster.py` | Cluster protocol + factory |
| `lib/fray/src/fray/v2/job.py` | Job handle |
| `lib/fray/src/fray/v2/actor/__init__.py` | Actor exports |
| `lib/fray/src/fray/v2/actor/server.py` | ActorServer |
| `lib/fray/src/fray/v2/actor/pool.py` | ActorPool, BroadcastResult |
| `lib/fray/src/fray/v2/actor/resolver.py` | Resolver, FixedResolver |
| `lib/fray/src/fray/v2/worker_pool.py` | WorkerPool |
| `lib/fray/src/fray/v2/backends/__init__.py` | Backend exports |
| `lib/fray/src/fray/v2/backends/local.py` | LocalCluster |
| `lib/fray/src/fray/v2/backends/iris.py` | IrisCluster |
| `lib/fray/src/fray/v2/backends/ray.py` | RayCluster |
| `lib/fray/tests/v2/test_types.py` | Unit tests for types |
| `lib/fray/tests/v2/test_local.py` | Unit tests for LocalCluster |
| `tests/test_frayv2_integration.py` | Integration tests |

---

## Implementation Phases

### Phase 1: Core Types + Local Backend
1. Create directory structure
2. Implement `types.py` with fresh types
3. Implement `cluster.py` with protocols + `current_cluster()`
4. Implement `backends/local.py`:
   - `LocalCluster` with thread-based jobs
   - `LocalResolver` with in-memory registry
   - `LocalWorkerPool` with `ThreadPoolExecutor`
   - `LocalActorServer` with cloudpickle serialization
5. Write unit tests: `lib/fray/tests/v2/test_local.py`

### Phase 2: Iris Backend
1. Implement `backends/iris.py`:
   - Type conversion functions
   - `IrisCluster` wrapping `IrisClient`
   - Pass-through for ActorServer, Resolver, WorkerPool
2. Add `"iris"` to `create_cluster()` parsing
3. Write tests with Iris local mode

### Phase 3: Ray Backend
1. Implement `backends/ray.py`:
   - `RayCluster` using Jobs API (not `ray.remote`)
   - `RayActorServer` with named actors
   - `RayResolver` using `ray.get_actor()`
   - `RayWorkerPool` with explicit worker actors
2. Add `"ray"` to `create_cluster()` parsing
3. Write Ray-specific tests

### Phase 4: Integration Tests
1. Parameterized tests for all backends
2. Parity verification tests

---

## Testing Strategy

### Unit Tests (`lib/fray/tests/v2/`)

```python
# test_types.py
def test_resource_spec_with_tpu():
    spec = ResourceSpec.with_tpu("v5litepod-16", replicas=2)
    assert spec.device_type == "tpu"
    assert spec.device_variant == "v5litepod-16"
    assert spec.replicas == 2

# test_local.py
def test_local_job_submit():
    cluster = LocalCluster()

    def my_job():
        return 42

    job = cluster.submit(Entrypoint.from_callable(my_job), "test", ResourceSpec())
    status = job.wait()
    assert status == JobStatus.SUCCEEDED

def test_local_worker_pool():
    cluster = LocalCluster()

    with cluster.worker_pool(num_workers=2, resources=ResourceSpec()) as pool:
        futures = [pool.submit(lambda x: x*x, i) for i in range(5)]
        results = [f.result() for f in futures]

    assert sorted(results) == [0, 1, 4, 9, 16]

def test_local_actor_roundtrip():
    cluster = LocalCluster()

    class Counter:
        def __init__(self):
            self.value = 0
        def incr(self, n=1):
            self.value += n
            return self.value

    server = ActorServer(cluster)
    server.register("counter", Counter())
    server.serve_background()

    pool = cluster.resolver().lookup("counter")
    pool.wait_for_size(1)

    assert pool.call().incr(5) == 5
    assert pool.call().incr(3) == 8
```

### Integration Tests (`tests/test_frayv2_integration.py`)

```python
@pytest.fixture(params=["local", "iris", "ray"])
def cluster(request):
    spec = request.param
    if spec == "iris":
        pytest.importorskip("iris")
    if spec == "ray":
        pytest.importorskip("ray")
    return create_cluster(spec)

def test_job_lifecycle(cluster):
    job = cluster.submit(
        Entrypoint.from_callable(lambda: 42),
        name="test-job",
        resources=ResourceSpec(),
    )
    status = job.wait()
    assert status == JobStatus.SUCCEEDED

def test_worker_pool_map(cluster):
    def square(x):
        return x * x

    with cluster.worker_pool(num_workers=2, resources=ResourceSpec()) as pool:
        futures = [pool.submit(square, i) for i in range(10)]
        results = sorted(f.result() for f in futures)

    assert results == [i*i for i in range(10)]
```

---

## Verification

Run tests with:

```bash
# Unit tests (local only, fast)
uv run pytest lib/fray/tests/v2/ -v

# Integration tests (all backends)
uv run pytest tests/test_frayv2_integration.py -v

# Quick smoke test
uv run python -c "
from fray.v2 import current_cluster, Entrypoint, ResourceSpec

cluster = current_cluster()
with cluster.worker_pool(num_workers=2, resources=ResourceSpec()) as pool:
    futures = [pool.submit(lambda x: x*x, i) for i in range(5)]
    results = [f.result() for f in futures]
print('Results:', results)
"
```

Expected output: `Results: [0, 1, 4, 9, 16]`
