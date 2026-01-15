# Fluster Context Management v0

## Overview

This document proposes a unified context management system for Fluster that:
1. Provides a single `fluster_ctx()` ContextVar for accessing execution context
2. Enables transparent local-vs-remote execution (LocalClient runs jobs in threads with proper context)
3. Simplifies and unifies namespace handling
4. Removes scattered `os.environ` reads in favor of context-based access

## Current State Problems

### 1. Scattered Environment Variable Reads

Multiple locations read directly from `os.environ`:

```python
# actor/resolver.py:109
self._namespace = namespace or Namespace(os.environ.get("FLUSTER_NAMESPACE", "<local>"))

# actor/resolver.py:232
self._namespace = namespace or Namespace(os.environ.get("FLUSTER_NAMESPACE", "<local>"))

# worker_pool.py:212-215
job_id = os.environ["FLUSTER_JOB_ID"]
namespace = os.environ["FLUSTER_NAMESPACE"]
port = int(os.environ["FLUSTER_PORT_ACTOR"])
controller_url = os.environ["FLUSTER_CONTROLLER_ADDRESS"]
```

This makes the code fragile and prevents local execution without setting environment variables.

### 2. Confusing `"<local>"` Namespace

The string `"<local>"` appears as default in 10+ locations:
- `FixedResolver` default
- `ClusterResolver` fallback
- `GcsResolver` fallback
- `WorkerPool._namespace`
- `ClusterClient.submit()` default
- `RpcClusterClient.submit()` default
- `Controller service.py` fallbacks

This magic string is confusing: it's not clear when `"<local>"` should be used vs when a real namespace should be inherited.

### 3. No LocalClient for Thread-based Execution

There's no way to run fluster jobs locally in threads. The only client is `RpcClusterClient` which requires a running controller. For testing and local development, users need a `LocalClient` that:
- Runs jobs in the current process (in threads)
- Sets appropriate context per job
- Works identically to the RPC path

### 4. Limited ActorContext

The existing `ActorContext` (in `actor/server.py`) is limited:
```python
@dataclass
class ActorContext:
    cluster: Any
    resolver: Resolver | None
    job_id: str
    namespace: str
```

It's only available during actor method calls and doesn't provide access to worker_id or other execution metadata.

## Proposed Design

### Core Context: `FlusterContext`

A new unified context dataclass with all execution information:

```python
# fluster/client/context.py

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class FlusterContext:
    """Unified execution context for Fluster.

    Available in any fluster job via `fluster_ctx()`. Contains all
    information about the current execution environment.
    """
    namespace: Namespace
    job_id: str
    worker_id: str | None
    controller: "ClusterController | None"  # Protocol for cluster ops
    ports: dict[str, int] = field(default_factory=dict)  # Allocated ports: name -> port

    # Derived accessors
    @property
    def resolver(self) -> "Resolver":
        """Get a resolver for actor discovery."""
        if self.controller is None:
            return NullResolver()
        return self.controller.resolver(self.namespace)

    def get_port(self, name: str) -> int:
        """Get an allocated port by name. Raises KeyError if not allocated."""
        return self.ports[name]

# Module-level ContextVar
_fluster_context: ContextVar[FlusterContext | None] = ContextVar(
    "fluster_context",
    default=None
)

def fluster_ctx() -> FlusterContext:
    """Get the current fluster context.

    Raises RuntimeError if called outside a fluster job.
    """
    ctx = _fluster_context.get()
    if ctx is None:
        raise RuntimeError(
            "fluster_ctx() called outside of a fluster job. "
            "Wrap your code in a fluster_ctx_scope() or run via LocalClient/RpcClusterClient."
        )
    return ctx

def get_fluster_ctx() -> FlusterContext | None:
    """Get the current context, or None if not in a job."""
    return _fluster_context.get()

@contextmanager
def fluster_ctx_scope(ctx: FlusterContext) -> Generator[FlusterContext, None, None]:
    """Set the fluster context for the duration of this scope."""
    token = _fluster_context.set(ctx)
    try:
        yield ctx
    finally:
        _fluster_context.reset(token)
```

### ClusterController Protocol

A protocol that abstracts controller operations:

```python
# fluster/context.py

class ClusterController(Protocol):
    """Protocol for cluster operations."""

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: ResourceSpec,
        environment: EnvironmentConfig | None = None,
        ports: list[str] | None = None,
    ) -> JobId:
        """Submit a job (namespace is inherited from context)."""
        ...

    def status(self, job_id: JobId) -> JobStatus: ...
    def wait(self, job_id: JobId, timeout: float = 300.0) -> JobStatus: ...
    def terminate(self, job_id: JobId) -> None: ...
    def resolver(self, namespace: Namespace) -> Resolver: ...

    @property
    def address(self) -> str: ...
```

### EndpointRegistry Protocol

A protocol for actor endpoint registration, used by `ActorServer`:

```python
# fluster/context.py

class EndpointRegistry(Protocol):
    """Protocol for registering actor endpoints."""

    def register(
        self,
        name: str,
        address: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Register an endpoint. Returns endpoint_id."""
        ...

    def unregister(self, endpoint_id: str) -> None:
        """Unregister an endpoint."""
        ...
```

The `ClusterController` provides access to an `EndpointRegistry`:

```python
class ClusterController(Protocol):
    # ... other methods ...

    @property
    def endpoint_registry(self) -> EndpointRegistry:
        """Get the endpoint registry for actor registration."""
        ...
```

### ActorServer Self-Registration

`ActorServer` handles its own registration using the context:

```python
# actor/server.py (updated)

class ActorServer:
    def serve_background(self, actor_name: str) -> int:
        """Start server and register with controller.

        Uses fluster_ctx() to get port allocation and endpoint registry.
        """
        ctx = fluster_ctx()

        # Get allocated port from context
        port = ctx.get_port("actor")

        # Start serving
        self._start_serving(port)

        # Register endpoint using registry from context
        if ctx.controller:
            self._endpoint_id = ctx.controller.endpoint_registry.register(
                name=actor_name,
                address=f"{self._get_hostname()}:{self._actual_port}",
                metadata={"job_id": ctx.job_id},
            )

        return self._actual_port

    def shutdown(self) -> None:
        """Stop server and unregister endpoint."""
        ctx = get_fluster_ctx()
        if self._endpoint_id and ctx and ctx.controller:
            ctx.controller.endpoint_registry.unregister(self._endpoint_id)
        # ... stop server ...
```

This allows `WorkerPool` to work with any cluster client:

```python
# worker_pool.py (simplified)

def worker_job_entrypoint(pool_id: str, worker_index: int) -> None:
    """Job entrypoint - ActorServer handles registration."""
    worker_name = f"_workerpool_{pool_id}:worker-{worker_index}"

    # Start actor server - it uses fluster_ctx() internally
    server = ActorServer(host="0.0.0.0")
    server.register(worker_name, TaskExecutorActor())
    server.serve_background(worker_name)  # Uses ctx.get_port("actor"), ctx.controller.endpoint_registry

    # Serve forever
    while True:
        time.sleep(1)
```

### Namespace Rules

Clear, explicit namespace inheritance:

1. **Jobs inherit namespace from their submitter's context**
   - If submitted from context with namespace "prod", job runs with namespace "prod"
   - No separate `namespace` parameter in `ClusterController.submit()`

2. **Root namespace for cluster initialization**
   - When creating a cluster/controller, specify the root namespace
   - All jobs submitted through that cluster inherit this namespace

3. **No more `"<local>"` magic string**
   - Replace with `Namespace.DEFAULT = Namespace("default")`
   - Clear semantics: "default" is the default namespace, not a special local-only mode

```python
# fluster/cluster/types.py

class Namespace(str):
    """Namespace for actor isolation."""
    DEFAULT: ClassVar["Namespace"]

Namespace.DEFAULT = Namespace("default")
```

### LocalClient Implementation

A client that runs jobs locally in threads with full actor support:

```python
# fluster/cluster/local_client.py

@dataclass
class LocalClientConfig:
    """Configuration for local job execution."""
    max_workers: int = 4
    namespace: Namespace = Namespace.DEFAULT
    port_range: tuple[int, int] = (50000, 60000)  # Port range for actor servers

class LocalEndpointRegistry:
    """In-memory endpoint registry for local execution."""

    def __init__(self):
        self._endpoints: dict[str, tuple[str, str, dict]] = {}  # id -> (name, address, metadata)
        self._lock = threading.RLock()

    def register(self, name: str, address: str, metadata: dict[str, str] | None = None) -> str:
        endpoint_id = f"local-ep-{uuid.uuid4().hex[:8]}"
        with self._lock:
            self._endpoints[endpoint_id] = (name, address, metadata or {})
        return endpoint_id

    def unregister(self, endpoint_id: str) -> None:
        with self._lock:
            self._endpoints.pop(endpoint_id, None)

    def lookup(self, name: str) -> list[tuple[str, str, dict]]:
        """Return all endpoints with matching name."""
        with self._lock:
            return [(addr, eid, meta) for eid, (n, addr, meta) in self._endpoints.items() if n == name]

class LocalControllerAdapter:
    """ClusterController implementation for LocalClient."""

    def __init__(self, client: "LocalClient"):
        self._client = client
        self._registry = LocalEndpointRegistry()

    def submit(self, entrypoint: Entrypoint, name: str, resources: ResourceSpec, ...) -> JobId:
        return self._client.submit(entrypoint, name, resources, ...)

    def status(self, job_id: JobId) -> cluster_pb2.JobStatus:
        return self._client.status(job_id)

    def wait(self, job_id: JobId, timeout: float = 300.0) -> cluster_pb2.JobStatus:
        return self._client.wait(job_id, timeout)

    def terminate(self, job_id: JobId) -> None:
        self._client.terminate(job_id)

    def resolver(self, namespace: Namespace) -> Resolver:
        return LocalResolver(self._registry, namespace)

    @property
    def endpoint_registry(self) -> EndpointRegistry:
        return self._registry

    @property
    def address(self) -> str:
        return "local://localhost"

class LocalResolver:
    """Resolver backed by LocalEndpointRegistry."""

    def __init__(self, registry: LocalEndpointRegistry, namespace: Namespace):
        self._registry = registry
        self._namespace = namespace

    @property
    def default_namespace(self) -> Namespace:
        return self._namespace

    def resolve(self, name: str, namespace: Namespace | None = None) -> ResolveResult:
        ns = namespace or self._namespace
        matches = self._registry.lookup(name)
        endpoints = [
            ResolvedEndpoint(url=f"http://{addr}", actor_id=eid, metadata=meta)
            for addr, eid, meta in matches
        ]
        return ResolveResult(name=name, namespace=ns, endpoints=endpoints)

class LocalClient:
    """Run fluster jobs locally in threads.

    Jobs execute in the current process with proper context injection.
    Supports full actor functionality via LocalEndpointRegistry.

    Example:
        config = LocalClientConfig(max_workers=4, namespace=Namespace("test"))
        with LocalClient(config) as client:
            # Submit a job - runs in a thread with fluster_ctx() available
            job_id = client.submit(my_entrypoint, "my-job", resources, ports=["actor"])
            client.wait(job_id)

            # WorkerPool works too!
            pool_config = WorkerPoolConfig(num_workers=3, resources=...)
            with WorkerPool(client, pool_config) as pool:
                future = pool.submit(my_fn, arg)
                result = future.result()
    """

    def __init__(self, config: LocalClientConfig | None = None):
        self._config = config or LocalClientConfig()
        self._executor: ThreadPoolExecutor | None = None
        self._jobs: dict[JobId, _LocalJob] = {}
        self._lock = threading.RLock()
        self._job_counter = 0
        self._next_port = self._config.port_range[0]
        self._controller = LocalControllerAdapter(self)

    def __enter__(self) -> "LocalClient":
        self._executor = ThreadPoolExecutor(max_workers=self._config.max_workers)
        return self

    def __exit__(self, *_):
        self.shutdown()

    @property
    def controller_address(self) -> str:
        """For compatibility with ClusterClient protocol."""
        return "local://localhost"

    def _allocate_port(self) -> int:
        """Allocate a port from the configured range."""
        with self._lock:
            port = self._next_port
            self._next_port += 1
            if self._next_port > self._config.port_range[1]:
                self._next_port = self._config.port_range[0]
        return port

    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: ResourceSpec,
        environment: EnvironmentConfig | None = None,
        ports: list[str] | None = None,
    ) -> JobId:
        """Submit a job for local execution."""
        with self._lock:
            self._job_counter += 1
            job_id = JobId(f"local-{self._job_counter}")

        # Allocate requested ports
        allocated_ports = {port_name: self._allocate_port() for port_name in (ports or [])}

        # Create context for this job
        ctx = FlusterContext(
            namespace=self._config.namespace,
            job_id=job_id,
            worker_id=f"local-worker-{threading.current_thread().ident}",
            controller=self._controller,
            ports=allocated_ports,
        )

        # Submit to thread pool with context
        future = self._executor.submit(
            self._run_job_with_context,
            ctx,
            entrypoint
        )

        with self._lock:
            self._jobs[job_id] = _LocalJob(
                job_id=job_id,
                future=future,
                state=JobState.RUNNING,
            )

        return job_id

    def _run_job_with_context(
        self,
        ctx: FlusterContext,
        entrypoint: Entrypoint
    ) -> Any:
        """Execute entrypoint with fluster context injected."""
        with fluster_ctx_scope(ctx):
            return entrypoint.callable(*entrypoint.args, **entrypoint.kwargs)
```

### Updating Existing Code

#### 1. ActorClient - Use Context Instead of os.environ

```python
# actor/client.py (current)
# No os.environ usage, but uses resolver directly

# actor/resolver.py (updated)
class ClusterResolver:
    def __init__(
        self,
        controller_address: str,
        namespace: Namespace | None = None,  # If None, reads from context
        timeout: float = 5.0,
    ):
        self._address = controller_address.rstrip("/")
        self._timeout = timeout
        # Use context if no namespace provided
        if namespace is not None:
            self._namespace = namespace
        else:
            ctx = get_fluster_ctx()
            self._namespace = ctx.namespace if ctx else Namespace.DEFAULT
```

#### 2. worker_pool.py - Use Context

```python
# worker_pool.py (updated)
def worker_job_entrypoint(pool_id: str, worker_index: int) -> None:
    """Job entrypoint that reads context from fluster_ctx()."""
    ctx = fluster_ctx()  # Raises if not set

    job_id = ctx.job_id
    namespace = ctx.namespace

    # Port still from env (allocated by worker)
    port = int(os.environ["FLUSTER_PORT_ACTOR"])

    # Controller from context
    controller_url = ctx.controller.address if ctx.controller else os.environ["FLUSTER_CONTROLLER_ADDRESS"]

    # ... rest of function
```

#### 3. Worker Job Execution - Set Context

```python
# cluster/worker/worker.py (updated)
def _execute_job(self, job: Job) -> None:
    """Execute a job with context injection."""
    # Build context
    ctx = FlusterContext(
        namespace=Namespace(env.get("FLUSTER_NAMESPACE", Namespace.DEFAULT)),
        job_id=job.job_id,
        worker_id=self._config.worker_id,
        controller=self._create_controller_client(),  # RPC client to controller
    )

    # Inject context before running entrypoint
    with fluster_ctx_scope(ctx):
        # Run the job
        entrypoint()
```

#### 4. RpcClusterClient - Namespace from Context

```python
# cluster/client.py (updated)
class RpcClusterClient:
    def submit(
        self,
        entrypoint: Entrypoint,
        name: str,
        resources: cluster_pb2.ResourceSpec,
        environment: cluster_pb2.EnvironmentConfig | None = None,
        # No namespace param - inherited from context
        ports: list[str] | None = None,
    ) -> JobId:
        # Get namespace from current context, or use default
        ctx = get_fluster_ctx()
        namespace = ctx.namespace if ctx else Namespace.DEFAULT

        # ... rest of function
```

## Implementation Plan

### Phase 1: Core Context Infrastructure

**Files to create:**
- `src/fluster/client/context.py` - FlusterContext, fluster_ctx(), ClusterController, EndpointRegistry protocols

**Files to modify:**
- `src/fluster/cluster/types.py` - Add Namespace.DEFAULT

```python
# client/context.py skeleton
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generator, Protocol

if TYPE_CHECKING:
    from fluster.actor.resolver import Resolver
    from fluster.cluster.types import Entrypoint, JobId, Namespace, ResourceSpec
    from fluster import cluster_pb2

class EndpointRegistry(Protocol):
    """Protocol for registering actor endpoints."""
    def register(self, name: str, address: str, metadata: dict[str, str] | None = None) -> str: ...
    def unregister(self, endpoint_id: str) -> None: ...

class ClusterController(Protocol):
    def submit(self, entrypoint: Entrypoint, name: str, resources: ResourceSpec, ...) -> JobId: ...
    def status(self, job_id: JobId) -> cluster_pb2.JobStatus: ...
    def wait(self, job_id: JobId, timeout: float = 300.0) -> cluster_pb2.JobStatus: ...
    def terminate(self, job_id: JobId) -> None: ...
    def resolver(self, namespace: Namespace) -> Resolver: ...
    @property
    def endpoint_registry(self) -> EndpointRegistry: ...
    @property
    def address(self) -> str: ...

@dataclass(frozen=True)
class FlusterContext:
    namespace: Namespace
    job_id: str
    worker_id: str | None
    controller: ClusterController | None
    ports: dict[str, int] = field(default_factory=dict)

    def get_port(self, name: str) -> int:
        return self.ports[name]

    @property
    def resolver(self) -> Resolver:
        if self.controller is None:
            raise RuntimeError("No controller in context")
        return self.controller.resolver(self.namespace)

_fluster_context: ContextVar[FlusterContext | None] = ContextVar("fluster_context", default=None)

def fluster_ctx() -> FlusterContext:
    ctx = _fluster_context.get()
    if ctx is None:
        raise RuntimeError("fluster_ctx() called outside of a fluster job")
    return ctx

def get_fluster_ctx() -> FlusterContext | None:
    return _fluster_context.get()

@contextmanager
def fluster_ctx_scope(ctx: FlusterContext) -> Generator[FlusterContext, None, None]:
    token = _fluster_context.set(ctx)
    try:
        yield ctx
    finally:
        _fluster_context.reset(token)
```

### Phase 2: Update Resolvers

**Files to modify:**
- `src/fluster/actor/resolver.py`
  - `ClusterResolver.__init__`: Use `get_fluster_ctx()` when no namespace provided
  - `GcsResolver.__init__`: Same pattern
  - Remove direct `os.environ.get("FLUSTER_NAMESPACE", ...)` calls

```python
# resolver.py changes
def __init__(self, controller_address: str, namespace: Namespace | None = None, timeout: float = 5.0):
    self._address = controller_address.rstrip("/")
    self._timeout = timeout

    # Determine namespace: explicit > context > default
    if namespace is not None:
        self._namespace = namespace
    else:
        ctx = get_fluster_ctx()
        self._namespace = ctx.namespace if ctx else Namespace.DEFAULT
```

### Phase 3: Worker Context Injection + ActorServer Self-Registration

**Files to modify:**
- `src/fluster/cluster/worker/worker.py`
  - Inject `FlusterContext` before executing job entrypoints
  - Include allocated ports in context
  - Create `RpcControllerAdapter` implementing `ClusterController`

- `src/fluster/actor/server.py`
  - Update `serve_background()` to use `fluster_ctx()` for port and registration
  - Add `shutdown()` method that unregisters endpoint
  - Remove explicit port parameter (get from context)

- `src/fluster/worker_pool.py`
  - Simplify `worker_job_entrypoint` - ActorServer handles registration

```python
# worker.py - RpcControllerAdapter for remote execution
class RpcControllerAdapter:
    """ClusterController implementation backed by RPC to controller."""

    def __init__(self, controller_address: str, namespace: Namespace):
        self._address = controller_address
        self._namespace = namespace
        self._client = ControllerServiceClientSync(address=controller_address)
        self._registry = RpcEndpointRegistry(self._client)

    @property
    def endpoint_registry(self) -> EndpointRegistry:
        return self._registry

    def resolver(self, namespace: Namespace) -> Resolver:
        return ClusterResolver(self._address, namespace)

    # ... other methods delegate to RPC client ...

class RpcEndpointRegistry:
    """EndpointRegistry that registers via RPC."""

    def __init__(self, client: ControllerServiceClientSync, job_id: str, namespace: Namespace):
        self._client = client
        self._job_id = job_id
        self._namespace = namespace

    def register(self, name: str, address: str, metadata: dict[str, str] | None = None) -> str:
        request = cluster_pb2.Controller.RegisterEndpointRequest(
            name=name,
            address=address,
            job_id=self._job_id,
            namespace=str(self._namespace),
            metadata=metadata or {},
        )
        response = self._client.register_endpoint(request)
        return response.endpoint_id

    def unregister(self, endpoint_id: str) -> None:
        # Controller handles cleanup on job termination; explicit unregister optional
        pass

# worker.py - in _execute_job
def _execute_job(self, job: Job) -> None:
    namespace = Namespace(job_env.get("FLUSTER_NAMESPACE", Namespace.DEFAULT))

    # Build port map from allocated ports
    ports = {name: port for name, port in job.ports.items()}

    ctx = FlusterContext(
        namespace=namespace,
        job_id=job.job_id,
        worker_id=self._config.worker_id,
        controller=RpcControllerAdapter(self._config.controller_address, namespace),
        ports=ports,
    )

    with fluster_ctx_scope(ctx):
        entrypoint()
```

### Phase 4: LocalClient Implementation

**Files to create:**
- `src/fluster/cluster/local_client.py`

**Key features:**
- `LocalClient` class implementing job execution in threads
- `LocalControllerAdapter` implementing `ClusterController` for local use
- Thread-safe job tracking
- Local actor endpoint resolution

```python
# local_client.py skeleton
@dataclass
class LocalClientConfig:
    max_workers: int = 4
    namespace: Namespace = field(default_factory=lambda: Namespace.DEFAULT)

class LocalClient:
    def __init__(self, config: LocalClientConfig | None = None): ...
    def __enter__(self) -> LocalClient: ...
    def __exit__(self, *_): ...
    def submit(self, entrypoint: Entrypoint, name: str, resources: ResourceSpec, ...) -> JobId: ...
    def status(self, job_id: JobId) -> cluster_pb2.JobStatus: ...
    def wait(self, job_id: JobId, timeout: float = 300.0) -> cluster_pb2.JobStatus: ...
    def terminate(self, job_id: JobId) -> None: ...
    def shutdown(self, wait: bool = True) -> None: ...
```

### Phase 5: Namespace Cleanup

**Files to modify:**
- `src/fluster/cluster/types.py` - Add `Namespace.DEFAULT`
- `src/fluster/cluster/client.py` - Remove namespace param, inherit from context
- `src/fluster/cluster/controller/service.py` - Replace `"<local>"` with `Namespace.DEFAULT`
- `src/fluster/cluster/controller/state.py` - Same
- All test files using `"<local>"`

```python
# types.py addition
class Namespace(str):
    DEFAULT: ClassVar[Namespace]

    def __new__(cls, value: str = "default") -> Namespace:
        return super().__new__(cls, value)

Namespace.DEFAULT = Namespace("default")
```

### Phase 6: Remove ActorContext

**Files to modify:**
- `src/fluster/actor/server.py`
  - Delete `ActorContext` class entirely
  - Delete `current_ctx()` and `_set_actor_context()` functions
  - Actor method calls now use `fluster_ctx_scope()` directly
  - Update `ActorServer.call()` to set `FlusterContext` instead of `ActorContext`

```python
# actor/server.py - updated call() method
async def call(self, request: actor_pb2.ActorCall, ctx: RequestContext) -> actor_pb2.ActorResponse:
    # ... find actor and method ...

    try:
        args = cloudpickle.loads(request.serialized_args) if request.serialized_args else ()
        kwargs = cloudpickle.loads(request.serialized_kwargs) if request.serialized_kwargs else {}

        # Use FlusterContext instead of ActorContext
        with fluster_ctx_scope(self._fluster_context):
            result = method(*args, **kwargs)

        return actor_pb2.ActorResponse(serialized_value=cloudpickle.dumps(result))
    except Exception as e:
        # ... error handling ...
```

User code migration:
```python
# Before:
from fluster.actor.server import current_ctx
ctx = current_ctx()
resolver = ctx.resolver

# After:
from fluster.client import fluster_ctx
ctx = fluster_ctx()
resolver = ctx.resolver
```

## Migration Path

1. Add `FlusterContext` and `fluster_ctx()` - non-breaking
2. Update resolvers to use context with fallback to environment - non-breaking
3. Add `LocalClient` - new functionality
4. Add `Namespace.DEFAULT` alongside `"<local>"` - non-breaking
5. Update worker to inject context - internal change
6. Deprecate direct env var reads in user-facing code
7. Eventually remove `"<local>"` string usage

## Verification

### Unit Tests
- `test_context.py` - FlusterContext creation, scoping, error on missing context
- `test_local_client.py` - LocalClient job submission, context propagation
- `test_namespace.py` - Namespace inheritance, default behavior

### Integration Tests
- Test that `fluster_ctx()` is available in actor methods
- Test that resolvers work with context-provided namespace
- Test LocalClient + RpcClusterClient produce identical behavior
- Test namespace isolation between jobs

### Example Test

```python
from fluster.client import FlusterContext, fluster_ctx, fluster_ctx_scope, LocalClient, LocalClientConfig

def test_fluster_ctx_available_in_job():
    """Context should be available in job code."""
    results = []

    def job_fn():
        ctx = fluster_ctx()
        results.append({
            "namespace": ctx.namespace,
            "job_id": ctx.job_id,
            "has_controller": ctx.controller is not None,
        })

    config = LocalClientConfig(namespace=Namespace("test-ns"))
    with LocalClient(config) as client:
        job_id = client.submit(Entrypoint(callable=job_fn), "test", ResourceSpec())
        client.wait(job_id)

    assert len(results) == 1
    assert results[0]["namespace"] == "test-ns"
    assert results[0]["job_id"].startswith("local-")
    assert results[0]["has_controller"] is True

def test_resolver_uses_context_namespace():
    """Resolver should use namespace from context when not explicitly provided."""
    ctx = FlusterContext(
        namespace=Namespace("from-context"),
        job_id="test-job",
        worker_id=None,
        controller=MockController("http://localhost:8080"),
    )

    with fluster_ctx_scope(ctx):
        resolver = ClusterResolver("http://localhost:8080")
        assert resolver.default_namespace == Namespace("from-context")
```

## Files Changed Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/fluster/client/context.py` | New | FlusterContext, fluster_ctx(), ClusterController, EndpointRegistry Protocols |
| `src/fluster/cluster/local_client.py` | New | LocalClient, LocalEndpointRegistry, LocalResolver, LocalControllerAdapter |
| `src/fluster/cluster/rpc_adapter.py` | New | RpcControllerAdapter, RpcEndpointRegistry for remote execution |
| `src/fluster/cluster/types.py` | Modify | Add Namespace.DEFAULT |
| `src/fluster/actor/resolver.py` | Modify | Use context for namespace |
| `src/fluster/actor/server.py` | Modify | Self-registration via ctx.controller.endpoint_registry; delete ActorContext, current_ctx() |
| `src/fluster/worker_pool.py` | Modify | Simplify - ActorServer handles registration; use fluster_ctx() |
| `src/fluster/cluster/worker/worker.py` | Modify | Inject FlusterContext with ports before job execution |
| `src/fluster/cluster/client.py` | Modify | Remove namespace param, use context |
| `src/fluster/cluster/controller/service.py` | Modify | Replace "<local>" with Namespace.DEFAULT |
| `src/fluster/cluster/controller/state.py` | Modify | Replace "<local>" with Namespace.DEFAULT |
| `tests/test_context.py` | New | Context unit tests |
| `tests/test_local_client.py` | New | LocalClient tests with WorkerPool integration |
