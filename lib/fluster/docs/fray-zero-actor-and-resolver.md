# fray-zero-actor-and-resolver

Ref: [fray-zero.md](fray-zero.md)
Ref: [controller-v1.md](controller-v1.md)
Ref: [impl-recipe.md](impl-recipe.md)
Ref: [controller-v2.md](controller-v2.md)

We're moving onto the _Actor_ and _Resolver_ components of the fray-zero system,
which will leverage our cluster controller and workers to run jobs.

## Resolvers

A resolver maps a _service name_ e.g. the name of an actor to a set of strings,
typically URLs which represent the location of the actor or gRPC endpoint. The
resolved URL indicates both the protocol to use (either grpc or actor), as well
as the host and port of the server(s) providing the service.

We provide 3 types of resolvers:

1. Controller Metadata Service
2. GCS VM Tags
3. Fixed Addresses

Example usage:

```
resolver = GcsResolver()
resolver.resolve("fluster-controller") -> ["grpc://host:port/controller"]
```

## Namespaces

In general we want actor names to be isolated across user jobs. To this end, by
default Fluster creates a new _namespace_ based on the initial job ID in a
Fluster run. A typical Fluster run involves a _leader_ job which requests
further resources. This leader job creates a new _namespace_ which is propagated
via the `FLUSTER_NAMESPACE` environment variable.

When the fluster context is used to launch child jobs, the parent namespace
environment variable is automatically propagated by default. Users may override this to specify a shared global namespace to give children unique namespaces as needed.

Thus a typical RL tree might look like:

```
<user-rl-leader-123>:
  trainer/0
  rollout/0
  rollout/1
  inference/0
  inference/1
```

The actor namespace for these jobs is by default shared across all jobs,
therefore when they attempt to resolve e.g. a curriculum or training actor, they
will resolve to the same actor.

From the perspective of the Actor/RPC system, it means we should:

* Accept a namespace argument for the MetadataResolver
* Accept a namespace argument for the ActorServer
* Default these arguments to the FLUSTER_NAMESPACE environment variable or equivalent context var
* Default the namespace to FLUSTER_NAMESPACE if not specified and FLUSTER_JOB_ID is set

The remaining work is handled by Flusters default injection of the
FLUSTER_JOB_ID variable and propagation of the namespace to child jobs via the
`fluster.launch` API.

## Actors and Actor Servers

Users define actors as a Python class with a set of methods which are registered
to an ActorServer. The ActorServer optionally registers itself with a cluster
controller to allow _discovery_ via the resolver pattern.

```
class ActorClass:
  def method_a(self, arg1: int, arg2: str) -> int:
    pass

# e.g. this can be provided by the cluster controller
class NameMapping(Protocol):
  def register(self, name: str, target_url: str):
    pass

server = ActorServer(host, port, NameMapping())
server.register("actor_name", ActorClass())
server.start()

class ActorServer:
  def register(self, name: str, klass):
    self.mapping.register(name, f"actor://{self.host}:{self.port}/{name}")
```

### Actor server implementation

The actor service is a _indirection_ over a generic "Actor" gRPC service. An
example implementation might look like:

```
message ActorRequest {
  string actor_handle = 1;
  string method = 2;
  bytes args = 3;
}

message ActorResponse {
  bytes result = 1;
}

service ActorService {
  rpc Invoke(ActorRequest) returns (ActorResponse);
  rpc ListMethods(ListMethodsRequest) returns (ListMethodsResponse);
  rpc ListActors(ListActorsRequest) returns (ListActorsResponse);
}
```

Note that we use an actor _handle_ to identify the actor, which ensures we
can detect if an actor was terminated between invocations.

## Actor Client and Name Resolution

The actor client handles both name resolution via the resolver as well as
invocation of the actor methods.

```python
from typing import Generic, TypeVar, ParamSpec, Callable, Awaitable, overload
from dataclasses import dataclass, replace

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

@dataclass(frozen=True)
class RpcOptions:
    timeout: float | None = None
    retries: int = 0

class RpcClient(Generic[T]):
    def __init__(self, resolver: Resolver, cls: type[T], options: RpcOptions = RpcOptions()):
        self._resolver = resolver
        self._cls = cls
        self._options = options

    def with_options(self, **kwargs) -> "RpcClient[T]":
        return RpcClient(self._resolver, self._cls, replace(self._options, **kwargs))

    def __getattr__(self, name: str) -> Callable[..., Awaitable]:
        assert hasattr(self._cls, name), f"Method `{name}` not found on class `{self._cls.__name__}`"
        return RpcMethod(self, name, self._options, self._cls)

class RpcMethod(Generic[P, R]):
    def __init__(self, client: RpcClient[T], name: str, options: RpcOptions, cls: type[T]):
        self._client = client
        self._name = name
        self._options = options.copy()
        self._cls = cls

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Awaitable[R]:
        # pickle arguments
        # resolve endpoint
        # send request
        # return result
        # potentially cache resolved endpoint and actor handle
        # re-resolve endpoint and actor if handle is not found, or host is unreachable
        # retry up to retries times
        # raise exception if all retries fail
```

---

# Implementation Plan

This section provides a **spiral** implementation plan for the Actor and Resolver
system. Each stage delivers a working, testable slice of functionality that builds
on the previous stage. Unlike a ladder approach (proto → types → resolver → server
→ client), each spiral stage touches multiple components to create something useful.

## Design Decisions

Based on discussion, the following decisions guide the implementation:

1. **ActorContext injection**: Via contextvar. ActorServer sets `_actor_context`
   contextvar before calling user code. User code calls `current_ctx()` to access.
   This is cleaner than signature inspection and works with any method signature.

2. **Proto design**: Use `bytes serialized_callable` + `bytes serialized_args` +
   `bytes serialized_kwargs`. Ignore the existing proto sketch - do the right thing.

3. **Resolver implementations**: All 3 types (ClusterResolver, FixedResolver,
   GcsResolver) live in `resolver.py`. Keep files simple.

4. **Resolver return type**: Resolvers return a `ResolveResult` (list of URLs +
   metadata), not ActorPool. ActorClient and ActorPool are layered on top.

5. **Default namespace**: `"<local>"` when running without a cluster.

6. **Namespace isolation**: Client-side convention, not enforced by controller.

7. **Broadcast semantics**: Returns `BroadcastFuture` with `wait_all()`,
   `wait_any()`, `as_completed()` methods.

8. **Failure handling**: `pool.call()` propagates exceptions without auto-retry.

9. **Endpoint lifecycle**: Controller monitors job status. When a job transitions
   to a terminal state, the controller removes all endpoints registered by that job.
   The controller always indirects through the jobs map when resolving - if the job
   is not RUNNING, the endpoint is not returned.

10. **Testing strategy**: Prefer real implementations with dummy backends (e.g.,
    tempdir) over mocks. Design APIs like GcsResolver to accept an injectable
    `GcsApi` interface for testing.

11. **RPC infrastructure**: Use Connect-RPC for everything (the generated code).

## Spiral Stages Overview

| Stage | Deliverable | Key Components | Test |
|-------|-------------|----------------|------|
| 1 | Minimal e2e actor call | proto, server, client (hardcoded URL) | call method, get result |
| 2 | Resolver integration | FixedResolver, update client | resolve then call |
| 3 | Controller endpoint registry | state.py, service.py, job lifecycle hooks | register, lookup, job cleanup |
| 4 | ClusterResolver | ClusterResolver, integrate with client | e2e with controller discovery |
| 5 | ActorPool | pool.py with round-robin, broadcast | load-balanced and broadcast calls |
| 6 | GcsResolver | GcsResolver with injectable GcsApi | mock-based testing |
| 7 | Introspection (optional) | ListMethods, ListActors RPCs | debugging helpers |
| 8 | Integration examples | cluster_example.py updates | full demo |

---

## Stage 1: Minimal End-to-End Actor Call

**Goal**: Get a working actor server + client with direct connection. This validates
the core RPC mechanism before adding resolution complexity.

**Files to modify/create**:
- `src/fluster/actor/proto/actor.proto` - update to final design
- `src/fluster/actor/server.py` - new
- `src/fluster/actor/client.py` - new (hardcoded URL version)
- `src/fluster/actor/types.py` - add `current_ctx()` contextvar
- `tests/actor/test_actor_e2e.py` - new

### Proto Changes

Replace `actor.proto` with a clean design:

```protobuf
syntax = "proto3";
package fluster.actor;
option py_generic_services = true;

message ActorCall {
  string method_name = 1;
  string actor_name = 2;           // Which actor on this server
  bytes serialized_args = 3;       // cloudpickle((arg1, arg2, ...))
  bytes serialized_kwargs = 4;     // cloudpickle({k1: v1, ...})
}

message ActorResponse {
  oneof result {
    bytes serialized_value = 1;    // cloudpickle(return_value)
    ActorError error = 2;
  }
}

message ActorError {
  string error_type = 1;
  string message = 2;
  bytes serialized_exception = 3;  // cloudpickle(exception) for re-raise
}

message Empty {}

message HealthResponse {
  bool healthy = 1;
}

service ActorService {
  rpc Call(ActorCall) returns (ActorResponse);
  rpc HealthCheck(Empty) returns (HealthResponse);
}
```

### ActorContext via contextvar

Update `types.py`:

```python
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fluster.actor.resolver import Resolver
    from fluster.cluster.client import Cluster

_actor_context: ContextVar["ActorContext | None"] = ContextVar("actor_context", default=None)

def current_ctx() -> "ActorContext":
    """Get the current ActorContext. Raises if not in an actor call."""
    ctx = _actor_context.get()
    if ctx is None:
        raise RuntimeError("current_ctx() called outside of actor method")
    return ctx

def _set_actor_context(ctx: "ActorContext | None") -> None:
    """Internal: set the actor context for the current call."""
    _actor_context.set(ctx)

@dataclass
class ActorContext:
    """Context available to actor methods via current_ctx()."""
    cluster: "Cluster | None"
    resolver: "Resolver | None"
    job_id: str
    namespace: str
```

### Minimal ActorServer

```python
# src/fluster/actor/server.py
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import cloudpickle
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route

from fluster.actor.types import ActorContext, _set_actor_context, ActorId


@dataclass
class RegisteredActor:
    name: str
    actor_id: ActorId
    instance: Any
    methods: dict[str, Callable]
    registered_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class ActorServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 0):
        self._host = host
        self._port = port
        self._actors: dict[str, RegisteredActor] = {}
        self._context: ActorContext | None = None
        self._app: Starlette | None = None
        self._actual_port: int | None = None

    @property
    def address(self) -> str:
        port = self._actual_port or self._port
        return f"{self._host}:{port}"

    def register(self, name: str, actor: Any) -> ActorId:
        actor_id = ActorId(f"{name}-{uuid.uuid4().hex[:8]}")
        methods = {
            m: getattr(actor, m)
            for m in dir(actor)
            if not m.startswith("_") and callable(getattr(actor, m))
        }
        self._actors[name] = RegisteredActor(
            name=name,
            actor_id=actor_id,
            instance=actor,
            methods=methods,
        )
        return actor_id

    def _create_app(self) -> Starlette:
        async def call_handler(request):
            from fluster import actor_pb2

            # Parse Connect-RPC request
            body = await request.body()
            call = actor_pb2.ActorCall()
            call.ParseFromString(body)

            # Find actor
            actor_name = call.actor_name or next(iter(self._actors), "")
            actor = self._actors.get(actor_name)
            if not actor:
                error = actor_pb2.ActorError(
                    error_type="NotFound",
                    message=f"Actor '{actor_name}' not found",
                )
                resp = actor_pb2.ActorResponse(error=error)
                return Response(resp.SerializeToString(), media_type="application/proto")

            method = actor.methods.get(call.method_name)
            if not method:
                error = actor_pb2.ActorError(
                    error_type="NotFound",
                    message=f"Method '{call.method_name}' not found",
                )
                resp = actor_pb2.ActorResponse(error=error)
                return Response(resp.SerializeToString(), media_type="application/proto")

            try:
                args = cloudpickle.loads(call.serialized_args) if call.serialized_args else ()
                kwargs = cloudpickle.loads(call.serialized_kwargs) if call.serialized_kwargs else {}

                # Set context for this call
                _set_actor_context(self._context)
                try:
                    result = method(*args, **kwargs)
                finally:
                    _set_actor_context(None)

                resp = actor_pb2.ActorResponse(
                    serialized_value=cloudpickle.dumps(result)
                )
                return Response(resp.SerializeToString(), media_type="application/proto")

            except Exception as e:
                error = actor_pb2.ActorError(
                    error_type=type(e).__name__,
                    message=str(e),
                    serialized_exception=cloudpickle.dumps(e),
                )
                resp = actor_pb2.ActorResponse(error=error)
                return Response(resp.SerializeToString(), media_type="application/proto")

        async def health_handler(request):
            from fluster import actor_pb2
            resp = actor_pb2.HealthResponse(healthy=True)
            return Response(resp.SerializeToString(), media_type="application/proto")

        return Starlette(routes=[
            Route("/fluster.actor.ActorService/Call", call_handler, methods=["POST"]),
            Route("/fluster.actor.ActorService/HealthCheck", health_handler, methods=["POST"]),
        ])

    def serve_background(self, context: ActorContext | None = None) -> int:
        """Start server in background thread. Returns actual port."""
        import threading
        import uvicorn
        import socket

        self._context = context
        self._app = self._create_app()

        # Find available port if port=0
        if self._port == 0:
            with socket.socket() as s:
                s.bind(("", 0))
                self._actual_port = s.getsockname()[1]
        else:
            self._actual_port = self._port

        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._actual_port,
            log_level="error",
        )
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Wait for server to be ready
        import time
        for _ in range(50):
            try:
                import httpx
                httpx.get(f"http://{self._host}:{self._actual_port}/", timeout=0.1)
            except Exception:
                pass
            time.sleep(0.1)
            if server.started:
                break

        return self._actual_port
```

### Minimal ActorClient (hardcoded URL)

```python
# src/fluster/actor/client.py
from typing import Any

import cloudpickle
import httpx

from fluster import actor_pb2


class ActorClient:
    """Simple actor client with hardcoded URL (Stage 1)."""

    def __init__(self, url: str, actor_name: str = ""):
        """
        Args:
            url: Direct URL to actor server (e.g., "http://localhost:8080")
            actor_name: Name of actor on the server
        """
        self._url = url.rstrip("/")
        self._actor_name = actor_name
        self._timeout = 30.0

    def __getattr__(self, method_name: str) -> "_RpcMethod":
        return _RpcMethod(self, method_name)


class _RpcMethod:
    def __init__(self, client: ActorClient, method_name: str):
        self._client = client
        self._method_name = method_name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        call = actor_pb2.ActorCall(
            method_name=self._method_name,
            actor_name=self._client._actor_name,
            serialized_args=cloudpickle.dumps(args),
            serialized_kwargs=cloudpickle.dumps(kwargs),
        )

        response = httpx.post(
            f"{self._client._url}/fluster.actor.ActorService/Call",
            content=call.SerializeToString(),
            headers={"Content-Type": "application/proto"},
            timeout=self._client._timeout,
        )
        response.raise_for_status()

        resp = actor_pb2.ActorResponse()
        resp.ParseFromString(response.content)

        if resp.HasField("error"):
            if resp.error.serialized_exception:
                raise cloudpickle.loads(resp.error.serialized_exception)
            raise RuntimeError(f"{resp.error.error_type}: {resp.error.message}")

        return cloudpickle.loads(resp.serialized_value)
```

### Test

```python
# tests/actor/test_actor_e2e.py
import pytest
from fluster.actor.server import ActorServer
from fluster.actor.client import ActorClient
from fluster.actor.types import current_ctx, ActorContext


class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b

    def multiply(self, a: int, b: int) -> int:
        return a * b

    def divide(self, a: int, b: int) -> float:
        return a / b  # May raise ZeroDivisionError


class ContextAwareActor:
    def get_job_id(self) -> str:
        return current_ctx().job_id


def test_basic_actor_call():
    server = ActorServer(host="127.0.0.1")
    server.register("calc", Calculator())
    port = server.serve_background()

    client = ActorClient(f"http://127.0.0.1:{port}", "calc")
    assert client.add(2, 3) == 5
    assert client.multiply(4, 5) == 20


def test_actor_exception_propagation():
    server = ActorServer(host="127.0.0.1")
    server.register("calc", Calculator())
    port = server.serve_background()

    client = ActorClient(f"http://127.0.0.1:{port}", "calc")
    with pytest.raises(ZeroDivisionError):
        client.divide(1, 0)


def test_actor_context_injection():
    server = ActorServer(host="127.0.0.1")
    server.register("ctx_actor", ContextAwareActor())

    ctx = ActorContext(cluster=None, resolver=None, job_id="test-job-123", namespace="<local>")
    port = server.serve_background(context=ctx)

    client = ActorClient(f"http://127.0.0.1:{port}", "ctx_actor")
    assert client.get_job_id() == "test-job-123"
```

**Run**: `cd lib/fluster && buf generate && uv run pytest tests/actor/test_actor_e2e.py -v`

---

## Stage 2: Resolver Integration

**Goal**: Add FixedResolver, update ActorClient to use resolvers.

**Files to modify/create**:
- `src/fluster/actor/resolver.py` - new (ResolveResult, Resolver protocol, FixedResolver)
- `src/fluster/actor/client.py` - update to accept Resolver
- `tests/actor/test_resolver.py` - new

### Resolver Types and FixedResolver

```python
# src/fluster/actor/resolver.py
from dataclasses import dataclass, field
from typing import Protocol

from fluster.cluster.types import Namespace


@dataclass
class ResolvedEndpoint:
    """A single resolved endpoint."""
    url: str                    # e.g., "http://host:port"
    actor_id: str               # Unique handle for staleness detection
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ResolveResult:
    """Result of resolving an actor name."""
    name: str
    namespace: Namespace
    endpoints: list[ResolvedEndpoint] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.endpoints) == 0

    def first(self) -> ResolvedEndpoint:
        if not self.endpoints:
            raise ValueError(f"No endpoints for '{self.name}' in namespace '{self.namespace}'")
        return self.endpoints[0]


class Resolver(Protocol):
    """Protocol for actor name resolution."""

    def resolve(self, name: str, namespace: Namespace | None = None) -> ResolveResult:
        ...

    @property
    def default_namespace(self) -> Namespace:
        ...


class FixedResolver:
    """Resolver with statically configured endpoints."""

    def __init__(
        self,
        endpoints: dict[str, str | list[str]],
        namespace: Namespace = Namespace("<local>"),
    ):
        self._namespace = namespace
        self._endpoints: dict[str, list[str]] = {}
        for name, urls in endpoints.items():
            if isinstance(urls, str):
                self._endpoints[name] = [urls]
            else:
                self._endpoints[name] = list(urls)

    @property
    def default_namespace(self) -> Namespace:
        return self._namespace

    def resolve(self, name: str, namespace: Namespace | None = None) -> ResolveResult:
        ns = namespace or self._namespace
        urls = self._endpoints.get(name, [])
        endpoints = [
            ResolvedEndpoint(url=url, actor_id=f"fixed-{name}-{i}")
            for i, url in enumerate(urls)
        ]
        return ResolveResult(name=name, namespace=ns, endpoints=endpoints)
```

### Updated ActorClient

```python
# src/fluster/actor/client.py - updated
from typing import Any

import cloudpickle
import httpx

from fluster import actor_pb2
from fluster.actor.resolver import Resolver, ResolveResult


class ActorClient:
    """Actor client with resolver-based discovery."""

    def __init__(
        self,
        resolver: Resolver,
        name: str,
        timeout: float = 30.0,
    ):
        self._resolver = resolver
        self._name = name
        self._timeout = timeout
        self._cached_result: ResolveResult | None = None

    def _resolve(self) -> ResolveResult:
        if self._cached_result is None or self._cached_result.is_empty:
            self._cached_result = self._resolver.resolve(self._name)
        return self._cached_result

    def _invalidate_cache(self) -> None:
        self._cached_result = None

    def __getattr__(self, method_name: str) -> "_RpcMethod":
        return _RpcMethod(self, method_name)


class _RpcMethod:
    def __init__(self, client: ActorClient, method_name: str):
        self._client = client
        self._method_name = method_name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self._client._resolve()
        if result.is_empty:
            raise RuntimeError(f"No endpoints found for actor '{self._client._name}'")

        endpoint = result.first()

        call = actor_pb2.ActorCall(
            method_name=self._method_name,
            actor_name=self._client._name,
            serialized_args=cloudpickle.dumps(args),
            serialized_kwargs=cloudpickle.dumps(kwargs),
        )

        try:
            response = httpx.post(
                f"{endpoint.url}/fluster.actor.ActorService/Call",
                content=call.SerializeToString(),
                headers={"Content-Type": "application/proto"},
                timeout=self._client._timeout,
            )
            response.raise_for_status()
        except httpx.RequestError:
            self._client._invalidate_cache()
            raise

        resp = actor_pb2.ActorResponse()
        resp.ParseFromString(response.content)

        if resp.HasField("error"):
            if resp.error.serialized_exception:
                raise cloudpickle.loads(resp.error.serialized_exception)
            raise RuntimeError(f"{resp.error.error_type}: {resp.error.message}")

        return cloudpickle.loads(resp.serialized_value)
```

### Test

```python
# tests/actor/test_resolver.py
import pytest
from fluster.actor.resolver import FixedResolver, ResolveResult
from fluster.actor.server import ActorServer
from fluster.actor.client import ActorClient
from fluster.cluster.types import Namespace


class Echo:
    def echo(self, msg: str) -> str:
        return f"echo: {msg}"


def test_fixed_resolver_single():
    resolver = FixedResolver({"svc": "http://localhost:8080"})
    result = resolver.resolve("svc")
    assert len(result.endpoints) == 1
    assert result.first().url == "http://localhost:8080"


def test_fixed_resolver_multiple():
    resolver = FixedResolver({"svc": ["http://h1:8080", "http://h2:8080"]})
    result = resolver.resolve("svc")
    assert len(result.endpoints) == 2


def test_fixed_resolver_missing():
    resolver = FixedResolver({})
    result = resolver.resolve("missing")
    assert result.is_empty


def test_client_with_resolver():
    server = ActorServer(host="127.0.0.1")
    server.register("echo", Echo())
    port = server.serve_background()

    resolver = FixedResolver({"echo": f"http://127.0.0.1:{port}"})
    client = ActorClient(resolver, "echo")

    assert client.echo("hello") == "echo: hello"
```

**Run**: `uv run pytest tests/actor/test_resolver.py -v`

---

## Stage 3: Controller Endpoint Registry

**Goal**: Implement endpoint registry in controller state and service. The controller
tracks endpoints by job, and automatically removes them when jobs terminate.

**Key design point**: The controller always indirects through the jobs map when
returning endpoints. If a job is not RUNNING, its endpoints are filtered out.

**Files to modify**:
- `src/fluster/cluster/controller/state.py` - add endpoint storage
- `src/fluster/cluster/controller/service.py` - implement RPC handlers
- `src/fluster/cluster/controller/heartbeat.py` - cleanup on job termination
- `tests/cluster/controller/test_endpoint_registry.py` - new

### State Changes

```python
# Add to state.py

@dataclass
class ControllerEndpoint:
    """An endpoint registered with the controller."""
    endpoint_id: str
    name: str
    address: str
    job_id: JobId
    namespace: str
    metadata: dict[str, str] = field(default_factory=dict)
    registered_at_ms: int = 0


class ControllerState:
    def __init__(self):
        # ... existing ...
        self._endpoints: dict[str, ControllerEndpoint] = {}
        self._endpoints_by_job: dict[JobId, set[str]] = {}

    def add_endpoint(self, endpoint: ControllerEndpoint) -> None:
        with self._lock:
            self._endpoints[endpoint.endpoint_id] = endpoint
            self._endpoints_by_job.setdefault(endpoint.job_id, set()).add(endpoint.endpoint_id)

    def remove_endpoint(self, endpoint_id: str) -> ControllerEndpoint | None:
        with self._lock:
            endpoint = self._endpoints.pop(endpoint_id, None)
            if endpoint:
                job_endpoints = self._endpoints_by_job.get(endpoint.job_id)
                if job_endpoints:
                    job_endpoints.discard(endpoint_id)
            return endpoint

    def lookup_endpoints(self, name: str, namespace: str) -> list[ControllerEndpoint]:
        """Find endpoints by name, filtering to only RUNNING jobs."""
        with self._lock:
            results = []
            for ep in self._endpoints.values():
                if ep.name != name or ep.namespace != namespace:
                    continue
                # Only return endpoints for running jobs
                job = self._jobs.get(ep.job_id)
                if job and job.state == cluster_pb2.JOB_STATE_RUNNING:
                    results.append(ep)
            return results

    def list_endpoints_by_prefix(self, prefix: str, namespace: str) -> list[ControllerEndpoint]:
        """List endpoints matching prefix, filtering to only RUNNING jobs."""
        with self._lock:
            results = []
            for ep in self._endpoints.values():
                if not ep.name.startswith(prefix) or ep.namespace != namespace:
                    continue
                job = self._jobs.get(ep.job_id)
                if job and job.state == cluster_pb2.JOB_STATE_RUNNING:
                    results.append(ep)
            return results

    def remove_endpoints_for_job(self, job_id: JobId) -> list[ControllerEndpoint]:
        """Remove all endpoints for a job. Called on job termination."""
        with self._lock:
            endpoint_ids = list(self._endpoints_by_job.get(job_id, []))
            removed = []
            for eid in endpoint_ids:
                ep = self.remove_endpoint(eid)
                if ep:
                    removed.append(ep)
            return removed
```

### Service Changes

```python
# Update service.py - replace the stub implementations

def register_endpoint(
    self,
    request: cluster_pb2.RegisterEndpointRequest,
    ctx: Any,
) -> cluster_pb2.RegisterEndpointResponse:
    endpoint_id = str(uuid.uuid4())

    # Validate job exists and is running
    job = self._state.get_job(JobId(request.job_id))
    if not job:
        raise ConnectError(Code.NOT_FOUND, f"Job {request.job_id} not found")
    if job.state != cluster_pb2.JOB_STATE_RUNNING:
        raise ConnectError(Code.FAILED_PRECONDITION, f"Job {request.job_id} is not running")

    endpoint = ControllerEndpoint(
        endpoint_id=endpoint_id,
        name=request.name,
        address=request.address,
        job_id=JobId(request.job_id),
        namespace=request.namespace or "<local>",
        metadata=dict(request.metadata),
        registered_at_ms=int(time.time() * 1000),
    )
    self._state.add_endpoint(endpoint)
    self._state.log_action(
        "endpoint_registered",
        job_id=job.job_id,
        details=f"{request.name} at {request.address}",
    )
    return cluster_pb2.RegisterEndpointResponse(endpoint_id=endpoint_id)


def unregister_endpoint(
    self,
    request: cluster_pb2.UnregisterEndpointRequest,
    ctx: Any,
) -> cluster_pb2.Empty:
    endpoint = self._state.remove_endpoint(request.endpoint_id)
    if endpoint:
        self._state.log_action(
            "endpoint_unregistered",
            job_id=endpoint.job_id,
            details=endpoint.name,
        )
    return cluster_pb2.Empty()


def lookup_endpoint(
    self,
    request: cluster_pb2.LookupEndpointRequest,
    ctx: Any,
) -> cluster_pb2.LookupEndpointResponse:
    namespace = request.namespace or "<local>"
    endpoints = self._state.lookup_endpoints(request.name, namespace)
    if not endpoints:
        return cluster_pb2.LookupEndpointResponse()

    e = endpoints[0]
    return cluster_pb2.LookupEndpointResponse(
        endpoint=cluster_pb2.Endpoint(
            endpoint_id=e.endpoint_id,
            name=e.name,
            address=e.address,
            job_id=e.job_id,
            namespace=e.namespace,
            metadata=e.metadata,
        )
    )


def list_endpoints(
    self,
    request: cluster_pb2.ListEndpointsRequest,
    ctx: Any,
) -> cluster_pb2.ListEndpointsResponse:
    namespace = request.namespace or "<local>"
    endpoints = self._state.list_endpoints_by_prefix(request.prefix, namespace)
    return cluster_pb2.ListEndpointsResponse(
        endpoints=[
            cluster_pb2.Endpoint(
                endpoint_id=e.endpoint_id,
                name=e.name,
                address=e.address,
                job_id=e.job_id,
                namespace=e.namespace,
                metadata=e.metadata,
            )
            for e in endpoints
        ]
    )
```

### Job Termination Cleanup

In `heartbeat.py` or wherever job state transitions are handled, add:

```python
def _handle_job_termination(self, job_id: JobId) -> None:
    """Clean up when a job transitions to terminal state."""
    removed = self._state.remove_endpoints_for_job(job_id)
    for ep in removed:
        self._state.log_action(
            "endpoint_removed_job_terminated",
            job_id=job_id,
            details=ep.name,
        )
```

### Test

```python
# tests/cluster/controller/test_endpoint_registry.py
import pytest
from fluster import cluster_pb2
from fluster.cluster.controller.state import ControllerState, ControllerJob, ControllerEndpoint
from fluster.cluster.types import JobId


@pytest.fixture
def state() -> ControllerState:
    return ControllerState()


def test_add_and_lookup_endpoint(state: ControllerState):
    # Create a running job first
    job = ControllerJob(
        job_id=JobId("job-1"),
        request=cluster_pb2.LaunchJobRequest(name="test"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job)

    # Register endpoint
    ep = ControllerEndpoint(
        endpoint_id="ep-1",
        name="my-actor",
        address="10.0.0.1:8080",
        job_id=JobId("job-1"),
        namespace="<local>",
    )
    state.add_endpoint(ep)

    # Lookup
    results = state.lookup_endpoints("my-actor", "<local>")
    assert len(results) == 1
    assert results[0].address == "10.0.0.1:8080"


def test_endpoint_not_returned_for_non_running_job(state: ControllerState):
    # Create a completed job
    job = ControllerJob(
        job_id=JobId("job-1"),
        request=cluster_pb2.LaunchJobRequest(name="test"),
        state=cluster_pb2.JOB_STATE_SUCCEEDED,
    )
    state.add_job(job)

    ep = ControllerEndpoint(
        endpoint_id="ep-1",
        name="my-actor",
        address="10.0.0.1:8080",
        job_id=JobId("job-1"),
        namespace="<local>",
    )
    state.add_endpoint(ep)

    # Should not return endpoint because job is not running
    results = state.lookup_endpoints("my-actor", "<local>")
    assert len(results) == 0


def test_remove_endpoints_on_job_termination(state: ControllerState):
    job = ControllerJob(
        job_id=JobId("job-1"),
        request=cluster_pb2.LaunchJobRequest(name="test"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job)

    ep = ControllerEndpoint(
        endpoint_id="ep-1",
        name="my-actor",
        address="10.0.0.1:8080",
        job_id=JobId("job-1"),
        namespace="<local>",
    )
    state.add_endpoint(ep)

    # Simulate job termination
    removed = state.remove_endpoints_for_job(JobId("job-1"))
    assert len(removed) == 1

    # Endpoint should be gone
    results = state.lookup_endpoints("my-actor", "<local>")
    assert len(results) == 0
```

**Run**: `uv run pytest tests/cluster/controller/test_endpoint_registry.py -v`

---

## Stage 4: ClusterResolver

**Goal**: Implement ClusterResolver that queries the controller for endpoints.

**Files to modify/create**:
- `src/fluster/actor/resolver.py` - add ClusterResolver
- `tests/actor/test_cluster_resolver.py` - new

### ClusterResolver

```python
# Add to resolver.py

import httpx
from fluster import cluster_pb2


class ClusterResolver:
    """Resolver backed by the cluster controller's endpoint registry."""

    def __init__(
        self,
        controller_address: str,
        namespace: Namespace | None = None,
        timeout: float = 5.0,
    ):
        self._address = controller_address.rstrip("/")
        self._timeout = timeout

        import os
        self._namespace = namespace or Namespace(
            os.environ.get("FLUSTER_NAMESPACE", "<local>")
        )

    @property
    def default_namespace(self) -> Namespace:
        return self._namespace

    def resolve(self, name: str, namespace: Namespace | None = None) -> ResolveResult:
        ns = namespace or self._namespace

        request = cluster_pb2.ListEndpointsRequest(
            prefix=name,
            namespace=ns,
        )

        response = httpx.post(
            f"{self._address}/fluster.cluster.ControllerService/ListEndpoints",
            content=request.SerializeToString(),
            headers={"Content-Type": "application/proto"},
            timeout=self._timeout,
        )
        response.raise_for_status()

        resp = cluster_pb2.ListEndpointsResponse()
        resp.ParseFromString(response.content)

        # Filter to exact name matches
        endpoints = [
            ResolvedEndpoint(
                url=f"http://{ep.address}",
                actor_id=ep.endpoint_id,
                metadata=dict(ep.metadata),
            )
            for ep in resp.endpoints
            if ep.name == name
        ]

        return ResolveResult(name=name, namespace=ns, endpoints=endpoints)
```

### Test (with real controller)

```python
# tests/actor/test_cluster_resolver.py
import threading
import pytest
import uvicorn
from starlette.applications import Starlette

from fluster import cluster_pb2
from fluster.cluster.controller.state import ControllerState, ControllerJob, ControllerEndpoint
from fluster.cluster.controller.service import ControllerServiceImpl
from fluster.cluster.controller.scheduler import Scheduler
from fluster.actor.resolver import ClusterResolver
from fluster.cluster.types import JobId, Namespace


def create_controller_app(state: ControllerState) -> Starlette:
    """Create a minimal controller app for testing."""
    from starlette.responses import Response
    from starlette.routing import Route

    scheduler = Scheduler(state, interval=1.0)
    service = ControllerServiceImpl(state, scheduler)

    async def list_endpoints_handler(request):
        body = await request.body()
        req = cluster_pb2.ListEndpointsRequest()
        req.ParseFromString(body)
        resp = service.list_endpoints(req, None)
        return Response(resp.SerializeToString(), media_type="application/proto")

    return Starlette(routes=[
        Route("/fluster.cluster.ControllerService/ListEndpoints", list_endpoints_handler, methods=["POST"]),
    ])


@pytest.fixture
def controller_with_endpoint():
    """Start a controller with a registered endpoint."""
    import socket

    state = ControllerState()

    # Add a running job
    job = ControllerJob(
        job_id=JobId("job-1"),
        request=cluster_pb2.LaunchJobRequest(name="test"),
        state=cluster_pb2.JOB_STATE_RUNNING,
    )
    state.add_job(job)

    # Add an endpoint
    ep = ControllerEndpoint(
        endpoint_id="ep-1",
        name="inference",
        address="10.0.0.1:8080",
        job_id=JobId("job-1"),
        namespace="<local>",
    )
    state.add_endpoint(ep)

    # Find free port
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    app = create_controller_app(state)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server
    import time
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)

    yield f"http://127.0.0.1:{port}", state


def test_cluster_resolver_finds_endpoint(controller_with_endpoint):
    address, state = controller_with_endpoint

    resolver = ClusterResolver(address, namespace=Namespace("<local>"))
    result = resolver.resolve("inference")

    assert len(result.endpoints) == 1
    assert "10.0.0.1:8080" in result.first().url


def test_cluster_resolver_missing_endpoint(controller_with_endpoint):
    address, state = controller_with_endpoint

    resolver = ClusterResolver(address, namespace=Namespace("<local>"))
    result = resolver.resolve("nonexistent")

    assert result.is_empty
```

**Run**: `uv run pytest tests/actor/test_cluster_resolver.py -v`

---

## Stage 5: ActorPool

**Goal**: Implement ActorPool for load-balanced and broadcast calls.

**Files to create**:
- `src/fluster/actor/pool.py`
- `tests/actor/test_actor_pool.py`

### Implementation

```python
# src/fluster/actor/pool.py
import itertools
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterator, TypeVar

import cloudpickle
import httpx

from fluster import actor_pb2
from fluster.actor.resolver import ResolveResult, ResolvedEndpoint, Resolver

T = TypeVar("T")


@dataclass
class CallResult:
    """Result of a single call in a broadcast."""
    endpoint: ResolvedEndpoint
    value: Any | None = None
    exception: BaseException | None = None

    @property
    def success(self) -> bool:
        return self.exception is None


class BroadcastFuture(Generic[T]):
    """Future for broadcast results."""

    def __init__(self, futures: list[tuple[ResolvedEndpoint, Future]]):
        self._futures = futures

    def wait_all(self, timeout: float | None = None) -> list[CallResult]:
        results = []
        for endpoint, future in self._futures:
            try:
                value = future.result(timeout=timeout)
                results.append(CallResult(endpoint=endpoint, value=value))
            except Exception as e:
                results.append(CallResult(endpoint=endpoint, exception=e))
        return results

    def wait_any(self, timeout: float | None = None) -> CallResult:
        for future in as_completed([f for _, f in self._futures], timeout=timeout):
            idx = next(i for i, (_, f) in enumerate(self._futures) if f is future)
            endpoint = self._futures[idx][0]
            try:
                value = future.result()
                return CallResult(endpoint=endpoint, value=value)
            except Exception as e:
                return CallResult(endpoint=endpoint, exception=e)
        raise TimeoutError("No results within timeout")

    def as_completed(self, timeout: float | None = None) -> Iterator[CallResult]:
        endpoint_map = {id(f): ep for ep, f in self._futures}
        for future in as_completed([f for _, f in self._futures], timeout=timeout):
            endpoint = endpoint_map[id(future)]
            try:
                value = future.result()
                yield CallResult(endpoint=endpoint, value=value)
            except Exception as e:
                yield CallResult(endpoint=endpoint, exception=e)


class ActorPool(Generic[T]):
    """Pool of actors for load-balanced and broadcast calls."""

    def __init__(self, resolver: Resolver, name: str, timeout: float = 30.0):
        self._resolver = resolver
        self._name = name
        self._timeout = timeout
        self._round_robin: itertools.cycle | None = None
        self._cached_result: ResolveResult | None = None
        self._executor = ThreadPoolExecutor(max_workers=32)

    def _resolve(self) -> ResolveResult:
        result = self._resolver.resolve(self._name)
        if self._cached_result is None or result.endpoints != self._cached_result.endpoints:
            self._round_robin = itertools.cycle(result.endpoints) if result.endpoints else None
            self._cached_result = result
        return result

    @property
    def size(self) -> int:
        return len(self._resolve().endpoints)

    @property
    def endpoints(self) -> list[ResolvedEndpoint]:
        return list(self._resolve().endpoints)

    def _call_endpoint(
        self,
        endpoint: ResolvedEndpoint,
        method_name: str,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        call = actor_pb2.ActorCall(
            method_name=method_name,
            actor_name=self._name,
            serialized_args=cloudpickle.dumps(args),
            serialized_kwargs=cloudpickle.dumps(kwargs),
        )

        response = httpx.post(
            f"{endpoint.url}/fluster.actor.ActorService/Call",
            content=call.SerializeToString(),
            headers={"Content-Type": "application/proto"},
            timeout=self._timeout,
        )
        response.raise_for_status()

        resp = actor_pb2.ActorResponse()
        resp.ParseFromString(response.content)

        if resp.HasField("error"):
            if resp.error.serialized_exception:
                raise cloudpickle.loads(resp.error.serialized_exception)
            raise RuntimeError(f"{resp.error.error_type}: {resp.error.message}")

        return cloudpickle.loads(resp.serialized_value)

    def call(self) -> "_PoolCallProxy[T]":
        return _PoolCallProxy(self)

    def broadcast(self) -> "_PoolBroadcastProxy[T]":
        return _PoolBroadcastProxy(self)


class _PoolCallProxy(Generic[T]):
    def __init__(self, pool: ActorPool[T]):
        self._pool = pool

    def __getattr__(self, method_name: str) -> Callable[..., Any]:
        def call(*args, **kwargs):
            self._pool._resolve()
            if self._pool._round_robin is None:
                raise RuntimeError(f"No endpoints for '{self._pool._name}'")
            endpoint = next(self._pool._round_robin)
            return self._pool._call_endpoint(endpoint, method_name, args, kwargs)
        return call


class _PoolBroadcastProxy(Generic[T]):
    def __init__(self, pool: ActorPool[T]):
        self._pool = pool

    def __getattr__(self, method_name: str) -> Callable[..., BroadcastFuture]:
        def broadcast(*args, **kwargs) -> BroadcastFuture:
            result = self._pool._resolve()
            futures = []
            for endpoint in result.endpoints:
                future = self._pool._executor.submit(
                    self._pool._call_endpoint,
                    endpoint,
                    method_name,
                    args,
                    kwargs,
                )
                futures.append((endpoint, future))
            return BroadcastFuture(futures)
        return broadcast
```

### Test

```python
# tests/actor/test_actor_pool.py
import pytest
from fluster.actor.server import ActorServer
from fluster.actor.pool import ActorPool
from fluster.actor.resolver import FixedResolver


class Counter:
    def __init__(self, start: int = 0):
        self._value = start

    def get(self) -> int:
        return self._value

    def increment(self) -> int:
        self._value += 1
        return self._value


def test_pool_round_robin():
    servers = []
    urls = []

    for i in range(3):
        server = ActorServer(host="127.0.0.1")
        server.register("counter", Counter(start=i * 100))
        port = server.serve_background()
        servers.append(server)
        urls.append(f"http://127.0.0.1:{port}")

    resolver = FixedResolver({"counter": urls})
    pool = ActorPool(resolver, "counter")

    assert pool.size == 3

    # Round-robin should cycle through servers
    results = [pool.call().get() for _ in range(6)]
    # Should see values from all three servers (0, 100, 200, 0, 100, 200)
    assert set(results) == {0, 100, 200}


def test_pool_broadcast():
    servers = []
    urls = []

    for i in range(3):
        server = ActorServer(host="127.0.0.1")
        server.register("counter", Counter(start=i))
        port = server.serve_background()
        servers.append(server)
        urls.append(f"http://127.0.0.1:{port}")

    resolver = FixedResolver({"counter": urls})
    pool = ActorPool(resolver, "counter")

    broadcast = pool.broadcast().get()
    results = broadcast.wait_all()

    assert len(results) == 3
    assert all(r.success for r in results)
    assert {r.value for r in results} == {0, 1, 2}
```

**Run**: `uv run pytest tests/actor/test_actor_pool.py -v`

---

## Stage 6: GcsResolver

**Goal**: Implement GcsResolver with injectable GcsApi for testing.

**Files to modify**:
- `src/fluster/actor/resolver.py` - add GcsResolver, GcsApi protocol
- `tests/actor/test_gcs_resolver.py`

### Implementation

```python
# Add to resolver.py

from typing import Protocol as TypingProtocol


class GcsApi(TypingProtocol):
    """Protocol for GCS Compute API operations."""

    def list_instances(self, project: str, zone: str) -> list[dict]:
        """List VM instances with metadata."""
        ...


class RealGcsApi:
    """Real GCS API using google-cloud-compute."""

    def list_instances(self, project: str, zone: str) -> list[dict]:
        from google.cloud import compute_v1

        client = compute_v1.InstancesClient()
        instances = []
        for instance in client.list(project=project, zone=zone):
            metadata = {}
            if instance.metadata and instance.metadata.items:
                for item in instance.metadata.items:
                    metadata[item.key] = item.value

            internal_ip = None
            if instance.network_interfaces:
                internal_ip = instance.network_interfaces[0].network_i_p

            instances.append({
                "name": instance.name,
                "internal_ip": internal_ip,
                "metadata": metadata,
                "status": instance.status,
            })
        return instances


class MockGcsApi:
    """Mock GCS API for testing."""

    def __init__(self, instances: list[dict] | None = None):
        self._instances = instances or []

    def set_instances(self, instances: list[dict]) -> None:
        self._instances = instances

    def list_instances(self, project: str, zone: str) -> list[dict]:
        return self._instances


class GcsResolver:
    """Resolver using GCS VM instance metadata tags."""

    ACTOR_PREFIX = "fluster_actor_"
    NAMESPACE_KEY = "fluster_namespace"

    def __init__(
        self,
        project: str,
        zone: str,
        namespace: Namespace | None = None,
        api: GcsApi | None = None,
    ):
        self._project = project
        self._zone = zone
        self._api = api or RealGcsApi()

        import os
        self._namespace = namespace or Namespace(
            os.environ.get("FLUSTER_NAMESPACE", "<local>")
        )

    @property
    def default_namespace(self) -> Namespace:
        return self._namespace

    def resolve(self, name: str, namespace: Namespace | None = None) -> ResolveResult:
        ns = namespace or self._namespace
        endpoints = []

        instances = self._api.list_instances(self._project, self._zone)

        for instance in instances:
            if instance.get("status") != "RUNNING":
                continue

            metadata = instance.get("metadata", {})
            instance_ns = metadata.get(self.NAMESPACE_KEY, "<local>")

            if instance_ns != ns:
                continue

            actor_key = f"{self.ACTOR_PREFIX}{name}"
            if actor_key in metadata:
                port = metadata[actor_key]
                ip = instance.get("internal_ip")
                if ip:
                    endpoints.append(ResolvedEndpoint(
                        url=f"http://{ip}:{port}",
                        actor_id=f"gcs-{instance['name']}-{name}",
                        metadata={"instance": instance["name"]},
                    ))

        return ResolveResult(name=name, namespace=ns, endpoints=endpoints)
```

### Test

```python
# tests/actor/test_gcs_resolver.py
import pytest
from fluster.actor.resolver import GcsResolver, MockGcsApi
from fluster.cluster.types import Namespace


def test_gcs_resolver_finds_actors():
    api = MockGcsApi([
        {
            "name": "worker-1",
            "internal_ip": "10.0.0.1",
            "status": "RUNNING",
            "metadata": {
                "fluster_namespace": "<local>",
                "fluster_actor_inference": "8080",
            },
        },
    ])
    resolver = GcsResolver("project", "zone", api=api)
    result = resolver.resolve("inference")

    assert len(result.endpoints) == 1
    assert "10.0.0.1:8080" in result.first().url


def test_gcs_resolver_filters_namespace():
    api = MockGcsApi([
        {
            "name": "worker-1",
            "internal_ip": "10.0.0.1",
            "status": "RUNNING",
            "metadata": {
                "fluster_namespace": "other-ns",
                "fluster_actor_inference": "8080",
            },
        },
    ])
    resolver = GcsResolver("project", "zone", namespace=Namespace("<local>"), api=api)
    result = resolver.resolve("inference")

    assert result.is_empty


def test_gcs_resolver_ignores_non_running():
    api = MockGcsApi([
        {
            "name": "worker-1",
            "internal_ip": "10.0.0.1",
            "status": "TERMINATED",
            "metadata": {
                "fluster_namespace": "<local>",
                "fluster_actor_inference": "8080",
            },
        },
    ])
    resolver = GcsResolver("project", "zone", api=api)
    result = resolver.resolve("inference")

    assert result.is_empty
```

**Run**: `uv run pytest tests/actor/test_gcs_resolver.py -v`

---

## Stage 7: Introspection RPCs (Optional)

**Goal**: Add ListMethods and ListActors for debugging.

This is a polish stage. Add to `actor.proto`:

```protobuf
message ListMethodsRequest {
  string actor_name = 1;
}

message MethodInfo {
  string name = 1;
  string signature = 2;
  string docstring = 3;
}

message ListMethodsResponse {
  repeated MethodInfo methods = 1;
}

message ListActorsRequest {}

message ActorInfo {
  string name = 1;
  string actor_id = 2;
  int64 registered_at_ms = 3;
  map<string, string> metadata = 4;
}

message ListActorsResponse {
  repeated ActorInfo actors = 1;
}

service ActorService {
  rpc Call(ActorCall) returns (ActorResponse);
  rpc HealthCheck(Empty) returns (HealthResponse);
  rpc ListMethods(ListMethodsRequest) returns (ListMethodsResponse);
  rpc ListActors(ListActorsRequest) returns (ListActorsResponse);
}
```

Then add handlers in `ActorServer`. This is deferred until the core path works.

---

## Stage 8: Integration Examples

**Goal**: Add examples to `cluster_example.py` demonstrating actor patterns.

**Implemented**: ✅

Added three comprehensive examples to `examples/cluster_example.py`:

### 1. Basic Actor Pattern (`example_actor_basic`)
Demonstrates:
- Creating and registering an actor server
- Registering endpoints with the controller
- Using ActorClient with ClusterResolver for discovery
- Calling actor methods with arguments and return values

Example actor: Calculator with add(), multiply(), and get_history() methods.

### 2. Coordinator Pattern (`example_actor_coordinator`)
Demonstrates:
- Coordinator actor managing a task queue
- Worker actors fetching tasks from coordinator
- Context injection via `current_ctx()` for actor-to-actor communication
- Workers using ActorClient to communicate with coordinator

Shows the pull-based task distribution pattern where workers fetch tasks, process them, and report results back.

### 3. Actor Pool Pattern (`example_actor_pool`)
Demonstrates:
- ActorPool for load-balanced calls across multiple instances
- Round-robin distribution for inference requests
- Broadcast operations (update_weights) to all instances
- Collecting results from broadcast with `wait_all()`

Example: Multiple inference servers that can be called via round-robin or broadcast.

### CLI Updates
Added `--mode` option to cluster_example.py:
- `--mode actors`: Run only actor examples (no Docker required)
- `--mode jobs`: Run only cluster job examples (requires Docker)
- `--mode all`: Run all examples (default)

**Run examples**:
```bash
cd lib/fluster
uv run python examples/cluster_example.py --mode actors
```

---

## Test Commands Summary

```bash
cd lib/fluster

# Stage 1: Minimal e2e
buf generate
uv run pytest tests/actor/test_actor_e2e.py -v

# Stage 2: Resolver
uv run pytest tests/actor/test_resolver.py -v

# Stage 3: Endpoint registry
uv run pytest tests/cluster/controller/test_endpoint_registry.py -v

# Stage 4: ClusterResolver
uv run pytest tests/actor/test_cluster_resolver.py -v

# Stage 5: Pool
uv run pytest tests/actor/test_actor_pool.py -v

# Stage 6: GcsResolver
uv run pytest tests/actor/test_gcs_resolver.py -v

# All actor tests
uv run pytest tests/actor/ -v
```
