# fray-zero-actor-and-resolver

Ref: [controller-v1.md](controller-v1.md)
Ref: [impl-recipe.md](impl-recipe.md)
Ref: [controller-v2.md](controller-v2.md)
Ref: [fray-zero.md](fray-zero.md)

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
