# Actor System Overview

The Actor system provides RPC communication between Fluster jobs. It allows Python objects to be exposed as network services without writing protocol definitions. Actors register with the Controller's endpoint registry for discovery. Clients use resolvers to locate actors and call their methods with automatic load balancing and retries. Arguments and return values are serialized with cloudpickle, supporting arbitrary Python objects.

## Components

| Component | Description |
|-----------|-------------|
| `ActorServer` | Hosts actor instances, handles incoming RPC calls |
| `ActorClient` | Calls actor methods with automatic retry |
| `ActorPool` | Manages multiple endpoints for load balancing and broadcast |
| `Resolver` | Discovers actor endpoints by name |
| `ActorContext` | Injected into actor methods, enables actors to call other actors |

## ActorServer

Hosts one or more actor instances and serves RPC requests. Each job runs at most one ActorServer.

```python
server = ActorServer(controller_address="http://controller:8080")
server.register("inference", InferenceModel())
server.serve()  # blocks, serving requests
```

| Method | Description |
|--------|-------------|
| `register(name, actor, metadata)` | Register an actor instance under a name |
| `serve()` | Start serving requests (blocking) |
| `serve_background()` | Start serving in background |
| `shutdown(grace_period)` | Stop the server |

When an actor is registered, the server notifies the Controller's endpoint registry. Multiple actors (across different jobs) can register under the same name to form a pool.

## ActorClient

Calls methods on a specific actor endpoint. Method calls look like local invocations:

```python
client = ActorClient(resolver, endpoint)
result = client.predict(data)  # calls actor.predict(ctx, data) remotely
```

The client automatically retries failed calls with exponential backoff. Remote exceptions are propagated to the caller.

## ActorPool

Manages multiple endpoints registered under the same actor name. Provides two calling patterns:

```python
pool = resolver.lookup("inference")
pool.wait_for_size(4)  # wait for 4 actors to register

# Round-robin: routes to one actor
result = pool.call().predict(data)

# Broadcast: calls all actors in parallel
futures = pool.broadcast().shutdown()
results = [f.result() for f in futures]
```

| Method | Description |
|--------|-------------|
| `size` | Current number of endpoints |
| `endpoints` | List of current endpoints |
| `wait_for_size(n, timeout)` | Block until at least n actors available |
| `call()` | Get a client for round-robin calls |
| `broadcast()` | Get a handle for calling all actors |

## Resolver

Discovers actor endpoints by name. Three implementations:

| Implementation | Use Case |
|----------------|----------|
| `ClusterResolver` | Production: queries Controller endpoint registry |
| `FixedResolver` | Testing: static endpoint configuration |
| `GcsResolver` | GCP: discovers from VM metadata |

```python
resolver = ClusterResolver("http://controller:8080")
pool = resolver.lookup("inference")
```

The resolver returns an `ActorPool` that tracks endpoints for the given name. As actors register or unregister, the pool updates automatically.

## ActorContext

Passed as the first argument to actor methods. Enables actors to call other actors:

```python
class CoordinatorActor:
    def process(self, ctx: ActorContext, data):
        workers = ctx.resolver.lookup("workers")
        results = workers.broadcast().transform(data)
        return aggregate([r.result() for r in results])
```

| Field | Description |
|-------|-------------|
| `controller_address` | Controller URL |
| `job_id` | ID of the job hosting this actor |
| `namespace` | Namespace for endpoint isolation |
| `resolver` | Resolver for calling other actors |

## Integration Points

```
┌─────────────────────────────────────────────────────────────────┐
│                         Controller                               │
│                                                                  │
│                    Endpoint Registry                             │
│                          ▲    │                                  │
│         RegisterEndpoint │    │ LookupEndpoint                   │
└──────────────────────────┼────┼──────────────────────────────────┘
                           │    │
          ┌────────────────┘    └────────────────┐
          │                                      │
          ▼                                      ▼
┌──────────────────┐                  ┌──────────────────┐
│   ActorServer    │                  │     Resolver     │
│                  │                  │                  │
│  register(name)  │                  │  lookup(name)    │
│  serve()         │                  │        │         │
└──────────────────┘                  │        ▼         │
          ▲                           │   ActorPool      │
          │                           │        │         │
          │  Call RPC                 │        ▼         │
          │                           │   ActorClient    │
          └───────────────────────────┴──────────────────┘
```

1. **ActorServer** registers endpoints with the Controller
2. **Resolver** queries the Controller to discover endpoints
3. **ActorPool** tracks available endpoints and load balances calls
4. **ActorClient** makes RPC calls to individual endpoints

## Usage Patterns

### Single Actor
```python
# Server
server = ActorServer(controller_address)
server.register("model", MyModel())
server.serve()

# Client
pool = resolver.lookup("model")
result = pool.call().predict(x)
```

### Actor Pool (Load Balancing)
```python
# Multiple servers register under same name
for _ in range(4):
    server = ActorServer(controller_address)
    server.register("inference", InferenceModel())
    server.serve_background()

# Client load balances across all
pool = resolver.lookup("inference")
pool.wait_for_size(4)
results = [pool.call().predict(batch) for batch in batches]
```

### Broadcast
```python
pool = resolver.lookup("workers")
futures = pool.broadcast().checkpoint()
for f in futures:
    f.result()  # wait for all to complete
```

## File Summary

| File | Purpose |
|------|---------|
| `server.py` | `ActorServer` hosting and registration |
| `client.py` | `ActorClient` RPC calls with retry |
| `pool.py` | `ActorPool` load balancing and broadcast |
| `resolver.py` | `Resolver` protocol and implementations |
| `types.py` | `ActorContext`, `ActorEndpoint`, type definitions |
