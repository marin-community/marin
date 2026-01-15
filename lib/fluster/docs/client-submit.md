# Client Submit API Fixes

## Issue

The `scheduling_timeout_seconds` parameter exists in the proto (`LaunchJobRequest`) but is not exposed through the client APIs:
- `RpcClusterClient.submit()`
- `ExampleCluster.submit()`

Additionally, `example_actor_job_workflow` in `cluster_example.py` uses the old `ClusterResolver` API which now requires `FlusterContext`.

## Required Fixes

### 1. Add `scheduling_timeout_seconds` to `RpcClusterClient.submit()`

**File:** `src/fluster/cluster/client.py`

```python
def submit(
    self,
    entrypoint: Entrypoint,
    name: str,
    resources: cluster_pb2.ResourceSpec,
    environment: cluster_pb2.EnvironmentConfig | None = None,
    ports: list[str] | None = None,
    scheduling_timeout_seconds: int = 0,  # ADD
) -> JobId:
```

Pass it in the request:
```python
request = cluster_pb2.Controller.LaunchJobRequest(
    name=name,
    serialized_entrypoint=serialized,
    resources=resources,
    environment=env_config,
    bundle_blob=self._get_bundle(),
    ports=ports or [],
    parent_job_id=parent_job_id,
    scheduling_timeout_seconds=scheduling_timeout_seconds,  # ADD
)
```

### 2. Add `scheduling_timeout_seconds` to `ExampleCluster.submit()`

**File:** `examples/cluster_example.py`

```python
def submit(
    self,
    fn,
    *args,
    name: str | None = None,
    env_vars: dict[str, str] | None = None,
    cpu: int = 1,
    memory: str = "1g",
    ports: list[str] | None = None,
    scheduling_timeout_seconds: int = 0,  # ADD
    **kwargs,
) -> str:
```

Pass it through:
```python
return self.get_client().submit(
    entrypoint=entrypoint,
    name=name or fn.__name__,
    resources=resources,
    environment=environment,
    ports=ports,
    scheduling_timeout_seconds=scheduling_timeout_seconds,  # ADD
)
```

### 3. Fix `example_actor_job_workflow` ClusterResolver usage

**File:** `examples/cluster_example.py`

The old code uses `ClusterResolver` with explicit namespace:
```python
resolver = ClusterResolver(cluster.controller_url, namespace=Namespace("default"))
```

`ClusterResolver` now requires `FlusterContext` (derives namespace from context). For external clients, query the controller directly and use `FixedResolver`:

```python
from fluster.actor import ActorClient, FixedResolver

# Query controller for all endpoints
list_request = cluster_pb2.Controller.ListEndpointsRequest(prefix="")
list_response = cluster._controller_client.list_endpoints(list_request)

# Find calculator endpoint (name is "{namespace}/calculator")
calculator_endpoint = next(
    (ep for ep in list_response.endpoints if ep.name.endswith("/calculator")),
    None
)
if not calculator_endpoint:
    print("Calculator endpoint not found!")
    return

# Use FixedResolver with discovered address
resolver = FixedResolver({"calculator": f"http://{calculator_endpoint.address}"})
client = ActorClient(resolver, "calculator")
```

## Verification

After making changes:
```bash
cd lib/fluster
uv run pytest tests/ -v
uv run python examples/cluster_example.py --mode=jobs
```
