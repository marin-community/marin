# Design: Multi-VM JAX Coordinator Bootstrap via Iris Endpoints

## Problem

Multi-VM JAX training requires every process to call `jax.distributed.initialize(coordinator_address, num_processes, process_id)` before any computation. The coordinator address is the IP of process 0, which is only known at runtime (Kubernetes schedules pods to arbitrary nodes). Iris currently has no mechanism for task 0 to advertise its address to sibling tasks 1..N-1.

**What exists today:**

- `JobInfo` (`job_info.py:25-74`) provides `task_index`, `num_tasks`, `advertise_host`, and `controller_address` to every task via env vars (`IRIS_TASK_ID`, `IRIS_NUM_TASKS`, `IRIS_ADVERTISE_HOST`).
- The endpoint registry (`protocol.py:80-90`) supports `register_endpoint(name, address, task_attempt)`, `list_endpoints(prefix, exact=True)`, and `unregister_endpoint(endpoint_id)`.
- `NamespacedEndpointRegistry` (`client.py:373-420`) auto-prefixes endpoint names with the job namespace.
- The controller validates attempt IDs on registration (`service.py:1050-1055`) and cascades endpoint deletion on job/task cleanup (`transitions.py:1550-1568`).
- `coreweave.yaml` only defines `num_vms: 1` scale groups (lines 70, 91). No multi-VM GPU group exists.

**What's missing:**

1. A module that wires task 0's advertise address into the endpoint registry and makes tasks 1..N-1 poll for it before calling `jax.distributed.initialize()`.
2. A multi-VM H100 scale group in the CoreWeave config.
3. Levanter's `DistributedConfig.initialize()` (`distributed.py:348-387`) only handles SLURM and JAX's built-in cluster detection — no Iris path.

## Goals

- Task 0 registers its coordinator address via the existing endpoint API; tasks 1..N-1 discover it by polling `list_endpoints`.
- `jax.distributed.initialize()` is called on all tasks with correct `coordinator_address`, `num_processes`, and `process_id`.
- Endpoint cleanup happens automatically on task retry (stale attempt rejection) and job termination (cascade delete).
- Single-task jobs (`num_tasks == 1`) skip the coordination protocol entirely.

**Non-goals:**

- Modifying the proto schema or adding new RPC methods (the existing endpoint API is sufficient).
- Automatic Levanter integration (dependency direction forbids `levanter` importing `iris`; user code calls `initialize_jax()` explicitly).
- Fault-tolerant JAX (if any task in the gang dies, the JAX runtime is unrecoverable — the job must restart all tasks).

## Proposed Solution

### `iris.runtime.jax_init` module

New file: `lib/iris/src/iris/runtime/jax_init.py`. Single public function:

```python
def initialize_jax(
    port: int = 0,
    endpoint_name: str = "jax_coordinator",
    poll_timeout: float = 300.0,
    poll_interval: float = 2.0,
) -> None:
```

**Task 0 flow:**
1. Read `JobInfo` via `get_job_info()` (`job_info.py:81`).
2. If `num_tasks == 1`: call `jax.distributed.initialize()` with defaults and return.
3. Bind coordinator port: use `IRIS_PORT_jax` if allocated (from `job_info.ports`), else the provided `port` arg (0 = OS-assigned).
4. Build coordinator address: `f"{job_info.advertise_host}:{bound_port}"`.
5. Create `IrisContext` via `iris_ctx()` (`client.py:1025`) — this gives access to the `NamespacedEndpointRegistry`.
6. Register endpoint: `ctx.endpoint_registry.register(endpoint_name, coordinator_address)`.
7. Call `jax.distributed.initialize(coordinator_address, num_tasks, 0)`.
8. Register `atexit` handler to unregister the endpoint.

**Task 1..N-1 flow:**
1. Same `get_job_info()` / `iris_ctx()` setup.
2. Poll `ctx.resolver.resolve(endpoint_name)` with exponential backoff until an endpoint appears or `poll_timeout` expires.
3. Extract `coordinator_address` from `ResolveResult.endpoints[0]`.
4. Call `jax.distributed.initialize(coordinator_address, num_tasks, task_index)`.

```python
def initialize_jax(
    port: int = 0,
    endpoint_name: str = "jax_coordinator",
    poll_timeout: float = 300.0,
    poll_interval: float = 2.0,
) -> None:
    job_info = get_job_info()
    if job_info is None or job_info.num_tasks <= 1:
        jax.distributed.initialize()
        return

    ctx = iris_ctx()
    task_index = job_info.task_index

    if task_index == 0:
        bound_port = job_info.ports.get("jax", port)
        address = f"{job_info.advertise_host}:{bound_port}"
        endpoint_id = ctx.endpoint_registry.register(endpoint_name, address)
        atexit.register(ctx.endpoint_registry.unregister, endpoint_id)
        coordinator = address
    else:
        coordinator = _poll_for_coordinator(ctx.resolver, endpoint_name, poll_timeout, poll_interval)

    jax.distributed.initialize(coordinator, job_info.num_tasks, task_index)
```

The `_poll_for_coordinator` helper uses `ExponentialBackoff` from `iris.time_utils` (per Iris conventions) to retry `resolver.resolve()` until an endpoint appears.

### CoreWeave config changes

Add a multi-VM scale group to `examples/coreweave.yaml`:

```yaml
  h100-16x:
    num_vms: 2
    resources:
      cpu: 128
      ram: 2048GB
      disk: 1TB
      device_type: gpu
      device_variant: H100
      device_count: 8
    worker:
      attributes:
        region: US-WEST-04A
        pool: h100-16x
    min_slices: 0
    max_slices: 1
    priority: 50
    slice_template:
      num_vms: 2
      coreweave:
        region: US-WEST-04A
        instance_type: gd-8xh100ib-i128
```

Jobs targeting this group submit with `replicas=2` and `coscheduling=CoschedulingConfig(group_by="pool")`. The existing coscheduling scheduler (`scheduler.py:648`) groups workers by the `pool` attribute and assigns all replicas to workers in the same group. The autoscaler feasibility check (`service.py:637-643`) validates that `num_vms` matches `replicas`.

### Endpoint lifecycle on restart

No new code needed. The existing mechanisms handle restarts:

1. **Attempt validation** (`service.py:1050-1055`): `register_endpoint` rejects stale attempts. When task 0 is retried with a new `attempt_id`, only the new attempt can register.
2. **INSERT OR REPLACE** (`transitions.py:1533`): Re-registration with the same `endpoint_id` overwrites the old row.
3. **Cascading delete**: Endpoints are associated with `task_id` in the DB (`0001_init.sql:116`) with `ON DELETE CASCADE`. When a task is cleaned up, its endpoints go with it.
4. **Coscheduled sibling cascade** (`transitions.py:1005-1010`): When any coscheduled task fails terminally, siblings are killed too — so all tasks restart together on retry, and task 0 re-registers.

## Implementation Outline

1. **Create `iris.runtime.jax_init`** — `lib/iris/src/iris/runtime/__init__.py` + `jax_init.py` with `initialize_jax()` and `_poll_for_coordinator()`. No proto changes needed.
2. **Add multi-VM scale group** to `examples/coreweave.yaml` — `h100-16x` with `num_vms: 2` and corresponding `slice_template`.
3. **Wire named port** — Add `"jax"` to the default `ports` list for multi-replica GPU jobs, so `IRIS_PORT_JAX` is available. This is a user-side concern (passed in `LaunchJobRequest.ports`), not a framework change.
4. **Integration test** — Use the in-process controller test harness to submit a 2-replica job where task 0 registers a coordinator endpoint and task 1 resolves it. Verify both tasks see the same address. Simulate task 0 restart and verify re-convergence. No real GPUs needed — mock `jax.distributed.initialize`.
5. **Levanter callsite** — In user-level training scripts (not in Levanter itself), replace `DistributedConfig(...).initialize()` with `initialize_jax()` when `IRIS_CONTROLLER_ADDRESS` is set. This respects the dependency direction (`iris` does not import `levanter`; user code imports both).

## Notes

- `initialize_jax()` imports `jax` at call time, not module level. Iris does not depend on JAX — this is a runtime utility that tasks opt into.
- The `NamespacedResolver` (`client.py:430-465`) uses `list_endpoints(prefix=name, exact=True)`, which filters out endpoints from terminal jobs automatically (`db.py:934-937`). No risk of resolving a stale endpoint from a previous job run with the same name.
- `hostNetwork: true` on CoreWeave (`coreweave.md:65`) means `advertise_host` is the node's real IP, routable across the VPC. No NAT traversal needed.
- Port 0 (OS-assigned) works because task 0 registers the actual bound port. Pre-allocating via `IRIS_PORT_JAX` is preferable for firewall predictability but not required.

## Future Work

- **Levanter-native integration**: Add an `IrisCluster` to JAX's `ClusterEnv` registry so `DistributedConfig.initialize()` auto-detects Iris without explicit user code changes.
- **Multi-node TPU on CoreWeave**: Same pattern applies — TPU slices already use `num_vms > 1` on GCE. CoreWeave TPU support would reuse `initialize_jax()`.
- **Health checking**: Periodically verify the coordinator endpoint is still reachable; surface connection failures earlier than JAX's internal timeout.
- **Elastic scaling**: Support adding/removing tasks at runtime (requires JAX elastic training support, which does not exist yet).
