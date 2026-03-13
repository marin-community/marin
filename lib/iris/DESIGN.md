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
    port: int = 8476,
    endpoint_name: str = "jax_coordinator",
    poll_timeout: float = 300.0,
    poll_interval: float = 2.0,
) -> None:
```

**Task 0 flow:**
1. Read `JobInfo` via `get_job_info()` (`job_info.py:81`).
2. If `num_tasks == 1`: call `jax.distributed.initialize()` with defaults and return.
3. Bind coordinator port: use `IRIS_PORT_jax` if allocated (from `job_info.ports`), else the provided `port` arg. An explicit port is required — JAX's gRPC coordinator binds internally and does not expose the actual bound port, so port 0 (OS-assigned) would register `host:0`, which is unusable.
4. Build coordinator address: `f"{job_info.advertise_host}:{bound_port}"`.
5. Create `IrisContext` via `iris_ctx()` (`client.py:1025`) — this gives access to the `NamespacedEndpointRegistry`.
6. Register endpoint: `ctx.registry.register(endpoint_name, coordinator_address)`.
7. Call `jax.distributed.initialize(coordinator_address, num_tasks, 0)`.
8. Register `atexit` handler to unregister the endpoint. This is best-effort cleanup — if the process crashes, the cascade delete on task cleanup (see "Endpoint lifecycle on restart") handles removal.

**Task 1..N-1 flow:**
1. Same `get_job_info()` / `iris_ctx()` setup.
2. Poll `ctx.resolver.resolve(endpoint_name)` using `ExponentialBackoff(initial=poll_interval)` from `iris.time_utils` until an endpoint appears or `poll_timeout` expires. The `poll_interval` parameter sets the initial backoff delay; subsequent retries grow exponentially per `ExponentialBackoff` semantics.
3. Extract `coordinator_address` from `ResolveResult.endpoints[0]`.
4. Call `jax.distributed.initialize(coordinator_address, num_tasks, task_index)`.

```python
def initialize_jax(
    port: int = 8476,
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
        endpoint_id = ctx.registry.register(endpoint_name, address)
        atexit.register(ctx.registry.unregister, endpoint_id)
        coordinator = address
    else:
        coordinator = _poll_for_coordinator(ctx.resolver, endpoint_name, poll_timeout, poll_interval)

    jax.distributed.initialize(coordinator, job_info.num_tasks, task_index)
```

The `_poll_for_coordinator` helper uses `ExponentialBackoff(initial=poll_interval)` from `iris.time_utils` to retry `resolver.resolve()` until an endpoint appears or `poll_timeout` is exceeded, raising `TimeoutError` on expiry.

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

Jobs targeting this group submit with `replicas=2` and `coscheduling=CoschedulingConfig(group_by="pool")`. The existing coscheduling scheduler (`scheduler.py:648`) groups workers by the `pool` attribute and assigns all replicas to workers in the same group. The autoscaler feasibility check (`autoscaler.py:1406`) validates that `replicas` is an exact multiple of `num_vms` (e.g., a group with `num_vms: 2` can serve 2, 4, 6, ... replicas).

### Endpoint lifecycle on restart

No new code needed. The existing mechanisms handle restarts:

1. **Attempt validation** (`service.py:1050-1055`): `register_endpoint` rejects stale attempts. When task 0 is retried with a new `attempt_id`, only the new attempt can register.
2. **INSERT OR REPLACE** (`transitions.py:1533`): Re-registration with the same `endpoint_id` overwrites the old row.
3. **Cascading delete**: Endpoints are associated with `task_id` in the DB (`0001_init.sql:116`) with `ON DELETE CASCADE`. When a task is cleaned up, its endpoints go with it.
4. **Coscheduled sibling cascade** (`transitions.py:1005-1010`): When any coscheduled task fails terminally, siblings are killed too — so all tasks restart together on retry, and task 0 re-registers.

## Implementation Outline

1. **Create `iris.runtime.jax_init`** — `lib/iris/src/iris/runtime/__init__.py` + `jax_init.py` with `initialize_jax()` and `_poll_for_coordinator()`. No proto changes needed.
2. **Add multi-VM scale group** to `examples/coreweave.yaml` — `h100-16x` with `num_vms: 2` and corresponding `slice_template`.
3. **Lift `num_vms > 1` restriction in `CoreweavePlatform.create_slice()`** — This is the hardest step. `coreweave.py:627-631` currently raises `ValueError` for `num_vms > 1`, and `coreweave.md:436-437` documents this as a known limitation. The changes needed:
   - Remove the `num_vms > 1` guard in `create_slice()`.
   - Create N worker Pods per slice (one per VM), each with a distinct pod name and unique `IRIS_WORKER_ID`.
   - Wait for all N Pods to reach Ready state before transitioning the slice to READY.
   - Wire the N Pods into a single `CoreweaveSliceHandle` that tracks all Pod names/IPs and tears them all down on release.
   - Handle partial failures: if Pod k/N fails to start, clean up Pods 0..k-1 and report slice FAILED.

   #### Multi-VM Pod lifecycle

   **Naming convention.** Each Pod in a multi-VM slice is named `iris-worker-{slice_id}-vm{i}` where `i` ranges from `0` to `num_vms - 1`. The existing single-VM naming (`iris-worker-{slice_id}`) is equivalent to the `vm0` case. Update `_worker_pod_name` to accept a `vm_index` parameter:

   ```python
   def _worker_pod_name(slice_id: str, vm_index: int = 0) -> str:
       return f"iris-worker-{slice_id}-vm{vm_index}"
   ```

   **ConfigMap sharing.** All Pods in a slice share a single ConfigMap (`iris-worker-{slice_id}-wc`). The ConfigMap contains the `WorkerConfig` proto which is identical for all VMs in the slice. `_worker_config_cm_name` is unchanged.

   **Pod creation.** `_monitor_slice` calls `_create_worker_pod` N times in a loop, once per `vm_index`. Each Pod receives the same ConfigMap mount but a distinct pod name. The Pod's `IRIS_WORKER_ID` env var is set to its pod name (`iris-worker-{slice_id}-vm{i}`), which uniquely identifies it to the controller. Task assignment is handled by the scheduler — the VM's position within the slice (`vm_index`) is not exposed as a task index.

   **Handle tracking.** `CoreweaveSliceHandle` gains a `_pod_names: list[str]` field, populated during creation as each Pod is created. This replaces the implicit single-name derivation via `_worker_pod_name(self._slice_id)`.

   **`terminate()` changes:**

   ```python
   def terminate(self) -> None:
       cm_name = _worker_config_cm_name(self._slice_id)
       for pod_name in self._pod_names:
           logger.info("Deleting worker Pod: %s", pod_name)
           self._kubectl.delete("pod", pod_name, force=True)
       self._kubectl.delete("configmap", cm_name)
       with self._lock:
           self._state = CloudSliceState.DELETING
   ```

   **Partial-failure cleanup in `_monitor_slice`:**

   ```python
   except Exception as e:
       logger.error("Slice %s monitoring failed: %s", handle.slice_id, e)
       try:
           for pod_name in handle._pod_names:
               self._kubectl.delete("pod", pod_name, force=True)
           self._kubectl.delete("configmap", _worker_config_cm_name(handle.slice_id))
       except Exception as cleanup_err:
           logger.warning("Cleanup after failure also failed for slice %s: %s",
                          handle.slice_id, cleanup_err)
       handle._set_state(CloudSliceState.FAILED, error_message=str(e))
   ```

   **Readiness.** `_monitor_slice` waits for all N Pods to reach Ready state (sequential `_wait_for_pod_ready` per pod). Only after all N succeed does the slice transition to READY, with one `CoreweaveWorkerHandle` per Pod.
4. **Document named port usage** — No framework code changes needed. Users pass `ports=["jax"]` in their `LaunchJobRequest` when submitting multi-replica GPU jobs; Iris allocates `IRIS_PORT_JAX` automatically via the existing port allocation mechanism. Add a usage example to `docs/coreweave.md` showing the `ports` field in a multi-VM job submission.
5. **Integration test** — Use the E2E test harness in `tests/e2e/conftest.py` (the `IrisTestCluster` fixture, which boots a local cluster via `connect_cluster()` + `make_local_config()`) to submit a 2-replica coscheduled job where task 0 registers a coordinator endpoint and task 1 resolves it. Verify both tasks see the same address. Simulate task 0 restart and verify re-convergence. Mock `jax.distributed.initialize` with `unittest.mock.patch("jax.distributed.initialize")` — no real GPUs needed.
6. **Levanter callsite** *(out of scope for this implementation; documented for future reference)* — In user-level training scripts (not in Levanter itself), replace `DistributedConfig(...).initialize()` with `initialize_jax()` when `IRIS_CONTROLLER_ADDRESS` is set. This respects the dependency direction (`iris` does not import `levanter`; user code imports both). This step is tracked separately and not part of the initial PR.

## Notes

- `initialize_jax()` imports `jax` at call time, not module level. Iris does not depend on JAX — this is a runtime utility that tasks opt into.
- The `NamespacedResolver` (`client.py:430-465`) uses `list_endpoints(prefix=name, exact=True)`, which filters out endpoints from terminal jobs automatically (`db.py:934-937`). No risk of resolving a stale endpoint from a previous job run with the same name.
- `hostNetwork: true` on CoreWeave (`coreweave.md:65`) means `advertise_host` is the node's real IP, routable across the VPC. No NAT traversal needed. It also provides implicit anti-affinity: two `hostNetwork: true` Pods binding the same port cannot schedule on the same node, so the scheduler naturally spreads multi-VM Pods across distinct nodes without an explicit `podAntiAffinity` rule.
- An explicit coordinator port is required (default: 8476). JAX's gRPC coordinator binds internally and does not expose the actual bound port, so port 0 (OS-assigned) would result in registering `host:0`. Either allocate via `IRIS_PORT_JAX` or pass a fixed port to `initialize_jax()`.

## Future Work

- **Levanter-native integration**: Add an `IrisCluster` to JAX's `ClusterEnv` registry so `DistributedConfig.initialize()` auto-detects Iris without explicit user code changes.
- **Multi-node TPU on CoreWeave**: Same pattern applies — TPU slices already use `num_vms > 1` on GCE. CoreWeave TPU support would reuse `initialize_jax()`.
- **Health checking**: Periodically verify the coordinator endpoint is still reachable; surface connection failures earlier than JAX's internal timeout.
- **Elastic scaling**: Support adding/removing tasks at runtime (requires JAX elastic training support, which does not exist yet).
