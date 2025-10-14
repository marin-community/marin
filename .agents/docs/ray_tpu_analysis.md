# Ray TPU Backend Analysis

## Overview

The `ray_tpu.py` module provides a sophisticated system for running workloads on Google Cloud TPU pods using Ray. It handles the unique challenges of TPU execution including preemption, multi-slice coordination, health monitoring, and resource cleanup.

## Core Concepts

### 1. Resource Hierarchy

TPU resources are organized in a three-level hierarchy:

```
SlicePoolManager (manages multiple slices)
    └── SliceActor (manages one TPU slice/pod)
        └── TPUHostActor (manages one TPU VM/host)
```

**SlicePoolManager** (`SlicePoolManager` class):
- Manages a pool of TPU slices
- Handles scaling up/down the number of slices
- Supports "flex multislice" - dynamically adjusting slice count based on availability
- Removes unhealthy slices and provisions new ones

**SliceActor** (`@ray.remote` actor):
- Represents a single TPU slice (pod)
- Manages all TPU hosts within that slice
- Schedules work across all hosts in the slice
- Monitors preemption status
- Inherits from `ResourcePoolManager[TPUHostInfo]`

**TPUHostActor** (`@ray.remote` actor):
- Represents a single TPU VM/host within a slice
- Executes the actual remote function on its assigned TPU resources
- Handles libtpu lockfile cleanup
- Schedules tasks with node affinity to ensure correct placement

### 2. Key Data Structures

**TPUConfig**:
```python
@dataclass(frozen=True)
class TPUConfig:
    name: str          # e.g., "v4-32"
    chip_count: int    # Total TPU chips
    host_count: int    # Physical hosts
    vm_count: int      # VMs (may differ from hosts for v5litepod)
    chips_per_vm: int  # TPU chips per VM
```

**MultisliceInfo**:
```python
@dataclass
class MultisliceInfo:
    coordinator_ip: str  # IP of the first slice (coordinator)
    slice_id: int        # This slice's ID (0-indexed)
    num_slices: int      # Total number of slices
    port: int           # Coordination port (default 8081)
```

Converted to environment variables:
- `MEGASCALE_COORDINATOR_ADDRESS`
- `MEGASCALE_NUM_SLICES`
- `MEGASCALE_SLICE_ID`
- `MEGASCALE_PORT`

**Result Types** (ADT-style with dataclasses):
- `TpuSuccess`: Task completed successfully
- `TpuPreempted`: TPU was preempted by GCP
- `TpuFailed`: Node-level failure (treated as preemption)
- `TpuRunError`: Application-level error
- `TpuCancelled`: Task was cancelled due to another failure

### 3. Main Execution Flow

The primary entry point is `run_on_pod()` which wraps `run_on_pod_ray()`:

```python
def run_on_pod(
    remote_fn: RemoteFunction | Callable,
    tpu_type: str,
    *,
    num_slices: int | Sequence[int] = 1,
    max_retries_preemption=10000,
    max_retries_failure=10,
)
```

**Parameters**:
- `remote_fn`: A Ray remote function (or callable that will be wrapped). MUST have `max_calls=1` to ensure cleanup between runs
- `tpu_type`: TPU configuration (e.g., "v4-32", "v5p-128")
- `num_slices`: Either a single int or a list of valid slice counts for flex multislice
- `max_retries_preemption`: Retry count for preemptions (default 10000)
- `max_retries_failure`: Retry count for application failures (default 10)

**Execution Steps**:

1. **Validation**: Check `num_slices` is valid
2. **Actor Pool Setup**:
   - Create `SlicePoolManager` for the TPU type
   - Scale to desired number of slices using `scale_multislice()`
   - Each slice automatically provisions its TPUHostActors
3. **Multislice Coordination** (if `num_slices > 1`):
   - First slice becomes coordinator
   - Generate `MultisliceInfo` for each slice with unique slice_id
   - Inject MEGASCALE environment variables into runtime_env
4. **Work Distribution**:
   - For each slice, call `slice_actor.run_remote_fn()` which returns a list of futures (one per host)
   - All futures across all slices are collected
5. **Monitoring Loop**:
   - Wait for futures to complete one at a time using `ray.wait()`
   - Periodically check actor health with `actor.healthy.remote()`
   - Detect preemption, failures, or success
   - If any task fails, cancel all pending tasks
   - Check if we should scale up multislice (every 3 hours, can add more slices every 12 hours)
6. **Result Processing**:
   - Collect all results and categorize them (success/preempted/failed/cancelled)
   - Increment failure or preemption counters
   - Retry if within retry limits
7. **Cleanup**:
   - Always drain the actor pool in `finally` block
   - Each actor's `teardown()` method cancels pending work and removes lockfiles

### 4. Special Handling & Challenges

#### Preemption Detection

TPU VMs can be preempted by GCP at any time. Detection happens at multiple levels:

1. **Actor-level health checks**: `SliceActor.healthy()` and `TPUHostActor.healthy()` check `get_current_tpu_is_preempted()`
2. **Error classification**: In `_handle_ray_error()`:
   - `NodeDiedError`, `ActorDiedError`, `WorkerCrashedError` → treated as preemption
   - `RayTaskError` → check if TPU is preempted, else treat as application error
   - Timeout errors → assumed to be preemption

#### libtpu Lockfile Cleanup

TPUs use a lockfile (`/tmp/libtpu_lockfile`) to ensure single-process access. Ray's long-running worker processes don't clean this up between tasks, causing subsequent tasks to hang.

**Solution**: `_hacky_remove_tpu_lockfile()` is called before each task execution in `TPUHostActor.run_remote_fn()`.

#### Resource Scheduling

Ray's normal scheduling doesn't guarantee tasks land on the correct TPU hardware. The code uses:

1. **Custom resources**:
   - `TPU-{tpu_type}-head` for SliceActors
   - `{slice_name}` for TPUHostActors (e.g., "tpu-0")
2. **Node affinity**: `NodeAffinitySchedulingStrategy(node_id, soft=False)` ensures tasks run on the exact host
3. **TPU resource claims**: `resources={"TPU": num_tpus}` to reserve TPU chips

#### Flex Multislice

Supports flexible slice counts for better resource utilization:

1. **Initial scaling**: Try to get maximum desired slice count, fall back to smaller valid count
2. **Periodic scale-up checks**: Every 3 hours, check if more slices are available
3. **Scale-up attempt**: If check succeeds, try to acquire more slices (throttled to every 12 hours)
4. **Restart if scaled**: If new slices acquired, cancel current work and restart with more slices

Implementation in `SlicePoolManager.check_should_scale_up_multislice()`.

#### Task Cancellation

Ray doesn't allow cancelling actor method calls easily. To work around this:

1. `TPUHostActor` stores `_awaitable` (the current task future)
2. Before starting new work, cancel previous task: `_cancel_tasks_and_wait([self._awaitable])`
3. Use `ray.cancel(task, force=True, recursive=True)` with timeout

## Public API Functions

### Primary Functions

**`run_on_pod()`**: Synchronous execution with retry
```python
result = run_on_pod(my_function, "v4-32", num_slices=1)
```

**`run_on_pod_ray()`**: Async Ray remote version
```python
ref = run_on_pod_ray.remote(my_function, "v4-32")
result = ray.get(ref)
```

**`run_on_pod_resumable()`**: Alias with high preemption retry count
```python
run_on_pod_resumable(my_function, "v4-32")  # max_retries_preemption=1M
```

**`run_on_pod_multislice()`**: Run without retries (for testing)
```python
results = run_on_pod_multislice(my_function, "v4-32", num_slices=4)
```

**`run_on_pod_multislice_resumable()`**: Multislice with retries
```python
run_on_pod_multislice_resumable(my_function, "v4-32", num_slices=[2, 4, 8])
```

**`run_docker_on_pod()`**: Run Docker container on TPUs
```python
run_docker_on_pod(
    image_id="gcr.io/my-image:latest",
    command=["python", "train.py"],
    tpu_type="v4-32",
    num_slices=1,
    env={"MY_VAR": "value"}
)
```

### Helper Functions

**`submit_tpu_job_on_ray()`**: Submit job to Ray cluster programmatically
- Used from local machine to submit jobs to remote Ray cluster
- Uses Ray JobSubmissionClient

## Design Patterns

### ResourcePoolManager Abstract Base Class

Provides reusable pool management logic:
- `_scale_actor_pool(desired_num_actors)`: Scale pool to exact size
- `_add_members_to_actor_pool(desired)`: Add actors
- `_remove_members_from_actor_pool(desired)`: Remove actors
- `_remove_unhealthy_members_from_actor_pool()`: Health check and cleanup
- `drain_actor_pool()`: Remove all actors

Subclasses implement:
- `get_actor_pool_name()`: For logging
- `get_actor_name_from_actor_info()`: For logging
- `create_actor()`: How to create a new actor

Both `SlicePoolManager` and `SliceActor` use this pattern.

### Result Type Pattern

Instead of exceptions for control flow, results are wrapped in typed dataclasses:
- Allows distinguishing between preemption (retry) and failure (count towards limit)
- Enables centralized error handling logic
- All results processed after all tasks complete or fail

### Future-of-Future Pattern

Ray actor methods return futures. When called remotely, you get a future-of-a-future:

```python
# Returns list[ObjectRef] when called locally
futures = slice_actor.run_remote_fn.remote(fn, env)

# Returns ObjectRef[list[ObjectRef]] when called remotely
future_of_futures = ray.get(slice_actor.run_remote_fn.remote(fn, env))

# Unwrap once
futures = ray.get(future_of_futures)
```

The code handles this explicitly in the `run_remote_fn` methods.

## Integration Considerations for Fray

### What Should Be Exposed

The core functionality that should be exposed through Fray's generic interface:

1. **Run function on TPU slices**: The primary use case
   - Specify TPU type
   - Specify number of slices (or flexible range)
   - Automatic retry on preemption
   - Return results from all hosts

2. **Run Docker on TPUs**: Secondary use case for containerized workloads
   - Image ID
   - Command
   - Environment variables
   - Same slice/retry semantics

3. **Job submission**: For remote clusters
   - Submit from local machine to Ray cluster
   - Track job status

### What Can Be Hidden

Implementation details that don't need to be in the generic interface:

1. **Actor pool management**: Internal to the backend
2. **Health checking**: Automatic within retry loop
3. **Lockfile cleanup**: Backend-specific hack
4. **Resource scheduling**: Backend-specific Ray details
5. **Multislice coordination**: Automatic based on num_slices parameter

### Abstraction Boundary

**ClusterContext level**:
- Should expose TPU job creation (similar to `run_on_pod`)
- Should handle resource allocation and retry logic
- Should return job ID for tracking

**JobContext level**:
- May not need TPU-specific methods
- The remote function runs in a standard JobContext
- TPU-specific coordination (multislice) is handled via environment variables

### Mocking for In-Memory Backend

For `LocalClusterContext`, TPU operations could:
1. **Simple mock**: Just run the function locally, ignore TPU-specific parameters
2. **Validation mock**: Verify parameters are valid but run locally
3. **Simulation mock**: Simulate preemption/failure for testing

Option 1 (simple mock) is probably sufficient for most testing:

```python
def run_on_tpu(self, fn, tpu_type, num_slices=1, ...):
    # Ignore TPU parameters, just run the function
    return fn()
```

### Open Questions

1. **Should TPU jobs use JobContext?**: The current code doesn't - it just runs a function. Should we wrap it to provide JobContext to the function?

2. **Return value semantics**: `run_on_pod` returns a list of results (one per host). Should we aggregate this or return it as-is?

3. **Async vs sync**: Should the interface be async (return job ID) or sync (block until complete)?

4. **Configuration**: Should `tpu_type`, `max_retries`, etc. be part of RuntimeEnv or separate parameters?

5. **Naming**: `run_on_tpu`? `create_tpu_job`? `execute_on_accelerator`?
