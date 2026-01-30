# Fray-Lite Design

**Status**: Draft (v4, post-review)
**Issue**: [#2552](https://github.com/marin-community/marin/issues/2552)
**Research**: [fray-lite-research.md](fray-lite-research.md)
**Related**: [#2553](https://github.com/marin-community/marin/issues/2553) — Iris preemptible worker attribute

## Overview

Fray-lite is a minimal Fray V2 interface that supports our three primary
workloads (Zephyr data processing, RL training, Levanter training) and can be
backed by either Ray or Iris. It lives in `fray.v2` alongside the existing
`fray` code, allowing incremental migration.

## API Surface

The entire public API consists of:

1. **`Client`** — submit jobs and create actors
2. **`ActorHandle`** — call actor methods with `.remote()` shim (Protocol)
3. **`ResourceConfig`** — resource requirements per job/actor
4. **`JobHandle`** — wait for / monitor submitted jobs

### Client

```python
# fray/v2/client.py

class Client(Protocol):
    def submit(self, request: JobRequest) -> JobHandle:
        """Submit a job for execution. Returns immediately."""
        ...

    def create_actor(
        self,
        actor_class: type,
        *args,
        name: str,
        resources: ResourceConfig = ResourceConfig(),
        **kwargs,
    ) -> ActorHandle:
        """Create a named actor instance. Returns a deferred handle immediately.

        The handle is usable right away — the first .remote() call will block
        until the actor's name is registered (i.e. the actor job has started
        and the actor server is serving). Internally polls for name registration.

        Sugar for create_actor_group(..., count=1).handles[0].
        The caller is responsible for creating actors — there is no implicit
        singleton/get-if-exists behavior.
        """
        ...

    def create_actor_group(
        self,
        actor_class: type,
        *args,
        name: str,
        count: int,
        resources: ResourceConfig = ResourceConfig(),
        **kwargs,
    ) -> ActorGroup:
        """Create N instances of an actor, returning a group handle.

        Returns immediately. Each instance runs as a separate job (named
        "{name}-0", "{name}-1", ...). The group exposes individual handles
        as they become ready; callers use wait_ready() to block until enough
        actors are available. Callers (e.g. Zephyr) coordinate work
        distribution and fault tolerance themselves.
        """
        ...

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the client and all managed resources."""
        ...


def current_client() -> Client:
    """Get the current client from context or environment.

    Resolution order:
    1. Explicitly set client (via set_current_client or context manager)
    2. FRAY_CLIENT_SPEC environment variable (see format below)
    3. LocalClient (default)

    FRAY_CLIENT_SPEC format:
        "local"                             → LocalClient()
        "local?threads=4"                   → LocalClient(max_threads=4)
        "ray"                               → RayClient(address="auto")
        "ray?namespace=my-ns"               → RayClient(namespace="my-ns")
        "iris://controller:10000"           → FrayIrisClient("controller:10000")
        "iris://controller:10000?ws=/path"  → FrayIrisClient("controller:10000", workspace="/path")
    """
    ...


def wait_all(
    jobs: Sequence[JobHandle],
    *,
    timeout: float | None = None,
    raise_on_failure: bool = True,
) -> list[JobStatus]:
    """Wait for all jobs to complete. Returns when all finish or one fails.

    Unlike sequential job.wait() calls, this monitors all jobs concurrently
    and raises immediately if any job fails (when raise_on_failure=True),
    without waiting for earlier jobs to finish first.
    """
    ...
```

### ResourceConfig

v2 copies the type definitions from v1 into `fray/v2/types.py`. The following
files from v1 (`fray.cluster.base`) are copied:

- `ResourceConfig`, `DeviceConfig`, `CpuConfig`, `GpuConfig`, `TpuConfig`
- `EnvironmentConfig`
- `Entrypoint`, `BinaryEntrypoint`, `CallableEntrypoint`
- `JobRequest`, `JobStatus`

These are standalone dataclasses with no v1 backend dependencies, so the copy is
mechanical. After Phase 6 (cleanup), v1 types are deleted.

`ResourceConfig` describes the resources for a single task/replica. The
`replicas` field has been moved to `JobRequest` (see below) since it controls
job-level gang scheduling, not per-task resources.

Under Iris, `preemptible=False` maps to a constraint
`Constraint(key="preemptible", op=EQ, value="false")` once workers expose
their preemptibility as an attribute ([#2553](https://github.com/marin-community/marin/issues/2553)).

```python
@dataclass
class ResourceConfig:
    cpu: int = 1
    ram: str = "128m"
    disk: str = "1g"
    device: DeviceConfig = field(default_factory=CpuConfig)
    preemptible: bool = True    # Maps to Iris constraint, Ray head_node pinning
    regions: Sequence[str] | None = None
```

### JobRequest & JobHandle

```python
# fray/v2/types.py

@dataclass
class JobRequest:
    name: str
    entrypoint: Entrypoint
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    environment: EnvironmentConfig | None = None
    replicas: int = 1  # Gang-scheduled replicas (TPU slices). Maps to Ray
                        # run_on_pod num_slices, Iris LaunchJobRequest.replicas.
    max_retries_failure: int = 0
    max_retries_preemption: int = 100


class JobHandle(Protocol):
    @property
    def job_id(self) -> str: ...

    def wait(self, timeout: float | None = None, *, raise_on_failure: bool = True) -> JobStatus:
        """Block until job completes. Default timeout is None (wait forever)."""
        ...

    def status(self) -> JobStatus: ...

    def terminate(self) -> None: ...


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"
```

### ActorHandle

The `ActorHandle` provides a `.remote()` shim so existing RL code needs
minimal changes. Under the hood, it translates to synchronous RPC calls
(Iris ActorClient) or Ray actor refs.

Both `ActorHandle` and `ActorMethod` are Protocols so that backend-specific
implementations (Ray, Iris, Local) can be type-checked without inheritance.

```python
# fray/v2/actor.py

class ActorHandle(Protocol):
    """Handle to a remote actor with .method.remote() calling convention.

    Handles are deferred: they can be created before the actor is ready.
    The first RPC call blocks until the actor's name is registered.
    Handles are picklable for passing to worker jobs.
    """

    def __getattr__(self, method_name: str) -> ActorMethod:
        ...


class ActorMethod(Protocol):
    def remote(self, *args, **kwargs) -> ActorFuture:
        """Invoke the method remotely. Returns a future."""
        ...

    def __call__(self, *args, **kwargs) -> Any:
        """Invoke the method synchronously (blocking)."""
        ...


class ActorFuture:
    """Future for an actor method call."""

    def result(self, timeout: float | None = None) -> Any:
        """Block until result is available."""
        ...


class ActorGroup:
    """Group of actor instances. Callers coordinate work distribution themselves.

    Returned immediately from create_actor_group(). Actors become available
    asynchronously; use wait_ready() to get a stable snapshot of ready handles.
    """

    @property
    def ready_count(self) -> int:
        """Number of actors that have started and are available for RPC."""
        ...

    def wait_ready(self, count: int | None = None, timeout: float = 300.0) -> list[ActorHandle]:
        """Block until `count` actors are ready (default: all).

        Returns a frozen snapshot of ready handles. The returned list is stable
        — it will not change as more actors come up. Call wait_ready() again
        to get a new snapshot with additional actors.
        """
        ...

    @property
    def jobs(self) -> list[JobHandle]:
        """Underlying job handles for lifecycle management."""
        ...

    def statuses(self) -> list[JobStatus]:
        """Return the job status of each actor in the group.

        Useful for detecting dead actors (FAILED/STOPPED) so callers can
        reassign work or request replacement.
        """
        return [job.status() for job in self.jobs]

    def shutdown(self) -> None:
        """Terminate all actor jobs."""
        for job in self.jobs:
            job.terminate()
```

Note: `ActorGroup` is intentionally thin — it does NOT provide round-robin or
broadcast. Callers like Zephyr implement their own dispatch, retry, and
rebalancing logic. This avoids baking scheduling policy into the framework.

## Backend Implementations

### LocalClient

For development and testing. Runs jobs as threads/subprocesses, actors as
in-process objects with lock-based thread safety.

```python
class LocalClient(Client):
    def submit(self, request: JobRequest) -> JobHandle:
        # Run entrypoint in a thread or subprocess
        ...

    def create_actor(self, actor_class, *args, name, resources, **kwargs) -> ActorHandle:
        # Create instance in-process, wrap with lock for thread safety
        # Return ActorHandle that calls methods directly (no network)
        ...
```

### RayClient

Wraps Ray cluster functionality. Jobs become Ray jobs, actors become Ray actors.

v2 copies the Ray integration code from v1 into `fray/v2/ray/` as a clean
break. This avoids a layering violation (v2 depending on v1) and lets us
mutate freely during migration. The v1 originals are deleted in Phase 6.

**Files copied from v1 → v2:**

| v1 source | v2 destination | What it provides |
|-----------|---------------|------------------|
| `fray.cluster.ray.cluster` | `fray.v2.ray.backend` | Job routing (TPU/binary/callable), runtime env setup |
| `fray.cluster.ray.deps` | `fray.v2.ray.deps` | `build_runtime_env_for_packages`, PYTHONPATH computation |
| `fray.cluster.ray.resources` | `fray.v2.ray.resources` | `as_remote_kwargs`, scheduling strategy helpers |
| `fray.cluster.ray.tpu.execution` | `fray.v2.ray.tpu` | `run_on_pod_ray`, `SliceActor`, `TPUHostActor`, retry logic |
| `fray.fn_thunk` | `fray.v2.ray.fn_thunk` | Callable → pickle → CLI entrypoint serialization |
| `fray.cluster.ray.auth` | `fray.v2.ray.auth` | Ray token authentication |

**Not copied** (unused by v2): dashboard proxy, cluster config discovery,
`monitor()` log streaming, `Cluster` base class, `LocalCluster`.

```python
class RayClient(Client):
    def submit(self, request: JobRequest) -> JobHandle:
        # Routes to _launch_tpu_job / _launch_binary_job / _launch_callable_job
        # (logic copied from v1 RayCluster.launch)
        # Handles TPU jobs via run_on_pod
        # JobRequest.replicas → run_on_pod num_slices
        # max_retries = max_retries_failure + max_retries_preemption
        ...

    def create_actor(self, actor_class, *args, name, resources, **kwargs) -> ActorHandle:
        # Returns a deferred RayActorHandle immediately
        # Internally: ray.remote(actor_class).options(name=name, ...).remote(...)
        # The handle wraps the Ray actor ObjectRef; first .remote() call
        # blocks until the actor is scheduled.
        # Map resources to Ray options:
        #   resources.preemptible=False → {"resources": {"head_node": 0.0001}}
        #   resources.cpu → num_cpus (default 0 for actors)
        ...
```

### IrisClient (wraps `iris.client.IrisClient`)

The fray-lite `IrisClient` sits on top of the existing Iris client library.
The Iris client already provides job submission, endpoint registry, and
resolver — we wrap it rather than reimplementing.

`create_actor` delegates to `create_actor_group(..., count=1)`. Only
`create_actor_group` contains Iris-specific logic.

```python
class FrayIrisClient(Client):
    """Iris cluster backend. Wraps iris.client.IrisClient."""

    def __init__(self, controller_address: str, workspace: Path | None = None):
        self._iris = IrisClientLib.remote(controller_address, workspace=workspace)

    def submit(self, request: JobRequest) -> JobHandle:
        # Convert ResourceConfig → Iris ResourceSpec
        #   resources.preemptible=False → Constraint(key="preemptible", op=EQ, value="false")
        # Convert Entrypoint → Iris Entrypoint
        # JobRequest.replicas → Iris LaunchJobRequest.replicas
        # Submit via self._iris.submit(...)
        # Return JobHandle wrapping iris.client.Job
        ...

    def create_actor_group(self, actor_class, *args, name, count, resources, **kwargs) -> ActorGroup:
        # 1. Submit `count` Iris jobs, each running _host_actor entrypoint
        #    with ports=["actor"]. Jobs named "{name}-0", "{name}-1", ...
        # 2. Return ActorGroup immediately
        # 3. ActorGroup.wait_ready() polls resolver for endpoint availability
        # 4. As endpoints appear, wait_ready() returns IrisActorHandles
        ...
```

The actor hosting entrypoint:

```python
def _host_actor(actor_class, args, kwargs, name):
    """Entrypoint for actor-hosting Iris jobs.

    If actor_class.__init__ raises, the job exits non-zero immediately
    rather than blocking forever with a dead actor.
    """
    from iris.actor import ActorServer
    from iris.client import iris_ctx

    ctx = iris_ctx()

    # Init outside try — let failures propagate and kill the job
    instance = actor_class(*args, **kwargs)

    server = ActorServer(host="0.0.0.0", port=ctx.get_port("actor"))
    server.register(name, instance)
    server.serve_background()

    # Register endpoint for discovery via Iris controller
    address = f"{_get_host_ip()}:{server.port}"
    ctx.registry.register(name, address)

    # Actor stays alive until its job is terminated via JobHandle.terminate()
    # or ActorGroup.shutdown(). No separate control channel needed — job
    # termination kills the container.
    server.wait_for_termination()
```

The `ActorHandle` for Iris wraps `iris.actor.ActorClient`:

```python
class IrisActorHandle(ActorHandle):
    """Handle to an Iris-hosted actor.

    Deferred: the handle is created with just a name. On first use, it
    resolves the name via iris_ctx().resolver to find the actor's address.

    Serialization: pickles as (actor_name,). On unpickle, reconstructs
    from the current IrisContext (available in any Iris job via iris_ctx()).
    This means handles are only usable within the same Iris namespace —
    which is always the case since child jobs inherit the parent's namespace.
    """

    def __init__(self, actor_name: str, client: IrisActorClient | None = None):
        self._actor_name = actor_name
        self._client = client  # Lazily resolved on first use if None

    def _resolve(self) -> IrisActorClient:
        if self._client is None:
            from iris.client import iris_ctx
            from iris.actor import ActorClient
            ctx = iris_ctx()
            self._client = ActorClient(ctx.resolver, self._actor_name)
        return self._client

    def __getattr__(self, method_name: str) -> ActorMethod:
        return _IrisActorMethod(self, method_name)

    def __getstate__(self):
        return {"actor_name": self._actor_name}

    def __setstate__(self, state):
        self._actor_name = state["actor_name"]
        self._client = None  # Will resolve lazily via iris_ctx()


class _IrisActorMethod(ActorMethod):
    def remote(self, *args, **kwargs) -> ActorFuture:
        # Run the synchronous Iris RPC call in a thread pool to return
        # a non-blocking future. The Iris ActorClient already handles
        # timeouts internally; we use its timeout and don't add a second one.
        client = self._handle._resolve()
        executor = _get_shared_executor()
        future = executor.submit(
            lambda: getattr(client, self._method)(*args, **kwargs)
        )
        return ActorFuture(future)

    def __call__(self, *args, **kwargs) -> Any:
        client = self._handle._resolve()
        return getattr(client, self._method)(*args, **kwargs)
```

## Zephyr Migration

Zephyr currently uses `JobContext.put/get/run/wait` to dispatch `run_stage`
calls to Ray workers. Under fray-lite, Zephyr will instead:

1. **Launch N worker jobs** via `client.create_actor_group()`, each hosting a
   `ZephyrWorker` actor
2. **Dispatch `run_stage` calls** to individual workers — Zephyr coordinates
   which worker gets which shard
3. **Manage fault tolerance** itself — if a worker fails, retry the shard on
   another worker
4. **Pass data via shared storage** (GCS paths), not via RPC args — workers
   read/write shard data from GCS, the RPC call just passes the path/metadata.
   This avoids serializing large datasets through cloudpickle over HTTP.

```python
class ActorBackend:
    """Distributed Zephyr backend using fray-lite actor groups.

    Use as a context manager to ensure actor cleanup:

        with ActorBackend(client, 8, resources) as backend:
            backend.execute(dataset, hints)
    """

    def __init__(self, client: Client, num_workers: int, resources: ResourceConfig):
        self.group = client.create_actor_group(
            ZephyrWorker,
            name="zephyr-worker",
            count=num_workers,
            resources=resources,
        )

    def __enter__(self) -> ActorBackend:
        return self

    def __exit__(self, *exc):
        self.group.shutdown()

    def execute(self, dataset: Dataset, hints: ExecutionHint) -> Sequence:
        handles = self.group.wait_ready()
        plan = compute_plan(dataset, hints)
        shards = self._shards_from_source_items(plan.source_items)
        for stage in plan.stages:
            shards = self._execute_stage(stage, shards, hints, handles)
        return list(self._materialize(shards))

    def _execute_stage(self, stage, shards, hints, handles):
        # Assign shards to workers round-robin
        # Each worker reads shard data from shared storage, processes, writes back
        # On worker failure, reassign shard to another worker
        futures = []
        for i, shard in enumerate(shards):
            worker = handles[i % len(handles)]
            future = worker.run_stage.remote(shard.storage_path, stage.operations)
            futures.append((future, shard, i))
        # Collect results, retry failures on other workers
        ...


class ZephyrWorker:
    """Actor that executes Zephyr stage operations.

    Workers read input from and write output to shared storage (GCS).
    The RPC call passes storage paths, not data.
    """

    def run_stage(self, input_path: str, operations: list[PhysicalOp]) -> str:
        """Process a shard and return the output path."""
        data = read_from_storage(input_path)
        result = apply_operations(data, operations)
        output_path = write_to_storage(result)
        return output_path
```

The existing `Backend` class with `JobContext` remains available for local/test
use (ThreadContext, SyncContext). The `ActorBackend` is used for distributed
execution on Iris or Ray clusters.

## RL Migration

### Actor creation moves to the controller job

In v1, actors are created with `get_if_exists=True` — any worker can create or
reconnect to a shared actor. In v2, actors are created **only** from the
controller job (`rl_job.py`) and **passed to workers** as arguments. This
eliminates the need for `get_if_exists` and makes the data flow explicit.

### Before (v1) — in `curriculum.py` (called from any worker)

```python
from fray.job import get_default_job_ctx

job_ctx = get_default_job_ctx()
actor = job_ctx.create_actor(
    Curriculum, config,
    name="curriculum",
    get_if_exists=True,
    preemptible=False,
    num_cpus=0,
)
future = actor.sample_lesson.remote(seed)
result = job_ctx.get(future)
```

### After (v2) — actor created in `rl_job.py`, passed to workers

```python
# In rl_job.py (controller):
from fray.v2 import current_client

client = current_client()

# create_actor returns a deferred handle immediately; first .remote()
# call will block until the actor is registered and serving.
curriculum_actor = client.create_actor(
    Curriculum, config,
    name="curriculum",
    resources=ResourceConfig(preemptible=False),
)

# Pass the actor handle to worker entrypoints
def make_rollout_task(curriculum: ActorHandle, worker_idx: int):
    def task():
        # Worker uses the handle directly — no discovery needed
        future = curriculum.sample_lesson.remote(seed)
        result = future.result()
    return task

for i in range(N):
    client.submit(JobRequest(
        name=f"rollout-{i}",
        entrypoint=Entrypoint.from_callable(make_rollout_task(curriculum_actor, i)),
        resources=rollout_resources,
    ))
```

### Migration summary for RL code

| v1 | v2 |
|----|-----|
| `get_default_job_ctx()` | `current_client()` (controller only) |
| `ctx.create_actor(..., get_if_exists=True)` | `client.create_actor(...)` (controller creates, workers receive) |
| `preemptible=False, num_cpus=0` | `resources=ResourceConfig(preemptible=False)` |
| `actor.method.remote()` | `actor.method.remote()` (unchanged) |
| `ctx.get(future)` | `future.result()` |

### Job orchestration (`rl_job.py`)

```python
# Before
cluster = current_cluster()
jobs = []
jobs.append(cluster.launch(JobRequest(name="train", ...)))
for i in range(N):
    jobs.append(cluster.launch(JobRequest(name=f"rollout-{i}", ...)))
cluster.wait(jobs, raise_on_failure=True)

# After
client = current_client()

# Create shared actors first (deferred handles — no blocking here)
curriculum = client.create_actor(Curriculum, config, name="curriculum",
                                  resources=ResourceConfig(preemptible=False))
weight_coord = client.create_actor(WeightTransferCoordinator, name="wt-coord",
                                    resources=ResourceConfig(preemptible=False))

# Launch workers, passing actor handles
jobs = []
jobs.append(client.submit(JobRequest(name="train",
    entrypoint=Entrypoint.from_callable(train_task, args=[curriculum, weight_coord]),
    resources=train_resources)))
for i in range(N):
    jobs.append(client.submit(JobRequest(name=f"rollout-{i}",
        entrypoint=Entrypoint.from_callable(rollout_task, args=[i, curriculum, weight_coord]),
        resources=rollout_resources)))

# Wait for all concurrently — fails fast if any job fails
wait_all(jobs, raise_on_failure=True)
```

## Levanter Training Migration

Minimal change — just swap cluster for client:

```python
# Before
cluster = current_cluster()
job_id = cluster.launch(JobRequest(...))
cluster.wait(job_id, raise_on_failure=True)

# After
client = current_client()
job = client.submit(JobRequest(...))
job.wait(raise_on_failure=True)
```

## Migration Plan

### Phase 0: Iris Prerequisites

Before fray-lite can use Iris as a backend, two Iris changes are needed:

1. **Preemptible worker attribute** ([#2553](https://github.com/marin-community/marin/issues/2553)) —
   Workers must expose `preemptible=true|false` as an attribute so that
   `FrayIrisClient` can map `ResourceConfig(preemptible=False)` to an Iris
   scheduling constraint. Without this, non-preemptible actors (curriculum,
   weight coordinator) cannot be placed correctly.

2. **Move `replicas` to `LaunchJobRequest`** — Iris currently has `replicas`
   on `ResourceSpecProto`. It should be a top-level field on
   `LaunchJobRequest` since it controls gang scheduling at the job level,
   not per-task resource requirements. This aligns with how fray-lite's
   `JobRequest.replicas` maps to the Iris proto.

These can be done in parallel with Phase 1 (core interface + LocalClient).

### Phase 1: Core Interface + LocalClient

1. Create `fray/v2/` package with `types.py`, `client.py`, `actor.py`
2. Copy type definitions from v1 `fray.cluster.base` into `fray/v2/types.py`
   — move `replicas` from `ResourceConfig` to `JobRequest`
3. Implement `LocalClient` backend (threads + in-process actors)
4. Implement `wait_all()` utility
5. Write tests for core API (submit, create_actor, actor group, wait_all, lifecycle)

### Phase 2: Ray Backend

1. Copy Ray integration code from v1 into `fray/v2/ray/` (see table above)
2. Implement `RayClient` using the copied code
3. Ray `ActorHandle` wrapping Ray actor refs with `.remote()` shim (deferred)
4. Map `preemptible=False` → `resources={"head_node": 0.0001}`
5. Map `JobRequest.replicas` → `run_on_pod num_slices`
6. Map `max_retries_failure + max_retries_preemption` → Ray retry count
7. Test with existing Ray cluster

### Phase 3: Iris Backend

1. Implement `FrayIrisClient` wrapping `iris.client.IrisClient`
2. Implement actor hosting entrypoint + endpoint registration
3. Implement `IrisActorHandle` with deferred resolution and context-based serialization
4. Map `preemptible=False` → Iris constraint (requires Phase 0)
5. Map `JobRequest.replicas` → Iris `LaunchJobRequest.replicas` (requires Phase 0)
6. Test with local Iris controller

### Phase 4: Migrate Callers

1. **Levanter training** (`marin.training.training`) — job submission only,
   simplest migration
2. **RL job orchestration** (`marin.rl.rl_job`) — job submission + actor
   creation. Refactor to create actors in controller, pass to workers.
3. **RL actors** (`marin.rl.curriculum`, `marin.rl.weight_transfer`) — update
   call sites from `ctx.get(future)` → `future.result()`. Remove
   `get_or_create_curriculum_actor` in favor of explicit creation in `rl_job.py`.
4. **Zephyr** (`zephyr.backends`) — implement `ActorBackend` with
   shared-storage data movement. Run full Zephyr test suite to validate.
5. **Remaining callers** (evaluation, experiments) — update `current_cluster()`
   → `current_client()`, `cluster.launch()` → `client.submit()`

### Phase 5: Testing

1. Run all Zephyr tests against `ActorBackend` with `LocalClient`
2. Run RL integration tests with refactored actor creation
3. Run Levanter training smoke test
4. Validate Iris backend with Iris integration tests

### Phase 6: Cleanup

1. Delete `fray.cluster`, `fray.job` (v1 code)
2. Move `fray.v2` → `fray` (top-level)
3. Update all imports
4. Remove `fray.queue` if no remaining users

## File Layout

```
lib/fray/src/fray/v2/
├── __init__.py          # Re-exports: Client, current_client, wait_all, JobRequest, etc.
├── types.py             # JobRequest, JobHandle, JobStatus, Entrypoint, EnvironmentConfig
│                        # (copied from v1 fray.cluster.base, replicas moved to JobRequest)
├── client.py            # Client protocol, current_client(), set_current_client(), wait_all()
├── actor.py             # ActorHandle, ActorMethod, ActorFuture, ActorGroup (all Protocols)
├── local.py             # LocalClient implementation
├── ray/
│   ├── __init__.py
│   ├── backend.py       # RayClient (copied+adapted from v1 ray.cluster)
│   ├── deps.py          # Runtime env building (copied from v1 ray.deps)
│   ├── resources.py     # Resource mapping helpers (copied from v1 ray.resources)
│   ├── tpu.py           # TPU orchestration (copied from v1 ray.tpu.execution)
│   ├── fn_thunk.py      # Callable serialization (copied from v1 fn_thunk)
│   └── auth.py          # Ray token auth (copied from v1 ray.auth)
└── iris_backend.py      # FrayIrisClient implementation (wraps iris.client.IrisClient)
```

## Design Decisions

### Why `create_actor` and `create_actor_group`?

`create_actor` is sugar for `create_actor_group(..., count=1).wait_ready()[0]`.
It exists because singleton actors are the common case in RL (curriculum, weight
coordinator). Forcing `group.wait_ready()[0]` for every singleton would be noisy.

Both return deferred handles — the handle is usable immediately but the first
RPC call blocks until the actor is registered. This means `create_actor` does
not block the caller either; it returns a handle that will resolve lazily.

### Why `create_actor_group` returns immediately?

Actor startup is slow (job scheduling + container boot + actor init). Returning
immediately lets callers overlap actor startup with other work. The group
exposes `ready_count` and `wait_ready(count)` so callers choose when to block.
Zephyr can start processing shards as soon as *some* workers are ready rather
than waiting for all N.

### Why `wait_ready()` returns handles?

`wait_ready()` returns a frozen `list[ActorHandle]` snapshot rather than
mutating a `.handles` property. This avoids races where a caller iterates
handles while the list is growing. Callers that need more handles later call
`wait_ready()` again for a new snapshot.

### Why `ActorGroup` instead of `ActorPool`?

Fray-lite provides a thin `ActorGroup` (list of handles + jobs) rather than a
smart `ActorPool` with round-robin/broadcast. Callers like Zephyr need custom
scheduling (retry logic, data locality, rebalancing) that a generic pool can't
anticipate. Keeping the group dumb pushes scheduling policy to the caller.

### Why actors are created only from the controller?

In v1, any worker could `create_actor(..., get_if_exists=True)` to lazily
create-or-connect. This requires a global actor registry and implicit singleton
semantics. In v2, the controller job (`rl_job.py`) creates all actors and
passes handles to workers explicitly. This:
- Makes the data flow explicit and debuggable
- Eliminates the need for `get_if_exists` / singleton semantics
- Works naturally with Iris (actor = job + endpoint registration)

### Why copy v1 Ray code instead of wrapping it?

Three options were considered:

1. **v2 wraps v1 directly** — fast but creates a layering violation (v2 depends
   on v1). Deleting v1 in Phase 6 requires untangling.
2. **Extract shared module** — no duplication, but refactoring overhead for an
   intermediate state that's deleted in Phase 6 anyway.
3. **Copy into v2** (chosen) — clean separation, ~6 files of code. v2 can
   mutate freely. v1 deletion in Phase 6 is just `rm -rf`. The TPU execution
   code (1000 lines) is the largest piece but is self-contained.

### Why `replicas` on `JobRequest` instead of `ResourceConfig`?

`replicas` controls gang scheduling — how many coordinated instances of a job
run together (e.g. TPU slices in a multislice training job). This is a job-level
concern, not a per-task resource requirement. Putting it on `JobRequest` aligns
with the Iris model (where `replicas` belongs on `LaunchJobRequest`) and avoids
confusion with `ResourceConfig` which describes what a single replica needs.

### Why separate `max_retries_failure` and `max_retries_preemption`?

Preemption and failure are different failure modes with different retry budgets.
Preemption is expected and cheap to retry; failures may indicate a bug.
Under Ray (which has a single retry count), we add them together.

### Why `preemptible` on `ResourceConfig`?

This is the caller's declaration of whether the workload can tolerate
preemption. Backends map it to their native mechanism:
- **Ray**: `preemptible=False` → pin to head node via `resources={"head_node": 0.0001}`
- **Iris**: `preemptible=False` → `Constraint(key="preemptible", op=EQ, value="false")`
  (requires [#2553](https://github.com/marin-community/marin/issues/2553))

### Why Zephyr workers use shared storage instead of RPC for data?

Zephyr shards can be GBs. Passing them through cloudpickle over HTTP (Iris
actor RPC) would be slow and memory-intensive. Instead, workers read/write
shard data from shared storage (GCS). The RPC call passes only paths and
metadata. This matches how Zephyr's `LoadFileOp` pipelines already work.

### Why `FrayIrisClient` wraps `iris.client.IrisClient`?

The Iris client library already handles job submission, endpoint registry,
namespace-based resolver, and port allocation. Wrapping it avoids reimplementing
gRPC plumbing and keeps the fray-lite Iris backend thin. The main work is
type conversion (`ResourceConfig` → `ResourceSpec`, `Entrypoint` mapping)
and the actor hosting entrypoint.

### Why IrisActorHandle serializes via context, not controller address?

Actor handles are pickled when passed to worker jobs via `Entrypoint.from_callable`.
On deserialization, the handle reconstructs its `ActorClient` from `iris_ctx()` —
the Iris context already available in every Iris job. This avoids baking the
controller address into the handle and works because child jobs inherit the
parent's namespace (so the resolver finds the same actors).

### Why no async Iris API?

The `_IrisActorMethod.remote()` wraps synchronous Iris RPC calls in a thread
pool executor to return non-blocking futures. This is sufficient — Iris's
`ActorClient` is a simple HTTP call, and the thread pool avoids blocking the
caller. Adding a native async API to Iris would be more complex for marginal
benefit. Revisit only if profiling shows thread pool overhead.

### Why actor termination via job kill, not a control channel?

The `_host_actor` entrypoint blocks forever until its job is terminated.
`ActorGroup.shutdown()` calls `job.terminate()` on each underlying job, which
kills the container. No separate control RPC is needed — this is simpler and
works identically across Ray and Iris. A graceful shutdown hook could be added
later if actors need cleanup logic, but the current workloads don't require it.

## Recommended Issues

The following issues should be filed to begin implementation:

### Iris prerequisites (can start immediately, in parallel)

- **Iris: Move `replicas` from `ResourceSpecProto` to `LaunchJobRequest`** —
  Gang scheduling is a job-level concern. Move the `replicas` field from
  `ResourceSpecProto` to `LaunchJobRequest` as a top-level field. Update
  the controller's scheduling logic and all clients accordingly.

- **Iris: Expose preemptible worker attribute** ([#2553](https://github.com/marin-community/marin/issues/2553)) —
  Already filed. Workers must report `preemptible=true|false` so the scheduler
  can enforce `Constraint(key="preemptible", op=EQ, value="false")`.

### Fray-lite core (start after or in parallel with Iris prereqs)

- **Fray-lite: Phase 1 — Core interface + LocalClient** —
  Create `fray/v2/` package. Define `Client` protocol, `ActorHandle` protocol,
  `ActorGroup`, `JobHandle`, `wait_all()`. Copy type definitions from v1.
  Implement `LocalClient`. Write tests.

- **Fray-lite: Phase 2 — Ray backend** —
  Copy v1 Ray code into `fray/v2/ray/`. Implement `RayClient` with deferred
  actor handles. Map `JobRequest.replicas` → `run_on_pod num_slices`.

- **Fray-lite: Phase 3 — Iris backend** —
  Implement `FrayIrisClient` wrapping `iris.client.IrisClient`. Deferred
  `IrisActorHandle` with context-based serialization. Depends on both Iris
  prerequisite issues.

- **Fray-lite: Phase 4 — Migrate callers** —
  Migrate Levanter, RL, Zephyr, and remaining callers from v1 to v2 API.
  Zephyr is last and largest (new `ActorBackend`).

- **Fray-lite: Phase 6 — Delete v1** —
  Remove `fray.cluster`, `fray.job`, `fray.queue`. Move `fray.v2` → `fray`.
  Update all imports.
