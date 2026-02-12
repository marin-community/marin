# Fray-Lite Design

**Status**: Draft (v5, updated migration plan)
**Issue**: [#2552](https://github.com/marin-community/marin/issues/2552)
**Research**: [fray-lite-research.md](fray-lite-research.md)
**Related**: [#2553](https://github.com/marin-community/marin/issues/2553) — Iris preemptible worker attribute (✅ done)

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
    """Get the current client from context or auto-detection.

    Resolution order:
    1. Explicitly set client (via set_current_client or context manager)
    2. Auto-detect Iris environment (via get_iris_ctx())
    3. Auto-detect Ray environment (via ray.is_initialized())
    4. LocalClient (default)
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

The plan follows a **spiral** approach: each phase produces independently
testable, shippable work. We build v2 core + Ray first, migrate all three
workloads on Ray, then add the Iris backend as a separate track. This lets us
validate the API surface against real callers before adding a second backend.

### Phase 0: Iris Prerequisites ✅ DONE

Both Iris prerequisites are complete:

1. **Preemptible worker attribute** ([#2553](https://github.com/marin-community/marin/issues/2553)) ✅ —
   Workers now expose `preemptible=true|false` via `env_probe.py`. The helper
   `preemptible_constraint()` in `iris.cluster.types` creates the scheduling
   constraint. Detection queries GCP metadata, falls back to
   `IRIS_WORKER_ATTRIBUTES` env var.

2. **`replicas` on `ResourceSpec`** ✅ — `iris.cluster.types.ResourceSpec` has
   `replicas: int = 0` which maps to the proto. (Note: Iris keeps `replicas`
   on `ResourceSpec` rather than `LaunchJobRequest` — fray-lite's
   `JobRequest.replicas` will map to `ResourceSpec.replicas` during conversion.)

### Phase 1: Core Interface + LocalClient + Smoke Test

**Goal**: A working `fray.v2` package that passes tests with `LocalClient`.
This is the foundation everything else builds on.

**Spiral step 1a — Types + Client protocol + LocalClient submit:**

1. Create `fray/v2/` package with `__init__.py`, `types.py`, `client.py`
2. Copy type definitions from v1 `fray.cluster.base` into `fray/v2/types.py`:
   - `ResourceConfig` (remove `replicas` field), `DeviceConfig`, `CpuConfig`,
     `GpuConfig`, `TpuConfig`
   - `EnvironmentConfig`
   - `Entrypoint`, `BinaryEntrypoint`, `CallableEntrypoint`
   - `JobRequest` (add `replicas: int = 1`), `JobStatus`
3. Define `Client` protocol and `JobHandle` protocol in `client.py`
4. Implement `LocalClient.submit()` — run `CallableEntrypoint` in a thread,
   return a `LocalJobHandle` that wraps `concurrent.futures.Future`
5. Implement `wait_all()`
6. Write tests: submit callable job, wait, check status transitions, wait_all
   with mixed success/failure

**Spiral step 1b — Actors:**

1. Create `actor.py` with `ActorHandle`, `ActorMethod`, `ActorFuture`,
   `ActorGroup` protocols
2. Implement `LocalClient.create_actor()` — instantiate class in-process,
   wrap with `threading.Lock` for thread safety, return `LocalActorHandle`
3. Implement `LocalClient.create_actor_group()` — N in-process instances
4. Write tests: create actor, call `.method.remote()`, verify result;
   create group, `wait_ready()`, dispatch to multiple actors

**Spiral step 1c — current_client + auto-detection:**

1. Implement `current_client()` with context var + auto-detection
2. Implement `set_current_client()` context manager
3. Write tests: default → LocalClient, explicit set, auto-detection

**Deliverable**: `LocalClient` passing all protocol tests. Callers can
`pip install` and use `fray.v2` for local development immediately.

### Phase 2: Ray Backend

**Goal**: `RayClient` that can submit jobs and create actors on a Ray cluster.
Validated against a local Ray cluster.

**Spiral step 2a — Job submission:**

1. Copy v1 Ray code into `fray/v2/ray/` (see files table above):
   `backend.py`, `deps.py`, `resources.py`, `tpu.py`, `fn_thunk.py`, `auth.py`
2. Implement `RayClient.submit()` — route to TPU/binary/callable launchers
   based on device type
3. Map `JobRequest.replicas` → `run_on_pod num_slices`
4. Map `max_retries_failure + max_retries_preemption` → Ray retry count
5. Implement `RayJobHandle` wrapping Ray job ID with poll-based `wait()`
6. Test: submit a callable job to local Ray, verify completion

**Spiral step 2b — Actor support:**

1. Implement `RayClient.create_actor()` — `ray.remote(cls).options(...).remote()`
2. Implement `RayActorHandle` wrapping Ray actor ObjectRef with `.remote()` shim
3. Map `preemptible=False` → `resources={"head_node": 0.0001}`
4. Implement `RayClient.create_actor_group()` — N Ray actors
5. Test: create actor on local Ray, call methods, verify results

**Spiral step 2c — Ray auto-detection:**

1. Wire Ray auto-detection via `ray.is_initialized()` into `current_client()`
2. Test: Ray auto-detection returns RayClient

**Deliverable**: `RayClient` passing the same protocol tests as `LocalClient`,
plus Ray-specific integration tests.

### Phase 3: Migrate Levanter Training (simplest caller)

**Goal**: `marin.training.training` uses `fray.v2` instead of `fray.cluster`.
This is the easiest migration — pure job submission, no actors.

1. Update `marin/training/training.py`:
   - `from fray.cluster import ... current_cluster` → `from fray.v2 import ... current_client`
   - `cluster = current_cluster()` → `client = current_client()`
   - `cluster.launch(request)` → `client.submit(request)` (returns `JobHandle`)
   - `cluster.wait(job_id, ...)` → `job.wait(...)`
2. Update `ResourceConfig` usage: `replicas` moves from `ResourceConfig` to
   `JobRequest`. v1's `ResourceConfig.with_tpu(...)` sets `replicas` based on
   topology — v2 equivalent passes `replicas` to `JobRequest` instead.
3. Run existing Levanter tests to verify no regression.

**Deliverable**: Levanter training works on Ray via `fray.v2`. Can be tested
end-to-end with Ray auto-detection.


### Phase 4: Migrate Zephyr

**Goal**: Zephyr uses `fray.v2` for distributed execution. `JobContext` is
deleted — Zephyr moves to the `ActorBackend` model with long-lived worker
actors that receive shard assignments via RPC.

**Current Zephyr architecture** (from `backends.py`, `plan.py`):
- `Backend` class wraps a `JobContext` (SyncContext, ThreadContext, or RayContext)
- `context.put(obj)` / `context.get(ref)` — object store for data references
- `context.run(fn, *args)` — submit a task, returns a future/generator
- `context.wait(refs, num_returns=1)` — wait for N results, streaming style
- `run_stage()` is the worker function that processes shards

**Spiral step 4a — ActorBackend + ZephyrWorker:**

1. Implement `ActorBackend` in `zephyr/backends.py` (as designed in the
   Zephyr Migration section above) using `client.create_actor_group()`
2. Implement `ZephyrWorker` actor that runs `run_stage` on storage paths
3. Replace `Backend._execute_stage()` dispatch logic with actor RPC calls
4. Wire `ActorBackend` into Zephyr's `cli.py` for distributed backends
5. Run Zephyr test suite with `ActorBackend` + `LocalClient`

**Spiral step 4b — Delete JobContext:**

1. Remove `fray.job.context` (SyncContext, ThreadContext, RayContext)
2. Remove all `fray.job` imports from Zephyr
3. Keep a simple in-process `SyncBackend` / `ThreadBackend` for local
   testing that doesn't use `JobContext` — these call `run_stage` directly
4. Run full Zephyr test suite

**Deliverable**: Zephyr works on Ray via `fray.v2` with `ActorBackend`.
`
### Phase 5: Migrate RL Training

**Goal**: `marin.rl` fully migrated to `fray.v2` in one shot — job submission,
actor creation, and all worker call sites. No mixed v1/v2 usage.

**Current RL architecture** (from `rl_job.py`):
- `RLJob.run()` calls `current_cluster().launch()` for 1 train job + N rollout jobs
- `cluster.wait(jobs, raise_on_failure=True)` waits for all
- Workers internally create their own actors via `fray.job.get_default_job_ctx()`
  for `create_actor(get_if_exists=True)` and `ctx.get(future)`

**All of the following ships as one change:**

1. Update `rl_job.py`:
   - `from fray.cluster import ...` → `from fray.v2 import ...`
   - `cluster = current_cluster()` → `client = current_client()`
   - `cluster.launch(JobRequest(...))` → `client.submit(JobRequest(...))`
   - `cluster.wait(jobs, raise_on_failure=True)` → `wait_all(jobs, raise_on_failure=True)`
   - Create curriculum + weight-transfer actors via `client.create_actor()`
     and pass handles to worker entrypoints
2. Update `curriculum.py`: remove `get_or_create_curriculum_actor`, accept
   handle as parameter
3. Update `weight_transfer/`: same pattern — accept handle, remove
   `get_if_exists`
4. Update all worker call sites: `ctx.get(future)` → `future.result()`
5. Remove all `fray.job` imports from RL code
6. Run RL integration tests

**Deliverable**: RL training works on Ray via `fray.v2` with explicit actor
creation from the controller. Zero v1 imports remain in `marin.rl`.JobContext` and `fray.job` are deleted. All Zephyr tests pass.

### Phase 6: Iris Backend

**Goal**: `FrayIrisClient` that can run all three workloads on Iris. Deferred
until after Ray migration is complete and validated.

**Spiral step 6a — Job submission:**

1. Implement `FrayIrisClient` in `fray/v2/iris_backend.py`
2. Convert `ResourceConfig` → Iris `ResourceSpec`:
   - `preemptible=False` → `preemptible_constraint(False)` (from `iris.cluster.types`)
   - `JobRequest.replicas` → `ResourceSpec.replicas`
   - Device config → Iris accelerator spec
3. Convert `Entrypoint` → Iris entrypoint format
4. Implement `IrisJobHandle` wrapping `iris.client.Job`
5. Test: submit job to local Iris controller

**Spiral step 6b — Actor support:**

1. Implement `_host_actor` entrypoint (as designed above)
2. Implement `IrisActorHandle` with deferred name resolution via `iris_ctx()`
3. Implement context-based pickle serialization
4. Implement `create_actor_group()` — submit N hosting jobs, return `ActorGroup`
5. Test: create actor on Iris, call methods

**Spiral step 6c — Integration testing:**

1. Wire `"iris://..."` into `current_client()` resolution
2. Run Levanter, RL, and Zephyr integration tests against Iris backend
3. Validate preemptible constraint enforcement

**Deliverable**: All three workloads run on Iris via `fray.v2`.

### Phase 7: Cleanup

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

### ✅ Completed

- **Iris: Expose preemptible worker attribute** ([#2553](https://github.com/marin-community/marin/issues/2553)) — Done.
- **Iris: `replicas` on `ResourceSpec`** — Done (kept on `ResourceSpec`).

### Ready to start

- **Fray-lite: Phase 1 — Core interface + LocalClient** —
  Create `fray/v2/` package. Define `Client` protocol, `ActorHandle` protocol,
  `ActorGroup`, `JobHandle`, `wait_all()`. Copy type definitions from v1.
  Implement `LocalClient`. Write tests. See Phase 1 spiral steps for breakdown.

- **Fray-lite: Phase 2 — Ray backend** —
  Copy v1 Ray code into `fray/v2/ray/`. Implement `RayClient` with deferred
  actor handles. Map `JobRequest.replicas` → `run_on_pod num_slices`.

### After Ray backend is ready

- **Fray-lite: Phase 3 — Migrate Levanter** —
  Swap `current_cluster()` → `current_client()` in `marin.training.training`.
  Simplest caller, validates the API surface.

- **Fray-lite: Phase 4 — Migrate RL** —
  Single atomic migration: swap all imports, refactor actor creation to
  controller-only, update all worker call sites. No mixed v1/v2.

- **Fray-lite: Phase 5 — Migrate Zephyr** —
  Implement `ActorBackend` with `ZephyrWorker` actors. Delete `JobContext`
  and `fray.job`.

### After all callers migrated on Ray

- **Fray-lite: Phase 6 — Iris backend** —
  Implement `FrayIrisClient` wrapping `iris.client.IrisClient`. Deferred
  `IrisActorHandle` with context-based serialization. Run all workloads
  against Iris.

- **Fray-lite: Phase 7 — Delete v1** —
  Remove `fray.cluster`, `fray.job`, `fray.queue`. Move `fray.v2` → `fray`.
  Update all imports.

## Recommendations

### 1. Iris `replicas` field location differs from original design

The original design called for moving `replicas` from `ResourceSpecProto` to
`LaunchJobRequest`. In practice, Iris keeps `replicas` on `ResourceSpec`
(`iris.cluster.types.ResourceSpec.replicas`). This is fine — fray-lite's
`FrayIrisClient` will map `JobRequest.replicas` → `ResourceSpec.replicas`
during conversion. No Iris change needed.
