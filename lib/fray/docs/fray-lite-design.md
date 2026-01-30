# Fray-Lite Design

**Status**: Draft (v2, post-review)
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
2. **`ActorHandle`** — call actor methods with `.remote()` shim
3. **`ResourceConfig`** — extended from v1 with `preemptible` field
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
        """Create a named actor instance.

        Launches a job hosting the actor, waits for it to become available,
        and returns a handle for RPC calls. The caller is responsible for
        creating actors — there is no implicit singleton/get-if-exists behavior.
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

        Each instance runs as a separate job. The group exposes
        individual handles; callers (e.g. Zephyr) coordinate work
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
    2. FRAY_CLIENT_SPEC environment variable
    3. LocalClient (default)
    """
    ...
```

### ResourceConfig

`ResourceConfig` is extended from v1. The `preemptible` field already exists
on v1's `ResourceConfig`. Under Iris, `preemptible=False` maps to a constraint
`Constraint(key="preemptible", op=EQ, value="false")` once workers expose
their preemptibility as an attribute ([#2553](https://github.com/marin-community/marin/issues/2553)).

```python
# Unchanged from v1 except emphasis on preemptible field:
@dataclass
class ResourceConfig:
    cpu: int = 1
    ram: str = "128m"
    disk: str = "1g"
    device: DeviceConfig = field(default_factory=CpuConfig)
    replicas: int = 1
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
    num_tasks: int = 1  # For gang-scheduled multi-task jobs (TPU slices)
    max_retries_failure: int = 0
    max_retries_preemption: int = 100


class JobHandle(Protocol):
    @property
    def job_id(self) -> str: ...

    def wait(self, timeout: float = 300.0, *, raise_on_failure: bool = True) -> JobStatus: ...

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

```python
# fray/v2/actor.py

class ActorHandle:
    """Handle to a remote actor with .method.remote() calling convention."""

    def __getattr__(self, method_name: str) -> ActorMethod:
        ...


class ActorMethod:
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
    """Group of actor instances. Callers coordinate work distribution themselves."""

    @property
    def handles(self) -> list[ActorHandle]:
        """Individual actor handles."""
        ...

    @property
    def jobs(self) -> list[JobHandle]:
        """Underlying job handles for lifecycle management."""
        ...
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

Wraps existing Ray cluster functionality. Jobs become Ray jobs, actors become
Ray actors.

```python
class RayClient(Client):
    def submit(self, request: JobRequest) -> JobHandle:
        # Delegates to existing RayCluster.launch()
        # Handles TPU jobs via run_on_pod
        # max_retries = max_retries_failure + max_retries_preemption
        ...

    def create_actor(self, actor_class, *args, name, resources, **kwargs) -> ActorHandle:
        # Map resources to Ray options:
        #   resources.preemptible=False → {"resources": {"head_node": 0.0001}}
        #   resources.cpu → num_cpus (default 0 for actors)
        # ray.remote(actor_class).options(name=name, ...).remote(*args, **kwargs)
        # Wraps ray actor handle in ActorHandle shim
        ...
```

### IrisClient (wraps `iris.client.IrisClient`)

The fray-lite `IrisClient` sits on top of the existing Iris client library.
The Iris client already provides job submission, endpoint registry, and
resolver — we wrap it rather than reimplementing.

```python
class FrayIrisClient(Client):
    """Iris cluster backend. Wraps iris.client.IrisClient."""

    def __init__(self, controller_address: str, workspace: Path | None = None):
        self._iris = IrisClientLib.remote(controller_address, workspace=workspace)

    def submit(self, request: JobRequest) -> JobHandle:
        # Convert ResourceConfig → Iris ResourceSpec
        #   resources.preemptible=False → Constraint(key="preemptible", op=EQ, value="false")
        # Convert Entrypoint → Iris Entrypoint
        # Submit via self._iris.submit(...)
        # Return JobHandle wrapping iris.client.Job
        ...

    def create_actor(self, actor_class, *args, name, resources, **kwargs) -> ActorHandle:
        # 1. Create Entrypoint wrapping _host_actor(actor_class, args, kwargs, name)
        # 2. Submit as Iris job with ports=["actor"]
        # 3. Wait for endpoint to appear in self._iris.resolver()
        # 4. Return ActorHandle wrapping Iris ActorClient
        ...
```

The actor hosting entrypoint:

```python
def _host_actor(actor_class, args, kwargs, name):
    """Entrypoint for actor-hosting Iris jobs."""
    from iris.actor import ActorServer
    from iris.client import iris_ctx

    ctx = iris_ctx()
    instance = actor_class(*args, **kwargs)

    server = ActorServer(host="0.0.0.0", port=ctx.get_port("actor"))
    server.register(name, instance)
    server.serve_background()

    # Register endpoint for discovery via Iris controller
    address = f"{_get_host_ip()}:{server.port}"
    ctx.registry.register(name, address)

    # Block until job is terminated
    import threading
    threading.Event().wait()
```

The `ActorHandle` for Iris wraps `iris.actor.ActorClient`:

```python
class IrisActorHandle(ActorHandle):
    def __init__(self, client: IrisActorClient):
        self._client = client

    def __getattr__(self, method_name: str) -> ActorMethod:
        return _IrisActorMethod(self._client, method_name)

class _IrisActorMethod(ActorMethod):
    def remote(self, *args, **kwargs) -> ActorFuture:
        # Run the synchronous Iris RPC call in a thread pool to return
        # a non-blocking future. The Iris ActorClient already handles
        # timeouts internally; we use its timeout and don't add a second one.
        executor = _get_shared_executor()
        future = executor.submit(
            lambda: getattr(self._client, self._method)(*args, **kwargs)
        )
        return ActorFuture(future)

    def __call__(self, *args, **kwargs) -> Any:
        return getattr(self._client, self._method)(*args, **kwargs)
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
    """Distributed Zephyr backend using fray-lite actor groups."""

    def __init__(self, client: Client, num_workers: int, resources: ResourceConfig):
        self.group = client.create_actor_group(
            ZephyrWorker,
            name="zephyr-worker",
            count=num_workers,
            resources=resources,
        )

    def execute(self, dataset: Dataset, hints: ExecutionHint) -> Sequence:
        plan = compute_plan(dataset, hints)
        shards = self._shards_from_source_items(plan.source_items)
        for stage in plan.stages:
            shards = self._execute_stage(stage, shards, hints)
        return list(self._materialize(shards))

    def _execute_stage(self, stage, shards, hints):
        # Assign shards to workers round-robin
        # Each worker reads shard data from shared storage, processes, writes back
        # On worker failure, reassign shard to another worker
        futures = []
        handles = self.group.handles
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

# Create shared actors first
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

# Wait for all
for job in jobs:
    job.wait(raise_on_failure=True)
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

### Phase 1: Core Interface + LocalClient

1. Create `fray/v2/` package with `types.py`, `client.py`, `actor.py`
2. Implement `LocalClient` backend (threads + in-process actors)
3. Write tests for core API (submit, create_actor, actor group, lifecycle)

### Phase 2: Ray Backend

1. Implement `RayClient` wrapping existing `RayCluster`
2. Ray `ActorHandle` wrapping Ray actor refs with `.remote()` shim
3. Map `preemptible=False` → `resources={"head_node": 0.0001}`
4. Map `max_retries_failure + max_retries_preemption` → Ray retry count
5. Test with existing Ray cluster

### Phase 3: Iris Backend

1. Implement `FrayIrisClient` wrapping `iris.client.IrisClient`
2. Implement actor hosting entrypoint + endpoint registration
3. Map `preemptible=False` → Iris constraint (requires [#2553](https://github.com/marin-community/marin/issues/2553))
4. Test with local Iris controller

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
├── __init__.py          # Re-exports: Client, current_client, JobRequest, etc.
├── types.py             # JobRequest, JobHandle, JobStatus, Entrypoint, EnvironmentConfig
├── client.py            # Client protocol, current_client(), set_current_client()
├── actor.py             # ActorHandle, ActorMethod, ActorFuture, ActorGroup
├── local.py             # LocalClient implementation
├── ray_backend.py       # RayClient implementation
└── iris_backend.py      # FrayIrisClient implementation (wraps iris.client.IrisClient)
```

## Design Decisions

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
