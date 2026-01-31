# Iris Backend Control Flow

This document explains how Zephyr executes on Iris, covering the full control flow from ZephyrContext through the Iris cluster infrastructure.

## Executive Summary

**Key Finding**: Iris automatically injects `IRIS_CONTROLLER_ADDRESS` into all task environments. The fray.v2 iris_backend should NOT manually set this - it's handled by Iris's worker infrastructure.

## High-Level Architecture

```
ZephyrContext
    ↓ (uses)
FrayIrisClient (fray.v2.iris_backend)
    ↓ (wraps)
IrisClient (iris.client.client)
    ↓ (submits jobs to)
Controller (iris.cluster.controller)
    ↓ (schedules tasks on)
Workers (iris.cluster.worker)
    ↓ (run tasks in)
Containers (Docker or Local subprocess)
    ↓ (inside task)
iris_ctx() → IrisContext (with IrisClient for inter-actor communication)
```

## Detailed Control Flow

### 1. Test/Application Layer

```python
# User code (e.g., test)
client = FrayIrisClient(controller_address="http://127.0.0.1:8080", workspace=Path("/workspace"))
ctx = ZephyrContext(client=client, num_workers=2)
ds = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)
results = list(ctx.execute(ds))
```

### 2. Actor Creation (ZephyrContext → FrayIrisClient)

When `ctx.execute(ds)` runs, ZephyrContext creates actors:

```python
# lib/zephyr/src/zephyr/execution.py:484-495
coordinator = client.create_actor(ZephyrCoordinator, name="zephyr-controller")
workers_group = client.create_actor_group(ZephyrWorker, name="zephyr-worker", count=2)
workers = workers_group.wait_ready()
```

**What happens**:
- `create_actor()` and `create_actor_group()` submit Iris jobs
- Each actor runs in its own job: `zephyr-controller-0`, `zephyr-worker-0`, `zephyr-worker-1`
- Jobs use `_host_actor()` entrypoint which starts an ActorServer

### 3. Job Submission (FrayIrisClient → IrisClient → Controller)

```python
# lib/fray/src/fray/v2/iris_backend.py:429
job = self._iris.submit(
    entrypoint=IrisEntrypoint.from_callable(_host_actor, actor_class, args, kwargs, actor_name),
    name=actor_name,
    resources=iris_resources,
    environment=iris_environment,  # User's env vars (NOT controller address!)
    ports=["actor"],
)
```

**Key Point**: User code does NOT set `IRIS_CONTROLLER_ADDRESS`. Iris injects it automatically.

### 4. Job Scheduling (Controller → Worker)

Controller receives job submission:
1. Creates JobState in controller
2. Autoscaler provisions resources
3. Controller assigns task to Worker via RPC: `Worker.RunTask(request)`

### 5. Task Execution (Worker → Container)

Worker receives RunTask RPC and creates TaskAttempt:

```python
# lib/iris/src/iris/cluster/worker/worker.py:443-454
attempt = TaskAttempt(
    config=config,
    worker_id=self._worker_id,
    controller_address=self._config.controller_address,  # ← Worker knows controller address
    ...
)
```

TaskAttempt builds environment for container:

```python
# lib/iris/src/iris/cluster/worker/task_attempt.py:391-397
iris_env = build_iris_env(
    self,
    self._worker_id,
    self._controller_address,  # ← Passed from Worker
    type(self._runtime),
)
env.update(iris_env)  # Iris env overrides user env
```

`build_iris_env()` injects system variables:

```python
# lib/iris/src/iris/cluster/worker/task_attempt.py:97-111
env["IRIS_JOB_ID"] = task.job_id
env["IRIS_TASK_ID"] = task.task_id
env["IRIS_WORKER_ID"] = worker_id

if controller_address:
    if runtime_type is DockerRuntime:
        env["IRIS_CONTROLLER_ADDRESS"] = _rewrite_address_for_container(controller_address)
    else:
        env["IRIS_CONTROLLER_ADDRESS"] = controller_address  # ← AUTOMATICALLY INJECTED
```

**Result**: Every task gets `IRIS_CONTROLLER_ADDRESS` without user intervention.

### 6. Inside the Task Container

When `_host_actor()` runs inside the container:

```python
# lib/fray/src/fray/v2/iris_backend.py:208-230
def _host_actor(actor_class: type, args: tuple, kwargs: dict, name: str) -> None:
    from iris.client.client import iris_ctx

    ctx = iris_ctx()  # ← Reads IRIS_CONTROLLER_ADDRESS from environment

    instance = actor_class(*args, **kwargs)
    server = ActorServer(host="0.0.0.0", port=ctx.get_port("actor"))
    server.register(name, instance)
    server.serve_background()

    address = f"{_get_host_ip()}:{server._actual_port}"
    ctx.registry.register(name, address)  # ← Registers actor endpoint in controller

    threading.Event().wait()  # Block forever
```

**iris_ctx() creates IrisContext**:

```python
# lib/iris/src/iris/client/client.py:963-990
def create_context_from_env() -> IrisContext:
    job_info = get_job_info()  # Reads IRIS_JOB_ID, IRIS_CONTROLLER_ADDRESS, etc.

    client = None
    if job_info.controller_address:  # ← Set by build_iris_env()
        client = IrisClient.remote(
            controller_address=job_info.controller_address,  # ← Now actor can call controller!
        )
        registry = client._registry

    return IrisContext.from_job_info(job_info, client=client, registry=registry)
```

**Result**: Every actor job has:
- `iris_ctx().client` - IrisClient to call controller and resolve other actors
- `iris_ctx().registry` - EndpointRegistry to register actor endpoints

### 7. Actor Registration and Discovery

**Registration** (happens in `_host_actor()`):

```python
ctx.registry.register(name, address)
```

This sends RPC to controller: `RegisterEndpoint(job_id=zephyr-worker-0, name=zephyr-worker-0, address=192.168.0.17:30001)`

Controller stores: `{job_namespace}/zephyr-worker-0 → 192.168.0.17:30001`

**Discovery** (happens when one actor calls another):

```python
# Worker calls coordinator
coordinator.add_message.remote(msg).result()
```

Flow:
1. `IrisActorHandle._resolve()` needs to find coordinator's address
2. Calls `iris_ctx().client.resolver_for_job(coordinator_job_id)`
3. Creates `NamespacedResolver` with coordinator's job namespace
4. Resolver queries controller: `ListEndpoints(namespace=zephyr-controller-0)`
5. Controller returns: `zephyr-controller-0 → 192.168.0.17:30000`
6. `ActorClient` connects to that address and makes RPC call

### 8. Namespaces and Cross-Job Communication

**Namespace Rules**:
- Each root job gets its own namespace derived from job_id
- Child jobs inherit parent's namespace
- Actors register in their job's namespace: `{job_namespace}/{actor_name}`

**Problem with Current Fray Implementation**:

When FrayIrisClient creates actors, it submits them as **separate root jobs**:
```
zephyr-controller-0  (namespace: zephyr-controller-0)
zephyr-worker-0      (namespace: zephyr-worker-0)
zephyr-worker-1      (namespace: zephyr-worker-1)
```

Each has a different namespace! Workers can't find coordinator in default namespace.

**Solution** (what `IrisActorHandle` does):

Store `job_id` with each handle:
```python
handle = IrisActorHandle(actor_name="zephyr-controller-0", job_id="zephyr-controller-0")
```

When resolving from a different job:
```python
# Inside zephyr-worker-0 job
ctx = iris_ctx()  # Has client because IRIS_CONTROLLER_ADDRESS was set
resolver = ctx.client.resolver_for_job("zephyr-controller-0")  # Look in coordinator's namespace
result = resolver.resolve("zephyr-controller-0")  # Find the coordinator
```

This requires:
1. ✅ `job_id` stored in IrisActorHandle
2. ✅ `iris_ctx().client` available (needs IRIS_CONTROLLER_ADDRESS)
3. ✅ Use `resolver_for_job()` instead of default resolver

## Environment Variable Injection Flow

### For Subprocess/Docker Entrypoints

```
ClusterManager.connect()
    ↓
LocalController(controller_address="http://127.0.0.1:8080")
    ↓
LocalVmManager(controller_address=...)  # Passed to scale groups
    ↓
Worker(WorkerConfig(controller_address=...))  # Worker knows how to reach controller
    ↓
TaskAttempt(..., controller_address=...)  # Each task attempt gets it
    ↓
build_iris_env() sets env["IRIS_CONTROLLER_ADDRESS"]  # Injected into container
    ↓
Container runs with IRIS_CONTROLLER_ADDRESS set
    ↓
iris_ctx() reads it and creates IrisClient
```

### For Callable Entrypoints (Local Platform)

When tasks run in-process via `_LocalContainer` (local platform, callable entrypoints like `_host_actor`), environment is propagated via **ContextVars** instead of subprocess environment:

```
_LocalContainer._execute() (threading.Thread)
    ├─ set_job_info() sets ContextVars:
    │    _job_info = JobInfo(
    │        job_id="zephyr-worker-0",
    │        task_id="task-123",
    │        controller_address="http://127.0.0.1:8080",  ← From TaskAttempt
    │        parent_job_id=None,  ← Parent job if nested
    │    )
    │
    └─ _host_actor(actor_class, args, kwargs, name)
         └─ ActorServer.serve_background()
              ├─ ctx = contextvars.copy_context()  ← Captures _job_info ContextVar
              └─ threading.Thread(target=ctx.run, args=(server.run,))
                   └─ uvicorn server (runs in copied context)
                        └─ asyncio event loop (inherits context)
                             └─ ActorServer.Call()
                                  └─ asyncio.to_thread(method, ...)  ← Copies task context
                                       └─ Handler thread
                                            └─ iris_ctx() → get_job_info()
                                                 └─ Reads _job_info ContextVar ✓
```

**Critical implementation detail**: `ActorServer.serve_background()` MUST use `copy_context()` to propagate ContextVars to the uvicorn thread. Without this, the thread starts with an empty context and handler threads cannot access `IRIS_CONTROLLER_ADDRESS`.

**Why not os.environ?** Local runs must NEVER mutate `os.environ` because:
- Makes tests non-hermetic (global state)
- Can cause flaky parallel tests
- Violates isolation principles
- Bad practice that could leak into production code

**Parent job relationship**: When copying the context, the `_job_info` ContextVar contains the current job's parent relationship. For root jobs submitted via `IrisClient.remote()`, `parent_job_id=None`. For nested jobs (jobs submitted from within other jobs), `parent_job_id` would be set, allowing proper job hierarchy tracking.

**Key Insight**: The controller address flows from the top (ClusterManager/bootstrap config) down through Workers into every task. For subprocess entrypoints, it's injected via environment variables. For callable entrypoints, it's propagated via ContextVars. User code NEVER sets it.

## What FrayIrisClient Should Do

**Current (WRONG)**:
```python
# DON'T DO THIS - Iris already injects it!
iris_environment = EnvironmentSpec(
    env_vars={"IRIS_CONTROLLER_ADDRESS": controller_address}
)
```

**Correct**:
```python
# Just convert user's environment, Iris handles the rest
iris_environment = convert_environment(environment)  # May be None
```

**Why this works**:
1. FrayIrisClient wraps an IrisClient that knows the controller address
2. When jobs are submitted, they go to that controller
3. Controller assigns tasks to Workers
4. Workers inject `IRIS_CONTROLLER_ADDRESS` automatically
5. Tasks can create `iris_ctx().client` to communicate

## Resolution: NamespacedResolver

```python
class NamespacedResolver:
    def __init__(self, cluster: RemoteClusterClient, namespace: Namespace):
        self._cluster = cluster
        self._namespace = namespace

    def resolve(self, name: str) -> ResolveResult:
        prefixed_name = f"{self._namespace}/{name}"
        response = self._cluster.list_endpoints(pattern=prefixed_name)
        return ResolveResult(endpoints=response.endpoints)
```

**How it's used**:
```python
# Inside zephyr-worker-0, need to call coordinator in zephyr-controller-0
coordinator_handle = IrisActorHandle("zephyr-controller-0", job_id="zephyr-controller-0")

# When handle._resolve() is called:
ctx = iris_ctx()  # Has client with controller address
resolver = ctx.client.resolver_for_job("zephyr-controller-0")  # Returns NamespacedResolver
result = resolver.resolve("zephyr-controller-0")  # Looks up "zephyr-controller-0/zephyr-controller-0"
```

## Summary: Who Sets What

| Component | Sets | How |
|-----------|------|-----|
| **User Code** | workspace, user env vars | FrayIrisClient constructor, EnvironmentConfig |
| **FrayIrisClient** | job_id in actor handles | IrisActorHandle(..., job_id=...) |
| **Iris Worker** | IRIS_CONTROLLER_ADDRESS, IRIS_JOB_ID, IRIS_TASK_ID, etc. | build_iris_env() called by TaskAttempt |
| **Task (iris_ctx)** | IrisClient, EndpointRegistry | create_context_from_env() |
| **Controller** | Endpoint mappings | RegisterEndpoint RPC from actors |

## The Bug

The current FrayIrisClient manually sets `IRIS_CONTROLLER_ADDRESS`, which is:
1. Redundant (Iris already sets it)
2. Potentially wrong (might conflict with automatic injection)
3. Against Iris design (environment injection is Worker's job, not client's job)

**Fix**: Remove manual environment variable injection from `create_actor_group()`.
