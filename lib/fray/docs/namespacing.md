# Iris Job Namespacing and Fray v2 Integration

## How Iris Namespacing Works

### Job ID Hierarchy

Iris uses hierarchical, `/`-delimited job IDs to express parent-child relationships.
When `IrisClient.submit()` is called from within a running job, it reads the current
`IrisContext` (via `get_iris_ctx()`) and prepends the parent's job ID:

```python
# In iris.client.client.IrisClient.submit():
ctx = get_iris_ctx()
parent_job_id = ctx.job_id if ctx else None

if parent_job_id:
    job_id = JobId(f"{parent_job_id}/{name}")   # e.g. "root-job/child-job"
else:
    job_id = JobId(name)                         # top-level job
```

For multi-replica jobs, each task gets a task ID of `{job_id}/task-{N}`.

### Namespace Derivation

The `Namespace` is always the **first** path component of a job ID:

```
"abc123"                    -> Namespace("abc123")
"abc123/worker-0"           -> Namespace("abc123")
"abc123/worker-0/sub-task"  -> Namespace("abc123")
```

All jobs in a hierarchy share the same namespace, which provides actor isolation:
actors registered in one namespace cannot be discovered from another.

### IrisContext (`iris_ctx()`)

When Iris launches a task, the worker process (`task_attempt.py`) sets environment
variables including `IRIS_JOB_ID`, `IRIS_CONTROLLER_ADDRESS`, `IRIS_TASK_INDEX`, etc.
On first call, `iris_ctx()` reads these env vars via `get_job_info()` and constructs
an `IrisContext` with:

- `job_id`: the hierarchical job ID for this task
- `client`: an `IrisClient` connected back to the controller
- `registry`: a `NamespacedEndpointRegistry` that auto-prefixes actor names with the namespace
- `ports`: allocated ports

The context is stored in a `ContextVar`, so it is thread-local and available to all
code running within the job.

### Actor Registration and Discovery

The `NamespacedEndpointRegistry.register()` method auto-prefixes actor names:

```python
prefixed_name = f"{namespace_prefix}/{name}"   # e.g. "abc123/my-actor"
```

The `NamespacedResolver.resolve()` does the same prefix when looking up actors.
This means actors are automatically namespaced without callers needing to know
the namespace.

## How Fray v2 Currently Breaks Namespacing

### Problem 1: FrayIrisClient Creates a Fresh IrisClient, Bypassing Context

`FrayIrisClient.__init__()` creates its own `IrisClientLib.remote(controller_address)`:

```python
class FrayIrisClient:
    def __init__(self, controller_address, ...):
        self._iris = IrisClientLib.remote(controller_address, ...)
```

When `FrayIrisClient.create_actor_group()` calls `self._iris.submit()`, this goes
through `IrisClient.submit()`, which checks `get_iris_ctx()` for the parent job ID.
However, the `FrayIrisClient` was constructed from `FRAY_CLIENT_SPEC` parsing
(in `_parse_client_spec`), which creates a fresh `IrisClient` that has **no
relationship** to the `IrisContext` set up by the Iris worker.

The critical issue: `IrisClient.submit()` calls `get_iris_ctx()` to find the parent
job ID. If the `FrayIrisClient` was constructed at module import time or in a
different thread, the `ContextVar` may not have been populated yet, or may not
be visible. This means:

- Jobs submitted by Fray get **flat** names like `"coordinator"` and `"worker"`
  instead of `"parent-job/coordinator"` and `"parent-job/worker"`.
- Since namespace is derived from the root job ID component, flat names create
  **separate namespaces** per job, breaking cross-job actor discovery.

### Problem 2: Actor Name Registration Uses Context, but Discovery Doesn't Match

In `_host_actor()`, the actor registers its endpoint via `ctx.registry.register()`,
which uses `NamespacedEndpointRegistry` and auto-prefixes with the namespace.
But `IrisActorGroup.discover_new()` uses `self._iris_client.resolver_for_job(job.job_id)`,
which derives the namespace from the job's ID. If the job ID is flat (e.g. `"worker"`)
instead of hierarchical (e.g. `"parent/worker"`), the namespace won't match the
parent's namespace, and the resolver will look in the wrong namespace.

### Problem 3: Multi-Replica Job ID Structure

`IrisActorGroup` stores a single `IrisJobHandle` wrapping the multi-replica job.
It iterates `range(self._count)` and uses `self._jobs[i]`, but since there's only
one job, `self._jobs[0]` is used for all replicas. The `resolver_for_job(job.job_id)`
call uses this single job ID. If the job ID is `"worker"` (flat), the namespace is
`Namespace("worker")`. Meanwhile, the parent job has namespace `Namespace("parent-job")`.
These don't match, so endpoint registration (under parent's namespace) and discovery
(under child's namespace) are looking in different places.

## How `FRAY_CLIENT_SPEC` Clashes with `iris_ctx()`

### The Duplication

When Iris launches a task, `task_attempt.py` injects **both**:

1. `IRIS_JOB_ID`, `IRIS_CONTROLLER_ADDRESS`, etc. (consumed by `iris_ctx()` / `get_job_info()`)
2. `FRAY_CLIENT_SPEC=iris://{controller_address}` (consumed by `current_client()`)

These are two independent paths to the same controller, but they carry different
amounts of context:

| Mechanism | Knows parent job ID | Knows namespace | Knows ports |
|-----------|-------------------|-----------------|-------------|
| `iris_ctx()` via `IRIS_*` env vars | Yes (`IRIS_JOB_ID` is hierarchical) | Yes (derived from job ID) | Yes (`IRIS_PORT_*`) |
| `FRAY_CLIENT_SPEC` | No (only has controller address) | No | No |

### The Clash

When Fray code calls `current_client()`, it gets a `FrayIrisClient` constructed
from `FRAY_CLIENT_SPEC`. This client has no knowledge of the current job's
hierarchical position. When it submits child jobs via `self._iris.submit(name=...)`,
the `IrisClient` inside it *does* check `get_iris_ctx()`, but:

1. `iris_ctx()` may not have been called yet in this process, so the `ContextVar`
   may be empty.
2. Even if `iris_ctx()` was called, `FrayIrisClient` creates a **second**
   `IrisClient` instance, independent of the one in the context. The context's
   client and Fray's client are separate objects talking to the same controller.

The net effect: Fray creates a parallel, context-unaware path to Iris that
bypasses the automatic job hierarchy that Iris provides.

## Recommended Fixes

### Fix 1: Use `iris_ctx()` Client Instead of Creating a New One

When running inside an Iris job (i.e., `IRIS_JOB_ID` is set), `FrayIrisClient`
should reuse the `IrisClient` from `iris_ctx()` rather than constructing a new one.
This ensures that `IrisClient.submit()` sees the parent job ID and constructs
hierarchical job IDs automatically.

```python
class FrayIrisClient:
    def __init__(self, controller_address, ...):
        ctx = get_iris_ctx()
        if ctx is not None and ctx.client is not None:
            self._iris = ctx.client
        else:
            self._iris = IrisClientLib.remote(controller_address, ...)
```

### Fix 2: Ensure `iris_ctx()` Is Initialized Before Fray Creates Its Client

The `_parse_client_spec("iris://...")` path should call `iris_ctx()` early to
ensure the `ContextVar` is populated before any `IrisClient.submit()` calls
check it. Alternatively, `FrayIrisClient` could call `iris_ctx()` in its
constructor to force initialization.

### Fix 3: Stop Injecting `FRAY_CLIENT_SPEC` from `task_attempt.py`

Since `iris_ctx()` already provides everything needed, Fray's `current_client()`
could detect that it's running inside an Iris job (via `IRIS_JOB_ID` being set)
and automatically construct a `FrayIrisClient` from the existing context, without
needing `FRAY_CLIENT_SPEC` at all. This eliminates the duplication entirely:

```python
def current_client() -> Client:
    client = _current_client_var.get()
    if client is not None:
        return client

    # Prefer Iris context if available (running inside an Iris job)
    if os.environ.get("IRIS_JOB_ID"):
        from fray.v2.iris_backend import FrayIrisClient
        return FrayIrisClient.from_iris_context()

    spec = os.environ.get("FRAY_CLIENT_SPEC")
    ...
```

### Fix 4: Pass Parent Job ID Explicitly in `FrayIrisClient.submit()`

As a more targeted fix, `FrayIrisClient` could read the parent job ID from
`iris_ctx()` and prepend it to the job name when calling `self._iris.submit()`:

```python
def submit(self, request: JobRequest) -> IrisJobHandle:
    ctx = get_iris_ctx()
    name = request.name
    if ctx and ctx.job_id:
        name = f"{ctx.job_id}/{request.name}"
    ...
```

However, this duplicates logic that `IrisClient.submit()` already has. Fix 1 or
Fix 3 are preferable because they eliminate the duplication rather than working
around it.

### Recommended Approach

Fix 1 + Fix 3 together provide the cleanest solution: detect Iris context
automatically in `current_client()`, and reuse the context's `IrisClient`
when available. This removes `FRAY_CLIENT_SPEC` injection from `task_attempt.py`
entirely and lets Iris's native hierarchy management do its job.
