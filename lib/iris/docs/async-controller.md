# Async Controller: WSGI → ASGI Migration

## Motivation

The Iris controller serves RPC via a synchronous WSGI application
(`ControllerServiceWSGIApplication`) wrapped in Starlette's `WSGIMiddleware`,
running inside uvicorn (an async server). Every incoming RPC call is dispatched
to a worker thread via `anyio.to_thread.run_sync()`. The default thread pool is
capped at **40 threads** by anyio's `CapacityLimiter`.

This creates two problems:

1. **Thread starvation under load.** Each RPC call occupies one thread for its
   entire duration, including time spent waiting on outbound I/O (fetching logs
   from workers, writing bundles to GCS). With 10 concurrent trials polling
   `get_job_status` every 5 seconds, plus dashboard users, plus log fetches
   that can take 10+ seconds per worker, the 40-thread pool can saturate.

2. **Sequential I/O in hot paths.** `get_task_logs` fetches logs from multiple
   workers sequentially (one RPC at a time, line 1030-1034 in `service.py`).
   An async implementation could issue these RPCs concurrently with
   `asyncio.gather`, dramatically reducing latency.

The ConnectRPC codegen already produces both sync and async interfaces. The
async service protocol (`ControllerService`), ASGI application
(`ControllerServiceASGIApplication`), and async client
(`WorkerServiceClient`) all exist in `cluster_connect.py`. The migration is
a matter of wiring them up.

## Current Architecture

```
uvicorn (async event loop)
  └── Starlette (ASGI)
        ├── Route("/")                    → async dashboard handler
        ├── Route("/job/{id}")            → async dashboard handler
        ├── Route("/health")              → async health handler
        └── Mount("/iris.cluster.ControllerService/")
              └── WSGIMiddleware          → bridges ASGI↔WSGI
                    └── ControllerServiceWSGIApplication (sync ConnectRPC)
                          └── ControllerServiceImpl (sync methods)
                                └── self._state (RLock-protected)
                                └── self._controller (stub_factory → sync stubs)
```

Each RPC request follows this path:
1. uvicorn receives HTTP request on async event loop
2. Starlette routes to `WSGIMiddleware`
3. `WSGIMiddleware` calls `anyio.to_thread.run_sync(wsgi_handler)` — blocks one
   of 40 pool threads
4. ConnectRPC WSGI app deserializes protobuf, calls sync service method
5. Service method accesses `ControllerState` (RLock), possibly makes outbound
   sync RPCs to workers
6. Response flows back through the same chain

## Target Architecture

```
uvicorn (async event loop)
  └── Starlette (ASGI)
        ├── Route("/")                    → async dashboard handler
        ├── Route("/job/{id}")            → async dashboard handler
        ├── Route("/health")              → async health handler
        └── Mount("/iris.cluster.ControllerService/")
              └── ControllerServiceASGIApplication (async ConnectRPC)
                    └── AsyncControllerService (async methods)
                          └── self._state (RLock-protected, via to_thread)
                          └── async worker stubs (WorkerServiceClient)
```

No `WSGIMiddleware`, no thread pool bottleneck. I/O-bound operations (worker
RPCs, bundle uploads) run as native async tasks on the event loop.

## Method-by-Method Analysis

Each service method falls into one of three categories:

### Category A: Pure state access (no I/O)

These methods only read/write `ControllerState` under RLock. The lock
acquisition is fast (microseconds) and never blocks on I/O.

| Method | Notes |
|--------|-------|
| `get_job_status` | Iterates tasks, builds proto. Expensive for pending jobs (scheduling diagnostics) but CPU-only |
| `list_jobs` | Aggregates all jobs, sorts, paginates. CPU-bound |
| `get_task_status` | Single task lookup |
| `list_tasks` | Lists tasks for a job |
| `terminate_job` | Recursive tree walk, buffers kill RPCs |
| `register` | Worker registration event |
| `notify_task_update` | Calls wake(), returns empty |
| `register_endpoint` | Creates endpoint in state |
| `unregister_endpoint` | Removes endpoint from state |
| `lookup_endpoint` | State lookup |
| `list_endpoints` | State prefix scan |
| `get_autoscaler_status` | Reads autoscaler + state |
| `get_process_logs` | Reads log ring buffer |
| `get_transactions` | Reads transaction log |

**Migration strategy:** Wrap the entire sync body in `asyncio.to_thread()`:

```python
async def get_job_status(self, request, ctx):
    return await asyncio.to_thread(self._sync.get_job_status, request, ctx)
```

This preserves the existing RLock semantics (RLock is acquired in the thread,
not on the event loop) and keeps the state access thread-safe with respect to
the background scheduling/heartbeat threads that also hold the lock.

### Category B: Outbound worker RPCs

These methods make synchronous RPCs to workers and benefit most from async.

| Method | I/O Pattern |
|--------|-------------|
| `get_task_logs` | Sequential RPCs to N workers via `stub.fetch_task_logs()` |
| `profile_task` | Single RPC to one worker via `stub.profile_task()` |

**Migration strategy:** Replace sync worker stubs with async stubs and use
`asyncio.gather` for concurrent fan-out:

```python
async def get_task_logs(self, request, ctx):
    # Phase 1-2: collect requests and compute timeouts (state access)
    fetch_requests, immediate_errors, per_worker_timeout_ms = (
        await asyncio.to_thread(self._prepare_log_fetch, request, ctx)
    )

    # Phase 3: fetch logs concurrently from all workers
    async def fetch_one(req):
        stub = self._async_stub_factory.get_stub(req.worker_address)
        try:
            resp = await stub.fetch_task_logs(
                worker_pb2.FetchTaskLogsRequest(...),
                timeout_ms=per_worker_timeout_ms,
            )
            return _LogFetchResult(task_id_wire=req.task_id_wire, ...)
        except Exception as e:
            return _LogFetchResult(task_id_wire=req.task_id_wire, error=str(e))

    fetch_results = await asyncio.gather(*[fetch_one(r) for r in fetch_requests])

    # Phase 4: aggregate (pure computation)
    return self._aggregate_log_results(immediate_errors, fetch_results, request)
```

This is the highest-value change: log fetching from 8 workers goes from
sequential (8 × timeout) to concurrent (1 × timeout).

### Category C: External storage I/O

| Method | I/O Pattern |
|--------|-------------|
| `launch_job` | `BundleStore.write_bundle()` — fsspec write to GCS or local FS |
| `get_vm_logs` | `autoscaler.get_init_log()` — may hit GCP APIs |

**Migration strategy:** Wrap in `asyncio.to_thread()`. fsspec doesn't have a
native async API worth depending on, and bundle uploads are infrequent
(once per job submission):

```python
async def launch_job(self, request, ctx):
    return await asyncio.to_thread(self._sync.launch_job, request, ctx)
```

## Implementation Plan

### Phase 1: Async wrapper (minimal change, immediate benefit)

Create `AsyncControllerService` that wraps the existing sync
`ControllerServiceImpl`. Every method delegates to `asyncio.to_thread()`:

```python
# controller/async_service.py

class AsyncControllerService:
    """Async wrapper around ControllerServiceImpl.

    Delegates all RPC methods to asyncio.to_thread() so they run in
    worker threads without blocking the event loop. This eliminates the
    WSGIMiddleware thread pool bottleneck while preserving the existing
    sync implementation unchanged.
    """

    def __init__(self, sync_service: ControllerServiceImpl):
        self._sync = sync_service

    async def get_job_status(self, request, ctx):
        return await asyncio.to_thread(self._sync.get_job_status, request, ctx)

    async def list_jobs(self, request, ctx):
        return await asyncio.to_thread(self._sync.list_jobs, request, ctx)

    # ... one line per method
```

Update `dashboard.py`:

```python
def _create_app(self) -> Starlette:
    async_service = AsyncControllerService(self._service)
    rpc_app = ControllerServiceASGIApplication(service=async_service)

    routes = [
        Route("/", self._dashboard),
        Route("/job/{job_id:path}", self._job_detail_page),
        Route("/vm/{vm_id:path}", self._vm_detail_page),
        Route("/health", self._health),
        Mount(rpc_app.path, app=rpc_app),  # Direct ASGI mount, no WSGIMiddleware
        static_files_mount(),
    ]
    return Starlette(routes=routes)
```

**What this gives us:**
- Eliminates `WSGIMiddleware` and its fixed 40-thread anyio limiter
- Each request still runs in a thread (via `to_thread`), but now uses
  asyncio's default executor which can grow as needed
- No changes to `ControllerServiceImpl` or `ControllerState`
- Low risk — behavior is identical, only the dispatch mechanism changes

**What this does NOT give us:**
- Concurrent worker log fetching (still sequential within `to_thread`)
- True async I/O benefits

### Phase 2: Async worker RPCs (high-value optimization)

Replace the sync worker stub factory with an async one for outbound RPCs,
and make `get_task_logs` and `profile_task` natively async.

**Step 2a: Async stub factory**

```python
# controller/controller.py

class AsyncRpcWorkerStubFactory:
    """Caches async WorkerServiceClient stubs by address."""

    def __init__(self, timeout: Duration = Duration.from_seconds(10.0)):
        self._timeout = timeout
        self._stubs: dict[str, WorkerServiceClient] = {}

    def get_stub(self, address: str) -> WorkerServiceClient:
        if address not in self._stubs:
            self._stubs[address] = WorkerServiceClient(
                address=f"http://{address}",
                timeout_ms=self._timeout.to_ms(),
            )
        return self._stubs[address]

    def evict(self, address: str) -> None:
        stub = self._stubs.pop(address, None)
        if stub:
            stub.close()
```

Note: the async `WorkerServiceClient` uses `httpx.AsyncClient` internally,
which provides connection pooling with proper async keepalive handling.

**Step 2b: Natively async log fetching**

Override `get_task_logs` in `AsyncControllerService` to use concurrent
fan-out instead of delegating to the sync version:

```python
async def get_task_logs(self, request, ctx):
    # Prepare fetch requests using sync state access
    fetch_requests, immediate_errors, per_worker_timeout_ms = (
        await asyncio.to_thread(self._sync._prepare_log_fetch, request, ctx)
    )

    # Fan out to all workers concurrently
    results = await asyncio.gather(*[
        self._fetch_worker_logs_async(req, per_worker_timeout_ms)
        for req in fetch_requests
    ])

    return self._sync._aggregate_log_results(
        immediate_errors, results, request
    )
```

**What this gives us:**
- Log fetching from N workers completes in O(1 × timeout) instead of
  O(N × timeout)
- Profile requests use async I/O, don't hold a thread while waiting

### Phase 3: Native async state access (optional, future)

Replace `RLock` with `asyncio.Lock` and make state methods async. This
eliminates all `to_thread` calls for pure state access.

**This is NOT recommended short-term** because:
- The background threads (scheduling loop, heartbeat loop, autoscaler loop)
  also access `ControllerState` and are not async
- Converting those loops to async tasks is a large refactor with concurrency
  risk
- The `to_thread` overhead for state access is negligible (microseconds)
  compared to the benefit of Phase 1+2

If we ever want this, the path is:
1. Convert background loops to `asyncio.Task` running on the uvicorn event loop
2. Replace `RLock` with `asyncio.Lock`
3. Remove all `to_thread` wrappers
4. This fundamentally changes the concurrency model from threads to
   cooperative async, which is a significant architectural change

## Concurrency Safety

### RLock + asyncio.to_thread

When `asyncio.to_thread(fn)` runs `fn` in a thread, that thread can safely
acquire the `RLock`. Multiple concurrent `to_thread` calls will block on the
lock as expected — this is identical to the current `WSGIMiddleware` behavior.

The background threads (scheduling, heartbeat, autoscaler) also acquire the
same `RLock`. This continues to work correctly because `to_thread` uses real
OS threads, not cooperative async tasks.

### Stub factory thread safety

The current sync `RpcWorkerStubFactory` uses a `threading.Lock` to protect
its cache. The async version doesn't need a lock because stub creation
happens on the event loop (single-threaded). However, if `get_stub` is ever
called from a `to_thread` context, it needs a lock. Safest approach: keep a
lock, it costs nothing in the uncontended case.

## Migration Checklist

### Phase 1
- [ ] Create `AsyncControllerService` in `controller/async_service.py`
- [ ] Update `ControllerDashboard._create_app()` to use ASGI app
- [ ] Remove `WSGIMiddleware` import from `dashboard.py`
- [ ] Update `ControllerDashboard.__init__` type hint if needed
- [ ] Run full test suite (`lib/iris/tests/cluster/`)
- [ ] Load test with concurrent `get_job_status` polling

### Phase 2
- [ ] Create `AsyncRpcWorkerStubFactory` with `WorkerServiceClient`
- [ ] Extract `_prepare_log_fetch` and `_aggregate_log_results` helpers
      from `ControllerServiceImpl.get_task_logs` (pure computation, no I/O)
- [ ] Implement async `get_task_logs` with `asyncio.gather` fan-out
- [ ] Implement async `profile_task` with async worker stub
- [ ] Add `evict` support to async stub factory
- [ ] Run log-fetching integration tests

## Files Changed

### Phase 1
| File | Change |
|------|--------|
| `cluster/controller/async_service.py` | New file: async wrapper |
| `cluster/controller/dashboard.py` | ASGI mounting, drop WSGIMiddleware |

### Phase 2
| File | Change |
|------|--------|
| `cluster/controller/async_service.py` | Native async `get_task_logs`, `profile_task` |
| `cluster/controller/controller.py` | Add `AsyncRpcWorkerStubFactory` |
| `cluster/controller/service.py` | Extract `_prepare_log_fetch`, `_aggregate_log_results` helpers |

## Risks

1. **`asyncio.to_thread` vs `loop.run_in_executor`**: The actor server uses
   `loop.run_in_executor(self._executor)` with a custom executor to avoid
   issues during process cleanup. If the controller has similar lifecycle
   concerns, we should use the same pattern. For now, `asyncio.to_thread` is
   simpler and sufficient since the controller has clean shutdown via
   `stop_event`.

2. **Starlette deprecation warning**: `starlette.middleware.wsgi` is deprecated
   and will be removed. Migrating to ASGI preempts this.

3. **ConnectRPC `ctx.timeout_ms()`**: Verify the ASGI `RequestContext` exposes
   the same `timeout_ms()` method as the WSGI one. The codegen uses the same
   `RequestContext` type for both, so this should work.
