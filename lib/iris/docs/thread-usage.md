# Iris Thread Usage Audit & Management Design

## Current Thread Inventory

### Summary

- **13 `threading.Thread` creations** across 7 components (all currently daemon)
- **8 `threading.Event` uses** for shutdown signaling
- **4 `ThreadPoolExecutor` instances** with proper shutdown
- **3 `threading.Lock` / `RLock` uses** for state protection

### Component-by-Component Audit

#### 1. Controller (`cluster/controller/controller.py`)

**Threads created (3):**

| Thread | Line | Purpose | Daemon | Shutdown Signal |
|--------|------|---------|--------|-----------------|
| Scheduling loop | ~326 | Finds task assignments, checks timeouts | Yes | `_stop` flag + `_wake_event` |
| Heartbeat loop | ~333 | Sends RPC heartbeats to workers | Yes | `_stop` flag + `_heartbeat_event` |
| Server thread | ~340 | Runs uvicorn dashboard HTTP server | Yes | `server.should_exit = True` |

**Thread pool:** Dispatch executor (~line 302), 32 workers for heartbeat dispatch. Shutdown with `wait=True, cancel_futures=False`.

**Locks:** `_dispatch_lock` protects per-worker outboxes.

**Shutdown quality:** Good — wakes all threads before joining, no timeout (relies on pytest-timeout).

#### 2. Worker (`cluster/worker/worker.py`)

**Threads created (2 + N per task):**

| Thread | Line | Purpose | Daemon | Shutdown Signal |
|--------|------|---------|--------|-----------------|
| Server thread | ~129 | RPC endpoint server | Yes | `server.should_exit = True` |
| Lifecycle thread | ~149 | Register, serve, reset on timeout | Yes | `_stop_event` |
| Per-task threads | ~393 | One daemon thread per submitted task | Yes | `attempt.should_stop` flag |

**Locks:** `_lock` protects `_tasks` dict.

**Shutdown quality:** Decent — joins with 5-second timeout, event-based signaling. Per-task threads are not explicitly joined on shutdown.

#### 3. Autoscaler (`cluster/vm/autoscaler.py`)

**Thread pool:** Scale-up executor (~line 200), dynamic max_workers. Proper shutdown in `shutdown()` method. Has `_wait_for_inflight()` for test synchronization.

No raw threads.

#### 4. Actor Pool (`actor/pool.py`)

**Thread pool:** Broadcast executor (~line 111), 32 workers for broadcast RPC. Shutdown via `__exit__` context manager.

**Locks:** `_lock` protects `_cached_result` and `_endpoint_index`.

No raw threads.

#### 5. Actor Server (`actor/server.py`) — BROKEN

**Threads created (1):**

| Thread | Line | Purpose | Daemon | Shutdown Signal |
|--------|------|---------|--------|-----------------|
| Server thread | ~227 | Runs uvicorn via `serve_background()` | Yes | **None** |

**Critical issues:**
- `shutdown()` method is empty (`pass`)
- Server instance is a local variable in `serve_background()` — never stored, so it can't be stopped
- Relies entirely on daemon thread + process exit for cleanup

#### 6. Worker Pool Dispatcher (`client/worker_pool.py`)

**Threads created (1 per worker):**

| Thread | Line | Purpose | Daemon | Shutdown Signal |
|--------|------|---------|--------|-----------------|
| WorkerDispatcher | ~220 | Discovers worker endpoint, dispatches tasks | Yes | `_shutdown` event |

**Shutdown quality:** Good — `WorkerPool.shutdown()` sets event and joins each dispatcher.

#### 7. Local Platform (`cluster/vm/local_platform.py`)

**Threads created (1 per container):**

| Thread | Line | Purpose | Daemon | Shutdown Signal |
|--------|------|---------|--------|-----------------|
| Container exec thread | ~110 | Runs callable or subprocess | Yes | `_killed` event |

**Shutdown quality:** Decent — `kill()` joins with 0.5-second timeout.

---

## Uvicorn Background Server Patterns

### How Iris Uses Uvicorn

Three components run uvicorn in background threads:

| Component | Stores Server Ref | Shutdown Method | Join Timeout | Status |
|-----------|-------------------|-----------------|--------------|--------|
| Worker | Yes (instance var) | `should_exit = True` | 5s | Correct |
| Controller | Yes (instance var) | `should_exit = True` | None | Missing timeout |
| ActorServer | **No** (local var) | `pass` | No join | **Broken** |

### Correct Pattern for Uvicorn in Threads

```python
# Store server reference as instance variable
self._server: uvicorn.Server | None = None
self._server_thread: threading.Thread | None = None

def _run_server(self) -> None:
    config = uvicorn.Config(app, host="0.0.0.0", port=port)
    self._server = uvicorn.Server(config)
    self._server.run()  # Blocking

def start(self) -> None:
    self._server_thread = threading.Thread(target=self._run_server)
    self._server_thread.start()
    # Wait for startup
    while self._server is None or not self._server.started:
        time.sleep(0.01)

def stop(self) -> None:
    if self._server:
        self._server.should_exit = True
    if self._server_thread:
        self._server_thread.join(timeout=5.0)
```

Key points:
- Modern uvicorn (>=0.23.0) automatically skips signal handler installation when not in the main thread
- `should_exit` is thread-safe to set from any thread
- Always use `join(timeout=...)` to prevent indefinite hangs

---

## Why Daemon Threads Are Harmful

Every thread in Iris is currently `daemon=True`. This is problematic:

1. **No cleanup guarantees.** Daemon threads are killed abruptly on process exit — no `finally` blocks, no resource cleanup, no flushing.
2. **Masks bugs.** If shutdown logic is broken (like ActorServer), daemon mode hides it because the thread dies with the process anyway.
3. **Logging to closed streams.** In pytest, the process doesn't exit between tests — daemon threads keep running and log to streams that pytest has already closed, causing `ValueError: I/O operation on closed file`.
4. **No join enforcement.** Since daemon threads don't block exit, developers skip writing proper `join()` calls.

**With proper managed lifecycle (Event signaling + explicit join), daemon mode is unnecessary.** Non-daemon threads with proper shutdown are strictly better: they guarantee cleanup runs, and if shutdown is broken, tests hang visibly instead of producing cryptic stream errors.

---

## Design Options for Managed Thread Abstraction

### Option A: `ManagedThread` Wrapper Class

A thin wrapper around `threading.Thread` that enforces the event-based shutdown pattern.

```python
class ManagedThread:
    def __init__(
        self,
        target: Callable[[threading.Event], None],
        name: str | None = None,
        registry: ThreadRegistry | None = None,
    ):
        self._shutdown = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=name, daemon=False
        )
        self._target = target
        self._registry = registry

    def _run(self) -> None:
        try:
            self._target(self._shutdown)
        finally:
            if self._registry:
                self._registry._remove(self)

    def start(self) -> None:
        if self._registry:
            self._registry._add(self)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> bool:
        self._shutdown.set()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def shutdown_event(self) -> threading.Event:
        return self._shutdown
```

**Pros:**
- Simple, explicit API
- Forces callers to accept a shutdown `Event` in their target function
- Non-daemon by default
- Optional registry integration

**Cons:**
- Requires refactoring every thread target to accept `Event` as first arg
- Doesn't handle `ThreadPoolExecutor` (different abstraction)

### Option B: `ThreadRegistry` / `ThreadManager` with Context Manager

A registry that tracks all threads and provides bulk shutdown. Components register their threads (whether ManagedThread or raw Thread).

```python
class ThreadRegistry:
    def __init__(self) -> None:
        self._threads: list[ManagedThread] = []
        self._lock = threading.Lock()

    def create_thread(
        self,
        target: Callable[[threading.Event], None],
        name: str | None = None,
    ) -> ManagedThread:
        thread = ManagedThread(target=target, name=name, registry=self)
        return thread

    def _add(self, thread: ManagedThread) -> None:
        with self._lock:
            self._threads.append(thread)

    def _remove(self, thread: ManagedThread) -> None:
        with self._lock:
            self._threads = [t for t in self._threads if t is not thread]

    def shutdown(self, timeout: float = 10.0) -> list[str]:
        """Stop all threads. Returns names of threads that didn't stop."""
        with self._lock:
            threads = list(self._threads)

        # Signal all threads first (parallel)
        for t in threads:
            t._shutdown.set()

        # Then join all with per-thread timeout
        per_thread = timeout / max(len(threads), 1)
        stuck = []
        for t in threads:
            t._thread.join(per_thread)
            if t._thread.is_alive():
                stuck.append(t._thread.name or "<unnamed>")
        return stuck

    def __enter__(self) -> ThreadRegistry:
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()
```

**Pros:**
- Bulk shutdown with a single call — ideal for test fixtures
- Context manager for scoped thread lifetime
- Tracks all threads centrally
- Can report stuck threads

**Cons:**
- Adds a shared mutable registry — requires passing it through components
- Components need to know about the registry (or receive it via DI)

### Option C: Implicit Global Registry via `threading.local` / Module-Level

A module-level registry that `ManagedThread` auto-registers into.

```python
_global_registry = ThreadRegistry()

class ManagedThread:
    def __init__(self, target, name=None):
        ...
        _global_registry._add(self)

def shutdown_all_threads(timeout=10.0):
    return _global_registry.shutdown(timeout)
```

**Pros:**
- Zero boilerplate — threads auto-register
- Simple `shutdown_all_threads()` call in test teardown

**Cons:**
- Global mutable state is hard to reason about
- Tests that run in parallel could interfere
- Harder to scope thread lifetimes to specific components

### Option D: Protocol-Based (No Wrapper Class)

Define a `Stoppable` protocol and require all thread-owning components to implement it. Use a registry of `Stoppable` objects rather than individual threads.

```python
class Stoppable(Protocol):
    def stop(self) -> None: ...
    def join(self, timeout: float | None = None) -> None: ...
    @property
    def is_alive(self) -> bool: ...

class ComponentRegistry:
    def __init__(self) -> None:
        self._components: list[Stoppable] = []

    def register(self, component: Stoppable) -> None:
        self._components.append(component)

    def shutdown(self, timeout: float = 10.0) -> None:
        for c in self._components:
            c.stop()
        for c in self._components:
            c.join(timeout / max(len(self._components), 1))
```

**Pros:**
- Works with existing components (Controller, Worker already have stop/join)
- No need to change thread creation patterns
- Protocol-based — type-safe, testable

**Cons:**
- Doesn't enforce proper internal thread management
- Components could still use daemon threads internally
- Coarser granularity (component-level, not thread-level)

---

## Recommendation

**Use Option A + B together: `ManagedThread` + `ThreadRegistry`.**

Rationale:
- `ManagedThread` enforces the correct pattern at the thread level (non-daemon, event-based shutdown, explicit join)
- `ThreadRegistry` provides the bulk shutdown mechanism needed for tests
- Components create threads via the registry, which provides test fixtures a single `shutdown()` call
- The registry is explicit (passed via DI), not global — avoids Option C's shared state problems
- Option D is too coarse — we need thread-level control to fix issues like ActorServer's broken shutdown

### Migration Strategy

1. Introduce `ManagedThread` and `ThreadRegistry` in a new module (`iris/threading.py` or similar)
2. Fix ActorServer first (most broken — store server ref, implement shutdown)
3. Migrate components one at a time: Controller → Worker → WorkerDispatcher → LocalPlatform
4. For each migration:
   - Change thread target to accept `shutdown: threading.Event`
   - Replace `threading.Thread(daemon=True)` with `ManagedThread(...)`
   - Remove component-specific `_stop_event` / `_stop` flag (use the one from ManagedThread)
   - Switch from `daemon=True` to `daemon=False`
5. Add test fixtures that create a `ThreadRegistry` and call `registry.shutdown()` in teardown
6. `ThreadPoolExecutor` instances don't need wrapping — they already have proper `shutdown()` — but the registry could track them separately if needed

### Eliminating Daemon Threads

With this design, **all daemon threads can be converted to non-daemon**:
- Every thread gets a `shutdown Event` and checks it in its loop
- Every thread is joined with a timeout during shutdown
- If a thread doesn't stop within the timeout, tests report it explicitly instead of silently leaking

The only case where daemon threads might still be appropriate is for truly fire-and-forget work that has no resources to clean up — but none of the current Iris threads fit that description.
