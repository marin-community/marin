# Iris Threading Model

**Status:** Design document for hierarchical cleanup migration
**Issue:** [#2604](https://github.com/marin-community/marin/issues/2604)
**Author:** Claude (investigating cleanup model)
**Date:** 2026-02-02

---

## Overview

This document provides a complete schematic of the Iris threading model for local execution, identifies cleanup issues with the current model, and proposes a hierarchical shutdown model that eliminates the need for ThreadRegistry.

**Key Finding:** Iris currently uses a flat, registry-based cleanup model where all threads register with a global `ThreadRegistry`. This creates problems when some threads remain unmanaged (outside the registry) and makes shutdown ordering non-hierarchical. The proposed model uses component ownership to create a clean shutdown cascade from `ClusterManager` → `Controller` → `Workers` → `Tasks`.

---

## Current Threading Model: Complete Inventory

### 1. Threading Primitives

#### ManagedThread (`managed_thread.py:34-70`)
Non-daemon thread wrapper with integrated shutdown:
```python
class ManagedThread:
    def __init__(self, target, *, name=None, args=()):
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=target,
            args=(self._stop_event, *args),
            daemon=False,  # Non-daemon for clean shutdown
            name=name,
        )
```

**Key Properties:**
- Target callable receives `stop_event` as first parameter
- Non-daemon by design (explicit shutdown required)
- Provides `start()`, `stop()`, `join()`, `is_alive`, `stop_event`

#### ThreadRegistry (`managed_thread.py:72-113`)
Global flat registry for bulk thread shutdown:
```python
class ThreadRegistry:
    def spawn(self, target, *, name=None, args=()):
        """Create, register, and start a ManagedThread."""
        thread = ManagedThread(target=target, name=name, args=args)
        with self._lock:
            self._threads.append(thread)
        thread.start()
        return thread

    def shutdown(self, timeout=None):
        """Stop all threads and join with deadline-based budget."""
        # Signal all threads to stop
        for thread in threads:
            thread.stop()
        # Join all threads (with optional timeout)
        for thread in threads:
            thread.join(timeout=remaining)
```

**Usage Pattern:**
- Global singleton: `_global_registry = ThreadRegistry()`
- Per-test swap: `set_thread_registry(registry)` in pytest fixtures
- Test cleanup: `registry.shutdown(timeout=5.0)` in teardown

#### ThreadContainer (`managed_thread.py:116-150`)
Component-scoped thread group with dual tracking:
```python
class ThreadContainer:
    def spawn(self, target, *, name=None, args=()):
        # Register in BOTH global registry AND local list
        thread = get_thread_registry().spawn(target=target, name=name, args=args)
        self._threads.append(thread)
        return thread

    def stop(self, timeout=5.0):
        """Signal all threads to stop, then join with shared deadline."""
        for thread in self._threads:
            thread.stop()
        # Join with deadline-based budget
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            remaining = max(0, deadline - time.monotonic())
            thread.join(timeout=remaining)
```

**Purpose:** Allows components to manage their own threads while still registering them globally for test cleanup.

---

### 2. Local Controller Threads

#### LocalController (`controller.py:953-1036`)
In-process controller for local testing:

**Initialization:**
- Creates temporary directory for bundles and cache
- Runs real `Controller` + `Autoscaler(LocalVmManagers)` in current process
- Workers are threads, not VMs

**No direct threads** - delegates to inner `Controller` (see below)

**Shutdown:**
```python
def stop(self):
    if self._controller:
        self._controller.stop()  # Cascade to inner Controller
        self._controller = None
```

---

### 3. Controller Threads

#### Controller (`controller.py:321-384`)

**Thread 1: Scheduling Loop** (`controller.py:326-330`)
```python
self._scheduling_loop_thread = threading.Thread(
    target=self._run_scheduling_loop,
    daemon=True,  # DAEMON - Problem for clean shutdown!
)
self._scheduling_loop_thread.start()
```
**Responsibilities:**
- Compute task assignments via scheduler
- Run autoscaler (calls `_autoscaler.run_once()`)
- Check worker timeout/heartbeat deadlines
- Wake on events: new job, worker registration, task completion

**Thread 2: Heartbeat Loop** (`controller.py:333-337`)
```python
self._heartbeat_loop_thread = threading.Thread(
    target=self._run_heartbeat_loop,
    daemon=True,  # DAEMON - Problem for clean shutdown!
)
self._heartbeat_loop_thread.start()
```
**Responsibilities:**
- Send heartbeats to all workers via `_heartbeat_all_workers()`
- Separate from scheduling so slow RPCs don't block scheduling
- Uses `ThreadPoolExecutor` for parallel dispatch (see below)

**Thread 3: Dashboard Server** (`controller.py:340-344`)
```python
self._server_thread = threading.Thread(
    target=self._run_server,
    daemon=True,  # DAEMON - Problem for clean shutdown!
)
self._server_thread.start()
```
**Responsibilities:**
- Run uvicorn HTTP server for dashboard UI
- Serve job/task/worker/VM status endpoints

**ThreadPoolExecutor: Dispatch Executor** (`controller.py:302-305`)
```python
self._dispatch_executor = ThreadPoolExecutor(
    max_workers=config.max_dispatch_parallelism,  # Default: 32
    thread_name_prefix="dispatch",
)
```
**Responsibilities:**
- Parallel heartbeat dispatch to workers
- Each worker heartbeat runs in separate thread from pool
- Prevents slow RPCs from blocking heartbeat loop

**Shutdown Cascade** (`controller.py:356-384`):
```python
def stop(self):
    self._stop = True
    self._wake_event.set()
    self._heartbeat_event.set()

    # 1. Join scheduling and heartbeat threads
    if self._scheduling_loop_thread:
        self._scheduling_loop_thread.join()  # No timeout - rely on pytest-timeout
    if self._heartbeat_loop_thread:
        self._heartbeat_loop_thread.join()

    # 2. Stop uvicorn server
    if self._server:
        self._server.should_exit = True
    if self._server_thread:
        self._server_thread.join()

    # 3. Shutdown dispatch executor
    self._dispatch_executor.shutdown(wait=True, cancel_futures=False)

    # 4. Shutdown autoscaler
    if self._autoscaler:
        self._autoscaler.shutdown()
```

---

### 4. Worker Threads

#### Worker (`worker.py:124-199`)

Each Worker (in LocalVmManager) spawns 2 lifecycle threads:

**Thread 1: Server Thread** (`worker.py:128-133`)
```python
self._server_thread = threading.Thread(
    target=self._run_server,
    daemon=True,  # DAEMON - Problem for clean shutdown!
)
self._server_thread.start()
```
**Responsibilities:**
- Run uvicorn HTTP server for RPC endpoints
- Handle incoming requests: `submit_task`, `kill_task`, `get_task`, etc.

**Thread 2: Lifecycle Thread** (`worker.py:149-153`)
```python
self._lifecycle_thread = threading.Thread(
    target=self._run_lifecycle,
    daemon=True,  # DAEMON - Problem for clean shutdown!
)
self._lifecycle_thread.start()
```
**Responsibilities:**
- Loop: reset → register → serve → timeout → repeat
- Register with controller on startup
- Handle heartbeat deadline expiration triggering re-registration
- Transition through states: IDLE → REGISTERING → RUNNING → IDLE

**Shutdown** (`worker.py:169-199`):
```python
def stop(self):
    self._stop_event.set()

    # Join lifecycle thread
    if self._lifecycle_thread:
        self._lifecycle_thread.join(timeout=5.0)

    # Stop server
    if self._server:
        self._server.should_exit = True
    if self._server_thread:
        self._server_thread.join(timeout=5.0)

    # Kill all running tasks
    for task_id in list(self._tasks.keys()):
        self.kill_task(task_id)

    # Release resources
    if self._port_allocator:
        self._port_allocator.release_all()
    if self._temp_dir:
        shutil.rmtree(self._temp_dir, ignore_errors=True)
```

---

### 5. Task Execution Threads (UNMANAGED!)

#### Worker.submit_task() (`worker.py:429-431`)

**⚠️ PROBLEM: Unmanaged daemon threads!**

```python
def submit_task(self, request: RunTaskRequest) -> str:
    # ... create TaskAttempt ...

    # Start execution in background
    thread = threading.Thread(
        target=attempt.run,
        daemon=True,  # DAEMON + UNTRACKED = Leak risk!
    )
    attempt.thread = thread
    thread.start()  # No registry, no ThreadContainer!

    return task_id
```

**Why This Is a Problem:**
1. **Not in ThreadRegistry** - test cleanup can't find these threads
2. **Daemon thread** - will be killed abruptly on process exit
3. **No stop event** - can't be signaled to exit gracefully
4. **Stored in `attempt.thread`** - but Worker.stop() just kills containers, doesn't join threads

**When Does This Cause Issues?**
- Tests that submit tasks then immediately tear down
- Worker shutdown while tasks are running
- Process exit with running tasks (daemon threads killed mid-execution)

**From Issue #2604:**
> Worker.submit_task() spawns a daemon thread at worker.py:370 that "cannot be cleanly shut down" because it's untracked. The fix: integrate with ThreadContainer and wire TaskAttempt.should_stop to stop events.

---

### 6. ManagedVM Threads (UNMANAGED!)

#### ManagedVM._run() (`managed_vm.py:345-346`)

**⚠️ PROBLEM: Unmanaged daemon threads!**

```python
class ManagedVM:
    def __init__(self, ...):
        # ...
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,  # DAEMON + UNTRACKED = Leak risk!
            name=f"vm-{vm_id}"
        )
        self._stop = threading.Event()

    def start(self):
        self._thread.start()  # No registry, no ThreadContainer!

    def stop(self):
        self._stop.set()  # Signal stop, but don't join!
```

**Why This Is a Problem:**
1. **Not in ThreadRegistry** - test cleanup can't find these threads
2. **Daemon thread** - will be killed abruptly on process exit
3. **No join in stop()** - just sets event, doesn't wait for thread to exit
4. **VM lifecycle threads** can be doing SSH operations, bootstrap, etc.

**When Does This Cause Issues?**
- Tests that start VMs then immediately tear down
- Autoscaler shutdown while VMs are bootstrapping
- Process exit with VMs in INITIALIZING state

**From Issue #2604:**
> ManagedVM — VM lifecycle threads at managed_vm.py:345 operate outside the registry. The recommendation is spawning via ThreadContainer with proper stop event checks instead of relying on daemon termination.

---

### 7. Bridge Threads (Intentional Daemons)

#### spawn_server() Bridge (`managed_thread.py:152-162`)

```python
def _stop_event_to_server(stop_event: threading.Event, server: Any) -> None:
    """Bridge a stop_event to a uvicorn Server's should_exit flag."""
    def _watch() -> None:
        stop_event.wait()
        server.should_exit = True

    threading.Thread(
        target=_watch,
        daemon=True,  # Intentional daemon - low risk
        name="stop-event-bridge"
    ).start()
```

**Why This Is Acceptable:**
- Very short-lived (just waits on event and sets flag)
- No resources to clean up
- Exits immediately when stop_event is set
- Documented as intentional daemon

**From Issue #2604:**
> _stop_event_to_server() — Bridge threads at managed_thread.py:162 are low-risk intentional daemons requiring no action unless full accounting is desired.

---

### 8. Local Container Execution Threads

#### _LocalContainer (`local_platform.py:98-181`)

**Thread per container:**
```python
class _LocalContainer:
    def start(self):
        self._thread = threading.Thread(
            target=self._execute,
            daemon=True,  # DAEMON - but owned by Worker lifecycle
        )
        self._thread.start()
```

**Why This Is Less Problematic:**
- Created by `_LocalContainerRuntime` which is owned by Worker
- Worker.stop() calls `self._runtime.kill()` which joins threads
- Threads check `self._killed` Event for graceful shutdown
- Indirectly managed through Worker lifecycle

**Execution Model:**
- Callable entrypoint: Direct Python function execution
- Command entrypoint: subprocess.run with captured output
- Streaming capture via `_StreamingCapture`

---

### 9. Autoscaler Threads

#### Autoscaler (`autoscaler.py:160-227`)

**ThreadPoolExecutor for scale-up:**
```python
self._scale_up_executor = ThreadPoolExecutor(
    max_workers=max(len(scale_groups), 4),
    thread_name_prefix="scale-up",
)
```

**Why This Works:**
- Properly shut down in `Autoscaler.shutdown()`
- Called from `Controller.stop()` cascade
- Uses `shutdown(wait=True)` to join all threads

**Shutdown:**
```python
def shutdown(self):
    self._scale_up_executor.shutdown(wait=True)
    for group in self._groups.values():
        group.terminate_all()  # Kills all VMs in group
```

---

### 10. Test Cleanup Infrastructure

#### Per-Test ThreadRegistry (`conftest.py:116-124`)

```python
@pytest.fixture(autouse=True)
def _thread_registry():
    """Fresh registry per test for thread leak detection."""
    registry = ThreadRegistry()
    old = set_thread_registry(registry)
    yield registry
    registry.shutdown(timeout=5.0)  # Block until all threads exit
    set_thread_registry(old)
```

**Purpose:**
- Isolate tests from each other's threads
- Catch hanging threads (timeout=5.0 will fail test)
- Ensure no threads leak between tests

**Session-End Hook** (`conftest.py:127-145`):
```python
def pytest_sessionfinish(session, exitstatus):
    """Print any non-daemon threads still alive at session end."""
    threads = [t for t in threading.enumerate() if not t.daemon and t != threading.current_thread()]
    if threads:
        print("\n=== Non-daemon threads still alive ===")
        for t in threads:
            print(f"  {t.name} (alive={t.is_alive()})")
```

---

## Thread Hierarchy Diagram: Current Model

```
Local Execution Thread Hierarchy (Current)
============================================

ThreadRegistry (Global Singleton)
├─ [All ManagedThreads registered here for test cleanup]
│
ClusterManager
└─ LocalController
   └─ Controller (in-process)
      ├─ scheduling_loop_thread (daemon) ────┐
      ├─ heartbeat_loop_thread (daemon) ─────┤
      ├─ server_thread (daemon) ─────────────┤  ALL REGISTERED IN
      ├─ dispatch_executor (ThreadPool) ─────┤  ThreadRegistry
      │  └─ dispatch-0..31 threads ──────────┤  (via ThreadContainer)
      └─ Autoscaler ─────────────────────────┤
         └─ scale_up_executor (ThreadPool) ──┘
            └─ scale-up-0..N threads

LocalVmManager (via Autoscaler)
└─ Worker (×N, in-process)
   ├─ server_thread (daemon) ──────────┐
   ├─ lifecycle_thread (daemon) ───────┤  REGISTERED IN
   │                                    │  ThreadRegistry
   └─ TaskAttempt Threads ─────────────┼─ ⚠️ UNMANAGED!
      └─ attempt.run() thread (daemon) ┘

ManagedVM (×M, for distributed execution)
└─ _run() thread (daemon) ───────────────── ⚠️ UNMANAGED!

Bridge Threads
└─ stop-event-bridge (daemon) ───────────── Intentional daemon (OK)
```

**Legend:**
- ✅ Managed: In ThreadRegistry, proper shutdown
- ⚠️ Unmanaged: NOT in ThreadRegistry, daemon, no cleanup

---

## Problems with Current Model

### 1. **Flat Registry Architecture**

**Problem:** All threads register with a global `ThreadRegistry`, creating a flat, non-hierarchical structure.

**Why This Is Bad:**
- No ownership hierarchy - can't tell which component owns which threads
- Shutdown order is non-deterministic (just iterates registry list)
- Can't cascade shutdown from parent to children
- Makes reasoning about lifecycle difficult

**Example:**
```python
# Controller creates threads and registers them globally
def start(self):
    self._scheduling_loop_thread = threading.Thread(...)
    ThreadRegistry.spawn(...)  # Registers in global list

# Worker creates threads and registers them globally
def start(self):
    self._server_thread = threading.Thread(...)
    ThreadRegistry.spawn(...)  # Registers in same global list

# Shutdown: Just iterate flat list - no hierarchy!
def shutdown(self):
    for thread in self._threads:  # Mixed Controller + Worker threads
        thread.stop()
```

### 2. **Unmanaged Daemon Threads**

**Problem:** Several critical thread creation sites bypass ThreadRegistry entirely.

**Unmanaged Locations:**
1. **Worker.submit_task()** (`worker.py:429`) - Task execution threads
2. **ManagedVM._thread** (`managed_vm.py:345`) - VM lifecycle threads
3. **Bridge threads** (`managed_thread.py:162`) - Low-risk, intentional

**Why This Is Bad:**
- Test cleanup can't find these threads
- Process exit kills them abruptly (daemon=True)
- No graceful shutdown possible
- Resource leaks in long-running processes

**From Issue #2604:**
> The Iris project contains daemon threads spawned outside the ThreadRegistry/ThreadContainer infrastructure, making them invisible to test cleanup mechanisms. This creates risks including "stuck pytest sessions" and "resource leaks in long-running processes."

### 3. **Inconsistent Daemon Usage**

**Problem:** Mix of daemon and non-daemon threads with unclear policy.

**Current State:**
- `ManagedThread`: **Non-daemon** by design (for clean shutdown)
- `Controller` threads: **Daemon** (scheduling, heartbeat, server)
- `Worker` threads: **Daemon** (server, lifecycle)
- `TaskAttempt` threads: **Daemon** (task execution)
- `ManagedVM` threads: **Daemon** (VM lifecycle)

**Why This Is Confusing:**
- `ManagedThread` explicitly avoids daemon for clean shutdown
- But most actual threads are daemon anyway!
- Daemon threads can't be joined gracefully
- Mixed approach makes lifecycle reasoning harder

### 4. **Dual-Tracking Overhead**

**Problem:** `ThreadContainer` registers threads in BOTH global registry AND local list.

**Why This Is Complex:**
- Every thread appears in two places
- Must maintain both lists
- Shutdown must coordinate both (or trust registry to handle it)
- Extra indirection and state

**Example:**
```python
class ThreadContainer:
    def spawn(self, ...):
        thread = get_thread_registry().spawn(...)  # Global registry
        self._threads.append(thread)  # Local list
        return thread
```

### 5. **No Hierarchical Shutdown**

**Problem:** Shutdown is not hierarchical - can't cascade from parent to children.

**Current Shutdown:**
```
Test Teardown
└─ ThreadRegistry.shutdown()
   └─ Iterate all threads (flat list)
      └─ Stop + Join each thread

NO HIERARCHY! Controller/Worker/Task relationships lost.
```

**What We Want:**
```
ClusterManager.stop()
└─ Controller.stop()
   ├─ Stop scheduling loop
   ├─ Stop heartbeat loop
   ├─ Stop server
   ├─ Shutdown dispatch executor
   └─ Autoscaler.shutdown()
      └─ LocalVmManager.stop()
         └─ Worker.stop() (×N)
            └─ TaskAttempt.stop() (×M)

HIERARCHICAL! Clean cascade from parent to children.
```

---

## Proposed Hierarchical Model

### Key Principles

1. **Component Ownership**: Each component owns its threads explicitly
2. **No Global Registry**: Remove ThreadRegistry, use hierarchical ownership instead
3. **Non-Daemon by Default**: All threads are non-daemon for explicit cleanup
4. **Cascading Shutdown**: Parent components stop children recursively
5. **Stop Events Everywhere**: Every thread receives a stop event, checks it regularly

### Thread Ownership Hierarchy

```
ClusterManager
└─ owns → Controller
   ├─ owns → SchedulingLoopThread
   ├─ owns → HeartbeatLoopThread
   ├─ owns → ServerThread
   ├─ owns → DispatchExecutor
   │  └─ owns → DispatchWorkerThread (×32)
   └─ owns → Autoscaler
      ├─ owns → ScaleUpExecutor
      │  └─ owns → ScaleUpWorkerThread (×N)
      └─ owns → LocalVmManager
         └─ owns → Worker (×N)
            ├─ owns → ServerThread
            ├─ owns → LifecycleThread
            └─ owns → TaskAttempt (×M)
               └─ owns → ExecutionThread
```

### Implementation Strategy

#### 1. Add ThreadOwner Protocol

```python
class ThreadOwner(Protocol):
    """Protocol for components that own and manage threads."""

    def stop(self, timeout: float = 5.0) -> None:
        """Stop all owned threads and child components.

        Must cascade to all children recursively.
        """
        ...
```

#### 2. Convert All Threads to Non-Daemon ManagedThreads

**Before (Controller):**
```python
self._scheduling_loop_thread = threading.Thread(
    target=self._run_scheduling_loop,
    daemon=True,  # ❌ Daemon - no clean shutdown
)
self._scheduling_loop_thread.start()
```

**After (Controller):**
```python
self._scheduling_loop = ManagedThread(
    target=self._run_scheduling_loop,
    name="controller-scheduling",
)
self._scheduling_loop.start()
```

#### 3. Add Component-Owned Thread Lists

**Before (Controller):**
```python
class Controller:
    def __init__(self):
        self._scheduling_loop_thread = None
        self._heartbeat_loop_thread = None
        self._server_thread = None
        # ... no explicit ownership tracking
```

**After (Controller):**
```python
class Controller:
    def __init__(self):
        self._threads: list[ManagedThread] = []
        # All threads added to this list for bulk management

    def start(self):
        self._threads.append(
            ManagedThread(target=self._run_scheduling_loop, name="controller-scheduling")
        )
        self._threads.append(
            ManagedThread(target=self._run_heartbeat_loop, name="controller-heartbeat")
        )
        self._threads.append(
            ManagedThread(target=self._run_server, name="controller-server")
        )
        for thread in self._threads:
            thread.start()
```

#### 4. Add Child Component Tracking

**Before (Controller):**
```python
class Controller:
    def __init__(self, autoscaler):
        self._autoscaler = autoscaler  # No explicit ownership
```

**After (Controller):**
```python
class Controller:
    def __init__(self, autoscaler):
        self._children: list[ThreadOwner] = []
        if autoscaler:
            self._children.append(autoscaler)
```

#### 5. Implement Cascading Shutdown

**Before (Controller):**
```python
def stop(self):
    self._stop = True
    if self._scheduling_loop_thread:
        self._scheduling_loop_thread.join()
    if self._heartbeat_loop_thread:
        self._heartbeat_loop_thread.join()
    # ... manual management of each thread
```

**After (Controller):**
```python
def stop(self, timeout: float = 5.0):
    """Stop all threads and child components with deadline-based budget."""
    deadline = time.monotonic() + timeout

    # 1. Signal all threads to stop
    for thread in self._threads:
        thread.stop()

    # 2. Stop all child components (cascade)
    for child in self._children:
        remaining = max(0, deadline - time.monotonic())
        child.stop(timeout=remaining)

    # 3. Join all threads
    for thread in self._threads:
        remaining = max(0, deadline - time.monotonic())
        thread.join(timeout=remaining)
```

#### 6. Fix Worker Task Execution Threads

**Before (Worker):**
```python
def submit_task(self, request):
    attempt = TaskAttempt(...)

    # ❌ Unmanaged daemon thread!
    thread = threading.Thread(target=attempt.run, daemon=True)
    attempt.thread = thread
    thread.start()
```

**After (Worker):**
```python
def submit_task(self, request):
    attempt = TaskAttempt(...)

    # ✅ Managed thread with stop event
    thread = ManagedThread(
        target=self._run_task_attempt,
        name=f"task-{task_id}",
        args=(attempt,)
    )
    attempt.thread = thread
    self._task_threads.append(thread)
    thread.start()

def _run_task_attempt(self, stop_event: threading.Event, attempt: TaskAttempt):
    """Run task attempt, checking stop event periodically."""
    attempt.run(stop_event=stop_event)

def stop(self, timeout: float = 5.0):
    deadline = time.monotonic() + timeout

    # Stop all lifecycle threads
    for thread in self._threads:
        thread.stop()

    # Stop all task threads
    for thread in self._task_threads:
        thread.stop()

    # Join all threads
    for thread in self._threads + self._task_threads:
        remaining = max(0, deadline - time.monotonic())
        thread.join(timeout=remaining)
```

#### 7. Fix ManagedVM Threads

**Before (ManagedVM):**
```python
class ManagedVM:
    def __init__(self):
        # ❌ Unmanaged daemon thread
        self._thread = threading.Thread(target=self._run, daemon=True)

    def stop(self):
        self._stop.set()  # ❌ No join - just signal!
```

**After (ManagedVM):**
```python
class ManagedVM:
    def __init__(self):
        # ✅ Managed thread
        self._lifecycle_thread: ManagedThread | None = None

    def start(self):
        self._lifecycle_thread = ManagedThread(
            target=self._run,
            name=f"vm-{self.info.vm_id}"
        )
        self._lifecycle_thread.start()

    def stop(self, timeout: float = 5.0):
        if self._lifecycle_thread:
            self._lifecycle_thread.stop()
            self._lifecycle_thread.join(timeout=timeout)
```

#### 8. Remove ThreadRegistry from Tests

**Before (conftest.py):**
```python
@pytest.fixture(autouse=True)
def _thread_registry():
    registry = ThreadRegistry()
    old = set_thread_registry(registry)
    yield registry
    registry.shutdown(timeout=5.0)
    set_thread_registry(old)
```

**After (conftest.py):**
```python
@pytest.fixture
def cluster_manager(config):
    """Provides ClusterManager with automatic cleanup."""
    manager = ClusterManager(config)
    yield manager
    # Hierarchical shutdown from the top!
    manager.stop(timeout=5.0)
```

**Benefits:**
- No global registry needed
- Tests just call `ClusterManager.stop()` which cascades
- Cleaner fixture dependency graph
- Each test gets isolated cluster lifecycle

---

## Migration Path

### Phase 1: Fix Unmanaged Threads (Minimal Change)

**Goal:** Get all threads into ThreadRegistry without architectural changes.

**Tasks:**
1. **Worker.submit_task()**: Use `ThreadContainer.spawn()` for task threads
   - Add `self._task_threads = ThreadContainer()` to Worker
   - Change `threading.Thread(...)` to `self._task_threads.spawn(...)`
   - Wire `TaskAttempt.should_stop` to stop event

2. **ManagedVM**: Use `ThreadContainer.spawn()` for lifecycle threads
   - Add `self._threads = ThreadContainer()` to ManagedVM
   - Change `threading.Thread(...)` to `self._threads.spawn(...)`
   - Check stop event in `_run()` loop

**Result:** All threads managed, but still using flat registry.

### Phase 2: Add Component Ownership Tracking (Infrastructure)

**Goal:** Add ownership infrastructure without changing behavior.

**Tasks:**
1. Add `ThreadOwner` protocol
2. Add `_threads: list[ManagedThread]` to Controller, Worker, ManagedVM
3. Add `_children: list[ThreadOwner]` to Controller, Autoscaler, LocalVmManager
4. Track threads in both old registry AND new ownership lists (dual mode)

**Result:** Infrastructure ready, but still using ThreadRegistry for cleanup.

### Phase 3: Convert to Hierarchical Shutdown

**Goal:** Replace ThreadRegistry with hierarchical `stop()` cascade.

**Tasks:**
1. Implement `stop(timeout)` with deadline-based cascading for all components
2. Update test fixtures to call `ClusterManager.stop()` instead of `registry.shutdown()`
3. Remove `ThreadRegistry` registration calls (keep `ManagedThread` class)
4. Convert daemon threads to non-daemon

**Result:** Full hierarchical shutdown, no ThreadRegistry needed.

### Phase 4: Cleanup and Documentation

**Goal:** Remove dead code and document new model.

**Tasks:**
1. Remove `ThreadRegistry` class (keep `ManagedThread`)
2. Remove `ThreadContainer` class (just use lists directly)
3. Remove `get_thread_registry()`, `set_thread_registry()` functions
4. Update documentation and docstrings
5. Add integration tests for shutdown behavior

**Result:** Clean, simple, hierarchical model.

---

## Benefits of Hierarchical Model

### 1. **Clear Ownership**
- Each component explicitly owns its threads
- Easy to reason about lifecycle
- No hidden global state

### 2. **Deterministic Shutdown Order**
- Parent stops before children
- Resources released in correct order
- No race conditions from unordered shutdown

### 3. **No Global Registry Needed**
- Simpler test fixtures
- No global state to manage
- Easier to understand

### 4. **Better Test Isolation**
- Each test gets fresh ClusterManager
- Stop propagates from top down
- No thread leaks between tests

### 5. **Explicit Resource Management**
- All threads are non-daemon (explicit cleanup)
- All threads have stop events (graceful shutdown)
- All components implement consistent interface

---

## Open Questions

1. **Timeout Budget Distribution**: How should parent components divide timeout budget among children?
   - Equal split?
   - Critical-first (e.g., stop tasks before workers)?
   - Greedy (use remaining time for each)?

2. **Stop vs Kill Semantics**: Should components have both `stop()` (graceful) and `kill()` (forceful)?
   - `stop(timeout)` tries graceful, falls back to kill?
   - Separate methods for different use cases?

3. **Error Handling in Shutdown**: What if a child component hangs during stop?
   - Propagate exception?
   - Log and continue?
   - Hard timeout with process kill?

4. **Bridge Threads**: Should we manage bridge threads (`_stop_event_to_server`) explicitly?
   - They're intentionally daemon and low-risk
   - But makes accounting incomplete
   - Could spawn via ManagedThread for completeness

---

## References

- **Issue:** [#2604 - iris: remaining unmanaged threads need cleanup](https://github.com/marin-community/marin/issues/2604)
- **Prior Work:** `iris/thread-management-migration` branch (Controller, Worker, WorkerPool already converted)
- **Code Locations:**
  - `lib/iris/src/iris/managed_thread.py` - Threading primitives
  - `lib/iris/src/iris/cluster/controller/controller.py` - Controller threads
  - `lib/iris/src/iris/cluster/worker/worker.py` - Worker threads (UNMANAGED task threads)
  - `lib/iris/src/iris/cluster/vm/managed_vm.py` - VM lifecycle threads (UNMANAGED)
  - `lib/iris/tests/conftest.py` - Test cleanup fixtures
