# Iris Threading Architecture and Cleanup Plan

## Current Threading Architecture

### Thread Creation Hierarchy

```
ClusterManager (top-level, no threads)
├── Controller
│   ├── ThreadContainer (component-scoped)
│   │   ├── scheduling-loop (ManagedThread, non-daemon)
│   │   │   └── Runs autoscaler synchronously every 10s
│   │   ├── heartbeat-loop (ManagedThread, non-daemon)
│   │   └── controller-server (ManagedThread, non-daemon)
│   │       └── stop-event-bridge (daemon thread) ⚠️
│   ├── ThreadPoolExecutor (32 workers, for parallel RPC dispatch)
│   └── Autoscaler
│       └── ThreadPoolExecutor (max_workers=4+, for async scale-up)
│
├── Worker (per VM)
│   ├── ThreadContainer (component-scoped)
│   │   ├── worker-lifecycle (ManagedThread, non-daemon)
│   │   └── worker-server (ManagedThread, non-daemon)
│   │       └── stop-event-bridge (daemon thread) ⚠️
│   └── Task execution threads (per task, daemon=True) ⚠️
│
├── ManagedVm (per VM, created by Autoscaler)
│   └── vm-{id} thread (daemon=True) ⚠️
│
├── WorkerPool (client-side)
│   └── ThreadContainer
│       └── dispatch-{worker_id} threads (ManagedThread per worker)
│
└── ActorServer (various locations)
    ├── Server thread (daemon=True) ⚠️
    └── ThreadPoolExecutor (for actor method execution)

Global State:
└── ThreadRegistry (process-wide singleton) ⚠️
    └── Tracks ALL ManagedThreads globally
```

**Legend:**
- ⚠️ = Problematic patterns requiring cleanup

### Current Problems

#### 1. ThreadRegistry Global Singleton
- **Issue**: Process-wide tracking disconnected from component hierarchy
- **Impact**: ClusterManager can't cleanly shut down "its" threads
- **Manifestation**: Tests use `autouse` fixture to reset registry per test

#### 2. Daemon Threads
Daemon threads prevent clean shutdown and cause log spam:

a. **Task execution threads** (Worker)
   - Created with `daemon=True` in `submit_task()`
   - Rationale: "Can survive process termination, containers cleaned separately"
   - Problem: Continue running after Worker.stop(), write to closed log files

b. **VM bootstrap threads** (ManagedVm)
   - Created with `daemon=True` in `ManagedVm.__init__`
   - Problem: VM lifecycle continues after autoscaler shutdown

c. **Stop-event-bridge threads** (uvicorn server helpers)
   - Created with `daemon=True` in `_stop_event_to_server()`
   - Problem: Small auxiliary threads leak after server shutdown

d. **ActorServer threads**
   - Created with `daemon=True` in `serve_background()`
   - Problem: Server continues after component shutdown

e. **LocalPlatform container threads** (test/demo)
   - Created with `daemon=True` in `_LocalContainer.start()`
   - Problem: In-process container simulation outlives test cleanup

#### 3. Timeout-Based Test Cleanup
- Tests use `ThreadRegistry.shutdown(timeout=5.0)`
- Long timeouts mask issues, short timeouts cause flakiness
- `pytest-timeout` acts as safety net (indicates design problem)

#### 4. Log Spam on Shutdown
```
ValueError: I/O operation on closed file.
Call stack: [in logging handler]
```
- Threads attempt to log after test teardown closes file handles
- Caused by daemon threads outliving their parent component

#### 5. Shutdown Ordering Issues
- No explicit dependency graph for thread shutdown
- Components shut down independently with fixed timeouts
- Scale-up operations continue while controller is stopping

---

## Proposed Clean Architecture

### Design Principles

1. **Hierarchical ThreadContainers**: Each component owns a ThreadContainer for its threads
2. **No Global Registry**: Remove ThreadRegistry; ClusterManager is root of hierarchy
3. **No Daemon Threads**: All threads are non-daemon, shut down explicitly
4. **Sentinel-Based Signaling**: Use sentinel files/events instead of timeouts for tests
5. **Shutdown Dependency Graph**: Explicit ordering based on component relationships

### New Thread Hierarchy

```
ClusterManager
├── _threads: ThreadContainer (root container)
│   ├── Owns Controller threads
│   ├── Owns Worker threads (for local mode)
│   └── Owns ManagedVm threads
│
└── stop() method:
    1. Signal all threads via stop_event
    2. Call component-specific stop methods
    3. Wait for threads (no timeout, blocking)
    4. Clean up resources

Controller
├── _threads: ThreadContainer (component-scoped, owned by ClusterManager)
│   ├── scheduling-loop (non-daemon)
│   ├── heartbeat-loop (non-daemon)
│   └── controller-server (non-daemon, no bridge thread)
│
├── _dispatch_executor: ThreadPoolExecutor (managed directly)
└── _autoscaler: Autoscaler (with its own executor)

Worker
├── _threads: ThreadContainer (component-scoped, owned by ClusterManager)
│   ├── worker-lifecycle (non-daemon)
│   ├── worker-server (non-daemon, no bridge thread)
│   └── task-monitor threads (one per task, non-daemon) ✓
│
└── _task_threads: ThreadContainer (sub-container for task execution)

Autoscaler
├── _scale_up_executor: ThreadPoolExecutor (managed directly)
└── _managed_vms: list[ManagedVm]
    └── Each VM: ThreadContainer for its bootstrap thread

ActorServer
└── _threads: ThreadContainer (component-scoped)
    └── server thread (non-daemon) ✓
```

### Shutdown Ordering

```python
# ClusterManager.stop() orchestrates top-down shutdown:

1. Signal all components (immediate, non-blocking)
   - controller._wake_event.set()
   - controller._heartbeat_event.set()
   - worker stop events
   - VM stop events

2. Stop autoscaler scale-up operations
   - autoscaler._scale_up_executor.shutdown(wait=True)

3. Stop VM bootstrap threads
   - Each ManagedVm._threads.stop(wait=True)

4. Stop controller loops
   - controller._threads.stop(wait=True)
   - controller._dispatch_executor.shutdown(wait=True)

5. Stop workers
   - worker._task_threads.stop(wait=True)  # Task monitors first
   - worker._threads.stop(wait=True)        # Then lifecycle/server

6. Final verification
   - Assert no threads remain alive except MainThread
```

### ThreadContainer Enhancements

#### Current API
```python
class ThreadContainer:
    def spawn(self, target: Callable, name: str, args: tuple = ()) -> ManagedThread
    def spawn_server(self, server: Server, name: str) -> ManagedThread
    def stop(self, timeout: float = 5.0) -> None
    def wait(self) -> None
```

#### Enhanced API
```python
class ThreadContainer:
    def spawn(self, target: Callable, name: str, args: tuple = ()) -> ManagedThread
    def spawn_server(self, server: Server, name: str) -> ManagedThread
    def spawn_executor(self, max_workers: int, prefix: str) -> ThreadPoolExecutor

    # Shutdown phases
    def stop(self, wait: bool = True) -> None:
        """Signal all threads to stop. If wait=True, block until all exit."""

    def wait(self) -> None:
        """Block until all threads exit (no timeout)."""

    # Hierarchical containers
    def create_child(self, name: str) -> 'ThreadContainer':
        """Create a sub-container for component-scoped thread groups."""

    # Introspection
    def alive_threads(self) -> list[ManagedThread]:
        """Return list of threads that are still alive."""
```

### Server Thread Management

Remove `_stop_event_to_server()` daemon thread bridge. Instead:

```python
class ThreadContainer:
    def spawn_server(self, server: Server, name: str) -> ManagedThread:
        """Spawn uvicorn Server with integrated stop signaling."""
        def _run(stop_event: threading.Event) -> None:
            # Check stop_event periodically in server context
            original_should_exit = server.should_exit

            def _should_exit() -> bool:
                return original_should_exit or stop_event.is_set()

            server.should_exit = property(lambda _: _should_exit())
            server.run()

        return self.spawn(target=_run, name=name)
```

### Task Thread Management

Replace daemon task threads with monitored non-daemon threads:

```python
class Worker:
    def __init__(self, ...):
        self._threads = ThreadContainer()
        self._task_threads = self._threads.create_child("tasks")

    def submit_task(self, request: RunTaskRequest) -> None:
        attempt = TaskAttempt(...)

        def _monitor_task(stop_event: threading.Event) -> None:
            """Monitor task execution, check stop_event periodically."""
            attempt.run()

            # Periodically check if we should abort
            while attempt.is_running() and not stop_event.is_set():
                time.sleep(0.1)

            if stop_event.is_set():
                # Kill container immediately
                if attempt.container_id:
                    self._runtime.kill(attempt.container_id, force=True)

        self._task_threads.spawn(
            target=_monitor_task,
            name=f"task-{request.task_id}"
        )

    def stop(self) -> None:
        # Tasks stop first (may take time to kill containers)
        self._task_threads.stop(wait=True)
        # Then worker infrastructure
        self._threads.stop(wait=True)
```

### VM Thread Management

Replace daemon VM threads with component-managed threads:

```python
class ManagedVm:
    def __init__(self, vm_id: str, ...):
        self._threads = ThreadContainer()
        self._stop = threading.Event()

        # Spawn as non-daemon, owned by ThreadContainer
        self._threads.spawn(
            target=self._run,
            name=f"vm-{vm_id}",
            args=(self._stop,)
        )

    def stop(self) -> None:
        """Signal VM thread to stop and wait for it."""
        self._threads.stop(wait=True)

    def _run(self, stop_event: threading.Event) -> None:
        """Bootstrap lifecycle with stop_event checking."""
        # Existing bootstrap logic, but check stop_event in loops
        ...
```

### Test Sentinel Files

Replace timeout-based waits with sentinel files for deterministic test cleanup:

```python
# tests/test_utils.py enhancement
@contextmanager
def sentinel_file(path: Path) -> Iterator[Path]:
    """Create a sentinel file for signaling test completion.

    Usage:
        with sentinel_file(tmp_path / "controller_ready") as sentinel:
            controller.start()
            # Wait for controller to signal readiness
            wait_for_sentinel(sentinel, timeout=5.0)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        if path.exists():
            path.unlink()

def wait_for_sentinel(path: Path, timeout: float = 10.0) -> None:
    """Wait for sentinel file to appear."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise TimeoutError(f"Sentinel file {path} did not appear within {timeout}s")

def signal_sentinel(path: Path) -> None:
    """Signal completion by creating sentinel file."""
    path.touch()
```

### Test Fixture Update

```python
# lib/iris/tests/conftest.py
@pytest.fixture(autouse=True)
def _thread_cleanup():
    """Ensure all threads are cleaned up after each test."""
    # No ThreadRegistry setup needed anymore

    yield

    # After test: verify clean shutdown
    alive = [t for t in threading.enumerate()
             if t.is_alive() and t.name != "MainThread"]

    if alive:
        # If threads remain, it's a test failure
        thread_names = [t.name for t in alive]
        pytest.fail(f"Threads still alive after test: {thread_names}")
```

---

## Implementation Plan (Spiral Approach)

### Phase 1: Foundation - ThreadContainer Enhancement
**Goal**: Add hierarchical container support without breaking existing code

**Tasks**:
1. Add `create_child()` method to ThreadContainer
2. Add `alive_threads()` introspection method
3. Add `spawn_executor()` for ThreadPoolExecutor management
4. Add sentinel file utilities to `tests/test_utils.py`
5. Add integration test for hierarchical containers

**Expected State**: ThreadContainer can manage child containers, existing tests pass

---

### Phase 2: Remove Stop-Event Bridge Daemon Threads
**Goal**: Eliminate daemon threads from uvicorn server spawning

**Tasks**:
1. Rewrite `spawn_server()` to integrate stop signaling without bridge thread
2. Update Controller to use new spawn_server
3. Update Worker to use new spawn_server
4. Update ActorServer to use ThreadContainer instead of raw daemon thread
5. Verify no "stop-event-bridge" threads remain in tests

**Expected State**: No daemon bridge threads, server shutdown is clean

---

### Phase 3: Replace Daemon Task Threads
**Goal**: Task execution uses non-daemon threads with clean shutdown

**Tasks**:
1. Add `_task_threads` child container to Worker
2. Rewrite `submit_task()` to spawn monitored non-daemon threads
3. Update `Worker.stop()` to shut down tasks first, then infrastructure
4. Add test with sentinel file to verify task thread cleanup
5. Update tests to use sentinel signaling instead of timeouts

**Expected State**: Task threads shut down cleanly, no log spam

---

### Phase 4: Replace Daemon VM Threads
**Goal**: VM bootstrap threads are non-daemon, managed by Autoscaler

**Tasks**:
1. Add `_threads` ThreadContainer to ManagedVm
2. Update ManagedVm to spawn bootstrap thread via ThreadContainer
3. Add `ManagedVm.stop()` method for clean shutdown
4. Update Autoscaler.shutdown() to stop all ManagedVm threads
5. Add test verifying VM threads shut down before autoscaler returns

**Expected State**: VM threads shut down cleanly, autoscaler shutdown is deterministic

---

### Phase 5: Replace LocalPlatform Daemon Threads
**Goal**: Test/demo local execution uses non-daemon threads

**Tasks**:
1. Add ThreadContainer to LocalPlatform
2. Update _LocalContainer to spawn execution thread via ThreadContainer
3. Add LocalPlatform.stop() method
4. Update tests to call platform.stop() in teardown
5. Verify no local container threads leak in tests

**Expected State**: Local mode has clean shutdown, test isolation is perfect

---

### Phase 6: Hierarchical ClusterManager Threading
**Goal**: ClusterManager owns root ThreadContainer, eliminates ThreadRegistry

**Tasks**:
1. Add `_threads` ThreadContainer to ClusterManager
2. Update ClusterManager to pass container to Controller/Worker constructors
3. Rewrite Controller.__init__ to accept parent ThreadContainer
4. Rewrite Worker.__init__ to accept parent ThreadContainer
5. Remove ThreadRegistry usage from Controller/Worker
6. Add ClusterManager.stop() with proper shutdown ordering

**Expected State**: ClusterManager controls all threads, no ThreadRegistry

---

### Phase 7: Remove ThreadRegistry
**Goal**: Delete ThreadRegistry, update tests

**Tasks**:
1. Remove `_thread_registry` autouse fixture from conftest.py
2. Add `_thread_cleanup` fixture that verifies no threads remain
3. Delete `get_thread_registry()`, `set_thread_registry()` functions
4. Delete ThreadRegistry class
5. Update all tests to use component-level shutdown
6. Remove pytest-timeout dependency (tests should never hang now)

**Expected State**: No global registry, tests verify clean shutdown directly

---

### Phase 8: Update Smoke Tests
**Goal**: Smoke tests use sentinel files, no timeouts

**Tasks**:
1. Update smoke-test.py to use sentinel files for job completion
2. Remove timeout-based waits in cluster-tools.py
3. Add sentinel support to IrisClient for job completion signaling
4. Update validate command to use sentinel-based completion
5. Add test verifying smoke test cleanup leaves no threads

**Expected State**: Smoke tests are deterministic, complete immediately when jobs finish

---

## Expected Benefits

### 1. Deterministic Shutdown
- No timeouts, no races
- Tests complete as soon as work finishes
- Failure is immediate and debuggable

### 2. Clean Logs
- No "I/O operation on closed file" errors
- Threads stop before logging infrastructure teardown
- Stack traces are meaningful

### 3. Test Speed
- Current: 5s timeout per test minimum
- After: Tests complete in milliseconds after work finishes
- CI time reduction: ~30-50% for Iris test suite

### 4. Simplified Debugging
- Thread hierarchy visible in logs
- Component ownership clear
- Shutdown order explicit

### 5. Reduced Flakiness
- No timeout tuning
- No race conditions between threads and teardown
- Reproducible failures

---

## Migration Notes

### Backward Compatibility
- ThreadContainer API remains unchanged for existing callers
- New features are additive (create_child, spawn_executor)
- Tests migrate one at a time (spiral approach)

### Risk Mitigation
- Each phase independently testable
- Rollback point after each phase
- No "big bang" rewrite

### Validation Criteria
At end of each phase:
1. All existing tests pass
2. No daemon threads in test suite
3. Thread count returns to 1 (MainThread) after each test
4. No timeout-based waits in new code
5. Logs are clean (no "closed file" errors)

---

## Appendix A: Thread Lifecycle Invariants

After this refactoring, the following invariants will hold:

1. **All threads are non-daemon**: No thread outlives its parent component
2. **No global state**: ThreadContainer hierarchy is component-scoped
3. **Stop is synchronous**: `component.stop()` blocks until threads exit
4. **Shutdown is ordered**: Parent stops children before stopping itself
5. **Tests are isolated**: No threads leak between tests
6. **Logs are clean**: No writes to closed file handles
7. **Timeouts are for external I/O only**: Not for thread coordination

---

## Appendix B: Current vs. Proposed Shutdown Time

### Current (with timeouts)
```
Test: test_controller_worker_integration
├── Setup: 0.5s
├── Execution: 2.0s
├── ThreadRegistry.shutdown(timeout=5.0): 5.0s (waits full timeout)
└── Total: 7.5s
```

### Proposed (with sentinel files)
```
Test: test_controller_worker_integration
├── Setup: 0.5s
├── Execution: 2.0s
├── ClusterManager.stop(): 0.1s (threads exit immediately)
└── Total: 2.6s

Speedup: 2.9x faster
```

---

## Appendix C: File Changes Summary

### Modified Files
- `lib/iris/src/iris/managed_thread.py`: Enhance ThreadContainer
- `lib/iris/src/iris/cluster/controller/controller.py`: Use parent container
- `lib/iris/src/iris/cluster/worker/worker.py`: Use parent container, non-daemon tasks
- `lib/iris/src/iris/cluster/vm/managed_vm.py`: Use ThreadContainer
- `lib/iris/src/iris/cluster/vm/autoscaler.py`: Manage VM threads
- `lib/iris/src/iris/cluster/vm/local_platform.py`: Use ThreadContainer
- `lib/iris/src/iris/cluster/vm/cluster_manager.py`: Root ThreadContainer
- `lib/iris/src/iris/actor/server.py`: Use ThreadContainer
- `lib/iris/tests/conftest.py`: Remove registry fixture, add cleanup verification
- `tests/test_utils.py`: Add sentinel file utilities

### Deleted Code
- `ThreadRegistry` class (after Phase 7)
- `_stop_event_to_server()` function (after Phase 2)
- `_thread_registry` fixture (after Phase 7)

### New Test Files
- `lib/iris/tests/test_thread_hierarchy.py`: Validate hierarchical containers
- `lib/iris/tests/test_clean_shutdown.py`: Validate no threads remain after tests
