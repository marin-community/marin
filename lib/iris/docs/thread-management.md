# Thread Management Design

> Design document for [#2596](https://github.com/marin-community/marin/issues/2596).
> See [thread-usage.md](thread-usage.md) for the full audit and option analysis.

## Problem

Background threads in Iris (Controller, Worker, ActorServer, WorkerDispatcher, etc.) use daemon threads with inconsistent lifecycle management. This causes:

1. **Logging to closed streams** — daemon threads outlive pytest's stream lifecycle
2. **No cleanup guarantees** — daemon threads skip `finally` blocks on process exit
3. **Hidden bugs** — ActorServer's `shutdown()` is literally `pass`; daemon mode masks this
4. **Flaky tests** — race conditions during cleanup

## Decision

Introduce `ManagedThread` + `ThreadRegistry` (Options A+B from the audit). Convert all daemon threads to non-daemon with explicit event-based shutdown.

## API

```python
# iris/managed_thread.py

class ManagedThread:
    """Non-daemon thread with integrated shutdown event."""

    def __init__(
        self,
        target: Callable[[threading.Event], None],
        name: str | None = None,
        registry: ThreadRegistry | None = None,
    ): ...

    def start(self) -> None: ...
    def stop(self, timeout: float = 5.0) -> bool: ...

    @property
    def is_alive(self) -> bool: ...
    @property
    def shutdown_event(self) -> threading.Event: ...


class ThreadRegistry:
    """Tracks ManagedThreads for bulk shutdown."""

    def create_thread(
        self,
        target: Callable[[threading.Event], None],
        name: str | None = None,
    ) -> ManagedThread: ...

    def shutdown(self, timeout: float = 10.0) -> list[str]:
        """Stop all threads. Returns names of stuck threads."""
        ...

    def __enter__(self) -> ThreadRegistry: ...
    def __exit__(self, *exc) -> None: ...
```

### Thread target contract

Thread targets must accept a `threading.Event` as their first argument and check it in their loop:

```python
def my_loop(shutdown: threading.Event) -> None:
    while not shutdown.is_set():
        do_work()
        shutdown.wait(timeout=1.0)  # Replaces time.sleep()
```

### Test fixture usage

```python
@pytest.fixture
def thread_registry():
    registry = ThreadRegistry()
    yield registry
    stuck = registry.shutdown(timeout=5.0)
    assert not stuck, f"Threads did not stop: {stuck}"
```

## Migration Order

1. **Add `ManagedThread` + `ThreadRegistry`** — new module, no existing code changes
2. **Fix ActorServer** — store uvicorn server ref, implement real `shutdown()`
3. **Migrate Controller** — 3 threads + 1 executor
4. **Migrate Worker** — 2 threads + per-task threads
5. **Migrate WorkerDispatcher** — 1 thread per worker
6. **Migrate LocalPlatform** — 1 thread per container
7. **Remove all `daemon=True`** — final sweep

Each step is independently testable per the spiral planning approach in AGENTS.md.

## Daemon Thread Elimination

With event-based shutdown + explicit join, daemon mode is unnecessary. Non-daemon threads are strictly better:

- Cleanup runs reliably (`finally` blocks execute)
- Broken shutdown logic surfaces as test hangs instead of cryptic stream errors
- `pytest-timeout` provides a safety net for truly stuck threads

The only pre-requisite is that every thread loop checks its shutdown event, which `ManagedThread` enforces by API design.
