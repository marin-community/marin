# Thread Safety and Test Synchronization

## Overview

Iris uses structured concurrency via `ManagedThread` and `ThreadContainer` to ensure all background threads can be cleanly shut down. This document describes best practices for writing thread-safe code and reliable tests.

## Thread Lifecycle Management

### ManagedThread

All background threads should be created via `ManagedThread`, which:
- Integrates a `stop_event` that signals when the thread should exit
- Is non-daemon by default to prevent abrupt process termination
- Logs exceptions that escape the thread target
- Can be stopped with optional timeout

### ThreadContainer

Components should use `ThreadContainer` to manage their thread lifecycle:

```python
class Worker:
    def __init__(self):
        self._threads = ThreadContainer("worker")

    def start(self):
        self._threads.spawn(self._background_loop, name="worker-loop")

    def stop(self):
        self._threads.stop(timeout=5.0)  # Signal all threads and wait

    def _background_loop(self, stop_event: threading.Event):
        while not stop_event.is_set():
            do_work()
            stop_event.wait(timeout=1.0)  # Sleep but check stop_event
```

### Key Rules

1. **Accept stop_event as first parameter**: All thread targets must accept `threading.Event` as their first argument.

2. **Check stop_event regularly**: Loops should check `stop_event.is_set()` at least every 1 second.

3. **Exit promptly when signaled**: Threads should exit within ~1 second of stop_event being set.

4. **Use stop_event.wait() instead of time.sleep()**: This allows the thread to wake up immediately when signaled.

## Common Patterns

### Background Loop

```python
def worker_loop(stop_event: threading.Event, config: Config) -> None:
    """Worker that processes items until stopped."""
    while not stop_event.is_set():
        try:
            item = queue.get(timeout=1.0)
            process(item)
        except queue.Empty:
            continue  # Check stop_event and retry
```

### Polling Loop

```python
def poll_status(stop_event: threading.Event, interval: float) -> None:
    """Poll status every interval seconds."""
    while not stop_event.is_set():
        check_and_update_status()
        stop_event.wait(timeout=interval)  # Sleep but check stop_event
```

### Server Thread

```python
def start(self):
    server = uvicorn.Server(config)
    self._threads.spawn_server(server, name="api-server")
```

The `spawn_server` helper automatically bridges the stop_event to `server.should_exit`.

## Anti-Patterns to Avoid

### ❌ Infinite loop without stop_event check

```python
def bad_worker(stop_event: threading.Event):
    while True:  # Never checks stop_event!
        time.sleep(10)
```

### ❌ Using time.sleep() instead of stop_event.wait()

```python
def bad_worker(stop_event: threading.Event):
    while not stop_event.is_set():
        do_work()
        time.sleep(10)  # Will wait full 10s even if stop requested
```

### ❌ Ignoring the stop_event parameter

```python
def bad_worker(stop_event: threading.Event):
    # stop_event is never checked!
    for item in infinite_stream():
        process(item)
```

## Test Synchronization

Tests should use **sentinel files** and **condition waiting** instead of `time.sleep()` for synchronization.

### Why Not time.sleep()?

- **Flaky**: Race conditions if timing assumptions are violated
- **Slow**: Must wait full duration even if condition is satisfied early
- **Non-deterministic**: Doesn't guarantee the condition is true

### Sentinel Files

Use sentinel files to signal when asynchronous operations complete:

```python
def test_worker_processes_task(tmp_path):
    # Create sentinel file context
    with sentinel_file(tmp_path / "task_done") as sentinel:
        # Start worker that will signal completion
        worker.start()
        worker.submit_task(task_id="test-task", on_complete=lambda: signal_sentinel(sentinel))

        # Wait for sentinel (fast and reliable)
        wait_for_sentinel(sentinel, timeout=5.0)

        # Now safe to check results
        assert worker.get_task_status("test-task") == "complete"
```

### Condition Waiting

For complex conditions, use `wait_for_condition()`:

```python
def test_controller_accepts_workers(controller):
    controller.start()

    # Wait for controller to be ready
    wait_for_condition(lambda: controller.is_ready(), timeout=5.0)

    # Register worker
    worker_id = controller.register_worker(address="localhost:8000")

    # Wait for worker to appear in controller state
    wait_for_condition(
        lambda: controller.get_worker(worker_id) is not None,
        timeout=5.0
    )
```

### Comparison

```python
# ❌ BAD - flaky and slow
worker.start()
time.sleep(0.5)  # Hope worker is ready
assert worker.is_ready()

# ✅ GOOD - reliable and fast
worker.start()
wait_for_condition(lambda: worker.is_ready(), timeout=5.0)
```

## Thread Leak Detection

The test suite includes automatic thread leak detection via the `_thread_cleanup` fixture. If your test fails with:

```
Warning: Threads leaked from test: ['worker-loop', 'heartbeat-sender']
```

This means you didn't clean up threads properly. Fix by:

1. Ensuring all components expose a `stop()` method
2. Calling `stop()` in test cleanup (or use a pytest fixture with teardown)
3. Using `ThreadContainer` which handles cleanup automatically

Example fix:

```python
def test_worker_lifecycle():
    worker = Worker()
    worker.start()

    try:
        # Test logic
        pass
    finally:
        worker.stop()  # Ensure cleanup happens
```

## Debugging Hanging Tests

If a test hangs or times out:

1. **Check stop_event**: Ensure all loops check `stop_event.is_set()`
2. **Check timeouts**: Use `stop_event.wait(timeout=...)` not `time.sleep()`
3. **Check ThreadContainer.stop()**: All components should call this in their stop() method
4. **Run with timeout**: Pytest timeout will dump stack traces showing where threads are stuck

```bash
# Run with verbose output to see which test is hanging
uv run pytest -v --timeout=30 lib/iris/tests/

# Get stack trace of hanging test
# The pytest timeout will show where threads are blocked
```

## References

- `lib/iris/src/iris/managed_thread.py` - Thread lifecycle management
- `lib/iris/tests/test_utils.py` - Test synchronization utilities
- `lib/iris/tests/conftest.py` - Thread leak detection fixture
- `lib/iris/tests/test_thread_hierarchy.py` - Examples of proper thread management
