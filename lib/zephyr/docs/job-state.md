# Zephyr Job State Management

**Problem**: When ZephyrContext creates actors (coordinator + workers), it uses static names like `"zephyr-controller"` and `"zephyr-worker"`. If the context is reused or multiple tests run in sequence against the same Iris cluster, job name collisions occur:

```
connectrpc.errors.ConnectError: Job zephyr-controller-0 already exists
```

## Root Cause

1. `ZephyrContext._get_or_create_coordinator()` creates actors with fixed names:
   ```python
   self._coordinator = self.client.create_actor(
       ZephyrCoordinator,
       name="zephyr-controller",  # ← Static name!
       resources=ResourceConfig(),
   )
   ```

2. After `execute()` completes, ZephyrContext resets `self._coordinator = None` but the Iris jobs are still registered in the controller.

3. Next `execute()` call tries to create actors with the same names → collision.

## Solution

### 1. Use Unique Job Names

Generate unique names per ZephyrContext instance:

```python
@dataclass
class ZephyrContext:
    client: Client
    num_workers: int
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    max_parallelism: int = 1024

    # Add unique instance ID
    _instance_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def _get_or_create_coordinator(self) -> ActorHandle:
        if self._coordinator is None:
            # Use instance_id to make names unique
            self._coordinator = self.client.create_actor(
                ZephyrCoordinator,
                name=f"zephyr-controller-{self._instance_id}",
                resources=ResourceConfig(),
            )
            group = self.client.create_actor_group(
                ZephyrWorker,
                name=f"zephyr-worker-{self._instance_id}",
                count=self.num_workers,
                resources=self.resources,
            )
            self._workers = group.wait_ready()
        return self._coordinator
```

### 2. Ensure Context Manager Usage

ZephyrContext is already a context manager, but we should enforce its use:

**Current (already implemented)**:
```python
def __enter__(self) -> ZephyrContext:
    return self

def __exit__(self, *exc) -> None:
    self.shutdown()
```

**Recommended usage**:
```python
# Good - context manager ensures cleanup
with ZephyrContext(client, num_workers=2) as ctx:
    results = ctx.execute(dataset)

# Bad - manual lifecycle management
ctx = ZephyrContext(client, num_workers=2)
results = ctx.execute(dataset)  # Actors may leak!
```

### 3. Proper Shutdown

The `shutdown()` method should:
1. Signal workers to stop (`coordinator.signal_done()`)
2. Wait for worker futures to complete
3. **Terminate the coordinator and worker jobs** (currently missing!)

**Current implementation**:
```python
def shutdown(self) -> None:
    if self._coordinator is not None:
        self._coordinator.signal_done.remote()
        for w in self._workers:
            with suppress(Exception):
                w.shutdown.remote()
        # Wait for futures
        for f in self._worker_futures:
            with suppress(Exception):
                f.result(timeout=5.0)
        self._coordinator = None
        self._workers = []
        self._worker_futures = []
```

**Problem**: This signals the actors to stop but doesn't terminate the underlying Iris jobs.

**Fix needed**:
```python
def shutdown(self) -> None:
    if self._coordinator is not None:
        # Signal actors to stop gracefully
        self._coordinator.signal_done.remote()
        for w in self._workers:
            with suppress(Exception):
                w.shutdown.remote()

        # Wait for futures
        for f in self._worker_futures:
            with suppress(Exception):
                f.result(timeout=5.0)

        # Terminate the underlying jobs
        # (Need to track job handles when creating actors)
        for job in self._coordinator_jobs + self._worker_jobs:
            with suppress(Exception):
                job.terminate()

        self._coordinator = None
        self._workers = []
        self._worker_futures = []
```

## Implementation Plan

1. **Add unique instance ID** to ZephyrContext
2. **Track job handles** when creating actors via `create_actor()` and `create_actor_group()`
3. **Terminate jobs** in `shutdown()` method
4. **Update test fixtures** to use context manager pattern

## Test Fixture Pattern

**Before**:
```python
@pytest.fixture
def zephyr_ctx(fray_client):
    ctx = ZephyrContext(client=fray_client, num_workers=2)
    yield ctx
    ctx.shutdown()  # May not be called if test fails
```

**After**:
```python
@pytest.fixture
def zephyr_ctx(fray_client):
    with ZephyrContext(client=fray_client, num_workers=2) as ctx:
        yield ctx
    # Cleanup guaranteed even on test failure
```

## Status

✅ **Implemented** (2026-01-30):
- Unique job names using `_instance_id`
- Job termination in `shutdown()` via `ActorGroup.shutdown()`
- Context manager pattern enforced in test fixtures

**Test results**: 6/9 iris tests passing (job collision errors eliminated)

**Remaining work**:
- Fix `test_shared_data` to use context manager instead of manual lifecycle
- Investigate empty results in 3 tests (likely related to manual context creation)

## References

- Job collision error: `lib/zephyr/tests/test_execution.py` full test run
- Context manager implementation: `lib/zephyr/src/zephyr/execution.py:512-516`
- Shutdown logic: `lib/zephyr/src/zephyr/execution.py:498-510`
- Implementation: commit with iris ContextVar fix + job state management
