# Design: Refactor LocalClusterClient to use ClusterManager

**Issue:** [#2547](https://github.com/marin-community/marin/issues/2547)

## Problem Statement

Three separate locations manually wire up Controller + Worker for local/test execution:

1. **`LocalClusterClient.create()`** (`lib/iris/src/iris/cluster/client/local_client.py:324-397`)
   Creates a `Controller` and a single `Worker` with local providers, managing temp dirs, ports, and lifecycle manually.

2. **`LocalController.start()`** (`lib/iris/src/iris/cluster/vm/controller.py:959-999`)
   Creates a `Controller` with an `Autoscaler` backed by `LocalVmManager`s, which in turn create `Worker` instances with the same local providers.

3. **`E2ECluster.__enter__()`** (`lib/iris/tests/cluster/test_e2e.py:81-153`)
   A third copy: creates `Controller` + N `Worker` instances with the same local providers and manual temp dir management.

All three share the same pattern:
- Create temp dir with bundles/, cache/, fake_bundle/ subdirectories
- Find free ports
- Instantiate `Controller` with `ControllerConfig`
- Instantiate `Worker`(s) with `_LocalBundleProvider`, `_LocalImageProvider`, `_LocalContainerRuntime`, `LocalEnvironmentProvider`
- Wait for worker registration

The canonical path already exists: `ClusterManager` + `LocalController` + `_create_local_autoscaler` + `LocalVmManager`. `LocalClusterClient` and `E2ECluster` bypass this entirely.

## Analysis: Duplicated vs Unique

### Shared across all three (duplicated)
- Temp directory creation (bundles/, cache/, fake_bundle/ + pyproject.toml)
- Free port allocation
- `Controller` + `ControllerConfig` instantiation
- Worker creation with local providers
- Worker registration polling

### Unique to LocalClusterClient
- `max_workers` parameter (currently creates exactly 1 worker, parameter unused beyond docs)
- `port_range` parameter (passed to `WorkerConfig`, not `PortAllocator`)
- Returns a `LocalClusterClient` wrapping a `RemoteClusterClient` for RPC access
- `_wait_for_worker_registration` uses a temporary RPC client

### Unique to E2ECluster
- `use_docker=True` path with real `DockerRuntime`/`BundleCache`/`ImageCache`
- `num_workers` parameter creating N workers
- Custom `LocalEnvironmentProvider(cpu=4, memory_gb=8)` for resource scheduling tests
- `submit()`, `wait()`, `status()` convenience methods wrapping raw RPC
- `get_client()` returns `IrisClient.remote()` with workspace bundle

### Unique to LocalController
- Autoscaler integration (dynamic worker creation/destruction)
- Multi-scale-group support (TPU topologies, different accelerator types)
- `PortAllocator` shared across workers

## Proposed Changes

### 1. Refactor `LocalClusterClient.create()` to use `ClusterManager`

Replace manual Controller+Worker wiring with `ClusterManager` + `make_local_config()`.

**Before** (simplified):
```python
@classmethod
def create(cls, max_workers=4, port_range=(50000, 60000)) -> Self:
    temp_dir = tempfile.TemporaryDirectory(...)
    # ... create bundle_path, cache_path, fake_bundle ...
    controller = Controller(ControllerConfig(...))
    controller.start()
    worker = Worker(WorkerConfig(...), bundle_provider=..., ...)
    worker.start()
    cls._wait_for_worker_registration(address)
    remote_client = RemoteClusterClient(...)
    return cls(temp_dir, controller, worker, remote_client)
```

**After:**
```python
@classmethod
def create(cls, max_workers=4, port_range=(50000, 60000)) -> Self:
    config = _make_local_cluster_config(max_workers)
    config = make_local_config(config)
    manager = ClusterManager(config)
    address = manager.start()
    cls._wait_for_worker_registration(address)
    remote_client = RemoteClusterClient(
        controller_address=address, timeout_ms=30000
    )
    return cls(manager, remote_client)


def _make_local_cluster_config(max_workers: int) -> config_pb2.IrisClusterConfig:
    """Build a minimal IrisClusterConfig for local execution."""
    config = config_pb2.IrisClusterConfig()
    sg = config_pb2.ScaleGroupConfig(
        name="local-cpu",
        min_slices=1,
        max_slices=max_workers,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
    )
    config.scale_groups["local-cpu"].CopyFrom(sg)
    return config
```

**`__init__` and `shutdown()` changes:**
```python
def __init__(self, manager: ClusterManager, remote_client: RemoteClusterClient):
    self._manager = manager
    self._remote_client = remote_client

def shutdown(self, wait: bool = True) -> None:
    del wait
    self._remote_client.shutdown()
    self._manager.stop()
```

This removes:
- `_find_free_port()` (use `local_platform.find_free_port` via LocalController)
- Manual Controller/Worker creation
- `self._temp_dir`, `self._controller`, `self._worker` fields

**Keep `_wait_for_worker_registration`**: The autoscaler creates workers asynchronously, so we still need to poll until at least one worker registers.

### 2. Refactor `E2ECluster` to use `ClusterManager`

The default (non-Docker) path delegates to `ClusterManager`:

```python
class E2ECluster:
    def __init__(self, num_workers: int = 1, use_docker: bool = False):
        self._num_workers = num_workers
        self._use_docker = use_docker
        self._manager: ClusterManager | None = None
        # ... docker-specific fields remain ...

    def __enter__(self):
        if self._use_docker:
            return self._enter_docker()

        config = _make_e2e_config(self._num_workers)
        config = make_local_config(config)
        self._manager = ClusterManager(config)
        address = self._manager.start()
        self._wait_for_workers(address, timeout=10.0)
        # ... create controller_client for raw RPC methods ...
        return self

    def __exit__(self, *args):
        if self._manager:
            self._manager.stop()
        # ... docker cleanup ...
```

The `use_docker=True` path remains as-is since it uses completely different providers.

The E2E tests that rely on `LocalEnvironmentProvider(cpu=4, memory_gb=8)` need consideration: currently `LocalVmManager` hardcodes `cpu=1000, memory_gb=1000`. We may need to make this configurable via `ScaleGroupConfig` or accept that tests work with the 1000-cpu default (the tests only check scheduling relative to available resources, so 4 CPUs works because the "big job" requests 100).

**Resolution:** The `_create_local_autoscaler` in `local_platform.py` already creates `LocalEnvironmentProvider(cpu=1000, memory_gb=1000)`. The E2E resource tests submit a job with `cpu=100` expecting it to be unschedulable on a 4-CPU worker. With 1000 CPUs, that job *would* be schedulable. We have two options:

- **Option A:** Make cpu/memory configurable on `LocalVmManager` via a field on `ScaleGroupConfig` or a custom metadata field.
- **Option B:** Update the E2E tests to use `cpu=10000` for the "too big" job.

**Recommendation: Option B.** It's simpler and the test intent is clear regardless of the exact number. The test should use a resource request that exceeds any reasonable local worker, not rely on a specific worker size.

### 3. Local providers stay in `local_client.py`

The local providers (`_LocalContainerRuntime`, `_LocalBundleProvider`, `_LocalImageProvider`, `LocalEnvironmentProvider`, `_LocalContainer`, `_StreamingCapture`) remain in `local_client.py`. `local_platform.py` already imports them from there. No circular dependency issues.

### 4. Update `IrisClient.local()` and `LocalClientConfig`

`IrisClient.local()` calls `LocalClusterClient.create()`, so it gets the refactor for free. The `port_range` parameter on `LocalClientConfig` will need to be threaded through to the config (or dropped, since `PortAllocator` in `local_platform.py` uses its own range `(30000, 40000)`).

**Decision:** Drop `port_range` from `LocalClientConfig` and `LocalClusterClient.create()`. The `PortAllocator` in `_create_local_autoscaler` already handles port allocation with a hardcoded range. If we need it configurable later, it can be added to the proto config.

## fakes.py Verdict

**Keep fakes.py.** It serves a fundamentally different purpose from `local_platform.py` and `chaos.py`:

| Aspect | fakes.py | local_platform.py | chaos.py |
|--------|----------|-------------------|----------|
| Purpose | Deterministic, tick-based autoscaler unit tests | Real in-process cluster for integration tests | Runtime failure injection |
| Workers | No real workers, just `FakeVm` state machines | Real `Worker` instances with local providers | N/A (injected into real code) |
| Time | Explicit `tick(ts)` calls, fully deterministic | Real-time, async worker registration | Real-time probabilistic |
| Tests | Autoscaler scaling logic, quota handling | E2E job lifecycle, resource scheduling | Distributed failure scenarios |

There is zero overlap. `FakeVm`/`FakeVmGroup` test the autoscaler's state machine logic in isolation. `LocalVmManager`/`LocalVmGroup` run real workers for integration tests. `chaos.py` injects runtime failures into production code paths.

## Migration Plan

### Step 1: Update E2E test resource values
- Change `test_small_job_skips_oversized_job` to use `cpu=10000` instead of `cpu=100`
- Change `test_scheduling_timeout` to use `cpu=10000` instead of `cpu=100`
- Verify tests pass with existing code

### Step 2: Refactor `LocalClusterClient`
- Add `_make_local_cluster_config()` helper
- Rewrite `create()` to use `ClusterManager` + `make_local_config()`
- Update `__init__` to store `ClusterManager` instead of separate components
- Update `shutdown()` to call `manager.stop()`
- Remove `_find_free_port()` (dead code after refactor)
- Drop `port_range` from `create()` signature and `LocalClientConfig`
- Run: `uv run pytest tests/cluster/client/test_local_client.py -xvs`
- Run: `uv run pytest tests/cluster/test_client.py -xvs`
- Run: `uv run pytest tests/client/test_worker_pool.py -xvs`

### Step 3: Refactor `E2ECluster`
- Rewrite `__enter__`/`__exit__` non-Docker path to use `ClusterManager`
- Keep `use_docker=True` path unchanged
- Remove duplicated `find_free_port()` from test_e2e.py (use `local_platform.find_free_port` if needed for Docker path)
- Run: `uv run pytest tests/cluster/test_e2e.py -xvs`

### Step 4: Clean up imports
- Remove unused imports from `local_client.py` and `test_e2e.py`
- Run pre-commit: `./infra/pre-commit.py --all-files`

## Risk Assessment

**Low risk:**
- The canonical path (`ClusterManager` -> `LocalController` -> `_create_local_autoscaler` -> `LocalVmManager`) is already tested and used by `ClusterManager.connect()`.
- All changes are in test infrastructure and the local-only client, not production paths.
- Local providers are untouched.

**Medium risk:**
- **Worker creation is now async via autoscaler.** `LocalClusterClient.create()` currently creates a worker synchronously and polls for registration. After refactoring, the autoscaler creates workers on its evaluation interval. `_wait_for_worker_registration` handles this, but the wait time may increase slightly. Mitigation: `make_local_config` already sets `evaluation_interval_seconds=0.5`, and `_create_local_autoscaler` sets `scale_up_cooldown_ms=1000`. Workers should appear within ~1.5s.
- **E2E tests with `LocalEnvironmentProvider(cpu=4)`** will now use `cpu=1000`. Tests that depend on the exact CPU count need updating (Step 1 handles this).

**No risk:**
- `fakes.py` is completely independent and unchanged.
- `chaos.py` is completely independent and unchanged.
- Production GCP/Manual controller paths are untouched.

## Review

**Reviewer:** Senior engineer, 2026-01-29

### Issues Found

#### 1. `_create_local_autoscaler` ignores `config.autoscaler` (Medium — will cause flaky tests)

`make_local_config()` sets `config.autoscaler.evaluation_interval_seconds = 0.5`, but `_create_local_autoscaler()` at line 342 of `local_platform.py` hardcodes:

```python
config=config_pb2.AutoscalerConfig(evaluation_interval_seconds=2.0),
```

This means the 0.5s interval set by `make_local_config` is silently discarded. After the refactor, worker creation will take up to ~3s (2s eval interval + 1s scale-up cooldown) instead of the ~1.5s stated in the Risk Assessment.

**Fix:** `_create_local_autoscaler` should use `config.autoscaler` instead of constructing a fresh `AutoscalerConfig`. Change line 342 to:

```python
config=config.autoscaler,
```

This also respects any future autoscaler config fields callers might set.

#### 2. `_wait_for_worker_registration` timeout too short (Low — related to #1)

The current timeout in `LocalClusterClient._wait_for_worker_registration` is 5.0s. With the autoscaler path (2s eval + 1s cooldown + worker startup), this is tight. Even after fixing issue #1 (0.5s eval), it's worth bumping to 10.0s to avoid flakes in CI, since the autoscaler runs on a background thread and timing is not deterministic.

#### 3. E2E resource test values: Option B is correct but the doc understates the scope

The design says "Change `test_small_job_skips_oversized_job` to use `cpu=10000`". Looking at the test code, the `submit()` helper on `E2ECluster` takes a `cpu` parameter that becomes `ResourceSpec(cpu=cpu)`. With 1000-CPU workers, `cpu=100` *will* be schedulable — the test would *pass the big job* and break the assertion `big_status["state"] == "JOB_STATE_PENDING"`.

Option B (cpu=10000) is the right call. The doc correctly identifies both tests. No additional tests appear to depend on the 4-CPU assumption.

#### 4. `port_range` removal is safe

Traced all callers:
- `LocalClientConfig.port_range` → only used in `IrisClient.local()` → passed to `LocalClusterClient.create()` → passed to `WorkerConfig.port_range`
- No external code references `LocalClientConfig.port_range` outside `client.py`
- After the refactor, `PortAllocator` in `_create_local_autoscaler` handles port allocation with its own range `(30000, 40000)`

Dropping it is clean. No backward-compat shim needed (per AGENTS.md).

#### 5. `use_docker=True` path in E2ECluster is unaffected — confirmed

The Docker path creates real `DockerRuntime`, `BundleCache`, `ImageCache` with `environment_provider=None` (probes real system). This path does not use `ClusterManager` or `LocalVmManager`. Leaving it as-is is correct. No `ClusterManager` equivalent exists for Docker-on-localhost.

#### 6. fakes.py verdict is correct

Confirmed: `FakeVm`/`FakeVmGroup` test autoscaler state machine logic with explicit `tick()` calls. `LocalVmManager` runs real workers. Zero overlap. Keeping fakes.py is the right call.

#### 7. Race condition: autoscaler eval timing (Low)

After `ClusterManager.start()` returns, the autoscaler thread may not have run its first evaluation yet. The design correctly notes that `_wait_for_worker_registration` is still needed. The only risk is if the autoscaler thread hasn't started by the time polling begins — but Python's `threading.Thread.start()` guarantees the thread is created, and the Controller's `start()` method starts the autoscaler before returning. The polling loop with 0.1s sleep handles this gracefully.

#### 8. No other callers break

`LocalClusterClient` callers: `IrisClient.local()`, `test_local_client.py`, `test_worker_pool.py`, `test_e2e.py` (imports providers only, not `LocalClusterClient` itself), `__init__.py` (re-export). All are covered by the migration plan.

### Summary

The design is sound. One concrete bug to fix before implementation: make `_create_local_autoscaler` use `config.autoscaler` instead of hardcoding a fresh config. Bump `_wait_for_worker_registration` timeout to 10s as a safety margin. Everything else checks out.
