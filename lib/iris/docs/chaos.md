# Iris Chaos Testing Framework

Multi-stage plan for building a chaos testing system that validates Iris resilience
against distributed system failure modes.

## Overview

The chaos framework is two things:

1. **`iris.chaos` module** — global chaos rule registry: `enable_chaos()`, `chaos()`, `reset_chaos()`
2. **Inline `chaos()` calls** — sprinkled at key points in production code (zero-cost when inactive)

Tests use `ClusterManager` + `make_local_config()` + `IrisClient.remote()` — the same
pattern as `screenshot-dashboard.py`. No E2ECluster.

## Stage 1: Core chaos module + infrastructure

**Assignee: ml-engineer**
**Validate: senior-engineer**

### 1a. Create `lib/iris/src/iris/chaos.py`

```python
"""Global chaos injection for testing distributed failure scenarios.

Usage:
    from iris.chaos import chaos, chaos_raise, enable_chaos, reset_chaos

    enable_chaos("controller.dispatch", failure_rate=0.3)

    # Site decides what to do:
    if chaos("controller.dispatch"):
        raise Exception("chaos: dispatch failed")

    # Or use helper for simple raise cases:
    chaos_raise("worker.bundle_download")

    reset_chaos()
"""

import random
import time
import threading
from dataclasses import dataclass, field


@dataclass
class ChaosRule:
    failure_rate: float = 1.0
    error: Exception | None = None
    delay_seconds: float = 0.0
    max_failures: int | None = None
    _failure_count: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def should_fire(self) -> bool:
        with self._lock:
            if self.max_failures is not None and self._failure_count >= self.max_failures:
                return False
            return random.random() < self.failure_rate

    def fire(self) -> None:
        with self._lock:
            self._failure_count += 1
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)


_rules: dict[str, ChaosRule] = {}


def enable_chaos(
    key: str,
    failure_rate: float = 1.0,
    error: Exception | None = None,
    delay_seconds: float = 0.0,
    max_failures: int | None = None,
) -> None:
    _rules[key] = ChaosRule(
        failure_rate=failure_rate,
        error=error,
        delay_seconds=delay_seconds,
        max_failures=max_failures,
    )


def chaos(key: str) -> bool:
    """Check if chaos should fire for this key.

    Returns True if chaos is active and fires, False otherwise.
    Injection sites decide what to do with the signal.
    """
    rule = _rules.get(key)
    if rule is None:
        return False
    if rule.should_fire():
        rule.fire()  # increment counter, apply delay
        return True
    return False


def chaos_raise(key: str) -> None:
    """Convenience: raise an exception if chaos fires for this key."""
    if chaos(key):
        rule = _rules[key]
        if rule.error:
            raise rule.error
        raise RuntimeError(f"chaos: {key}")


def reset_chaos() -> None:
    _rules.clear()
```

### 1b. Inject `chaos()` calls at all key points in production code

Each site decides what to do when chaos fires. Use `if chaos(key):` for control flow,
or `chaos_raise(key)` for simple failure cases. Zero-cost when `_rules` is empty.

**Controller — `controller.py`**

```python
# In _send_run_task_rpc, inside retry loop before stub.run_task(request):
if chaos("controller.dispatch"):
    raise Exception("chaos: dispatch unavailable")
```

**Worker — `worker.py`**

```python
# In _heartbeat_loop, before register_worker() RPC (both initial and periodic):
if chaos("worker.heartbeat"):
    logger.debug("chaos: skipping heartbeat")
    continue  # skip this heartbeat attempt

# In _report_task_state, before report RPC:
if chaos("worker.report_task_state"):
    logger.debug("chaos: skipping report_task_state")
    return

# In submit_task, at method entry:
if chaos("worker.submit_task"):
    raise RuntimeError("chaos: worker rejecting task")

# In _execute_task, before get_bundle():
chaos_raise("worker.bundle_download")

# In _execute_task, before create_container():
chaos_raise("worker.create_container")

# In _monitor_task, at top of poll loop:
if chaos("worker.task_monitor"):
    task.transition_to(cluster_pb2.TASK_STATE_FAILED, error="chaos: monitor crashed")
    break
```

All injection points are in `worker.py` because the local ClusterManager runs real
Worker instances in-process. We control everything.

### 1c. Create test infrastructure

`lib/iris/tests/chaos/__init__.py` — empty

`lib/iris/tests/chaos/conftest.py`:
```python
import time
import pytest
from pathlib import Path
from iris.chaos import reset_chaos
from iris.cluster.platform.cluster_manager import ClusterManager, make_local_config
from iris.config import load_config
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, ResourceSpec, EnvironmentSpec
from iris.rpc import cluster_pb2

IRIS_ROOT = Path(__file__).resolve().parents[2]  # lib/iris
DEFAULT_CONFIG = IRIS_ROOT / "examples" / "demo.yaml"

TERMINAL_STATES = {
    cluster_pb2.JOB_STATE_SUCCEEDED,
    cluster_pb2.JOB_STATE_FAILED,
    cluster_pb2.JOB_STATE_KILLED,
    cluster_pb2.JOB_STATE_WORKER_FAILED,
}


@pytest.fixture(autouse=True)
def _reset_chaos():
    yield
    reset_chaos()


@pytest.fixture
def cluster():
    """Boots a local cluster via ClusterManager, yields (url, client)."""
    config = load_config(DEFAULT_CONFIG)
    config = make_local_config(config)
    manager = ClusterManager(config)
    with manager.connect() as url:
        client = IrisClient.remote(url, workspace=IRIS_ROOT)
        yield url, client


def submit(client, fn, name, **kw):
    return client.submit(
        entrypoint=Entrypoint.from_callable(fn),
        name=name,
        resources=ResourceSpec(cpu=1, memory="1g"),
        environment=EnvironmentSpec(),
        **kw,
    )


def wait(client, job, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.status(str(job.job_id))
        if status.state in TERMINAL_STATES:
            return status
        time.sleep(0.5)
    return client.status(str(job.job_id))
```

### 1d. Smoke test: `lib/iris/tests/chaos/test_smoke.py`

```python
from iris.rpc import cluster_pb2
from iris.tests.chaos.conftest import submit, wait


def _ok():
    return 42


def test_smoke(cluster):
    url, client = cluster
    job = submit(client, _ok, "chaos-smoke")
    status = wait(client, job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
```

### 1e. Validation

```bash
uv run pytest lib/iris/tests/chaos/test_smoke.py -v
```

**IMPORTANT**

For the below tests, DO NOT attempt to fix them if they fail, except for failures in the test code itself. The goal is to discover multiple failure modes and implement the chaos system. We will handle repair later.

---

## Stage 2: RPC failure tests 1–5

**Assignee: ml-engineer**
**Validate: senior-engineer**

### `lib/iris/tests/chaos/test_rpc_failures.py`

```python
"""RPC failure chaos tests.

Tests that Iris handles RPC failures gracefully:
- Dispatch retries (4x with exponential backoff)
- Heartbeat timeout (60s)
- Heartbeat reconciliation (running_tasks)
"""
from iris.chaos import enable_chaos
from iris.rpc import cluster_pb2
from iris.tests.chaos.conftest import submit, wait


def _quick():
    return 1

def _slow():
    import time; time.sleep(120)
```

**Test 1** — Intermittent dispatch failure (30%). Controller retries dispatch 4x with
backoff. Task should eventually succeed.
```python
def test_dispatch_intermittent_failure(cluster):
    url, client = cluster
    enable_chaos("controller.dispatch", failure_rate=0.3)
    job = submit(client, _quick, "intermittent-dispatch")
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
```

**Test 2** — Permanent dispatch failure. All 4 retries fail → WorkerFailedEvent → task
rescheduled to other workers → all fail → job FAILED.
```python
def test_dispatch_permanent_failure(cluster):
    url, client = cluster
    enable_chaos("controller.dispatch", failure_rate=1.0)
    job = submit(client, _quick, "permanent-dispatch")
    status = wait(client, job, timeout=120)
    assert status.state == cluster_pb2.JOB_STATE_FAILED
```

**Test 3** — Heartbeat fails 3 times (30s gap), but worker timeout is 60s. Worker should
NOT be marked failed. Task should still succeed.
```python
def test_heartbeat_temporary_failure(cluster):
    url, client = cluster
    enable_chaos("worker.heartbeat", failure_rate=1.0, max_failures=3)
    job = submit(client, _quick, "temp-hb-fail")
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
```

**Test 4** — Heartbeat permanently fails. After 60s, worker marked failed, tasks
become WORKER_FAILED.
```python
def test_heartbeat_permanent_failure(cluster):
    url, client = cluster
    enable_chaos("worker.heartbeat", failure_rate=1.0)
    job = submit(client, _slow, "perm-hb-fail")
    status = wait(client, job, timeout=90)
    assert status.state in (cluster_pb2.JOB_STATE_FAILED,
                            cluster_pb2.JOB_STATE_WORKER_FAILED)
```

**Test 5** — `report_task_state` always fails. Controller should detect task completion
via heartbeat reconciliation (`running_tasks` goes empty when task finishes).
```python
def test_report_task_state_failure(cluster):
    url, client = cluster
    enable_chaos("worker.report_task_state", failure_rate=1.0)
    job = submit(client, _quick, "report-fail")
    status = wait(client, job, timeout=60)
    assert status.state in (cluster_pb2.JOB_STATE_SUCCEEDED,
                            cluster_pb2.JOB_STATE_FAILED)
```

### Validation

```bash
uv run pytest lib/iris/tests/chaos/test_rpc_failures.py -v
```

---

## Stage 3: Worker failure tests 6–10

**Assignee: ml-engineer**
**Validate: senior-engineer**

All worker failures are injected via inline `chaos()` in `worker.py`. The local
ClusterManager runs real Worker instances in-process, so we control everything
through the chaos module — no need to reach into manager internals.

### `lib/iris/tests/chaos/test_worker_failures.py`

```python
"""Worker failure chaos tests.

Tests worker crashes, delayed registration, stale state, and task-level retries.
All chaos is injected inline in worker.py.
"""
import time
from iris.chaos import enable_chaos
from iris.rpc import cluster_pb2
from iris.tests.chaos.conftest import submit, wait


def _quick():
    return 1

def _slow():
    import time; time.sleep(120)
```

**Test 6** — Worker task monitor crashes mid-task. Task fails, controller detects
via heartbeat reconciliation or report_task_state.
```python
def test_worker_crash_mid_task(cluster):
    url, client = cluster
    # task_monitor chaos kills the monitoring loop — task fails with error
    enable_chaos("worker.task_monitor", failure_rate=1.0,
                 error=RuntimeError("chaos: monitor crashed"))
    job = submit(client, _quick, "crash-mid-task")
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_FAILED
```

**Test 7** — Worker heartbeat delayed by 10s on first attempt. Task pends, then
schedules once registration completes.
```python
def test_worker_delayed_registration(cluster):
    url, client = cluster
    enable_chaos("worker.heartbeat", delay_seconds=10.0, max_failures=1)
    job = submit(client, _quick, "delayed-reg")
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
```

**Test 8** — Sequential jobs verify reconciliation works across job boundaries.
Worker state is consistent between tasks.
```python
def test_worker_sequential_jobs(cluster):
    url, client = cluster
    for i in range(3):
        job = submit(client, _quick, f"seq-{i}")
        status = wait(client, job, timeout=30)
        assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
```

**Test 9** — All workers' heartbeats fail permanently. Tasks eventually FAILED after
heartbeat timeout (workers become unreachable from controller's perspective).
```python
def test_all_workers_fail(cluster):
    url, client = cluster
    enable_chaos("worker.heartbeat", failure_rate=1.0)
    job = submit(client, _slow, "all-workers-fail")
    status = wait(client, job, timeout=90)
    assert status.state == cluster_pb2.JOB_STATE_FAILED
```

**Test 10** — Container creation fails once, succeeds on retry (with
`max_retries_failure=1`).
```python
def test_task_fails_once_then_succeeds(cluster):
    url, client = cluster
    enable_chaos("worker.create_container", failure_rate=1.0, max_failures=1,
                 error=RuntimeError("chaos: transient container failure"))
    job = submit(client, _quick, "retry-once", max_retries_failure=1)
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
```

### Validation

```bash
uv run pytest lib/iris/tests/chaos/test_worker_failures.py -v
```

---

## Stage 4: Task lifecycle + scheduling tests 11–17

**Assignee: ml-engineer**
**Validate: senior-engineer**

### `lib/iris/tests/chaos/test_task_lifecycle.py`

**Test 11** — Bundle download fails intermittently, task retries handle it.
```python
def test_bundle_download_intermittent(cluster):
    url, client = cluster
    enable_chaos("worker.bundle_download", failure_rate=0.5, max_failures=2,
                 error=RuntimeError("chaos: download failed"))
    job = submit(client, _quick, "bundle-fail", max_retries_failure=3)
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
```

**Test 12** — Task times out, marked FAILED.
```python
def test_task_timeout(cluster):
    url, client = cluster
    def hang(): import time; time.sleep(300)
    job = submit(client, hang, "timeout-test", timeout_seconds=5)
    status = wait(client, job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_FAILED
```

**Test 13** — Coscheduled job: one replica fails → all siblings killed.
```python
def test_coscheduled_sibling_failure(cluster):
    url, client = cluster
    # Fail container creation once — hits one replica, sibling cascade kills all
    enable_chaos("worker.create_container", failure_rate=0.5, max_failures=1,
                 error=RuntimeError("chaos: replica fail"))
    job = submit(client, _quick, "cosched-fail", replicas=2,
                 coscheduling=CoschedulingConfig(group_by="worker"))
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_FAILED
```

**Test 14** — Task fails exactly N-1 times, succeeds on last attempt.
```python
def test_retry_budget_exact(cluster):
    url, client = cluster
    enable_chaos("worker.create_container", failure_rate=1.0, max_failures=2,
                 error=RuntimeError("chaos: transient"))
    job = submit(client, _quick, "exact-retry", max_retries_failure=2)
    status = wait(client, job, timeout=60)
    # 2 failures consumed by chaos, 3rd attempt succeeds (chaos exhausted)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
```

**Test 15** — Workers at capacity, task pends, schedules when capacity frees.
```python
def test_capacity_wait(cluster):
    url, client = cluster
    def blocker(): import time; time.sleep(8); return 1
    blockers = [submit(client, blocker, f"blocker-{i}", cpu=4) for i in range(2)]
    time.sleep(1)
    pending = submit(client, _quick, "pending", cpu=1)
    status = client.status(str(pending.job_id))
    assert status.state == cluster_pb2.JOB_STATE_PENDING
    for b in blockers:
        wait(client, b, timeout=30)
    status = wait(client, pending, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
```

**Test 16** — Scheduling timeout exceeded → UNSCHEDULABLE.
```python
def test_scheduling_timeout(cluster):
    url, client = cluster
    job = submit(client, _quick, "unsched", cpu=9999, scheduling_timeout_seconds=2)
    status = wait(client, job, timeout=10)
    assert status.state in (cluster_pb2.JOB_STATE_FAILED,
                            cluster_pb2.JOB_STATE_UNSCHEDULABLE)
```

**Test 17** — Dispatch delayed by chaos, but eventually goes through.
```python
def test_dispatch_delayed(cluster):
    url, client = cluster
    enable_chaos("controller.dispatch", delay_seconds=3.0, failure_rate=1.0,
                 max_failures=2)
    job = submit(client, _quick, "delayed-dispatch")
    status = wait(client, job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
```

### Validation

```bash
uv run pytest lib/iris/tests/chaos/test_task_lifecycle.py -v
```

---

## Stage 5: Autoscaler/VM failure tests 18–20

**Assignee: ml-engineer**
**Validate: senior-engineer**

### `lib/iris/tests/chaos/test_vm_failures.py`

These use `FakeVmManager` directly since they test the autoscaler layer.
No `cluster` fixture needed.

**Test 18** — VM creation fails with quota exceeded, retry after clearing.
```python
from iris.tests.cluster.vm.fakes import FakeVmManager, FakeVmManagerConfig, FailureMode
from iris.rpc import config_pb2, vm_pb2
from iris.time_utils import now_ms


def test_quota_exceeded_retry():
    config = config_pb2.ScaleGroupConfig(
        name="test", accelerator_variant="v4-8",
        min_slices=0, max_slices=3, zones=["us-central1-a"],
    )
    manager = FakeVmManager(FakeVmManagerConfig(config=config,
                            failure_mode=FailureMode.QUOTA_EXCEEDED))
    try:
        manager.create_vm_group()
        assert False, "should have raised"
    except Exception:
        pass
    manager.set_failure_mode(FailureMode.NONE)
    group = manager.create_vm_group()
    manager.tick(now_ms())
    status = group.status()
    assert any(vm.state == vm_pb2.VM_STATE_READY for vm in status.vms)
```

**Test 19** — VM boots but worker never initializes (stuck in INITIALIZING).
```python
def test_vm_init_stuck():
    config = config_pb2.ScaleGroupConfig(
        name="stuck", accelerator_variant="v4-8",
        min_slices=0, max_slices=3, zones=["us-central1-a"],
    )
    manager = FakeVmManager(FakeVmManagerConfig(config=config, init_delay_ms=999_999_999))
    group = manager.create_vm_group()
    manager.tick(now_ms())
    status = group.status()
    assert all(vm.state != vm_pb2.VM_STATE_READY for vm in status.vms)
```

**Test 20** — VM preempted (terminated).
```python
def test_vm_preempted():
    config = config_pb2.ScaleGroupConfig(
        name="preempt", accelerator_variant="v4-8",
        min_slices=0, max_slices=3, zones=["us-central1-a"],
    )
    manager = FakeVmManager(FakeVmManagerConfig(config=config))
    group = manager.create_vm_group()
    manager.tick(now_ms())
    assert any(vm.state == vm_pb2.VM_STATE_READY for vm in group.status().vms)
    group.terminate()
    status = group.status()
    assert all(vm.state == vm_pb2.VM_STATE_TERMINATED for vm in status.vms)
```

### Validation

```bash
uv run pytest lib/iris/tests/chaos/test_vm_failures.py -v
```

---

## Stage 6: Final review

**Assignee: senior-engineer**

1. Verify all 20 tests pass or have documented expected-failure rationale
2. Verify `chaos()` placement — inside vs outside retry loops must be deliberate:
   - `controller.dispatch` is inside the retry loop (correct: each attempt can fail independently)
   - `worker.heartbeat` is before the RPC (correct: simulates network failure)
   - `worker.task_monitor` is inside the poll loop (correct: simulates monitor crash)
3. Verify `reset_chaos()` autouse fixture prevents cross-test contamination
4. Verify exception types match what the code actually catches/handles
5. Document findings: which failure modes Iris handles, which need fixes
6. Write final report to `logs/chaos/summary.md`

---

## Chaos Injection Points (complete list)

| Key | File | Location | What it guards |
|-----|------|----------|---------------|
| `controller.dispatch` | `controller.py` | `_send_run_task_rpc`, before `stub.run_task()` | Controller→Worker dispatch RPC |
| `worker.heartbeat` | `worker.py` | `_heartbeat_loop`, before both registration RPCs | Worker→Controller heartbeat |
| `worker.report_task_state` | `worker.py` | `_report_task_state`, before RPC | Worker→Controller state report |
| `worker.submit_task` | `worker.py` | `submit_task`, at method entry | Worker task acceptance |
| `worker.bundle_download` | `worker.py` | `_execute_task`, before `get_bundle()` | Bundle download |
| `worker.create_container` | `worker.py` | `_execute_task`, before `create_container()` | Container creation |
| `worker.task_monitor` | `worker.py` | `_monitor_task`, top of poll loop | Task monitoring/liveness |

## Files Modified (production)

| File | Change |
|------|--------|
| `lib/iris/src/iris/chaos.py` | **New** — ~50 lines |
| `lib/iris/src/iris/cluster/controller/controller.py` | 1 `chaos()` call |
| `lib/iris/src/iris/cluster/worker/worker.py` | 6 `chaos()` calls |

## Files Created (tests)

| File | Purpose |
|------|---------|
| `lib/iris/tests/chaos/__init__.py` | Package |
| `lib/iris/tests/chaos/conftest.py` | Fixtures + helpers |
| `lib/iris/tests/chaos/test_smoke.py` | Smoke test |
| `lib/iris/tests/chaos/test_rpc_failures.py` | Tests 1–5 |
| `lib/iris/tests/chaos/test_worker_failures.py` | Tests 6–10 |
| `lib/iris/tests/chaos/test_task_lifecycle.py` | Tests 11–17 |
| `lib/iris/tests/chaos/test_vm_failures.py` | Tests 18–20 |

## Test Inventory

| # | Test | Chaos Key | Failure Mode | Expected |
|---|------|-----------|-------------|----------|
| 1 | dispatch_intermittent_failure | controller.dispatch | 30% fail | SUCCEEDED |
| 2 | dispatch_permanent_failure | controller.dispatch | 100% fail | FAILED |
| 3 | heartbeat_temporary_failure | worker.heartbeat | 3 failures then stops | SUCCEEDED |
| 4 | heartbeat_permanent_failure | worker.heartbeat | 100% fail | FAILED (timeout) |
| 5 | report_task_state_failure | worker.report_task_state | 100% fail | Reconciled |
| 6 | worker_crash_mid_task | worker.task_monitor | Monitor crashes | FAILED |
| 7 | worker_delayed_registration | worker.heartbeat | 10s delay, 1x | SUCCEEDED |
| 8 | worker_sequential_jobs | (none) | Reconciliation check | SUCCEEDED |
| 9 | all_workers_fail | worker.heartbeat | 100% fail | FAILED |
| 10 | task_fails_once_then_succeeds | worker.create_container | 1 failure + retry | SUCCEEDED |
| 11 | bundle_download_intermittent | worker.bundle_download | 50%, 2 max | SUCCEEDED |
| 12 | task_timeout | (timeout param) | 5s timeout | FAILED |
| 13 | coscheduled_sibling_failure | worker.create_container | 1 replica fails | FAILED |
| 14 | retry_budget_exact | worker.create_container | N-1 failures | SUCCEEDED |
| 15 | capacity_wait | (resource limits) | Full → free | SUCCEEDED |
| 16 | scheduling_timeout | (impossible resources) | 2s timeout | UNSCHEDULABLE |
| 17 | dispatch_delayed | controller.dispatch | 3s delay, 2 max | SUCCEEDED |
| 18 | quota_exceeded_retry | (FakeVmManager) | Quota error | VM created |
| 19 | vm_init_stuck | (FakeVmManager) | Infinite init | Not READY |
| 20 | vm_preempted | (FakeVmManager) | Terminate | TERMINATED |

## Stage 7: Virtual time enhancement (optional)

**Purpose:** Speed up chaos tests by replacing real sleeps with virtual time. Tests that
take 60-90s can run in milliseconds.

**Trade-offs:**
- ✅ Fast: 60s timeout tests run in ~100ms
- ✅ Deterministic: no flaky timing issues
- ❌ Not testing real timing: validates logic but not actual time behavior
- ❌ Requires all code uses `time.sleep()` (not asyncio, select, etc.)

### Implementation

`lib/iris/tests/chaos/chronos.py`:
```python
"""Virtual time for chaos tests - makes time.sleep() controllable."""
import time
import threading
import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class SleepEvent:
    wake_time: float
    event: threading.Event = field(compare=False)


class VirtualClock:
    """Thread-safe virtual clock that can be ticked forward for testing.

    Key insight: Lock protects shared state (current_time, sleepers queue).
    Event.wait() happens outside the lock. tick() wakes threads by setting events.
    """

    def __init__(self):
        self._current_time = 0.0
        self._sleepers = []  # min-heap of SleepEvent
        self._lock = threading.Lock()

    def time(self):
        """Returns current virtual time."""
        with self._lock:
            return self._current_time

    def sleep(self, seconds):
        """Blocks thread until chronos.tick() advances past wake time."""
        if seconds <= 0:
            return

        event = threading.Event()
        with self._lock:
            wake_time = self._current_time + seconds
            heapq.heappush(self._sleepers, SleepEvent(wake_time, event))

        # Block WITHOUT holding lock - will be woken by tick()
        event.wait()

    def tick(self, duration=None):
        """Advance time and wake sleeping threads.

        If duration=None, advances to next sleeper's wake time.
        If duration specified, advances by that amount.
        """
        with self._lock:
            if duration is None:
                if not self._sleepers:
                    return  # No sleepers, nothing to do
                target_time = self._sleepers[0].wake_time
            else:
                target_time = self._current_time + duration

            # Wake all threads whose wake_time <= target_time
            while self._sleepers and self._sleepers[0].wake_time <= target_time:
                sleeper = heapq.heappop(self._sleepers)
                self._current_time = sleeper.wake_time
                sleeper.event.set()  # Wake the thread

            # Advance to target even if no sleepers woke
            self._current_time = max(self._current_time, target_time)

    def tick_until_idle(self, max_iterations=1000):
        """Keep ticking until no new sleepers appear.

        Useful for advancing through all pending work. Adds small real delays
        to let threads run between ticks.
        """
        for i in range(max_iterations):
            with self._lock:
                num_sleepers = len(self._sleepers)

            if num_sleepers == 0:
                return  # All done

            self.tick()  # Wake next sleeper
            time.sleep(0.001)  # Small real sleep to let thread run

        raise TimeoutError(f"tick_until_idle exceeded {max_iterations} iterations")
```

### Update conftest.py

Add chronos fixture and update wait() helper:

```python
# Add at top
from iris.tests.chaos.chronos import VirtualClock

# Add fixture
@pytest.fixture
def chronos(monkeypatch):
    """Virtual time fixture - makes time.sleep() controllable for fast tests."""
    clock = VirtualClock()

    # Patch time module
    monkeypatch.setattr(time, "time", clock.time)
    monkeypatch.setattr(time, "monotonic", clock.time)
    monkeypatch.setattr(time, "sleep", clock.sleep)

    return clock

# Update wait() to support optional chronos parameter
def wait(client, job, timeout=60, chronos=None):
    """Wait for job to reach terminal state.

    If chronos is provided, uses virtual time.
    Otherwise uses real time.sleep().
    """
    if chronos:
        # Virtual time: tick until job completes or timeout
        start_time = chronos.time()
        while chronos.time() - start_time < timeout:
            status = client.status(str(job.job_id))
            if status.state in TERMINAL_STATES:
                return status
            chronos.tick(0.5)  # Advance by poll interval
        return client.status(str(job.job_id))
    else:
        # Real time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = client.status(str(job.job_id))
            if status.state in TERMINAL_STATES:
                return status
            time.sleep(0.5)
        return client.status(str(job.job_id))
```

### Usage examples

```python
# Without virtual time (real 90s wait)
def test_heartbeat_permanent_failure(cluster):
    url, client = cluster
    enable_chaos("worker.heartbeat", failure_rate=1.0)
    job = submit(client, _slow, "perm-hb-fail")
    status = wait(client, job, timeout=90)
    assert status.state == cluster_pb2.JOB_STATE_FAILED

# With virtual time (~100ms)
def test_heartbeat_permanent_failure_fast(cluster, chronos):
    url, client = cluster
    enable_chaos("worker.heartbeat", failure_rate=1.0)
    job = submit(client, _slow, "perm-hb-fail")
    status = wait(client, job, timeout=90, chronos=chronos)
    assert status.state == cluster_pb2.JOB_STATE_FAILED
```

### Recommendation

Start without virtual time. Add it later if:
1. Test suite becomes too slow (>5 minutes)
2. You need to run chaos tests frequently in CI
3. You want to test extreme timeouts (hours) quickly

For initial validation, real time is fine and tests actual timing behavior.

---

## Execution

```
Stage 1 (ml-engineer) → senior-engineer review
Stage 2 (ml-engineer) → senior-engineer review
Stage 3 (ml-engineer) → senior-engineer review
Stage 4 (ml-engineer) → senior-engineer review
Stage 5 (ml-engineer) → senior-engineer review
Stage 6: Final senior-engineer review + report
Stage 7 (optional): Add virtual time if tests become too slow
```

Execution log: `logs/chaos/summary.md`
