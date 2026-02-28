# Iris E2E Testing Design

## Problem

Iris has three overlapping test surfaces that evolved independently:

| Surface | Location | What it tests | Cluster setup |
|---------|----------|---------------|---------------|
| **Chaos tests** | `tests/chaos/` | Failure injection (RPC, heartbeat, task lifecycle, VM) | `connect_cluster()` with `demo.yaml` + `make_local_config()` |
| **E2E tests** | `tests/cluster/test_e2e.py` | Job lifecycle, scheduling, ports, endpoints, Docker, TPU sim | `E2ECluster` context manager (manual Controller+Worker or LocalController) |
| **Screenshot script** | `scripts/screenshot-dashboard.py` | Visual dashboard rendering | `connect_cluster()` with `demo.yaml` + Playwright |

These share significant infrastructure (cluster boot, job submission, wait loops)
but use incompatible fixtures and patterns. The screenshot script isn't a test at
all — it's a standalone CLI that reimplements cluster setup and job submission.

**Concrete pain points:**

1. No dashboard regression testing. The screenshot script captures images but
   doesn't validate them.
2. Chaos tests and E2E tests boot separate clusters with different configs.
   Tests that need both chaos injection _and_ multi-worker scheduling require
   duplicating setup.
3. Adding a new test requires choosing between three frameworks, each with
   different conventions for cluster setup, job submission, and state polling.
4. The screenshot script submits 14 jobs with hardcoded names and bespoke
   wait logic that duplicates what the test fixtures already provide.

## Goal

A single `tests/e2e/` directory where every test has the signature:

```python
def test_something(cluster, page, chaos):
    ...
```

Each test gets:
- **`cluster`** — a booted local cluster with an `IrisClient` and RPC access
- **`page`** — a Playwright `Page` pointed at the controller dashboard
- **`chaos`** — the chaos injection context (auto-reset between tests)

Screenshot capture is a utility available to any test, not a separate script.
Tests validate both programmatic state (via RPC) _and_ visual state (via
Playwright assertions or saved screenshots).

## Design

### Directory Structure

```
lib/iris/tests/e2e/
├── conftest.py              # Fixtures: cluster, page, chaos, screenshot helpers
├── test_smoke.py            # Basic job lifecycle (submit, succeed, fail, kill)
├── test_rpc_failures.py     # Chaos: dispatch, heartbeat, notify failures
├── test_task_lifecycle.py   # Chaos: retries, timeouts, coscheduling failures
├── test_worker_failures.py  # Chaos: worker crash, registration delay, all-fail
├── test_vm_lifecycle.py     # Platform: quota exhaustion, stuck init, preemption
├── test_scheduling.py       # Resource scheduling, multi-worker distribution
├── test_heartbeat.py        # Controller heartbeat threshold, timeout detection
├── test_dashboard.py        # Dashboard tabs render, job detail pages, log viewer
├── test_endpoints.py        # Endpoint registration, prefix matching
├── test_high_concurrency.py # 128+ tasks, race condition reproduction
├── test_docker.py           # Docker-only: OOM detection, TPU sim, command→callable
├── chronos.py               # VirtualClock (moved from chaos/)
└── helpers.py               # Shared job functions (_quick, _slow, _block, etc.)
```

### Core Fixtures

#### `cluster` fixture

Boots a local Iris cluster using `connect_cluster()` with a test config derived
from `demo.yaml`. Yields a `TestCluster` dataclass that wraps the url and client:

```python
@dataclass
class TestCluster:
    url: str
    client: IrisClient
    controller_client: ControllerServiceClientSync

    def submit(self, fn, name, *args, **kw) -> Job: ...
    def wait(self, job, timeout=60, chronos=None) -> JobStatus: ...
    def status(self, job) -> JobStatus: ...
    def task_status(self, job, task_index=0) -> TaskStatus: ...
    def kill(self, job) -> None: ...
    def get_task_logs(self, job, task_index=0) -> list[str]: ...
```

This merges the best of both worlds: the chaos conftest's `connect_cluster()`
bootstrap (simpler, uses production code paths) with E2ECluster's convenience
methods (status/wait/kill/logs).

The fixture is **module-scoped** for the cluster boot (expensive) but chaos
state is reset per-test via an autouse fixture.

Config includes the coscheduling group from the current chaos conftest.

```python
@pytest.fixture(scope="module")
def cluster():
    config = load_config(DEFAULT_CONFIG)
    _add_coscheduling_group(config)
    config = make_local_config(config)
    with connect_cluster(config) as url:
        client = IrisClient.remote(url, workspace=IRIS_ROOT)
        controller_client = ControllerServiceClientSync(
            address=url, timeout_ms=30000
        )
        yield TestCluster(url=url, client=client, controller_client=controller_client)
        controller_client.close()
```

Tests that need multiple workers or custom config get a separate fixture:

```python
@pytest.fixture
def multi_worker_cluster():
    config = load_config(DEFAULT_CONFIG)
    config = make_local_config(config, num_workers=4)
    with connect_cluster(config) as url:
        ...
```

#### Why the default cluster doesn't use Docker

`connect_cluster()` + `LocalController` runs workers as in-process threads via
`LocalPlatform`. There is no Docker runtime involved — tasks execute as
functions in the worker thread. This is intentional: it's fast (~2s boot), needs
no Docker daemon, and exercises the same controller/scheduler/RPC code paths
that production uses.

Docker mode requires a fundamentally different setup: manual `Controller` +
`Worker` instances with a `DockerRuntime`, temp directories for bundle caching,
explicit port allocation, and `uv sync` inside each container (~30-60s
overhead). This is the `E2ECluster(use_docker=True)` path.

Since most tests (chaos injection, scheduling, heartbeat, dashboard rendering)
don't care whether tasks run as threads or containers, they use the fast local
cluster. Only tests that exercise Docker-specific behavior (OOM detection via
cgroups, JAX coordinator env vars from `_build_device_env_vars`, command
entrypoints that need a real filesystem) need Docker.

#### `docker_cluster` fixture

For Docker-specific tests, we keep a separate fixture based on `E2ECluster`:

```python
@pytest.fixture(scope="module")
def docker_cluster(shared_cache):
    with E2ECluster(use_docker=True, cache_dir=shared_cache) as cluster:
        yield cluster
```

This lives in `test_docker.py` and is only used by tests marked `@pytest.mark.docker`.

#### `page` fixture

Provides a Playwright `Page` connected to the cluster's dashboard:

```python
@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()

@pytest.fixture
def page(browser, cluster):
    pg = browser.new_page(viewport={"width": 1400, "height": 900})
    pg.goto(f"{cluster.url}/")
    pg.wait_for_load_state("domcontentloaded")
    _wait_for_dashboard_ready(pg)
    yield pg
    pg.close()
```

Tests that don't need Playwright simply don't request the `page` fixture and
pay zero overhead.

#### `chaos` fixture

The chaos context is trivially the auto-reset pattern:

```python
@pytest.fixture(autouse=True)
def _reset_chaos():
    yield
    reset_chaos()
```

Tests call `enable_chaos(...)` directly. No wrapper needed.

#### `chronos` fixture

Unchanged from current implementation — monkeypatches `time.time`,
`time.monotonic`, and `time.sleep` with a `VirtualClock`. Used by tests that
need deterministic time control (heartbeat timeout tests).

#### `screenshot` fixture

Captures and saves screenshots with metadata:

```python
@pytest.fixture
def screenshot(page, request, tmp_path):
    output_dir = Path(os.environ.get(
        "IRIS_SCREENSHOT_DIR",
        tmp_path / "screenshots"
    ))
    output_dir.mkdir(parents=True, exist_ok=True)

    def capture(label: str) -> Path:
        path = output_dir / f"{request.node.name}-{label}.png"
        page.screenshot(path=str(path), full_page=True)
        return path

    return capture
```

Usage:

```python
def test_jobs_tab_renders(cluster, page, screenshot):
    cluster.submit(_quick, "render-test")
    cluster.wait(job, timeout=30)
    page.click('button.tab-btn:has-text("Jobs")')
    screenshot("jobs-tab")
    # Assert job appears in table
    assert page.locator("text=render-test").is_visible()
```

### Test Categories

#### Smoke & Lifecycle (`test_smoke.py`)

Migrated from both `chaos/test_smoke.py` and `test_e2e.py::TestJobLifecycle`:

```python
def test_submit_and_succeed(cluster):
    job = cluster.submit(lambda: 42, "smoke-succeed")
    status = cluster.wait(job, timeout=30)
    assert status.state == JOB_STATE_SUCCEEDED

def test_job_failure_propagates(cluster):
    job = cluster.submit(lambda: 1/0, "smoke-fail")
    status = cluster.wait(job, timeout=30)
    assert status.state == JOB_STATE_FAILED

def test_kill_running_job(cluster, sentinel):
    job = cluster.submit(_block, "smoke-kill", sentinel)
    cluster.wait_for_state(job, JOB_STATE_RUNNING, timeout=10)
    cluster.kill(job)
    sentinel.signal()
    status = cluster.wait(job, timeout=30)
    assert status.state == JOB_STATE_KILLED
```

#### RPC Failures (`test_rpc_failures.py`)

Migrated from `chaos/test_rpc_failures.py`. Unchanged in structure — these
tests are already well-designed.

#### Dashboard Validation (`test_dashboard.py`)

New tests that replace the screenshot script. Each test exercises a specific
dashboard view and validates both RPC state and visual rendering:

```python
def test_jobs_tab_shows_all_states(cluster, page, screenshot):
    """Submit jobs in various states, verify the Jobs tab renders them."""
    jobs = {}
    jobs["succeeded"] = cluster.submit(_quick, "dash-succeeded")
    jobs["failed"] = cluster.submit(_failing, "dash-failed")
    jobs["running"] = cluster.submit(_slow, "dash-running")

    cluster.wait(jobs["succeeded"], timeout=30)
    cluster.wait(jobs["failed"], timeout=30)

    page.goto(f"{cluster.url}/")
    _wait_for_dashboard_ready(page)

    # Verify job names appear
    for name in ["dash-succeeded", "dash-failed", "dash-running"]:
        assert page.locator(f"text={name}").is_visible(timeout=5000)

    screenshot("jobs-all-states")

def test_job_detail_page(cluster, page, screenshot):
    """Job detail page shows task status and logs."""
    job = cluster.submit(_quick, "dash-detail")
    cluster.wait(job, timeout=30)

    page.goto(f"{cluster.url}/job/{job.job_id}")
    _wait_for_dashboard_ready(page)

    assert page.locator("text=SUCCEEDED").is_visible(timeout=5000)
    screenshot("job-detail-succeeded")

def test_fleet_tab(cluster, page, screenshot):
    """Fleet tab shows machines with health status."""
    page.goto(f"{cluster.url}/")
    _wait_for_dashboard_ready(page)
    page.click('button.tab-btn:has-text("Fleet")')

    assert page.locator("text=ready").first.is_visible(timeout=5000)
    screenshot("fleet-tab")

def test_autoscaler_tab(cluster, page, screenshot):
    """Autoscaler tab shows scale groups."""
    page.goto(f"{cluster.url}/")
    _wait_for_dashboard_ready(page)
    page.click('button.tab-btn:has-text("Autoscaler")')

    screenshot("autoscaler-tab")

def test_controller_logs(cluster, page, screenshot):
    """Logs page shows controller log entries."""
    page.goto(f"{cluster.url}#logs")
    page.wait_for_load_state("domcontentloaded")
    # Wait for log lines to appear
    page.wait_for_selector(".log-line", timeout=10000)
    screenshot("controller-logs")
```

#### VM Lifecycle (`test_vm_lifecycle.py`)

Migrated from `chaos/test_vm_failures.py`. These tests use `FakePlatform`
directly and don't need the full cluster fixture. They can optionally use
Playwright to validate the VMs tab renders correctly:

```python
def test_quota_exceeded_retry():
    """VM creation fails with quota, retries after clearing."""
    # Uses FakePlatform directly — no cluster needed
    ...

def test_fleet_tab_shows_states(cluster, page, screenshot):
    """Fleet tab renders machine states from autoscaler."""
    page.goto(f"{cluster.url}/")
    _wait_for_dashboard_ready(page)
    page.click('button.tab-btn:has-text("Fleet")')
    screenshot("fleet-tab")
```

### What Gets Deleted

Once migration is complete:

| Path | Action |
|------|--------|
| `tests/chaos/` | Delete entirely (all tests moved to `tests/e2e/`) |
| `tests/cluster/test_e2e.py` | Delete entirely. Non-Docker tests move to `test_smoke.py`, `test_scheduling.py`, `test_endpoints.py`, `test_high_concurrency.py`. Docker tests (`TestDockerOOM`, `TestTPUSimulation`, `TestCommandParentCallableChild`) move to `tests/e2e/test_docker.py`. `E2ECluster` moves to `conftest.py` as the `docker_cluster` fixture. |
| `scripts/screenshot-dashboard.py` | Delete. Dashboard coverage moves to `test_dashboard.py`. If a standalone screenshot tool is still wanted, it becomes `IRIS_SCREENSHOT_DIR=/tmp/shots uv run pytest tests/e2e/test_dashboard.py`. |

### `test_docker.py`

Docker tests use `E2ECluster(use_docker=True)` which manually wires up
Controller + Workers with a `DockerRuntime`. These tests validate behavior
that only manifests inside real containers:

```python
pytestmark = [pytest.mark.e2e, pytest.mark.docker]

@pytest.fixture(scope="module")
def docker_cluster(shared_cache):
    with E2ECluster(use_docker=True, cache_dir=shared_cache) as cluster:
        yield cluster

def test_oom_detection(docker_cluster):
    """Container killed by OOM reports oom_killed in error message."""
    ...

def test_jax_coordinator_address_format(tpu_sim_cluster):
    """JAX_COORDINATOR_ADDRESS has correct host:port format."""
    ...

def test_command_parent_callable_child(docker_cluster):
    """Command entrypoint parent submits callable child via iris_ctx()."""
    ...
```

The `use_docker` parametrization from the current `test_e2e.py` is removed.
Tests that work identically with or without Docker (job lifecycle, scheduling,
ports, endpoints) just use the local cluster. Tests that specifically need
Docker are explicitly in `test_docker.py`.

### Pytest Marks

The existing `chaos` mark is retired. All tests in `tests/e2e/` share a single
`e2e` mark. Docker tests get an additional `docker` mark for selective
exclusion (Docker tests are slow and need a daemon).

```ini
[tool.pytest.ini_options]
markers = [
    "e2e: end-to-end cluster tests (chaos, dashboard, scheduling)",
    "docker: tests requiring Docker runtime (slow, needs daemon)",
]
```

Every test file in `tests/e2e/` uses `pytestmark = pytest.mark.e2e` at module
level. No per-test `@pytest.mark.chaos` — if you're in `tests/e2e/`, you're an
E2E test. The `e2e` mark is the single selector for the whole suite:

```bash
# Run all E2E tests (chaos + dashboard + scheduling + everything)
uv run pytest lib/iris/tests/e2e/ -m e2e

# Skip Docker tests (fast local-only run)
uv run pytest lib/iris/tests/e2e/ -m "e2e and not docker"

# Only Docker tests
uv run pytest lib/iris/tests/e2e/ -m docker
```

The `IRIS_SCREENSHOT_DIR` environment variable controls where screenshots are
saved. When unset, screenshots go to pytest's `tmp_path` and are cleaned up
normally. When set (e.g. in CI), screenshots persist for artifact collection.

### Handling Scope and Performance

**Module-scoped cluster:** Booting a local cluster takes 2-3 seconds. Tests
within a module share a single cluster to avoid this overhead. Chaos state is
reset per-test (function-scoped autouse fixture), so tests remain isolated.

**Module-scoped browser:** Playwright browser launch is ~1 second. Shared
across tests in a module. Each test gets its own `Page` (function-scoped) for
isolation.

**Tests that need custom cluster config** (multi-worker, custom scale groups)
get their own fixture and are grouped in their own module to avoid conflicts
with the shared cluster.

### Golden Screenshot Testing (Future)

The screenshot infrastructure enables future golden-image comparison:

1. Capture baseline screenshots: `IRIS_SCREENSHOT_DIR=golden/ pytest tests/e2e/test_dashboard.py`
2. Compare against baselines in CI using pixel-diff tools (e.g. `pixelmatch`)
3. Fail on visual regressions above a threshold

This is explicitly a follow-up. The initial implementation saves screenshots
for manual review and uses Playwright's DOM assertions for functional
validation.

### Migration Plan

Phase 1: Create `tests/e2e/` with shared fixtures, migrate `test_smoke.py`.
Verify the new cluster fixture boots correctly and tests pass.

Phase 2: Migrate chaos tests (`test_rpc_failures.py`, `test_task_lifecycle.py`,
`test_worker_failures.py`, `test_heartbeat.py`). These are nearly copy-paste
with the fixture signature changed from `cluster` (tuple) to `cluster`
(TestCluster).

Phase 3: Migrate E2E tests from `test_e2e.py`. Adapt `E2ECluster` convenience
methods into `TestCluster`.

Phase 4: Add `test_dashboard.py` with Playwright-based dashboard validation.
This replaces `screenshot-dashboard.py`.

Phase 5: Delete old test locations and the screenshot script. Update
`AGENTS.md` to reference the new test location.

### AGENTS.md Updates

After migration, update `lib/iris/AGENTS.md` to add:

```markdown
## Testing

All Iris E2E tests live in `tests/e2e/`. Every test is marked `e2e`.
Tests use three core fixtures:

- `cluster`: Booted local cluster with `IrisClient` and RPC access
- `page`: Playwright page pointed at the dashboard (request only when needed)
- `screenshot`: Capture labeled screenshots to `IRIS_SCREENSHOT_DIR`

Chaos injection is auto-reset between tests. Call `enable_chaos()` directly.
Docker tests use a separate `docker_cluster` fixture and are marked `docker`.

Run all E2E tests:
    uv run pytest lib/iris/tests/e2e/ -m e2e

Run E2E tests without Docker (fast):
    uv run pytest lib/iris/tests/e2e/ -m "e2e and not docker"

Run Docker-only tests:
    uv run pytest lib/iris/tests/e2e/ -m docker

Run dashboard tests with saved screenshots:
    IRIS_SCREENSHOT_DIR=/tmp/shots uv run pytest lib/iris/tests/e2e/test_dashboard.py

When modifying the dashboard:
    uv run pytest lib/iris/tests/e2e/test_dashboard.py -x
```
