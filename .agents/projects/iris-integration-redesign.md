# Iris Integration Test Redesign

## Problem

The iris integration test suite at `tests/integration/iris/` has several issues:

1. **Fake pipeline test** — `test_marin_on_iris.py:24-71` uses a fake JSON-file-shuffling pipeline instead of reusing the real marin integration test from `tests/integration_test.py:127-354` (`create_steps()`).

2. **`IntegrationJobs` class** — `tests/integration/iris/jobs.py:12` wraps simple static methods in a class for no reason. These should be top-level functions.

3. **Local cluster fallback in conftest** — `tests/integration/iris/conftest.py:354-362` boots a local in-process cluster when `--controller-url` is not provided. This creates a hidden 10-worker cluster with coscheduling groups that makes tests slow and hard to reason about. The conftest should only connect to an externally-provided controller.

4. **Tests that need their own local clusters are mixed in** — `test_checkpoint_restore` (`:299`), `test_gpu_worker_metadata` (`:387`), `test_static_auth_rpc_access` (`:475`), and `test_static_auth_job_ownership` (`:510`) all create standalone `LocalCluster` instances. These test iris-specific features and belong under `lib/iris/tests/`.

5. **No CLI path coverage** — All tests submit jobs via the Python `IrisClient` API. None exercise the `iris job run` CLI (`lib/iris/src/iris/cli/job.py:681`).

6. **No GitHub workflow for iris integration tests** — The existing `marin-itest.yaml` only runs `tests/integration_test.py` on Ray. The iris integration tests at `tests/integration/iris/` have no CI workflow.

## Proposed Solution

### Approach

- **Strip conftest down to a connector**: Remove the local cluster fallback, all config builders, and capability discovery. The conftest becomes ~30 lines: parse `--controller-url`, connect `IrisClient` + `ControllerServiceClientSync`, yield an `IrisIntegrationCluster`.

- **Move iris-specific tests to `lib/iris/tests/`**: Auth tests and checkpoint/restore tests create their own local clusters — they don't use the shared `integration_cluster` fixture. They belong in `lib/iris/tests/test_auth.py` and `lib/iris/tests/test_checkpoint_restore.py`.

- **Replace fake pipeline with real `create_steps()`**: Import `create_steps` from `tests/integration_test` and run it via the marin executor inside an Iris job.

- **Add CLI dispatch tests**: Use `subprocess.run(["iris", "job", "run", ...])` to submit jobs via CLI, then poll status via the Python API.

- **New GitHub workflow**: Start a local iris cluster via `iris cluster start --local`, pass the URL to pytest.

### Why this approach

The alternative of keeping the local cluster fallback "for convenience" means conftest has 300+ lines of config builders and every test run boots 10 workers. Making `--controller-url` mandatory forces the workflow to explicitly manage the cluster lifecycle, which is more transparent and matches how real deployments work.

Moving auth/checkpoint tests to `lib/iris/tests/` rather than keeping them in a separate file under `tests/integration/iris/` is correct because these tests are iris unit/integration tests that don't need the marin workspace — they only use iris APIs and create their own clusters.

## Implementation Plan

### Item 1: Slim down `tests/integration/iris/conftest.py`

**Files**: `tests/integration/iris/conftest.py`

**Changes**:
- Remove `_make_integration_config`, `_add_cpu_group`, `_add_coscheduling_group`, `_add_coscheduling_group_4vm`, `_add_multi_region_groups`, `INTEGRATION_WORKER_COUNT` — all config builders for the local cluster fallback
- Remove `ClusterCapabilities` and `discover_capabilities` — move to `lib/iris/tests/conftest.py` if needed there
- Remove the `capabilities` fixture
- Remove `pytest_collection_modifyitems` (timeout adjustment for remote clusters can be a simple marker)
- Make `--controller-url` required (raise `pytest.UsageError` if not provided)
- Keep `IrisIntegrationCluster` as the test helper class — it's useful
- Keep `IRIS_ROOT`, `DEFAULT_CONFIG`

New conftest structure:

```python
import logging
import pytest
from iris.client.client import IrisClient
from iris.rpc.cluster_connect import ControllerServiceClientSync
from pathlib import Path

from .cluster import IrisIntegrationCluster

logger = logging.getLogger(__name__)

IRIS_ROOT = Path(__file__).resolve().parents[3] / "lib" / "iris"


def pytest_addoption(parser):
    parser.addoption("--controller-url", required=True, help="Iris controller URL")


@pytest.fixture(scope="module")
def integration_cluster(request):
    """Connect to an existing Iris controller."""
    url = request.config.getoption("--controller-url")
    client = IrisClient.remote(url, workspace=IRIS_ROOT)
    controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
    tc = IrisIntegrationCluster(
        url=url,
        client=client,
        controller_client=controller_client,
        job_timeout=120.0,
    )
    yield tc
    controller_client.close()
```

Move `IrisIntegrationCluster` to a new `tests/integration/iris/cluster.py` (extracted from conftest, replacing `jobs.py`).

**Tests**: Verify conftest loads correctly with `pytest --co --controller-url http://localhost:10000`.

### Item 2: Replace `IntegrationJobs` class with top-level functions in `jobs.py`

**Files**: `tests/integration/iris/jobs.py`, `tests/integration/iris/test_iris_integration.py`

**Changes**:
- Convert every `IntegrationJobs.xxx` static method to a top-level function: `quick()`, `sleep()`, `fail()`, `noop()`, `busy_loop()`, `log_verbose()`, `register_endpoint()`, `validate_ports()`, `validate_job_context()`
- Update all call sites: `IntegrationJobs.quick` → `quick`, `IntegrationJobs.fail` → `fail`, etc.
- Delete the `IntegrationJobs` class

```python
# tests/integration/iris/jobs.py
def quick():
    return 1

def sleep(duration: float):
    import time
    time.sleep(duration)
    return 1

def fail():
    raise ValueError("intentional failure")

# ... etc
```

**Tests**: All existing tests that reference `IntegrationJobs.xxx` must be updated and still pass.

### Item 3: Move auth tests to `lib/iris/tests/test_auth.py`

**Files**:
- Remove from: `tests/integration/iris/test_iris_integration.py:461-551`
- Create: `lib/iris/tests/test_auth.py`

**What moves**:
- `_AUTH_TOKEN`, `_AUTH_USER` constants
- `_login_for_jwt()` helper
- `test_static_auth_rpc_access()` — creates its own `LocalCluster` with auth config
- `test_static_auth_job_ownership()` — creates its own `LocalCluster` with multi-user auth

These tests don't use the `integration_cluster` fixture at all. They create standalone clusters with auth enabled. They import `_make_controller_only_config` which also needs to move.

Also move the helper `_make_controller_only_config` since both auth tests and the GPU metadata test need it.

```python
# lib/iris/tests/test_auth.py
import pytest
from iris.cluster.config import load_config, make_local_config
from iris.cluster.local_cluster import LocalCluster
from iris.rpc import cluster_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync

IRIS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = IRIS_ROOT / "examples" / "test.yaml"

def _make_controller_only_config():
    """Build a local config with no auto-scaled workers."""
    config = load_config(DEFAULT_CONFIG)
    config.scale_groups.clear()
    sg = config.scale_groups["placeholder"]
    # ... same as current
    return make_local_config(config)

def test_static_auth_rpc_access():
    # ... moved verbatim

def test_static_auth_job_ownership():
    # ... moved verbatim
```

**Tests**: `cd lib/iris && uv run pytest tests/test_auth.py -v`

### Item 4: Move checkpoint/restore and GPU metadata tests to `lib/iris/tests/`

**Files**:
- Remove from: `tests/integration/iris/test_iris_integration.py:299-454`
- Create: `lib/iris/tests/test_checkpoint_restore.py`
- Create: `lib/iris/tests/test_gpu_worker_metadata.py`

**What moves**:
- `test_checkpoint_restore()` — creates its own `LocalCluster`, tests checkpoint/restore lifecycle
- `test_gpu_worker_metadata()` — creates its own cluster with mocked nvidia-smi
- `_NVIDIA_SMI_H100_8X` constant
- `_make_controller_only_config()` — shared with auth tests, put in a common helper or duplicate (it's 10 lines)

These tests are self-contained iris-level tests. They don't need the marin workspace at all.

**Tests**: `cd lib/iris && uv run pytest tests/test_checkpoint_restore.py tests/test_gpu_worker_metadata.py -v`

### Item 5: Add CLI dispatch tests to `tests/integration/iris/test_cli_dispatch.py`

**Files**: Create `tests/integration/iris/test_cli_dispatch.py`

**Changes**: Tests that submit jobs via `iris job run` CLI subprocess, then verify completion via the Python API.

```python
import subprocess
import pytest
from iris.rpc import cluster_pb2

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_cli_submit_and_succeed(integration_cluster):
    """Submit a simple command via 'iris job run' CLI."""
    result = subprocess.run(
        [
            "iris",
            "--controller-url", integration_cluster.url,
            "job", "run",
            "--no-wait",
            "--name", "itest-cli-simple",
            "--cpu", "1",
            "--memory", "1g",
            "--", "python", "-c", "print('hello from cli')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Extract job ID from CLI output and poll via API
    # The CLI prints the job ID on submission with --no-wait
    job_id = _extract_job_id(result.stdout)
    _wait_for_job(integration_cluster, job_id, cluster_pb2.JOB_STATE_SUCCEEDED)


def test_cli_submit_failing_command(integration_cluster):
    """CLI-submitted job that exits non-zero is reported as FAILED."""
    result = subprocess.run(
        [
            "iris",
            "--controller-url", integration_cluster.url,
            "job", "run",
            "--no-wait",
            "--name", "itest-cli-fail",
            "--cpu", "1",
            "--memory", "1g",
            "--", "python", "-c", "raise SystemExit(1)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0  # CLI itself succeeds, job fails
    job_id = _extract_job_id(result.stdout)
    _wait_for_job(integration_cluster, job_id, cluster_pb2.JOB_STATE_FAILED)


def _extract_job_id(stdout: str) -> str:
    """Extract job ID from 'iris job run --no-wait' output."""
    # The CLI prints something like "Job submitted: /user/job-name (id: xxx)"
    # Parse accordingly — inspect actual output format
    for line in stdout.strip().split("\n"):
        if "job" in line.lower() and "/" in line:
            # Extract the job name/ID
            ...
    raise ValueError(f"Could not extract job ID from: {stdout}")


def _wait_for_job(cluster, job_id: str, expected_state: int, timeout: float = 60.0):
    """Poll job status until terminal state."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        request = cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        response = cluster.controller_client.get_job_status(request)
        if response.job.state == expected_state:
            return
        if response.job.state in (
            cluster_pb2.JOB_STATE_SUCCEEDED,
            cluster_pb2.JOB_STATE_FAILED,
            cluster_pb2.JOB_STATE_KILLED,
        ):
            assert response.job.state == expected_state, (
                f"Expected {expected_state}, got {response.job.state}"
            )
        time.sleep(1)
    raise TimeoutError(f"Job did not reach state {expected_state} in {timeout}s")
```

**Note**: The exact CLI output format for `--no-wait` needs to be verified by inspecting `lib/iris/src/iris/cli/job.py:681` `run()` function's `--no-wait` path. The implementer should check what gets printed and parse accordingly.

**Tests**: Run with `--controller-url` pointing to a live cluster.

### Item 6: Port real marin pipeline test to `tests/integration/iris/test_marin_on_iris.py`

**Files**: `tests/integration/iris/test_marin_on_iris.py`

**Changes**:
- Remove the fake `_marin_data_pipeline_step` function (`:24-71`)
- Remove `test_multi_step_pipeline_on_iris` (`:85-100`) — trivial sequential job submission, not testing marin
- Replace with a single test that runs the real marin integration pipeline via Iris

The challenge: `create_steps()` from `tests/integration_test.py:127` returns `ExecutorStep` objects that need the marin executor to run. The current Ray test (`integration_test.py:357-403`) calls `executor_main(config, steps=steps)` which uses Ray. For Iris, we need to either:

**Option A**: Submit the entire `integration_test.py` as an Iris command-line job:
```python
def test_marin_pipeline_on_iris(integration_cluster):
    """Run the full marin integration pipeline as an Iris job."""
    result = subprocess.run(
        [
            "iris",
            "--controller-url", integration_cluster.url,
            "job", "run",
            "--no-wait",
            "--name", "itest-marin-pipeline",
            "--cpu", "4",
            "--memory", "8g",
            "--timeout", "600",
            "--", "python", "tests/integration_test.py",
            "--prefix", "/tmp/iris-itest",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # ... poll for completion
```

**Option B**: Create a thin wrapper that imports `create_steps` and runs `executor_main` inside an Iris task callable:
```python
def _run_marin_pipeline():
    """Callable submitted as an Iris job that runs the marin pipeline."""
    import os, sys, dataclasses
    from marin.execution.executor import ExecutorMainConfig, executor_main
    from tests.integration_test import create_steps

    prefix = "/tmp/iris-marin-itest"
    os.environ["MARIN_PREFIX"] = prefix
    config = ExecutorMainConfig(prefix=prefix, executor_info_base_path=os.path.join(prefix, "experiments"))
    steps = create_steps("itest-marin", "./tests/quickstart-data")
    executor_main(config, steps=steps)
```

**Recommended: Option A** via CLI. This exercises the CLI path, doesn't require cloudpickle to serialize the entire marin import tree, and matches how users actually run jobs on Iris. The `integration_test.py` script already has a `main()` entrypoint that accepts `--prefix`.

However, Option A requires the Iris worker to have the marin package installed. In CI with a local cluster (`iris cluster start --local`), the worker processes share the same Python environment, so this works. For remote clusters, the task image must include marin.

**Tests**: This is slow (~5-10 min). Mark as `@pytest.mark.marin_pipeline` so it can be skipped in quick runs.

### Item 7: Update GitHub workflow for iris integration tests

**Files**: Create `.github/workflows/iris-integration.yaml`

**Changes**: New workflow that:
1. Installs dependencies (same as `marin-itest.yaml`)
2. Starts a local iris cluster in the background
3. Runs pytest against `tests/integration/iris/`
4. Cleans up

```yaml
name: Iris - Integration Tests

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

jobs:
  iris-itest:
    if: github.event_name == 'push' || github.event.pull_request.head.repo.full_name == github.repository
    runs-on: ubuntu-latest
    timeout-minutes: 30
    concurrency:
      group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}
      cancel-in-progress: true

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - uses: actions/setup-node@v4
        with:
          node-version: "22"
      - uses: astral-sh/setup-uv@v7
        with:
          enable-cache: true

      - name: Install dependencies
        run: uv sync --all-packages --extra=cpu --extra=dedup --no-default-groups

      - name: Start local Iris cluster
        run: |
          # Start local cluster in background, capture URL
          uv run iris --config lib/iris/examples/test.yaml \
            cluster start --local > /tmp/iris-cluster.log 2>&1 &
          CLUSTER_PID=$!
          echo "CLUSTER_PID=$CLUSTER_PID" >> "$GITHUB_ENV"

          # Wait for "Controller started at" line
          for i in $(seq 1 60); do
            if grep -q "Controller started at" /tmp/iris-cluster.log 2>/dev/null; then
              URL=$(grep "Controller started at" /tmp/iris-cluster.log | awk '{print $NF}')
              echo "IRIS_CONTROLLER_URL=$URL" >> "$GITHUB_ENV"
              echo "Cluster ready at $URL"
              break
            fi
            sleep 1
          done

          if [ -z "${IRIS_CONTROLLER_URL:-}" ]; then
            echo "Cluster failed to start"
            cat /tmp/iris-cluster.log
            exit 1
          fi

      - name: Run integration tests
        run: |
          uv run pytest tests/integration/iris/ \
            --controller-url "$IRIS_CONTROLLER_URL" \
            -v --tb=short --timeout=120
        env:
          WANDB_MODE: offline
          JAX_TRACEBACK_FILTERING: off

      - name: Stop cluster
        if: always()
        run: kill $CLUSTER_PID 2>/dev/null || true
```

**Note**: The `iris cluster start --local` command (`:218-260` in `cluster.py`) prints `Controller started at {address}` and then blocks. The workflow captures this URL from stdout.

The marin pipeline test (Item 6) may need to be excluded from the default CI run if it takes too long, or run in a separate workflow with more timeout.

**Tests**: The workflow itself is the test.

### Item 8: Clean up `tests/integration/iris/test_iris_integration.py`

**Files**: `tests/integration/iris/test_iris_integration.py`

**Changes after items 1-4**:
- Remove auth tests (moved to `lib/iris/tests/test_auth.py`)
- Remove `test_checkpoint_restore` (moved to `lib/iris/tests/test_checkpoint_restore.py`)
- Remove `test_gpu_worker_metadata` and `_make_controller_only_config`, `_NVIDIA_SMI_H100_8X` (moved to `lib/iris/tests/test_gpu_worker_metadata.py`)
- Remove `capabilities` fixture usage — tests that need `capabilities` (`test_region_constrained_routing`, `test_port_allocation`, `test_log_levels_populated`, `test_log_level_filter`) should either skip based on a simpler check or move to a separate file
- Update imports: `IntegrationJobs.quick` → `quick` from `.jobs`
- Remove unused imports (`subprocess`, `patch`, `uuid`, `ProcessRuntime`, `Worker`, `WorkerConfig`, `ThreadContainer`, etc.)

Tests remaining in `test_iris_integration.py`:
- `test_submit_and_succeed`
- `test_submit_and_fail`
- `test_cancel_job_releases_resources`
- `test_endpoint_registration`
- `test_port_allocation` (needs capability check → skip on CLI-only local cluster)
- `test_reservation_gates_scheduling`
- `test_log_levels_populated` / `test_log_level_filter` (need capability check)
- `test_region_constrained_routing` (need multi-region → skip on simple local cluster)
- `test_profile_running_task`
- `test_exec_in_container`
- `test_stress_50_tasks`

For capability-dependent tests, replace the `capabilities` fixture with inline skip logic that queries workers directly.

## Dependency Graph

```
Item 1 (conftest)  ──┐
Item 2 (jobs.py)   ──┤
                     ├──> Item 8 (clean up test_iris_integration.py)
Item 3 (auth)      ──┤
Item 4 (ckpt/gpu)  ──┘

Item 5 (CLI tests) — independent, needs Item 1

Item 6 (marin pipeline) — independent, needs Item 1

Item 7 (workflow) — depends on all others being done
```

Items 1-4 can be done in parallel. Item 8 merges their results. Items 5 and 6 are independent of 3/4. Item 7 comes last.

## Risks and Open Questions

1. **`iris job run --no-wait` output format**: The CLI dispatch tests need to parse the job ID from stdout. The exact format needs to be verified. If the output is not machine-parseable, we may need to add `--json` output support to `iris job run`.

2. **Marin pipeline test on local cluster**: Running `executor_main` inside a local iris worker process means the worker needs all marin dependencies. In CI, `uv sync --all-packages` installs everything, so this should work. But the worker process is a subprocess, not the main process — we need to verify that `tests/quickstart-data/` is accessible from the worker's cwd.

3. **`create_steps` imports**: `tests/integration_test.py` is a script with a `@draccus.wrap()` main. Importing `create_steps` from it will trigger top-level imports of `ray`, `fray`, etc. The integration test file should be refactored to make `create_steps` importable without side effects, or the marin pipeline test should invoke the script as a subprocess (Option A).

4. **Worker count for local cluster**: The current conftest boots 10 workers. The CI workflow will boot a local cluster using `test.yaml` config — that config has many scale groups aimed at GCP. `--local` converts them to local workers. We may need a simpler CI-specific config (e.g., `test-local.yaml`) with just 2-4 CPU workers to keep CI fast.

5. **Capability-dependent tests**: Tests like `test_region_constrained_routing` need multi-region workers. On a simple local cluster, these will always skip. This is acceptable — these tests are meant for cloud smoke runs.
