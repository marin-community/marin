# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Core fixtures for Iris E2E tests.

Boots a local cluster via connect_cluster() + make_local_config() and provides
a TestCluster dataclass that wraps the IrisClient and ControllerServiceClientSync
with convenience methods for job submission, waiting, and status queries.

The cluster fixture is module-scoped (expensive boot ~2-3s), while chaos state
is reset per-test via an autouse fixture.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from iris.chaos import reset_chaos
from iris.client.client import IrisClient, Job
from iris.cluster.config import load_config, make_local_config
from iris.cluster.manager import connect_cluster
from iris.cluster.types import (
    CoschedulingConfig,
    Constraint,
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
    is_job_finished,
)
from iris.rpc import cluster_pb2, config_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.time_utils import Duration

from .chronos import VirtualClock

IRIS_ROOT = Path(__file__).resolve().parents[2]  # lib/iris
DEFAULT_CONFIG = IRIS_ROOT / "examples" / "demo.yaml"


@dataclass
class TestCluster:
    """Wraps a booted local cluster with convenience methods for E2E tests.

    Combines the chaos conftest's connect_cluster() bootstrap with E2ECluster-style
    convenience methods. Methods return protobuf types directly rather than dicts.
    """

    url: str
    client: IrisClient
    controller_client: ControllerServiceClientSync

    def submit(
        self,
        fn,
        name: str,
        *args,
        cpu: int = 1,
        memory: str = "1g",
        ports: list[str] | None = None,
        scheduling_timeout: Duration | None = None,
        replicas: int = 1,
        max_retries_failure: int = 0,
        max_retries_preemption: int = 100,
        timeout: Duration | None = None,
        coscheduling: CoschedulingConfig | None = None,
        constraints: list[Constraint] | None = None,
    ) -> Job:
        """Submit a callable as a job. Returns a Job handle."""
        return self.client.submit(
            entrypoint=Entrypoint.from_callable(fn, *args),
            name=name,
            resources=ResourceSpec(cpu=cpu, memory=memory),
            environment=EnvironmentSpec(),
            ports=ports,
            scheduling_timeout=scheduling_timeout,
            replicas=replicas,
            max_retries_failure=max_retries_failure,
            max_retries_preemption=max_retries_preemption,
            timeout=timeout,
            coscheduling=coscheduling,
            constraints=constraints,
        )

    def status(self, job: Job) -> cluster_pb2.JobStatus:
        """Get the current JobStatus protobuf for a job."""
        job_id = job.job_id.to_wire()
        request = cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        response = self.controller_client.get_job_status(request)
        return response.job

    def task_status(self, job: Job, task_index: int = 0) -> cluster_pb2.TaskStatus:
        """Get the current TaskStatus protobuf for a specific task."""
        task_id = job.job_id.task(task_index).to_wire()
        request = cluster_pb2.Controller.GetTaskStatusRequest(task_id=task_id)
        response = self.controller_client.get_task_status(request)
        return response.task

    def wait(
        self,
        job: Job,
        timeout: float = 60.0,
        chronos: VirtualClock | None = None,
        poll_interval: float = 0.5,
    ) -> cluster_pb2.JobStatus:
        """Poll until a job reaches a terminal state. Returns the final JobStatus.

        If chronos is provided, uses virtual time for deterministic tests.
        Raises TimeoutError if the job doesn't finish within the deadline.
        """
        if chronos is not None:
            start_time = chronos.time()
            while chronos.time() - start_time < timeout:
                status = self.status(job)
                if is_job_finished(status.state):
                    return status
                chronos.tick(poll_interval)
            raise TimeoutError(f"Job {job.job_id} did not complete in {timeout}s (virtual time)")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.status(job)
            if is_job_finished(status.state):
                return status
            time.sleep(poll_interval)
        raise TimeoutError(f"Job {job.job_id} did not complete in {timeout}s")

    def wait_for_state(
        self,
        job: Job,
        state: int,
        timeout: float = 10.0,
        poll_interval: float = 0.1,
    ) -> cluster_pb2.JobStatus:
        """Poll until a job reaches a specific state (e.g. JOB_STATE_RUNNING)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.status(job)
            if status.state == state:
                return status
            time.sleep(poll_interval)
        raise TimeoutError(f"Job {job.job_id} did not reach state {state} in {timeout}s " f"(current: {status.state})")

    def kill(self, job: Job) -> None:
        """Terminate a running job."""
        job_id = job.job_id.to_wire()
        request = cluster_pb2.Controller.TerminateJobRequest(job_id=job_id)
        self.controller_client.terminate_job(request)

    def wait_for_workers(self, min_workers: int, timeout: float = 30.0) -> None:
        """Wait until at least min_workers healthy workers are registered."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            request = cluster_pb2.Controller.ListWorkersRequest()
            response = self.controller_client.list_workers(request)
            healthy = [w for w in response.workers if w.healthy]
            if len(healthy) >= min_workers:
                return
            time.sleep(0.5)
        raise TimeoutError(f"Only {len(healthy)} of {min_workers} workers registered in {timeout}s")

    def get_task_logs(self, job: Job, task_index: int = 0) -> list[str]:
        """Fetch log lines for a task."""
        task_id = job.job_id.task(task_index).to_wire()
        request = cluster_pb2.Controller.GetTaskLogsRequest(id=task_id)
        response = self.controller_client.get_task_logs(request)
        lines = []
        for batch in response.task_logs:
            for entry in batch.logs:
                lines.append(f"{entry.source}: {entry.data}")
        return lines


def _add_coscheduling_group(config: config_pb2.IrisClusterConfig) -> None:
    """Add a scale group with num_vms=2 so coscheduling tests can find a match.

    v5litepod-16 has vm_count=2, so the local platform creates 2 workers per slice
    sharing the same tpu-name. Setting num_vms=2 lets the demand router match
    coscheduled jobs with replicas=2.
    """
    sg = config.scale_groups["tpu_cosched_2"]
    sg.name = "tpu_cosched_2"
    sg.accelerator_type = config_pb2.ACCELERATOR_TYPE_TPU
    sg.accelerator_variant = "v5litepod-16"
    sg.num_vms = 2
    sg.min_slices = 1
    sg.max_slices = 2
    sg.resources.cpu = 128
    sg.resources.memory_bytes = 128 * 1024 * 1024 * 1024
    sg.resources.disk_bytes = 1024 * 1024 * 1024 * 1024
    sg.slice_template.preemptible = True
    sg.slice_template.num_vms = 2
    sg.slice_template.accelerator_type = config_pb2.ACCELERATOR_TYPE_TPU
    sg.slice_template.accelerator_variant = "v5litepod-16"
    sg.slice_template.local.SetInParent()


@pytest.fixture(scope="module")
def cluster():
    """Boots a local cluster. Yields a TestCluster with IrisClient and RPC access."""
    config = load_config(DEFAULT_CONFIG)
    _add_coscheduling_group(config)
    config = make_local_config(config)
    with connect_cluster(config) as url:
        client = IrisClient.remote(url, workspace=IRIS_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        yield TestCluster(url=url, client=client, controller_client=controller_client)
        controller_client.close()


def _make_multi_worker_config(num_workers: int) -> config_pb2.IrisClusterConfig:
    """Build a local config with a single CPU scale group providing num_workers workers."""
    config = load_config(DEFAULT_CONFIG)
    config.scale_groups.clear()
    sg = config.scale_groups["local-cpu"]
    sg.name = "local-cpu"
    sg.accelerator_type = config_pb2.ACCELERATOR_TYPE_CPU
    sg.num_vms = 1
    sg.min_slices = num_workers
    sg.max_slices = num_workers
    sg.resources.cpu = 8
    sg.resources.memory_bytes = 16 * 1024**3
    sg.resources.disk_bytes = 50 * 1024**3
    sg.slice_template.local.SetInParent()
    return make_local_config(config)


@pytest.fixture(scope="module")
def multi_worker_cluster():
    """Boots a local cluster with 4 workers for distribution and concurrency tests.

    Waits for all workers to register before yielding, since the autoscaler
    scales up one slice per evaluation interval (~0.5s each).
    """
    num_workers = 4
    config = _make_multi_worker_config(num_workers)
    with connect_cluster(config) as url:
        client = IrisClient.remote(url, workspace=IRIS_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        tc = TestCluster(url=url, client=client, controller_client=controller_client)
        tc.wait_for_workers(num_workers, timeout=30)
        yield tc
        controller_client.close()


@pytest.fixture(autouse=True)
def _reset_chaos():
    yield
    reset_chaos()


@pytest.fixture
def chronos(monkeypatch):
    """Virtual time fixture - makes time.sleep() controllable for fast tests."""
    clock = VirtualClock()
    monkeypatch.setattr(time, "time", clock.time)
    monkeypatch.setattr(time, "monotonic", clock.time)
    monkeypatch.setattr(time, "sleep", clock.sleep)
    return clock


class _NoOpPage:
    """Stub page that provides no-op methods for all Playwright page operations."""

    def goto(self, url, **kwargs):
        pass

    def wait_for_load_state(self, state=None, **kwargs):
        pass

    def wait_for_function(self, expression, **kwargs):
        pass

    def click(self, selector, **kwargs):
        pass

    def wait_for_selector(self, selector, **kwargs):
        pass

    def locator(self, selector, **kwargs):
        return _NoOpLocator()

    def screenshot(self, **kwargs):
        pass

    def close(self):
        pass


class _NoOpLocator:
    """Stub locator that provides no-op methods."""

    @property
    def first(self):
        return self

    def is_visible(self, **kwargs):
        return False

    def text_content(self, **kwargs):
        return ""

    def count(self):
        return 0


class _NoOpBrowser:
    """Stub browser that provides no-op methods."""

    def new_page(self, **kwargs):
        return _NoOpPage()

    def close(self):
        pass


def _is_noop_page(page) -> bool:
    return isinstance(page, _NoOpPage)


def assert_visible(page, selector: str, *, timeout: int = 5000) -> None:
    """Assert a selector is visible. No-op when Playwright is unavailable."""
    if _is_noop_page(page):
        return
    assert page.locator(selector).first.is_visible(timeout=timeout)


def dashboard_click(page, selector: str) -> None:
    """Click a selector. No-op when Playwright is unavailable."""
    if _is_noop_page(page):
        return
    page.click(selector)


def dashboard_goto(page, url: str) -> None:
    """Navigate to URL. No-op when Playwright is unavailable."""
    if _is_noop_page(page):
        return
    page.goto(url)


@pytest.fixture(scope="module")
def browser():
    """Lazily launches a Chromium browser for Playwright-based tests.

    Returns a no-op stub if playwright is not installed or browser executable
    is missing (common in CI without 'playwright install'), allowing tests to
    run but skip screenshot operations.
    """
    try:
        import playwright.sync_api as pw

        with pw.sync_playwright() as p:
            b = p.chromium.launch()
            yield b
            b.close()
    except (ImportError, Exception):
        # Playwright not available or browser not installed - return stub
        yield _NoOpBrowser()


def wait_for_dashboard_ready(page) -> None:
    """Wait for Preact to render the dashboard root.

    Waits until the #root element has children and no longer shows "Loading...".
    Used by dashboard assertion tests across multiple test modules.
    """
    if _is_noop_page(page):
        return
    page.wait_for_function(
        "() => document.getElementById('root').children.length > 0"
        " && !document.getElementById('root').textContent.includes('Loading...')",
        timeout=30000,
    )


@pytest.fixture
def page(browser, cluster):
    """Per-test Playwright page pointed at the cluster dashboard.

    Returns a no-op stub if Playwright is unavailable, allowing tests to
    run but skip dashboard assertions.
    """
    pg = browser.new_page(viewport={"width": 1400, "height": 900})
    if not isinstance(browser, _NoOpBrowser):
        pg.goto(f"{cluster.url}/")
        pg.wait_for_load_state("domcontentloaded")
    yield pg
    pg.close()


@pytest.fixture
def screenshot(page, request, tmp_path):
    """Capture labeled screenshots. Set IRIS_SCREENSHOT_DIR to persist them.

    Returns a no-op callable if Playwright is unavailable, allowing tests to
    run but skip screenshot capture.
    """
    # Check if page is a no-op stub
    if isinstance(page, _NoOpPage):

        def noop_capture(label: str) -> Path:
            return tmp_path / f"{request.node.name}-{label}.png"

        return noop_capture

    output_dir = Path(
        os.environ.get(
            "IRIS_SCREENSHOT_DIR",
            str(tmp_path / "screenshots"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    def capture(label: str) -> Path:
        path = output_dir / f"{request.node.name}-{label}.png"
        page.screenshot(path=str(path), full_page=True)
        return path

    return capture
