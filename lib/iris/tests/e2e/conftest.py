# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Core fixtures for Iris E2E tests.

Boots a local cluster via connect_cluster() + make_local_config() and provides
a IrisTestCluster dataclass that wraps the IrisClient and ControllerServiceClientSync
with convenience methods for job submission, waiting, and status queries.

The cluster fixture is function-scoped so each test gets a fresh cluster with no
stale worker state or chaos bleed. Chaos state is also reset per-test via an
autouse fixture.
"""

import fcntl
import logging
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from iris.chaos import reset_chaos
from iris.client.client import IrisClient, Job
from iris.cluster.config import connect_cluster, load_config, make_local_config
from iris.cluster.constraints import Constraint, WellKnownAttribute
from iris.cluster.types import (
    CoschedulingConfig,
    Entrypoint,
    EnvironmentSpec,
    ReservationEntry,
    ResourceSpec,
    is_job_finished,
)
from iris.rpc import config_pb2, logging_pb2
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.rpc.logging_connect import LogServiceClientSync
from rigging.timing import Duration

from .chronos import VirtualClock

MARIN_ROOT = Path(__file__).resolve().parents[4]  # repo root
IRIS_ROOT = MARIN_ROOT / "lib" / "iris"
DEFAULT_CONFIG = IRIS_ROOT / "examples" / "test.yaml"


@pytest.fixture(scope="session", autouse=True)
def _ensure_dashboard_built(tmp_path_factory):
    """Build dashboard assets once per session so dashboard tests have content to render.

    With pytest-xdist each worker gets its own session fixture, so all 8 workers
    race to run ``npm ci`` in the same directory — corrupting node_modules.
    A filelock serialises this so only one worker installs at a time.
    """
    dashboard_dir = IRIS_ROOT / "dashboard"
    if not (dashboard_dir / "package.json").exists():
        return
    if shutil.which("npm") is None:
        logging.getLogger(__name__).warning("npm not found, skipping dashboard build for tests")
        return

    lock_path = tmp_path_factory.getbasetemp().parent / "dashboard_build.lock"
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            subprocess.run(["npm", "ci"], cwd=dashboard_dir, check=True, capture_output=True)
            subprocess.run(["npm", "run", "build"], cwd=dashboard_dir, check=True, capture_output=True)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def pytest_addoption(parser):
    """Cloud mode CLI options for running smoke tests against remote clusters."""
    parser.addoption("--iris-controller-url", default=None, help="Connect to existing controller")


# Cloud mode needs much longer timeouts: GCE provisioning can take 20 minutes,
# and individual tests need time for remote job execution.
_LOCAL_E2E_TIMEOUT = 30  # local e2e tests boot clusters + run jobs
_CLOUD_FIXTURE_TIMEOUT = 1200  # 20 min for cluster provisioning
_CLOUD_TEST_TIMEOUT = 120  # 2 min per test


def pytest_collection_modifyitems(config, items):
    """Set appropriate timeouts for e2e tests.

    Local mode: 30s default (cluster boot + job execution).
    Cloud mode: 20 min for first smoke test (provisioning), 2 min for the rest.
    """
    is_cloud = config.getoption("--iris-controller-url") is not None

    import pytest

    first_smoke_test = True
    for item in items:
        if item.get_closest_marker("timeout"):
            continue
        if is_cloud:
            if "smoke_cluster" in getattr(item, "fixturenames", ()):
                if first_smoke_test:
                    item.add_marker(pytest.mark.timeout(_CLOUD_FIXTURE_TIMEOUT))
                    first_smoke_test = False
                else:
                    item.add_marker(pytest.mark.timeout(_CLOUD_TEST_TIMEOUT))
            else:
                item.add_marker(pytest.mark.timeout(_CLOUD_TEST_TIMEOUT))
        else:
            item.add_marker(pytest.mark.timeout(_LOCAL_E2E_TIMEOUT))


@dataclass
class IrisTestCluster:
    """Wraps a booted local cluster with convenience methods for E2E tests.

    Combines the chaos conftest's connect_cluster() bootstrap with E2ECluster-style
    convenience methods. Methods return protobuf types directly rather than dicts.
    """

    url: str
    client: IrisClient
    controller_client: ControllerServiceClientSync
    log_client: LogServiceClientSync
    job_timeout: float = 60.0
    is_cloud: bool = False

    # Cloud task pods run uv sync per pod, needing ~4GB. Local workers
    # share a pre-built venv so 1GB is fine.
    _CLOUD_MEMORY_DEFAULT = "4g"
    _LOCAL_MEMORY_DEFAULT = "1g"

    def submit(
        self,
        fn,
        name: str,
        *args,
        cpu: float = 1,
        memory: str | None = None,
        ports: list[str] | None = None,
        scheduling_timeout: Duration | None = None,
        replicas: int = 1,
        max_retries_failure: int = 0,
        max_retries_preemption: int = 1000,
        timeout: Duration | None = None,
        coscheduling: CoschedulingConfig | None = None,
        constraints: list[Constraint] | None = None,
        reservation: list[ReservationEntry] | None = None,
    ) -> Job:
        """Submit a callable as a job. Returns a Job handle."""
        if memory is None:
            memory = self._CLOUD_MEMORY_DEFAULT if self.is_cloud else self._LOCAL_MEMORY_DEFAULT
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
            reservation=reservation,
        )

    def status(self, job: Job) -> job_pb2.JobStatus:
        """Get the current JobStatus protobuf for a job."""
        job_id = job.job_id.to_wire()
        request = controller_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        response = self.controller_client.get_job_status(request)
        return response.job

    def task_status(self, job: Job, task_index: int = 0) -> job_pb2.TaskStatus:
        """Get the current TaskStatus protobuf for a specific task."""
        task_id = job.job_id.task(task_index).to_wire()
        request = controller_pb2.Controller.GetTaskStatusRequest(task_id=task_id)
        response = self.controller_client.get_task_status(request)
        return response.task

    def wait(
        self,
        job: Job,
        timeout: float = 60.0,
        chronos: VirtualClock | None = None,
        poll_interval: float = 0.5,
    ) -> job_pb2.JobStatus:
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
    ) -> job_pb2.JobStatus:
        """Poll until a job reaches a specific state (e.g. JOB_STATE_RUNNING)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.status(job)
            if status.state == state:
                return status
            time.sleep(poll_interval)
        raise TimeoutError(f"Job {job.job_id} did not reach state {state} in {timeout}s " f"(current: {status.state})")

    @contextmanager
    def launched_job(self, fn, name: str, *args, **kwargs):
        """Submit a job and guarantee it's killed on exit, even if the test fails.

        kill() is safe on already-finished jobs (controller silently returns),
        so this works for both pending and completed jobs.
        """
        job = self.submit(fn, name, *args, **kwargs)
        try:
            yield job
        finally:
            self.kill(job)

    def kill(self, job: Job) -> None:
        """Terminate a running job."""
        job_id = job.job_id.to_wire()
        request = controller_pb2.Controller.TerminateJobRequest(job_id=job_id)
        self.controller_client.terminate_job(request)

    def wait_for_workers(self, min_workers: int, timeout: float = 30.0) -> None:
        """Wait until at least min_workers healthy workers are registered."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            request = controller_pb2.Controller.ListWorkersRequest()
            response = self.controller_client.list_workers(request)
            healthy = [w for w in response.workers if w.healthy]
            if len(healthy) >= min_workers:
                return
            time.sleep(0.5)
        raise TimeoutError(f"Only {len(healthy)} of {min_workers} workers registered in {timeout}s")

    def get_task_logs(self, job: Job, task_index: int = 0) -> list[str]:
        """Fetch log lines for a task."""
        task_id = job.job_id.task(task_index).to_wire()
        request = logging_pb2.FetchLogsRequest(source=re.escape(task_id) + ":.*")
        response = self.log_client.fetch_logs(request)
        return [f"{e.source}: {e.data}" for e in response.entries]


@dataclass(frozen=True)
class ClusterCapabilities:
    """What the smoke cluster fleet provides, discovered from live workers."""

    regions: tuple[str, ...]
    device_types: frozenset[str]
    has_coscheduling: bool
    has_workers: bool

    @property
    def has_multi_region(self) -> bool:
        return len(self.regions) > 1

    @property
    def has_gpu(self) -> bool:
        return "gpu" in self.device_types

    @property
    def has_tpu(self) -> bool:
        return "tpu" in self.device_types


def discover_capabilities(controller_client: ControllerServiceClientSync) -> ClusterCapabilities:
    """Probe the live worker fleet to determine cluster capabilities."""
    request = controller_pb2.Controller.ListWorkersRequest()
    response = controller_client.list_workers(request)
    healthy = [w for w in response.workers if w.healthy]

    regions: set[str] = set()
    device_types: set[str] = set()
    tpu_names: set[str] = set()

    for w in healthy:
        attrs = w.metadata.attributes
        region_attr = attrs.get(WellKnownAttribute.REGION)
        if region_attr and region_attr.HasField("string_value"):
            regions.add(region_attr.string_value)
        device_attr = attrs.get(WellKnownAttribute.DEVICE_TYPE)
        if device_attr and device_attr.HasField("string_value"):
            device_types.add(device_attr.string_value)
        tpu_attr = attrs.get(WellKnownAttribute.TPU_NAME)
        if tpu_attr and tpu_attr.HasField("string_value"):
            tpu_names.add(tpu_attr.string_value)

    return ClusterCapabilities(
        regions=tuple(sorted(regions)),
        device_types=frozenset(device_types),
        has_coscheduling=len(tpu_names) > 0,
        has_workers=len(healthy) > 0,
    )


def _add_coscheduling_group(config: config_pb2.IrisClusterConfig) -> None:
    """Add a scale group with num_vms=2 so coscheduling tests can find a match.

    v5litepod-16 has vm_count=2, so the local platform creates 2 workers per slice
    sharing the same tpu-name. Setting num_vms=2 lets the demand router match
    coscheduled jobs with replicas=2.
    """
    sg = config.scale_groups["tpu_cosched_2"]
    sg.name = "tpu_cosched_2"
    sg.num_vms = 2
    sg.buffer_slices = 1
    sg.max_slices = 2
    sg.resources.cpu_millicores = 128000
    sg.resources.memory_bytes = 128 * 1024 * 1024 * 1024
    sg.resources.disk_bytes = 1024 * 1024 * 1024 * 1024
    sg.resources.device_type = config_pb2.ACCELERATOR_TYPE_TPU
    sg.resources.device_variant = "v5litepod-16"
    sg.resources.capacity_type = config_pb2.CAPACITY_TYPE_PREEMPTIBLE
    sg.slice_template.capacity_type = config_pb2.CAPACITY_TYPE_PREEMPTIBLE
    sg.slice_template.num_vms = 2
    sg.slice_template.accelerator_type = config_pb2.ACCELERATOR_TYPE_TPU
    sg.slice_template.accelerator_variant = "v5litepod-16"
    sg.slice_template.local.SetInParent()


@pytest.fixture
def cluster():
    """Boots a local cluster. Yields a IrisTestCluster with IrisClient and RPC access."""
    config = load_config(DEFAULT_CONFIG)
    _add_coscheduling_group(config)
    config = make_local_config(config)
    with connect_cluster(config) as url:
        client = IrisClient.remote(url, workspace=MARIN_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        log_client = LogServiceClientSync(address=url, timeout_ms=30000)
        yield IrisTestCluster(url=url, client=client, controller_client=controller_client, log_client=log_client)
        log_client.close()
        controller_client.close()


def _make_multi_worker_config(num_workers: int) -> config_pb2.IrisClusterConfig:
    """Build a local config with a single CPU scale group providing num_workers workers."""
    config = load_config(DEFAULT_CONFIG)
    config.scale_groups.clear()
    sg = config.scale_groups["local-cpu"]
    sg.name = "local-cpu"
    sg.num_vms = 1
    sg.buffer_slices = num_workers
    sg.max_slices = num_workers
    sg.resources.cpu_millicores = 8000
    sg.resources.memory_bytes = 16 * 1024**3
    sg.resources.disk_bytes = 50 * 1024**3
    sg.resources.device_type = config_pb2.ACCELERATOR_TYPE_CPU
    sg.resources.capacity_type = config_pb2.CAPACITY_TYPE_ON_DEMAND
    sg.slice_template.local.SetInParent()
    return make_local_config(config)


@pytest.fixture
def multi_worker_cluster():
    """Boots a local cluster with 4 workers for distribution and concurrency tests.

    Waits for all workers to register before yielding, since the autoscaler
    scales up one slice per evaluation interval (~0.5s each).
    """
    num_workers = 4
    config = _make_multi_worker_config(num_workers)
    with connect_cluster(config) as url:
        client = IrisClient.remote(url, workspace=MARIN_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        log_client = LogServiceClientSync(address=url, timeout_ms=30000)
        tc = IrisTestCluster(url=url, client=client, controller_client=controller_client, log_client=log_client)
        tc.wait_for_workers(num_workers, timeout=60)
        yield tc
        log_client.close()
        controller_client.close()


@pytest.fixture(autouse=True)
def _reset_chaos():
    yield
    reset_chaos()


logger = logging.getLogger(__name__)


def _open_fds() -> dict[int, Path]:
    """Snapshot all open file descriptors for the current process via /proc or lsof."""
    pid = os.getpid()
    proc_fd = Path(f"/proc/{pid}/fd")

    if proc_fd.is_dir():
        fds: dict[int, Path] = {}
        for entry in proc_fd.iterdir():
            try:
                fd = int(entry.name)
                target = entry.resolve()
                fds[fd] = target
            except (ValueError, OSError):
                continue
        return fds

    # macOS: fall back to lsof
    try:
        result = subprocess.run(
            ["lsof", "-p", str(pid), "-Fn"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}

    fds = {}
    current_fd: int | None = None
    for line in result.stdout.splitlines():
        if line.startswith("f") and line[1:].isdigit():
            current_fd = int(line[1:])
        elif line.startswith("n") and current_fd is not None:
            fds[current_fd] = Path(line[1:])
            current_fd = None
    return fds


@pytest.fixture(autouse=True)
def _detect_fd_leaks(request):
    """Log file descriptors that were opened but not closed during a test."""
    before = _open_fds()
    yield
    after = _open_fds()
    leaked = {fd: path for fd, path in after.items() if fd not in before}
    if leaked:
        lines = [f"  fd {fd} -> {path}" for fd, path in sorted(leaked.items())]
        logger.warning(
            "Test %s leaked %d file descriptor(s):\n%s",
            request.node.nodeid,
            len(leaked),
            "\n".join(lines),
        )


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

    def fill(self, selector, value, **kwargs):
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


def _is_noop_page(page) -> bool:
    return isinstance(page, _NoOpPage)


def assert_visible(page, selector: str, *, timeout: int = 10_000) -> None:
    """Assert a selector is visible. No-op when Playwright is unavailable."""
    if _is_noop_page(page):
        return
    from playwright.sync_api import expect

    expect(page.locator(selector).first).to_be_visible(timeout=timeout)


def dashboard_click(page, selector: str) -> None:
    """Click a selector. No-op when Playwright is unavailable."""
    if _is_noop_page(page):
        return
    page.click(selector)


def dashboard_goto(page, url: str) -> None:
    """Navigate to URL, converting paths to hash-based URLs for Vue Router.

    Vue Router uses createWebHashHistory, so /job/X must become /#/job/X.
    """
    if _is_noop_page(page):
        return
    from urllib.parse import urlparse

    parsed = urlparse(url)
    path = parsed.path
    if path and path != "/":
        base = f"{parsed.scheme}://{parsed.netloc}"
        url = f"{base}/#{path}"
    page.goto(url)


def wait_for_dashboard_ready(page) -> None:
    """Wait for the Vue 3 dashboard to mount and render children into #app."""
    if _is_noop_page(page):
        return
    page.wait_for_function(
        "() => {"
        "  const app = document.getElementById('app');"
        "  return app !== null && app.children.length > 0;"
        "}",
        timeout=30000,
    )
