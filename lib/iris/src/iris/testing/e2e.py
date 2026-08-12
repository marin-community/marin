# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Core drivers and helpers for Iris E2E tests.

Boots a local cluster via connect_cluster() + make_local_config() and provides
a IrisTestCluster dataclass that wraps the IrisClient and ControllerServiceClientSync
with convenience methods for job submission, waiting, and status queries.

Each local cluster driver boots fresh state for its caller.
"""

import fcntl
import logging
import os
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from finelog.rpc import logging_pb2
from finelog.rpc.logging_connect import LogServiceClientSync
from rigging.connect import proxy_path
from rigging.timing import Duration, ExponentialBackoff

from iris.client.client import IrisClient, Job
from iris.cluster.config import (
    IrisClusterConfig,
    LocalSliceConfig,
    ScaleGroupConfig,
    ScaleGroupResources,
    SliceConfig,
    load_config,
    make_local_config,
)
from iris.cluster.constraints import Constraint, WellKnownAttribute
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.lifecycle import connect_cluster
from iris.cluster.types import (
    AcceleratorType,
    CapacityType,
    is_job_finished,
)
from iris.resources.execution import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.resources.job import CoschedulingConfig
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync

MARIN_ROOT = Path(__file__).resolve().parents[5]  # repo root
IRIS_ROOT = MARIN_ROOT / "lib" / "iris"
DEFAULT_CONFIG = IRIS_ROOT / "config" / "ci-test.yaml"


def ensure_dashboard_built(tmp_path_factory) -> None:
    """Build dashboard assets once per session so dashboard tests have content to render.

    With pytest-xdist each worker calls this once per session, so all workers
    race to run ``npm ci`` in the same directory. A file lock serializes the
    install.
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
        max_task_failures: int = 0,
        timeout: Duration | None = None,
        coscheduling: CoschedulingConfig | None = None,
        constraints: list[Constraint] | None = None,
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
            max_task_failures=max_task_failures,
            timeout=timeout,
            coscheduling=coscheduling,
            constraints=constraints,
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
        poll_interval: float = 0.5,
    ) -> job_pb2.JobStatus:
        """Poll until a job reaches a terminal state. Returns the final JobStatus."""
        status = job_pb2.JobStatus()

        def job_is_finished() -> bool:
            nonlocal status
            status = self.status(job)
            return is_job_finished(status.state)

        ExponentialBackoff(initial=poll_interval, maximum=poll_interval, factor=1, jitter=0).wait_until_or_raise(
            job_is_finished,
            timeout=Duration.from_seconds(timeout),
            error_message=f"Job {job.job_id} did not complete in {timeout}s",
        )
        return status

    def wait_for_state(
        self,
        job: Job,
        state: int,
        timeout: float = 10.0,
        poll_interval: float = 0.1,
    ) -> job_pb2.JobStatus:
        """Poll until a job reaches a specific state (e.g. JOB_STATE_RUNNING)."""
        status = job_pb2.JobStatus()

        def job_reached_state() -> bool:
            nonlocal status
            status = self.status(job)
            return status.state == state

        ExponentialBackoff(initial=poll_interval, maximum=poll_interval, factor=1, jitter=0).wait_until_or_raise(
            job_reached_state,
            timeout=Duration.from_seconds(timeout),
            error_message=f"Job {job.job_id} did not reach state {state} in {timeout}s (current: {status.state})",
        )
        return status

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
        healthy = []

        def enough_workers_are_healthy() -> bool:
            nonlocal healthy
            request = controller_pb2.Controller.ListWorkersRequest()
            response = self.controller_client.list_workers(request)
            healthy = [w for w in response.workers if w.healthy]
            return len(healthy) >= min_workers

        ExponentialBackoff(initial=0.1, maximum=0.5).wait_until_or_raise(
            enough_workers_are_healthy,
            timeout=Duration.from_seconds(timeout),
            error_message=f"Only {len(healthy)} of {min_workers} workers registered in {timeout}s",
        )

    def get_task_logs(self, job: Job, task_index: int = 0) -> list[str]:
        """Fetch log lines for a task."""
        task_id = job.job_id.task(task_index).to_wire()
        request = logging_pb2.FetchLogsRequest(
            source=f"{task_id}:",
            match_scope=logging_pb2.MATCH_SCOPE_PREFIX,
        )
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


def _add_coscheduling_group(config: IrisClusterConfig) -> None:
    """Add a scale group with num_vms=2 so coscheduling tests can find a match.

    v5litepod-16 has vm_count=2, so the local platform creates 2 workers per slice
    sharing the same tpu-name. Setting num_vms=2 lets the demand router match
    coscheduled jobs with replicas=2.
    """
    config.scale_groups["tpu_cosched_2"] = ScaleGroupConfig(
        name="tpu_cosched_2",
        num_vms=2,
        buffer_slices=1,
        max_slices=2,
        resources=ScaleGroupResources(
            cpu_millicores=128000,
            memory_bytes=128 * 1024 * 1024 * 1024,
            disk_bytes=1024 * 1024 * 1024 * 1024,
            device_type=AcceleratorType.TPU,
            device_variant="v5litepod-16",
            capacity_type=CapacityType.PREEMPTIBLE,
        ),
        slice_template=SliceConfig(num_vms=2, local=LocalSliceConfig()),
    )


def local_test_cluster():
    """Boot a local cluster and yield its test client."""
    config = load_config(DEFAULT_CONFIG)
    _add_coscheduling_group(config)
    config = make_local_config(config)
    with connect_cluster(config) as url:
        client = IrisClient.remote(url, workspace=MARIN_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        log_client = LogServiceClientSync(
            address=f"{url.rstrip('/')}{proxy_path(LOG_SERVER_ENDPOINT_NAME)}",
            timeout_ms=30000,
        )
        yield IrisTestCluster(url=url, client=client, controller_client=controller_client, log_client=log_client)
        log_client.close()
        controller_client.close()


def _make_multi_worker_config(num_workers: int) -> IrisClusterConfig:
    """Build a local config with a single CPU scale group providing num_workers workers."""
    config = load_config(DEFAULT_CONFIG)
    config.scale_groups.clear()
    config.scale_groups["local-cpu"] = ScaleGroupConfig(
        name="local-cpu",
        num_vms=1,
        buffer_slices=num_workers,
        max_slices=num_workers,
        resources=ScaleGroupResources(
            cpu_millicores=8000,
            memory_bytes=16 * 1024**3,
            disk_bytes=50 * 1024**3,
            device_type=AcceleratorType.CPU,
            capacity_type=CapacityType.ON_DEMAND,
        ),
        slice_template=SliceConfig(local=LocalSliceConfig()),
    )
    return make_local_config(config)


def local_multi_worker_test_cluster():
    """Boot a local cluster with four workers.

    Waits for all workers to register before yielding, since the autoscaler
    scales up one slice per evaluation interval (~0.5s each).
    """
    num_workers = 4
    config = _make_multi_worker_config(num_workers)
    with connect_cluster(config) as url:
        client = IrisClient.remote(url, workspace=MARIN_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        log_client = LogServiceClientSync(
            address=f"{url.rstrip('/')}{proxy_path(LOG_SERVER_ENDPOINT_NAME)}",
            timeout_ms=30000,
        )
        tc = IrisTestCluster(url=url, client=client, controller_client=controller_client, log_client=log_client)
        tc.wait_for_workers(num_workers, timeout=60)
        yield tc
        log_client.close()
        controller_client.close()


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


def detect_fd_leaks(request):
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


def assert_visible(page, selector: str, *, timeout: int = 10_000) -> None:
    from playwright.sync_api import expect  # noqa: PLC0415  # pyrefly: ignore[missing-import]

    expect(page.locator(selector).first).to_be_visible(timeout=timeout)


def dashboard_goto(page, url: str) -> None:
    """Navigate to URL, converting paths to hash-based URLs for Vue Router.

    Vue Router uses createWebHashHistory, so /job/X must become /#/job/X.
    """
    parsed = urlparse(url)
    path = parsed.path
    if path and path != "/":
        base = f"{parsed.scheme}://{parsed.netloc}"
        url = f"{base}/#{path}"
    page.goto(url)


def wait_for_dashboard_ready(page) -> None:
    """Wait for the Vue 3 dashboard to mount and render children into #app."""
    page.wait_for_function(
        "() => {"
        "  const app = document.getElementById('app');"
        "  return app !== null && app.children.length > 0;"
        "}",
        timeout=30000,
    )
