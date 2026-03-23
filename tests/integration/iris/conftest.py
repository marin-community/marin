# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for Iris integration tests.

Accepts --controller-url to run against an existing controller (e.g. iris-dev).
Without it, boots a local in-process cluster for offline testing.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
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
from iris.rpc import cluster_pb2, config_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.time_utils import Duration

logger = logging.getLogger(__name__)

IRIS_ROOT = Path(__file__).resolve().parents[3] / "lib" / "iris"
DEFAULT_CONFIG = IRIS_ROOT / "examples" / "test.yaml"


def pytest_addoption(parser):
    """CLI options for running integration tests against a remote or local cluster."""
    parser.addoption("--controller-url", default=None, help="Connect to existing Iris controller")


# Remote clusters need longer timeouts for provisioning and job execution.
_REMOTE_FIXTURE_TIMEOUT = 1200
_REMOTE_TEST_TIMEOUT = 120


def pytest_collection_modifyitems(config, items):
    """Set appropriate timeouts for remote-cluster tests."""
    if config.getoption("--controller-url") is None:
        return
    first = True
    for item in items:
        if item.get_closest_marker("timeout"):
            continue
        if first:
            item.add_marker(pytest.mark.timeout(_REMOTE_FIXTURE_TIMEOUT))
            first = False
        else:
            item.add_marker(pytest.mark.timeout(_REMOTE_TEST_TIMEOUT))


@dataclass(frozen=True)
class ClusterCapabilities:
    """What the live worker fleet provides."""

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
    request = cluster_pb2.Controller.ListWorkersRequest()
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


@dataclass
class IrisIntegrationCluster:
    """Wraps a cluster connection with convenience methods for integration tests.

    Unlike the E2E IrisTestCluster, this is designed for tests that exercise
    job submission and lifecycle without dashboard/screenshot concerns.
    """

    url: str
    client: IrisClient
    controller_client: ControllerServiceClientSync
    job_timeout: float = 60.0
    is_remote: bool = False

    _REMOTE_MEMORY_DEFAULT = "4g"
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
        max_retries_preemption: int = 100,
        timeout: Duration | None = None,
        coscheduling: CoschedulingConfig | None = None,
        constraints: list[Constraint] | None = None,
        reservation: list[ReservationEntry] | None = None,
    ) -> Job:
        """Submit a callable as a job."""
        if memory is None:
            memory = self._REMOTE_MEMORY_DEFAULT if self.is_remote else self._LOCAL_MEMORY_DEFAULT
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

    def status(self, job: Job) -> cluster_pb2.JobStatus:
        job_id = job.job_id.to_wire()
        request = cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id)
        response = self.controller_client.get_job_status(request)
        return response.job

    def task_status(self, job: Job, task_index: int = 0) -> cluster_pb2.TaskStatus:
        task_id = job.job_id.task(task_index).to_wire()
        request = cluster_pb2.Controller.GetTaskStatusRequest(task_id=task_id)
        response = self.controller_client.get_task_status(request)
        return response.task

    def wait(self, job: Job, timeout: float = 60.0, poll_interval: float = 0.5) -> cluster_pb2.JobStatus:
        """Poll until a job reaches a terminal state."""
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
        deadline = time.monotonic() + timeout
        status = self.status(job)
        while time.monotonic() < deadline:
            status = self.status(job)
            if status.state == state:
                return status
            time.sleep(poll_interval)
        raise TimeoutError(f"Job {job.job_id} did not reach state {state} in {timeout}s (current: {status.state})")

    @contextmanager
    def launched_job(self, fn, name: str, *args, **kwargs):
        """Submit a job and guarantee it's killed on exit."""
        job = self.submit(fn, name, *args, **kwargs)
        try:
            yield job
        finally:
            self.kill(job)

    def kill(self, job: Job) -> None:
        job_id = job.job_id.to_wire()
        request = cluster_pb2.Controller.TerminateJobRequest(job_id=job_id)
        self.controller_client.terminate_job(request)

    def wait_for_workers(self, min_workers: int, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        healthy = []
        while time.monotonic() < deadline:
            request = cluster_pb2.Controller.ListWorkersRequest()
            response = self.controller_client.list_workers(request)
            healthy = [w for w in response.workers if w.healthy]
            if len(healthy) >= min_workers:
                return
            time.sleep(0.5)
        raise TimeoutError(f"Only {len(healthy)} of {min_workers} workers registered in {timeout}s")

    def get_task_logs(self, job: Job, task_index: int = 0) -> list[str]:
        task_id = job.job_id.task(task_index).to_wire()
        request = cluster_pb2.Controller.GetTaskLogsRequest(id=task_id)
        response = self.controller_client.get_task_logs(request)
        lines = []
        for batch in response.task_logs:
            for entry in batch.logs:
                lines.append(f"{entry.source}: {entry.data}")
        return lines


def _add_coscheduling_group(config: config_pb2.IrisClusterConfig) -> None:
    """Add a TPU coscheduling group with num_vms=2 for coscheduling tests."""
    sg = config.scale_groups["tpu_cosched_2"]
    sg.name = "tpu_cosched_2"
    sg.num_vms = 2
    sg.min_slices = 1
    sg.max_slices = 2
    sg.resources.cpu_millicores = 128000
    sg.resources.memory_bytes = 128 * 1024 * 1024 * 1024
    sg.resources.disk_bytes = 1024 * 1024 * 1024 * 1024
    sg.resources.device_type = config_pb2.ACCELERATOR_TYPE_TPU
    sg.resources.device_variant = "v5litepod-16"
    sg.resources.preemptible = True
    sg.slice_template.preemptible = True
    sg.slice_template.num_vms = 2
    sg.slice_template.accelerator_type = config_pb2.ACCELERATOR_TYPE_TPU
    sg.slice_template.accelerator_variant = "v5litepod-16"
    sg.slice_template.local.SetInParent()


def _add_cpu_group(config: config_pb2.IrisClusterConfig, num_workers: int = 4) -> None:
    """CPU scale group with multiple workers."""
    sg = config.scale_groups["local-cpu"]
    sg.name = "local-cpu"
    sg.num_vms = 1
    sg.min_slices = num_workers
    sg.max_slices = num_workers
    sg.resources.cpu_millicores = 8000
    sg.resources.memory_bytes = 16 * 1024**3
    sg.resources.disk_bytes = 50 * 1024**3
    sg.resources.device_type = config_pb2.ACCELERATOR_TYPE_CPU
    sg.slice_template.local.SetInParent()


def _add_coscheduling_group_4vm(config: config_pb2.IrisClusterConfig) -> None:
    """4-VM TPU coscheduling group for reservation and large-job tests."""
    sg = config.scale_groups["tpu_cosched_4"]
    sg.name = "tpu_cosched_4"
    sg.num_vms = 4
    sg.min_slices = 1
    sg.max_slices = 1
    sg.resources.cpu_millicores = 128000
    sg.resources.memory_bytes = 128 * 1024**3
    sg.resources.disk_bytes = 1024 * 1024**3
    sg.resources.device_type = config_pb2.ACCELERATOR_TYPE_TPU
    sg.resources.device_variant = "v5litepod-32"
    sg.resources.preemptible = True
    sg.slice_template.preemptible = True
    sg.slice_template.num_vms = 4
    sg.slice_template.accelerator_type = config_pb2.ACCELERATOR_TYPE_TPU
    sg.slice_template.accelerator_variant = "v5litepod-32"
    sg.slice_template.local.SetInParent()


def _add_multi_region_groups(config: config_pb2.IrisClusterConfig) -> None:
    """Two CPU scale groups in different regions for constraint routing tests."""
    for name, region in [("cpu-region-a", "us-central1"), ("cpu-region-b", "europe-west4")]:
        sg = config.scale_groups[name]
        sg.name = name
        sg.num_vms = 1
        sg.min_slices = 1
        sg.max_slices = 2
        sg.resources.cpu_millicores = 8000
        sg.resources.memory_bytes = 16 * 1024**3
        sg.resources.disk_bytes = 50 * 1024**3
        sg.resources.device_type = config_pb2.ACCELERATOR_TYPE_CPU
        sg.slice_template.local.SetInParent()
        sg.worker.attributes[WellKnownAttribute.REGION] = region


# Total local-mode workers: 2 (cpu) + 2 (cosched_2) + 4 (cosched_4) + 2 (region) = 10
INTEGRATION_WORKER_COUNT = 10


def _make_integration_config() -> config_pb2.IrisClusterConfig:
    """Build a local config with CPU, TPU (coscheduling), and multi-region workers."""
    config = load_config(DEFAULT_CONFIG)
    config.scale_groups.clear()
    _add_cpu_group(config, num_workers=2)
    _add_coscheduling_group(config)
    _add_coscheduling_group_4vm(config)
    _add_multi_region_groups(config)
    return make_local_config(config)


@pytest.fixture(scope="module")
def integration_cluster(request):
    """Module-scoped cluster for integration tests.

    Remote mode: connect to existing cluster via --controller-url.
    Local mode: boot in-process cluster with CPU + TPU + multi-region groups.
    """
    controller_url = request.config.getoption("--controller-url")

    if controller_url:
        client = IrisClient.remote(controller_url, workspace=IRIS_ROOT)
        controller_client = ControllerServiceClientSync(address=controller_url, timeout_ms=30000)
        tc = IrisIntegrationCluster(
            url=controller_url,
            client=client,
            controller_client=controller_client,
            job_timeout=600.0,
            is_remote=True,
        )
        workers = controller_client.list_workers(cluster_pb2.Controller.ListWorkersRequest()).workers
        if workers:
            tc.wait_for_workers(1, timeout=600)
        yield tc
        controller_client.close()
        return

    config = _make_integration_config()
    with connect_cluster(config) as url:
        client = IrisClient.remote(url, workspace=IRIS_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        tc = IrisIntegrationCluster(url=url, client=client, controller_client=controller_client)
        tc.wait_for_workers(INTEGRATION_WORKER_COUNT, timeout=60)
        yield tc
        controller_client.close()


@pytest.fixture(scope="module")
def capabilities(integration_cluster) -> ClusterCapabilities:
    """Discover cluster capabilities from live workers."""
    return discover_capabilities(integration_cluster.controller_client)
