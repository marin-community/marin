# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive smoke tests exercising Iris cluster features.

All tests share a single module-scoped cluster (smoke_cluster). Each test
submits its own jobs and is independently runnable. In local mode the cluster
has workers across CPU, TPU coscheduling, and multi-region scale groups.
"""

import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from iris.cli.cluster import _build_cluster_images, _pin_latest_images
from iris.client.client import IrisClient
from iris.cluster.config import IrisConfig, load_config, make_local_config
from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute, region_constraint
from iris.cluster.manager import connect_cluster
from iris.cluster.runtime.process import ProcessRuntime
from iris.cluster.types import (
    ReservationEntry,
    ResourceSpec,
    gpu_device,
)
from iris.cluster.worker.env_probe import DefaultEnvironmentProvider
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.managed_thread import ThreadContainer
from iris.rpc import cluster_pb2, config_pb2, logging_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.time_utils import Duration, ExponentialBackoff

from .conftest import (
    DEFAULT_CONFIG,
    IRIS_ROOT,
    ClusterCapabilities,
    IrisTestCluster,
    _NoOpPage,
    _add_coscheduling_group,
    assert_visible,
    dashboard_goto,
    discover_capabilities,
    wait_for_dashboard_ready,
)
from .helpers import TestJobs

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Smoke-test cluster configuration helpers
# ---------------------------------------------------------------------------


def _add_cpu_group(config: config_pb2.IrisClusterConfig, num_workers: int = 4) -> None:
    """CPU scale group with multiple workers for scheduling diversity and bin-packing."""
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


# Total local-mode workers:
# 4 (local-cpu) + 2 (cosched_2) + 4 (cosched_4) + 2 (region-a + region-b) = 12
SMOKE_WORKER_COUNT = 12


def _make_smoke_config() -> config_pb2.IrisClusterConfig:
    """Build a local config with CPU, TPU (coscheduling), and multi-region workers."""
    config = load_config(DEFAULT_CONFIG)
    config.scale_groups.clear()
    _add_cpu_group(config, num_workers=4)
    _add_coscheduling_group(config)
    _add_coscheduling_group_4vm(config)
    _add_multi_region_groups(config)
    return make_local_config(config)


def _cloud_smoke_cluster(config_path: str, mode: str, label_prefix: str | None = None):
    """Manage full cloud cluster lifecycle: stop old → build images → start → test → stop.

    Uses the Iris Python API directly instead of subprocess to avoid stdout
    parsing and to get proper tunnel management via the platform abstraction.
    """
    config = load_config(config_path)

    if label_prefix:
        config.platform.label_prefix = label_prefix
        # Isolate snapshot storage so this run doesn't restore stale state
        # from previous runs that shared the default bundle_prefix.
        config.storage.bundle_prefix = f"gs://marin-tmp-eu-west4/ttl=7d/iris/bundles/{label_prefix}"

    logger.info("Pinning and building cluster images...")
    _pin_latest_images(config)
    _build_cluster_images(config, verbose=False)

    iris_config = IrisConfig(config)
    platform = iris_config.platform()

    # Tear down any existing cluster for a clean slate
    logger.info("Stopping any existing cluster...")
    try:
        platform.stop_all(config)
    except Exception:
        logger.info("No existing cluster to stop (or stop failed), continuing")

    logger.info("Starting fresh controller...")
    address = platform.start_controller(config)
    logger.info("Controller started at %s", address)

    try:
        with platform.tunnel(address) as url:
            logger.info("Tunnel ready: %s", url)
            client = IrisClient.remote(url, workspace=IRIS_ROOT)
            controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
            tc = IrisTestCluster(
                url=url, client=client, controller_client=controller_client, job_timeout=600.0, is_cloud=True
            )
            tc.wait_for_workers(1, timeout=600)
            yield tc
            controller_client.close()
    finally:
        if mode != "keep":
            logger.info("Stopping cluster...")
            try:
                platform.stop_all(config)
            except Exception:
                logger.warning("Cluster stop failed during teardown", exc_info=True)


# ---------------------------------------------------------------------------
# Smoke-test fixtures (module-scoped so all smoke tests share one cluster)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_cluster(request):
    """Module-scoped cluster shared across all smoke tests.

    Local mode: boots in-process cluster with CPU + TPU + multi-region groups.
    Cloud mode: connects to existing cluster via --iris-controller-url or
    starts one via CLI using --iris-config.
    """
    controller_url = request.config.getoption("--iris-controller-url")
    config_path = request.config.getoption("--iris-config")
    mode = request.config.getoption("--iris-mode")
    label_prefix = request.config.getoption("--iris-label-prefix")

    is_cloud = mode != "local"
    timeout = 600.0 if is_cloud else 60.0

    if is_cloud and not controller_url:
        assert label_prefix, "--iris-label-prefix is required in cloud mode to avoid stomping on production clusters"

    if controller_url:
        client = IrisClient.remote(controller_url, workspace=IRIS_ROOT)
        controller_client = ControllerServiceClientSync(address=controller_url, timeout_ms=30000)
        tc = IrisTestCluster(
            url=controller_url,
            client=client,
            controller_client=controller_client,
            job_timeout=timeout,
            is_cloud=is_cloud,
        )
        if is_cloud:
            tc.wait_for_workers(1, timeout=timeout)
        yield tc
        controller_client.close()
        return

    if config_path and mode != "local":
        yield from _cloud_smoke_cluster(config_path, mode, label_prefix=label_prefix)
        return

    config = _make_smoke_config()
    with connect_cluster(config) as url:
        client = IrisClient.remote(url, workspace=IRIS_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        tc = IrisTestCluster(url=url, client=client, controller_client=controller_client)
        tc.wait_for_workers(SMOKE_WORKER_COUNT, timeout=60)
        yield tc
        controller_client.close()


@pytest.fixture(scope="module")
def smoke_page(smoke_cluster):
    """Module-scoped Playwright page for smoke dashboard tests."""
    try:
        import playwright.sync_api as pw

        with pw.sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1400, "height": 900})
            pg.goto(f"{smoke_cluster.url}/")
            pg.wait_for_load_state("domcontentloaded")
            yield pg
            pg.close()
            b.close()
    except (ImportError, Exception):
        yield _NoOpPage()


@pytest.fixture(scope="module")
def smoke_screenshot(smoke_page, tmp_path_factory):
    """Module-scoped screenshot capture for smoke dashboard tests."""
    if isinstance(smoke_page, _NoOpPage):

        def noop_capture(label: str) -> Path:
            return tmp_path_factory.mktemp("screenshots") / f"smoke-{label}.png"

        return noop_capture

    output_dir = Path(
        os.environ.get(
            "IRIS_SCREENSHOT_DIR",
            str(tmp_path_factory.mktemp("screenshots")),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    def capture(label: str) -> Path:
        path = output_dir / f"smoke-{label}.png"
        smoke_page.screenshot(path=str(path), full_page=True)
        return path

    return capture


@pytest.fixture(scope="module")
def verbose_job(smoke_cluster):
    """Shared verbose log job — submits once, used by log-related tests."""
    job = smoke_cluster.submit(TestJobs.log_verbose, "smoke-verbose")
    smoke_cluster.wait(job, timeout=smoke_cluster.job_timeout)
    return job


@pytest.fixture(scope="module")
def capabilities(smoke_cluster) -> ClusterCapabilities:
    """Discover cluster capabilities from live workers for topology-dependent tests."""
    return discover_capabilities(smoke_cluster.controller_client)


# ============================================================================
# Cluster readiness
# ============================================================================


def test_workers_ready(smoke_cluster, smoke_page, smoke_screenshot):
    """Verify workers are healthy, screenshot fleet tab."""
    request = cluster_pb2.Controller.ListWorkersRequest()
    response = smoke_cluster.controller_client.list_workers(request)
    healthy = [w for w in response.workers if w.healthy]
    assert len(healthy) > 0, "No healthy workers registered"

    dashboard_goto(smoke_page, f"{smoke_cluster.url}/fleet")
    wait_for_dashboard_ready(smoke_page)
    smoke_page.wait_for_function(
        "() => document.body.textContent.includes('Healthy')",
        timeout=10000,
    )
    smoke_screenshot("workers-ready")


# ============================================================================
# Dashboard tests
# ============================================================================


def test_dashboard_jobs_tab(smoke_cluster, smoke_page, smoke_screenshot):
    """Jobs tab shows diverse states."""
    quick = smoke_cluster.submit(TestJobs.quick, "smoke-simple")
    failed = smoke_cluster.submit(TestJobs.fail, "smoke-failed")
    running = smoke_cluster.submit(TestJobs.sleep, "smoke-running", 30)

    smoke_cluster.wait(quick, timeout=smoke_cluster.job_timeout)
    smoke_cluster.wait(failed, timeout=smoke_cluster.job_timeout)
    smoke_cluster.wait_for_state(running, cluster_pb2.JOB_STATE_RUNNING, timeout=smoke_cluster.job_timeout)

    dashboard_goto(smoke_page, f"{smoke_cluster.url}/")
    wait_for_dashboard_ready(smoke_page)
    for name in ["smoke-simple", "smoke-failed", "smoke-running"]:
        assert_visible(smoke_page, f"text={name}")
    smoke_screenshot("jobs-tab")

    smoke_cluster.kill(running)


def test_dashboard_job_detail(smoke_cluster, smoke_page, smoke_screenshot):
    """SUCCEEDED job detail page."""
    job = smoke_cluster.submit(TestJobs.quick, "smoke-detail")
    smoke_cluster.wait(job, timeout=smoke_cluster.job_timeout)

    dashboard_goto(smoke_page, f"{smoke_cluster.url}/job/{job.job_id.to_wire()}")
    wait_for_dashboard_ready(smoke_page)
    smoke_page.wait_for_function(
        "() => document.body.textContent.includes('Succeeded')",
        timeout=10000,
    )
    smoke_screenshot("job-detail")


def test_dashboard_task_logs(smoke_cluster, verbose_job, smoke_page, smoke_screenshot):
    """Task logs show lines and substring filter on the task detail page."""
    task_status = smoke_cluster.task_status(verbose_job)
    task_id = task_status.task_id
    job_id = verbose_job.job_id.to_wire()

    dashboard_goto(smoke_page, f"{smoke_cluster.url}/job/{job_id}/task/{task_id}")
    wait_for_dashboard_ready(smoke_page)

    smoke_page.wait_for_function(
        "() => document.body.textContent.includes('DONE: all lines emitted')",
        timeout=10000,
    )
    smoke_screenshot("task-logs-default")

    # "validation failed" only appears in ERROR lines
    smoke_page.fill("input[placeholder='Filter logs...']", "validation failed")
    smoke_page.wait_for_function(
        "() => document.body.textContent.includes('validation failed') && "
        "!document.body.textContent.includes('processing data batch')",
        timeout=5000,
    )
    smoke_screenshot("task-logs-filtered")


def test_dashboard_constraints(smoke_cluster, smoke_page, smoke_screenshot):
    """Constraint chips rendered on job detail."""
    constraints = [
        Constraint(key="region", op=ConstraintOp.EQ, value="local"),
        Constraint(key="env-tag", op=ConstraintOp.EXISTS),
        Constraint(key="device-variant", op=ConstraintOp.IN, values=("v5p-8", "v6e-4")),
    ]
    with smoke_cluster.launched_job(TestJobs.quick, "smoke-constraints", constraints=constraints) as job:
        time.sleep(3)

        dashboard_goto(smoke_page, f"{smoke_cluster.url}/job/{job.job_id.to_wire()}")
        wait_for_dashboard_ready(smoke_page)

        smoke_page.wait_for_function(
            "() => document.body.textContent.includes('Constraints')",
            timeout=5000,
        )
        assert_visible(smoke_page, "text=region")
        smoke_screenshot("constraints")


def test_dashboard_scheduling_diagnostic(smoke_cluster, smoke_page, smoke_screenshot):
    """Scheduling diagnostic shows pending reason for oversized job."""
    smoke_cluster.wait_for_workers(1, timeout=smoke_cluster.job_timeout)
    with smoke_cluster.launched_job(TestJobs.quick, "smoke-diag-cpu", cpu=999_999) as job:
        status = smoke_cluster.status(job)
        assert status.state == cluster_pb2.JOB_STATE_PENDING
        assert "cpu" in status.pending_reason.lower()

        dashboard_goto(smoke_page, f"{smoke_cluster.url}/job/{job.job_id.to_wire()}")
        wait_for_dashboard_ready(smoke_page)
        assert_visible(smoke_page, "text=Scheduling Diagnostic")
        smoke_screenshot("scheduling-diagnostic")


def test_dashboard_workers_tab(smoke_cluster, smoke_page, smoke_screenshot):
    """Workers tab shows healthy workers."""
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/fleet")
    wait_for_dashboard_ready(smoke_page)
    smoke_page.wait_for_function(
        "() => document.body.textContent.includes('Healthy')",
        timeout=10000,
    )
    smoke_screenshot("workers-tab")


def test_dashboard_worker_detail(smoke_cluster, smoke_page, smoke_screenshot):
    """Worker detail page shows info, task history, metric cards."""
    job = smoke_cluster.submit(TestJobs.quick, "smoke-worker-detail")
    smoke_cluster.wait(job, timeout=smoke_cluster.job_timeout)

    task_status = smoke_cluster.task_status(job)
    worker_id = task_status.worker_id
    assert worker_id

    dashboard_goto(smoke_page, f"{smoke_cluster.url}/worker/{worker_id}")
    wait_for_dashboard_ready(smoke_page)

    smoke_page.wait_for_function(
        f"() => document.body.textContent.includes('{worker_id}') && " "document.body.textContent.includes('Healthy')",
        timeout=10000,
    )
    smoke_screenshot("worker-detail")


def test_dashboard_autoscaler_tab(smoke_cluster, smoke_page, smoke_screenshot):
    """Autoscaler tab shows scale groups."""
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/autoscaler")
    wait_for_dashboard_ready(smoke_page)
    smoke_screenshot("autoscaler-tab")


def test_dashboard_status_tab(smoke_cluster, smoke_page, smoke_screenshot):
    """Status tab renders process info and log viewer."""
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/status")
    wait_for_dashboard_ready(smoke_page)
    # Status tab renders process info when available, or an error message.
    # Wait for either to appear to confirm the tab loaded and made the RPC call.
    smoke_page.wait_for_function(
        "() => document.body.textContent.includes('Process') || "
        "document.body.textContent.includes('GetProcessStatus')",
        timeout=10000,
    )
    smoke_screenshot("status-tab")


# ============================================================================
# Scheduling & endpoint verification
# ============================================================================


def test_small_job_skips_oversized(smoke_cluster):
    """Small job gets scheduled even when a large unschedulable job is queued."""
    with smoke_cluster.launched_job(TestJobs.quick, "smoke-big", cpu=10000) as big_job:
        small_job = smoke_cluster.submit(TestJobs.quick, "smoke-small", cpu=1)
        status = smoke_cluster.wait(small_job, timeout=smoke_cluster.job_timeout)
        assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
        big_status = smoke_cluster.status(big_job)
        assert big_status.state == cluster_pb2.JOB_STATE_PENDING


def test_endpoint_registration(smoke_cluster):
    """Endpoint registered from inside job via RPC."""
    prefix = f"smoke-ep-{uuid.uuid4().hex[:8]}"
    job = smoke_cluster.submit(TestJobs.register_endpoint, "smoke-endpoint", prefix)
    status = smoke_cluster.wait(job, timeout=smoke_cluster.job_timeout)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_port_allocation(smoke_cluster):
    """Port allocation job succeeded."""
    job = smoke_cluster.submit(TestJobs.validate_ports, "smoke-ports", ports=["http", "grpc"])
    status = smoke_cluster.wait(job, timeout=smoke_cluster.job_timeout)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_reservation_gates_scheduling(smoke_cluster):
    """Unsatisfiable reservation blocks scheduling; regular jobs proceed."""
    with smoke_cluster.launched_job(
        TestJobs.quick,
        "smoke-reserved",
        reservation=[
            ReservationEntry(resources=ResourceSpec(cpu=1, memory="1g", device=gpu_device("NONEXISTENT-GPU-9999", 99)))
        ],
    ) as reserved:
        reserved_status = smoke_cluster.status(reserved)
        assert reserved_status.state == cluster_pb2.JOB_STATE_PENDING

        regular = smoke_cluster.submit(TestJobs.quick, "smoke-regular-while-reserved")
        status = smoke_cluster.wait(regular, timeout=smoke_cluster.job_timeout)
        assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


# ============================================================================
# Log level verification
# ============================================================================


def test_log_levels_populated(smoke_cluster, verbose_job):
    """Task logs have level field (INFO, WARNING, ERROR)."""
    task_id = verbose_job.job_id.task(0).to_wire()

    deadline = time.monotonic() + smoke_cluster.job_timeout
    entries = []
    while time.monotonic() < deadline:
        request = cluster_pb2.Controller.GetTaskLogsRequest(id=task_id)
        response = smoke_cluster.controller_client.get_task_logs(request)
        entries = []
        for batch in response.task_logs:
            entries.extend(batch.logs)
        if any("info-marker" in e.data for e in entries):
            break
        time.sleep(0.5)

    markers_found = {}
    for entry in entries:
        for marker in ("info-marker", "warning-marker", "error-marker"):
            if marker in entry.data:
                markers_found[marker] = entry.level

    assert "info-marker" in markers_found, f"info-marker not found after 60s. Got {len(entries)} entries"
    assert markers_found["info-marker"] == logging_pb2.LOG_LEVEL_INFO
    assert markers_found.get("warning-marker") == logging_pb2.LOG_LEVEL_WARNING
    assert markers_found.get("error-marker") == logging_pb2.LOG_LEVEL_ERROR


def test_log_level_filter(smoke_cluster, verbose_job):
    """min_level=WARNING excludes INFO."""
    task_id = verbose_job.job_id.task(0).to_wire()

    request = cluster_pb2.Controller.GetTaskLogsRequest(id=task_id, min_level="WARNING")
    response = smoke_cluster.controller_client.get_task_logs(request)
    filtered = []
    for batch in response.task_logs:
        filtered.extend(batch.logs)

    filtered_data = [e.data for e in filtered]
    assert any("warning-marker" in d for d in filtered_data), f"warning-marker missing: {filtered_data}"
    assert any("error-marker" in d for d in filtered_data), f"error-marker missing: {filtered_data}"
    assert not any("info-marker" in d for d in filtered_data if d), "info-marker should be filtered out"


# ============================================================================
# Multi-region routing
# ============================================================================


def test_region_constrained_routing(smoke_cluster, capabilities):
    """Job with region constraint lands on correct worker."""
    if not capabilities.has_multi_region:
        pytest.skip("No multi-region workers in cluster")

    target_region = capabilities.regions[0]
    job = smoke_cluster.submit(
        TestJobs.noop,
        "smoke-region",
        constraints=[region_constraint([target_region])],
    )
    smoke_cluster.wait(job, timeout=smoke_cluster.job_timeout)

    task = smoke_cluster.task_status(job, task_index=0)
    assert task.worker_id

    request = cluster_pb2.Controller.ListWorkersRequest()
    response = smoke_cluster.controller_client.list_workers(request)
    worker = next(
        (w for w in response.workers if w.worker_id == task.worker_id or w.address == task.worker_id),
        None,
    )
    assert worker is not None
    region_attr = worker.metadata.attributes.get(WellKnownAttribute.REGION)
    if region_attr and region_attr.HasField("string_value"):
        assert region_attr.string_value == target_region, f"Expected {target_region}, got {region_attr.string_value}"


# ============================================================================
# Profiling
# ============================================================================


def test_profile_running_task(smoke_cluster):
    """Profile a running task, verify data returned."""
    if smoke_cluster.is_cloud:
        pytest.skip("py-spy races with short-lived containers in cloud mode")
    job = smoke_cluster.submit(TestJobs.busy_loop, name="smoke-profile")

    last_state = "unknown"

    def _is_running():
        nonlocal last_state
        task = smoke_cluster.task_status(job, task_index=0)
        last_state = task.state
        return last_state == cluster_pb2.TASK_STATE_RUNNING

    ExponentialBackoff(initial=0.1, maximum=2.0).wait_until_or_raise(
        _is_running,
        timeout=Duration.from_seconds(smoke_cluster.job_timeout),
        error_message=f"Task did not reach RUNNING within {smoke_cluster.job_timeout}s, last state: {last_state}",
    )
    task_id = smoke_cluster.task_status(job, task_index=0).task_id

    request = cluster_pb2.ProfileTaskRequest(
        target=task_id,
        duration_seconds=1,
        profile_type=cluster_pb2.ProfileType(cpu=cluster_pb2.CpuProfile(format=cluster_pb2.CpuProfile.FLAMEGRAPH)),
    )
    response = smoke_cluster.controller_client.profile_task(request, timeout_ms=3000)
    assert len(response.profile_data) > 0
    assert not response.error

    smoke_cluster.wait(job, timeout=smoke_cluster.job_timeout)


# ============================================================================
# Checkpoint / restore
# ============================================================================


def test_checkpoint_restore():
    """Controller restart resumes from checkpoint: completed jobs visible, cluster functional.

    Uses a dedicated cluster (not the shared smoke_cluster). A single platform
    instance restarts its controller between phases so the persistent DB dir
    (held by LocalController across stop/start) preserves checkpoint state.
    Phase 1 — run a job and write a checkpoint.
    Phase 2 — restart the controller and verify the job is still SUCCEEDED
              and the cluster can accept new work.
    """
    config = load_config(DEFAULT_CONFIG)
    config = make_local_config(config)

    platform = IrisConfig(config).platform()
    url = platform.start_controller(config)
    try:
        # Phase 1: complete a job, write checkpoint, restart controller.
        client = IrisClient.remote(url, workspace=IRIS_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        tc = IrisTestCluster(url=url, client=client, controller_client=controller_client)
        tc.wait_for_workers(1, timeout=30)

        job = tc.submit(TestJobs.quick, "pre-restart")
        tc.wait(job, timeout=30)
        saved_job_id = job.job_id.to_wire()

        ckpt = controller_client.begin_checkpoint(cluster_pb2.Controller.BeginCheckpointRequest())
        assert ckpt.checkpoint_path, "begin_checkpoint returned empty path"
        assert ckpt.job_count >= 1
        controller_client.close()

        url = platform.restart_controller(config)

        # Phase 2: verify restored state and submit new work.
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        tc = IrisTestCluster(
            url=url, client=IrisClient.remote(url, workspace=IRIS_ROOT), controller_client=controller_client
        )

        resp = controller_client.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id=saved_job_id))
        assert (
            resp.job.state == cluster_pb2.JOB_STATE_SUCCEEDED
        ), f"Pre-restart job has state {resp.job.state} after restore"

        tc.wait_for_workers(1, timeout=30)
        post_job = tc.submit(TestJobs.quick, "post-restart")
        status = tc.wait(post_job, timeout=30)
        assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

        controller_client.close()
    finally:
        platform.stop_controller(config)
        platform.shutdown()


# ============================================================================
# Stress test
# ============================================================================


@pytest.mark.timeout(1200)
def test_stress_200_tasks(smoke_cluster):
    """200 tasks exercises scheduler concurrency and bin-packing."""
    job = smoke_cluster.submit(
        TestJobs.quick,
        "smoke-stress-200",
        cpu=0,
        memory="100m",
        replicas=200,
    )
    status = smoke_cluster.wait(job, timeout=smoke_cluster.job_timeout * 2)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


# ============================================================================
# GPU metadata (local-only, creates standalone cluster with mocked nvidia-smi)
# ============================================================================

_NVIDIA_SMI_H100_8X = "\n".join(["NVIDIA H100 80GB HBM3, 81559"] * 8)


def _make_controller_only_config() -> config_pb2.IrisClusterConfig:
    """Build a local config with no auto-scaled workers."""
    config = load_config(DEFAULT_CONFIG)
    config.scale_groups.clear()
    sg = config.scale_groups["placeholder"]
    sg.name = "placeholder"
    sg.num_vms = 1
    sg.min_slices = 0
    sg.max_slices = 0
    sg.resources.cpu_millicores = 1000
    sg.resources.memory_bytes = 1 * 1024**3
    sg.resources.disk_bytes = 10 * 1024**3
    sg.resources.device_type = config_pb2.ACCELERATOR_TYPE_CPU
    sg.slice_template.local.SetInParent()
    return make_local_config(config)


def test_gpu_worker_metadata(tmp_path):
    """Mocked nvidia-smi registers GPU metadata on worker.

    Creates a standalone local cluster (not the shared smoke_cluster) with a
    manually started worker using mocked nvidia-smi output. This test only
    works in local mode — it doesn't run against real GPU/TPU clusters.
    """
    config = _make_controller_only_config()
    with connect_cluster(config) as url:
        original_run = subprocess.run
        with patch(
            "iris.cluster.worker.env_probe.subprocess.run",
            side_effect=lambda cmd, *a, **kw: (
                subprocess.CompletedProcess(args=cmd, returncode=0, stdout=_NVIDIA_SMI_H100_8X, stderr="")
                if isinstance(cmd, list) and cmd and cmd[0] == "nvidia-smi"
                else original_run(cmd, *a, **kw)
            ),
        ):
            env_provider = DefaultEnvironmentProvider()
            threads = ThreadContainer(name="test-gpu-worker")
            cache_dir = tmp_path / "cache"
            cache_dir.mkdir()

            worker_config = WorkerConfig(
                host="127.0.0.1",
                port=0,
                cache_dir=cache_dir,
                controller_address=url,
                worker_id=f"test-gpu-worker-{uuid.uuid4().hex[:8]}",
                poll_interval=Duration.from_seconds(0.1),
            )
            worker = Worker(
                worker_config,
                container_runtime=ProcessRuntime(),
                environment_provider=env_provider,
                threads=threads,
            )
            worker.start()

            try:
                controller_client = ControllerServiceClientSync(address=url, timeout_ms=10000)
                deadline = time.monotonic() + 15.0
                workers = []
                while time.monotonic() < deadline:
                    request = cluster_pb2.Controller.ListWorkersRequest()
                    response = controller_client.list_workers(request)
                    workers = [w for w in response.workers if w.healthy]
                    if workers:
                        break
                    time.sleep(0.5)

                assert workers, "Worker did not register within timeout"
                w = workers[0]
                meta = w.metadata
                assert meta.gpu_count == 8
                assert "H100" in meta.gpu_name
                assert meta.gpu_memory_mb == 81559
                assert meta.device.gpu.count == 8
                assert "H100" in meta.device.gpu.variant

                attrs = meta.attributes
                assert WellKnownAttribute.GPU_VARIANT in attrs
                assert "H100" in attrs[WellKnownAttribute.GPU_VARIANT].string_value
                assert WellKnownAttribute.GPU_COUNT in attrs
                assert attrs[WellKnownAttribute.GPU_COUNT].int_value == 8

                controller_client.close()
            finally:
                worker.stop()
                threads.stop(timeout=Duration.from_seconds(5.0))
