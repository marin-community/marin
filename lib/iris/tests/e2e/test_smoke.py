# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive smoke tests exercising Iris cluster features.

All tests share a single module-scoped cluster (smoke_cluster). Each test
submits its own jobs and is independently runnable. In local mode the cluster
has workers across CPU, TPU coscheduling, and multi-region scale groups.
"""

import logging
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from connectrpc.errors import ConnectError
from iris.client.client import IrisClient
from iris.cluster.config import connect_cluster, load_config, make_local_config
from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute, region_constraint
from iris.cluster.runtime.process import ProcessRuntime
from iris.cluster.types import (
    Entrypoint,
    ReservationEntry,
    ResourceSpec,
    gpu_device,
)
from iris.cluster.worker.env_probe import DefaultEnvironmentProvider
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.managed_thread import ThreadContainer
from iris.rpc import config_pb2, logging_pb2
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.rpc.logging_connect import LogServiceClientSync
from rigging.timing import Duration, ExponentialBackoff

from .conftest import (
    DEFAULT_CONFIG,
    MARIN_ROOT,
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
    sg.buffer_slices = num_workers
    sg.max_slices = num_workers
    sg.resources.cpu_millicores = 8000
    sg.resources.memory_bytes = 16 * 1024**3
    sg.resources.disk_bytes = 50 * 1024**3
    sg.resources.device_type = config_pb2.ACCELERATOR_TYPE_CPU
    sg.resources.capacity_type = config_pb2.CAPACITY_TYPE_ON_DEMAND
    sg.slice_template.local.SetInParent()


def _add_coscheduling_group_4vm(config: config_pb2.IrisClusterConfig) -> None:
    """4-VM TPU coscheduling group for reservation and large-job tests."""
    sg = config.scale_groups["tpu_cosched_4"]
    sg.name = "tpu_cosched_4"
    sg.num_vms = 4
    sg.buffer_slices = 1
    sg.max_slices = 1
    sg.resources.cpu_millicores = 128000
    sg.resources.memory_bytes = 128 * 1024**3
    sg.resources.disk_bytes = 1024 * 1024**3
    sg.resources.device_type = config_pb2.ACCELERATOR_TYPE_TPU
    sg.resources.device_variant = "v5litepod-32"
    sg.resources.capacity_type = config_pb2.CAPACITY_TYPE_PREEMPTIBLE
    sg.slice_template.capacity_type = config_pb2.CAPACITY_TYPE_PREEMPTIBLE
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
        sg.buffer_slices = 1
        sg.max_slices = 2
        sg.resources.cpu_millicores = 8000
        sg.resources.memory_bytes = 16 * 1024**3
        sg.resources.disk_bytes = 50 * 1024**3
        sg.resources.device_type = config_pb2.ACCELERATOR_TYPE_CPU
        sg.resources.capacity_type = config_pb2.CAPACITY_TYPE_ON_DEMAND
        sg.slice_template.local.SetInParent()
        sg.worker.attributes[WellKnownAttribute.REGION] = region


# Total local-mode workers:
# 2 (local-cpu) + 2 (cosched_2) + 4 (cosched_4) + 2 (region-a + region-b) = 10
SMOKE_WORKER_COUNT = 10


def _make_smoke_config() -> config_pb2.IrisClusterConfig:
    """Build a local config with CPU, TPU (coscheduling), and multi-region workers."""
    config = load_config(DEFAULT_CONFIG)
    config.scale_groups.clear()
    _add_cpu_group(config, num_workers=2)
    _add_coscheduling_group(config)
    _add_coscheduling_group_4vm(config)
    _add_multi_region_groups(config)
    return make_local_config(config)


# ---------------------------------------------------------------------------
# Smoke-test fixtures (module-scoped so all smoke tests share one cluster)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_cluster(request):
    """Module-scoped cluster shared across all smoke tests.

    Cloud mode: connect to existing cluster via --iris-controller-url.
    Local mode: boot in-process cluster with CPU + TPU + multi-region groups.
    """
    controller_url = request.config.getoption("--iris-controller-url")

    if controller_url:
        client = IrisClient.remote(controller_url, workspace=MARIN_ROOT)
        controller_client = ControllerServiceClientSync(address=controller_url, timeout_ms=30000)
        log_client = LogServiceClientSync(address=controller_url, timeout_ms=30000)
        tc = IrisTestCluster(
            url=controller_url,
            client=client,
            controller_client=controller_client,
            log_client=log_client,
            job_timeout=600.0,
            is_cloud=True,
        )
        # Only wait for workers on platforms with persistent worker VMs (GCP).
        # kubernetes_provider (CoreWeave) runs tasks as ephemeral pods.
        workers = controller_client.list_workers(controller_pb2.Controller.ListWorkersRequest()).workers
        if workers:
            tc.wait_for_workers(1, timeout=600)
        yield tc
        controller_client.close()
        return

    config = _make_smoke_config()
    with connect_cluster(config) as url:
        client = IrisClient.remote(url, workspace=MARIN_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        log_client = LogServiceClientSync(address=url, timeout_ms=30000)
        tc = IrisTestCluster(url=url, client=client, controller_client=controller_client, log_client=log_client)
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

        def noop_capture(label: str, description: str = "") -> Path:
            return tmp_path_factory.mktemp("screenshots") / f"smoke-{label}.png"

        return noop_capture

    output_dir = Path(
        os.environ.get(
            "IRIS_SCREENSHOT_DIR",
            str(tmp_path_factory.mktemp("screenshots")),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    def capture(label: str, description: str = "") -> Path:
        path = output_dir / f"smoke-{label}.png"
        smoke_page.screenshot(path=str(path), full_page=True)
        if description:
            desc_path = output_dir / f"smoke-{label}.txt"
            desc_path.write_text(description)
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
# ============================================================================
# Dashboard tests
# ============================================================================


def test_dashboard_jobs_tab(smoke_cluster, smoke_page, smoke_screenshot):
    """Jobs tab shows diverse states."""
    quick = smoke_cluster.submit(TestJobs.quick, "smoke-simple")
    failed = smoke_cluster.submit(TestJobs.fail, "smoke-failed")
    running = smoke_cluster.submit(TestJobs.sleep, "smoke-running", 300)

    smoke_cluster.wait(quick, timeout=smoke_cluster.job_timeout)
    smoke_cluster.wait(failed, timeout=smoke_cluster.job_timeout)
    smoke_cluster.wait_for_state(running, job_pb2.JOB_STATE_RUNNING, timeout=smoke_cluster.job_timeout)

    dashboard_goto(smoke_page, f"{smoke_cluster.url}/")
    wait_for_dashboard_ready(smoke_page)
    for name in ["smoke-simple", "smoke-failed", "smoke-running"]:
        assert_visible(smoke_page, f"text={name}")
    smoke_screenshot(
        "jobs-tab", "Jobs tab listing smoke-simple (succeeded), smoke-failed (failed), and smoke-running (running)"
    )

    smoke_cluster.kill(running)


def _parent_with_two_children():
    """Parent callable that submits two child jobs and waits for both."""
    from iris.client.client import iris_ctx
    from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec

    ctx = iris_ctx()
    res = ResourceSpec(cpu=1, memory="1g")
    env = EnvironmentSpec()

    job_a = ctx.client.submit(
        Entrypoint.from_command("sh", "-c", "echo CHILD_A"),
        "child-a",
        res,
        environment=env,
    )
    job_b = ctx.client.submit(
        Entrypoint.from_command("sh", "-c", "echo CHILD_B"),
        "child-b",
        res,
        environment=env,
    )
    job_a.wait(timeout=30, raise_on_failure=True)
    job_b.wait(timeout=30, raise_on_failure=True)


def test_dashboard_job_expand(smoke_cluster, smoke_page, smoke_screenshot):
    """Expanding a parent job in the jobs tab shows its children."""
    parent = smoke_cluster.submit(_parent_with_two_children, "smoke-expand-parent")
    smoke_cluster.wait(parent, timeout=smoke_cluster.job_timeout)

    dashboard_goto(smoke_page, f"{smoke_cluster.url}/")
    wait_for_dashboard_ready(smoke_page)
    assert_visible(smoke_page, "text=smoke-expand-parent")

    # The parent should have an expand arrow (▶)
    row = smoke_page.locator("tr", has_text="smoke-expand-parent")
    expand_btn = row.get_by_role("button", name="▶")
    expand_btn.click()

    # After clicking, children should appear (wait for the child names to render)
    smoke_page.wait_for_function(
        "() => document.body.textContent.includes('child-a') && " "document.body.textContent.includes('child-b')",
        timeout=10000,
    )

    # Verify the arrow changed to ▼
    row.get_by_role("button", name="▼").wait_for(timeout=5000)

    smoke_screenshot("job-expand", "Jobs tab with expanded parent showing child-a and child-b indented beneath")


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
    smoke_screenshot(
        "job-detail", "Job detail page for succeeded job with state badge, task table, and job-level log viewer"
    )


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
    smoke_screenshot(
        "task-logs-default",
        "Task detail page with a log viewer panel displaying log output lines. "
        "Should have structural elements like a status card and resource info.",
    )

    # "validation failed" only appears in ERROR lines
    smoke_page.fill("input[placeholder='Filter logs...']", "validation failed")
    smoke_page.wait_for_function(
        "() => document.body.textContent.includes('validation failed') && "
        "!document.body.textContent.includes('processing data batch')",
        timeout=5000,
    )
    smoke_screenshot(
        "task-logs-filtered",
        "Task detail page with log filter input populated and filtered log lines visible in the log viewer.",
    )


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
        smoke_screenshot(
            "constraints", "Job detail page showing constraint chips for region, env-tag, and device-variant"
        )


def test_dashboard_scheduling_diagnostic(smoke_cluster, smoke_page, smoke_screenshot, capabilities):
    """Scheduling diagnostic shows pending reason for oversized job."""
    if not capabilities.has_workers:
        pytest.skip("No persistent workers")
    smoke_cluster.wait_for_workers(1, timeout=smoke_cluster.job_timeout)
    with smoke_cluster.launched_job(TestJobs.quick, "smoke-diag-cpu", cpu=999_999) as job:
        # Poll until the scheduler has evaluated the job and produced a
        # CPU-specific pending reason (avoids racing the scheduler loop).
        deadline = time.monotonic() + smoke_cluster.job_timeout
        while time.monotonic() < deadline:
            status = smoke_cluster.status(job)
            if "cpu" in status.pending_reason.lower():
                break
            time.sleep(0.2)
        assert status.state == job_pb2.JOB_STATE_PENDING
        assert "cpu" in status.pending_reason.lower()

        dashboard_goto(smoke_page, f"{smoke_cluster.url}/job/{job.job_id.to_wire()}")
        wait_for_dashboard_ready(smoke_page)
        assert_visible(smoke_page, "text=Scheduling Diagnostic")
        smoke_screenshot(
            "scheduling-diagnostic", "Job detail page with yellow scheduling diagnostic banner explaining CPU capacity"
        )


def test_dashboard_workers_tab(smoke_cluster, smoke_page, smoke_screenshot, capabilities):
    """Workers tab shows healthy workers."""
    if not capabilities.has_workers:
        pytest.skip("No persistent workers")
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/fleet")
    wait_for_dashboard_ready(smoke_page)
    smoke_page.wait_for_function(
        "() => document.body.textContent.includes('Healthy')",
        timeout=10000,
    )
    smoke_screenshot("workers-tab", "Fleet tab showing worker list with health status badges")


def test_dashboard_worker_detail(smoke_cluster, smoke_page, smoke_screenshot, capabilities):
    """Worker detail page shows info, task history, metric cards."""
    if not capabilities.has_workers:
        pytest.skip("No persistent workers")
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
    smoke_screenshot(
        "worker-detail", "Worker detail page with identity info, health badge, metric sparklines, and task history"
    )


def test_dashboard_autoscaler_tab(smoke_cluster, smoke_page, smoke_screenshot):
    """Autoscaler tab shows scale groups."""
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/autoscaler")
    wait_for_dashboard_ready(smoke_page)
    smoke_page.wait_for_function(
        "() => document.body.textContent.includes('Autoscaler') || "
        "document.body.textContent.includes('Scale Group') || "
        "document.body.textContent.includes('scale group')",
        timeout=10000,
    )
    smoke_screenshot("autoscaler-tab", "Autoscaler tab showing scale group configuration")


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
    smoke_screenshot("status-tab", "Status tab showing controller process info or GetProcessStatus error")


def test_dashboard_job_detail_with_logs(smoke_cluster, verbose_job, smoke_page, smoke_screenshot):
    """Job detail page shows combined log viewer for all tasks."""
    job_id = verbose_job.job_id.to_wire()
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/job/{job_id}")
    wait_for_dashboard_ready(smoke_page)
    smoke_page.wait_for_function(
        "() => document.body.textContent.includes('Job Logs')",
        timeout=10000,
    )
    smoke_screenshot(
        "job-detail-logs",
        "Job detail page showing task table and combined job-level log viewer with log lines",
    )


def test_dashboard_scheduler_tab(smoke_cluster, smoke_page, smoke_screenshot):
    """Scheduler tab shows pending queue, user budgets, and running tasks."""
    running = smoke_cluster.submit(TestJobs.sleep, "smoke-sched-running", 300)
    smoke_cluster.wait_for_state(running, job_pb2.JOB_STATE_RUNNING, timeout=smoke_cluster.job_timeout)

    dashboard_goto(smoke_page, f"{smoke_cluster.url}/scheduler")
    wait_for_dashboard_ready(smoke_page)
    smoke_page.wait_for_function(
        "() => document.body.textContent.includes('Pending Queue') || "
        "document.body.textContent.includes('User Budgets')",
        timeout=10000,
    )
    assert_visible(smoke_page, "text=Pending Queue")
    assert_visible(smoke_page, "text=User Budgets")
    assert_visible(smoke_page, "text=Running Tasks")
    smoke_screenshot(
        "scheduler-tab",
        "Scheduler tab showing pending queue, user budgets, and running tasks",
    )

    smoke_cluster.kill(running)


# ============================================================================
# Scheduling & endpoint verification
# ============================================================================


def test_endpoint_registration(smoke_cluster):
    """Endpoint registered from inside job via RPC."""
    prefix = f"smoke-ep-{uuid.uuid4().hex[:8]}"
    job = smoke_cluster.submit(TestJobs.register_endpoint, "smoke-endpoint", prefix)
    status = smoke_cluster.wait(job, timeout=smoke_cluster.job_timeout)
    assert status.state == job_pb2.JOB_STATE_SUCCEEDED


def test_port_allocation(smoke_cluster, capabilities):
    """Port allocation job succeeded."""
    if not capabilities.has_workers:
        pytest.skip("kubernetes_provider does not inject port allocations into task pods yet")
    job = smoke_cluster.submit(TestJobs.validate_ports, "smoke-ports", ports=["http", "grpc"])
    status = smoke_cluster.wait(job, timeout=smoke_cluster.job_timeout)
    assert status.state == job_pb2.JOB_STATE_SUCCEEDED


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
        assert reserved_status.state == job_pb2.JOB_STATE_PENDING

        regular = smoke_cluster.submit(TestJobs.quick, "smoke-regular-while-reserved")
        status = smoke_cluster.wait(regular, timeout=smoke_cluster.job_timeout)
        assert status.state == job_pb2.JOB_STATE_SUCCEEDED


def test_cancel_job_releases_resources(smoke_cluster):
    """Cancelling a running job decommits worker resources so new jobs can schedule.

    Submits a resource-heavy job, cancels it, then verifies a second job with
    the same resource requirements succeeds — proving the worker's committed
    resources were fully released by cancel_job().

    Regression test for #3553.
    """
    # Use most of a single worker's CPU so the followup job can't schedule
    # until the heavy job is cancelled. Local workers have 1000 cores, cloud
    # TPU VMs have 128 — pick a value that works in both modes.
    heavy_cpu = 8 if smoke_cluster.is_cloud else 900

    job = smoke_cluster.submit(TestJobs.sleep, "smoke-cancel-heavy", 30, cpu=heavy_cpu)
    smoke_cluster.wait_for_state(job, job_pb2.JOB_STATE_RUNNING, timeout=smoke_cluster.job_timeout)

    smoke_cluster.kill(job)
    killed_status = smoke_cluster.wait(job, timeout=smoke_cluster.job_timeout)
    assert killed_status.state == job_pb2.JOB_STATE_KILLED

    # If resources weren't released, this job would stay PENDING forever.
    followup = smoke_cluster.submit(TestJobs.quick, "smoke-cancel-followup", cpu=heavy_cpu)
    followup_status = smoke_cluster.wait(followup, timeout=smoke_cluster.job_timeout)
    assert followup_status.state == job_pb2.JOB_STATE_SUCCEEDED


# ============================================================================
# Log level verification
# ============================================================================


def test_log_levels_populated(smoke_cluster, verbose_job, capabilities):
    """Task logs have level field (INFO, WARNING, ERROR)."""
    if not capabilities.has_workers:
        pytest.skip("kubernetes_provider log collection does not parse structured levels yet")

    task_id = verbose_job.job_id.task(0).to_wire()

    deadline = time.monotonic() + smoke_cluster.job_timeout
    entries = []
    while time.monotonic() < deadline:
        request = logging_pb2.FetchLogsRequest(source=re.escape(task_id) + ":.*")
        response = smoke_cluster.log_client.fetch_logs(request)
        entries = list(response.entries)
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


def test_log_level_filter(smoke_cluster, verbose_job, capabilities):
    """min_level=WARNING excludes INFO."""
    if not capabilities.has_workers:
        pytest.skip("kubernetes_provider log collection does not parse structured levels yet")

    task_id = verbose_job.job_id.task(0).to_wire()

    request = logging_pb2.FetchLogsRequest(source=re.escape(task_id) + ":.*", min_level="WARNING")
    response = smoke_cluster.log_client.fetch_logs(request)
    filtered = list(response.entries)

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

    request = controller_pb2.Controller.ListWorkersRequest()
    response = smoke_cluster.controller_client.list_workers(request)
    worker = next(
        (w for w in response.workers if w.worker_id == task.worker_id or w.address == task.worker_id),
        None,
    )
    assert worker is not None
    region_attr = worker.metadata.attributes.get(WellKnownAttribute.REGION)
    if region_attr and region_attr.HasField("string_value"):
        assert region_attr.string_value == target_region, f"Expected {target_region}, got {region_attr.string_value}"


def test_capacity_type_propagates_to_worker_attributes(smoke_cluster):
    """Workers from preemptible groups register preemptible=true, on-demand groups false.

    Catches regressions where config.capacity_type gets lost on the way to
    worker metadata (e.g. LOCAL-mode fake deriving it from the wrong source).
    """
    request = controller_pb2.Controller.ListWorkersRequest()
    response = smoke_cluster.controller_client.list_workers(request)
    assert response.workers, "Expected registered workers"

    for w in response.workers:
        attrs = w.metadata.attributes
        preemptible_attr = attrs.get(WellKnownAttribute.PREEMPTIBLE)
        assert preemptible_attr is not None, f"Worker {w.worker_id} missing preemptible attribute"

        device_attr = attrs.get(WellKnownAttribute.DEVICE_TYPE)
        device_type = device_attr.string_value if device_attr else "cpu"

        # Smoke cluster: TPU groups are preemptible, CPU groups are on-demand
        if device_type == "tpu":
            assert (
                preemptible_attr.string_value == "true"
            ), f"TPU worker {w.worker_id} should be preemptible=true, got {preemptible_attr.string_value}"
        else:
            assert (
                preemptible_attr.string_value == "false"
            ), f"CPU worker {w.worker_id} should be preemptible=false, got {preemptible_attr.string_value}"


# ============================================================================
# Profiling
# ============================================================================


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="py-spy ptrace can segfault worker threads in CI")
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
        return last_state == job_pb2.TASK_STATE_RUNNING

    ExponentialBackoff(initial=0.1, maximum=2.0).wait_until_or_raise(
        _is_running,
        timeout=Duration.from_seconds(smoke_cluster.job_timeout),
        error_message=f"Task did not reach RUNNING within {smoke_cluster.job_timeout}s, last state: {last_state}",
    )
    task_id = smoke_cluster.task_status(job, task_index=0).task_id

    request = job_pb2.ProfileTaskRequest(
        target=task_id,
        duration_seconds=1,
        profile_type=job_pb2.ProfileType(cpu=job_pb2.CpuProfile(format=job_pb2.CpuProfile.FLAMEGRAPH)),
    )
    response = smoke_cluster.controller_client.profile_task(request, timeout_ms=3000)
    assert len(response.profile_data) > 0
    assert not response.error

    smoke_cluster.wait(job, timeout=smoke_cluster.job_timeout)


# ============================================================================
# Exec in container
# ============================================================================


@pytest.mark.timeout(300)
def test_exec_in_container(smoke_cluster):
    """Exec a command in a running task's container."""
    job = smoke_cluster.submit(TestJobs.sleep, "smoke-exec", 120)
    smoke_cluster.wait_for_state(job, job_pb2.JOB_STATE_RUNNING, timeout=smoke_cluster.job_timeout)

    # Wait for the task itself to reach RUNNING (job can be RUNNING while task is still BUILDING)
    task_id = smoke_cluster.task_status(job, task_index=0).task_id
    deadline = time.monotonic() + smoke_cluster.job_timeout
    while time.monotonic() < deadline:
        task = smoke_cluster.task_status(job, task_index=0)
        if task.state == job_pb2.TASK_STATE_RUNNING:
            break
        time.sleep(0.5)
    assert task.state == job_pb2.TASK_STATE_RUNNING, f"Task stuck in {job_pb2.TaskState.Name(task.state)}"

    request = controller_pb2.Controller.ExecInContainerRequest(
        task_id=task_id,
        command=["echo", "hello"],
    )
    response = smoke_cluster.controller_client.exec_in_container(request)
    assert not response.error, f"exec failed: {response.error}"
    assert response.exit_code == 0
    assert "hello" in response.stdout

    smoke_cluster.kill(job)


@pytest.mark.timeout(180)
def test_worker_restart_preserves_task(smoke_cluster):
    """Restarting a worker preserves its running task via container adoption.

    Submits a task that logs every second, restarts the worker running it,
    then verifies the task is still RUNNING and eventually succeeds. Fetches
    logs to confirm continuity across the restart.
    """
    # Use shorter duration in local mode (restart is a no-op there)
    is_local = not smoke_cluster.is_cloud
    task_duration = 30 if is_local else 90

    job = smoke_cluster.submit(TestJobs.log_periodic, "smoke-restart", task_duration, 1.0)
    smoke_cluster.wait_for_state(job, job_pb2.JOB_STATE_RUNNING, timeout=smoke_cluster.job_timeout)

    # Wait for task itself to be RUNNING (not just BUILDING)
    deadline = time.monotonic() + smoke_cluster.job_timeout
    while time.monotonic() < deadline:
        task = smoke_cluster.task_status(job, task_index=0)
        if task.state == job_pb2.TASK_STATE_RUNNING:
            break
        time.sleep(0.5)
    assert task.state == job_pb2.TASK_STATE_RUNNING, f"Task stuck in {job_pb2.TaskState.Name(task.state)}"

    # Let a few ticks log before we restart
    time.sleep(3)

    # Find the worker running this task
    worker_id = task.worker_id
    assert worker_id, "Task has no worker_id"

    # Restart the worker via the controller RPC
    restart_resp = smoke_cluster.controller_client.restart_worker(
        controller_pb2.Controller.RestartWorkerRequest(worker_id=worker_id),
        timeout_ms=60_000,
    )
    assert restart_resp.accepted, f"Restart rejected: {restart_resp.error}"

    # In cloud mode, wait for the worker to come back as healthy after real restart.
    # In local mode, restart_worker is a no-op so the worker stays healthy.
    if not is_local:
        worker_back = False
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            time.sleep(5)
            workers_resp = smoke_cluster.controller_client.list_workers(controller_pb2.Controller.ListWorkersRequest())
            for w in workers_resp.workers:
                if w.worker_id == worker_id and w.healthy:
                    worker_back = True
                    break
            if worker_back:
                break
        assert worker_back, f"Worker {worker_id} did not re-register within 120s"

    # Wait for the job to complete — the task should either be adopted by
    # the new worker (RUNNING → SUCCEEDED) or retried by the controller
    # (WORKER_FAILED → re-scheduled → SUCCEEDED). Either path validates
    # that the restart didn't permanently break the task.
    final = smoke_cluster.wait(job, timeout=120)
    assert (
        final.state == job_pb2.JOB_STATE_SUCCEEDED
    ), f"Job should succeed after restart, got {job_pb2.JobState.Name(final.state)}"


# ============================================================================
# Checkpoint / restore
# ============================================================================


def test_checkpoint_restore():
    """Controller restart resumes from checkpoint: completed jobs visible, cluster functional.

    Uses a dedicated LocalCluster (not the shared smoke_cluster). The persistent DB dir
    (held by LocalCluster across stop/start) preserves checkpoint state.
    Phase 1 — run a job and write a checkpoint.
    Phase 2 — restart the controller and verify the job is still SUCCEEDED
              and the cluster can accept new work.
    """
    from iris.cluster.providers.local.cluster import LocalCluster

    config = load_config(DEFAULT_CONFIG)
    config = make_local_config(config)

    cluster = LocalCluster(config)
    url = cluster.start()
    try:
        # Phase 1: complete a job, write checkpoint, restart controller.
        client = IrisClient.remote(url, workspace=MARIN_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        log_client = LogServiceClientSync(address=url, timeout_ms=30000)
        tc = IrisTestCluster(url=url, client=client, controller_client=controller_client, log_client=log_client)
        tc.wait_for_workers(1, timeout=30)

        job = tc.submit(TestJobs.quick, "pre-restart")
        tc.wait(job, timeout=30)
        saved_job_id = job.job_id.to_wire()

        ckpt = controller_client.begin_checkpoint(controller_pb2.Controller.BeginCheckpointRequest())
        assert ckpt.checkpoint_path, "begin_checkpoint returned empty path"
        assert ckpt.job_count >= 1
        controller_client.close()

        url = cluster.restart()

        # Phase 2: verify restored state and submit new work.
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        log_client = LogServiceClientSync(address=url, timeout_ms=30000)
        tc = IrisTestCluster(
            url=url,
            client=IrisClient.remote(url, workspace=MARIN_ROOT),
            controller_client=controller_client,
            log_client=log_client,
        )

        resp = controller_client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=saved_job_id))
        assert resp.job.state == job_pb2.JOB_STATE_SUCCEEDED, f"Pre-restart job has state {resp.job.state} after restore"

        tc.wait_for_workers(1, timeout=30)
        post_job = tc.submit(TestJobs.quick, "post-restart")
        status = tc.wait(post_job, timeout=30)
        assert status.state == job_pb2.JOB_STATE_SUCCEEDED

        controller_client.close()
    finally:
        cluster.close()


# ============================================================================
# Stress test
# ============================================================================


@pytest.mark.timeout(600)
def test_stress_50_tasks(smoke_cluster):
    """50 concurrent tasks exercises scheduler concurrency and bin-packing."""
    job = smoke_cluster.submit(
        TestJobs.quick,
        "smoke-stress-50",
        cpu=0,
        replicas=50,
    )
    status = smoke_cluster.wait(job, timeout=smoke_cluster.job_timeout * 2)
    assert status.state == job_pb2.JOB_STATE_SUCCEEDED


# ============================================================================
# Workdir file offload (large files externalized to blob store)
# ============================================================================

OFFLOAD_FILE_SIZE = 200 * 1024  # 200KB — exceeds the 100KB offload threshold


def test_workdir_file_offload(smoke_cluster):
    """A job with a >100KB workdir file succeeds after blob-store offloading."""
    entrypoint = Entrypoint.from_callable(TestJobs.verify_workdir_file, "large_payload.bin", OFFLOAD_FILE_SIZE)
    entrypoint.workdir_files["large_payload.bin"] = b"\xab" * OFFLOAD_FILE_SIZE
    job = smoke_cluster.client.submit(
        entrypoint=entrypoint,
        name=f"smoke-offload-{uuid.uuid4().hex[:8]}",
        resources=ResourceSpec(cpu=1, memory="1g"),
    )
    status = smoke_cluster.wait(job, timeout=smoke_cluster.job_timeout)
    assert status.state == job_pb2.JOB_STATE_SUCCEEDED


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
    sg.buffer_slices = 0
    sg.max_slices = 0
    sg.resources.cpu_millicores = 1000
    sg.resources.memory_bytes = 1 * 1024**3
    sg.resources.disk_bytes = 10 * 1024**3
    sg.resources.device_type = config_pb2.ACCELERATOR_TYPE_CPU
    sg.resources.capacity_type = config_pb2.CAPACITY_TYPE_ON_DEMAND
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
                container_runtime=ProcessRuntime(cache_dir=cache_dir),
                environment_provider=env_provider,
                threads=threads,
            )
            worker.start()

            try:
                controller_client = ControllerServiceClientSync(address=url, timeout_ms=10000)
                deadline = time.monotonic() + 15.0
                workers = []
                while time.monotonic() < deadline:
                    request = controller_pb2.Controller.ListWorkersRequest()
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


# ============================================================================
# Dashboard authentication flow (standalone cluster with auth enabled)
# ============================================================================

_AUTH_TOKEN = "e2e-test-token"
_AUTH_USER = "test-user"


def test_dashboard_login_flow():
    """Dashboard shows login page when auth is enabled, allows token login, and supports logout.

    Creates a standalone local cluster with static auth via config. Exercises the
    full browser auth flow: redirect to login, paste token, verify RPC data loads,
    then logout back to the login page.
    """
    from iris.cluster.providers.local.cluster import LocalCluster

    try:
        import playwright.sync_api as pw
    except ImportError:
        pytest.skip("playwright not installed")

    config = _make_controller_only_config()
    config.auth.static.tokens[_AUTH_TOKEN] = _AUTH_USER
    controller = LocalCluster(config)
    url = controller.start()

    # Run Playwright in a separate thread to avoid conflict with the asyncio
    # event loop that AnyIO worker threads may have installed.
    errors: list[Exception] = []

    def _run_browser_flow():
        try:
            with pw.sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page(viewport={"width": 1400, "height": 900})

                # Navigate to dashboard root — auth guard should redirect to login
                page.goto(f"{url}/")
                page.wait_for_load_state("domcontentloaded")
                wait_for_dashboard_ready(page)

                page.wait_for_function(
                    "() => window.location.hash.includes('/login')",
                    timeout=10000,
                )

                # Login page should show the token textarea and login button
                assert_visible(page, "text=Iris Dashboard")
                assert_visible(page, "text=bearer token")
                assert_visible(page, "button:has-text('Login')")

                # Enter the token and submit
                page.fill("textarea#token", _AUTH_TOKEN)
                page.click("button:has-text('Login')")

                # After login, should redirect to jobs tab (root hash)
                page.wait_for_function(
                    "() => !window.location.hash.includes('/login')",
                    timeout=10000,
                )
                wait_for_dashboard_ready(page)

                # Verify the dashboard loaded — the Jobs tab heading or
                # an empty "No jobs" state should be visible
                page.wait_for_function(
                    "() => document.body.textContent.includes('Jobs') || "
                    "document.body.textContent.includes('No jobs')",
                    timeout=10000,
                )

                # Logout should redirect back to login
                page.click("button:has-text('Logout')")
                page.wait_for_function(
                    "() => window.location.hash.includes('/login')",
                    timeout=10000,
                )
                assert_visible(page, "text=bearer token")

                page.close()
                browser.close()
        except Exception as exc:
            errors.append(exc)

    import threading

    t = threading.Thread(target=_run_browser_flow)
    t.start()
    t.join(timeout=60)

    try:
        if errors:
            raise errors[0]
        if t.is_alive():
            raise TimeoutError("Browser flow did not complete within 60s")
    finally:
        controller.close()


def _login_for_jwt(url: str, identity_token: str) -> str:
    """Exchange a raw identity token for a JWT via the Login RPC."""
    client = ControllerServiceClientSync(address=url, timeout_ms=10000)
    try:
        resp = client.login(job_pb2.LoginRequest(identity_token=identity_token))
        return resp.token
    finally:
        client.close()


def test_static_auth_rpc_access():
    """Static auth rejects unauthenticated and wrong-token RPCs, accepts valid JWT."""
    from iris.cluster.providers.local.cluster import LocalCluster
    from iris.rpc.auth import AuthTokenInjector, StaticTokenProvider

    config = _make_controller_only_config()
    config.auth.static.tokens[_AUTH_TOKEN] = _AUTH_USER
    controller = LocalCluster(config)
    url = controller.start()

    try:
        list_req = controller_pb2.Controller.ListWorkersRequest()

        # Unauthenticated: should be rejected with 401
        unauth_client = ControllerServiceClientSync(address=url, timeout_ms=5000)
        with pytest.raises(ConnectError, match=r"(?i)authenticat"):
            unauth_client.list_workers(list_req)
        unauth_client.close()

        # Wrong token: should be rejected
        wrong_injector = AuthTokenInjector(StaticTokenProvider("wrong-token"))
        wrong_client = ControllerServiceClientSync(address=url, timeout_ms=5000, interceptors=[wrong_injector])
        with pytest.raises(ConnectError, match=r"(?i)authenticat"):
            wrong_client.list_workers(list_req)
        wrong_client.close()

        # Exchange static token for JWT, then use JWT
        jwt_token = _login_for_jwt(url, _AUTH_TOKEN)
        valid_injector = AuthTokenInjector(StaticTokenProvider(jwt_token))
        valid_client = ControllerServiceClientSync(address=url, timeout_ms=5000, interceptors=[valid_injector])
        response = valid_client.list_workers(list_req)
        assert response is not None
        valid_client.close()
    finally:
        controller.close()


def test_static_auth_job_ownership():
    """Job ownership: user A cannot terminate user B's job.

    Submits a job as user-a via the RPC layer (no workers needed; job stays
    PENDING). Verifies user-b gets PERMISSION_DENIED when trying to terminate
    it, while user-a can terminate their own job.
    """
    from iris.cluster.providers.local.cluster import LocalCluster
    from iris.rpc.auth import AuthTokenInjector, StaticTokenProvider

    _TOKEN_A = "token-user-a"
    _TOKEN_B = "token-user-b"

    config = _make_controller_only_config()
    config.auth.static.tokens[_TOKEN_A] = "user-a"
    config.auth.static.tokens[_TOKEN_B] = "user-b"
    controller = LocalCluster(config)
    url = controller.start()

    try:
        # Exchange static tokens for JWTs via Login RPC
        jwt_a = _login_for_jwt(url, _TOKEN_A)
        jwt_b = _login_for_jwt(url, _TOKEN_B)

        # User A submits a job (stays PENDING since no workers)
        injector_a = AuthTokenInjector(StaticTokenProvider(jwt_a))
        client_a = ControllerServiceClientSync(address=url, timeout_ms=10000, interceptors=[injector_a])

        entrypoint = Entrypoint.from_callable(TestJobs.quick)
        launch_req = controller_pb2.Controller.LaunchJobRequest(
            name="/user-a/auth-owned-job",
            entrypoint=entrypoint.to_proto(),
            resources=ResourceSpec(cpu=1, memory="1g").to_proto(),
        )
        resp = client_a.launch_job(launch_req)
        job_id = resp.job_id

        # User B tries to terminate user A's job — should fail
        injector_b = AuthTokenInjector(StaticTokenProvider(jwt_b))
        client_b = ControllerServiceClientSync(address=url, timeout_ms=10000, interceptors=[injector_b])
        with pytest.raises(ConnectError, match="cannot access resources owned by"):
            client_b.terminate_job(controller_pb2.Controller.TerminateJobRequest(job_id=job_id))

        # User A can terminate their own job
        client_a.terminate_job(controller_pb2.Controller.TerminateJobRequest(job_id=job_id))

        client_a.close()
        client_b.close()
    finally:
        controller.close()
