# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive smoke tests exercising Iris cluster features.

All tests share a single module-scoped cluster (smoke_cluster) with 12 workers
across CPU, TPU coscheduling, and multi-region scale groups. Tests are numbered
for execution order and share a module-level _jobs dict for cross-test job
handle passing.
"""

import subprocess
import time
import uuid
from unittest.mock import patch

import pytest
from iris.client.client import Job
from iris.cluster.config import load_config, make_local_config
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
    _is_noop_page,
    assert_visible,
    dashboard_click,
    dashboard_goto,
    wait_for_dashboard_ready,
)
from .helpers import (
    _emit_multi_level_logs,
    _failing,
    _noop,
    _port_job,
    _quick,
    _quick_task_job,
    _register_endpoint_job,
    _slow,
    _verbose_task,
)

pytestmark = pytest.mark.e2e

_jobs: dict[str, Job] = {}


def _busy_task():
    """Busy-loop with real Python work for profiling."""
    import time

    end = time.monotonic() + 3
    while time.monotonic() < end:
        sum(range(1000))


# ============================================================================
# Phase 1: Cluster readiness
# ============================================================================


def test_smoke_01_workers_ready(smoke_cluster, smoke_page, smoke_screenshot):
    """Wait for all 12 workers, screenshot fleet tab."""
    smoke_cluster.wait_for_workers(12, timeout=60)
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/")
    wait_for_dashboard_ready(smoke_page)
    dashboard_click(smoke_page, 'button.tab-btn:has-text("Workers")')
    assert_visible(smoke_page, "text=healthy")
    smoke_screenshot("01-workers-ready")


# ============================================================================
# Phase 2: Submit diverse jobs
# ============================================================================


def test_smoke_02_submit_diverse_jobs(smoke_cluster, smoke_page, smoke_screenshot):
    """Submit diverse jobs in parallel and wait for expected states."""
    _jobs["simple"] = smoke_cluster.submit(_quick, "smoke-simple")

    for i in range(3):
        _jobs[f"task-{i}"] = smoke_cluster.submit(_quick_task_job, f"smoke-task-{i}", i)

    _jobs["verbose"] = smoke_cluster.submit(_verbose_task, "smoke-verbose")
    _jobs["log-levels"] = smoke_cluster.submit(_emit_multi_level_logs, "smoke-log-levels")
    _jobs["failed"] = smoke_cluster.submit(_failing, "smoke-failed")
    _jobs["running"] = smoke_cluster.submit(_slow, "smoke-running")

    prefix = f"smoke-ep-{uuid.uuid4().hex[:8]}"
    _jobs["_ep_prefix"] = prefix  # type: ignore[assignment]
    _jobs["endpoint"] = smoke_cluster.submit(_register_endpoint_job, "smoke-endpoint", prefix)

    _jobs["ports"] = smoke_cluster.submit(_port_job, "smoke-ports", ports=["http", "grpc"])

    _jobs["region"] = smoke_cluster.submit(
        _noop,
        "smoke-region-a",
        constraints=[region_constraint(["us-central1"])],
    )

    _jobs["unschedulable"] = smoke_cluster.submit(
        _quick,
        "smoke-unsched",
        cpu=9999,
        scheduling_timeout=Duration.from_seconds(5),
    )

    _jobs["reserved"] = smoke_cluster.submit(
        _quick,
        "smoke-reserved",
        reservation=[ReservationEntry(resources=ResourceSpec(cpu=1, memory="1g", device=gpu_device("H100", 8)))],
    )

    for key in ["simple", "task-0", "task-1", "task-2", "verbose", "log-levels", "endpoint", "ports", "region"]:
        status = smoke_cluster.wait(_jobs[key], timeout=60)
        assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED, f"Job {key} failed: {status}"

    status = smoke_cluster.wait(_jobs["failed"], timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_FAILED

    status = smoke_cluster.wait(_jobs["unschedulable"], timeout=15)
    assert status.state in (cluster_pb2.JOB_STATE_FAILED, cluster_pb2.JOB_STATE_UNSCHEDULABLE)

    smoke_cluster.wait_for_state(_jobs["running"], cluster_pb2.JOB_STATE_RUNNING, timeout=15)

    reserved_status = smoke_cluster.status(_jobs["reserved"])
    assert reserved_status.state == cluster_pb2.JOB_STATE_PENDING

    smoke_screenshot("02-jobs-submitted")


# ============================================================================
# Phase 3: Dashboard screenshots
# ============================================================================


def test_smoke_03_dashboard_jobs_tab(smoke_cluster, smoke_page, smoke_screenshot):
    """Jobs tab shows diverse states."""
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/")
    wait_for_dashboard_ready(smoke_page)
    for name in ["smoke-simple", "smoke-failed", "smoke-running"]:
        assert_visible(smoke_page, f"text={name}")
    smoke_screenshot("03-jobs-tab")


def test_smoke_04_dashboard_job_detail(smoke_cluster, smoke_page, smoke_screenshot):
    """SUCCEEDED job detail page."""
    job = _jobs["simple"]
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/job/{job.job_id.to_wire()}")
    wait_for_dashboard_ready(smoke_page)
    assert_visible(smoke_page, "text=SUCCEEDED")
    smoke_screenshot("04-job-detail")


def test_smoke_05_dashboard_task_logs(smoke_cluster, smoke_page, smoke_screenshot):
    """Task logs show 200 lines, truncation buttons, substring filter."""
    job = _jobs["verbose"]
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/job/{job.job_id.to_wire()}")
    wait_for_dashboard_ready(smoke_page)

    if not _is_noop_page(smoke_page):
        smoke_page.wait_for_function(
            "() => document.querySelector('pre') && "
            "document.querySelector('pre').textContent.includes('DONE: all 200 lines emitted')",
            timeout=10000,
        )
        smoke_screenshot("05-task-logs-default")

        smoke_page.click("button:has-text('100')")
        smoke_page.wait_for_function(
            "() => document.querySelector('span') && document.body.textContent.includes('(truncated)')",
            timeout=5000,
        )
        smoke_screenshot("05-task-logs-truncated")

        smoke_page.fill("input[placeholder='substring']", "ERROR")
        smoke_page.click("button:has-text('Apply')")
        smoke_page.wait_for_function(
            "() => document.querySelector('pre') && "
            "!document.querySelector('pre').textContent.includes('[INFO]') && "
            "document.querySelector('pre').textContent.includes('[ERROR]')",
            timeout=5000,
        )
        smoke_screenshot("05-task-logs-filtered")


def test_smoke_06_dashboard_constraints(smoke_cluster, smoke_page, smoke_screenshot):
    """Constraint chips rendered on job detail."""
    constraints = [
        Constraint(key="region", op=ConstraintOp.EQ, value="local"),
        Constraint(key="env-tag", op=ConstraintOp.EXISTS),
        Constraint(key="device-variant", op=ConstraintOp.IN, values=("v5p-8", "v6e-4")),
    ]
    job = smoke_cluster.submit(_quick, "smoke-constraints", constraints=constraints)
    time.sleep(3)

    dashboard_goto(smoke_page, f"{smoke_cluster.url}/job/{job.job_id.to_wire()}")
    wait_for_dashboard_ready(smoke_page)

    if not _is_noop_page(smoke_page):
        smoke_page.click("text=Job Request")
        smoke_page.wait_for_function(
            "() => document.querySelector('.constraint-chip') !== null",
            timeout=5000,
        )
        chips = smoke_page.locator(".constraint-chip")
        texts = [chips.nth(i).text_content() for i in range(chips.count())]
        assert any("region = local" in t for t in texts), f"Expected region constraint in {texts}"

    smoke_screenshot("06-constraints")


def test_smoke_07_dashboard_scheduling_diagnostic(smoke_cluster, smoke_page, smoke_screenshot):
    """Scheduling diagnostic shows 'Insufficient CPU' for oversized job."""
    smoke_cluster.wait_for_workers(1, timeout=30)
    job = smoke_cluster.submit(lambda: None, "smoke-diag-cpu", cpu=10000)

    status = smoke_cluster.status(job)
    assert status.state == cluster_pb2.JOB_STATE_PENDING
    assert "cpu" in status.pending_reason.lower()

    dashboard_goto(smoke_page, f"{smoke_cluster.url}/job/{job.job_id.to_wire()}")
    wait_for_dashboard_ready(smoke_page)
    assert_visible(smoke_page, "text=Scheduling Diagnostic")
    assert_visible(smoke_page, "text=Insufficient CPU")
    smoke_screenshot("07-scheduling-diagnostic")


def test_smoke_08_dashboard_workers_tab(smoke_cluster, smoke_page, smoke_screenshot):
    """Workers tab shows healthy workers."""
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/")
    wait_for_dashboard_ready(smoke_page)
    dashboard_click(smoke_page, 'button.tab-btn:has-text("Workers")')
    assert_visible(smoke_page, "text=healthy")
    smoke_screenshot("08-workers-tab")


def test_smoke_09_dashboard_worker_detail(smoke_cluster, smoke_page, smoke_screenshot):
    """Worker detail page shows info, task history, metric cards."""
    job = _jobs["simple"]
    task_status = smoke_cluster.task_status(job)
    worker_id = task_status.worker_id
    assert worker_id

    dashboard_goto(smoke_page, f"{smoke_cluster.url}/worker/{worker_id}")

    if not _is_noop_page(smoke_page):
        smoke_page.wait_for_function(
            "() => document.querySelector('.worker-detail-grid') !== null"
            " || document.querySelector('.error-message') !== null",
            timeout=10000,
        )
    assert_visible(smoke_page, f"text={worker_id}")
    assert_visible(smoke_page, "text=Healthy")
    assert_visible(smoke_page, "text=Task History")
    smoke_screenshot("09-worker-detail")


def test_smoke_10_dashboard_autoscaler_tab(smoke_cluster, smoke_page, smoke_screenshot):
    """Autoscaler tab shows scale groups."""
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/")
    wait_for_dashboard_ready(smoke_page)
    dashboard_click(smoke_page, 'button.tab-btn:has-text("Autoscaler")')
    smoke_screenshot("10-autoscaler-tab")


def test_smoke_11_dashboard_status_tab(smoke_cluster, smoke_page, smoke_screenshot):
    """Status tab renders process info and log viewer."""
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/")
    wait_for_dashboard_ready(smoke_page)
    dashboard_click(smoke_page, 'button.tab-btn:has-text("Status")')
    if not _is_noop_page(smoke_page):
        smoke_page.wait_for_selector(".log-container", timeout=10000)
    smoke_screenshot("11-status-tab")


# ============================================================================
# Phase 4: Scheduling & endpoint verification
# ============================================================================


def test_smoke_12_small_job_skips_oversized(smoke_cluster):
    """Small job gets scheduled even when a large unschedulable job is queued."""
    big_job = smoke_cluster.submit(lambda: None, "smoke-big", cpu=10000)
    small_job = smoke_cluster.submit(lambda: "done", "smoke-small", cpu=1)
    status = smoke_cluster.wait(small_job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
    big_status = smoke_cluster.status(big_job)
    assert big_status.state == cluster_pb2.JOB_STATE_PENDING


def test_smoke_13_endpoint_registration(smoke_cluster):
    """Endpoint registration from inside job succeeded."""
    assert _jobs["endpoint"] is not None
    status = smoke_cluster.status(_jobs["endpoint"])
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_smoke_14_port_allocation(smoke_cluster):
    """Port allocation job succeeded."""
    assert _jobs["ports"] is not None
    status = smoke_cluster.status(_jobs["ports"])
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_smoke_15_reservation_gates_scheduling(smoke_cluster):
    """Unsatisfiable reservation blocks scheduling; regular jobs proceed."""
    reserved_status = smoke_cluster.status(_jobs["reserved"])
    assert reserved_status.state == cluster_pb2.JOB_STATE_PENDING

    regular = smoke_cluster.submit(_quick, "smoke-regular-while-reserved")
    status = smoke_cluster.wait(regular, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


# ============================================================================
# Phase 5: Log level verification
# ============================================================================


def test_smoke_16_log_levels_populated(smoke_cluster):
    """Task logs have level field (INFO, WARNING, ERROR)."""
    job = _jobs["log-levels"]
    task_id = job.job_id.task(0).to_wire()

    deadline = time.monotonic() + 60
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


def test_smoke_17_log_level_filter(smoke_cluster):
    """min_level=WARNING excludes INFO."""
    job = _jobs["log-levels"]
    task_id = job.job_id.task(0).to_wire()

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
# Phase 6: Multi-region routing
# ============================================================================


def test_smoke_18_region_constrained_routing(smoke_cluster):
    """Job with region constraint lands on correct worker."""
    job = _jobs["region"]
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
        assert region_attr.string_value == "us-central1", f"Expected us-central1, got {region_attr.string_value}"


# ============================================================================
# Phase 7: Profiling
# ============================================================================


def test_smoke_19_profile_running_task(smoke_cluster):
    """Profile a running task, verify data returned."""
    job = smoke_cluster.submit(_busy_task, name="smoke-profile")

    last_state = "unknown"

    def _is_running():
        nonlocal last_state
        task = smoke_cluster.task_status(job, task_index=0)
        last_state = task.state
        return last_state == cluster_pb2.TASK_STATE_RUNNING

    ExponentialBackoff(initial=0.1, maximum=2.0).wait_until_or_raise(
        _is_running,
        timeout=Duration.from_seconds(30),
        error_message=f"Task did not reach RUNNING within 30s, last state: {last_state}",
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

    smoke_cluster.wait(job, timeout=30)


# ============================================================================
# Phase 8: GPU metadata (standalone cluster)
# ============================================================================

_NVIDIA_SMI_H100_8X = "\n".join(["NVIDIA H100 80GB HBM3, 81559"] * 8)


def _make_controller_only_config() -> config_pb2.IrisClusterConfig:
    """Build a local config with no auto-scaled workers."""
    from .conftest import DEFAULT_CONFIG

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


def test_smoke_20_gpu_worker_metadata(tmp_path):
    """Mocked nvidia-smi registers GPU metadata on worker."""
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


# ============================================================================
# Cleanup
# ============================================================================


def test_smoke_99_cleanup(smoke_cluster):
    """Kill any long-running jobs."""
    for key in ["running", "reserved"]:
        if key in _jobs and isinstance(_jobs[key], Job):
            try:
                smoke_cluster.kill(_jobs[key])
            except Exception:
                pass
