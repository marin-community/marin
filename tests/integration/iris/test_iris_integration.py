# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris integration tests exercising job lifecycle, scheduling, and cluster features.

These tests run against an existing controller specified via --controller-url.
No dashboard screenshots are taken here; those remain in lib/iris/tests/e2e/test_smoke.py.
"""

import logging
import os
import re
import time
import uuid

import pytest
from iris.cluster.constraints import region_constraint
from iris.cluster.types import (
    ReservationEntry,
    ResourceSpec,
    gpu_device,
)
from iris.rpc import logging_pb2
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from rigging.timing import Duration, ExponentialBackoff

from .jobs import (
    busy_loop,
    log_verbose,
    noop,
    quick,
    fail,
    sleep,
    register_endpoint,
)

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


# ============================================================================
# Job lifecycle
# ============================================================================


def test_submit_and_succeed(integration_cluster):
    """Submit a simple job and verify it succeeds."""
    job = integration_cluster.submit(quick, "itest-simple")
    status = integration_cluster.wait(job, timeout=integration_cluster.job_timeout)
    assert status.state == job_pb2.JOB_STATE_SUCCEEDED


def test_submit_and_fail(integration_cluster):
    """Submit a failing job and verify it reports failure."""
    job = integration_cluster.submit(fail, "itest-fail")
    status = integration_cluster.wait(job, timeout=integration_cluster.job_timeout)
    assert status.state == job_pb2.JOB_STATE_FAILED


def test_cancel_job_releases_resources(integration_cluster):
    """Cancelling a running job decommits worker resources so new jobs can schedule.

    Regression test for #3553.
    """
    heavy_cpu = 2
    job = integration_cluster.submit(sleep, "itest-cancel-heavy", 30, cpu=heavy_cpu)
    integration_cluster.wait_for_state(job, job_pb2.JOB_STATE_RUNNING, timeout=integration_cluster.job_timeout)

    integration_cluster.kill(job)
    killed_status = integration_cluster.wait(job, timeout=integration_cluster.job_timeout)
    assert killed_status.state == job_pb2.JOB_STATE_KILLED

    followup = integration_cluster.submit(quick, "itest-cancel-followup", cpu=heavy_cpu)
    followup_status = integration_cluster.wait(followup, timeout=integration_cluster.job_timeout)
    assert followup_status.state == job_pb2.JOB_STATE_SUCCEEDED


# ============================================================================
# Scheduling & endpoint verification
# ============================================================================


def test_endpoint_registration(integration_cluster):
    """Endpoint registered from inside job via RPC."""
    prefix = f"itest-ep-{uuid.uuid4().hex[:8]}"
    job = integration_cluster.submit(register_endpoint, "itest-endpoint", prefix)
    status = integration_cluster.wait(job, timeout=integration_cluster.job_timeout)
    assert status.state == job_pb2.JOB_STATE_SUCCEEDED


def test_reservation_gates_scheduling(integration_cluster):
    """Unsatisfiable reservation blocks scheduling; regular jobs proceed."""
    with integration_cluster.launched_job(
        quick,
        "itest-reserved",
        reservation=[
            ReservationEntry(resources=ResourceSpec(cpu=1, memory="1g", device=gpu_device("NONEXISTENT-GPU-9999", 99)))
        ],
    ) as reserved:
        reserved_status = integration_cluster.status(reserved)
        assert reserved_status.state == job_pb2.JOB_STATE_PENDING

        regular = integration_cluster.submit(quick, "itest-regular-while-reserved")
        status = integration_cluster.wait(regular, timeout=integration_cluster.job_timeout)
        assert status.state == job_pb2.JOB_STATE_SUCCEEDED


# ============================================================================
# Log level verification
# ============================================================================


@pytest.fixture(scope="module")
def verbose_job(integration_cluster):
    """Shared verbose log job used by log-related tests."""
    job = integration_cluster.submit(log_verbose, "itest-verbose")
    integration_cluster.wait(job, timeout=integration_cluster.job_timeout)
    return job


def test_log_levels_populated(integration_cluster, verbose_job):
    """Task logs have level field (INFO, WARNING, ERROR)."""
    task_id = verbose_job.job_id.task(0).to_wire()

    deadline = time.monotonic() + integration_cluster.job_timeout
    entries = []
    source = re.escape(task_id) + ":.*"
    while time.monotonic() < deadline:
        response = integration_cluster.fetch_logs(source)
        entries = list(response.entries)
        if any("info-marker" in e.data for e in entries):
            break
        time.sleep(0.5)

    markers_found = {}
    for entry in entries:
        for marker in ("info-marker", "warning-marker", "error-marker"):
            if marker in entry.data:
                markers_found[marker] = entry.level

    assert "info-marker" in markers_found, f"info-marker not found. Got {len(entries)} entries"
    assert markers_found["info-marker"] == logging_pb2.LOG_LEVEL_INFO
    assert markers_found.get("warning-marker") == logging_pb2.LOG_LEVEL_WARNING
    assert markers_found.get("error-marker") == logging_pb2.LOG_LEVEL_ERROR


def test_log_level_filter(integration_cluster, verbose_job):
    """min_level=WARNING excludes INFO."""
    task_id = verbose_job.job_id.task(0).to_wire()

    response = integration_cluster.fetch_logs(re.escape(task_id) + ":.*", min_level="WARNING")
    filtered = list(response.entries)

    filtered_data = [e.data for e in filtered]
    assert any("warning-marker" in d for d in filtered_data), f"warning-marker missing: {filtered_data}"
    assert any("error-marker" in d for d in filtered_data), f"error-marker missing: {filtered_data}"
    assert not any("info-marker" in d for d in filtered_data if d), "info-marker should be filtered out"


# ============================================================================
# Multi-region routing
# ============================================================================


def test_region_constrained_routing(integration_cluster):
    """Job with region constraint lands on correct worker."""
    # Query workers to check for multi-region support
    workers = integration_cluster.list_workers()

    from iris.cluster.constraints import WellKnownAttribute

    regions = set()
    for w in workers:
        region_attr = w.metadata.attributes.get(WellKnownAttribute.REGION)
        if region_attr and region_attr.HasField("string_value"):
            regions.add(region_attr.string_value)

    if len(regions) < 2:
        pytest.skip("No multi-region workers in cluster")

    target_region = sorted(regions)[0]
    job = integration_cluster.submit(
        noop,
        "itest-region",
        constraints=[region_constraint([target_region])],
    )
    integration_cluster.wait(job, timeout=integration_cluster.job_timeout)

    task = integration_cluster.task_status(job, task_index=0)
    assert task.worker_id

    # Re-fetch workers after job completes in case autoscaling added new nodes
    # to satisfy the region constraint.
    post_workers = integration_cluster.list_workers()
    worker = next(
        (w for w in post_workers if w.worker_id == task.worker_id or w.address == task.worker_id),
        None,
    )
    assert worker is not None, f"Worker {task.worker_id} not found in {[w.worker_id for w in post_workers]}"
    region_attr = worker.metadata.attributes.get(WellKnownAttribute.REGION)
    if region_attr and region_attr.HasField("string_value"):
        assert region_attr.string_value == target_region, f"Expected {target_region}, got {region_attr.string_value}"


# ============================================================================
# Profiling
# ============================================================================


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="py-spy ptrace can segfault worker threads in CI")
def test_profile_running_task(integration_cluster):
    """Profile a running task, verify data returned."""
    job = integration_cluster.submit(busy_loop, name="itest-profile")

    last_state = "unknown"

    def _is_running():
        nonlocal last_state
        task = integration_cluster.task_status(job, task_index=0)
        last_state = task.state
        return last_state == job_pb2.TASK_STATE_RUNNING

    ExponentialBackoff(initial=0.1, maximum=2.0).wait_until_or_raise(
        _is_running,
        timeout=Duration.from_seconds(integration_cluster.job_timeout),
        error_message=f"Task did not reach RUNNING within {integration_cluster.job_timeout}s, last state: {last_state}",
    )
    task_id = integration_cluster.task_status(job, task_index=0).task_id

    request = job_pb2.ProfileTaskRequest(
        target=task_id,
        duration_seconds=1,
        profile_type=job_pb2.ProfileType(cpu=job_pb2.CpuProfile(format=job_pb2.CpuProfile.FLAMEGRAPH)),
    )
    response = integration_cluster.controller_client.profile_task(request, timeout_ms=3000)
    assert len(response.profile_data) > 0
    assert not response.error

    integration_cluster.wait(job, timeout=integration_cluster.job_timeout)


# ============================================================================
# Exec in container
# ============================================================================


@pytest.mark.timeout(300)
def test_exec_in_container(integration_cluster):
    """Exec a command in a running task's container."""
    job = integration_cluster.submit(sleep, "itest-exec", 120)
    task = integration_cluster.wait_for_task_state(
        job, job_pb2.TASK_STATE_RUNNING, timeout=integration_cluster.job_timeout
    )
    task_id = task.task_id

    request = controller_pb2.Controller.ExecInContainerRequest(
        task_id=task_id,
        command=["echo", "hello"],
    )
    response = integration_cluster.controller_client.exec_in_container(request)
    assert not response.error, f"exec failed: {response.error}"
    assert response.exit_code == 0
    assert "hello" in response.stdout

    integration_cluster.kill(job)


# ============================================================================
# Stress test
# ============================================================================


@pytest.mark.timeout(600)
def test_stress_50_tasks(integration_cluster):
    """50 concurrent tasks exercises scheduler concurrency and bin-packing."""
    job = integration_cluster.submit(
        quick,
        "itest-stress-50",
        cpu=0,
        replicas=50,
    )
    status = integration_cluster.wait(job, timeout=integration_cluster.job_timeout * 2)
    assert status.state == job_pb2.JOB_STATE_SUCCEEDED
