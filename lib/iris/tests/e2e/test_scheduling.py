# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scheduling and multi-worker distribution tests.

Migrated from tests/cluster/test_e2e.py::TestResourceScheduling and TestMultiWorker.
"""

import pytest
from iris.rpc import cluster_pb2
from iris.time_utils import Duration

from .conftest import assert_visible, dashboard_goto, wait_for_dashboard_ready

pytestmark = pytest.mark.e2e


def test_small_job_skips_oversized_job(cluster):
    """Small job gets scheduled even when a large unschedulable job is queued first."""
    big_job = cluster.submit(lambda: None, "big-job", cpu=10000)
    small_job = cluster.submit(lambda: "done", "small-job", cpu=1)

    status = cluster.wait(small_job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    big_status = cluster.status(big_job)
    assert big_status.state == cluster_pb2.JOB_STATE_PENDING


def test_scheduling_timeout(cluster):
    """Job that can never be scheduled becomes UNSCHEDULABLE after timeout."""
    job = cluster.submit(
        lambda: None,
        "impossible-job",
        cpu=10000,
        scheduling_timeout=Duration.from_seconds(1),
    )
    status = cluster.wait(job, timeout=10)
    assert status.state == cluster_pb2.JOB_STATE_UNSCHEDULABLE


def _brief_task():
    """Task that runs long enough for scheduling to distribute across workers."""
    import time

    time.sleep(0.5)
    return 42


def test_multi_worker_execution(multi_worker_cluster):
    """Replicated job distributes tasks across multiple workers.

    Each task sleeps briefly so multiple tasks are pending/running simultaneously,
    forcing the scheduler to distribute them. With cpu=5 and workers having cpu=8,
    each worker can only run one task at a time.
    """
    job = multi_worker_cluster.submit(
        _brief_task,
        "mw-job",
        cpu=5,
        replicas=6,
    )

    status = multi_worker_cluster.wait(job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    workers_used = set()
    for task_idx in range(6):
        task = multi_worker_cluster.task_status(job, task_index=task_idx)
        if task.worker_id:
            workers_used.add(task.worker_id)

    assert len(workers_used) > 1, f"Tasks should distribute across workers, but all ran on: {workers_used}"


# ---------------------------------------------------------------------------
# Dashboard assertions (require Playwright via the `page` fixture)
# ---------------------------------------------------------------------------


def test_pending_job_visible_in_dashboard(cluster, page, screenshot):
    """An unschedulable job should appear as PENDING in the dashboard."""
    cluster.submit(lambda: None, "dash-pending", cpu=10000)

    dashboard_goto(page, f"{cluster.url}/")
    wait_for_dashboard_ready(page)

    assert_visible(page, "text=dash-pending")
    assert_visible(page, "text=PENDING")
    screenshot("jobs-pending")


def test_scheduling_diagnostic_shown_on_job_detail(cluster, page, screenshot):
    """Job detail page shows scheduling diagnostics for unschedulable pending jobs.

    Submits a job requesting more CPU than any worker can provide, then verifies
    the job detail page renders the scheduling diagnostic with the actual rejection
    reason (e.g. "Insufficient CPU") rather than a generic "no resources" message.
    """
    # Ensure workers are registered so diagnostics show resource rejection, not "no workers"
    cluster.wait_for_workers(1, timeout=30)

    job = cluster.submit(lambda: None, "diag-cpu-too-high", cpu=10000)

    # Verify the job is pending via RPC first
    status = cluster.status(job)
    assert status.state == cluster_pb2.JOB_STATE_PENDING

    # Verify the RPC returns a meaningful pending_reason
    assert status.pending_reason, "GetJobStatus should return a non-empty pending_reason for unschedulable jobs"
    assert (
        "cpu" in status.pending_reason.lower()
    ), f"pending_reason should mention CPU as the bottleneck, got: {status.pending_reason}"

    # Navigate to job detail page and verify the diagnostic is rendered
    dashboard_goto(page, f"{cluster.url}/job/{job.job_id.to_wire()}")
    wait_for_dashboard_ready(page)

    assert_visible(page, "text=Scheduling Diagnostic")
    assert_visible(page, "text=Insufficient CPU")
    screenshot("job-detail-scheduling-diagnostic")


def test_scheduling_diagnostic_shows_device_mismatch(cluster, page, screenshot):
    """Job detail page shows device variant mismatch when requesting a non-existent TPU type.

    Uses IrisClient.submit directly to specify a device config that no worker matches.
    """
    from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec

    cluster.wait_for_workers(1, timeout=30)

    device_config = cluster_pb2.DeviceConfig()
    device_config.tpu.variant = "v5litepod-999"
    device_config.tpu.count = 1

    job = cluster.client.submit(
        entrypoint=Entrypoint.from_callable(lambda: None),
        name="diag-device-mismatch",
        resources=ResourceSpec(cpu=1, memory="1g", device=device_config),
        environment=EnvironmentSpec(),
    )

    # Verify the RPC returns a meaningful pending_reason about device mismatch
    status = cluster.status(job)
    assert status.state == cluster_pb2.JOB_STATE_PENDING
    assert status.pending_reason, "GetJobStatus should return a non-empty pending_reason for device mismatch"
    assert (
        "device" in status.pending_reason.lower() or "variant" in status.pending_reason.lower()
    ), f"pending_reason should mention device or variant mismatch, got: {status.pending_reason}"

    # Navigate to job detail page
    dashboard_goto(page, f"{cluster.url}/job/{job.job_id.to_wire()}")
    wait_for_dashboard_ready(page)

    assert_visible(page, "text=Scheduling Diagnostic")
    screenshot("job-detail-device-mismatch-diagnostic")
