# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scheduling and multi-worker distribution tests.

Migrated from tests/cluster/test_e2e.py::TestResourceScheduling and TestMultiWorker.
"""

import pytest
from iris.rpc import cluster_pb2
from iris.time_utils import Duration

from .conftest import wait_for_dashboard_ready

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


def test_pending_job_visible_in_dashboard(cluster, page):
    """An unschedulable job should appear as PENDING in the dashboard."""
    cluster.submit(lambda: None, "dash-pending", cpu=10000)

    page.goto(f"{cluster.url}/")
    wait_for_dashboard_ready(page)

    assert page.locator("text=dash-pending").first.is_visible(timeout=5000)
    assert page.locator("text=PENDING").first.is_visible(timeout=5000)
