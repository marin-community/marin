# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker failure E2E tests.

Tests worker crashes, delayed registration, stale state, and task-level retries.
All chaos is injected inline in worker.py.
"""

import pytest
from iris.chaos import enable_chaos
from iris.rpc import cluster_pb2
from iris.time_utils import Duration

from .conftest import wait_for_dashboard_ready
from .helpers import _quick, _slow

pytestmark = pytest.mark.e2e


def test_worker_crash_mid_task(cluster):
    """Worker task monitor crashes mid-task. Task fails, controller detects
    via heartbeat reconciliation or report_task_state."""
    enable_chaos("worker.task_monitor", failure_rate=1.0)
    job = cluster.submit(_quick, "crash-mid-task")
    status = cluster.wait(job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_FAILED


def test_worker_delayed_registration(cluster):
    """Worker registration delayed by 5s on first attempt. Task pends, then
    schedules once registration completes."""
    enable_chaos("worker.register", delay_seconds=5.0, max_failures=1)
    job = cluster.submit(_quick, "delayed-reg")
    status = cluster.wait(job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_worker_sequential_jobs(cluster):
    """Sequential jobs verify reconciliation works across job boundaries.
    Worker state is consistent between tasks."""
    for i in range(3):
        job = cluster.submit(_quick, f"seq-{i}")
        status = cluster.wait(job, timeout=30)
        assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


@pytest.mark.timeout(60)
def test_all_workers_fail(cluster):
    """All workers' registration fails permanently. With scheduling timeout,
    job transitions to FAILED/UNSCHEDULABLE when no workers register.
    """
    enable_chaos("worker.register", failure_rate=1.0, error=RuntimeError("chaos: registration failed"))
    job = cluster.submit(_slow, "all-workers-fail", scheduling_timeout=Duration.from_seconds(15))
    status = cluster.wait(job, timeout=30)
    assert status.state in (cluster_pb2.JOB_STATE_FAILED, cluster_pb2.JOB_STATE_UNSCHEDULABLE)


def test_task_fails_once_then_succeeds(cluster):
    """Container creation fails once, succeeds on retry."""
    enable_chaos(
        "worker.create_container",
        failure_rate=1.0,
        max_failures=1,
        error=RuntimeError("chaos: transient container failure"),
    )
    job = cluster.submit(_quick, "retry-once", max_retries_failure=2)
    status = cluster.wait(job, timeout=60)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


# ---------------------------------------------------------------------------
# Dashboard assertions (require Playwright via the `page` fixture)
# ---------------------------------------------------------------------------


def test_worker_health_in_dashboard(cluster, page):
    """Workers tab shows at least one healthy worker."""
    page.goto(f"{cluster.url}/")
    wait_for_dashboard_ready(page)
    page.click('button.tab-btn:has-text("Workers")')

    assert page.locator("text=healthy").first.is_visible(timeout=5000)
