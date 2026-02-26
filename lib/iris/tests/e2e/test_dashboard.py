# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dashboard validation tests using Playwright.

Replaces scripts/screenshot-dashboard.py with proper test assertions. Each test
exercises a specific dashboard view and validates both DOM state and visual
rendering via screenshots.
"""

import pytest
from iris.rpc import cluster_pb2

from .conftest import _is_noop_page, assert_visible, dashboard_click, dashboard_goto, wait_for_dashboard_ready
from .helpers import _failing, _quick, _slow

pytestmark = pytest.mark.e2e


def _quick_with_log_marker():
    print("TASK_LOG_MARKER: dashboard-task-log-visible")
    return 1


def test_jobs_tab_shows_all_states(cluster, page, screenshot):
    """Submit jobs in various states, verify the Jobs tab renders them."""
    succeeded_job = cluster.submit(_quick, "dash-succeeded")
    failed_job = cluster.submit(_failing, "dash-failed")
    running_job = cluster.submit(_slow, "dash-running")

    cluster.wait(succeeded_job, timeout=30)
    cluster.wait(failed_job, timeout=30)
    cluster.wait_for_state(running_job, cluster_pb2.JOB_STATE_RUNNING, timeout=15)

    dashboard_goto(page, f"{cluster.url}/")
    wait_for_dashboard_ready(page)

    for name in ["dash-succeeded", "dash-failed", "dash-running"]:
        assert_visible(page, f"text={name}")

    screenshot("jobs-all-states")


def test_job_detail_page(cluster, page, screenshot):
    """Job detail page shows task status."""
    job = cluster.submit(_quick, "dash-detail")
    cluster.wait(job, timeout=30)

    dashboard_goto(page, f"{cluster.url}/job/{job.job_id.to_wire()}")
    wait_for_dashboard_ready(page)

    assert_visible(page, "text=SUCCEEDED")
    screenshot("job-detail-succeeded")


def test_job_detail_shows_task_logs(cluster, page, screenshot):
    """Job detail Task Logs panel shows task stdout logs."""
    job = cluster.submit(_quick_with_log_marker, "dash-task-logs")
    cluster.wait(job, timeout=30)

    dashboard_goto(page, f"{cluster.url}/job/{job.job_id.to_wire()}")
    wait_for_dashboard_ready(page)

    if not _is_noop_page(page):
        page.wait_for_function(
            "() => document.querySelector('pre') && "
            "document.querySelector('pre').textContent.includes('TASK_LOG_MARKER: dashboard-task-log-visible')",
            timeout=10000,
        )
    assert_visible(page, "text=TASK_LOG_MARKER: dashboard-task-log-visible")
    screenshot("job-detail-task-logs")


def test_fleet_tab(cluster, page, screenshot):
    """Fleet tab shows machines with health status."""
    cluster.wait_for_workers(1)
    dashboard_goto(page, f"{cluster.url}/")
    wait_for_dashboard_ready(page)
    dashboard_click(page, 'button.tab-btn:has-text("Fleet")')

    assert_visible(page, "text=ready")
    screenshot("fleet-tab")


def test_worker_detail_page(cluster, page, screenshot):
    """Worker detail page shows worker info, task history, and logs."""
    # Run a job so the worker has real task history
    job = cluster.submit(_quick, "worker-detail-ok")
    cluster.wait(job, timeout=30)

    # Find which worker ran the task and navigate to its detail page
    task_status = cluster.task_status(job)
    worker_id = task_status.worker_id
    assert worker_id, "Expected task to be assigned to a worker"

    dashboard_goto(page, f"{cluster.url}/worker/{worker_id}")

    if not _is_noop_page(page):
        # Wait for either success or error to render
        page.wait_for_function(
            "() => document.querySelector('.worker-detail-grid') !== null"
            " || document.querySelector('.error-message') !== null",
            timeout=10000,
        )
    screenshot("worker-detail")
    assert_visible(page, f"text={worker_id}")
    assert_visible(page, "text=Healthy")
    assert_visible(page, "text=Task History")


def test_autoscaler_tab(cluster, page, screenshot):
    """Autoscaler tab shows scale groups."""
    wait_for_dashboard_ready(page)
    dashboard_click(page, 'button.tab-btn:has-text("Autoscaler")')

    screenshot("autoscaler-tab")


def test_controller_logs(cluster, page, screenshot):
    """Logs tab renders the log viewer component."""
    wait_for_dashboard_ready(page)
    dashboard_click(page, 'button.tab-btn:has-text("Logs")')

    if not _is_noop_page(page):
        page.wait_for_selector("#log-container", timeout=10000)
        page.wait_for_function(
            "() => document.querySelectorAll('.log-line').length > 0"
            " || document.querySelector('.empty-state') !== null",
            timeout=10000,
        )
    screenshot("controller-logs")
