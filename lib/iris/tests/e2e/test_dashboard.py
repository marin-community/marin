# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dashboard validation tests using Playwright.

Replaces scripts/screenshot-dashboard.py with proper test assertions. Each test
exercises a specific dashboard view and validates both DOM state and visual
rendering via screenshots.
"""

import pytest
from iris.rpc import cluster_pb2

from .conftest import wait_for_dashboard_ready
from .helpers import _failing, _quick, _slow

pytest.importorskip("playwright")

pytestmark = pytest.mark.e2e


def test_jobs_tab_shows_all_states(cluster, page, screenshot):
    """Submit jobs in various states, verify the Jobs tab renders them."""
    succeeded_job = cluster.submit(_quick, "dash-succeeded")
    failed_job = cluster.submit(_failing, "dash-failed")
    running_job = cluster.submit(_slow, "dash-running")

    cluster.wait(succeeded_job, timeout=30)
    cluster.wait(failed_job, timeout=30)
    cluster.wait_for_state(running_job, cluster_pb2.JOB_STATE_RUNNING, timeout=15)

    page.goto(f"{cluster.url}/")
    wait_for_dashboard_ready(page)

    for name in ["dash-succeeded", "dash-failed", "dash-running"]:
        assert page.locator(f"text={name}").first.is_visible(timeout=5000)

    screenshot("jobs-all-states")


def test_job_detail_page(cluster, page, screenshot):
    """Job detail page shows task status."""
    job = cluster.submit(_quick, "dash-detail")
    cluster.wait(job, timeout=30)

    page.goto(f"{cluster.url}/job/{job.job_id.to_wire()}")
    wait_for_dashboard_ready(page)

    assert page.locator("text=SUCCEEDED").first.is_visible(timeout=5000)
    screenshot("job-detail-succeeded")


def test_workers_tab(cluster, page, screenshot):
    """Workers tab shows healthy workers."""
    wait_for_dashboard_ready(page)
    page.click('button.tab-btn:has-text("Workers")')

    assert page.locator("text=healthy").first.is_visible(timeout=5000)
    screenshot("workers-tab")


def test_autoscaler_tab(cluster, page, screenshot):
    """Autoscaler tab shows scale groups."""
    wait_for_dashboard_ready(page)
    page.click('button.tab-btn:has-text("Autoscaler")')

    screenshot("autoscaler-tab")


def test_controller_logs(cluster, page, screenshot):
    """Logs tab renders the log viewer component."""
    wait_for_dashboard_ready(page)
    page.click('button.tab-btn:has-text("Logs")')

    # The log viewer always renders #log-container; wait for it plus either
    # log lines or the "No logs found" empty state (both are valid).
    page.wait_for_selector("#log-container", timeout=10000)
    page.wait_for_function(
        "() => document.querySelectorAll('.log-line').length > 0" " || document.querySelector('.empty-state') !== null",
        timeout=10000,
    )
    screenshot("controller-logs")
