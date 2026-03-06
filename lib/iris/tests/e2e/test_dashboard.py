# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dashboard validation tests using Playwright.

Replaces scripts/screenshot-dashboard.py with proper test assertions. Each test
exercises a specific dashboard view and validates both DOM state and visual
rendering via screenshots.
"""

import time

import pytest
from iris.rpc import cluster_pb2

from .conftest import _is_noop_page, assert_visible, dashboard_click, dashboard_goto, wait_for_dashboard_ready
from .helpers import _failing, _quick, _slow

pytestmark = pytest.mark.e2e


def _quick_with_log_marker():
    print("TASK_LOG_MARKER: dashboard-task-log-visible")
    return 1


def _verbose_task():
    """Emit 200 numbered log lines with categorized prefixes for filter testing."""
    for i in range(200):
        if i % 3 == 0:
            print(f"[INFO] step {i}: processing data batch")
        elif i % 3 == 1:
            print(f"[WARN] step {i}: slow operation detected")
        else:
            print(f"[ERROR] step {i}: validation failed for item")
    print("DONE: all 200 lines emitted")
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
    """Job detail Task Logs panel shows logs, line limit buttons, and regex filter."""
    job = cluster.submit(_verbose_task, "dash-task-logs")
    cluster.wait(job, timeout=30)

    dashboard_goto(page, f"{cluster.url}/job/{job.job_id.to_wire()}")
    wait_for_dashboard_ready(page)

    if _is_noop_page(page):
        return

    # Wait for logs to appear (default 1K limit shows all 201 lines)
    page.wait_for_function(
        "() => document.querySelector('pre') && "
        "document.querySelector('pre').textContent.includes('DONE: all 200 lines emitted')",
        timeout=10000,
    )
    screenshot("task-logs-default-1k")

    # Click "100" button — should truncate to 100 lines
    page.click("button:has-text('100')")
    page.wait_for_function(
        "() => document.querySelector('span')  && " "document.body.textContent.includes('(truncated)')",
        timeout=5000,
    )
    screenshot("task-logs-limit-100")

    # Click "All" to restore
    page.click("button:has-text('All')")
    page.wait_for_function(
        "() => document.querySelector('pre') && "
        "document.querySelector('pre').textContent.includes('DONE: all 200 lines emitted')",
        timeout=5000,
    )
    screenshot("task-logs-limit-all")

    # Test regex filter — filter for ERROR lines only
    page.fill("input[placeholder='regex']", "ERROR")
    page.click("button:has-text('Apply')")
    page.wait_for_function(
        "() => document.querySelector('pre') && "
        "!document.querySelector('pre').textContent.includes('[INFO]') && "
        "document.querySelector('pre').textContent.includes('[ERROR]')",
        timeout=5000,
    )
    screenshot("task-logs-regex-error")

    # Clear filter — all lines return
    page.click("button:has-text('Clear')")
    page.wait_for_function(
        "() => document.querySelector('pre') && " "document.querySelector('pre').textContent.includes('[INFO]')",
        timeout=5000,
    )
    screenshot("task-logs-regex-cleared")


def test_workers_tab(cluster, page, screenshot):
    """Workers tab shows workers with health status."""
    cluster.wait_for_workers(1)
    dashboard_goto(page, f"{cluster.url}/")
    wait_for_dashboard_ready(page)
    dashboard_click(page, 'button.tab-btn:has-text("Workers")')

    assert_visible(page, "text=healthy")
    screenshot("workers-tab")


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


def test_worker_detail_metric_cards(cluster, page, screenshot):
    """Worker detail page renders MetricCard tiles for key stats."""
    job = cluster.submit(_quick, "worker-metric-cards")
    cluster.wait(job, timeout=30)

    task_status = cluster.task_status(job)
    worker_id = task_status.worker_id
    assert worker_id

    dashboard_goto(page, f"{cluster.url}/worker/{worker_id}")

    if not _is_noop_page(page):
        page.wait_for_function(
            "() => document.querySelector('.metric-card') !== null",
            timeout=10000,
        )
    # MetricCard components should be present (running tasks, CPU/memory, accelerator)
    assert_visible(page, "text=Running Tasks")
    screenshot("worker-detail-metric-cards")


def test_worker_detail_shows_network_and_disk_sparklines(cluster, page, screenshot):
    """Worker detail page shows network bandwidth and disk sparkline in Live Utilization."""
    job = cluster.submit(_quick, "worker-net-disk")
    cluster.wait(job, timeout=30)

    task_status = cluster.task_status(job)
    worker_id = task_status.worker_id
    assert worker_id

    dashboard_goto(page, f"{cluster.url}/worker/{worker_id}")

    if not _is_noop_page(page):
        page.wait_for_function(
            "() => document.querySelector('.utilization-panel') !== null",
            timeout=10000,
        )
    # Live Utilization panel should show CPU, Memory, Disk, and Network
    assert_visible(page, "text=Live Utilization")
    assert_visible(page, "text=CPU")
    assert_visible(page, "text=Disk")
    assert_visible(page, "text=Network")
    screenshot("worker-detail-net-disk-sparklines")


def _hold_for_heartbeats():
    """Sleep long enough for multiple heartbeat cycles to accumulate resource history."""
    import time

    time.sleep(6)
    return 1


def test_worker_detail_sparklines_with_history(cluster, page, screenshot):
    """Worker detail sparklines render once resource history has accumulated.

    Submits a task that holds long enough for multiple heartbeat cycles (local
    heartbeat_interval=0.5s) so the resource_history deque has enough entries
    for the Sparkline SVGs to render.
    """
    job = cluster.submit(_hold_for_heartbeats, "worker-sparkline-history")
    cluster.wait_for_state(job, cluster_pb2.JOB_STATE_RUNNING, timeout=15)

    task_status = cluster.task_status(job)
    worker_id = task_status.worker_id
    assert worker_id

    # Wait for heartbeats to accumulate (local interval is 0.5s, need >=2 entries)
    time.sleep(3)

    dashboard_goto(page, f"{cluster.url}/worker/{worker_id}")

    if not _is_noop_page(page):
        # Wait for the utilization panel with SVG sparklines to render
        page.wait_for_function(
            "() => document.querySelector('.utilization-panel') !== null"
            " && document.querySelectorAll('.utilization-panel svg.sparkline').length >= 3",
            timeout=15000,
        )

    # Verify all four utilization metric sections are present
    assert_visible(page, "text=Live Utilization")
    assert_visible(page, "text=CPU")
    assert_visible(page, "text=Memory")
    assert_visible(page, "text=Disk")
    assert_visible(page, "text=Network")

    # Verify SVG sparklines rendered (not hidden due to empty data)
    if not _is_noop_page(page):
        sparkline_count = page.locator(".utilization-panel svg.sparkline").count()
        assert sparkline_count >= 3, f"Expected at least 3 sparkline SVGs in utilization panel, got {sparkline_count}"

    screenshot("worker-detail-sparklines-with-history")

    cluster.wait(job, timeout=30)


def _allocate_memory_and_wait():
    """Allocate ~50 MB and sleep long enough for at least one stats collection cycle."""
    import time

    data = bytearray(50 * 1024 * 1024)  # 50 MB
    time.sleep(8)
    del data
    return 1


def test_job_detail_shows_human_readable_resources(cluster, page, screenshot):
    """Job detail page shows human-readable resource values (GB/MB, cores) in the Resource Request card.

    This verifies the fix for incomprehensible memory/CPU bars — the dashboard
    should show absolute values like '1 GB' rather than opaque percentages.
    """
    job = cluster.submit(_quick, "dash-resources", memory="4g", cpu=8)
    cluster.wait(job, timeout=30)

    dashboard_goto(page, f"{cluster.url}/job/{job.job_id.to_wire()}")
    wait_for_dashboard_ready(page)

    # The Resource Request card should show human-readable memory (4 GB) and CPU (8)
    assert_visible(page, "text=4 GB")
    assert_visible(page, "text=Resource Request")
    screenshot("job-detail-human-readable-resources")


def test_job_detail_task_table_shows_resource_values(cluster, page, screenshot):
    """Task table Mem/CPU columns show human-readable values for running tasks with stats.

    Submits a job that allocates memory and runs for ~8s (enough for at least
    one stats collection cycle), then verifies that the task table inline gauges
    display absolute values (e.g. 'MB' or 'GB') instead of raw percentages.
    """
    job = cluster.submit(_allocate_memory_and_wait, "dash-task-resources", memory="4g", cpu=8)
    cluster.wait_for_state(job, cluster_pb2.JOB_STATE_RUNNING, timeout=15)

    # Wait for stats collection (poll interval is 5s)
    time.sleep(7)

    dashboard_goto(page, f"{cluster.url}/job/{job.job_id.to_wire()}")
    wait_for_dashboard_ready(page)

    if not _is_noop_page(page):
        # Wait for the task table to render with inline-gauge elements
        page.wait_for_function(
            "() => document.querySelectorAll('.inline-gauge').length > 0",
            timeout=10000,
        )
        # Verify inline gauge text contains human-readable units (MB or GB)
        mem_gauge_text = page.locator(".inline-gauge__text").first.text_content()
        assert (
            "MB" in mem_gauge_text or "GB" in mem_gauge_text
        ), f"Expected memory gauge to show MB or GB units, got: '{mem_gauge_text}'"

    screenshot("job-detail-task-resource-values")

    # Clean up: kill the running job
    cluster.kill(job)
    cluster.wait(job, timeout=30)


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
