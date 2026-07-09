# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive smoke tests exercising Iris cluster features.

All tests share a single module-scoped cluster (smoke_cluster). Each test
submits its own jobs and is independently runnable. In local mode the cluster
has workers across CPU, TPU coscheduling, and multi-region scale groups.
"""

import logging
import os
import time
import uuid
from pathlib import Path

import pytest
from finelog.rpc import logging_pb2
from finelog.rpc.logging_connect import LogServiceClientSync
from iris.client.client import IrisClient, iris_ctx
from iris.cluster.config import (
    IrisClusterConfig,
    LocalSliceConfig,
    ScaleGroupConfig,
    ScaleGroupResources,
    SliceConfig,
    load_config,
    make_local_config,
)
from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute, region_constraint
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.lifecycle import connect_cluster
from iris.cluster.local_cluster import LocalCluster
from iris.cluster.types import AcceleratorType, CapacityType, Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from rigging.connect import proxy_path
from rigging.timing import Duration, ExponentialBackoff

from .conftest import (
    DEFAULT_CONFIG,
    MARIN_ROOT,
    ClusterCapabilities,
    IrisTestCluster,
    _add_coscheduling_group,
    _NoOpPage,
    assert_visible,
    dashboard_goto,
    discover_capabilities,
    wait_for_dashboard_ready,
)
from .helpers import TestJobs

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.requires_cluster


# ---------------------------------------------------------------------------
# Smoke-test cluster configuration helpers
# ---------------------------------------------------------------------------


def _add_cpu_group(config: IrisClusterConfig, num_workers: int = 4) -> None:
    """CPU scale group with multiple workers for scheduling diversity and bin-packing."""
    config.scale_groups["local-cpu"] = ScaleGroupConfig(
        name="local-cpu",
        num_vms=1,
        buffer_slices=num_workers,
        max_slices=num_workers,
        resources=ScaleGroupResources(
            cpu_millicores=8000,
            memory_bytes=16 * 1024**3,
            disk_bytes=50 * 1024**3,
            device_type=AcceleratorType.CPU,
            capacity_type=CapacityType.ON_DEMAND,
        ),
        slice_template=SliceConfig(local=LocalSliceConfig()),
    )


def _add_coscheduling_group_4vm(config: IrisClusterConfig) -> None:
    """4-VM TPU coscheduling group for large-job tests."""
    config.scale_groups["tpu_cosched_4"] = ScaleGroupConfig(
        name="tpu_cosched_4",
        num_vms=4,
        buffer_slices=1,
        max_slices=1,
        resources=ScaleGroupResources(
            cpu_millicores=128000,
            memory_bytes=128 * 1024**3,
            disk_bytes=1024 * 1024**3,
            device_type=AcceleratorType.TPU,
            device_variant="v5litepod-32",
            capacity_type=CapacityType.PREEMPTIBLE,
        ),
        slice_template=SliceConfig(num_vms=4, local=LocalSliceConfig()),
    )


# Total local-mode workers:
# 2 (local-cpu) + 2 (cosched_2) + 4 (cosched_4) = 8
SMOKE_WORKER_COUNT = 8


def _make_smoke_config() -> IrisClusterConfig:
    """Build a local config with CPU and TPU (coscheduling) workers."""
    config = load_config(DEFAULT_CONFIG)
    config.scale_groups.clear()
    _add_cpu_group(config, num_workers=2)
    _add_coscheduling_group(config)
    _add_coscheduling_group_4vm(config)
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
        log_client = LogServiceClientSync(
            address=f"{controller_url.rstrip('/')}{proxy_path(LOG_SERVER_ENDPOINT_NAME)}",
            timeout_ms=30000,
        )
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
        log_client.close()
        controller_client.close()
        return

    config = _make_smoke_config()
    with connect_cluster(config) as url:
        client = IrisClient.remote(url, workspace=MARIN_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        log_client = LogServiceClientSync(
            address=f"{url.rstrip('/')}{proxy_path(LOG_SERVER_ENDPOINT_NAME)}",
            timeout_ms=30000,
        )
        tc = IrisTestCluster(url=url, client=client, controller_client=controller_client, log_client=log_client)
        tc.wait_for_workers(SMOKE_WORKER_COUNT, timeout=60)
        yield tc
        log_client.close()
        controller_client.close()


@pytest.fixture(scope="module")
def smoke_page(smoke_cluster):
    """Module-scoped Playwright page for smoke dashboard tests."""
    try:
        import playwright.sync_api as pw  # noqa: PLC0415  # optional dep: playwright

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


def _await_stable_screenshot(page, check: str, *, arg=None) -> None:
    """Wait for a screenshot-readiness predicate, settle briefly, then re-verify.

    Dashboard pages render structural content only after their first RPC resolves,
    and an SPA route swap (lazy-imported components) can leave the previous page's
    DOM mounted while the next chunk loads. The settle + re-verify catches a
    predicate that passes on such a transient state and then flips back, so the
    screenshot lands on the stable loaded page. Timing lives here so both detail
    pages share one cadence.
    """
    page.wait_for_function(check, arg=arg, timeout=15000)
    page.wait_for_timeout(250)
    page.wait_for_function(check, arg=arg, timeout=5000)


def _wait_for_worker_detail_screenshot_ready(page, worker_id: str) -> None:
    # WorkerDetail.vue uniquely nulls `data` in its workerId watch, so a late
    # re-fire can flip the page back to the "Loading worker..." overlay after a
    # naive wait passes. Anchor on h3 sections that only render in the
    # v-else-if="data" branch, then settle + re-verify to catch the transient.
    check = """
        (workerId) => {
            const text = document.body.textContent || "";
            const routeReady = decodeURIComponent(window.location.hash) === `#/worker/${workerId}`;
            const headings = Array.from(document.querySelectorAll("h3"))
                .map((heading) => (heading.textContent || "").trim().toLowerCase());
            return routeReady
                && !text.includes("Loading worker...")
                && text.includes(workerId)
                && text.includes("Healthy")
                && headings.includes("identity")
                && headings.includes("task history");
        }
    """
    _await_stable_screenshot(page, check, arg=worker_id)


def _wait_for_job_detail_screenshot_ready(page, job_id: str) -> None:
    page.wait_for_function(
        """
        (jobId) => {
            const text = document.body.textContent || "";
            const routeReady = decodeURIComponent(window.location.hash) === `#/job/${jobId}`;
            const headings = Array.from(document.querySelectorAll("h3"))
                .map((heading) => (heading.textContent || "").trim().toLowerCase());
            const taskRowReady = Array.from(document.querySelectorAll("table tbody tr"))
                .some((row) => (row.textContent || "").includes("Succeeded"));
            const pageHeight = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
            return routeReady
                && !text.includes("Loading...")
                && text.includes("Job Status")
                && text.includes("Task Summary")
                && headings.includes("tasks")
                && headings.includes("job logs")
                && taskRowReady
                && pageHeight > window.innerHeight;
        }
        """,
        arg=job_id,
        timeout=10000,
    )


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
    """Landing page groups jobs by owner; drilling into the owner shows states."""
    quick = smoke_cluster.submit(TestJobs.quick, "smoke-simple")
    failed = smoke_cluster.submit(TestJobs.fail, "smoke-failed")
    running = smoke_cluster.submit(TestJobs.sleep, "smoke-running", 300)

    smoke_cluster.wait(quick, timeout=smoke_cluster.job_timeout)
    smoke_cluster.wait(failed, timeout=smoke_cluster.job_timeout)
    smoke_cluster.wait_for_state(running, job_pb2.JOB_STATE_RUNNING, timeout=smoke_cluster.job_timeout)

    user = quick.job_id.user

    # Landing page is the per-owner overview, not a flat job list.
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/")
    wait_for_dashboard_ready(smoke_page)
    assert_visible(smoke_page, f"text={user}")

    # Drill into this owner to see their individual jobs and states.
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/#/?user={user}")
    wait_for_dashboard_ready(smoke_page)
    for name in ["smoke-simple", "smoke-failed", "smoke-running"]:
        assert_visible(smoke_page, f"text={name}")
    # The Cluster column is always rendered, blank for local jobs — a
    # single-cluster smoke deployment shows the header with only "—" cells and
    # never a peer annotation.
    assert_visible(smoke_page, "th:has-text('Cluster')")
    smoke_screenshot(
        "jobs-tab",
        f"Jobs for user {user}: smoke-simple (succeeded), smoke-failed (failed), and smoke-running (running)",
    )

    smoke_cluster.kill(running)


def _parent_with_two_children():
    """Parent callable that submits two child jobs and waits for both."""

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

    # Route through the owner overview first so the subsequent drill-in is a
    # genuine hash change (the smoke page is shared module-scope and may already
    # be parked on this owner's view from an earlier test).
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/")
    wait_for_dashboard_ready(smoke_page)
    # Open the owner's scoped job list (the landing page groups by owner).
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/#/?user={parent.job_id.user}")
    wait_for_dashboard_ready(smoke_page)
    assert_visible(smoke_page, "text=smoke-expand-parent")

    # The parent row exposes a keyboard-accessible expand toggle.
    row = smoke_page.locator("tr", has_text="smoke-expand-parent")
    expand_btn = row.get_by_role("button", name="Expand children")
    expand_btn.click()

    # After clicking, children should appear (wait for the child names to render)
    smoke_page.wait_for_function(
        "() => document.body.textContent.includes('child-a') && " "document.body.textContent.includes('child-b')",
        timeout=10000,
    )

    # Once expanded, the toggle flips to a collapse affordance.
    row.get_by_role("button", name="Collapse children").wait_for(timeout=5000)

    smoke_screenshot("job-expand", "Jobs tab with expanded parent showing child-a and child-b indented beneath")


def test_dashboard_job_detail(smoke_cluster, smoke_page, smoke_screenshot):
    """SUCCEEDED job detail page."""
    job = smoke_cluster.submit(TestJobs.quick, "smoke-detail")
    smoke_cluster.wait(job, timeout=smoke_cluster.job_timeout)

    job_id = job.job_id.to_wire()
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/job/{job_id}")
    wait_for_dashboard_ready(smoke_page)
    _wait_for_job_detail_screenshot_ready(smoke_page, job_id)
    smoke_screenshot(
        "job-detail", "Job detail page for succeeded job with state badge, task table, and job-level log viewer"
    )


def _wait_for_task_log_marker(
    cluster: IrisTestCluster, task_id: str, attempt_id: int, marker: str, *, timeout: float = 60.0
) -> None:
    """Poll the log server until the task attempt's EXACT-source logs contain ``marker``.

    Log shipping is asynchronous: a worker flushes buffered lines after the task
    exits, so a just-completed task's logs are not immediately queryable. This
    mirrors the LogViewer's own EXACT ``{task}:{attempt}`` query so a dashboard test
    can wait for the logs to land before asserting on a single page load.
    """
    source = f"{task_id}:{attempt_id}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        request = logging_pb2.FetchLogsRequest(
            source=source, match_scope=logging_pb2.MATCH_SCOPE_EXACT, tail=True, max_lines=1000
        )
        if any(marker in entry.data for entry in cluster.log_client.fetch_logs(request).entries):
            return
        time.sleep(0.5)
    raise AssertionError(f"log marker {marker!r} for {source} not queryable within {timeout:.0f}s")


def test_dashboard_task_logs(smoke_cluster, verbose_job, smoke_page, smoke_screenshot):
    """Task logs show lines and substring filter on the task detail page."""
    task_status = smoke_cluster.task_status(verbose_job)
    task_id = task_status.task_id
    job_id = verbose_job.job_id.to_wire()

    # The LogViewer issues a single EXACT {task}:{attempt} fetch on mount and only
    # re-polls every 30s, so its first fetch must not race the worker's asynchronous
    # post-completion log flush. Wait for the attempt's logs to be queryable first
    # (as test_log_levels_populated does) so the page renders them on load.
    _wait_for_task_log_marker(smoke_cluster, task_id, task_status.current_attempt_id, "DONE: all lines emitted")

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

    # This job logs plenty of ERROR-level lines but never crashes. Exception
    # navigation keys off tracebacks and fatal banners, not severity, so it must
    # find nothing here.
    assert_visible(smoke_page, "text=No exceptions")

    # Search marks matching lines in place. "validation failed" only appears in
    # ERROR lines, so the INFO lines around them must survive — that is the whole
    # point of search being distinct from filter.
    search_input = "input[placeholder^='Search loaded lines']"
    smoke_page.fill(search_input, "validation failed")
    smoke_page.wait_for_function(
        "() => document.querySelectorAll('mark').length > 0 && "
        "document.body.textContent.includes('processing data batch')",
        timeout=5000,
    )
    smoke_screenshot(
        "task-logs-searched",
        "Task detail page with a log search box populated; matching text is highlighted in place "
        "and non-matching log lines are still visible around the highlights.",
    )

    # The filter re-queries the server and drops non-matching lines entirely. It
    # applies on Enter, not on every keystroke.
    filter_input = "input[placeholder^='Filter:']"
    smoke_page.fill(filter_input, "validation failed")
    smoke_page.press(filter_input, "Enter")
    smoke_page.wait_for_function(
        "() => document.body.textContent.includes('validation failed') && "
        "!document.body.textContent.includes('processing data batch')",
        timeout=5000,
    )
    smoke_screenshot(
        "task-logs-filtered",
        "Task detail page with log filter input populated and filtered log lines visible in the log viewer.",
    )


def test_dashboard_jump_to_exception(smoke_cluster, smoke_page, smoke_screenshot):
    """A crashed task's log viewer finds and steps to the traceback."""
    failed = smoke_cluster.submit(TestJobs.fail, "smoke-exception")
    smoke_cluster.wait(failed, timeout=smoke_cluster.job_timeout)

    task_status = smoke_cluster.task_status(failed)
    _wait_for_task_log_marker(smoke_cluster, task_status.task_id, task_status.current_attempt_id, "Traceback")

    dashboard_goto(
        smoke_page,
        f"{smoke_cluster.url}/job/{failed.job_id.to_wire()}/task/{task_status.task_id}",
    )
    wait_for_dashboard_ready(smoke_page)

    # The control counts failures, not ERROR-level lines, and a whole traceback
    # collapses to a single stop.
    jump = smoke_page.locator("button", has_text="Jump to exception")
    jump.wait_for(timeout=10000)
    jump.click()
    assert_visible(smoke_page, "text=/Exception 1 \\/ \\d+/")
    smoke_screenshot(
        "task-logs-exception",
        "Task detail page for a failed task; the log viewer's exception control reads 'Exception 1 / N' "
        "and a highlighted traceback line is visible in the log output.",
    )


def test_dashboard_constraints(smoke_cluster, smoke_page, smoke_screenshot):
    """Constraint chips rendered on job detail."""
    # Use soft constraints to avoid submit-time routing feasibility rejection;
    # the test only checks that constraint chips render on the dashboard.
    constraints = [
        Constraint.create(key="region", op=ConstraintOp.EQ, value="local", mode=job_pb2.CONSTRAINT_MODE_PREFERRED),
        Constraint.create(key="env-tag", op=ConstraintOp.EXISTS),
        Constraint.create(
            key="device-variant",
            op=ConstraintOp.IN,
            values=["v5p-8", "v6e-4"],
            mode=job_pb2.CONSTRAINT_MODE_PREFERRED,
        ),
    ]
    with smoke_cluster.launched_job(TestJobs.quick, "smoke-constraints", constraints=constraints) as job:
        time.sleep(3)

        dashboard_goto(smoke_page, f"{smoke_cluster.url}/job/{job.job_id.to_wire()}")
        wait_for_dashboard_ready(smoke_page)

        # Placement constraints live in the collapsible "Scheduling" pane, which
        # stays collapsed unless the job has pending tasks. Open it to inspect the chips.
        smoke_page.wait_for_function(
            "() => document.body.textContent.includes('Scheduling')",
            timeout=5000,
        )
        smoke_page.evaluate(
            "() => [...document.querySelectorAll('details')]"
            ".filter(d => d.querySelector('summary')?.textContent.includes('Scheduling'))"
            ".forEach(d => { d.open = true })"
        )
        assert_visible(smoke_page, "text=region")
        smoke_screenshot(
            "constraints", "Job detail Scheduling pane showing constraint chips for region, env-tag, and device-variant"
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
    """Worker detail page shows info, task history, metric cards, and links each
    task id to its task-detail page."""
    if not capabilities.has_workers:
        pytest.skip("No persistent workers")
    job = smoke_cluster.submit(TestJobs.quick, "smoke-worker-detail")
    smoke_cluster.wait(job, timeout=smoke_cluster.job_timeout)

    task_status = smoke_cluster.task_status(job)
    worker_id = task_status.worker_id
    assert worker_id

    dashboard_goto(smoke_page, f"{smoke_cluster.url}/worker/{worker_id}")
    wait_for_dashboard_ready(smoke_page)
    _wait_for_worker_detail_screenshot_ready(smoke_page, worker_id)

    # The Task History task id links to the task-detail route — the
    # "links to tasks" contract for this page.
    smoke_page.wait_for_selector("a[href*='/task/']", timeout=10000)
    href = smoke_page.locator("a[href*='/task/']").first.get_attribute("href")
    assert href is not None and "/job/" in href and "/task/" in href

    smoke_screenshot(
        "worker-detail",
        "Worker detail page with identity info, health badge, metric sparklines, "
        "and task history with per-task resource columns and task-id links",
    )


def _wait_for_capacity_screenshot_ready(page) -> None:
    # CapacityTab.vue shows only a "Loading capacity & scheduling…" spinner until its
    # first RPC resolves. Route components are lazy-imported, so during the SPA swap
    # the previously-viewed page (e.g. worker detail, which renders "Scale Group" +
    # the "local-cpu" group name) is still mounted while the capacity chunk loads.
    # A body-text wait keyed on those strings false-positives on that stale DOM, so
    # the screenshot then catches the spinner. Anchor on the route hash plus section
    # headings that only render in the loaded (v-else) branch so the match can't be
    # satisfied by another page. Substring-match the headings (not exact) so dynamic
    # counts (e.g. "Pending Jobs (3)") and unicode dashes don't break the wait.
    check = """
        () => {
            const text = document.body.textContent || "";
            const routeReady = decodeURIComponent(window.location.hash) === "#/capacity";
            const headings = Array.from(document.querySelectorAll("h3"))
                .map((heading) => (heading.textContent || "").trim().toLowerCase());
            const has = (needle) => headings.some((heading) => heading.includes(needle));
            return routeReady
                && !text.includes("Loading capacity & scheduling")
                && has("pools")
                && has("pending jobs")
                && has("users & quotas");
        }
    """
    _await_stable_screenshot(page, check)


def test_dashboard_capacity_tab(smoke_cluster, smoke_page, smoke_screenshot):
    """Capacity & Scheduling tab shows scale groups, pending jobs, and user quotas."""
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/capacity")
    wait_for_dashboard_ready(smoke_page)
    _wait_for_capacity_screenshot_ready(smoke_page)
    smoke_screenshot("capacity-tab", "Capacity & Scheduling tab: pools, demand, pending jobs, quotas")


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


def _wait_for_backends_tab_ready(page) -> None:
    # The combined execution-targets tab renders backend (and peer) cards only
    # after ListBackends resolves. Anchor on the route hash plus the "N backend(s)"
    # count subtitle in the tab's h2 — that subtitle renders only once the RPC
    # resolves, unlike the persistent "Backends"/"Workers" nav links, so it marks
    # the tab content (not just the shell) as loaded.
    check = """
        () => {
            const routeReady = decodeURIComponent(window.location.hash) === "#/backends";
            const heading = Array.from(document.querySelectorAll("h2"))
                .find((h) => (h.textContent || "").trim().startsWith("Backends"));
            const loaded = !!heading && /\\d+\\s+backend/.test(heading.textContent || "");
            return routeReady && loaded;
        }
    """
    _await_stable_screenshot(page, check)


def test_dashboard_backends_tab(smoke_cluster, smoke_page, smoke_screenshot):
    """Combined execution-targets tab renders local backends; no peers configured.

    The smoke cluster has no federation peers, so this asserts graceful-empty
    rendering: backend cards/rows show and no peer card ("peer" tag) appears.
    """
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/backends")
    wait_for_dashboard_ready(smoke_page)
    # The readiness gate requires the "N backend(s)" subtitle, so reaching here
    # already proves the backend cards rendered. With no peers configured, the
    # roster is empty: no peer tag.
    _wait_for_backends_tab_ready(smoke_page)
    if not isinstance(smoke_page, _NoOpPage):
        from playwright.sync_api import expect  # noqa: PLC0415  # optional dep: playwright

        expect(smoke_page.get_by_text("peer", exact=True)).to_have_count(0)
    smoke_screenshot(
        "backends-tab",
        "Execution-targets tab: local backend cards, no federation peers configured",
    )


def test_dashboard_backends_tab_with_peer(smoke_cluster, smoke_page, smoke_screenshot):
    """A configured peer renders as a card alongside backends (route-mocked ListPeers).

    The smoke cluster has no real peers, so ListPeers is stubbed at the browser to
    prove the peer card renders: health dot, "peer" tag, aggregated device caps,
    worker/task counts, and an inward link to the parent's cluster-filtered jobs
    (never an outbound link to the peer's own dashboard).
    """
    if isinstance(smoke_page, _NoOpPage):
        pytest.skip("Playwright unavailable")

    import json  # noqa: PLC0415

    peer_body = json.dumps(
        {
            "peers": [
                {
                    "peerId": "cw-smoke-peer",
                    "controllerAddress": "https://cw.example:8443",
                    "reachable": True,
                    "lastContactMs": "1720000000000",
                    "activeFederatedJobs": 2,
                    "backends": [
                        {
                            "backendId": "cw-h100",
                            "name": "cw-h100",
                            "kind": "kubernetes",
                            "capabilities": ["gpu"],
                            "advertisedAttributes": {"accelerator": {"values": ["H100"]}},
                            "scaleGroups": [],
                            "workerCount": 4,
                            "pendingTaskCount": 1,
                            "runningTaskCount": 3,
                            "hasAutoscaler": True,
                            "capacityHealth": {},
                        }
                    ],
                }
            ]
        }
    )

    def _fulfill_peers(route):
        route.fulfill(status=200, content_type="application/json", body=peer_body)

    smoke_page.route("**/ListPeers", _fulfill_peers)
    try:
        dashboard_goto(smoke_page, f"{smoke_cluster.url}/backends")
        # The prior test already sits on #/backends, and a same-hash goto does not
        # reload — the tab would keep its peerless roster and never re-issue
        # ListPeers under the mock. Reload to force a fresh mount + refetch.
        smoke_page.reload()
        wait_for_dashboard_ready(smoke_page)
        _wait_for_backends_tab_ready(smoke_page)
        smoke_page.wait_for_function(
            "() => document.body.textContent.includes('cw-smoke-peer')",
            timeout=10000,
        )
        # Target the peer card heading, not the peer's (hidden) <option> in the
        # scope <select>, which also carries the peer id.
        assert_visible(smoke_page, "h3:has-text('cw-smoke-peer')")
        # The peer links inward to the parent's cluster-filtered jobs, not out to
        # the peer's own dashboard (which users can't reach).
        assert_visible(smoke_page, "a[href*='cluster=cw-smoke-peer']")
        smoke_screenshot(
            "backends-tab-peer",
            "Execution-targets tab with a federation peer card: health dot, peer tag, "
            "aggregated caps, worker/task counts, and an inward link to the parent's "
            "cluster-filtered jobs",
        )
    finally:
        smoke_page.unroute("**/ListPeers", _fulfill_peers)


def test_dashboard_job_detail_with_logs(smoke_cluster, verbose_job, smoke_page, smoke_screenshot):
    """Job detail page shows combined log viewer for all tasks."""
    job_id = verbose_job.job_id.to_wire()
    # Same asynchronous-log-shipping race as test_dashboard_task_logs: wait for the
    # task's logs to land before the single page-load fetch (EXACT is a superset of
    # the job view's PREFIX query).
    task_status = smoke_cluster.task_status(verbose_job)
    _wait_for_task_log_marker(
        smoke_cluster, task_status.task_id, task_status.current_attempt_id, "DONE: all lines emitted"
    )
    dashboard_goto(smoke_page, f"{smoke_cluster.url}/job/{job_id}")
    wait_for_dashboard_ready(smoke_page)
    _wait_for_job_detail_screenshot_ready(smoke_page, job_id)
    smoke_page.wait_for_function(
        "() => document.body.textContent.includes('DONE: all lines emitted')",
        timeout=10000,
    )
    smoke_screenshot(
        "job-detail-logs",
        "Job detail page showing task table and combined job-level log viewer with log lines",
    )


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


def test_cancel_job_releases_resources(smoke_cluster):
    """Cancelling a running job decommits worker resources so new jobs can schedule.

    Submits a resource-heavy job, cancels it, then verifies a second job with
    the same resource requirements succeeds — proving the worker's committed
    resources were fully released by cancel_job().

    Regression test for #3553.
    """
    # Use most of a single worker's CPU so the followup job can't schedule on
    # that worker until the heavy job is cancelled. Workers now advertise their
    # scale-group declared CPU (local-cpu: 8 cores; cloud TPU VMs: 128) rather
    # than a probed host count — pick a value that fills most of one worker.
    heavy_cpu = 8 if smoke_cluster.is_cloud else 7

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
        request = logging_pb2.FetchLogsRequest(
            source=f"{task_id}:",
            match_scope=logging_pb2.MATCH_SCOPE_PREFIX,
        )
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

    request = logging_pb2.FetchLogsRequest(
        source=f"{task_id}:",
        match_scope=logging_pb2.MATCH_SCOPE_PREFIX,
        min_level="WARNING",
    )
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


# ============================================================================
# Checkpoint / restore
# ============================================================================


@pytest.mark.timeout(120)
def test_checkpoint_restore():
    """Controller restart resumes from checkpoint: completed jobs visible, cluster functional.

    Uses a dedicated LocalCluster (not the shared smoke_cluster). The persistent DB dir
    (held by LocalCluster across stop/start) preserves checkpoint state.
    Phase 1 — run a job and write a checkpoint.
    Phase 2 — restart the controller and verify the job is still SUCCEEDED
              and the cluster can accept new work.
    """

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
    status = smoke_cluster.wait(job, timeout=smoke_cluster.job_timeout * 4)
    assert status.state == job_pb2.JOB_STATE_SUCCEEDED


# ============================================================================
# Workdir file offload (large files externalized to blob store)
# ============================================================================

OFFLOAD_FILE_SIZE = 32 * 1024  # 32KB — exceeds the 10KB offload threshold


def test_workdir_file_offload(smoke_cluster):
    """A job with a workdir file above the offload threshold succeeds after blob-store offloading."""
    entrypoint = Entrypoint.from_callable(TestJobs.verify_workdir_file, "large_payload.bin", OFFLOAD_FILE_SIZE)
    entrypoint.workdir_files["large_payload.bin"] = b"\xab" * OFFLOAD_FILE_SIZE
    job = smoke_cluster.client.submit(
        entrypoint=entrypoint,
        name=f"smoke-offload-{uuid.uuid4().hex[:8]}",
        resources=ResourceSpec(cpu=1, memory="1g"),
    )
    status = smoke_cluster.wait(job, timeout=smoke_cluster.job_timeout)
    assert status.state == job_pb2.JOB_STATE_SUCCEEDED
