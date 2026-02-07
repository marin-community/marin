#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture dashboard screenshots for documentation and visual review.

Boots a local Iris cluster, submits jobs in various states (succeeded, failed,
running, pending/unschedulable, coscheduled), then screenshots each dashboard
tab and selected job detail pages using Playwright.

Usage:
    uv run lib/iris/scripts/screenshot-dashboard.py
    uv run lib/iris/scripts/screenshot-dashboard.py --output-dir /tmp/screenshots
    uv run lib/iris/scripts/screenshot-dashboard.py --config lib/iris/examples/demo.yaml
    uv run lib/iris/scripts/screenshot-dashboard.py --stay-open
"""

import logging
import time
from pathlib import Path

import click
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright

from iris.client.client import IrisClient
from iris.cluster.types import (
    CoschedulingConfig,
    Constraint,
    ConstraintOp,
    JobName,
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
)
from iris.cluster.vm.cluster_manager import ClusterManager
from iris.cluster.vm.config import load_config, make_local_config
from iris.rpc import cluster_pb2

IRIS_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = IRIS_ROOT / "examples" / "demo.yaml"

DASHBOARD_TABS = ["jobs", "workers", "endpoints", "vms", "autoscaler", "transactions"]

logger = logging.getLogger(__name__)


# =============================================================================
# Demo job functions (serialized via cloudpickle to workers)
# =============================================================================


def _succeeded_job():
    print("Snapshot succeeded job completed.")
    return "ok"


def _failed_job():
    raise RuntimeError("Intentional failure for screenshot demo")


def _slow_job():
    import time as _time

    _time.sleep(120)
    return "done"


def _building_job():
    """Simple job for testing BUILDING state (used with chaos delay injection)."""
    return "ok"


def _retry_job():
    """Job that fails on first attempt, succeeds on retry (for attempt visibility demo)."""
    from iris.cluster.client import get_job_info

    info = get_job_info()
    if info is None:
        raise RuntimeError("JobInfo not available")

    attempt_id = info.attempt_id
    print(f"Retry job: attempt_id={attempt_id}")

    if attempt_id == 0:
        raise RuntimeError("Intentional failure on first attempt for retry demo")
    return "success after retry"


def _pipeline_parent_job():
    """Parent job that submits child jobs to demonstrate hierarchical tree view."""
    import time as _time

    from iris.client.client import iris_ctx
    from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec

    ctx = iris_ctx()
    resources = ResourceSpec(cpu=1, memory="1g")
    env = EnvironmentSpec()

    # Submit child jobs - these will appear as children of this job in the tree view
    def _child_succeeded():
        print("Child job succeeded")
        return "ok"

    def _child_slow():
        import time

        time.sleep(120)
        return "done"

    # Submit multiple child jobs
    child1 = ctx.client.submit(
        Entrypoint.from_callable(_child_succeeded),
        "tokenize",
        resources,
        environment=env,
    )
    _child2 = ctx.client.submit(
        Entrypoint.from_callable(_child_slow),
        "train",
        resources,
        environment=env,
    )
    child3 = ctx.client.submit(
        Entrypoint.from_callable(_child_succeeded),
        "evaluate",
        resources,
        environment=env,
    )

    # Wait for the quick ones to complete
    child1.wait(timeout=60, raise_on_failure=True)
    child3.wait(timeout=60, raise_on_failure=True)

    # Keep running so the train job stays visible
    _time.sleep(60)
    return "pipeline complete"


# =============================================================================
# Job submission
# =============================================================================


def submit_demo_jobs(client: IrisClient) -> dict[str, str]:
    """Submit a variety of jobs to populate the dashboard.

    Returns a mapping of logical name -> job_id for later reference.
    """
    from iris.chaos import enable_chaos

    job_ids: dict[str, str] = {}

    # 1. Building job (with chaos delay)
    # Enable chaos to inject a 30-second delay during the BUILDING state
    # max_failures=1 ensures only the first job hits the delay
    enable_chaos("worker.building_delay", delay_seconds=30.0, max_failures=1)

    job = client.submit(
        entrypoint=Entrypoint.from_callable(_building_job),
        name="snapshot-building",
        resources=ResourceSpec(cpu=1, memory="2g"),
        environment=EnvironmentSpec(),
    )
    job_ids["building"] = str(job.job_id)
    logger.info("Submitted building job (with 30s chaos delay): %s", job.job_id)

    # 2. Succeeded job
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_succeeded_job),
        name="snapshot-succeeded",
        resources=ResourceSpec(cpu=2, memory="4g"),
        environment=EnvironmentSpec(),
    )
    job_ids["succeeded"] = str(job.job_id)
    logger.info("Submitted succeeded job: %s", job.job_id)

    # 3. Failed job
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_failed_job),
        name="snapshot-failed",
        resources=ResourceSpec(cpu=2, memory="4g"),
        environment=EnvironmentSpec(),
    )
    job_ids["failed"] = str(job.job_id)
    logger.info("Submitted failed job: %s", job.job_id)

    # 4. Running/slow job
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_slow_job),
        name="snapshot-running",
        resources=ResourceSpec(cpu=2, memory="4g"),
        environment=EnvironmentSpec(),
    )
    job_ids["running"] = str(job.job_id)
    logger.info("Submitted running job: %s", job.job_id)

    # 5. Second running job (more realistic)
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_slow_job),
        name="snapshot-running-2",
        resources=ResourceSpec(cpu=1, memory="2g"),
        environment=EnvironmentSpec(),
    )
    job_ids["running-2"] = str(job.job_id)
    logger.info("Submitted second running job: %s", job.job_id)

    # 6. Pending unschedulable job (constraint for nonexistent TPU)
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_succeeded_job),
        name="snapshot-pending-constraint",
        resources=ResourceSpec(cpu=4, memory="8g"),
        environment=EnvironmentSpec(),
        constraints=[
            Constraint(key="tpu-name", op=ConstraintOp.EQ, value="nonexistent-tpu-xyz"),
        ],
    )
    job_ids["pending-constraint"] = str(job.job_id)
    logger.info("Submitted pending/unschedulable job: %s", job.job_id)

    # 7. Coscheduled multi-replica job (also unschedulable)
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_succeeded_job),
        name="snapshot-coscheduled",
        resources=ResourceSpec(cpu=1, memory="2g"),
        environment=EnvironmentSpec(),
        constraints=[
            Constraint(key="tpu-name", op=ConstraintOp.EQ, value="nonexistent-tpu-xyz"),
        ],
        coscheduling=CoschedulingConfig(group_by="tpu-name"),
        replicas=4,
    )
    job_ids["coscheduled"] = str(job.job_id)
    logger.info("Submitted coscheduled job: %s", job.job_id)

    # 8. Job targeting zero-quota scale group (quota exhaustion scenario)
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_succeeded_job),
        name="snapshot-quota-exhausted",
        resources=ResourceSpec(cpu=2, memory="4g"),
        environment=EnvironmentSpec(),
        constraints=[
            Constraint(key="scale-group", op=ConstraintOp.EQ, value="tpu_v6e_256_no_quota"),
        ],
    )
    job_ids["quota-exhausted"] = str(job.job_id)
    logger.info("Submitted quota-exhausted job: %s", job.job_id)

    # 9. Job targeting disabled scale group
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_succeeded_job),
        name="snapshot-disabled-group",
        resources=ResourceSpec(cpu=1, memory="2g"),
        environment=EnvironmentSpec(),
        constraints=[
            Constraint(key="scale-group", op=ConstraintOp.EQ, value="gpu_disabled"),
        ],
    )
    job_ids["disabled-group"] = str(job.job_id)
    logger.info("Submitted disabled-group job: %s", job.job_id)

    # 10. Large coscheduled job that needs more workers than available
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_succeeded_job),
        name="snapshot-insufficient-capacity",
        resources=ResourceSpec(cpu=1, memory="1g"),
        environment=EnvironmentSpec(),
        coscheduling=CoschedulingConfig(group_by="scale-group"),
        replicas=100,
    )
    job_ids["insufficient-capacity"] = str(job.job_id)
    logger.info("Submitted insufficient-capacity job: %s", job.job_id)

    # 11. Killed job — submit a slow job, then terminate it
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_slow_job),
        name="snapshot-killed",
        resources=ResourceSpec(cpu=1, memory="2g"),
        environment=EnvironmentSpec(),
    )
    job_ids["killed"] = str(job.job_id)
    logger.info("Submitted killed job (will terminate after brief delay): %s", job.job_id)

    # 12. Retry job — fails on first attempt, succeeds on second (shows attempt visibility)
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_retry_job),
        name="snapshot-retry",
        resources=ResourceSpec(cpu=1, memory="2g"),
        environment=EnvironmentSpec(),
        max_retries_failure=1,
    )
    job_ids["retry"] = str(job.job_id)
    logger.info("Submitted retry job (will fail then retry): %s", job.job_id)

    # 13. Hierarchical pipeline job — parent that submits child jobs (demonstrates tree view)
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_pipeline_parent_job),
        name="snapshot-pipeline",
        resources=ResourceSpec(cpu=1, memory="1g"),
        environment=EnvironmentSpec(),
    )
    job_ids["pipeline"] = str(job.job_id)
    logger.info("Submitted pipeline parent job (will spawn children): %s", job.job_id)

    return job_ids


def wait_for_terminal_jobs(client: IrisClient, job_ids: dict[str, str], timeout: float = 60.0):
    """Wait for the succeeded, failed, and killed jobs to reach terminal states.

    Also verifies that the building job shows BUILDING state with appropriate
    task status message before eventually completing.
    """
    terminal_states = (
        cluster_pb2.JOB_STATE_SUCCEEDED,
        cluster_pb2.JOB_STATE_FAILED,
        cluster_pb2.JOB_STATE_KILLED,
        cluster_pb2.JOB_STATE_WORKER_FAILED,
    )

    # Wait for succeeded, failed, and retry jobs first
    deadline = time.monotonic() + timeout
    for name in ["succeeded", "failed", "retry"]:
        jid = JobName.from_wire(job_ids[name])
        logger.info("Waiting for %s job (%s) to finish...", name, jid)
        while time.monotonic() < deadline:
            status = client.status(jid)
            if status.state in terminal_states:
                state_name = cluster_pb2.JobState.Name(status.state)
                logger.info("  %s -> %s", name, state_name)
                break
            time.sleep(0.5)
        else:
            logger.warning("  %s job did not finish within %.0fs", name, timeout)

    # Verify building job is in BUILDING state with appropriate task status
    # Wait for the building job to enter BUILDING state (may take a few seconds for worker dispatch)
    building_jid = JobName.from_wire(job_ids["building"]) if job_ids.get("building") else None
    if building_jid:
        logger.info("Waiting for building job (%s) to enter BUILDING state...", building_jid)
        build_state_deadline = time.monotonic() + 10.0
        while time.monotonic() < build_state_deadline:
            status = client.status(building_jid)
            if status.tasks:
                task_status = status.tasks[0]
                if task_status.state == cluster_pb2.TASK_STATE_BUILDING:
                    logger.info("  building task entered BUILDING state")
                    break
            time.sleep(0.5)
        else:
            status = client.status(building_jid)
            if status.tasks:
                task_status = status.tasks[0]
                task_state_name = cluster_pb2.TaskState.Name(task_status.state)
                logger.warning("  Building job did not enter BUILDING state within timeout (state: %s)", task_state_name)

    # Wait for running job's task to actually reach RUNNING state (not just ASSIGNED)
    running_jid = JobName.from_wire(job_ids["running"]) if job_ids.get("running") else None
    if running_jid:
        logger.info("Waiting for running job (%s) task to enter RUNNING state...", running_jid)
        run_deadline = time.monotonic() + 15.0
        while time.monotonic() < run_deadline:
            status = client.status(running_jid)
            if status.tasks and status.tasks[0].state == cluster_pb2.TASK_STATE_RUNNING:
                logger.info("  running task entered RUNNING state")
                break
            time.sleep(0.5)
        else:
            status = client.status(running_jid)
            if status.tasks:
                task_state_name = cluster_pb2.TaskState.Name(status.tasks[0].state)
                logger.warning(
                    "  Running job task did not enter RUNNING state within timeout (state: %s)", task_state_name
                )

    # Wait for pipeline job to spawn its child jobs (demonstrates tree view)
    pipeline_jid = JobName.from_wire(job_ids["pipeline"]) if job_ids.get("pipeline") else None
    if pipeline_jid:
        logger.info("Waiting for pipeline job (%s) to spawn child jobs...", pipeline_jid)
        pipeline_deadline = time.monotonic() + 30.0
        child_jobs_found = False
        while time.monotonic() < pipeline_deadline:
            # Check if child jobs exist by looking for jobs with pipeline as parent
            all_jobs = client.list_jobs()
            child_jobs = [j for j in all_jobs if j.name.startswith(str(pipeline_jid) + "/")]
            if len(child_jobs) >= 2:  # At least tokenize and train should exist
                logger.info("  Found %d child jobs under pipeline", len(child_jobs))
                child_jobs_found = True
                break
            time.sleep(0.5)
        if not child_jobs_found:
            logger.warning("  Pipeline child jobs did not appear within timeout")

    # Kill the "killed" job after a brief delay so it has time to start
    killed_jid = JobName.from_wire(job_ids["killed"]) if job_ids.get("killed") else None
    if killed_jid:
        time.sleep(1.0)
        logger.info("Terminating killed job (%s)...", killed_jid)
        client.terminate(killed_jid)
        while time.monotonic() < deadline:
            status = client.status(killed_jid)
            if status.state in terminal_states:
                state_name = cluster_pb2.JobState.Name(status.state)
                logger.info("  killed -> %s", state_name)
                break
            time.sleep(0.5)
        else:
            logger.warning("  killed job did not reach terminal state within timeout")


# =============================================================================
# Screenshot capture
# =============================================================================


def capture_screenshots(dashboard_url: str, job_ids: dict[str, str], output_dir: Path):
    """Use Playwright to screenshot each dashboard tab and job detail pages."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})

        def wait_for_ready(expression: str, label: str) -> None:
            try:
                page.wait_for_function(expression, timeout=30000)
            except PlaywrightTimeoutError:
                logger.warning("Timed out waiting for %s; capturing anyway", label)

        # Load dashboard once, wait for Preact to render, then screenshot each tab
        page.goto(f"{dashboard_url}/")
        page.wait_for_load_state("domcontentloaded")
        wait_for_ready(
            "() => document.getElementById('root').children.length > 0"
            " && !document.getElementById('root').textContent.includes('Loading...')",
            "dashboard root",
        )

        for tab in DASHBOARD_TABS:
            logger.info("Capturing tab: %s", tab)
            if tab != "jobs":
                page.click(f'button.tab-btn:has-text("{tab.capitalize()}")')
                page.wait_for_timeout(300)

            # For jobs tab, expand hierarchical jobs to show tree view
            if tab == "jobs":
                try:
                    # Click the expand arrow (▶) next to jobs that have children
                    expand_arrows = page.locator("span:has-text('▶')")
                    count = expand_arrows.count()
                    if count > 0:
                        logger.info("  Expanding %d hierarchical jobs", count)
                        for i in range(count):
                            expand_arrows.nth(i).click()
                            page.wait_for_timeout(100)
                except Exception as e:
                    logger.warning("  Failed to expand hierarchical jobs: %s", e)

            path = output_dir / f"tab-{tab}.png"
            page.screenshot(path=str(path), full_page=True)
            saved.append(path)

        # Screenshot job detail pages
        for name, label in [("building", "building"), ("failed", "failed"), ("running", "running")]:
            jid = job_ids.get(name)
            if not jid:
                continue
            url = f"{dashboard_url}/job/{jid}"
            logger.info("Capturing job detail: %s (%s)", name, jid)
            page.goto(url)
            page.wait_for_load_state("domcontentloaded")
            wait_for_ready(
                "() => document.getElementById('root').children.length > 0"
                " && !document.body.textContent.includes('Loading')",
                f"job detail {label}",
            )

            path = output_dir / f"job-{label}.png"
            page.screenshot(path=str(path), full_page=True)
            saved.append(path)

        # Screenshot retry job (with expanded attempts to show attempt visibility)
        retry_jid = job_ids.get("retry")
        if retry_jid:
            url = f"{dashboard_url}/job/{retry_jid}"
            logger.info("Capturing retry job detail (with attempts expanded): %s", retry_jid)
            page.goto(url)
            page.wait_for_load_state("domcontentloaded")
            wait_for_ready(
                "() => document.getElementById('root').children.length > 0"
                " && !document.body.textContent.includes('Loading')",
                "job detail retry",
            )
            # Click the attempts count cell to expand attempt history
            # The cell with ▶ contains the attempt count and is clickable
            try:
                page.click("td:has-text('▶')", timeout=3000)
                page.wait_for_timeout(300)
            except PlaywrightTimeoutError:
                logger.info("No expandable attempts found (may already be expanded or single attempt)")
            path = output_dir / "job-retry.png"
            page.screenshot(path=str(path), full_page=True)
            saved.append(path)

        # Screenshot killed and pending and coscheduled job detail pages
        for name, label in [("killed", "killed"), ("pending-constraint", "pending"), ("coscheduled", "coscheduled")]:
            jid = job_ids.get(name)
            if not jid:
                continue
            url = f"{dashboard_url}/job/{jid}"
            logger.info("Capturing %s job detail: %s", label, jid)
            page.goto(url)
            page.wait_for_load_state("domcontentloaded")
            wait_for_ready(
                "() => document.getElementById('root').children.length > 0"
                " && !document.body.textContent.includes('Loading')",
                f"job detail {label}",
            )
            path = output_dir / f"job-{label}.png"
            page.screenshot(path=str(path), full_page=True)
            saved.append(path)

        # Screenshot VM detail page (find a real VM from autoscaler)
        try:
            import json
            import urllib.request

            autoscaler_resp = urllib.request.urlopen(
                urllib.request.Request(
                    f"{dashboard_url}/iris.cluster.ControllerService/GetAutoscalerStatus",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                )
            )
            autoscaler_data = json.loads(autoscaler_resp.read())
            status = autoscaler_data.get("status", {})
            for group in status.get("groups", []):
                for slice_info in group.get("slices", []):
                    for vm in slice_info.get("vms", []):
                        vm_id = vm.get("vmId")
                        if vm_id:
                            url = f"{dashboard_url}/vm/{vm_id}"
                            logger.info("Capturing VM detail: %s", vm_id)
                            page.goto(url)
                            page.wait_for_load_state("domcontentloaded")
                            wait_for_ready(
                                "() => document.getElementById('root').children.length > 0"
                                " && !document.getElementById('root').textContent.includes('Loading...')",
                                "vm detail",
                            )
                            path = output_dir / "vm-detail.png"
                            page.screenshot(path=str(path), full_page=True)
                            saved.append(path)
                            raise StopIteration
        except StopIteration:
            pass
        except Exception as e:
            logger.warning("Failed to capture VM detail: %s", e)

        # Screenshot controller logs page
        logger.info("Capturing controller logs page")
        page.goto(f"{dashboard_url}#logs")
        page.wait_for_load_state("domcontentloaded")
        wait_for_ready(
            "() => document.querySelectorAll('.log-line').length > 0",
            "controller logs",
        )
        path = output_dir / "controller-logs.png"
        page.screenshot(path=str(path), full_page=True)
        saved.append(path)

        browser.close()

    return saved


def capture_worker_screenshots(controller_url: str, output_dir: Path) -> list[Path]:
    """Screenshot the worker dashboard for each worker in the cluster."""
    import json
    import urllib.request

    saved: list[Path] = []

    # Get list of workers from controller
    try:
        workers_resp = urllib.request.urlopen(
            urllib.request.Request(
                f"{controller_url}/iris.cluster.ControllerService/ListWorkers",
                data=b"{}",
                headers={"Content-Type": "application/json"},
            )
        )
        workers_data = json.loads(workers_resp.read())
        workers = workers_data.get("workers", [])
    except Exception as e:
        logger.warning("Failed to get workers list: %s", e)
        return saved

    if not workers:
        logger.info("No workers found, skipping worker screenshots")
        return saved

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})

        def wait_for_ready(expression: str, label: str) -> None:
            try:
                page.wait_for_function(expression, timeout=30000)
            except PlaywrightTimeoutError:
                logger.warning("Timed out waiting for %s; capturing anyway", label)

        for i, worker in enumerate(workers):
            worker_address = worker.get("address", "")
            if not worker_address:
                continue

            # Worker address is host:port, construct URL
            worker_url = f"http://{worker_address}"
            logger.info("Capturing worker dashboard: %s", worker_url)

            try:
                page.goto(f"{worker_url}/")
                page.wait_for_load_state("domcontentloaded")
                wait_for_ready(
                    "() => document.getElementById('root').children.length > 0"
                    " && !document.getElementById('root').textContent.includes('Loading...')",
                    "worker dashboard",
                )

                path = output_dir / f"worker-{i}-dashboard.png"
                page.screenshot(path=str(path), full_page=True)
                saved.append(path)

                # Also capture task detail page if there are tasks
                # Try to get task list from worker
                try:
                    tasks_resp = urllib.request.urlopen(
                        urllib.request.Request(
                            f"{worker_url}/iris.cluster.WorkerService/ListTasks",
                            data=b"{}",
                            headers={"Content-Type": "application/json"},
                        )
                    )
                    tasks_data = json.loads(tasks_resp.read())
                    tasks = tasks_data.get("tasks", [])

                    # Screenshot first running task if any
                    for task in tasks:
                        task_id = task.get("taskId", "")
                        task_state = task.get("state", "")
                        if task_state == "TASK_STATE_RUNNING" and task_id:
                            task_url = f"{worker_url}/task/{task_id}"
                            logger.info("Capturing task detail: %s", task_id)
                            page.goto(task_url)
                            page.wait_for_load_state("domcontentloaded")
                            wait_for_ready(
                                "() => document.getElementById('root').children.length > 0",
                                "task detail",
                            )
                            path = output_dir / f"worker-{i}-task-running.png"
                            page.screenshot(path=str(path), full_page=True)
                            saved.append(path)
                            break
                except Exception as e:
                    logger.debug("Failed to get tasks from worker: %s", e)

            except Exception as e:
                logger.warning("Failed to capture worker %s dashboard: %s", worker_address, e)

        browser.close()

    return saved


# =============================================================================
# Main
# =============================================================================


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_CONFIG,
    help="Cluster config YAML (default: examples/demo.yaml)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory for saved screenshots",
)
@click.option(
    "--stay-open",
    is_flag=True,
    help="Keep cluster running after screenshots for manual browsing",
)
def main(config: Path, output_dir: Path, stay_open: bool):
    from iris.logging import configure_logging

    configure_logging(level=logging.INFO)

    logger.info("Loading config from %s", config)
    cluster_config = load_config(config)
    cluster_config = make_local_config(cluster_config)

    manager = ClusterManager(cluster_config)

    with manager.connect() as url:
        logger.info("Cluster running at %s", url)

        client = IrisClient.remote(url, workspace=IRIS_ROOT)

        logger.info("Submitting demo jobs...")
        job_ids = submit_demo_jobs(client)

        logger.info("Waiting for terminal jobs...")
        wait_for_terminal_jobs(client, job_ids)

        # Brief pause so the running job has time to start
        time.sleep(2.0)

        logger.info("Capturing controller screenshots to %s", output_dir)
        saved = capture_screenshots(url, job_ids, output_dir)
        logger.info("Controller screenshots: %d saved", len(saved))

        logger.info("Capturing worker screenshots...")
        worker_saved = capture_worker_screenshots(url, output_dir)
        saved.extend(worker_saved)
        logger.info("Worker screenshots: %d saved", len(worker_saved))

        logger.info("Summary: %d total screenshots saved", len(saved))
        for path in saved:
            logger.info("  %s", path)

        # After screenshots, wait for building job to complete
        # (chaos delay is 30s, should complete shortly after screenshots)
        building_jid = JobName.from_wire(job_ids["building"]) if job_ids.get("building") else None
        if building_jid:
            logger.info("Waiting for building job to complete after chaos delay...")
            deadline = time.monotonic() + 60.0
            terminal_states = (
                cluster_pb2.JOB_STATE_SUCCEEDED,
                cluster_pb2.JOB_STATE_FAILED,
                cluster_pb2.JOB_STATE_KILLED,
                cluster_pb2.JOB_STATE_WORKER_FAILED,
            )
            while time.monotonic() < deadline:
                status = client.status(building_jid)
                if status.state in terminal_states:
                    state_name = cluster_pb2.JobState.Name(status.state)
                    logger.info("  building -> %s", state_name)
                    break
                time.sleep(0.5)
            else:
                logger.warning("  building job did not complete within timeout")

        print(f"\nDashboard URL: {url}")

        if not stay_open:
            # Terminate all non-terminal jobs so the controller can shut down cleanly
            # without waiting for in-flight heartbeat dispatches to long-running workers.
            terminal_states = (
                cluster_pb2.JOB_STATE_SUCCEEDED,
                cluster_pb2.JOB_STATE_FAILED,
                cluster_pb2.JOB_STATE_KILLED,
                cluster_pb2.JOB_STATE_WORKER_FAILED,
            )
            for name, jid in job_ids.items():
                job_id = JobName.from_wire(jid)
                status = client.status(job_id)
                if status.state not in terminal_states:
                    logger.info("Terminating remaining job %s (%s)", name, jid)
                    client.terminate(job_id)

        if stay_open:
            print("Cluster will remain running. Press Ctrl-C to shutdown...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down cluster...")


if __name__ == "__main__":
    main()
