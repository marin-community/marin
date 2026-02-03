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

DASHBOARD_TABS = ["jobs", "workers", "endpoints", "vms", "autoscaler"]

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

    # Wait for succeeded and failed jobs first
    deadline = time.monotonic() + timeout
    for name in ["succeeded", "failed"]:
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
    default=None,
    help="Directory for saved screenshots (optional, skips screenshots if not provided)",
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

        if output_dir:
            logger.info("Capturing screenshots to %s", output_dir)
            saved = capture_screenshots(url, job_ids, output_dir)
            logger.info("Summary: %d screenshots saved", len(saved))
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
        else:
            logger.info("Skipping screenshots (no --output-dir specified)")

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
