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
from playwright.sync_api import sync_playwright

from iris.client.client import IrisClient
from iris.cluster.types import (
    CoschedulingConfig,
    Constraint,
    ConstraintOp,
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
)
from iris.cluster.vm.cluster_manager import ClusterManager, make_local_config
from iris.cluster.vm.config import load_config
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

    _time.sleep(30)
    return "done"


# =============================================================================
# Job submission
# =============================================================================


def submit_demo_jobs(client: IrisClient) -> dict[str, str]:
    """Submit a variety of jobs to populate the dashboard.

    Returns a mapping of logical name -> job_id for later reference.
    """
    job_ids: dict[str, str] = {}

    # 1. Succeeded job
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_succeeded_job),
        name="snapshot-succeeded",
        resources=ResourceSpec(cpu=2, memory="4g"),
        environment=EnvironmentSpec(),
    )
    job_ids["succeeded"] = str(job.job_id)
    logger.info("Submitted succeeded job: %s", job.job_id)

    # 2. Failed job
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_failed_job),
        name="snapshot-failed",
        resources=ResourceSpec(cpu=2, memory="4g"),
        environment=EnvironmentSpec(),
    )
    job_ids["failed"] = str(job.job_id)
    logger.info("Submitted failed job: %s", job.job_id)

    # 3. Running/slow job
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_slow_job),
        name="snapshot-running",
        resources=ResourceSpec(cpu=2, memory="4g"),
        environment=EnvironmentSpec(),
    )
    job_ids["running"] = str(job.job_id)
    logger.info("Submitted running job: %s", job.job_id)

    # 4. Second running job (more realistic)
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_slow_job),
        name="snapshot-running-2",
        resources=ResourceSpec(cpu=1, memory="2g"),
        environment=EnvironmentSpec(),
    )
    job_ids["running-2"] = str(job.job_id)
    logger.info("Submitted second running job: %s", job.job_id)

    # 5. Pending unschedulable job (constraint for nonexistent TPU)
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

    # 6. Coscheduled multi-replica job (also unschedulable)
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_succeeded_job),
        name="snapshot-coscheduled",
        resources=ResourceSpec(cpu=1, memory="2g", replicas=4),
        environment=EnvironmentSpec(),
        constraints=[
            Constraint(key="tpu-name", op=ConstraintOp.EQ, value="nonexistent-tpu-xyz"),
        ],
        coscheduling=CoschedulingConfig(group_by="tpu-name"),
    )
    job_ids["coscheduled"] = str(job.job_id)
    logger.info("Submitted coscheduled job: %s", job.job_id)

    # 7. Job targeting zero-quota scale group (quota exhaustion scenario)
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

    # 8. Job targeting disabled scale group
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

    # 9. Large coscheduled job that needs more workers than available
    job = client.submit(
        entrypoint=Entrypoint.from_callable(_succeeded_job),
        name="snapshot-insufficient-capacity",
        resources=ResourceSpec(cpu=1, memory="1g", replicas=100),
        environment=EnvironmentSpec(),
        coscheduling=CoschedulingConfig(group_by="scale-group"),
    )
    job_ids["insufficient-capacity"] = str(job.job_id)
    logger.info("Submitted insufficient-capacity job: %s", job.job_id)

    return job_ids


def wait_for_terminal_jobs(client: IrisClient, job_ids: dict[str, str], timeout: float = 60.0):
    """Wait for the succeeded and failed jobs to reach terminal states."""
    terminal_names = ["succeeded", "failed"]
    deadline = time.monotonic() + timeout

    for name in terminal_names:
        jid = job_ids[name]
        logger.info("Waiting for %s job (%s) to finish...", name, jid)
        while time.monotonic() < deadline:
            status = client.status(jid)
            if status.state in (
                cluster_pb2.JOB_STATE_SUCCEEDED,
                cluster_pb2.JOB_STATE_FAILED,
                cluster_pb2.JOB_STATE_KILLED,
                cluster_pb2.JOB_STATE_WORKER_FAILED,
            ):
                state_name = cluster_pb2.JobState.Name(status.state)
                logger.info("  %s -> %s", name, state_name)
                break
            time.sleep(0.5)
        else:
            logger.warning("  %s job did not finish within %.0fs", name, timeout)


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

        # Load dashboard once, wait for data, then screenshot each tab
        page.goto(f"{dashboard_url}/")
        page.wait_for_load_state("networkidle")
        # Wait for jobs table to populate (Loading... replaced by actual rows)
        page.wait_for_function(
            "() => document.getElementById('jobs-body').textContent.trim() !== 'Loading...'",
            timeout=10000,
        )

        for tab in DASHBOARD_TABS:
            logger.info("Capturing tab: %s", tab)
            # Switch tab via JS click
            if tab != "jobs":
                page.evaluate(
                    f"""() => {{
                    document.querySelector('.tab-btn[data-tab="{tab}"]').click();
                }}"""
                )
                page.wait_for_timeout(300)

            path = output_dir / f"tab-{tab}.png"
            page.screenshot(path=str(path), full_page=True)
            saved.append(path)

        # Screenshot job detail pages
        for name, label in [("failed", "failed"), ("running", "running")]:
            jid = job_ids.get(name)
            if not jid:
                continue
            url = f"{dashboard_url}/job/{jid}"
            logger.info("Capturing job detail: %s (%s)", name, jid)
            page.goto(url)
            page.wait_for_load_state("networkidle")
            page.wait_for_function(
                "() => document.getElementById('job-state').textContent !== '-'",
                timeout=10000,
            )

            path = output_dir / f"job-{label}.png"
            page.screenshot(path=str(path), full_page=True)
            saved.append(path)

        # Screenshot pending and coscheduled job detail pages
        for name, label in [("pending-constraint", "pending"), ("coscheduled", "coscheduled")]:
            jid = job_ids.get(name)
            if not jid:
                continue
            url = f"{dashboard_url}/job/{jid}"
            logger.info("Capturing %s job detail: %s", label, jid)
            page.goto(url)
            page.wait_for_load_state("networkidle")
            page.wait_for_function(
                "() => document.getElementById('job-state').textContent !== '-'",
                timeout=10000,
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
                            page.wait_for_load_state("networkidle")
                            page.wait_for_function(
                                "() => document.getElementById('vm-state').textContent !== '-'",
                                timeout=10000,
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
        page.goto(f"{dashboard_url}/logs")
        page.wait_for_load_state("networkidle")
        page.wait_for_function(
            "() => !document.body.textContent.includes('Loading')",
            timeout=10000,
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
        else:
            logger.info("Skipping screenshots (no --output-dir specified)")

        print(f"\nDashboard URL: {url}")

        if stay_open:
            print("Cluster will remain running. Press Ctrl-C to shutdown...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down cluster...")


if __name__ == "__main__":
    main()
