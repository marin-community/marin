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

"""Automated cleanup utilities for cluster management.

Preempted TPUs count against our quota until they are deleted.
This provides a periodic cron job which attempts to cleanup any preempted or
terminated TPUs, as well as cleaning up any TPU lockfiles on active workers.
"""

import datetime
import itertools
import json
import logging
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ray

from .gcp import cleanup_preempted_tpus

logger = logging.getLogger(__name__)


@dataclass
class TpuCleanupResult:
    """Result of cleaning TPU processes on workers."""

    workers_targeted: int
    workers_cleaned: int
    errors: list[str]


@dataclass
class CleanupIterationResult:
    """Result of a single cleanup iteration."""

    deleted_tpus: list[str]
    tpu_cleanup: TpuCleanupResult


def get_tpu_worker_resources() -> dict[str, int]:
    """Get list of unique worker resource names from Ray cluster.

    Queries the Ray GCS to find all worker nodes with TPU resources and returns
    their unique resource identifiers.

    Returns:
        List of worker resource names like:
        - ray-marin-us-east5-a-worker-09614199-tpu
        - ray-marin-us-east5-a-worker-2f248a95-tpu
    """
    from .ray import list_nodes

    worker_resources = {}
    try:
        nodes = list_nodes()

        for node in nodes:
            # Only process alive nodes
            if node.get("state") != "ALIVE":
                continue

            resources = node.get("resources_total", {})
            # Find worker-specific resources (format: ray-*-worker-*-tpu)
            for resource_name in resources.keys():
                if resource_name.startswith("ray-") and "-worker-" in resource_name and resource_name.endswith("-tpu"):
                    worker_resources[resource_name] = resources[resource_name]

        logger.info(f"Found {len(worker_resources)} TPU worker resources")
    except Exception as e:
        logger.error(f"Failed to get TPU worker resources: {e}")

    return worker_resources


def cleanup_tpu_processes() -> TpuCleanupResult:
    @ray.remote(num_cpus=0)
    def _cleanup_worker():
        """Kill any TPU processes on a specific worker and cleanup the lockfile."""
        subprocess.run(["sudo", "rm", "-f", "/tmp/libtpu_lockfile"], check=False)

        # Find and kill any running TPU processes by traversing /proc
        accels = []
        for path in ["/dev/vfio/", "/dev/accel"]:
            for chip in range(8):
                accels.append(f"{path}{chip}")

        pids = _find_pids_using_device(set(accels))
        for pid in pids:
            subprocess.run(["sudo", "--non-interactive", "kill", "-9", str(pid)], check=False)

    def _find_pids_using_device(accels: set[str]) -> list[int]:
        pids = []
        proc_dir = Path("/proc")

        for pid_dir in proc_dir.iterdir():
            if not pid_dir.name.isdigit():
                continue

            fd_dir = pid_dir / "fd"
            if not fd_dir.exists():
                continue

            try:
                for fd in fd_dir.iterdir():
                    target = fd.resolve()
                    if str(target) in accels:
                        pids.append(int(pid_dir.name))
                        break
            except (OSError, PermissionError):
                # Can't access this process's fds, skip it
                continue

        return pids

    worker_resources = get_tpu_worker_resources()

    if not worker_resources:
        logger.info("No TPU workers found for cleanup")
        return TpuCleanupResult(workers_targeted=0, workers_cleaned=0, errors=[])

    logger.info(f"Cleaning TPU on {len(worker_resources)} workers")

    futures = {}
    for resource_name, count in worker_resources.items():
        # Schedule with resource requirement to target the given slice.
        # This won't guarantee we hit every host, as we may be scheduled
        # onto the same host multiple times.
        for i in range(int(count)):
            task = _cleanup_worker.options(
                resources={resource_name: 1},
                enable_task_events=False,
                retry_exceptions=False,
            ).remote()
            futures[f"{resource_name}:{i}"] = task

    errors: list[str] = []
    workers_cleaned = 0
    ready, not_ready = ray.wait(list(futures.values()), num_returns=len(futures), timeout=60)

    # Process completed tasks
    for resource_name, future in futures.items():
        if future in ready:
            try:
                ray.get(future)
                workers_cleaned += 1
            except Exception as e:
                errors.append(f"{resource_name}: {e}")
        elif future in not_ready:
            errors.append(f"{resource_name}: Timed out after 60 seconds")

    for task in not_ready:
        ray.cancel(task, force=True)

    logger.info(f"Cleaned {workers_cleaned}/{len(worker_resources)} workers")

    if errors:
        logger.warning(f"Encountered {len(errors)} errors during cleanup")

    return TpuCleanupResult(
        workers_targeted=len(worker_resources),
        workers_cleaned=workers_cleaned,
        errors=errors,
    )


CLEANUP_JOB_PREFIX = "marin-cleanup-cron"


def cleanup_iteration(project: str, zone: str, dry_run: bool = False) -> CleanupIterationResult:
    """Run one iteration of cleanup checks."""
    # Cleanup preempted/terminated TPU nodes
    deleted = cleanup_preempted_tpus(project, zone, dry_run=dry_run)

    if deleted:
        logger.info(f"Cleaned up {len(deleted)} preempted/terminated TPUs: {deleted}")

    if not dry_run:
        tpu_cleanup = cleanup_tpu_processes()
        logger.info(f"TPU lockfile cleanup: {tpu_cleanup.workers_cleaned}/{tpu_cleanup.workers_targeted} workers")
    else:
        tpu_cleanup = TpuCleanupResult(workers_targeted=0, workers_cleaned=0, errors=[])

    return CleanupIterationResult(deleted_tpus=deleted, tpu_cleanup=tpu_cleanup)


def run_cleanup_loop(gcp_project: str, zone: str, interval: int = 600) -> None:
    """Main cleanup cron loop - runs indefinitely checking for issues periodically.

    This function runs as a Ray job and performs cleanup operations every `interval` seconds.
    It will continue running until the job is explicitly stopped.
    """
    logger.info(f"Starting cleanup cron loop (interval={interval}s)")
    logger.info(f"Monitoring project={gcp_project}, zone={zone}")

    for iteration in itertools.count(1):
        logger.info(f"=== Cleanup iteration {iteration} ===")

        try:
            results = cleanup_iteration(gcp_project, zone, dry_run=False)
            logger.info(f"Iteration {iteration}: {len(results.deleted_tpus)} TPUs terminated")
            workers_cleaned = results.tpu_cleanup.workers_cleaned
            workers_targeted = results.tpu_cleanup.workers_targeted
            logger.info(f"Iteration {iteration}: cleanup - {workers_cleaned}/{workers_targeted} workers")
            if results.tpu_cleanup.errors:
                logger.warning(f"Iteration {iteration}: cleanup errors - {results.tpu_cleanup.errors}")
        except Exception as e:
            logger.error(f"Cleanup iteration {iteration} failed: {e}", exc_info=True)
            logger.info(f"Will retry in {interval}s...")

        logger.info(f"Sleeping {interval}s until next check...")
        time.sleep(interval)


def running_cleanup_jobs() -> Sequence[dict[str, Any]]:
    """Return a list of job dicts for any currently running or pending cleanup jobs."""
    from .ray import list_jobs

    running_jobs = list_jobs(filters=["status=RUNNING"])
    pending_jobs = list_jobs(filters=["status=PENDING"])

    active_jobs = {}

    for job in running_jobs + pending_jobs:
        submission_id = job.get("submission_id") or ""
        if submission_id.startswith(CLEANUP_JOB_PREFIX):
            active_jobs[submission_id] = job

    return active_jobs.values()


def submit_cleanup_cron_job(project: str, cluster: str, zone: str, interval: int = 600) -> str:
    """Submit the cleanup cron job to the Ray cluster.

    If a cleanup job is already running, stops it first before starting a new one.
    """
    for job in running_cleanup_jobs():
        logger.info(f"Stopping existing cleanup job: {json.dumps(job, indent=2)}")
        subprocess.run(
            ["ray", "job", "stop", job["submission_id"]],
            check=True,
        )

    entrypoint = (
        f"PYTHONPATH=src/ python ./scripts/ray/cleanup_tpus.py --project {project} --zone {zone} --interval {interval}"
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_id = f"{CLEANUP_JOB_PREFIX}-{timestamp}"

    logger.info(f"Submitting cleanup cron job: {submission_id}")
    logger.info(f"Entrypoint: {entrypoint}")

    subprocess.run(
        [
            "uv",
            "run",
            "src/marin/run/ray_run.py",
            f"--submission-id={submission_id}",
            f"--cluster={cluster}",
            "--no_wait",
            "--",
            *entrypoint.split(),
        ],
        capture_output=False,
        check=True,
    )

    logger.info(f"Cleanup job submitted: {submission_id}")
    return submission_id
