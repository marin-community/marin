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

This module provides a cron-based cleanup system that periodically checks for
and removes terminated/preempted TPU nodes.
"""

import argparse
import datetime
import itertools
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import ray

from .gcp import cleanup_preempted_tpus

logger = logging.getLogger(__name__)


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


def cleanup_tpu_processes() -> dict[str, Any]:
    """Clean TPU lockfiles across all worker nodes."""

    @ray.remote(num_cpus=0)
    def _cleanup_worker():
        """Remote task kill any TPU processes on a specific worker and cleanup the lockfile."""
        subprocess.run(["sudo", "rm", "-f", "/tmp/libtpu_lockfile"], check=False)

        # Find and kill any running TPU processes by traversing /proc
        accels = []
        for path in ["/dev/vfio/", "/dev/accel"]:
            for chip in range(8):
                accels.append(f"{path}{chip}")

        pids = _find_pids_using_device(accels)
        for pid in pids:
            subprocess.run(["sudo", "--non-interactive", "kill", "-9", str(pid)], check=False)

    def _find_pids_using_device(accels: list[str]) -> list[int]:
        """Find PIDs that have the given device file open by traversing /proc.

        Ignores processes we don't have permission to inspect.
        """
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
        return {"workers_targeted": 0, "workers_cleaned": 0, "errors": []}

    logger.info(f"Cleaning TPU on {len(worker_resources)} workers")

    # Schedule cleanup task on each worker using its specific resource
    futures = {}
    for resource_name, count in worker_resources.items():
        # Schedule with resource requirement to target the given slice.
        # For slices, we best-effort target one task per host.
        for i in range(int(count)):
            task = _cleanup_worker.options(resources={resource_name: 1}).remote()
            futures[f"{resource_name}:{i}"] = task

    results = {"workers_targeted": len(worker_resources), "workers_cleaned": 0, "errors": []}
    ready, not_ready = ray.wait(list(futures.values()), num_returns=len(futures), timeout=60)

    # Process completed tasks
    for resource_name, future in futures.items():
        if future in ready:
            try:
                ray.get(future)
                results["workers_cleaned"] += 1
            except Exception as e:
                results["errors"].append(f"{resource_name}: {e}")
        elif future in not_ready:
            results["errors"].append(f"{resource_name}: Timed out after 60 seconds")

    logger.info(f"Cleaned {results['workers_cleaned']}/{results['workers_targeted']} workers")

    if results["errors"]:
        logger.warning(f"Encountered {len(results['errors'])} errors during cleanup")

    return results


CLEANUP_JOB_PREFIX = "marin-cleanup-cron"


def cleanup_iteration(project: str, zone: str, dry_run: bool = False) -> dict[str, Any]:
    """Run one iteration of cleanup checks.

    Performs:
    - Removes terminated/preempted TPU nodes
    - Terminates any processes using the TPU on idle workers

    Args:
        project: GCP project ID
        zone: GCP zone
        dry_run: If True, only report what would be cleaned without making changes

    Returns:
        Dict with cleanup results containing:
        - deleted_tpus: List of deleted TPU names
        - lockfile_cleanup: Dict with lockfile cleanup statistics
    """
    results = {"deleted_tpus": [], "lockfile_cleanup": {}}

    # Cleanup preempted/terminated TPU nodes
    deleted = cleanup_preempted_tpus(project, zone, dry_run=dry_run)
    results["deleted_tpus"] = deleted

    if deleted:
        logger.info(f"Cleaned up {len(deleted)} preempted/terminated TPUs: {deleted}")

    # Clean TPU lockfiles on active workers (skip during dry-run)
    if not dry_run:
        lockfile_results = cleanup_tpu_processes()
        results["lockfile_cleanup"] = lockfile_results
        logger.info(
            f"TPU lockfile cleanup: {lockfile_results['workers_cleaned']}/{lockfile_results['workers_targeted']} workers"
        )

    return results


def run_cleanup_loop(project: str, zone: str, interval: int = 600) -> None:
    """Main cleanup cron loop - runs indefinitely checking for issues periodically.

    This function runs as a Ray job and performs cleanup operations every `interval` seconds.
    It will continue running until the job is explicitly stopped.

    Args:
        project: GCP project ID
        zone: GCP zone
        interval: Time between cleanup checks in seconds (default 600 = 10 minutes)
    """
    logger.info(f"Starting cleanup cron loop (interval={interval}s)")
    logger.info(f"Monitoring project={project}, zone={zone}")

    for iteration in itertools.count(1):
        logger.info(f"=== Cleanup iteration {iteration} ===")

        try:
            results = cleanup_iteration(project, zone, dry_run=False)

            if results["deleted_tpus"]:
                logger.info(f"Iteration {iteration}: {len(results['deleted_tpus'])} TPUs terminated")

            lockfile_stats = results.get("lockfile_cleanup", {})
            workers_cleaned = lockfile_stats.get("workers_cleaned", 0)
            workers_targeted = lockfile_stats.get("workers_targeted", 0)
            logger.info(f"Iteration {iteration}: Lockfile cleanup - {workers_cleaned}/{workers_targeted} workers")
            if lockfile_stats.get("errors"):
                logger.warning(f"Iteration {iteration}: Lockfile cleanup errors - {lockfile_stats['errors']}")

            if not results["deleted_tpus"] and not lockfile_stats:
                logger.info(f"Iteration {iteration} complete: No cleanup needed")
        except Exception as e:
            logger.error(f"Cleanup iteration {iteration} failed: {e}", exc_info=True)
            logger.info(f"Will retry in {interval}s...")

        logger.info(f"Sleeping {interval}s until next check...")
        time.sleep(interval)


def list_jobs() -> list[dict[str, Any]]:
    """List all Ray jobs.

    Returns:
        List of job info dicts
    """
    result = subprocess.run(
        ["ray", "job", "list", "--format=json"],
        capture_output=True,
        text=True,
        check=True,
    )

    return json.loads(result.stdout)


def get_running_cleanup_job() -> dict[str, Any] | None:
    """Check if a cleanup cron job is currently running.

    Returns:
        Job info dict if found, None otherwise
    """
    jobs = list_jobs()
    for job in jobs:
        submission_id = job.get("submission_id", "")
        if submission_id.startswith(CLEANUP_JOB_PREFIX):
            status = job.get("status", "")
            if status in ["RUNNING", "PENDING"]:
                return job

    return None


def stop_cleanup_job(job_id: str) -> None:
    """Stop a cleanup cron job.

    Args:
        job_id: Ray job ID to stop
    """
    logger.info(f"Stopping cleanup job: {job_id}")

    subprocess.run(
        ["ray", "job", "stop", job_id],
        capture_output=True,
        text=True,
        check=True,
    )
    logger.info(f"Successfully stopped cleanup job: {job_id}")


def submit_cleanup_cron_job(project: str, zone: str, interval: int = 600) -> str:
    """Submit the cleanup cron job to the Ray cluster.

    If a cleanup job is already running, stops it first before starting a new one.

    Args:
        project: GCP project ID
        zone: GCP zone
        interval: Time between cleanup checks in seconds (default 600)

    Returns:
        Job ID of the submitted cleanup job
    """
    # Stop existing cleanup job if running
    existing_job = get_running_cleanup_job()
    if existing_job:
        logger.info(f"Stopping existing cleanup job: {existing_job['job_id']}")
        stop_cleanup_job(existing_job["job_id"])

    # Build entrypoint command
    entrypoint = f"python -m marin.cluster.cleanup " f"--project {project} " f"--zone {zone} " f"--interval {interval}"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_id = f"{CLEANUP_JOB_PREFIX}-{timestamp}"

    logger.info(f"Submitting cleanup cron job: {submission_id}")
    logger.info(f"Entrypoint: {entrypoint}")

    # Submit job using ray CLI
    subprocess.run(
        [
            "ray",
            "job",
            "submit",
            "--submission-id",
            submission_id,
            "--no-wait",
            "--",
            *entrypoint.split(),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    logger.info(f"Cleanup job submitted: {submission_id}")
    return submission_id


def main():
    """CLI entry point for cleanup utilities."""
    parser = argparse.ArgumentParser(description="Automated cluster cleanup utilities")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--zone", required=True, help="GCP zone")
    parser.add_argument("--interval", type=int, default=600, help="Cleanup interval in seconds (default: 600)")

    args = parser.parse_args()

    # Run the cleanup loop
    run_cleanup_loop(
        project=args.project,
        zone=args.zone,
        interval=args.interval,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
