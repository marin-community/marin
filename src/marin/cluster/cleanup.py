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
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable

import ray

from .gcp import cleanup_preempted_tpus, delete_tpu_node, find_tpu_by_ip

logger = logging.getLogger(__name__)


@dataclass
class WorkerTaskResult:
    """Result from running tasks across workers."""

    workers_targeted: int
    workers_completed: int
    results: list[tuple[str, Any]] = field(default_factory=list)  # (resource_name, return_value) pairs
    errors: list[str] = field(default_factory=list)


@dataclass
class DiskUsageInfo:
    """Disk usage information for a worker."""

    node_ip: str
    disk_usage_pct: float
    disk_free_pct: float
    disk_free_gb: float
    disk_total_gb: float
    resource_name: str = ""


@dataclass
class DiskUsageResult:
    """Result from disk usage check across workers."""

    workers_checked: int
    workers_with_low_disk: int
    low_disk_workers: list[DiskUsageInfo] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class CleanupResult:
    """Result from a cleanup iteration."""

    deleted_tpus: list[str] = field(default_factory=list)
    lockfile_cleanup: WorkerTaskResult | None = None
    disk_usage_check: DiskUsageResult | None = None
    low_disk_terminated: list[str] = field(default_factory=list)


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


def run_task_on_workers(task_fn: Callable, timeout: int = 5) -> WorkerTaskResult:
    """Run a Ray remote task on all TPU workers and collect results.

    Args:
        task_fn: Ray remote function to run on each worker
        timeout: Timeout in seconds for waiting on tasks

    Returns:
        WorkerTaskResult with results from completed tasks
    """
    worker_resources = get_tpu_worker_resources()

    if not worker_resources:
        logger.info("No TPU workers found")
        return WorkerTaskResult(workers_targeted=0, workers_completed=0)

    # Schedule task on each worker using its specific resource
    futures = {}
    for resource_name, count in worker_resources.items():
        for i in range(int(count)):
            task = task_fn.options(resources={resource_name: 1}).remote()
            futures[f"{resource_name}:{i}"] = task

    result = WorkerTaskResult(workers_targeted=len(worker_resources), workers_completed=0)
    ready, not_ready = ray.wait(list(futures.values()), num_returns=len(futures), timeout=timeout)

    # Collect results from completed tasks
    for resource_name, future in futures.items():
        if future in ready:
            try:
                value = ray.get(future)
                result.results.append((resource_name, value))
                result.workers_completed += 1
            except Exception as e:
                result.errors.append(f"{resource_name}: {e}")
        elif future in not_ready:
            result.errors.append(f"{resource_name}: Timed out after {timeout} seconds")

    return result


@ray.remote(num_cpus=0)
def _cleanup_worker_tpu():
    """Remote task to kill any TPU processes on a specific worker and cleanup the lockfile."""
    import subprocess

    subprocess.run(["sudo", "rm", "-f", "/tmp/libtpu_lockfile"], check=False)

    # Find and kill any running TPU processes by traversing /proc
    accels = []
    for path in ["/dev/vfio/", "/dev/accel"]:
        for chip in range(8):
            accels.append(f"{path}{chip}")

    # Find PIDs that have TPU device files open
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
            continue

    # Kill the processes
    for pid in pids:
        subprocess.run(["sudo", "--non-interactive", "kill", "-9", str(pid)], check=False)


def cleanup_worker_tpus() -> WorkerTaskResult:
    """Clean TPU lockfiles and processes across all worker nodes."""

    logger.info("Cleaning TPU lockfiles on workers")
    result = run_task_on_workers(_cleanup_worker_tpu)
    logger.info(f"Cleaned {result.workers_completed}/{result.workers_targeted} workers")

    if result.errors:
        logger.warning(f"Encountered {len(result.errors)} errors during cleanup")

    return result


@ray.remote(num_cpus=0)
def _check_worker_disk_usage() -> dict:
    """Remote task to check disk usage on a specific worker."""
    import socket

    usage = shutil.disk_usage("/")
    return {
        "node_ip": socket.gethostbyname(socket.gethostname()),
        "disk_usage_pct": (usage.used / usage.total) * 100,
        "disk_free_pct": (usage.free / usage.total) * 100,
        "disk_free_gb": usage.free / (1024**3),
        "disk_total_gb": usage.total / (1024**3),
    }


def check_worker_disk_usage(disk_threshold_pct: float = 1.0) -> DiskUsageResult:
    """Check disk usage across all worker nodes and identify workers with low disk space.

    Args:
        disk_threshold_pct: Percentage threshold below which a worker is considered to have low disk (default 1.0%)

    Returns:
        DiskUsageResult with workers that have low disk space
    """
    logger.info("Checking disk usage on workers")
    task_result = run_task_on_workers(_check_worker_disk_usage)

    result = DiskUsageResult(
        workers_checked=task_result.workers_targeted, workers_with_low_disk=0, errors=task_result.errors
    )

    # Process disk usage results
    for resource_name, disk_info_dict in task_result.results:
        # Convert dict to DiskUsageInfo
        disk_info_dict["resource_name"] = resource_name.split(":")[0]
        disk_info = DiskUsageInfo(**disk_info_dict)

        if disk_info.disk_free_pct < disk_threshold_pct:
            result.low_disk_workers.append(disk_info)
            result.workers_with_low_disk += 1
            logger.warning(f"Worker {resource_name} has low disk space: {disk_info.disk_free_pct:.2f}% free")

    logger.info(f"Disk usage check: {result.workers_with_low_disk}/{result.workers_checked} workers with low disk")
    if result.errors:
        logger.warning(f"Encountered {len(result.errors)} errors during disk usage check")

    return result


def terminate_workers_by_ip(worker_ips: list[str], project: str, zone: str) -> list[str]:
    """Terminate worker nodes by their IP addresses.

    For TPU slices, multiple IPs may map to the same TPU node. This function
    groups by TPU name first to avoid duplicate deletion attempts.

    Args:
        worker_ips: List of worker node IP addresses to terminate
        project: GCP project ID
        zone: GCP zone

    Returns:
        List of terminated TPU node names
    """
    if not worker_ips:
        return []

    # Group IPs by TPU name (handles slices where multiple IPs -> one TPU)
    tpu_to_ips: dict[str, list[str]] = {}
    for worker_ip in worker_ips:
        try:
            tpu_info = find_tpu_by_ip(worker_ip, project, zone)
            if not tpu_info:
                logger.warning(f"Could not find TPU node for IP {worker_ip}")
                continue

            tpu_name, tpu_zone, worker_index = tpu_info
            if tpu_name not in tpu_to_ips:
                tpu_to_ips[tpu_name] = []
            tpu_to_ips[tpu_name].append(worker_ip)
        except Exception as e:
            logger.error(f"Failed to lookup TPU for IP {worker_ip}: {e}")

    # Delete each unique TPU once
    terminated = []
    for tpu_name, ips in tpu_to_ips.items():
        try:
            logger.info(f"Terminating TPU {tpu_name} (low disk on workers: {', '.join(ips)})")
            delete_tpu_node(tpu_name, project, zone, quiet=True)
            terminated.append(tpu_name)
        except Exception as e:
            logger.error(f"Failed to terminate TPU {tpu_name}: {e}")

    return terminated


CLEANUP_JOB_PREFIX = "marin-cleanup-cron"


def cleanup_iteration(project: str, zone: str, dry_run: bool = False, disk_threshold_pct: float = 1.0) -> CleanupResult:
    """Run one iteration of cleanup checks.

    Performs:
    - Removes terminated/preempted TPU nodes
    - Terminates any processes using the TPU on idle workers
    - Checks disk usage and terminates workers with low disk space

    Args:
        project: GCP project ID
        zone: GCP zone
        dry_run: If True, only report what would be cleaned without making changes
        disk_threshold_pct: Percentage threshold for low disk space (default 1.0%)

    Returns:
        CleanupResult with cleanup statistics
    """
    result = CleanupResult()

    # Cleanup preempted/terminated TPU nodes
    result.deleted_tpus = cleanup_preempted_tpus(project, zone, dry_run=dry_run)
    if result.deleted_tpus:
        logger.info(f"Cleaned up {len(result.deleted_tpus)} preempted/terminated TPUs")

    if not dry_run:
        # Clean TPU lockfiles on active workers
        result.lockfile_cleanup = cleanup_worker_tpus()
        completed = result.lockfile_cleanup.workers_completed
        targeted = result.lockfile_cleanup.workers_targeted
        logger.info(f"TPU lockfile cleanup: {completed}/{targeted} workers")

        # Check disk usage and terminate workers with low disk
        result.disk_usage_check = check_worker_disk_usage(disk_threshold_pct)
        if result.disk_usage_check.workers_with_low_disk > 0:
            logger.warning(
                f"Found {result.disk_usage_check.workers_with_low_disk} workers with <{disk_threshold_pct}% free"
            )
            worker_ips = [w.node_ip for w in result.disk_usage_check.low_disk_workers]
            result.low_disk_terminated = terminate_workers_by_ip(worker_ips, project, zone)
            if result.low_disk_terminated:
                logger.info(f"Terminated {len(result.low_disk_terminated)} workers due to low disk space")

    return result


def run_cleanup_loop(project: str, zone: str, interval: int = 600, disk_threshold_pct: float = 1.0) -> None:
    """Main cleanup cron loop - runs indefinitely checking for issues periodically.

    This function runs as a Ray job and performs cleanup operations every `interval` seconds.
    It will continue running until the job is explicitly stopped.

    Args:
        project: GCP project ID
        zone: GCP zone
        interval: Time between cleanup checks in seconds (default 600 = 10 minutes)
        disk_threshold_pct: Percentage threshold for low disk space (default 1.0%)
    """
    logger.info(f"Starting cleanup cron loop (interval={interval}s, disk_threshold={disk_threshold_pct}%)")
    logger.info(f"Monitoring project={project}, zone={zone}")

    for iteration in itertools.count(1):
        logger.info(f"=== Cleanup iteration {iteration} ===")

        try:
            result = cleanup_iteration(project, zone, dry_run=False, disk_threshold_pct=disk_threshold_pct)
            logger.info(f"Iteration {iteration}: {result}")

            # Log errors if any
            if result.lockfile_cleanup and result.lockfile_cleanup.errors:
                logger.warning(f"Lockfile cleanup errors: {result.lockfile_cleanup.errors}")
            if result.disk_usage_check and result.disk_usage_check.errors:
                logger.warning(f"Disk usage check errors: {result.disk_usage_check.errors}")
        except Exception as e:
            logger.error(f"Cleanup iteration {iteration} failed: {e}", exc_info=True)

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


def submit_cleanup_cron_job(project: str, zone: str, interval: int = 600, disk_threshold_pct: float = 1.0) -> str:
    """Submit the cleanup cron job to the Ray cluster.

    If a cleanup job is already running, stops it first before starting a new one.

    Args:
        project: GCP project ID
        zone: GCP zone
        interval: Time between cleanup checks in seconds (default 600)
        disk_threshold_pct: Percentage threshold for low disk space (default 1.0%)

    Returns:
        Job ID of the submitted cleanup job
    """
    # Stop existing cleanup job if running
    existing_job = get_running_cleanup_job()
    if existing_job:
        logger.info(f"Stopping existing cleanup job: {existing_job['job_id']}")
        stop_cleanup_job(existing_job["job_id"])

    # Build entrypoint command
    entrypoint = (
        f"python -m marin.cluster.cleanup "
        f"--project {project} "
        f"--zone {zone} "
        f"--interval {interval} "
        f"--disk-threshold {disk_threshold_pct}"
    )

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
    parser.add_argument(
        "--disk-threshold",
        type=float,
        default=1.0,
        help="Disk free space threshold percentage for terminating workers (default: 1.0%%)",
    )

    args = parser.parse_args()

    # Run the cleanup loop
    run_cleanup_loop(
        project=args.project,
        zone=args.zone,
        interval=args.interval,
        disk_threshold_pct=args.disk_threshold,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
