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
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
class TpuCleanupResult:
    """Result of cleaning TPU processes on workers."""

    workers_targeted: int
    workers_cleaned: int
    errors: list[str]


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
class CleanupIterationResult:
    """Result of a single cleanup iteration."""

    deleted_tpus: list[str]
    tpu_cleanup: TpuCleanupResult
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


def run_task_on_workers(task_fn: Callable, timeout: int = 60) -> WorkerTaskResult:
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
            task = task_fn.options(
                resources={resource_name: 1},
                enable_task_events=False,
                retry_exceptions=False,
            ).remote()
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

    # Cancel incomplete tasks
    for task in not_ready:
        ray.cancel(task, force=True)

    return result


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

            tpu_name, _tpu_zone, _worker_index = tpu_info
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


def cleanup_iteration(
    project: str, zone: str, dry_run: bool = False, disk_threshold_pct: float = 1.0
) -> CleanupIterationResult:
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
        CleanupIterationResult with cleanup statistics
    """
    # Cleanup preempted/terminated TPU nodes
    deleted = cleanup_preempted_tpus(project, zone, dry_run=dry_run)

    if deleted:
        logger.info(f"Cleaned up {len(deleted)} preempted/terminated TPUs: {deleted}")

    disk_usage_check = None
    low_disk_terminated = []

    if not dry_run:
        tpu_cleanup = cleanup_tpu_processes()
        logger.info(f"TPU lockfile cleanup: {tpu_cleanup.workers_cleaned}/{tpu_cleanup.workers_targeted} workers")

        # Check disk usage and terminate workers with low disk
        disk_usage_check = check_worker_disk_usage(disk_threshold_pct)
        if disk_usage_check.workers_with_low_disk > 0:
            logger.warning(f"Found {disk_usage_check.workers_with_low_disk} workers with <{disk_threshold_pct}% free")
            worker_ips = [w.node_ip for w in disk_usage_check.low_disk_workers]
            low_disk_terminated = terminate_workers_by_ip(worker_ips, project, zone)
            if low_disk_terminated:
                logger.info(f"Terminated {len(low_disk_terminated)} workers due to low disk space")
    else:
        tpu_cleanup = TpuCleanupResult(workers_targeted=0, workers_cleaned=0, errors=[])

    return CleanupIterationResult(
        deleted_tpus=deleted,
        tpu_cleanup=tpu_cleanup,
        disk_usage_check=disk_usage_check,
        low_disk_terminated=low_disk_terminated,
    )


def run_cleanup_loop(gcp_project: str, zone: str, interval: int = 600, disk_threshold_pct: float = 1.0) -> None:
    """Main cleanup cron loop - runs indefinitely checking for issues periodically.

    This function runs as a Ray job and performs cleanup operations every `interval` seconds.
    It will continue running until the job is explicitly stopped.

    Args:
        gcp_project: GCP project ID
        zone: GCP zone
        interval: Time between cleanup checks in seconds (default 600 = 10 minutes)
        disk_threshold_pct: Percentage threshold for low disk space (default 1.0%)
    """
    logger.info(f"Starting cleanup cron loop (interval={interval}s, disk_threshold={disk_threshold_pct}%)")
    logger.info(f"Monitoring project={gcp_project}, zone={zone}")

    for iteration in itertools.count(1):
        logger.info(f"=== Cleanup iteration {iteration} ===")

        try:
            results = cleanup_iteration(gcp_project, zone, dry_run=False, disk_threshold_pct=disk_threshold_pct)
            logger.info(f"Iteration {iteration}: {len(results.deleted_tpus)} TPUs terminated")
            workers_cleaned = results.tpu_cleanup.workers_cleaned
            workers_targeted = results.tpu_cleanup.workers_targeted
            logger.info(f"Iteration {iteration}: cleanup - {workers_cleaned}/{workers_targeted} workers")
            if results.tpu_cleanup.errors:
                logger.warning(f"Iteration {iteration}: cleanup errors - {results.tpu_cleanup.errors}")
            if results.disk_usage_check and results.disk_usage_check.errors:
                logger.warning(f"Iteration {iteration}: disk check errors - {results.disk_usage_check.errors}")
            if results.low_disk_terminated:
                logger.info(f"Iteration {iteration}: terminated {len(results.low_disk_terminated)} low-disk workers")
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


def submit_cleanup_cron_job(
    project: str, cluster: str, zone: str, interval: int = 600, disk_threshold_pct: float = 1.0
) -> str:
    """Submit the cleanup cron job to the Ray cluster.

    If a cleanup job is already running, stops it first before starting a new one.

    Args:
        project: GCP project ID
        cluster: Cluster name
        zone: GCP zone
        interval: Time between cleanup checks in seconds (default 600)
        disk_threshold_pct: Percentage threshold for low disk space (default 1.0%)

    Returns:
        Job ID of the submitted cleanup job
    """
    for job in running_cleanup_jobs():
        logger.info(f"Stopping existing cleanup job: {json.dumps(job, indent=2)}")
        subprocess.run(
            ["ray", "job", "stop", job["submission_id"]],
            check=True,
        )

    entrypoint = " ".join(
        [
            "PYTHONPATH=src/",
            "python",
            "./scripts/ray/cleanup_tpus.py",
            f"--project {project}" f"--zone {zone}",
            f"--interval {interval}",
            f"--disk-threshold {disk_threshold_pct}",
        ]
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
