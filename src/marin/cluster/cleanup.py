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
from typing import Any

from .gcp import cleanup_preempted_tpus

logger = logging.getLogger(__name__)

CLEANUP_JOB_PREFIX = "marin-cleanup-cron"


def cleanup_iteration(project: str, zone: str, dry_run: bool) -> list[str]:
    """Run one iteration of cleanup checks.

    Removes terminated/preempted TPU nodes.

    Args:
        project: GCP project ID
        zone: GCP zone

    Returns:
        List of deleted TPU names
    """
    deleted = cleanup_preempted_tpus(project, zone, dry_run=dry_run)

    if deleted:
        logger.info(f"Cleaned up {len(deleted)} preempted/terminated TPUs: {deleted}")

    return deleted


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
            deleted = cleanup_iteration(project, zone)

            if deleted:
                logger.info(f"Iteration {iteration} complete: {len(deleted)} TPUs terminated")
            else:
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
