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

"""Cleanup temporary files & working directories from finished Ray jobs.

Ray does not automatically clean up job working directories, which can lead to
workers running out of temp space over time."""

import argparse
import asyncio
import datetime
import json
import logging
import os
import subprocess
import time
from typing import Any

import ray

logger = logging.getLogger(__name__)


def get_finished_jobs() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ray", "list", "jobs", "--format=json", "--limit=10000"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )

    jobs = json.loads(result.stdout)
    finished_jobs = [job for job in jobs if job.get("status") in ["SUCCEEDED", "FAILED", "STOPPED"]]

    logger.info(f"Found {len(finished_jobs)} finished jobs out of {len(jobs)} total jobs")
    return finished_jobs


def get_worker_nodes() -> list[str]:
    result = subprocess.run(
        ["ray", "list", "workers", "--format=json", "--limit=10000"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )

    workers = json.loads(result.stdout)
    unique_ips = list(set(worker.get("ip") for worker in workers if worker.get("ip")))

    logger.info(f"Found {len(unique_ips)} unique worker nodes")
    return unique_ips


def cleanup_temp_dirs_on_node(job_ids: list[str], dry_run: bool = False) -> dict[str, Any]:
    """Clean up temporary directories for finished jobs on the current node."""
    import glob
    import shutil

    hostname = os.uname().nodename
    cleaned_dirs = []
    errors = []

    # Find all Ray session directories
    ray_session_dirs = glob.glob("/tmp/ray/session_*")

    for session_dir in ray_session_dirs:
        runtime_dir = os.path.join(session_dir, "runtime_resources", "working_dir_files")
        if not os.path.exists(runtime_dir):
            continue

        pkg_dirs = glob.glob(os.path.join(runtime_dir, "_ray_pkg_*"))

        for pkg_dir in pkg_dirs:
            logger.info("Deleting Ray package directory: %s", pkg_dir)
            if dry_run:
                cleaned_dirs.append(pkg_dir)
            else:
                try:
                    shutil.rmtree(pkg_dir)
                except Exception as e:
                    errors.append(f"Failed to delete {pkg_dir}: {e}")
                cleaned_dirs.append(pkg_dir)

    result = {
        "hostname": hostname,
        "cleaned_dirs": cleaned_dirs,
        "errors": errors,
        "dry_run": dry_run,
    }

    logger.info(f"Worker {hostname}: Cleaned {len(cleaned_dirs)} directories")
    return result


def run_cleanup_on_all_workers(job_ids: list[str], worker_ips: list[str], dry_run: bool = False):
    """Run cleanup on all workers in the cluster."""

    @ray.remote(max_calls=1)
    def _cleanup_temp_dirs():
        return cleanup_temp_dirs_on_node(job_ids, dry_run)

    # Submit cleanup tasks to all workers
    futures = []
    for ip in worker_ips:
        try:
            # Target specific nodes using resource constraints.
            # By requiring the TPU resource we ensure we're the only task on the node.
            future = _cleanup_temp_dirs.options(resources={f"node:{ip}": 0.01, "TPU": 4}).remote()
            futures.append((ip, future))
        except Exception as e:
            logger.warning(f"Failed to submit cleanup task to {ip}: {e}")

    # Collect results
    results = []
    for ip, future in futures:
        try:
            result = ray.get(future, timeout=60)
            results.append(result)
        except Exception as e:
            logger.error(f"Cleanup task failed on {ip}", exc_info=True)
            results.append(
                {
                    "hostname": f"{ip} (failed)",
                    "errors": [str(e)],
                    "dry_run": dry_run,
                }
            )

    return results


async def submit_cleanup_job(dry_run: bool = False, max_retries: int = 3):
    """Submit cleanup job to Ray cluster."""

    # N.B. We don't have access to the normal `@ray.remote` functionality
    # when submitting jobs from laptops - this is only available when RAY_ADDRESS
    # points to a GCS service, not the dashboard. For whatever reason, the job
    # submission client still works in this case.
    from ray.job_submission import JobSubmissionClient

    from marin.run.vars import REMOTE_DASHBOARD_URL

    client = JobSubmissionClient(REMOTE_DASHBOARD_URL)

    runtime_dict = {
        "working_dir": "src/marin/run/",
        "config": {"setup_timeout_seconds": 180},
    }

    # Call this script with --internal-run flag
    script_path = "./clean_ray_temp.py"
    dry_run_flag = "--dry-run" if dry_run else ""
    entrypoint = f"python {script_path} --internal-run {dry_run_flag}".strip()

    logger.info("Submitting Ray temp cleanup job")
    logger.info(f"Entrypoint: {entrypoint}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_id = f"ray_temp_cleanup-{timestamp}"

    # Submit job
    client.submit_job(
        submission_id=submission_id,
        entrypoint=entrypoint,
        runtime_env=runtime_dict,
    )
    logger.info(f"Job submitted with ID: {submission_id}")
    logger.info(f"Job URL: http://localhost:8265/#/jobs/{submission_id}")

    # Stream logs
    async for lines in client.tail_job_logs(submission_id):
        print(lines, end="")

    result = client.get_job_status(submission_id)
    logger.info(f"Job finished with status: {result}")


def run_cleanup_internal(dry_run: bool = False):
    """Internal cleanup function that runs on the Ray cluster."""
    try:
        ray.init(address="auto", ignore_reinit_error=True)

        # Get finished jobs and worker nodes
        finished_jobs = get_finished_jobs()
        worker_ips = get_worker_nodes()

        job_ids = [job.get("job_id", job.get("submission_id", "")) for job in finished_jobs]
        job_ids = [jid for jid in job_ids if jid]

        logger.info(f"Starting cleanup for {len(job_ids)} finished jobs across {len(worker_ips)} workers")

        # Run cleanup on all workers
        results = run_cleanup_on_all_workers(job_ids, worker_ips, dry_run)

        # Summary
        total_dirs_cleaned = sum(len(r["cleaned_dirs"]) for r in results)
        total_errors = sum(len(r["errors"]) for r in results)

        action = "would clean" if dry_run else "cleaned"
        logger.info(f"Cleanup summary: {action} {total_dirs_cleaned} directories, {total_errors} errors")

        for result in results:
            if result["cleaned_dirs"] or result["errors"]:
                logger.info(f"Node {result['hostname']}: {len(result['cleaned_dirs'])} dirs cleaned")
                for error in result["errors"]:
                    logger.warning(f"Node {result['hostname']}: {error}")

    finally:
        ray.shutdown()


def cron_mode(dry_run: bool = False, interval_hours: int = 1):
    """Run cleanup in cron mode - continuously every interval_hours."""
    logger.info(f"Starting cron mode - cleanup every {interval_hours} hour(s)")

    while True:
        try:
            logger.info("Starting scheduled cleanup...")
            asyncio.run(submit_cleanup_job(dry_run))
            logger.info("Scheduled cleanup completed")
        except Exception as e:
            logger.error(f"Error in scheduled cleanup: {e}")

        # Wait for next run
        sleep_seconds = interval_hours * 3600
        logger.info(f"Sleeping for {interval_hours} hour(s) until next cleanup...")
        try:
            time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            logger.info("Cron mode interrupted by user")
            break


def main():
    parser = argparse.ArgumentParser(description="Clean up Ray temporary directories from finished jobs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without actually deleting",
    )
    parser.add_argument("--cron", action="store_true", help="Run in cron mode - continuously every hour")
    parser.add_argument("--interval", type=int, default=1, help="Interval in hours for cron mode (default: 1)")
    parser.add_argument(
        "--internal-run",
        action="store_true",
        help="Internal flag to run cleanup directly on cluster",
    )

    args = parser.parse_args()

    if args.internal_run:
        # Run cleanup directly on the cluster
        run_cleanup_internal(args.dry_run)
    elif args.cron:
        # Run in cron mode
        cron_mode(args.dry_run, args.interval)
    else:
        # Submit one-off cleanup job
        asyncio.run(submit_cleanup_job(args.dry_run))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
