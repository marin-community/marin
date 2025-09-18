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

"""Cleanup utilities for cluster management."""

import argparse
import asyncio
import datetime
import glob
import json
import logging
import os
import signal
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def kill_tpu_processes(tpu_type: str = "v4-8", remote_dashboard_url: str | None = None) -> None:
    """Kill TPU processes holding TPU resources.

    This submits a Ray job that runs across all TPU workers to check for and kill
    any processes holding TPU device files.
    """
    # Import here to avoid circular dependencies
    import asyncio

    logger.info(f"Submitting TPU cleanup job for TPU type: {tpu_type}")
    asyncio.run(submit_tpu_cleanup_job(tpu_type))


def _kill_active_tpu_processes() -> dict[str, Any]:
    """Check and kill any processes holding TPU resources on the current node."""
    hostname = os.uname().nodename
    result = {
        "hostname": hostname,
        "devices_found": 0,
        "killed": False,
        "error": None,
    }

    # Find TPU devices
    for _ in glob.glob("/sys/class/accel/accel*"):
        result["devices_found"] += 1

    try:
        subprocess_result = subprocess.run(
            ["lsof", "/sys/class/accel/accel*"],
            capture_output=True,
            text=True,
            timeout=10,
            shell=True,
        )

        if subprocess_result.returncode == 0:
            lines = subprocess_result.stdout.strip().split("\n")
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 2:
                    pid = int(parts[1])
                    logger.info(f"Found PID {pid} holding TPU resources, killing...")
                    os.kill(pid, signal.SIGKILL)
                    result["killed"] = True
                    logger.info(f"Successfully killed process {pid}")

        return result
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error checking/killing TPU processes: {e}")

    return result


def _list_worker_nodes() -> list[str]:
    """Get unique worker node IPs from Ray cluster."""
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


async def submit_tpu_cleanup_job(tpu_type: str):
    """Submit TPU cleanup job to Ray cluster."""
    from ray.job_submission import JobSubmissionClient

    from ..run.vars import REMOTE_DASHBOARD_URL

    client = JobSubmissionClient(REMOTE_DASHBOARD_URL)

    # Request resources for the entire slice
    entrypoint_resources = {f"TPU-{tpu_type}-head": 1}

    runtime_dict = {
        "working_dir": ".",
        "config": {"setup_timeout_seconds": 60},
    }

    entrypoint = f"python src/marin/cluster/cleanup.py --internal-run tpu --tpu-type {tpu_type}"

    print(f"Submitting TPU cleanup job for TPU type: {tpu_type}")
    print(f"Entrypoint: {entrypoint}")
    print(f"Resources: {json.dumps(entrypoint_resources, indent=2)}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_id = f"tpu_cleanup-{timestamp}"

    client.submit_job(
        submission_id=submission_id,
        entrypoint=entrypoint,
        runtime_env=runtime_dict,
        entrypoint_resources=entrypoint_resources,
    )
    print(f"Job submitted with ID: {submission_id}")
    print(f"Job URL: http://localhost:8265/#/jobs/{submission_id}")

    # Stream logs
    async for lines in client.tail_job_logs(submission_id):
        print(lines, end="")

    result = client.get_job_status(submission_id)
    print(f"Job finished with status: {result}")


def run_tpu_cleanup_on_all_workers(tpu_type: str):
    """Run TPU cleanup on all workers in the cluster."""
    import ray
    from levanter.infra.ray_tpu import run_on_pod

    @ray.remote(max_calls=1)
    def _cleanup_tpu_process():
        _kill_active_tpu_processes()

    run_on_pod(
        _cleanup_tpu_process,
        tpu_type,
        num_slices=1,
        max_retries_failure=0,
        max_retries_preemption=0,
    )
    ray.shutdown()


def main():
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(description="Cluster cleanup utilities")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Temp cleanup subcommand
    temp_parser = subparsers.add_parser("temp", help="Clean up Ray temporary files")
    temp_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without actually deleting",
    )
    temp_parser.add_argument(
        "--internal-run",
        action="store_true",
        help="Internal flag to run cleanup directly on cluster",
    )

    # TPU cleanup subcommand
    tpu_parser = subparsers.add_parser("tpu", help="Clean up TPU processes by PID")
    tpu_parser.add_argument(
        "--tpu-type",
        type=str,
        default="v4-8",
        help="TPU type to target (default: v4-8)",
    )
    tpu_parser.add_argument("--no-wait", action="store_true", help="Don't wait for job to complete")
    tpu_parser.add_argument("--internal-run", action="store_true", help="Internal flag to run cleanup directly")

    args = parser.parse_args()

    if args.command == "tpu":
        if args.internal_run:
            run_tpu_cleanup_on_all_workers(args.tpu_type)
        else:
            asyncio.run(submit_tpu_cleanup_job(args.tpu_type))
    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
