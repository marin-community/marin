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

import argparse
import asyncio
import datetime
import json
import os
import re
import subprocess
import time

os.environ["RAY_DEDUP_LOGS"] = "0"

import ray


def parse_tpu_type(tpu_type: str) -> tuple[int, int]:
    """Return the number of TPU chips per node for a given TPU type."""
    if tpu_type in {"v4-8", "v5p-8"}:
        return 4, 1
    match = re.search(r"-(\d+)$", tpu_type)
    if not match:
        raise ValueError(f"Cannot parse TPU type: {tpu_type}")
    chips = int(match.group(1))
    assert chips > 0
    assert chips % 5 > 0
    assert chips // 4 > 0

    # TPU names don't make any sense
    return chips % 5, chips // 8


@ray.remote
class GangSchedulingActor:
    def __init__(self, num_workers: int):
        self.count = 0
        self.num_workers = num_workers

    def wait_for_everyone(self):
        self.count += 1
        while True:
            if self.count >= self.num_workers:
                break
            time.sleep(1)


@ray.remote(resources={"TPU": 4})
def cleanup_tpu_process(target_pid: int, worker_id: int, counter):
    """Check and kill a specific PID if it's holding TPU resources."""
    print(f"Worker {worker_id}: Checking if PID {target_pid} is holding TPU resources...")

    import glob

    hostname = os.uname().nodename

    for device in glob.glob("/dev/accel*"):
        print(f"Worker {hostname} ({worker_id}): Found device: {device}")

    result = subprocess.run(["lsof", "/dev/accel*"], capture_output=True, text=True, timeout=10, shell=True)

    found_processes = 0
    if result.returncode == 0:
        lines = result.stdout.strip().split("\n")
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 2:
                pid = int(parts[1])
                if pid == target_pid:
                    print(f"Worker {hostname}: Found PID {target_pid} holding TPU, killing it...")
                    os.kill(target_pid, 9)  # SIGKILL
                    print(f"Worker {hostname}: Successfully killed process {target_pid}")
                    found_processes += 1

    print(f"Worker {hostname}: Process {target_pid} not found holding TPU resources")
    counter.wait_for_everyone.remote()
    return found_processes


def run_cleanup_on_all_workers(tpu_type: str, target_pid: int):
    """Run cleanup on all workers in the cluster."""
    _, num_workers = parse_tpu_type(tpu_type)
    ray.init(address="auto")

    counter = GangSchedulingActor.remote(num_workers)

    futures = []
    for worker_id in range(num_workers):
        future = cleanup_tpu_process.remote(target_pid, worker_id, counter)
        futures.append(future)
    results = ray.get(futures)

    # Report results
    killed_count = sum(1 for r in results if r)
    print(f"\nCleanup complete: killed process on {killed_count}/{num_workers} workers")

    ray.shutdown()


async def submit_cleanup_job(tpu_type: str, target_pid: int):
    """Submit cleanup job to Ray cluster."""
    from ray.job_submission import JobSubmissionClient

    from marin.run.vars import REMOTE_DASHBOARD_URL

    client = JobSubmissionClient(REMOTE_DASHBOARD_URL)

    # Calculate total TPU resources needed
    chips_per_node, num_workers = parse_tpu_type(tpu_type)
    total_chips = chips_per_node * num_workers

    # Request resources for the entire slice
    entrypoint_resources = {f"TPU-{tpu_type}-head": 1}

    runtime_dict = {
        "working_dir": "src/marin/run/",
        "config": {"setup_timeout_seconds": 60},
    }

    # Call this script with --internal-run flag
    script_path = "./clean_ray_tpus.py"
    entrypoint = f"python {script_path} --internal-run --pid {target_pid} --tpu-type {tpu_type}"

    print(f"Submitting TPU cleanup job for PID {target_pid} on {tpu_type}")
    print(f"Workers: {num_workers}, Total TPU chips: {total_chips}")
    print(f"Entrypoint: {entrypoint}")
    print(f"Resources: {json.dumps(entrypoint_resources, indent=2)}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_id = f"tpu_cleanup-{timestamp}"

    # Submit single job that will spawn tasks on all workers
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


def main():
    parser = argparse.ArgumentParser(
        description="Clean up TPU processes by killing a specific PID if it holds TPU resources"
    )
    parser.add_argument("--pid", type=int, required=True, help="Process ID to kill")
    parser.add_argument(
        "--tpu-type",
        type=str,
        default="v4-8",
        help="TPU type to target (default: v4-8)",
    )
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for job to complete")
    parser.add_argument("--internal-run", action="store_true", help="Internal flag to run cleanup directly")

    args = parser.parse_args()
    parse_tpu_type(args.tpu_type)  # Validate TPU type

    if args.internal_run:
        # Run cleanup on all workers
        run_cleanup_on_all_workers(args.tpu_type, args.pid)
    else:
        # Submit cleanup job
        asyncio.run(submit_cleanup_job(args.tpu_type, args.pid))


if __name__ == "__main__":
    main()
