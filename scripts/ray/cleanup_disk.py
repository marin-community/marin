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

"""
Cleanup disk space on Ray TPU workers by restarting Docker containers.

This script:
1. Identifies all manual workers for a cluster
2. Checks disk space on each worker
3. Restarts ray_docker containers on workers with low disk space
4. Verifies containers are running after restart

Usage:
    python scripts/ray/cleanup_disk.py --config cluster.yaml [--threshold 20] [--dry-run]
"""

import argparse
import json
import logging
import subprocess
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class WorkerDiskInfo:
    tpu_name: str
    worker_id: int
    worker_ip: str
    free_pct: int
    available: str


@dataclass
class RestartResult:
    tpu_name: str
    worker_id: int
    success: bool
    error: str | None = None


def run_gcloud_ssh(
    tpu_name: str,
    worker_id: int,
    zone: str,
    project: str,
    command: str,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    """Execute a command on a TPU worker via gcloud ssh."""
    return subprocess.run(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            tpu_name,
            f"--zone={zone}",
            f"--project={project}",
            f"--worker={worker_id}",
            "--command",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _parse_df_output(output: str) -> tuple[int, str] | None:
    """Parse 'df -h /' output. Returns (free_pct, available) or None."""
    parts = output.strip().split()
    if len(parts) >= 5:
        try:
            used_pct = int(parts[4].rstrip("%"))
            return (100 - used_pct, parts[3])
        except (ValueError, IndexError):
            return None
    return None


def _stop_container(tpu_name: str, worker_id: int, zone: str, project: str) -> bool:
    """Stop and remove ray_docker container. Try graceful stop first, then force remove."""
    commands = [
        "docker stop ray_docker && docker rm ray_docker",
        "docker rm -f ray_docker",
    ]

    for i, cmd in enumerate(commands):
        try:
            result = run_gcloud_ssh(tpu_name, worker_id, zone, project, cmd, timeout=60)
            if result.returncode == 0:
                return True
            if i == 0:
                logger.warning("  Normal stop failed, trying force remove...")
        except subprocess.TimeoutExpired:
            logger.error(f"  Timeout running '{cmd}'")
        except Exception as e:
            logger.error(f"  Error running '{cmd}': {e}")

    return False


def _start_container(
    tpu_name: str, worker_id: int, zone: str, project: str, docker_image: str
) -> tuple[bool, str | None]:
    """Start ray_docker container. Returns (success, error_message)."""
    docker_run_cmd = f"""docker run -d \
  --net=host \
  --name=ray_docker \
  --init \
  --privileged \
  -v /tmp:/tmp \
  -v /var/run/docker.sock:/var/run/docker.sock \
  {docker_image} \
  /bin/bash /tmp/entry.sh"""

    try:
        result = run_gcloud_ssh(tpu_name, worker_id, zone, project, docker_run_cmd, timeout=60)
        if result.returncode == 0:
            return (True, None)
        return (False, result.stderr.strip())
    except subprocess.TimeoutExpired:
        return (False, "Timeout starting container")
    except Exception as e:
        return (False, str(e))


def get_cluster_config(config_file: str) -> dict:
    """Load cluster configuration from YAML file."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def list_cluster_workers(cluster_name: str, zone: str, project: str) -> list[str]:
    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            f"--zone={zone}",
            f"--project={project}",
            "--format=value(name)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    all_tpus = result.stdout.strip().split("\n")
    return all_tpus


def get_worker_disk_info(
    tpu_name: str, worker_id: int, worker_ip: str, zone: str, project: str
) -> WorkerDiskInfo | None:
    """Get disk space information for a single worker."""
    try:
        result = run_gcloud_ssh(tpu_name, worker_id, zone, project, "df -h / | tail -n1")

        if result.returncode == 0 and result.stdout.strip():
            parsed = _parse_df_output(result.stdout)
            if parsed:
                free_pct, available = parsed
                return WorkerDiskInfo(
                    tpu_name=tpu_name,
                    worker_id=worker_id,
                    worker_ip=worker_ip,
                    free_pct=free_pct,
                    available=available,
                )
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout checking {tpu_name} worker {worker_id}")
    except Exception as e:
        logger.error(f"Error checking {tpu_name} worker {worker_id}: {e}")

    return None


def get_tpu_workers(tpu_name: str, zone: str, project: str) -> list[dict]:
    """Get worker information for a TPU."""
    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            tpu_name,
            f"--zone={zone}",
            f"--project={project}",
            "--format=json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    tpu_info = json.loads(result.stdout)

    workers = []
    for idx, endpoint in enumerate(tpu_info.get("networkEndpoints", [])):
        if "ipAddress" not in endpoint:
            continue
        workers.append({"tpu_name": tpu_name, "worker_id": idx, "worker_ip": endpoint["ipAddress"]})
    return workers


def restart_ray_docker(
    tpu_name: str, worker_id: int, zone: str, project: str, docker_image: str, dry_run: bool = False
) -> RestartResult:
    """Stop, remove, and recreate the ray_docker container on a worker."""
    logger.info(f"Restarting {tpu_name} worker {worker_id}")

    if dry_run:
        logger.info(f"  [DRY RUN] Would restart {tpu_name} worker {worker_id}")
        return RestartResult(tpu_name=tpu_name, worker_id=worker_id, success=True)

    _stop_container(tpu_name, worker_id, zone, project)

    success, error = _start_container(tpu_name, worker_id, zone, project, docker_image)
    if success:
        logger.info(f"  ✓ {tpu_name} worker {worker_id}: Container restarted")
        return RestartResult(tpu_name=tpu_name, worker_id=worker_id, success=True)
    else:
        logger.error(f"  ✗ {tpu_name} worker {worker_id}: Failed - {error}")
        return RestartResult(tpu_name=tpu_name, worker_id=worker_id, success=False, error=error)


def verify_container_running(tpu_name: str, worker_id: int, zone: str, project: str) -> bool:
    """Verify that ray_docker container is running on a worker."""
    try:
        result = run_gcloud_ssh(
            tpu_name, worker_id, zone, project, "docker ps --filter name=ray_docker --format '{{.Status}}'"
        )
        if result.returncode == 0:
            return "Up" in result.stdout.strip()
    except Exception:
        pass

    return False


def main():
    parser = argparse.ArgumentParser(description="Cleanup disk space on Ray TPU workers by restarting Docker containers")
    parser.add_argument("--config", type=str, required=True, help="Path to cluster configuration YAML file")
    parser.add_argument("--cluster-name", type=str, help="Cluster name (overrides config file)")
    parser.add_argument(
        "--threshold",
        type=int,
        default=20,
        help="Restart workers with less than this percentage of free disk space (default: 20)",
    )
    parser.add_argument(
        "--workers", type=str, nargs="+", help="Specific TPU worker names to check (default: all manual workers)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without actually restarting containers"
    )
    parser.add_argument("--parallel", type=int, default=32, help="Number of parallel operations (default: 32)")

    args = parser.parse_args()

    config = get_cluster_config(args.config)
    cluster_name = args.cluster_name or config["cluster_name"]
    zone = config["provider"]["availability_zone"]
    project = config["provider"].get("project", "hai-gcp-models")
    docker_image = config["docker"]["image"]

    logger.info(f"Cluster: {cluster_name}")
    logger.info(f"Zone: {zone}")
    logger.info(f"Project: {project}")
    logger.info(f"Threshold: {args.threshold}% free space")
    logger.info(f"Dry run: {args.dry_run}")

    if args.workers:
        tpu_names = args.workers
        logger.info(f"Checking {len(tpu_names)} specified workers")
    else:
        tpu_names = list_cluster_workers(cluster_name, zone, project)
        logger.info(f"Found {len(tpu_names)} manual workers for cluster")

    if not tpu_names:
        logger.error("No workers found")
        sys.exit(1)

    logger.info("\nCollecting worker information...")
    all_workers = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(get_tpu_workers, tpu_name, zone, project): tpu_name for tpu_name in tpu_names}
        for future in tqdm(as_completed(futures), total=len(futures)):
            tpu_name = futures[future]
            result = future.result()
            all_workers.extend(result)
            logger.info(f"  {tpu_name}: {len(result)} workers")

    logger.info(f"\nChecking disk space on {len(all_workers)} workers...")
    disk_info: list[WorkerDiskInfo] = []

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(get_worker_disk_info, w["tpu_name"], w["worker_id"], w["worker_ip"], zone, project): w
            for w in all_workers
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                disk_info.append(result)
                if result.free_pct < args.threshold:
                    logger.warning(
                        f"  ⚠ {result.tpu_name} worker {result.worker_id}: "
                        f"{result.free_pct}% free ({result.available} available)"
                    )

    workers_to_restart = [w for w in disk_info if w.free_pct < args.threshold]

    if not workers_to_restart:
        logger.info(f"\n✓ All workers have >{args.threshold}% free disk space")
        return

    logger.info(f"\n{'='*80}")
    logger.info(f"Found {len(workers_to_restart)} workers with <{args.threshold}% free space:")
    for w in sorted(workers_to_restart, key=lambda x: x.free_pct):
        logger.info(f"  {w.tpu_name} worker {w.worker_id}: {w.free_pct}% free ({w.available})")

    logger.info(f"\n{'='*80}")
    logger.info("Restarting ray_docker containers...")

    restart_results: list[RestartResult] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(restart_ray_docker, w.tpu_name, w.worker_id, zone, project, docker_image, args.dry_run): w
            for w in workers_to_restart
        }

        for future in as_completed(futures):
            result = future.result()
            restart_results.append(result)

    if not args.dry_run:
        logger.info(f"\n{'='*80}")
        logger.info("Verifying containers are running...")

        verification_results = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {
                executor.submit(verify_container_running, w.tpu_name, w.worker_id, zone, project): w
                for w in workers_to_restart
            }

            for future in as_completed(futures):
                worker = futures[future]
                is_running = future.result()
                verification_results.append((worker, is_running))

                if is_running:
                    logger.info(f"  ✓ {worker.tpu_name} worker {worker.worker_id}: Running")
                else:
                    logger.error(f"  ✗ {worker.tpu_name} worker {worker.worker_id}: NOT running")

    logger.info(f"\n{'='*80}")
    logger.info("Summary:")
    success_count = sum(1 for r in restart_results if r.success)
    logger.info(f"  Successfully restarted: {success_count}/{len(restart_results)}")

    if not args.dry_run:
        running_count = sum(1 for _, running in verification_results if running)
        logger.info(f"  Verified running: {running_count}/{len(verification_results)}")

    failures = [r for r in restart_results if not r.success]
    if failures:
        logger.error("\nFailed restarts:")
        for f in failures:
            logger.error(f"  - {f.tpu_name} worker {f.worker_id}: {f.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
