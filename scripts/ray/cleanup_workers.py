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
Cleanup Ray TPU workers: handle preempted TPUs and low disk space.

Usage:
    uv run python scripts/ray/cleanup_workers.py                    # all clusters
    uv run python scripts/ray/cleanup_workers.py --config infra/marin-us-central1.yaml
    uv run python scripts/ray/cleanup_workers.py --dry-run
"""

import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from glob import glob

import click
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class WorkerDiskInfo:
    tpu_name: str
    worker_id: int
    worker_ip: str
    free_pct: int
    available: str
    is_preemptible: bool = False
    topology: str = ""


@dataclass
class RestartResult:
    tpu_name: str
    worker_id: int
    success: bool
    error: str | None = None


def run_gcloud_ssh(tpu_name: str, worker_id: int, zone: str, project: str, command: str, timeout: int = 30):
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


def cleanup_preempted_tpus(zone: str, project: str, dry_run: bool = False) -> list[str]:
    """Delete TPUs in PREEMPTED or TERMINATED state."""
    result = subprocess.run(
        ["gcloud", "compute", "tpus", "tpu-vm", "list", f"--zone={zone}", f"--project={project}", "--format=json"],
        capture_output=True,
        text=True,
        check=True,
    )
    tpus = json.loads(result.stdout) if result.stdout.strip() else []

    preempted = [tpu.get("name", "").split("/")[-1] for tpu in tpus if tpu.get("state") in ("PREEMPTED", "TERMINATED")]

    if not preempted:
        return []

    for name in preempted:
        logger.info(f"{'[DRY RUN] Would delete' if dry_run else 'Deleting'} preempted TPU: {name}")

    if dry_run:
        return preempted

    def delete_tpu(name: str) -> tuple[str, bool, str | None]:
        try:
            subprocess.run(
                [
                    "gcloud",
                    "compute",
                    "tpus",
                    "tpu-vm",
                    "delete",
                    name,
                    f"--zone={zone}",
                    f"--project={project}",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return (name, True, None)
        except subprocess.CalledProcessError as e:
            return (name, False, e.stderr)

    with ThreadPoolExecutor(max_workers=16) as executor:
        for name, success, error in tqdm(
            executor.map(delete_tpu, preempted), total=len(preempted), desc="Deleting TPUs"
        ):
            if not success:
                logger.error(f"Failed to delete TPU {name}: {error}")

    return preempted


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
    return [name for name in result.stdout.strip().split("\n") if name]


def get_tpu_workers(tpu_name: str, zone: str, project: str) -> list[dict]:
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
    is_preemptible = tpu_info.get("schedulingConfig", {}).get("preemptible", False)
    topology = tpu_info.get("acceleratorType", "")
    workers = []
    for idx, endpoint in enumerate(tpu_info.get("networkEndpoints", [])):
        if "ipAddress" not in endpoint:
            continue
        workers.append(
            {
                "tpu_name": tpu_name,
                "worker_id": idx,
                "worker_ip": endpoint["ipAddress"],
                "is_preemptible": is_preemptible,
                "topology": topology,
            }
        )
    return workers


def get_worker_disk_info(
    tpu_name: str,
    worker_id: int,
    worker_ip: str,
    zone: str,
    project: str,
    is_preemptible: bool = False,
    topology: str = "",
) -> WorkerDiskInfo | None:
    try:
        result = run_gcloud_ssh(tpu_name, worker_id, zone, project, "df -h / | tail -n1")
        if result.returncode != 0 or not result.stdout.strip():
            return None
        parts = result.stdout.strip().split()
        if len(parts) >= 5:
            used_pct = int(parts[4].rstrip("%"))
            return WorkerDiskInfo(tpu_name, worker_id, worker_ip, 100 - used_pct, parts[3], is_preemptible, topology)
    except (subprocess.TimeoutExpired, ValueError, IndexError) as e:
        logger.error(f"Error checking {tpu_name} worker {worker_id}: {e}")
    return None


def print_workers_table(workers: list[WorkerDiskInfo]) -> None:
    """Print a simple text table of all workers."""
    if not workers:
        return

    # Column headers and widths
    headers = ["Worker Name", "Type", "Topology", "Free Disk"]
    rows = []
    for w in sorted(workers, key=lambda x: (x.tpu_name, x.worker_id)):
        worker_type = "preemptible" if w.is_preemptible else "manual"
        rows.append([f"{w.tpu_name}:{w.worker_id}", worker_type, w.topology, f"{w.free_pct}% ({w.available})"])

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Build table lines
    def format_row(cells: list[str]) -> str:
        return "  " + "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells))

    separator = "  " + "  ".join("-" * w for w in col_widths)
    lines = ["\nWorker Summary:", separator, format_row(headers), separator]
    for row in rows:
        lines.append(format_row(row))
    lines.append(separator)

    logger.info("\n".join(lines))


def restart_worker(
    tpu_name: str, worker_id: int, zone: str, project: str, docker_image: str, dry_run: bool
) -> RestartResult:
    if dry_run:
        logger.info(f"[DRY RUN] Would restart {tpu_name} worker {worker_id}")
        return RestartResult(tpu_name, worker_id, True)

    # Stop container
    for cmd in ["docker stop ray_docker && docker rm ray_docker", "docker rm -f ray_docker"]:
        try:
            if run_gcloud_ssh(tpu_name, worker_id, zone, project, cmd, timeout=60).returncode == 0:
                break
        except subprocess.TimeoutExpired:
            pass

    # Start container
    docker_cmd = f"docker run -d --net=host --name=ray_docker --init --privileged -v /tmp:/tmp -v /var/run/docker.sock:/var/run/docker.sock {docker_image} /bin/bash /tmp/entry.sh"  # noqa: E501
    try:
        result = run_gcloud_ssh(tpu_name, worker_id, zone, project, docker_cmd, timeout=60)
        if result.returncode == 0:
            logger.info(f"✓ {tpu_name} worker {worker_id}: restarted")
            return RestartResult(tpu_name, worker_id, True)
        return RestartResult(tpu_name, worker_id, False, result.stderr.strip())
    except subprocess.TimeoutExpired:
        return RestartResult(tpu_name, worker_id, False, "timeout")


def process_cluster(config_path: str, threshold: int, dry_run: bool, parallel: int) -> bool:
    """Process a single cluster. Returns True if successful."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    cluster_name = config["cluster_name"]
    zone = config["provider"]["availability_zone"]
    project = config["provider"].get("project", "hai-gcp-models")
    docker_image = config["docker"]["image"]

    logger.info(f"\n{'=' * 60}\nCluster: {cluster_name} ({zone})\n{'=' * 60}")

    deleted = cleanup_preempted_tpus(zone, project, dry_run)
    if deleted:
        logger.info(f"Cleaned up {len(deleted)} preempted/terminated TPUs")

    tpu_names = list_cluster_workers(cluster_name, zone, project)
    if not tpu_names:
        logger.info("No manual workers found")
        return True

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        worker_lists = list(
            tqdm(
                executor.map(lambda t: get_tpu_workers(t, zone, project), tpu_names),
                total=len(tpu_names),
                desc="Fetching workers",
            )
        )
    all_workers = [w for workers in worker_lists for w in workers]

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        disk_info = list(
            filter(
                None,
                tqdm(
                    executor.map(
                        lambda w: get_worker_disk_info(
                            w["tpu_name"],
                            w["worker_id"],
                            w["worker_ip"],
                            zone,
                            project,
                            w["is_preemptible"],
                            w["topology"],
                        ),
                        all_workers,
                    ),
                    total=len(all_workers),
                    desc="Checking disk",
                ),
            )
        )

    logger.info(f"Retrieved disk info for {len(disk_info)} workers")
    print_workers_table(disk_info)

    low_disk = [w for w in disk_info if w.free_pct < threshold]
    if not low_disk:
        logger.info(f"All workers have >{threshold}% free disk space")
        return True

    logger.info(f"Found {len(low_disk)} workers with <{threshold}% free:")
    for w in sorted(low_disk, key=lambda x: x.free_pct):
        logger.warning(f"  {w.tpu_name} worker {w.worker_id}: {w.free_pct}% free ({w.available})")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            tqdm(
                executor.map(
                    lambda w: restart_worker(w.tpu_name, w.worker_id, zone, project, docker_image, dry_run), low_disk
                ),
                total=len(low_disk),
                desc="Restarting",
            )
        )

    failures = [r for r in results if not r.success]
    if failures:
        for f in failures:
            logger.error(f"✗ {f.tpu_name} worker {f.worker_id}: {f.error}")
        return False

    return True


@click.command()
@click.option("--config", help="Path to cluster config YAML (default: all infra/*.yaml)")
@click.option("--threshold", type=int, default=10, help="Restart workers with less than this % free disk")
@click.option("--dry-run", is_flag=True, help="Show what would be done without making changes")
@click.option("--parallel", type=int, default=32, help="Number of parallel operations")
def main(config, threshold, dry_run, parallel):
    """Cleanup Ray TPU workers: handle preempted TPUs and low disk space."""
    if config:
        configs = [config]
    else:
        configs = sorted(glob("infra/marin-*.yaml"))
        if not configs:
            logger.error("No config files found in infra/marin-*.yaml")
            sys.exit(1)
        logger.info(f"Processing {len(configs)} clusters")

    failed = []
    for cfg in configs:
        try:
            if not process_cluster(cfg, threshold, dry_run, parallel):
                failed.append(cfg)
        except Exception as e:
            logger.error(f"Error processing {cfg}: {e}")
            failed.append(cfg)

    if failed:
        logger.error(f"\nFailed clusters: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
