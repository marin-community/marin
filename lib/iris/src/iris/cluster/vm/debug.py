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

"""Debug utilities for log collection and VM cleanup."""

from __future__ import annotations

import logging
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def collect_docker_logs(
    vm_name: str,
    container_name: str,
    zone: str,
    project: str,
    output_dir: Path,
    is_tpu: bool = False,
) -> Path | None:
    """SSH to VM and collect docker logs.

    Args:
        vm_name: GCE VM or TPU name
        container_name: Docker container name (e.g., "iris-worker", "iris-controller")
        zone: GCP zone
        project: GCP project
        output_dir: Directory to write logs
        is_tpu: If True, use `gcloud compute tpus tpu-vm ssh`

    Returns:
        Path to log file, or None if collection failed
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = output_dir / f"{vm_name}-{container_name}-{timestamp}.log"
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_tpu:
        cmd = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            vm_name,
            f"--zone={zone}",
            f"--project={project}",
            "--command",
            f"sudo docker logs {container_name} 2>&1",
        ]
    else:
        cmd = [
            "gcloud",
            "compute",
            "ssh",
            vm_name,
            f"--zone={zone}",
            f"--project={project}",
            "--command",
            f"sudo docker logs {container_name} 2>&1",
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output_file.write_text(result.stdout + result.stderr)
        logger.info("Collected logs from %s:%s to %s", vm_name, container_name, output_file)
        return output_file
    except subprocess.TimeoutExpired:
        logger.warning("Timeout collecting logs from %s:%s", vm_name, container_name)
        return None
    except Exception as e:
        logger.warning("Failed to collect logs from %s: %s", vm_name, e)
        return None


def cleanup_iris_resources(zone: str, project: str, dry_run: bool = True) -> list[str]:
    """Delete all iris-* VMs and TPU slices in a zone.

    Args:
        zone: GCP zone
        project: GCP project
        dry_run: If True, only list resources without deleting

    Returns:
        List of resource names that were (or would be) deleted
    """
    deleted = []

    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            "--filter=name~^iris-",
            f"--zones={zone}",
            f"--project={project}",
            "--format=value(name)",
        ],
        capture_output=True,
        text=True,
    )
    vms = [v.strip() for v in result.stdout.strip().split("\n") if v.strip()]

    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            "--filter=name~^iris-",
            f"--zone={zone}",
            f"--project={project}",
            "--format=value(name)",
        ],
        capture_output=True,
        text=True,
    )
    tpus = [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]

    for vm in vms:
        if dry_run:
            logger.info("[DRY RUN] Would delete VM: %s", vm)
        else:
            subprocess.run(
                [
                    "gcloud",
                    "compute",
                    "instances",
                    "delete",
                    vm,
                    f"--zone={zone}",
                    f"--project={project}",
                    "-q",
                ],
                capture_output=True,
            )
            logger.info("Deleted VM: %s", vm)
        deleted.append(vm)

    for tpu in tpus:
        if dry_run:
            logger.info("[DRY RUN] Would delete TPU: %s", tpu)
        else:
            subprocess.run(
                [
                    "gcloud",
                    "compute",
                    "tpus",
                    "tpu-vm",
                    "delete",
                    tpu,
                    f"--zone={zone}",
                    f"--project={project}",
                    "-q",
                ],
                capture_output=True,
            )
            logger.info("Deleted TPU: %s", tpu)
        deleted.append(tpu)

    return deleted


def list_iris_tpus(zone: str, project: str) -> list[str]:
    """List all iris TPU VMs in the zone.

    Args:
        zone: GCP zone
        project: GCP project

    Returns:
        List of TPU names matching iris-* pattern
    """
    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            f"--project={project}",
            f"--zone={zone}",
            "--filter=name~^iris-",
            "--format=value(name)",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [n.strip() for n in result.stdout.strip().split("\n") if n.strip()]


def discover_controller_vm(zone: str, project: str) -> str | None:
    """Find iris controller VM by name pattern.

    Args:
        zone: GCP zone
        project: GCP project

    Returns:
        Controller VM name, or None if not found
    """
    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={project}",
            f"--zones={zone}",
            "--filter=name~^iris-controller",
            "--format=value(name)",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    names = [n.strip() for n in result.stdout.strip().split("\n") if n.strip()]
    if len(names) > 1:
        logger.warning(
            "Multiple controller VMs found: %s. Using first: %s",
            ", ".join(names),
            names[0],
        )
    return names[0] if names else None


def wait_for_port(port: int, host: str = "localhost", timeout: float = 30.0) -> bool:
    """Wait for a port to become available.

    Args:
        port: Port number to check
        host: Host to connect to (default: localhost)
        timeout: Maximum time to wait in seconds (default: 30.0)

    Returns:
        True if port is ready, False if timeout
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError, TimeoutError):
            time.sleep(0.5)
    return False


@contextmanager
def controller_tunnel(
    zone: str,
    project: str,
    local_port: int = 10000,
    tunnel_logger: logging.Logger | None = None,
    timeout: float = 60.0,
) -> Iterator[str]:
    """Establish SSH tunnel to controller and yield the local URL.

    Args:
        zone: GCP zone
        project: GCP project
        local_port: Local port to forward to (default: 10000)
        tunnel_logger: Optional logger for progress messages
        timeout: Timeout in seconds for tunnel establishment (default: 60.0)

    Yields:
        Local controller URL (e.g., "http://localhost:10000")

    Raises:
        RuntimeError: If no controller VM found or tunnel fails to establish

    Example:
        with controller_tunnel("europe-west4-b", "hai-gcp-models") as url:
            client = IrisClient.remote(url)
            job = client.submit(...)
    """
    vm_name = discover_controller_vm(zone, project)
    if not vm_name:
        raise RuntimeError(f"No controller VM found in zone {zone}")

    if tunnel_logger:
        tunnel_logger.info("Establishing SSH tunnel to %s...", vm_name)

    proc = subprocess.Popen(
        [
            "gcloud",
            "compute",
            "ssh",
            vm_name,
            f"--project={project}",
            f"--zone={zone}",
            "--",
            "-L",
            f"{local_port}:localhost:10000",
            "-N",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        if not wait_for_port(local_port, timeout=timeout):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            proc.terminate()
            proc.wait()
            raise RuntimeError(f"SSH tunnel failed to establish: {stderr}")

        if tunnel_logger:
            tunnel_logger.info("Tunnel ready: localhost:%d -> %s:10000", local_port, vm_name)

        yield f"http://localhost:{local_port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
