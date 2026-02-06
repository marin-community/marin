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

import logging
import re
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


def _docker_log_ssh_command(
    vm_name: str,
    container_name: str,
    zone: str,
    project: str,
    is_tpu: bool,
    follow: bool = False,
) -> list[str]:
    """Build gcloud SSH command for docker log collection."""
    docker_cmd = "sudo docker logs"
    if follow:
        docker_cmd += " -f"
    docker_cmd += f" {container_name} 2>&1"

    base = ["gcloud", "compute"]
    if is_tpu:
        base += ["tpus", "tpu-vm"]
    return [*base, "ssh", vm_name, f"--zone={zone}", f"--project={project}", "--command", docker_cmd]


def stream_docker_logs(
    vm_name: str,
    container_name: str,
    zone: str,
    project: str,
    output_file: Path,
    is_tpu: bool = False,
    stop_event: threading.Event | None = None,
) -> None:
    """Stream docker logs from a VM to a file until stop_event is set.

    Blocking call. Uses `docker logs -f` via gcloud SSH.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = _docker_log_ssh_command(vm_name, container_name, zone, project, is_tpu, follow=True)

    with open(output_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        try:
            if stop_event:
                while not stop_event.is_set():
                    if proc.poll() is not None:
                        break
                    stop_event.wait(1.0)
            else:
                proc.wait()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def list_docker_containers(
    vm_name: str,
    zone: str,
    project: str,
    label_filter: str,
    is_tpu: bool = False,
) -> list[str]:
    """List docker container names on a VM matching a label filter.

    Args:
        vm_name: VM or TPU name to SSH into
        zone: GCP zone
        project: GCP project
        label_filter: Docker label filter (e.g., "iris.managed=true")
        is_tpu: Whether the target is a TPU VM

    Returns:
        List of container names matching the filter
    """
    docker_cmd = f"sudo docker ps -a --filter label={label_filter} --format '{{{{.Names}}}}'"

    base = ["gcloud", "compute"]
    if is_tpu:
        base += ["tpus", "tpu-vm"]
    cmd = [*base, "ssh", vm_name, f"--zone={zone}", f"--project={project}", "--command", docker_cmd]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        logger.debug("Failed to list containers on %s: %s", vm_name, result.stderr.strip()[:200])
        return []
    return [name.strip() for name in result.stdout.strip().split("\n") if name.strip()]


def cleanup_iris_resources(
    zone: str,
    project: str,
    label_prefix: str = "iris",
    dry_run: bool = True,
) -> list[str]:
    """Delete VMs and TPU slices matching the label prefix in a zone.

    Args:
        zone: GCP zone
        project: GCP project
        label_prefix: Prefix used for resource naming (default: "iris").
                     Controller VMs are named "iris-controller-{prefix}",
                     TPU slices are named "{prefix}-{scale_group}-{timestamp}".
        dry_run: If True, only list resources without deleting

    Returns:
        List of resource names that were (or would be) deleted
    """
    deleted = []

    # Controller VMs are named "iris-controller-{label_prefix}"
    controller_vm = discover_controller_vm(zone, project, label_prefix)
    vms = [controller_vm] if controller_vm else []

    # TPU slices are named "{label_prefix}-{scale_group}-{timestamp}"
    # and are labeled with "{label_prefix}-managed=true"
    tpu_label_filter = f"labels.{label_prefix}-managed=true"
    try:
        result = subprocess.run(
            [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "list",
                f"--filter={tpu_label_filter}",
                f"--zone={zone}",
                f"--project={project}",
                "--format=value(name)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        tpus = [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]
    except subprocess.TimeoutExpired:
        logger.warning("TPU list command timed out after 30s, skipping TPU cleanup")
        tpus = []

    for vm in vms:
        if dry_run:
            logger.info("[DRY RUN] Would delete VM: %s", vm)
        else:
            try:
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
                    timeout=60,
                )
                logger.info("Deleted VM: %s", vm)
            except subprocess.TimeoutExpired:
                logger.warning("VM delete timed out after 60s: %s", vm)
        deleted.append(vm)

    for tpu in tpus:
        if dry_run:
            logger.info("[DRY RUN] Would delete TPU: %s", tpu)
        else:
            try:
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
                    timeout=60,
                )
                logger.info("Deleted TPU: %s", tpu)
            except subprocess.TimeoutExpired:
                logger.warning("TPU delete timed out after 60s: %s", tpu)
        deleted.append(tpu)

    return deleted


def list_iris_tpus(zone: str, project: str, label_prefix: str = "iris") -> list[str]:
    """List all TPU VMs matching the label prefix in the zone.

    Args:
        zone: GCP zone
        project: GCP project
        label_prefix: Prefix used for resource naming (default: "iris").
                     TPUs are filtered by the label "{label_prefix}-managed=true".

    Returns:
        List of TPU names matching the label filter
    """
    label_filter = f"labels.{label_prefix}-managed=true"
    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            f"--project={project}",
            f"--zone={zone}",
            f"--filter={label_filter}",
            "--format=value(name)",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    # Filter client-side: gcloud TPU --filter with regex is unreliable
    return [n.strip() for n in result.stdout.strip().split("\n") if n.strip() and n.strip().startswith("iris-")]


def discover_controller_vm(zone: str, project: str, label_prefix: str = "iris") -> str | None:
    """Find controller VM by name pattern for the given prefix.

    Args:
        zone: GCP zone
        project: GCP project
        label_prefix: Prefix used for resource naming (default: "iris").
                     Controller VMs are named "iris-controller-{label_prefix}".

    Returns:
        Controller VM name, or None if not found
    """
    # Controller VMs are named "iris-controller-{label_prefix}"
    name_filter = f"name~^iris-controller-{re.escape(label_prefix)}$"
    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={project}",
            f"--zones={zone}",
            f"--filter={name_filter}",
            "--format=value(name)",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    names = [n.strip() for n in result.stdout.strip().split("\n") if n.strip()]
    if len(names) > 1:
        raise RuntimeError(
            f"Multiple controller VMs found for prefix '{label_prefix}': {', '.join(names)}. "
            "This indicates a resource leak or configuration error."
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
    label_prefix: str = "iris",
) -> Iterator[str]:
    """Establish SSH tunnel to controller and yield the local URL.

    Args:
        zone: GCP zone
        project: GCP project
        local_port: Local port to forward to (default: 10000)
        tunnel_logger: Optional logger for progress messages
        timeout: Timeout in seconds for tunnel establishment (default: 60.0)
        label_prefix: Prefix used for resource naming (default: "iris").

    Yields:
        Local controller URL (e.g., "http://localhost:10000")

    Raises:
        RuntimeError: If no controller VM found or tunnel fails to establish

    Example:
        with controller_tunnel("europe-west4-b", "hai-gcp-models") as url:
            client = IrisClient.remote(url)
            job = client.submit(...)
    """
    vm_name = discover_controller_vm(zone, project, label_prefix)
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
            "-o",
            "ServerAliveInterval=60",
            "-o",
            "ServerAliveCountMax=3",
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
