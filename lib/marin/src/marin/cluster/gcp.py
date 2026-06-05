# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCP utilities for cluster management.

This provides functions to get access to the current GCP configuration, list and
connect to TPUs, and find TPUs by IP address.
"""

import json
import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def run_gcloud_command(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a gcloud command with error handling."""
    try:
        logger.info(f"Running {' '.join(cmd)}")
        return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gcloud command failed: {' '.join(cmd)}\nError: {e.stderr}") from e


def get_project_id() -> str | None:
    """Get the current GCP project ID."""
    try:
        result = run_gcloud_command(["gcloud", "config", "get-value", "project"])
        return result.stdout.strip() or None
    except RuntimeError:
        return None


def list_tpu_nodes(project: str, zone: str | None = None, filter_expr: str = "") -> list[dict[str, Any]]:
    """List TPU nodes in a zone."""
    zone_arg = zone or "-"
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "list",
        f"--project={project}",
        f"--zone={zone_arg}",
        "--format=json",
    ]
    if filter_expr:
        cmd.append(f"--filter={filter_expr}")

    result = run_gcloud_command(cmd)
    return json.loads(result.stdout)


def _parse_tpu_node_name(full_name: str, fallback_zone: str) -> tuple[str, str]:
    name_parts = full_name.split("/")
    if len(name_parts) >= 6:
        return name_parts[5], name_parts[3]
    return full_name, fallback_zone


def _find_tpu_by_name_in_nodes(
    target_name: str,
    tpu_nodes: list[dict[str, Any]],
    *,
    fallback_zone: str,
) -> tuple[str, str] | None:
    for node in tpu_nodes:
        simple_name, node_zone = _parse_tpu_node_name(node["name"], fallback_zone)
        if simple_name == target_name or node["name"] == target_name:
            return simple_name, node_zone

    return None


def find_tpu_by_name(target_name: str, project: str, zone: str | None = None) -> tuple[str, str] | None:
    """Find TPU node by name.

    Searches all zones by default.

    Returns:
        Tuple of (tpu_name, zone) or None if not found.
    """
    tpu_nodes = list_tpu_nodes(project, zone)

    return _find_tpu_by_name_in_nodes(target_name, tpu_nodes, fallback_zone=zone or "-")


def _find_tpu_by_ip_in_nodes(
    target_ip: str,
    tpu_nodes: list[dict[str, Any]],
    *,
    fallback_zone: str,
) -> tuple[str, str, int] | None:
    for node in tpu_nodes:
        network_endpoints = node.get("networkEndpoints", [])
        for worker_index, endpoint in enumerate(network_endpoints):
            # Iris worker RPC addresses can surface either TPU-internal IPs or NATed external IPs.
            external_ip = (endpoint.get("accessConfig") or {}).get("externalIp")
            if endpoint.get("ipAddress") == target_ip or external_ip == target_ip:
                simple_name, node_zone = _parse_tpu_node_name(node["name"], fallback_zone)
                return simple_name, node_zone, worker_index

    return None


def find_tpu_by_ip(target_ip: str, project: str, zone: str | None = None) -> tuple[str, str, int] | None:
    """Find TPU node by its internal or external IP address.

    Searches all zones by default.

    Returns:
        Tuple of (tpu_name, zone, worker_index) or None if not found
    """
    tpu_nodes = list_tpu_nodes(project, zone)

    return _find_tpu_by_ip_in_nodes(target_ip, tpu_nodes, fallback_zone=zone or "-")


def find_vm_by_ip(target_ip: str, project: str) -> tuple[str, str] | None:
    """Find a GCE VM by its internal IP address.

    Searches all zones.

    Returns:
        Tuple of (instance_name, zone) or None if not found
    """
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--project={project}",
        f"--filter=networkInterfaces[0].networkIP={target_ip}",
        "--format=json(name,zone)",
    ]
    result = run_gcloud_command(cmd)
    instances = json.loads(result.stdout)
    if not instances:
        return None

    if len(instances) > 1:
        details = ", ".join(f"{i['name']} ({i['zone'].split('/')[-1]})" for i in instances)
        raise RuntimeError(f"Multiple VMs found with IP {target_ip}: {details}")

    instance = instances[0]
    name = instance["name"]
    # zone is a full URL like .../zones/us-central1-a
    zone = instance["zone"].split("/")[-1]
    return name, zone


def ssh_to_vm(instance_name: str, zone: str, project: str, extra_args: list[str] | None = None) -> None:
    """SSH into a GCE VM."""
    cmd = [
        "gcloud",
        "compute",
        "ssh",
        instance_name,
        f"--zone={zone}",
        f"--project={project}",
    ]

    if extra_args:
        cmd.extend(["--", *extra_args])

    subprocess.run(cmd, check=True)


def ssh_to_tpu(tpu_name: str, zone: str, project: str, extra_args: list[str] | None = None, worker_id: int = 0) -> None:
    """SSH into a TPU node.

    Args:
        tpu_name: Name of the TPU
        zone: GCP zone
        project: GCP project
        extra_args: Additional SSH arguments
        worker_id: Worker index for multi-worker TPUs (default: 0)
    """
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        tpu_name,
        f"--zone={zone}",
        f"--project={project}",
        f"--worker={worker_id}",
    ]

    if extra_args:
        cmd.extend(["--", *extra_args])

    subprocess.run(cmd, check=True)
