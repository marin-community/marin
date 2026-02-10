# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCP utilities for cluster management.

This provides functions to get access to the current GCP configuration, list and
connect to TPUs, and find TPUs by IP address.
"""

import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def get_default_zone() -> str | None:
    """Get the default GCP zone."""
    try:
        result = run_gcloud_command(["gcloud", "config", "get-value", "compute/zone"])
        return result.stdout.strip() or None
    except RuntimeError:
        return None


# Compute instance utilities


def list_instances(project: str, zone: str, filter_expr: str | None = None) -> list[dict[str, Any]]:
    """List GCP compute instances."""
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--project={project}",
        f"--zones={zone}",
        "--format=json",
    ]

    if filter_expr:
        cmd.append(f"--filter={filter_expr}")

    result = run_gcloud_command(cmd)
    return json.loads(result.stdout)


def list_tpu_nodes(project: str, zone: str, filter_expr: str = "") -> list[dict[str, Any]]:
    """List TPU nodes in a zone."""
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "list",
        f"--project={project}",
        f"--zone={zone}",
        "--format=json",
    ]
    if filter_expr:
        cmd.append(f"--filter={filter_expr}")

    result = run_gcloud_command(cmd)
    return json.loads(result.stdout)


def delete_tpu_node(node_name: str, project: str, zone: str, quiet: bool = False) -> None:
    """Delete a TPU node."""
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "delete",
        node_name,
        f"--project={project}",
        f"--zone={zone}",
    ]

    if quiet:
        cmd.append("--quiet")

    run_gcloud_command(cmd)


def find_tpu_by_ip(target_ip: str, project: str, zone: str) -> tuple[str, str, int] | None:
    """Find TPU node by its internal IP address.

    Returns:
        Tuple of (tpu_name, zone, worker_index) or None if not found
    """
    tpu_nodes = list_tpu_nodes(project, zone)

    for node in tpu_nodes:
        network_endpoints = node.get("networkEndpoints", [])
        for worker_index, endpoint in enumerate(network_endpoints):
            if endpoint.get("ipAddress") == target_ip:
                # Extract simple name from full resource path
                full_name = node["name"]
                name_parts = full_name.split("/")
                if len(name_parts) >= 6:
                    simple_name = name_parts[5]  # nodes/simple-name
                    node_zone = name_parts[3]  # locations/zone
                    return simple_name, node_zone, worker_index
                else:
                    # Fallback for different naming schemes
                    return full_name, zone, worker_index

    return None


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


def terminate_tpus_in_cluster(project: str, zone: str, cluster_name: str) -> list[str]:
    """Terminate TPU nodes belonging to a specific cluster in parallel."""
    # Find TPUs with the correct node name.
    # Note we don't use the zone as a tag (we might have different clusters in a zone)
    # Nor the cluster name tag, as some clusters do not include the zone name and
    # are ambiguous e.g. eu-west4 vs eu-west4-a
    label_filter = f"labels.ray-node-name:ray-{cluster_name}-worker"
    nodes = list_tpu_nodes(project, zone, label_filter)
    terminated_nodes = []

    if not nodes:
        logger.info(f"No TPU nodes found for cluster {cluster_name} in zone {zone}")
        return terminated_nodes

    def delete_single_tpu(node: dict[str, Any]) -> str | None:
        node_name = node.get("name", "").split("/")[-1]
        try:
            delete_tpu_node(node_name, project, zone, quiet=True)
            logger.info(f"Terminated TPU node: {node_name}")
            return node_name
        except Exception as e:
            logger.error(f"Failed to terminate TPU node {node_name}: {e}")
            return None

    # Use ThreadPoolExecutor to terminate TPUs in parallel
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_node = {executor.submit(delete_single_tpu, node): node for node in nodes}

        for future in as_completed(future_to_node):
            result = future.result()
            if result is not None:
                terminated_nodes.append(result)

    logger.info(f"Terminated {len(terminated_nodes)} TPU nodes for cluster {cluster_name} in zone {zone}")
    return terminated_nodes


def terminate_head_node(cluster_name: str, project: str, zone: str) -> str | None:
    """Terminate the Ray cluster head node directly via gcloud."""
    # Use ray-cluster-name label for cluster identification
    filter_expr = f"labels.ray-node-type=head AND labels.ray-node-name=ray-{cluster_name}-head"
    instances = list_instances(project, zone, filter_expr)

    if not instances:
        logger.warning(f"No head node found for cluster {cluster_name} in zone {zone}")
        return None

    head_name = instances[0]["name"]

    cmd = [
        "gcloud",
        "compute",
        "instances",
        "delete",
        head_name,
        f"--project={project}",
        f"--zone={zone}",
        "--quiet",
    ]

    try:
        run_gcloud_command(cmd)
        logger.info(f"Terminated head node: {head_name}")
        return head_name
    except Exception as e:
        logger.error(f"Failed to terminate head node {head_name}: {e}")
        raise
