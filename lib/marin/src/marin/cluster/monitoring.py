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

"""Monitoring utilities for cluster management."""

import logging
import time
from collections import Counter
from typing import Any

from .gcp import list_tpu_nodes
from .ray import get_cluster_utilization, list_nodes

logger = logging.getLogger(__name__)


def collect_tpu_metrics(project: str, zones: list[str]) -> dict[str, Any]:
    """Collect TPU metrics across multiple zones."""
    all_metrics = Counter()
    all_vms_to_delete = {}

    for zone in zones:
        try:
            zone_metrics, vms_to_delete = collect_zone_tpu_metrics(project, zone)
            all_metrics.update(zone_metrics)
            if vms_to_delete:
                all_vms_to_delete[zone] = vms_to_delete
        except Exception as e:
            logger.error(f"Failed to collect TPU metrics for zone {zone}: {e}")

    return {"metrics": dict(all_metrics), "vms_to_delete": all_vms_to_delete, "timestamp": time.time()}


def collect_zone_tpu_metrics(project: str, zone: str) -> tuple[dict[str, Any], list[str]]:
    """Collect TPU metrics for a specific zone."""
    nodes = list_tpu_nodes(project, zone)

    total_devices_zone = 0
    total_preemptible_devices_zone = 0
    nodes_types_zone = Counter()
    tpu_by_generation = Counter()
    vms_to_delete = []

    BAD_STATES = ["PREEMPTED", "TERMINATED"]
    GOOD_STATES = ["READY"]

    for node in nodes:
        state = node.get("state", "UNKNOWN")
        node_name = node.get("name", "").split("/")[-1]

        if state in BAD_STATES:
            logger.info(f"Node {node_name} is in bad state {state}, marking for deletion")
            vms_to_delete.append(node_name)
            continue

        if state not in GOOD_STATES:
            logger.info(f"Node {node_name} is in non-ready state {state}, skipping")
            continue

        # Get TPU configuration (e.g., v4-64)
        tpu_config = node.get("acceleratorType", "")
        if not tpu_config:
            continue

        # Calculate number of chips for this TPU type
        try:
            total_devices_this_tpu = int(tpu_config.split("-")[-1])
        except (ValueError, IndexError):
            logger.warning(f"Could not parse TPU type: {tpu_config}")
            continue

        total_devices_zone += total_devices_this_tpu
        generation = tpu_config.split("-")[0]

        # Check if preemptible
        is_preemptible = node.get("schedulingConfig", {}).get("preemptible", False)
        if is_preemptible:
            total_preemptible_devices_zone += total_devices_this_tpu

        nodes_types_zone[tpu_config] += 1
        tpu_by_generation[generation] += total_devices_this_tpu

    # Build metrics dict
    metrics = {}

    for tpu_type, count in nodes_types_zone.items():
        metrics[f"{zone}/devices/{tpu_type}"] = count
        metrics[f"devices/{tpu_type}"] = count

    for generation, count in tpu_by_generation.items():
        metrics[f"{zone}/devices/{generation}"] = count
        metrics[f"devices/{generation}"] = count

    metrics[f"{zone}/devices/total"] = total_devices_zone
    metrics[f"{zone}/devices/total_preemptible"] = total_preemptible_devices_zone
    metrics["devices/total"] = total_devices_zone
    metrics["devices/total_preemptible"] = total_preemptible_devices_zone

    return metrics, vms_to_delete


def collect_ray_metrics() -> dict[str, Any]:
    """Collect Ray cluster metrics."""
    try:
        utilization = get_cluster_utilization()
        nodes = list_nodes()

        # Count nodes by status
        node_status = Counter()
        for node in nodes:
            status = node.get("Alive", False)
            node_status["alive" if status else "dead"] += 1

        return {"utilization": utilization, "node_status": dict(node_status), "timestamp": time.time()}
    except Exception as e:
        logger.error("Failed to collect Ray metrics.", exc_info=True)
        return {"error": str(e), "timestamp": time.time()}


def log_metrics_to_wandb(metrics: dict[str, Any], project: str = "marin-monitoring") -> None:
    """Log metrics to Weights & Biases."""
    try:
        import wandb

        # Extract just the numeric metrics for wandb
        wandb_metrics = {}
        for key, value in metrics.get("metrics", {}).items():
            if isinstance(value, int | float):
                wandb_metrics[key] = value

        if wandb_metrics:
            wandb.init(project=project, reinit=True)
            wandb.log(wandb_metrics)
            wandb.finish()
            logger.info(f"Logged {len(wandb_metrics)} metrics to wandb")
        else:
            logger.warning("No numeric metrics found to log to wandb")

    except ImportError:
        logger.warning("wandb not available, skipping metric logging")
    except Exception as e:
        logger.error(f"Failed to log metrics to wandb: {e}")


def monitor_cluster_health(
    config_zones: dict[str, list[str]], project: str | None = None, log_to_wandb: bool = False
) -> dict[str, Any]:
    """Monitor overall cluster health across all zones.

    Args:
        config_zones: Dict mapping region names to list of zones
        project: GCP project ID
        log_to_wandb: Whether to log metrics to wandb

    Returns:
        Combined health metrics
    """
    if not project:
        from .gcp.auth import get_project_id

        project = get_project_id()

    all_zones = []
    for zones in config_zones.values():
        all_zones.extend(zones)

    # Collect TPU metrics
    tpu_metrics = collect_tpu_metrics(project, all_zones)

    # Collect Ray metrics
    ray_metrics = collect_ray_metrics()

    combined_metrics = {
        "tpu": tpu_metrics,
        "ray": ray_metrics,
        "summary": {
            "total_zones": len(all_zones),
            "zones_with_issues": len(tpu_metrics.get("vms_to_delete", {})),
            "timestamp": time.time(),
        },
    }

    if log_to_wandb:
        log_metrics_to_wandb(tpu_metrics)

    return combined_metrics


def get_cluster_health_summary(health_data: dict[str, Any]) -> str:
    """Generate a human-readable health summary."""
    summary_lines = []

    # TPU summary
    tpu_metrics = health_data.get("tpu", {}).get("metrics", {})
    total_devices = tpu_metrics.get("devices/total", 0)
    total_preemptible = tpu_metrics.get("devices/total_preemptible", 0)

    summary_lines.append(f"TPU Devices: {total_devices} total ({total_preemptible} preemptible)")

    # Ray summary
    ray_data = health_data.get("ray", {})
    if "error" in ray_data:
        summary_lines.append(f"Ray Status: ERROR - {ray_data['error']}")
    else:
        utilization = ray_data.get("utilization", {})
        total_nodes = utilization.get("total_nodes", 0)
        total_workers = utilization.get("total_workers", 0)
        summary_lines.append(f"Ray Cluster: {total_nodes} nodes, {total_workers} workers")

    # Issues summary
    vms_to_delete = health_data.get("tpu", {}).get("vms_to_delete", {})
    if vms_to_delete:
        total_bad_vms = sum(len(vms) for vms in vms_to_delete.values())
        summary_lines.append(f"Issues: {total_bad_vms} TPU nodes need cleanup")
    else:
        summary_lines.append("Issues: None detected")

    return "\n".join(summary_lines)
