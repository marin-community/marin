#!/usr/bin/env python
"""Reconnect TPU VMs that are tagged with their Ray cluster name."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml as pyyaml
from google.cloud import tpu_v2alpha1


def reconnect_reserved_tpus(cluster_yaml: str) -> None:
    # Load the cluster configuration
    try:
        config_data = pyyaml.safe_load(Path(cluster_yaml).read_text())
    except Exception as e:
        print(f"Failed to load {cluster_yaml}: {e}")
        return

    cluster_name = config_data.get("cluster_name")
    zone = config_data.get("provider", {}).get("availability_zone")
    project_id = config_data.get("provider", {}).get("project_id")

    if not cluster_name or not zone or not project_id:
        print(
            f"Missing required fields in {cluster_yaml}: cluster_name, provider.availability_zone, "
            f"or provider.project_id"
        )
        return

    tpu_client = tpu_v2alpha1.TpuClient()

    parent = f"projects/{project_id}/locations/{zone}"
    try:
        nodes = tpu_client.list_nodes(parent=parent)
    except Exception as e:
        print(f"Failed to list nodes in {zone}: {e}")
        return

    for node in nodes:
        labels = dict(node.labels)
        if labels.get("marin-ray-cluster-name") != cluster_name:
            continue
        if labels.get("marin-ray-worker-type") != "manual":
            continue

        tpu_name = node.name.split("/")[-1]
        print(f"Reconnecting {tpu_name} in {zone} to {cluster_name}")
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    str(Path(__file__).parent / "manual_ray_worker_launch.py"),
                    "--cluster_yaml",
                    str(cluster_yaml),
                    "--reserved",
                    "--tpu_type",
                    node.accelerator_type,
                    "--tpu_name",
                    tpu_name,
                    "--zone",
                    zone,
                ]
            )
        except Exception as e:
            print(f"Failed to reconnect {tpu_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Reconnect TPU VMs that are tagged with their Ray cluster name")
    parser.add_argument("cluster_yaml", help="Path to the cluster YAML configuration file")

    args = parser.parse_args()
    reconnect_reserved_tpus(args.cluster_yaml)


if __name__ == "__main__":
    main()
