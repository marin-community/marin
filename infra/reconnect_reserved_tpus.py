#!/usr/bin/env python
"""Reconnect TPU VMs that are tagged with their Ray cluster name."""

import subprocess
import sys
from pathlib import Path

import yaml as pyyaml
from google.cloud import tpu_v2alpha1
from tpu_monitor import PROJECT_NAME, YAML_FILES


def reconnect_reserved_tpus() -> None:
    tpu_client = tpu_v2alpha1.TpuClient()

    for yaml_file in YAML_FILES:
        try:
            config_data = pyyaml.safe_load(Path(yaml_file).read_text())
        except Exception as e:
            print(f"Failed to load {yaml_file}: {e}")
            continue

        cluster_name = config_data.get("cluster_name")
        zone = config_data.get("provider", {}).get("availability_zone")
        if not cluster_name or not zone:
            continue

        parent = f"projects/{PROJECT_NAME}/locations/{zone}"
        try:
            nodes = tpu_client.list_nodes(parent=parent)
        except Exception as e:
            print(f"Failed to list nodes in {zone}: {e}")
            continue

        for node in nodes:
            labels = dict(node.labels)
            if labels.get("ray-cluster-name") != cluster_name:
                continue

            tpu_name = node.name.split("/")[-1]
            print(f"Reconnecting {tpu_name} in {zone} to {cluster_name}")
            try:
                subprocess.check_call(
                    [
                        sys.executable,
                        str(Path(__file__).parent / "manual_ray_worker_launch.py"),
                        "--cluster_yaml",
                        str(yaml_file),
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


if __name__ == "__main__":
    reconnect_reserved_tpus()
