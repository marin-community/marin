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
SSH into a TPU node by its internal IP address.

Usage: python scripts/ray/ssh_by_ip.py <ip_address>
Example: python scripts/ray/ssh_by_ip.py 10.130.1.40
"""

import argparse
import json
import subprocess
import sys


def get_gcloud_config():
    """Get current gcloud project and zone configuration."""
    project_result = subprocess.run(
        ["gcloud", "config", "get-value", "project"], capture_output=True, text=True, check=True
    )
    project = project_result.stdout.strip()

    zone_result = subprocess.run(
        ["gcloud", "config", "get-value", "compute/zone"], capture_output=True, text=True, check=True
    )
    zone = zone_result.stdout.strip()

    return {"project": project, "zone": zone}


def list_tpu_nodes(project, zone):
    """List all TPU nodes and their internal IPs."""
    cmd = ["gcloud", "compute", "tpus", "tpu-vm", "list", f"--project={project}", f"--zone={zone}", "--format=json"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def find_tpu_by_ip(tpu_nodes, target_ip):
    """Find TPU node with matching internal IP."""
    for node in tpu_nodes:
        network_endpoints = node.get("networkEndpoints", [])
        for endpoint in network_endpoints:
            if endpoint.get("ipAddress") == target_ip:
                # Extract zone and simple name from full resource path
                # e.g. "projects/hai-gcp-models/locations/us-central2-b/nodes/ray-marin-us-central2-vllm-worker-52842f06-tpu"
                full_name = node["name"]
                name_parts = full_name.split("/")
                zone = name_parts[3]  # locations/us-central2-b
                simple_name = name_parts[5]  # nodes/simple-name
                return simple_name, zone
    return None, None


def ssh_to_tpu(tpu_name, zone, project, extra_args=None):
    """SSH into the TPU node."""
    cmd = ["gcloud", "compute", "tpus", "tpu-vm", "ssh", tpu_name, f"--zone={zone}", f"--project={project}"]

    if extra_args:
        cmd.extend(["--"] + extra_args)

    print(f"Connecting to TPU node {tpu_name} in zone {zone}...")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="SSH into a TPU node by its internal IP address")
    parser.add_argument("ip", help="Internal IP address of the TPU node")
    parser.add_argument("--project", help="GCP project (defaults to current gcloud config)")
    parser.add_argument("--zone", help="GCP zone (defaults to current gcloud config)")
    parser.add_argument("extra_args", nargs="*", help="Extra arguments to pass to SSH command")

    args = parser.parse_args()

    # Get gcloud config if project/zone not specified
    if not args.project or not args.zone:
        config = get_gcloud_config()
        project = args.project or config["project"]
        zone = args.zone or config["zone"]
    else:
        project = args.project
        zone = args.zone

    if not project:
        raise ValueError("No GCP project specified and none found in gcloud config")

    if not zone:
        raise ValueError("No GCP zone specified and none found in gcloud config")

    # List TPU nodes and find matching IP
    print(f"Looking for TPU node with IP {args.ip} in project {project}, zone {zone}...")
    tpu_nodes = list_tpu_nodes(project, zone)

    tpu_name, tpu_zone = find_tpu_by_ip(tpu_nodes, args.ip)

    if not tpu_name:
        raise ValueError(f"No TPU node found with IP address {args.ip}")

    # SSH to the found TPU
    ssh_to_tpu(tpu_name, tpu_zone, project, args.extra_args)


if __name__ == "__main__":
    main()
