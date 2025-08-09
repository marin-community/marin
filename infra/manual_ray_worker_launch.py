#!/usr/bin/python
# This script manually spins up a TPU slice and connects it to the provided Ray cluster head node.
# Usage: python infra/manual_ray_worker_launch.py --cluster_yaml <cluster_yaml> [--head <head-node-ip>]\
#          --tpu_type <tpu_type> [--project <project> --zone <zone> --tpu_name <tpu_name> --version <version>]\
#          [--reserved | --preemptible | --best_effort]

import argparse
import json
import logging
import subprocess

import levanter.infra.cli_helpers as cli
import yaml
from levanter.infra.tpus import setup_vm_docker, start_tpu_vm_queued_resources, tpu_ssh

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

TPU_TYPE_TO_VM_IMAGE = {
    "v5litepod": "v2-alpha-tpuv5-lite",
    "v5p": "v2-alpha-tpuv5",
    "v6e": "v2-alpha-tpuv6e",
}


def get_head_node_ip(cluster_name, region, zone):
    """Get the internal IP of the head node by querying GCP instances with Ray cluster labels."""
    try:
        # Query GCP instances with the appropriate Ray cluster labels
        cmd = [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--filter=labels.ray-node-type=head AND labels.ray-cluster-name={cluster_name}",
            "--format=json",
            f"--zones={zone}",  # Use the exact zone from the cluster config
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        result.check_returncode()

        instances = json.loads(result.stdout)
        if not instances:
            raise RuntimeError(f"No head node found for cluster {cluster_name} in zone {zone}")

        # Get the internal IP of the first matching instance
        return instances[0]["networkInterfaces"][0]["networkIP"]
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to query GCP instances: {e.stderr}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to get head node IP: {e!s}") from e


def main():
    parser = argparse.ArgumentParser()
    config = cli.load_config()

    cli.add_arg(parser, config, ["--cluster_yaml"], required=True)
    cli.add_arg(
        parser,
        config,
        ["--head"],
        required=False,
        help="address of head node. If not provided, will automatically find the head node of the specified cluster",
    )
    cli.add_capacity_type_args(parser, config)
    cli.add_arg(parser, config, ["--project"], default=cli.gcloud_config()["project"])
    cli.add_arg(parser, config, ["--tpu_name"], required=False, default=None)
    cli.add_arg(parser, config, ["--tpu_type"], required=True)
    cli.add_arg(parser, config, ["--version"], default=None)
    cli.add_arg(parser, config, ["--zone"], default=None, type=str, required=False)

    parser.add_argument("command", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # Load cluster config once and reuse it
    with open(args.cluster_yaml, "r") as f:
        cluster_config = yaml.safe_load(f)

    cluster_name = cluster_config["cluster_name"]
    region = cluster_config["provider"]["region"]
    zone = cluster_config["provider"]["availability_zone"]

    capacity_type = args.capacity_type

    tpu_type = args.tpu_type
    tpu_name = args.tpu_name
    if tpu_name is None:
        # generate a random unique name
        tpu_name = f"ray-worker-manual-{cli.default_run_id()}"
    tpu_gen = tpu_type.split("-")[0]
    version = args.version or TPU_TYPE_TO_VM_IMAGE.get(tpu_gen, "tpu-ubuntu2204-base")

    head = args.head
    if head is None:
        head = get_head_node_ip(cluster_name, region, zone)
        logger.info(f"Found head node IP for cluster {cluster_name}: {head}")

    if zone is None:
        zone = cli.gcloud_config()["zone"]

    if zone is None:
        raise ValueError("Zone must be specified or set in gcloud config.")

    image_id = cluster_config["docker"]["image"]
    container_name = cluster_config["docker"].get("container_name", "ray")
    worker_run_options = cluster_config["docker"].get("worker_run_options", ["-v", "/tmp:/tmp"])

    initialization_commands = cluster_config.get("initialization_commands", [])
    setup_commands = cluster_config.get("setup_commands", [])

    entry_command = f"ray start --address={head}:6379 --block"
    # TODO: would be friendlier to also sniff out the head and docker image from the cluster yaml

    # Extract bucket name - all clusters follow pattern: marin-{region}
    bucket_name = f"marin-{region}"
    logger.info(f"Using bucket: {bucket_name}")

    logger.info(f"Creating TPU with name: {tpu_name}")
    start_tpu_vm_queued_resources(
        tpu_name=tpu_name,
        tpu_type=tpu_type,
        capacity_type=capacity_type,
        version=version,
        zone=zone,
        node_count=1,
    )

    setup_vm_docker(
        tpu_name=tpu_name,
        zone=zone,
        node_count=1,
    )
    tpu_ssh(tpu_name, zone, 1, f"docker rm -f {container_name} || true")

    logger.info(f"Running on tpu_name... {tpu_name}")

    # run all initialization commands on HOST first
    for command in initialization_commands:
        tpu_ssh(tpu_name, zone, 1, command)

    # SIMPLE SOLUTION: Use sleep container + direct docker exec commands
    docker_command = [
        "docker",
        "run",
        "-d",
        "--net=host",
        f"--name={container_name}",
        "--init",
        "--privileged",
        # Add environment variables needed for gcsfuse
        "-e",
        f"BUCKET={bucket_name}",
        "-e",
        f"MARIN_PREFIX=gs://{bucket_name}",
        "-e",
        "AUTOSCALER_HEARTBEAT_TIMEOUT_S=600",
        *worker_run_options,
        image_id,
        "sleep",
        "3600",  # Keep container alive for direct command execution
    ]

    logger.info(f"Starting container: {' '.join(docker_command)}")
    tpu_ssh(tpu_name, zone, 1, *docker_command)

    # Run essential setup commands inside container (excluding gcsfuse which we handle separately)
    logger.info("Running setup commands inside container...")
    for command in setup_commands:
        # Skip gcsfuse command - we handle it separately with proper error handling
        if command.startswith("gcsfuse "):
            continue
        # Run other setup commands via docker exec
        setup_cmd = f"docker exec {container_name} bash -c '{command}'"
        tpu_ssh(tpu_name, zone, 1, setup_cmd)

    # Mount gcsfuse directly via docker exec (the simple solution!)
    logger.info("Mounting gcsfuse...")
    gcsfuse_command = (
        f"docker exec {container_name} gcsfuse --implicit-dirs --only-dir gcsfuse_mount {bucket_name} /opt/gcsfuse_mount"
    )
    tpu_ssh(tpu_name, zone, 1, gcsfuse_command)

    # Start Ray worker directly via docker exec
    logger.info("Starting Ray worker...")
    ray_start_command = f"docker exec -d {container_name} {entry_command}"
    tpu_ssh(tpu_name, zone, 1, ray_start_command)

    logger.info("Manual worker setup complete with gcsfuse!")
    logger.info("Verify with:")
    logger.info(f"  gcsfuse: docker exec {container_name} mount | grep gcsfuse")
    logger.info(f"  files: docker exec {container_name} ls /opt/gcsfuse_mount/models/")


if __name__ == "__main__":
    main()
