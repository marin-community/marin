#!/usr/bin/python
# This script manually spins up a TPU slice and connects it to the provided Ray cluster head node.
# Usage: python infra/manual_ray_worker_launch.py --cluster_yaml <cluster_yaml> --head <head-node-ip>\
#          --tpu_type <tpu_type> [--project <project> --zone <zone> --tpu_name <tpu_name> --version <version>]\
#          [--reserved | --preemptible | --best_effort]

import argparse
import tempfile

import levanter.infra.cli_helpers as cli
import yaml
from levanter.infra.tpus import run_command, setup_vm_docker, start_tpu_vm_queued_resources, tpu_ssh


def main():
    parser = argparse.ArgumentParser()
    config = cli.load_config()

    cli.add_arg(parser, config, ["--cluster_yaml"], required=True)
    cli.add_arg(parser, config, ["--head"], required=True, help="address of head node")
    cli.add_capacity_type_args(parser, config)
    cli.add_arg(parser, config, ["--project"], default=cli.gcloud_config()["project"])
    cli.add_arg(parser, config, ["--tpu_name"], required=False, default=None)
    cli.add_arg(parser, config, ["--tpu_type"], required=True)
    cli.add_arg(parser, config, ["--version"], default="tpu-ubuntu2204-base")
    cli.add_arg(parser, config, ["--zone"], default=None, type=str, required=False)

    parser.add_argument("command", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    capacity_type = args.capacity_type

    tpu_type = args.tpu_type
    tpu_name = args.tpu_name
    if tpu_name is None:
        # generate a random unique name
        tpu_name = f"ray-worker-manual-{cli.default_run_id()}"
    version = args.version
    zone = args.zone

    head = args.head

    if zone is None:
        zone = cli.gcloud_config()["zone"]

    if zone is None:
        raise ValueError("Zone must be specified or set in gcloud config.")

    with open(args.cluster_yaml, "r") as f:
        cluster_yaml = yaml.safe_load(f)

    image_id = cluster_yaml["docker"]["image"]

    setup_commands = [line for line in cluster_yaml.get("setup_commands", [])]

    entry_command = f"ray start --address={head}:6379 --block"
    # TODO: would be friendlier to also sniff out the head and docker image from the cluster yaml

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
    tpu_ssh(tpu_name, zone, 1, "docker rm -f ray || true")

    # first we want to make a new entrypoint that starts ray and runs the setup commands
    print(f"Running on tpu_name... {tpu_name}")
    with tempfile.NamedTemporaryFile("w", prefix="entry", suffix=".sh") as f:
        f.write("#!/bin/bash\n")
        for command in setup_commands:
            f.write(command + "\n")

        f.write(entry_command + "\n")

        f.flush()

        # copy the entrypoint to the tpu
        run_command(
            *(f"gcloud compute tpus tpu-vm scp {f.name} {tpu_name}:/tmp/entry.sh --zone={zone} --worker=all".split(" "))
        )
        # chmod the entrypoint
        tpu_ssh(tpu_name, zone, 1, "chmod a+rwx /tmp/entry.sh")

        docker_command = [
            "docker",
            "run",
            "-d",
            "--name=ray",
            "--privileged",
            "--shm-size=32gb",
            "--net=host",
            "--init",
            "-v",
            "/tmp:/tmp",
            image_id,
            "/bin/bash",  # Use bash as entrypoint to set up entry.sh
            "/tmp/entry.sh",
        ]

        tpu_ssh(tpu_name, zone, 1, *docker_command)


if __name__ == "__main__":
    main()
