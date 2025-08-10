#!/usr/bin/python
import argparse
import getpass
import os
import subprocess
import sys
import time
from pathlib import Path

import levanter.infra.cli_helpers as cli
import levanter.infra.docker as docker
import levanter.infra.tpus
from levanter.infra.tpus import launch_job


# default: tpu-ubuntu2204-base
TPU_TYPE_TO_VM_IMAGE = {
    "v5litepod": "v2-alpha-tpuv5-lite",
    "v5p": "v2-alpha-tpuv5",
    "v6e": "v2-alpha-tpuv6e",
}


def main():
    parser = argparse.ArgumentParser()
    config = cli.load_config()

    cli.add_arg(
        parser, config, ["--autodelete"], default=False, action="store_true", help="Delete TPU after job completes."
    )
    cli.add_arg(parser, config, ["--docker_base_image"], default="ghcr.io/stanford-crfm/levanter-base:latest")
    cli.add_arg(parser, config, ["--docker_repository"], default="levanter")
    cli.add_arg(parser, config, ["--foreground"], default=False, action="store_true")
    cli.add_arg(parser, config, ["--image_name"], default=f"levanter-{getpass.getuser()}")
    cli.add_capacity_type_args(parser, config)
    cli.add_arg(parser, config, ["--project"], default=cli.gcloud_config()["project"])
    cli.add_arg(parser, config, ["--tpu_name"], required=True)
    cli.add_arg(parser, config, ["--tpu_type"], required=True)
    cli.add_arg(parser, config, ["--node_count"], default=1, type=int)
    cli.add_arg(parser, config, ["--version"], default=None)
    cli.add_arg(parser, config, ["--zone"], default=None, type=str, required=False)
    cli.add_arg(parser, config, ["--retries"], default=10, type=int)
    cli.add_arg(parser, config, ["--run_id"], default=cli.default_run_id(), type=str)
    cli.add_arg(parser, config, ["--docker_registry"], default="gcp", choices=["gcp", "ghcr"])
    cli.add_arg(parser, config, ["--github_user"], type=str)
    cli.add_arg(parser, config, ["--github_token"], type=str)
    cli.add_arg(parser, config, ["--extra_context"], type=Path, required=False, default=None)

    parser.add_argument("-e", "--env", action="append", nargs=2, metavar=("KEY", "VALUE"))
    parser.add_argument("command", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    autodelete = args.autodelete
    command = args.command
    docker_repository = args.docker_repository
    foreground = args.foreground
    image_id = args.image_name
    capacity_type = args.capacity_type
    project = args.project
    if args.retries < 0:
        retries = 10000000
    else:
        retries = args.retries
    tpu_name = args.tpu_name
    tpu_type = args.tpu_type

    tpu_gen = tpu_type.split("-")[0]
    version = args.version or TPU_TYPE_TO_VM_IMAGE.get(tpu_gen, "tpu-ubuntu2204-base")

    if not args.version:
        print(f"Using default version: {version}", file=sys.stderr)

    node_count = args.node_count
    zone = args.zone
    run_id = args.run_id
    registry = args.docker_registry
    github_user = args.github_user
    github_token = args.github_token or os.getenv("GITHUB_DOCKER_TOKEN")
    extra_context = args.extra_context

    if zone is None:
        zone = cli.gcloud_config()["zone"]

    if zone is None:
        raise ValueError("Zone must be specified or set in gcloud config.")

    region = "-".join(zone.split("-")[:-1])

    env = config.env_for_accel(tpu_type)
    for key, value in args.env or []:
        env[key] = value

    if "WANDB_PROJECT" not in env:
        env["WANDB_PROJECT"] = "levanter"

    env["GIT_COMMIT"] = cli.get_git_commit()
    env["RUN_ID"] = run_id
    env["WANDB_DOCKER"] = image_id

    if command[0] == "--":
        command = command[1:]

    # make an image tag based on the unix timestamp to ensure we always pull the latest image
    tag = int(time.time())

    with docker.copy_extra_ctx(extra_context) as extra_context:
        build_args = {"EXTRA_CTX": extra_context} if extra_context else {}
        base_image, base_tag = docker.split_image_and_tag(args.docker_base_image)
        build_args["IMAGE"] = base_image
        build_args["TAG"] = base_tag

        local_id = docker.build_docker(
            docker_file="docker/tpu/Dockerfile.incremental", image_name=image_id, tag=tag, build_args=build_args
        )

    if registry == "ghcr":
        full_image_id = docker.push_to_github(
            local_id=local_id,
            github_user=github_user,
            github_token=github_token,
        )
    elif registry == "gcp":
        full_image_id = docker.push_to_gcp(
            local_id=local_id,
            project_id=project,
            region=region,
            repository=docker_repository,
        )
    else:
        raise ValueError(f"Unknown docker registry: {registry}")

    failure = None

    for i in range(retries + 1):
        try:
            launch_job(
                command=command,
                tpu_name=tpu_name,
                tpu_type=tpu_type,
                capacity_type=capacity_type,
                zone=zone,
                node_count=node_count,
                full_image_id=full_image_id,
                env=env,
                foreground=foreground,
                version=version,
            )
        except subprocess.CalledProcessError as e:  # noqa: F841
            print(f"Error running command {e.cmd}")
            if i < retries - 1:
                print("Retrying... %d/%d" % (i + 1, retries))
            else:
                print("Retries exhausted. Raising error.")
                print(f"Error running command {e.cmd}")
                print(f"Output: {e.output}")
                failure = e
        else:
            print("Job finished with no error.")
            break

    try:
        if autodelete:
            print("Autodelete is set to True. Tearing down machine...")
            levanter.infra.tpus.run_command(
                "gcloud",
                "alpha",
                "compute",
                "tpus",
                "queued-resources",
                "delete",
                tpu_name,
                "--quiet",
                f"--zone={zone}",
                "--force",
            )
    except Exception as e:
        print(f"Error tearing down TPU {tpu_name}: {e}")
        if failure:
            raise failure
        else:
            raise e

    if failure:
        raise failure


if __name__ == "__main__":
    main()
