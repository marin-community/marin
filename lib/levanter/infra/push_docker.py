#!/usr/bin/python

# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Build and deploy the Levanter base image to Artifact Registry or Docker Hub.

It is not necessary to run this yourself unless you are deploying a new base image: the launch
script will automatically build and deploy an image based on your current code.
"""
import argparse

from levanter.infra import cli_helpers as cli
from levanter.infra import docker
from levanter.infra.docker import build_docker, push_to_gcp, push_to_github


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and push Docker image to Artifact Registry.")
    config = cli.load_config()
    cli.add_arg(parser, config, ["--project"], help="GCP project ID")
    cli.add_arg(parser, config, ["--region"], help="Artifact Registry region (e.g., us-west4)")
    cli.add_arg(parser, config, ["--repository"], default="levanter", help="Artifact Registry repository name")
    cli.add_arg(parser, config, ["--image"], default="levanter", help="Docker image name.")
    cli.add_arg(parser, config, ["--tag"], default="latest", help="Docker image tag.")
    cli.add_arg(parser, config, ["--github_user"], default=None, help="Github user name.")
    cli.add_arg(parser, config, ["--github_token"], default=None, help="Github token.")
    cli.add_arg(parser, config, ["--docker_file"], default="docker/tpu/Dockerfile.base", help="Dockerfile to use.")
    cli.add_arg(parser, config, ["--extra_context"], required=False, default=None)

    # push to either github or GCP
    cli.add_arg(parser, config, ["--docker_target"], choices=["github", "gcp", "ghcr"], required=True)

    args = parser.parse_args()

    with docker.copy_extra_ctx(args.extra_context) as extra_ctx:
        build_args = {"EXTRA_CTX": extra_ctx} if extra_ctx else None
        local_id = build_docker(docker_file=args.docker_file, image_name=args.image, tag=args.tag)

    if args.docker_target in ["github", "ghcr"]:
        assert args.github_user, "Must specify --github_user when pushing to Github"
        assert args.github_token, "Must specify --github_token when pushing to Github"
        push_to_github(local_id=local_id, github_user=args.github_user, github_token=args.github_token)
    else:
        assert args.region, "Must specify --region when pushing to GCP"
        assert args.project, "Must specify --project when pushing to GCP"
        assert args.repository, "Must specify --repository when pushing to GCP"

        push_to_gcp(local_id, args.project, args.region, args.repository)
