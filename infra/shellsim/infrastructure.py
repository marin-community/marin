# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the source-pinned image used by Shellsim's CoreWeave workers.

Public release storage and HTTPS belong to the shared buckets stack. The
worker image has no serving process or long-lived object-store reader key.
"""

import re

import pulumi
import pulumi_docker_build as docker_build

IMAGE_TAG = "ghcr.io/marin-community/iris-task:buildomatic"


def deploy(source_revision: str) -> pulumi.Output[str]:
    """Publish the worker image without building or pushing during preview."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("source_revision must be a full Git SHA")
    image = docker_build.Image(
        "ports-image",
        context=docker_build.BuildContextArgs(location=f"https://github.com/rjpower/shellsim.git#{source_revision}"),
        dockerfile=docker_build.DockerfileArgs(
            location=(
                "https://raw.githubusercontent.com/rjpower/shellsim/"
                f"{source_revision}/ports/buildomatic/images/Dockerfile"
            )
        ),
        platforms=[docker_build.Platform.LINUX_AMD64],
        pull=True,
        tags=[IMAGE_TAG],
        build_on_preview=False,
        push=True,
    )
    return image.ref
