# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared Docker image build policy for Pulumi application deployments."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pulumi
import pulumi_docker_build as docker_build

CACHE_COMPRESSION_LEVEL = 3


@dataclass(frozen=True)
class DockerImageConfig:
    """Inputs for one cached, pushed linux/amd64 image."""

    build_context: str
    dockerfile: str
    build_args: Mapping[str, pulumi.Input[str]]
    cache_ref: pulumi.Input[str]
    tags: Sequence[pulumi.Input[str]]


def cached_amd64_image(
    name: str,
    config: DockerImageConfig,
    *,
    parent: pulumi.ComponentResource,
    provider: pulumi.ProviderResource | None = None,
    depends_on: Sequence[pulumi.Resource] = (),
) -> docker_build.Image:
    """Build and push an amd64 image with a registry-backed BuildKit cache."""
    return docker_build.Image(
        name,
        context=docker_build.BuildContextArgs(location=config.build_context),
        dockerfile=docker_build.DockerfileArgs(location=f"{config.build_context}/{config.dockerfile}"),
        build_args=config.build_args,
        cache_from=[
            docker_build.CacheFromArgs(
                registry=docker_build.CacheFromRegistryArgs(ref=config.cache_ref),
            )
        ],
        cache_to=[
            docker_build.CacheToArgs(
                registry=docker_build.CacheToRegistryArgs(
                    ref=config.cache_ref,
                    mode=docker_build.CacheMode.MAX,
                    compression=docker_build.CompressionType.ZSTD,
                    compression_level=CACHE_COMPRESSION_LEVEL,
                    oci_media_types=True,
                    image_manifest=True,
                ),
            )
        ],
        platforms=[docker_build.Platform.LINUX_AMD64],
        tags=config.tags,
        push=True,
        build_on_preview=False,
        opts=pulumi.ResourceOptions(
            parent=parent,
            provider=provider,
            depends_on=list(depends_on),
        ),
    )
