# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for config-driven image build/push paths (GHCR only)."""

from unittest.mock import patch

from iris.cli.cluster import _build_cluster_images
from iris.cluster.config import (
    ControllerVmConfig,
    DefaultsConfig,
    GcpControllerConfig,
    GcpSliceConfig,
    IrisClusterConfig,
    ScaleGroupConfig,
    SliceConfig,
    WorkerConfig,
)


def test_build_cluster_images_pushes_worker_controller_and_task_to_ghcr() -> None:
    """Cluster image build should build and push all three image types to GHCR."""
    config = IrisClusterConfig(
        defaults=DefaultsConfig(
            worker=WorkerConfig(
                docker_image="ghcr.io/marin-community/iris-worker:v1",
                default_task_image="ghcr.io/marin-community/iris-task:v1",
            )
        ),
        controller=ControllerVmConfig(
            image="ghcr.io/marin-community/iris-controller:v1",
            gcp=GcpControllerConfig(zone="us-west1-b"),
        ),
        scale_groups={
            "east": ScaleGroupConfig(slice_template=SliceConfig(gcp=GcpSliceConfig(zone="us-east1-d"))),
            "eu": ScaleGroupConfig(slice_template=SliceConfig(gcp=GcpSliceConfig(zone="europe-west4-b"))),
        },
    )

    with patch("iris.cli.cluster.build_image") as build_image:

        built = _build_cluster_images(
            config,
            git_sha="abc",
            platform="linux/amd64",
            cargo_profile="fast",
        )

    assert built == {
        "worker": "ghcr.io/marin-community/iris-worker:v1",
        "controller": "ghcr.io/marin-community/iris-controller:v1",
        "task": "ghcr.io/marin-community/iris-task:v1",
    }
    assert build_image.call_count == 3
    image_types = [call.kwargs["image_type"] for call in build_image.call_args_list]
    assert sorted(image_types) == ["controller", "task", "worker"]
    for call in build_image.call_args_list:
        assert call.kwargs["tag"].endswith(":v1")
        assert call.kwargs["git_sha"] == "abc"
        assert call.kwargs["platform"] == "linux/amd64"
        assert call.kwargs["cargo_profile"] == "fast"
