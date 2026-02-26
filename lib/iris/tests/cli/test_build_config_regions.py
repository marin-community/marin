# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for config-driven image build/push paths (GHCR only)."""

from unittest.mock import patch

from iris.rpc import config_pb2


def test_build_cluster_images_pushes_worker_controller_and_task_to_ghcr() -> None:
    """Cluster image build should build and push all three image types to GHCR."""
    config = config_pb2.IrisClusterConfig()
    config.defaults.worker.docker_image = "ghcr.io/marin-community/iris-worker:v1"
    config.controller.image = "ghcr.io/marin-community/iris-controller:v1"
    config.defaults.worker.default_task_image = "ghcr.io/marin-community/iris-task:v1"
    config.controller.gcp.zone = "us-west1-b"
    config.scale_groups["east"].slice_template.gcp.zone = "us-east1-d"
    config.scale_groups["eu"].slice_template.gcp.zone = "europe-west4-b"

    with (
        patch("iris.cli.cluster._build_and_push_for_tag") as build_and_push_for_tag,
        patch("iris.cli.cluster._build_and_push_task_image") as build_and_push_task,
    ):
        from iris.cli.cluster import _build_cluster_images

        built = _build_cluster_images(config)

    assert built == {
        "worker": "ghcr.io/marin-community/iris-worker:v1",
        "controller": "ghcr.io/marin-community/iris-controller:v1",
        "task": "ghcr.io/marin-community/iris-task:v1",
    }
    assert build_and_push_for_tag.call_count == 2
    for call in build_and_push_for_tag.call_args_list:
        assert call.args[0].startswith("ghcr.io/")

    build_and_push_task.assert_called_once_with(
        "ghcr.io/marin-community/iris-task:v1",
        verbose=False,
    )
