# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for config-driven multi-region image push/build paths."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from iris.cli import iris
from iris.cluster.config import load_config
from iris.cluster.platform.bootstrap import collect_all_regions
from iris.rpc import config_pb2


def test_build_all_uses_config_regions_for_gcp_registry() -> None:
    """`iris build all --registry gcp` should derive regions/project from --config."""
    runner = CliRunner()
    config_path = Path(__file__).resolve().parents[2] / "examples" / "marin.yaml"
    config = load_config(config_path)
    expected_regions = tuple(sorted(collect_all_regions(config)))
    expected_project = config.platform.gcp.project_id

    captured_calls: list[tuple] = []

    def _capture_build_image(*args, **kwargs):
        captured_calls.append(args)

    with patch("iris.cli.build.get_git_sha", return_value="abc123"), patch(
        "iris.cli.build.build_image", side_effect=_capture_build_image
    ):
        result = runner.invoke(
            iris,
            ["--config", str(config_path), "build", "all", "--registry", "gcp"],
        )

    assert result.exit_code == 0, result.output
    assert len(captured_calls) == 3
    for call in captured_calls:
        assert call[6] == "gcp"
        assert call[8] == expected_regions
        assert call[9] == expected_project


def test_build_cluster_images_pushes_worker_controller_and_task_to_extra_regions() -> None:
    """Cluster image build should fan out all AR image types to discovered regions."""
    config = config_pb2.IrisClusterConfig()
    config.defaults.bootstrap.docker_image = "us-central2-docker.pkg.dev/test-project/marin/iris-worker:v1"
    config.controller.image = "us-central2-docker.pkg.dev/test-project/marin/iris-controller:v1"
    config.defaults.default_task_image = "us-central2-docker.pkg.dev/test-project/marin/iris-task:v1"
    config.controller.gcp.zone = "us-west1-b"
    config.scale_groups["east"].slice_template.gcp.zone = "us-east1-d"
    config.scale_groups["eu"].slice_template.gcp.zone = "europe-west4-b"

    with patch("iris.cli.cluster._build_and_push_for_tag") as build_and_push_for_tag, patch(
        "iris.cli.cluster._build_and_push_task_image"
    ) as build_and_push_task, patch("iris.cli.cluster._push_image_to_extra_regions") as push_to_extra:
        from iris.cli.cluster import _build_cluster_images

        built = _build_cluster_images(config)

    assert built == {
        "worker": "us-central2-docker.pkg.dev/test-project/marin/iris-worker:v1",
        "controller": "us-central2-docker.pkg.dev/test-project/marin/iris-controller:v1",
        "task": "us-central2-docker.pkg.dev/test-project/marin/iris-task:v1",
    }
    assert build_and_push_for_tag.call_count == 2
    build_and_push_task.assert_called_once_with("us-central2-docker.pkg.dev/test-project/marin/iris-task:v1")

    assert push_to_extra.call_count == 3
    expected_extra_regions = {"us-east1", "europe-west4", "us-west1"}
    image_types = set()
    for call in push_to_extra.call_args_list:
        params = call.args[0]
        image_types.add(params.image_type)
        assert call.args[1] == expected_extra_regions

    assert image_types == {"worker", "controller", "task"}
