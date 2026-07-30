# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from iris.cli import build as image_build
from iris.cli.cluster import _pin_latest_images
from iris.cluster.config import (
    ControllerVmConfig,
    DefaultsConfig,
    IrisClusterConfig,
    KubernetesProviderConfig,
    WorkerConfig,
)
from rigging.provenance import Provenance


def _provenance(dirty: bool) -> Provenance:
    return Provenance(
        tree_hash="abc1234",
        base_commit="abc1234",
        dirty=dirty,
        branch="feature",
        built_by="tester",
    )


@pytest.mark.parametrize(
    ("dirty", "dirty_suffix"),
    [
        (False, ""),
        (True, "-dirty"),
    ],
)
def test_pin_latest_images_marks_only_dirty_worktrees(dirty: bool, dirty_suffix: str):
    config = IrisClusterConfig(
        controller=ControllerVmConfig(image="ghcr.io/marin-community/iris-controller:latest"),
        defaults=DefaultsConfig(
            worker=WorkerConfig(
                docker_image="ghcr.io/marin-community/iris-worker:latest",
                default_task_image="ghcr.io/marin-community/iris-task:latest",
                runtime="kubernetes",
            )
        ),
        kubernetes_provider=KubernetesProviderConfig(),
    )

    _pin_latest_images(config, _provenance(dirty))

    assert config.controller.image == f"ghcr.io/marin-community/iris-controller:abc1234{dirty_suffix}"
    assert config.defaults.worker.docker_image == f"ghcr.io/marin-community/iris-worker:abc1234-amd64{dirty_suffix}"
    assert config.defaults.worker.default_task_image == f"ghcr.io/marin-community/iris-task:abc1234{dirty_suffix}"


def test_build_image_push_does_not_publish_latest(monkeypatch):
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(image_build.subprocess, "run", run)

    with pytest.raises(image_build.click.ClickException):
        image_build.build_image(
            image_type="task",
            tag="iris-task:latest",
            push=True,
            context=None,
            platform="linux/arm64",
            provenance=_provenance(False),
        )

    assert commands == []


def test_push_to_ghcr_does_not_publish_latest(monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(image_build.subprocess, "run", lambda command, **_kwargs: commands.append(command))

    with pytest.raises(image_build.click.ClickException):
        image_build.push_to_ghcr("iris-task:latest")

    assert commands == []


def test_pin_latest_images_single_platform_task_uses_architecture_suffix():
    config = IrisClusterConfig(
        controller=ControllerVmConfig(image="ghcr.io/marin-community/iris-controller:latest"),
        defaults=DefaultsConfig(
            worker=WorkerConfig(
                default_task_image="ghcr.io/marin-community/iris-task:latest",
                runtime="kubernetes",
            )
        ),
        kubernetes_provider=KubernetesProviderConfig(),
    )

    _pin_latest_images(config, _provenance(False), "linux/amd64")

    assert config.controller.image == "ghcr.io/marin-community/iris-controller:abc1234"
    assert config.defaults.worker.default_task_image == "ghcr.io/marin-community/iris-task:abc1234-amd64"
