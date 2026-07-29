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


def test_pin_latest_images_pins_kubernetes_task_pods_to_deploy_sha():
    config = IrisClusterConfig(
        controller=ControllerVmConfig(image="ghcr.io/marin-community/iris-controller:latest"),
        defaults=DefaultsConfig(
            worker=WorkerConfig(
                docker_image="ghcr.io/marin-community/iris-worker:latest",
                default_task_image="ghcr.io/marin-community/iris-task:latest",
                runtime="kubernetes",
            )
        ),
        kubernetes_provider=KubernetesProviderConfig(
            default_image="ghcr.io/marin-community/iris-task:latest",
        ),
    )

    _pin_latest_images(config, "abc1234")

    assert config.controller.image == "ghcr.io/marin-community/iris-controller:abc1234"
    assert config.defaults.worker.docker_image == "ghcr.io/marin-community/iris-worker:abc1234-amd64"
    assert config.defaults.worker.default_task_image == "ghcr.io/marin-community/iris-task:abc1234"
    assert config.kubernetes_provider.default_image == "ghcr.io/marin-community/iris-task:abc1234"


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
            git_sha="abc1234",
        )

    assert commands == []


def test_push_to_ghcr_does_not_publish_latest(monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(image_build.subprocess, "run", lambda command, **_kwargs: commands.append(command))

    with pytest.raises(image_build.click.ClickException):
        image_build.push_to_ghcr("iris-task:latest")

    assert commands == []


def test_build_image_push_publishes_only_requested_sha_tag(monkeypatch):
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(image_build.subprocess, "run", run)

    image_build.build_image(
        image_type="task",
        tag="iris-task:abc1234-arm64",
        push=True,
        context=None,
        platform="linux/arm64",
        git_sha="abc1234",
    )

    pushed_tags = [commands[-1][index + 1] for index, arg in enumerate(commands[-1]) if arg == "-t"]
    assert pushed_tags == ["ghcr.io/marin-community/iris-task:abc1234-arm64"]


def test_pin_latest_images_single_platform_task_uses_architecture_suffix():
    config = IrisClusterConfig(
        controller=ControllerVmConfig(image="ghcr.io/marin-community/iris-controller:latest"),
        defaults=DefaultsConfig(
            worker=WorkerConfig(
                default_task_image="ghcr.io/marin-community/iris-task:latest",
                runtime="kubernetes",
            )
        ),
        kubernetes_provider=KubernetesProviderConfig(
            default_image="ghcr.io/marin-community/iris-task:latest",
        ),
    )

    _pin_latest_images(config, "abc1234", "linux/amd64")

    assert config.controller.image == "ghcr.io/marin-community/iris-controller:abc1234"
    assert config.defaults.worker.default_task_image == "ghcr.io/marin-community/iris-task:abc1234-amd64"
    assert config.kubernetes_provider.default_image == "ghcr.io/marin-community/iris-task:abc1234-amd64"


def test_pin_latest_images_rejects_different_kubernetes_task_images():
    config = IrisClusterConfig(
        controller=ControllerVmConfig(image="ghcr.io/marin-community/iris-controller:latest"),
        defaults=DefaultsConfig(
            worker=WorkerConfig(
                default_task_image="ghcr.io/marin-community/iris-task:latest",
                runtime="kubernetes",
            )
        ),
        kubernetes_provider=KubernetesProviderConfig(
            default_image="ghcr.io/example/other-task:latest",
        ),
    )

    with pytest.raises(image_build.click.ClickException):
        _pin_latest_images(config, "abc1234")
