# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from iris.cli import build as image_build
from iris.cli import cluster as cluster_cli
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


def _kubernetes_config() -> IrisClusterConfig:
    return IrisClusterConfig(
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


@pytest.mark.parametrize(
    ("dirty", "dirty_suffix"),
    [
        (False, ""),
        (True, "-dirty"),
    ],
)
def test_cluster_start_pins_latest_images_seen_by_provider(
    monkeypatch: pytest.MonkeyPatch,
    dirty: bool,
    dirty_suffix: str,
) -> None:
    config = _kubernetes_config()
    observed_images: list[tuple[str, str, str]] = []
    docker_commands: list[list[str]] = []

    class ControllerProvider:
        def start_controller(self, started_config, *, fresh=False):
            observed_images.append(
                (
                    started_config.controller.image,
                    started_config.defaults.worker.docker_image,
                    started_config.defaults.worker.default_task_image,
                )
            )
            return "https://controller.example"

    def run(command, **_kwargs):
        docker_commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cluster_cli, "get_git_provenance", lambda: _provenance(dirty))
    monkeypatch.setattr(cluster_cli, "provider_bundle", lambda _config: SimpleNamespace(controller=ControllerProvider()))
    monkeypatch.setattr(image_build.subprocess, "run", run)
    # This test covers image pinning; there is no controller to reach.
    monkeypatch.setattr(cluster_cli, "_wait_controller_reachable", lambda _bundle, _address: None)

    result = CliRunner().invoke(cluster_cli.cluster_start, obj={"config": config, "verbose": False})

    assert result.exit_code == 0, result.output
    assert observed_images == [
        (
            f"ghcr.io/marin-community/iris-controller:abc1234{dirty_suffix}",
            f"ghcr.io/marin-community/iris-worker:abc1234-amd64{dirty_suffix}",
            f"ghcr.io/marin-community/iris-task:abc1234{dirty_suffix}",
        )
    ]
    assert docker_commands


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


@pytest.mark.parametrize(
    ("platform", "expected_cache_ref"),
    [
        ("linux/amd64", "ghcr.io/marin-community/iris-cache:controller-amd64"),
        ("linux/amd64,linux/arm64", "ghcr.io/marin-community/iris-cache:controller"),
    ],
)
def test_build_image_push_accepts_qualified_tag_and_preserves_cache_scopes(
    monkeypatch, platform: str, expected_cache_ref: str
):
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(image_build.subprocess, "run", run)

    image_build.build_image(
        image_type="controller",
        tag="ghcr.io/marin-community/iris-controller:abc1234",
        push=True,
        context=None,
        platform=platform,
        provenance=_provenance(False),
    )

    command = commands[-1]
    tag_index = command.index("-t")
    assert command[tag_index + 1] == "ghcr.io/marin-community/iris-controller:abc1234"
    cache_index = command.index("--cache-to")
    assert f"ref={expected_cache_ref}" in command[cache_index + 1]


def test_push_to_ghcr_does_not_publish_latest(monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(image_build.subprocess, "run", lambda command, **_kwargs: commands.append(command))

    with pytest.raises(image_build.click.ClickException):
        image_build.push_to_ghcr("iris-task:latest")

    assert commands == []


def test_cluster_start_single_platform_task_uses_architecture_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _kubernetes_config()
    observed_task_images: list[str] = []

    class ControllerProvider:
        def start_controller(self, started_config, *, fresh=False):
            observed_task_images.append(started_config.defaults.worker.default_task_image)
            return "https://controller.example"

    monkeypatch.setattr(cluster_cli, "get_git_provenance", lambda: _provenance(False))
    monkeypatch.setattr(cluster_cli, "provider_bundle", lambda _config: SimpleNamespace(controller=ControllerProvider()))
    monkeypatch.setattr(
        image_build.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    # This test covers image pinning; there is no controller to reach.
    monkeypatch.setattr(cluster_cli, "_wait_controller_reachable", lambda _bundle, _address: None)

    result = CliRunner().invoke(
        cluster_cli.cluster_start,
        ["--image-platform", "linux/amd64"],
        obj={"config": config, "verbose": False},
    )

    assert result.exit_code == 0, result.output
    assert observed_task_images == ["ghcr.io/marin-community/iris-task:abc1234-amd64"]


def test_controller_restart_rejects_prebuilt_image_missing_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _kubernetes_config()
    manifest = {
        "digest": f"sha256:{'b' * 64}",
        "manifests": [{"platform": {"os": "linux", "architecture": "amd64"}}],
    }
    monkeypatch.setattr(
        cluster_cli.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(manifest), stderr=""),
    )
    monkeypatch.setattr(
        cluster_cli,
        "provider_bundle",
        lambda _config: SimpleNamespace(controller=SimpleNamespace(preflight_controller=lambda _config: None)),
    )

    result = CliRunner().invoke(
        cluster_cli.controller_restart,
        ["--skip-checkpoint", "--prebuilt-tag", "abc1234"],
        obj={"config": config, "controller_url": "http://127.0.0.1:1", "verbose": False},
    )

    assert result.exit_code == 1
    assert "missing platforms: linux/arm64" in result.output
