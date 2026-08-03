# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from iris.cli import build as image_build
from iris.cli import cluster as cluster_cli
from iris.cli.cluster import _pin_latest_images
from iris.cluster.config import (
    ControllerVmConfig,
    CoreweaveControllerConfig,
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


def test_controller_restart_with_prebuilt_tag_skips_build_and_deploys_digests(monkeypatch):
    config = IrisClusterConfig(
        controller=ControllerVmConfig(
            image="ghcr.io/marin-community/iris-controller:latest",
            coreweave=CoreweaveControllerConfig(),
        ),
        defaults=DefaultsConfig(
            worker=WorkerConfig(
                default_task_image="ghcr.io/marin-community/iris-task:latest",
                runtime="kubernetes",
            )
        ),
        kubernetes_provider=KubernetesProviderConfig(),
    )
    deployed: dict[str, str] = {}

    class Controller:
        def preflight_controller(self, _config):
            pass

        def restart_controller(self, deploy_config):
            deployed["controller"] = deploy_config.controller.image
            deployed["task"] = deploy_config.defaults.worker.default_task_image
            return "iris-controller.example:10000"

    monkeypatch.setattr(cluster_cli, "provider_bundle", lambda _config: SimpleNamespace(controller=Controller()))
    monkeypatch.setattr(cluster_cli, "require_controller_url", lambda _ctx: "http://controller.example")

    def inspect_manifest(command, **_kwargs):
        image = command[4]
        digest_character = "a" if "iris-controller:" in image else "b"
        manifest = {
            "digest": f"sha256:{digest_character * 64}",
            "manifests": [
                {"platform": {"os": "linux", "architecture": "amd64"}},
                {"platform": {"os": "linux", "architecture": "arm64"}},
            ],
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(manifest), stderr="")

    monkeypatch.setattr(cluster_cli.subprocess, "run", inspect_manifest)

    result = CliRunner().invoke(
        cluster_cli.controller_restart,
        ["--skip-checkpoint", "--prebuilt-tag", "abc1234"],
        obj={"config": config},
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert deployed == {
        "controller": f"ghcr.io/marin-community/iris-controller@sha256:{'a' * 64}",
        "task": f"ghcr.io/marin-community/iris-task@sha256:{'b' * 64}",
    }


def test_resolve_prebuilt_image_rejects_missing_platform(monkeypatch):
    manifest = {
        "digest": f"sha256:{'b' * 64}",
        "manifests": [{"platform": {"os": "linux", "architecture": "amd64"}}],
    }
    monkeypatch.setattr(
        cluster_cli.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(manifest), stderr=""),
    )

    with pytest.raises(cluster_cli.click.ClickException):
        cluster_cli._resolve_prebuilt_image("ghcr.io/marin-community/iris-controller:abc1234")
