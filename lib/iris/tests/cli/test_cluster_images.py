# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from iris.cli import build as image_build
from iris.cli import cluster as cluster_cli
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
def test_cluster_start_pins_latest_images_seen_by_provider(
    kubernetes_cluster_config,
    run_cluster_start,
    stub_controller_health,
    stub_controller_provider,
    dirty: bool,
    dirty_suffix: str,
) -> None:
    provider = stub_controller_provider(stub_controller_health().url)

    result = run_cluster_start(provider, kubernetes_cluster_config, dirty=dirty)

    assert result.exit_code == 0, result.output
    started = provider.started[-1]
    assert (
        started.controller.image,
        started.defaults.worker.docker_image,
        started.defaults.worker.default_task_image,
    ) == (
        f"ghcr.io/marin-community/iris-controller:abc1234{dirty_suffix}",
        f"ghcr.io/marin-community/iris-worker:abc1234-amd64{dirty_suffix}",
        f"ghcr.io/marin-community/iris-task:abc1234{dirty_suffix}",
    )
    assert provider.docker_commands


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


def test_cluster_start_single_platform_task_uses_architecture_suffix(
    kubernetes_cluster_config, run_cluster_start, stub_controller_health, stub_controller_provider
) -> None:
    provider = stub_controller_provider(stub_controller_health().url)

    result = run_cluster_start(provider, kubernetes_cluster_config, args=["--image-platform", "linux/amd64"])

    assert result.exit_code == 0, result.output
    assert provider.started[-1].defaults.worker.default_task_image == "ghcr.io/marin-community/iris-task:abc1234-amd64"


def test_controller_restart_rejects_prebuilt_image_missing_platform(
    monkeypatch: pytest.MonkeyPatch, kubernetes_cluster_config
) -> None:
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
        obj={"config": kubernetes_cluster_config, "controller_url": "http://127.0.0.1:1", "verbose": False},
    )

    assert result.exit_code == 1
    assert "missing platforms: linux/arm64" in result.output
