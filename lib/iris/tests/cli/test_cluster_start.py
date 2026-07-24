# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from click.testing import CliRunner
from iris.cli import cluster as cluster_cli


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        controller=SimpleNamespace(controller_kind=lambda: "gcp"),
        platform=SimpleNamespace(label_prefix=None),
        storage=SimpleNamespace(remote_state_dir=None),
    )


def test_cluster_start_passes_requested_image_build_settings(monkeypatch):
    config = _config()
    build = Mock(return_value={})
    controller = SimpleNamespace(start_controller=Mock(return_value="http://controller"))
    monkeypatch.setattr(cluster_cli, "get_git_sha", lambda: "tree-hash")
    monkeypatch.setattr(cluster_cli, "_pin_latest_images", Mock())
    monkeypatch.setattr(cluster_cli, "_build_cluster_images", build)
    monkeypatch.setattr(cluster_cli, "provider_bundle", lambda _: SimpleNamespace(controller=controller))

    result = CliRunner().invoke(
        cluster_cli.cluster_start,
        ["--image-platform", "linux/amd64", "--cargo-profile", "release"],
        obj={"config": config},
    )

    assert result.exit_code == 0, result.output
    build.assert_called_once_with(
        config,
        "tree-hash",
        verbose=False,
        platform="linux/amd64",
        cargo_profile="release",
    )


def test_cluster_start_smoke_passes_requested_image_build_settings(monkeypatch, tmp_path):
    config = _config()
    build = Mock(return_value={})
    controller = SimpleNamespace(
        stop_all=Mock(),
        start_controller=Mock(return_value="controller-address"),
        tunnel=Mock(return_value=nullcontext("http://controller")),
    )
    rpc_client = SimpleNamespace(
        list_workers=Mock(return_value=SimpleNamespace(workers=[SimpleNamespace(healthy=True)]))
    )
    stop_event = SimpleNamespace(wait=Mock())

    monkeypatch.setattr(cluster_cli, "get_git_sha", lambda: "tree-hash")
    monkeypatch.setattr(cluster_cli, "_pin_latest_images", Mock())
    monkeypatch.setattr(cluster_cli, "_build_cluster_images", build)
    monkeypatch.setattr(cluster_cli, "provider_bundle", lambda _: SimpleNamespace(controller=controller))
    monkeypatch.setattr(cluster_cli, "marin_temp_bucket", lambda **_: "gs://state")
    monkeypatch.setattr(cluster_cli, "clear_remote_state", Mock())
    monkeypatch.setattr(cluster_cli, "rpc_client_for_ctx", lambda *_args, **_kwargs: nullcontext(rpc_client))
    monkeypatch.setattr(cluster_cli.threading, "Event", lambda: stop_event)
    monkeypatch.setattr(cluster_cli.signal, "signal", Mock())

    result = CliRunner().invoke(
        cluster_cli.cluster_start_smoke,
        [
            "--label-prefix",
            "smoke",
            "--url-file",
            str(tmp_path / "url"),
            "--image-platform",
            "linux/amd64",
            "--cargo-profile",
            "fast",
        ],
        obj={"config": config},
    )

    assert result.exit_code == 0, result.output
    build.assert_called_once_with(
        config,
        "tree-hash",
        verbose=False,
        platform="linux/amd64",
        cargo_profile="fast",
    )
