# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster import dashboard_common
from iris.rpc import config_pb2


def test_dashboard_title_from_config_prefers_name() -> None:
    config = config_pb2.IrisClusterConfig()
    config.name = "marin"
    config.platform.label_prefix = "marin-dev"

    assert dashboard_common.dashboard_title_from_config(config) == "marin"


def test_dashboard_title_from_config_falls_back_to_label_prefix() -> None:
    config = config_pb2.IrisClusterConfig()
    config.platform.label_prefix = "cw"

    assert dashboard_common.dashboard_title_from_config(config) == "cw"


def test_html_shell_uses_deployed_dashboard_title(monkeypatch, tmp_path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "controller.html").write_text(
        "<!doctype html><html><head><title>Iris Dashboard</title></head><body></body></html>"
    )

    monkeypatch.setattr(dashboard_common, "VUE_DIST_DIR", dist)
    monkeypatch.setattr(dashboard_common, "DOCKER_VUE_DIST_DIR", tmp_path / "missing")
    monkeypatch.setenv(dashboard_common.DASHBOARD_TITLE_ENV_VAR, "marin-dev")

    html = dashboard_common.html_shell("Iris Controller", "controller")

    assert "<title>marin-dev | Iris</title>" in html


def test_html_shell_escapes_deployed_dashboard_title(monkeypatch, tmp_path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "controller.html").write_text(
        "<!doctype html><html><head><title>Iris Dashboard</title></head><body></body></html>"
    )

    monkeypatch.setattr(dashboard_common, "VUE_DIST_DIR", dist)
    monkeypatch.setattr(dashboard_common, "DOCKER_VUE_DIST_DIR", tmp_path / "missing")
    monkeypatch.setenv(dashboard_common.DASHBOARD_TITLE_ENV_VAR, "<cw>")

    html = dashboard_common.html_shell("Iris Controller", "controller")

    assert "<title>&lt;cw&gt; | Iris</title>" in html
