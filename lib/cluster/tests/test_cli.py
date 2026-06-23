# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the marin-cluster CLI surface (config group + delegation wiring)."""

from click.testing import CliRunner
from marin_cluster import config as cluster_config
from marin_cluster.cli import main


def test_config_show_resolves_repo_marin_manifest(monkeypatch):
    # The umbrella resolves the committed config/marin.yaml from the workspace root.
    monkeypatch.delenv("MARIN_CLUSTER", raising=False)
    result = CliRunner().invoke(main, ["--cluster", "marin", "config", "show"])
    assert result.exit_code == 0, result.output
    assert "cluster:       marin" in result.output
    assert "project=hai-gcp-models" in result.output
    assert "domain=iris-marin.oa.dev" in result.output


def test_config_use_then_list_marks_pinned(tmp_path, monkeypatch):
    monkeypatch.setattr(cluster_config, "_CURRENT_CLUSTER_POINTER", tmp_path / "cluster")
    monkeypatch.delenv("MARIN_CLUSTER", raising=False)
    runner = CliRunner()

    used = runner.invoke(main, ["config", "use", "marin"])
    assert used.exit_code == 0, used.output
    assert cluster_config.current_cluster() == "marin"

    listed = runner.invoke(main, ["config", "list"])
    assert listed.exit_code == 0, listed.output
    assert "marin *" in listed.output


def test_delegated_client_verbs_are_mounted():
    # iris/finelog are installed in the dev workspace, so their verbs delegate.
    assert {"job", "cluster", "logs"} <= set(main.commands)
