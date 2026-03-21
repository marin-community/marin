# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from levanter.infra import cli_helpers


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    return repo


def test_load_config_prefers_marin_over_levanter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    repo = _init_repo(tmp_path)
    (repo / ".marin.yaml").write_text("zone: us-east5-a\nproject: from-marin\n", encoding="utf-8")
    (repo / ".levanter.yaml").write_text("zone: us-central1-a\nproject: from-levanter\n", encoding="utf-8")

    monkeypatch.chdir(repo)
    with pytest.warns(UserWarning, match=r"Both \.marin\.yaml and \.levanter\.yaml found"):
        config = cli_helpers.load_config()

    assert config.zone == "us-east5-a"
    assert config.project == "from-marin"


def test_load_config_reads_marin_from_repo_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    repo = _init_repo(tmp_path)
    (repo / ".marin.yaml").write_text("zone: us-east1-d\n", encoding="utf-8")
    nested = repo / "experiments" / "rl"
    nested.mkdir(parents=True)

    monkeypatch.chdir(nested)
    config = cli_helpers.load_config()

    assert config.zone == "us-east1-d"


def test_load_config_falls_back_to_levanter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    repo = _init_repo(tmp_path)
    (repo / ".levanter.yaml").write_text("zone: europe-west4-a\n", encoding="utf-8")

    monkeypatch.chdir(repo)
    config = cli_helpers.load_config()

    assert config.zone == "europe-west4-a"


def test_load_config_warns_on_legacy_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    repo = _init_repo(tmp_path)
    (repo / ".config").write_text("zone: us-central1-a\n", encoding="utf-8")

    monkeypatch.chdir(repo)
    with pytest.warns(UserWarning, match=r"Using deprecated \.config file"):
        config = cli_helpers.load_config()

    assert config.zone == "us-central1-a"


def test_load_config_does_not_search_above_repo_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    (tmp_path / ".marin.yaml").write_text("zone: should-not-be-used\n", encoding="utf-8")
    repo = _init_repo(tmp_path)
    nested = repo / "subdir"
    nested.mkdir()

    monkeypatch.chdir(nested)
    config = cli_helpers.load_config()

    assert config.zone is None
