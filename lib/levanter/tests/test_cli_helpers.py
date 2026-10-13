# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from levanter.infra.cli_helpers import load_config


def test_load_config_ignores_xdg_config_directory(tmp_path, monkeypatch):
    (tmp_path / ".config").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    load_config()


def test_load_config_still_reads_legacy_config_file(tmp_path, monkeypatch):
    (tmp_path / ".config").write_text("zone: us-central2-b\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    config = load_config()
    assert config.zone == "us-central2-b"
