# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from levanter.infra.cli_helpers import CliConfig, load_config


def test_load_config_ignores_xdg_directory_and_keeps_legacy_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))

    legacy_path = tmp_path / ".config"
    legacy_path.mkdir()
    assert load_config() == CliConfig()

    legacy_path.rmdir()
    legacy_path.write_text("project: legacy\n")
    assert load_config().project == "legacy"
