#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import types

import pytest

from marin.run.ray_run import (
    _temporarily_ignore_gitignore,
    load_env_vars_from_cli_config,
    maybe_include_levanter_config,
    tpus_per_node,
)


def test_tpus_per_node():
    assert tpus_per_node("v4-8") == 4
    assert tpus_per_node("v5p-8") == 4
    assert tpus_per_node("v5e-4") == 4
    assert tpus_per_node("v5e-2") == 2
    with pytest.raises(ValueError):
        tpus_per_node("v5e-16")


class _DummyConfig:
    def __init__(self):
        self.env = {"FOO": "bar", "EMPTY": None}

    def env_for_accel(self, accel_type: str):
        env = self.env.copy()
        env["ACCEL"] = accel_type
        return env


def _install_fake_levanter(monkeypatch: pytest.MonkeyPatch, config: _DummyConfig) -> None:
    cli_helpers_module = types.ModuleType("levanter.infra.cli_helpers")
    cli_helpers_module.load_config = lambda: config

    infra_module = types.ModuleType("levanter.infra")
    infra_module.cli_helpers = cli_helpers_module

    levanter_module = types.ModuleType("levanter")
    levanter_module.infra = infra_module

    monkeypatch.setitem(sys.modules, "levanter", levanter_module)
    monkeypatch.setitem(sys.modules, "levanter.infra", infra_module)
    monkeypatch.setitem(sys.modules, "levanter.infra.cli_helpers", cli_helpers_module)


def test_load_env_vars_from_cli_config_without_tpu(monkeypatch: pytest.MonkeyPatch):
    config = _DummyConfig()
    _install_fake_levanter(monkeypatch, config)

    env = load_env_vars_from_cli_config()

    assert env == {"FOO": "bar"}


def test_load_env_vars_from_cli_config_with_tpu(monkeypatch: pytest.MonkeyPatch):
    config = _DummyConfig()
    _install_fake_levanter(monkeypatch, config)

    env = load_env_vars_from_cli_config("v5e-4")

    assert env == {"FOO": "bar", "ACCEL": "v5e-4"}


def test_maybe_include_levanter_config(tmp_path):
    runtime_env = {"excludes": [".git"]}
    (tmp_path / ".levanter.yaml").write_text("env: {}\n")
    (tmp_path / ".gitignore").write_text("# comment\n.levanter.yaml\nfoo/\n")

    updated, ignore = maybe_include_levanter_config(runtime_env, os.fspath(tmp_path))

    assert ignore is True
    assert updated["excludes"] == [".git", ".levanter.yaml", "foo/", "!.levanter.yaml"]

    updated_again, ignore_again = maybe_include_levanter_config(updated, os.fspath(tmp_path))
    assert updated_again["excludes"] == updated["excludes"]
    assert ignore_again is True


def test_maybe_include_levanter_config_without_gitignore(tmp_path):
    runtime_env = {"excludes": [".git"]}
    (tmp_path / ".levanter.yaml").write_text("env: {}\n")

    updated, ignore = maybe_include_levanter_config(runtime_env, os.fspath(tmp_path))

    assert ignore is False
    assert updated["excludes"] == [".git"]


def test_maybe_include_respects_nested_gitignore(tmp_path):
    runtime_env = {"excludes": []}
    (tmp_path / ".levanter.yaml").write_text("env: {}\n")

    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (nested_dir / ".gitignore").write_text("bar/\n!.keep\n/foo\n/baz/\n")

    updated, ignore = maybe_include_levanter_config(runtime_env, os.fspath(tmp_path))

    assert ignore is True
    assert "nested/**/bar/" in updated["excludes"]
    assert "!nested/**/.keep" in updated["excludes"]
    assert "nested/foo" in updated["excludes"]
    assert "nested/baz/" in updated["excludes"]
    assert "!.levanter.yaml" in updated["excludes"]


def test_temporarily_ignore_gitignore(monkeypatch: pytest.MonkeyPatch):
    env_var = "RAY_RUNTIME_ENV_IGNORE_GITIGNORE"
    monkeypatch.delenv(env_var, raising=False)

    with _temporarily_ignore_gitignore(True):
        assert os.environ[env_var] == "1"

    assert env_var not in os.environ

    monkeypatch.setenv(env_var, "0")
    with _temporarily_ignore_gitignore(True):
        assert os.environ[env_var] == "1"
    assert os.environ[env_var] == "0"

    with _temporarily_ignore_gitignore(False):
        assert os.environ[env_var] == "0"


def test_maybe_include_handles_legacy_config(tmp_path):
    runtime_env = {"excludes": []}
    (tmp_path / ".config").write_text("env: {}\n")
    (tmp_path / ".gitignore").write_text(".config\n")

    updated, ignore = maybe_include_levanter_config(runtime_env, os.fspath(tmp_path))

    assert ignore is True
    assert ".config" in updated["excludes"]
    assert "!.config" in updated["excludes"]


def test_maybe_include_preserves_anchored_patterns(tmp_path):
    runtime_env = {"excludes": []}
    (tmp_path / ".levanter.yaml").write_text("env: {}\n")

    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (nested_dir / ".gitignore").write_text("/only-here\nunanchored\n")

    updated, ignore = maybe_include_levanter_config(runtime_env, os.fspath(tmp_path))

    assert ignore is True
    assert "nested/only-here" in updated["excludes"]
    assert "nested/**/unanchored" in updated["excludes"]
