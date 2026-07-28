# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""External runtime pins must come from their independent uv lockfiles."""

import tomllib
from pathlib import Path

from marin.external_dependencies import EXTERNAL_DEPENDENCIES, HARBOR

ROOT = Path(__file__).parents[2]


def _locked_git_source(config_name: str, distribution: str) -> tuple[str, str]:
    lock = tomllib.loads((ROOT / "config" / "external" / config_name / "uv.lock").read_text())
    package = next(package for package in lock["package"] if package["name"] == distribution)
    source = package["source"]["git"]
    repository, _, commit = source.partition("#")
    repository, _, _ = repository.partition("?")
    return repository, commit


def _locked_version(config_name: str, distribution: str) -> str:
    lock = tomllib.loads((ROOT / "config" / "external" / config_name / "uv.lock").read_text())
    package = next(package for package in lock["package"] if package["name"] == distribution)
    return package["version"]


def test_external_dependency_pins_match_lockfiles():
    for dependency in EXTERNAL_DEPENDENCIES:
        repository, commit = _locked_git_source(dependency.config_name, dependency.distribution)
        assert dependency.repository == repository
        assert dependency.commit == commit
        for requirement in dependency.runtime_requirements:
            distribution, separator, version = requirement.partition("==")
            assert separator
            assert version == _locked_version(dependency.config_name, distribution)


def test_harbor_workspace_source_matches_external_pin():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    source = pyproject["tool"]["uv"]["sources"]["harbor"]
    assert source == {"git": HARBOR.repository, "rev": HARBOR.commit}
