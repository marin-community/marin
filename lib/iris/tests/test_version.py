# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.version client revision date resolution."""

import subprocess

import pytest
from iris import version as iris_version


@pytest.fixture(autouse=True)
def _reset_cache():
    iris_version._reset_cache_for_tests()
    yield
    iris_version._reset_cache_for_tests()


def test_client_revision_date_uses_build_info(monkeypatch):
    """A wheel build exposes its stamped revision date."""
    monkeypatch.setattr(iris_version, "BUILD_DATE", "2026-01-15")

    assert iris_version.client_revision_date() == "2026-01-15"


def test_client_revision_date_falls_back_to_git(monkeypatch):
    """Editable install (BUILD_DATE empty) falls back to git log on lib/iris."""
    monkeypatch.setattr(iris_version, "BUILD_DATE", "")
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "2026-02-02\n")

    assert iris_version.client_revision_date() == "2026-02-02"


def test_client_revision_date_empty_when_git_fails(monkeypatch):
    """Subprocess failure (no git, no repo, etc.) yields an empty string."""
    monkeypatch.setattr(iris_version, "BUILD_DATE", "")

    def _raise(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "check_output", _raise)
    assert iris_version.client_revision_date() == ""
