# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.version client revision date resolution."""

import re
import subprocess

import pytest

from iris import version as iris_version


@pytest.fixture(autouse=True)
def _reset_cache():
    iris_version._reset_cache_for_tests()
    yield
    iris_version._reset_cache_for_tests()


def test_client_revision_date_uses_build_info(monkeypatch):
    """When BUILD_DATE is set (wheel build), the resolver short-circuits to it."""
    monkeypatch.setattr(iris_version, "BUILD_DATE", "2026-01-15")

    def _fail(*args, **kwargs):
        raise AssertionError("git should not be invoked when BUILD_DATE is set")

    monkeypatch.setattr(iris_version, "_git_iris_date", _fail)

    assert iris_version.client_revision_date() == "2026-01-15"


def test_client_revision_date_falls_back_to_git(monkeypatch):
    """Editable install (BUILD_DATE empty) falls back to git log on lib/iris."""
    monkeypatch.setattr(iris_version, "BUILD_DATE", "")
    result = iris_version.client_revision_date()
    # Inside this repo we expect a real ISO date back from `git log`.
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", result), f"got {result!r}"


def test_client_revision_date_empty_when_git_fails(monkeypatch):
    """Subprocess failure (no git, no repo, etc.) yields an empty string."""
    monkeypatch.setattr(iris_version, "BUILD_DATE", "")

    def _raise(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "check_output", _raise)
    assert iris_version.client_revision_date() == ""


def test_client_revision_date_is_cached(monkeypatch):
    """Resolver computes once per process; subsequent calls reuse the result."""
    monkeypatch.setattr(iris_version, "BUILD_DATE", "")
    calls = {"n": 0}

    def _count(*args, **kwargs):
        calls["n"] += 1
        return "2026-02-02"

    monkeypatch.setattr(iris_version, "_git_iris_date", _count)
    iris_version.client_revision_date()
    iris_version.client_revision_date()
    iris_version.client_revision_date()
    assert calls["n"] == 1
