# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the load-test harness smoke tests.

The `loadtest` marker is registered in `pyproject.toml`; default test runs
exclude it via `addopts`. These fixtures are only used when a caller opts in
with `-m loadtest`. All production load-test code now lives under
``iris.loadtest``; this conftest only provides fixtures for the smoke test.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

from iris.loadtest.configs import DEFAULT_SNAPSHOT_PATH


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect load-test tests unless `-m loadtest` is explicitly requested.

    The load-test harness is heavy (opens a 1.5 GB sqlite, runs ANALYZE) and
    must not be collected by the default test run. pyproject `addopts`
    already excludes the marker, but a caller passing `-m` on the CLI
    overrides that. This hook is the belt-and-braces guard: any item inside
    `tests/loadtest/` is removed unless the active marker expression mentions
    ``loadtest`` explicitly.
    """
    marker_expr = config.getoption("-m") or ""
    if "loadtest" in marker_expr:
        return

    kept: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        if "loadtest" in item.keywords and item.fspath.strpath.find("/tests/loadtest/") >= 0:
            deselected.append(item)
        else:
            kept.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


@pytest.fixture
def snapshot_path() -> Path:
    """Path to the captured controller sqlite snapshot.

    Skips cleanly when the snapshot is not present so the harness tests do not
    fail on machines that have not pulled the 1.5 GB checkpoint.
    """
    if not DEFAULT_SNAPSHOT_PATH.exists():
        pytest.skip(f"Snapshot {DEFAULT_SNAPSHOT_PATH} not present; run the loadtest manually after copying it.")
    return DEFAULT_SNAPSHOT_PATH


@pytest.fixture
def snapshot_copy_dir(tmp_path: Path, snapshot_path: Path) -> Path:
    """Copy the snapshot into a per-test temp dir and yield the directory.

    We copy rather than mount the original so migrations, WAL files, and any
    accidental mutations never touch the source of truth.
    """
    dst_dir = tmp_path / "db"
    dst_dir.mkdir()
    shutil.copy2(snapshot_path, dst_dir / "controller.sqlite3")
    return dst_dir


@pytest.fixture
def harness_logger() -> logging.Logger:
    return logging.getLogger("iris.loadtest")
