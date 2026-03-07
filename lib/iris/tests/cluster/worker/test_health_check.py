# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker health checks."""

import os
from pathlib import Path

import pytest
from iris.cluster.worker.env_probe import (
    MIN_DISK_FREE_FRACTION,
    HealthCheckResult,
    check_worker_health,
)


def test_health_check_passes_on_healthy_disk(tmp_path):
    """Health check passes on a directory with plenty of free space."""
    result = check_worker_health(disk_path=str(tmp_path))
    assert result.healthy
    assert result.error == ""


def test_health_check_tempfile_write_failure(tmp_path):
    """Health check fails if it cannot write a tempfile."""
    # Use a nonexistent directory to trigger write failure
    bad_path = str(tmp_path / "does_not_exist")
    result = check_worker_health(disk_path=bad_path)
    assert not result.healthy
    assert "tempfile write failed" in result.error


def test_health_check_result_dataclass():
    """HealthCheckResult is frozen and has expected fields."""
    ok = HealthCheckResult(healthy=True)
    assert ok.healthy
    assert ok.error == ""

    bad = HealthCheckResult(healthy=False, error="disk full")
    assert not bad.healthy
    assert bad.error == "disk full"


def test_min_disk_free_fraction_constant():
    """MIN_DISK_FREE_FRACTION is 5%."""
    assert MIN_DISK_FREE_FRACTION == 0.05
