# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker health checks."""

import stat

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


def test_health_check_skips_nonexistent_dir(tmp_path):
    """Health check returns healthy when the directory does not exist (teardown)."""
    bad_path = str(tmp_path / "does_not_exist")
    result = check_worker_health(disk_path=bad_path)
    assert result.healthy


def test_health_check_tempfile_write_failure(tmp_path):
    """Health check fails if it cannot write a tempfile."""
    # Make the directory read-only to trigger write failure
    read_only = tmp_path / "read_only"
    read_only.mkdir()
    read_only.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        result = check_worker_health(disk_path=str(read_only))
        assert not result.healthy
        assert "tempfile write failed" in result.error
    finally:
        # Restore write permissions for cleanup
        read_only.chmod(stat.S_IRWXU)


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
