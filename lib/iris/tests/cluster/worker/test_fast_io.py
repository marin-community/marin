# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for tmpfs-based fast IO directory selection."""

import os
from pathlib import Path
from unittest.mock import patch

from iris.cluster.runtime.types import get_fast_io_dir


def test_fast_io_dir_uses_tmpfs_when_available(tmp_path: Path) -> None:
    """When tmpfs directory exists and has sufficient space, use it."""
    fake_tmpfs = tmp_path / "shm" / "iris"
    fake_tmpfs.mkdir(parents=True)

    with (
        patch.object(
            os,
            "statvfs",
            return_value=os.statvfs_result((4096, 4096, 1000000, 900000, 800000, 1000000, 900000, 800000, 0, 255)),
        ),
        patch("iris.cluster.runtime.types._TMPFS_DIR", fake_tmpfs),
    ):
        result = get_fast_io_dir(tmp_path / "cache")

    assert result == fake_tmpfs


def test_fast_io_dir_falls_back_when_tmpfs_too_small(tmp_path: Path) -> None:
    """When tmpfs has insufficient space, fall back to cache_dir."""
    fake_tmpfs = tmp_path / "shm" / "iris"
    fake_tmpfs.mkdir(parents=True)
    cache_dir = tmp_path / "cache"

    # f_bavail * f_frsize < _TMPFS_MIN_FREE_BYTES
    with (
        patch.object(
            os,
            "statvfs",
            return_value=os.statvfs_result((4096, 4096, 100, 50, 50, 100, 50, 50, 0, 255)),
        ),
        patch("iris.cluster.runtime.types._TMPFS_DIR", fake_tmpfs),
    ):
        result = get_fast_io_dir(cache_dir)

    assert result == cache_dir


def test_fast_io_dir_falls_back_when_tmpfs_missing(tmp_path: Path) -> None:
    """When tmpfs directory doesn't exist, fall back to cache_dir."""
    cache_dir = tmp_path / "cache"
    nonexistent = tmp_path / "nonexistent" / "iris"

    with patch("iris.cluster.runtime.types._TMPFS_DIR", nonexistent):
        result = get_fast_io_dir(cache_dir)

    assert result == cache_dir


def test_fast_io_dir_falls_back_on_oserror(tmp_path: Path) -> None:
    """When statvfs raises OSError, fall back to cache_dir."""
    fake_tmpfs = tmp_path / "shm" / "iris"
    fake_tmpfs.mkdir(parents=True)
    cache_dir = tmp_path / "cache"

    with (
        patch.object(os, "statvfs", side_effect=OSError("permission denied")),
        patch("iris.cluster.runtime.types._TMPFS_DIR", fake_tmpfs),
    ):
        result = get_fast_io_dir(cache_dir)

    assert result == cache_dir
