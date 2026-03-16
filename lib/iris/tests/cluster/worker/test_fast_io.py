# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for fast IO directory and prepare_workdir."""

from pathlib import Path

from iris.cluster.runtime.docker import DockerRuntime, _DEV_SHM


def test_fast_io_dir_always_uses_dev_shm(tmp_path: Path) -> None:
    """DockerRuntime always uses /dev/shm as the fast IO directory."""
    runtime = DockerRuntime(cache_dir=tmp_path / "cache")
    assert runtime._fast_io_dir == _DEV_SHM
