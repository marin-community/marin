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

"""Tests for Docker container statistics collection."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock

import docker.errors
import pytest

from fluster.cluster.worker.stats import (
    ContainerStats,
    collect_container_stats,
    collect_workdir_size_mb,
)


def check_docker_available():
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.asyncio
async def test_collect_workdir_size_mb_with_temp_directory(tmp_path):
    """Test workdir size calculation with a temporary directory."""
    # Create some files in temp directory
    (tmp_path / "file1.txt").write_text("x" * 1024 * 100)  # 100 KB
    (tmp_path / "file2.txt").write_text("y" * 1024 * 100)  # 100 KB

    size_mb = await collect_workdir_size_mb(tmp_path)

    # Size should be at least 1 MB (200 KB rounded up)
    assert size_mb >= 1


@pytest.mark.asyncio
async def test_collect_workdir_size_mb_nonexistent_directory():
    """Test workdir size returns 0 for non-existent directory."""
    nonexistent = Path("/nonexistent/path/does/not/exist")

    size_mb = await collect_workdir_size_mb(nonexistent)

    assert size_mb == 0


@pytest.mark.asyncio
async def test_collect_container_stats_invalid_container():
    """Test that collect_container_stats returns available=False for invalid container ID."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    import docker

    docker_client = docker.from_env()
    invalid_container_id = "nonexistent_container_12345"

    stats = await collect_container_stats(docker_client, invalid_container_id)

    assert isinstance(stats, ContainerStats)
    assert stats.available is False
    assert stats.memory_mb == 0
    assert stats.cpu_percent == 0
    assert stats.process_count == 0


@pytest.mark.asyncio
async def test_collect_container_stats_with_mock():
    """Test collect_container_stats parsing with mocked Docker client."""
    # Create mock Docker client
    mock_client = MagicMock()
    mock_container = Mock()

    # Mock stats response with realistic Docker stats format
    mock_stats = {
        "memory_stats": {
            "usage": 100 * 1024 * 1024,  # 100 MB in bytes
        },
        "cpu_stats": {
            "cpu_usage": {"total_usage": 2000000000},
            "system_cpu_usage": 10000000000,
            "online_cpus": 4,
        },
        "precpu_stats": {
            "cpu_usage": {"total_usage": 1000000000},
            "system_cpu_usage": 9000000000,
        },
        "pids_stats": {
            "current": 5,
        },
    }

    mock_container.stats.return_value = mock_stats
    mock_client.containers.get.return_value = mock_container

    stats = await collect_container_stats(mock_client, "test_container")

    assert stats.available is True
    assert stats.memory_mb == 100
    assert stats.cpu_percent == 400  # (1000000000 / 1000000000) * 4 * 100
    assert stats.process_count == 5


@pytest.mark.asyncio
async def test_collect_container_stats_not_found_exception():
    """Test that NotFound exception returns available=False."""
    mock_client = MagicMock()
    mock_client.containers.get.side_effect = docker.errors.NotFound("Container not found")

    stats = await collect_container_stats(mock_client, "missing_container")

    assert stats.available is False


@pytest.mark.asyncio
async def test_collect_container_stats_api_error_exception():
    """Test that APIError exception returns available=False."""
    mock_client = MagicMock()
    mock_client.containers.get.side_effect = docker.errors.APIError("API error")

    stats = await collect_container_stats(mock_client, "error_container")

    assert stats.available is False
