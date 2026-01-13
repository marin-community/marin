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
from unittest.mock import MagicMock, patch

import docker.errors
import pytest

from fluster.cluster.worker.docker import ContainerStats, DockerRuntime
from fluster.cluster.worker.worker_types import collect_workdir_size_mb


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


def test_collect_workdir_size_mb_with_temp_directory(tmp_path):
    """Test workdir size calculation with a temporary directory."""
    # Create some files in temp directory
    (tmp_path / "file1.txt").write_text("x" * 1024 * 100)  # 100 KB
    (tmp_path / "file2.txt").write_text("y" * 1024 * 100)  # 100 KB

    size_mb = collect_workdir_size_mb(tmp_path)

    # Size should be at least 1 MB (200 KB rounded up)
    assert size_mb >= 1


def test_collect_workdir_size_mb_nonexistent_directory():
    """Test workdir size returns 0 for non-existent directory."""
    nonexistent = Path("/nonexistent/path/does/not/exist")

    size_mb = collect_workdir_size_mb(nonexistent)

    assert size_mb == 0


def test_get_stats_invalid_container():
    """Test that get_stats returns available=False for invalid container ID."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    runtime = DockerRuntime()
    invalid_container_id = "nonexistent_container_12345"

    stats = runtime.get_stats(invalid_container_id)

    assert isinstance(stats, ContainerStats)
    assert stats.available is False
    assert stats.memory_mb == 0
    assert stats.cpu_percent == 0
    assert stats.process_count == 0


def test_get_stats_with_mock():
    """Test get_stats parsing with mocked Docker client."""
    runtime = DockerRuntime()

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

    mock_container = MagicMock()
    mock_container.stats.return_value = mock_stats

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container

    with patch("fluster.cluster.worker.docker.docker") as mock_docker:
        mock_docker.from_env.return_value = mock_client
        stats = runtime.get_stats("test_container")

    assert stats.available is True
    assert stats.memory_mb == 100
    assert stats.cpu_percent == 400  # (1000000000 / 1000000000) * 4 * 100
    assert stats.process_count == 5


def test_get_stats_not_found_exception():
    """Test that NotFound exception returns available=False."""
    runtime = DockerRuntime()

    mock_client = MagicMock()
    mock_client.containers.get.side_effect = docker.errors.NotFound("Container not found")

    with patch("fluster.cluster.worker.docker.docker") as mock_docker:
        mock_docker.from_env.return_value = mock_client
        mock_docker.errors = docker.errors
        stats = runtime.get_stats("missing_container")

    assert stats.available is False


def test_get_stats_api_error_exception():
    """Test that APIError exception returns available=False."""
    runtime = DockerRuntime()

    mock_client = MagicMock()
    mock_client.containers.get.side_effect = docker.errors.APIError("API error")

    with patch("fluster.cluster.worker.docker.docker") as mock_docker:
        mock_docker.from_env.return_value = mock_client
        mock_docker.errors = docker.errors
        stats = runtime.get_stats("error_container")

    assert stats.available is False
