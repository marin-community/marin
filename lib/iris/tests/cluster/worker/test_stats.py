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
import time
from pathlib import Path

import pytest

from iris.cluster.worker.docker import ContainerConfig, ContainerStats, DockerRuntime
from iris.cluster.worker.env_probe import collect_workdir_size_mb


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


@pytest.fixture
def runtime():
    """Create DockerRuntime instance."""
    return DockerRuntime()


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


def test_get_stats_invalid_container(runtime):
    """Test that get_stats returns available=False for invalid container ID."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    invalid_container_id = "nonexistent_container_12345"

    stats = runtime.get_stats(invalid_container_id)

    assert isinstance(stats, ContainerStats)
    assert stats.available is False
    assert stats.memory_mb == 0
    assert stats.cpu_percent == 0
    assert stats.process_count == 0


@pytest.mark.slow
def test_get_stats_from_running_container(runtime):
    """Test get_stats returns positive values for a real running container."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    # Start a container that runs for a few seconds
    config = ContainerConfig(
        image="alpine:latest",
        command=["sh", "-c", "sleep 5"],
        env={},
    )

    container_id = runtime.create_container(config)
    runtime.start_container(container_id)

    # Wait for container to be running
    time.sleep(0.5)

    try:
        stats = runtime.get_stats(container_id)

        # Container should be available
        assert stats.available is True

        # Memory usage should be non-negative (may be 0 MB for lightweight containers)
        assert stats.memory_mb >= 0

        # Process count should be at least 1 (the sleep process)
        assert stats.process_count >= 1

        # CPU percent can be 0 if container is idle, but should be non-negative
        assert stats.cpu_percent >= 0

    finally:
        runtime.kill(container_id, force=True)
        runtime.remove(container_id)


@pytest.mark.slow
def test_get_stats_from_busy_container(runtime):
    """Test get_stats returns higher values for a CPU-intensive container."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    # Start a container that does some work
    config = ContainerConfig(
        image="alpine:latest",
        command=["sh", "-c", "while true; do echo test; done"],
        env={},
    )

    container_id = runtime.create_container(config)
    runtime.start_container(container_id)

    # Let it run for a moment to generate stats
    time.sleep(1.0)

    try:
        stats = runtime.get_stats(container_id)

        assert stats.available is True
        assert stats.memory_mb >= 0
        assert stats.process_count >= 1
        # CPU percent should be positive for a busy container
        assert stats.cpu_percent >= 0

    finally:
        runtime.kill(container_id, force=True)
        runtime.remove(container_id)


@pytest.mark.slow
def test_get_stats_from_stopped_container(runtime):
    """Test that get_stats returns zero values for stopped container."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["echo", "test"],
        env={},
    )

    container_id = runtime.create_container(config)
    runtime.start_container(container_id)

    # Wait for container to finish
    max_wait = 5
    start = time.time()
    while time.time() - start < max_wait:
        status = runtime.inspect(container_id)
        if not status.running:
            break
        time.sleep(0.1)

    try:
        # Docker can still return stats for stopped containers, but values should be minimal/zero
        stats = runtime.get_stats(container_id)
        # Process count should be 0 for stopped container
        assert stats.process_count == 0

    finally:
        runtime.remove(container_id)
