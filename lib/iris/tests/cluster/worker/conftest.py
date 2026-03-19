# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for worker tests (both mock and Docker-based)."""

from unittest.mock import Mock

import pytest

from iris.cluster.bundle import BundleStore
from iris.cluster.runtime.docker import DockerRuntime
from iris.cluster.runtime.types import ContainerPhase, ContainerStats, ContainerStatus
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.time_utils import Duration


@pytest.fixture
def docker_runtime(tmp_path):
    """DockerRuntime that cleans up its own containers after the test."""
    rt = DockerRuntime(cache_dir=tmp_path / "cache")
    yield rt
    rt.cleanup()


@pytest.fixture
def mock_bundle_store(tmp_path):
    """Create mock BundleStore with a real temp directory."""
    cache = Mock(spec=BundleStore)
    cache.extract_bundle_to = Mock()
    return cache


@pytest.fixture
def mock_runtime():
    """Create mock DockerRuntime.

    By default, simulates a container that runs and completes successfully.
    """
    runtime = Mock(spec=DockerRuntime)

    handle = Mock()
    handle.container_id = "container123"
    handle.build = Mock(return_value=[])
    handle.run = Mock()

    call_count = [0]
    status_sequence = [
        ContainerStatus(phase=ContainerPhase.RUNNING),
        ContainerStatus(phase=ContainerPhase.STOPPED, exit_code=0),
    ]

    def status_side_effect():
        idx = min(call_count[0], len(status_sequence) - 1)
        call_count[0] += 1
        return status_sequence[idx]

    handle.status = Mock(side_effect=status_side_effect)
    handle.stop = Mock()

    log_reader_mock = Mock()
    log_reader_mock.read = Mock(return_value=[])
    log_reader_mock.read_all = Mock(return_value=[])
    handle.log_reader = Mock(return_value=log_reader_mock)
    handle.stats = Mock(return_value=ContainerStats(memory_mb=100, cpu_percent=50, process_count=5, available=True))
    handle.disk_usage_mb = Mock(return_value=0)
    handle.cleanup = Mock()

    runtime.create_container = Mock(return_value=handle)
    runtime.stage_bundle = Mock()
    runtime.list_iris_containers = Mock(return_value=[])
    runtime.remove_all_iris_containers = Mock(return_value=0)
    runtime.cleanup = Mock()
    return runtime


@pytest.fixture
def mock_worker(mock_bundle_store, mock_runtime, tmp_path):
    """Create Worker with mocked dependencies."""
    config = WorkerConfig(
        port=0,
        port_range=(50000, 50100),
        poll_interval=Duration.from_seconds(0.1),
        cache_dir=tmp_path / "cache",
        default_task_image="mock-image",
    )
    return Worker(
        config,
        bundle_store=mock_bundle_store,
        container_runtime=mock_runtime,
    )
