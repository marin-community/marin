# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for worker tests (both mock and Docker-based)."""

from collections.abc import Callable
from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest
from iris.cluster.bundle import BundleStore
from iris.cluster.runtime.docker import DockerRuntime
from iris.cluster.runtime.types import ContainerPhase, ContainerStats, ContainerStatus
from iris.cluster.types import Entrypoint, JobName
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.cluster.worker.worker_types import LogLine
from iris.rpc import job_pb2
from iris.time_proto import duration_to_proto
from rigging.timing import Duration


@pytest.fixture
def docker_runtime(tmp_path):
    """DockerRuntime that cleans up its own containers after the test."""
    rt = DockerRuntime(cache_dir=tmp_path / "cache")
    yield rt
    rt.cleanup()


@pytest.fixture
def mock_bundle_store(tmp_path):
    cache = Mock(spec=BundleStore)
    cache.extract_bundle_to = Mock()
    return cache


@dataclass
class FakeLogReader:
    """In-memory RuntimeLogReader for tests."""

    _logs: list[LogLine] = field(default_factory=list)
    _cursor: int = 0

    def read(self) -> list[LogLine]:
        new = self._logs[self._cursor :]
        self._cursor = len(self._logs)
        return new

    def read_all(self) -> list[LogLine]:
        return list(self._logs)


class FakeContainerHandle:
    """In-memory ContainerHandle for tests.

    Replaces the MagicMock-based create_mock_container_handle with a
    type-checkable class that implements the ContainerHandle protocol.
    Supports failure injection via build_error, run_error, and stop_hook.
    """

    def __init__(
        self,
        status_sequence: list[ContainerStatus] | None = None,
        run_error: Exception | None = None,
    ):
        if status_sequence is None:
            status_sequence = [
                ContainerStatus(phase=ContainerPhase.RUNNING),
                ContainerStatus(phase=ContainerPhase.STOPPED, exit_code=0),
            ]
        self._status_sequence = status_sequence
        self._status_cursor = 0
        self._run_error = run_error
        self.build_error: Exception | None = None
        self.stop_hook: object = None  # Callable[[bool], None] | None — set by tests for slow_stop etc.
        self.stop_calls: list[dict[str, object]] = []
        self._cleaned_up = False

    @property
    def container_id(self) -> str | None:
        return "container123"

    def build(self, on_logs: Callable[[list[LogLine]], None] | None = None) -> list[LogLine]:
        if self.build_error is not None:
            raise self.build_error
        return []

    def run(self) -> None:
        if self._run_error is not None:
            raise self._run_error

    def stop(self, force: bool = False) -> None:
        self.stop_calls.append({"force": force})
        if self.stop_hook is not None:
            self.stop_hook(force)  # type: ignore[operator]

    def status(self) -> ContainerStatus:
        idx = min(self._status_cursor, len(self._status_sequence) - 1)
        self._status_cursor += 1
        return self._status_sequence[idx]

    def log_reader(self) -> FakeLogReader:
        return FakeLogReader()

    def stats(self) -> ContainerStats:
        return ContainerStats(memory_mb=100, cpu_millicores=500, process_count=5, available=True)

    def disk_usage_mb(self) -> int:
        return 0

    def profile(self, duration_seconds: int, profile_type: job_pb2.ProfileType) -> bytes:
        raise RuntimeError("profiling not supported in FakeContainerHandle")

    def cleanup(self) -> None:
        self._cleaned_up = True


def create_mock_container_handle(
    status_sequence: list[ContainerStatus] | None = None,
    run_side_effect: Exception | None = None,
) -> FakeContainerHandle:
    return FakeContainerHandle(
        status_sequence=status_sequence,
        run_error=run_side_effect,
    )


@pytest.fixture
def mock_runtime():
    """Mock DockerRuntime that produces FakeContainerHandle instances.

    The runtime itself stays as a Mock because tests need mock assertions
    on create_container (assert_called, call_args, side_effect injection).
    """
    runtime = Mock(spec=DockerRuntime)
    runtime.create_container = Mock(side_effect=lambda config: create_mock_container_handle())
    runtime.stage_bundle = Mock()
    runtime.list_iris_containers = Mock(return_value=[])
    runtime.remove_all_iris_containers = Mock(return_value=0)
    runtime.remove_containers = Mock(return_value=0)
    runtime.discover_containers = Mock(return_value=[])
    runtime.adopt_container = Mock(side_effect=lambda cid: create_mock_container_handle())
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


def create_run_task_request(
    task_id: str = JobName.root("test-user", "test-task").task(0).to_wire(),
    num_tasks: int = 1,
    ports: list[str] | None = None,
    attempt_id: int = 0,
    task_image: str = "",
):
    def test_fn():
        print("Hello from test")

    entrypoint_proto = Entrypoint.from_callable(test_fn).to_proto()

    env_config = job_pb2.EnvironmentConfig(
        env_vars={
            "TEST_VAR": "value",
            "TASK_VAR": "task_value",
        },
        extras=["dev"],
        dockerfile="FROM python:3.11-slim\nRUN echo test",
    )

    resources = job_pb2.ResourceSpecProto(cpu_millicores=2000, memory_bytes=4 * 1024**3)

    request = job_pb2.RunTaskRequest(
        task_id=task_id,
        num_tasks=num_tasks,
        attempt_id=attempt_id,
        entrypoint=entrypoint_proto,
        environment=env_config,
        bundle_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        resources=resources,
        ports=ports or [],
        task_image=task_image,
    )
    request.timeout.CopyFrom(duration_to_proto(Duration.from_seconds(300)))
    return request
