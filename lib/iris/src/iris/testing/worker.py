# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker fakes and factories."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from unittest.mock import Mock

from rigging.timing import Duration

from iris.cluster.bundle import BundleStore
from iris.cluster.runtime.docker import DockerRuntime
from iris.cluster.runtime.entrypoint import build_runtime_entrypoint
from iris.cluster.runtime.types import ContainerPhase, ContainerStats, ContainerStatus
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.cluster.worker.worker_types import LogLine
from iris.resources.attempt import AttemptLaunch, AttemptLaunchTemplate
from iris.resources.endpoint import ProfileConfiguration
from iris.resources.execution import Entrypoint, Environment, ResourceSpec
from iris.resources.job import ContainerProfile, PriorityBand
from iris.resources.names import (
    AttemptUid,
    JobName,
)


def make_docker_runtime(tmp_path) -> DockerRuntime:
    return DockerRuntime(cache_dir=tmp_path / "cache")


def make_mock_bundle_store() -> Mock:
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
        self._killed = False

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
        if force:
            # Model real runtimes: once SIGKILL has been delivered the container
            # reports STOPPED on the next inspect. Tests that need to simulate a
            # wedged container should override _killed back to False.
            self._killed = True

    def status(self) -> ContainerStatus:
        if self._killed:
            return ContainerStatus(phase=ContainerPhase.STOPPED, exit_code=137)
        idx = min(self._status_cursor, len(self._status_sequence) - 1)
        self._status_cursor += 1
        return self._status_sequence[idx]

    def log_reader(self) -> FakeLogReader:
        return FakeLogReader()

    def stats(self) -> ContainerStats:
        return ContainerStats(memory_mb=100, cpu_millicores=500, process_count=5, available=True)

    def disk_usage_mb(self) -> int:
        return 0

    def profile(self, duration_seconds: int, profile_type: ProfileConfiguration) -> bytes:
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


def make_mock_runtime() -> Mock:
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


def make_mock_worker(mock_bundle_store, mock_runtime, tmp_path) -> Worker:
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


def create_attempt_launch(
    task_id: str = JobName.root("test-user", "test-task").task(0).to_wire(),
    num_tasks: int = 1,
    ports: list[str] | None = None,
    attempt_id: int = 0,
    task_image: str = "",
    attempt_uid: str | None = None,
):
    # Worker.submit_task requires a non-empty attempt_uid. Default to a value
    # derived from the task identity so tests that don't care about UID still
    # get a unique-per-attempt one.
    if attempt_uid is None:
        digest = hashlib.sha1(f"{task_id}#{attempt_id}".encode()).hexdigest()
        attempt_uid = digest[:16]

    def test_fn():
        print("Hello from test")

    environment = Environment(
        env_vars={
            "TEST_VAR": "value",
            "TASK_VAR": "task_value",
        },
        setup_scripts=["uv sync\n"],
    )

    return AttemptLaunch(
        task_id=JobName.from_wire(task_id),
        attempt_id=attempt_id,
        attempt_uid=AttemptUid(attempt_uid),
        template=AttemptLaunchTemplate(
            num_tasks=num_tasks,
            entrypoint=build_runtime_entrypoint(Entrypoint.from_callable(test_fn), Environment({}, ())),
            environment=environment,
            bundle_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            resources=ResourceSpec(cpu=2, memory=4 * 1024**3),
            timeout=Duration.from_seconds(300),
            ports=tuple(ports or ()),
            constraints=(),
            task_image=task_image,
            coscheduling=None,
            priority_band=PriorityBand.INHERIT,
            container_profile=ContainerProfile.UNSPECIFIED,
        ),
    )
