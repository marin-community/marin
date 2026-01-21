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

"""Tests for Controller parallel dispatch functionality.

These tests verify that the Controller correctly dispatches tasks in parallel
and handles dispatch failures properly.
"""

import threading
import time

import pytest

from iris.cluster.controller.controller import (
    Controller,
    ControllerConfig,
    WorkerStubFactory,
)
from iris.cluster.controller.events import (
    JobSubmittedEvent,
    WorkerRegisteredEvent,
)
from iris.cluster.types import JobId, WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import now_ms

# =============================================================================
# Test Fixtures
# =============================================================================


class MockWorkerStub:
    """Mock worker stub for testing dispatch behavior."""

    def __init__(
        self,
        *,
        delay: float = 0.0,
        fail: bool = False,
        fail_message: str = "Mock RPC failure",
    ):
        self.delay = delay
        self.fail = fail
        self.fail_message = fail_message
        self.run_task_calls: list[cluster_pb2.Worker.RunTaskRequest] = []
        self.kill_task_calls: list[cluster_pb2.Worker.KillTaskRequest] = []

    def run_task(self, request: cluster_pb2.Worker.RunTaskRequest) -> cluster_pb2.Worker.RunTaskResponse:
        self.run_task_calls.append(request)
        if self.delay > 0:
            time.sleep(self.delay)
        if self.fail:
            raise ConnectionError(self.fail_message)
        return cluster_pb2.Worker.RunTaskResponse(task_id=request.task_id)

    def get_task_status(self, request: cluster_pb2.Worker.GetTaskStatusRequest) -> cluster_pb2.TaskStatus:
        return cluster_pb2.TaskStatus()

    def list_tasks(self, request: cluster_pb2.Worker.ListTasksRequest) -> cluster_pb2.Worker.ListTasksResponse:
        return cluster_pb2.Worker.ListTasksResponse()

    def health_check(self, request: cluster_pb2.Empty) -> cluster_pb2.Worker.HealthResponse:
        return cluster_pb2.Worker.HealthResponse(healthy=True)

    def kill_task(self, request: cluster_pb2.Worker.KillTaskRequest) -> cluster_pb2.Empty:
        self.kill_task_calls.append(request)
        return cluster_pb2.Empty()


class MockWorkerStubFactory(WorkerStubFactory):
    """Factory that returns mock stubs based on address."""

    def __init__(self):
        self.stubs: dict[str, MockWorkerStub] = {}
        self._default_stub: MockWorkerStub | None = None

    def set_stub(self, address: str, stub: MockWorkerStub) -> None:
        self.stubs[address] = stub

    def set_default_stub(self, stub: MockWorkerStub) -> None:
        self._default_stub = stub

    def get_stub(self, address: str) -> MockWorkerStub:
        if address in self.stubs:
            return self.stubs[address]
        if self._default_stub is not None:
            return self._default_stub
        # Return a default working stub
        stub = MockWorkerStub()
        self.stubs[address] = stub
        return stub


@pytest.fixture
def job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(
        name: str = "test-job",
        replicas: int = 1,
        coscheduling_group_by: str | None = None,
    ) -> cluster_pb2.Controller.LaunchJobRequest:
        req = cluster_pb2.Controller.LaunchJobRequest(
            name=name,
            serialized_entrypoint=b"test",
            resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3, replicas=replicas),
            environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        )
        if coscheduling_group_by:
            req.coscheduling.group_by = coscheduling_group_by
        return req

    return _make


@pytest.fixture
def worker_metadata():
    """Create WorkerMetadata for testing."""

    def _make(
        cpu: int = 10,
        memory_bytes: int = 10 * 1024**3,
        disk_bytes: int = 10 * 1024**3,
        tpu_name: str = "",
        tpu_worker_id: int = 0,
    ) -> cluster_pb2.WorkerMetadata:
        device = cluster_pb2.DeviceConfig()
        if tpu_name:
            device.tpu.CopyFrom(cluster_pb2.TpuDevice(variant=tpu_name))
        else:
            device.cpu.CopyFrom(cluster_pb2.CpuDevice(variant="cpu"))

        meta = cluster_pb2.WorkerMetadata(
            hostname="test-worker",
            ip_address="127.0.0.1",
            cpu_count=cpu,
            memory_bytes=memory_bytes,
            disk_bytes=disk_bytes,
            device=device,
            tpu_name=tpu_name,
            tpu_worker_id=str(tpu_worker_id),
        )
        # Add attributes for coscheduling
        if tpu_name:
            meta.attributes["tpu-name"].string_value = tpu_name
            meta.attributes["tpu-worker-id"].int_value = tpu_worker_id

        return meta

    return _make


def register_worker(
    controller: Controller,
    worker_id: str,
    address: str,
    metadata: cluster_pb2.WorkerMetadata,
) -> WorkerId:
    """Register a worker via state event."""
    wid = WorkerId(worker_id)
    controller.state.handle_event(
        WorkerRegisteredEvent(
            worker_id=wid,
            address=address,
            metadata=metadata,
            timestamp_ms=now_ms(),
        )
    )
    return wid


def submit_job(controller: Controller, job_id: str, request: cluster_pb2.Controller.LaunchJobRequest) -> JobId:
    """Submit a job via state event."""
    jid = JobId(job_id)
    controller.state.handle_event(
        JobSubmittedEvent(
            job_id=jid,
            request=request,
            timestamp_ms=now_ms(),
        )
    )
    return jid


# =============================================================================
# Parallel Dispatch Tests
# =============================================================================


def test_multiple_tasks_dispatched_in_parallel(job_request, worker_metadata):
    """Verify that multiple tasks are dispatched in parallel, not serially.

    If tasks were dispatched serially with 100ms delay each, 4 tasks would take 400ms+.
    With parallel dispatch, they should complete in ~100ms (plus overhead).
    """
    stub_factory = MockWorkerStubFactory()
    config = ControllerConfig(port=0, scheduler_interval_seconds=10.0)  # Disable auto-scheduling
    controller = Controller(config=config, worker_stub_factory=stub_factory)

    # Create stubs with a delay to measure parallelism
    delay_per_task = 0.1  # 100ms
    for i in range(4):
        stub = MockWorkerStub(delay=delay_per_task)
        stub_factory.set_stub(f"addr{i}:8080", stub)

    # Register 4 workers
    for i in range(4):
        register_worker(controller, f"w{i}", f"addr{i}:8080", worker_metadata())

    # Submit job with 4 tasks
    submit_job(controller, "j1", job_request(replicas=4))

    # Run scheduling manually (don't start background thread)
    start_time = time.monotonic()
    controller._run_scheduling()
    elapsed = time.monotonic() - start_time

    # Verify all tasks were dispatched
    total_dispatched = sum(len(stub.run_task_calls) for stub in stub_factory.stubs.values())
    assert total_dispatched == 4, f"Expected 4 dispatched tasks, got {total_dispatched}"

    # Verify parallel dispatch: should complete in ~delay_per_task, not 4*delay_per_task
    # Allow some overhead (2x the single task delay)
    max_expected_time = delay_per_task * 2
    assert elapsed < max_expected_time, f"Dispatch took {elapsed:.3f}s, expected <{max_expected_time:.3f}s (parallel)"

    controller.stop()


def test_dispatch_failure_marks_worker_failed(job_request, worker_metadata):
    """When dispatch RPC fails, the worker should be marked as failed."""
    stub_factory = MockWorkerStubFactory()
    config = ControllerConfig(port=0, scheduler_interval_seconds=10.0)
    controller = Controller(config=config, worker_stub_factory=stub_factory)

    # Create a failing stub
    stub = MockWorkerStub(fail=True, fail_message="Connection refused")
    stub_factory.set_stub("addr1:8080", stub)

    # Register worker
    worker_id = register_worker(controller, "w1", "addr1:8080", worker_metadata())

    # Submit job
    submit_job(controller, "j1", job_request())

    # Verify worker is healthy before dispatch
    worker = controller.state.get_worker(worker_id)
    assert worker is not None
    assert worker.healthy is True

    # Run scheduling
    controller._run_scheduling()

    # Verify dispatch was attempted
    assert len(stub.run_task_calls) == 1

    # Verify worker is now marked as unhealthy
    worker = controller.state.get_worker(worker_id)
    assert worker is not None
    assert worker.healthy is False

    controller.stop()


def test_partial_dispatch_failure_in_coscheduled_job(job_request, worker_metadata):
    """When some tasks in a coscheduled job fail to dispatch, those workers should be marked failed."""
    stub_factory = MockWorkerStubFactory()
    config = ControllerConfig(port=0, scheduler_interval_seconds=10.0)
    controller = Controller(config=config, worker_stub_factory=stub_factory)

    # Create stubs: 2 succeed, 2 fail
    for i in range(2):
        stub_factory.set_stub(f"addr{i}:8080", MockWorkerStub())
    for i in range(2, 4):
        stub_factory.set_stub(f"addr{i}:8080", MockWorkerStub(fail=True))

    # Register 4 workers with same tpu-name for coscheduling
    for i in range(4):
        register_worker(
            controller,
            f"w{i}",
            f"addr{i}:8080",
            worker_metadata(tpu_name="tpu-a", tpu_worker_id=i),
        )

    # Submit coscheduled job with 4 tasks
    submit_job(controller, "j1", job_request(replicas=4, coscheduling_group_by="tpu-name"))

    # Run scheduling
    controller._run_scheduling()

    # Verify first 2 workers still healthy, last 2 unhealthy
    for i in range(2):
        worker = controller.state.get_worker(WorkerId(f"w{i}"))
        assert worker is not None
        assert worker.healthy is True, f"Worker w{i} should be healthy"

    for i in range(2, 4):
        worker = controller.state.get_worker(WorkerId(f"w{i}"))
        assert worker is not None
        assert worker.healthy is False, f"Worker w{i} should be unhealthy"

    controller.stop()


def test_dispatch_executor_shutdown_gracefully(job_request, worker_metadata):
    """Verify that the dispatch executor shuts down properly when controller stops."""
    stub_factory = MockWorkerStubFactory()
    config = ControllerConfig(port=0, scheduler_interval_seconds=10.0)
    controller = Controller(config=config, worker_stub_factory=stub_factory)

    # Add a slow stub that will still be running when we stop
    slow_stub = MockWorkerStub(delay=1.0)
    stub_factory.set_default_stub(slow_stub)

    register_worker(controller, "w1", "addr1:8080", worker_metadata())
    submit_job(controller, "j1", job_request())

    # Start dispatch in background (will be slow)
    def run_scheduling():
        controller._run_scheduling()

    thread = threading.Thread(target=run_scheduling)
    thread.start()

    # Give it time to start the dispatch
    time.sleep(0.1)

    # Stop should not hang even with slow dispatch
    start = time.monotonic()
    controller.stop()
    elapsed = time.monotonic() - start

    # Should complete quickly (executor shutdown cancels pending futures)
    assert elapsed < 2.0, f"Stop took {elapsed:.3f}s, should be quick"

    thread.join(timeout=2.0)
