# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Worker class (includes PortAllocator and task management)."""

import hashlib
import socket
import time
import zipfile
from unittest.mock import Mock

import pytest
from connectrpc.request import RequestContext

from iris.cluster.runtime.docker import DockerRuntime
from iris.cluster.runtime.types import (
    ContainerErrorKind,
    ContainerInfraError,
    ContainerPhase,
    ContainerStatus,
    DiscoveredContainer,
    ExecutionStage,
)
from iris.cluster.types import Entrypoint, JobName
from iris.cluster.worker.task_attempt import TaskAttempt
from iris.cluster.worker.port_allocator import PortAllocator
from iris.cluster.worker.service import WorkerServiceImpl
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.rpc import job_pb2
from rigging.timing import Duration
from iris.cluster.worker.worker_types import LogLine
from tests.cluster.worker.conftest import (
    FakeContainerHandle,
    FakeLogReader,
    create_mock_container_handle,
    create_run_task_request,
)
from iris.test_util import wait_for_condition

pytestmark = pytest.mark.timeout(10)

# ============================================================================
# PortAllocator Tests
# ============================================================================


@pytest.fixture
def allocator():
    return PortAllocator(port_range=(40000, 40100))


def test_allocated_ports_are_usable(allocator):
    ports = allocator.allocate(count=3)

    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", port))


def test_no_port_reuse_before_release(allocator):
    ports1 = allocator.allocate(count=5)
    ports2 = allocator.allocate(count=5)

    assert len(set(ports1) & set(ports2)) == 0


def test_concurrent_allocations(allocator):
    import threading

    results = []

    def allocate_ports():
        ports = allocator.allocate(count=5)
        results.append(ports)

    threads = [threading.Thread(target=allocate_ports) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    all_ports = []
    for ports in results:
        all_ports.extend(ports)

    assert len(all_ports) == len(set(all_ports))


# ============================================================================
# Worker Tests (with mocked dependencies)
# ============================================================================


def test_task_lifecycle_phases(mock_worker):
    """Test task transitions through PENDING -> BUILDING -> RUNNING -> SUCCEEDED."""
    request = create_run_task_request()
    task_id = mock_worker.submit_task(request)

    task = mock_worker.get_task(task_id)
    task.thread.join(timeout=15.0)

    final_task = mock_worker.get_task(task_id)
    assert final_task.status == job_pb2.TASK_STATE_SUCCEEDED
    assert final_task.exit_code == 0


def test_runtime_stage_bundle_receives_workdir_files(mock_worker, mock_runtime):
    request = create_run_task_request()
    request.entrypoint.workdir_files["extra.txt"] = b"extra"
    task_id = mock_worker.submit_task(request)

    task = mock_worker.get_task(task_id)
    task.thread.join(timeout=15.0)

    assert mock_runtime.stage_bundle.called
    kwargs = mock_runtime.stage_bundle.call_args.kwargs
    assert kwargs["bundle_id"] == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert kwargs["workdir_files"]["extra.txt"] == b"extra"


def test_task_with_ports(mock_worker):
    """Test task with port allocation."""
    request = create_run_task_request(ports=["http", "grpc"])
    task_id = mock_worker.submit_task(request)

    task = mock_worker.get_task(task_id)

    # Ports are allocated in the task thread during setup, so wait for the
    # task to move past PENDING before checking.
    wait_for_condition(lambda: task.status != job_pb2.TASK_STATE_PENDING)

    assert len(task.ports) == 2
    assert "http" in task.ports
    assert "grpc" in task.ports
    assert task.ports["http"] != task.ports["grpc"]

    task.thread.join(timeout=15.0)


def test_task_failure_on_nonzero_exit(mock_worker, mock_runtime):
    """Test task fails when container exits with non-zero code."""
    # Update the mock handle's status to return failure immediately
    mock_handle = create_mock_container_handle(
        status_sequence=[ContainerStatus(phase=ContainerPhase.STOPPED, exit_code=1)]
    )
    mock_runtime.create_container = Mock(return_value=mock_handle)

    request = create_run_task_request()
    task_id = mock_worker.submit_task(request)

    task = mock_worker.get_task(task_id)
    task.thread.join(timeout=15.0)

    final_task = mock_worker.get_task(task_id)
    assert final_task.status == job_pb2.TASK_STATE_FAILED
    assert final_task.exit_code == 1
    assert "Exit code: 1" in final_task.error


def test_tpu_bad_node_stderr_promotes_to_worker_failed(mock_worker, mock_runtime):
    """Non-zero exit with TPU bad-node stderr -> WORKER_FAILED (issue #4783)."""
    bad_node_stderr = [
        LogLine.now(source="stdout", data="startup: launching vLLM engine"),
        LogLine.now(
            source="stderr",
            data=(
                "jax.errors.JaxRuntimeError: UNKNOWN: TPU initialization failed: "
                "open(/dev/vfio/0): Device or resource busy: Device or resource busy; "
                "Couldn't open iommu group /dev/vfio/0"
            ),
        ),
    ]
    populated_reader = FakeLogReader(_logs=list(bad_node_stderr))

    class _HandleWithStderr(FakeContainerHandle):
        def log_reader(self) -> FakeLogReader:
            return populated_reader

    mock_handle = _HandleWithStderr(
        status_sequence=[
            ContainerStatus(phase=ContainerPhase.RUNNING),
            ContainerStatus(phase=ContainerPhase.STOPPED, exit_code=1),
        ]
    )
    mock_runtime.create_container = Mock(return_value=mock_handle)

    request = create_run_task_request()
    task_id = mock_worker.submit_task(request)

    task = mock_worker.get_task(task_id)
    task.thread.join(timeout=15.0)

    final_task = mock_worker.get_task(task_id)
    assert final_task.status == job_pb2.TASK_STATE_WORKER_FAILED
    assert final_task.exit_code == 1
    assert final_task.error is not None
    assert "TPU init failure" in final_task.error
    assert "Couldn't open iommu group" in final_task.error


def test_non_tpu_stderr_still_maps_to_failed(mock_worker, mock_runtime):
    """Non-zero exit with unrelated stderr stays FAILED (no false promotion)."""
    user_stderr = [
        LogLine.now(source="stderr", data="Traceback (most recent call last):"),
        LogLine.now(source="stderr", data='ValueError: bad user config: expected "foo"'),
    ]
    populated_reader = FakeLogReader(_logs=list(user_stderr))

    class _HandleWithStderr(FakeContainerHandle):
        def log_reader(self) -> FakeLogReader:
            return populated_reader

    mock_handle = _HandleWithStderr(
        status_sequence=[
            ContainerStatus(phase=ContainerPhase.RUNNING),
            ContainerStatus(phase=ContainerPhase.STOPPED, exit_code=1),
        ]
    )
    mock_runtime.create_container = Mock(return_value=mock_handle)

    request = create_run_task_request()
    task_id = mock_worker.submit_task(request)

    task = mock_worker.get_task(task_id)
    task.thread.join(timeout=15.0)

    final_task = mock_worker.get_task(task_id)
    assert final_task.status == job_pb2.TASK_STATE_FAILED
    assert final_task.exit_code == 1


def test_task_failure_on_error(mock_worker, mock_runtime):
    """Test task fails when container returns error."""
    # Update the mock handle's status to return error after first poll
    mock_handle = create_mock_container_handle(
        status_sequence=[
            ContainerStatus(phase=ContainerPhase.RUNNING),
            ContainerStatus(phase=ContainerPhase.STOPPED, exit_code=1, error="Container crashed"),
        ]
    )
    mock_runtime.create_container = Mock(return_value=mock_handle)

    request = create_run_task_request()
    task_id = mock_worker.submit_task(request)

    task = mock_worker.get_task(task_id)
    task.thread.join(timeout=10.0)

    final_task = mock_worker.get_task(task_id)
    assert final_task.status == job_pb2.TASK_STATE_FAILED
    assert final_task.error == "Container crashed"


def test_task_infra_not_found_error_maps_to_worker_failed(mock_worker, mock_runtime):
    """Infrastructure disappearance should consume preemption budget, not failure budget."""
    mock_handle = create_mock_container_handle(
        status_sequence=[
            ContainerStatus(
                phase=ContainerPhase.STOPPED,
                exit_code=1,
                error="Task pod not found after retry window: name=iris-task-abc, namespace=iris",
                error_kind=ContainerErrorKind.INFRA_NOT_FOUND,
            )
        ]
    )
    mock_runtime.create_container = Mock(return_value=mock_handle)

    request = create_run_task_request()
    task_id = mock_worker.submit_task(request)

    task = mock_worker.get_task(task_id)
    task.thread.join(timeout=10.0)

    final_task = mock_worker.get_task(task_id)
    assert final_task.status == job_pb2.TASK_STATE_WORKER_FAILED
    assert "Task pod not found" in (final_task.error or "")


def test_docker_create_infra_error_maps_to_worker_failed(mock_worker, mock_runtime):
    """ContainerInfraError during build() should transition to WORKER_FAILED (preemption budget)."""
    mock_handle = create_mock_container_handle()
    mock_handle.build_error = ContainerInfraError(
        "Failed to create container (infra): error getting credentials - "
        "err: exit status 1, out: `You do not currently have an active account selected.`"
    )
    mock_runtime.create_container = Mock(return_value=mock_handle)

    request = create_run_task_request()
    task_id = mock_worker.submit_task(request)

    task = mock_worker.get_task(task_id)
    task.thread.join(timeout=15.0)

    final_task = mock_worker.get_task(task_id)
    assert final_task.status == job_pb2.TASK_STATE_WORKER_FAILED
    assert "error getting credentials" in (final_task.error or "")


def test_docker_create_user_error_still_maps_to_failed(mock_worker, mock_runtime):
    """A plain RuntimeError during build() should still transition to TASK_STATE_FAILED."""
    mock_handle = create_mock_container_handle()
    mock_handle.build_error = RuntimeError("Build failed with exit_code=1")
    mock_runtime.create_container = Mock(return_value=mock_handle)

    request = create_run_task_request()
    task_id = mock_worker.submit_task(request)

    task = mock_worker.get_task(task_id)
    task.thread.join(timeout=15.0)

    final_task = mock_worker.get_task(task_id)
    assert final_task.status == job_pb2.TASK_STATE_FAILED
    assert "Build failed" in (final_task.error or "")


def test_task_exception_handling(mock_worker):
    """Test task handles exceptions during execution."""
    mock_worker._runtime.stage_bundle = Mock(side_effect=Exception("Bundle download failed"))

    request = create_run_task_request()
    task_id = mock_worker.submit_task(request)

    task = mock_worker.get_task(task_id)
    task.thread.join(timeout=15.0)

    final_task = mock_worker.get_task(task_id)
    assert final_task.status == job_pb2.TASK_STATE_FAILED
    assert "Bundle download failed" in final_task.error


def test_list_tasks(mock_worker):
    """Test listing all tasks."""
    requests = [
        create_run_task_request(task_id=JobName.root("test-user", "test-job").task(i).to_wire()) for i in range(3)
    ]

    for request in requests:
        mock_worker.submit_task(request)

    tasks = mock_worker.list_tasks()
    assert len(tasks) == 3


def test_kill_running_task(mock_worker, mock_runtime):
    """Test killing a running task with graceful timeout."""
    # Create a handle that stays running until killed
    mock_handle = create_mock_container_handle(
        status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100
    )  # Stay running
    mock_runtime.create_container = Mock(return_value=mock_handle)

    request = create_run_task_request()
    task_id = mock_worker.submit_task(request)

    # Wait for task thread to reach RUNNING state
    task = mock_worker.get_task(task_id)
    wait_for_condition(lambda: task.status == job_pb2.TASK_STATE_RUNNING and task.container_id)

    result = mock_worker.kill_task(task_id, term_timeout_ms=100)
    assert result is True

    task.thread.join(timeout=15.0)

    assert task.status == job_pb2.TASK_STATE_KILLED
    assert any(c["force"] for c in mock_handle.stop_calls)


def test_new_attempt_supersedes_old(mock_worker, mock_runtime):
    """New attempt for same task_id kills the old attempt and starts a new one."""
    # Create a handle that stays running until killed
    mock_handle = create_mock_container_handle(
        status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100
    )  # Stay running
    mock_runtime.create_container = Mock(return_value=mock_handle)

    request_0 = create_run_task_request(task_id=JobName.root("test-user", "retry-task").task(0).to_wire(), attempt_id=0)
    mock_worker.submit_task(request_0)

    # Wait for attempt 0 to be running
    task_id = JobName.root("test-user", "retry-task").task(0).to_wire()
    old_task = mock_worker.get_task(task_id)
    wait_for_condition(lambda: old_task.status == job_pb2.TASK_STATE_RUNNING and old_task.container_id)
    assert old_task.attempt_id == 0

    # Submit attempt 1 for the same task_id — should kill attempt 0
    request_1 = create_run_task_request(task_id=JobName.root("test-user", "retry-task").task(0).to_wire(), attempt_id=1)
    mock_worker.submit_task(request_1)

    # Old attempt should have been killed
    assert old_task.should_stop is True

    # The new attempt should now be tracked with the new attempt_id
    new_task = mock_worker.get_task(task_id)
    assert new_task.attempt_id == 1
    assert new_task is not old_task

    # Clean up
    mock_worker.kill_task(task_id)
    new_task.thread.join(timeout=15.0)


def test_duplicate_attempt_rejected(mock_worker, mock_runtime):
    """Same attempt_id for an existing non-terminal task is rejected."""
    # Create a handle that stays running until killed
    mock_handle = create_mock_container_handle(
        status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100
    )  # Stay running
    mock_runtime.create_container = Mock(return_value=mock_handle)

    request = create_run_task_request(task_id=JobName.root("test-user", "dup-task").task(0).to_wire(), attempt_id=0)
    mock_worker.submit_task(request)

    # Wait for it to be running
    task_id = JobName.root("test-user", "dup-task").task(0).to_wire()
    task = mock_worker.get_task(task_id)
    wait_for_condition(lambda: task.status == job_pb2.TASK_STATE_RUNNING)

    # Submit same attempt_id again — should be rejected (task unchanged)
    mock_worker.submit_task(create_run_task_request(task_id=task_id, attempt_id=0))
    assert mock_worker.get_task(task_id) is task  # Same object, not replaced

    # Clean up
    mock_worker.kill_task(task_id)
    task.thread.join(timeout=15.0)


def test_heartbeat_kill_blocks_until_stopped(mock_worker, mock_runtime):
    """Heartbeat blocks on tasks_to_kill so the old task releases resources before returning."""
    mock_handle = create_mock_container_handle(status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 1000)

    def slow_stop(force=False):
        time.sleep(0.3)

    mock_handle.stop_hook = slow_stop
    mock_runtime.create_container = Mock(return_value=mock_handle)

    task_id_wire = JobName.root("test-user", "hb-kill-task").task(0).to_wire()
    request = create_run_task_request(task_id=task_id_wire)
    mock_worker.submit_task(request)

    # Wait for task to reach RUNNING
    task = mock_worker.get_task(task_id_wire)
    wait_for_condition(lambda: task.status == job_pb2.TASK_STATE_RUNNING)

    # Build a heartbeat that requests this task be killed.
    # Include the task in expected_tasks so the reconciliation path doesn't
    # also try to kill it -- we only want to exercise the tasks_to_kill path.
    heartbeat_req = job_pb2.HeartbeatRequest(
        tasks_to_kill=[task_id_wire],
        expected_tasks=[
            job_pb2.WorkerTaskStatus(task_id=task_id_wire, attempt_id=0),
        ],
    )

    response = mock_worker.handle_heartbeat(heartbeat_req)

    assert task.should_stop is True
    # The task is KILLED by the time handle_heartbeat returns — kill is blocking
    assert task.status == job_pb2.TASK_STATE_KILLED

    # The response should still include the task's current state
    assert len(response.tasks) >= 1


def test_heartbeat_reconciliation_kill_is_non_blocking(mock_worker, mock_runtime):
    """Tasks not in expected_tasks are killed asynchronously during heartbeat reconciliation."""
    mock_handle = create_mock_container_handle(status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 1000)

    def slow_stop(force=False):
        time.sleep(0.5)

    mock_handle.stop_hook = slow_stop
    mock_runtime.create_container = Mock(return_value=mock_handle)

    task_id_wire = JobName.root("test-user", "reconcile-kill").task(0).to_wire()
    request = create_run_task_request(task_id=task_id_wire)
    mock_worker.submit_task(request)

    task = mock_worker.get_task(task_id_wire)
    wait_for_condition(lambda: task.status == job_pb2.TASK_STATE_RUNNING)

    # Send heartbeat with empty expected_tasks -- the worker should kill
    # the running task because it's no longer expected
    heartbeat_req = job_pb2.HeartbeatRequest(expected_tasks=[])

    mock_worker.handle_heartbeat(heartbeat_req)

    # should_stop is set synchronously by the heartbeat before the async stop() runs
    assert task.should_stop is True
    # The task is not yet KILLED because slow_stop is async — confirms kill is non-blocking
    assert task.status != job_pb2.TASK_STATE_KILLED

    task.thread.join(timeout=15.0)
    assert task.status == job_pb2.TASK_STATE_KILLED


def test_kill_nonexistent_task(mock_worker):
    """Test killing a nonexistent task returns False."""
    result = mock_worker.kill_task(JobName.root("test-user", "nonexistent-task").task(0).to_wire())
    assert result is False


def test_port_env_vars_set(mock_worker, mock_runtime):
    """Test that IRIS_PORT_* environment variables are set for requested ports."""
    request = create_run_task_request(ports=["web", "api", "metrics"])
    task_id = mock_worker.submit_task(request)

    task = mock_worker.get_task(task_id)
    task.thread.join(timeout=15.0)

    assert mock_runtime.create_container.called
    call_args = mock_runtime.create_container.call_args
    config = call_args[0][0]

    assert "IRIS_PORT_WEB" in config.env
    assert "IRIS_PORT_API" in config.env
    assert "IRIS_PORT_METRICS" in config.env

    ports = {
        int(config.env["IRIS_PORT_WEB"]),
        int(config.env["IRIS_PORT_API"]),
        int(config.env["IRIS_PORT_METRICS"]),
    }
    assert len(ports) == 3


def test_env_merge_precedence(mock_bundle_store, mock_runtime, tmp_path):
    """Job-level env vars win over task_env, which wins over iris system vars.

    The merge order in _create_container is:
      1. iris system vars (IRIS_TASK_ID, etc.)
      2. task_env (worker-level defaults, overrides iris vars)
      3. job-level env_vars (from the request, wins over everything user-visible)

    This test verifies the observable precedence: job > default > absent.
    """
    config = WorkerConfig(
        port=0,
        port_range=(50000, 50100),
        poll_interval=Duration.from_seconds(0.1),
        cache_dir=tmp_path / "cache",
        default_task_image="mock-image",
        task_env={"SHARED_KEY": "default_value", "DEFAULT_ONLY": "from_default"},
    )
    w = Worker(config, bundle_store=mock_bundle_store, container_runtime=mock_runtime)

    # Build a request whose env_vars override SHARED_KEY but leave DEFAULT_ONLY untouched.
    def _fn():
        pass

    request = job_pb2.RunTaskRequest(
        task_id=JobName.root("test-user", "env-test").task(0).to_wire(),
        num_tasks=1,
        attempt_id=0,
        entrypoint=Entrypoint.from_callable(_fn).to_proto(),
        environment=job_pb2.EnvironmentConfig(
            env_vars={"SHARED_KEY": "job_value", "JOB_ONLY": "from_job"},
        ),
        bundle_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=512 * 1024**2),
    )

    task_id = w.submit_task(request)
    task = w.get_task(task_id)
    task.thread.join(timeout=15.0)

    assert mock_runtime.create_container.called
    env = mock_runtime.create_container.call_args[0][0].env

    # Job-level wins over task_env.
    assert env["SHARED_KEY"] == "job_value"
    # task_env key present when job doesn't override it.
    assert env["DEFAULT_ONLY"] == "from_default"
    # Job-only key propagates.
    assert env["JOB_ONLY"] == "from_job"
    # Iris system vars are always injected.
    assert "IRIS_TASK_ID" in env


def test_task_image_override_uses_request_value(mock_bundle_store, mock_runtime, tmp_path):
    """Per-task task_image overrides the worker's default_task_image."""
    config = WorkerConfig(
        port=0,
        port_range=(50000, 50100),
        poll_interval=Duration.from_seconds(0.1),
        cache_dir=tmp_path / "cache",
        default_task_image="default/cluster-image:latest",
    )
    w = Worker(config, bundle_store=mock_bundle_store, container_runtime=mock_runtime)

    request = create_run_task_request(task_image="custom/swetrace:dev")
    task_id = w.submit_task(request)
    task = w.get_task(task_id)
    task.thread.join(timeout=15.0)

    assert mock_runtime.create_container.called
    container_config = mock_runtime.create_container.call_args[0][0]
    assert container_config.image == "custom/swetrace:dev"


def test_task_image_default_used_when_override_empty(mock_bundle_store, mock_runtime, tmp_path):
    """Empty task_image falls back to the cluster default."""
    config = WorkerConfig(
        port=0,
        port_range=(50000, 50100),
        poll_interval=Duration.from_seconds(0.1),
        cache_dir=tmp_path / "cache",
        default_task_image="default/cluster-image:latest",
    )
    w = Worker(config, bundle_store=mock_bundle_store, container_runtime=mock_runtime)

    request = create_run_task_request()  # task_image="" by default
    task_id = w.submit_task(request)
    task = w.get_task(task_id)
    task.thread.join(timeout=15.0)

    assert mock_runtime.create_container.called
    container_config = mock_runtime.create_container.call_args[0][0]
    assert container_config.image == "default/cluster-image:latest"


def test_port_binding_failure(mock_bundle_store, tmp_path):
    """Test that task fails when port binding fails.

    With --network=host, port binding happens in the application, not Docker.
    If the app fails to bind (port in use by external process), the task fails.
    """
    runtime = Mock(spec=DockerRuntime)

    mock_handle = create_mock_container_handle(
        run_side_effect=RuntimeError("failed to bind host port: address already in use")
    )
    runtime.create_container = Mock(return_value=mock_handle)
    runtime.cleanup = Mock()

    config = WorkerConfig(
        port=0,
        port_range=(50000, 50100),
        poll_interval=Duration.from_seconds(0.1),
        cache_dir=tmp_path / "cache",
        default_task_image="mock-image",
    )
    worker = Worker(
        config,
        bundle_store=mock_bundle_store,
        container_runtime=runtime,
    )

    request = create_run_task_request(ports=["actor"])
    task_id = worker.submit_task(request)

    task = worker.get_task(task_id)
    assert task is not None
    assert task.thread is not None
    task.thread.join(timeout=15.0)

    final_task = worker.get_task(task_id)
    assert final_task is not None
    assert final_task.status == job_pb2.TASK_STATE_FAILED
    assert final_task.error is not None
    assert "address already in use" in final_task.error


# ============================================================================
# Remote log handler attach tests (regression for #4794)
# ============================================================================


def _log_server_endpoints(address: str):
    from iris.rpc import controller_pb2

    return controller_pb2.Controller.ListEndpointsResponse(
        endpoints=[
            controller_pb2.Controller.Endpoint(
                endpoint_id="/system/log-server",
                name="/system/log-server",
                address=address,
            )
        ]
    )


def test_attach_log_handler_uses_worker_log_key_before_register(mock_bundle_store, mock_runtime, tmp_path):
    """Worker known locally (e.g. via slice_id) attaches under worker_log_key
    *before* register so pre-register failures ship remote logs."""
    from iris.cluster.log_store import worker_log_key

    config = WorkerConfig(
        port=0,
        port_range=(50000, 50100),
        cache_dir=tmp_path / "cache",
        default_task_image="mock-image",
        worker_id="w-1",
    )
    worker = Worker(config, bundle_store=mock_bundle_store, container_runtime=mock_runtime)
    worker._controller_client = Mock(list_endpoints=Mock(return_value=_log_server_endpoints("http://log:9000")))

    try:
        worker._attach_log_handler()
        assert worker._log_handler is not None
        assert worker._log_handler.key == worker_log_key("w-1")
    finally:
        worker._detach_log_handler()


def test_attach_log_handler_defers_resolution_to_pusher(mock_bundle_store, mock_runtime, tmp_path):
    """A ListEndpoints RPC failure at attach time must not crash the lifecycle
    thread. Resolution happens lazily inside LogPusher on the first push, so
    the handler is attached even if the controller is not yet reachable."""
    config = WorkerConfig(
        port=0,
        port_range=(50000, 50100),
        cache_dir=tmp_path / "cache",
        default_task_image="mock-image",
        worker_id="w-1",
    )
    worker = Worker(config, bundle_store=mock_bundle_store, container_runtime=mock_runtime)
    worker._controller_client = Mock(list_endpoints=Mock(side_effect=ConnectionError("controller down")))

    try:
        worker._attach_log_handler()
        # Handler attached; pusher exists but has no cached RPC client yet.
        assert worker._log_handler is not None
        assert worker._log_pusher is not None
        # The controller mock was NOT called at attach time — resolution
        # is deferred to the first send.
        assert worker._controller_client.list_endpoints.call_count == 0
    finally:
        worker._detach_log_handler()


def test_attach_log_handler_idempotent_renames_key(mock_bundle_store, mock_runtime, tmp_path):
    """Re-attach under a new worker_id renames the handler's key in place
    rather than rebuilding the LogPusher — the pusher owns its own
    resolve-on-failure loop across lifecycle iterations."""
    from iris.cluster.log_store import worker_log_key

    config = WorkerConfig(
        port=0,
        port_range=(50000, 50100),
        cache_dir=tmp_path / "cache",
        default_task_image="mock-image",
        worker_id="w-1",
    )
    worker = Worker(config, bundle_store=mock_bundle_store, container_runtime=mock_runtime)
    worker._controller_client = Mock(list_endpoints=Mock(return_value=_log_server_endpoints("http://log:9000")))

    try:
        worker._attach_log_handler()
        first_pusher = worker._log_pusher
        first_handler = worker._log_handler
        assert first_pusher is not None
        assert first_handler is not None

        # Re-attach with same id → identity preserved.
        worker._attach_log_handler()
        assert worker._log_pusher is first_pusher
        assert worker._log_handler is first_handler

        # Re-attach under a different worker_id → same pusher, same handler,
        # renamed key.
        worker._worker_id = "w-2"
        worker._attach_log_handler()
        assert worker._log_pusher is first_pusher
        assert worker._log_handler is first_handler
        assert first_handler.key == worker_log_key("w-2")
    finally:
        worker._detach_log_handler()


# ============================================================================
# Integration Tests (with real Docker)
# ============================================================================


def create_test_bundle(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    (bundle_dir / "pyproject.toml").write_text(
        """[project]
name = "test-task"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []
"""
    )

    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in bundle_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(bundle_dir))

    bundle_bytes = zip_path.read_bytes()
    return hashlib.sha256(bundle_bytes).hexdigest(), zip_path


def create_integration_entrypoint():
    def test_fn():
        print("Hello from test task!")
        return 42

    return Entrypoint.from_callable(test_fn)


def create_integration_run_task_request(bundle_id: str, task_id: str):
    entrypoint = create_integration_entrypoint()

    return job_pb2.RunTaskRequest(
        task_id=task_id,
        num_tasks=1,
        entrypoint=entrypoint.to_proto(),
        bundle_id=bundle_id,
        environment=job_pb2.EnvironmentConfig(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=512 * 1024**2),
    )


@pytest.fixture
def cache_dir(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    return cache


@pytest.fixture
def test_bundle(tmp_path):
    return create_test_bundle(tmp_path)


@pytest.fixture
def real_worker(cache_dir):
    runtime = DockerRuntime(cache_dir=cache_dir)
    config = WorkerConfig(
        port=0,
        cache_dir=cache_dir,
        port_range=(40000, 40100),
        poll_interval=Duration.from_seconds(0.5),  # Faster polling for tests
        default_task_image="iris-task:latest",
    )
    worker = Worker(config, container_runtime=runtime)
    yield worker
    worker.stop()
    runtime.cleanup()


@pytest.fixture
def real_service(real_worker):
    return WorkerServiceImpl(real_worker)


class TestWorkerIntegration:
    """Integration tests for Worker with real components."""

    @pytest.mark.docker
    def test_submit_task_lifecycle(self, real_worker, test_bundle, cache_dir):
        bundle_id, bundle_zip_path = test_bundle
        bundle_store_zip = cache_dir / "bundles" / f"{bundle_id}.zip"
        bundle_store_zip.parent.mkdir(parents=True, exist_ok=True)
        bundle_store_zip.write_bytes(bundle_zip_path.read_bytes())

        expected_task_id = JobName.root("test-user", "integration-test").task(0).to_wire()
        request = create_integration_run_task_request(bundle_id, expected_task_id)

        task_id = real_worker.submit_task(request)
        assert task_id == expected_task_id

        # Poll for task completion with shorter intervals
        deadline = time.time() + 30.0
        while time.time() < deadline:
            task = real_worker.get_task(task_id)
            if task.status in (
                job_pb2.TASK_STATE_SUCCEEDED,
                job_pb2.TASK_STATE_FAILED,
                job_pb2.TASK_STATE_KILLED,
            ):
                break
            time.sleep(0.5)

        task = real_worker.get_task(task_id)
        assert task.status in (
            job_pb2.TASK_STATE_SUCCEEDED,
            job_pb2.TASK_STATE_FAILED,
        ), f"Task did not complete in time, final status: {task.status}"


class TestWorkerServiceIntegration:
    """Integration tests for WorkerService RPC implementation."""

    @pytest.mark.docker
    def test_health_check_rpc(self, real_service):
        ctx = Mock(spec=RequestContext)

        response = real_service.health_check(job_pb2.Empty(), ctx)

        assert response.healthy
        assert response.uptime.milliseconds >= 0


# ============================================================================
# Container Adoption Tests
# ============================================================================


def _make_discovered_container(
    task_id: str = JobName.root("test-user", "test-job").task(0).to_wire(),
    attempt_id: int = 0,
    worker_id: str = "",
    phase: ExecutionStage = ExecutionStage.RUN,
    running: bool = True,
    workdir_host_path: str = "/tmp/workdirs/test",
) -> DiscoveredContainer:
    return DiscoveredContainer(
        container_id="abc123def456",
        task_id=task_id,
        attempt_id=attempt_id,
        job_id=JobName.root("test-user", "test-job").to_wire(),
        worker_id=worker_id,
        phase=phase,
        running=running,
        exit_code=None if running else 0,
        started_at="2025-01-01T00:00:00Z",
        workdir_host_path=workdir_host_path,
    )


def test_adopt_creates_task_in_running_state(mock_worker, mock_runtime):
    """Adoption creates a TaskAttempt in RUNNING state."""
    container = _make_discovered_container()
    mock_runtime.discover_containers = Mock(return_value=[container])

    adopted = mock_worker.adopt_running_containers()

    assert adopted == 1
    task = mock_worker.get_task(container.task_id, container.attempt_id)
    assert task is not None
    assert task.status == job_pb2.TASK_STATE_RUNNING


def test_adopt_skips_build_phase_containers(mock_worker, mock_runtime):
    """Build-phase containers should be cleaned up, not adopted."""
    container = _make_discovered_container(phase=ExecutionStage.BUILD)
    mock_runtime.discover_containers = Mock(return_value=[container])

    adopted = mock_worker.adopt_running_containers()

    assert adopted == 0


def test_adopt_skips_exited_containers(mock_worker, mock_runtime):
    """Exited containers should be cleaned up, not adopted."""
    container = _make_discovered_container(running=False)
    mock_runtime.discover_containers = Mock(return_value=[container])

    adopted = mock_worker.adopt_running_containers()

    assert adopted == 0


def test_adopt_skips_wrong_worker_id(mock_worker, mock_runtime, tmp_path):
    """Containers from a different worker should be cleaned up."""
    # Set the mock_worker's worker_id
    mock_worker._worker_id = "worker-1"
    container = _make_discovered_container(worker_id="worker-2")
    mock_runtime.discover_containers = Mock(return_value=[container])

    adopted = mock_worker.adopt_running_containers()

    assert adopted == 0


def test_adopt_accepts_matching_worker_id(mock_worker, mock_runtime):
    """Containers from the same worker should be adopted."""
    mock_worker._worker_id = "worker-1"
    container = _make_discovered_container(worker_id="worker-1")
    mock_runtime.discover_containers = Mock(return_value=[container])

    adopted = mock_worker.adopt_running_containers()

    assert adopted == 1


def test_heartbeat_after_adoption_reports_running(mock_worker, mock_runtime):
    """After adoption, heartbeat reconciliation should report the task as RUNNING."""
    container = _make_discovered_container()
    mock_runtime.discover_containers = Mock(return_value=[container])
    mock_worker.adopt_running_containers()

    heartbeat_req = job_pb2.HeartbeatRequest(
        expected_tasks=[
            job_pb2.WorkerTaskStatus(
                task_id=container.task_id,
                attempt_id=container.attempt_id,
            )
        ],
    )
    response = mock_worker.handle_heartbeat(heartbeat_req)

    assert len(response.tasks) == 1
    task_status = response.tasks[0]
    assert task_status.task_id == container.task_id
    assert task_status.state == job_pb2.TASK_STATE_RUNNING


def test_heartbeat_without_adoption_reports_worker_failed(mock_worker, mock_runtime):
    """Without adoption, expected tasks should report WORKER_FAILED."""
    mock_runtime.discover_containers = Mock(return_value=[])

    heartbeat_req = job_pb2.HeartbeatRequest(
        expected_tasks=[
            job_pb2.WorkerTaskStatus(
                task_id=JobName.root("test-user", "test-job").task(0).to_wire(),
                attempt_id=0,
            )
        ],
    )
    response = mock_worker.handle_heartbeat(heartbeat_req)

    assert len(response.tasks) == 1
    assert response.tasks[0].state == job_pb2.TASK_STATE_WORKER_FAILED


def test_stop_preserve_containers_does_not_kill_tasks(mock_worker, mock_runtime):
    """stop(preserve_containers=True) should not kill running tasks."""
    container = _make_discovered_container()
    # Use a handle that stays RUNNING indefinitely so the monitor thread
    # doesn't exit before we call stop().
    always_running = [ContainerStatus(phase=ContainerPhase.RUNNING)] * 1000
    mock_runtime.discover_containers = Mock(return_value=[container])
    mock_runtime.adopt_container = Mock(
        side_effect=lambda cid: create_mock_container_handle(status_sequence=always_running)
    )
    mock_worker.adopt_running_containers()

    # Give the monitoring thread time to start
    time.sleep(0.2)

    task = mock_worker.get_task(container.task_id, container.attempt_id)
    assert task is not None
    assert task.status == job_pb2.TASK_STATE_RUNNING

    mock_worker.stop(preserve_containers=True)
    # The task should still be in RUNNING state (not KILLED)
    assert task.status == job_pb2.TASK_STATE_RUNNING


def test_task_attempt_adopt_factory():
    """TaskAttempt.adopt() creates a properly initialized attempt."""
    port_allocator = PortAllocator(port_range=(50000, 50100))
    container = _make_discovered_container()
    handle = create_mock_container_handle()

    attempt = TaskAttempt.adopt(
        discovered=container,
        container_handle=handle,
        log_pusher=None,
        port_allocator=port_allocator,
    )

    assert attempt.status == job_pb2.TASK_STATE_RUNNING
    assert attempt.task_id == JobName.from_wire(container.task_id)
    assert attempt.attempt_id == container.attempt_id
    assert attempt.container_id == "container123"
    assert attempt.has_container
    assert attempt.error is None
    assert attempt.exit_code is None

    # to_proto should work
    proto = attempt.to_proto()
    assert proto.state == job_pb2.TASK_STATE_RUNNING
    assert proto.current_attempt_id == container.attempt_id


# ============================================================================
# Docker-based Adoption Integration Tests
# ============================================================================


@pytest.mark.docker
def test_docker_container_has_adoption_labels(docker_runtime, tmp_path):
    """Containers created by DockerRuntime should have adoption labels."""
    import subprocess as sp

    from iris.cluster.runtime.types import ContainerConfig, MountKind, MountSpec

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    config = ContainerConfig(
        image="iris-task:latest",
        entrypoint=job_pb2.RuntimeEntrypoint(
            run_command=job_pb2.CommandEntrypoint(argv=["echo", "hello"]),
        ),
        env={},
        mounts=[MountSpec("/app", kind=MountKind.WORKDIR)],
        workdir_host_path=workdir,
        task_id="/test-user/test-job/0",
        attempt_id=3,
        job_id="/test-user/test-job",
        worker_id="worker-42",
    )

    handle = docker_runtime.create_container(config)
    try:
        handle.run()

        # Inspect the container's labels
        cid = handle.container_id
        result = sp.run(
            ["docker", "inspect", "--format", "{{json .Config.Labels}}", cid],
            capture_output=True,
            text=True,
            check=True,
        )

        import json

        labels = json.loads(result.stdout)
        assert labels["iris.managed"] == "true"
        assert labels["iris.task_id"] == "/test-user/test-job/0"
        assert labels["iris.attempt_id"] == "3"
        assert labels["iris.worker_id"] == "worker-42"
        assert labels["iris.phase"] == "run"
        assert labels["iris.job_id"] == "/test-user/test-job"
    finally:
        handle.cleanup()


@pytest.mark.docker
def test_docker_discover_containers(docker_runtime, tmp_path):
    """discover_containers() should find iris-managed containers."""
    from iris.cluster.runtime.types import ContainerConfig, MountKind, MountSpec

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    config = ContainerConfig(
        image="iris-task:latest",
        entrypoint=job_pb2.RuntimeEntrypoint(
            run_command=job_pb2.CommandEntrypoint(argv=["sleep", "60"]),
        ),
        env={},
        mounts=[MountSpec("/app", kind=MountKind.WORKDIR)],
        workdir_host_path=workdir,
        task_id="/test-user/discover-job/0",
        attempt_id=5,
        job_id="/test-user/discover-job",
        worker_id="worker-99",
    )

    handle = docker_runtime.create_container(config)
    try:
        handle.run()

        discovered = docker_runtime.discover_containers()
        matching = [d for d in discovered if d.task_id == "/test-user/discover-job/0"]
        assert len(matching) == 1

        d = matching[0]
        assert d.attempt_id == 5
        assert d.worker_id == "worker-99"
        assert d.phase == "run"
        assert d.running is True
        assert d.workdir_host_path == str(workdir)
    finally:
        handle.cleanup()


@pytest.mark.docker
def test_docker_adopt_container(docker_runtime, tmp_path):
    """adopt_container() should wrap an existing container."""
    from iris.cluster.runtime.types import ContainerConfig, ContainerPhase, MountKind, MountSpec

    workdir = tmp_path / "workdir"
    workdir.mkdir()

    config = ContainerConfig(
        image="iris-task:latest",
        entrypoint=job_pb2.RuntimeEntrypoint(
            run_command=job_pb2.CommandEntrypoint(argv=["sleep", "60"]),
        ),
        env={},
        mounts=[MountSpec("/app", kind=MountKind.WORKDIR)],
        workdir_host_path=workdir,
        task_id="/test-user/adopt-job/0",
        attempt_id=0,
        job_id="/test-user/adopt-job",
    )

    handle = docker_runtime.create_container(config)
    try:
        handle.run()
        cid = handle.container_id

        # Adopt the container via a new handle
        adopted_handle = docker_runtime.adopt_container(cid)
        status = adopted_handle.status()
        assert status.phase == ContainerPhase.RUNNING
        assert adopted_handle.container_id == cid
    finally:
        handle.cleanup()
