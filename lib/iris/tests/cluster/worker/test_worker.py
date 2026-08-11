# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Worker class (includes PortAllocator and task management)."""

import hashlib
import json
import socket
import subprocess as sp
import threading
import zipfile
from typing import cast
from unittest.mock import Mock

import pytest
from connectrpc.request import RequestContext
from finelog.client import LogClient
from finelog.rpc import logging_pb2
from iris.cluster.log_keys import worker_log_key
from iris.cluster.runtime.docker import DockerRuntime
from iris.cluster.runtime.types import (
    ContainerConfig,
    ContainerErrorKind,
    ContainerInfraError,
    ContainerPhase,
    ContainerStatus,
    DiscoveredContainer,
    ExecutionStage,
    MountKind,
    MountSpec,
)
from iris.cluster.stats.tables import TASK_STATS_NAMESPACE, WORKER_STATS_NAMESPACE, IrisTaskStat, IrisWorkerStat
from iris.cluster.types import Entrypoint, JobName
from iris.cluster.worker.port_allocator import PortAllocator
from iris.cluster.worker.service import WorkerServiceImpl
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.cluster.worker.worker_types import LogLine
from iris.managed_thread import ThreadContainer
from iris.rpc import controller_pb2, job_pb2, worker_pb2
from iris.test_util import wait_for_condition
from rigging.timing import Duration
from tests.cluster.worker.conftest import (
    FakeContainerHandle,
    FakeLogReader,
    create_mock_container_handle,
    create_run_task_request,
)

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


def test_task_exception_handling(mock_worker, mock_runtime):
    """Test task handles exceptions during execution."""
    mock_runtime.stage_bundle = Mock(side_effect=Exception("Bundle download failed"))

    request = create_run_task_request()
    task_id = mock_worker.submit_task(request)

    task = mock_worker.get_task(task_id)
    task.thread.join(timeout=15.0)

    final_task = mock_worker.get_task(task_id)
    assert final_task.status == job_pb2.TASK_STATE_FAILED
    assert "Bundle download failed" in final_task.error


def test_list_tasks_with_submitted_tasks_returns_live_task_set(mock_worker):
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


def test_resubmit_same_composite_fresh_uid_is_distinct_attempt(mock_worker, mock_runtime):
    """A resubmit of (task_id, attempt_id=0) with a fresh UID is a distinct attempt.

    Regression: the worker retains the terminal attempt for log access. A
    re-submitted composite carrying a *new* UID is a new incarnation and must
    run, not be rejected as a duplicate of the retained terminal attempt.
    """
    # Container exits immediately so the first attempt becomes terminal.
    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.STOPPED, exit_code=0)],
        )
    )
    task_id = JobName.root("test-user", "resubmit-task").task(0).to_wire()

    mock_worker.submit_task(create_run_task_request(task_id=task_id, attempt_id=0, attempt_uid="uid-first"))
    first = mock_worker.task_by_uid("uid-first")
    assert first is not None
    first.thread.join(timeout=15.0)
    assert first.status == job_pb2.TASK_STATE_SUCCEEDED

    # Resubmit the same (task_id, attempt_id=0) with a fresh UID.
    mock_worker.submit_task(create_run_task_request(task_id=task_id, attempt_id=0, attempt_uid="uid-second"))

    second = mock_worker.task_by_uid("uid-second")
    assert second is not None, "Resubmit with a fresh UID must produce a new attempt"
    assert second is not first, "New incarnation must be a distinct TaskAttempt"
    # Both incarnations coexist in the public worker task listing.
    assert {task.attempt_uid for task in mock_worker.list_tasks()} == {"uid-first", "uid-second"}

    if second.thread:
        second.thread.join(timeout=15.0)


def _terminal_and_live_twins(mock_worker, mock_runtime):
    """Submit two attempts sharing (task_id, attempt_id=0), distinguished by UID.

    The first attempt's container exits immediately (terminal, retained for
    log access); the second stays running. Returns ``(task_id, terminal, live)``.
    The terminal twin is appended to ``_tasks`` first — the order in which the
    composite-lookup bug surfaces.
    """
    terminal_handle = create_mock_container_handle(
        status_sequence=[ContainerStatus(phase=ContainerPhase.STOPPED, exit_code=0)],
    )
    live_handle = create_mock_container_handle(
        status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 1000,
    )
    # Respond to SIGTERM promptly so kill() need not wait out the term timeout.
    live_handle.stop_hook = lambda force: setattr(live_handle, "_killed", True)
    mock_runtime.create_container = Mock(side_effect=[terminal_handle, live_handle])

    task_id = JobName.root("test-user", "twin-task").task(0).to_wire()
    mock_worker.submit_task(create_run_task_request(task_id=task_id, attempt_id=0, attempt_uid="uid-terminal"))
    terminal = mock_worker.task_by_uid("uid-terminal")
    assert terminal is not None
    terminal.thread.join(timeout=15.0)
    assert terminal.status == job_pb2.TASK_STATE_SUCCEEDED

    mock_worker.submit_task(create_run_task_request(task_id=task_id, attempt_id=0, attempt_uid="uid-live"))
    live = mock_worker.task_by_uid("uid-live")
    assert live is not None and live is not terminal
    wait_for_condition(lambda: live.status == job_pb2.TASK_STATE_RUNNING)
    return task_id, terminal, live


def test_composite_lookups_prefer_live_twin(mock_worker, mock_runtime):
    """Composite resolution returns the live attempt, not a retained terminal twin.

    Regression (#5862 review): task_by_attempt / current_attempt returned the
    first matching list entry — the terminal attempt appended first — so kill
    and status paths acted on the wrong incarnation.
    """
    task_id, _terminal, live = _terminal_and_live_twins(mock_worker, mock_runtime)

    assert mock_worker.task_by_attempt(task_id, 0) is live
    assert mock_worker.current_attempt(task_id) is live
    assert mock_worker.get_task(task_id, attempt_id=0) is live

    mock_worker.kill_task(task_id)
    live.thread.join(timeout=15.0)


def test_stop_intent_by_uid_kills_live_twin(mock_worker, mock_runtime):
    """A reconcile stop intent routed by the live attempt's UID kills that
    attempt, leaving the retained terminal twin untouched.

    Regression (#5862 review by yonromai): _process_stop_intent resolved the
    live attempt by UID, then re-resolved by composite when killing — landing
    on the terminal twin, so the live attempt kept running.
    """
    _task_id, terminal, live = _terminal_and_live_twins(mock_worker, mock_runtime)

    mock_worker.handle_reconcile(
        worker_pb2.Worker.ReconcileRequest(
            desired=[
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid="uid-live",
                    stop=worker_pb2.Worker.STOP_REASON_CANCELLED,
                )
            ]
        )
    )

    wait_for_condition(lambda: live.status == job_pb2.TASK_STATE_KILLED)
    live.thread.join(timeout=15.0)
    assert terminal.status == job_pb2.TASK_STATE_SUCCEEDED


def test_resubmit_same_uid_is_rejected_as_duplicate(mock_worker, mock_runtime):
    """A resubmit carrying the *same* UID is still rejected as a true duplicate."""
    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
        )
    )
    task_id = JobName.root("test-user", "dup-uid-task").task(0).to_wire()

    mock_worker.submit_task(create_run_task_request(task_id=task_id, attempt_id=0, attempt_uid="uid-dup"))
    task = mock_worker.task_by_uid("uid-dup")
    assert task is not None
    wait_for_condition(lambda: task.status == job_pb2.TASK_STATE_RUNNING)

    # Resubmit with the identical UID — must be rejected, public list unchanged.
    mock_worker.submit_task(create_run_task_request(task_id=task_id, attempt_id=0, attempt_uid="uid-dup"))
    assert [candidate.attempt_uid for candidate in mock_worker.list_tasks()] == ["uid-dup"]

    mock_worker.kill_task(task_id)
    task.thread.join(timeout=15.0)


def test_task_by_uid_and_attempt_resolution(mock_worker, mock_runtime):
    """task_by_uid / task_by_attempt resolve correctly; empty UID resolves to None."""
    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
        )
    )
    task_id = JobName.root("test-user", "resolve-task").task(0).to_wire()
    mock_worker.submit_task(create_run_task_request(task_id=task_id, attempt_id=0, attempt_uid="uid-resolve"))

    task = mock_worker.task_by_uid("uid-resolve")
    assert task is not None
    assert mock_worker.task_by_attempt(task_id, 0) is task
    # An empty UID never identifies an attempt, even though one is tracked.
    assert mock_worker.task_by_uid("") is None
    # An unknown UID / composite resolves to None.
    assert mock_worker.task_by_uid("uid-nope") is None
    assert mock_worker.task_by_attempt(task_id, 99) is None

    mock_worker.kill_task(task_id)
    task.thread.join(timeout=15.0)


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
        attempt_uid="uid-env-test",
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
# Worker telemetry tests
# ============================================================================


class _RecordingStatsTable:
    def __init__(self, write_error: Exception | None = None):
        self.rows: list[object] = []
        self.write_error = write_error

    def write(self, rows) -> None:
        self.rows.extend(rows)
        if self.write_error is not None:
            raise self.write_error


class _RecordingLogClient:
    """In-memory finelog boundary used to observe worker emissions."""

    def __init__(self, table_errors: dict[str, Exception] | None = None):
        errors = table_errors or {}
        self.tables: dict[str, _RecordingStatsTable] = {
            WORKER_STATS_NAMESPACE: _RecordingStatsTable(errors.get(WORKER_STATS_NAMESPACE)),
            TASK_STATS_NAMESPACE: _RecordingStatsTable(errors.get(TASK_STATS_NAMESPACE)),
        }
        self.log_batches: list[tuple[str, list[logging_pb2.LogEntry]]] = []

    def get_table(self, namespace, _schema):
        return self.tables.setdefault(namespace, _RecordingStatsTable())

    def write_batch(self, key: str, entries) -> None:
        self.log_batches.append((key, list(entries)))

    def flush(self, timeout=None):
        return True

    def close(self) -> None:
        pass

    def rows(self, namespace: str) -> list[object]:
        return list(self.tables[namespace].rows)

    def log_lines(self, key: str) -> list[logging_pb2.LogEntry]:
        return [entry for batch_key, entries in self.log_batches if batch_key == key for entry in entries]


def _worker_with_log_sink(
    mock_bundle_store,
    mock_runtime,
    tmp_path,
    *,
    worker_id: str = "w-test",
    table_errors: dict[str, Exception] | None = None,
) -> tuple[Worker, _RecordingLogClient]:
    sink = _RecordingLogClient(table_errors)
    worker = Worker(
        WorkerConfig(
            port=0,
            port_range=(50000, 50100),
            poll_interval=Duration.from_seconds(0.01),
            cache_dir=tmp_path / "cache",
            default_task_image="mock-image",
            worker_id=worker_id,
        ),
        bundle_store=mock_bundle_store,
        container_runtime=mock_runtime,
        log_client=cast(LogClient, sink),
        threads=ThreadContainer(name=f"worker-{worker_id}"),
    )
    return worker, sink


def test_start_publishes_worker_logs_before_controller_registration(
    mock_bundle_store, mock_runtime, tmp_path, monkeypatch
):
    sink = _RecordingLogClient()

    class _ControllerBoundary:
        def __init__(self):
            self.registration_seen = threading.Event()
            self.worker_log_visible = False

        def register(self, request):
            self.worker_log_visible = bool(sink.log_lines(worker_log_key("worker-log-test")))
            self.registration_seen.set()
            return controller_pb2.Controller.RegisterResponse(accepted=True, worker_id=request.worker_id)

        def close(self) -> None:
            pass

    class _EndpointBoundary:
        def close(self) -> None:
            pass

    controller = _ControllerBoundary()
    monkeypatch.setattr("iris.cluster.worker.worker.ControllerServiceClientSync", lambda **_kwargs: controller)
    monkeypatch.setattr("iris.cluster.worker.worker.EndpointServiceClientSync", lambda **_kwargs: _EndpointBoundary())
    worker = Worker(
        WorkerConfig(
            port=0,
            port_range=(50000, 50100),
            cache_dir=tmp_path / "cache",
            default_task_image="mock-image",
            controller_address="http://controller.test",
            worker_id="worker-log-test",
        ),
        bundle_store=mock_bundle_store,
        container_runtime=mock_runtime,
        log_client=cast(LogClient, sink),
        threads=ThreadContainer(name="worker-log-test"),
    )

    worker.start()
    wait_for_condition(controller.registration_seen.is_set)

    assert controller.worker_log_visible
    worker.stop()


def test_handle_reconcile_publishes_worker_stat(mock_bundle_store, mock_runtime, tmp_path):
    worker, sink = _worker_with_log_sink(mock_bundle_store, mock_runtime, tmp_path)

    worker.handle_reconcile(worker_pb2.Worker.ReconcileRequest())

    rows = sink.rows(WORKER_STATS_NAMESPACE)
    assert len(rows) == 1
    stat = rows[0]
    assert isinstance(stat, IrisWorkerStat)
    assert stat.worker_id == "w-test"
    assert stat.mem_total_bytes >= 0
    assert stat.cpu_pct >= 0.0
    worker.stop()


def test_handle_reconcile_when_stats_sink_rejects_row_propagates_error(mock_bundle_store, mock_runtime, tmp_path):
    worker, _sink = _worker_with_log_sink(
        mock_bundle_store,
        mock_runtime,
        tmp_path,
        table_errors={WORKER_STATS_NAMESPACE: TypeError("schema mismatch")},
    )

    with pytest.raises(TypeError, match="schema mismatch"):
        worker.handle_reconcile(worker_pb2.Worker.ReconcileRequest())
    worker.stop()


def test_handle_reconcile_without_log_sink_returns_health(mock_worker):
    response = mock_worker.handle_reconcile(worker_pb2.Worker.ReconcileRequest())
    assert response.health.healthy


def test_task_resource_poll_publishes_task_stat(mock_bundle_store, mock_runtime, tmp_path):
    worker, sink = _worker_with_log_sink(mock_bundle_store, mock_runtime, tmp_path)
    request = create_run_task_request()
    task_id = worker.submit_task(request)
    task = worker.get_task(task_id)
    assert task is not None
    task.thread.join(timeout=15.0)
    final = worker.get_task(task_id)
    assert final is not None
    assert final.status == job_pb2.TASK_STATE_SUCCEEDED

    rows = sink.rows(TASK_STATS_NAMESPACE)
    assert rows
    stat = rows[0]
    assert isinstance(stat, IrisTaskStat)
    assert stat.task_id == request.task_id
    assert stat.attempt_id == request.attempt_id
    assert stat.worker_id == "w-test"
    worker.stop()


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
requires-python = ">=3.12"
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

        wait_for_condition(
            lambda: real_worker.get_task(task_id).status
            in (
                job_pb2.TASK_STATE_SUCCEEDED,
                job_pb2.TASK_STATE_FAILED,
                job_pb2.TASK_STATE_KILLED,
            ),
            timeout=Duration.from_seconds(30.0),
        )

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
    attempt_uid: str = "",
    worker_id: str = "",
    phase: ExecutionStage = ExecutionStage.RUN,
    running: bool = True,
    workdir_host_path: str = "/tmp/workdirs/test",
    ports: dict[str, int] | None = None,
) -> DiscoveredContainer:
    return DiscoveredContainer(
        container_id="abc123def456",
        task_id=task_id,
        attempt_id=attempt_id,
        attempt_uid=attempt_uid,
        job_id=JobName.root("test-user", "test-job").to_wire(),
        worker_id=worker_id,
        phase=phase,
        running=running,
        exit_code=None if running else 0,
        started_at="2025-01-01T00:00:00Z",
        workdir_host_path=workdir_host_path,
        ports=ports or {},
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


def test_adopt_skips_wrong_worker_id(mock_bundle_store, mock_runtime, tmp_path):
    """Containers from a different worker should be cleaned up."""
    worker = Worker(
        WorkerConfig(
            port=0,
            port_range=(50000, 50100),
            cache_dir=tmp_path / "cache",
            default_task_image="mock-image",
            worker_id="worker-1",
        ),
        bundle_store=mock_bundle_store,
        container_runtime=mock_runtime,
    )
    container = _make_discovered_container(worker_id="worker-2")
    mock_runtime.discover_containers = Mock(return_value=[container])

    adopted = worker.adopt_running_containers()

    assert adopted == 0


def test_adopt_accepts_matching_worker_id(mock_bundle_store, mock_runtime, tmp_path):
    """Containers from the same worker should be adopted."""
    worker = Worker(
        WorkerConfig(
            port=0,
            port_range=(50000, 50100),
            cache_dir=tmp_path / "cache",
            default_task_image="mock-image",
            worker_id="worker-1",
        ),
        bundle_store=mock_bundle_store,
        container_runtime=mock_runtime,
        threads=ThreadContainer(name="matching-worker"),
    )
    container = _make_discovered_container(worker_id="worker-1")
    mock_runtime.discover_containers = Mock(return_value=[container])

    adopted = worker.adopt_running_containers()

    assert adopted == 1
    task = worker.get_task(container.task_id, container.attempt_id)
    assert task is not None
    wait_for_condition(lambda: task.status == job_pb2.TASK_STATE_SUCCEEDED)
    worker.stop()


def test_adopt_with_uid_label_carries_uid(mock_worker, mock_runtime):
    """A discovered container WITH an iris.attempt_uid label yields an attempt carrying that UID."""
    container = _make_discovered_container(attempt_uid="uid-adopted")
    mock_runtime.discover_containers = Mock(return_value=[container])

    adopted = mock_worker.adopt_running_containers()

    assert adopted == 1
    task = mock_worker.task_by_uid("uid-adopted")
    assert task is not None
    assert task.attempt_uid == "uid-adopted"


def test_adopt_without_uid_label_has_empty_uid(mock_worker, mock_runtime):
    """A discovered container WITHOUT the label yields an attempt with an empty UID."""
    container = _make_discovered_container(attempt_uid="")
    mock_runtime.discover_containers = Mock(return_value=[container])

    adopted = mock_worker.adopt_running_containers()

    assert adopted == 1
    task = mock_worker.get_task(container.task_id, container.attempt_id)
    assert task is not None
    assert task.attempt_uid == ""
    # An empty UID never resolves via task_by_uid.
    assert mock_worker.task_by_uid("") is None


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

    task = mock_worker.get_task(container.task_id, container.attempt_id)
    assert task is not None
    wait_for_condition(lambda: task.status == job_pb2.TASK_STATE_RUNNING)

    mock_worker.stop(preserve_containers=True)
    # The task should still be in RUNNING state (not KILLED)
    assert task.status == job_pb2.TASK_STATE_RUNNING
    # This process remains responsible for its fake monitoring thread after the
    # preservation assertion; a normal stop provides deterministic test cleanup.
    mock_worker.stop()


def test_adopted_attempt_publishes_logs_and_stats(mock_bundle_store, mock_runtime, tmp_path):
    """A restarted worker keeps observing the surviving container through public sinks."""
    container = _make_discovered_container(worker_id="worker-rt", attempt_uid="uid-roundtrip")
    mock_runtime.discover_containers = Mock(return_value=[container])
    reader = FakeLogReader(_logs=[LogLine.now(source="stdout", data="resumed output")])

    class _AdoptedHandle(FakeContainerHandle):
        def log_reader(self) -> FakeLogReader:
            return reader

    mock_runtime.adopt_container = Mock(
        return_value=_AdoptedHandle(
            status_sequence=[
                ContainerStatus(phase=ContainerPhase.RUNNING),
                ContainerStatus(phase=ContainerPhase.STOPPED, exit_code=0),
            ]
        )
    )
    worker, sink = _worker_with_log_sink(
        mock_bundle_store,
        mock_runtime,
        tmp_path,
        worker_id="worker-rt",
    )
    assert worker.adopt_running_containers() == 1
    task = worker.task_by_uid("uid-roundtrip")
    assert task is not None
    wait_for_condition(lambda: task.status == job_pb2.TASK_STATE_SUCCEEDED)

    log_key = f"{container.task_id}:{container.attempt_id}"
    assert [entry.data for entry in sink.log_lines(log_key)] == ["resumed output"]
    task_rows = sink.rows(TASK_STATS_NAMESPACE)
    assert any(
        isinstance(row, IrisTaskStat) and row.task_id == container.task_id and row.worker_id == "worker-rt"
        for row in task_rows
    )
    worker.stop()


# ============================================================================
# Docker-based Adoption Integration Tests
# ============================================================================


@pytest.mark.docker
def test_docker_container_has_adoption_labels(docker_runtime, tmp_path):
    """Containers created by DockerRuntime should have adoption labels."""
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    config = ContainerConfig(
        image="iris-task:latest",
        entrypoint=job_pb2.RuntimeEntrypoint(
            run_command=job_pb2.CommandEntrypoint(argv=["echo", "hello"]),
        ),
        env={},
        mounts=[MountSpec("app", "/app", kind=MountKind.WORKDIR)],
        workdir_host_path=workdir,
        task_id="/test-user/test-job/0",
        attempt_id=3,
        attempt_uid="abcd1234abcd1234",
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

        labels = json.loads(result.stdout)
        assert labels["iris.managed"] == "true"
        assert labels["iris.task_id"] == "/test-user/test-job/0"
        assert labels["iris.attempt_id"] == "3"
        assert labels["iris.attempt_uid"] == "abcd1234abcd1234"
        assert labels["iris.worker_id"] == "worker-42"
        assert labels["iris.phase"] == "run"
        assert labels["iris.job_id"] == "/test-user/test-job"
    finally:
        handle.cleanup()


@pytest.mark.docker
def test_docker_discover_containers(docker_runtime, tmp_path):
    """discover_containers() should find iris-managed containers."""
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    config = ContainerConfig(
        image="iris-task:latest",
        entrypoint=job_pb2.RuntimeEntrypoint(
            run_command=job_pb2.CommandEntrypoint(argv=["sleep", "60"]),
        ),
        env={},
        mounts=[MountSpec("app", "/app", kind=MountKind.WORKDIR)],
        workdir_host_path=workdir,
        task_id="/test-user/discover-job/0",
        attempt_id=5,
        attempt_uid="cafe9999cafe9999",
        job_id="/test-user/discover-job",
        worker_id="worker-99",
        ports={"http": 30000, "grpc": 30001},
    )

    handle = docker_runtime.create_container(config)
    try:
        handle.run()

        discovered = docker_runtime.discover_containers()
        matching = [d for d in discovered if d.task_id == "/test-user/discover-job/0"]
        assert len(matching) == 1

        d = matching[0]
        assert d.attempt_id == 5
        assert d.attempt_uid == "cafe9999cafe9999"
        assert d.worker_id == "worker-99"
        assert d.phase == "run"
        assert d.running is True
        assert d.workdir_host_path == str(workdir)
        assert d.ports == {"http": 30000, "grpc": 30001}
    finally:
        handle.cleanup()


@pytest.mark.docker
def test_docker_adopt_container(docker_runtime, tmp_path):
    """adopt_container() should wrap an existing container."""
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    config = ContainerConfig(
        image="iris-task:latest",
        entrypoint=job_pb2.RuntimeEntrypoint(
            run_command=job_pb2.CommandEntrypoint(argv=["sleep", "60"]),
        ),
        env={},
        mounts=[MountSpec("app", "/app", kind=MountKind.WORKDIR)],
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


@pytest.mark.docker
def test_docker_worker_restart_round_trip_adopts_surviving_container(docker_runtime, mock_bundle_store, tmp_path):
    """End-to-end round trip with a real DockerRuntime.

    Container is created out of band, worker A starts and adopts it, then
    stops with preserve_containers=True. Worker B starts against the same
    DockerRuntime and adopts the still-running container with matching
    identity. This exercises the real discover_containers / adopt_container
    path through Worker.start(), which the mock-runtime test cannot.
    """
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    task_id = "/test-user/restart-job/0"
    job_id = "/test-user/restart-job"
    worker_id = "worker-restart"

    cfg = ContainerConfig(
        image="iris-task:latest",
        entrypoint=job_pb2.RuntimeEntrypoint(
            run_command=job_pb2.CommandEntrypoint(argv=["sleep", "60"]),
        ),
        env={},
        mounts=[MountSpec("app", "/app", kind=MountKind.WORKDIR)],
        workdir_host_path=workdir,
        task_id=task_id,
        attempt_id=0,
        job_id=job_id,
        worker_id=worker_id,
    )

    pre_handle = docker_runtime.create_container(cfg)
    pre_handle.run()
    container_id = pre_handle.container_id
    assert container_id is not None

    def make_config():
        return WorkerConfig(
            port=0,
            port_range=(50000, 50200),
            cache_dir=tmp_path / "cache",
            default_task_image="iris-task:latest",
            poll_interval=Duration.from_seconds(0.1),
            worker_id=worker_id,
        )

    # Worker A: real DockerRuntime, start, adopt the surviving container.
    worker_a = Worker(
        make_config(),
        bundle_store=mock_bundle_store,
        container_runtime=docker_runtime,
        threads=ThreadContainer(name="worker-a"),
    )
    try:
        worker_a.start()

        task_a = worker_a.get_task(task_id, 0)
        assert task_a is not None, "Worker A failed to adopt the pre-existing container"
        assert task_a.task_id == JobName.from_wire(task_id)
        assert task_a.attempt_id == 0
        assert task_a.container_id == container_id
        assert task_a.has_container
        assert task_a.status == job_pb2.TASK_STATE_RUNNING
    finally:
        worker_a.stop(preserve_containers=True)

    # Container should still be running after preserve_containers stop.
    discovered = docker_runtime.discover_containers()
    survivors = [d for d in discovered if d.task_id == task_id]
    assert len(survivors) == 1, "Container should survive preserve_containers stop"
    assert survivors[0].running is True

    # Worker B: fresh worker, same DockerRuntime, adopts the survivor.
    worker_b = Worker(
        make_config(),
        bundle_store=mock_bundle_store,
        container_runtime=docker_runtime,
        threads=ThreadContainer(name="worker-b"),
    )
    try:
        worker_b.start()

        task_b = worker_b.get_task(task_id, 0)
        assert task_b is not None, "Worker B failed to adopt the surviving container"
        assert task_b is not task_a
        assert task_b.task_id == task_a.task_id
        assert task_b.attempt_id == task_a.attempt_id
        assert task_b.container_id == container_id
        assert task_b.has_container
        assert task_b.status == job_pb2.TASK_STATE_RUNNING
    finally:
        # A normal stop on B kills the surviving container. A's detached monitor
        # then observes the exit; the second stop joins that process-local thread.
        worker_b.stop()
        worker_a.stop()
