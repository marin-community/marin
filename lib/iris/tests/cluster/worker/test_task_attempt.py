# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TaskAttempt._monitor() BUILDING→RUNNING transition."""

from unittest.mock import Mock

import pytest

from iris.cluster.runtime.types import ContainerPhase, ContainerStats, ContainerStatus
from iris.cluster.types import Entrypoint, JobName
from iris.cluster.worker.port_allocator import PortAllocator
from iris.cluster.worker.task_attempt import TaskAttempt, TaskAttemptConfig
from iris.rpc import cluster_pb2


def _make_request():
    def noop():
        pass

    ep = Entrypoint.from_callable(noop).to_proto()
    return cluster_pb2.Worker.RunTaskRequest(
        task_id=JobName.root("test").task(0).to_wire(),
        num_tasks=1,
        attempt_id=0,
        entrypoint=ep,
        bundle_gcs_path="gs://test/bundle",
    )


def _make_mock_handle(status_sequence):
    """Create a mock ContainerHandle with a scripted status() sequence."""
    handle = Mock()
    handle.container_id = "test-container"
    handle.build = Mock(return_value=[])
    handle.run = Mock()
    handle.stop = Mock()
    handle.cleanup = Mock()

    call_count = [0]

    def status_fn():
        idx = min(call_count[0], len(status_sequence) - 1)
        call_count[0] += 1
        return status_sequence[idx]

    handle.status = Mock(side_effect=status_fn)

    log_reader = Mock()
    log_reader.read = Mock(return_value=[])
    log_reader.read_all = Mock(return_value=[])
    handle.log_reader = Mock(return_value=log_reader)
    handle.stats = Mock(return_value=ContainerStats(memory_mb=0, cpu_percent=0, process_count=0, available=False))

    return handle


def _make_task(tmp_path, mock_handle, *, task_ref=None):
    """Construct a TaskAttempt wired to a mock handle.

    If task_ref (a one-element list) is provided, the status() side-effect
    is wrapped to record task.status at each poll for later assertion.
    """
    states_during_polls: list[int] = []

    if task_ref is not None:
        original_side_effect = mock_handle.status.side_effect

        def capturing_status():
            if task_ref[0] is not None:
                states_during_polls.append(task_ref[0].status)
            return original_side_effect()

        mock_handle.status = Mock(side_effect=capturing_status)

    mock_runtime = Mock()
    mock_runtime.create_container = Mock(return_value=mock_handle)
    mock_runtime.stage_bundle = Mock()

    mock_log_sink = Mock()
    mock_log_sink.log_path = str(tmp_path / "logs")

    workdir = tmp_path / "workdir"
    workdir.mkdir(exist_ok=True)

    config = TaskAttemptConfig(
        task_id=JobName.root("test").task(0),
        num_tasks=1,
        attempt_id=0,
        request=_make_request(),
        ports={},
        workdir=workdir,
        cache_dir=tmp_path / "cache",
    )

    task = TaskAttempt(
        config=config,
        bundle_provider=Mock(),
        container_runtime=mock_runtime,
        worker_metadata=cluster_pb2.WorkerMetadata(),
        worker_id="test-worker",
        controller_address=None,
        default_task_env={},
        default_task_image="test-image",
        resolve_image=lambda x: x,
        port_allocator=PortAllocator(port_range=(40000, 40100)),
        report_state=lambda: None,
        log_sink=mock_log_sink,
        poll_interval_seconds=0.001,
    )

    if task_ref is not None:
        task_ref[0] = task

    return task, states_during_polls


@pytest.mark.parametrize("pending_polls", [0, 3], ids=["immediate_ready", "deferred_ready"])
def test_monitor_defers_running_until_container_ready(tmp_path, pending_polls):
    """_monitor() keeps the task in BUILDING until the container reports ready=True.

    pending_polls=0 simulates process/docker runtime (ready from the start).
    pending_polls=3 simulates K8S where the pod spends time in Pending phase.

    We capture task.status at each handle.status() call (before the monitor
    processes the result), then verify BUILDING is held during ready=False
    polls and RUNNING appears only after the first ready=True poll.
    """
    status_seq = (
        [ContainerStatus(phase=ContainerPhase.PENDING)] * pending_polls
        + [ContainerStatus(phase=ContainerPhase.RUNNING)]
        + [ContainerStatus(phase=ContainerPhase.STOPPED, exit_code=0)]
    )

    task_ref = [None]
    mock_handle = _make_mock_handle(status_seq)
    task, states = _make_task(tmp_path, mock_handle, task_ref=task_ref)

    task.run()

    assert task.status == cluster_pb2.TASK_STATE_SUCCEEDED
    assert task.exit_code == 0

    # During ready=False polls AND the ready=True poll (captured before
    # transition_to runs), the task must still be in BUILDING.
    for i in range(pending_polls + 1):
        assert (
            states[i] == cluster_pb2.TASK_STATE_BUILDING
        ), f"Poll {i}: expected BUILDING, got {cluster_pb2.TaskState.Name(states[i])}"

    # On the next poll (running=False), the transition to RUNNING has fired.
    assert states[pending_polls + 1] == cluster_pb2.TASK_STATE_RUNNING
