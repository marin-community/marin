# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the worker's reconcile-via-poll mode (Phase A.2)."""

from unittest.mock import Mock

import pytest
from iris.cluster.constraints import ProtocolMode, WellKnownAttribute
from iris.cluster.runtime.types import ContainerPhase, ContainerStatus
from iris.cluster.types import JobName
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.rpc import job_pb2, worker_pb2
from iris.test_util import wait_for_condition
from rigging.timing import Duration

from tests.cluster.worker.conftest import create_mock_container_handle, create_run_task_request

pytestmark = pytest.mark.timeout(10)


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


def _make_reconcile_worker(mock_bundle_store, mock_runtime, tmp_path) -> Worker:
    """Worker configured in reconcile-via-poll mode."""
    config = WorkerConfig(
        port=0,
        port_range=(50100, 50200),
        poll_interval=Duration.from_seconds(0.1),
        cache_dir=tmp_path / "cache",
        default_task_image="mock-image",
        reconcile_via_poll=True,
    )
    return Worker(
        config,
        bundle_store=mock_bundle_store,
        container_runtime=mock_runtime,
    )


@pytest.fixture
def reconcile_worker(mock_bundle_store, mock_runtime, tmp_path):
    return _make_reconcile_worker(mock_bundle_store, mock_runtime, tmp_path)


def _expected_entry(task_id: str, attempt_id: int = 0) -> job_pb2.WorkerTaskStatus:
    return job_pb2.WorkerTaskStatus(task_id=task_id, attempt_id=attempt_id)


# ----------------------------------------------------------------------------
# Protocol mode advertised via worker attributes
# ----------------------------------------------------------------------------


def test_protocol_mode_attribute_reconcile_via_poll(reconcile_worker):
    attrs = reconcile_worker._worker_metadata.attributes
    assert WellKnownAttribute.PROTOCOL_MODE in attrs
    assert attrs[WellKnownAttribute.PROTOCOL_MODE].string_value == ProtocolMode.RECONCILE_VIA_POLL


def test_protocol_mode_attribute_default_start_stop(mock_worker):
    attrs = mock_worker._worker_metadata.attributes
    assert WellKnownAttribute.PROTOCOL_MODE in attrs
    assert attrs[WellKnownAttribute.PROTOCOL_MODE].string_value == ProtocolMode.START_STOP


# ----------------------------------------------------------------------------
# Legacy mode: PollTasks does NOT fetch missing specs
# ----------------------------------------------------------------------------


def test_legacy_mode_poll_does_not_fetch_missing_specs(mock_worker):
    """In start_stop mode, PollTasks reports WORKER_FAILED for unknown expected keys.

    No call into ``_fetch_attempt_info`` is made — that path is gated on
    reconcile-via-poll mode.
    """
    fetch_spy = Mock(side_effect=AssertionError("must not be called in legacy mode"))
    mock_worker._fetch_attempt_info = fetch_spy

    task_id = JobName.root("test-user", "legacy-missing").task(0).to_wire()
    response = mock_worker.handle_poll_tasks(
        worker_pb2.Worker.PollTasksRequest(expected_tasks=[_expected_entry(task_id, attempt_id=0)])
    )

    fetch_spy.assert_not_called()
    assert len(response.tasks) == 1
    status = response.tasks[0]
    assert status.task_id == task_id
    assert status.state == job_pb2.TASK_STATE_WORKER_FAILED
    assert status.error == "Task not found on worker"


def test_legacy_mode_poll_kill_path_still_works(mock_worker, mock_runtime):
    """In start_stop mode, PollTasks with an unexpected local task still kills it."""
    mock_handle = create_mock_container_handle(
        status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
    )
    mock_runtime.create_container = Mock(return_value=mock_handle)
    task_id = JobName.root("test-user", "legacy-kill").task(0).to_wire()
    mock_worker.submit_task(create_run_task_request(task_id=task_id))
    task = mock_worker.get_task(task_id)
    wait_for_condition(lambda: task.status == job_pb2.TASK_STATE_RUNNING)
    mock_worker._recent_submissions.clear()

    mock_worker.handle_poll_tasks(worker_pb2.Worker.PollTasksRequest(expected_tasks=[]))

    assert task.should_stop is True
    task.thread.join(timeout=15.0)
    assert task.status == job_pb2.TASK_STATE_KILLED


# ----------------------------------------------------------------------------
# Reconcile-via-poll mode: handle_start_tasks rejects
# ----------------------------------------------------------------------------


def test_reconcile_mode_rejects_start_tasks(reconcile_worker):
    task_id = JobName.root("test-user", "no-push").task(0).to_wire()
    response = reconcile_worker.handle_start_tasks(
        worker_pb2.Worker.StartTasksRequest(tasks=[create_run_task_request(task_id=task_id)])
    )

    assert len(response.acks) == 1
    ack = response.acks[0]
    assert ack.task_id == task_id
    assert ack.accepted is False
    assert "reconcile-via-poll" in ack.error
    assert "GetTaskAttemptInfo" in ack.error
    # And no task was actually submitted.
    assert reconcile_worker.get_task(task_id) is None


# ----------------------------------------------------------------------------
# Reconcile-via-poll mode: fetch + submit on expected_keys - local_keys
# ----------------------------------------------------------------------------


def test_reconcile_mode_poll_fetches_and_submits_missing(reconcile_worker, mock_runtime):
    """expected_keys absent from local state trigger a fetch + submit."""
    mock_handle = create_mock_container_handle(
        status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
    )
    mock_runtime.create_container = Mock(return_value=mock_handle)

    task_id_a = JobName.root("test-user", "fetch-a").task(0).to_wire()
    task_id_b = JobName.root("test-user", "fetch-b").task(0).to_wire()

    canned_a = create_run_task_request(task_id=task_id_a, attempt_id=0)
    canned_b = create_run_task_request(task_id=task_id_b, attempt_id=2)

    fetch_calls: list[tuple[str, int]] = []

    def fake_fetch(task_id: str, attempt_id: int) -> job_pb2.RunTaskRequest:
        fetch_calls.append((task_id, attempt_id))
        if task_id == task_id_a:
            return canned_a
        if task_id == task_id_b:
            return canned_b
        raise AssertionError(f"unexpected fetch: {task_id} {attempt_id}")

    reconcile_worker._fetch_attempt_info = fake_fetch

    reconcile_worker.handle_poll_tasks(
        worker_pb2.Worker.PollTasksRequest(
            expected_tasks=[
                _expected_entry(task_id_a, attempt_id=0),
                _expected_entry(task_id_b, attempt_id=2),
            ]
        )
    )

    assert sorted(fetch_calls) == sorted([(task_id_a, 0), (task_id_b, 2)])

    # Both tasks were submitted locally.
    task_a = reconcile_worker.get_task(task_id_a, attempt_id=0)
    task_b = reconcile_worker.get_task(task_id_b, attempt_id=2)
    assert task_a is not None
    assert task_b is not None

    # Specs were cached.
    assert (task_id_a, 0) in reconcile_worker._spec_cache
    assert (task_id_b, 2) in reconcile_worker._spec_cache

    # Cleanup: stop running threads.
    reconcile_worker._recent_submissions.clear()
    reconcile_worker.handle_poll_tasks(worker_pb2.Worker.PollTasksRequest(expected_tasks=[]))
    if task_a.thread:
        task_a.thread.join(timeout=15.0)
    if task_b.thread:
        task_b.thread.join(timeout=15.0)


def test_reconcile_mode_kill_path_still_works(reconcile_worker, mock_runtime):
    """local_keys - expected_keys is killed exactly as in legacy mode."""
    mock_handle = create_mock_container_handle(
        status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
    )
    mock_runtime.create_container = Mock(return_value=mock_handle)

    task_id = JobName.root("test-user", "reconcile-kill").task(0).to_wire()
    canned = create_run_task_request(task_id=task_id, attempt_id=0)

    # Seed the worker with a running task by going through the fetch path
    # (handle_start_tasks is rejected in reconcile mode).
    reconcile_worker._fetch_attempt_info = lambda t, a: canned
    reconcile_worker.handle_poll_tasks(worker_pb2.Worker.PollTasksRequest(expected_tasks=[_expected_entry(task_id, 0)]))
    task = reconcile_worker.get_task(task_id, attempt_id=0)
    wait_for_condition(lambda: task.status == job_pb2.TASK_STATE_RUNNING)

    # Now the controller drops the task from expected_tasks. The worker must
    # kill it. Clear recent-submissions so the grace window doesn't protect
    # the task we just installed.
    reconcile_worker._recent_submissions.clear()
    reconcile_worker.handle_poll_tasks(worker_pb2.Worker.PollTasksRequest(expected_tasks=[]))

    assert task.should_stop is True
    task.thread.join(timeout=15.0)
    assert task.status == job_pb2.TASK_STATE_KILLED


# ----------------------------------------------------------------------------
# Spec cache behavior
# ----------------------------------------------------------------------------


def test_spec_cache_hit_avoids_refetch(reconcile_worker, mock_runtime):
    """A second fetch for the same (task_id, attempt_id) reuses the cached spec."""
    mock_runtime.create_container = Mock(
        side_effect=lambda config: create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
        )
    )

    task_id = JobName.root("test-user", "cache-hit").task(0).to_wire()
    canned = create_run_task_request(task_id=task_id, attempt_id=0)

    fetch_count = 0

    def counting_fetch(t: str, a: int) -> job_pb2.RunTaskRequest:
        nonlocal fetch_count
        # Honour the cache check that lives in _fetch_attempt_info: when
        # the entry is already cached we should never reach here.
        cached = reconcile_worker._spec_cache.get((t, a))
        if cached is not None:
            return cached
        fetch_count += 1
        return canned

    reconcile_worker._fetch_attempt_info = counting_fetch

    # First poll: cold cache, one fetch.
    reconcile_worker.handle_poll_tasks(worker_pb2.Worker.PollTasksRequest(expected_tasks=[_expected_entry(task_id, 0)]))
    assert fetch_count == 1

    # Simulate the task disappearing locally so the next poll re-attempts
    # the fetch path. (Removes both _tasks entry and recent-submissions.)
    with reconcile_worker._lock:
        reconcile_worker._tasks.clear()
        reconcile_worker._recent_submissions.clear()

    # Second poll: warm cache, no additional fetch beyond the cache hit.
    reconcile_worker.handle_poll_tasks(worker_pb2.Worker.PollTasksRequest(expected_tasks=[_expected_entry(task_id, 0)]))
    assert fetch_count == 1

    # The newly-submitted task should be tracked.
    task = reconcile_worker.get_task(task_id, attempt_id=0)
    assert task is not None

    # Cleanup.
    reconcile_worker._recent_submissions.clear()
    reconcile_worker.handle_poll_tasks(worker_pb2.Worker.PollTasksRequest(expected_tasks=[]))
    if task.thread:
        task.thread.join(timeout=15.0)


def test_spec_cache_lru_eviction(reconcile_worker):
    """The spec cache evicts least-recently-used entries beyond its capacity."""
    cap = Worker._SPEC_CACHE_CAPACITY
    for i in range(cap + 5):
        reconcile_worker._record_attempt_spec(
            f"task-{i}",
            0,
            create_run_task_request(task_id=f"task-{i}", attempt_id=0),
        )
    assert len(reconcile_worker._spec_cache) == cap
    # The first five entries are evicted; the last ``cap`` survive.
    assert ("task-0", 0) not in reconcile_worker._spec_cache
    assert ("task-4", 0) not in reconcile_worker._spec_cache
    assert ("task-5", 0) in reconcile_worker._spec_cache
    assert (f"task-{cap + 4}", 0) in reconcile_worker._spec_cache


# ----------------------------------------------------------------------------
# Stub helper behavior (NotImplementedError until A.1 lands)
# ----------------------------------------------------------------------------


def test_fetch_attempt_info_stub_raises_until_wired(reconcile_worker):
    """Without a controller client and an empty cache, the stub raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="GetTaskAttemptInfo"):
        reconcile_worker._fetch_attempt_info("task-x", 0)


def test_fetch_attempt_info_returns_cached_entry(reconcile_worker):
    """The stub helper still returns cached entries — only the network path is stubbed."""
    canned = create_run_task_request(task_id="task-y", attempt_id=3)
    reconcile_worker._record_attempt_spec("task-y", 3, canned)
    got = reconcile_worker._fetch_attempt_info("task-y", 3)
    assert got is canned


# ----------------------------------------------------------------------------
# Grace window protects fresh fetch submissions
# ----------------------------------------------------------------------------


def test_recent_submissions_protect_just_fetched_task(reconcile_worker, mock_runtime):
    """A task fetched and submitted on one poll is not re-fetched on the next."""
    mock_runtime.create_container = Mock(
        side_effect=lambda config: create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
        )
    )
    task_id = JobName.root("test-user", "grace").task(0).to_wire()
    canned = create_run_task_request(task_id=task_id, attempt_id=0)

    fetch_count = 0

    def counting_fetch(t: str, a: int) -> job_pb2.RunTaskRequest:
        nonlocal fetch_count
        fetch_count += 1
        return canned

    reconcile_worker._fetch_attempt_info = counting_fetch

    reconcile_worker.handle_poll_tasks(worker_pb2.Worker.PollTasksRequest(expected_tasks=[_expected_entry(task_id, 0)]))
    assert fetch_count == 1

    # Second poll arrives with the same expected_tasks: task is locally
    # present, no fetch.
    reconcile_worker.handle_poll_tasks(worker_pb2.Worker.PollTasksRequest(expected_tasks=[_expected_entry(task_id, 0)]))
    assert fetch_count == 1

    # Cleanup.
    reconcile_worker._recent_submissions.clear()
    reconcile_worker.handle_poll_tasks(worker_pb2.Worker.PollTasksRequest(expected_tasks=[]))
    task = reconcile_worker.get_task(task_id, attempt_id=0)
    if task and task.thread:
        task.thread.join(timeout=15.0)
