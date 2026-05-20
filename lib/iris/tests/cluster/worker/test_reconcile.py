# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``Worker.handle_reconcile``.

These exercise the Reconcile RPC handler in-process by calling
``worker.handle_reconcile(request)`` directly on a real Worker instance.
"""

from unittest.mock import Mock

import pytest
from iris.cluster.runtime.types import ContainerPhase, ContainerStatus
from iris.cluster.types import JobName
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.rpc import job_pb2, worker_pb2
from iris.test_util import wait_for_condition
from rigging.timing import Duration

from tests.cluster.worker.conftest import create_mock_container_handle, create_run_task_request

pytestmark = pytest.mark.timeout(10)


@pytest.fixture
def worker(mock_bundle_store, mock_runtime, tmp_path) -> Worker:
    config = WorkerConfig(
        port=0,
        port_range=(50200, 50300),
        poll_interval=Duration.from_seconds(0.05),
        cache_dir=tmp_path / "cache",
        default_task_image="mock-image",
        worker_id="w-reconcile-test",
    )
    return Worker(config, bundle_store=mock_bundle_store, container_runtime=mock_runtime)


def _task_id(name: str = "reconcile-task") -> str:
    return JobName.root("test-user", name).task(0).to_wire()


def _run_desired(
    task_id: str,
    attempt_id: int,
    run_request: job_pb2.RunTaskRequest | None = None,
    attempt_uid: str = "",
) -> worker_pb2.Worker.DesiredAttempt:
    """Build a DesiredAttempt with intent=run, optionally with inline spec."""
    spec = (
        worker_pb2.Worker.AttemptSpec(request=run_request)
        if run_request is not None
        else worker_pb2.Worker.AttemptSpec()
    )
    return worker_pb2.Worker.DesiredAttempt(
        attempt_uid=attempt_uid,
        task_id=task_id,
        attempt_id=attempt_id,
        run=spec,
    )


def _stop_desired(task_id: str, attempt_id: int, attempt_uid: str = "") -> worker_pb2.Worker.DesiredAttempt:
    return worker_pb2.Worker.DesiredAttempt(
        attempt_uid=attempt_uid,
        task_id=task_id,
        attempt_id=attempt_id,
        stop=worker_pb2.Worker.STOP_REASON_CANCELLED,
    )


def _reconcile(
    worker: Worker,
    desired: list[worker_pb2.Worker.DesiredAttempt],
) -> worker_pb2.Worker.ReconcileResponse:
    return worker.handle_reconcile(worker_pb2.Worker.ReconcileRequest(desired=desired))


def _observations_by_key(
    response: worker_pb2.Worker.ReconcileResponse,
) -> dict[tuple[str, int], worker_pb2.Worker.AttemptObservation]:
    return {(obs.task_id, obs.attempt_id): obs for obs in response.observed}


def test_inline_spec_enqueues_and_reports_building(worker, mock_runtime):
    """First reconcile with an inline spec enqueues the attempt; response is BUILDING."""
    task_id = _task_id("inline-spec")
    run_req = create_run_task_request(task_id=task_id, attempt_id=0)

    # Use a container handle that stays pending indefinitely so we can inspect state.
    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
        )
    )

    response = _reconcile(worker, [_run_desired(task_id, 0, run_request=run_req)])

    obs = _observations_by_key(response)
    key = (task_id, 0)
    assert key in obs, "Expected observation for the newly-enqueued attempt"
    # Workers map PENDING -> BUILDING in the observation (never expose PENDING to controller).
    assert obs[key].state in (
        job_pb2.TASK_STATE_BUILDING,
        job_pb2.TASK_STATE_RUNNING,
    ), f"Unexpected state {obs[key].state} — expected BUILDING or RUNNING"

    # The attempt must now be tracked locally.
    assert worker.get_task(task_id, attempt_id=0) is not None

    # Clean up.
    worker.kill_task(task_id)
    task = worker.get_task(task_id)
    if task and task.thread:
        task.thread.join(timeout=5.0)


def test_second_reconcile_without_spec_is_cache_hit(worker, mock_runtime):
    """A follow-up reconcile with no inline spec must not re-enqueue the attempt.

    After the first call enqueues the attempt, the second call observes the
    locally-tracked task and reports its current state instead of MISSING.
    """
    task_id = _task_id("cache-hit")
    run_req = create_run_task_request(task_id=task_id, attempt_id=0)

    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
        )
    )
    submit_spy = Mock(wraps=worker.submit_task)
    worker.submit_task = submit_spy  # type: ignore[method-assign]

    # First call: spec inline, enqueues attempt.
    _reconcile(worker, [_run_desired(task_id, 0, run_request=run_req)])
    call_count_after_first = submit_spy.call_count

    # Second call: no inline spec.
    response2 = _reconcile(worker, [_run_desired(task_id, 0, run_request=None)])

    # No additional submit_task call.
    assert submit_spy.call_count == call_count_after_first, "Unexpected re-enqueue on second reconcile"

    # Observation still present and NOT MISSING.
    obs = _observations_by_key(response2)
    assert (task_id, 0) in obs
    assert obs[(task_id, 0)].state != job_pb2.TASK_STATE_MISSING

    worker.kill_task(task_id)
    task = worker.get_task(task_id)
    if task and task.thread:
        task.thread.join(timeout=5.0)


def test_no_spec_unknown_attempt_reports_missing(worker):
    """intent=run for an unknown attempt with no inline spec → TASK_STATE_MISSING."""
    task_id = _task_id("missing")

    # Worker has no prior knowledge of this attempt.
    assert worker.get_task(task_id) is None

    response = _reconcile(worker, [_run_desired(task_id, 0, run_request=None)])

    obs = _observations_by_key(response)
    assert (task_id, 0) in obs
    assert obs[(task_id, 0)].state == job_pb2.TASK_STATE_MISSING

    # Nothing should have been enqueued.
    assert worker.get_task(task_id) is None


def test_stop_intent_kills_running_attempt(worker, mock_runtime):
    """intent=stop signals the worker to kill the attempt asynchronously."""
    task_id = _task_id("stop-intent")
    run_req = create_run_task_request(task_id=task_id, attempt_id=0)

    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
        )
    )

    # Submit a task so it's locally known.
    worker.submit_task(run_req)
    task = worker.get_task(task_id, attempt_id=0)
    assert task is not None
    wait_for_condition(lambda: task.status == job_pb2.TASK_STATE_RUNNING)

    # Reconcile with intent=stop.
    response = _reconcile(worker, [_stop_desired(task_id, 0)])

    # The response should include the attempt's current observation (running or transitioning).
    obs = _observations_by_key(response)
    assert (task_id, 0) in obs

    # should_stop is set synchronously before handle_reconcile returns.
    assert task.should_stop is True

    task.thread.join(timeout=5.0)
    assert task.status == job_pb2.TASK_STATE_KILLED


def test_zombie_attempt_is_killed(worker, mock_runtime):
    """Attempt running locally but absent from desired is killed (zombie kill)."""
    task_id = _task_id("zombie")
    run_req = create_run_task_request(task_id=task_id, attempt_id=0)

    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
        )
    )

    worker.submit_task(run_req)
    task = worker.get_task(task_id, attempt_id=0)
    assert task is not None
    wait_for_condition(lambda: task.status == job_pb2.TASK_STATE_RUNNING)

    # Reconcile with an empty desired set — the running attempt becomes a zombie.
    _reconcile(worker, [])

    # Zombie kill is async; should_stop is set synchronously.
    assert task.should_stop is True

    task.thread.join(timeout=5.0)
    assert task.status == job_pb2.TASK_STATE_KILLED


def test_zombie_kill_skips_terminal_tasks(worker, mock_runtime):
    """Attempts already in a terminal state are not re-killed during zombie detection."""
    task_id = _task_id("zombie-terminal")
    run_req = create_run_task_request(task_id=task_id, attempt_id=0)

    # Container exits immediately with success.
    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.STOPPED, exit_code=0)],
        )
    )

    worker.submit_task(run_req)
    task = worker.get_task(task_id, attempt_id=0)
    assert task is not None
    task.thread.join(timeout=5.0)
    assert task.status == job_pb2.TASK_STATE_SUCCEEDED

    # Reconcile with empty desired; terminal attempt must not be re-killed.
    _reconcile(worker, [])
    assert task.status == job_pb2.TASK_STATE_SUCCEEDED


def test_terminal_attempt_retained_and_observed_on_later_reconcile(worker, mock_runtime):
    """A terminal attempt stays in the worker's list and is still observed.

    Terminal attempts are retained so their logs remain accessible; a follow-up
    reconcile with no inline spec must report the terminal state, not MISSING.
    """
    task_id = _task_id("retain")
    run_req = create_run_task_request(task_id=task_id, attempt_id=0)

    # Container exits immediately with success.
    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.STOPPED, exit_code=0)],
        )
    )

    # First reconcile: spec inline → enqueues the attempt.
    _reconcile(worker, [_run_desired(task_id, 0, run_request=run_req)])

    # Wait for the task to terminate.
    task = worker.get_task(task_id, attempt_id=0)
    assert task is not None
    task.thread.join(timeout=5.0)
    assert task.status == job_pb2.TASK_STATE_SUCCEEDED

    # Second reconcile tick with no inline spec: the terminal attempt is still
    # tracked and observed (retained for log access), not synthesized as MISSING.
    response = _reconcile(worker, [_run_desired(task_id, 0, run_request=None)])
    assert worker.get_task(task_id, attempt_id=0) is task
    obs = _observations_by_key(response)
    assert (task_id, 0) in obs
    assert obs[(task_id, 0)].state == job_pb2.TASK_STATE_SUCCEEDED


def test_reconcile_response_includes_worker_id(worker):
    """ReconcileResponse always includes the worker_id."""
    response = _reconcile(worker, [])
    assert response.worker_id == "w-reconcile-test"


def test_reconcile_response_includes_health(worker):
    """ReconcileResponse always includes a WorkerHealth message."""
    response = _reconcile(worker, [])
    assert response.HasField("health")


def test_reconcile_empty_desired_empty_observed_when_no_tasks(worker):
    """With no desired and no local tasks, observed list is empty."""
    response = _reconcile(worker, [])
    assert list(response.observed) == []


def test_reconcile_pending_state_reported_as_building(worker, mock_runtime):
    """Workers never expose TASK_STATE_PENDING; it is mapped to TASK_STATE_BUILDING."""
    task_id = _task_id("pending-map")
    run_req = create_run_task_request(task_id=task_id, attempt_id=0)

    # Container never starts so the attempt stays in PENDING / BUILDING.
    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
        )
    )

    # Submit directly so attempt is in _tasks before reconcile is called.
    worker.submit_task(run_req)

    response = _reconcile(worker, [_run_desired(task_id, 0, run_request=run_req)])

    obs = _observations_by_key(response)
    assert (task_id, 0) in obs
    # PENDING must be hidden from the controller.
    assert obs[(task_id, 0)].state != job_pb2.TASK_STATE_PENDING

    worker.kill_task(task_id)
    task = worker.get_task(task_id, attempt_id=0)
    if task and task.thread:
        task.thread.join(timeout=5.0)


# ============================================================================
# UID routing
# ============================================================================


def test_observation_carries_attempt_uid(worker, mock_runtime):
    """An observation built for an attempt reports that attempt's UID."""
    task_id = _task_id("obs-uid")
    run_req = create_run_task_request(task_id=task_id, attempt_id=0, attempt_uid="uid-obs-1")

    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
        )
    )
    worker.submit_task(run_req)

    response = _reconcile(worker, [_run_desired(task_id, 0, attempt_uid="uid-obs-1")])

    obs = _observations_by_key(response)
    assert (task_id, 0) in obs
    assert obs[(task_id, 0)].attempt_uid == "uid-obs-1"

    worker.kill_task(task_id)
    task = worker.get_task(task_id, attempt_id=0)
    if task and task.thread:
        task.thread.join(timeout=5.0)


def test_run_intent_routes_to_attempt_by_uid(worker, mock_runtime):
    """A run intent carrying a known UID resolves to that attempt — no re-enqueue."""
    task_id = _task_id("route-uid")
    run_req = create_run_task_request(task_id=task_id, attempt_id=0, attempt_uid="uid-route-1")

    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
        )
    )
    worker.submit_task(run_req)
    submit_spy = Mock(wraps=worker.submit_task)
    worker.submit_task = submit_spy  # type: ignore[method-assign]

    # Run intent with the same UID but no inline spec: must resolve by UID, no re-enqueue.
    response = _reconcile(worker, [_run_desired(task_id, 0, run_request=None, attempt_uid="uid-route-1")])

    assert submit_spy.call_count == 0, "Run intent matching by UID must not re-enqueue"
    obs = _observations_by_key(response)
    assert (task_id, 0) in obs
    assert obs[(task_id, 0)].state != job_pb2.TASK_STATE_MISSING

    worker.kill_task(task_id)
    task = worker.get_task(task_id, attempt_id=0)
    if task and task.thread:
        task.thread.join(timeout=5.0)


def test_run_intent_unknown_attempt_no_spec_reports_missing(worker):
    """A run intent with a UID matching no attempt and no inline spec → MISSING."""
    task_id = _task_id("uid-missing")

    response = _reconcile(worker, [_run_desired(task_id, 0, run_request=None, attempt_uid="uid-unknown")])

    obs = _observations_by_key(response)
    assert (task_id, 0) in obs
    missing = obs[(task_id, 0)]
    assert missing.state == job_pb2.TASK_STATE_MISSING
    # MISSING observation echoes the UID so the controller can route it.
    assert missing.attempt_uid == "uid-unknown"
    assert worker.get_task(task_id) is None


def test_stop_intent_routes_to_attempt_by_uid(worker, mock_runtime):
    """A stop intent carrying a known UID kills that attempt."""
    task_id = _task_id("stop-uid")
    run_req = create_run_task_request(task_id=task_id, attempt_id=0, attempt_uid="uid-stop-1")

    mock_runtime.create_container = Mock(
        return_value=create_mock_container_handle(
            status_sequence=[ContainerStatus(phase=ContainerPhase.RUNNING)] * 100,
        )
    )
    worker.submit_task(run_req)
    task = worker.get_task(task_id, attempt_id=0)
    assert task is not None
    wait_for_condition(lambda: task.status == job_pb2.TASK_STATE_RUNNING)

    _reconcile(worker, [_stop_desired(task_id, 0, attempt_uid="uid-stop-1")])

    assert task.should_stop is True
    task.thread.join(timeout=5.0)
    assert task.status == job_pb2.TASK_STATE_KILLED
