# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Phase-A legacy wire translators.

Tests ``legacy_translator_request`` and ``legacy_translator_response`` from
``controller/reconcile.py``. All tests are pure-function: no DB, no asyncio,
no RPC.

Covers:
- Round-trip: each non-terminal dispatch cell from §5.3 lands in the right
  wire list (start_tasks, expected_tasks, stop_tasks).
- ``legacy_translator_response`` converts poll_updates into AttemptObservations
  with correct field mapping.
- ``poll_error`` → empty observations list.
- Absent entry (worker didn't return it) → no observation produced.
"""

from dataclasses import dataclass

import pytest
from iris.cluster.controller.reconcile import (
    AttemptSpec,
    DesiredAttempt,
    ReconcileRequest,
    StopReason,
    WorkerReconcilePlan,
    legacy_translator_request,
    legacy_translator_response,
)
from iris.cluster.controller.transitions import RunningTaskEntry
from iris.cluster.types import AttemptUid, JobName, WorkerId
from iris.rpc import job_pb2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_W1 = WorkerId("worker-1")
_TASK1 = JobName.from_string("/alice/job1/0")
_TASK2 = JobName.from_string("/alice/job1/1")
_TASK3 = JobName.from_string("/alice/job1/2")
_TASK4 = JobName.from_string("/alice/job1/3")
_ADDRESS = "localhost:9999"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec(task_id: JobName = _TASK1, attempt_id: int = 1) -> job_pb2.RunTaskRequest:
    """Build a minimal RunTaskRequest for use as an ASSIGNED spec."""
    req = job_pb2.RunTaskRequest()
    req.num_tasks = 1
    return req


def _desired_run_with_spec(
    task_id: JobName = _TASK1,
    attempt_id: int = 1,
    spec: job_pb2.RunTaskRequest | None = None,
) -> DesiredAttempt:
    return DesiredAttempt(
        attempt_uid=AttemptUid(""),
        intent_run=AttemptSpec(request=spec or _spec(task_id, attempt_id)),
        task_id=task_id.to_wire(),
        attempt_id=attempt_id,
    )


def _desired_run_no_spec(task_id: JobName = _TASK2, attempt_id: int = 2) -> DesiredAttempt:
    return DesiredAttempt(
        attempt_uid=AttemptUid(""),
        intent_run=AttemptSpec(request=None),
        task_id=task_id.to_wire(),
        attempt_id=attempt_id,
    )


def _desired_stop(
    task_id: JobName = _TASK3,
    attempt_id: int = 3,
    stop_reason: StopReason = StopReason.CANCELLED,
) -> DesiredAttempt:
    return DesiredAttempt(
        attempt_uid=AttemptUid(""),
        intent_stop=stop_reason,
        task_id=task_id.to_wire(),
        attempt_id=attempt_id,
    )


def _plan(desired: list[DesiredAttempt], worker_id: str = str(_W1)) -> WorkerReconcilePlan:
    return WorkerReconcilePlan(
        request=ReconcileRequest(worker_id=worker_id, desired=desired),
    )


@dataclass
class _FakeResult:
    """Minimal stand-in for WorkerReconcileResult for translation tests."""

    poll_updates: list | None
    poll_error: str | None = None
    start_error: str | None = None


@dataclass
class _FakeTaskUpdate:
    """Minimal stand-in for TaskUpdate for translation tests."""

    task_id: JobName
    attempt_id: int
    new_state: int
    error: str | None = None
    exit_code: int | None = None
    container_id: str | None = None


# ---------------------------------------------------------------------------
# legacy_translator_request — round-trip tests covering each §5.3 cell
# ---------------------------------------------------------------------------


def test_assigned_with_spec_lands_in_start_and_expected() -> None:
    """ASSIGNED-with-spec entry → start_tasks AND expected_tasks."""
    da = _desired_run_with_spec(_TASK1, attempt_id=1)
    dispatch = legacy_translator_request(_plan([da]), _ADDRESS)

    assert len(dispatch.start_tasks) == 1
    assert dispatch.start_tasks[0].task_id == _TASK1.to_wire()
    assert dispatch.start_tasks[0].attempt_id == 1

    assert len(dispatch.expected_tasks) == 1
    assert dispatch.expected_tasks[0] == RunningTaskEntry(task_id=_TASK1, attempt_id=1)

    assert dispatch.stop_tasks == []


@pytest.mark.parametrize("attempt_id", [2, 5])
def test_active_no_spec_only_in_expected(attempt_id: int) -> None:
    """BUILDING/RUNNING (run, no spec) → expected_tasks only; not in start_tasks.

    BUILDING and RUNNING share the same dispatch shape — the translator does
    not distinguish them; both produce an expected_tasks entry and nothing else.
    """
    da = _desired_run_no_spec(_TASK2, attempt_id=attempt_id)
    dispatch = legacy_translator_request(_plan([da]), _ADDRESS)

    assert dispatch.start_tasks == []
    assert dispatch.expected_tasks == [RunningTaskEntry(task_id=_TASK2, attempt_id=attempt_id)]
    assert dispatch.stop_tasks == []


@pytest.mark.parametrize(
    "task_id,attempt_id,stop_reason",
    [
        (_TASK3, 3, StopReason.CANCELLED),
        (_TASK4, 4, StopReason.PREEMPTED),
    ],
)
def test_stop_intent_in_stop_only(task_id: JobName, attempt_id: int, stop_reason: StopReason) -> None:
    """CANCELLED/PREEMPTED row (intent_stop set) → stop_tasks only."""
    da = _desired_stop(task_id, attempt_id=attempt_id, stop_reason=stop_reason)
    dispatch = legacy_translator_request(_plan([da]), _ADDRESS)

    assert dispatch.start_tasks == []
    assert dispatch.expected_tasks == []
    assert dispatch.stop_tasks == [task_id.to_wire()]


def test_mixed_plan_all_cells() -> None:
    """Plan with ASSIGNED, BUILDING, RUNNING, CANCELLED, PREEMPTED entries."""
    assigned = _desired_run_with_spec(_TASK1, attempt_id=1)
    building = _desired_run_no_spec(_TASK2, attempt_id=2)
    cancelled = _desired_stop(_TASK3, attempt_id=3, stop_reason=StopReason.CANCELLED)
    preempted = _desired_stop(_TASK4, attempt_id=4, stop_reason=StopReason.PREEMPTED)

    dispatch = legacy_translator_request(_plan([assigned, building, cancelled, preempted]), _ADDRESS)

    # start_tasks: only ASSIGNED-with-spec
    assert len(dispatch.start_tasks) == 1
    assert dispatch.start_tasks[0].task_id == _TASK1.to_wire()

    # expected_tasks: ASSIGNED + BUILDING (not stop entries)
    assert len(dispatch.expected_tasks) == 2
    expected_task_ids = {e.task_id for e in dispatch.expected_tasks}
    assert expected_task_ids == {_TASK1, _TASK2}

    # stop_tasks: CANCELLED + PREEMPTED
    assert set(dispatch.stop_tasks) == {_TASK3.to_wire(), _TASK4.to_wire()}


def test_worker_id_and_address_propagated() -> None:
    dispatch = legacy_translator_request(_plan([], worker_id="w-xyz"), "1.2.3.4:9000")
    assert dispatch.worker_id == WorkerId("w-xyz")
    assert dispatch.address == "1.2.3.4:9000"


def test_empty_plan_produces_empty_dispatch() -> None:
    dispatch = legacy_translator_request(_plan([]), _ADDRESS)
    assert dispatch.start_tasks == []
    assert dispatch.expected_tasks == []
    assert dispatch.stop_tasks == []


def test_start_task_stamps_task_id_and_attempt_id() -> None:
    """RunTaskRequest in start_tasks must have task_id and attempt_id stamped."""
    da = _desired_run_with_spec(_TASK1, attempt_id=7)
    dispatch = legacy_translator_request(_plan([da]), _ADDRESS)
    req = dispatch.start_tasks[0]
    assert req.task_id == _TASK1.to_wire()
    assert req.attempt_id == 7


# ---------------------------------------------------------------------------
# legacy_translator_response
# ---------------------------------------------------------------------------


def test_poll_error_returns_empty_observations() -> None:
    """poll_error set → empty observation list regardless of plan."""
    result = _FakeResult(poll_updates=None, poll_error="connection refused")
    obs = legacy_translator_response(_plan([]), result)
    assert obs == []


def test_none_poll_updates_returns_empty() -> None:
    """poll_updates=None (no poll ran) → empty list."""
    result = _FakeResult(poll_updates=None, poll_error=None)
    obs = legacy_translator_response(_plan([]), result)
    assert obs == []


def test_empty_poll_updates_returns_empty() -> None:
    result = _FakeResult(poll_updates=[], poll_error=None)
    obs = legacy_translator_response(_plan([]), result)
    assert obs == []


def test_building_update_converted_to_observation() -> None:
    update = _FakeTaskUpdate(
        task_id=_TASK1,
        attempt_id=1,
        new_state=job_pb2.TASK_STATE_BUILDING,
    )
    result = _FakeResult(poll_updates=[update])
    obs = legacy_translator_response(_plan([]), result)

    assert len(obs) == 1
    assert obs[0].state == job_pb2.TASK_STATE_BUILDING
    assert obs[0].task_id == _TASK1.to_wire()
    assert obs[0].attempt_id_compat == 1
    assert obs[0].exit_code is None
    assert obs[0].error is None
    assert obs[0].container_id is None


def test_running_update_converted_to_observation() -> None:
    update = _FakeTaskUpdate(
        task_id=_TASK2,
        attempt_id=2,
        new_state=job_pb2.TASK_STATE_RUNNING,
        container_id="c-abc123",
    )
    result = _FakeResult(poll_updates=[update])
    obs = legacy_translator_response(_plan([]), result)

    assert len(obs) == 1
    assert obs[0].state == job_pb2.TASK_STATE_RUNNING
    assert obs[0].container_id == "c-abc123"


def test_terminal_succeeded_update_converted() -> None:
    update = _FakeTaskUpdate(
        task_id=_TASK1,
        attempt_id=1,
        new_state=job_pb2.TASK_STATE_SUCCEEDED,
        exit_code=0,
    )
    result = _FakeResult(poll_updates=[update])
    obs = legacy_translator_response(_plan([]), result)

    assert len(obs) == 1
    assert obs[0].state == job_pb2.TASK_STATE_SUCCEEDED
    assert obs[0].exit_code == 0
    assert obs[0].task_id == _TASK1.to_wire()


def test_terminal_failed_update_converted() -> None:
    update = _FakeTaskUpdate(
        task_id=_TASK1,
        attempt_id=3,
        new_state=job_pb2.TASK_STATE_FAILED,
        exit_code=1,
        error="OOM",
    )
    result = _FakeResult(poll_updates=[update])
    obs = legacy_translator_response(_plan([]), result)

    assert len(obs) == 1
    assert obs[0].state == job_pb2.TASK_STATE_FAILED
    assert obs[0].exit_code == 1
    assert obs[0].error == "OOM"


def test_multiple_updates_all_converted() -> None:
    """Multiple entries in poll_updates produce one observation each."""
    updates = [
        _FakeTaskUpdate(task_id=_TASK1, attempt_id=1, new_state=job_pb2.TASK_STATE_BUILDING),
        _FakeTaskUpdate(task_id=_TASK2, attempt_id=2, new_state=job_pb2.TASK_STATE_RUNNING),
        _FakeTaskUpdate(task_id=_TASK3, attempt_id=3, new_state=job_pb2.TASK_STATE_SUCCEEDED, exit_code=0),
    ]
    result = _FakeResult(poll_updates=updates)
    obs = legacy_translator_response(_plan([]), result)

    assert len(obs) == 3
    states = {o.task_id: o.state for o in obs}
    assert states[_TASK1.to_wire()] == job_pb2.TASK_STATE_BUILDING
    assert states[_TASK2.to_wire()] == job_pb2.TASK_STATE_RUNNING
    assert states[_TASK3.to_wire()] == job_pb2.TASK_STATE_SUCCEEDED


def test_absent_expected_entry_produces_no_observation() -> None:
    """Worker doesn't return status for an expected entry → no observation.

    This is the Phase-A MISSING quirk: the legacy PollTasks wire has no
    MISSING state. The worker learns about the attempt on the next tick via
    the GetTaskAttemptInfo pull path. No observation is fabricated here.
    """
    # Plan has TASK1 in expected, but poll_updates only has TASK2.
    building = _desired_run_no_spec(_TASK1, attempt_id=1)
    running_update = _FakeTaskUpdate(task_id=_TASK2, attempt_id=2, new_state=job_pb2.TASK_STATE_RUNNING)
    result = _FakeResult(poll_updates=[running_update])
    obs = legacy_translator_response(_plan([building]), result)

    # Only TASK2 observation; TASK1 absent → no observation produced.
    assert len(obs) == 1
    assert obs[0].task_id == _TASK2.to_wire()


def test_start_error_does_not_suppress_poll_observations() -> None:
    """start_error does not affect poll observation translation.

    A.4 handles the start_error path; legacy_translator_response only
    checks poll_error.
    """
    update = _FakeTaskUpdate(task_id=_TASK1, attempt_id=1, new_state=job_pb2.TASK_STATE_RUNNING)
    result = _FakeResult(poll_updates=[update], start_error="start rpc failed")
    obs = legacy_translator_response(_plan([]), result)
    assert len(obs) == 1
    assert obs[0].state == job_pb2.TASK_STATE_RUNNING
