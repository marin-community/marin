# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for apply_reconcile_response.

Each test sets up real SQLite DB state via ControllerTransitions helpers,
calls apply_reconcile_response, and asserts on post-state and cascade firing.

Covers all required cases from the kata:
- Terminal SUCCEEDED observation transitions attempt + fires _recompute_job_state.
- Terminal FAILED observation transitions attempt + fires cascades.
- MISSING observation → attempt transitions to FAILED("worker_lost_spec") + cascades.
- Duplicate terminal observation is idempotent (no finished_at overwrite).
- RPC error → no state change for non-ASSIGNED; synthetic WORKER_FAILED for ASSIGNED.
- Cross-worker preemption cascade fires when terminal observation is applied.
"""

from iris.cluster.controller.reconcile import (
    AttemptObservation,
    AttemptSpec,
    DesiredAttempt,
    ReconcileRequest,
    WorkerReconcilePlan,
)
from iris.cluster.controller.transitions import ControllerTransitions
from iris.cluster.types import AttemptUid, JobName, WorkerId
from iris.rpc import job_pb2
from rigging.timing import Timestamp

from .conftest import (
    dispatch_task,
    make_controller_state,
    make_job_request,
    make_worker_metadata,
    query_attempt,
    query_job,
    query_task,
    register_worker,
    submit_job,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_W1 = "worker-1"
_W2 = "worker-2"
_NOW = Timestamp(1_000_000_000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan(worker_id: str, desired: list[DesiredAttempt] | None = None) -> WorkerReconcilePlan:
    return WorkerReconcilePlan(
        request=ReconcileRequest(
            worker_id=worker_id,
            desired=desired or [],
        ),
    )


def _obs(
    task_id: JobName,
    attempt_id: int,
    state: int,
    *,
    exit_code: int | None = None,
    error: str | None = None,
) -> AttemptObservation:
    return AttemptObservation(
        attempt_uid=AttemptUid(""),
        state=state,
        task_id=task_id.to_wire(),
        attempt_id_compat=attempt_id,
        exit_code=exit_code,
        error=error,
    )


def _setup_running_task(
    state: ControllerTransitions,
    worker_id: str,
    job_name: str = "test-job",
) -> tuple[JobName, int]:
    """Register worker, submit job, dispatch, transition to RUNNING.

    Returns (task_id, attempt_id).
    """
    wid = WorkerId(worker_id)
    register_worker(state, worker_id, f"{worker_id}:8080", make_worker_metadata())
    tasks = submit_job(state, job_name, make_job_request(name=job_name))
    task_row = tasks[0]
    dispatch_task(state, task_row, wid)
    refreshed = query_task(state, task_row.task_id)
    assert refreshed is not None
    return task_row.task_id, refreshed.current_attempt_id


def _setup_assigned_task(
    state: ControllerTransitions,
    worker_id: str,
    job_name: str = "test-job",
) -> tuple[JobName, int]:
    """Register worker and submit job, queue assignment only (stays ASSIGNED).

    Returns (task_id, attempt_id).
    """
    from iris.cluster.controller.transitions import Assignment

    wid = WorkerId(worker_id)
    register_worker(state, worker_id, f"{worker_id}:8080", make_worker_metadata())
    tasks = submit_job(state, job_name, make_job_request(name=job_name))
    task_row = tasks[0]
    with state._db.transaction() as cur:
        state.queue_assignments(cur, [Assignment(task_id=task_row.task_id, worker_id=wid)])
    refreshed = query_task(state, task_row.task_id)
    assert refreshed is not None
    assert refreshed.state == job_pb2.TASK_STATE_ASSIGNED
    return task_row.task_id, refreshed.current_attempt_id


# ---------------------------------------------------------------------------
# Tests: successful observations
# ---------------------------------------------------------------------------


def test_succeeded_observation_transitions_attempt_and_recomputes_job():
    with make_controller_state() as state:
        task_id, attempt_id = _setup_running_task(state, _W1)
        plan = _make_plan(_W1)
        observations = [_obs(task_id, attempt_id, job_pb2.TASK_STATE_SUCCEEDED, exit_code=0)]
        with state._db.transaction() as cur:
            state.apply_reconcile_response(cur, plan, observations, None, _NOW)

        task = query_task(state, task_id)
        attempt = query_attempt(state, task_id, attempt_id)
        assert task is not None
        assert task.state == job_pb2.TASK_STATE_SUCCEEDED
        assert attempt is not None
        assert attempt.finished_at_ms is not None
        assert attempt.state == job_pb2.TASK_STATE_SUCCEEDED

        job_id = task.job_id
        job = query_job(state, job_id)
        assert job is not None
        assert job.state == job_pb2.JOB_STATE_SUCCEEDED


def test_failed_observation_transitions_attempt_and_fires_cascades():
    with make_controller_state() as state:
        task_id, attempt_id = _setup_running_task(state, _W1)
        plan = _make_plan(_W1)
        observations = [_obs(task_id, attempt_id, job_pb2.TASK_STATE_FAILED, error="segfault")]
        with state._db.transaction() as cur:
            state.apply_reconcile_response(cur, plan, observations, None, _NOW)

        task = query_task(state, task_id)
        assert task is not None
        assert task.state == job_pb2.TASK_STATE_FAILED

        job_id = task.job_id
        job = query_job(state, job_id)
        assert job is not None
        assert job.state == job_pb2.JOB_STATE_FAILED


def test_missing_observation_fails_attempt_with_worker_lost_spec():
    with make_controller_state() as state:
        task_id, attempt_id = _setup_running_task(state, _W1)
        plan = _make_plan(_W1)
        observations = [_obs(task_id, attempt_id, job_pb2.TASK_STATE_MISSING)]
        with state._db.transaction() as cur:
            state.apply_reconcile_response(cur, plan, observations, None, _NOW)

        task = query_task(state, task_id)
        assert task is not None
        assert task.state == job_pb2.TASK_STATE_FAILED
        assert task.error == "worker_lost_spec"

        attempt = query_attempt(state, task_id, attempt_id)
        assert attempt is not None
        assert attempt.state == job_pb2.TASK_STATE_FAILED
        assert attempt.error == "worker_lost_spec"


def test_missing_observation_on_building_task_fails_with_worker_lost_spec():
    """BUILDING + MISSING → FAILED("worker_lost_spec"): spec was lost before task started."""
    with make_controller_state() as state:
        from iris.cluster.controller.transitions import Assignment, HeartbeatApplyRequest, TaskUpdate

        wid = WorkerId(_W1)
        register_worker(state, _W1, f"{_W1}:8080", make_worker_metadata())
        tasks = submit_job(state, "build-job", make_job_request(name="build-job"))
        task_row = tasks[0]
        task_id = task_row.task_id

        with state._db.transaction() as cur:
            state.queue_assignments(cur, [Assignment(task_id=task_id, worker_id=wid)])
        refreshed = query_task(state, task_id)
        attempt_id = refreshed.current_attempt_id

        with state._db.transaction() as cur:
            state.apply_task_updates(
                cur,
                HeartbeatApplyRequest(
                    worker_id=wid,
                    updates=[TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_BUILDING)],
                ),
            )

        task_after_build = query_task(state, task_id)
        assert task_after_build.state == job_pb2.TASK_STATE_BUILDING

        plan = _make_plan(_W1)
        observations = [_obs(task_id, attempt_id, job_pb2.TASK_STATE_MISSING)]
        with state._db.transaction() as cur:
            state.apply_reconcile_response(cur, plan, observations, None, _NOW)

        task = query_task(state, task_id)
        assert task.state == job_pb2.TASK_STATE_FAILED
        assert task.error == "worker_lost_spec"


# ---------------------------------------------------------------------------
# Tests: idempotency
# ---------------------------------------------------------------------------


def test_duplicate_terminal_observation_does_not_overwrite_finished_at():
    with make_controller_state() as state:
        task_id, attempt_id = _setup_running_task(state, _W1)
        plan = _make_plan(_W1)
        observations = [_obs(task_id, attempt_id, job_pb2.TASK_STATE_SUCCEEDED, exit_code=0)]

        with state._db.transaction() as cur:
            state.apply_reconcile_response(cur, plan, observations, None, _NOW)

        attempt_after_first = query_attempt(state, task_id, attempt_id)
        assert attempt_after_first is not None
        finished_at_first = attempt_after_first.finished_at_ms

        # Apply the same observation again.
        with state._db.transaction() as cur:
            state.apply_reconcile_response(cur, plan, observations, None, _NOW)

        attempt_after_second = query_attempt(state, task_id, attempt_id)
        assert attempt_after_second is not None
        assert attempt_after_second.finished_at_ms == finished_at_first


def test_stale_running_observation_does_not_revive_cancelled_task():
    """A RUNNING observation after a task is CANCELLED must not revert the state."""
    with make_controller_state() as state:
        task_id, attempt_id = _setup_running_task(state, _W1)

        with state._db.transaction() as cur:
            task_row = query_task(state, task_id)
            state.cancel_job(cur, task_row.job_id, "user_cancel")

        task_after_cancel = query_task(state, task_id)
        assert task_after_cancel.state == job_pb2.TASK_STATE_KILLED

        plan = _make_plan(_W1)
        observations = [_obs(task_id, attempt_id, job_pb2.TASK_STATE_RUNNING)]
        with state._db.transaction() as cur:
            state.apply_reconcile_response(cur, plan, observations, None, _NOW)

        task = query_task(state, task_id)
        assert task.state == job_pb2.TASK_STATE_KILLED


# ---------------------------------------------------------------------------
# Tests: RPC error handling
# ---------------------------------------------------------------------------


def test_rpc_error_no_state_change_for_running_task():
    """On RPC error, non-ASSIGNED tasks are not modified."""
    with make_controller_state() as state:
        task_id, attempt_id = _setup_running_task(state, _W1)

        plan = _make_plan(
            _W1,
            desired=[
                DesiredAttempt(
                    attempt_uid=AttemptUid(""),
                    intent_run=AttemptSpec(),
                    task_id=task_id.to_wire(),
                    attempt_id=attempt_id,
                )
            ],
        )
        with state._db.transaction() as cur:
            state.apply_reconcile_response(cur, plan, [], "connection refused", _NOW)

        task = query_task(state, task_id)
        assert task.state == job_pb2.TASK_STATE_RUNNING


def test_rpc_error_applies_worker_failed_to_assigned_tasks():
    """On RPC error, ASSIGNED tasks with inline spec get synthetic WORKER_FAILED."""
    with make_controller_state() as state:
        task_id, attempt_id = _setup_assigned_task(state, _W1)

        plan = _make_plan(
            _W1,
            desired=[
                DesiredAttempt(
                    attempt_uid=AttemptUid(""),
                    intent_run=AttemptSpec(request=object()),
                    task_id=task_id.to_wire(),
                    attempt_id=attempt_id,
                )
            ],
        )
        with state._db.transaction() as cur:
            state.apply_reconcile_response(cur, plan, [], "timeout", _NOW)

        task = query_task(state, task_id)
        assert task is not None
        assert task.state == job_pb2.TASK_STATE_PENDING


def test_rpc_error_no_cascades():
    """On RPC error with no ASSIGNED tasks, return empty result with no cascades."""
    with make_controller_state() as state:
        register_worker(state, _W1, f"{_W1}:8080", make_worker_metadata())
        tasks = submit_job(state, "idle-job", make_job_request(name="idle-job"))
        task_id = tasks[0].task_id

        plan = _make_plan(_W1)
        with state._db.transaction() as cur:
            result = state.apply_reconcile_response(cur, plan, [], "connection refused", _NOW)

        assert result.tasks_to_kill == set()
        task = query_task(state, task_id)
        assert task.state == job_pb2.TASK_STATE_PENDING


# ---------------------------------------------------------------------------
# Tests: cross-worker cascade
# ---------------------------------------------------------------------------


def test_coscheduled_sibling_cascade_fires_on_terminal_observation():
    """Failure of one task in a coscheduled job cascades to sibling tasks."""
    with make_controller_state() as state:
        from iris.cluster.controller.transitions import Assignment

        wid1 = WorkerId(_W1)
        wid2 = WorkerId(_W2)
        register_worker(state, _W1, f"{_W1}:8080", make_worker_metadata())
        register_worker(state, _W2, f"{_W2}:8080", make_worker_metadata())

        req = make_job_request(name="cosched-job", replicas=2)
        req.coscheduling.group_by = "job"
        tasks = submit_job(state, "cosched-job", req)
        assert len(tasks) == 2

        task_id_1 = tasks[0].task_id
        task_id_2 = tasks[1].task_id

        with state._db.transaction() as cur:
            state.queue_assignments(
                cur,
                [
                    Assignment(task_id=task_id_1, worker_id=wid1),
                    Assignment(task_id=task_id_2, worker_id=wid2),
                ],
            )

        from iris.cluster.controller.transitions import HeartbeatApplyRequest, TaskUpdate

        t1 = query_task(state, task_id_1)
        t2 = query_task(state, task_id_2)
        attempt_id_1 = t1.current_attempt_id
        attempt_id_2 = t2.current_attempt_id

        for wid, task_id, attempt_id in [
            (wid1, task_id_1, attempt_id_1),
            (wid2, task_id_2, attempt_id_2),
        ]:
            with state._db.transaction() as cur:
                state.apply_task_updates(
                    cur,
                    HeartbeatApplyRequest(
                        worker_id=wid,
                        updates=[
                            TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_RUNNING)
                        ],
                    ),
                )

        plan = _make_plan(_W1)
        observations = [_obs(task_id_1, attempt_id_1, job_pb2.TASK_STATE_FAILED, error="oom")]
        with state._db.transaction() as cur:
            state.apply_reconcile_response(cur, plan, observations, None, _NOW)

        task1_final = query_task(state, task_id_1)
        task2_final = query_task(state, task_id_2)

        assert task1_final.state == job_pb2.TASK_STATE_FAILED
        assert task2_final.state in (
            job_pb2.TASK_STATE_KILLED,
            job_pb2.TASK_STATE_PREEMPTED,
            job_pb2.TASK_STATE_PENDING,
            job_pb2.TASK_STATE_COSCHED_FAILED,
        ), f"sibling state should have cascaded, got {task2_final.state}"


def test_unknown_worker_returns_empty_result():
    """apply_reconcile_response on a non-existent worker returns empty TxResult."""
    with make_controller_state() as state:
        plan = _make_plan("ghost-worker")
        with state._db.transaction() as cur:
            result = state.apply_reconcile_response(cur, plan, [], None, _NOW)

        assert result.tasks_to_kill == set()
        assert result.task_kill_workers == {}
