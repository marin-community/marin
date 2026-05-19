# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidated tests for the controller reconcile pipeline.

Three layers, exercised in order:

1. **Pure compute** — ``reconcile_workers`` builds one ``ReconcileRequest``
   proto per worker from a ``ReconcileInputs`` snapshot. No DB.
2. **Wire & dispatch** — ``WorkerProvider.reconcile_workers`` fans out via a
   fake stub factory and synthesizes ``ReconcileResult.observations`` for both
   the ``Reconcile`` RPC wire (``use_reconcile_rpc=True``) and the legacy
   ``StartTasks`` + ``PollTasks`` wire (``use_reconcile_rpc=False``).
3. **Apply + e2e** — ``ControllerTransitions.apply_reconcile_result`` against
   real SQLite DB state, plus a handful of end-to-end convergence ticks driven
   through ``Controller._reconcile_worker_batch``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from iris.cluster.controller.reconcile import (
    ReconcileInputs,
    ReconcileResult,
    ReconcileRow,
    WorkerReconcilePlan,
    reconcile_workers,
)
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    HeartbeatApplyRequest,
    TaskUpdate,
)
from iris.cluster.controller.worker_provider import WorkerProvider
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2, worker_pb2
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

_W1 = "worker-1"
_W2 = "worker-2"
_W1_ADDR = "worker-1:8080"
_NOW = Timestamp(1_000_000_000)


def _job_id(name: str) -> JobName:
    return JobName.root("test-user", name)


def _task_id(name: str, index: int = 0) -> JobName:
    """Matches the deterministic name ``submit_job`` produces."""
    return _job_id(name).child(f"task-{index}")


def _make_plan(
    worker_id: str,
    desired: list[worker_pb2.Worker.DesiredAttempt] | None = None,
) -> WorkerReconcilePlan:
    return WorkerReconcilePlan(
        worker_id=WorkerId(worker_id),
        request=worker_pb2.Worker.ReconcileRequest(
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
) -> worker_pb2.Worker.AttemptObservation:
    kwargs: dict = {
        "attempt_uid": "",
        "state": state,
        "task_id": task_id.to_wire(),
        "attempt_id": attempt_id,
    }
    if exit_code is not None:
        kwargs["exit_code"] = exit_code
    if error is not None:
        kwargs["error"] = error
    return worker_pb2.Worker.AttemptObservation(**kwargs)


def _desired_run(task_id: JobName, attempt_id: int, *, spec: job_pb2.RunTaskRequest | None = None):
    return worker_pb2.Worker.DesiredAttempt(
        attempt_uid="",
        run=worker_pb2.Worker.AttemptSpec(request=spec) if spec is not None else worker_pb2.Worker.AttemptSpec(),
        task_id=task_id.to_wire(),
        attempt_id=attempt_id,
    )


def _desired_stop(task_id: JobName, attempt_id: int, *, reason=worker_pb2.Worker.STOP_REASON_CANCELLED):
    return worker_pb2.Worker.DesiredAttempt(
        attempt_uid="",
        stop=reason,
        task_id=task_id.to_wire(),
        attempt_id=attempt_id,
    )


# ===========================================================================
# Section 1: pure-compute (reconcile_worker)
# ===========================================================================


def _row(task_state: int, *, task_id: str = "task-a", attempt_id: int = 0, job: str = "job-a") -> ReconcileRow:
    return ReconcileRow(
        worker_id=WorkerId(_W1),
        task_id=_task_id(task_id),
        attempt_id=attempt_id,
        task_state=task_state,
        attempt_state=job_pb2.TASK_STATE_PENDING,  # unused by reconcile_worker today
        job_id=_job_id(job),
    )


def _plan_for(
    rows: list[ReconcileRow],
    *,
    job_specs: dict[JobName, job_pb2.RunTaskRequest] | None = None,
) -> WorkerReconcilePlan:
    """Run the pure-compute layer for one worker and return its plan."""
    wid = WorkerId(_W1)
    inputs = ReconcileInputs(
        job_specs=dict(job_specs or {}),
        worker_ids=[wid],
        rows_by_worker={wid: list(rows)},
    )
    return reconcile_workers(inputs)[0]


def _spec(image: str = "spec-image") -> job_pb2.RunTaskRequest:
    return job_pb2.RunTaskRequest(task_image=image)


def test_reconcile_worker_empty_rows_yields_empty_plan():
    plan = _plan_for([])
    assert plan.worker_id == WorkerId(_W1)
    assert plan.request.worker_id == _W1
    assert list(plan.request.desired) == []


def test_reconcile_worker_assigned_with_spec_emits_run_with_inline_spec():
    row = _row(job_pb2.TASK_STATE_ASSIGNED, attempt_id=7, job="job-a", task_id="task-a")
    plan = _plan_for([row], job_specs={_job_id("job-a"): _spec("custom-image")})

    assert len(plan.request.desired) == 1
    desired = plan.request.desired[0]
    assert desired.task_id == row.task_id.to_wire()
    assert desired.attempt_id == 7
    assert desired.HasField("run")
    assert desired.run.HasField("request")
    # Spec is stamped with the routing key.
    assert desired.run.request.task_image == "custom-image"
    assert desired.run.request.task_id == row.task_id.to_wire()
    assert desired.run.request.attempt_id == 7


def test_reconcile_worker_assigned_without_spec_is_omitted():
    """ASSIGNED with no cached job spec is dropped (scheduler reissues later)."""
    plan = _plan_for([_row(job_pb2.TASK_STATE_ASSIGNED, job="job-missing")], job_specs={})
    assert list(plan.request.desired) == []


@pytest.mark.parametrize(
    "task_state",
    [job_pb2.TASK_STATE_BUILDING, job_pb2.TASK_STATE_RUNNING],
)
def test_reconcile_worker_executing_states_emit_run_without_inline_spec(task_state):
    """BUILDING / RUNNING: run intent but no inline spec (cache-hit invariant)."""
    plan = _plan_for([_row(task_state, attempt_id=3)])
    assert len(plan.request.desired) == 1
    desired = plan.request.desired[0]
    assert desired.HasField("run")
    assert not desired.run.HasField("request")
    assert desired.attempt_id == 3


@pytest.mark.parametrize(
    "task_state",
    [
        job_pb2.TASK_STATE_SUCCEEDED,
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_UNSCHEDULABLE,
        job_pb2.TASK_STATE_COSCHED_FAILED,
    ],
)
def test_reconcile_worker_terminal_rows_emit_run_without_inline_spec(task_state):
    """Worker-bound terminal rows stay expected until their attempt is finalized."""
    plan = _plan_for([_row(task_state, attempt_id=4)])

    assert len(plan.request.desired) == 1
    desired = plan.request.desired[0]
    assert desired.HasField("run")
    assert not desired.run.HasField("request")
    assert desired.attempt_id == 4


@pytest.mark.parametrize(
    "task_state,expected_reason",
    [
        (job_pb2.TASK_STATE_KILLED, worker_pb2.Worker.STOP_REASON_CANCELLED),
        (job_pb2.TASK_STATE_PREEMPTED, worker_pb2.Worker.STOP_REASON_PREEMPTED),
    ],
)
def test_reconcile_worker_stop_states_emit_stop_with_reason(task_state, expected_reason):
    plan = _plan_for([_row(task_state, attempt_id=2)])
    assert len(plan.request.desired) == 1
    desired = plan.request.desired[0]
    assert desired.HasField("stop")
    assert desired.stop == expected_reason
    assert desired.attempt_id == 2


@pytest.mark.parametrize(
    "task_state",
    [
        job_pb2.TASK_STATE_PENDING,
        job_pb2.TASK_STATE_UNSPECIFIED,
    ],
)
def test_reconcile_worker_unrecognised_states_are_omitted(task_state):
    plan = _plan_for([_row(task_state)])
    assert list(plan.request.desired) == []


def test_reconcile_worker_mixed_rows_per_axis():
    """A worker holding tasks across every axis builds one desired entry per worker-bound row."""
    rows = [
        _row(job_pb2.TASK_STATE_ASSIGNED, task_id="a", attempt_id=1, job="j1"),
        _row(job_pb2.TASK_STATE_RUNNING, task_id="b", attempt_id=2, job="j2"),
        _row(job_pb2.TASK_STATE_KILLED, task_id="c", attempt_id=3, job="j3"),
        _row(job_pb2.TASK_STATE_SUCCEEDED, task_id="d", attempt_id=4, job="j4"),
    ]
    plan = _plan_for(rows, job_specs={_job_id("j1"): _spec("img-j1")})

    by_task = {d.task_id: d for d in plan.request.desired}
    assert set(by_task) == {
        _task_id("a").to_wire(),
        _task_id("b").to_wire(),
        _task_id("c").to_wire(),
        _task_id("d").to_wire(),
    }
    assert by_task[_task_id("a").to_wire()].run.HasField("request")
    assert not by_task[_task_id("b").to_wire()].run.HasField("request")
    assert by_task[_task_id("c").to_wire()].stop == worker_pb2.Worker.STOP_REASON_CANCELLED
    assert not by_task[_task_id("d").to_wire()].run.HasField("request")


# ===========================================================================
# Section 2: wire & dispatch (WorkerProvider.reconcile_workers)
# ===========================================================================


@dataclass
class _FakeWorkerStub:
    """In-process WorkerServiceClient stand-in.

    Records every call and returns canned responses (or raises) according to
    pre-configured fields. Only the methods used by ``reconcile_workers`` are
    implemented.
    """

    address: str
    reconcile_calls: list[worker_pb2.Worker.ReconcileRequest] = field(default_factory=list)
    start_calls: list[worker_pb2.Worker.StartTasksRequest] = field(default_factory=list)
    poll_calls: list[worker_pb2.Worker.PollTasksRequest] = field(default_factory=list)
    reconcile_response: worker_pb2.Worker.ReconcileResponse | None = None
    reconcile_exc: Exception | None = None
    start_response: worker_pb2.Worker.StartTasksResponse | None = None
    start_exc: Exception | None = None
    poll_response: worker_pb2.Worker.PollTasksResponse | None = None
    poll_exc: Exception | None = None

    async def reconcile(self, request, *, timeout_ms=None):
        del timeout_ms
        self.reconcile_calls.append(request)
        if self.reconcile_exc is not None:
            raise self.reconcile_exc
        return self.reconcile_response or worker_pb2.Worker.ReconcileResponse()

    async def start_tasks(self, request, *, timeout_ms=None):
        del timeout_ms
        self.start_calls.append(request)
        if self.start_exc is not None:
            raise self.start_exc
        return self.start_response or worker_pb2.Worker.StartTasksResponse()

    async def poll_tasks(self, request, *, timeout_ms=None):
        del timeout_ms
        self.poll_calls.append(request)
        if self.poll_exc is not None:
            raise self.poll_exc
        return self.poll_response or worker_pb2.Worker.PollTasksResponse()


@dataclass
class _FakeStubFactory:
    stubs: dict[str, _FakeWorkerStub] = field(default_factory=dict)

    def get_stub(self, address: str) -> Any:
        if address not in self.stubs:
            self.stubs[address] = _FakeWorkerStub(address=address)
        return self.stubs[address]

    def evict(self, address: str) -> None:
        self.stubs.pop(address, None)

    def close(self) -> None:
        self.stubs.clear()


def _provider_with_stub(stub: _FakeWorkerStub | None = None) -> tuple[WorkerProvider, _FakeWorkerStub]:
    """Build a provider with a single stub at ``_W1_ADDR`` (creating a default if needed)."""
    if stub is None:
        stub = _FakeWorkerStub(address=_W1_ADDR)
    factory = _FakeStubFactory(stubs={_W1_ADDR: stub})
    return WorkerProvider(stub_factory=factory), stub


def _reconcile_one(provider: WorkerProvider, plan: WorkerReconcilePlan, *, rpc: bool, address: str = _W1_ADDR):
    return provider.reconcile_workers([plan], {WorkerId(_W1): address}, use_reconcile_rpc=rpc)


def test_reconcile_workers_empty_short_circuits():
    provider, _ = _provider_with_stub()
    assert provider.reconcile_workers([], {}, use_reconcile_rpc=True) == []
    assert provider.reconcile_workers([], {}, use_reconcile_rpc=False) == []


# --- New wire (use_reconcile_rpc=True) ---------------------------------------


def test_reconcile_rpc_forwards_observations_and_skips_legacy_wire():
    """One Reconcile RPC per plan; observed observations surface verbatim; no legacy calls."""
    observation = _obs(_task_id("a"), 0, job_pb2.TASK_STATE_RUNNING)
    stub = _FakeWorkerStub(
        address=_W1_ADDR,
        reconcile_response=worker_pb2.Worker.ReconcileResponse(observed=[observation]),
    )
    provider, _ = _provider_with_stub(stub)
    plan = _make_plan(_W1, desired=[_desired_run(_task_id("a"), 0)])

    results = _reconcile_one(provider, plan, rpc=True)

    assert len(stub.reconcile_calls) == 1
    assert stub.reconcile_calls[0].worker_id == _W1
    assert stub.start_calls == [] and stub.poll_calls == []
    assert len(results) == 1
    assert results[0].worker_id == WorkerId(_W1)
    assert results[0].error is None
    assert list(results[0].observations) == [observation]


def test_reconcile_rpc_failure_returns_error_and_empty_observations():
    stub = _FakeWorkerStub(address=_W1_ADDR, reconcile_exc=RuntimeError("boom"))
    provider, _ = _provider_with_stub(stub)

    results = _reconcile_one(provider, _make_plan(_W1), rpc=True)

    assert results[0].error == "boom"
    assert list(results[0].observations) == []


# --- Legacy wire (use_reconcile_rpc=False) -----------------------------------


def test_legacy_wire_splits_desired_into_start_expected_stop():
    """run-with-spec → StartTasks + expected; run-without-spec → only expected; stop → omitted from polls."""
    provider, stub = _provider_with_stub()

    tid_a = _task_id("a")
    tid_b = _task_id("b")
    tid_c = _task_id("c")
    plan = _make_plan(
        _W1,
        desired=[
            _desired_run(tid_a, 1, spec=_spec("img-a")),  # → StartTasks + expected
            _desired_run(tid_b, 2, spec=None),  # → only expected
            _desired_stop(tid_c, 3),  # → neither
        ],
    )

    results = provider.reconcile_workers([plan], {WorkerId(_W1): _W1_ADDR}, use_reconcile_rpc=False)

    assert len(stub.start_calls) == 1
    started = list(stub.start_calls[0].tasks)
    assert [(s.task_id, s.attempt_id, s.task_image) for s in started] == [(tid_a.to_wire(), 1, "img-a")]

    assert len(stub.poll_calls) == 1
    expected_keys = {(e.task_id, e.attempt_id) for e in stub.poll_calls[0].expected_tasks}
    assert expected_keys == {(tid_a.to_wire(), 1), (tid_b.to_wire(), 2)}

    assert stub.reconcile_calls == []
    # Default ack/poll responses: no observations.
    assert len(results) == 1
    assert results[0].error is None
    assert list(results[0].observations) == []


def test_legacy_wire_forwards_poll_updates_as_observations():
    tid_a = _task_id("a")
    poll_resp = worker_pb2.Worker.PollTasksResponse(
        tasks=[
            job_pb2.WorkerTaskStatus(task_id=tid_a.to_wire(), attempt_id=1, state=job_pb2.TASK_STATE_RUNNING),
        ]
    )
    provider, _ = _provider_with_stub(_FakeWorkerStub(address=_W1_ADDR, poll_response=poll_resp))

    plan = _make_plan(_W1, desired=[_desired_run(tid_a, 1, spec=None)])
    results = _reconcile_one(provider, plan, rpc=False)

    assert results[0].error is None
    observations = list(results[0].observations)
    assert len(observations) == 1
    assert (observations[0].task_id, observations[0].attempt_id, observations[0].state) == (
        tid_a.to_wire(),
        1,
        job_pb2.TASK_STATE_RUNNING,
    )


def test_legacy_wire_rejected_start_ack_emits_worker_failed_observation():
    """Non-accepted TaskAck → synthetic WORKER_FAILED observation."""
    tid_a = _task_id("a")
    start_resp = worker_pb2.Worker.StartTasksResponse(
        acks=[worker_pb2.Worker.TaskAck(task_id=tid_a.to_wire(), accepted=False, error="bundle missing")]
    )
    provider, _ = _provider_with_stub(_FakeWorkerStub(address=_W1_ADDR, start_response=start_resp))

    plan = _make_plan(_W1, desired=[_desired_run(tid_a, 5, spec=_spec("img"))])
    results = _reconcile_one(provider, plan, rpc=False)

    observations = list(results[0].observations)
    assert len(observations) == 1
    assert observations[0].state == job_pb2.TASK_STATE_WORKER_FAILED
    assert "bundle missing" in observations[0].error
    assert (observations[0].task_id, observations[0].attempt_id) == (tid_a.to_wire(), 5)


def test_legacy_wire_start_failure_surfaces_as_error():
    provider, _ = _provider_with_stub(_FakeWorkerStub(address=_W1_ADDR, start_exc=RuntimeError("connection refused")))
    plan = _make_plan(_W1, desired=[_desired_run(_task_id("a"), 1, spec=_spec("img"))])

    results = _reconcile_one(provider, plan, rpc=False)

    assert results[0].error == "connection refused"
    assert list(results[0].observations) == []


def test_legacy_wire_poll_failure_surfaces_as_error():
    """PollTasks failure surfaces as a reconcile error."""
    provider, _ = _provider_with_stub(_FakeWorkerStub(address=_W1_ADDR, poll_exc=RuntimeError("timeout")))
    plan = _make_plan(_W1, desired=[_desired_run(_task_id("a"), 1, spec=None)])

    results = _reconcile_one(provider, plan, rpc=False)

    assert results[0].error == "timeout"
    assert list(results[0].observations) == []


# ===========================================================================
# Section 3: apply (transitions) + e2e (controller tick)
# ===========================================================================


def _setup_running_task(state: ControllerTransitions, worker_id: str = _W1) -> tuple[JobName, int]:
    """Register worker, submit job, dispatch, drive to RUNNING."""
    wid = WorkerId(worker_id)
    register_worker(state, worker_id, f"{worker_id}:8080", make_worker_metadata())
    tasks = submit_job(state, "test-job", make_job_request(name="test-job"))
    task_row = tasks[0]
    dispatch_task(state, task_row, wid)
    refreshed = query_task(state, task_row.task_id)
    assert refreshed is not None
    return task_row.task_id, refreshed.current_attempt_id


def _setup_assigned_task(state: ControllerTransitions, worker_id: str = _W1) -> tuple[JobName, int]:
    """Register worker, submit job, queue assignment (no heartbeat → stays ASSIGNED)."""
    wid = WorkerId(worker_id)
    register_worker(state, worker_id, f"{worker_id}:8080", make_worker_metadata())
    tasks = submit_job(state, "test-job", make_job_request(name="test-job"))
    task_row = tasks[0]
    with state._db.transaction() as cur:
        state.queue_assignments(cur, [Assignment(task_id=task_row.task_id, worker_id=wid)])
    refreshed = query_task(state, task_row.task_id)
    assert refreshed is not None
    assert refreshed.state == job_pb2.TASK_STATE_ASSIGNED
    return task_row.task_id, refreshed.current_attempt_id


def _apply_observations(
    state: ControllerTransitions,
    worker_id: str,
    observations: list[worker_pb2.Worker.AttemptObservation],
):
    plan = _make_plan(worker_id)
    result = ReconcileResult(worker_id=WorkerId(worker_id), observations=observations, error=None)
    with state._db.transaction() as cur:
        return state.apply_reconcile_result(cur, plan, result, _NOW)


def _apply_failure(
    state: ControllerTransitions,
    worker_id: str,
    plan: WorkerReconcilePlan,
    error: str,
):
    result = ReconcileResult(worker_id=WorkerId(worker_id), observations=[], error=error)
    with state._db.transaction() as cur:
        return state.apply_reconcile_result(cur, plan, result, _NOW)


# --- Terminal observations transition tasks + propagate to jobs --------------


@pytest.mark.parametrize(
    "obs_state, expected_task_state, expected_job_state, attempt_kwargs",
    [
        (
            job_pb2.TASK_STATE_SUCCEEDED,
            job_pb2.TASK_STATE_SUCCEEDED,
            job_pb2.JOB_STATE_SUCCEEDED,
            {"exit_code": 0},
        ),
        (
            job_pb2.TASK_STATE_FAILED,
            job_pb2.TASK_STATE_FAILED,
            job_pb2.JOB_STATE_FAILED,
            {"error": "segfault"},
        ),
    ],
)
def test_terminal_observation_transitions_task_and_job(
    obs_state, expected_task_state, expected_job_state, attempt_kwargs
):
    with make_controller_state() as state:
        task_id, attempt_id = _setup_running_task(state)
        _apply_observations(state, _W1, [_obs(task_id, attempt_id, obs_state, **attempt_kwargs)])

        task = query_task(state, task_id)
        attempt = query_attempt(state, task_id, attempt_id)
        assert task is not None and attempt is not None
        assert task.state == expected_task_state
        assert attempt.state == expected_task_state
        assert attempt.finished_at_ms is not None

        job = query_job(state, task.job_id)
        assert job is not None
        assert job.state == expected_job_state


def test_missing_observation_fails_attempt_with_worker_lost_spec():
    with make_controller_state() as state:
        task_id, attempt_id = _setup_running_task(state)
        _apply_observations(state, _W1, [_obs(task_id, attempt_id, job_pb2.TASK_STATE_MISSING)])

        task = query_task(state, task_id)
        attempt = query_attempt(state, task_id, attempt_id)
        assert task is not None and attempt is not None
        assert task.state == job_pb2.TASK_STATE_FAILED
        assert task.error == "worker_lost_spec"
        assert attempt.state == job_pb2.TASK_STATE_FAILED
        assert attempt.error == "worker_lost_spec"


def test_duplicate_terminal_observation_does_not_overwrite_finished_at():
    with make_controller_state() as state:
        task_id, attempt_id = _setup_running_task(state)
        observations = [_obs(task_id, attempt_id, job_pb2.TASK_STATE_SUCCEEDED, exit_code=0)]

        _apply_observations(state, _W1, observations)
        first = query_attempt(state, task_id, attempt_id)
        assert first is not None

        _apply_observations(state, _W1, observations)
        second = query_attempt(state, task_id, attempt_id)
        assert second is not None
        assert second.finished_at_ms == first.finished_at_ms


def test_stale_running_observation_does_not_revive_cancelled_task():
    """RUNNING after cancellation must not roll the task forward."""
    with make_controller_state() as state:
        task_id, attempt_id = _setup_running_task(state)
        with state._db.transaction() as cur:
            task_row = query_task(state, task_id)
            assert task_row is not None
            state.cancel_job(cur, task_row.job_id, "user_cancel")
        assert query_task(state, task_id).state == job_pb2.TASK_STATE_KILLED

        _apply_observations(state, _W1, [_obs(task_id, attempt_id, job_pb2.TASK_STATE_RUNNING)])

        assert query_task(state, task_id).state == job_pb2.TASK_STATE_KILLED


# --- RPC failure (result.error set) ----------------------------------------


def test_rpc_failure_leaves_running_task_unchanged():
    """RPC failure does not mutate non-ASSIGNED tasks."""
    with make_controller_state() as state:
        task_id, attempt_id = _setup_running_task(state)
        plan = _make_plan(_W1, desired=[_desired_run(task_id, attempt_id, spec=None)])
        _apply_failure(state, _W1, plan, "connection refused")
        assert query_task(state, task_id).state == job_pb2.TASK_STATE_RUNNING


def test_rpc_failure_bounces_assigned_task_back_to_pending():
    """RPC failure on an ASSIGNED dispatch synthesizes WORKER_FAILED, returning the task to PENDING."""
    with make_controller_state() as state:
        task_id, attempt_id = _setup_assigned_task(state)
        plan = _make_plan(_W1, desired=[_desired_run(task_id, attempt_id, spec=_spec())])
        _apply_failure(state, _W1, plan, "timeout")
        # Synthetic WORKER_FAILED bounces the task back to PENDING so it can be re-dispatched.
        assert query_task(state, task_id).state == job_pb2.TASK_STATE_PENDING


def test_rpc_failure_with_no_assigned_attempts_is_a_noop():
    with make_controller_state() as state:
        register_worker(state, _W1, f"{_W1}:8080", make_worker_metadata())
        tasks = submit_job(state, "idle-job", make_job_request(name="idle-job"))
        task_id = tasks[0].task_id

        result = _apply_failure(state, _W1, _make_plan(_W1), "connection refused")

        assert result.tasks_to_kill == set()
        assert query_task(state, task_id).state == job_pb2.TASK_STATE_PENDING


def test_apply_result_on_unknown_worker_is_a_noop():
    with make_controller_state() as state:
        result = _apply_observations(state, "ghost-worker", [])
        assert result.tasks_to_kill == set()
        assert result.task_kill_workers == {}


# --- Coscheduled cascade ----------------------------------------------------


def test_coscheduled_sibling_cascade_fires_on_terminal_observation():
    """Failure of one coscheduled task cascades to its siblings on the other worker."""
    with make_controller_state() as state:
        wid1 = WorkerId(_W1)
        wid2 = WorkerId(_W2)
        register_worker(state, _W1, f"{_W1}:8080", make_worker_metadata())
        register_worker(state, _W2, f"{_W2}:8080", make_worker_metadata())

        req = make_job_request(name="cosched-job", replicas=2)
        req.coscheduling.group_by = "job"
        tasks = submit_job(state, "cosched-job", req)
        assert len(tasks) == 2
        task_id_1, task_id_2 = tasks[0].task_id, tasks[1].task_id

        with state._db.transaction() as cur:
            state.queue_assignments(
                cur,
                [
                    Assignment(task_id=task_id_1, worker_id=wid1),
                    Assignment(task_id=task_id_2, worker_id=wid2),
                ],
            )

        attempt_id_1 = query_task(state, task_id_1).current_attempt_id
        attempt_id_2 = query_task(state, task_id_2).current_attempt_id

        # Drive both to RUNNING.
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

        _apply_observations(state, _W1, [_obs(task_id_1, attempt_id_1, job_pb2.TASK_STATE_FAILED, error="oom")])

        assert query_task(state, task_id_1).state == job_pb2.TASK_STATE_FAILED
        sibling_state = query_task(state, task_id_2).state
        assert sibling_state in (
            job_pb2.TASK_STATE_KILLED,
            job_pb2.TASK_STATE_PREEMPTED,
            job_pb2.TASK_STATE_PENDING,
            job_pb2.TASK_STATE_COSCHED_FAILED,
        ), f"sibling state should have cascaded, got {sibling_state}"


# --- End-to-end: full controller tick over both wires ----------------------


@dataclass
class _ScriptedProvider:
    """In-process TaskProvider whose ``reconcile_workers`` returns scripted observations.

    Each tick consumes one ``script`` entry (a callable taking the plan and
    returning a list of observations). Records every call so tests can assert
    the right wire was selected and the right plans were dispatched.
    """

    use_reconcile_rpc_expected: bool
    script: list[Any] = field(default_factory=list)
    calls: list[tuple[list[WorkerReconcilePlan], dict, bool]] = field(default_factory=list)

    def get_process_status(self, *_args, **_kwargs):
        raise NotImplementedError

    def on_worker_failed(self, *_args, **_kwargs):
        pass

    def profile_task(self, *_args, **_kwargs):
        raise NotImplementedError

    def ping_workers(self, workers):
        return []

    def reconcile_workers(self, plans, addresses, *, use_reconcile_rpc):
        self.calls.append((list(plans), dict(addresses), use_reconcile_rpc))
        assert (
            use_reconcile_rpc == self.use_reconcile_rpc_expected
        ), f"expected use_reconcile_rpc={self.use_reconcile_rpc_expected}, got {use_reconcile_rpc}"
        tick = len(self.calls) - 1
        responder = self.script[tick] if tick < len(self.script) else (lambda plan: [])
        return [ReconcileResult(worker_id=p.worker_id, observations=responder(p), error=None) for p in plans]

    def close(self):
        pass


def _observation_for_all_run(plan: WorkerReconcilePlan, state: int, **kwargs):
    """Build one observation per run-intent in the plan."""
    return [
        worker_pb2.Worker.AttemptObservation(
            attempt_uid="",
            state=state,
            task_id=d.task_id,
            attempt_id=d.attempt_id,
            **kwargs,
        )
        for d in plan.request.desired
        if d.HasField("run")
    ]


@pytest.mark.parametrize("flag", [True, False])
def test_e2e_converges_to_succeeded_through_both_wires(flag, make_controller):
    """Full ASSIGNED → RUNNING → SUCCEEDED convergence over either wire."""
    script = [
        lambda _plan: [],  # tick 1: ASSIGNED dispatch, worker hasn't started
        lambda plan: _observation_for_all_run(plan, job_pb2.TASK_STATE_RUNNING),
        lambda plan: _observation_for_all_run(plan, job_pb2.TASK_STATE_SUCCEEDED, exit_code=0),
    ]
    provider = _ScriptedProvider(use_reconcile_rpc_expected=flag, script=script)
    ctrl = make_controller(provider=provider, reconcile_rpc_enabled=flag)
    state = ctrl._transitions

    wid = register_worker(state, _W1, _W1_ADDR, make_worker_metadata())
    tasks = submit_job(state, "e2e-job", make_job_request(name="e2e-job"))
    task_id = tasks[0].task_id

    with state._db.transaction() as cur:
        state.queue_assignments(cur, [Assignment(task_id=task_id, worker_id=wid)])

    # Tick 1: ASSIGNED — controller dispatches the inline spec.
    ctrl._reconcile_worker_batch()
    tick1_desired = list(provider.calls[0][0][0].request.desired)
    assert len(tick1_desired) == 1
    assert tick1_desired[0].HasField("run") and tick1_desired[0].run.HasField(
        "request"
    ), "first tick should carry inline spec"

    # Tick 2: worker reports RUNNING.
    ctrl._reconcile_worker_batch()
    assert query_task(state, task_id).state == job_pb2.TASK_STATE_RUNNING

    # Tick 3: subsequent run intents must not carry inline spec (cache-hit invariant).
    ctrl._reconcile_worker_batch()
    tick3_desired = list(provider.calls[2][0][0].request.desired)
    assert tick3_desired and tick3_desired[0].HasField("run")
    assert not tick3_desired[0].run.HasField("request"), "subsequent ticks must not carry inline spec"

    task_final = query_task(state, task_id)
    assert task_final.state == job_pb2.TASK_STATE_SUCCEEDED
    assert query_job(state, task_final.job_id).state == job_pb2.JOB_STATE_SUCCEEDED


def test_e2e_missing_observation_fails_attempt_with_worker_lost_spec(make_controller):
    """End-to-end MISSING cascade: dispatch → MISSING → FAILED("worker_lost_spec")."""
    script = [
        lambda _plan: [],  # tick 1: ASSIGNED dispatch
        lambda plan: _observation_for_all_run(plan, job_pb2.TASK_STATE_MISSING),
    ]
    provider = _ScriptedProvider(use_reconcile_rpc_expected=True, script=script)
    ctrl = make_controller(provider=provider, reconcile_rpc_enabled=True)
    state = ctrl._transitions

    wid = register_worker(state, _W1, _W1_ADDR, make_worker_metadata())
    tasks = submit_job(state, "missing-job", make_job_request(name="missing-job"))
    task_id = tasks[0].task_id

    with state._db.transaction() as cur:
        state.queue_assignments(cur, [Assignment(task_id=task_id, worker_id=wid)])

    ctrl._reconcile_worker_batch()
    ctrl._reconcile_worker_batch()

    task = query_task(state, task_id)
    assert task.state == job_pb2.TASK_STATE_FAILED
    assert task.error == "worker_lost_spec"
