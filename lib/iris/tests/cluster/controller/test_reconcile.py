# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidated tests for the controller reconcile pipeline.

Three layers, exercised in order:

1. **Pure compute** — ``reconcile_worker`` builds a ``ReconcileRequest`` proto
   from a hand-built ``WorkerReconcileInputs``. No DB.
2. **Wire & dispatch** — ``WorkerProvider.reconcile_workers`` fans out via a
   fake stub factory and synthesizes ``ReconcileResult.observations`` for both
   the new ``Reconcile`` RPC wire (``use_reconcile_rpc=True``) and the legacy
   ``StartTasks`` + ``PollTasks`` wire (``use_reconcile_rpc=False``).
3. **Apply + e2e** — ``ControllerTransitions.apply_reconcile_observations`` /
   ``apply_reconcile_failure`` against real SQLite DB state, plus a handful of
   end-to-end convergence ticks driven through ``Controller._reconcile_worker_batch``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from iris.cluster.controller.reconcile import (
    ReconcileRow,
    WorkerReconcileInputs,
    WorkerReconcilePlan,
    WorkerRow,
    reconcile_worker,
)
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    HeartbeatApplyRequest,
    TaskUpdate,
)
from iris.cluster.controller.worker_provider import ReconcileResult, WorkerProvider
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

# ---------------------------------------------------------------------------
# Shared constants & helpers
# ---------------------------------------------------------------------------

_W1 = "worker-1"
_W2 = "worker-2"
_W1_ADDR = "worker-1:8080"
_NOW = Timestamp(1_000_000_000)


def _job_id(name: str) -> JobName:
    return JobName.root("test-user", name)


def _task_id(name: str, index: int = 0) -> JobName:
    """Build a deterministic task name (matches what ``submit_job`` produces)."""
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


# ===========================================================================
# Section 1: pure-compute (reconcile_worker)
# ===========================================================================


def _inputs(
    rows: list[ReconcileRow],
    *,
    job_specs: dict[JobName, job_pb2.RunTaskRequest] | None = None,
    worker_id: str = _W1,
) -> WorkerReconcileInputs:
    return WorkerReconcileInputs(
        worker=WorkerRow(worker_id=WorkerId(worker_id), address=_W1_ADDR),
        rows=list(rows),
        job_specs=dict(job_specs or {}),
    )


def _row(task_state: int, *, task_id: str = "task-a", attempt_id: int = 0, job: str = "job-a") -> ReconcileRow:
    """Build a ReconcileRow for the pure-compute tests.

    ``attempt_state`` is unused by ``reconcile_worker`` today; pass a
    plausible PENDING placeholder.
    """
    return ReconcileRow(
        worker_id=WorkerId(_W1),
        task_id=_task_id(task_id),
        attempt_id=attempt_id,
        task_state=task_state,
        attempt_state=job_pb2.TASK_STATE_PENDING,
        job_id=_job_id(job),
    )


def _spec(image: str = "spec-image") -> job_pb2.RunTaskRequest:
    return job_pb2.RunTaskRequest(task_image=image)


def test_reconcile_worker_empty_rows_yields_empty_desired():
    plan = reconcile_worker(_inputs([]))
    assert plan.worker_id == WorkerId(_W1)
    assert plan.request.worker_id == _W1
    assert list(plan.request.desired) == []


def test_reconcile_worker_assigned_with_spec_emits_run_with_inline_spec():
    row = _row(job_pb2.TASK_STATE_ASSIGNED, attempt_id=7, job="job-a", task_id="task-a")
    spec = _spec("custom-image")
    plan = reconcile_worker(_inputs([row], job_specs={_job_id("job-a"): spec}))

    assert len(plan.request.desired) == 1
    desired = plan.request.desired[0]
    assert desired.attempt_uid == ""
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
    row = _row(job_pb2.TASK_STATE_ASSIGNED, job="job-missing")
    plan = reconcile_worker(_inputs([row], job_specs={}))
    assert list(plan.request.desired) == []


@pytest.mark.parametrize(
    "task_state",
    [job_pb2.TASK_STATE_BUILDING, job_pb2.TASK_STATE_RUNNING],
)
def test_reconcile_worker_executing_states_emit_run_with_empty_spec(task_state):
    """BUILDING / RUNNING: run intent but no inline spec (cache-hit invariant)."""
    row = _row(task_state, attempt_id=3)
    plan = reconcile_worker(_inputs([row]))

    assert len(plan.request.desired) == 1
    desired = plan.request.desired[0]
    assert desired.HasField("run")
    assert not desired.run.HasField("request"), "EXECUTING tasks must not carry inline spec"
    assert desired.attempt_id == 3
    assert desired.attempt_uid == ""


@pytest.mark.parametrize(
    "task_state,expected_reason",
    [
        (job_pb2.TASK_STATE_KILLED, worker_pb2.Worker.STOP_REASON_CANCELLED),
        (job_pb2.TASK_STATE_PREEMPTED, worker_pb2.Worker.STOP_REASON_PREEMPTED),
    ],
)
def test_reconcile_worker_stop_states_emit_stop_with_reason(task_state, expected_reason):
    row = _row(task_state, attempt_id=2)
    plan = reconcile_worker(_inputs([row]))

    assert len(plan.request.desired) == 1
    desired = plan.request.desired[0]
    assert desired.HasField("stop")
    assert desired.stop == expected_reason
    assert desired.attempt_id == 2


@pytest.mark.parametrize(
    "task_state",
    [
        job_pb2.TASK_STATE_SUCCEEDED,
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_PENDING,
        job_pb2.TASK_STATE_UNSPECIFIED,
    ],
)
def test_reconcile_worker_terminal_and_unrecognised_states_are_omitted(task_state):
    row = _row(task_state)
    plan = reconcile_worker(_inputs([row]))
    assert list(plan.request.desired) == []


def test_reconcile_worker_mixed_rows_per_axis():
    """A worker holding tasks across every axis builds one desired entry per non-terminal row."""
    rows = [
        _row(job_pb2.TASK_STATE_ASSIGNED, task_id="a", attempt_id=1, job="j1"),
        _row(job_pb2.TASK_STATE_RUNNING, task_id="b", attempt_id=2, job="j2"),
        _row(job_pb2.TASK_STATE_KILLED, task_id="c", attempt_id=3, job="j3"),
        _row(job_pb2.TASK_STATE_SUCCEEDED, task_id="d", attempt_id=4, job="j4"),
    ]
    plan = reconcile_worker(
        _inputs(
            rows,
            job_specs={_job_id("j1"): _spec("img-j1")},
        )
    )

    by_task = {d.task_id: d for d in plan.request.desired}
    # SUCCEEDED row is dropped.
    assert set(by_task) == {
        _task_id("a").to_wire(),
        _task_id("b").to_wire(),
        _task_id("c").to_wire(),
    }
    assert by_task[_task_id("a").to_wire()].run.HasField("request")
    assert not by_task[_task_id("b").to_wire()].run.HasField("request")
    assert by_task[_task_id("c").to_wire()].stop == worker_pb2.Worker.STOP_REASON_CANCELLED


# ===========================================================================
# Section 2: wire & dispatch (WorkerProvider.reconcile_workers)
# ===========================================================================


@dataclass
class _StubCallLog:
    """Records every wire-level call made through fake stubs."""

    reconcile: list[worker_pb2.Worker.ReconcileRequest] = field(default_factory=list)
    start_tasks: list[worker_pb2.Worker.StartTasksRequest] = field(default_factory=list)
    poll_tasks: list[worker_pb2.Worker.PollTasksRequest] = field(default_factory=list)


@dataclass
class _FakeWorkerStub:
    """In-process WorkerServiceClient stand-in.

    Records every call and returns canned responses (or raises) according to
    pre-configured fields. Only the methods used by ``reconcile_workers`` are
    implemented — anything else asserts.
    """

    address: str
    log: _StubCallLog
    reconcile_response: worker_pb2.Worker.ReconcileResponse | None = None
    reconcile_exc: Exception | None = None
    start_response: worker_pb2.Worker.StartTasksResponse | None = None
    start_exc: Exception | None = None
    poll_response: worker_pb2.Worker.PollTasksResponse | None = None
    poll_exc: Exception | None = None

    async def reconcile(self, request, *, timeout_ms=None):
        del timeout_ms
        self.log.reconcile.append(request)
        if self.reconcile_exc is not None:
            raise self.reconcile_exc
        return self.reconcile_response or worker_pb2.Worker.ReconcileResponse()

    async def start_tasks(self, request, *, timeout_ms=None):
        del timeout_ms
        self.log.start_tasks.append(request)
        if self.start_exc is not None:
            raise self.start_exc
        return self.start_response or worker_pb2.Worker.StartTasksResponse()

    async def poll_tasks(self, request, *, timeout_ms=None):
        del timeout_ms
        self.log.poll_tasks.append(request)
        if self.poll_exc is not None:
            raise self.poll_exc
        return self.poll_response or worker_pb2.Worker.PollTasksResponse()


@dataclass
class _FakeStubFactory:
    """WorkerStubFactory that hands out per-address fake stubs."""

    stubs: dict[str, _FakeWorkerStub] = field(default_factory=dict)

    def get_stub(self, address: str) -> Any:
        if address not in self.stubs:
            # Default stub: empty responses. Tests that care preinstall stubs.
            self.stubs[address] = _FakeWorkerStub(address=address, log=_StubCallLog())
        return self.stubs[address]

    def evict(self, address: str) -> None:
        self.stubs.pop(address, None)

    def close(self) -> None:
        self.stubs.clear()


def _provider_with_stubs(stubs: dict[str, _FakeWorkerStub]) -> tuple[WorkerProvider, _FakeStubFactory]:
    factory = _FakeStubFactory(stubs=stubs)
    return WorkerProvider(stub_factory=factory), factory


def test_reconcile_workers_empty_short_circuits():
    provider, _ = _provider_with_stubs({})
    assert provider.reconcile_workers([], {}, use_reconcile_rpc=True) == []
    assert provider.reconcile_workers([], {}, use_reconcile_rpc=False) == []


# --- New wire (use_reconcile_rpc=True) ---------------------------------------


def test_reconcile_rpc_issues_one_call_per_plan_and_forwards_observations():
    """Flag=True: one ``Reconcile`` RPC per plan; ``observed`` is surfaced verbatim."""
    log = _StubCallLog()
    observation = worker_pb2.Worker.AttemptObservation(
        attempt_uid="",
        state=job_pb2.TASK_STATE_RUNNING,
        task_id=_task_id("a").to_wire(),
        attempt_id=0,
    )
    stub = _FakeWorkerStub(
        address=_W1_ADDR,
        log=log,
        reconcile_response=worker_pb2.Worker.ReconcileResponse(observed=[observation]),
    )
    provider, _ = _provider_with_stubs({_W1_ADDR: stub})

    plan = _make_plan(
        _W1,
        desired=[
            worker_pb2.Worker.DesiredAttempt(
                attempt_uid="",
                run=worker_pb2.Worker.AttemptSpec(),
                task_id=_task_id("a").to_wire(),
                attempt_id=0,
            )
        ],
    )
    results = provider.reconcile_workers(
        [plan],
        {WorkerId(_W1): _W1_ADDR},
        use_reconcile_rpc=True,
    )

    assert len(log.reconcile) == 1
    assert log.reconcile[0].worker_id == _W1
    assert log.start_tasks == [] and log.poll_tasks == []
    assert len(results) == 1
    assert results[0].worker_id == WorkerId(_W1)
    assert results[0].error is None
    assert list(results[0].observations) == [observation]


def test_reconcile_rpc_failure_returns_error_and_empty_observations():
    log = _StubCallLog()
    stub = _FakeWorkerStub(address=_W1_ADDR, log=log, reconcile_exc=RuntimeError("boom"))
    provider, _ = _provider_with_stubs({_W1_ADDR: stub})

    plan = _make_plan(_W1)
    results = provider.reconcile_workers(
        [plan],
        {WorkerId(_W1): _W1_ADDR},
        use_reconcile_rpc=True,
    )

    assert len(results) == 1
    assert results[0].error == "boom"
    assert list(results[0].observations) == []


def test_reconcile_rpc_missing_address_yields_error_without_call():
    log = _StubCallLog()
    stub = _FakeWorkerStub(address=_W1_ADDR, log=log)
    provider, _ = _provider_with_stubs({_W1_ADDR: stub})

    plan = _make_plan(_W1)
    results = provider.reconcile_workers([plan], {WorkerId(_W1): None}, use_reconcile_rpc=True)

    assert log.reconcile == []
    assert results[0].error is not None
    assert "no address" in results[0].error


# --- Legacy wire (use_reconcile_rpc=False) -----------------------------------


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


def test_legacy_wire_splits_desired_into_start_expected_stop():
    """Flag=False: run-with-spec → StartTasks; run-without-spec → only expected; stop → omitted from polls."""
    log = _StubCallLog()
    stub = _FakeWorkerStub(address=_W1_ADDR, log=log)
    provider, _ = _provider_with_stubs({_W1_ADDR: stub})

    tid_a = _task_id("a")
    tid_b = _task_id("b")
    tid_c = _task_id("c")
    plan = _make_plan(
        _W1,
        desired=[
            _desired_run(tid_a, 1, spec=_spec("img-a")),  # → StartTasks + expected
            _desired_run(tid_b, 2, spec=None),  # → only expected (cache-hit)
            _desired_stop(tid_c, 3),  # → neither
        ],
    )

    results = provider.reconcile_workers([plan], {WorkerId(_W1): _W1_ADDR}, use_reconcile_rpc=False)

    # StartTasks only for the run-with-spec entry.
    assert len(log.start_tasks) == 1
    started = list(log.start_tasks[0].tasks)
    assert len(started) == 1
    assert started[0].task_id == tid_a.to_wire()
    assert started[0].attempt_id == 1
    assert started[0].task_image == "img-a"

    # PollTasks expected list covers both run intents, never the stop.
    assert len(log.poll_tasks) == 1
    expected = list(log.poll_tasks[0].expected_tasks)
    expected_keys = {(e.task_id, e.attempt_id) for e in expected}
    assert expected_keys == {(tid_a.to_wire(), 1), (tid_b.to_wire(), 2)}

    # No Reconcile RPC.
    assert log.reconcile == []
    assert len(results) == 1


def test_legacy_wire_skips_start_tasks_when_no_run_with_spec():
    """No run-with-spec → no StartTasks; PollTasks still fires."""
    log = _StubCallLog()
    stub = _FakeWorkerStub(address=_W1_ADDR, log=log)
    provider, _ = _provider_with_stubs({_W1_ADDR: stub})

    plan = _make_plan(
        _W1,
        desired=[_desired_run(_task_id("a"), 1, spec=None)],
    )
    provider.reconcile_workers([plan], {WorkerId(_W1): _W1_ADDR}, use_reconcile_rpc=False)

    assert log.start_tasks == []
    assert len(log.poll_tasks) == 1


def test_legacy_wire_forwards_poll_updates_as_observations():
    """PollTasks response updates surface as proto AttemptObservations."""
    log = _StubCallLog()
    tid_a = _task_id("a")
    poll_resp = worker_pb2.Worker.PollTasksResponse(
        tasks=[
            job_pb2.WorkerTaskStatus(
                task_id=tid_a.to_wire(),
                attempt_id=1,
                state=job_pb2.TASK_STATE_RUNNING,
            ),
        ]
    )
    stub = _FakeWorkerStub(address=_W1_ADDR, log=log, poll_response=poll_resp)
    provider, _ = _provider_with_stubs({_W1_ADDR: stub})

    plan = _make_plan(_W1, desired=[_desired_run(tid_a, 1, spec=None)])
    results = provider.reconcile_workers([plan], {WorkerId(_W1): _W1_ADDR}, use_reconcile_rpc=False)

    assert len(results) == 1
    assert results[0].error is None
    observations = list(results[0].observations)
    assert len(observations) == 1
    assert observations[0].task_id == tid_a.to_wire()
    assert observations[0].attempt_id == 1
    assert observations[0].state == job_pb2.TASK_STATE_RUNNING


def test_legacy_wire_rejected_start_ack_emits_worker_failed_observation():
    """Non-accepted TaskAck → synthetic WORKER_FAILED observation."""
    log = _StubCallLog()
    tid_a = _task_id("a")
    start_resp = worker_pb2.Worker.StartTasksResponse(
        acks=[
            worker_pb2.Worker.TaskAck(task_id=tid_a.to_wire(), accepted=False, error="bundle missing"),
        ]
    )
    stub = _FakeWorkerStub(address=_W1_ADDR, log=log, start_response=start_resp)
    provider, _ = _provider_with_stubs({_W1_ADDR: stub})

    plan = _make_plan(_W1, desired=[_desired_run(tid_a, 5, spec=_spec("img"))])
    results = provider.reconcile_workers([plan], {WorkerId(_W1): _W1_ADDR}, use_reconcile_rpc=False)

    assert len(results) == 1
    observations = list(results[0].observations)
    assert len(observations) == 1
    assert observations[0].state == job_pb2.TASK_STATE_WORKER_FAILED
    assert "bundle missing" in observations[0].error
    assert observations[0].task_id == tid_a.to_wire()
    assert observations[0].attempt_id == 5


def test_legacy_wire_accepted_acks_do_not_emit_observations():
    log = _StubCallLog()
    tid_a = _task_id("a")
    start_resp = worker_pb2.Worker.StartTasksResponse(
        acks=[worker_pb2.Worker.TaskAck(task_id=tid_a.to_wire(), accepted=True)]
    )
    stub = _FakeWorkerStub(address=_W1_ADDR, log=log, start_response=start_resp)
    provider, _ = _provider_with_stubs({_W1_ADDR: stub})

    plan = _make_plan(_W1, desired=[_desired_run(tid_a, 1, spec=_spec("img"))])
    results = provider.reconcile_workers([plan], {WorkerId(_W1): _W1_ADDR}, use_reconcile_rpc=False)

    assert results[0].observations == []


def test_legacy_wire_start_failure_surfaces_as_error():
    """StartTasks exception → ReconcileResult.error set, no observations."""
    log = _StubCallLog()
    stub = _FakeWorkerStub(address=_W1_ADDR, log=log, start_exc=RuntimeError("connection refused"))
    provider, _ = _provider_with_stubs({_W1_ADDR: stub})

    plan = _make_plan(_W1, desired=[_desired_run(_task_id("a"), 1, spec=_spec("img"))])
    results = provider.reconcile_workers([plan], {WorkerId(_W1): _W1_ADDR}, use_reconcile_rpc=False)

    assert results[0].error == "connection refused"
    assert list(results[0].observations) == []


def test_legacy_wire_poll_failure_is_logged_not_fatal():
    """PollTasks failure: observations are empty but error stays None (logged at debug)."""
    log = _StubCallLog()
    stub = _FakeWorkerStub(address=_W1_ADDR, log=log, poll_exc=RuntimeError("timeout"))
    provider, _ = _provider_with_stubs({_W1_ADDR: stub})

    plan = _make_plan(_W1, desired=[_desired_run(_task_id("a"), 1, spec=None)])
    results = provider.reconcile_workers([plan], {WorkerId(_W1): _W1_ADDR}, use_reconcile_rpc=False)

    assert results[0].error is None
    assert list(results[0].observations) == []


# ===========================================================================
# Section 3: apply (transitions) + e2e (controller tick)
# ===========================================================================


def _setup_running_task(
    state: ControllerTransitions,
    worker_id: str,
    job_name: str = "test-job",
) -> tuple[JobName, int]:
    """Register worker, submit job, dispatch, drive to RUNNING."""
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
    """Register worker, submit job, queue assignment (no heartbeat → stays ASSIGNED)."""
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


# --- apply_reconcile_observations / apply_reconcile_failure ----------------


def test_succeeded_observation_transitions_attempt_and_recomputes_job():
    with make_controller_state() as state:
        task_id, attempt_id = _setup_running_task(state, _W1)
        plan = _make_plan(_W1)
        observations = [_obs(task_id, attempt_id, job_pb2.TASK_STATE_SUCCEEDED, exit_code=0)]
        with state._db.transaction() as cur:
            state.apply_reconcile_observations(cur, plan, observations, _NOW)

        task = query_task(state, task_id)
        attempt = query_attempt(state, task_id, attempt_id)
        assert task is not None
        assert task.state == job_pb2.TASK_STATE_SUCCEEDED
        assert attempt is not None
        assert attempt.finished_at_ms is not None
        assert attempt.state == job_pb2.TASK_STATE_SUCCEEDED

        job = query_job(state, task.job_id)
        assert job is not None
        assert job.state == job_pb2.JOB_STATE_SUCCEEDED


def test_failed_observation_transitions_attempt_and_fires_cascades():
    with make_controller_state() as state:
        task_id, attempt_id = _setup_running_task(state, _W1)
        plan = _make_plan(_W1)
        observations = [_obs(task_id, attempt_id, job_pb2.TASK_STATE_FAILED, error="segfault")]
        with state._db.transaction() as cur:
            state.apply_reconcile_observations(cur, plan, observations, _NOW)

        task = query_task(state, task_id)
        assert task is not None
        assert task.state == job_pb2.TASK_STATE_FAILED

        job = query_job(state, task.job_id)
        assert job is not None
        assert job.state == job_pb2.JOB_STATE_FAILED


def test_missing_observation_fails_attempt_with_worker_lost_spec():
    with make_controller_state() as state:
        task_id, attempt_id = _setup_running_task(state, _W1)
        plan = _make_plan(_W1)
        observations = [_obs(task_id, attempt_id, job_pb2.TASK_STATE_MISSING)]
        with state._db.transaction() as cur:
            state.apply_reconcile_observations(cur, plan, observations, _NOW)

        task = query_task(state, task_id)
        assert task is not None
        assert task.state == job_pb2.TASK_STATE_FAILED
        assert task.error == "worker_lost_spec"

        attempt = query_attempt(state, task_id, attempt_id)
        assert attempt is not None
        assert attempt.state == job_pb2.TASK_STATE_FAILED
        assert attempt.error == "worker_lost_spec"


def test_missing_observation_on_building_task_fails_with_worker_lost_spec():
    """BUILDING + MISSING → FAILED("worker_lost_spec"): spec lost before task started."""
    with make_controller_state() as state:
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

        assert query_task(state, task_id).state == job_pb2.TASK_STATE_BUILDING

        plan = _make_plan(_W1)
        observations = [_obs(task_id, attempt_id, job_pb2.TASK_STATE_MISSING)]
        with state._db.transaction() as cur:
            state.apply_reconcile_observations(cur, plan, observations, _NOW)

        task = query_task(state, task_id)
        assert task.state == job_pb2.TASK_STATE_FAILED
        assert task.error == "worker_lost_spec"


def test_duplicate_terminal_observation_does_not_overwrite_finished_at():
    with make_controller_state() as state:
        task_id, attempt_id = _setup_running_task(state, _W1)
        plan = _make_plan(_W1)
        observations = [_obs(task_id, attempt_id, job_pb2.TASK_STATE_SUCCEEDED, exit_code=0)]

        with state._db.transaction() as cur:
            state.apply_reconcile_observations(cur, plan, observations, _NOW)
        attempt_after_first = query_attempt(state, task_id, attempt_id)
        assert attempt_after_first is not None
        finished_at_first = attempt_after_first.finished_at_ms

        with state._db.transaction() as cur:
            state.apply_reconcile_observations(cur, plan, observations, _NOW)
        attempt_after_second = query_attempt(state, task_id, attempt_id)
        assert attempt_after_second is not None
        assert attempt_after_second.finished_at_ms == finished_at_first


def test_stale_running_observation_does_not_revive_cancelled_task():
    """RUNNING after cancellation must not roll the task forward."""
    with make_controller_state() as state:
        task_id, attempt_id = _setup_running_task(state, _W1)

        with state._db.transaction() as cur:
            task_row = query_task(state, task_id)
            state.cancel_job(cur, task_row.job_id, "user_cancel")

        assert query_task(state, task_id).state == job_pb2.TASK_STATE_KILLED

        plan = _make_plan(_W1)
        observations = [_obs(task_id, attempt_id, job_pb2.TASK_STATE_RUNNING)]
        with state._db.transaction() as cur:
            state.apply_reconcile_observations(cur, plan, observations, _NOW)

        assert query_task(state, task_id).state == job_pb2.TASK_STATE_KILLED


def test_rpc_error_no_state_change_for_running_task():
    """RPC failure does not mutate non-ASSIGNED tasks."""
    with make_controller_state() as state:
        task_id, attempt_id = _setup_running_task(state, _W1)
        plan = _make_plan(
            _W1,
            desired=[_desired_run(task_id, attempt_id, spec=None)],
        )
        with state._db.transaction() as cur:
            state.apply_reconcile_failure(cur, plan, "connection refused", _NOW)

        assert query_task(state, task_id).state == job_pb2.TASK_STATE_RUNNING


def test_rpc_error_applies_worker_failed_to_assigned_tasks():
    """RPC failure on an ASSIGNED dispatch attempt synthesizes WORKER_FAILED."""
    with make_controller_state() as state:
        task_id, attempt_id = _setup_assigned_task(state, _W1)
        plan = _make_plan(
            _W1,
            desired=[_desired_run(task_id, attempt_id, spec=_spec())],
        )
        with state._db.transaction() as cur:
            state.apply_reconcile_failure(cur, plan, "timeout", _NOW)

        task = query_task(state, task_id)
        assert task is not None
        # Synthetic WORKER_FAILED bounces the task back to PENDING so it can be re-dispatched.
        assert task.state == job_pb2.TASK_STATE_PENDING


def test_rpc_error_no_cascades_when_no_assigned_tasks():
    with make_controller_state() as state:
        register_worker(state, _W1, f"{_W1}:8080", make_worker_metadata())
        tasks = submit_job(state, "idle-job", make_job_request(name="idle-job"))
        task_id = tasks[0].task_id

        plan = _make_plan(_W1)
        with state._db.transaction() as cur:
            result = state.apply_reconcile_failure(cur, plan, "connection refused", _NOW)

        assert result.tasks_to_kill == set()
        assert query_task(state, task_id).state == job_pb2.TASK_STATE_PENDING


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
            state.apply_reconcile_observations(cur, plan, observations, _NOW)

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
    """``apply_reconcile_observations`` on an unknown worker is a no-op."""
    with make_controller_state() as state:
        plan = _make_plan("ghost-worker")
        with state._db.transaction() as cur:
            result = state.apply_reconcile_observations(cur, plan, [], _NOW)

        assert result.tasks_to_kill == set()
        assert result.task_kill_workers == {}


# --- End-to-end: full controller tick over both wires ----------------------


@dataclass
class _ScriptedProvider:
    """In-process TaskProvider whose ``reconcile_workers`` returns scripted observations.

    Records every call so tests can assert that the right wire was used and
    that the right plans were dispatched. Each tick consumes one ``script``
    entry (a callable taking the plan and returning a list of observations).
    """

    use_reconcile_rpc_expected: bool
    script: list[Any] = field(default_factory=list)
    calls: list[tuple[list[WorkerReconcilePlan], dict, bool]] = field(default_factory=list)

    def get_process_status(self, *_args, **_kwargs):
        raise NotImplementedError("scripted provider: get_process_status not used")

    def on_worker_failed(self, *_args, **_kwargs):
        pass

    def profile_task(self, *_args, **_kwargs):
        raise NotImplementedError("scripted provider: profile_task not used")

    def ping_workers(self, workers):
        return []

    def reconcile_workers(self, plans, addresses, *, use_reconcile_rpc):
        self.calls.append((list(plans), dict(addresses), use_reconcile_rpc))
        assert use_reconcile_rpc == self.use_reconcile_rpc_expected, (
            f"expected use_reconcile_rpc={self.use_reconcile_rpc_expected}, "
            f"controller called with {use_reconcile_rpc}"
        )
        tick = len(self.calls) - 1
        responder = self.script[tick] if tick < len(self.script) else (lambda plan: [])
        return [ReconcileResult(worker_id=p.worker_id, observations=responder(p), error=None) for p in plans]

    def close(self):
        pass


def _converge_to_succeeded(
    flag: bool,
    *,
    make_controller,
) -> None:
    """One job, one worker, one task: tick three times to reach SUCCEEDED."""

    def tick1(_plan):
        # ASSIGNED → BUILDING. No observation needed; worker hasn't started yet.
        return []

    def tick2(plan):
        # BUILDING → RUNNING.
        return [
            worker_pb2.Worker.AttemptObservation(
                attempt_uid="",
                state=job_pb2.TASK_STATE_RUNNING,
                task_id=d.task_id,
                attempt_id=d.attempt_id,
            )
            for d in plan.request.desired
            if d.HasField("run")
        ]

    def tick3(plan):
        return [
            worker_pb2.Worker.AttemptObservation(
                attempt_uid="",
                state=job_pb2.TASK_STATE_SUCCEEDED,
                exit_code=0,
                task_id=d.task_id,
                attempt_id=d.attempt_id,
            )
            for d in plan.request.desired
            if d.HasField("run")
        ]

    provider = _ScriptedProvider(use_reconcile_rpc_expected=flag, script=[tick1, tick2, tick3])
    ctrl = make_controller(provider=provider, reconcile_rpc_enabled=flag)
    state = ctrl._transitions

    wid = register_worker(state, _W1, _W1_ADDR, make_worker_metadata())
    tasks = submit_job(state, "e2e-job", make_job_request(name="e2e-job"))
    task_id = tasks[0].task_id

    with state._db.transaction() as cur:
        state.queue_assignments(cur, [Assignment(task_id=task_id, worker_id=wid)])

    # Tick 1: ASSIGNED — controller dispatches the inline spec.
    ctrl._reconcile_worker_batch()
    plans1, _addrs1, _ = provider.calls[0]
    assert len(plans1) == 1
    tick1_desired = list(plans1[0].request.desired)
    assert len(tick1_desired) == 1
    assert tick1_desired[0].HasField("run")
    assert tick1_desired[0].run.HasField("request"), "first tick should carry inline spec"

    # Tick 2: ASSIGNED (no observation moved it forward) — still RUN intent.
    ctrl._reconcile_worker_batch()
    # Worker reports RUNNING; controller transitions task to RUNNING.
    assert query_task(state, task_id).state == job_pb2.TASK_STATE_RUNNING

    # Tick 3: RUNNING — run intent without inline spec (cache-hit invariant).
    ctrl._reconcile_worker_batch()
    plans3, _addrs3, _ = provider.calls[2]
    tick3_desired = list(plans3[0].request.desired)
    assert tick3_desired and tick3_desired[0].HasField("run")
    assert not tick3_desired[0].run.HasField("request"), "subsequent ticks must not carry inline spec"

    task_final = query_task(state, task_id)
    assert task_final.state == job_pb2.TASK_STATE_SUCCEEDED
    assert query_job(state, task_final.job_id).state == job_pb2.JOB_STATE_SUCCEEDED


@pytest.mark.parametrize("flag", [True, False])
def test_e2e_converges_to_succeeded_through_both_wires(flag, make_controller):
    """Full ASSIGNED → RUNNING → SUCCEEDED convergence over either wire."""
    _converge_to_succeeded(flag, make_controller=make_controller)


def test_e2e_missing_observation_fails_attempt_with_worker_lost_spec(make_controller):
    """End-to-end MISSING cascade: dispatch → MISSING → FAILED("worker_lost_spec")."""

    def tick_dispatch(_plan):
        # First reconcile tick: ASSIGNED dispatch. No observation yet.
        return []

    def tick_missing(plan):
        return [
            worker_pb2.Worker.AttemptObservation(
                attempt_uid="",
                state=job_pb2.TASK_STATE_MISSING,
                task_id=d.task_id,
                attempt_id=d.attempt_id,
            )
            for d in plan.request.desired
            if d.HasField("run")
        ]

    provider = _ScriptedProvider(use_reconcile_rpc_expected=True, script=[tick_dispatch, tick_missing])
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
