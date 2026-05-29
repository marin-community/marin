# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidated tests for the controller reconcile pipeline.

Three layers, exercised in order:

1. **Pure compute** — ``reconcile_workers`` builds one ``ReconcileRequest``
   proto per worker from a ``ReconcileInputs`` snapshot. No DB.
2. **Wire & dispatch** — ``WorkerProvider.reconcile_workers`` fans out via a
   fake stub factory and synthesizes ``ReconcileResult.observations``.
3. **Apply + e2e** — ``apply_reconcile`` against real SQLite DB state, plus a
   handful of end-to-end convergence ticks driven through
   ``Controller._reconcile_worker_batch``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest
from iris.cluster.controller import ops, writes
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.ops.worker import apply_reconcile_observations as apply_reconcile
from iris.cluster.controller.reconcile.loader import load_closed_snapshot
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.reconcile.worker import (
    ReconcileInputs,
    ReconcileResult,
    ReconcileRow,
    WorkerReconcilePlan,
    reconcile_workers,
)
from iris.cluster.controller.reconcile.worker import (
    observations_to_updates as worker_observations_to_updates,
)
from iris.cluster.controller.schema import task_attempts_table
from iris.cluster.controller.worker_provider import WorkerProvider
from iris.cluster.types import AttemptUid, JobName, WorkerId
from iris.rpc import job_pb2, worker_pb2
from rigging.timing import Timestamp
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from tests.cluster.controller._test_support import ControllerTestState
from tests.cluster.controller.transition_driver import WorkerTaskUpdates, apply_task_observations

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
    attempt_uid: str,
    state: int,
    *,
    exit_code: int | None = None,
    error: str | None = None,
) -> worker_pb2.Worker.AttemptObservation:
    kwargs: dict = {
        "attempt_uid": attempt_uid,
        "state": state,
    }
    if exit_code is not None:
        kwargs["exit_code"] = exit_code
    if error is not None:
        kwargs["error"] = error
    return worker_pb2.Worker.AttemptObservation(**kwargs)


def _desired_run(attempt_uid: str, *, spec: job_pb2.RunTaskRequest | None = None):
    return worker_pb2.Worker.DesiredAttempt(
        attempt_uid=attempt_uid,
        run=worker_pb2.Worker.AttemptSpec(request=spec) if spec is not None else worker_pb2.Worker.AttemptSpec(),
    )


def _desired_stop(attempt_uid: str, *, reason=worker_pb2.Worker.STOP_REASON_CANCELLED):
    return worker_pb2.Worker.DesiredAttempt(
        attempt_uid=attempt_uid,
        stop=reason,
    )


# ===========================================================================
# Section 0: attempt_uid minting (writes.insert_attempt)
# ===========================================================================
#
# insert_attempt is the single task_attempts INSERT chokepoint: it mints the
# controller-side attempt_uid, writes the row, and returns the uid. A
# UNIQUE-index collision on attempt_uid is re-minted; any other integrity
# error propagates.


def _submit_pending_task(state: ControllerTestState, job: str = "mint-job") -> JobName:
    """Submit a one-task job and return its task_id (task stays PENDING, no attempt)."""
    tasks = submit_job(state, job, make_job_request(name=job))
    return tasks[0].task_id


def _insert_attempt(state: ControllerTestState, task_id: JobName, attempt_id: int) -> str:
    with state._db.transaction() as cur:
        return writes.insert_attempt(
            cur,
            task_id=task_id,
            attempt_id=attempt_id,
            worker_id=None,
            state=job_pb2.TASK_STATE_ASSIGNED,
            created_at_ms=1_000,
        )


def test_insert_attempt_mints_16_hex_uid_and_persists_it():
    """insert_attempt returns a 16-hex attempt_uid and stores it on the row."""
    with make_controller_state() as state:
        task_id = _submit_pending_task(state)
        uid = _insert_attempt(state, task_id, attempt_id=0)

        assert len(uid) == 16
        assert all(c in "0123456789abcdef" for c in uid)
        # Returned uid is exactly what landed in the row.
        assert _attempt_uid(state, task_id, 0) == uid


def test_insert_attempt_mints_distinct_uids_for_distinct_attempts():
    """Distinct insert_attempt calls yield distinct attempt_uids."""
    with make_controller_state() as state:
        task_a = _submit_pending_task(state, job="mint-a")
        task_b = _submit_pending_task(state, job="mint-b")

        uid_a = _insert_attempt(state, task_a, attempt_id=0)
        uid_b = _insert_attempt(state, task_b, attempt_id=0)

        assert uid_a != uid_b


def test_insert_attempt_remints_on_uid_collision(monkeypatch):
    """A UNIQUE collision on attempt_uid is re-minted; the row lands with the fresh value.

    secrets.token_hex is the mint seam. Force it to hand back an
    already-stored uid once, then a fresh value, and assert the second attempt
    row lands with the fresh uid rather than aborting the transaction.
    """
    with make_controller_state() as state:
        task_a = _submit_pending_task(state, job="collide-a")
        task_b = _submit_pending_task(state, job="collide-b")

        first_uid = _insert_attempt(state, task_a, attempt_id=0)

        # token_hex hands back the existing uid (collision), then a fresh one.
        scripted = iter([first_uid, "cafebabecafebabe"])
        monkeypatch.setattr("iris.cluster.controller.writes.secrets.token_hex", lambda _n: next(scripted))

        second_uid = _insert_attempt(state, task_b, attempt_id=0)

        assert second_uid == "cafebabecafebabe"
        assert _attempt_uid(state, task_b, 0) == "cafebabecafebabe"
        # The collided first value still belongs to task_a, untouched.
        assert _attempt_uid(state, task_a, 0) == first_uid


def test_insert_attempt_exhausts_retries_when_every_mint_collides(monkeypatch):
    """If every minted uid collides, insert_attempt raises rather than looping forever."""
    with make_controller_state() as state:
        task_a = _submit_pending_task(state, job="exhaust-a")
        task_b = _submit_pending_task(state, job="exhaust-b")

        first_uid = _insert_attempt(state, task_a, attempt_id=0)
        # Every mint returns the already-stored uid.
        monkeypatch.setattr("iris.cluster.controller.writes.secrets.token_hex", lambda _n: first_uid)

        with pytest.raises(RuntimeError, match="exhausted attempt_uid retries"):
            _insert_attempt(state, task_b, attempt_id=0)


def test_insert_attempt_propagates_non_uid_integrity_error():
    """A composite-PK collision (same task_id, attempt_id) is a real bug — it must propagate.

    The retry loop is scoped to attempt_uid collisions only; re-inserting an
    existing (task_id, attempt_id) trips the PRIMARY KEY constraint and must
    raise instead of being silently retried.
    """
    with make_controller_state() as state:
        task_id = _submit_pending_task(state, job="pk-collide")
        _insert_attempt(state, task_id, attempt_id=0)

        with pytest.raises(IntegrityError):
            _insert_attempt(state, task_id, attempt_id=0)


# ===========================================================================
# Section 1: pure-compute (reconcile_worker)
# ===========================================================================


def _row(
    task_state: int,
    *,
    task_id: str = "task-a",
    attempt_id: int = 0,
    job: str = "job-a",
    attempt_uid: str = "deadbeefdeadbeef",
    attempt_state: int = job_pb2.TASK_STATE_PENDING,
) -> ReconcileRow:
    return ReconcileRow(
        worker_id=WorkerId(_W1),
        task_id=_task_id(task_id),
        attempt_id=attempt_id,
        task_state=task_state,
        attempt_state=attempt_state,
        job_id=_job_id(job),
        attempt_uid=AttemptUid(attempt_uid),
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
    assert desired.attempt_uid == row.attempt_uid
    assert desired.HasField("run")
    assert desired.run.HasField("request")
    # The inline RunTaskRequest is stamped with the routing key: the worker's
    # submit_task reads attempt_uid from the request.
    assert desired.run.request.task_image == "custom-image"
    assert desired.run.request.task_id == row.task_id.to_wire()
    assert desired.run.request.attempt_id == 7
    assert desired.run.request.attempt_uid == row.attempt_uid


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
    row = _row(task_state, attempt_id=3)
    plan = _plan_for([row])
    assert len(plan.request.desired) == 1
    desired = plan.request.desired[0]
    assert desired.HasField("run")
    assert not desired.run.HasField("request")
    assert desired.attempt_uid == row.attempt_uid


@pytest.mark.parametrize(
    "task_state,expected_reason",
    [
        # Execution timeout marks the task FAILED; the cosched cascade and the
        # other controller-induced terminals map to JOB_TERMINATED.
        (job_pb2.TASK_STATE_FAILED, worker_pb2.Worker.STOP_REASON_TASK_TIMEOUT),
        (job_pb2.TASK_STATE_COSCHED_FAILED, worker_pb2.Worker.STOP_REASON_JOB_TERMINATED),
        (job_pb2.TASK_STATE_WORKER_FAILED, worker_pb2.Worker.STOP_REASON_JOB_TERMINATED),
        (job_pb2.TASK_STATE_UNSCHEDULABLE, worker_pb2.Worker.STOP_REASON_JOB_TERMINATED),
        (job_pb2.TASK_STATE_SUCCEEDED, worker_pb2.Worker.STOP_REASON_JOB_TERMINATED),
    ],
)
def test_reconcile_worker_controller_terminal_with_terminal_attempt_emits_stop(task_state, expected_reason):
    """A terminal task whose attempt is itself terminal is a controller-induced
    terminal where the worker may still be running the process. The planner must
    emit a 'stop' (not a no-op 'run') so the worker tears the process down."""
    row = _row(task_state, attempt_id=4, attempt_state=task_state)
    plan = _plan_for([row])

    assert len(plan.request.desired) == 1
    desired = plan.request.desired[0]
    assert desired.HasField("stop")
    assert desired.stop == expected_reason
    assert desired.attempt_uid == row.attempt_uid


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
def test_reconcile_worker_stranded_terminal_with_live_attempt_emits_run(task_state):
    """A terminal task whose attempt is NOT yet terminal is a stranded attempt:
    re-poll (run, no spec) so the worker re-reports its real status or the
    daemon synthesizes MISSING. Switching it to 'stop' would drop that recovery."""
    row = _row(task_state, attempt_id=4, attempt_state=job_pb2.TASK_STATE_RUNNING)
    plan = _plan_for([row])

    assert len(plan.request.desired) == 1
    desired = plan.request.desired[0]
    assert desired.HasField("run")
    assert not desired.run.HasField("request")
    assert desired.attempt_uid == row.attempt_uid


@pytest.mark.parametrize(
    "task_state,expected_reason",
    [
        (job_pb2.TASK_STATE_KILLED, worker_pb2.Worker.STOP_REASON_CANCELLED),
        (job_pb2.TASK_STATE_PREEMPTED, worker_pb2.Worker.STOP_REASON_PREEMPTED),
    ],
)
def test_reconcile_worker_stop_states_emit_stop_with_reason(task_state, expected_reason):
    row = _row(task_state, attempt_id=2)
    plan = _plan_for([row])
    assert len(plan.request.desired) == 1
    desired = plan.request.desired[0]
    assert desired.HasField("stop")
    assert desired.stop == expected_reason
    assert desired.attempt_uid == row.attempt_uid


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
        _row(job_pb2.TASK_STATE_ASSIGNED, task_id="a", attempt_id=1, job="j1", attempt_uid="aaaaaaaaaaaaaaaa"),
        _row(job_pb2.TASK_STATE_RUNNING, task_id="b", attempt_id=2, job="j2", attempt_uid="bbbbbbbbbbbbbbbb"),
        _row(job_pb2.TASK_STATE_KILLED, task_id="c", attempt_id=3, job="j3", attempt_uid="cccccccccccccccc"),
        _row(job_pb2.TASK_STATE_SUCCEEDED, task_id="d", attempt_id=4, job="j4", attempt_uid="dddddddddddddddd"),
    ]
    plan = _plan_for(rows, job_specs={_job_id("j1"): _spec("img-j1")})

    by_uid = {d.attempt_uid: d for d in plan.request.desired}
    assert set(by_uid) == {row.attempt_uid for row in rows}
    assert by_uid["aaaaaaaaaaaaaaaa"].run.HasField("request")
    assert not by_uid["bbbbbbbbbbbbbbbb"].run.HasField("request")
    assert by_uid["cccccccccccccccc"].stop == worker_pb2.Worker.STOP_REASON_CANCELLED
    assert not by_uid["dddddddddddddddd"].run.HasField("request")


# --- attempt_uid is stamped on every emit site ------------------------------


@pytest.mark.parametrize(
    "task_state",
    [
        job_pb2.TASK_STATE_ASSIGNED,  # assign/run emit site
        job_pb2.TASK_STATE_BUILDING,  # executing emit site
        job_pb2.TASK_STATE_RUNNING,  # executing emit site
        job_pb2.TASK_STATE_KILLED,  # killed → stop emit site
        job_pb2.TASK_STATE_PREEMPTED,  # preempted → stop emit site
        job_pb2.TASK_STATE_SUCCEEDED,  # terminal-expected emit site
        job_pb2.TASK_STATE_FAILED,  # terminal-expected emit site
    ],
)
def test_reconcile_worker_stamps_attempt_uid_on_every_emit_site(task_state):
    """Every ``DesiredAttempt`` _reconcile_worker emits carries the row's attempt_uid.

    All five emit branches (assign/run, executing, killed, preempted,
    terminal-expected) used to hardcode ``attempt_uid=""``; Phase 2 fills them
    from ``ReconcileRow.attempt_uid``.
    """
    uid = "00112233aabbccdd"
    row = _row(task_state, attempt_id=9, job="job-uid", attempt_uid=uid)
    # ASSIGNED needs a cached spec or the row is dropped before any emit.
    job_specs = {_job_id("job-uid"): _spec()} if task_state == job_pb2.TASK_STATE_ASSIGNED else None
    plan = _plan_for([row], job_specs=job_specs)

    assert len(plan.request.desired) == 1
    assert plan.request.desired[0].attempt_uid == uid


def test_reconcile_worker_emits_distinct_uids_for_distinct_rows():
    """Each row's own attempt_uid lands on its own DesiredAttempt — no cross-talk."""
    rows = [
        _row(job_pb2.TASK_STATE_ASSIGNED, task_id="a", attempt_id=1, job="j1", attempt_uid="1111111111111111"),
        _row(job_pb2.TASK_STATE_RUNNING, task_id="b", attempt_id=2, job="j2", attempt_uid="2222222222222222"),
        _row(job_pb2.TASK_STATE_KILLED, task_id="c", attempt_id=3, job="j3", attempt_uid="3333333333333333"),
    ]
    plan = _plan_for(rows, job_specs={_job_id("j1"): _spec()})

    uids = {d.attempt_uid for d in plan.request.desired}
    assert uids == {"1111111111111111", "2222222222222222", "3333333333333333"}


# ===========================================================================
# Section 2: wire & dispatch (WorkerProvider.reconcile_workers)
# ===========================================================================


@dataclass
class _FakeWorkerStub:
    """In-process WorkerServiceClient stand-in.

    Records every reconcile call and returns canned responses (or raises)
    according to pre-configured fields.
    """

    address: str
    reconcile_calls: list[worker_pb2.Worker.ReconcileRequest] = field(default_factory=list)
    reconcile_response: worker_pb2.Worker.ReconcileResponse | None = None
    reconcile_exc: Exception | None = None

    async def reconcile(self, request, *, timeout_ms=None):
        del timeout_ms
        self.reconcile_calls.append(request)
        if self.reconcile_exc is not None:
            raise self.reconcile_exc
        return self.reconcile_response or worker_pb2.Worker.ReconcileResponse()


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


def _reconcile_one(provider: WorkerProvider, plan: WorkerReconcilePlan, *, address: str = _W1_ADDR):
    return provider.reconcile_workers([plan], {WorkerId(_W1): address})


def test_reconcile_workers_empty_short_circuits():
    provider, _ = _provider_with_stub()
    assert provider.reconcile_workers([], {}) == []


def test_reconcile_rpc_forwards_observations():
    """One Reconcile RPC per plan; observed observations surface verbatim."""
    observation = _obs("uid-a", job_pb2.TASK_STATE_RUNNING)
    stub = _FakeWorkerStub(
        address=_W1_ADDR,
        reconcile_response=worker_pb2.Worker.ReconcileResponse(observed=[observation]),
    )
    provider, _ = _provider_with_stub(stub)
    plan = _make_plan(_W1, desired=[_desired_run("uid-a")])

    results = _reconcile_one(provider, plan)

    assert len(stub.reconcile_calls) == 1
    assert stub.reconcile_calls[0].worker_id == _W1
    assert len(results) == 1
    assert results[0].worker_id == WorkerId(_W1)
    assert results[0].error is None
    assert list(results[0].observations) == [observation]


def test_reconcile_rpc_failure_returns_error_and_empty_observations():
    stub = _FakeWorkerStub(address=_W1_ADDR, reconcile_exc=RuntimeError("boom"))
    provider, _ = _provider_with_stub(stub)

    results = _reconcile_one(provider, _make_plan(_W1))

    assert results[0].error == "boom"
    assert list(results[0].observations) == []


# ===========================================================================
# Section 3: apply (transitions) + e2e (controller tick)
# ===========================================================================


def _attempt_uid(state: ControllerTestState, task_id: JobName, attempt_id: int) -> str:
    """Read the controller-minted attempt_uid for one attempt row."""
    with state._db.read_snapshot() as tx:
        row = tx.execute(
            select(task_attempts_table.c.attempt_uid).where(
                task_attempts_table.c.task_id == task_id,
                task_attempts_table.c.attempt_id == attempt_id,
            )
        ).first()
    assert row is not None
    return row.attempt_uid


def _setup_running_task(state: ControllerTestState, worker_id: str = _W1) -> tuple[JobName, int, str]:
    """Register worker, submit job, dispatch, drive to RUNNING.

    Returns ``(task_id, attempt_id, attempt_uid)``.
    """
    wid = WorkerId(worker_id)
    register_worker(state, worker_id, f"{worker_id}:8080", make_worker_metadata())
    tasks = submit_job(state, "test-job", make_job_request(name="test-job"))
    task_row = tasks[0]
    dispatch_task(state, task_row, wid)
    refreshed = query_task(state, task_row.task_id)
    assert refreshed is not None
    uid = _attempt_uid(state, task_row.task_id, refreshed.current_attempt_id)
    return task_row.task_id, refreshed.current_attempt_id, uid


def _setup_assigned_task(state: ControllerTestState, worker_id: str = _W1) -> tuple[JobName, int, str]:
    """Register worker, submit job, queue assignment (no heartbeat → stays ASSIGNED).

    Returns ``(task_id, attempt_id, attempt_uid)``.
    """
    wid = WorkerId(worker_id)
    register_worker(state, worker_id, f"{worker_id}:8080", make_worker_metadata())
    tasks = submit_job(state, "test-job", make_job_request(name="test-job"))
    task_row = tasks[0]
    with state._db.transaction() as cur:
        ops.task.queue_assignments(cur, [Assignment(task_id=task_row.task_id, worker_id=wid)], health=state._health)
    refreshed = query_task(state, task_row.task_id)
    assert refreshed is not None
    assert refreshed.state == job_pb2.TASK_STATE_ASSIGNED
    uid = _attempt_uid(state, task_row.task_id, refreshed.current_attempt_id)
    return task_row.task_id, refreshed.current_attempt_id, uid


def _apply_observations(
    state: ControllerTestState,
    worker_id: str,
    observations: list[worker_pb2.Worker.AttemptObservation],
    *,
    plan: WorkerReconcilePlan | None = None,
):
    """Apply observations under a plan that requests them by default.

    The controller drops observations whose ``attempt_uid`` is not in the
    per-worker plan (see ``_filter_observations_to_plan``). Tests focused on
    observation semantics get a plan that asks for every supplied observation;
    tests that need to exercise the filter pass an explicit ``plan``.
    """
    if plan is None:
        desired = [
            worker_pb2.Worker.DesiredAttempt(
                attempt_uid=obs.attempt_uid,
                run=worker_pb2.Worker.AttemptSpec(),
            )
            for obs in observations
            if obs.attempt_uid
        ]
        plan = _make_plan(worker_id, desired=desired)
    result = ReconcileResult(worker_id=WorkerId(worker_id), observations=observations, error=None)
    with state._db.transaction() as cur:
        return apply_reconcile(
            cur,
            {plan.worker_id: plan},
            [result],
            health=state._health,
            endpoints=state._endpoints,
            now=_NOW,
        )


def _apply_failure(
    state: ControllerTestState,
    worker_id: str,
    plan: WorkerReconcilePlan,
    error: str,
):
    result = ReconcileResult(worker_id=WorkerId(worker_id), observations=[], error=error)
    with state._db.transaction() as cur:
        return apply_reconcile(
            cur,
            {plan.worker_id: plan},
            [result],
            health=state._health,
            endpoints=state._endpoints,
            now=_NOW,
        )


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
        task_id, attempt_id, uid = _setup_running_task(state)
        _apply_observations(state, _W1, [_obs(uid, obs_state, **attempt_kwargs)])

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
        task_id, attempt_id, uid = _setup_running_task(state)
        _apply_observations(state, _W1, [_obs(uid, job_pb2.TASK_STATE_MISSING)])

        task = query_task(state, task_id)
        attempt = query_attempt(state, task_id, attempt_id)
        assert task is not None and attempt is not None
        assert task.state == job_pb2.TASK_STATE_FAILED
        assert task.error == "worker_lost_spec"
        assert attempt.state == job_pb2.TASK_STATE_FAILED
        assert attempt.error == "worker_lost_spec"


def test_duplicate_terminal_observation_does_not_overwrite_finished_at():
    with make_controller_state() as state:
        task_id, attempt_id, uid = _setup_running_task(state)
        observations = [_obs(uid, job_pb2.TASK_STATE_SUCCEEDED, exit_code=0)]

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
        task_id, _attempt_id, uid = _setup_running_task(state)
        with state._db.transaction() as cur:
            task_row = query_task(state, task_id)
            assert task_row is not None
            ops.job.cancel(
                cur, job_id=task_row.job_id, reason="user_cancel", endpoints=state._endpoints, health=state._health
            )
        assert query_task(state, task_id).state == job_pb2.TASK_STATE_KILLED

        _apply_observations(state, _W1, [_obs(uid, job_pb2.TASK_STATE_RUNNING)])

        assert query_task(state, task_id).state == job_pb2.TASK_STATE_KILLED


# --- RPC failure (result.error set) ----------------------------------------


def test_rpc_failure_leaves_running_task_unchanged():
    """RPC failure does not mutate non-ASSIGNED tasks."""
    with make_controller_state() as state:
        task_id, _attempt_id, uid = _setup_running_task(state)
        plan = _make_plan(_W1, desired=[_desired_run(uid, spec=None)])
        _apply_failure(state, _W1, plan, "connection refused")
        assert query_task(state, task_id).state == job_pb2.TASK_STATE_RUNNING


def test_rpc_failure_bounces_assigned_task_back_to_pending():
    """RPC failure on an ASSIGNED dispatch synthesizes WORKER_FAILED, returning the task to PENDING."""
    with make_controller_state() as state:
        task_id, attempt_id, uid = _setup_assigned_task(state)
        spec = _spec()
        spec.task_id = task_id.to_wire()
        spec.attempt_id = attempt_id
        spec.attempt_uid = uid
        plan = _make_plan(_W1, desired=[_desired_run(uid, spec=spec)])
        _apply_failure(state, _W1, plan, "timeout")
        # Synthetic WORKER_FAILED bounces the task back to PENDING so it can be re-dispatched.
        assert query_task(state, task_id).state == job_pb2.TASK_STATE_PENDING


def test_rpc_failure_with_no_assigned_attempts_is_a_noop():
    with make_controller_state() as state:
        register_worker(state, _W1, f"{_W1}:8080", make_worker_metadata())
        tasks = submit_job(state, "idle-job", make_job_request(name="idle-job"))
        task_id = tasks[0].task_id

        _apply_failure(state, _W1, _make_plan(_W1), "connection refused")

        assert query_task(state, task_id).state == job_pb2.TASK_STATE_PENDING


def test_apply_result_on_unknown_worker_is_a_noop():
    with make_controller_state() as state:
        result = _apply_observations(state, "ghost-worker", [])
        assert not result.tasks
        assert not result.attempts
        assert not result.jobs


def test_observation_outside_plan_is_dropped():
    """An observation whose attempt is not in the per-worker plan is dropped.

    Defends against the prod waste case: an old or out-of-sync worker
    volunteering observations for attempts the controller has forgotten about.
    Each accepted observation would otherwise drive a DB write.
    """
    with make_controller_state() as state:
        live_task, _live_attempt, live_uid = _setup_running_task(state)

        # The plan only desires the live attempt; the worker also sends a
        # terminal observation for a stale attempt the controller never asked
        # about.
        plan = _make_plan(_W1, desired=[_desired_run(live_uid, spec=None)])
        observations = [
            _obs(live_uid, job_pb2.TASK_STATE_RUNNING),
            _obs("ffffffffffffffff", job_pb2.TASK_STATE_SUCCEEDED, exit_code=0),
        ]
        _apply_observations(state, _W1, observations, plan=plan)

        # Live attempt's RUNNING was applied; stale observation was dropped
        # without raising even though the attempt has no row.
        assert query_task(state, live_task).state == job_pb2.TASK_STATE_RUNNING


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
            ops.task.queue_assignments(
                cur,
                [
                    Assignment(task_id=task_id_1, worker_id=wid1),
                    Assignment(task_id=task_id_2, worker_id=wid2),
                ],
                health=state._health,
            )

        attempt_id_1 = query_task(state, task_id_1).current_attempt_id
        attempt_id_2 = query_task(state, task_id_2).current_attempt_id

        # Drive both to RUNNING.
        for wid, task_id, attempt_id in [
            (wid1, task_id_1, attempt_id_1),
            (wid2, task_id_2, attempt_id_2),
        ]:
            with state._db.transaction() as cur:
                apply_task_observations(
                    cur,
                    [
                        WorkerTaskUpdates(
                            worker_id=wid,
                            updates=[
                                TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_RUNNING)
                            ],
                        )
                    ],
                    health=state._health,
                    endpoints=state._endpoints,
                    now=Timestamp.now(),
                )

        uid_1 = _attempt_uid(state, task_id_1, attempt_id_1)
        _apply_observations(state, _W1, [_obs(uid_1, job_pb2.TASK_STATE_FAILED, error="oom")])

        assert query_task(state, task_id_1).state == job_pb2.TASK_STATE_FAILED
        sibling_state = query_task(state, task_id_2).state
        assert sibling_state in (
            job_pb2.TASK_STATE_KILLED,
            job_pb2.TASK_STATE_PREEMPTED,
            job_pb2.TASK_STATE_PENDING,
            job_pb2.TASK_STATE_COSCHED_FAILED,
        ), f"sibling state should have cascaded, got {sibling_state}"


# --- UID routing in _observations_to_updates --------------------------------


def _observations_to_updates(
    state: ControllerTestState,
    observations: list[worker_pb2.Worker.AttemptObservation],
) -> list[TaskUpdate]:
    uids = [AttemptUid(obs.attempt_uid) for obs in observations if obs.attempt_uid]
    with state._db.transaction() as cur:
        snapshot = load_closed_snapshot(cur, now=Timestamp.now(), observation_uids=uids)
        return worker_observations_to_updates(snapshot, observations)


def test_observation_routed_by_attempt_uid():
    """A resolvable ``attempt_uid`` is mapped to its ``(task_id, attempt_id)``."""
    with make_controller_state() as state:
        task_id, attempt_id, uid = _setup_running_task(state)

        obs = _obs(uid, job_pb2.TASK_STATE_SUCCEEDED, exit_code=0)
        updates = _observations_to_updates(state, [obs])

        assert len(updates) == 1
        assert updates[0].task_id == task_id
        assert updates[0].attempt_id == attempt_id
        assert updates[0].new_state == job_pb2.TASK_STATE_SUCCEEDED


def test_unresolvable_observation_uid_is_dropped_and_logged(caplog):
    """An observation whose uid resolves to no attempt is dropped and logged inline.

    The diagnostic is emitted inline by ``observations_to_updates`` (logger
    ``iris.cluster.controller.reconcile.worker``), not via a deferred effect.
    """
    with make_controller_state() as state:
        _setup_running_task(state)
        obs = _obs("does-not-exist-uid", job_pb2.TASK_STATE_SUCCEEDED, exit_code=0)
        with caplog.at_level(logging.WARNING, logger="iris.cluster.controller.reconcile.worker"):
            updates = _observations_to_updates(state, [obs])

    assert updates == []
    assert any("did not resolve to an attempt row" in r.getMessage() for r in caplog.records), caplog.text


def _setup_running_task_named(state: ControllerTestState, job: str, worker_id: str) -> tuple[JobName, int, str]:
    """Register a worker, submit a uniquely-named job, dispatch, drive to RUNNING."""
    wid = WorkerId(worker_id)
    register_worker(state, worker_id, f"{worker_id}:8080", make_worker_metadata())
    tasks = submit_job(state, job, make_job_request(name=job))
    dispatch_task(state, tasks[0], wid)
    refreshed = query_task(state, tasks[0].task_id)
    assert refreshed is not None
    uid = _attempt_uid(state, tasks[0].task_id, refreshed.current_attempt_id)
    return tasks[0].task_id, refreshed.current_attempt_id, uid


def test_observations_to_updates_routes_batch_by_uid():
    """A batch of UID-routed observations on different tasks/workers routes each correctly."""
    with make_controller_state() as state:
        task_a, attempt_a, uid_a = _setup_running_task_named(state, "mixed-a", _W1)
        task_b, attempt_b, uid_b = _setup_running_task_named(state, "mixed-b", _W2)

        obs_a = _obs(uid_a, job_pb2.TASK_STATE_SUCCEEDED, exit_code=0)
        obs_b = _obs(uid_b, job_pb2.TASK_STATE_FAILED, error="oom")
        updates = _observations_to_updates(state, [obs_a, obs_b])

        by_task = {u.task_id: u for u in updates}
        assert set(by_task) == {task_a, task_b}
        assert by_task[task_a].attempt_id == attempt_a
        assert by_task[task_a].new_state == job_pb2.TASK_STATE_SUCCEEDED
        assert by_task[task_b].attempt_id == attempt_b
        assert by_task[task_b].new_state == job_pb2.TASK_STATE_FAILED


# --- End-to-end: full controller tick over both wires ----------------------


@dataclass
class _ScriptedProvider:
    """In-process TaskProvider whose ``reconcile_workers`` returns scripted observations.

    Each tick consumes one ``script`` entry (a callable taking the plan and
    returning a list of observations). Records every call so tests can assert
    that the right plans were dispatched.
    """

    script: list[Any] = field(default_factory=list)
    calls: list[tuple[list[WorkerReconcilePlan], dict]] = field(default_factory=list)

    def get_process_status(self, *_args, **_kwargs):
        raise NotImplementedError

    def on_worker_failed(self, *_args, **_kwargs):
        pass

    def profile_task(self, *_args, **_kwargs):
        raise NotImplementedError

    def ping_workers(self, workers):
        return []

    def reconcile_workers(self, plans, addresses):
        self.calls.append((list(plans), dict(addresses)))
        tick = len(self.calls) - 1
        responder = self.script[tick] if tick < len(self.script) else (lambda plan: [])
        return [ReconcileResult(worker_id=p.worker_id, observations=responder(p), error=None) for p in plans]

    def close(self):
        pass


def _observation_for_all_run(plan: WorkerReconcilePlan, state: int, **kwargs):
    """Build one observation per run-intent in the plan, echoing the controller-minted UID."""
    return [
        worker_pb2.Worker.AttemptObservation(
            attempt_uid=d.attempt_uid,
            state=state,
            **kwargs,
        )
        for d in plan.request.desired
        if d.HasField("run")
    ]


def test_e2e_converges_to_succeeded(make_controller):
    """Full ASSIGNED → RUNNING → SUCCEEDED convergence over the Reconcile RPC."""
    script = [
        lambda _plan: [],  # tick 1: ASSIGNED dispatch, worker hasn't started
        lambda plan: _observation_for_all_run(plan, job_pb2.TASK_STATE_RUNNING),
        lambda plan: _observation_for_all_run(plan, job_pb2.TASK_STATE_SUCCEEDED, exit_code=0),
    ]
    provider = _ScriptedProvider(script=script)
    ctrl = make_controller(provider=provider)
    state = ControllerTestState(
        ctrl._db,
        health=ctrl._health,
        endpoints=ctrl._endpoints,
        worker_attrs=ctrl._worker_attrs,
        run_template_cache=ctrl._run_template_cache,
    )

    wid = register_worker(state, _W1, _W1_ADDR, make_worker_metadata())
    tasks = submit_job(state, "e2e-job", make_job_request(name="e2e-job"))
    task_id = tasks[0].task_id

    with state._db.transaction() as cur:
        ops.task.queue_assignments(cur, [Assignment(task_id=task_id, worker_id=wid)], health=state._health)

    # Tick 1: ASSIGNED — controller dispatches the inline spec.
    ctrl._reconcile_worker_batch()
    tick1_desired = list(provider.calls[0][0][0].request.desired)
    assert len(tick1_desired) == 1
    assert tick1_desired[0].HasField("run") and tick1_desired[0].run.HasField(
        "request"
    ), "first tick should carry inline spec"
    assert tick1_desired[0].attempt_uid, "controller must emit a non-empty attempt_uid"

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
    provider = _ScriptedProvider(script=script)
    ctrl = make_controller(provider=provider)
    state = ControllerTestState(
        ctrl._db,
        health=ctrl._health,
        endpoints=ctrl._endpoints,
        worker_attrs=ctrl._worker_attrs,
        run_template_cache=ctrl._run_template_cache,
    )

    wid = register_worker(state, _W1, _W1_ADDR, make_worker_metadata())
    tasks = submit_job(state, "missing-job", make_job_request(name="missing-job"))
    task_id = tasks[0].task_id

    with state._db.transaction() as cur:
        ops.task.queue_assignments(cur, [Assignment(task_id=task_id, worker_id=wid)], health=state._health)

    ctrl._reconcile_worker_batch()
    ctrl._reconcile_worker_batch()

    task = query_task(state, task_id)
    assert task.state == job_pb2.TASK_STATE_FAILED
    assert task.error == "worker_lost_spec"
