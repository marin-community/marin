# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidated tests for the controller reconcile pipeline.

Three layers, exercised in order:

1. **Pure compute** — ``build_reconcile_plans`` builds one ``ReconcileRequest``
   proto per worker from a ``ReconcileInputs`` snapshot. No DB.
2. **Wire & dispatch** — ``RpcTaskBackend.reconcile`` fans out via a
   fake stub factory and synthesizes ``WorkerReconcileResult.observations``.
3. **Apply + e2e** — ``apply_reconcile`` against real SQLite DB state, plus a
   handful of end-to-end convergence ticks driven through the production control
   tick's reconcile phase (``reconcile_once``).
"""

import threading
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import pytest
from iris.cluster.backends.rpc.backend import (
    WORKER_RECONCILE_TEARDOWN_REASON,
    FleetObservation,
    RpcTaskBackend,
)
from iris.cluster.controller import ops, writes
from iris.cluster.controller.backend import (
    AutoscaleRequest,
    AutoscaleResult,
    BackendCapability,
    BackendRuntime,
    ReconcileRequest,
    ReconcileResult,
    ScheduleRequest,
    ScheduleResult,
    plans_from_snapshot,
)
from iris.cluster.controller.backend_store import BackendWorkerStore
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.reads import ControlSnapshot
from iris.cluster.controller.reconcile.loader import load_closed_snapshot
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.reconcile.worker import (
    ReconcileInputs,
    ReconcileRow,
    WorkerReconcilePlan,
    WorkerReconcileResult,
    build_reconcile_plans,
)
from iris.cluster.controller.reconcile.worker import (
    observations_to_updates as worker_observations_to_updates,
)
from iris.cluster.controller.scheduling.scheduler import Scheduler
from iris.cluster.controller.schema import task_attempts_table
from iris.cluster.controller.worker_health import (
    BUILD_FAILURE_THRESHOLD,
    MIN_UNREACHABLE_FAILURES,
    WorkerHealthEvent,
    WorkerHealthEventKind,
    WorkerHealthTracker,
)
from iris.cluster.types import AttemptUid, JobName, WorkerId
from iris.rpc import job_pb2, worker_pb2
from rigging.timing import Duration, Timestamp
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from tests.cluster.controller._test_support import ControllerTestState
from tests.cluster.controller.transition_driver import (
    WorkerTaskUpdates,
    apply_task_observations,
    commit_reconcile,
)

from .conftest import (
    dispatch_task,
    make_controller_state,
    make_job_request,
    make_worker_metadata,
    query_attempt,
    query_job,
    query_task,
    query_worker,
    reconcile_once,
    register_worker,
    run_worker_daemon_reconcile,
    run_worker_daemon_schedule,
    store_from_runtime,
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
    return build_reconcile_plans(inputs)[0]


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
    # PENDING here carries the default non-terminal attempt_state, so there is
    # no worker-bound terminal attempt to stop (see the PENDING+terminal-attempt
    # case below). UNSPECIFIED is never planned for.
    plan = _plan_for([_row(task_state)])
    assert list(plan.request.desired) == []


def test_reconcile_worker_pending_with_terminal_attempt_emits_stop():
    """A task rolled back to PENDING for retry, whose old worker-bound attempt is
    already terminal (PREEMPTED), must be planned as a 'stop' — not omitted.

    Omitting it drops the attempt from the desired set, so the worker's terminal
    observation for it is filtered out by ``filter_observations_to_plan`` and the
    attempt is never finalized: its chips leak. Emitting 'stop' keeps it in the
    plan so the worker tears it down and the terminal observation releases the
    slot. Covers preemption-with-budget and coscheduled-sibling requeue, which
    both leave the sibling's attempt unfinished and worker-bound.
    """
    row = _row(job_pb2.TASK_STATE_PENDING, attempt_id=5, attempt_state=job_pb2.TASK_STATE_PREEMPTED)
    plan = _plan_for([row])

    assert len(plan.request.desired) == 1
    desired = plan.request.desired[0]
    assert desired.HasField("stop")
    assert desired.stop == worker_pb2.Worker.STOP_REASON_PREEMPTED
    assert desired.attempt_uid == row.attempt_uid


def test_reconcile_worker_holds_assigned_run_while_preemption_victim_occupies_worker():
    """A preemptor ASSIGNED onto the worker its victim frees gets no run-intent
    while the victim's PREEMPTED attempt is still worker-bound — the victim's
    stop still fires, but the preemptor waits so it never races the live process."""
    victim = _row(
        job_pb2.TASK_STATE_PENDING,
        task_id="victim",
        job="job-victim",
        attempt_id=4,
        attempt_uid="aaaaaaaaaaaaaaaa",
        attempt_state=job_pb2.TASK_STATE_PREEMPTED,
    )
    preemptor = _row(
        job_pb2.TASK_STATE_ASSIGNED,
        task_id="preemptor",
        job="job-preemptor",
        attempt_id=1,
        attempt_uid="bbbbbbbbbbbbbbbb",
    )
    plan = _plan_for([victim, preemptor], job_specs={_job_id("job-preemptor"): _spec()})

    by_uid = {d.attempt_uid: d for d in plan.request.desired}
    assert by_uid[victim.attempt_uid].stop == worker_pb2.Worker.STOP_REASON_PREEMPTED
    assert (
        preemptor.attempt_uid not in by_uid
    ), "preemptor run-intent must be withheld while the victim occupies the worker"


def test_reconcile_worker_dispatches_assigned_once_victim_finalized():
    """Once the victim's attempt is finalized it drops from the reconcile snapshot,
    so the preemptor — now alone on the worker — is dispatched."""
    preemptor = _row(
        job_pb2.TASK_STATE_ASSIGNED,
        task_id="preemptor",
        job="job-preemptor",
        attempt_id=1,
        attempt_uid="bbbbbbbbbbbbbbbb",
    )
    plan = _plan_for([preemptor], job_specs={_job_id("job-preemptor"): _spec()})

    by_uid = {d.attempt_uid: d for d in plan.request.desired}
    assert by_uid[preemptor.attempt_uid].HasField("run")


def test_reconcile_worker_running_cotenant_does_not_hold_assigned_dispatch():
    """A merely-RUNNING co-tenant does not gate a freshly ASSIGNED task on the
    same worker — only a preemption victim being torn down does."""
    running = _row(
        job_pb2.TASK_STATE_RUNNING,
        task_id="cotenant",
        job="job-run",
        attempt_id=2,
        attempt_uid="cccccccccccccccc",
        attempt_state=job_pb2.TASK_STATE_RUNNING,
    )
    assigned = _row(
        job_pb2.TASK_STATE_ASSIGNED,
        task_id="fresh",
        job="job-fresh",
        attempt_id=1,
        attempt_uid="dddddddddddddddd",
    )
    plan = _plan_for([running, assigned], job_specs={_job_id("job-fresh"): _spec()})

    by_uid = {d.attempt_uid: d for d in plan.request.desired}
    assert by_uid[assigned.attempt_uid].HasField("run")
    assert by_uid[running.attempt_uid].HasField("run")


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
# Section 2: wire & dispatch (RpcTaskBackend.reconcile)
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


def _provider_with_stub(stub: _FakeWorkerStub | None = None) -> tuple[RpcTaskBackend, _FakeWorkerStub]:
    """Build a backend with a single stub at ``_W1_ADDR`` (creating a default if needed)."""
    if stub is None:
        stub = _FakeWorkerStub(address=_W1_ADDR)
    factory = _FakeStubFactory(stubs={_W1_ADDR: stub})
    return RpcTaskBackend(stub_factory=factory), stub


def _reconcile_snapshot(worker_addresses: dict[WorkerId, str]) -> ControlSnapshot:
    return ControlSnapshot(
        worker_addresses=worker_addresses,
        reconcile_rows=[],
        timeout_rows=[],
    )


@dataclass
class _StubWorkerStore:
    """A worker store that hands the backend a fixed reconcile snapshot.

    The dispatch-layer tests exercise ``RpcTaskBackend.reconcile``'s fan-out and
    health-event derivation given a known snapshot; the backend now sources that
    snapshot itself, so the test supplies it through this stub.
    """

    snapshot: ControlSnapshot

    def reconcile_snapshot(self) -> ControlSnapshot:
        return self.snapshot

    def scheduling_inputs(self):
        raise NotImplementedError

    def worker_status(self):
        raise NotImplementedError


def _reconcile_with(provider: RpcTaskBackend, worker_addresses: dict[WorkerId, str]) -> FleetObservation:
    provider._store = cast(BackendWorkerStore, _StubWorkerStore(_reconcile_snapshot(worker_addresses)))
    return provider._observe_fleet()


def _reconcile_one(provider: RpcTaskBackend, plan: WorkerReconcilePlan, *, address: str = _W1_ADDR):
    # The backend now builds plans from the snapshot; ``plan`` here only fixes
    # which worker is reconciled. The RPC fan-out and observation surfacing are
    # what these dispatch-layer tests exercise.
    observation = _reconcile_with(provider, {plan.worker_id: address})
    return [r for _, r in observation.worker_results]


def test_dispatch_reconcile_plans_empty_short_circuits():
    provider, _ = _provider_with_stub()
    assert _reconcile_with(provider, {}).worker_results == []


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


def test_reconcile_matching_responder_id_is_reached():
    """A healthy reply stamped with the targeted worker's id counts as REACHED."""
    stub = _FakeWorkerStub(
        address=_W1_ADDR,
        reconcile_response=worker_pb2.Worker.ReconcileResponse(
            worker_id=_W1, health=worker_pb2.Worker.WorkerHealth(healthy=True)
        ),
    )
    factory = _FakeStubFactory(stubs={_W1_ADDR: stub})
    provider = RpcTaskBackend(stub_factory=factory)

    result = _reconcile_with(provider, {WorkerId(_W1): _W1_ADDR})

    assert result.transport_events == [WorkerHealthEvent(WorkerId(_W1), WorkerHealthEventKind.REACHED)]
    assert _W1_ADDR in factory.stubs  # healthy worker's stub kept


def test_reconcile_recycled_address_is_unreachable_not_reached():
    """A healthy reply stamped with a DIFFERENT worker_id (recycled IP) is UNREACHABLE.

    Regression: after a worker's VM is deleted GCP recycles its internal IP onto
    a new VM. Reconciling the dead worker at its stale address then reaches the
    *new* worker, which answers healthy. Folding that as REACHED would reset the
    dead worker's failure count and keep it schedulable forever — a black hole
    that accepts and kills every task assigned to it. The mismatched id must mark
    the dead worker UNREACHABLE so it is reaped, and the impostor's stub dropped.
    """
    stub = _FakeWorkerStub(
        address=_W1_ADDR,
        reconcile_response=worker_pb2.Worker.ReconcileResponse(
            worker_id=_W2, health=worker_pb2.Worker.WorkerHealth(healthy=True)
        ),
    )
    factory = _FakeStubFactory(stubs={_W1_ADDR: stub})
    provider = RpcTaskBackend(stub_factory=factory)

    result = _reconcile_with(provider, {WorkerId(_W1): _W1_ADDR})

    assert result.transport_events == [WorkerHealthEvent(WorkerId(_W1), WorkerHealthEventKind.UNREACHABLE)]
    # The stale stub is evicted so the next tick re-resolves the address.
    assert _W1_ADDR not in factory.stubs


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
        ops.task.assign(cur, [Assignment(task_id=task_row.task_id, worker_id=wid)], health=state._health)
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
    result = WorkerReconcileResult(worker_id=WorkerId(worker_id), observations=observations, error=None)
    with state._db.transaction() as cur:
        return commit_reconcile(
            cur,
            [(plan, result)],
            now=_NOW,
        )


def _apply_failure(
    state: ControllerTestState,
    worker_id: str,
    plan: WorkerReconcilePlan,
    error: str,
):
    result = WorkerReconcileResult(worker_id=WorkerId(worker_id), observations=[], error=error)
    with state._db.transaction() as cur:
        return commit_reconcile(
            cur,
            [(plan, result)],
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


def test_missing_observation_on_active_task_charges_preemption_budget():
    """A MISSING observation for a still-active task is worker loss, not app failure.

    With ``max_retries_preemption=0`` the active RUNNING task goes terminal
    WORKER_FAILED (preemption budget exhausted) rather than FAILED. The
    distinction matters: a worker that restarts and loses a still-running
    container reports MISSING, which must consume the preemption budget — not
    fail the task at ``max_retries_failure=0``.
    """
    with make_controller_state() as state:
        task_id, attempt_id, uid = _setup_running_task(state)
        _apply_observations(state, _W1, [_obs(uid, job_pb2.TASK_STATE_MISSING)])

        task = query_task(state, task_id)
        attempt = query_attempt(state, task_id, attempt_id)
        assert task is not None and attempt is not None
        assert task.state == job_pb2.TASK_STATE_WORKER_FAILED
        assert task.error == "worker_lost_spec"
        assert task.preemption_count == 1
        assert task.failure_count == 0
        assert attempt.state == job_pb2.TASK_STATE_WORKER_FAILED


def test_missing_observation_on_active_task_retries_with_preemption_budget():
    """A MISSING observation for an active task with preemption budget retries it."""
    with make_controller_state() as state:
        register_worker(state, _W1, f"{_W1}:8080", make_worker_metadata())
        tasks = submit_job(
            state, "retry-job", make_job_request(name="retry-job", max_retries_preemption=2, max_retries_failure=0)
        )
        task_row = tasks[0]
        dispatch_task(state, task_row, WorkerId(_W1))
        refreshed = query_task(state, task_row.task_id)
        assert refreshed is not None
        uid = _attempt_uid(state, task_row.task_id, refreshed.current_attempt_id)

        _apply_observations(state, _W1, [_obs(uid, job_pb2.TASK_STATE_MISSING)])

        task = query_task(state, task_row.task_id)
        assert task is not None
        # Retried to PENDING on the preemption budget, NOT failed at
        # max_retries_failure=0.
        assert task.state == job_pb2.TASK_STATE_PENDING
        assert task.preemption_count == 1
        assert task.failure_count == 0


def test_missing_observation_on_terminal_task_finalizes_failed():
    """A MISSING observation for an already-terminal task finalizes the stranded attempt.

    This is the stranded-terminal-attempt case: the task already reached a
    terminal state but the worker-bound attempt's ``finished_at_ms`` is NULL. A
    re-poll surfaces MISSING, which stays FAILED and stamps the attempt so its
    capacity is released.
    """
    with make_controller_state() as state:
        task_id, attempt_id, uid = _setup_running_task(state)
        # Drive the task terminal FAILED first (max_retries_failure=0).
        _apply_observations(state, _W1, [_obs(uid, job_pb2.TASK_STATE_FAILED, error="boom")])
        terminal = query_task(state, task_id)
        assert terminal is not None and terminal.state == job_pb2.TASK_STATE_FAILED

        # A later MISSING re-poll for the same now-terminal task stays FAILED and
        # finalizes the attempt rather than charging the preemption budget.
        _apply_observations(state, _W1, [_obs(uid, job_pb2.TASK_STATE_MISSING)])
        task = query_task(state, task_id)
        attempt = query_attempt(state, task_id, attempt_id)
        assert task is not None and attempt is not None
        assert task.state == job_pb2.TASK_STATE_FAILED
        assert task.preemption_count == 0
        assert attempt.finished_at_ms is not None


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
            ops.job.cancel(cur, job_id=task_row.job_id, reason="user_cancel")
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
            ops.task.assign(
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
                    now=Timestamp.now(),
                )

        uid_1 = _attempt_uid(state, task_id_1, attempt_id_1)
        _apply_observations(state, _W1, [_obs(uid_1, job_pb2.TASK_STATE_FAILED, error="oom")])

        assert query_task(state, task_id_1).state == job_pb2.TASK_STATE_FAILED
        # max_retries_failure=0, so the failure is terminal (not a retry): the
        # coscheduled sibling is terminated, not requeued.
        sibling_state = query_task(state, task_id_2).state
        assert (
            sibling_state == job_pb2.TASK_STATE_COSCHED_FAILED
        ), f"sibling should cascade terminal, got {sibling_state}"


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


def test_unresolvable_observation_uid_is_dropped():
    """An observation whose uid resolves to no attempt row produces no update."""
    with make_controller_state() as state:
        _setup_running_task(state)
        obs = _obs("does-not-exist-uid", job_pb2.TASK_STATE_SUCCEEDED, exit_code=0)
        updates = _observations_to_updates(state, [obs])

    assert updates == []


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


@pytest.mark.parametrize(
    ("obs_state", "exit_code", "expected"),
    [
        # proto3 has no scalar presence, so a real exit 0 (wire-0) is collapsed to
        # None to avoid clobbering a recorded code via the commit-time coalesce;
        # success is conveyed by the SUCCEEDED state, not by exit 0.
        (job_pb2.TASK_STATE_SUCCEEDED, 0, None),
        # A genuine non-zero code must survive.
        (job_pb2.TASK_STATE_FAILED, 137, 137),
    ],
)
def test_exit_code_zero_coalesced_to_none(obs_state, exit_code, expected):
    """exit_code 0 is intentionally collapsed to None."""
    with make_controller_state() as state:
        _, _, uid = _setup_running_task(state)
        [update] = _observations_to_updates(state, [_obs(uid, obs_state, exit_code=exit_code)])
    assert update.exit_code == expected


# --- End-to-end: full controller tick over both wires ----------------------


@dataclass
class _ScriptedProvider:
    """In-process worker-daemon TaskBackend whose ``reconcile`` returns scripted observations.

    Builds per-worker plans from the snapshot (mirroring ``RpcTaskBackend``),
    then consumes one ``script`` entry per tick (a callable taking the plan and
    returning a list of observations). Records every call so tests can assert
    that the right plans were dispatched.
    """

    script: list[Any] = field(default_factory=list)
    calls: list[tuple[list[WorkerReconcilePlan], dict]] = field(default_factory=list)
    name: str = "worker"
    autoscaler: Any = None
    _store: BackendWorkerStore | None = None
    health: WorkerHealthTracker = field(default_factory=WorkerHealthTracker)
    advertised: dict[str, set[str]] = field(default_factory=dict)
    capabilities: ClassVar[frozenset[BackendCapability]] = frozenset(
        {BackendCapability.WORKER_DAEMON, BackendCapability.IRIS_AUTOSCALER}
    )
    _scheduler: Scheduler = field(default_factory=Scheduler, init=False, repr=False)
    _pending_dead: list[WorkerId] = field(default_factory=list, init=False, repr=False)

    def advertised_attributes(self) -> dict[str, set[str]]:
        return self.advertised

    def configure_routing(self, advertised: dict[str, set[str]]) -> None:
        self.advertised = advertised

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        return run_worker_daemon_schedule(self._scheduler, self._store, request)

    def get_process_status(self, *_args, **_kwargs):
        raise NotImplementedError

    def bind_runtime(self, runtime: BackendRuntime) -> None:
        self._store = store_from_runtime(runtime, self.health, self.autoscale)

    def seed_liveness(self) -> None:
        assert self._store is not None
        worker_ids = self._store.owned_worker_ids()
        if worker_ids:
            self.health.heartbeat(worker_ids, Timestamp.now().epoch_ms())

    def profile_task(self, *_args, **_kwargs):
        raise NotImplementedError

    def autoscale(self, request: AutoscaleRequest) -> AutoscaleResult:
        return AutoscaleResult()

    def reconcile(self, request: ReconcileRequest) -> ReconcileResult:
        assert self._store is not None, "_ScriptedProvider.reconcile called before worker store attached"
        snapshot = self._store.reconcile_snapshot()
        plans = plans_from_snapshot(snapshot)
        self.calls.append((plans, dict(snapshot.worker_addresses)))
        tick = len(self.calls) - 1
        responder = self.script[tick] if tick < len(self.script) else (lambda plan: [])
        worker_results = [
            (p, WorkerReconcileResult(worker_id=p.worker_id, observations=responder(p), error=None)) for p in plans
        ]
        events = [WorkerHealthEvent(p.worker_id, WorkerHealthEventKind.REACHED) for p in plans]
        result, dead = run_worker_daemon_reconcile(self._store, self.health, worker_results, events)
        self._pending_dead.extend(dead)
        return result

    def run_teardown(self) -> None:
        dead = self._pending_dead
        self._pending_dead = []
        self.teardown(dead, reason=WORKER_RECONCILE_TEARDOWN_REASON)

    def teardown(self, dead_workers: list[WorkerId], *, reason: str) -> None:
        assert self._store is not None, "_ScriptedProvider.teardown called before worker store attached"
        self._store.reap_workers(dead_workers, reason=reason)

    def prune_dead_workers(self, *, cutoff_ms: int, stop_event: threading.Event | None, pause: float) -> int:
        assert self._store is not None, "_ScriptedProvider.prune_dead_workers called before worker store attached"
        return self._store.prune_dead_workers(cutoff_ms=cutoff_ms, stop_event=stop_event, pause=pause)

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
        health=ctrl.provider.health,
    )

    wid = register_worker(state, _W1, _W1_ADDR, make_worker_metadata())
    tasks = submit_job(state, "e2e-job", make_job_request(name="e2e-job"))
    task_id = tasks[0].task_id

    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task_id, worker_id=wid)], health=state._health)

    # Tick 1: ASSIGNED — controller dispatches the inline spec.
    reconcile_once(ctrl)
    tick1_desired = list(provider.calls[0][0][0].request.desired)
    assert len(tick1_desired) == 1
    assert tick1_desired[0].HasField("run") and tick1_desired[0].run.HasField(
        "request"
    ), "first tick should carry inline spec"
    assert tick1_desired[0].attempt_uid, "controller must emit a non-empty attempt_uid"

    # Tick 2: worker reports RUNNING.
    reconcile_once(ctrl)
    assert query_task(state, task_id).state == job_pb2.TASK_STATE_RUNNING

    # Tick 3: subsequent run intents must not carry inline spec (cache-hit invariant).
    reconcile_once(ctrl)
    tick3_desired = list(provider.calls[2][0][0].request.desired)
    assert tick3_desired and tick3_desired[0].HasField("run")
    assert not tick3_desired[0].run.HasField("request"), "subsequent ticks must not carry inline spec"

    task_final = query_task(state, task_id)
    assert task_final.state == job_pb2.TASK_STATE_SUCCEEDED
    assert query_job(state, task_final.job_id).state == job_pb2.JOB_STATE_SUCCEEDED


def test_e2e_missing_observation_on_assigned_task_retries_to_pending(make_controller):
    """End-to-end MISSING cascade on an ASSIGNED task: dispatch → MISSING → PENDING retry.

    A worker that accepted the assignment but lost the spec reports MISSING. The
    task is still ASSIGNED (worker loss before the process ran), so it retries to
    PENDING without charging any budget rather than going terminal FAILED.
    """
    script = [
        lambda _plan: [],  # tick 1: ASSIGNED dispatch
        lambda plan: _observation_for_all_run(plan, job_pb2.TASK_STATE_MISSING),
    ]
    provider = _ScriptedProvider(script=script)
    ctrl = make_controller(provider=provider)
    state = ControllerTestState(
        ctrl._db,
        health=ctrl.provider.health,
    )

    wid = register_worker(state, _W1, _W1_ADDR, make_worker_metadata())
    tasks = submit_job(state, "missing-job", make_job_request(name="missing-job"))
    task_id = tasks[0].task_id

    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task_id, worker_id=wid)], health=state._health)

    reconcile_once(ctrl)
    reconcile_once(ctrl)

    task = query_task(state, task_id)
    assert task.state == job_pb2.TASK_STATE_PENDING
    assert task.preemption_count == 0
    assert task.failure_count == 0


@dataclass
class _UnreachableProvider:
    """Worker-daemon backend that reports ``unreachable`` workers UNREACHABLE each tick.

    Drives the reconcile-fail → teardown path with NO ping loop: liveness comes
    purely from ``reconcile`` health events, and teardown rides
    ``autoscale(dead_workers=...)``, which here reports each dead worker's slice
    siblings so the controller fails them too. Records every ``autoscale``
    ``dead_workers`` argument so tests can prove teardown was triggered by the
    reconcile pass rather than any separate ping channel.
    """

    unreachable: set[str] = field(default_factory=set)
    unhealthy: set[str] = field(default_factory=set)
    siblings: dict[str, list[str]] = field(default_factory=dict)
    autoscale_calls: list[list[WorkerId]] = field(default_factory=list)
    name: str = "worker"
    autoscaler: Any = None
    _store: BackendWorkerStore | None = None
    health: WorkerHealthTracker = field(default_factory=WorkerHealthTracker)
    advertised: dict[str, set[str]] = field(default_factory=dict)
    capabilities: ClassVar[frozenset[BackendCapability]] = frozenset(
        {BackendCapability.WORKER_DAEMON, BackendCapability.IRIS_AUTOSCALER}
    )
    _scheduler: Scheduler = field(default_factory=Scheduler, init=False, repr=False)
    _pending_dead: list[WorkerId] = field(default_factory=list, init=False, repr=False)

    def advertised_attributes(self) -> dict[str, set[str]]:
        return self.advertised

    def configure_routing(self, advertised: dict[str, set[str]]) -> None:
        self.advertised = advertised

    def bind_runtime(self, runtime: BackendRuntime) -> None:
        self._store = store_from_runtime(runtime, self.health, self.autoscale)

    def seed_liveness(self) -> None:
        assert self._store is not None
        worker_ids = self._store.owned_worker_ids()
        if worker_ids:
            self.health.heartbeat(worker_ids, Timestamp.now().epoch_ms())

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        return run_worker_daemon_schedule(self._scheduler, self._store, request)

    def reconcile(self, request: ReconcileRequest) -> ReconcileResult:
        assert self._store is not None, "_UnreachableProvider.reconcile called before worker store attached"
        snapshot = self._store.reconcile_snapshot()
        plans = plans_from_snapshot(snapshot)
        worker_results: list[tuple[WorkerReconcilePlan, WorkerReconcileResult]] = []
        events: list[WorkerHealthEvent] = []
        for plan in plans:
            wid = str(plan.worker_id)
            if wid in self.unreachable:
                # RPC failed outright: error set, no observations.
                worker_results.append(
                    (plan, WorkerReconcileResult(worker_id=plan.worker_id, observations=[], error="rpc unreachable"))
                )
                events.append(WorkerHealthEvent(plan.worker_id, WorkerHealthEventKind.UNREACHABLE))
            elif wid in self.unhealthy:
                # RPC succeeded but the worker self-reported unhealthy. Mirror
                # RpcTaskBackend: error=None + self_healthy=False ⇒ UNREACHABLE.
                worker_results.append(
                    (
                        plan,
                        WorkerReconcileResult(worker_id=plan.worker_id, observations=[], error=None, self_healthy=False),
                    )
                )
                events.append(WorkerHealthEvent(plan.worker_id, WorkerHealthEventKind.UNREACHABLE))
            else:
                worker_results.append(
                    (plan, WorkerReconcileResult(worker_id=plan.worker_id, observations=[], error=None))
                )
                events.append(WorkerHealthEvent(plan.worker_id, WorkerHealthEventKind.REACHED))
        result, dead = run_worker_daemon_reconcile(self._store, self.health, worker_results, events)
        self._pending_dead.extend(dead)
        return result

    def run_teardown(self) -> None:
        dead = self._pending_dead
        self._pending_dead = []
        self.teardown(dead, reason=WORKER_RECONCILE_TEARDOWN_REASON)

    def teardown(self, dead_workers: list[WorkerId], *, reason: str) -> None:
        assert self._store is not None, "_UnreachableProvider.teardown called before worker store attached"
        self._store.reap_workers(dead_workers, reason=reason)

    def prune_dead_workers(self, *, cutoff_ms: int, stop_event: threading.Event | None, pause: float) -> int:
        assert self._store is not None, "_UnreachableProvider.prune_dead_workers called before worker store attached"
        return self._store.prune_dead_workers(cutoff_ms=cutoff_ms, stop_event=stop_event, pause=pause)

    def autoscale(self, request: AutoscaleRequest) -> AutoscaleResult:
        self.autoscale_calls.append(list(request.dead_workers))
        removed: list[WorkerId] = list(request.dead_workers)
        for dead in request.dead_workers:
            removed.extend(WorkerId(sib) for sib in self.siblings.get(str(dead), []))
        return AutoscaleResult(removed_workers=removed)

    def get_process_status(self, *_args, **_kwargs):
        raise NotImplementedError

    def profile_task(self, *_args, **_kwargs):
        raise NotImplementedError

    def close(self):
        pass


# Worker-death detection is time-based: a worker is reaped once it has been
# continuously unreachable for worker_unreachable_grace (and past the failure
# floor). Tests force the grace to have elapsed with ``_expire_grace`` rather than
# sleeping, since the control tick reads the real wall clock.
_GRACE = Duration.from_seconds(4)


def _expire_grace(ctrl, wid: WorkerId) -> None:
    """Backdate a worker's last heartbeat so the unreachable grace has elapsed."""
    aged = Timestamp.now().epoch_ms() - _GRACE.to_ms() - 1
    ctrl.provider.health.set_last_heartbeat_for_test(wid, aged)


@pytest.mark.parametrize(
    "provider_kwargs",
    [
        pytest.param({"unreachable": {_W1}}, id="rpc_unreachable"),
        pytest.param({"unhealthy": {_W1}}, id="responded_but_unhealthy"),
    ],
)
def test_reconcile_failure_tears_down_worker_without_ping_loop(make_controller, provider_kwargs):
    """A worker the backend can't keep alive is torn down by the reconcile pass
    alone — no ping loop, no separate liveness channel.

    Two failure modes fold to the same UNREACHABLE signal: the reconcile RPC
    fails outright (``rpc_unreachable``), or it succeeds but the worker
    self-reports unhealthy — e.g. failed disk (``responded_but_unhealthy``,
    ``error=None`` + ``self_healthy=False``). In both, once the worker has been
    continuously unreachable for the grace, the controller fails the worker,
    drives ``backend.autoscale(dead_workers=...)`` to reap the slice, and forgets
    it.
    """
    provider = _UnreachableProvider(**provider_kwargs)
    ctrl = make_controller(provider=provider, worker_unreachable_grace=_GRACE)
    state = ControllerTestState(
        ctrl._db,
        health=ctrl.provider.health,
    )

    wid = register_worker(state, _W1, _W1_ADDR, make_worker_metadata())

    # Many consecutive failures, but all within the wall-clock grace window:
    # detection is time-based, so the worker is still tracked, not torn down.
    for _ in range(MIN_UNREACHABLE_FAILURES + 5):
        reconcile_once(ctrl)
    assert query_worker(state, wid) is not None
    assert provider.autoscale_calls == []

    # Once it has been unreachable for the full grace, the next failed reconcile
    # tears it down and forgets it.
    _expire_grace(ctrl, wid)
    reconcile_once(ctrl)
    assert provider.autoscale_calls == [[wid]]
    assert query_worker(state, wid) is None, "failed worker row should be removed"
    assert wid not in ctrl.provider.health.all(), "failed worker should be forgotten from the tracker"


def test_reconcile_failure_reaps_slice_siblings(make_controller):
    """Failing one worker on a multi-VM slice reaps its healthy siblings too.

    ``backend.autoscale`` reports the dead worker's slice siblings in
    ``removed_workers``; the controller fails those siblings and forgets the
    whole slice, even though the siblings were reachable every tick.
    """
    provider = _UnreachableProvider(unreachable={_W1}, siblings={_W1: [_W2]})
    ctrl = make_controller(provider=provider, worker_unreachable_grace=_GRACE)
    state = ControllerTestState(
        ctrl._db,
        health=ctrl.provider.health,
    )

    dead = register_worker(state, _W1, _W1_ADDR, make_worker_metadata())
    sibling = register_worker(state, _W2, f"{_W2}:8080", make_worker_metadata())

    # Accrue failures past the floor, then expire the grace so the next tick reaps.
    for _ in range(MIN_UNREACHABLE_FAILURES):
        reconcile_once(ctrl)
    _expire_grace(ctrl, dead)
    reconcile_once(ctrl)

    assert provider.autoscale_calls == [[dead]]
    assert query_worker(state, dead) is None
    assert query_worker(state, sibling) is None, "reachable slice sibling should be reaped too"
    assert ctrl.provider.health.all() == {}, "whole slice should be forgotten from the tracker"


def test_request_worker_eviction_tears_down_on_next_tick(make_controller):
    """A queued eviction fails the worker and reaps its slice on the next tick.

    The Register RPC queues a recycled-IP prior owner off the control-loop thread,
    where reaping a slice via the autoscaler is unsafe. The tick drains it through
    the same fail-and-teardown path as a reconcile failure --
    ``backend.autoscale(dead_workers=...)`` -- even though the worker answers every
    reconcile: eviction is driven by the queue, not by liveness.
    """
    provider = _UnreachableProvider()  # the worker stays reachable every tick
    ctrl = make_controller(provider=provider, worker_unreachable_grace=_GRACE)
    state = ControllerTestState(
        ctrl._db,
        health=ctrl.provider.health,
    )

    wid = register_worker(state, _W1, _W1_ADDR, make_worker_metadata())

    ctrl.request_worker_eviction([wid])
    reconcile_once(ctrl)

    assert provider.autoscale_calls == [[wid]], "drain must drive teardown via backend.autoscale"
    assert query_worker(state, wid) is None, "evicted worker row should be removed"
    assert wid not in ctrl.provider.health.all(), "evicted worker should be forgotten from the tracker"


def _fail_first_held(plan: WorkerReconcilePlan) -> list[worker_pb2.Worker.AttemptObservation]:
    """Report the worker's first still-held run attempt as a launch failure.

    The kernel attributes an ASSIGNED -> WORKER_FAILED transition observed over
    reconcile to the worker (a host that couldn't bring the attempt up), so
    reporting exactly one held attempt WORKER_FAILED per tick charges exactly one
    build failure. The attempt retries to an unassigned PENDING task without
    consuming budget and drops out of the worker's held set, so the next tick's
    first held attempt is a fresh one.
    """
    for desired in plan.request.desired:
        if desired.HasField("run"):
            return [
                worker_pb2.Worker.AttemptObservation(
                    attempt_uid=desired.attempt_uid,
                    state=job_pb2.TASK_STATE_WORKER_FAILED,
                    error="host failed to launch attempt",
                )
            ]
    return []


def test_reconcile_reaps_worker_at_build_failure_threshold(make_controller):
    """The in-process backend's reconcile fold reaps a build-failing worker at the
    threshold -- not one failure before.

    Each reconcile pass charges one build failure (a worker-reported
    ASSIGNED -> FAILED transition). The liveness fold now lives in the backend, so
    the worker that crosses ``BUILD_FAILURE_THRESHOLD`` rides back as
    ``dead_workers`` and the controller tears it down. The boundary is exact: the
    worker survives the first ``BUILD_FAILURE_THRESHOLD - 1`` failures and is
    removed on the one that reaches the threshold.
    """
    provider = _ScriptedProvider(script=[_fail_first_held] * BUILD_FAILURE_THRESHOLD)
    ctrl = make_controller(provider=provider)
    state = ControllerTestState(
        ctrl._db,
        health=ctrl.provider.health,
    )

    wid = register_worker(state, _W1, _W1_ADDR, make_worker_metadata())
    # One task per build failure we mean to charge, all assigned up front so the
    # responder always has a fresh held attempt to fail. Generous retry budget
    # keeps every task retrying (never terminal) so the job stays active and the
    # worker keeps its place across all the passes.
    tasks = submit_job(
        state,
        "build-fail-job",
        make_job_request(
            name="build-fail-job",
            replicas=BUILD_FAILURE_THRESHOLD,
            max_retries_failure=BUILD_FAILURE_THRESHOLD,
        ),
    )
    with state._db.transaction() as cur:
        ops.task.assign(
            cur,
            [Assignment(task_id=task.task_id, worker_id=wid) for task in tasks],
            health=state._health,
        )

    # The first THRESHOLD - 1 build failures accrue but stay under the bar.
    for expected_failures in range(1, BUILD_FAILURE_THRESHOLD):
        reconcile_once(ctrl)
        assert query_worker(state, wid) is not None, "worker reaped before reaching the build-failure threshold"
        assert ctrl.provider.health.liveness(wid).build_failures == expected_failures

    # The THRESHOLD-th build failure trips the bar: the backend's fold returns the
    # worker dead and the controller reaps it.
    reconcile_once(ctrl)
    assert query_worker(state, wid) is None, "worker should be reaped at the build-failure threshold"
    assert wid not in ctrl.provider.health.all(), "reaped worker should be forgotten from the tracker"


# ===========================================================================
# Section 6: same-batch coscheduling split-slice corruption (#2 / #3)
# ===========================================================================
#
# ``apply_reconcile`` (the batch verb) shares one Overlay overlay across
# every worker in a batch. When a coscheduled member's terminal update requeues
# its sibling, the sibling's PENDING task state + PREEMPTED attempt state are
# written into the overlay only. Reconcile guards that later read the raw
# snapshot miss those mutations, so the outcome depends on the (non-deterministic)
# per-worker processing order and can split a coscheduled gang. The fix routes
# the attempt-state guards through ``Overlay.attempt_state`` /
# ``attempt_finished_at`` and the RPC-failure synthesis through the overlay task
# state. These regressions drive both worker orderings and assert the gang
# converges identically (order-independence).


@dataclass
class _CoschedPair:
    state: ControllerTestState
    t0: JobName
    t1: JobName
    a0: int
    a1: int
    u0: str
    u1: str


def _setup_coscheduled_running_pair(
    state: ControllerTestState,
    *,
    max_retries_failure: int = 0,
    max_retries_preemption: int = 0,
    max_task_failures: int = 0,
    sibling_assigned: bool = False,
) -> _CoschedPair:
    """Submit a 2-replica coscheduled job and place both replicas on workers.

    Replica 0 lands RUNNING on ``_W1``. Replica 1 lands on ``_W2``: RUNNING by
    default, or left ASSIGNED (never heartbeated) when ``sibling_assigned`` is
    set — the state the RPC-failure WORKER_FAILED synthesis targets (#2).
    """
    wid1, wid2 = WorkerId(_W1), WorkerId(_W2)
    register_worker(state, _W1, f"{_W1}:8080", make_worker_metadata())
    register_worker(state, _W2, f"{_W2}:8080", make_worker_metadata())

    req = make_job_request(
        name="cosched-job",
        replicas=2,
        max_retries_failure=max_retries_failure,
        max_retries_preemption=max_retries_preemption,
        max_task_failures=max_task_failures,
    )
    req.coscheduling.group_by = "job"
    tasks = submit_job(state, "cosched-job", req)
    assert len(tasks) == 2
    t0, t1 = tasks[0].task_id, tasks[1].task_id

    with state._db.transaction() as cur:
        ops.task.assign(
            cur,
            [Assignment(task_id=t0, worker_id=wid1), Assignment(task_id=t1, worker_id=wid2)],
            health=state._health,
        )
    a0 = query_task(state, t0).current_attempt_id
    a1 = query_task(state, t1).current_attempt_id

    running = [WorkerTaskUpdates(wid1, [TaskUpdate(t0, a0, job_pb2.TASK_STATE_RUNNING)])]
    if not sibling_assigned:
        running.append(WorkerTaskUpdates(wid2, [TaskUpdate(t1, a1, job_pb2.TASK_STATE_RUNNING)]))
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            running,
            health=state._health,
            now=_NOW,
        )
    return _CoschedPair(
        state=state,
        t0=t0,
        t1=t1,
        a0=a0,
        a1=a1,
        u0=_attempt_uid(state, t0, a0),
        u1=_attempt_uid(state, t1, a1),
    )


def _run_plan(worker_id: str, task_id: JobName, attempt_id: int, attempt_uid: str) -> WorkerReconcilePlan:
    """A single-attempt plan whose desired entry carries an inline ``run.request``.

    The RPC-failure synthesis only fires for desired entries that carry a
    ``run.request`` (it reads ``task_id`` / ``attempt_id`` from it), and the
    success path drops any observation whose ``attempt_uid`` is not in the plan.
    """
    spec = _spec()
    spec.task_id = task_id.to_wire()
    spec.attempt_id = attempt_id
    spec.attempt_uid = attempt_uid
    return _make_plan(worker_id, desired=[_desired_run(attempt_uid, spec=spec)])


def _apply_batch(
    state: ControllerTestState,
    plans: dict[WorkerId, WorkerReconcilePlan],
    results: list[WorkerReconcileResult],
):
    """Apply a multi-worker reconcile batch through the production verb.

    ``results`` order is the per-worker processing order (``apply_reconcile``
    iterates it in order), so it controls which worker is seen first.
    """
    plan_results = [(plans[r.worker_id], r) for r in results]
    with state._db.transaction() as cur:
        return commit_reconcile(
            cur,
            plan_results,
            now=_NOW,
        )


def test_coscheduled_running_repoll_does_not_revive_after_sibling_requeue():
    """#3: a same-batch RUNNING re-poll must not revive a sibling requeued to PENDING.

    One batch: worker A reports t0=FAILED (with retry budget, so t0 -> PENDING,
    which requeues sibling t1 to PENDING + its attempt PREEMPTED in the overlay),
    and worker B reports t1=RUNNING. Processing A first means B's RUNNING re-poll
    lands while t1's attempt is PREEMPTED-in-overlay. Pre-fix the guard read the
    raw snapshot (attempt still RUNNING), revived t1, and split the gang.
    """
    with make_controller_state() as state:
        pair = _setup_coscheduled_running_pair(state, max_retries_failure=1, max_task_failures=1)
        plans = {
            WorkerId(_W1): _run_plan(_W1, pair.t0, pair.a0, pair.u0),
            WorkerId(_W2): _run_plan(_W2, pair.t1, pair.a1, pair.u1),
        }
        # Process A (the FAILED trigger) FIRST, then B's RUNNING re-poll.
        results = [
            WorkerReconcileResult(
                worker_id=WorkerId(_W1), observations=[_obs(pair.u0, job_pb2.TASK_STATE_FAILED)], error=None
            ),
            WorkerReconcileResult(
                worker_id=WorkerId(_W2), observations=[_obs(pair.u1, job_pb2.TASK_STATE_RUNNING)], error=None
            ),
        ]
        _apply_batch(state, plans, results)

        # The gang must not split: the trigger retries to PENDING and the sibling
        # stays PENDING (its RUNNING re-poll is dropped, not applied to revive it).
        assert query_task(state, pair.t0).state == job_pb2.TASK_STATE_PENDING
        assert query_task(state, pair.t1).state == job_pb2.TASK_STATE_PENDING
        # The sibling's old attempt is terminal (COSCHED_FAILED — it was bounced
        # by its gang peer's failure, not preempted) in the overlay; the re-poll
        # must not revive it back to RUNNING.
        assert query_attempt(state, pair.t1, pair.a1).state == job_pb2.TASK_STATE_COSCHED_FAILED


def test_coscheduled_rpc_failure_does_not_split_slice():
    """#2: an RPC failure must not fabricate WORKER_FAILED for a same-batch requeued sibling.

    One batch through the production ``apply_reconcile`` verb: worker W0 succeeds
    reporting t0=WORKER_FAILED (with preemption budget, so t0 -> PENDING, which
    requeues sibling t1 to PENDING in the overlay), and worker W1's reconcile RPC
    fails. Processing W0 first means the RPC-failure synthesis for W1 runs after
    t1 is already PENDING in the overlay. Pre-fix the synthesis gated on the raw
    snapshot (t1 still ASSIGNED-equivalent), fabricated a synthetic WORKER_FAILED
    for t1, drove it terminal, and split the gang.
    """
    with make_controller_state() as state:
        # t1 stays ASSIGNED so the RPC-failure WORKER_FAILED synthesis is in
        # scope for it (the gate only fires on ASSIGNED tasks). t0 is RUNNING.
        pair = _setup_coscheduled_running_pair(state, max_retries_preemption=1, sibling_assigned=True)
        assert query_task(state, pair.t1).state == job_pb2.TASK_STATE_ASSIGNED
        plans = {
            WorkerId(_W1): _run_plan(_W1, pair.t0, pair.a0, pair.u0),
            WorkerId(_W2): _run_plan(_W2, pair.t1, pair.a1, pair.u1),
        }
        # Process W0 (the WORKER_FAILED trigger) FIRST, then W1's RPC failure.
        # W0's WORKER_FAILED requeues sibling t1 to PENDING in the overlay; W1's
        # RPC failure then runs while t1's raw snapshot still reads ASSIGNED.
        results = [
            WorkerReconcileResult(
                worker_id=WorkerId(_W1), observations=[_obs(pair.u0, job_pb2.TASK_STATE_WORKER_FAILED)], error=None
            ),
            WorkerReconcileResult(worker_id=WorkerId(_W2), observations=[], error="rpc boom"),
        ]
        _apply_batch(state, plans, results)

        # No split: both members return to PENDING for a fresh coscheduled placement.
        assert query_task(state, pair.t0).state == job_pb2.TASK_STATE_PENDING
        assert query_task(state, pair.t1).state == job_pb2.TASK_STATE_PENDING
        # The sibling must NOT be driven terminal by a fabricated WORKER_FAILED.
        assert query_attempt(state, pair.t1, pair.a1).state != job_pb2.TASK_STATE_WORKER_FAILED
        assert query_attempt(state, pair.t1, pair.a1).finished_at_ms is None


@pytest.mark.parametrize("trigger_first", [True, False])
def test_reconcile_batch_order_independent_coscheduled_failure(trigger_first):
    """Durable guard: the same coscheduled-failure batch converges to the same DB
    state under BOTH worker orderings.

    Builds the #3 batch (worker A: t0=FAILED-with-budget; worker B: t1=RUNNING)
    and applies it with A-first and B-first. Both orderings must reach the same
    final task states, attempt states, attempt-finished-ness, and preemption
    counts. Pre-fix the two orderings diverged (one revived t1 to RUNNING).
    """

    def run(trigger_first_order: bool) -> dict[str, Any]:
        with make_controller_state() as state:
            pair = _setup_coscheduled_running_pair(state, max_retries_failure=1, max_task_failures=1)
            plans = {
                WorkerId(_W1): _run_plan(_W1, pair.t0, pair.a0, pair.u0),
                WorkerId(_W2): _run_plan(_W2, pair.t1, pair.a1, pair.u1),
            }
            trigger = WorkerReconcileResult(
                worker_id=WorkerId(_W1), observations=[_obs(pair.u0, job_pb2.TASK_STATE_FAILED)], error=None
            )
            repoll = WorkerReconcileResult(
                worker_id=WorkerId(_W2), observations=[_obs(pair.u1, job_pb2.TASK_STATE_RUNNING)], error=None
            )
            results = [trigger, repoll] if trigger_first_order else [repoll, trigger]
            _apply_batch(state, plans, results)
            return {
                "t0_state": query_task(state, pair.t0).state,
                "t1_state": query_task(state, pair.t1).state,
                "t0_attempt_state": query_attempt(state, pair.t0, pair.a0).state,
                "t1_attempt_state": query_attempt(state, pair.t1, pair.a1).state,
                "t1_attempt_finished": query_attempt(state, pair.t1, pair.a1).finished_at_ms is not None,
                "t1_preemption_count": query_task(state, pair.t1).preemption_count,
            }

    observed = run(trigger_first)
    reference = run(True)
    # Both orderings converge to the same final state, and that state is the
    # un-split gang (both PENDING, sibling attempt terminal not RUNNING).
    assert observed == reference
    assert observed["t0_state"] == job_pb2.TASK_STATE_PENDING
    assert observed["t1_state"] == job_pb2.TASK_STATE_PENDING
    # The bounced sibling is stamped COSCHED_FAILED, not PREEMPTED: it was
    # requeued because its gang peer failed, so deriving preemption_count from
    # attempts must not charge it (see reconcile/peers.requeue_coscheduled_siblings).
    assert observed["t1_attempt_state"] == job_pb2.TASK_STATE_COSCHED_FAILED
    assert observed["t1_preemption_count"] == 0
