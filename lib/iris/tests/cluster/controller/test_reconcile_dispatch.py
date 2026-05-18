# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for controller dispatch via Reconcile RPC (B.4).

Tests cover:
- Flag off: all workers receive legacy StartTasks+PollTasks RPCs; Reconcile is
  never called.
- Flag on: all workers receive the Reconcile RPC; legacy start_tasks/poll_tasks
  are not called.
- Flag on, end-to-end: ASSIGNED → BUILDING → RUNNING → SUCCEEDED through the
  Reconcile wire over multiple ticks.
- Flag on, MISSING observation: apply_reconcile_response fails the attempt as
  worker_lost_spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from iris.cluster.controller.provider import ProviderError
from iris.cluster.controller.reconcile import WorkerReconcileDispatch
from iris.cluster.controller.transitions import Assignment, HeartbeatApplyRequest, TaskUpdate
from iris.cluster.controller.worker_provider import ReconcileRpcResult, WorkerReconcileResult
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2, worker_pb2

from .conftest import (
    make_job_request,
    make_worker_metadata,
    query_task,
    register_worker,
    submit_job,
)

# ---------------------------------------------------------------------------
# Recording providers
# ---------------------------------------------------------------------------

_W1 = "worker-1"
_WORKER_ADDR = "127.0.0.1:9999"


@dataclass
class RecordingLegacyProvider:
    """Records reconcile_workers (legacy wire) calls; never calls reconcile_workers_via_reconcile."""

    calls: list[list[WorkerReconcileDispatch]] = field(default_factory=list)
    reconcile_calls: list[Any] = field(default_factory=list)

    def reconcile_workers(self, plans: list[WorkerReconcileDispatch]) -> list[WorkerReconcileResult]:
        self.calls.append(list(plans))
        return [
            WorkerReconcileResult(
                worker_id=plan.worker_id,
                start_response=worker_pb2.Worker.StartTasksResponse() if plan.start_tasks else None,
                start_error=None,
                poll_updates=[],
                poll_error=None,
            )
            for plan in plans
        ]

    def reconcile_workers_via_reconcile(self, plans_with_addresses: Any) -> list[ReconcileRpcResult]:
        self.reconcile_calls.append(plans_with_addresses)
        raise AssertionError("reconcile_workers_via_reconcile must NOT be called when flag is off")

    def ping_workers(self, workers: Any) -> list[Any]:
        return []

    def get_process_status(self, worker_id: Any, address: Any, request: Any) -> Any:
        raise ProviderError("recording provider: get_process_status not supported")

    def on_worker_failed(self, worker_id: Any, address: Any) -> None:
        pass

    def profile_task(self, address: Any, request: Any, timeout_ms: int) -> Any:
        raise ProviderError("recording provider: profile_task not supported")

    def close(self) -> None:
        pass


@dataclass
class RecordingReconcileProvider:
    """Records reconcile_workers_via_reconcile (new wire) calls; raises if legacy wire is used."""

    calls: list[list[Any]] = field(default_factory=list)
    # State machine for responses: maps worker_id -> list of (state, exit_code, error)
    # to return on successive calls; cycles through the list.
    response_sequence: list[int] = field(default_factory=list)
    _call_index: int = field(default=0, init=False)

    def reconcile_workers_via_reconcile(self, plans_with_addresses: Any) -> list[ReconcileRpcResult]:
        self.calls.append(list(plans_with_addresses))
        results: list[ReconcileRpcResult] = []
        for plan, _address, _proto_req in plans_with_addresses:
            worker_id = WorkerId(plan.request.worker_id)
            # Build an empty response by default (no observations).
            response = worker_pb2.Worker.ReconcileResponse(
                worker_id=str(worker_id),
                observed=[],
            )
            results.append(ReconcileRpcResult(worker_id=worker_id, response=response, error=None))
        return results

    def reconcile_workers(self, plans: list[WorkerReconcileDispatch]) -> list[WorkerReconcileResult]:
        raise AssertionError("Legacy reconcile_workers must NOT be called when Reconcile RPC flag is on")

    def ping_workers(self, workers: Any) -> list[Any]:
        return []

    def get_process_status(self, worker_id: Any, address: Any, request: Any) -> Any:
        raise ProviderError("recording provider: get_process_status not supported")

    def on_worker_failed(self, worker_id: Any, address: Any) -> None:
        pass

    def profile_task(self, address: Any, request: Any, timeout_ms: int) -> Any:
        raise ProviderError("recording provider: profile_task not supported")

    def close(self) -> None:
        pass


@dataclass
class ObservingReconcileProvider:
    """Returns configurable observations per tick for end-to-end state machine tests."""

    # List of (observations) to return on successive ticks across ALL workers.
    # Each element is a list of worker_pb2.Worker.AttemptObservation protos.
    tick_observations: list[list[worker_pb2.Worker.AttemptObservation]] = field(default_factory=list)
    _tick: int = field(default=0, init=False)

    def reconcile_workers_via_reconcile(self, plans_with_addresses: Any) -> list[ReconcileRpcResult]:
        observations = []
        if self._tick < len(self.tick_observations):
            observations = self.tick_observations[self._tick]
        self._tick += 1

        results: list[ReconcileRpcResult] = []
        for plan, _address, _proto_req in plans_with_addresses:
            worker_id = WorkerId(plan.request.worker_id)
            response = worker_pb2.Worker.ReconcileResponse(
                worker_id=str(worker_id),
                observed=observations,
            )
            results.append(ReconcileRpcResult(worker_id=worker_id, response=response, error=None))
        return results

    def reconcile_workers(self, plans: list[WorkerReconcileDispatch]) -> list[WorkerReconcileResult]:
        raise AssertionError("Legacy reconcile_workers must NOT be called when Reconcile RPC flag is on")

    def ping_workers(self, workers: Any) -> list[Any]:
        return []

    def get_process_status(self, worker_id: Any, address: Any, request: Any) -> Any:
        raise ProviderError("recording provider: get_process_status not supported")

    def on_worker_failed(self, worker_id: Any, address: Any) -> None:
        pass

    def profile_task(self, address: Any, request: Any, timeout_ms: int) -> Any:
        raise ProviderError("recording provider: profile_task not supported")

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _register_worker(state, worker_id: str = _W1, address: str = _WORKER_ADDR) -> WorkerId:
    return register_worker(state, worker_id, address, make_worker_metadata(cpu=8, memory_bytes=32 * 1024**3))


def _assign_task(state, task_id: JobName, worker_id: WorkerId) -> int:
    """Queue an assignment and return the attempt_id."""
    with state._store.transaction() as cur:
        state.queue_assignments(cur, [Assignment(task_id=task_id, worker_id=worker_id)])
    task = query_task(state, task_id)
    assert task is not None
    return task.current_attempt_id


def _transition(state, task_id: JobName, worker_id: WorkerId, attempt_id: int, new_state: int) -> None:
    with state._store.transaction() as cur:
        state.apply_task_updates(
            cur,
            HeartbeatApplyRequest(
                worker_id=worker_id,
                updates=[TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=new_state)],
            ),
        )


# ---------------------------------------------------------------------------
# Test: flag off → legacy wire only
# ---------------------------------------------------------------------------


def test_flag_off_uses_legacy_wire(make_controller, tmp_path):
    """Flag off: reconcile_workers (legacy) is called; Reconcile RPC is not."""
    provider = RecordingLegacyProvider()
    ctrl = make_controller(provider=provider, reconcile_rpc_enabled=False)
    state = ctrl._transitions

    wid = _register_worker(state)
    tasks = submit_job(state, "job-a", make_job_request(name="job-a", replicas=1))
    assert tasks
    with state._store.transaction() as cur:
        state.queue_assignments(cur, [Assignment(task_id=tasks[0].task_id, worker_id=wid)])

    ctrl._reconcile_worker_batch()

    # Legacy wire was called; new wire was not.
    assert len(provider.calls) == 1
    assert provider.reconcile_calls == []


# ---------------------------------------------------------------------------
# Test: flag on → Reconcile RPC only
# ---------------------------------------------------------------------------


def test_flag_on_uses_reconcile_rpc(make_controller, tmp_path):
    """Flag on: reconcile_workers_via_reconcile is called; legacy wire is not."""
    provider = RecordingReconcileProvider()
    ctrl = make_controller(provider=provider, reconcile_rpc_enabled=True)
    state = ctrl._transitions

    wid = _register_worker(state)
    tasks = submit_job(state, "job-b", make_job_request(name="job-b", replicas=1))
    assert tasks
    with state._store.transaction() as cur:
        state.queue_assignments(cur, [Assignment(task_id=tasks[0].task_id, worker_id=wid)])

    ctrl._reconcile_worker_batch()

    # New wire was called; legacy was not (provider raises AssertionError if called).
    assert len(provider.calls) == 1
    # Each call element is a list of (plan, address, proto_request) triples.
    assert len(provider.calls[0]) == 1  # one worker


# ---------------------------------------------------------------------------
# Test: end-to-end ASSIGNED → BUILDING → RUNNING → SUCCEEDED via Reconcile wire
# ---------------------------------------------------------------------------


def test_e2e_assigned_to_succeeded_via_reconcile(make_controller, tmp_path):
    """End-to-end: task progresses ASSIGNED → BUILDING → RUNNING → SUCCEEDED via Reconcile ticks."""
    # Phase 1: set up controller with empty observations provider, get task_id and attempt_id.
    ctrl = make_controller(provider=ObservingReconcileProvider(), reconcile_rpc_enabled=True)
    state = ctrl._transitions

    wid = _register_worker(state)
    tasks = submit_job(state, "job-e2e", make_job_request(name="job-e2e", replicas=1))
    assert tasks
    task_id = tasks[0].task_id
    attempt_id = _assign_task(state, task_id, wid)

    # Phase 2: swap in a provider that knows the task_id/attempt_id and drives the state machine.
    ctrl._provider = ObservingReconcileProvider(
        tick_observations=[
            [],  # tick 1: no report yet → task stays ASSIGNED
            [  # tick 2: BUILDING
                worker_pb2.Worker.AttemptObservation(
                    task_id=task_id.to_wire(),
                    attempt_id=attempt_id,
                    state=job_pb2.TASK_STATE_BUILDING,
                )
            ],
            [  # tick 3: RUNNING
                worker_pb2.Worker.AttemptObservation(
                    task_id=task_id.to_wire(),
                    attempt_id=attempt_id,
                    state=job_pb2.TASK_STATE_RUNNING,
                )
            ],
            [  # tick 4: SUCCEEDED
                worker_pb2.Worker.AttemptObservation(
                    task_id=task_id.to_wire(),
                    attempt_id=attempt_id,
                    state=job_pb2.TASK_STATE_SUCCEEDED,
                    exit_code=0,
                )
            ],
        ]
    )

    # Tick 1: no observations, task stays ASSIGNED.
    ctrl._reconcile_worker_batch()
    task_after_tick1 = query_task(state, task_id)
    assert task_after_tick1 is not None
    assert task_after_tick1.state == job_pb2.TASK_STATE_ASSIGNED

    # Tick 2: BUILDING observation.
    ctrl._reconcile_worker_batch()
    task_after_tick2 = query_task(state, task_id)
    assert task_after_tick2 is not None
    assert task_after_tick2.state == job_pb2.TASK_STATE_BUILDING

    # Tick 3: RUNNING observation.
    ctrl._reconcile_worker_batch()
    task_after_tick3 = query_task(state, task_id)
    assert task_after_tick3 is not None
    assert task_after_tick3.state == job_pb2.TASK_STATE_RUNNING

    # Tick 4: SUCCEEDED observation.
    ctrl._reconcile_worker_batch()
    task_after_tick4 = query_task(state, task_id)
    assert task_after_tick4 is not None
    assert task_after_tick4.state == job_pb2.TASK_STATE_SUCCEEDED


# ---------------------------------------------------------------------------
# Test: flag on, MISSING observation → worker_lost_spec
# ---------------------------------------------------------------------------


def test_missing_observation_via_reconcile_fails_attempt(make_controller):
    """Reconcile wire MISSING observation causes attempt to fail as worker_lost_spec."""
    provider_placeholder = RecordingReconcileProvider()
    ctrl = make_controller(provider=provider_placeholder, reconcile_rpc_enabled=True)
    state = ctrl._transitions

    wid = _register_worker(state)
    tasks = submit_job(state, "job-missing", make_job_request(name="job-missing", replicas=1))
    assert tasks
    task_id = tasks[0].task_id
    attempt_id = _assign_task(state, task_id, wid)

    # Advance to RUNNING so we have a non-ASSIGNED executing task.
    _transition(state, task_id, wid, attempt_id, job_pb2.TASK_STATE_BUILDING)
    _transition(state, task_id, wid, attempt_id, job_pb2.TASK_STATE_RUNNING)

    task_running = query_task(state, task_id)
    assert task_running is not None
    assert task_running.state == job_pb2.TASK_STATE_RUNNING

    # Now inject a MISSING observation via the Reconcile wire.
    missing_obs = [
        worker_pb2.Worker.AttemptObservation(
            task_id=task_id.to_wire(),
            attempt_id=attempt_id,
            state=job_pb2.TASK_STATE_MISSING,
        )
    ]
    ctrl._provider = ObservingReconcileProvider(tick_observations=[missing_obs])

    ctrl._reconcile_worker_batch()

    task_final = query_task(state, task_id)
    assert task_final is not None
    assert task_final.state == job_pb2.TASK_STATE_FAILED
    assert task_final.error == "worker_lost_spec"


# ---------------------------------------------------------------------------
# Test: flag on, empty worker still calls Reconcile
# ---------------------------------------------------------------------------


def test_flag_on_empty_worker_still_calls_reconcile(make_controller):
    """With flag on and no tasks, Reconcile is still called (worker auto-kills strays)."""
    provider = RecordingReconcileProvider()
    ctrl = make_controller(provider=provider, reconcile_rpc_enabled=True)
    state = ctrl._transitions

    _register_worker(state)

    ctrl._reconcile_worker_batch()

    assert len(provider.calls) == 1
    # One worker, empty desired list.
    assert len(provider.calls[0]) == 1
    _, _, proto_req = provider.calls[0][0]
    assert len(proto_req.desired) == 0


# ---------------------------------------------------------------------------
# Test: reconcile_request_from_plan proto shape
# ---------------------------------------------------------------------------


def test_reconcile_request_from_plan_shape():
    """reconcile_request_from_plan produces correct proto from a WorkerReconcilePlan."""
    from iris.cluster.controller.reconcile import (
        AttemptSpec,
        DesiredAttempt,
        ReconcileRequest,
        StopReason,
        WorkerReconcilePlan,
        reconcile_request_from_plan,
    )
    from iris.cluster.types import AttemptUid

    plan = WorkerReconcilePlan(
        request=ReconcileRequest(
            worker_id="w1",
            desired=[
                DesiredAttempt(
                    attempt_uid=AttemptUid(""),
                    intent_run=AttemptSpec(),
                    task_id="/u/t:0",
                    attempt_id=0,
                ),
                DesiredAttempt(
                    attempt_uid=AttemptUid(""),
                    intent_stop=StopReason.CANCELLED,
                    task_id="/u/t:1",
                    attempt_id=0,
                ),
            ],
        )
    )

    proto = reconcile_request_from_plan(plan)

    assert proto.worker_id == "w1"
    assert len(proto.desired) == 2

    run_entry = proto.desired[0]
    assert run_entry.task_id == "/u/t:0"
    assert run_entry.HasField("run")
    assert not run_entry.HasField("stop")

    stop_entry = proto.desired[1]
    assert stop_entry.task_id == "/u/t:1"
    assert stop_entry.stop == worker_pb2.Worker.STOP_REASON_CANCELLED
    assert not stop_entry.HasField("run")


# ---------------------------------------------------------------------------
# Test: observations_from_reconcile_response round-trip
# ---------------------------------------------------------------------------


def test_observations_from_reconcile_response_round_trip():
    """observations_from_reconcile_response correctly maps proto fields."""
    from iris.cluster.controller.reconcile import observations_from_reconcile_response

    response = worker_pb2.Worker.ReconcileResponse(
        worker_id="w1",
        observed=[
            worker_pb2.Worker.AttemptObservation(
                task_id="/u/t:0",
                attempt_id=2,
                state=job_pb2.TASK_STATE_RUNNING,
                container_id="c1",
            ),
            worker_pb2.Worker.AttemptObservation(
                task_id="/u/t:1",
                attempt_id=1,
                state=job_pb2.TASK_STATE_MISSING,
            ),
        ],
    )

    obs = observations_from_reconcile_response(response)

    assert len(obs) == 2
    assert obs[0].task_id == "/u/t:0"
    assert obs[0].attempt_id_compat == 2
    assert obs[0].state == job_pb2.TASK_STATE_RUNNING
    assert obs[0].container_id == "c1"

    assert obs[1].task_id == "/u/t:1"
    assert obs[1].state == job_pb2.TASK_STATE_MISSING
