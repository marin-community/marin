# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curated replay scenarios.

Each scenario is a function ``def scenario_NAME(transitions, clock) -> None``
that drives ``ControllerTransitions`` through a sequence of mutations. The
caller (the pytest fixture or an ad-hoc runner) freezes the clock to a
deterministic monotonic counter so the DB state is byte-identical across
runs and the committed goldens stay stable.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.controller.schema import EndpointRow
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    HeartbeatApplyRequest,
    ReservationClaim,
    TaskUpdate,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2
from rigging import timing
from rigging.timing import Duration, Timestamp

from tests.cluster.controller.replay.events import (
    AddEndpoint,
    ApplyDirectProviderUpdates,
    ApplyTaskUpdates,
    BufferDirectKill,
    CancelJob,
    CancelTasksForTimeout,
    DrainForDirectProvider,
    PreemptTask,
    QueueAssignments,
    RegisterOrRefreshWorker,
    RemoveEndpoint,
    ReplaceReservationClaims,
    SubmitJob,
    apply_event,
)


class FrozenClock:
    """Monotonic counter standing in for ``Timestamp.now()`` during a scenario.

    Every call to :meth:`now` returns the next millisecond; :meth:`at`
    returns the current value without advancing. The class is a plain
    substrate — the ``frozen_clock`` fixture (conftest.py) patches
    ``Timestamp.now`` / ``rigging.timing._now_ms`` to route through it,
    and scenarios read timestamps via ``clock.at()`` so goldens encode
    the exact count of internal ``Timestamp.now()`` calls performed by
    the transition methods.
    """

    # Baseline epoch for all scenarios: 2024-01-01 00:00:00 UTC. Concrete
    # value chosen so reasoning about timestamps in the goldens is easier
    # than with epoch=0 (which renders as 1970 dates that look like bugs).
    EPOCH_MS: int = 1_704_067_200_000

    def __init__(self, start_ms: int = EPOCH_MS) -> None:
        self._t = start_ms

    def now(self) -> Timestamp:
        ts = Timestamp.from_ms(self._t)
        self._t += 1
        return ts

    def at(self) -> Timestamp:
        return Timestamp.from_ms(self._t)

    def advance_ms(self, ms: int) -> None:
        self._t += ms


@contextmanager
def frozen_clock() -> Iterator[FrozenClock]:
    """Patch ``Timestamp.now`` / ``rigging.timing._now_ms`` with a monotonic counter.

    Must wrap scenario execution only — enter AFTER ``ControllerDB``
    construction so schema migrations use real time. The returned
    :class:`FrozenClock` is shared by scenario code (via ``clock.at()``)
    and every internal ``Timestamp.now()`` call inside transitions.

    ``Timestamp.now`` is a ``classmethod``. Reading ``Timestamp.now``
    goes through the descriptor protocol and returns a bound method,
    not the descriptor itself — assigning that bound method back to
    the class as a "restore" would leave ``Timestamp.now`` as a plain
    method and break subclass binding. We save/restore the raw
    descriptor via ``Timestamp.__dict__`` so the original classmethod
    semantics are preserved byte-for-byte.
    """
    clock = FrozenClock()
    saved_now_desc = Timestamp.__dict__["now"]
    saved_now_ms = timing._now_ms
    Timestamp.now = classmethod(lambda cls: clock.now())  # type: ignore[method-assign]
    timing._now_ms = lambda: clock.now().epoch_ms()  # type: ignore[assignment]
    try:
        yield clock
    finally:
        Timestamp.now = saved_now_desc  # type: ignore[method-assign]
        timing._now_ms = saved_now_ms  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Scenario building blocks
# ---------------------------------------------------------------------------


def _make_metadata(*, cpu: int = 8, memory_bytes: int = 16 * 1024**3) -> job_pb2.WorkerMetadata:
    """Plain CPU worker with the well-known device attributes populated."""
    device = job_pb2.DeviceConfig()
    device.cpu.CopyFrom(job_pb2.CpuDevice(variant="cpu"))
    meta = job_pb2.WorkerMetadata(
        hostname="replay-worker",
        ip_address="127.0.0.1",
        cpu_count=cpu,
        memory_bytes=memory_bytes,
        disk_bytes=memory_bytes,
        device=device,
    )
    meta.attributes[WellKnownAttribute.DEVICE_TYPE].string_value = "cpu"
    return meta


def _entrypoint() -> job_pb2.RuntimeEntrypoint:
    ep = job_pb2.RuntimeEntrypoint()
    ep.run_command.argv[:] = ["python", "-c", "pass"]
    return ep


def _job_request(
    name: str,
    *,
    replicas: int = 1,
    max_retries_failure: int = 0,
    max_retries_preemption: int = 0,
    coscheduled: bool = False,
    reservation_entries: int = 0,
) -> tuple[JobName, controller_pb2.Controller.LaunchJobRequest]:
    job_name = JobName.root("test-user", name)
    request = controller_pb2.Controller.LaunchJobRequest(
        name=job_name.to_wire(),
        entrypoint=_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        max_retries_failure=max_retries_failure,
        max_retries_preemption=max_retries_preemption,
        replicas=replicas,
    )
    if coscheduled:
        request.coscheduling.group_by = "task_index"
    if reservation_entries > 0:
        for _ in range(reservation_entries):
            entry = request.reservation.entries.add()
            entry.resources.CopyFrom(
                job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            )
    return job_name, request


def _register_worker(
    transitions: ControllerTransitions,
    clock: FrozenClock,
    worker_id: str,
    *,
    address: str | None = None,
) -> WorkerId:
    wid = WorkerId(worker_id)
    apply_event(
        transitions,
        RegisterOrRefreshWorker(
            worker_id=wid,
            address=address or f"{worker_id}:8080",
            metadata=_make_metadata(),
            ts=clock.at(),
        ),
    )
    return wid


def _submit(
    transitions: ControllerTransitions,
    clock: FrozenClock,
    name: str,
    **kw,
) -> JobName:
    job_id, req = _job_request(name, **kw)
    apply_event(transitions, SubmitJob(job_id=job_id, request=req, ts=clock.at()))
    return job_id


def _task_ids(transitions: ControllerTransitions, job_id: JobName) -> list[JobName]:
    with transitions._store.read_snapshot() as snap:
        rows = snap.fetchall(
            "SELECT task_id FROM tasks WHERE job_id = ? ORDER BY task_index ASC",
            (job_id.to_wire(),),
        )
    return [JobName.from_wire(str(row["task_id"])) for row in rows]


def _current_attempt(transitions: ControllerTransitions, task_id: JobName) -> int:
    with transitions._store.read_snapshot() as snap:
        row = snap.fetchone(
            "SELECT current_attempt_id FROM tasks WHERE task_id = ?",
            (task_id.to_wire(),),
        )
    assert row is not None, f"task missing: {task_id}"
    return int(row["current_attempt_id"])


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenario_submit_simple(transitions: ControllerTransitions, clock: FrozenClock) -> None:
    """Submit a single 1-replica job."""
    _submit(transitions, clock, "simple-job")


def scenario_submit_with_reservation(transitions: ControllerTransitions, clock: FrozenClock) -> None:
    """Submit a job that carries a reservation entry — exercises holder creation."""
    _submit(transitions, clock, "reservation-job", reservation_entries=1)


def scenario_register_assign_run_succeed(transitions: ControllerTransitions, clock: FrozenClock) -> None:
    """Full happy-path lifecycle: register worker, submit, assign, run, succeed."""
    worker_id = _register_worker(transitions, clock, "w-happy")
    job_id = _submit(transitions, clock, "happy-job")
    (task_id,) = _task_ids(transitions, job_id)
    apply_event(
        transitions,
        QueueAssignments(assignments=[Assignment(task_id=task_id, worker_id=worker_id)]),
    )
    attempt = _current_attempt(transitions, task_id)
    apply_event(
        transitions,
        ApplyTaskUpdates(
            request=HeartbeatApplyRequest(
                worker_id=worker_id,
                worker_resource_snapshot=None,
                updates=[TaskUpdate(task_id=task_id, attempt_id=attempt, new_state=job_pb2.TASK_STATE_RUNNING)],
            ),
        ),
    )
    apply_event(
        transitions,
        ApplyTaskUpdates(
            request=HeartbeatApplyRequest(
                worker_id=worker_id,
                worker_resource_snapshot=None,
                updates=[TaskUpdate(task_id=task_id, attempt_id=attempt, new_state=job_pb2.TASK_STATE_SUCCEEDED)],
            ),
        ),
    )


def scenario_task_failure_with_retry(transitions: ControllerTransitions, clock: FrozenClock) -> None:
    """Submit with retry budget, fail once, observe retry to PENDING, then succeed."""
    worker_id = _register_worker(transitions, clock, "w-retry")
    job_id = _submit(transitions, clock, "retry-job", max_retries_failure=2)
    (task_id,) = _task_ids(transitions, job_id)
    apply_event(transitions, QueueAssignments([Assignment(task_id=task_id, worker_id=worker_id)]))
    first_attempt = _current_attempt(transitions, task_id)
    apply_event(
        transitions,
        ApplyTaskUpdates(
            HeartbeatApplyRequest(
                worker_id=worker_id,
                worker_resource_snapshot=None,
                updates=[
                    TaskUpdate(
                        task_id=task_id,
                        attempt_id=first_attempt,
                        new_state=job_pb2.TASK_STATE_FAILED,
                        error="boom",
                    )
                ],
            )
        ),
    )
    apply_event(transitions, QueueAssignments([Assignment(task_id=task_id, worker_id=worker_id)]))
    second_attempt = _current_attempt(transitions, task_id)
    apply_event(
        transitions,
        ApplyTaskUpdates(
            HeartbeatApplyRequest(
                worker_id=worker_id,
                worker_resource_snapshot=None,
                updates=[TaskUpdate(task_id=task_id, attempt_id=second_attempt, new_state=job_pb2.TASK_STATE_SUCCEEDED)],
            )
        ),
    )


def scenario_worker_failure_cascade(transitions: ControllerTransitions, clock: FrozenClock) -> None:
    """Register → assign → fail the worker via ``fail_workers`` (multi-tx orchestrator)."""
    worker_id = _register_worker(transitions, clock, "w-doomed")
    job_id = _submit(transitions, clock, "cascade-job")
    (task_id,) = _task_ids(transitions, job_id)
    apply_event(transitions, QueueAssignments([Assignment(task_id=task_id, worker_id=worker_id)]))
    # Direct call: fail_workers is intentionally not an IrisEvent.
    transitions.fail_workers([(worker_id, "w-doomed:8080", "node lost")])


def scenario_cancel_running_job(transitions: ControllerTransitions, clock: FrozenClock) -> None:
    """Submit a 2-replica job, assign both tasks, then cancel."""
    worker_id = _register_worker(transitions, clock, "w-cancel", address="w-cancel:8080")
    job_id = _submit(transitions, clock, "cancel-job", replicas=2)
    tasks = _task_ids(transitions, job_id)
    apply_event(
        transitions,
        QueueAssignments([Assignment(task_id=t, worker_id=worker_id) for t in tasks]),
    )
    apply_event(transitions, CancelJob(job_id=job_id, reason="user-cancel"))


def scenario_preempt_task(transitions: ControllerTransitions, clock: FrozenClock) -> None:
    """Assign a task and preempt it terminally (no preemption-retry budget)."""
    worker_id = _register_worker(transitions, clock, "w-preempt")
    job_id = _submit(transitions, clock, "preempt-job", max_retries_preemption=0)
    (task_id,) = _task_ids(transitions, job_id)
    apply_event(transitions, QueueAssignments([Assignment(task_id=task_id, worker_id=worker_id)]))
    apply_event(transitions, PreemptTask(task_id=task_id, reason="reclaim"))


def scenario_coscheduled_timeout(transitions: ControllerTransitions, clock: FrozenClock) -> None:
    """Coscheduled 2-replica job; timeout one task and observe sibling cascade."""
    worker_a = _register_worker(transitions, clock, "w-cosched-a", address="w-cosched-a:8080")
    worker_b = _register_worker(transitions, clock, "w-cosched-b", address="w-cosched-b:8080")
    job_id = _submit(transitions, clock, "cosched-job", replicas=2, coscheduled=True)
    tasks = _task_ids(transitions, job_id)
    apply_event(
        transitions,
        QueueAssignments(
            [
                Assignment(task_id=tasks[0], worker_id=worker_a),
                Assignment(task_id=tasks[1], worker_id=worker_b),
            ],
        ),
    )
    apply_event(transitions, CancelTasksForTimeout(task_ids=frozenset({tasks[0]}), reason="execution-timeout"))


def scenario_coscheduled_failure_retry_bounces_siblings(transitions: ControllerTransitions, clock: FrozenClock) -> None:
    """Coscheduled 2-replica job; one task hits a transient failure with retry
    budget remaining. Siblings must bounce to PENDING so the retry re-coschedules
    atomically — otherwise the lone PENDING retry can land on a different slice
    and split the SPMD mesh.
    """
    worker_a = _register_worker(transitions, clock, "w-cosched-fail-a", address="w-cosched-fail-a:8080")
    worker_b = _register_worker(transitions, clock, "w-cosched-fail-b", address="w-cosched-fail-b:8080")
    job_id = _submit(
        transitions,
        clock,
        "cosched-fail",
        replicas=2,
        coscheduled=True,
        max_retries_failure=2,
    )
    tasks = _task_ids(transitions, job_id)
    apply_event(
        transitions,
        QueueAssignments(
            [
                Assignment(task_id=tasks[0], worker_id=worker_a),
                Assignment(task_id=tasks[1], worker_id=worker_b),
            ],
        ),
    )
    # Drive task-0 to RUNNING then fail it transiently.
    a0 = _current_attempt(transitions, tasks[0])
    a1 = _current_attempt(transitions, tasks[1])
    apply_event(
        transitions,
        ApplyTaskUpdates(
            HeartbeatApplyRequest(
                worker_id=worker_a,
                worker_resource_snapshot=None,
                updates=[TaskUpdate(task_id=tasks[0], attempt_id=a0, new_state=job_pb2.TASK_STATE_RUNNING)],
            )
        ),
    )
    apply_event(
        transitions,
        ApplyTaskUpdates(
            HeartbeatApplyRequest(
                worker_id=worker_b,
                worker_resource_snapshot=None,
                updates=[TaskUpdate(task_id=tasks[1], attempt_id=a1, new_state=job_pb2.TASK_STATE_RUNNING)],
            )
        ),
    )
    apply_event(
        transitions,
        ApplyTaskUpdates(
            HeartbeatApplyRequest(
                worker_id=worker_a,
                worker_resource_snapshot=None,
                updates=[
                    TaskUpdate(
                        task_id=tasks[0],
                        attempt_id=a0,
                        new_state=job_pb2.TASK_STATE_FAILED,
                        error="transient-tpu-init",
                    )
                ],
            )
        ),
    )


def scenario_coscheduled_preempt_retry_bounces_siblings(transitions: ControllerTransitions, clock: FrozenClock) -> None:
    """Coscheduled 2-replica job; controller preempts one task with budget remaining.
    Siblings must bounce to PENDING and the original worker must be in the kill set
    so a stale TPU process doesn't outlive its bookkeeping.
    """
    worker_a = _register_worker(transitions, clock, "w-cosched-preempt-a", address="w-cosched-preempt-a:8080")
    worker_b = _register_worker(transitions, clock, "w-cosched-preempt-b", address="w-cosched-preempt-b:8080")
    job_id = _submit(
        transitions,
        clock,
        "cosched-preempt",
        replicas=2,
        coscheduled=True,
        max_retries_preemption=2,
    )
    tasks = _task_ids(transitions, job_id)
    apply_event(
        transitions,
        QueueAssignments(
            [
                Assignment(task_id=tasks[0], worker_id=worker_a),
                Assignment(task_id=tasks[1], worker_id=worker_b),
            ],
        ),
    )
    apply_event(transitions, PreemptTask(task_id=tasks[0], reason="evicted-by-prod"))


def scenario_direct_provider_cycle(transitions: ControllerTransitions, clock: FrozenClock) -> None:
    """Submit a job with no worker, drain to direct-provider, then mark RUNNING."""
    job_id = _submit(transitions, clock, "direct-job")
    (task_id,) = _task_ids(transitions, job_id)
    apply_event(transitions, DrainForDirectProvider(max_promotions=4))
    attempt = _current_attempt(transitions, task_id)
    apply_event(
        transitions,
        ApplyDirectProviderUpdates(
            updates=[TaskUpdate(task_id=task_id, attempt_id=attempt, new_state=job_pb2.TASK_STATE_RUNNING)],
        ),
    )


def scenario_prune_old_data(transitions: ControllerTransitions, clock: FrozenClock) -> None:
    """Submit, mark task succeeded, age out, then call prune_old_data directly."""
    worker_id = _register_worker(transitions, clock, "w-prune")
    job_id = _submit(transitions, clock, "prune-job")
    (task_id,) = _task_ids(transitions, job_id)
    apply_event(transitions, QueueAssignments([Assignment(task_id=task_id, worker_id=worker_id)]))
    attempt = _current_attempt(transitions, task_id)
    apply_event(
        transitions,
        ApplyTaskUpdates(
            HeartbeatApplyRequest(
                worker_id=worker_id,
                worker_resource_snapshot=None,
                updates=[TaskUpdate(task_id=task_id, attempt_id=attempt, new_state=job_pb2.TASK_STATE_SUCCEEDED)],
            )
        ),
    )
    # Backdate finished_at so the prune sweep picks it up.
    with transitions._store.transaction() as cur:
        cur.execute(
            "UPDATE jobs SET finished_at_ms = 1 WHERE job_id = ?",
            (job_id.to_wire(),),
        )
    # Advance the clock so retention math classifies the row as old.
    clock.advance_ms(10_000)
    # Direct call: prune_old_data is intentionally not an IrisEvent.
    transitions.prune_old_data(
        job_retention=Duration.from_seconds(0),
        worker_retention=Duration.from_seconds(3600),
        profile_retention=Duration.from_seconds(3600),
        pause_between_s=0.0,
    )


def scenario_endpoint_register_remove(transitions: ControllerTransitions, clock: FrozenClock) -> None:
    """Add an endpoint to a non-terminal task, then remove it."""
    worker_id = _register_worker(transitions, clock, "w-endpoint")
    job_id = _submit(transitions, clock, "endpoint-job")
    (task_id,) = _task_ids(transitions, job_id)
    apply_event(transitions, QueueAssignments([Assignment(task_id=task_id, worker_id=worker_id)]))
    endpoint = EndpointRow(
        endpoint_id="ep-replay",
        name="api",
        address="endpoint:9000",
        task_id=task_id,
        metadata={"protocol": "grpc"},
        registered_at=clock.at(),
    )
    apply_event(transitions, AddEndpoint(endpoint=endpoint))
    apply_event(transitions, RemoveEndpoint(endpoint_id="ep-replay"))


def scenario_replace_reservation_claims(transitions: ControllerTransitions, clock: FrozenClock) -> None:
    """Register two workers, replace reservation claims twice (with then without entries)."""
    wa = _register_worker(transitions, clock, "w-claim-a", address="w-claim-a:8080")
    wb = _register_worker(transitions, clock, "w-claim-b", address="w-claim-b:8080")
    job_id = _submit(transitions, clock, "claim-job", reservation_entries=2)
    claims = {
        wa: ReservationClaim(job_id=job_id.to_wire(), entry_idx=0),
        wb: ReservationClaim(job_id=job_id.to_wire(), entry_idx=1),
    }
    apply_event(transitions, ReplaceReservationClaims(claims=claims))
    apply_event(transitions, ReplaceReservationClaims(claims={}))


def scenario_buffer_direct_kill(transitions: ControllerTransitions, clock: FrozenClock) -> None:
    """Buffer a kill request for a direct-provider task."""
    job_id = _submit(transitions, clock, "buffer-direct")
    (task_id,) = _task_ids(transitions, job_id)
    apply_event(transitions, BufferDirectKill(task_id=task_id.to_wire()))


SCENARIOS: dict[str, Callable[[ControllerTransitions, FrozenClock], None]] = {
    "buffer_direct_kill": scenario_buffer_direct_kill,
    "cancel_running_job": scenario_cancel_running_job,
    "coscheduled_failure_retry_bounces_siblings": scenario_coscheduled_failure_retry_bounces_siblings,
    "coscheduled_preempt_retry_bounces_siblings": scenario_coscheduled_preempt_retry_bounces_siblings,
    "coscheduled_timeout": scenario_coscheduled_timeout,
    "direct_provider_cycle": scenario_direct_provider_cycle,
    "endpoint_register_remove": scenario_endpoint_register_remove,
    "preempt_task": scenario_preempt_task,
    "prune_old_data": scenario_prune_old_data,
    "register_assign_run_succeed": scenario_register_assign_run_succeed,
    "replace_reservation_claims": scenario_replace_reservation_claims,
    "submit_simple": scenario_submit_simple,
    "submit_with_reservation": scenario_submit_with_reservation,
    "task_failure_with_retry": scenario_task_failure_with_retry,
    "worker_failure_cascade": scenario_worker_failure_cascade,
}

SCENARIO_NAMES: list[str] = sorted(SCENARIOS.keys())
