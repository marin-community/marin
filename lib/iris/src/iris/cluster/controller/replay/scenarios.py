# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curated replay scenarios.

Each scenario is a function ``def scenario_NAME(transitions) -> None`` that
drives ``ControllerTransitions`` through a sequence of mutations. Scenarios
freeze ``Timestamp.now()`` to a deterministic monotonic counter so the DB
state is byte-identical across runs (the goldens are committed to the
repo).
"""

from collections.abc import Callable
from contextlib import contextmanager
from collections.abc import Iterator

from iris.cluster.controller.replay.dispatcher import apply_event
from iris.cluster.controller.replay.events import (
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
)
from iris.cluster.controller.schema import EndpointRow
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    HeartbeatApplyRequest,
    ReservationClaim,
    TaskUpdate,
)
from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2
from rigging import timing
from rigging.timing import Duration, Timestamp

# ---------------------------------------------------------------------------
# Deterministic clock
# ---------------------------------------------------------------------------

# Baseline epoch for all scenarios: 2024-01-01 00:00:00 UTC. Concrete value
# chosen so reasoning about timestamps in the goldens is easier than with
# epoch=0 (which renders 1970 dates that look like bugs).
SCENARIO_EPOCH_MS: int = 1_704_067_200_000


class _FakeClock:
    """Monotonic counter standing in for ``Timestamp.now()`` during a scenario."""

    def __init__(self, start_ms: int) -> None:
        self._t = start_ms

    def now(self) -> Timestamp:
        ts = Timestamp.from_ms(self._t)
        self._t += 1
        return ts

    def at(self) -> Timestamp:
        """Return the current timestamp without advancing."""
        return Timestamp.from_ms(self._t)

    def advance_ms(self, ms: int) -> None:
        self._t += ms


@contextmanager
def _frozen_clock() -> Iterator[_FakeClock]:
    """Patch ``Timestamp.now`` to return monotonic counters for the duration."""
    clock = _FakeClock(SCENARIO_EPOCH_MS)
    saved_now = Timestamp.now
    saved_now_ms = timing._now_ms
    Timestamp.now = classmethod(lambda cls: clock.now())  # type: ignore[method-assign]
    timing._now_ms = lambda: clock.now().epoch_ms()  # type: ignore[assignment]
    try:
        yield clock
    finally:
        Timestamp.now = saved_now  # type: ignore[method-assign]
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
    clock: _FakeClock,
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
    clock: _FakeClock,
    name: str,
    **kw,
) -> JobName:
    job_id, req = _job_request(name, **kw)
    apply_event(transitions, SubmitJob(job_id=job_id, request=req, ts=clock.at()))
    return job_id


def _task_ids(transitions: ControllerTransitions, job_id: JobName) -> list[JobName]:
    """Return the task ids of ``job_id`` in deterministic order."""
    with transitions._db.read_snapshot() as snap:
        rows = snap.fetchall(
            "SELECT task_id FROM tasks WHERE job_id = ? ORDER BY task_index ASC",
            (job_id.to_wire(),),
        )
    return [JobName.from_wire(str(row["task_id"])) for row in rows]


def _current_attempt(transitions: ControllerTransitions, task_id: JobName) -> int:
    with transitions._db.read_snapshot() as snap:
        row = snap.fetchone(
            "SELECT current_attempt_id FROM tasks WHERE task_id = ?",
            (task_id.to_wire(),),
        )
    assert row is not None, f"task missing: {task_id}"
    return int(row["current_attempt_id"])


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenario_submit_simple(transitions: ControllerTransitions) -> None:
    """Submit a single 1-replica job."""
    with _frozen_clock() as clock:
        _submit(transitions, clock, "simple-job")


def scenario_submit_with_reservation(transitions: ControllerTransitions) -> None:
    """Submit a job that carries a reservation entry — exercises reservation holder creation."""
    with _frozen_clock() as clock:
        _submit(transitions, clock, "reservation-job", reservation_entries=1)


def scenario_register_assign_run_succeed(transitions: ControllerTransitions) -> None:
    """Full happy-path lifecycle: register worker, submit, assign, run, succeed."""
    with _frozen_clock() as clock:
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


def scenario_task_failure_with_retry(transitions: ControllerTransitions) -> None:
    """Submit with retry budget, fail once, observe retry to PENDING, then succeed."""
    with _frozen_clock() as clock:
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
        # Retry: re-assign and succeed on the second attempt.
        apply_event(transitions, QueueAssignments([Assignment(task_id=task_id, worker_id=worker_id)]))
        second_attempt = _current_attempt(transitions, task_id)
        apply_event(
            transitions,
            ApplyTaskUpdates(
                HeartbeatApplyRequest(
                    worker_id=worker_id,
                    worker_resource_snapshot=None,
                    updates=[
                        TaskUpdate(task_id=task_id, attempt_id=second_attempt, new_state=job_pb2.TASK_STATE_SUCCEEDED)
                    ],
                )
            ),
        )


def scenario_worker_failure_cascade(transitions: ControllerTransitions) -> None:
    """Register → assign → fail the worker via ``fail_workers`` (multi-tx orchestrator)."""
    with _frozen_clock() as clock:
        worker_id = _register_worker(transitions, clock, "w-doomed")
        job_id = _submit(transitions, clock, "cascade-job")
        (task_id,) = _task_ids(transitions, job_id)
        apply_event(transitions, QueueAssignments([Assignment(task_id=task_id, worker_id=worker_id)]))
        # Direct call: fail_workers is intentionally not an IrisEvent.
        transitions.fail_workers([(worker_id, "w-doomed:8080", "node lost")])


def scenario_cancel_running_job(transitions: ControllerTransitions) -> None:
    """Submit a 2-replica job, assign both tasks, then cancel."""
    with _frozen_clock() as clock:
        worker_id = _register_worker(transitions, clock, "w-cancel", address="w-cancel:8080")
        job_id = _submit(transitions, clock, "cancel-job", replicas=2)
        tasks = _task_ids(transitions, job_id)
        apply_event(
            transitions,
            QueueAssignments([Assignment(task_id=t, worker_id=worker_id) for t in tasks]),
        )
        apply_event(transitions, CancelJob(job_id=job_id, reason="user-cancel"))


def scenario_preempt_task(transitions: ControllerTransitions) -> None:
    """Assign a task and preempt it terminally (no preemption-retry budget)."""
    with _frozen_clock() as clock:
        worker_id = _register_worker(transitions, clock, "w-preempt")
        job_id = _submit(transitions, clock, "preempt-job", max_retries_preemption=0)
        (task_id,) = _task_ids(transitions, job_id)
        apply_event(transitions, QueueAssignments([Assignment(task_id=task_id, worker_id=worker_id)]))
        apply_event(transitions, PreemptTask(task_id=task_id, reason="reclaim"))


def scenario_coscheduled_timeout(transitions: ControllerTransitions) -> None:
    """Coscheduled 2-replica job; timeout one task and observe sibling cascade."""
    with _frozen_clock() as clock:
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


def scenario_direct_provider_cycle(transitions: ControllerTransitions) -> None:
    """Submit a job with no worker, drain to direct-provider, then mark RUNNING."""
    with _frozen_clock() as clock:
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


def scenario_prune_old_data(transitions: ControllerTransitions) -> None:
    """Submit, mark task succeeded, age out, then call prune_old_data directly."""
    with _frozen_clock() as clock:
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
        transitions._db.execute(
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


def scenario_endpoint_register_remove(transitions: ControllerTransitions) -> None:
    """Add an endpoint to a non-terminal task, then remove it."""
    with _frozen_clock() as clock:
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


def scenario_replace_reservation_claims(transitions: ControllerTransitions) -> None:
    """Register two workers, replace reservation claims twice (with then without entries)."""
    with _frozen_clock() as clock:
        wa = _register_worker(transitions, clock, "w-claim-a", address="w-claim-a:8080")
        wb = _register_worker(transitions, clock, "w-claim-b", address="w-claim-b:8080")
        job_id = _submit(transitions, clock, "claim-job", reservation_entries=2)
        claims = {
            wa: ReservationClaim(job_id=job_id.to_wire(), entry_idx=0),
            wb: ReservationClaim(job_id=job_id.to_wire(), entry_idx=1),
        }
        apply_event(transitions, ReplaceReservationClaims(claims=claims))
        apply_event(transitions, ReplaceReservationClaims(claims={}))


def scenario_buffer_direct_kill(transitions: ControllerTransitions) -> None:
    """Buffer a kill request for a direct-provider task."""
    with _frozen_clock() as clock:
        job_id = _submit(transitions, clock, "buffer-direct")
        (task_id,) = _task_ids(transitions, job_id)
        apply_event(transitions, BufferDirectKill(task_id=task_id.to_wire()))


SCENARIOS: dict[str, Callable[[ControllerTransitions], None]] = {
    "submit_simple": scenario_submit_simple,
    "submit_with_reservation": scenario_submit_with_reservation,
    "register_assign_run_succeed": scenario_register_assign_run_succeed,
    "task_failure_with_retry": scenario_task_failure_with_retry,
    "worker_failure_cascade": scenario_worker_failure_cascade,
    "cancel_running_job": scenario_cancel_running_job,
    "preempt_task": scenario_preempt_task,
    "coscheduled_timeout": scenario_coscheduled_timeout,
    "direct_provider_cycle": scenario_direct_provider_cycle,
    "prune_old_data": scenario_prune_old_data,
    "endpoint_register_remove": scenario_endpoint_register_remove,
    "replace_reservation_claims": scenario_replace_reservation_claims,
    "buffer_direct_kill": scenario_buffer_direct_kill,
}
"""Registry of curated replay scenarios."""

SCENARIO_NAMES: list[str] = sorted(SCENARIOS.keys())


def run_scenario(name: str, transitions: ControllerTransitions) -> None:
    """Run the named scenario against ``transitions``."""
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario: {name!r} (available: {SCENARIO_NAMES})")
    SCENARIOS[name](transitions)
