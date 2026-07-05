# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curated replay scenarios.

Each scenario is a function ``def scenario_NAME(transitions, clock) -> None``
that drives ``ControllerTestState`` through a sequence of mutations. The
caller (the pytest fixture or an ad-hoc runner) freezes the clock to a
deterministic monotonic counter so the DB state is byte-identical across
runs and the committed goldens stay stable.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.controller import ops
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.projections.endpoints import EndpointRow
from iris.cluster.controller.pruner import prune_old_data
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.schema import jobs_table, tasks_table
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2
from rigging import timing
from rigging.timing import Duration, Timestamp
from sqlalchemy import select
from sqlalchemy import update as sa_update
from tests.cluster.controller._test_support import ControllerTestState
from tests.cluster.controller.conftest import worker_daemon_backends_for_prune
from tests.cluster.controller.replay.events import (
    AddEndpoint,
    ApplyDirectProviderUpdates,
    ApplyTaskUpdates,
    CancelJob,
    CancelTasksForTimeout,
    DrainForDirectProvider,
    PreemptTask,
    QueueAssignments,
    RegisterOrRefreshWorker,
    RemoveEndpoint,
    SubmitJob,
    apply_event,
)
from tests.cluster.controller.transition_driver import WorkerTaskUpdates


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
    max_task_failures: int = 0,
    coscheduled: bool = False,
) -> tuple[JobName, controller_pb2.Controller.LaunchJobRequest]:
    job_name = JobName.root("test-user", name)
    request = controller_pb2.Controller.LaunchJobRequest(
        name=job_name.to_wire(),
        entrypoint=_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        max_retries_failure=max_retries_failure,
        max_retries_preemption=max_retries_preemption,
        max_task_failures=max_task_failures,
        replicas=replicas,
    )
    if coscheduled:
        request.coscheduling.group_by = "task_index"
    return job_name, request


def _register_worker(
    transitions: ControllerTestState,
    clock: FrozenClock,
    worker_id: str,
    *,
    address: str | None = None,
    metadata: job_pb2.WorkerMetadata | None = None,
) -> WorkerId:
    wid = WorkerId(worker_id)
    apply_event(
        transitions,
        RegisterOrRefreshWorker(
            worker_id=wid,
            address=address or f"{worker_id}:8080",
            metadata=metadata or _make_metadata(),
            ts=clock.at(),
        ),
    )
    return wid


def _submit(
    transitions: ControllerTestState,
    clock: FrozenClock,
    name: str,
    **kw,
) -> JobName:
    job_id, req = _job_request(name, **kw)
    apply_event(transitions, SubmitJob(job_id=job_id, request=req, ts=clock.at()))
    return job_id


def _task_ids(transitions: ControllerTestState, job_id: JobName) -> list[JobName]:
    with transitions._db.read_snapshot() as snap:
        rows = snap.execute(
            select(tasks_table.c.task_id)
            .where(tasks_table.c.job_id == job_id.to_wire())
            .order_by(tasks_table.c.task_index.asc())
        ).all()
    return [JobName.from_wire(str(row.task_id)) for row in rows]


def _current_attempt(transitions: ControllerTestState, task_id: JobName) -> int:
    with transitions._db.read_snapshot() as snap:
        row = snap.execute(
            select(tasks_table.c.current_attempt_id).where(tasks_table.c.task_id == task_id.to_wire())
        ).first()
    assert row is not None, f"task missing: {task_id}"
    return int(row.current_attempt_id)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenario_submit_simple(transitions: ControllerTestState, clock: FrozenClock) -> None:
    """Submit a single 1-replica job."""
    _submit(transitions, clock, "simple-job")


def scenario_register_assign_run_succeed(transitions: ControllerTestState, clock: FrozenClock) -> None:
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
            request=WorkerTaskUpdates(
                worker_id=worker_id,
                updates=[TaskUpdate(task_id=task_id, attempt_id=attempt, new_state=job_pb2.TASK_STATE_RUNNING)],
            ),
        ),
    )
    apply_event(
        transitions,
        ApplyTaskUpdates(
            request=WorkerTaskUpdates(
                worker_id=worker_id,
                updates=[TaskUpdate(task_id=task_id, attempt_id=attempt, new_state=job_pb2.TASK_STATE_SUCCEEDED)],
            ),
        ),
    )


def scenario_task_failure_with_retry(transitions: ControllerTestState, clock: FrozenClock) -> None:
    """Submit with retry budget, fail once, observe retry to PENDING, then succeed.

    ``max_task_failures`` must cover the cumulative failure so the job tolerates
    the one failed attempt instead of failing on it.
    """
    worker_id = _register_worker(transitions, clock, "w-retry")
    job_id = _submit(transitions, clock, "retry-job", max_retries_failure=2, max_task_failures=2)
    (task_id,) = _task_ids(transitions, job_id)
    apply_event(transitions, QueueAssignments([Assignment(task_id=task_id, worker_id=worker_id)]))
    first_attempt = _current_attempt(transitions, task_id)
    apply_event(
        transitions,
        ApplyTaskUpdates(
            WorkerTaskUpdates(
                worker_id=worker_id,
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
            WorkerTaskUpdates(
                worker_id=worker_id,
                updates=[TaskUpdate(task_id=task_id, attempt_id=second_attempt, new_state=job_pb2.TASK_STATE_SUCCEEDED)],
            )
        ),
    )


def scenario_worker_failure_cascade(transitions: ControllerTestState, clock: FrozenClock) -> None:
    """Register → assign → fail the worker via ``ops.worker.fail`` (multi-tx orchestrator)."""
    worker_id = _register_worker(transitions, clock, "w-doomed")
    job_id = _submit(transitions, clock, "cascade-job")
    (task_id,) = _task_ids(transitions, job_id)
    apply_event(transitions, QueueAssignments([Assignment(task_id=task_id, worker_id=worker_id)]))
    # Direct call: fail_workers is intentionally not an IrisEvent.
    ops.worker.fail(
        transitions._db,
        worker_ids=[str(worker_id)],
        reason="node lost",
        health=transitions._health,
    )


def scenario_cancel_running_job(transitions: ControllerTestState, clock: FrozenClock) -> None:
    """Submit a 2-replica job, assign both tasks, then cancel."""
    worker_id = _register_worker(transitions, clock, "w-cancel", address="w-cancel:8080")
    job_id = _submit(transitions, clock, "cancel-job", replicas=2)
    tasks = _task_ids(transitions, job_id)
    apply_event(
        transitions,
        QueueAssignments([Assignment(task_id=t, worker_id=worker_id) for t in tasks]),
    )
    apply_event(transitions, CancelJob(job_id=job_id, reason="user-cancel"))


def scenario_preempt_task(transitions: ControllerTestState, clock: FrozenClock) -> None:
    """Assign a task and preempt it terminally (no preemption-retry budget)."""
    worker_id = _register_worker(transitions, clock, "w-preempt")
    job_id = _submit(transitions, clock, "preempt-job", max_retries_preemption=0)
    (task_id,) = _task_ids(transitions, job_id)
    apply_event(transitions, QueueAssignments([Assignment(task_id=task_id, worker_id=worker_id)]))
    apply_event(transitions, PreemptTask(task_id=task_id, reason="reclaim"))


def scenario_coscheduled_timeout(transitions: ControllerTestState, clock: FrozenClock) -> None:
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


def scenario_coscheduled_failure_retry_bounces_siblings(transitions: ControllerTestState, clock: FrozenClock) -> None:
    """Coscheduled 2-replica job; one task hits a transient failure with retry
    budget remaining. Siblings must bounce to PENDING so the retry re-coschedules
    atomically — otherwise the lone PENDING retry can land on a different slice
    and split the SPMD mesh.

    ``max_task_failures`` covers the cumulative failure so the gang retries the
    round instead of the job failing on the first crash.
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
        max_task_failures=2,
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
            WorkerTaskUpdates(
                worker_id=worker_a,
                updates=[TaskUpdate(task_id=tasks[0], attempt_id=a0, new_state=job_pb2.TASK_STATE_RUNNING)],
            )
        ),
    )
    apply_event(
        transitions,
        ApplyTaskUpdates(
            WorkerTaskUpdates(
                worker_id=worker_b,
                updates=[TaskUpdate(task_id=tasks[1], attempt_id=a1, new_state=job_pb2.TASK_STATE_RUNNING)],
            )
        ),
    )
    apply_event(
        transitions,
        ApplyTaskUpdates(
            WorkerTaskUpdates(
                worker_id=worker_a,
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


def scenario_coscheduled_preempt_retry_bounces_siblings(transitions: ControllerTestState, clock: FrozenClock) -> None:
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


def scenario_coscheduled_preempt_terminal_cascades_siblings(
    transitions: ControllerTestState, clock: FrozenClock
) -> None:
    """Coscheduled 2-replica job RUNNING with no preemption budget; the controller
    preempts one task terminally. The sibling must cascade to COSCHED_FAILED — not
    bounce to PENDING — so no task is left active on a slice whose partner is gone.

    The terminal-preempt complement of ``coscheduled_preempt_retry_bounces_siblings``:
    the retry case requeues siblings, the no-budget case terminates them.
    """
    worker_a = _register_worker(transitions, clock, "w-cosched-pt-a", address="w-cosched-pt-a:8080")
    worker_b = _register_worker(transitions, clock, "w-cosched-pt-b", address="w-cosched-pt-b:8080")
    job_id = _submit(
        transitions,
        clock,
        "cosched-preempt-term",
        replicas=2,
        coscheduled=True,
        max_retries_preemption=0,
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
    attempts = [_current_attempt(transitions, t) for t in tasks]
    _observe(transitions, worker_a, tasks[0], attempts[0], job_pb2.TASK_STATE_RUNNING)
    _observe(transitions, worker_b, tasks[1], attempts[1], job_pb2.TASK_STATE_RUNNING)
    apply_event(transitions, PreemptTask(task_id=tasks[0], reason="evicted-by-prod"))


def scenario_dispatch_cycle(transitions: ControllerTestState, clock: FrozenClock) -> None:
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


def scenario_prune_old_data(transitions: ControllerTestState, clock: FrozenClock) -> None:
    """Submit, mark task succeeded, age out, then call prune_old_data directly."""
    worker_id = _register_worker(transitions, clock, "w-prune")
    job_id = _submit(transitions, clock, "prune-job")
    (task_id,) = _task_ids(transitions, job_id)
    apply_event(transitions, QueueAssignments([Assignment(task_id=task_id, worker_id=worker_id)]))
    attempt = _current_attempt(transitions, task_id)
    apply_event(
        transitions,
        ApplyTaskUpdates(
            WorkerTaskUpdates(
                worker_id=worker_id,
                updates=[TaskUpdate(task_id=task_id, attempt_id=attempt, new_state=job_pb2.TASK_STATE_SUCCEEDED)],
            )
        ),
    )
    # Backdate finished_at so the prune sweep picks it up.
    with transitions._db.transaction() as cur:
        cur.execute(sa_update(jobs_table).where(jobs_table.c.job_id == job_id.to_wire()).values(finished_at_ms=1))
    # Advance the clock so retention math classifies the row as old.
    clock.advance_ms(10_000)
    # Direct call: prune_old_data is intentionally not an IrisEvent.
    prune_old_data(
        transitions._db,
        worker_daemon_backends_for_prune(transitions),
        job_retention=Duration.from_seconds(0),
        worker_retention=Duration.from_seconds(3600),
        slice_retention=Duration.from_seconds(3600),
        pause_between_s=0.0,
    )


def scenario_endpoint_register_remove(transitions: ControllerTestState, clock: FrozenClock) -> None:
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


def _observe(
    transitions: ControllerTestState,
    worker_id: WorkerId,
    task_id: JobName,
    attempt: int,
    state: int,
    *,
    error: str | None = None,
) -> None:
    """Land a single worker observation for one attempt through the reconcile path."""
    apply_event(
        transitions,
        ApplyTaskUpdates(
            WorkerTaskUpdates(
                worker_id=worker_id,
                updates=[TaskUpdate(task_id=task_id, attempt_id=attempt, new_state=state, error=error or "")],
            )
        ),
    )


def scenario_coscheduled_five_tasks_one_fails_all_terminal(transitions: ControllerTestState, clock: FrozenClock) -> None:
    """Coscheduled 5-replica job with no retry budget: one task fails non-retryably
    and every sibling cascades to COSCHED_FAILED so no task is left active."""
    workers = [_register_worker(transitions, clock, f"w-cosched5-{i}", address=f"w-cosched5-{i}:8080") for i in range(5)]
    job_id = _submit(transitions, clock, "cosched5-job", replicas=5, coscheduled=True, max_retries_failure=0)
    tasks = _task_ids(transitions, job_id)
    apply_event(
        transitions,
        QueueAssignments([Assignment(task_id=t, worker_id=w) for t, w in zip(tasks, workers, strict=True)]),
    )
    attempts = [_current_attempt(transitions, t) for t in tasks]
    for task_id, worker_id, attempt in zip(tasks, workers, attempts, strict=True):
        _observe(transitions, worker_id, task_id, attempt, job_pb2.TASK_STATE_RUNNING)
    # Task index 2 fails terminally; siblings must all cascade to COSCHED_FAILED.
    _observe(transitions, workers[2], tasks[2], attempts[2], job_pb2.TASK_STATE_FAILED, error="boom")


def scenario_independent_five_tasks_one_fails(transitions: ControllerTestState, clock: FrozenClock) -> None:
    """Non-coscheduled 5-replica job: one task fails while the others run to success.

    The negative control for the coscheduled cascade — siblings must NOT be
    terminated, so the four survivors reach SUCCEEDED even though the job itself
    fails on the single non-retryable task.
    """
    workers = [_register_worker(transitions, clock, f"w-indep5-{i}", address=f"w-indep5-{i}:8080") for i in range(5)]
    job_id = _submit(transitions, clock, "indep5-job", replicas=5, max_retries_failure=0)
    tasks = _task_ids(transitions, job_id)
    apply_event(
        transitions,
        QueueAssignments([Assignment(task_id=t, worker_id=w) for t, w in zip(tasks, workers, strict=True)]),
    )
    attempts = [_current_attempt(transitions, t) for t in tasks]
    for task_id, worker_id, attempt in zip(tasks, workers, attempts, strict=True):
        _observe(transitions, worker_id, task_id, attempt, job_pb2.TASK_STATE_RUNNING)
    # Survivors complete independently first — proving the failure does not cascade.
    # (Were they still running when task 2 fails, the job-terminal finalize would
    # kill them; succeeding them first isolates the no-peer-cascade contract.)
    for i in (0, 1, 3, 4):
        _observe(transitions, workers[i], tasks[i], attempts[i], job_pb2.TASK_STATE_SUCCEEDED)
    _observe(transitions, workers[2], tasks[2], attempts[2], job_pb2.TASK_STATE_FAILED, error="boom")


SCENARIOS: dict[str, Callable[[ControllerTestState, FrozenClock], None]] = {
    "cancel_running_job": scenario_cancel_running_job,
    "coscheduled_failure_retry_bounces_siblings": scenario_coscheduled_failure_retry_bounces_siblings,
    "coscheduled_preempt_retry_bounces_siblings": scenario_coscheduled_preempt_retry_bounces_siblings,
    "coscheduled_preempt_terminal_cascades_siblings": scenario_coscheduled_preempt_terminal_cascades_siblings,
    "coscheduled_timeout": scenario_coscheduled_timeout,
    "dispatch_cycle": scenario_dispatch_cycle,
    "coscheduled_five_tasks_one_fails_all_terminal": scenario_coscheduled_five_tasks_one_fails_all_terminal,
    "endpoint_register_remove": scenario_endpoint_register_remove,
    "independent_five_tasks_one_fails": scenario_independent_five_tasks_one_fails,
    "preempt_task": scenario_preempt_task,
    "prune_old_data": scenario_prune_old_data,
    "register_assign_run_succeed": scenario_register_assign_run_succeed,
    "submit_simple": scenario_submit_simple,
    "task_failure_with_retry": scenario_task_failure_with_retry,
    "worker_failure_cascade": scenario_worker_failure_cascade,
}

SCENARIO_NAMES: list[str] = sorted(SCENARIOS.keys())
