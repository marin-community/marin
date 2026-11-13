# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression test for issue #11289: superseded PREEMPTED attempts never finalized.

A preempted worker-bound attempt keeps ``finished_at_ms`` NULL, and so keeps its
capacity charge, until the worker confirms termination. When a retry assigns the
task a newer attempt first, the old attempt must stay in the worker's reconcile
plan; the worker's KILLED or MISSING report for it must stamp only that attempt.
Before the fix the reconcile read joined on the current attempt, so the old one
left the plan, its report was dropped, and the host stayed charged forever: a
full-slice gang could never be admitted onto that slice again.
"""

import pytest
from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.controller import ops, reads
from iris.cluster.controller.backend import plans_from_snapshot
from iris.cluster.controller.codec import constraints_from_json, device_counts_from_json, device_variant_from_json
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind
from iris.cluster.controller.reconcile.worker import WorkerReconcilePlan, WorkerReconcileResult
from iris.cluster.controller.scheduling.scheduler import (
    DEFAULT_MAX_ASSIGNMENTS_PER_WORKER,
    JobRequirements,
    Scheduler,
    SchedulingContext,
    worker_snapshot_from_row,
)
from iris.cluster.types import JobName, UserBudgetDefaults, WorkerId
from iris.rpc import controller_pb2, job_pb2, worker_pb2
from iris.testing.controller import (
    assign_task,
    building_counts,
    check_task_can_be_scheduled,
    dispatch_task,
    healthy_active_workers,
    make_test_entrypoint,
    make_worker_metadata,
    query_attempt,
    query_job,
    query_task,
    query_tasks_for_job,
    register_worker,
    schedulable_tasks,
    submit_job,
)
from iris.testing.controller_state import ControllerTestState
from iris.testing.transitions import commit_reconcile
from rigging.timing import Timestamp

CHIPS_PER_HOST = 4
HOSTS_PER_SLICE = 4


def _register_slice(state: ControllerTestState, name: str) -> list[WorkerId]:
    worker_ids = []
    for index in range(HOSTS_PER_SLICE):
        metadata = make_worker_metadata(cpu=208, memory_bytes=448 * 1024**3, tpu_name="v5p-32")
        metadata.device.tpu.count = CHIPS_PER_HOST
        metadata.attributes[WellKnownAttribute.TPU_NAME].string_value = name
        metadata.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = index
        worker_id = f"{name}-w{index}"
        register_worker(state, worker_id, f"10.0.{len(name)}.{index}", metadata)
        worker_ids.append(WorkerId(worker_id))
    return worker_ids


def _tpu_request(name: str, replicas: int, *, gang: bool) -> controller_pb2.Controller.LaunchJobRequest:
    request = controller_pb2.Controller.LaunchJobRequest(
        name=name,
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=32_000,
            memory_bytes=128 * 1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5p-32", count=CHIPS_PER_HOST)),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=replicas,
        max_retries_preemption=10,
    )
    if gang:
        request.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    return request


def _tpu_used(state: ControllerTestState, worker_id: WorkerId) -> int:
    with state._db.read_snapshot() as snap:
        usage = reads.resource_usage_by_worker(snap).get(worker_id)
    return usage.tpu_count if usage is not None else 0


def _schedule(state: ControllerTestState) -> list[tuple[JobName, WorkerId]]:
    """Run one scheduling decision over the pending tasks and commit its assignments."""
    tasks = [task for task in schedulable_tasks(state) if check_task_can_be_scheduled(task)]
    jobs = {}
    for task in tasks:
        job = query_job(state, task.job_id)
        devices = device_counts_from_json(job.res_device_json)
        jobs[task.job_id] = JobRequirements(
            req_cpu_millicores=job.res_cpu_millicores,
            req_memory_bytes=job.res_memory_bytes,
            req_gpu_count=devices.gpu,
            req_tpu_count=devices.tpu,
            device_variant=device_variant_from_json(job.res_device_json),
            constraints=constraints_from_json(job.constraints_json),
            is_coscheduled=job.has_coscheduling,
            coscheduling_group_by=job.coscheduling_group_by if job.has_coscheduling else None,
        )
    with state._db.read_snapshot() as snap:
        usage = reads.resource_usage_by_worker(snap)
    scheduler = Scheduler()
    context = SchedulingContext(
        workers=[worker_snapshot_from_row(w, usage.get(w.worker_id)) for w in healthy_active_workers(state)],
        building_counts=building_counts(state),
        max_building_tasks=scheduler.max_building_tasks_per_worker,
        max_assignments_per_worker=DEFAULT_MAX_ASSIGNMENTS_PER_WORKER,
        pending_tasks=[task.task_id for task in tasks],
        jobs=jobs,
        pending_task_rows=[],
        user_spend={},
        user_budget_limits={},
        requested_bands={},
        user_budget_defaults=UserBudgetDefaults(),
    )
    assignments = scheduler.find_assignments(context).assignments
    if assignments:
        with state._db.transaction() as cur:
            ops.task.assign(
                cur, [Assignment(task_id=tid, worker_id=wid) for tid, wid in assignments], health=state._health
            )
    return assignments


def _preempt(state: ControllerTestState, task_ids: list[JobName]) -> None:
    decisions = [
        TerminalDecision(kind=TerminalKind.PREEMPT, task_id=tid, reason="preempted by test") for tid in task_ids
    ]
    with state._db.transaction() as cur:
        ops.task.finalize(cur, decisions, now=Timestamp.now())


def _plan_for(state: ControllerTestState, worker_id: WorkerId) -> WorkerReconcilePlan:
    with state._db.read_snapshot() as snap:
        snapshot = reads.load_control_snapshot(snap, state._health, scan_timeouts=False)
    return next(plan for plan in plans_from_snapshot(snapshot) if plan.worker_id == worker_id)


def _desired_by_uid(plan: WorkerReconcilePlan) -> dict[str, worker_pb2.Worker.DesiredAttempt]:
    return {desired.attempt_uid: desired for desired in plan.request.desired}


def _report(state: ControllerTestState, plan: WorkerReconcilePlan, attempt_uid: str, observed: int) -> None:
    """The worker answers its reconcile plan with one observation for ``attempt_uid``."""
    result = WorkerReconcileResult(
        worker_id=plan.worker_id,
        observations=[worker_pb2.Worker.AttemptObservation(attempt_uid=attempt_uid, state=observed)],
    )
    with state._db.transaction() as cur:
        commit_reconcile(cur, [(plan, result)], now=Timestamp.now())


@pytest.mark.parametrize("observed", [job_pb2.TASK_STATE_KILLED, job_pb2.TASK_STATE_MISSING])
def test_superseded_preempted_attempt_is_finalized_by_worker_report(state, observed):
    old_worker, new_worker = _register_slice(state, "slice-a")[:2]
    submit_job(state, "victim", _tpu_request("victim", 1, gang=False))
    task = query_tasks_for_job(state, JobName.root("test-user", "victim"))[0]
    dispatch_task(state, task, old_worker)

    _preempt(state, [task.task_id])
    assert query_task(state, task.task_id).state == job_pb2.TASK_STATE_PENDING
    # Capacity stays reserved until the worker confirms the process is gone.
    assert _tpu_used(state, old_worker) == CHIPS_PER_HOST

    # The retry assigns attempt 1 before attempt 0's termination is observed.
    assign_task(state, task, new_worker)
    assert query_task(state, task.task_id).current_attempt_id == 1
    old_attempt = query_attempt(state, task.task_id, 0)

    # The superseded attempt stays in its worker's plan, as a stop of its own process.
    plan = _plan_for(state, old_worker)
    desired = _desired_by_uid(plan)[old_attempt.attempt_uid]
    assert desired.stop == worker_pb2.Worker.STOP_REASON_PREEMPTED

    _report(state, plan, old_attempt.attempt_uid, observed)

    assert query_attempt(state, task.task_id, 0).finished_at_ms is not None
    assert _tpu_used(state, old_worker) == 0
    assert old_attempt.attempt_uid not in _desired_by_uid(_plan_for(state, old_worker))
    # The newer attempt and the task are untouched.
    current = query_task(state, task.task_id)
    assert current.current_attempt_id == 1
    assert current.state == job_pb2.TASK_STATE_ASSIGNED
    assert current.current_worker_id == new_worker
    assert query_attempt(state, task.task_id, 1).finished_at_ms is None
    assert _tpu_used(state, new_worker) == CHIPS_PER_HOST


def test_superseded_preempted_gang_member_blocks_full_slice_until_confirmed(state):
    slice_a = _register_slice(state, "slice-a")
    slice_b = _register_slice(state, "slice-b")
    gang_a = JobName.root("test-user", "gang-a")
    submit_job(state, "gang-a", _tpu_request("gang-a", HOSTS_PER_SLICE, gang=True))
    for task_id, worker_id in _schedule(state):
        dispatch_task(state, query_task(state, task_id), worker_id)
    first_tasks = query_tasks_for_job(state, gang_a)
    first_slice = {task.current_worker_id for task in first_tasks}
    assert first_slice in (set(slice_a), set(slice_b))
    other_slice = set(slice_b) if first_slice == set(slice_a) else set(slice_a)

    _preempt(state, [task.task_id for task in first_tasks])
    # Every host but one confirms its preempted attempt; the last host's report is still outstanding.
    leaked = first_tasks[-1]
    for task in first_tasks[:-1]:
        plan = _plan_for(state, task.current_worker_id)
        _report(state, plan, query_attempt(state, task.task_id, 0).attempt_uid, job_pb2.TASK_STATE_KILLED)
    assert _tpu_used(state, leaked.current_worker_id) == CHIPS_PER_HOST

    # The gang retries on the other slice: attempt 1 supersedes the unconfirmed attempt 0.
    assert {wid for _tid, wid in _schedule(state)} == other_slice
    assert query_task(state, leaked.task_id).current_attempt_id == 1

    # A second full-slice gang cannot land while one host of the first slice is still charged.
    gang_b = JobName.root("test-user", "gang-b")
    submit_job(state, "gang-b", _tpu_request("gang-b", HOSTS_PER_SLICE, gang=True))
    assert _schedule(state) == []

    plan = _plan_for(state, leaked.current_worker_id)
    leaked_uid = query_attempt(state, leaked.task_id, 0).attempt_uid
    assert leaked_uid in _desired_by_uid(plan)
    _report(state, plan, leaked_uid, job_pb2.TASK_STATE_MISSING)

    assert {wid for _tid, wid in _schedule(state)} == first_slice
    assert {task.current_worker_id for task in query_tasks_for_job(state, gang_b)} == first_slice
    # Gang A's retried attempts keep their placement on the other slice.
    retried = query_tasks_for_job(state, gang_a)
    assert {task.current_worker_id for task in retried} == other_slice
    assert all(task.current_attempt_id == 1 for task in retried)
