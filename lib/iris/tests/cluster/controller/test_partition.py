# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for slicing a global per-tick snapshot into per-backend views.

The controller builds one global read snapshot per tick and partitions it by
backend. These tests pin the two invariants that keep the multi-backend path
safe: (1) each backend sees only its own workers/tasks/running work, and (2) a
single backend with constant predicates keeps the whole snapshot unchanged.
"""

from iris.cluster.constraints import AttributeValue
from iris.cluster.controller.partition import partition_control_snapshot, partition_scheduling_context
from iris.cluster.controller.reads import ControlSnapshot
from iris.cluster.controller.reconcile.worker import ReconcileRow
from iris.cluster.controller.scheduling.scheduler import (
    JobRequirements,
    RunningTaskInfo,
    SchedulingContext,
    WorkerSnapshot,
)
from iris.cluster.types import AttemptUid, JobName, PendingTask, UserBudgetDefaults, WorkerId, WorkerStatus
from iris.rpc import job_pb2
from rigging.timing import Timestamp


def _worker(worker_id: str) -> WorkerSnapshot:
    return WorkerSnapshot(
        worker_id=WorkerId(worker_id),
        total_cpu_millicores=64_000,
        total_memory_bytes=1 << 30,
        total_gpu_count=0,
        total_tpu_count=0,
        committed_cpu_millicores=0,
        committed_memory_bytes=0,
        committed_gpu_count=0,
        committed_tpu_count=0,
        attributes={"pool": AttributeValue("default")},
    )


def _pending(job: JobName, backend_id: str) -> PendingTask:
    return PendingTask(
        task_id=job.task(0),
        job_id=job,
        backend_id=backend_id,
        state=job_pb2.TASK_STATE_PENDING,
        current_attempt_id=0,
        failure_count=0,
        preemption_count=0,
        max_retries_failure=0,
        max_retries_preemption=0,
        submitted_at_ms=Timestamp.from_ms(0),
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
        priority_neg_depth=0,
        priority_root_submitted_ms=0,
        priority_insertion=0,
        job_state=job_pb2.JOB_STATE_RUNNING,
        scheduling_deadline_epoch_ms=None,
        scheduling_timeout_ms=None,
        has_coscheduling=False,
        coscheduling_group_by=None,
        constraints_json=None,
        res_cpu_millicores=1000,
        res_memory_bytes=1 << 20,
        res_disk_bytes=0,
        res_device_json=None,
    )


def _job_req() -> JobRequirements:
    return JobRequirements(
        req_cpu_millicores=1000,
        req_memory_bytes=1 << 20,
        req_gpu_count=0,
        req_tpu_count=0,
        device_variant=None,
        constraints=[],
        is_coscheduled=False,
        coscheduling_group_by=None,
    )


def _running(task: JobName, worker_id: str) -> RunningTaskInfo:
    return RunningTaskInfo(
        task_id=task,
        worker_id=WorkerId(worker_id),
        band_sort_key=2,
        resource_value=1,
        is_coscheduled=False,
        cpu_millicores=1000,
        memory_bytes=1 << 20,
        gpu_count=0,
        tpu_count=0,
    )


# Two-backend fixture: worker wa / job ja on backend "a"; worker wb / job jb on "b".
JOB_A = JobName.root("alice", "ja")
JOB_B = JobName.root("bob", "jb")
WORKER_BACKENDS = {WorkerId("wa"): "a", WorkerId("wb"): "b"}
JOB_BACKENDS = {JOB_A: "a", JOB_B: "b"}


def _backend_of_worker(worker_id: WorkerId) -> str:
    return WORKER_BACKENDS[worker_id]


def _backend_of_job(job_id: JobName) -> str:
    return JOB_BACKENDS[job_id]


def _two_backend_context() -> SchedulingContext:
    return SchedulingContext(
        workers=[_worker("wa"), _worker("wb")],
        building_counts={WorkerId("wa"): 1, WorkerId("wb"): 2},
        max_building_tasks=4,
        max_assignments_per_worker=8,
        pending_tasks=[JOB_A.task(0), JOB_B.task(0)],
        jobs={JOB_A: _job_req(), JOB_B: _job_req()},
        pending_task_rows=[_pending(JOB_A, "a"), _pending(JOB_B, "b")],
        user_spend={"alice": 10, "bob": 20},
        user_budget_limits={},
        requested_bands={JOB_A: job_pb2.PRIORITY_BAND_INTERACTIVE, JOB_B: job_pb2.PRIORITY_BAND_BATCH},
        user_budget_defaults=UserBudgetDefaults(),
        running_for_preemption=[_running(JOB_A.task(0), "wa"), _running(JOB_B.task(0), "wb")],
    )


def test_scheduling_context_keeps_only_one_backend_slice():
    ctx = _two_backend_context()

    part = partition_scheduling_context(ctx, "a", _backend_of_worker, _backend_of_job)

    assert [w.worker_id for w in part.workers] == [WorkerId("wa")]
    assert part.building_counts == {WorkerId("wa"): 1}
    assert [t.task_id for t in part.pending_task_rows] == [JOB_A.task(0)]
    assert part.pending_tasks == [JOB_A.task(0)]
    assert set(part.jobs) == {JOB_A}
    assert set(part.requested_bands) == {JOB_A}
    assert [r.worker_id for r in part.running_for_preemption] == [WorkerId("wa")]


def test_running_task_follows_its_worker_not_its_job():
    # A task whose job routes to "a" but is somehow running on backend "b"'s
    # worker belongs to "b"'s partition (a task runs on exactly one worker).
    ctx = _two_backend_context()
    ctx.running_for_preemption.append(_running(JOB_A.task(5), "wb"))

    part_b = partition_scheduling_context(ctx, "b", _backend_of_worker, _backend_of_job)

    assert {r.task_id for r in part_b.running_for_preemption} == {JOB_B.task(0), JOB_A.task(5)}


def test_user_spend_override_threads_through():
    ctx = _two_backend_context()

    part = partition_scheduling_context(ctx, "a", _backend_of_worker, _backend_of_job, user_spend={"alice": 999})

    assert part.user_spend == {"alice": 999}


def test_single_backend_scheduling_partition_is_identity():
    ctx = _two_backend_context()

    # One backend, constant predicates: every worker and every task is kept.
    part = partition_scheduling_context(ctx, "only", lambda _w: "only", lambda _j: "only")

    assert [w.worker_id for w in part.workers] == [w.worker_id for w in ctx.workers]
    assert part.building_counts == ctx.building_counts
    assert [t.task_id for t in part.pending_task_rows] == [t.task_id for t in ctx.pending_task_rows]
    assert part.pending_tasks == ctx.pending_tasks
    assert part.jobs == ctx.jobs
    assert part.requested_bands == ctx.requested_bands
    assert part.user_spend == ctx.user_spend
    assert [r.task_id for r in part.running_for_preemption] == [r.task_id for r in ctx.running_for_preemption]


def _reconcile_row(task: JobName, worker_id: str) -> ReconcileRow:
    return ReconcileRow(
        worker_id=WorkerId(worker_id),
        task_id=task,
        attempt_id=0,
        task_state=job_pb2.TASK_STATE_ASSIGNED,
        attempt_state=job_pb2.TASK_STATE_ASSIGNED,
        job_id=task.parent if task.parent is not None else task,
        attempt_uid=AttemptUid(f"{task}#0"),
    )


def _control_snapshot() -> ControlSnapshot:
    return ControlSnapshot(
        worker_addresses={WorkerId("wa"): "10.0.0.1:9", WorkerId("wb"): "10.0.0.2:9"},
        reconcile_rows=[_reconcile_row(JOB_A.task(0), "wa"), _reconcile_row(JOB_B.task(0), "wb")],
        timeout_rows=[],
        job_specs={JOB_A: job_pb2.RunTaskRequest(), JOB_B: job_pb2.RunTaskRequest()},
        worker_status_map={
            "wa": WorkerStatus(worker_id="wa", running_task_ids=frozenset()),
            "wb": WorkerStatus(worker_id="wb", running_task_ids=frozenset()),
        },
    )


def test_control_snapshot_keeps_only_one_backend_slice():
    control = _control_snapshot()

    part = partition_control_snapshot(control, "a", _backend_of_worker)

    assert set(part.worker_addresses) == {WorkerId("wa")}
    assert [r.worker_id for r in part.reconcile_rows] == [WorkerId("wa")]
    assert set(part.job_specs) == {JOB_A}
    assert set(part.worker_status_map) == {WorkerId("wa")}


def test_single_backend_control_partition_is_identity():
    control = _control_snapshot()

    part = partition_control_snapshot(control, "only", lambda _w: "only")

    assert part.worker_addresses == control.worker_addresses
    assert [r.worker_id for r in part.reconcile_rows] == [r.worker_id for r in control.reconcile_rows]
    assert part.job_specs == control.job_specs
    assert part.worker_status_map == control.worker_status_map
