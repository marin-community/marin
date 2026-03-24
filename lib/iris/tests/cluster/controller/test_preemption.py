# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the preemption loop — higher-priority tasks evict lower-priority running tasks."""


from iris.cluster.controller.controller import (
    PreemptionCandidate,
    RunningTaskInfo,
    _get_running_tasks_with_band_and_value,
    _run_preemption_pass,
)
from iris.cluster.controller.scheduler import JobRequirements, WorkerCapacity
from iris.cluster.types import JobName, WorkerId
from iris.rpc import cluster_pb2

from .conftest import (
    ControllerTestHarness,
    make_controller_state,
    query_task,
)


def _make_simple_context(workers: list[WorkerCapacity]) -> "FakeSchedulingContext":
    """Create a minimal scheduling context for preemption tests."""
    return FakeSchedulingContext(
        capacities={w.worker_id: w for w in workers},
    )


class FakeSchedulingContext:
    """Minimal stand-in for SchedulingContext used by _run_preemption_pass."""

    def __init__(self, capacities: dict[WorkerId, WorkerCapacity]):
        self.capacities = capacities


def _cpu_resources(cpu_cores: int = 1) -> cluster_pb2.ResourceSpecProto:
    return cluster_pb2.ResourceSpecProto(cpu_millicores=cpu_cores * 1000, memory_bytes=1024**3)


def _cpu_requirements(cpu_cores: int = 1) -> JobRequirements:
    return JobRequirements(
        resources=_cpu_resources(cpu_cores),
        constraints=[],
        is_coscheduled=False,
        coscheduling_group_by=None,
    )


# ---------------------------------------------------------------------------
# Unit tests for _run_preemption_pass
# ---------------------------------------------------------------------------


def test_production_preempts_batch():
    """PRODUCTION task preempts a BATCH task on the same worker."""
    w1 = WorkerId("w1")
    # Worker with 4 CPUs, all committed (0 available)
    cap = WorkerCapacity(
        worker_id=w1,
        available_cpu_millicores=0,
        available_memory=0,
        available_gpus=0,
        available_tpus=0,
    )
    ctx = _make_simple_context([cap])

    victim = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/batch-job:0"),
        worker_id=w1,
        band_sort_key=cluster_pb2.PRIORITY_BAND_BATCH,
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    preemptor_id = JobName.from_wire("/bob/prod-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), cluster_pb2.PRIORITY_BAND_PRODUCTION),
    ]

    preemptions = _run_preemption_pass(unscheduled, [victim], ctx)
    assert len(preemptions) == 1
    assert preemptions[0] == (preemptor_id, victim.task_id)


def test_interactive_preempts_batch():
    """INTERACTIVE task preempts a BATCH task."""
    w1 = WorkerId("w1")
    cap = WorkerCapacity(
        worker_id=w1,
        available_cpu_millicores=0,
        available_memory=0,
        available_gpus=0,
        available_tpus=0,
    )
    ctx = _make_simple_context([cap])

    victim = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/batch-job:0"),
        worker_id=w1,
        band_sort_key=cluster_pb2.PRIORITY_BAND_BATCH,
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    preemptor_id = JobName.from_wire("/bob/interactive-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), cluster_pb2.PRIORITY_BAND_INTERACTIVE),
    ]

    preemptions = _run_preemption_pass(unscheduled, [victim], ctx)
    assert len(preemptions) == 1
    assert preemptions[0] == (preemptor_id, victim.task_id)


def test_interactive_does_not_preempt_production():
    """INTERACTIVE cannot preempt PRODUCTION."""
    w1 = WorkerId("w1")
    cap = WorkerCapacity(
        worker_id=w1,
        available_cpu_millicores=0,
        available_memory=0,
        available_gpus=0,
        available_tpus=0,
    )
    ctx = _make_simple_context([cap])

    victim = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/prod-job:0"),
        worker_id=w1,
        band_sort_key=cluster_pb2.PRIORITY_BAND_PRODUCTION,
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    preemptor_id = JobName.from_wire("/bob/interactive-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), cluster_pb2.PRIORITY_BAND_INTERACTIVE),
    ]

    preemptions = _run_preemption_pass(unscheduled, [victim], ctx)
    assert len(preemptions) == 0


def test_batch_never_preempts():
    """BATCH tasks never trigger preemption even when higher-priority victims exist."""
    w1 = WorkerId("w1")
    cap = WorkerCapacity(
        worker_id=w1,
        available_cpu_millicores=0,
        available_memory=0,
        available_gpus=0,
        available_tpus=0,
    )
    ctx = _make_simple_context([cap])

    # Even with a batch victim, batch preemptor should not preempt
    victim = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/interactive-job:0"),
        worker_id=w1,
        band_sort_key=cluster_pb2.PRIORITY_BAND_INTERACTIVE,
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    preemptor_id = JobName.from_wire("/bob/batch-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), cluster_pb2.PRIORITY_BAND_BATCH),
    ]

    preemptions = _run_preemption_pass(unscheduled, [victim], ctx)
    assert len(preemptions) == 0


def test_same_band_no_preemption():
    """Two tasks in the same band don't preempt each other."""
    w1 = WorkerId("w1")
    cap = WorkerCapacity(
        worker_id=w1,
        available_cpu_millicores=0,
        available_memory=0,
        available_gpus=0,
        available_tpus=0,
    )
    ctx = _make_simple_context([cap])

    victim = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/job-a:0"),
        worker_id=w1,
        band_sort_key=cluster_pb2.PRIORITY_BAND_INTERACTIVE,
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    preemptor_id = JobName.from_wire("/bob/job-b:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), cluster_pb2.PRIORITY_BAND_INTERACTIVE),
    ]

    preemptions = _run_preemption_pass(unscheduled, [victim], ctx)
    assert len(preemptions) == 0


def test_coscheduled_not_preempted():
    """Coscheduled tasks are skipped as victims."""
    w1 = WorkerId("w1")
    cap = WorkerCapacity(
        worker_id=w1,
        available_cpu_millicores=0,
        available_memory=0,
        available_gpus=0,
        available_tpus=0,
    )
    ctx = _make_simple_context([cap])

    victim = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/gang-job:0"),
        worker_id=w1,
        band_sort_key=cluster_pb2.PRIORITY_BAND_BATCH,
        resource_value=1000,
        is_coscheduled=True,
        resources=_cpu_resources(1),
    )

    preemptor_id = JobName.from_wire("/bob/prod-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), cluster_pb2.PRIORITY_BAND_PRODUCTION),
    ]

    preemptions = _run_preemption_pass(unscheduled, [victim], ctx)
    assert len(preemptions) == 0


# ---------------------------------------------------------------------------
# Integration tests using ControllerTransitions
# ---------------------------------------------------------------------------


def test_preempted_task_retries():
    """Preempted task transitions to PENDING (retries) when preemption budget remains."""
    with make_controller_state() as state:
        harness = ControllerTestHarness(state)
        w1 = harness.add_worker("w1", cpu=4)

        # Submit a batch job with preemption retries
        tasks = harness.submit(
            "/alice/batch-job",
            cpu=1,
            replicas=1,
            max_retries_preemption=5,
        )
        task = tasks[0]

        # Dispatch and advance to RUNNING
        harness.dispatch(task, w1)
        assert query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_RUNNING

        # Preempt
        state.preempt_task(task.task_id, reason="Preempted by /bob/prod-job:0")

        # Task should be PENDING (retry)
        updated = query_task(state, task.task_id)
        assert updated.state == cluster_pb2.TASK_STATE_PENDING
        assert updated.preemption_count == 1
        assert updated.error == "Preempted by /bob/prod-job:0"


def test_preempted_task_exhausted_retries():
    """Preempted task transitions to PREEMPTED when preemption budget exhausted."""
    with make_controller_state() as state:
        harness = ControllerTestHarness(state)
        w1 = harness.add_worker("w1", cpu=4)

        tasks = harness.submit(
            "/alice/batch-job",
            cpu=1,
            replicas=1,
            max_retries_preemption=0,
        )
        task = tasks[0]

        harness.dispatch(task, w1)
        assert query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_RUNNING

        state.preempt_task(task.task_id, reason="preempted")

        updated = query_task(state, task.task_id)
        assert updated.state == cluster_pb2.TASK_STATE_PREEMPTED
        assert updated.preemption_count == 1


def test_preemption_skips_if_capacity_available():
    """No preemption when the worker already has capacity for the preemptor."""
    w1 = WorkerId("w1")
    # Worker with plenty of available resources
    cap = WorkerCapacity(
        worker_id=w1,
        available_cpu_millicores=4000,
        available_memory=4 * 1024**3,
        available_gpus=0,
        available_tpus=0,
    )
    ctx = _make_simple_context([cap])

    victim = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/batch-job:0"),
        worker_id=w1,
        band_sort_key=cluster_pb2.PRIORITY_BAND_BATCH,
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    preemptor_id = JobName.from_wire("/bob/prod-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), cluster_pb2.PRIORITY_BAND_PRODUCTION),
    ]

    # Should not preempt since capacity is available
    preemptions = _run_preemption_pass(unscheduled, [victim], ctx)
    assert len(preemptions) == 0


def test_preemption_picks_cheapest_victim():
    """When multiple victims are available, the cheapest one is preempted first."""
    w1 = WorkerId("w1")
    cap = WorkerCapacity(
        worker_id=w1,
        available_cpu_millicores=0,
        available_memory=0,
        available_gpus=0,
        available_tpus=0,
    )
    ctx = _make_simple_context([cap])

    expensive_victim = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/big-batch:0"),
        worker_id=w1,
        band_sort_key=cluster_pb2.PRIORITY_BAND_BATCH,
        resource_value=5000,
        is_coscheduled=False,
        resources=_cpu_resources(4),
    )
    cheap_victim = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/small-batch:0"),
        worker_id=w1,
        band_sort_key=cluster_pb2.PRIORITY_BAND_BATCH,
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    preemptor_id = JobName.from_wire("/bob/prod-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), cluster_pb2.PRIORITY_BAND_PRODUCTION),
    ]

    preemptions = _run_preemption_pass(unscheduled, [expensive_victim, cheap_victim], ctx)
    assert len(preemptions) == 1
    assert preemptions[0][1] == cheap_victim.task_id


def test_get_running_tasks_skips_claimed_workers():
    """_get_running_tasks_with_band_and_value skips tasks on reservation-claimed workers."""
    with make_controller_state() as state:
        harness = ControllerTestHarness(state)
        w1 = harness.add_worker("w1", cpu=4)
        w2 = harness.add_worker("w2", cpu=4)

        tasks1 = harness.submit("/alice/job1", cpu=1)
        tasks2 = harness.submit("/bob/job2", cpu=1)

        harness.dispatch(tasks1[0], w1)
        harness.dispatch(tasks2[0], w2)

        # w1 is claimed by reservation
        claimed = {w1}
        running = _get_running_tasks_with_band_and_value(state._db, claimed)

        # Only tasks on w2 should be returned
        task_ids = {r.task_id for r in running}
        assert tasks2[0].task_id in task_ids
        assert tasks1[0].task_id not in task_ids
