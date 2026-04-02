# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the preemption loop — higher-priority tasks evict lower-priority running tasks."""


from iris.cluster.controller.budget import compute_effective_band
from iris.cluster.controller.transitions import _resolve_task_failure_state
from iris.cluster.controller.controller import (
    PreemptionCandidate,
    RunningTaskInfo,
    _get_running_tasks_with_band_and_value,
    _run_preemption_pass,
)
from iris.cluster.controller.scheduler import JobRequirements, WorkerCapacity
from iris.cluster.types import JobName, WorkerId
from iris.rpc import cluster_pb2

from iris.cluster.controller.transitions import Assignment

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


def test_over_budget_user_tasks_preemptible():
    """Over-budget user's INTERACTIVE running tasks become BATCH victims for preemption."""
    w1 = WorkerId("w1")
    cap = WorkerCapacity(
        worker_id=w1,
        available_cpu_millicores=0,
        available_memory=0,
        available_gpus=0,
        available_tpus=0,
    )
    ctx = _make_simple_context([cap])

    # Alice is over budget — her INTERACTIVE task should have effective band BATCH
    user_spend = {"alice": 10000}
    user_budget_limits = {"alice": 5000}
    effective = compute_effective_band(cluster_pb2.PRIORITY_BAND_INTERACTIVE, "alice", user_spend, user_budget_limits)
    assert effective == cluster_pb2.PRIORITY_BAND_BATCH

    victim = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/interactive-job:0"),
        worker_id=w1,
        band_sort_key=effective,  # BATCH due to budget
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    # Bob's INTERACTIVE task should be able to preempt alice's downgraded task
    preemptor_id = JobName.from_wire("/bob/interactive-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), cluster_pb2.PRIORITY_BAND_INTERACTIVE),
    ]

    preemptions = _run_preemption_pass(unscheduled, [victim], ctx)
    assert len(preemptions) == 1
    assert preemptions[0] == (preemptor_id, victim.task_id)


def test_over_budget_production_not_preemptible():
    """Over-budget user's PRODUCTION tasks are NOT downgraded and stay non-preemptible by INTERACTIVE."""
    user_spend = {"alice": 10000}
    user_budget_limits = {"alice": 5000}
    effective = compute_effective_band(cluster_pb2.PRIORITY_BAND_PRODUCTION, "alice", user_spend, user_budget_limits)
    assert effective == cluster_pb2.PRIORITY_BAND_PRODUCTION


def test_running_tasks_use_effective_band():
    """_get_running_tasks_with_band_and_value applies budget down-weighting to running tasks."""
    with make_controller_state() as state:
        harness = ControllerTestHarness(state)
        w1 = harness.add_worker("w1", cpu=4)

        # Submit an INTERACTIVE job for alice
        tasks = harness.submit("/alice/interactive-job", cpu=1)
        harness.dispatch(tasks[0], w1)

        # Set alice's budget: over budget
        user_spend = {"alice": 10000}
        user_budget_limits = {"alice": 5000}

        running = _get_running_tasks_with_band_and_value(
            state._db, set(), user_spend=user_spend, user_budget_limits=user_budget_limits
        )

        assert len(running) == 1
        # Should be downgraded to BATCH
        assert running[0].band_sort_key == cluster_pb2.PRIORITY_BAND_BATCH


# ---------------------------------------------------------------------------
# Additional preemption edge cases
# ---------------------------------------------------------------------------


def test_preempted_assigned_task_always_retries():
    """ASSIGNED task always retries on preemption regardless of preemption budget."""
    with make_controller_state() as state:
        harness = ControllerTestHarness(state)
        w1 = harness.add_worker("w1", cpu=4)

        # max_retries_preemption=0 — but ASSIGNED tasks always retry
        tasks = harness.submit("/alice/assigned-job", cpu=1, replicas=1, max_retries_preemption=0)
        task = tasks[0]

        # Only assign, don't advance to RUNNING
        state.queue_assignments([Assignment(task_id=task.task_id, worker_id=w1)])
        assert query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_ASSIGNED

        state.preempt_task(task.task_id, reason="preempted while assigned")

        updated = query_task(state, task.task_id)
        assert updated.state == cluster_pb2.TASK_STATE_PENDING, "ASSIGNED tasks should always retry on preemption"


def test_preemption_multiple_victims_one_pass():
    """Multiple preemptors can each preempt different victims in a single pass."""
    w1 = WorkerId("w1")
    cap = WorkerCapacity(
        worker_id=w1,
        available_cpu_millicores=0,
        available_memory=0,
        available_gpus=0,
        available_tpus=0,
    )
    ctx = _make_simple_context([cap])

    victim1 = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/batch-job-1:0"),
        worker_id=w1,
        band_sort_key=cluster_pb2.PRIORITY_BAND_BATCH,
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )
    victim2 = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/batch-job-2:0"),
        worker_id=w1,
        band_sort_key=cluster_pb2.PRIORITY_BAND_BATCH,
        resource_value=2000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    preemptor1 = PreemptionCandidate(
        JobName.from_wire("/bob/prod-1:0"),
        _cpu_requirements(1),
        cluster_pb2.PRIORITY_BAND_PRODUCTION,
    )
    preemptor2 = PreemptionCandidate(
        JobName.from_wire("/bob/prod-2:0"),
        _cpu_requirements(1),
        cluster_pb2.PRIORITY_BAND_PRODUCTION,
    )

    preemptions = _run_preemption_pass([preemptor1, preemptor2], [victim1, victim2], ctx)
    assert len(preemptions) == 2
    victims_preempted = {p[1] for p in preemptions}
    assert victim1.task_id in victims_preempted
    assert victim2.task_id in victims_preempted


def test_preemption_across_multiple_workers():
    """Preemption selects victims from different workers."""
    w1 = WorkerId("w1")
    w2 = WorkerId("w2")
    cap1 = WorkerCapacity(
        worker_id=w1,
        available_cpu_millicores=0,
        available_memory=0,
        available_gpus=0,
        available_tpus=0,
    )
    cap2 = WorkerCapacity(
        worker_id=w2,
        available_cpu_millicores=0,
        available_memory=0,
        available_gpus=0,
        available_tpus=0,
    )
    ctx = _make_simple_context([cap1, cap2])

    victim_w1 = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/batch-w1:0"),
        worker_id=w1,
        band_sort_key=cluster_pb2.PRIORITY_BAND_BATCH,
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )
    victim_w2 = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/batch-w2:0"),
        worker_id=w2,
        band_sort_key=cluster_pb2.PRIORITY_BAND_BATCH,
        resource_value=500,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    # Preemptor needs 1 CPU — should pick cheapest victim (w2)
    preemptor = PreemptionCandidate(
        JobName.from_wire("/bob/prod:0"),
        _cpu_requirements(1),
        cluster_pb2.PRIORITY_BAND_PRODUCTION,
    )

    preemptions = _run_preemption_pass([preemptor], [victim_w1, victim_w2], ctx)
    assert len(preemptions) == 1
    assert preemptions[0][1] == victim_w2.task_id


def test_preemption_nonexistent_task_is_noop():
    """Preempting a non-existent task is a no-op."""
    with make_controller_state() as state:
        result = state.preempt_task(JobName.from_wire("/ghost/job:0"), reason="does not exist")
        assert result.tasks_to_kill == set()


def test_preemption_terminal_task_is_noop():
    """Preempting an already-finished task is a no-op."""
    with make_controller_state() as state:
        harness = ControllerTestHarness(state)
        w1 = harness.add_worker("w1", cpu=4)

        tasks = harness.submit("/alice/done-job", cpu=1, replicas=1)
        task = tasks[0]
        harness.dispatch(task, w1)

        # Succeed the task
        harness.transition(task.task_id, cluster_pb2.TASK_STATE_SUCCEEDED)
        assert query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_SUCCEEDED

        # Preempt should be no-op
        state.preempt_task(task.task_id, reason="too late")
        assert query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_SUCCEEDED


# ---------------------------------------------------------------------------
# Unit tests for _resolve_task_failure_state
# ---------------------------------------------------------------------------


def test_resolve_failure_assigned_always_retries():
    """ASSIGNED tasks always retry regardless of preemption budget."""
    new_state, count = _resolve_task_failure_state(
        cluster_pb2.TASK_STATE_ASSIGNED,
        preemption_count=0,
        max_preemptions=0,
        terminal_state=cluster_pb2.TASK_STATE_PREEMPTED,
    )
    assert new_state == cluster_pb2.TASK_STATE_PENDING
    assert count == 0  # preemption_count not incremented for ASSIGNED


def test_resolve_failure_running_retries_within_budget():
    """RUNNING task retries when preemption budget remains."""
    new_state, count = _resolve_task_failure_state(
        cluster_pb2.TASK_STATE_RUNNING,
        preemption_count=0,
        max_preemptions=3,
        terminal_state=cluster_pb2.TASK_STATE_PREEMPTED,
    )
    assert new_state == cluster_pb2.TASK_STATE_PENDING
    assert count == 1


def test_resolve_failure_running_terminal_when_budget_exhausted():
    """RUNNING task goes terminal when preemption budget is exhausted."""
    new_state, count = _resolve_task_failure_state(
        cluster_pb2.TASK_STATE_RUNNING,
        preemption_count=3,
        max_preemptions=3,
        terminal_state=cluster_pb2.TASK_STATE_PREEMPTED,
    )
    assert new_state == cluster_pb2.TASK_STATE_PREEMPTED
    assert count == 4


def test_resolve_failure_building_retries_within_budget():
    """BUILDING task (executing state) retries when budget remains."""
    new_state, count = _resolve_task_failure_state(
        cluster_pb2.TASK_STATE_BUILDING,
        preemption_count=0,
        max_preemptions=1,
        terminal_state=cluster_pb2.TASK_STATE_WORKER_FAILED,
    )
    assert new_state == cluster_pb2.TASK_STATE_PENDING
    assert count == 1


def test_resolve_failure_building_terminal_when_exhausted():
    """BUILDING task goes terminal when preemption budget is exhausted."""
    new_state, count = _resolve_task_failure_state(
        cluster_pb2.TASK_STATE_BUILDING,
        preemption_count=1,
        max_preemptions=1,
        terminal_state=cluster_pb2.TASK_STATE_WORKER_FAILED,
    )
    assert new_state == cluster_pb2.TASK_STATE_WORKER_FAILED
    assert count == 2
