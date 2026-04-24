# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the preemption loop — higher-priority tasks evict lower-priority running tasks."""

from iris.cluster.controller.budget import UserBudgetDefaults, compute_effective_band
from iris.cluster.controller.transitions import RESERVATION_HOLDER_JOB_NAME, _resolve_task_failure_state
from iris.cluster.controller.controller import (
    PreemptionCandidate,
    RunningTaskInfo,
    _get_running_tasks_with_band_and_value,
    _run_preemption_pass,
)
from iris.cluster.controller.scheduler import JobRequirements, WorkerCapacity
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2
from iris.rpc import controller_pb2

from iris.cluster.controller.transitions import Assignment, HeartbeatApplyRequest, TaskUpdate

from .conftest import (
    ControllerTestHarness,
    dispatch_task,
    make_controller_state,
    make_test_entrypoint,
    make_worker_metadata,
    query_attempt,
    query_job,
    query_task,
    query_tasks_for_job,
    register_worker,
    submit_job,
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


def _cpu_resources(cpu_cores: int = 1) -> job_pb2.ResourceSpecProto:
    return job_pb2.ResourceSpecProto(cpu_millicores=cpu_cores * 1000, memory_bytes=1024**3)


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
        band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    preemptor_id = JobName.from_wire("/bob/prod-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), job_pb2.PRIORITY_BAND_PRODUCTION),
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
        band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    preemptor_id = JobName.from_wire("/bob/interactive-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), job_pb2.PRIORITY_BAND_INTERACTIVE),
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
        band_sort_key=job_pb2.PRIORITY_BAND_PRODUCTION,
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    preemptor_id = JobName.from_wire("/bob/interactive-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), job_pb2.PRIORITY_BAND_INTERACTIVE),
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
        band_sort_key=job_pb2.PRIORITY_BAND_INTERACTIVE,
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    preemptor_id = JobName.from_wire("/bob/batch-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), job_pb2.PRIORITY_BAND_BATCH),
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
        band_sort_key=job_pb2.PRIORITY_BAND_INTERACTIVE,
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    preemptor_id = JobName.from_wire("/bob/job-b:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), job_pb2.PRIORITY_BAND_INTERACTIVE),
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
        band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
        resource_value=1000,
        is_coscheduled=True,
        resources=_cpu_resources(1),
    )

    preemptor_id = JobName.from_wire("/bob/prod-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), job_pb2.PRIORITY_BAND_PRODUCTION),
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
        assert query_task(state, task.task_id).state == job_pb2.TASK_STATE_RUNNING

        # Preempt
        state.preempt_task(task.task_id, reason="Preempted by /bob/prod-job:0")

        # Task should be PENDING (retry)
        updated = query_task(state, task.task_id)
        assert updated.state == job_pb2.TASK_STATE_PENDING
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
        assert query_task(state, task.task_id).state == job_pb2.TASK_STATE_RUNNING

        state.preempt_task(task.task_id, reason="preempted")

        updated = query_task(state, task.task_id)
        assert updated.state == job_pb2.TASK_STATE_PREEMPTED
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
        band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    preemptor_id = JobName.from_wire("/bob/prod-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), job_pb2.PRIORITY_BAND_PRODUCTION),
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
        band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
        resource_value=5000,
        is_coscheduled=False,
        resources=_cpu_resources(4),
    )
    cheap_victim = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/small-batch:0"),
        worker_id=w1,
        band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    preemptor_id = JobName.from_wire("/bob/prod-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), job_pb2.PRIORITY_BAND_PRODUCTION),
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
    defaults = UserBudgetDefaults()
    effective = compute_effective_band(
        job_pb2.PRIORITY_BAND_INTERACTIVE, "alice", user_spend, user_budget_limits, defaults
    )
    assert effective == job_pb2.PRIORITY_BAND_BATCH

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
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), job_pb2.PRIORITY_BAND_INTERACTIVE),
    ]

    preemptions = _run_preemption_pass(unscheduled, [victim], ctx)
    assert len(preemptions) == 1
    assert preemptions[0] == (preemptor_id, victim.task_id)


def test_over_budget_production_not_preemptible():
    """Over-budget user's PRODUCTION tasks are NOT downgraded and stay non-preemptible by INTERACTIVE."""
    user_spend = {"alice": 10000}
    user_budget_limits = {"alice": 5000}
    defaults = UserBudgetDefaults()
    effective = compute_effective_band(
        job_pb2.PRIORITY_BAND_PRODUCTION, "alice", user_spend, user_budget_limits, defaults
    )
    assert effective == job_pb2.PRIORITY_BAND_PRODUCTION


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
            state._db,
            set(),
            user_spend=user_spend,
            user_budget_limits=user_budget_limits,
            user_budget_defaults=UserBudgetDefaults(),
        )

        assert len(running) == 1
        # Should be downgraded to BATCH
        assert running[0].band_sort_key == job_pb2.PRIORITY_BAND_BATCH


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
        assert query_task(state, task.task_id).state == job_pb2.TASK_STATE_ASSIGNED

        state.preempt_task(task.task_id, reason="preempted while assigned")

        updated = query_task(state, task.task_id)
        assert updated.state == job_pb2.TASK_STATE_PENDING, "ASSIGNED tasks should always retry on preemption"


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
        band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )
    victim2 = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/batch-job-2:0"),
        worker_id=w1,
        band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
        resource_value=2000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    preemptor1 = PreemptionCandidate(
        JobName.from_wire("/bob/prod-1:0"),
        _cpu_requirements(1),
        job_pb2.PRIORITY_BAND_PRODUCTION,
    )
    preemptor2 = PreemptionCandidate(
        JobName.from_wire("/bob/prod-2:0"),
        _cpu_requirements(1),
        job_pb2.PRIORITY_BAND_PRODUCTION,
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
        band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
        resource_value=1000,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )
    victim_w2 = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/batch-w2:0"),
        worker_id=w2,
        band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
        resource_value=500,
        is_coscheduled=False,
        resources=_cpu_resources(1),
    )

    # Preemptor needs 1 CPU — should pick cheapest victim (w2)
    preemptor = PreemptionCandidate(
        JobName.from_wire("/bob/prod:0"),
        _cpu_requirements(1),
        job_pb2.PRIORITY_BAND_PRODUCTION,
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
        harness.transition(task.task_id, job_pb2.TASK_STATE_SUCCEEDED)
        assert query_task(state, task.task_id).state == job_pb2.TASK_STATE_SUCCEEDED

        # Preempt should be no-op
        state.preempt_task(task.task_id, reason="too late")
        assert query_task(state, task.task_id).state == job_pb2.TASK_STATE_SUCCEEDED


# ---------------------------------------------------------------------------
# Unit tests for _resolve_task_failure_state
# ---------------------------------------------------------------------------


def test_resolve_failure_assigned_always_retries():
    """ASSIGNED tasks always retry regardless of preemption budget."""
    new_state, count = _resolve_task_failure_state(
        job_pb2.TASK_STATE_ASSIGNED,
        preemption_count=0,
        max_preemptions=0,
        terminal_state=job_pb2.TASK_STATE_PREEMPTED,
    )
    assert new_state == job_pb2.TASK_STATE_PENDING
    assert count == 0  # preemption_count not incremented for ASSIGNED


def test_resolve_failure_running_retries_within_budget():
    """RUNNING task retries when preemption budget remains."""
    new_state, count = _resolve_task_failure_state(
        job_pb2.TASK_STATE_RUNNING,
        preemption_count=0,
        max_preemptions=3,
        terminal_state=job_pb2.TASK_STATE_PREEMPTED,
    )
    assert new_state == job_pb2.TASK_STATE_PENDING
    assert count == 1


def test_resolve_failure_running_terminal_when_budget_exhausted():
    """RUNNING task goes terminal when preemption budget is exhausted."""
    new_state, count = _resolve_task_failure_state(
        job_pb2.TASK_STATE_RUNNING,
        preemption_count=3,
        max_preemptions=3,
        terminal_state=job_pb2.TASK_STATE_PREEMPTED,
    )
    assert new_state == job_pb2.TASK_STATE_PREEMPTED
    assert count == 4


def test_resolve_failure_building_retries_within_budget():
    """BUILDING task (executing state) retries when budget remains."""
    new_state, count = _resolve_task_failure_state(
        job_pb2.TASK_STATE_BUILDING,
        preemption_count=0,
        max_preemptions=1,
        terminal_state=job_pb2.TASK_STATE_WORKER_FAILED,
    )
    assert new_state == job_pb2.TASK_STATE_PENDING
    assert count == 1


def test_resolve_failure_building_terminal_when_exhausted():
    """BUILDING task goes terminal when preemption budget is exhausted."""
    new_state, count = _resolve_task_failure_state(
        job_pb2.TASK_STATE_BUILDING,
        preemption_count=1,
        max_preemptions=1,
        terminal_state=job_pb2.TASK_STATE_WORKER_FAILED,
    )
    assert new_state == job_pb2.TASK_STATE_WORKER_FAILED
    assert count == 2


# ---------------------------------------------------------------------------
# Integration tests: preempt_task attempt state and coscheduled cascade
# ---------------------------------------------------------------------------


def test_preempt_task_retries_when_budget_remains():
    """Preempted running task retries to PENDING with attempt marked PREEMPTED."""
    with make_controller_state() as state:
        harness = ControllerTestHarness(state)
        w1 = harness.add_worker("w1", cpu=4)

        tasks = harness.submit(
            "/alice/batch-job",
            cpu=1,
            replicas=1,
            max_retries_preemption=3,
        )
        task = tasks[0]
        harness.dispatch(task, w1)
        assert query_task(state, task.task_id).state == job_pb2.TASK_STATE_RUNNING

        attempt_id_before = query_task(state, task.task_id).current_attempt_id
        state.preempt_task(task.task_id, reason="Evicted by /bob/prod:0")

        # Task retries to PENDING
        updated = query_task(state, task.task_id)
        assert updated.state == job_pb2.TASK_STATE_PENDING
        assert updated.preemption_count == 1

        # The attempt is marked PREEMPTED even though the task retries
        attempt = query_attempt(state, task.task_id, attempt_id_before)
        assert attempt is not None
        assert attempt.state == job_pb2.TASK_STATE_PREEMPTED


def test_preempt_task_terminal_when_budget_exhausted():
    """Preempted running task becomes terminal PREEMPTED when budget is spent."""
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

        result = state.preempt_task(task.task_id, reason="budget gone")

        updated = query_task(state, task.task_id)
        assert updated.state == job_pb2.TASK_STATE_PREEMPTED
        assert updated.preemption_count == 1
        assert updated.finished_at is not None

        # The preempted task is included in tasks_to_kill so the controller
        # can send a kill RPC to the worker.
        assert task.task_id in result.tasks_to_kill

        # Attempt is also PREEMPTED
        attempt = query_attempt(state, task.task_id, updated.current_attempt_id)
        assert attempt is not None
        assert attempt.state == job_pb2.TASK_STATE_PREEMPTED


def test_preempt_task_cascades_coscheduled_siblings():
    """When all coscheduled tasks are preempted to terminal, the job finalizes and kills survivors.

    preempt_task does not directly cascade coscheduled siblings (unlike
    WORKER_FAILED via heartbeat). Instead, siblings are killed when the job
    reaches a terminal state through _finalize_terminal_job.
    """
    from iris.cluster.constraints import WellKnownAttribute

    with make_controller_state() as state:
        # Register 2 workers with TPU attributes for coscheduling
        for i in range(2):
            meta = make_worker_metadata()
            meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
            meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
            register_worker(state, f"w{i}", f"addr{i}:8080", meta)

        # Submit a coscheduled job with 2 replicas, no preemption retries
        req = controller_pb2.Controller.LaunchJobRequest(
            name="cosched-preempt",
            entrypoint=make_test_entrypoint(),
            resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            replicas=2,
            environment=job_pb2.EnvironmentConfig(),
            max_retries_preemption=0,
        )
        req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
        tasks = submit_job(state, "cosched-preempt", req)
        assert len(tasks) == 2

        # Dispatch both tasks
        for i, task in enumerate(tasks):
            dispatch_task(state, task, WorkerId(f"w{i}"))

        # Preempt the first task — it goes terminal PREEMPTED
        result0 = state.preempt_task(tasks[0].task_id, reason="preempted by prod")
        assert query_task(state, tasks[0].task_id).state == job_pb2.TASK_STATE_PREEMPTED

        # Second task is still running (preempt_task doesn't directly cascade siblings)
        assert query_task(state, tasks[1].task_id).state == job_pb2.TASK_STATE_RUNNING

        # Preempt the second task — now ALL tasks are terminal, job finalizes
        result1 = state.preempt_task(tasks[1].task_id, reason="preempted by prod")
        assert query_task(state, tasks[1].task_id).state == job_pb2.TASK_STATE_PREEMPTED

        # Both tasks should be in the combined kill set
        all_kills = result0.tasks_to_kill | result1.tasks_to_kill
        assert tasks[0].task_id in all_kills
        assert tasks[1].task_id in all_kills


# ---------------------------------------------------------------------------
# Reservation holder survival during preemption retry
# ---------------------------------------------------------------------------


def test_preemption_retry_preserves_reservation_holder():
    """When a parent with a reservation retries after preemption, the :reservation: child is NOT killed.

    Non-reservation children (e.g. train_lm) must still be killed by the cascade.
    This prevents a deadlock where the killed reservation can never be re-satisfied,
    leaving the parent stuck PENDING forever.
    """

    with make_controller_state() as state:
        harness = ControllerTestHarness(state)
        w1 = harness.add_worker("w1", cpu=4)
        w2 = harness.add_worker("w2", cpu=4)

        # Submit parent job with a reservation (has_reservation=1)
        parent_job_id = JobName.root("test-user", "res-parent")
        parent_req = controller_pb2.Controller.LaunchJobRequest(
            name=parent_job_id.to_wire(),
            entrypoint=make_test_entrypoint(),
            resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            environment=job_pb2.EnvironmentConfig(),
            replicas=1,
            max_retries_preemption=5,
        )
        parent_req.reservation.entries.append(
            job_pb2.ReservationEntry(
                resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            )
        )
        submit_job(state, parent_job_id.to_wire(), parent_req)

        holder_job_id = parent_job_id.child(RESERVATION_HOLDER_JOB_NAME)

        # Verify the reservation holder child was created
        holder_tasks = [t for t in query_tasks_for_job(state, holder_job_id)]
        assert len(holder_tasks) == 1, "reservation holder job should have 1 task"

        # Submit a non-reservation child job under the parent (simulating train_lm)
        child_job_id = parent_job_id.child("train_lm")
        child_req = controller_pb2.Controller.LaunchJobRequest(
            name=child_job_id.to_wire(),
            entrypoint=make_test_entrypoint(),
            resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            environment=job_pb2.EnvironmentConfig(),
            replicas=1,
            max_retries_preemption=0,
        )
        submit_job(state, child_job_id.to_wire(), child_req)
        child_tasks = query_tasks_for_job(state, child_job_id)
        assert len(child_tasks) == 1

        # Dispatch parent task and advance to RUNNING
        parent_tasks = query_tasks_for_job(state, parent_job_id)
        assert len(parent_tasks) == 1
        parent_task = parent_tasks[0]
        dispatch_task(state, parent_task, w1)
        assert query_task(state, parent_task.task_id).state == job_pb2.TASK_STATE_RUNNING

        # Dispatch reservation holder task to w2
        dispatch_task(state, holder_tasks[0], w2)

        # Dispatch child task to w2
        dispatch_task(state, child_tasks[0], w2)

        # Preempt the parent task — it should retry (go PENDING)
        state.preempt_task(parent_task.task_id, reason="Preempted by higher priority")

        # Parent task should be PENDING (retry)
        updated_parent = query_task(state, parent_task.task_id)
        assert updated_parent.state == job_pb2.TASK_STATE_PENDING
        assert updated_parent.preemption_count == 1

        # Reservation holder job should NOT be killed
        holder_job = query_job(state, holder_job_id)
        assert (
            holder_job.state != job_pb2.JOB_STATE_KILLED
        ), "reservation holder job must survive parent preemption retry"

        # Reservation holder task should NOT be killed
        holder_task_updated = query_task(state, holder_tasks[0].task_id)
        assert (
            holder_task_updated.state != job_pb2.TASK_STATE_KILLED
        ), "reservation holder task must survive parent preemption retry"

        # Non-reservation child job SHOULD be killed
        child_job = query_job(state, child_job_id)
        assert (
            child_job.state == job_pb2.JOB_STATE_KILLED
        ), "non-reservation child job must be killed on parent preemption retry"

        # Non-reservation child task SHOULD be killed
        child_task_updated = query_task(state, child_tasks[0].task_id)
        assert (
            child_task_updated.state == job_pb2.TASK_STATE_KILLED
        ), "non-reservation child task must be killed on parent preemption retry"


def test_late_heartbeat_after_preempt_to_pending_does_not_revive_attempt():
    """Regression: after preempt_task retries a task (state -> PENDING, attempt -> PREEMPTED),
    a late worker heartbeat for the dead attempt_id must NOT revive the attempt row back
    to RUNNING while leaving `error` and `finished_at_ms` set.

    Observed in production (job /eczech/iris-run-exp109_bolinas_sweep_eval-...): the
    attempt ended up in the impossible mixed state
        state=RUNNING, error="Preempted by ...", finished_at_ms=<set>
    because preempt_task leaves `tasks.current_attempt_id` pointing at the dead
    attempt, so _apply_task_transitions' stale-attempt guard fails to fire and
    overwrites `state` on the attempt row (COALESCE only protects
    finished_at_ms / error / exit_code).
    """
    with make_controller_state() as state:
        harness = ControllerTestHarness(state)
        worker_id = harness.add_worker("w1", cpu=4)

        tasks = harness.submit(
            "/alice/batch-job",
            cpu=1,
            replicas=1,
            max_retries_preemption=5,
        )
        task = tasks[0]

        harness.dispatch(task, worker_id)
        assert query_task(state, task.task_id).state == job_pb2.TASK_STATE_RUNNING
        dead_attempt_id = query_task(state, task.task_id).current_attempt_id
        assert dead_attempt_id == 0

        state.preempt_task(task.task_id, reason="Preempted by /bob/prod-job:0")

        # Sanity: task went to PENDING (budget remains), attempt row is PREEMPTED-terminal.
        assert query_task(state, task.task_id).state == job_pb2.TASK_STATE_PENDING
        attempt_after_preempt = query_attempt(state, task.task_id, dead_attempt_id)
        assert attempt_after_preempt is not None
        assert attempt_after_preempt.state == job_pb2.TASK_STATE_PREEMPTED
        assert attempt_after_preempt.finished_at is not None
        assert attempt_after_preempt.error == "Preempted by /bob/prod-job:0"

        # Late heartbeat for the (now-dead) attempt 0 arrives: worker still thinks
        # it is RUNNING. This simulates the RPC-in-flight race.
        state.apply_task_updates(
            HeartbeatApplyRequest(
                worker_id=worker_id,
                worker_resource_snapshot=None,
                updates=[
                    TaskUpdate(
                        task_id=task.task_id,
                        attempt_id=dead_attempt_id,
                        new_state=job_pb2.TASK_STATE_RUNNING,
                    )
                ],
            )
        )

        # The attempt row must remain in a consistent terminal state — NOT flipped
        # back to RUNNING with preemption error/finished_at still set.
        attempt_final = query_attempt(state, task.task_id, dead_attempt_id)
        assert attempt_final is not None, "attempt row disappeared"
        assert attempt_final.state == job_pb2.TASK_STATE_PREEMPTED, (
            f"attempt {dead_attempt_id} was revived to state={attempt_final.state} "
            f"(expected PREEMPTED={job_pb2.TASK_STATE_PREEMPTED}); "
            f"error={attempt_final.error!r}, finished_at={attempt_final.finished_at}"
        )
        assert attempt_final.finished_at is not None
        assert attempt_final.error == "Preempted by /bob/prod-job:0"
