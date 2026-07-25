# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the preemption loop — higher-priority tasks evict lower-priority running tasks."""

from iris.cluster.constraints import AttributeValue, Constraint, ConstraintIndex, ConstraintOp, WellKnownAttribute
from iris.cluster.controller import ops, reads
from iris.cluster.controller.budget import compute_effective_band
from iris.cluster.controller.ops.task import Assignment, finalize
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind
from iris.cluster.controller.reconcile.task import resolve_task_failure_state as _resolve_task_failure_state
from iris.cluster.controller.scheduling.policy import (
    PreemptionCandidate,
    _sort_pending_tasks_by_resolved_band,
    get_running_tasks_with_band_and_value,
    run_preemption_pass,
)
from iris.cluster.controller.scheduling.scheduler import JobRequirements, RunningTaskInfo, WorkerCapacity
from iris.cluster.types import TERMINAL_JOB_STATES, JobName, UserBudgetDefaults, WorkerId
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Timestamp
from tests.cluster.controller.transition_driver import WorkerTaskUpdates, apply_task_observations

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
    """Minimal stand-in for SchedulingContext used by run_preemption_pass.

    Builds the same constraint index the real context does, so the coscheduled
    partial-host fallback (which groups candidate workers by attribute) behaves
    faithfully: attribute-less workers form no groups, so it no-ops.
    """

    def __init__(self, capacities: dict[WorkerId, WorkerCapacity]):
        self.capacities = capacities
        self._str_to_wid = {str(wid): wid for wid in capacities}
        entity_attrs = {str(wid): dict(cap.attributes) for wid, cap in capacities.items()}
        self.index = ConstraintIndex.build(entity_attrs)
        self._soft_score_cache: dict[tuple[WorkerId, tuple[Constraint, ...]], int] = {}

    def matching_workers(self, constraints: list[Constraint]) -> set[WorkerId]:
        return {self._str_to_wid[s] for s in self.index.matching_entities(constraints)}

    def workers_by_group(self, group_by: str, matching_worker_ids: set[WorkerId]) -> dict[str, list[WorkerId]]:
        matching_strs = {str(wid) for wid in matching_worker_ids}
        str_groups = self.index.entities_by_group(group_by, matching_strs)
        return {key: [self._str_to_wid[s] for s in ids] for key, ids in str_groups.items()}


def _cpu_requirements(cpu_cores: int = 1) -> JobRequirements:
    return JobRequirements(
        req_cpu_millicores=cpu_cores * 1000,
        req_memory_bytes=1024**3,
        req_gpu_count=0,
        req_tpu_count=0,
        device_variant=None,
        constraints=[],
        is_coscheduled=False,
        coscheduling_group_by=None,
    )


def _tpu_requirements(
    variant: str,
    *,
    count: int = 4,
    is_coscheduled: bool = False,
    constraints: list[Constraint] | None = None,
) -> JobRequirements:
    return JobRequirements(
        req_cpu_millicores=1000,
        req_memory_bytes=1024**3,
        req_gpu_count=0,
        req_tpu_count=count,
        device_variant=variant,
        constraints=constraints or [],
        is_coscheduled=is_coscheduled,
        coscheduling_group_by="tpu-name" if is_coscheduled else None,
    )


def _tpu_capacity(worker_id: WorkerId, *, attributes: dict[str, AttributeValue] | None = None) -> WorkerCapacity:
    """Worker fully committed to a TPU task (0 available)."""
    return WorkerCapacity(
        worker_id=worker_id,
        available_cpu_millicores=0,
        available_memory=0,
        available_gpus=0,
        available_tpus=0,
        attributes=attributes or {},
    )


# ---------------------------------------------------------------------------
# Unit tests for run_preemption_pass
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
        cpu_millicores=1000,
        memory_bytes=1024**3,
        gpu_count=0,
        tpu_count=0,
    )

    preemptor_id = JobName.from_wire("/bob/prod-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), job_pb2.PRIORITY_BAND_PRODUCTION),
    ]

    preemptions = run_preemption_pass(unscheduled, [victim], ctx).evictions
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
        cpu_millicores=1000,
        memory_bytes=1024**3,
        gpu_count=0,
        tpu_count=0,
    )

    preemptor_id = JobName.from_wire("/bob/interactive-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), job_pb2.PRIORITY_BAND_INTERACTIVE),
    ]

    preemptions = run_preemption_pass(unscheduled, [victim], ctx).evictions
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
        cpu_millicores=1000,
        memory_bytes=1024**3,
        gpu_count=0,
        tpu_count=0,
    )

    preemptor_id = JobName.from_wire("/bob/interactive-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), job_pb2.PRIORITY_BAND_INTERACTIVE),
    ]

    preemptions = run_preemption_pass(unscheduled, [victim], ctx).evictions
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
        cpu_millicores=1000,
        memory_bytes=1024**3,
        gpu_count=0,
        tpu_count=0,
    )

    preemptor_id = JobName.from_wire("/bob/batch-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), job_pb2.PRIORITY_BAND_BATCH),
    ]

    preemptions = run_preemption_pass(unscheduled, [victim], ctx).evictions
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
        cpu_millicores=1000,
        memory_bytes=1024**3,
        gpu_count=0,
        tpu_count=0,
    )

    preemptor_id = JobName.from_wire("/bob/job-b:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), job_pb2.PRIORITY_BAND_INTERACTIVE),
    ]

    preemptions = run_preemption_pass(unscheduled, [victim], ctx).evictions
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
        cpu_millicores=1000,
        memory_bytes=1024**3,
        gpu_count=0,
        tpu_count=0,
    )

    preemptor_id = JobName.from_wire("/bob/prod-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), job_pb2.PRIORITY_BAND_PRODUCTION),
    ]

    preemptions = run_preemption_pass(unscheduled, [victim], ctx).evictions
    assert len(preemptions) == 0


# ---------------------------------------------------------------------------
# Same-variant gating + slice eviction
# ---------------------------------------------------------------------------


def test_solo_preempts_same_variant_tpu():
    """A solo PRODUCTION TPU task evicts a solo BATCH victim of the same variant."""
    w1 = WorkerId("w1")
    ctx = _make_simple_context([_tpu_capacity(w1)])

    victim = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/batch-job:0"),
        worker_id=w1,
        band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
        resource_value=1000,
        is_coscheduled=False,
        cpu_millicores=1000,
        memory_bytes=1024**3,
        gpu_count=0,
        tpu_count=4,
        device_variant="v5p-8",
    )

    preemptor_id = JobName.from_wire("/bob/prod-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _tpu_requirements("v5p-8"), job_pb2.PRIORITY_BAND_PRODUCTION),
    ]

    preemptions = run_preemption_pass(unscheduled, [victim], ctx).evictions
    assert preemptions == [(preemptor_id, victim.task_id)]


def test_solo_does_not_preempt_different_variant():
    """A v5p-256 preemptor cannot evict a v5p-8 solo victim (variant mismatch)."""
    w1 = WorkerId("w1")
    ctx = _make_simple_context([_tpu_capacity(w1)])

    victim = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/batch-job:0"),
        worker_id=w1,
        band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
        resource_value=1000,
        is_coscheduled=False,
        cpu_millicores=1000,
        memory_bytes=1024**3,
        gpu_count=0,
        tpu_count=4,
        device_variant="v5p-8",
    )

    preemptor_id = JobName.from_wire("/bob/prod-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _tpu_requirements("v5p-256"), job_pb2.PRIORITY_BAND_PRODUCTION),
    ]

    preemptions = run_preemption_pass(unscheduled, [victim], ctx).evictions
    assert preemptions == []


def test_coscheduled_preemptor_evicts_same_variant_slice():
    """A coscheduled PROD job of N tasks evicts an entire coscheduled BATCH slice
    of the same variant; one slice eviction satisfies all N preemptor siblings."""
    workers = [WorkerId(f"w{i}") for i in range(4)]
    ctx = _make_simple_context([_tpu_capacity(w) for w in workers])

    victim_job = JobName.from_wire("/alice/cosched-batch")
    victims = [
        RunningTaskInfo(
            task_id=victim_job.child(str(i)),
            worker_id=workers[i],
            band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
            resource_value=1000,
            is_coscheduled=True,
            cpu_millicores=1000,
            memory_bytes=1024**3,
            gpu_count=0,
            tpu_count=4,
            device_variant="v5p-8",
        )
        for i in range(4)
    ]

    preemptor_job = JobName.from_wire("/bob/cosched-prod")
    req = _tpu_requirements("v5p-8", is_coscheduled=True)
    unscheduled = [
        PreemptionCandidate(preemptor_job.child(str(i)), req, job_pb2.PRIORITY_BAND_PRODUCTION) for i in range(4)
    ]

    preemptions = run_preemption_pass(unscheduled, victims, ctx).evictions
    # Exactly N pairs emitted, one preemptor task per victim sibling.
    assert len(preemptions) == 4
    assert {p[1] for p in preemptions} == {v.task_id for v in victims}
    # All pairs are attributed to a single preemptor sibling — the rest
    # short-circuit via the satisfied_preemptor_jobs guard.
    preemptors_used = {p[0] for p in preemptions}
    assert len(preemptors_used) == 1


def test_coscheduled_preemptor_is_placed_on_the_freed_slice():
    """Each preemptor sibling is bound to one worker of the slice it evicts, so the
    gang is committed onto the freed capacity rather than re-competing next tick."""
    workers = [WorkerId(f"w{i}") for i in range(4)]
    ctx = _make_simple_context([_tpu_capacity(w) for w in workers])

    victim_job = JobName.from_wire("/alice/cosched-batch")
    victims = [
        RunningTaskInfo(
            task_id=victim_job.child(str(i)),
            worker_id=workers[i],
            band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
            resource_value=1000,
            is_coscheduled=True,
            cpu_millicores=1000,
            memory_bytes=1024**3,
            gpu_count=0,
            tpu_count=4,
            device_variant="v5p-8",
        )
        for i in range(4)
    ]

    preemptor_job = JobName.from_wire("/bob/cosched-prod")
    req = _tpu_requirements("v5p-8", is_coscheduled=True)
    unscheduled = [
        PreemptionCandidate(preemptor_job.child(str(i)), req, job_pb2.PRIORITY_BAND_PRODUCTION) for i in range(4)
    ]

    plan = run_preemption_pass(unscheduled, victims, ctx)
    assert len(plan.placements) == 4
    assert {task for task, _ in plan.placements} == {preemptor_job.child(str(i)) for i in range(4)}
    assert {worker for _, worker in plan.placements} == set(workers)


def test_coscheduled_preemptor_does_not_evict_different_variant_slice():
    """v5p-256 coscheduled preemptor cannot tear down a v5p-8 slice."""
    workers = [WorkerId(f"w{i}") for i in range(4)]
    ctx = _make_simple_context([_tpu_capacity(w) for w in workers])

    victim_job = JobName.from_wire("/alice/cosched-batch")
    victims = [
        RunningTaskInfo(
            task_id=victim_job.child(str(i)),
            worker_id=workers[i],
            band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
            resource_value=1000,
            is_coscheduled=True,
            cpu_millicores=1000,
            memory_bytes=1024**3,
            gpu_count=0,
            tpu_count=4,
            device_variant="v5p-8",
        )
        for i in range(4)
    ]

    preemptor_job = JobName.from_wire("/bob/cosched-prod")
    req = _tpu_requirements("v5p-256", is_coscheduled=True)
    unscheduled = [
        PreemptionCandidate(preemptor_job.child(str(i)), req, job_pb2.PRIORITY_BAND_PRODUCTION) for i in range(4)
    ]

    preemptions = run_preemption_pass(unscheduled, victims, ctx).evictions
    assert preemptions == []


def test_coscheduled_preemptor_skips_slice_failing_hard_constraint():
    """A coscheduled preemptor must not evict a same-variant slice whose workers
    fail the preemptor's hard constraints (e.g. wrong region).

    Same device variant is necessary but not sufficient: if the freed slice does
    not satisfy the preemptor's placement constraints, the preemptor can never be
    scheduled onto it, so the eviction is pure waste. Mirrors the solo path's
    matches_constraints gate.
    """
    workers = [WorkerId(f"w{i}") for i in range(4)]
    # Victim slice workers are in us-east1; the preemptor demands us-west1.
    ctx = _make_simple_context([_tpu_capacity(w, attributes={"region": AttributeValue("us-east1")}) for w in workers])

    victim_job = JobName.from_wire("/alice/cosched-batch")
    victims = [
        RunningTaskInfo(
            task_id=victim_job.child(str(i)),
            worker_id=workers[i],
            band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
            resource_value=1000,
            is_coscheduled=True,
            cpu_millicores=1000,
            memory_bytes=1024**3,
            gpu_count=0,
            tpu_count=4,
            device_variant="v5p-8",
        )
        for i in range(4)
    ]

    preemptor_job = JobName.from_wire("/bob/cosched-prod")
    req = _tpu_requirements(
        "v5p-8",
        is_coscheduled=True,
        constraints=[Constraint.create(key="region", op=ConstraintOp.EQ, value="us-west1")],
    )
    unscheduled = [
        PreemptionCandidate(preemptor_job.child(str(i)), req, job_pb2.PRIORITY_BAND_PRODUCTION) for i in range(4)
    ]

    preemptions = run_preemption_pass(unscheduled, victims, ctx).evictions
    assert preemptions == []


def test_coscheduled_preemptor_evicts_slice_satisfying_hard_constraint():
    """Positive control for the constraint gate: when the slice's workers DO
    satisfy the preemptor's hard constraint, the eviction proceeds. A soft
    constraint the workers fail must not block it."""
    workers = [WorkerId(f"w{i}") for i in range(4)]
    ctx = _make_simple_context([_tpu_capacity(w, attributes={"region": AttributeValue("us-west1")}) for w in workers])

    victim_job = JobName.from_wire("/alice/cosched-batch")
    victims = [
        RunningTaskInfo(
            task_id=victim_job.child(str(i)),
            worker_id=workers[i],
            band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
            resource_value=1000,
            is_coscheduled=True,
            cpu_millicores=1000,
            memory_bytes=1024**3,
            gpu_count=0,
            tpu_count=4,
            device_variant="v5p-8",
        )
        for i in range(4)
    ]

    preemptor_job = JobName.from_wire("/bob/cosched-prod")
    req = _tpu_requirements(
        "v5p-8",
        is_coscheduled=True,
        constraints=[
            Constraint.create(key="region", op=ConstraintOp.EQ, value="us-west1"),
            # An unmet *soft* preference must not veto the eviction.
            Constraint.create(
                key="zone", op=ConstraintOp.EQ, value="us-west1-b", mode=job_pb2.CONSTRAINT_MODE_PREFERRED
            ),
        ],
    )
    unscheduled = [
        PreemptionCandidate(preemptor_job.child(str(i)), req, job_pb2.PRIORITY_BAND_PRODUCTION) for i in range(4)
    ]

    preemptions = run_preemption_pass(unscheduled, victims, ctx).evictions
    assert {p[1] for p in preemptions} == {v.task_id for v in victims}


def test_coscheduled_preemptor_skips_undersized_slice():
    """Slice eviction requires len(victim_group) >= preemptor sibling count."""
    workers = [WorkerId(f"w{i}") for i in range(2)]
    ctx = _make_simple_context([_tpu_capacity(w) for w in workers])

    victim_job = JobName.from_wire("/alice/small-batch")
    victims = [
        RunningTaskInfo(
            task_id=victim_job.child(str(i)),
            worker_id=workers[i],
            band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
            resource_value=1000,
            is_coscheduled=True,
            cpu_millicores=1000,
            memory_bytes=1024**3,
            gpu_count=0,
            tpu_count=4,
            device_variant="v5p-8",
        )
        for i in range(2)
    ]

    preemptor_job = JobName.from_wire("/bob/big-prod")
    req = _tpu_requirements("v5p-8", is_coscheduled=True)
    unscheduled = [
        PreemptionCandidate(preemptor_job.child(str(i)), req, job_pb2.PRIORITY_BAND_PRODUCTION)
        for i in range(4)  # needs 4, slice has 2
    ]

    preemptions = run_preemption_pass(unscheduled, victims, ctx).evictions
    assert preemptions == []


def test_solo_preemptor_does_not_tear_down_slice():
    """A non-coscheduled preemptor never evicts a coscheduled slice, even on a variant match."""
    workers = [WorkerId(f"w{i}") for i in range(4)]
    ctx = _make_simple_context([_tpu_capacity(w) for w in workers])

    victim_job = JobName.from_wire("/alice/cosched-batch")
    victims = [
        RunningTaskInfo(
            task_id=victim_job.child(str(i)),
            worker_id=workers[i],
            band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
            resource_value=1000,
            is_coscheduled=True,
            cpu_millicores=1000,
            memory_bytes=1024**3,
            gpu_count=0,
            tpu_count=4,
            device_variant="v5p-8",
        )
        for i in range(4)
    ]

    preemptor_id = JobName.from_wire("/bob/solo-prod:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _tpu_requirements("v5p-8"), job_pb2.PRIORITY_BAND_PRODUCTION),
    ]

    preemptions = run_preemption_pass(unscheduled, victims, ctx).evictions
    assert preemptions == []


# ---------------------------------------------------------------------------
# Coscheduled partial-host fallback: a blocked gang evicts lower-band solo
# co-tenants squatting on the few hosts it needs.
# ---------------------------------------------------------------------------

_GB = 1024**3
_HOST_CPU = 4000  # millicores free on a host with no squatter
_FULL_TPUS = 4  # chips on a whole TPU host
_FREE_RAM = 200 * _GB  # RAM free on a host that fits the gang
_BLOCKED_RAM = 10 * _GB  # RAM free on a host a squatter is hogging
_SQUATTER_RAM = 200 * _GB  # RAM a squatter holds; freeing it unblocks its host
_GANG_RAM = 128 * _GB  # per-host RAM the gang requests


def _pod_capacity(
    worker_id: WorkerId,
    *,
    pod: str = "pod-a",
    cpu_millicores: int = _HOST_CPU,
    memory_bytes: int,
    tpus: int = _FULL_TPUS,
) -> WorkerCapacity:
    """A TPU host in coscheduling group ``pod`` with the given free resources."""
    return WorkerCapacity(
        worker_id=worker_id,
        available_cpu_millicores=cpu_millicores,
        available_memory=memory_bytes,
        available_gpus=0,
        available_tpus=tpus,
        attributes={WellKnownAttribute.TPU_NAME: AttributeValue(pod)},
    )


def _gang_req(
    *,
    variant: str = "v4-2048",
    tpus: int = _FULL_TPUS,
    cpu_millicores: int = 1000,
    memory_bytes: int = _GANG_RAM,
) -> JobRequirements:
    return JobRequirements(
        req_cpu_millicores=cpu_millicores,
        req_memory_bytes=memory_bytes,
        req_gpu_count=0,
        req_tpu_count=tpus,
        device_variant=variant,
        constraints=[],
        is_coscheduled=True,
        coscheduling_group_by=WellKnownAttribute.TPU_NAME,
    )


def _solo_victim(
    task_id: JobName,
    worker_id: WorkerId,
    *,
    band: int,
    cpu_millicores: int = 0,
    memory_bytes: int = 0,
    tpus: int = 0,
    variant: str | None = None,
    resource_value: int = 1000,
) -> RunningTaskInfo:
    return RunningTaskInfo(
        task_id=task_id,
        worker_id=worker_id,
        band_sort_key=band,
        resource_value=resource_value,
        is_coscheduled=False,
        cpu_millicores=cpu_millicores,
        memory_bytes=memory_bytes,
        gpu_count=0,
        tpu_count=tpus,
        device_variant=variant,
    )


def _gang_unscheduled(job: JobName, req: JobRequirements, n: int) -> list[PreemptionCandidate]:
    return [PreemptionCandidate(job.child(str(i)), req, job_pb2.PRIORITY_BAND_PRODUCTION) for i in range(n)]


def test_gang_preempts_cpu_squatter_on_blocking_host():
    """The reserved-pod repro: a PRODUCTION gang needs every host in its pod; N-1
    fit and one is blocked only by a BATCH CPU-only squatter's RAM. The gang
    evicts the squatter (which has no device variant), freeing the host."""
    workers = [WorkerId(f"w{i}") for i in range(4)]
    req = _gang_req()
    # w0-w2 fit; w3 has TPUs free but its RAM is held by the squatter.
    caps = [_pod_capacity(workers[i], memory_bytes=_FREE_RAM) for i in range(3)]
    caps.append(_pod_capacity(workers[3], memory_bytes=_BLOCKED_RAM))
    ctx = _make_simple_context(caps)

    squatter = _solo_victim(
        JobName.from_wire("/michael/ft-prep:0"),
        workers[3],
        band=job_pb2.PRIORITY_BAND_BATCH,
        cpu_millicores=64000,
        memory_bytes=_SQUATTER_RAM,  # freeing it lifts w3 past the gang's RAM ask
    )

    gang = JobName.from_wire("/larry/grug-moe")
    preemptions = run_preemption_pass(_gang_unscheduled(gang, req, 4), [squatter], ctx).evictions

    assert len(preemptions) == 1
    assert preemptions[0][1] == squatter.task_id
    assert preemptions[0][0].parent == gang  # attributed to one gang sibling


def test_gang_partial_host_places_gang_on_freed_and_fitting_hosts():
    """The gang is committed onto every host it will use — the already-fitting
    hosts plus the one freed by evicting the squatter."""
    workers = [WorkerId(f"w{i}") for i in range(4)]
    req = _gang_req()
    caps = [_pod_capacity(workers[i], memory_bytes=_FREE_RAM) for i in range(3)]
    caps.append(_pod_capacity(workers[3], memory_bytes=_BLOCKED_RAM))
    ctx = _make_simple_context(caps)

    squatter = _solo_victim(
        JobName.from_wire("/michael/ft-prep:0"),
        workers[3],
        band=job_pb2.PRIORITY_BAND_BATCH,
        cpu_millicores=64000,
        memory_bytes=_SQUATTER_RAM,
    )

    gang = JobName.from_wire("/larry/grug-moe")
    plan = run_preemption_pass(_gang_unscheduled(gang, req, 4), [squatter], ctx)

    assert [v for _, v in plan.evictions] == [squatter.task_id]
    assert len(plan.placements) == 4
    assert {task for task, _ in plan.placements} == {gang.child(str(i)) for i in range(4)}
    assert {worker for _, worker in plan.placements} == set(workers)


def test_gang_does_not_preempt_same_band_squatter():
    """A squatter at or above the gang's band is never evicted by the fallback."""
    workers = [WorkerId(f"w{i}") for i in range(2)]
    req = _gang_req()
    caps = [
        _pod_capacity(workers[0], memory_bytes=_FREE_RAM),
        _pod_capacity(workers[1], memory_bytes=_BLOCKED_RAM),
    ]
    ctx = _make_simple_context(caps)

    # Same-band (PRODUCTION) squatter — strictly-lower-band gate rejects it.
    squatter = _solo_victim(
        JobName.from_wire("/peer/prod-cpu:0"),
        workers[1],
        band=job_pb2.PRIORITY_BAND_PRODUCTION,
        memory_bytes=_SQUATTER_RAM,
    )

    gang = JobName.from_wire("/larry/grug-moe")
    preemptions = run_preemption_pass(_gang_unscheduled(gang, req, 2), [squatter], ctx).evictions
    assert preemptions == []


def test_gang_partial_host_skips_when_not_enough_recoverable():
    """No eviction when fewer hosts can be freed than the gang needs — freeing a
    strict subset would be wasted (the gang still can't place)."""
    workers = [WorkerId(f"w{i}") for i in range(4)]
    req = _gang_req()
    # w0,w1 fit; w2,w3 both RAM-blocked but only w2 has an evictable victim.
    caps = [
        _pod_capacity(workers[0], memory_bytes=_FREE_RAM),
        _pod_capacity(workers[1], memory_bytes=_FREE_RAM),
        _pod_capacity(workers[2], memory_bytes=_BLOCKED_RAM),
        _pod_capacity(workers[3], memory_bytes=_BLOCKED_RAM),
    ]
    ctx = _make_simple_context(caps)

    only_victim = _solo_victim(
        JobName.from_wire("/m/batch:0"),
        workers[2],
        band=job_pb2.PRIORITY_BAND_BATCH,
        memory_bytes=_SQUATTER_RAM,
    )

    gang = JobName.from_wire("/larry/grug-moe")
    preemptions = run_preemption_pass(_gang_unscheduled(gang, req, 4), [only_victim], ctx).evictions
    assert preemptions == []


def test_gang_partial_host_no_preemption_when_enough_hosts_free():
    """When the group has enough free hosts for the gang already, the squatter on
    a spare host is left alone (the gang places without preemption)."""
    workers = [WorkerId(f"w{i}") for i in range(5)]
    req = _gang_req()
    # 4 free hosts (>= n_required=4) plus one squatted spare; no preemption needed.
    caps = [_pod_capacity(workers[i], memory_bytes=_FREE_RAM) for i in range(4)]
    caps.append(_pod_capacity(workers[4], memory_bytes=_BLOCKED_RAM))
    ctx = _make_simple_context(caps)

    squatter = _solo_victim(
        JobName.from_wire("/m/batch:0"),
        workers[4],
        band=job_pb2.PRIORITY_BAND_BATCH,
        memory_bytes=_SQUATTER_RAM,
    )

    gang = JobName.from_wire("/larry/grug-moe")
    preemptions = run_preemption_pass(_gang_unscheduled(gang, req, 4), [squatter], ctx).evictions
    assert preemptions == []


def test_gang_partial_host_commits_minimal_evictions():
    """With more recoverable hosts than needed, evict only the cheapest `needed`."""
    workers = [WorkerId(f"w{i}") for i in range(5)]
    req = _gang_req()
    # w0,w1 fit; w2,w3,w4 each blocked with one evictable victim. Gang needs 3, so
    # only one host (the cheapest victim) is freed.
    caps = [
        _pod_capacity(workers[0], memory_bytes=_FREE_RAM),
        _pod_capacity(workers[1], memory_bytes=_FREE_RAM),
        _pod_capacity(workers[2], memory_bytes=_BLOCKED_RAM),
        _pod_capacity(workers[3], memory_bytes=_BLOCKED_RAM),
        _pod_capacity(workers[4], memory_bytes=_BLOCKED_RAM),
    ]
    ctx = _make_simple_context(caps)

    victims = [
        _solo_victim(
            JobName.from_wire("/m/batch-a:0"),
            workers[2],
            band=job_pb2.PRIORITY_BAND_BATCH,
            memory_bytes=_SQUATTER_RAM,
            resource_value=9000,
        ),
        _solo_victim(
            JobName.from_wire("/m/batch-b:0"),
            workers[3],
            band=job_pb2.PRIORITY_BAND_BATCH,
            memory_bytes=_SQUATTER_RAM,
            resource_value=1000,  # cheapest
        ),
        _solo_victim(
            JobName.from_wire("/m/batch-c:0"),
            workers[4],
            band=job_pb2.PRIORITY_BAND_BATCH,
            memory_bytes=_SQUATTER_RAM,
            resource_value=5000,
        ),
    ]

    gang = JobName.from_wire("/larry/grug-moe")
    preemptions = run_preemption_pass(_gang_unscheduled(gang, req, 3), victims, ctx).evictions
    assert len(preemptions) == 1
    assert preemptions[0][1] == victims[1].task_id  # the cheapest victim's host


def test_gang_partial_host_ignores_coscheduled_cotenant():
    """The fallback never evicts a *coscheduled* co-tenant — only whole-slice
    eviction (``_preempt_coscheduled``) may touch a gang, and only at slice size."""
    workers = [WorkerId(f"w{i}") for i in range(2)]
    req = _gang_req()
    caps = [
        _pod_capacity(workers[0], memory_bytes=_FREE_RAM),
        _pod_capacity(workers[1], memory_bytes=_BLOCKED_RAM),
    ]
    ctx = _make_simple_context(caps)

    # A coscheduled BATCH co-tenant (single member) holds w1's RAM. It is not a
    # solo victim, and its slice (size 1) is smaller than the gang (size 2).
    cosched_cotenant = RunningTaskInfo(
        task_id=JobName.from_wire("/other/cosched-batch").child("0"),
        worker_id=workers[1],
        band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
        resource_value=1000,
        is_coscheduled=True,
        cpu_millicores=0,
        memory_bytes=_SQUATTER_RAM,
        gpu_count=0,
        tpu_count=0,
        device_variant="v4-2048",
    )

    gang = JobName.from_wire("/larry/grug-moe")
    preemptions = run_preemption_pass(_gang_unscheduled(gang, req, 2), [cosched_cotenant], ctx).evictions
    assert preemptions == []


def test_gang_preempts_solo_tpu_cotenant_on_blocking_host():
    """Intended broader behavior: a BATCH *solo TPU* task occupying chips the gang
    needs is preemptible too (on a host matching the gang's group, any TPU solo
    victim is necessarily the same variant)."""
    workers = [WorkerId(f"w{i}") for i in range(2)]
    req = _gang_req(variant="v5p-8", memory_bytes=_GB)
    # w0 fits; w1 has only 2 of 4 chips free (a solo BATCH task holds the other 2).
    caps = [
        _pod_capacity(workers[0], memory_bytes=_FREE_RAM),
        _pod_capacity(workers[1], memory_bytes=_FREE_RAM, tpus=2),
    ]
    ctx = _make_simple_context(caps)

    tpu_cotenant = _solo_victim(
        JobName.from_wire("/m/batch-tpu:0"),
        workers[1],
        band=job_pb2.PRIORITY_BAND_BATCH,
        tpus=2,  # freeing restores w1 to 4 chips
        variant="v5p-8",
    )

    gang = JobName.from_wire("/larry/grug-moe")
    preemptions = run_preemption_pass(_gang_unscheduled(gang, req, 2), [tpu_cotenant], ctx).evictions
    assert len(preemptions) == 1
    assert preemptions[0][1] == tpu_cotenant.task_id


def test_two_gangs_do_not_double_book_hosts():
    """Two gangs contending for one pod don't double-book hosts: the first claims
    the freed pod; the second finds every host reserved and preempts nothing."""
    workers = [WorkerId(f"w{i}") for i in range(4)]
    req = _gang_req()
    # 3 hosts fit; w3 is RAM-blocked with one evictable BATCH squatter.
    caps = [_pod_capacity(workers[i], memory_bytes=_FREE_RAM) for i in range(3)]
    caps.append(_pod_capacity(workers[3], memory_bytes=_BLOCKED_RAM))
    ctx = _make_simple_context(caps)

    squatter = _solo_victim(
        JobName.from_wire("/m/batch:0"),
        workers[3],
        band=job_pb2.PRIORITY_BAND_BATCH,
        memory_bytes=_SQUATTER_RAM,
    )

    gang_a = JobName.from_wire("/larry/gang-a")
    gang_b = JobName.from_wire("/larry/gang-b")
    unscheduled = _gang_unscheduled(gang_a, req, 4) + _gang_unscheduled(gang_b, req, 4)

    preemptions = run_preemption_pass(unscheduled, [squatter], ctx).evictions
    # Only gang A's single squatter eviction; gang B sees every host reserved.
    assert len(preemptions) == 1
    assert preemptions[0][1] == squatter.task_id
    assert preemptions[0][0].parent == gang_a


# ---------------------------------------------------------------------------
# Integration tests using ControllerTestState
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
        with state._db.transaction() as cur:
            finalize(
                cur,
                [TerminalDecision(TerminalKind.PREEMPT, task.task_id, "Preempted by /bob/prod-job:0")],
                now=Timestamp.now(),
            )

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

        with state._db.transaction() as cur:
            finalize(
                cur,
                [TerminalDecision(TerminalKind.PREEMPT, task.task_id, "preempted")],
                now=Timestamp.now(),
            )

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
        cpu_millicores=1000,
        memory_bytes=1024**3,
        gpu_count=0,
        tpu_count=0,
    )

    preemptor_id = JobName.from_wire("/bob/prod-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), job_pb2.PRIORITY_BAND_PRODUCTION),
    ]

    # Should not preempt since capacity is available
    preemptions = run_preemption_pass(unscheduled, [victim], ctx).evictions
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
        cpu_millicores=4000,
        memory_bytes=1024**3,
        gpu_count=0,
        tpu_count=0,
    )
    cheap_victim = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/small-batch:0"),
        worker_id=w1,
        band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
        resource_value=1000,
        is_coscheduled=False,
        cpu_millicores=1000,
        memory_bytes=1024**3,
        gpu_count=0,
        tpu_count=0,
    )

    preemptor_id = JobName.from_wire("/bob/prod-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), job_pb2.PRIORITY_BAND_PRODUCTION),
    ]

    preemptions = run_preemption_pass(unscheduled, [expensive_victim, cheap_victim], ctx).evictions
    assert len(preemptions) == 1
    assert preemptions[0][1] == cheap_victim.task_id


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
        cpu_millicores=1000,
        memory_bytes=1024**3,
        gpu_count=0,
        tpu_count=0,
    )

    # Bob's INTERACTIVE task should be able to preempt alice's downgraded task
    preemptor_id = JobName.from_wire("/bob/interactive-job:0")
    unscheduled = [
        PreemptionCandidate(preemptor_id, _cpu_requirements(1), job_pb2.PRIORITY_BAND_INTERACTIVE),
    ]

    preemptions = run_preemption_pass(unscheduled, [victim], ctx).evictions
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


def test_running_tasks_report_stamped_band():
    """get_running_tasks_with_band_and_value returns the band stamped at assign time.

    The over-budget downgrade is applied once in ``_commit_assignments`` and
    persisted in ``tasks.priority_band``; the lookup must not re-derive the
    band from current spend (which is what previously caused two same-band
    users at the budget cliff to mutually preempt each other).
    """
    with make_controller_state() as state:
        harness = ControllerTestHarness(state)
        w1 = harness.add_worker("w1", cpu=4)
        w2 = harness.add_worker("w2", cpu=4)

        tasks_alice = harness.submit("/alice/interactive-job", cpu=1)
        tasks_bob = harness.submit("/bob/interactive-job", cpu=1)

        # Alice's task is stamped INTERACTIVE (she was under budget at schedule time).
        # Bob's task is stamped BATCH (he was over budget at schedule time).
        _dispatch_with_band(state, tasks_alice[0], w1, job_pb2.PRIORITY_BAND_INTERACTIVE)
        _dispatch_with_band(state, tasks_bob[0], w2, job_pb2.PRIORITY_BAND_BATCH)

        running = {r.task_id: r.band_sort_key for r in get_running_tasks_with_band_and_value(state._db)}
        assert running == {
            tasks_alice[0].task_id: job_pb2.PRIORITY_BAND_INTERACTIVE,
            tasks_bob[0].task_id: job_pb2.PRIORITY_BAND_BATCH,
        }


def test_demoted_task_re_promotes_after_user_returns_under_budget():
    """Stamping the effective band on assign must not pin a task to BATCH for life.

    Sequence: alice submits INTERACTIVE → over-budget at assign time, scheduler
    stamps ``tasks.priority_band = BATCH`` → task is preempted back to PENDING
    → alice now under budget. The next scheduling tick must source the
    requested band from ``job_config`` (immutable since submission), not from
    the stamped ``tasks.priority_band`` (BATCH). Otherwise
    ``compute_effective_band`` — which only demotes — can never restore
    INTERACTIVE, and a momentary over-budget blip becomes a permanent
    downgrade that did not exist before this PR's stamping change.
    """
    with make_controller_state() as state:
        harness = ControllerTestHarness(state)
        w1 = harness.add_worker("w1", cpu=4)

        tasks = harness.submit("/alice/interactive-job", cpu=1, priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE)
        task = tasks[0]
        job_id = task.job_id

        # Scheduler decides alice is over budget and stamps BATCH at assign
        # time. We bypass the over-budget computation by passing the band
        # directly; ``assign_task`` is the same code path the scheduler hits.
        _dispatch_with_band(state, task, w1, job_pb2.PRIORITY_BAND_BATCH)

        # Confirm tasks.priority_band has been overwritten to BATCH.
        assert query_task(state, task.task_id).priority_band == job_pb2.PRIORITY_BAND_BATCH

        # job_config still reflects what alice asked for.
        with state._db.read_snapshot() as snap:
            requested = reads.get_priority_bands(snap, [job_id])
        assert requested == {job_id: job_pb2.PRIORITY_BAND_INTERACTIVE}

        # And under-budget alice gets her INTERACTIVE band back from
        # compute_effective_band — the scheduler now feeds it the job_config
        # value so the next assignment will re-stamp INTERACTIVE.
        assert (
            compute_effective_band(
                requested[job_id],
                "alice",
                user_spend={"alice": 0},
                user_budgets={"alice": 5000},
                defaults=UserBudgetDefaults(),
            )
            == job_pb2.PRIORITY_BAND_INTERACTIVE
        )


def test_pending_child_order_uses_parent_job_config_not_stamped_task_band():
    """Pending order resolves parent bands from job_config, not stamped task rows."""
    with make_controller_state() as state:
        harness = ControllerTestHarness(state)
        w1 = harness.add_worker("w1", cpu=4)

        parent_tasks = harness.submit(
            "/alice/parent-prod",
            cpu=1,
            priority_band=job_pb2.PRIORITY_BAND_PRODUCTION,
        )
        parent_task = parent_tasks[0]
        parent_id = parent_task.job_id

        # Simulate an assignment-time effective-band stamp that differs from
        # the parent's immutable requested band. Child inheritance and pending
        # ordering must not read this stamped value.
        _dispatch_with_band(state, parent_task, w1, job_pb2.PRIORITY_BAND_BATCH)
        assert query_task(state, parent_task.task_id).priority_band == job_pb2.PRIORITY_BAND_BATCH

        child_id = parent_id.child("child")
        child_req = controller_pb2.Controller.LaunchJobRequest(
            name=child_id.to_wire(),
            entrypoint=make_test_entrypoint(),
            resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            environment=job_pb2.EnvironmentConfig(),
            replicas=1,
        )
        with state._db.transaction() as cur:
            ops.job.submit(cur, job_id=child_id, request=child_req, ts=Timestamp.now())
        interactive_tasks = harness.submit(
            "/bob/interactive",
            cpu=1,
            priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
        )

        child_task = query_tasks_for_job(state, child_id)[0]
        assert child_task.priority_band == job_pb2.PRIORITY_BAND_INTERACTIVE

        with state._db.read_snapshot() as tx:
            pending = reads.pending_tasks_with_jobs(tx)
            bands = reads.get_priority_bands(tx, {t.job_id for t in pending})
        ordered = _sort_pending_tasks_by_resolved_band(pending, bands)
        ordered_ids = [task.task_id for task in ordered]

        assert ordered_ids.index(child_task.task_id) < ordered_ids.index(interactive_tasks[0].task_id)


def _dispatch_with_band(state, task, worker_id, priority_band: int) -> None:
    """Dispatch task with an explicit stamped band, advancing it to RUNNING."""
    with state._db.transaction() as cur:
        ops.task.assign(
            cur,
            [Assignment(task_id=task.task_id, worker_id=worker_id, priority_band=priority_band)],
            health=state._health,
        )
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=worker_id,
                    updates=[
                        TaskUpdate(
                            task_id=task.task_id,
                            attempt_id=query_task(state, task.task_id).current_attempt_id,
                            new_state=job_pb2.TASK_STATE_RUNNING,
                        )
                    ],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )


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
        with state._db.transaction() as cur:
            ops.task.assign(cur, [Assignment(task_id=task.task_id, worker_id=w1)], health=state._health)
        assert query_task(state, task.task_id).state == job_pb2.TASK_STATE_ASSIGNED

        with state._db.transaction() as cur:
            finalize(
                cur,
                [TerminalDecision(TerminalKind.PREEMPT, task.task_id, "preempted while assigned")],
                now=Timestamp.now(),
            )

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
        cpu_millicores=1000,
        memory_bytes=1024**3,
        gpu_count=0,
        tpu_count=0,
    )
    victim2 = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/batch-job-2:0"),
        worker_id=w1,
        band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
        resource_value=2000,
        is_coscheduled=False,
        cpu_millicores=1000,
        memory_bytes=1024**3,
        gpu_count=0,
        tpu_count=0,
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

    preemptions = run_preemption_pass([preemptor1, preemptor2], [victim1, victim2], ctx).evictions
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
        cpu_millicores=1000,
        memory_bytes=1024**3,
        gpu_count=0,
        tpu_count=0,
    )
    victim_w2 = RunningTaskInfo(
        task_id=JobName.from_wire("/alice/batch-w2:0"),
        worker_id=w2,
        band_sort_key=job_pb2.PRIORITY_BAND_BATCH,
        resource_value=500,
        is_coscheduled=False,
        cpu_millicores=1000,
        memory_bytes=1024**3,
        gpu_count=0,
        tpu_count=0,
    )

    # Preemptor needs 1 CPU — should pick cheapest victim (w2)
    preemptor = PreemptionCandidate(
        JobName.from_wire("/bob/prod:0"),
        _cpu_requirements(1),
        job_pb2.PRIORITY_BAND_PRODUCTION,
    )

    preemptions = run_preemption_pass([preemptor], [victim_w1, victim_w2], ctx).evictions
    assert len(preemptions) == 1
    assert preemptions[0][1] == victim_w2.task_id


def test_solo_preemptor_is_placed_on_the_freed_worker():
    """The solo preemptor is bound to the worker of the victim it evicts."""
    w1, w2 = WorkerId("w1"), WorkerId("w2")
    ctx = _make_simple_context(
        [
            WorkerCapacity(
                worker_id=w1, available_cpu_millicores=0, available_memory=0, available_gpus=0, available_tpus=0
            ),
            WorkerCapacity(
                worker_id=w2, available_cpu_millicores=0, available_memory=0, available_gpus=0, available_tpus=0
            ),
        ]
    )
    # w2's victim is cheaper, so it is the one evicted; the preemptor lands there.
    victim_w1 = _solo_victim(
        JobName.from_wire("/alice/batch-w1:0"),
        w1,
        band=job_pb2.PRIORITY_BAND_BATCH,
        cpu_millicores=1000,
        memory_bytes=1024**3,
        resource_value=1000,
    )
    victim_w2 = _solo_victim(
        JobName.from_wire("/alice/batch-w2:0"),
        w2,
        band=job_pb2.PRIORITY_BAND_BATCH,
        cpu_millicores=1000,
        memory_bytes=1024**3,
        resource_value=500,
    )

    preemptor = JobName.from_wire("/bob/prod:0")
    plan = run_preemption_pass(
        [PreemptionCandidate(preemptor, _cpu_requirements(1), job_pb2.PRIORITY_BAND_PRODUCTION)],
        [victim_w1, victim_w2],
        ctx,
    )
    assert plan.evictions == [(preemptor, victim_w2.task_id)]
    assert plan.placements == [(preemptor, w2)]


def test_preemption_nonexistent_task_is_noop():
    """Preempting a non-existent task is a no-op."""
    with make_controller_state() as state:
        with state._db.transaction() as cur:
            result = finalize(
                cur,
                [TerminalDecision(TerminalKind.PREEMPT, JobName.from_wire("/ghost/job:0"), "does not exist")],
                now=Timestamp.now(),
            )
        assert not result.tasks
        assert not result.attempts
        assert not result.jobs


def test_preempt_then_worker_terminal_heartbeat_stamps_finished_at_ms():
    """Regression for #5918: worker's post-preempt terminal heartbeat must finalize the attempt.

    ``preempt_task`` marks the attempt PREEMPTED via
    ``task.merge_task_termination(stamp_attempt_finished=False)``, which
    deliberately leaves ``finished_at_ms`` NULL and relies on the worker's
    subsequent terminal-state heartbeat to stamp it. Without the stamp the row
    stays counted by ``resource_usage_by_worker`` and ghost-pins the worker's
    capacity for as long as the worker lives.
    """
    with make_controller_state() as state:
        harness = ControllerTestHarness(state)
        w1 = harness.add_worker("w1", cpu=4)

        tasks = harness.submit("/alice/job", cpu=1, replicas=1, max_retries_preemption=5)
        task = tasks[0]
        harness.dispatch(task, w1)
        attempt_id = query_task(state, task.task_id).current_attempt_id

        # Producer transition: attempt PREEMPTED, finished_at_ms left NULL on purpose.
        with state._db.transaction() as cur:
            finalize(
                cur,
                [TerminalDecision(TerminalKind.PREEMPT, task.task_id, "preempted by /bob/prod-job:0")],
                now=Timestamp.now(),
            )
        attempt = query_attempt(state, task.task_id, attempt_id)
        assert attempt.state == job_pb2.TASK_STATE_PREEMPTED
        assert attempt.finished_at_ms is None, "producer transition should leave finalization for heartbeat"

        # Worker's heartbeat for the now-terminal attempt — the deferred finalization.
        with state._db.transaction() as cur:
            apply_task_observations(
                cur,
                [
                    WorkerTaskUpdates(
                        worker_id=w1,
                        updates=[
                            TaskUpdate(
                                task_id=task.task_id,
                                attempt_id=attempt_id,
                                new_state=job_pb2.TASK_STATE_KILLED,
                            )
                        ],
                    )
                ],
                health=state._health,
                now=Timestamp.now(),
            )

        attempt = query_attempt(state, task.task_id, attempt_id)
        assert attempt.finished_at_ms is not None, (
            "worker's terminal-state heartbeat must stamp finished_at_ms on the preempted "
            "attempt; otherwise the row stays in resource_usage_by_worker and ghost-pins capacity"
        )


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
        with state._db.transaction() as cur:
            finalize(
                cur,
                [TerminalDecision(TerminalKind.PREEMPT, task.task_id, "too late")],
                now=Timestamp.now(),
            )
        assert query_task(state, task.task_id).state == job_pb2.TASK_STATE_SUCCEEDED


# ---------------------------------------------------------------------------
# Unit tests for _resolve_task_failure_state
# ---------------------------------------------------------------------------


def test_resolve_failure_assigned_always_retries():
    """ASSIGNED tasks always retry regardless of preemption budget."""
    new_state = _resolve_task_failure_state(
        job_pb2.TASK_STATE_ASSIGNED,
        preemption_count=0,
        max_preemptions=0,
        terminal_state=job_pb2.TASK_STATE_PREEMPTED,
    )
    assert new_state == job_pb2.TASK_STATE_PENDING


def test_resolve_failure_running_retries_within_budget():
    """RUNNING task retries when preemption budget remains."""
    new_state = _resolve_task_failure_state(
        job_pb2.TASK_STATE_RUNNING,
        preemption_count=0,
        max_preemptions=3,
        terminal_state=job_pb2.TASK_STATE_PREEMPTED,
    )
    assert new_state == job_pb2.TASK_STATE_PENDING


def test_resolve_failure_running_terminal_when_budget_exhausted():
    """RUNNING task goes terminal when preemption budget is exhausted."""
    new_state = _resolve_task_failure_state(
        job_pb2.TASK_STATE_RUNNING,
        preemption_count=3,
        max_preemptions=3,
        terminal_state=job_pb2.TASK_STATE_PREEMPTED,
    )
    assert new_state == job_pb2.TASK_STATE_PREEMPTED


def test_resolve_failure_building_retries_within_budget():
    """BUILDING task (executing state) retries when budget remains."""
    new_state = _resolve_task_failure_state(
        job_pb2.TASK_STATE_BUILDING,
        preemption_count=0,
        max_preemptions=1,
        terminal_state=job_pb2.TASK_STATE_WORKER_FAILED,
    )
    assert new_state == job_pb2.TASK_STATE_PENDING


def test_resolve_failure_building_terminal_when_exhausted():
    """BUILDING task goes terminal when preemption budget is exhausted."""
    new_state = _resolve_task_failure_state(
        job_pb2.TASK_STATE_BUILDING,
        preemption_count=1,
        max_preemptions=1,
        terminal_state=job_pb2.TASK_STATE_WORKER_FAILED,
    )
    assert new_state == job_pb2.TASK_STATE_WORKER_FAILED


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
        with state._db.transaction() as cur:
            result = finalize(
                cur,
                [TerminalDecision(TerminalKind.PREEMPT, task.task_id, "Evicted by /bob/prod:0")],
                now=Timestamp.now(),
            )

        # Task retries to PENDING
        updated = query_task(state, task.task_id)
        assert updated.state == job_pb2.TASK_STATE_PENDING
        assert updated.preemption_count == 1

        # The attempt is marked PREEMPTED even though the task retries
        attempt = query_attempt(state, task.task_id, attempt_id_before)
        assert attempt is not None
        assert attempt.state == job_pb2.TASK_STATE_PREEMPTED

        # The task_preempted log event confirms the preemption was recorded.
        preempted_ids = {ev.entity_id for ev in result.log_events if ev.action == "task_preempted"}
        assert task.task_id.to_wire() in preempted_ids


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

        with state._db.transaction() as cur:
            result = finalize(
                cur,
                [TerminalDecision(TerminalKind.PREEMPT, task.task_id, "budget gone")],
                now=Timestamp.now(),
            )

        updated = query_task(state, task.task_id)
        assert updated.state == job_pb2.TASK_STATE_PREEMPTED
        assert updated.preemption_count == 1
        assert updated.finished_at_ms is not None

        # The task_preempted log event confirms the preemption was recorded.
        preempted_ids = {ev.entity_id for ev in result.log_events if ev.action == "task_preempted"}
        assert task.task_id.to_wire() in preempted_ids

        # Attempt is also PREEMPTED
        attempt = query_attempt(state, task.task_id, updated.current_attempt_id)
        assert attempt is not None
        assert attempt.state == job_pb2.TASK_STATE_PREEMPTED


def test_preempt_task_requeues_coscheduled_siblings_on_retry():
    """When a coscheduled task is preempted but retries (PENDING), siblings are
    bounced to PENDING so the job re-coschedules atomically. Without this, the
    retry could land on a different slice from the still-RUNNING siblings,
    splitting the SPMD mesh."""
    with make_controller_state() as state:
        for i in range(2):
            meta = make_worker_metadata()
            meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
            meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
            register_worker(state, f"w{i}", f"addr{i}:8080", meta)

        req = controller_pb2.Controller.LaunchJobRequest(
            name="cosched-preempt-retry",
            entrypoint=make_test_entrypoint(),
            resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            replicas=2,
            environment=job_pb2.EnvironmentConfig(),
            max_retries_preemption=3,
        )
        req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
        tasks = submit_job(state, "cosched-preempt-retry", req)
        assert len(tasks) == 2

        for i, task in enumerate(tasks):
            dispatch_task(state, task, WorkerId(f"w{i}"))

        with state._db.transaction() as cur:
            result = finalize(
                cur,
                [TerminalDecision(TerminalKind.PREEMPT, tasks[0].task_id, "evicted")],
                now=Timestamp.now(),
            )

        # Preempted task retries to PENDING with attempt PREEMPTED.
        preempted = query_task(state, tasks[0].task_id)
        assert preempted.state == job_pb2.TASK_STATE_PENDING
        assert preempted.preemption_count == 1

        # Sibling bounced to PENDING so the job re-coschedules atomically;
        # its preemption budget is preserved (only the original victim pays).
        sibling = query_task(state, tasks[1].task_id)
        assert sibling.state == job_pb2.TASK_STATE_PENDING
        assert sibling.preemption_count == 0

        # The preempted trigger task is recorded via a task_preempted log event.
        preempted_ids = {ev.entity_id for ev in result.log_events if ev.action == "task_preempted"}
        assert tasks[0].task_id.to_wire() in preempted_ids
        # The sibling was bounced to PENDING (requeue, not terminal preempt).
        assert tasks[1].task_id in result.tasks
        assert result.tasks[tasks[1].task_id].state == job_pb2.TASK_STATE_PENDING


def test_preempt_task_cascades_coscheduled_siblings():
    """Terminally preempting one coscheduled task cascades its siblings to
    COSCHED_FAILED in the same batch, tearing the gang down atomically.

    A terminal preemption is a FAILURE-class transition, so the shared peer
    cascade terminates siblings (it only requeues them when the trigger task
    rolls back to PENDING with retry budget). The job then finalizes terminal
    with no task left active on a half-gone slice.
    """
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

        # Preempt the first task terminally (no retry budget).
        with state._db.transaction() as cur:
            result0 = finalize(
                cur,
                [TerminalDecision(TerminalKind.PREEMPT, tasks[0].task_id, "preempted by prod")],
                now=Timestamp.now(),
            )

        # Direct victim is PREEMPTED; the coscheduled sibling cascades to
        # COSCHED_FAILED in the same batch — no second preempt needed.
        assert query_task(state, tasks[0].task_id).state == job_pb2.TASK_STATE_PREEMPTED
        assert query_task(state, tasks[1].task_id).state == job_pb2.TASK_STATE_COSCHED_FAILED

        # The job finalizes terminal once no task is active.
        parent_job_id, _ = tasks[0].task_id.require_task()
        assert query_job(state, parent_job_id).state in TERMINAL_JOB_STATES

        # The direct victim emitted a task_preempted event; the cascaded sibling
        # is terminated by the peer cascade, not a separate preempt decision.
        preempted = {ev.entity_id for ev in result0.log_events if ev.action == "task_preempted"}
        assert preempted == {tasks[0].task_id.to_wire()}


def test_late_heartbeat_after_preempt_to_pending_does_not_revive_attempt():
    """Regression: after preempt_task retries a task (state -> PENDING, attempt -> PREEMPTED),
    a late worker heartbeat for the dead attempt_id must NOT revive the attempt row back
    to RUNNING while leaving `error` and `finished_at_ms` set.

    Observed in production (job /eczech/iris-run-exp109_bolinas_sweep_eval-...): the
    attempt ended up in the impossible mixed state
        state=RUNNING, error="Preempted by ...", finished_at_ms=<set>
    because preempt_task leaves `tasks.current_attempt_id` pointing at the dead
    attempt, so task.apply_one_transition's stale-attempt guard fails to fire and
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

        with state._db.transaction() as cur:
            finalize(
                cur,
                [TerminalDecision(TerminalKind.PREEMPT, task.task_id, "Preempted by /bob/prod-job:0")],
                now=Timestamp.now(),
            )

        # Sanity: task went to PENDING (budget remains), attempt row is in
        # PREEMPTED reporting state. ``preempt_task`` is a producer transition
        # (``stamp_attempt_finished=False``), so ``finished_at_ms`` is intentionally
        # left NULL — the worker still holds the chips until a terminal
        # heartbeat (or worker-failure synthesis) lands.
        assert query_task(state, task.task_id).state == job_pb2.TASK_STATE_PENDING
        attempt_after_preempt = query_attempt(state, task.task_id, dead_attempt_id)
        assert attempt_after_preempt is not None
        assert attempt_after_preempt.state == job_pb2.TASK_STATE_PREEMPTED
        assert (
            attempt_after_preempt.finished_at_ms is None
        ), "producer-side preempt must not stamp finished_at_ms; that is the heartbeat path's job"
        assert attempt_after_preempt.error == "Preempted by /bob/prod-job:0"

        # Late heartbeat for the (now-dead) attempt 0 arrives: worker still thinks
        # it is RUNNING. This simulates the RPC-in-flight race.
        with state._db.transaction() as cur:
            apply_task_observations(
                cur,
                [
                    WorkerTaskUpdates(
                        worker_id=worker_id,
                        updates=[
                            TaskUpdate(
                                task_id=task.task_id,
                                attempt_id=dead_attempt_id,
                                new_state=job_pb2.TASK_STATE_RUNNING,
                            )
                        ],
                    )
                ],
                health=state._health,
                now=Timestamp.now(),
            )

        # The attempt row must remain in a consistent state — NOT flipped
        # back to RUNNING. ``finished_at`` may still be NULL because the
        # producer-side preempt deliberately leaves it that way.
        attempt_final = query_attempt(state, task.task_id, dead_attempt_id)
        assert attempt_final is not None, "attempt row disappeared"
        assert attempt_final.state == job_pb2.TASK_STATE_PREEMPTED, (
            f"attempt {dead_attempt_id} was revived to state={attempt_final.state} "
            f"(expected PREEMPTED={job_pb2.TASK_STATE_PREEMPTED}); "
            f"error={attempt_final.error!r}, finished_at={attempt_final.finished_at_ms}"
        )
        assert attempt_final.error == "Preempted by /bob/prod-job:0"
