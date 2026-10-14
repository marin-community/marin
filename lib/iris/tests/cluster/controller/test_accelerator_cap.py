# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hard per-identity accelerator cap: held at placement on both the worker
scheduler path and the Kubernetes dispatch path."""

import pytest
from iris.cluster.controller import ops, writes
from iris.cluster.controller.backend import ScheduleRequest, ScheduleResult, run_scheduling_decision
from iris.cluster.controller.budget import user_cap_status
from iris.cluster.controller.reconcile import dispatch
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.scheduling.policy import build_scheduling_context, compute_demand_entries
from iris.cluster.controller.scheduling.scheduler import Scheduler
from iris.cluster.types import JobName, UserBudgetDefaults
from iris.rpc import job_pb2
from iris.testing.controller import (
    inject_device_constraints,
    make_job_request,
    make_worker_metadata,
    query_task,
    query_tasks_for_job,
    register_worker,
)
from iris.testing.controller_state import ControllerTestState, set_task_state_for_test, submit_job_in_tx
from iris.testing.transitions import WorkerTaskUpdates, apply_task_observations
from rigging.timing import Timestamp

AGENT = "belay-agent@hai-gcp-models.iam.gserviceaccount.com"
GPUS_PER_NODE = 8
NO_SOFT_BUDGET = UserBudgetDefaults(budget_limit=0)


def _gpu_request(name: str, *, replicas: int = 1, gpus: int = GPUS_PER_NODE, band: int = 0, coscheduled: bool = False):
    request = make_job_request(name=name, cpu=1, replicas=replicas, priority_band=band)
    request.resources.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="H100", count=gpus))
    if coscheduled:
        request.coscheduling.group_by = "pool"
    inject_device_constraints(request)
    return request


def _submit(
    state: ControllerTestState,
    job_id: JobName,
    *,
    submitting_user: str | None = AGENT,
    replicas: int = 1,
    band: int = 0,
    coscheduled: bool = False,
) -> list[JobName]:
    request = _gpu_request(job_id.to_wire(), replicas=replicas, band=band, coscheduled=coscheduled)
    with state._db.transaction() as cur:
        submit_job_in_tx(cur, job_id=job_id, request=request, submitting_user=submitting_user)
    return [row.task_id for row in query_tasks_for_job(state, job_id)]


def _add_gpu_workers(state: ControllerTestState, count: int) -> None:
    for i in range(count):
        metadata = make_worker_metadata(cpu=64, memory_bytes=512 * 1024**3, gpu_count=GPUS_PER_NODE, gpu_name="H100")
        metadata.attributes["pool"].string_value = "a"
        register_worker(state, f"gpu-{i}", f"addr-{i}", metadata)


def _set_cap(state: ControllerTestState, user: str, max_accelerators: int) -> None:
    with state._db.transaction() as cur:
        writes.set_user_cap(
            cur,
            user,
            max_accelerators,
            Timestamp.now(),
            default_budget_limit=0,
            default_max_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
        )


def _schedule(state: ControllerTestState, defaults: UserBudgetDefaults = NO_SOFT_BUDGET) -> ScheduleResult:
    """Run one full scheduling pass and commit its assignments."""
    with state._db.read_snapshot() as snap:
        context = build_scheduling_context(snap, state._health, state._worker_attrs, defaults)
    result = run_scheduling_decision(Scheduler(), ScheduleRequest(context=context, max_tasks_per_job_per_cycle=100))
    with state._db.transaction() as cur:
        ops.task.assign(cur, result.assignments, health=state._health)
    return result


def _placed(result: ScheduleResult) -> set[JobName]:
    return {assignment.task_id for assignment in result.assignments}


def _finish(state: ControllerTestState, task_id: JobName) -> None:
    task = query_task(state, task_id)
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=task.current_worker_id,
                    updates=[
                        TaskUpdate(
                            task_id=task_id,
                            attempt_id=task.current_attempt_id,
                            new_state=job_pb2.TASK_STATE_SUCCEEDED,
                        )
                    ],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )


def _state_of(state: ControllerTestState, task_id: JobName) -> int:
    return query_task(state, task_id).state


def test_tasks_beyond_cap_stay_pending_until_capacity_frees(state):
    _add_gpu_workers(state, 4)
    _set_cap(state, AGENT, 2 * GPUS_PER_NODE)
    tasks = _submit(state, JobName.root("belay", "train"), replicas=4)

    first = _schedule(state)

    assert len(first.assignments) == 2
    held = [t for t in tasks if t not in _placed(first)]
    assert all(_state_of(state, t) == job_pb2.TASK_STATE_PENDING for t in held)
    assert first.diagnostics[JobName.root("belay", "train").to_wire()].startswith("held: user cap 16 accelerators")

    assert _schedule(state).assignments == []

    _finish(state, next(iter(_placed(first))))
    second = _schedule(state)
    assert len(second.assignments) == 1


def test_batch_tasks_count_toward_cap(state):
    _add_gpu_workers(state, 2)
    _set_cap(state, AGENT, GPUS_PER_NODE)
    [batch_task] = _submit(state, JobName.root("belay", "batch"), band=job_pb2.PRIORITY_BAND_BATCH)
    assert _placed(_schedule(state)) == {batch_task}

    [interactive_task] = _submit(state, JobName.root("belay", "interactive"))
    assert _schedule(state).assignments == []
    assert _state_of(state, interactive_task) == job_pb2.TASK_STATE_PENDING


def test_zero_cap_places_without_limit(state):
    _add_gpu_workers(state, 3)
    _set_cap(state, AGENT, 0)
    tasks = _submit(state, JobName.root("belay", "train"), replicas=3)

    assert _placed(_schedule(state)) == set(tasks)


def test_default_cap_applies_to_users_without_a_budget_row(state):
    _add_gpu_workers(state, 3)
    _submit(state, JobName.root("belay", "train"), replicas=3)

    result = _schedule(state, UserBudgetDefaults(budget_limit=0, max_accelerators=GPUS_PER_NODE))

    assert len(result.assignments) == 1


def test_one_pass_does_not_overshoot_across_jobs(state):
    _add_gpu_workers(state, 6)
    _set_cap(state, AGENT, 3 * GPUS_PER_NODE)
    for i in range(3):
        _submit(state, JobName.root("belay", f"job-{i}"), replicas=2)

    assert len(_schedule(state).assignments) == 3


def test_coscheduled_gang_exceeding_remaining_cap_is_held_whole(state):
    _add_gpu_workers(state, 4)
    _set_cap(state, AGENT, 3 * GPUS_PER_NODE)
    [solo] = _submit(state, JobName.root("belay", "solo"))
    assert _placed(_schedule(state)) == {solo}

    gang = _submit(state, JobName.root("belay", "gang"), replicas=3, coscheduled=True)
    result = _schedule(state)

    assert result.assignments == []
    assert all(_state_of(state, t) == job_pb2.TASK_STATE_PENDING for t in gang)


def test_child_jobs_count_against_root_submitter(state):
    _add_gpu_workers(state, 3)
    _set_cap(state, AGENT, 2 * GPUS_PER_NODE)
    root = JobName.root("belay", "parent")
    _submit(state, root)
    # A child submitted from inside the job carries no principal of its own; it
    # inherits the root's submitter. Were it charged to the job owner "belay"
    # instead, all three tasks would place.
    _submit(state, root.child("child"), submitting_user=None, replicas=2)

    assert len(_schedule(state).assignments) == 2


def test_cap_holds_only_the_capped_identity(state):
    _add_gpu_workers(state, 3)
    _set_cap(state, AGENT, GPUS_PER_NODE)
    _submit(state, JobName.root("belay", "train"), replicas=2)
    other = _submit(state, JobName.root("alice", "train"), submitting_user="alice@example.com")

    placed = _placed(_schedule(state))

    assert len(placed) == 2
    assert set(other) <= placed


def test_raising_cap_releases_held_tasks_and_lowering_never_kills(state):
    _add_gpu_workers(state, 3)
    _set_cap(state, AGENT, GPUS_PER_NODE)
    tasks = _submit(state, JobName.root("belay", "train"), replicas=3)
    assert len(_schedule(state).assignments) == 1

    _set_cap(state, AGENT, 3 * GPUS_PER_NODE)
    assert len(_schedule(state).assignments) == 2

    _set_cap(state, AGENT, GPUS_PER_NODE)
    assert _schedule(state).assignments == []
    assert all(_state_of(state, t) == job_pb2.TASK_STATE_ASSIGNED for t in tasks)


@pytest.mark.parametrize("cap_nodes,expect_preemption", [(1, False), (0, True)])
def test_held_preemptor_does_not_evict_lower_band_work(state, cap_nodes, expect_preemption):
    _add_gpu_workers(state, 1)
    _set_cap(state, AGENT, cap_nodes * GPUS_PER_NODE)
    alice = "alice@example.com"
    [squatter] = _submit(state, JobName.root("alice", "batch"), submitting_user=alice, band=job_pb2.PRIORITY_BAND_BATCH)
    assert _placed(_schedule(state)) == {squatter}
    set_task_state_for_test(state, squatter, job_pb2.TASK_STATE_RUNNING)
    # belay already holds one node's worth elsewhere, filling a 1-node cap.
    [holding] = _submit(state, JobName.root("belay", "holding"))
    set_task_state_for_test(state, holding, job_pb2.TASK_STATE_RUNNING)

    _submit(state, JobName.root("belay", "urgent"), band=job_pb2.PRIORITY_BAND_PRODUCTION)
    result = _schedule(state)

    assert [p.task_id for p in result.preemptions] == ([squatter] if expect_preemption else [])


def test_held_tasks_emit_no_autoscaler_demand(state):
    _set_cap(state, AGENT, GPUS_PER_NODE)
    _submit(state, JobName.root("belay", "train"), replicas=3)

    with state._db.read_snapshot() as snap:
        context = build_scheduling_context(snap, state._health, state._worker_attrs, NO_SOFT_BUDGET)
    demand = compute_demand_entries(context, Scheduler())

    assert sum(len(entry.task_ids) for entry in demand) == 1


# --- Kubernetes direct dispatch -------------------------------------------------


def _drain(state: ControllerTestState, defaults: UserBudgetDefaults = NO_SOFT_BUDGET) -> dispatch.DispatchBatch:
    with state._db.transaction() as cur:
        return dispatch.drain_for_dispatch(cur, defaults=defaults)


def test_k8s_dispatch_creates_no_pod_beyond_cap_until_tasks_finish(state):
    _set_cap(state, AGENT, 2 * GPUS_PER_NODE)
    job_id = JobName.root("belay", "train")
    tasks = _submit(state, job_id, replicas=3)

    def pending() -> list[JobName]:
        return [t for t in tasks if _state_of(state, t) == job_pb2.TASK_STATE_PENDING]

    batch = _drain(state)

    assert len(pending()) == 1
    assert batch.cap_holds[job_id.to_wire()].startswith("held: user cap 16 accelerators (16 in use")
    _drain(state)
    assert len(pending()) == 1

    [held] = pending()
    set_task_state_for_test(state, next(t for t in tasks if t != held), job_pb2.TASK_STATE_SUCCEEDED)
    released = _drain(state)
    assert pending() == []
    assert released.cap_holds == {}


def test_k8s_dispatch_holds_whole_gang_over_remaining_cap(state):
    _set_cap(state, AGENT, 2 * GPUS_PER_NODE)
    [solo] = _submit(state, JobName.root("belay", "solo"))
    gang_id = JobName.root("belay", "gang")
    gang = _submit(state, gang_id, replicas=2, coscheduled=True)

    batch = _drain(state)

    assert [request.task_id for request in batch.tasks_to_run] == [solo.to_wire()]
    assert gang_id.to_wire() in batch.cap_holds
    assert all(_state_of(state, t) == job_pb2.TASK_STATE_PENDING for t in gang)


def test_cap_status_reports_usage_and_held_work(state):
    _add_gpu_workers(state, 3)
    _set_cap(state, AGENT, GPUS_PER_NODE)
    _submit(state, JobName.root("belay", "train"), replicas=3)
    _schedule(state)

    with state._db.read_snapshot() as snap:
        status = user_cap_status(snap, AGENT, NO_SOFT_BUDGET)

    assert (status.max_accelerators, status.accelerators_in_use) == (GPUS_PER_NODE, GPUS_PER_NODE)
    assert (status.held_jobs, status.held_tasks) == (1, 2)
