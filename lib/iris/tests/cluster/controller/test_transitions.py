# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for controller state management.

These tests exercise end-to-end observable behavior through the event-driven API (handle_event).
They focus on:
- Full workflows (submit job -> dispatch tasks -> complete/fail)
- Important edge cases (retry exhaustion, worker failure cascades, failure domains)
- Final state verification rather than intermediate steps
"""

import threading

import pytest
from finelog.rpc import logging_pb2
from iris.cluster.constraints import DeviceType, WellKnownAttribute
from iris.cluster.controller import ops, reads, writes
from iris.cluster.controller.codec import constraints_from_json, device_counts_from_json, device_variant_from_json
from iris.cluster.controller.ops.task import Assignment, finalize
from iris.cluster.controller.projections.endpoints import EndpointQuery, EndpointRow
from iris.cluster.controller.projections.run_templates import RunTemplatesProjection
from iris.cluster.controller.pruner import PruneResult, prune_old_data
from iris.cluster.controller.reads import WorkerResourceUsage

# =============================================================================
# Test Helpers
# =============================================================================
from iris.cluster.controller.reconcile import dispatch
from iris.cluster.controller.reconcile.effects import JobRowDelta
from iris.cluster.controller.reconcile.job import recompute_state
from iris.cluster.controller.reconcile.overlay import Overlay
from iris.cluster.controller.reconcile.policy import MAX_REPLICAS_PER_JOB
from iris.cluster.controller.reconcile.snapshot import (
    JobStateBasis,
    TaskHistogramRow,
    TaskUpdate,
    TransitionSnapshot,
)
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind
from iris.cluster.controller.scheduling.policy import build_scheduling_context, compute_demand_entries
from iris.cluster.controller.scheduling.scheduler import (
    DEFAULT_MAX_ASSIGNMENTS_PER_WORKER,
    JobRequirements,
    Scheduler,
    SchedulingContext,
    worker_snapshot_from_row,
)
from iris.cluster.controller.schema import jobs_table, slices_table, task_attempts_table, tasks_table, workers_table
from iris.cluster.log_keys import task_log_key
from iris.cluster.types import TERMINAL_TASK_STATES, JobName, TaskAttempt, UserBudgetDefaults, WorkerId
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Duration, Timestamp
from sqlalchemy import func, insert, select
from sqlalchemy import update as sa_update
from tests.cluster.controller._test_support import ControllerTestState, create_attempt_for_test
from tests.cluster.controller.transition_driver import (
    WorkerTaskUpdates,
    apply_task_observations,
    commit_dispatch_updates,
)

from .conftest import (
    building_counts as _building_counts,
)
from .conftest import (
    check_task_can_be_scheduled,
    check_task_is_finished,
    dispatch_task,
    fail_worker,
    healthy_active_workers,
    make_job_request,
    make_worker_metadata,
    register_worker,
    submit_job,
    transition_task,
    worker_daemon_backends_for_prune,
    worker_running_tasks,
)
from .conftest import (
    make_test_entrypoint as _make_test_entrypoint,
)
from .conftest import (
    query_attempt as _query_attempt,
)
from .conftest import (
    query_job as _query_job,
)
from .conftest import (
    query_task as _query_task,
)
from .conftest import (
    query_worker as _query_worker,
)
from .conftest import (
    schedulable_tasks as _schedulable_tasks,
)

_ZERO_USAGE = WorkerResourceUsage(0, 0, 0, 0)


def _usage_for_worker(state: ControllerTestState, worker_id: WorkerId) -> WorkerResourceUsage:
    """Derived per-worker usage (replaces the old ``workers.committed_*`` cache)."""
    with state._db.read_snapshot() as snap:
        return reads.resource_usage_by_worker(snap).get(worker_id, _ZERO_USAGE)


def _endpoints(state: ControllerTestState, query: EndpointQuery = EndpointQuery()) -> list[EndpointRow]:
    rows = state._endpoints.query(query)
    # Mirror the original helper's ordering (registered_at DESC, endpoint_id ASC).
    return sorted(rows, key=lambda r: (-r.registered_at.epoch_ms(), r.endpoint_id))


def _demand_entries(state: ControllerTestState):
    """Compute autoscaler demand through the shared scheduling-context path.

    Mirrors the production demand path: build the per-tick scheduling context
    from the live DB and run the single demand computation over it.
    """
    with state._db.read_snapshot() as snap:
        ctx = build_scheduling_context(snap, state._health, state._worker_attrs, UserBudgetDefaults(), {})
    return compute_demand_entries(ctx, Scheduler(), {})


def _build_scheduling_context(scheduler: Scheduler, state: ControllerTestState):
    pending = _schedulable_tasks(state)
    workers = healthy_active_workers(state)
    task_ids = [t.task_id for t in pending]
    jobs: dict[JobName, JobRequirements] = {}
    for t in pending:
        job_id = t.task_id.parent
        if job_id and job_id not in jobs:
            job = _query_job(state, job_id)
            if job:
                dc = device_counts_from_json(job.res_device_json)
                jobs[job_id] = JobRequirements(
                    req_cpu_millicores=job.res_cpu_millicores,
                    req_memory_bytes=job.res_memory_bytes,
                    req_gpu_count=dc.gpu,
                    req_tpu_count=dc.tpu,
                    device_variant=device_variant_from_json(job.res_device_json),
                    constraints=constraints_from_json(job.constraints_json),
                    is_coscheduled=job.has_coscheduling,
                    coscheduling_group_by=job.coscheduling_group_by if job.has_coscheduling else None,
                )
    with state._db.read_snapshot() as snap:
        usage = reads.resource_usage_by_worker(snap)
    snapshots = [worker_snapshot_from_row(w, usage.get(w.worker_id)) for w in workers]
    return SchedulingContext(
        workers=snapshots,
        building_counts=_building_counts(state),
        max_building_tasks=scheduler.max_building_tasks_per_worker,
        max_assignments_per_worker=DEFAULT_MAX_ASSIGNMENTS_PER_WORKER,
        pending_tasks=task_ids,
        jobs=jobs,
        pending_task_rows=[],
        user_spend={},
        user_budget_limits={},
        requested_bands={},
        user_budget_defaults=UserBudgetDefaults(),
    )


def test_sa_core_select_returns_typed_rows(state) -> None:
    """SA Core reads return TypeDecorator-decoded values (JobName, Timestamp, etc.)."""
    request = make_job_request("typed-rows")
    tasks = submit_job(state, "typed-rows", request)

    job_id = JobName.root("test-user", "typed-rows")
    with state._db.read_snapshot() as tx:
        job_row = reads.get_job_detail(tx, job_id)
        task_count = tx.execute(select(jobs_table.c.num_tasks).where(jobs_table.c.job_id == job_id)).first()

    assert job_row is not None
    assert job_row.submitted_at_ms is not None
    assert job_row.job_id == job_id
    assert task_count.num_tasks == len(tasks)


def test_sa_core_typed_values_roundtrip(state) -> None:
    """TypeDecorators round-trip correctly through SA Core read+write paths."""
    wid = register_worker(state, "proj-worker", "addr", make_worker_metadata())
    request = controller_pb2.Controller.LaunchJobRequest(
        name=JobName.root("test-user", "projection").to_wire(),
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    [task] = submit_job(state, "projection", request)
    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task.task_id, worker_id=wid)], health=state._health)

    running = worker_running_tasks(state, wid)

    assert len(running) == 1
    assert task.task_id in running


def test_sa_core_read_snapshot_finds_workers(state) -> None:
    register_worker(state, "exists-worker", "addr", make_worker_metadata())

    with state._db.read_snapshot() as tx:
        row = tx.execute(select(workers_table.c.worker_id).where(workers_table.c.worker_id == "exists-worker")).first()
    assert row is not None


# =============================================================================
# Job/Task Lifecycle Integration Tests
# =============================================================================


def test_job_lifecycle_success(harness):
    """E2E: Submit job -> dispatch task -> succeed -> verify final state."""
    worker_id = harness.add_worker("w1")
    tasks = harness.submit("j1", replicas=2)

    assert len(tasks) == 2
    assert harness.query_job(JobName.root("test-user", "j1")).state == job_pb2.JOB_STATE_PENDING

    for task in tasks:
        harness.dispatch(task, worker_id)
        harness.transition(task.task_id, job_pb2.TASK_STATE_SUCCEEDED)

    assert harness.query_job(JobName.root("test-user", "j1")).state == job_pb2.JOB_STATE_SUCCEEDED
    for task in tasks:
        assert harness.query_task(task.task_id).state == job_pb2.TASK_STATE_SUCCEEDED
    assert len(_schedulable_tasks(harness.state)) == 0


def test_job_lifecycle_failure_exhausted_retries(harness):
    """E2E: Task failure with no retries -> job fails."""
    worker_id = harness.add_worker("w1")
    [task] = harness.submit("j1")
    job_id = JobName.root("test-user", "j1")

    harness.dispatch(task, worker_id)
    harness.transition(task.task_id, job_pb2.TASK_STATE_FAILED, error="Task failed")

    assert harness.query_task(task.task_id).state == job_pb2.TASK_STATE_FAILED
    assert check_task_is_finished(harness.query_task(task.task_id))
    assert harness.query_job(job_id).state == job_pb2.JOB_STATE_FAILED


def test_task_failure_with_retry_requeues(harness):
    """E2E: Task failure with retries -> task requeued, job stays running."""
    worker_id = harness.add_worker("w1")

    req = make_job_request("job1")
    req.max_task_failures = 1
    req.max_retries_failure = 1
    tasks = submit_job(harness.state, "j1", req)
    task = tasks[0]
    job_id = JobName.root("test-user", "j1")

    harness.dispatch(task, worker_id)
    harness.transition(task.task_id, job_pb2.TASK_STATE_FAILED)

    assert harness.query_task(task.task_id).state == job_pb2.TASK_STATE_PENDING
    assert check_task_can_be_scheduled(harness.query_task(task.task_id))
    assert harness.query_job(job_id).state == job_pb2.JOB_STATE_RUNNING
    pending = _schedulable_tasks(harness.state)
    assert len(pending) == 1
    assert pending[0].task_id == task.task_id


def test_unschedulable_task_finalizes_job_with_timeout_error(harness):
    """E2E: Task UNSCHEDULABLE propagates timeout-style error to final job state."""
    worker_id = harness.add_worker("w1")
    [task] = harness.submit("j1", scheduling_timeout_seconds=300)
    job_id = JobName.root("test-user", "j1")

    harness.dispatch(task, worker_id)
    harness.transition(task.task_id, job_pb2.TASK_STATE_UNSCHEDULABLE)

    assert harness.query_task(task.task_id).state == job_pb2.TASK_STATE_UNSCHEDULABLE
    assert harness.query_task(task.task_id).error == "Scheduling timeout exceeded"
    assert harness.query_job(job_id).state == job_pb2.JOB_STATE_UNSCHEDULABLE
    assert harness.query_job(job_id).error == "Scheduling timeout exceeded"


def test_job_cancellation_kills_all_tasks(harness):
    """E2E: Job cancellation -> all tasks killed."""
    worker_id = harness.add_worker("w1")
    tasks = harness.submit("j1", replicas=3)
    job_id = JobName.root("test-user", "j1")

    harness.dispatch(tasks[0], worker_id)
    harness.dispatch(tasks[1], worker_id)

    with harness.state._db.transaction() as cur:
        ops.job.cancel(cur, job_id=job_id, reason="User cancelled")

    assert harness.query_job(job_id).state == job_pb2.JOB_STATE_KILLED
    for task in tasks:
        assert harness.query_task(task.task_id).state == job_pb2.TASK_STATE_KILLED


def test_cancel_job_holds_resources_until_heartbeat_finalization(harness):
    """Verify ``cancel_job`` is a producer transition: the attempt's
    ``finished_at_ms`` stays NULL so the worker keeps owning its slice in
    the scheduler's derived usage view until the heartbeat path stamps the
    timestamp.
    """
    w1 = harness.add_worker("w1")
    w2 = harness.add_worker("w2")
    tasks = harness.submit("j1", replicas=3)

    harness.dispatch(tasks[0], w1)
    harness.dispatch(tasks[1], w2)

    assert _usage_for_worker(harness.state, w1).cpu_millicores == 1000
    assert _usage_for_worker(harness.state, w1).memory_bytes == 1024**3
    assert _usage_for_worker(harness.state, w2).cpu_millicores == 1000

    with harness.state._db.transaction() as cur:
        ops.job.cancel(
            cur,
            job_id=JobName.root("test-user", "j1"),
            reason="User cancelled",
        )

    # Producer-side cancel: usage stays held — finished_at_ms is still NULL.
    assert _usage_for_worker(harness.state, w1).cpu_millicores == 1000
    assert _usage_for_worker(harness.state, w2).cpu_millicores == 1000

    # Tasks transition to KILLED, removing them from the running-task view.
    assert len(worker_running_tasks(harness.state, w1)) == 0
    assert len(worker_running_tasks(harness.state, w2)) == 0


def test_cancel_job_rolls_attempt_state_without_finalizing(harness):
    """Verify ``cancel_job`` rolls the in-flight attempt's reporting state to
    KILLED so dashboards don't see a "killed but still RUNNING" row, while
    leaving ``finished_at_ms`` NULL so the scheduler retains capacity for
    the worker until heartbeat finalization.
    """
    w1 = harness.add_worker("w1")
    w2 = harness.add_worker("w2")
    tasks = harness.submit("j1", replicas=2)

    harness.dispatch(tasks[0], w1)
    harness.dispatch(tasks[1], w2)

    attempt_ids = {t.task_id: harness.query_task(t.task_id).current_attempt_id for t in tasks}
    assert all(aid >= 0 for aid in attempt_ids.values())
    for t in tasks:
        att = _query_attempt(harness.state, t.task_id, attempt_ids[t.task_id])
        assert att is not None
        assert att.state not in TERMINAL_TASK_STATES
        assert att.finished_at_ms is None

    with harness.state._db.transaction() as cur:
        ops.job.cancel(
            cur,
            job_id=JobName.root("test-user", "j1"),
            reason="User cancelled",
        )

    for t in tasks:
        att = _query_attempt(harness.state, t.task_id, attempt_ids[t.task_id])
        assert att is not None
        assert att.state in TERMINAL_TASK_STATES, f"orphan attempt left active for task {t.task_id} (state={att.state})"
        # Producer-side cancel does not stamp finished_at_ms — the
        # reconcile-observation path owns that write so the scheduler keeps
        # capacity held.
        assert att.finished_at_ms is None


def test_heartbeat_finalizes_stranded_attempt_after_producer_terminal(harness):
    """A producer transition (cancel) leaves the task terminal but the attempt
    unfinalized; a subsequent terminal observation — the case the reconcile
    planner now re-issues (via the worker's desired set) for worker-bound
    attempts with NULL finished_at_ms — must stamp finished_at_ms without
    rewriting the task's terminal state.
    """
    w1 = harness.add_worker("w1")
    tasks = harness.submit("j1", replicas=1)
    harness.dispatch(tasks[0], w1)

    task = tasks[0]
    attempt_id = harness.query_task(task.task_id).current_attempt_id

    with harness.state._db.transaction() as cur:
        ops.job.cancel(
            cur,
            job_id=JobName.root("test-user", "j1"),
            reason="User cancelled",
        )

    pre = _query_attempt(harness.state, task.task_id, attempt_id)
    assert pre is not None
    assert pre.worker_id is not None
    assert pre.state in TERMINAL_TASK_STATES
    assert pre.finished_at_ms is None
    pre_task_state = _query_task(harness.state, task.task_id).state

    with harness.state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=w1,
                    updates=[
                        TaskUpdate(
                            task_id=task.task_id,
                            attempt_id=attempt_id,
                            new_state=job_pb2.TASK_STATE_WORKER_FAILED,
                            error="Task not found on worker",
                        )
                    ],
                )
            ],
            health=harness.state._health,
            now=Timestamp.now(),
        )

    post = _query_attempt(harness.state, task.task_id, attempt_id)
    assert post is not None
    assert post.finished_at_ms is not None, "stranded attempt should be finalized by heartbeat"
    assert (
        _query_task(harness.state, task.task_id).state == pre_task_state
    ), "task's terminal state must not be rewritten by the late heartbeat"


def test_cancel_job_preserves_kill_worker_mapping_after_clearing_tasks(harness):
    """cancel_job marks tasks KILLED and clears their worker binding."""
    w1 = harness.add_worker("w1")
    w2 = harness.add_worker("w2")
    tasks = harness.submit("j1", replicas=2)

    harness.dispatch(tasks[0], w1)
    harness.dispatch(tasks[1], w2)

    with harness.state._db.transaction() as cur:
        ops.job.cancel(
            cur,
            job_id=JobName.root("test-user", "j1"),
            reason="User cancelled",
        )

    assert harness.query_task(tasks[0].task_id).state == job_pb2.TASK_STATE_KILLED
    assert harness.query_task(tasks[1].task_id).state == job_pb2.TASK_STATE_KILLED
    assert harness.query_task(tasks[0].task_id).current_worker_id is None
    assert harness.query_task(tasks[1].task_id).current_worker_id is None


def test_cancel_job_removes_endpoints_for_job_tree(state):

    parent_worker = register_worker(state, "w1", "host1:8080", make_worker_metadata())
    child_worker = register_worker(state, "w2", "host2:8080", make_worker_metadata())

    parent_tasks = submit_job(state, "parent", make_job_request("parent"))
    child_req = make_job_request("child")
    child_req.name = JobName.from_string("/test-user/parent/child").to_wire()
    child_tasks = submit_job(state, "/test-user/parent/child", child_req)

    dispatch_task(state, parent_tasks[0], parent_worker)
    dispatch_task(state, child_tasks[0], child_worker)

    with state._db.transaction() as cur:
        state._endpoints.add(
            cur,
            EndpointRow(
                endpoint_id="parent-ep",
                name="parent/actor",
                address="host1:9000",
                task_id=parent_tasks[0].task_id,
                metadata={},
                registered_at=Timestamp.now(),
            ),
        )
    with state._db.transaction() as cur:
        state._endpoints.add(
            cur,
            EndpointRow(
                endpoint_id="child-ep",
                name="parent/child/actor",
                address="host2:9000",
                task_id=child_tasks[0].task_id,
                metadata={},
                registered_at=Timestamp.now(),
            ),
        )

    assert len(_endpoints(state, EndpointQuery())) == 2

    with state._db.transaction() as cur:
        ops.job.cancel(
            cur,
            job_id=JobName.root("test-user", "parent"),
            reason="User cancelled",
        )

    assert _endpoints(state, EndpointQuery()) == []


def test_cancelled_job_tasks_excluded_from_demand(harness):
    """Regression test for issue #2777: Killed tasks with no attempts should not appear in demand entries."""
    worker_id = harness.add_worker("w1")
    tasks = harness.submit("j1", replicas=3)
    job_id = JobName.root("test-user", "j1")

    harness.dispatch(tasks[0], worker_id)
    with harness.state._db.transaction() as cur:
        ops.job.cancel(cur, job_id=job_id, reason="User cancelled")

    assert harness.query_job(job_id).state == job_pb2.JOB_STATE_KILLED
    for task in tasks:
        assert harness.query_task(task.task_id).state == job_pb2.TASK_STATE_KILLED
        assert not check_task_can_be_scheduled(harness.query_task(task.task_id))

    assert len(_schedulable_tasks(harness.state)) == 0
    assert len(_demand_entries(harness.state)) == 0


# =============================================================================
# Worker Failure Cascade Tests
# =============================================================================


def test_worker_failure_cascades_to_running_tasks(harness):
    """E2E: Worker failure -> running tasks transition to WORKER_FAILED and requeue."""
    worker_id = harness.add_worker("w1")
    req = make_job_request("job1")
    req.max_retries_preemption = 1
    tasks = submit_job(harness.state, "j1", req)
    task = tasks[0]

    harness.dispatch(task, worker_id)
    fail_worker(harness.state, worker_id, "Connection lost")

    assert _query_worker(harness.state, worker_id) is None
    assert harness.query_task(task.task_id).state == job_pb2.TASK_STATE_PENDING
    assert check_task_can_be_scheduled(harness.query_task(task.task_id))
    assert len(_schedulable_tasks(harness.state)) == 1


def test_failed_worker_is_pruned_from_state(state):
    """E2E: Worker failure removes worker from state, preventing dead worker accumulation."""

    w1 = register_worker(state, "w1", "host1:8080", make_worker_metadata())
    w2 = register_worker(state, "w2", "host2:8080", make_worker_metadata())

    req = make_job_request("job1")
    req.max_retries_preemption = 1
    tasks = submit_job(state, "j1", req)
    dispatch_task(state, tasks[0], w1)

    # Worker w1 fails
    fail_worker(state, w1, "Connection lost")

    # w1 is gone from state entirely
    assert _query_worker(state, w1) is None
    # w2 is still present
    assert _query_worker(state, w2) is not None

    # list_all_workers only returns w2
    with state._db.read_snapshot() as tx:
        all_workers = tx.execute(select(workers_table.c.worker_id)).all()
    assert len(all_workers) == 1
    assert all_workers[0].worker_id == w2

    # Task was requeued despite worker removal
    assert tasks[0].state == job_pb2.TASK_STATE_PENDING
    assert check_task_can_be_scheduled(tasks[0])

    # A re-registering worker creates a fresh entry
    w1_again = register_worker(state, "w1", "host1:8080", make_worker_metadata())
    assert _query_worker(state, w1_again) is not None
    assert _query_worker(state, w1_again).healthy is True
    with state._db.read_snapshot() as tx:
        assert len(tx.execute(select(workers_table.c.worker_id)).all()) == 2


def test_dispatch_failure_marks_worker_failed_and_requeues_task(state):
    """E2E: Dispatch RPC failure (task in PENDING) -> worker failed event cascades to task."""

    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = make_job_request("job1")
    req.max_retries_preemption = 1
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    # Task gets assigned (creates attempt, puts in ASSIGNED state)
    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task.task_id, worker_id=worker_id)], health=state._health)
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_ASSIGNED
    assert _query_task(state, task.task_id).current_attempt_id == 0

    # Dispatch RPC fails -> WORKER_FAILED event
    fail_worker(state, worker_id, "Dispatch RPC failed: Connection refused")

    # Verify cascade:
    # 1. Worker marked unhealthy
    assert _query_worker(state, worker_id) is None

    # 2. Task requeued (back to PENDING for retry).
    #    Since the task was still ASSIGNED (never confirmed BUILDING/RUNNING),
    #    this is a delivery failure — no budget consumed at all.
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_PENDING
    assert _query_task(state, task.task_id).preemption_count == 0
    assert _query_task(state, task.task_id).failure_count == 0
    assert check_task_can_be_scheduled(_query_task(state, task.task_id))

    # 3. Task should be requeued for retry
    pending = _schedulable_tasks(state)
    assert len(pending) == 1
    assert pending[0].task_id == task.task_id

    # 4. Worker no longer has task assigned
    assert _query_worker(state, worker_id) is None


def test_task_assigned_to_missing_worker_is_ignored(state):
    """Stale assignments to pruned workers are skipped without crashing."""

    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())
    tasks = submit_job(state, "j1", make_job_request("job1"))
    task = tasks[0]

    # Worker disappears between scheduling and assignment commit.
    with state._db.transaction() as cur:
        writes.remove_worker(cur, worker_id, health=state._health)
    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task.task_id, worker_id=worker_id)], health=state._health)

    # Task remains schedulable and no attempt/resources are committed.
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_PENDING
    assert _query_task(state, task.task_id).current_attempt_id == -1
    assert check_task_can_be_scheduled(_query_task(state, task.task_id))
    assert task.task_id in {t.task_id for t in _schedulable_tasks(state)}


# =============================================================================
# Failure Domain Tests (max_task_failures)
# =============================================================================


def test_failure_domain_kills_remaining_tasks(state):
    """E2E: One task fails beyond retries -> remaining tasks killed, job fails."""

    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = controller_pb2.Controller.LaunchJobRequest(
        name="multi-task-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        max_task_failures=0,
        replicas=3,
    )
    tasks = submit_job(state, "j1", req)
    job = _query_job(state, JobName.root("test-user", "j1"))

    # Dispatch 2 tasks, leave 1 pending
    dispatch_task(state, tasks[0], worker_id)
    dispatch_task(state, tasks[1], worker_id)

    # Task-0 fails
    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_FAILED, error="Task failed")

    # Verify final state
    assert _query_job(state, job.job_id).state == job_pb2.JOB_STATE_FAILED
    assert _query_task(state, tasks[0].task_id).state == job_pb2.TASK_STATE_FAILED
    assert _query_task(state, tasks[1].task_id).state == job_pb2.TASK_STATE_KILLED
    assert _query_task(state, tasks[2].task_id).state == job_pb2.TASK_STATE_KILLED


@pytest.mark.parametrize("success_first", [True, False])
def test_batch_success_and_failure_is_order_independent(state, success_first):
    """A success + a job-failing failure in one batch is order-independent.

    With ``max_task_failures=0``, one task FAILED fails the job. A sibling that
    reaches SUCCEEDED in the same batch must stay SUCCEEDED (terminal finalize
    only kills NON-terminal tasks), and the job must end FAILED, regardless of
    the order of the two updates within the batch.
    """
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = controller_pb2.Controller.LaunchJobRequest(
        name="multi-task-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        max_task_failures=0,
        replicas=3,
    )
    tasks = submit_job(state, "j1", req)
    job_id = JobName.root("test-user", "j1")

    for task in tasks:
        dispatch_task(state, task, worker_id)

    succeed_update = TaskUpdate(
        task_id=tasks[0].task_id,
        attempt_id=_query_task(state, tasks[0].task_id).current_attempt_id,
        new_state=job_pb2.TASK_STATE_SUCCEEDED,
    )
    fail_update = TaskUpdate(
        task_id=tasks[1].task_id,
        attempt_id=_query_task(state, tasks[1].task_id).current_attempt_id,
        new_state=job_pb2.TASK_STATE_FAILED,
        error="boom",
    )
    updates = [succeed_update, fail_update] if success_first else [fail_update, succeed_update]

    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [WorkerTaskUpdates(worker_id=worker_id, updates=updates)],
            health=state._health,
            now=Timestamp.now(),
        )

    assert _query_task(state, tasks[0].task_id).state == job_pb2.TASK_STATE_SUCCEEDED
    assert _query_task(state, tasks[1].task_id).state == job_pb2.TASK_STATE_FAILED
    assert _query_task(state, tasks[2].task_id).state == job_pb2.TASK_STATE_KILLED
    assert _query_job(state, job_id).state == job_pb2.JOB_STATE_FAILED


def _report_worker_state(state, worker_id, task, new_state):
    """Feed one worker-reported task observation for ``task``'s current attempt."""
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=worker_id,
                    updates=[
                        TaskUpdate(
                            task_id=task.task_id,
                            attempt_id=_query_task(state, task.task_id).current_attempt_id,
                            new_state=new_state,
                        )
                    ],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )


def test_worker_reported_killed_on_live_attempt_retries_under_preemption_budget(state):
    """A worker-reported KILLED for the current attempt retries; it does not fail the job.

    A worker only reports KILLED when its container is stopped out-of-band — a
    higher-priority job reclaiming the slice, a node drain, a spot/preemptible
    reclaim, or a stop directive the controller issued without recording a task
    transition. None of these are application failures, so the kill is charged to
    the preemption budget (like WORKER_FAILED) and the task rolls back to PENDING.

    Regression for a production incident: a v5p training task was preempted
    (attempt 0), retried onto a fresh worker (attempt 1), and that worker reported
    KILLED ~13s in. The controller terminated the whole job (JOB_STATE_KILLED) with
    99 of 100 preemption retries unused, and the parent driver's wait() raised.
    """
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())
    req = controller_pb2.Controller.LaunchJobRequest(
        name="train",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
        max_retries_preemption=100,
    )
    [task] = submit_job(state, "train", req)
    job_id = JobName.root("test-user", "train")

    dispatch_task(state, task, worker_id)
    # Run the attempt first so the kill is charged against the preemption budget
    # (an EXECUTING task), exactly as in the incident.
    _report_worker_state(state, worker_id, task, job_pb2.TASK_STATE_RUNNING)

    _report_worker_state(state, worker_id, task, job_pb2.TASK_STATE_KILLED)

    retried = _query_task(state, task.task_id)
    assert retried.state == job_pb2.TASK_STATE_PENDING, "a live-attempt KILLED must roll the task back to PENDING"
    assert retried.preemption_count == 1, "the kill must be charged to the preemption budget"
    assert _query_job(state, job_id).state == job_pb2.JOB_STATE_RUNNING, "the job must stay alive across the retry"


def test_worker_reported_killed_terminal_when_preemption_budget_exhausted(state):
    """With the preemption budget exhausted, a worker-reported KILLED finalizes
    the task (as WORKER_FAILED) and the job — the retry is bounded, not infinite."""
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())
    req = controller_pb2.Controller.LaunchJobRequest(
        name="train",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
        max_retries_preemption=0,
    )
    [task] = submit_job(state, "train", req)
    job_id = JobName.root("test-user", "train")

    dispatch_task(state, task, worker_id)
    _report_worker_state(state, worker_id, task, job_pb2.TASK_STATE_RUNNING)

    _report_worker_state(state, worker_id, task, job_pb2.TASK_STATE_KILLED)

    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_WORKER_FAILED
    assert _query_job(state, job_id).state == job_pb2.JOB_STATE_WORKER_FAILED


def test_worker_reported_killed_on_coscheduled_task_requeues_the_gang(state):
    """A worker-reported KILLED on one coscheduled task bounces the whole gang.

    The retry must preserve the coscheduling invariant: when one gang member is
    stopped out-of-band, every sibling is requeued to PENDING so the job
    re-coschedules atomically onto a single slice — not left half-running with
    one member pending. (Mirrors the WORKER_FAILED peer cascade.)
    """
    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-killed",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
        max_retries_preemption=100,
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j-cosched-killed", req)
    job_id = JobName.root("test-user", "j-cosched-killed")

    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))
    # The coscheduled requeue cascade only fires from EXECUTING states.
    for i, task in enumerate(tasks):
        _report_worker_state(state, WorkerId(f"w{i}"), task, job_pb2.TASK_STATE_RUNNING)

    # One member is stopped out-of-band -> reported KILLED with budget to spare.
    _report_worker_state(state, WorkerId("w0"), tasks[0], job_pb2.TASK_STATE_KILLED)

    # The whole gang bounces to PENDING for atomic re-scheduling; the job survives.
    for task in tasks:
        assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_PENDING
    assert _query_task(state, tasks[0].task_id).preemption_count == 1
    assert _query_job(state, job_id).state == job_pb2.JOB_STATE_RUNNING


def test_max_task_failures_tolerance(state):
    """E2E: Job tolerates max_task_failures, then fails on next failure."""

    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = controller_pb2.Controller.LaunchJobRequest(
        name="tolerant-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=3,
        environment=job_pb2.EnvironmentConfig(),
        max_task_failures=1,
    )
    tasks = submit_job(state, "j1", req)
    job = _query_job(state, JobName.root("test-user", "j1"))

    for task in tasks:
        dispatch_task(state, task, worker_id)

    # First failure - job should keep running
    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_FAILED, error="First")
    assert _query_job(state, job.job_id).state == job_pb2.JOB_STATE_RUNNING

    # Second task succeeds
    transition_task(state, tasks[1].task_id, job_pb2.TASK_STATE_SUCCEEDED)
    assert _query_job(state, job.job_id).state == job_pb2.JOB_STATE_RUNNING

    # Third task fails - exceeds threshold, job fails
    transition_task(state, tasks[2].task_id, job_pb2.TASK_STATE_FAILED, error="Second")
    assert _query_job(state, job.job_id).state == job_pb2.JOB_STATE_FAILED


def test_preemption_does_not_count_toward_max_task_failures(state):
    """E2E: Worker failures (preemptions) don't count toward max_task_failures."""

    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = controller_pb2.Controller.LaunchJobRequest(
        name="preemption-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=2,
        environment=job_pb2.EnvironmentConfig(),
        max_task_failures=0,
        max_retries_preemption=1,
    )
    tasks = submit_job(state, "j1", req)
    job = _query_job(state, JobName.root("test-user", "j1"))

    dispatch_task(state, tasks[0], worker_id)
    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_WORKER_FAILED, error="Worker died")

    # Preemption doesn't count toward failure threshold; task requeued to PENDING
    assert tasks[0].state == job_pb2.TASK_STATE_PENDING
    assert check_task_can_be_scheduled(tasks[0])
    assert _query_job(state, job.job_id).state == job_pb2.JOB_STATE_RUNNING


# =============================================================================
# Endpoint Cleanup Tests
# =============================================================================


def test_terminal_states_clean_up_endpoints(state):
    """E2E: Task reaching terminal state removes associated endpoints."""

    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = make_job_request("job1")
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    dispatch_task(state, task, worker_id)

    ep = EndpointRow(
        endpoint_id="ep1",
        name="j1/actor",
        address="a:1",
        task_id=task.task_id,
        metadata={},
        registered_at=Timestamp.now(),
    )
    with state._db.transaction() as cur:
        state._endpoints.add(cur, ep)

    # Verify endpoint visible while running
    assert len(_endpoints(state, EndpointQuery(exact_name="j1/actor"))) == 1

    # Task succeeds
    transition_task(state, task.task_id, job_pb2.TASK_STATE_SUCCEEDED)

    # Endpoint removed
    assert _endpoints(state, EndpointQuery(exact_name="j1/actor")) == []


def test_endpoint_visibility_by_job_state(state):
    """Endpoints associated with a task are deleted when the task reaches a terminal state."""

    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = make_job_request("test")
    tasks = submit_job(state, "ns-1", req)
    job = _query_job(state, JobName.root("test-user", "ns-1"))
    task = tasks[0]

    ep = EndpointRow(
        endpoint_id="ep-1",
        name="ns-1/actor",
        address="10.0.0.1:8080",
        task_id=task.task_id,
        metadata={},
        registered_at=Timestamp.now(),
    )
    with state._db.transaction() as cur:
        state._endpoints.add(cur, ep)

    # Visible while pending
    assert len(_endpoints(state, EndpointQuery(exact_name="ns-1/actor"))) == 1

    # Still visible after transition to running
    dispatch_task(state, task, worker_id)
    assert _query_job(state, job.job_id).state == job_pb2.JOB_STATE_RUNNING
    assert len(_endpoints(state, EndpointQuery(exact_name="ns-1/actor"))) == 1

    # Deleted when task reaches terminal state
    transition_task(state, task.task_id, job_pb2.TASK_STATE_SUCCEEDED)
    assert _query_job(state, job.job_id).state == job_pb2.JOB_STATE_SUCCEEDED
    assert len(_endpoints(state, EndpointQuery(exact_name="ns-1/actor"))) == 0


def test_endpoint_deleted_on_task_failure_with_retry(state):
    """Endpoints are cleaned up when a task fails even if it retries back to PENDING."""

    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = make_job_request("test")
    req.max_retries_failure = 1
    req.max_task_failures = 1
    tasks = submit_job(state, "ns-1", req)
    task = tasks[0]

    dispatch_task(state, task, worker_id)

    ep = EndpointRow(
        endpoint_id="ep-1",
        name="ns-1/actor",
        address="10.0.0.1:8080",
        task_id=task.task_id,
        metadata={},
        registered_at=Timestamp.now(),
    )
    with state._db.transaction() as cur:
        state._endpoints.add(cur, ep)
    assert len(_endpoints(state, EndpointQuery(exact_name="ns-1/actor"))) == 1

    # Task fails but retries (goes back to PENDING)
    transition_task(state, task.task_id, job_pb2.TASK_STATE_FAILED, error="crash")
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_PENDING

    # Stale endpoints should be deleted even though the task retried
    assert len(_endpoints(state, EndpointQuery(exact_name="ns-1/actor"))) == 0


def test_endpoint_deleted_on_worker_failure(state):
    """Endpoints are cleaned up when the worker dies, even if the task retries."""

    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = make_job_request("test")
    req.max_retries_preemption = 1
    tasks = submit_job(state, "ns-1", req)
    task = tasks[0]

    dispatch_task(state, task, worker_id)

    ep = EndpointRow(
        endpoint_id="ep-1",
        name="ns-1/actor",
        address="10.0.0.1:8080",
        task_id=task.task_id,
        metadata={},
        registered_at=Timestamp.now(),
    )
    with state._db.transaction() as cur:
        state._endpoints.add(cur, ep)
    assert len(_endpoints(state, EndpointQuery(exact_name="ns-1/actor"))) == 1

    # Worker fails -> task retries to PENDING
    fail_worker(state, worker_id, "Connection lost")
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_PENDING

    # Endpoints should be cleaned up because the worker is dead
    assert len(_endpoints(state, EndpointQuery(exact_name="ns-1/actor"))) == 0


def test_endpoint_survives_building_state(state):
    """Endpoints registered during BUILDING are not deleted by subsequent BUILDING updates."""

    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = make_job_request("test")
    tasks = submit_job(state, "ns-1", req)
    task = tasks[0]

    # Assign task and transition to BUILDING
    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task.task_id, worker_id=worker_id)], health=state._health)
    task = _query_task(state, task.task_id)
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=worker_id,
                    updates=[
                        TaskUpdate(
                            task_id=task.task_id,
                            attempt_id=task.current_attempt_id,
                            new_state=job_pb2.TASK_STATE_BUILDING,
                        )
                    ],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )

    # Register endpoint during BUILDING (e.g. jax_init.py pre-registration)
    ep = EndpointRow(
        endpoint_id="ep-1",
        name="ns-1/actor",
        address="10.0.0.1:8080",
        task_id=task.task_id,
        metadata={},
        registered_at=Timestamp.now(),
    )
    with state._db.transaction() as cur:
        state._endpoints.add(cur, ep)
    assert len(_endpoints(state, EndpointQuery(exact_name="ns-1/actor"))) == 1

    # Transition to RUNNING — endpoint should survive
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=worker_id,
                    updates=[
                        TaskUpdate(
                            task_id=task.task_id,
                            attempt_id=_query_task(state, task.task_id).current_attempt_id,
                            new_state=job_pb2.TASK_STATE_RUNNING,
                        )
                    ],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )
    assert len(_endpoints(state, EndpointQuery(exact_name="ns-1/actor"))) == 1


def test_namespace_isolation(state):
    """E2E: Endpoints are isolated by namespace prefix."""

    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req1 = make_job_request("test1")
    req2 = make_job_request("test2")

    tasks1 = submit_job(state, "ns-1", req1)
    tasks2 = submit_job(state, "ns-2", req2)

    # Dispatch tasks to transition jobs to RUNNING state
    dispatch_task(state, tasks1[0], worker_id)
    dispatch_task(state, tasks2[0], worker_id)

    with state._db.transaction() as cur:
        state._endpoints.add(
            cur,
            EndpointRow(
                endpoint_id="ep-1",
                name="ns-1/actor",
                address="10.0.0.1:8080",
                task_id=tasks1[0].task_id,
                metadata={},
                registered_at=Timestamp.now(),
            ),
        )
    with state._db.transaction() as cur:
        state._endpoints.add(
            cur,
            EndpointRow(
                endpoint_id="ep-2",
                name="ns-2/actor",
                address="10.0.0.2:8080",
                task_id=tasks2[0].task_id,
                metadata={},
                registered_at=Timestamp.now(),
            ),
        )

    # Each namespace only sees its own endpoint
    results_ns1 = _endpoints(state, EndpointQuery(exact_name="ns-1/actor"))
    assert len(results_ns1) == 1
    assert results_ns1[0].address == "10.0.0.1:8080"

    results_ns2 = _endpoints(state, EndpointQuery(exact_name="ns-2/actor"))
    assert len(results_ns2) == 1
    assert results_ns2[0].address == "10.0.0.2:8080"


# =============================================================================
# Queue and Worker State Tests
# =============================================================================


def test_task_queue_fifo_order(state):
    """Tasks are returned in FIFO order."""

    req1 = make_job_request("job1")
    req2 = make_job_request("job2")
    submit_job(state, "j1", req1)
    submit_job(state, "j2", req2)

    pending = _schedulable_tasks(state)
    assert len(pending) == 2
    assert pending[0].job_id == JobName.root("test-user", "j1")
    assert pending[1].job_id == JobName.root("test-user", "j2")


def test_hierarchical_job_tracking(state):
    """Parent-child job relationships are tracked correctly."""

    parent_req = make_job_request("parent")
    submit_job(state, "parent", parent_req)

    child1_req = make_job_request("child1")
    submit_job(state, "/test-user/parent/child1", child1_req)

    child2_req = make_job_request("child2")
    submit_job(state, "/test-user/parent/child2", child2_req)

    grandchild_req = make_job_request("grandchild")
    submit_job(state, "/test-user/parent/child1/grandchild", grandchild_req)

    # get_children only returns direct children
    parent_id = JobName.root("test-user", "parent")
    with state._db.read_snapshot() as tx:
        children = tx.execute(select(jobs_table.c.job_id).where(jobs_table.c.parent_job_id == parent_id)).all()
    assert len(children) == 2
    assert {c.job_id for c in children} == {
        JobName.from_string("/test-user/parent/child1"),
        JobName.from_string("/test-user/parent/child2"),
    }

    # No children for leaf nodes
    grandchild_id = JobName.from_string("/test-user/parent/child1/grandchild")
    with state._db.read_snapshot() as tx:
        leaf_children = tx.execute(select(jobs_table.c.job_id).where(jobs_table.c.parent_job_id == grandchild_id)).all()
    assert leaf_children == []


def test_thread_safety(state):
    """Concurrent access doesn't corrupt state."""
    num_threads = 10
    jobs_per_thread = 50
    barrier = threading.Barrier(num_threads)
    errors = []

    def add_jobs(thread_id: int):
        try:
            barrier.wait()
            for i in range(jobs_per_thread):
                job_id = f"t{thread_id}_j{i}"
                req = make_job_request(f"job-{job_id}")
                submit_job(state, job_id, req)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=add_jobs, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    expected_count = num_threads * jobs_per_thread
    pending = _schedulable_tasks(state)
    assert len(pending) == expected_count


# =============================================================================
# Validation Tests
# =============================================================================


def test_excessive_replicas_fails_job(state):
    """E2E: Job with replicas exceeding MAX_REPLICAS_PER_JOB fails immediately."""

    req = make_job_request("too-many-replicas")
    req.replicas = MAX_REPLICAS_PER_JOB + 1

    tasks = submit_job(state, "j1", req)
    job = _query_job(state, JobName.root("test-user", "j1"))

    assert job is not None
    assert _query_job(state, job.job_id).state == job_pb2.JOB_STATE_FAILED
    assert f"exceeds max {MAX_REPLICAS_PER_JOB}" in _query_job(state, job.job_id).error
    assert len(tasks) == 0
    assert len(_schedulable_tasks(state)) == 0


# =============================================================================
# Worker Resource Commitment Tests
# =============================================================================


def test_worker_cannot_accept_task_when_resources_committed(state):
    """E2E: A worker with committed resources cannot accept tasks that exceed remaining capacity."""

    # Worker with 4 CPUs
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata(cpu=4))

    # First job uses 3 CPUs
    tasks1 = submit_job(state, "j1", make_job_request(cpu=3))
    dispatch_task(state, tasks1[0], worker_id)

    # Second job needs 2 CPUs - should not fit (only 1 CPU remaining)
    submit_job(state, "j2", make_job_request(cpu=2))

    # Scheduler should not assign the second task to this worker
    pending = _schedulable_tasks(state)
    assert len(pending) == 1  # j2's task is still pending

    scheduler = Scheduler()
    context = _build_scheduling_context(scheduler, state)
    result = scheduler.find_assignments(context)

    # The task cannot be scheduled - no worker has sufficient capacity
    assert len(result.assignments) == 0
    assert pending[0].job_id == JobName.root("test-user", "j2")


def test_worker_can_accept_new_task_after_previous_completes(state):
    """E2E: After a task completes, its resources are freed and new tasks can be scheduled.

    This verifies that task completion releases committed resources back to the worker.
    """

    # Worker with 4 CPUs
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata(cpu=4))

    # First job uses 3 CPUs
    tasks1 = submit_job(state, "j1", make_job_request(cpu=3))
    dispatch_task(state, tasks1[0], worker_id)

    # Second job needs 3 CPUs - cannot fit while first is running
    submit_job(state, "j2", make_job_request(cpu=3))

    scheduler = Scheduler()

    # Verify second task cannot be scheduled yet
    context = _build_scheduling_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 0

    # Complete the first task
    transition_task(state, tasks1[0].task_id, job_pb2.TASK_STATE_SUCCEEDED)

    # Now the second task can be scheduled
    context = _build_scheduling_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 1
    assert result.assignments[0][0].parent == JobName.root("test-user", "j2")


def test_multiple_small_tasks_fill_worker_capacity(state):
    """E2E: cumulative resource usage across tasks fills a worker and blocks the rest.

    Verifies the scheduler tracks committed CPU across tasks: a 4-CPU worker takes
    two 2-CPU tasks (capacity-limited, within the per-cycle assignment cap), and the
    third does not fit and stays pending.
    """

    # Worker with 4 CPUs
    register_worker(state, "w1", "host:8080", make_worker_metadata(cpu=4))

    # Submit 3 jobs, each using 2 CPUs
    for i in range(3):
        submit_job(state, f"j{i}", make_job_request(cpu=2))

    scheduler = Scheduler()

    # One cycle fills the worker to capacity: two 2-CPU tasks on 4 CPUs.
    context = _build_scheduling_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 2
    for task_id, worker_id in result.assignments:
        task = _query_task(state, task_id)
        dispatch_task(state, task, worker_id)

    # Third task remains pending: the worker is out of CPU.
    pending = _schedulable_tasks(state)
    assert len(pending) == 1
    assert pending[0].job_id == JobName.root("test-user", "j2")

    # Scheduler should not assign the third task (no capacity - 4 CPUs used)
    context = _build_scheduling_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 0


# =============================================================================
# Coscheduled Failure Cascade Tests
# =============================================================================


def test_coscheduled_task_failure_kills_siblings(state):
    """When one coscheduled task fails terminally, all running siblings are killed."""

    # Register 4 workers (one per task)
    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    # Create coscheduled job with 4 tasks
    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j1", req)

    job = _query_job(state, JobName.root("test-user", "j1"))
    assert job.has_coscheduling

    # Dispatch all tasks
    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Fail task-0 (terminal failure with no retries)
    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_FAILED, error="OOM")

    # Task-0 should be FAILED. Siblings cascade to COSCHED_FAILED — a
    # dedicated terminal state so we don't have to lie about
    # ``preemption_count`` (the historical "+1 tombstone" pattern), and so
    # dashboards can distinguish cascade kills from operator-initiated
    # terminations.
    assert _query_task(state, tasks[0].task_id).state == job_pb2.TASK_STATE_FAILED
    for task in tasks[1:]:
        sib = _query_task(state, task.task_id)
        assert sib.state == job_pb2.TASK_STATE_COSCHED_FAILED
        assert sib.preemption_count == 0, "sibling counter must stay honest"
        assert check_task_is_finished(sib), "COSCHED_FAILED must be unconditionally terminal"


def test_coscheduled_cascade_ignores_late_sibling_heartbeats(state):
    """A late heartbeat for a cascade-killed sibling does not resurrect it.

    The worker's container is still up until the next poll diff stops it, so
    heartbeats for the sibling task may arrive after the cascade has marked
    it COSCHED_FAILED. The heartbeat update loop short-circuits via
    ``task_is_finished``; without an unconditionally-terminal sibling
    state, a stale ``WORKER_FAILED`` heartbeat with retry budget remaining
    would flip the sibling back to PENDING and undo the cascade.
    """
    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    req = controller_pb2.Controller.LaunchJobRequest(
        name="late-heartbeat",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
        max_retries_preemption=100,  # plenty of budget — the tombstone, not the budget, must guard the cascade
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j-late", req)
    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_FAILED, error="OOM")
    for task in tasks[1:]:
        assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_COSCHED_FAILED

    # Late heartbeat for a sibling — the worker hasn't been diffed off yet
    # and reports its container is still WORKER_FAILED'ing into the void.
    transition_task(state, tasks[1].task_id, job_pb2.TASK_STATE_WORKER_FAILED, error="stale heartbeat")

    sib = _query_task(state, tasks[1].task_id)
    assert sib.state == job_pb2.TASK_STATE_COSCHED_FAILED, "late heartbeat must not un-kill the sibling"
    assert sib.preemption_count == 0, "no spurious budget consumption on a terminal task"


def test_coscheduled_cascade_survives_same_batch_sibling_update(state):
    """A sibling's own update in the SAME batch must not undo the cascade.

    Regression for the batch-overlay staleness gap: when the cascade that
    terminates a coscheduled sibling and that sibling's own heartbeat land in
    one ``apply_task_observations`` batch, the per-update transition core must read the
    sibling's prospective (overlay) state, not the stale snapshot row. With
    retry budget remaining, processing the sibling from the stale RUNNING row
    would flip its COSCHED_FAILED tombstone back to PENDING and resurrect it.
    """
    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    req = controller_pb2.Controller.LaunchJobRequest(
        name="same-batch-cascade",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
        max_retries_preemption=100,  # plenty of budget — only the tombstone may guard the sibling
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j-same-batch", req)
    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    trigger = _query_task(state, tasks[0].task_id)
    sibling = _query_task(state, tasks[1].task_id)

    # One batch: task-0 FAILED (fires the sibling cascade) AND task-1 reports
    # WORKER_FAILED. The cascade marks task-1 COSCHED_FAILED before task-1's own
    # update is processed; the overlay-aware guard must drop the late update.
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=WorkerId("w0"),
                    updates=[
                        TaskUpdate(
                            task_id=tasks[0].task_id,
                            attempt_id=trigger.current_attempt_id,
                            new_state=job_pb2.TASK_STATE_FAILED,
                            error="OOM",
                        )
                    ],
                ),
                WorkerTaskUpdates(
                    worker_id=WorkerId("w1"),
                    updates=[
                        TaskUpdate(
                            task_id=tasks[1].task_id,
                            attempt_id=sibling.current_attempt_id,
                            new_state=job_pb2.TASK_STATE_WORKER_FAILED,
                            error="stale same-batch heartbeat",
                        )
                    ],
                ),
            ],
            health=state._health,
            now=Timestamp.now(),
        )

    sib = _query_task(state, tasks[1].task_id)
    assert sib.state == job_pb2.TASK_STATE_COSCHED_FAILED, "same-batch sibling update must not un-kill the cascade"
    assert sib.preemption_count == 0, "no spurious budget consumption on a tombstoned sibling"


def test_worker_failures_batch_does_not_double_process_cascaded_sibling(state):
    """Two coscheduled workers failing in one batch must not re-finalize a sibling.

    Regression for the worker-failures overlay-staleness gap: when both workers
    holding a coscheduled slice fail in the same ``ops.worker.fail`` batch, the
    first failure's peer cascade marks the second worker's task COSCHED_FAILED in
    the overlay. The loop must re-read overlay state per row and skip the now-
    tombstoned sibling instead of overwriting it with WORKER_FAILED from the
    stale snapshot row.
    """
    for i in range(2):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    req = controller_pb2.Controller.LaunchJobRequest(
        name="batch-worker-fail",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=2,
        environment=job_pb2.EnvironmentConfig(),
        max_retries_preemption=0,  # worker failure is immediately terminal → cascade fires
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j-batch-fail", req)
    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Fail both slice workers in one batch (single chunk, single kernel pass).
    ops.worker.fail(
        state._db,
        worker_ids=["w0", "w1"],
        reason="slice reaped",
        health=state._health,
    )

    task0 = _query_task(state, tasks[0].task_id)
    task1 = _query_task(state, tasks[1].task_id)
    # Exactly one task is the direct WORKER_FAILED victim; the cascaded sibling
    # is COSCHED_FAILED, not double-processed into WORKER_FAILED.
    states = {task0.state, task1.state}
    assert states == {
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_COSCHED_FAILED,
    }, f"expected one WORKER_FAILED + one COSCHED_FAILED, got {states}"


@pytest.mark.parametrize("fail_both", [False, True])
def test_worker_failure_drives_coscheduled_job_terminal(state, fail_both):
    """A coscheduled worker failure must drive the JOB terminal once every task
    is terminal — not leave it stranded RUNNING.

    Regression for the recompute-before-cascade ordering in
    ``ReconcileState.fail_workers``: the per-task job recompute must observe the
    cascaded COSCHED_FAILED siblings. Before the fix it recomputed while the
    siblings were still active (job stayed RUNNING) and never recomputed after
    the cascade, so a job whose direct victim exhausted its retry budget was
    left RUNNING with all tasks terminal. Covers both the common single-worker
    slice reap and the both-workers-in-one-batch case.
    """
    for i in range(2):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    req = controller_pb2.Controller.LaunchJobRequest(
        name="cosched-worker-fail",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=2,
        environment=job_pb2.EnvironmentConfig(),
        max_retries_preemption=0,  # the direct victim is immediately terminal
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j-cosched-fail", req)
    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    failed = ["w0", "w1"] if fail_both else ["w0"]
    ops.worker.fail(
        state._db,
        worker_ids=failed,
        reason="slice reaped",
        health=state._health,
    )

    task0 = _query_task(state, tasks[0].task_id)
    task1 = _query_task(state, tasks[1].task_id)
    assert {task0.state, task1.state} == {
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_COSCHED_FAILED,
    }, f"expected WORKER_FAILED + COSCHED_FAILED, got {task0.state}, {task1.state}"

    job_id, _ = tasks[0].task_id.require_task()
    job = _query_job(state, job_id)
    assert (
        job.state == job_pb2.JOB_STATE_WORKER_FAILED
    ), f"job stranded in state {job.state}; expected terminal WORKER_FAILED"


def test_coscheduled_cascade_holds_sibling_resources_until_heartbeat(state):
    """Coscheduled sibling cascade keeps siblings' chips reserved until their
    heartbeats finalize them.

    Under the new derived-usage contract, ``_terminate_coscheduled_siblings``
    runs as a producer (``stamp_attempt_finished=False``) — it transitions the
    siblings' tasks to COSCHED_FAILED but leaves their attempts unfinished so
    the worker's chips stay accounted for. Only the originating task's worker
    (w0) gets capacity back, because that release came from the heartbeat
    that delivered the FAILED state in the first place.
    """

    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    req = controller_pb2.Controller.LaunchJobRequest(
        name="leak-test",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=2000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j-leak", req)

    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Verify resources are committed before failure
    for i in range(4):
        usage = _usage_for_worker(state, WorkerId(f"w{i}"))
        assert usage.cpu_millicores == 2000
        assert len(worker_running_tasks(state, WorkerId(f"w{i}"))) == 1

    # Fail task-0 terminally → cascade marks siblings COSCHED_FAILED but leaves
    # their attempts unfinished (producer path, no finished_at_ms stamp).
    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_FAILED, error="OOM")

    # The trigger worker's attempt is finalized by the heartbeat that
    # delivered FAILED.
    assert _usage_for_worker(state, WorkerId("w0")).cpu_millicores == 0

    # Sibling workers still hold their chips until their own terminal
    # heartbeats arrive (or worker-failure synthesis stamps finished_at_ms).
    for i in range(1, 4):
        usage = _usage_for_worker(state, WorkerId(f"w{i}"))
        assert (
            usage.cpu_millicores == 2000
        ), f"w{i} sibling must keep its reservation until terminal heartbeat: usage={usage}"
        assert usage.memory_bytes == 1024**3


def test_coscheduled_task_worker_failure_kills_siblings(state):
    """WORKER_FAILED triggers sibling kill when retries exhausted; bounces them
    to PENDING when retries remain."""

    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    # Use max_retries_preemption=1 so second worker failure is terminal.
    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
        max_retries_preemption=1,  # Allow one retry, so second failure is terminal
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j1", req)

    # Dispatch all tasks
    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # First WORKER_FAILED is retriable (retries remaining). Task-0 returns to
    # PENDING and its slice siblings get bounced too so the job can re-cosched.
    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_WORKER_FAILED, error="Worker crashed (first)")

    assert _query_task(state, tasks[0].task_id).preemption_count == 1
    assert check_task_can_be_scheduled(_query_task(state, tasks[0].task_id))
    # Siblings bounced to PENDING with their preemption budget untouched.
    for task in tasks[1:]:
        sib = _query_task(state, task.task_id)
        assert sib.state == job_pb2.TASK_STATE_PENDING
        assert sib.preemption_count == 0

    # Re-dispatch the whole slice atomically (mimicking re-coscheduling).
    for i, task in enumerate(tasks):
        refreshed = _query_task(state, task.task_id)
        if refreshed.state == job_pb2.TASK_STATE_PENDING:
            dispatch_task(state, refreshed, WorkerId(f"w{i}"))

    # Second WORKER_FAILED on task-0 exhausts retries → terminal; siblings die.
    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_WORKER_FAILED, error="Worker crashed (second)")

    assert _query_task(state, tasks[0].task_id).state == job_pb2.TASK_STATE_WORKER_FAILED
    assert check_task_is_finished(_query_task(state, tasks[0].task_id))
    for task in tasks[1:]:
        sib = _query_task(state, task.task_id)
        assert sib.state == job_pb2.TASK_STATE_COSCHED_FAILED
        assert sib.preemption_count == 0, "sibling counter must stay honest"


def test_coscheduled_task_success_does_not_affect_siblings(state):
    """Task success does NOT kill siblings."""

    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j1", req)

    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Task-0 succeeds
    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_SUCCEEDED)

    # Task-0 succeeded, siblings still running
    assert _query_task(state, tasks[0].task_id).state == job_pb2.TASK_STATE_SUCCEEDED
    for task in tasks[1:]:
        assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_RUNNING


def test_non_coscheduled_task_failure_does_not_kill_siblings(state):
    """Regular jobs don't cascade failures to siblings."""

    for i in range(4):
        register_worker(state, f"w{i}", f"addr{i}:8080", make_worker_metadata())

    # Regular job (no coscheduling)
    req = controller_pb2.Controller.LaunchJobRequest(
        name="regular-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
        max_task_failures=3,  # Allow failures without killing the job
    )
    tasks = submit_job(state, "j1", req)

    job = _query_job(state, JobName.root("test-user", "j1"))
    assert not job.has_coscheduling

    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Fail task-0
    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_FAILED, error="OOM")

    # Task-0 failed, but siblings are still running (no cascade)
    assert _query_task(state, tasks[0].task_id).state == job_pb2.TASK_STATE_FAILED
    for task in tasks[1:]:
        assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_RUNNING


def test_coscheduled_retriable_failure_bounces_siblings_to_pending(state):
    """A retriable failure of one coscheduled task bounces all siblings to
    PENDING so the job re-coschedules atomically. Only the preempted task
    uses its retry budget; sibling preemption and failure counts are
    unchanged."""

    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
        max_retries_failure=1,  # Allow one retry
        max_task_failures=4,  # Don't fail job on task failure
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j1", req)

    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Capture the siblings' worker-bound attempts before the failure so we can
    # assert the requeue leaves them unfinished (holding their slice).
    sibling_attempt_ids = {t.task_id: _query_task(state, t.task_id).current_attempt_id for t in tasks[1:]}

    # Fail task-0 (first failure, has retry remaining)
    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_FAILED, error="OOM")

    # Task-0 retried to PENDING and bears the failure_count.
    failed = _query_task(state, tasks[0].task_id)
    assert failed.state == job_pb2.TASK_STATE_PENDING
    assert failed.failure_count == 1
    assert check_task_can_be_scheduled(failed)
    assert not check_task_is_finished(failed)

    # Siblings bounced to PENDING with their counters preserved, so they don't
    # forfeit retries for someone else's failure.
    for task in tasks[1:]:
        sib = _query_task(state, task.task_id)
        assert sib.state == job_pb2.TASK_STATE_PENDING
        assert sib.failure_count == 0
        assert sib.preemption_count == 0
        # Producer-side requeue: the sibling's old attempt stays unfinished and
        # worker-bound, holding the slice. The reconcile planner sends a 'stop'
        # for it (PENDING + terminal-attempt branch) so the worker's terminal
        # observation later stamps finished_at_ms and releases capacity.
        old = _query_attempt(state, task.task_id, sibling_attempt_ids[task.task_id])
        assert old.finished_at_ms is None

    # Under the new derived-usage contract, sibling resources stay held by
    # their unfinished attempts until heartbeats arrive. The producer-side
    # requeue path only flips task.state to PENDING; capacity is released by
    # the heartbeat that subsequently reports KILLED.
    for i in range(1, 4):
        usage = _usage_for_worker(state, WorkerId(f"w{i}"))
        assert usage.cpu_millicores == 1000


def test_coscheduled_worker_failure_bounces_siblings(state):
    """Reaper-driven worker death (fail_workers path) must also clear siblings
    on surviving slice workers so the bounced task doesn't end up on a
    different slice from the rest of its job."""

    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
        max_retries_preemption=2,
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j-w", req)

    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Capture the siblings' worker-bound attempts before the failure so we can
    # assert the requeue leaves them unfinished (holding their slice).
    sibling_attempt_ids = {t.task_id: _query_task(state, t.task_id).current_attempt_id for t in tasks[1:]}

    fail_worker(state, WorkerId("w0"), "host disappeared")

    # Task-0 retried to PENDING with one preemption charge; siblings bounced
    # to PENDING but kept their full preemption budget.
    failed = _query_task(state, tasks[0].task_id)
    assert failed.state == job_pb2.TASK_STATE_PENDING
    assert failed.preemption_count == 1
    for task in tasks[1:]:
        sib = _query_task(state, task.task_id)
        assert sib.state == job_pb2.TASK_STATE_PENDING
        assert sib.preemption_count == 0
        # The requeue is producer-side: it flips the sibling task to PENDING but
        # leaves its old attempt unfinished and worker-bound, so the slice stays
        # reserved. The reconcile planner then sends a 'stop' for that terminal
        # attempt (PENDING + terminal-attempt branch), and the worker's resulting
        # terminal observation stamps finished_at_ms and releases the slot.
        old = _query_attempt(state, task.task_id, sibling_attempt_ids[task.task_id])
        assert old.finished_at_ms is None

    # Surviving workers' chips remain held by unfinished sibling attempts
    # under the new derived-usage contract. They are released when each
    # sibling's terminal heartbeat (or worker-failure synthesis) finalizes
    # the attempt. The bounced sibling tasks are already back to PENDING so
    # the job can re-coschedule once those heartbeats land.
    for i in range(1, 4):
        usage = _usage_for_worker(state, WorkerId(f"w{i}"))
        assert usage.cpu_millicores == 1000


def test_coscheduled_terminal_preempt_cascades_siblings(state):
    """A *terminal* preemption (no retry budget) of one coscheduled task must
    drive every sibling terminal, just like the retry case bounces them.

    The preempt path routes through the shared peer cascade, which terminates
    siblings when the trigger lands on a FAILURE-class state and only requeues
    them on a PENDING rollback. The terminal branch previously skipped the
    cascade outright, leaving siblings RUNNING on their slice — the split-slice
    precondition the ``test_no_split_coscheduled_active_tasks`` invariant flags.
    """

    for i in range(4):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-preempt-terminal",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
        max_retries_preemption=0,  # no budget -> preempt is terminal
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j-preempt", req)

    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    with state._db.transaction() as cur:
        finalize(
            cur,
            [TerminalDecision(TerminalKind.PREEMPT, tasks[0].task_id, "reclaim")],
            now=Timestamp.now(),
        )

    # Direct victim is terminally preempted...
    assert _query_task(state, tasks[0].task_id).state == job_pb2.TASK_STATE_PREEMPTED
    # ...and every coscheduled sibling cascades terminal, so no task is left
    # active to be re-placed onto a fresh slice.
    for task in tasks[1:]:
        sib = _query_task(state, task.task_id)
        assert sib.state == job_pb2.TASK_STATE_COSCHED_FAILED
        assert sib.preemption_count == 0, "sibling counter must stay honest"


def test_coscheduled_bounced_job_recoschedules_to_single_slice(state):
    """End-to-end: after a transient failure bounces a coscheduled slice,
    the next scheduling pass must place all tasks on a single tpu-name
    group, not split across the freed slice and a parallel one."""

    # Two slices: tpu-a (workers 0-3) and tpu-b (workers 4-7).
    for i in range(8):
        meta = make_worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a" if i < 4 else "tpu-b"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i % 4
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    req = controller_pb2.Controller.LaunchJobRequest(
        name="recosched",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=job_pb2.EnvironmentConfig(),
        max_retries_failure=1,
        max_task_failures=4,
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j-rc", req)

    # Initial dispatch: place all 4 tasks on tpu-a (workers 0-3).
    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Transient failure on task-0 bounces the whole slice to PENDING.
    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_FAILED, error="transient")

    # All four tasks are now PENDING and the scheduler must re-place them on
    # ONE group, even though tpu-b has 4 idle workers and tpu-a has 4 freshly
    # freed ones (either group is valid; what matters is that the assignment
    # is single-slice).
    scheduler = Scheduler()
    ctx = _build_scheduling_context(scheduler, state)
    result = scheduler.find_assignments(ctx)

    assignments = {tid: wid for tid, wid in result.assignments}
    assert set(assignments.keys()) == {
        t.task_id for t in tasks
    }, f"expected all 4 tasks scheduled, got {assignments.keys()}"
    chosen_tpu_names = {
        ctx.capacities[wid].attributes[WellKnownAttribute.TPU_NAME].value for wid in assignments.values()
    }
    assert len(chosen_tpu_names) == 1, f"job split across slices: {chosen_tpu_names}"


# =============================================================================
# compute_demand_entries Tests
# =============================================================================


# =============================================================================
# Stale Attempt Tracking Tests
# =============================================================================


def test_stale_attempt_ignored(state):
    """Stale attempt report does not change task state."""
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = make_job_request("job1")
    req.max_retries_preemption = 2
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    # First attempt: dispatch, then fail via worker failure (retriable)
    dispatch_task(state, task, worker_id)
    old_attempt_id = _query_task(state, task.task_id).current_attempt_id
    assert old_attempt_id == 0

    transition_task(state, task.task_id, job_pb2.TASK_STATE_WORKER_FAILED, error="Worker died")

    # Second attempt
    dispatch_task(state, task, worker_id)
    assert _query_task(state, task.task_id).current_attempt_id == 1
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_RUNNING

    # Stale report from old attempt should be ignored
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=worker_id,
                    updates=[
                        TaskUpdate(
                            task_id=task.task_id,
                            attempt_id=old_attempt_id,
                            new_state=job_pb2.TASK_STATE_SUCCEEDED,
                        )
                    ],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )

    # Task should still be RUNNING on the new attempt
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_RUNNING
    assert _query_task(state, task.task_id).current_attempt_id == 1


def test_stale_attempt_for_non_terminal_is_dropped(state):
    """A stale attempt report for a non-terminal old attempt is dropped, not applied."""
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = make_job_request("job1")
    req.max_retries_preemption = 2
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    # First attempt
    dispatch_task(state, task, worker_id)

    # Manually create a second attempt without properly terminating the first.
    # This simulates a scenario where the controller created a new attempt
    # but the old one is still non-terminal (a precondition violation).
    create_attempt_for_test(state, task.task_id, worker_id)
    assert _query_task(state, task.task_id).current_attempt_id == 1
    # The old attempt (0) is still in RUNNING state (non-terminal)
    with state._db.read_snapshot() as tx:
        attempts = tx.execute(
            select(
                task_attempts_table.c.task_id,
                task_attempts_table.c.attempt_id,
                task_attempts_table.c.worker_id,
                task_attempts_table.c.state,
                task_attempts_table.c.created_at_ms,
                task_attempts_table.c.started_at_ms,
                task_attempts_table.c.finished_at_ms,
                task_attempts_table.c.exit_code,
                task_attempts_table.c.error,
            )
            .where(task_attempts_table.c.task_id == task.task_id)
            .order_by(task_attempts_table.c.attempt_id.asc())
        ).all()
    assert attempts[0].state not in TERMINAL_TASK_STATES

    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=worker_id,
                    updates=[TaskUpdate(task_id=task.task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_SUCCEEDED)],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )

    # The stale update is dropped: attempt 0 is not revived to SUCCEEDED and the
    # current-attempt pointer is untouched.
    assert _query_task(state, task.task_id).current_attempt_id == 1
    stale = _query_attempt(state, task.task_id, 0)
    assert stale.state != job_pb2.TASK_STATE_SUCCEEDED
    assert stale.finished_at_ms is None


# =============================================================================
# Heartbeat Log Forwarding Tests
# =============================================================================


def test_log_service_direct_push(state, log_service):
    """Log entries pushed via LogService are queryable."""
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    tasks = submit_job(state, "j1", make_job_request("job1"))
    task = tasks[0]
    dispatch_task(state, task, worker_id)

    attempt_id = _query_task(state, task.task_id).current_attempt_id
    log_key = task_log_key(TaskAttempt(task_id=task.task_id, attempt_id=attempt_id))

    # Simulate push-based log delivery (worker pushes via LogService)
    log_entry = logging_pb2.LogEntry(source="stdout", data="hello world")
    log_entry.timestamp.epoch_ms = 1000
    push_req = logging_pb2.PushLogsRequest(key=log_key, entries=[log_entry])
    log_service.push_logs(push_req)

    fetch_resp = log_service.fetch_logs(logging_pb2.FetchLogsRequest(source=log_key))
    assert len(fetch_resp.entries) == 1
    assert fetch_resp.entries[0].data == "hello world"


def test_log_service_accumulates_pushes(state, log_service):
    """Multiple pushes accumulate logs in the service."""
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    tasks = submit_job(state, "j1", make_job_request("job1"))
    task = tasks[0]
    dispatch_task(state, task, worker_id)

    attempt_id = _query_task(state, task.task_id).current_attempt_id
    log_key = task_log_key(TaskAttempt(task_id=task.task_id, attempt_id=attempt_id))

    for i in range(3):
        entry = logging_pb2.LogEntry(source="stdout", data=f"line {i}")
        entry.timestamp.epoch_ms = 1000 + i
        log_service.push_logs(logging_pb2.PushLogsRequest(key=log_key, entries=[entry]))

    fetch_resp = log_service.fetch_logs(logging_pb2.FetchLogsRequest(source=log_key))
    assert len(fetch_resp.entries) == 3
    assert [e.data for e in fetch_resp.entries] == ["line 0", "line 1", "line 2"]


# =============================================================================
# compute_demand_entries Tests
# =============================================================================


def test_compute_demand_entries_counts_coscheduled_job_once(state):
    """Coscheduled job with 4 tasks should count as 1 slice demand, not 4."""
    req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5litepod-16")),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=4,
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    submit_job(state, "j1", req)

    demand = _demand_entries(state)
    assert len(demand) == 1
    assert demand[0].normalized.device_type == DeviceType.TPU
    assert demand[0].normalized.device_variants == frozenset({"v5litepod-16"})
    assert demand[0].task_ids == ("/test-user/j1/0", "/test-user/j1/1", "/test-user/j1/2", "/test-user/j1/3")
    assert demand[0].coschedule_group_id == "/test-user/j1"


def test_compute_demand_entries_counts_non_coscheduled_tasks_individually(state):
    """Non-coscheduled job with 4 tasks should count as 4 slices demand."""
    req = controller_pb2.Controller.LaunchJobRequest(
        name="regular-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5litepod-16")),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=4,
    )
    # No coscheduling set
    submit_job(state, "j1", req)

    demand = _demand_entries(state)
    assert len(demand) == 4
    for entry in demand:
        assert entry.normalized.device_type == DeviceType.TPU
        assert entry.normalized.device_variants == frozenset({"v5litepod-16"})
        assert entry.coschedule_group_id is None
        assert len(entry.task_ids) == 1


def test_compute_demand_entries_mixed_coscheduled_and_regular(state):
    """Mix of coscheduled and regular jobs should count correctly."""

    # Coscheduled job with 4 tasks -> 1 slice
    coscheduled_req = controller_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5litepod-16")),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=4,
    )
    coscheduled_req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    submit_job(state, "j1", coscheduled_req)

    # Regular job with 2 tasks -> 2 slices
    regular_req = controller_pb2.Controller.LaunchJobRequest(
        name="regular-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5litepod-16")),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=2,
    )
    submit_job(state, "j2", regular_req)

    demand = _demand_entries(state)
    assert len(demand) == 3
    coscheduled = [entry for entry in demand if entry.coschedule_group_id == "/test-user/j1"]
    regular = [entry for entry in demand if entry.coschedule_group_id is None]
    assert len(coscheduled) == 1
    assert len(regular) == 2
    assert coscheduled[0].task_ids == ("/test-user/j1/0", "/test-user/j1/1", "/test-user/j1/2", "/test-user/j1/3")
    for entry in regular:
        assert entry.normalized.device_type == DeviceType.TPU
        assert entry.normalized.device_variants == frozenset({"v5litepod-16"})


def test_compute_demand_entries_separates_by_preemptible_constraint(state):
    """Jobs with different preemptible constraints produce separate demand entries."""

    # Job requiring preemptible workers
    preemptible_req = controller_pb2.Controller.LaunchJobRequest(
        name="preemptible-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5p-8")),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
        constraints=[
            job_pb2.Constraint(
                key=WellKnownAttribute.PREEMPTIBLE,
                op=job_pb2.CONSTRAINT_OP_EQ,
                value=job_pb2.AttributeValue(string_value="true"),
            )
        ],
    )
    submit_job(state, "j1", preemptible_req)

    # Job requiring non-preemptible workers
    on_demand_req = controller_pb2.Controller.LaunchJobRequest(
        name="on-demand-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5p-8")),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
        constraints=[
            job_pb2.Constraint(
                key=WellKnownAttribute.PREEMPTIBLE,
                op=job_pb2.CONSTRAINT_OP_EQ,
                value=job_pb2.AttributeValue(string_value="false"),
            )
        ],
    )
    submit_job(state, "j2", on_demand_req)

    demand = _demand_entries(state)
    assert len(demand) == 2

    by_preemptible = {d.normalized.preemptible: d for d in demand}
    assert by_preemptible[True].normalized.device_type == DeviceType.TPU
    assert by_preemptible[True].task_ids == ("/test-user/j1/0",)
    assert by_preemptible[False].normalized.device_type == DeviceType.TPU
    assert by_preemptible[False].task_ids == ("/test-user/j2/0",)


def test_compute_demand_entries_no_preemptible_constraint_gives_none(state):
    """Job without preemptible constraint produces demand with preemptible=None."""

    req = controller_pb2.Controller.LaunchJobRequest(
        name="unconstrained-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5p-8")),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j1", req)

    demand = _demand_entries(state)
    assert len(demand) == 1
    assert demand[0].normalized.preemptible is None


def test_compute_demand_entries_extracts_required_region(state):
    req = controller_pb2.Controller.LaunchJobRequest(
        name="regional-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5p-8")),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
        constraints=[
            job_pb2.Constraint(
                key=WellKnownAttribute.REGION,
                op=job_pb2.CONSTRAINT_OP_EQ,
                value=job_pb2.AttributeValue(string_value="us-west4"),
            )
        ],
    )
    submit_job(state, "j1", req)

    demand = _demand_entries(state)
    assert len(demand) == 1
    assert demand[0].normalized.required_regions == frozenset({"us-west4"})
    assert demand[0].invalid_reason is None


def test_compute_demand_entries_marks_invalid_on_conflicting_region_constraints(state):
    req = controller_pb2.Controller.LaunchJobRequest(
        name="invalid-regional-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5p-8")),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
        constraints=[
            job_pb2.Constraint(
                key=WellKnownAttribute.REGION,
                op=job_pb2.CONSTRAINT_OP_EQ,
                value=job_pb2.AttributeValue(string_value="us-west4"),
            ),
            job_pb2.Constraint(
                key=WellKnownAttribute.REGION,
                op=job_pb2.CONSTRAINT_OP_EQ,
                value=job_pb2.AttributeValue(string_value="eu-west4"),
            ),
        ],
    )
    submit_job(state, "j1", req)

    demand = _demand_entries(state)
    assert len(demand) == 1
    assert demand[0].invalid_reason is not None


def _h100_device() -> job_pb2.DeviceConfig:
    return job_pb2.DeviceConfig(gpu=job_pb2.GpuDevice(variant="H100", count=8))


def _a100_device() -> job_pb2.DeviceConfig:
    return job_pb2.DeviceConfig(gpu=job_pb2.GpuDevice(variant="A100", count=8))


def test_demand_passes_all_tasks(state):
    """Job emits all task demand entries."""
    req = controller_pb2.Controller.LaunchJobRequest(
        name="regular-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=3,
    )
    submit_job(state, "j1", req)

    demand = _demand_entries(state)
    assert len(demand) == 3


# =============================================================================
# Depth-First Scheduling Priority Tests
# =============================================================================


def test_peek_pending_tasks_deeper_job_before_shallow(state):
    """Depth-first priority: deeper jobs come before shallow ones in queue order."""

    # Submit root job and child job (both with 1 CPU)
    submit_job(state, "root", make_job_request("root"), timestamp_ms=1000)
    submit_job(state, "/test-user/root/child", make_job_request("child"), timestamp_ms=2000)

    pending = _schedulable_tasks(state)
    assert len(pending) == 2
    # Child (depth 2) should come first
    assert pending[0].job_id == JobName.from_string("/test-user/root/child")
    assert pending[1].job_id == JobName.root("test-user", "root")


def test_peek_pending_tasks_older_root_tree_preferred(state):
    """At same depth, older root tree is preferred."""

    # Submit two root jobs at different timestamps
    req_a = make_job_request("user-a-job")
    submit_job(state, "user-a-job", req_a, timestamp_ms=1000)

    req_b = make_job_request("user-b-job")
    submit_job(state, "user-b-job", req_b, timestamp_ms=2000)

    pending = _schedulable_tasks(state)
    assert len(pending) == 2
    # user-a-job submitted first, should come first
    assert pending[0].job_id == JobName.root("test-user", "user-a-job")
    assert pending[1].job_id == JobName.root("test-user", "user-b-job")


def test_peek_pending_tasks_child_of_older_tree_beats_newer_root(state):
    """Child of older tree beats root of newer tree."""

    # Submit old tree
    submit_job(state, "old-tree", make_job_request("old-tree"), timestamp_ms=1000)

    # Submit new tree
    submit_job(state, "new-tree", make_job_request("new-tree"), timestamp_ms=2000)

    # Submit child of old tree (depth 2) after new tree
    submit_job(state, "/test-user/old-tree/child", make_job_request("child"), timestamp_ms=3000)

    pending = _schedulable_tasks(state)
    assert len(pending) == 3

    # Expected order: child (depth 2), old-tree (depth 1, older), new-tree (depth 1, newer)
    assert pending[0].job_id == JobName.from_string("/test-user/old-tree/child")
    assert pending[1].job_id == JobName.root("test-user", "old-tree")
    assert pending[2].job_id == JobName.root("test-user", "new-tree")


def test_peek_pending_tasks_fifo_within_same_depth_and_tree(state):
    """FIFO within same depth and tree."""

    # Submit parent first
    submit_job(state, "tree", make_job_request("tree"), timestamp_ms=1000)

    # Submit two children at different times
    submit_job(state, "/test-user/tree/child-a", make_job_request("child-a"), timestamp_ms=2000)
    submit_job(state, "/test-user/tree/child-b", make_job_request("child-b"), timestamp_ms=3000)

    pending = _schedulable_tasks(state)
    assert len(pending) == 3

    # Both children at depth 2, same root tree — child-a submitted first
    child_tasks = [t for t in pending if t.job_id.parent == JobName.root("test-user", "tree")]
    assert len(child_tasks) == 2
    assert child_tasks[0].job_id == JobName.from_string("/test-user/tree/child-a")
    assert child_tasks[1].job_id == JobName.from_string("/test-user/tree/child-b")


def test_child_job_inherits_root_submitted_at(state):
    """Child job inherits root_submitted_at from parent."""

    # Submit parent at known time
    parent_req = make_job_request("parent")
    submit_job(state, "parent", parent_req, timestamp_ms=1000)
    parent_job = _query_job(state, JobName.root("test-user", "parent"))
    parent_submitted = parent_job.submitted_at_ms

    # Submit child later
    child_req = make_job_request("child")
    submit_job(state, "/test-user/parent/child", child_req, timestamp_ms=2000)
    child_job = _query_job(state, JobName.from_string("/test-user/parent/child"))

    # Child's root_submitted_at should equal parent's
    assert child_job.root_submitted_at_ms == parent_submitted
    assert child_job.root_submitted_at_ms == parent_job.root_submitted_at_ms


def test_requeued_task_maintains_priority_position(state):
    """Requeued task maintains its priority position (deeper job still prioritized)."""

    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    # Submit a deep job (under an explicit parent tree) and a shallow job
    submit_job(state, "tree", make_job_request("tree"), timestamp_ms=500)
    submit_job(state, "/test-user/tree/deep", make_job_request("deep"), timestamp_ms=1000)
    submit_job(state, "shallow", make_job_request("shallow"), timestamp_ms=2000)

    # Initially: deep job comes first
    pending = _schedulable_tasks(state)
    assert len(pending) == 3
    assert pending[0].job_id == JobName.from_string("/test-user/tree/deep")
    assert pending[1].job_id == JobName.root("test-user", "tree")
    assert pending[2].job_id == JobName.root("test-user", "shallow")

    # Dispatch and fail the deep job's task (with retries enabled)
    deep_req = make_job_request("deep")
    deep_req.max_retries_failure = 1
    deep_req.max_task_failures = 1
    deep_tasks = submit_job(state, "/test-user/tree/deep-retry", deep_req, timestamp_ms=3000)
    submit_job(state, "shallow-2", make_job_request("shallow-2"), timestamp_ms=4000)

    dispatch_task(state, deep_tasks[0], worker_id)
    transition_task(state, deep_tasks[0].task_id, job_pb2.TASK_STATE_FAILED, error="Retriable failure")

    # Verify task was requeued
    assert deep_tasks[0].state == job_pb2.TASK_STATE_PENDING
    assert check_task_can_be_scheduled(deep_tasks[0])

    # Check queue order — requeued deep job should still come before shallow
    pending = _schedulable_tasks(state)
    deep_pending = [t for t in pending if t.job_id == JobName.from_string("/test-user/tree/deep-retry")]
    shallow_pending = [t for t in pending if t.job_id == JobName.root("test-user", "shallow-2")]

    assert len(deep_pending) == 1
    assert len(shallow_pending) == 1

    # Find indices
    deep_idx = pending.index(deep_pending[0])
    shallow_idx = pending.index(shallow_pending[0])
    assert deep_idx < shallow_idx, "Requeued deep task should still come before shallow task"


def test_worker_failed_from_assigned_is_delivery_failure(state):
    """WORKER_FAILED on a task still in ASSIGNED state is a delivery failure.

    When a task was assigned but never confirmed running (BUILDING/RUNNING),
    a WORKER_FAILED is a delivery failure — no budget is consumed. This
    prevents preemption count inflation from repeated 'Task not found' reports.
    """
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = make_job_request("job1")
    req.max_retries_preemption = 5
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    # Assign but do NOT transition to RUNNING
    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task.task_id, worker_id=worker_id)], health=state._health)
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_ASSIGNED

    # Worker reports WORKER_FAILED (e.g., "Task not found on worker")
    transition_task(
        state,
        task.task_id,
        job_pb2.TASK_STATE_WORKER_FAILED,
        error="Task not found on worker",
    )

    # Delivery failure: no budget consumed at all
    assert _query_task(state, task.task_id).preemption_count == 0
    assert _query_task(state, task.task_id).failure_count == 0
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_PENDING
    assert check_task_can_be_scheduled(_query_task(state, task.task_id))


def test_worker_failed_from_running_counts_as_preemption(state):
    """WORKER_FAILED on a task in RUNNING state counts as a preemption."""
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = make_job_request("job1")
    req.max_retries_preemption = 5
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    # Full lifecycle: assign and transition to RUNNING
    dispatch_task(state, task, worker_id)
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_RUNNING

    # Worker dies
    transition_task(
        state,
        task.task_id,
        job_pb2.TASK_STATE_WORKER_FAILED,
        error="Worker crashed",
    )

    # Real preemption: counts against preemption budget
    assert _query_task(state, task.task_id).preemption_count == 1
    assert _query_task(state, task.task_id).failure_count == 0
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_PENDING
    assert check_task_can_be_scheduled(_query_task(state, task.task_id))


def test_worker_failed_from_building_counts_as_preemption(state):
    """WORKER_FAILED on a task in BUILDING state counts as a preemption."""
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = make_job_request("job1")
    req.max_retries_preemption = 5
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    # Assign and transition to BUILDING (worker confirmed it received the task)
    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task.task_id, worker_id=worker_id)], health=state._health)
    transition_task(state, task.task_id, job_pb2.TASK_STATE_BUILDING)
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_BUILDING

    # Worker dies
    transition_task(
        state,
        task.task_id,
        job_pb2.TASK_STATE_WORKER_FAILED,
        error="Worker crashed",
    )

    # Real preemption: worker had started processing the task
    assert _query_task(state, task.task_id).preemption_count == 1
    assert _query_task(state, task.task_id).failure_count == 0


def test_worker_failed_from_assigned_bumps_health_tracker(state):
    """ASSIGNED -> WORKER_FAILED attributes the failure to the worker.

    Regression for the TPU-iommu co-schedule loop: the task retries to PENDING
    (no preemption-budget cost) but the health tracker must still bump so that
    a host that repeatedly fails launches eventually crosses the threshold and
    gets reaped.
    """
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())
    req = make_job_request("job1")
    req.max_retries_preemption = 5
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task.task_id, worker_id=worker_id)], health=state._health)
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_ASSIGNED
    # No build failures recorded yet (worker registered, but no failure events).
    assert state._health.snapshot().get(worker_id, (0, 0))[1] == 0

    transition_task(
        state,
        task.task_id,
        job_pb2.TASK_STATE_WORKER_FAILED,
        error='TPU init failure ("Couldn\'t open iommu group")',
    )

    # Task retries without consuming preemption budget...
    t = _query_task(state, task.task_id)
    assert t.state == job_pb2.TASK_STATE_PENDING
    assert t.preemption_count == 0
    # ...but the worker is charged a build failure.
    _, build_failures = state._health.snapshot()[worker_id]
    assert build_failures == 1


def test_failed_from_building_bumps_health_tracker(state):
    """FAILED originating from BUILDING increments the build failure counter.

    A task that never reaches RUNNING and then reports FAILED almost always
    reflects infrastructure trouble (image pull, disk, DNS) rather than user
    code. The tracker should record one build failure for that worker.
    """
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())
    req = make_job_request("job1", max_retries_failure=5)
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task.task_id, worker_id=worker_id)], health=state._health)
    transition_task(state, task.task_id, job_pb2.TASK_STATE_BUILDING)
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_BUILDING

    # No build failures recorded yet.
    assert state._health.snapshot().get(worker_id, (0, 0))[1] == 0

    transition_task(
        state,
        task.task_id,
        job_pb2.TASK_STATE_FAILED,
        error="image pull failed",
    )

    assert _query_task(state, task.task_id).failure_count == 1
    _, build_failures = state._health.snapshot()[worker_id]
    assert build_failures == 1


def test_worker_failed_from_building_bumps_health_tracker(state):
    """WORKER_FAILED originating from BUILDING charges the build-failure reaper.

    A bad-TPU/infra-missing host commonly fails the build phase via
    WORKER_FAILED-from-BUILDING (the worker announces BUILDING before the
    container runs). That host must be charged a build failure — same as the
    FAILED-from-BUILDING and WORKER_FAILED-from-ASSIGNED cases — so the reaper
    eventually trips instead of letting the host re-attract and poison jobs.
    """
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())
    req = make_job_request("job1")
    req.max_retries_preemption = 5
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task.task_id, worker_id=worker_id)], health=state._health)
    transition_task(state, task.task_id, job_pb2.TASK_STATE_BUILDING)
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_BUILDING
    # No build failures recorded yet.
    assert state._health.snapshot().get(worker_id, (0, 0))[1] == 0

    transition_task(
        state,
        task.task_id,
        job_pb2.TASK_STATE_WORKER_FAILED,
        error="TPU bad node during build",
    )

    # Charged the preemption budget (worker had started bringing the task up)...
    t = _query_task(state, task.task_id)
    assert t.preemption_count == 1
    # ...and the worker is charged a build failure, like the ASSIGNED/FAILED twins.
    _, build_failures = state._health.snapshot()[worker_id]
    assert build_failures == 1


def test_failed_from_running_does_not_bump_health_tracker(state):
    """FAILED from RUNNING is treated as user code and must NOT move the score."""
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())
    req = make_job_request("job1", max_retries_failure=5)
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    dispatch_task(state, task, worker_id)
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_RUNNING

    transition_task(
        state,
        task.task_id,
        job_pb2.TASK_STATE_FAILED,
        error="user code raised",
    )

    # FAILED from RUNNING must NOT bump the build failure counter.
    assert state._health.snapshot().get(worker_id, (0, 0))[1] == 0


def test_fail_workers_by_ids_cascades_tasks(state):
    """fail_workers_by_ids fails sibling workers and cascades their tasks."""

    meta1 = make_worker_metadata()
    w1 = register_worker(state, "w1", "host1:8080", meta1)

    meta2 = make_worker_metadata()
    w2 = register_worker(state, "w2", "host2:8080", meta2)

    tasks1 = submit_job(state, "j1", make_job_request("job1"))
    dispatch_task(state, tasks1[0], w1)

    tasks2 = submit_job(state, "j2", make_job_request("job2"))
    dispatch_task(state, tasks2[0], w2)

    assert _query_task(state, tasks1[0].task_id).state == job_pb2.TASK_STATE_RUNNING
    assert _query_task(state, tasks2[0].task_id).state == job_pb2.TASK_STATE_RUNNING

    result = ops.worker.fail(
        state._db,
        worker_ids=["w2"],
        reason="slice terminated",
        health=state._health,
    )

    assert len(result.removed_workers) == 1
    assert result.removed_workers[0][0] == w2
    assert result.removed_workers[0][1] == "host2:8080"

    t2 = _query_task(state, tasks2[0].task_id)
    assert t2.state in (job_pb2.TASK_STATE_WORKER_FAILED, job_pb2.TASK_STATE_PENDING)

    assert _query_task(state, tasks1[0].task_id).state == job_pb2.TASK_STATE_RUNNING
    assert _query_worker(state, w1) is not None
    assert _query_worker(state, w2) is None


def test_fail_workers_batch_skips_unknown(state):
    """ops.worker.fail returns empty for unknown worker IDs."""
    meta = make_worker_metadata()
    register_worker(state, "w1", "host1:8080", meta)

    result = ops.worker.fail(
        state._db,
        worker_ids=["w-unknown"],
        reason="unknown",
        health=state._health,
    )
    assert result.removed_workers == []

    w = _query_worker(state, WorkerId("w1"))
    assert w is not None
    assert w.healthy


def test_fail_workers_batch_force_removes_without_threshold(state):
    """ops.worker.fail removes targets immediately instead of incrementing failures."""
    meta = make_worker_metadata()
    worker_id = register_worker(state, "w1", "host1:8080", meta)

    result = ops.worker.fail(
        state._db,
        worker_ids=["w1"],
        reason="slice terminated",
        health=state._health,
    )

    assert len(result.removed_workers) == 1
    assert result.removed_workers[0][0] == worker_id
    assert _query_worker(state, worker_id) is None


def test_fail_workers_batch_does_not_block_readers(state):
    """ops.worker.fail uses read_snapshot for lookups, so concurrent reads don't block.

    Verifies that read_snapshot() (not write-locked snapshot()) is used for the
    worker lookup query. We hold a write transaction open on a second thread while
    calling ops.worker.fail from the main thread; if the lookup used
    snapshot() (write lock), it would deadlock/timeout.
    """
    meta = make_worker_metadata()
    w1 = register_worker(state, "w1", "host1:8080", meta)
    register_worker(state, "w2", "host2:8080", meta)

    tasks = submit_job(state, "j1", make_job_request("job1"))
    dispatch_task(state, tasks[0], w1)

    barrier = threading.Event()
    done = threading.Event()

    def hold_write_lock():
        """Hold the DB write lock to prove ops.worker.fail doesn't need it for reads."""
        with state._db.transaction():
            barrier.set()
            done.wait(timeout=5)

    t = threading.Thread(target=hold_write_lock, daemon=True)
    t.start()
    barrier.wait(timeout=5)

    # fail_workers should still complete even though the write lock is held,
    # because its lookup query uses read_snapshot (WAL reader).
    # The inner write transaction for actually failing workers still needs the
    # write lock, so we test with unknown IDs to isolate the read path.
    result = ops.worker.fail(
        state._db,
        worker_ids=["w-nonexistent"],
        reason="test",
        health=state._health,
    )
    assert result.removed_workers == []

    done.set()
    t.join(timeout=5)


# =============================================================================
# Demand Dry-Run Scheduling Tests
#
# These tests verify that compute_demand_entries runs a dry-run scheduling pass
# to absorb tasks into existing worker capacity, and only emits demand for
# truly unschedulable tasks (not building-limited ones).
# =============================================================================


def _gpu_make_worker_metadata(
    *,
    cpu: int = 128,
    memory_gb: int = 256,
    variant: str = "H100",
    gpu_count: int = 8,
) -> job_pb2.WorkerMetadata:
    """Create worker metadata for a GPU worker with scheduling attributes."""
    return job_pb2.WorkerMetadata(
        hostname="gpu-worker",
        ip_address="10.0.0.1",
        cpu_count=cpu,
        memory_bytes=memory_gb * 1024**3,
        disk_bytes=100 * 1024**3,
        device=job_pb2.DeviceConfig(
            gpu=job_pb2.GpuDevice(variant=variant, count=gpu_count),
        ),
        attributes={
            WellKnownAttribute.DEVICE_TYPE: job_pb2.AttributeValue(string_value="gpu"),
            WellKnownAttribute.DEVICE_VARIANT: job_pb2.AttributeValue(string_value=variant.lower()),
            WellKnownAttribute.PREEMPTIBLE: job_pb2.AttributeValue(string_value="false"),
        },
    )


def _cpu_make_worker_metadata(
    *,
    cpu: int = 128,
    memory_gb: int = 256,
) -> job_pb2.WorkerMetadata:
    return job_pb2.WorkerMetadata(
        hostname="cpu-worker",
        ip_address="10.0.0.1",
        cpu_count=cpu,
        memory_bytes=memory_gb * 1024**3,
        disk_bytes=100 * 1024**3,
        device=job_pb2.DeviceConfig(
            cpu=job_pb2.CpuDevice(variant="cpu"),
        ),
        attributes={
            WellKnownAttribute.DEVICE_TYPE: job_pb2.AttributeValue(string_value="cpu"),
            WellKnownAttribute.PREEMPTIBLE: job_pb2.AttributeValue(string_value="false"),
        },
    )


def test_demand_excludes_building_limited_tasks(state):
    """Worker has resources but is at building limit -> no demand emitted."""

    # Register a CPU worker with plenty of capacity
    wid = register_worker(state, "w1", "10.0.0.1:8080", _cpu_make_worker_metadata(cpu=128, memory_gb=256))

    # Submit a job with 1 pending CPU task
    req = controller_pb2.Controller.LaunchJobRequest(
        name="cpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j1", req)

    # Fill the worker with 2 building tasks (at the building limit).
    # These use minimal resources so the worker still has plenty of capacity.
    build_req = controller_pb2.Controller.LaunchJobRequest(
        name="build-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=100,
            memory_bytes=1024**2,
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=2,
    )
    build_tasks = submit_job(state, "build-job", build_req)
    for bt in build_tasks:
        dispatch_task(state, bt, wid)
        transition_task(state, bt.task_id, job_pb2.TASK_STATE_BUILDING)

    # Now w1 has 2 building tasks (at limit), but has plenty of CPU/memory.
    # The pending task from j1 should be building-limited, not truly unschedulable.
    demand = _demand_entries(state)
    task_demand = list(demand)
    assert len(task_demand) == 0, "Building-limited task should not generate demand"


def test_demand_includes_truly_unschedulable_tasks(state):
    """No worker with matching device type -> demand IS emitted."""

    # Register a CPU-only worker
    register_worker(state, "w1", "10.0.0.1:8080", _cpu_make_worker_metadata())

    # Submit a job requiring H100 GPUs
    req = controller_pb2.Controller.LaunchJobRequest(
        name="gpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j1", req)

    demand = _demand_entries(state)
    task_demand = list(demand)
    assert len(task_demand) == 1, "Task with no matching device should generate demand"


def test_demand_includes_resource_exhausted_tasks(state):
    """Worker has right device but insufficient CPU -> demand IS emitted."""

    # Register a GPU worker with only 1 CPU core
    register_worker(state, "w1", "10.0.0.1:8080", _gpu_make_worker_metadata(cpu=1))

    # Submit a job requiring 4 CPU cores
    req = controller_pb2.Controller.LaunchJobRequest(
        name="gpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=4000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j1", req)

    demand = _demand_entries(state)
    task_demand = list(demand)
    assert len(task_demand) == 1, "Task exceeding worker CPU should generate demand"


def test_demand_absorbs_capacity_before_emitting(state):
    """2 workers fit 1 task each, 3 pending tasks -> only 1 demand entry."""

    # Register 2 GPU workers, each with enough capacity for 1 task
    register_worker(state, "w1", "10.0.0.1:8080", _gpu_make_worker_metadata(cpu=2, memory_gb=4))
    register_worker(state, "w2", "10.0.0.2:8080", _gpu_make_worker_metadata(cpu=2, memory_gb=4))

    # Submit 3 tasks each needing 2 CPU cores (each worker fits exactly 1)
    req = controller_pb2.Controller.LaunchJobRequest(
        name="gpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=2000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=3,
    )
    submit_job(state, "j1", req)

    demand = _demand_entries(state)
    task_demand = list(demand)
    assert len(task_demand) == 1, "Only 1 of 3 tasks should generate demand (2 absorbed)"


def test_demand_no_workers_falls_back_to_all_pending(state):
    """When no workers provided, all pending tasks generate demand (backward compat)."""

    req = controller_pb2.Controller.LaunchJobRequest(
        name="gpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=3,
    )
    submit_job(state, "j1", req)

    # No scheduler, no workers -> all tasks become demand
    demand = _demand_entries(state)
    task_demand = list(demand)
    assert len(task_demand) == 3


def test_demand_building_limited_with_multiple_workers(state):
    """All matching workers at building limit -> no demand, even with multiple workers."""

    # Register 2 CPU workers
    wid1 = register_worker(state, "w1", "10.0.0.1:8080", _cpu_make_worker_metadata())
    wid2 = register_worker(state, "w2", "10.0.0.2:8080", _cpu_make_worker_metadata())

    # Fill both workers with 1 building task each (at limit since max=1).
    # Use minimal resources so workers retain plenty of capacity.
    for i, wid in enumerate([wid1, wid2]):
        build_req = controller_pb2.Controller.LaunchJobRequest(
            name=f"build-{i}",
            entrypoint=_make_test_entrypoint(),
            resources=job_pb2.ResourceSpecProto(
                cpu_millicores=100,
                memory_bytes=1024**2,
            ),
            environment=job_pb2.EnvironmentConfig(),
            replicas=1,
        )
        build_tasks = submit_job(state, f"build-{i}", build_req)
        dispatch_task(state, build_tasks[0], wid)
        transition_task(state, build_tasks[0].task_id, job_pb2.TASK_STATE_BUILDING)

    # Submit a new task
    req = controller_pb2.Controller.LaunchJobRequest(
        name="pending-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "pending-job", req)

    demand = _demand_entries(state)
    task_demand = list(demand)
    assert len(task_demand) == 0, "All workers at building limit -> no demand"


def test_demand_mixed_building_limited_and_unschedulable(state):
    """Some tasks building-limited, some truly unschedulable -> only unschedulable emit demand."""

    # Register 1 GPU worker at building limit.
    # Use a minimal CPU task to fill the building slot so GPU capacity stays intact.
    wid = register_worker(state, "w1", "10.0.0.1:8080", _gpu_make_worker_metadata())
    build_req = controller_pb2.Controller.LaunchJobRequest(
        name="build-0",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=100,
            memory_bytes=1024**2,
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    build_tasks = submit_job(state, "build-0", build_req)
    dispatch_task(state, build_tasks[0], wid)
    transition_task(state, build_tasks[0].task_id, job_pb2.TASK_STATE_BUILDING)

    # Task 1: H100 job (building-limited, worker has resources but at limit)
    h100_req = controller_pb2.Controller.LaunchJobRequest(
        name="h100-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "h100-job", h100_req)

    # Task 2: A100 job (truly unschedulable, no A100 workers exist)
    a100_req = controller_pb2.Controller.LaunchJobRequest(
        name="a100-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_a100_device(),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "a100-job", a100_req)

    demand = _demand_entries(state)
    task_demand = list(demand)

    assert len(task_demand) == 1
    assert "a100-job" in task_demand[0].task_ids[0], "Only A100 task should emit demand"


# =============================================================================
# Worker Death Cascade + Preemption Policy Tests
# =============================================================================


def test_worker_death_cascades_children_terminal(state):
    """Single-task parent exhausts preemption retries -> job terminal -> children killed."""
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    # Parent job with 0 preemption retries so worker death => WORKER_FAILED (terminal task)
    parent_req = make_job_request("parent")
    parent_req.max_retries_preemption = 0
    parent_req.max_task_failures = 0
    parent_tasks = submit_job(state, "parent", parent_req)
    dispatch_task(state, parent_tasks[0], worker_id)

    # Child job under parent
    child_req = make_job_request("child")
    child_tasks = submit_job(state, "/test-user/parent/child", child_req)

    # Register new worker for child and dispatch
    w2 = register_worker(state, "w2", "host2:8080", make_worker_metadata())
    dispatch_task(state, child_tasks[0], w2)
    assert _query_task(state, child_tasks[0].task_id).state == job_pb2.TASK_STATE_RUNNING

    # Worker w1 dies — parent task exhausts preemption retries
    fail_worker(state, worker_id, "Connection lost")

    # Parent task should be terminal (WORKER_FAILED)
    parent_task = _query_task(state, parent_tasks[0].task_id)
    assert parent_task.state == job_pb2.TASK_STATE_WORKER_FAILED

    # Child should be killed via cascade
    child_task = _query_task(state, child_tasks[0].task_id)
    assert child_task.state == job_pb2.TASK_STATE_KILLED

    child_job = _query_job(state, JobName.from_string("/test-user/parent/child"))
    assert child_job.state == job_pb2.JOB_STATE_KILLED


def test_worker_death_preemption_policy_terminate(state):
    """Single-task parent retried after worker death -> children killed (default TERMINATE)."""
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    # Parent with retries so task goes back to PENDING
    parent_req = make_job_request("parent")
    parent_req.max_retries_preemption = 5
    parent_tasks = submit_job(state, "parent", parent_req)
    dispatch_task(state, parent_tasks[0], worker_id)

    # Child job
    child_req = make_job_request("child")
    child_tasks = submit_job(state, "/test-user/parent/child", child_req)
    w2 = register_worker(state, "w2", "host2:8080", make_worker_metadata())
    dispatch_task(state, child_tasks[0], w2)
    assert _query_task(state, child_tasks[0].task_id).state == job_pb2.TASK_STATE_RUNNING

    # Worker w1 dies — parent task retried (goes to PENDING)
    fail_worker(state, worker_id, "Connection lost")

    # Parent task should be retried
    parent_task = _query_task(state, parent_tasks[0].task_id)
    assert parent_task.state == job_pb2.TASK_STATE_PENDING

    # Default policy for single-task job is TERMINATE_CHILDREN: child killed
    child_task = _query_task(state, child_tasks[0].task_id)
    assert child_task.state == job_pb2.TASK_STATE_KILLED

    child_job = _query_job(state, JobName.from_string("/test-user/parent/child"))
    assert child_job.state == job_pb2.JOB_STATE_KILLED


def test_worker_death_preemption_policy_preserve(state):
    """Parent with PRESERVE_CHILDREN policy -> children survive worker death retry."""
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    # Parent with PRESERVE policy
    parent_req = make_job_request("parent")
    parent_req.max_retries_preemption = 5
    parent_req.preemption_policy = job_pb2.JOB_PREEMPTION_POLICY_PRESERVE_CHILDREN
    parent_tasks = submit_job(state, "parent", parent_req)
    dispatch_task(state, parent_tasks[0], worker_id)

    # Child job
    child_req = make_job_request("child")
    child_tasks = submit_job(state, "/test-user/parent/child", child_req)
    w2 = register_worker(state, "w2", "host2:8080", make_worker_metadata())
    dispatch_task(state, child_tasks[0], w2)
    assert _query_task(state, child_tasks[0].task_id).state == job_pb2.TASK_STATE_RUNNING

    # Worker w1 dies — parent task retried
    fail_worker(state, worker_id, "Connection lost")

    # Parent task goes back to PENDING
    parent_task = _query_task(state, parent_tasks[0].task_id)
    assert parent_task.state == job_pb2.TASK_STATE_PENDING

    # PRESERVE_CHILDREN: child stays alive
    child_task = _query_task(state, child_tasks[0].task_id)
    assert child_task.state == job_pb2.TASK_STATE_RUNNING

    child_job = _query_job(state, JobName.from_string("/test-user/parent/child"))
    assert child_job.state == job_pb2.JOB_STATE_RUNNING


def test_worker_death_terminal_job_respects_preserve_children_policy(state):
    """A multi-task parent that goes *terminal* on worker death must honour its
    PRESERVE_CHILDREN policy for descendant jobs.

    The terminal-finalize child cascade is policy-gated (same gate as the
    reconcile path's ``_finalize_terminal_job``). The worker-failure batch
    previously cascaded children unconditionally once the parent went terminal,
    killing descendants the policy says to keep — the inconsistency that the
    PENDING-rollback policy gate (``test_worker_death_preemption_policy_preserve``)
    already covered but the terminal branch did not.
    """
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    # Multi-task parent with no preemption budget: worker death makes both tasks
    # terminal, taking the whole job terminal. Multi-task default policy is
    # PRESERVE_CHILDREN.
    parent_req = controller_pb2.Controller.LaunchJobRequest(
        name="parent",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=2,
        max_retries_preemption=0,
        max_task_failures=0,
    )
    parent_tasks = submit_job(state, "parent", parent_req)
    for task in parent_tasks:
        dispatch_task(state, task, worker_id)

    # Child job under the parent, running on its own worker.
    child_req = make_job_request("child")
    child_tasks = submit_job(state, "/test-user/parent/child", child_req)
    w2 = register_worker(state, "w2", "host2:8080", make_worker_metadata())
    dispatch_task(state, child_tasks[0], w2)
    assert _query_task(state, child_tasks[0].task_id).state == job_pb2.TASK_STATE_RUNNING

    # w1 dies: both parent tasks exhaust budget -> WORKER_FAILED -> parent terminal.
    fail_worker(state, worker_id, "Connection lost")
    for task in parent_tasks:
        assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_WORKER_FAILED

    # PRESERVE_CHILDREN: the child survives the parent's terminal finalize.
    assert _query_task(state, child_tasks[0].task_id).state == job_pb2.TASK_STATE_RUNNING
    child_job = _query_job(state, JobName.from_string("/test-user/parent/child"))
    assert child_job.state == job_pb2.JOB_STATE_RUNNING


def test_multi_task_parent_preserves_children(state):
    """Multi-task parent (replicas > 1) -> children preserved by default on retry."""
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    # Multi-task parent (replicas=2) — default policy is PRESERVE_CHILDREN
    parent_req = controller_pb2.Controller.LaunchJobRequest(
        name="multi-parent",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=2,
        max_retries_preemption=5,
    )
    parent_tasks = submit_job(state, "parent", parent_req)
    dispatch_task(state, parent_tasks[0], worker_id)

    # Leave second parent task pending, dispatch child
    child_req = make_job_request("child")
    child_tasks = submit_job(state, "/test-user/parent/child", child_req)
    w2 = register_worker(state, "w2", "host2:8080", make_worker_metadata())
    dispatch_task(state, child_tasks[0], w2)
    assert _query_task(state, child_tasks[0].task_id).state == job_pb2.TASK_STATE_RUNNING

    # Worker w1 dies — parent task[0] retried
    fail_worker(state, worker_id, "Connection lost")

    parent_task = _query_task(state, parent_tasks[0].task_id)
    assert parent_task.state == job_pb2.TASK_STATE_PENDING

    # Multi-task default is PRESERVE_CHILDREN: child stays running
    child_task = _query_task(state, child_tasks[0].task_id)
    assert child_task.state == job_pb2.TASK_STATE_RUNNING


def test_task_update_worker_failed_cascades_children(state):
    """apply_task_observations with WORKER_FAILED terminal task cascades children via preemption policy."""
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    # Parent job with 0 preemption retries so WORKER_FAILED makes the task terminal
    parent_req = make_job_request("parent")
    parent_req.max_retries_preemption = 0
    parent_req.max_task_failures = 0
    parent_tasks = submit_job(state, "parent", parent_req)
    dispatch_task(state, parent_tasks[0], worker_id)

    # Child job under parent
    child_req = make_job_request("child")
    child_tasks = submit_job(state, "/test-user/parent/child", child_req)
    w2 = register_worker(state, "w2", "host2:8080", make_worker_metadata())
    dispatch_task(state, child_tasks[0], w2)
    assert _query_task(state, child_tasks[0].task_id).state == job_pb2.TASK_STATE_RUNNING

    # Report WORKER_FAILED via heartbeat update (goes through apply_task_observations)
    transition_task(state, parent_tasks[0].task_id, job_pb2.TASK_STATE_WORKER_FAILED, error="Worker crashed")

    # Parent task terminal
    parent_task = _query_task(state, parent_tasks[0].task_id)
    assert parent_task.state == job_pb2.TASK_STATE_WORKER_FAILED

    # Parent job should be WORKER_FAILED
    parent_job = _query_job(state, JobName.root("test-user", "parent"))
    assert parent_job.state == job_pb2.JOB_STATE_WORKER_FAILED

    # Child should be killed via cascade — last occurrence in file
    child_task = _query_task(state, child_tasks[0].task_id)
    assert child_task.state == job_pb2.TASK_STATE_KILLED

    child_job = _query_job(state, JobName.from_string("/test-user/parent/child"))
    assert child_job.state == job_pb2.JOB_STATE_KILLED


def test_reconcile_worker_failed_pending_rollback_cascades_children(state):
    """Worker-reported WORKER_FAILED that retries a parent to PENDING cascades children.

    Reconcile-path analogue of ``test_worker_death_preemption_policy_terminate``:
    the failure arrives as a worker observation (through ``apply_task_observations``
    -> ``ReconcileState.reconcile``) rather than as a controller-asserted worker death.
    With retry budget the parent task rolls back to PENDING (its job stays RUNNING,
    so terminal finalize never fires), and the single-task default
    ``TERMINATE_CHILDREN`` policy must still kill descendant jobs.
    """
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    # Parent with retry budget so WORKER_FAILED rolls the task back to PENDING
    # rather than taking it (and the job) terminal.
    parent_req = make_job_request("parent")
    parent_req.max_retries_preemption = 5
    parent_tasks = submit_job(state, "parent", parent_req)
    dispatch_task(state, parent_tasks[0], worker_id)
    assert _query_task(state, parent_tasks[0].task_id).state == job_pb2.TASK_STATE_RUNNING

    # Child job running on its own worker.
    child_req = make_job_request("child")
    child_tasks = submit_job(state, "/test-user/parent/child", child_req)
    w2 = register_worker(state, "w2", "host2:8080", make_worker_metadata())
    dispatch_task(state, child_tasks[0], w2)
    assert _query_task(state, child_tasks[0].task_id).state == job_pb2.TASK_STATE_RUNNING

    # Worker reports WORKER_FAILED for the parent task via the reconcile path.
    transition_task(state, parent_tasks[0].task_id, job_pb2.TASK_STATE_WORKER_FAILED, error="Worker crashed")

    # Parent task rolled back to PENDING; parent job stays RUNNING (not terminal).
    parent_task = _query_task(state, parent_tasks[0].task_id)
    assert parent_task.state == job_pb2.TASK_STATE_PENDING
    parent_job = _query_job(state, JobName.root("test-user", "parent"))
    assert parent_job.state == job_pb2.JOB_STATE_RUNNING

    # TERMINATE_CHILDREN: the child task/job are cascaded to terminal.
    child_task = _query_task(state, child_tasks[0].task_id)
    assert child_task.state == job_pb2.TASK_STATE_KILLED
    child_job = _query_job(state, JobName.from_string("/test-user/parent/child"))
    assert child_job.state == job_pb2.JOB_STATE_KILLED


def test_endpoint_registered_after_task_terminal_is_orphaned(state):
    """Reproduce endpoint leak: register_endpoint succeeds for already-terminal tasks.

    When a task completes, apply_task_observations deletes its endpoints. But
    register_endpoint doesn't check task state — only attempt_id. So a slow
    register_endpoint call arriving after the task is terminal inserts an
    orphaned endpoint that is never cleaned up.
    """
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = make_job_request("leak")
    tasks = submit_job(state, "leak", req)
    task = tasks[0]

    dispatch_task(state, task, worker_id)

    # Task succeeds — any existing endpoints would be cleaned up here.
    transition_task(state, task.task_id, job_pb2.TASK_STATE_SUCCEEDED)
    task_after = _query_task(state, task.task_id)
    assert task_after.state == job_pb2.TASK_STATE_SUCCEEDED

    # Now a slow register_endpoint arrives AFTER the task is terminal.
    # This simulates the task process still alive briefly after the
    # controller processed the terminal heartbeat.
    ep = EndpointRow(
        endpoint_id="orphan-ep",
        name="leak/actor",
        address="a:1",
        task_id=task.task_id,
        metadata={},
        registered_at=Timestamp.now(),
    )
    with state._db.transaction() as cur:
        state._endpoints.add(cur, ep)

    # BUG: The endpoint is now orphaned — the task is terminal so no
    # future transition will clean it up.
    leaked = _endpoints(state, EndpointQuery(exact_name="leak/actor"))
    assert leaked == [], (
        f"Expected no endpoints for terminal task, but found {len(leaked)}. "
        "register_endpoint/add_endpoint must reject inserts for terminal tasks."
    )


# =============================================================================
# Pruning Tests
# =============================================================================
def test_prune_old_terminal_jobs(state):
    """Terminal jobs older than retention are pruned; recent and active jobs are kept."""
    wid = register_worker(state, "w1", "host:8080", make_worker_metadata())

    # Submit two jobs and complete them
    old_req = make_job_request("old-job")
    old_tasks = submit_job(state, "old-job", old_req)
    dispatch_task(state, old_tasks[0], wid)
    transition_task(state, old_tasks[0].task_id, job_pb2.TASK_STATE_SUCCEEDED)

    recent_req = make_job_request("recent-job")
    recent_tasks = submit_job(state, "recent-job", recent_req)
    dispatch_task(state, recent_tasks[0], wid)
    transition_task(state, recent_tasks[0].task_id, job_pb2.TASK_STATE_SUCCEEDED)

    # Also submit an active (non-terminal) job
    active_req = make_job_request("active-job")
    submit_job(state, "active-job", active_req)

    old_job_id = JobName.root("test-user", "old-job")
    recent_job_id = JobName.root("test-user", "recent-job")
    active_job_id = JobName.root("test-user", "active-job")

    # Backdate old-job's finished_at_ms to epoch so it falls outside retention
    with state._db.transaction() as _tx:
        _tx.execute(sa_update(jobs_table).where(jobs_table.c.job_id == old_job_id).values(finished_at_ms=1000))

    # All three jobs exist
    assert _query_job(state, old_job_id) is not None
    assert _query_job(state, recent_job_id) is not None
    assert _query_job(state, active_job_id) is not None

    # Prune with a 1-day retention — old-job finished at ~epoch, recent-job finished just now
    result = prune_old_data(
        state._db,
        worker_daemon_backends_for_prune(state),
        job_retention=Duration.from_seconds(86400),
        worker_retention=Duration.from_seconds(86400),
        slice_retention=Duration.from_seconds(86400),
    )

    assert result.jobs_deleted == 1
    assert _query_job(state, old_job_id) is None  # pruned
    assert _query_job(state, recent_job_id) is not None  # kept (recent)
    assert _query_job(state, active_job_id) is not None  # kept (non-terminal)

    # Tasks for old job should also be gone (CASCADE)
    assert _query_task(state, old_tasks[0].task_id) is None


def test_prune_old_inactive_workers(state):
    """Inactive workers with stale heartbeats are pruned; active workers are kept.

    Liveness state lives in :class:`WorkerHealthTracker` rather than the
    SQLite ``workers`` row, so the test mutates the tracker directly to age
    out the stale worker.
    """
    active_wid = register_worker(state, "active-w", "host:8080", make_worker_metadata())
    stale_wid = register_worker(state, "stale-w", "host:8081", make_worker_metadata())

    # Mark the stale worker as unhealthy with an old heartbeat in the tracker.
    state._health.set_health_for_test(stale_wid, healthy=False)
    state._health.set_last_heartbeat_for_test(stale_wid, last_heartbeat_ms=1000)

    assert _query_worker(state, active_wid) is not None
    assert _query_worker(state, stale_wid) is not None

    result = prune_old_data(
        state._db,
        worker_daemon_backends_for_prune(state),
        job_retention=Duration.from_seconds(86400),
        worker_retention=Duration.from_seconds(86400),
        slice_retention=Duration.from_seconds(86400),
    )

    assert result.workers_deleted == 1
    assert _query_worker(state, active_wid) is not None  # kept (healthy+active)
    assert _query_worker(state, stale_wid) is None  # pruned


def test_prune_noop_when_nothing_old(state):
    """Pruning with no old data returns zero counts."""

    result = prune_old_data(
        state._db,
        worker_daemon_backends_for_prune(state),
        job_retention=Duration.from_seconds(86400),
        worker_retention=Duration.from_seconds(86400),
        slice_retention=Duration.from_seconds(86400),
    )

    assert result == PruneResult()
    assert result.total == 0


def _insert_slice(state, slice_id, *, scale_group, worker_ids, created_at_ms, lifecycle="ready"):
    with state._db.transaction() as cur:
        cur.execute(
            insert(slices_table).values(
                slice_id=slice_id,
                scale_group=scale_group,
                lifecycle=lifecycle,
                worker_ids=worker_ids,
                created_at_ms=created_at_ms,
                error_message="",
            )
        )


def _query_slice(state, slice_id):
    with state._db.read_snapshot() as snap:
        return snap.execute(select(slices_table.c.slice_id).where(slices_table.c.slice_id == slice_id)).first()


def test_prune_orphaned_slices(state):
    """Old slices with no backing worker row are pruned; recent or worker-backed slices are kept.

    The orphan-slice sweep keys off ``workers.slice_id`` (the authoritative
    liveness signal), is independent of whether the scale group still exists in
    config, and is age-gated so a freshly-booting slice that hasn't registered
    its workers yet survives.
    """
    now_ms = Timestamp.now().epoch_ms()

    # (1) Old slice, no worker references it, in a scale group that is gone from
    #     config — exactly the "stranded by a rename" case. Should be deleted.
    _insert_slice(state, "orphan-old", scale_group="retired_group-zone", worker_ids=[], created_at_ms=1000)

    # (2) Recent slice with no workers yet (still booting). Should be kept by the grace age.
    _insert_slice(state, "fresh-booting", scale_group="retired_group-zone", worker_ids=[], created_at_ms=now_ms)

    # (3) Old slice still backed by a live worker row. Should be kept regardless of age.
    register_worker(
        state, "w-live", "host:9001", make_worker_metadata(), slice_id="live-old", scale_group="some_group-zone"
    )
    _insert_slice(state, "live-old", scale_group="some_group-zone", worker_ids=["w-live"], created_at_ms=1000)

    result = prune_old_data(
        state._db,
        worker_daemon_backends_for_prune(state),
        job_retention=Duration.from_seconds(86400),
        worker_retention=Duration.from_seconds(86400),
        slice_retention=Duration.from_seconds(3600),
        pause_between_s=0.0,
    )

    assert result.slices_deleted == 1
    assert _query_slice(state, "orphan-old") is None  # pruned (old + no worker)
    assert _query_slice(state, "fresh-booting") is not None  # kept (within grace)
    assert _query_slice(state, "live-old") is not None  # kept (worker row references it)


def test_prune_keeps_slice_with_live_worker_despite_empty_worker_ids(state):
    """Regression: liveness is ``workers.slice_id``, not the slice's ``worker_ids`` JSON.

    A slice whose ``worker_ids`` list is stale/empty but which a live ``workers``
    row still points at (via ``workers.slice_id``) must NOT be pruned — deleting
    it would orphan a worker that is still running tasks.
    """
    register_worker(
        state, "w-attached", "host:9002", make_worker_metadata(), slice_id="slice-empty-json", scale_group="g-zone"
    )
    # JSON worker_ids is empty even though w-attached references the slice.
    _insert_slice(state, "slice-empty-json", scale_group="g-zone", worker_ids=[], created_at_ms=1000)

    result = prune_old_data(
        state._db,
        worker_daemon_backends_for_prune(state),
        job_retention=Duration.from_seconds(86400),
        worker_retention=Duration.from_seconds(86400),
        slice_retention=Duration.from_seconds(3600),
        pause_between_s=0.0,
    )

    assert result.slices_deleted == 0
    assert _query_slice(state, "slice-empty-json") is not None  # kept (worker row references it)


def test_dispatch_propagates_task_image(state):
    """task_image set on the LaunchJobRequest is copied into the per-job RunTaskRequest template."""
    register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = make_job_request("img-job", task_image="custom/swetrace:dev")
    tasks = submit_job(state, "img-job", req)
    job_id = tasks[0].job_id
    with state._db.read_snapshot() as snap:
        template = snap.caches[RunTemplatesProjection].get(snap, job_id)
    assert template is not None
    assert template.task_image == "custom/swetrace:dev"


def test_run_request_template_does_not_leak_workdir_files_across_jobs(state):
    """Two jobs with identical entrypoint_json must get independent workdir_files.

    proto_from_json caches parsed protos by JSON string; two jobs with the same
    serialized RuntimeEntrypoint (same setup_commands + run_command) share the
    cached instance. Mutating the cached instance to attach per-job
    workdir_files would leak files from one job's template into another's.
    Regression test for the cached-proto mutation bug.
    """
    register_worker(state, "w1", "host:8080", make_worker_metadata())

    # Two jobs with identical entrypoint (so the cache key collides) but
    # different inline workdir files.
    req_a = make_job_request("job-a")
    req_a.entrypoint.workdir_files["a.txt"] = b"A"
    req_b = make_job_request("job-b")
    req_b.entrypoint.workdir_files["b.txt"] = b"B"

    tasks_a = submit_job(state, "job-a", req_a)
    tasks_b = submit_job(state, "job-b", req_b)

    with state._db.read_snapshot() as snap:
        template_a = snap.caches[RunTemplatesProjection].get(snap, tasks_a[0].job_id)
        template_b = snap.caches[RunTemplatesProjection].get(snap, tasks_b[0].job_id)

    assert template_a is not None
    assert template_b is not None
    assert dict(template_a.entrypoint.workdir_files) == {"a.txt": b"A"}
    assert dict(template_b.entrypoint.workdir_files) == {"b.txt": b"B"}


def test_resubmit_invalidates_run_template_cache(state):
    """Resubmitting a job with the same JobName serves the NEW payload, not the old one.

    The projection invalidates post-commit so a reader that opened a snapshot
    before the resubmit cannot store a stale template back into the cache after
    the new row commits.  Without post-commit invalidation the old template
    would survive until the next eviction.
    """
    job_id = JobName.root("test-user", "my-job")

    req_v1 = make_job_request("my-job", task_image="image:v1")
    with state._db.transaction() as cur:
        ops.job.submit(cur, job_id=job_id, request=req_v1, ts=Timestamp.now())

    # Warm the cache by reading the first submission's template.
    with state._db.read_snapshot() as snap:
        template_v1 = snap.caches[RunTemplatesProjection].get(snap, job_id)
    assert template_v1 is not None
    assert template_v1.task_image == "image:v1"

    # Cancel and purge the first job so the job_id slot is free for resubmission.
    with state._db.transaction() as cur:
        ops.job.cancel(cur, job_id=job_id, reason="resubmit test")
    with state._db.transaction() as cur:
        ops.job.remove_finished(cur, job_id)

    # Resubmit with a different payload under the same job name.
    req_v2 = make_job_request("my-job", task_image="image:v2")
    with state._db.transaction() as cur:
        ops.job.submit(cur, job_id=job_id, request=req_v2, ts=Timestamp.now())

    # The cache must return the new template, not the old one.
    with state._db.read_snapshot() as snap:
        template_v2 = snap.caches[RunTemplatesProjection].get(snap, job_id)
    assert template_v2 is not None
    assert template_v2.task_image == "image:v2"


def test_prune_old_data_short_circuits_when_nothing_prunable(state):
    """prune_old_data skips the write lock when a read_snapshot shows nothing to prune."""
    wid = register_worker(state, "w1", "host:8080", make_worker_metadata())
    req = make_job_request("active-job")
    tasks = submit_job(state, "active-job", req)
    dispatch_task(state, tasks[0], wid)

    result = prune_old_data(
        state._db,
        worker_daemon_backends_for_prune(state),
        job_retention=Duration.from_seconds(86400),
        worker_retention=Duration.from_seconds(86400),
        slice_retention=Duration.from_seconds(86400),
    )

    assert result == PruneResult()
    assert result.total == 0


# =============================================================================
# Direct Provider Transition Tests
# =============================================================================


def _submit_job_direct(
    state: ControllerTestState,
    job_id_str: str,
    *,
    replicas: int = 1,
    max_retries_failure: int = 0,
    max_retries_preemption: int = 0,
    max_task_failures: int = 0,
) -> list[JobName]:
    job_id = JobName.from_wire(job_id_str)
    request = controller_pb2.Controller.LaunchJobRequest(
        name="test-job",
        replicas=replicas,
        max_retries_failure=max_retries_failure,
        max_retries_preemption=max_retries_preemption,
        max_task_failures=max_task_failures,
    )
    with state._db.transaction() as cur:
        ops.job.submit(cur, job_id=job_id, request=request, ts=Timestamp.now())
    return [job_id.task(idx) for idx in range(replicas)]


def _task_state_direct(state: ControllerTestState, task_id: JobName) -> int:
    with state._db.read_snapshot() as tx:
        row = reads.get_task_detail(tx, task_id)
    assert row is not None
    return int(row.state)


def _task_row_direct(state: ControllerTestState, task_id: JobName):
    with state._db.read_snapshot() as tx:
        row = reads.get_task_detail(tx, task_id)
    assert row is not None
    return row


def _run_direct_tasks(state: ControllerTestState, task_ids: list[JobName]) -> None:
    """Drain and transition tasks to RUNNING via direct provider."""
    with state._db.transaction() as cur:
        dispatch.drain_for_dispatch(cur)
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [TaskUpdate(task_id=t, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING) for t in task_ids],
            now=Timestamp.now(),
        )


def test_drain_pending_creates_attempt_rows(state):
    """drain_for_dispatch promotes PENDING tasks to ASSIGNED with NULL worker_id."""
    task_ids = _submit_job_direct(state, "/user/job1")
    task_id = task_ids[0]

    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)

    assert len(batch.tasks_to_run) == 1
    assert batch.tasks_to_run[0].task_id == task_id.to_wire()
    assert batch.tasks_to_run[0].attempt_id == 0
    assert _task_state_direct(state, task_id) == job_pb2.TASK_STATE_ASSIGNED

    # Verify attempt row was created with NULL worker_id.
    with state._db.read_snapshot() as _snap:
        row = _snap.execute(
            select(task_attempts_table.c.worker_id, task_attempts_table.c.state).where(
                (task_attempts_table.c.task_id == task_id) & (task_attempts_table.c.attempt_id == 0)
            )
        ).first()
    assert row is not None
    assert row.worker_id is None
    assert int(row.state) == job_pb2.TASK_STATE_ASSIGNED


def _count_pending(state: ControllerTestState) -> int:
    with state._db.read_snapshot() as q:
        row = q.execute(
            select(func.count().label("c"))
            .select_from(tasks_table)
            .where(tasks_table.c.state == job_pb2.TASK_STATE_PENDING)
        ).first()
    return int(row.c)


def test_drain_redrives_assigned_until_executing(state):
    """ASSIGNED+null-worker rows are redriven each cycle so a missed pod-apply
    is recovered, and they are also in running_tasks so the same-cycle poll
    transitions them out of ASSIGNED. Once the task reaches BUILDING/RUNNING,
    it leaves tasks_to_run but stays in running_tasks."""
    task_ids = _submit_job_direct(state, "/user/job1")
    task_id = task_ids[0]

    # First drain: PENDING -> ASSIGNED, dispatched and polled.
    with state._db.transaction() as cur:
        batch1 = dispatch.drain_for_dispatch(cur)
    assert len(batch1.tasks_to_run) == 1
    assert [(e.task_id, e.attempt_id) for e in batch1.running_tasks] == [(task_id, 0)]

    # Second drain (e.g. previous _apply_pod failed or controller crashed):
    # row is still ASSIGNED, redriven in tasks_to_run with same attempt_id.
    with state._db.transaction() as cur:
        batch2 = dispatch.drain_for_dispatch(cur)
    assert len(batch2.tasks_to_run) == 1
    assert batch2.tasks_to_run[0].task_id == task_id.to_wire()
    assert batch2.tasks_to_run[0].attempt_id == 0
    assert [(e.task_id, e.attempt_id) for e in batch2.running_tasks] == [(task_id, 0)]

    # Once the task reaches RUNNING it leaves tasks_to_run; running_tasks still
    # contains it so the next poll observes terminal transitions.
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING)],
            now=Timestamp.now(),
        )
    with state._db.transaction() as cur:
        batch3 = dispatch.drain_for_dispatch(cur)
    assert len(batch3.tasks_to_run) == 0
    assert len(batch3.running_tasks) == 1
    assert batch3.running_tasks[0].task_id == task_id


def test_drain_caps_promotions_per_cycle(state):
    """``max_promotions`` caps how many PENDING rows are promoted per cycle.
    Redrives of already-ASSIGNED rows do not count against the cap."""
    _submit_job_direct(state, "/user/big-job", replicas=200)
    assert _count_pending(state) == 200

    with state._db.transaction() as cur:
        batch1 = dispatch.drain_for_dispatch(cur, max_promotions=128)
    # All 128 dispatched are freshly promoted (no prior ASSIGNED rows).
    assert len(batch1.tasks_to_run) == 128
    assert _count_pending(state) == 72

    # Second drain: 72 newly promoted, 128 redriven.
    with state._db.transaction() as cur:
        batch2 = dispatch.drain_for_dispatch(cur, max_promotions=128)
    assert len(batch2.tasks_to_run) == 200
    assert _count_pending(state) == 0


def test_drain_max_promotions_limits_batch(state):
    """``max_promotions`` is a per-cycle PENDING-promotion budget, not a cap
    on total dispatch (which also includes ASSIGNED redrives)."""
    _submit_job_direct(state, "/user/cap-job", replicas=250)

    with state._db.transaction() as cur:
        batch1 = dispatch.drain_for_dispatch(cur, max_promotions=50)
    assert len(batch1.tasks_to_run) == 50
    assert _count_pending(state) == 200

    # 50 newly promoted + 50 prior ASSIGNED redriven.
    with state._db.transaction() as cur:
        batch2 = dispatch.drain_for_dispatch(cur, max_promotions=50)
    assert len(batch2.tasks_to_run) == 100
    assert _count_pending(state) == 150


def test_apply_running(state):
    """Applying a RUNNING update transitions task from ASSIGNED to RUNNING."""
    task_ids = _submit_job_direct(state, "/user/job1")
    task_id = task_ids[0]
    with state._db.transaction() as cur:
        dispatch.drain_for_dispatch(cur)

    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING),
            ],
            now=Timestamp.now(),
        )

    assert _task_state_direct(state, task_id) == job_pb2.TASK_STATE_RUNNING


def test_apply_succeeded(state):
    """Applying SUCCEEDED transitions task to terminal state with exit_code=0."""
    task_ids = _submit_job_direct(state, "/user/job1")
    task_id = task_ids[0]
    with state._db.transaction() as cur:
        dispatch.drain_for_dispatch(cur)

    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING),
            ],
            now=Timestamp.now(),
        )
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_SUCCEEDED),
            ],
            now=Timestamp.now(),
        )

    task = _task_row_direct(state, task_id)
    assert task.state == job_pb2.TASK_STATE_SUCCEEDED
    assert task.exit_code == 0
    assert task.finished_at_ms is not None


def test_apply_failed_with_retry(state):
    """FAILED with retries remaining returns task to PENDING."""
    task_ids = _submit_job_direct(state, "/user/job1", max_retries_failure=1, max_task_failures=1)
    task_id = task_ids[0]
    with state._db.transaction() as cur:
        dispatch.drain_for_dispatch(cur)

    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING),
            ],
            now=Timestamp.now(),
        )
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_FAILED, error="boom"),
            ],
            now=Timestamp.now(),
        )

    # Task should be PENDING again (1 failure <= 1 max_retries_failure).
    assert _task_state_direct(state, task_id) == job_pb2.TASK_STATE_PENDING

    # The dead attempt 0 must have finished_at_ms stamped even though the task
    # itself rolled back to PENDING. Otherwise the row is indistinguishable from
    # a still-assigned attempt. Regression guard for the terminal_ms conflation.
    with state._db.read_snapshot() as tx:
        attempts = tx.execute(
            select(
                task_attempts_table.c.task_id,
                task_attempts_table.c.attempt_id,
                task_attempts_table.c.worker_id,
                task_attempts_table.c.state,
                task_attempts_table.c.created_at_ms,
                task_attempts_table.c.started_at_ms,
                task_attempts_table.c.finished_at_ms,
                task_attempts_table.c.exit_code,
                task_attempts_table.c.error,
            )
            .where(task_attempts_table.c.task_id == task_id)
            .order_by(task_attempts_table.c.attempt_id.asc())
        ).all()
    assert len(attempts) == 1
    assert attempts[0].state == job_pb2.TASK_STATE_FAILED
    assert attempts[0].finished_at_ms is not None

    # Draining again should promote it for a second attempt.
    with state._db.transaction() as cur:
        batch = dispatch.drain_for_dispatch(cur)
    assert len(batch.tasks_to_run) == 1
    assert batch.tasks_to_run[0].attempt_id == 1


def test_apply_failed_no_retry(state):
    """FAILED with no retries remaining leaves task in FAILED terminal state."""
    task_ids = _submit_job_direct(state, "/user/job1", max_retries_failure=0)
    task_id = task_ids[0]
    with state._db.transaction() as cur:
        dispatch.drain_for_dispatch(cur)

    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING),
            ],
            now=Timestamp.now(),
        )
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_FAILED, error="boom"),
            ],
            now=Timestamp.now(),
        )

    task = _task_row_direct(state, task_id)
    assert task.state == job_pb2.TASK_STATE_FAILED
    assert task.failure_count == 1
    assert task.finished_at_ms is not None


def test_apply_worker_failed(state):
    """WORKER_FAILED on a RUNNING task increments preemption_count and retries if allowed."""
    task_ids = _submit_job_direct(state, "/user/job1", max_retries_preemption=1)
    task_id = task_ids[0]
    with state._db.transaction() as cur:
        dispatch.drain_for_dispatch(cur)

    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING),
            ],
            now=Timestamp.now(),
        )
    with state._db.transaction() as cur:
        commit_dispatch_updates(
            cur,
            [
                TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_WORKER_FAILED, error="node died"),
            ],
            now=Timestamp.now(),
        )

    # Should be retried (preemption_count=1 <= max_retries_preemption=1).
    assert _task_state_direct(state, task_id) == job_pb2.TASK_STATE_PENDING
    task = _task_row_direct(state, task_id)
    assert task.preemption_count == 1


def test_cancel_job_kills_dispatch_tasks(state):
    """cancel_job moves NULL-worker_id (direct-provider) tasks to KILLED."""
    task_ids = _submit_job_direct(state, "/user/job1", replicas=2)
    _run_direct_tasks(state, task_ids)

    with state._db.transaction() as cur:
        ops.job.cancel(
            cur,
            job_id=JobName.from_wire("/user/job1"),
            reason="test cancel",
        )

    for tid in task_ids:
        assert _task_state_direct(state, tid) == job_pb2.TASK_STATE_KILLED


def test_kill_non_terminal_dispatch_tasks(state):
    """cancel_job kills NULL-worker_id tasks via _kill_non_terminal_tasks cascade."""
    task_ids = _submit_job_direct(state, "/user/job1")
    _run_direct_tasks(state, task_ids)

    # Trigger via cancel_job which calls _kill_non_terminal_tasks indirectly through
    # cascade, or call it via a job failure path. Use cancel_job for simplicity.
    with state._db.transaction() as cur:
        ops.job.cancel(
            cur,
            job_id=JobName.from_wire("/user/job1"),
            reason="test kill",
        )

    assert _task_state_direct(state, task_ids[0]) == job_pb2.TASK_STATE_KILLED


def test_max_failures_kills_dispatch_tasks(state):
    """When a task fails and triggers kill of siblings, direct-provider tasks are killed via task_mutations."""
    task_ids = _submit_job_direct(state, "/user/job1", replicas=2, max_retries_failure=0)
    _run_direct_tasks(state, task_ids)

    # Fail one task — with max_task_failures=0 (default) this should kill the job,
    # triggering _kill_non_terminal_tasks for the sibling.
    with state._db.transaction() as cur:
        result = commit_dispatch_updates(
            cur,
            [TaskUpdate(task_id=task_ids[0], attempt_id=0, new_state=job_pb2.TASK_STATE_FAILED, error="boom")],
            now=Timestamp.now(),
        )

    # The sibling task (task_ids[1]) should have a KILLED task delta.
    killed_task_ids = {tid for tid, delta in result.tasks.items() if delta.state == job_pb2.TASK_STATE_KILLED}
    assert task_ids[1] in killed_task_ids


# =============================================================================
# Job state lifecycle tests (merged from test_job.py)
# =============================================================================


def test_job_becomes_succeeded_when_all_tasks_succeed(harness) -> None:
    worker_id = harness.add_worker("w1")
    tasks = harness.submit("all-succeeded", replicas=2)

    for task in tasks:
        harness.dispatch(task, worker_id)
        harness.transition(task.task_id, job_pb2.TASK_STATE_SUCCEEDED)

    assert harness.query_job(JobName.root("test-user", "all-succeeded")).state == job_pb2.JOB_STATE_SUCCEEDED


def test_job_failure_threshold_applies(harness) -> None:
    worker_id = harness.add_worker("w1")
    tasks = harness.submit("fail-fast", replicas=2)

    harness.dispatch(tasks[0], worker_id)
    harness.transition(tasks[0].task_id, job_pb2.TASK_STATE_FAILED)

    assert harness.query_job(JobName.root("test-user", "fail-fast")).state == job_pb2.JOB_STATE_FAILED


def test_job_expands_to_replicas_and_retry_limits(harness) -> None:
    tasks = harness.submit("expand", replicas=3, max_retries_failure=3, max_retries_preemption=7)

    jid = JobName.root("test-user", "expand")
    assert len(tasks) == 3
    for idx, task in enumerate(tasks):
        assert task.task_id == jid.task(idx)
        task_row = harness.query_task(task.task_id)
        assert task_row.max_retries_failure == 3
        assert task_row.max_retries_preemption == 7


def test_job_becomes_unschedulable_when_task_unschedulable(harness) -> None:
    tasks = harness.submit("unsched", replicas=2)
    with harness.state._db.transaction() as cur:
        finalize(
            cur,
            [TerminalDecision(TerminalKind.UNSCHEDULABLE, tasks[0].task_id, "no capacity")],
            now=Timestamp.now(),
        )
    assert harness.query_job(JobName.root("test-user", "unsched")).state == job_pb2.JOB_STATE_UNSCHEDULABLE


def test_job_cancel_marks_job_killed(harness) -> None:
    harness.submit("killed", replicas=2)
    jid = JobName.root("test-user", "killed")
    with harness.state._db.transaction() as cur:
        ops.job.cancel(cur, job_id=jid, reason="manual")
    assert harness.query_job(jid).state == job_pb2.JOB_STATE_KILLED


def _empty_snapshot(*, job_state_basis=None):
    return TransitionSnapshot(
        now=Timestamp.from_ms(1000),
        tasks={},
        attempts={},
        attempt_uid_index={},
        job_configs={},
        job_state_basis=job_state_basis or {},
        job_descendants={},
        all_tasks_by_job={},
        active_tasks_by_job={},
        active_workers=frozenset(),
    )


def _basis(job_id: JobName, state: int):
    return JobStateBasis(
        job_id=job_id,
        state=state,
        started_at=None,
        max_task_failures=0,
        task_state_counts={},
        total_failures=0,
        first_task_error=None,
    )


def test_cascade_kill_noops_on_worker_failed_without_allow_overwrite():
    """A cascade kill is a no-op on a WORKER_FAILED job unless allow_overwrite is set."""
    jid = JobName.from_wire("/u/j")
    snap = _empty_snapshot(job_state_basis={jid: _basis(jid, job_pb2.JOB_STATE_WORKER_FAILED)})
    ws = Overlay(snap)
    ws.merge_cascade_kill(
        JobRowDelta(
            job_id=jid,
            state=job_pb2.JOB_STATE_KILLED,
            error="cascade",
            finished_at=Timestamp.from_ms(100),
            is_cascade_kill=True,
            allow_overwrite_worker_failed=False,
        )
    )
    # Guarded: WORKER_FAILED is in TERMINAL_JOB_STATES -> no delta recorded.
    assert jid not in ws.effects.jobs


def test_cascade_kill_overwrites_worker_failed_with_allow_overwrite():
    """allow_overwrite_worker_failed widens the guard so WORKER_FAILED is killed (cancel)."""
    jid = JobName.from_wire("/u/j")
    snap = _empty_snapshot(job_state_basis={jid: _basis(jid, job_pb2.JOB_STATE_WORKER_FAILED)})
    ws = Overlay(snap)
    ws.merge_cascade_kill(
        JobRowDelta(
            job_id=jid,
            state=job_pb2.JOB_STATE_KILLED,
            error="cancel",
            finished_at=Timestamp.from_ms(100),
            is_cascade_kill=True,
            allow_overwrite_worker_failed=True,
        )
    )
    assert ws.effects.jobs[jid].state == job_pb2.JOB_STATE_KILLED
    assert ws.effects.jobs[jid].error == "cancel"


def test_cascade_kill_noops_on_already_terminal_job():
    jid = JobName.from_wire("/u/j")
    snap = _empty_snapshot(job_state_basis={jid: _basis(jid, job_pb2.JOB_STATE_SUCCEEDED)})
    ws = Overlay(snap)
    ws.merge_cascade_kill(
        JobRowDelta(
            job_id=jid,
            state=job_pb2.JOB_STATE_KILLED,
            error="cascade",
            finished_at=Timestamp.from_ms(100),
            is_cascade_kill=True,
        )
    )
    assert jid not in ws.effects.jobs


def _recompute_snapshot(
    job_id: JobName,
    task_states: list[int],
    max_task_failures: int,
    failure_counts: list[int] | None = None,
):
    """A RUNNING job snapshot whose tasks carry ``task_states``.

    ``started_at`` is set on the basis so a job that does not otherwise finalize
    falls through to the RUNNING strand the #5 fix closes. ``job_basis`` rebuilds
    the task histogram and cumulative failure count from ``all_tasks_by_job``, so
    both live there. ``failure_counts`` defaults to one charged failure per task
    currently in FAILED; pass it explicitly to model failures that have already
    been retried back to PENDING/RUNNING (the coscheduled crash-loop case).
    """
    if failure_counts is None:
        failure_counts = [1 if st == job_pb2.TASK_STATE_FAILED else 0 for st in task_states]
    basis = JobStateBasis(
        job_id=job_id,
        state=job_pb2.JOB_STATE_RUNNING,
        started_at=Timestamp.from_ms(500),
        max_task_failures=max_task_failures,
        task_state_counts={},
        total_failures=sum(failure_counts),
        first_task_error="boom",
    )
    rows = tuple(
        TaskHistogramRow(
            task_id=JobName.from_wire(f"{job_id.to_wire()}/{i}"),
            task_index=i,
            state=st,
            failure_count=fc,
            error="boom" if st == job_pb2.TASK_STATE_FAILED else None,
        )
        for i, (st, fc) in enumerate(zip(task_states, failure_counts, strict=True))
    )
    return TransitionSnapshot(
        now=Timestamp.from_ms(1000),
        tasks={},
        attempts={},
        attempt_uid_index={},
        job_configs={},
        job_state_basis={job_id: basis},
        job_descendants={},
        all_tasks_by_job={job_id: rows},
        active_tasks_by_job={},
        active_workers=frozenset(),
    )


@pytest.mark.parametrize(
    "task_states",
    [
        [job_pb2.TASK_STATE_FAILED],
        [job_pb2.TASK_STATE_FAILED, job_pb2.TASK_STATE_SUCCEEDED, job_pb2.TASK_STATE_SUCCEEDED],
    ],
    ids=["lone-failed", "failed-with-succeeded-siblings"],
)
def test_recompute_fails_job_when_all_tasks_terminal_with_a_failure(task_states):
    """Once every task is terminal, a lone FAILED task fails the job.

    The failure is within ``max_task_failures`` (so the cumulative-budget branch
    did not fire) and no worker/preempt/cosched terminal state is present, but a
    terminally FAILED task can never succeed, so the job as a whole fails instead
    of hanging RUNNING.
    """
    jid = JobName.from_wire("/u/terminal-failure")
    ws = Overlay(_recompute_snapshot(jid, task_states, max_task_failures=1))

    new_state = recompute_state(ws, jid)

    assert new_state == job_pb2.JOB_STATE_FAILED
    assert ws.effects.jobs[jid].state == job_pb2.JOB_STATE_FAILED
    assert ws.effects.jobs[jid].finished_at is not None


@pytest.mark.parametrize(
    "task_states, failure_counts, max_task_failures",
    [
        ([job_pb2.TASK_STATE_PENDING, job_pb2.TASK_STATE_RUNNING], [1, 0], 0),
        (
            [job_pb2.TASK_STATE_PENDING, job_pb2.TASK_STATE_PENDING, job_pb2.TASK_STATE_RUNNING],
            [1, 1, 0],
            1,
        ),
    ],
    ids=["one-failure-retried", "failures-spread-across-tasks"],
)
def test_recompute_fails_job_on_cumulative_failures_while_active(task_states, failure_counts, max_task_failures):
    """Cumulative hard failures fail the job even with no task currently FAILED.

    Models a coscheduled gang mid-crash-loop: each crashed round charges a task's
    ``failure_count`` and bounces it back to PENDING, so the instantaneous
    histogram holds only a live RUNNING sibling and no FAILED task. The running
    total of failures still exceeds ``max_task_failures`` and fails the job,
    rather than letting the gang crash-loop forever because the failure keeps
    landing on a different task that never exhausts its own per-task retries.
    """
    jid = JobName.from_wire("/u/gang")
    snap = _recompute_snapshot(jid, task_states, max_task_failures=max_task_failures, failure_counts=failure_counts)
    ws = Overlay(snap)

    new_state = recompute_state(ws, jid)

    assert new_state == job_pb2.JOB_STATE_FAILED
    assert ws.effects.jobs[jid].state == job_pb2.JOB_STATE_FAILED


@pytest.mark.parametrize(
    "task_states, failure_counts, max_task_failures",
    [
        ([job_pb2.TASK_STATE_FAILED, job_pb2.TASK_STATE_RUNNING], None, 1),
        ([job_pb2.TASK_STATE_PENDING, job_pb2.TASK_STATE_RUNNING], [1, 0], 2),
    ],
    ids=["failed-with-active-sibling", "within-cumulative-budget"],
)
def test_recompute_keeps_job_running_within_budget(task_states, failure_counts, max_task_failures):
    """An active task keeps the job RUNNING while failures stay within budget.

    A task currently in FAILED does not finalize the job while a sibling is still
    active, and failures retried back to PENDING keep the job RUNNING as long as
    the cumulative total stays within ``max_task_failures`` so the retries can
    proceed.
    """
    jid = JobName.from_wire("/u/active")
    snap = _recompute_snapshot(jid, task_states, max_task_failures=max_task_failures, failure_counts=failure_counts)
    ws = Overlay(snap)

    new_state = recompute_state(ws, jid)

    assert new_state == job_pb2.JOB_STATE_RUNNING
