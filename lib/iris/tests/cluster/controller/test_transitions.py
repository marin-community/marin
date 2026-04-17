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

from iris.cluster.constraints import DeviceType, WellKnownAttribute, constraints_from_resources
from iris.cluster.controller.codec import constraints_from_json, resource_spec_from_scalars
from iris.cluster.controller.autoscaler.models import DemandEntry
from iris.cluster.controller.controller import compute_demand_entries
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.store import attempt_is_terminal
from iris.cluster.controller.schema import (
    ATTEMPT_PROJECTION,
    JOB_DETAIL_PROJECTION,
    TASK_DETAIL_PROJECTION,
    WORKER_DETAIL_PROJECTION,
    EndpointRow,
)
from iris.cluster.controller.store import EndpointQuery
from iris.cluster.controller.scheduler import JobRequirements, Scheduler
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    HEARTBEAT_FAILURE_THRESHOLD,
    HeartbeatAction,
    HeartbeatApplyRequest,
    MAX_REPLICAS_PER_JOB,
    PruneResult,
    TaskUpdate,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from iris.rpc import logging_pb2
from rigging.timing import Duration, Timestamp

from ._testing import (
    create_attempt as _create_attempt,
    set_worker_consecutive_failures as _set_worker_consecutive_failures,
)
from .conftest import (
    building_counts as _building_counts,
    check_task_can_be_scheduled,
    check_task_is_finished,
    dispatch_task,
    fail_worker,
    healthy_active_workers,
    make_job_request,
    make_test_entrypoint as _make_test_entrypoint,
    make_worker_metadata,
    query_job as _query_job,
    query_task as _query_task,
    query_tasks_for_job as _query_tasks_for_job,
    query_worker as _query_worker,
    register_worker,
    schedulable_tasks as _schedulable_tasks,
    submit_job,
    transition_task,
    worker_running_tasks,
)

# =============================================================================
# Test Helpers
# =============================================================================


def _queued_dispatch(
    state: ControllerTransitions, worker_id: WorkerId
) -> tuple[list[job_pb2.RunTaskRequest], list[str]]:
    rows = state._db.fetchall(
        "SELECT kind, payload_proto, task_id FROM dispatch_queue WHERE worker_id = ? ORDER BY id ASC",
        (str(worker_id),),
    )
    tasks_to_run: list[job_pb2.RunTaskRequest] = []
    tasks_to_kill: list[str] = []
    for row in rows:
        if str(row["kind"]) == "run" and row["payload_proto"] is not None:
            req = job_pb2.RunTaskRequest()
            req.ParseFromString(bytes(row["payload_proto"]))
            tasks_to_run.append(req)
        elif row["task_id"] is not None:
            tasks_to_kill.append(str(row["task_id"]))
    return tasks_to_run, tasks_to_kill


def _endpoints(state: ControllerTransitions, query: EndpointQuery = EndpointQuery()) -> list[EndpointRow]:
    rows = state._stores.endpoints.query(query)
    # Mirror the original helper's ordering (registered_at DESC, endpoint_id ASC).
    return sorted(rows, key=lambda r: (-r.registered_at.epoch_ms(), r.endpoint_id))


def _build_scheduling_context(scheduler: Scheduler, state: ControllerTransitions):
    pending = _schedulable_tasks(state)
    workers = healthy_active_workers(state)
    task_ids = [t.task_id for t in pending]
    jobs: dict[JobName, JobRequirements] = {}
    for t in pending:
        job_id = t.task_id.parent
        if job_id and job_id not in jobs:
            job = _query_job(state, job_id)
            if job:
                resources = resource_spec_from_scalars(
                    job.resources.cpu_millicores,
                    job.resources.memory_bytes,
                    job.resources.disk_bytes,
                    job.resources.device_json,
                )
                jobs[job_id] = JobRequirements(
                    resources=resources,
                    constraints=constraints_from_json(job.constraints_json),
                    is_coscheduled=job.has_coscheduling,
                    coscheduling_group_by=job.coscheduling_group_by if job.has_coscheduling else None,
                )
    return scheduler.create_scheduling_context(
        workers,
        building_counts=_building_counts(state),
        pending_tasks=task_ids,
        jobs=jobs,
    )


def test_db_snapshot_select_returns_typed_rows(state) -> None:
    request = make_job_request("typed-rows")
    tasks = submit_job(state, "typed-rows", request)

    job_wire = JobName.root("test-user", "typed-rows").to_wire()
    with state._db.snapshot() as q:
        jobs = JOB_DETAIL_PROJECTION.decode(q.fetchall("SELECT * FROM jobs WHERE job_id = ?", (job_wire,)))
        task_count = q.fetchone("SELECT COUNT(*) FROM tasks WHERE job_id = ?", (job_wire,))[0]

    assert len(jobs) == 1
    assert jobs[0].submitted_at is not None
    assert jobs[0].job_id == JobName.root("test-user", "typed-rows")
    assert task_count == len(tasks)


def test_db_snapshot_projection_inferrs_typed_values(state) -> None:
    wid = register_worker(state, "proj-worker", "addr", make_worker_metadata())
    request = controller_pb2.Controller.LaunchJobRequest(
        name=JobName.root("test-user", "projection").to_wire(),
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    [task] = submit_job(state, "projection", request)
    state.queue_assignments([Assignment(task_id=task.task_id, worker_id=wid)])

    running = worker_running_tasks(state, wid)

    assert len(running) == 1
    assert task.task_id in running


def test_db_snapshot_exists_for_workers(state) -> None:
    register_worker(state, "exists-worker", "addr", make_worker_metadata())

    with state._db.snapshot() as q:
        assert q.fetchone("SELECT 1 FROM workers WHERE worker_id = ?", ("exists-worker",)) is not None


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

    harness.state.cancel_job(job_id, reason="User cancelled")

    assert harness.query_job(job_id).state == job_pb2.JOB_STATE_KILLED
    for task in tasks:
        assert harness.query_task(task.task_id).state == job_pb2.TASK_STATE_KILLED


def test_cancel_job_releases_committed_worker_resources(harness):
    """cancel_job must decommit resources on workers that had active tasks.

    Regression: cancel_job marked tasks KILLED without calling _decommit_worker_resources.
    apply_heartbeat then skipped the update (task already finished), so committed resources
    were never released, permanently blocking scheduling on those workers.
    """
    w1 = harness.add_worker("w1")
    w2 = harness.add_worker("w2")
    tasks = harness.submit("j1", replicas=3)

    harness.dispatch(tasks[0], w1)
    harness.dispatch(tasks[1], w2)

    assert _query_worker(harness.state, w1).committed_cpu_millicores == 1000
    assert _query_worker(harness.state, w1).committed_mem == 1024**3
    assert _query_worker(harness.state, w2).committed_cpu_millicores == 1000

    harness.state.cancel_job(JobName.root("test-user", "j1"), reason="User cancelled")

    assert _query_worker(harness.state, w1).committed_cpu_millicores == 0, "w1 leaked committed_cpu_millicores"
    assert _query_worker(harness.state, w1).committed_mem == 0, "w1 leaked committed_mem"
    assert _query_worker(harness.state, w2).committed_cpu_millicores == 0, "w2 leaked committed_cpu_millicores"
    assert _query_worker(harness.state, w2).committed_mem == 0, "w2 leaked committed_mem"

    assert len(worker_running_tasks(harness.state, w1)) == 0
    assert len(worker_running_tasks(harness.state, w2)) == 0


def test_cancel_job_preserves_kill_worker_mapping_after_clearing_tasks(harness):
    """cancel_job returns worker routing for kill RPCs before current_worker_id is cleared."""
    w1 = harness.add_worker("w1")
    w2 = harness.add_worker("w2")
    tasks = harness.submit("j1", replicas=2)

    harness.dispatch(tasks[0], w1)
    harness.dispatch(tasks[1], w2)

    result = harness.state.cancel_job(JobName.root("test-user", "j1"), reason="User cancelled")

    assert result.tasks_to_kill == {tasks[0].task_id, tasks[1].task_id}
    assert result.task_kill_workers == {
        tasks[0].task_id: w1,
        tasks[1].task_id: w2,
    }
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

    state.add_endpoint(
        EndpointRow(
            endpoint_id="parent-ep",
            name="parent/actor",
            address="host1:9000",
            task_id=parent_tasks[0].task_id,
            metadata={},
            registered_at=Timestamp.now(),
        ),
    )
    state.add_endpoint(
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

    state.cancel_job(JobName.root("test-user", "parent"), reason="User cancelled")

    assert _endpoints(state, EndpointQuery()) == []


def test_cancelled_job_tasks_excluded_from_demand(harness):
    """Regression test for issue #2777: Killed tasks with no attempts should not appear in demand entries."""
    worker_id = harness.add_worker("w1")
    tasks = harness.submit("j1", replicas=3)
    job_id = JobName.root("test-user", "j1")

    harness.dispatch(tasks[0], worker_id)
    harness.state.cancel_job(job_id, reason="User cancelled")

    assert harness.query_job(job_id).state == job_pb2.JOB_STATE_KILLED
    for task in tasks:
        assert harness.query_task(task.task_id).state == job_pb2.TASK_STATE_KILLED
        assert not check_task_can_be_scheduled(harness.query_task(task.task_id))

    assert len(_schedulable_tasks(harness.state)) == 0
    assert len(compute_demand_entries(harness.state._stores)) == 0


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
    with state._db.snapshot() as q:
        all_workers = WORKER_DETAIL_PROJECTION.decode(q.fetchall("SELECT * FROM workers"))
    assert len(all_workers) == 1
    assert all_workers[0].worker_id == w2

    # Task was requeued despite worker removal
    assert tasks[0].state == job_pb2.TASK_STATE_PENDING
    assert check_task_can_be_scheduled(tasks[0])

    # A re-registering worker creates a fresh entry
    w1_again = register_worker(state, "w1", "host1:8080", make_worker_metadata())
    assert _query_worker(state, w1_again) is not None
    assert _query_worker(state, w1_again).healthy is True
    with state._db.snapshot() as q:
        assert len(WORKER_DETAIL_PROJECTION.decode(q.fetchall("SELECT * FROM workers"))) == 2


def test_dispatch_failure_marks_worker_failed_and_requeues_task(state):
    """E2E: Dispatch RPC failure (task in PENDING) -> worker failed event cascades to task."""

    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = make_job_request("job1")
    req.max_retries_preemption = 1
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    # Task gets assigned (creates attempt, puts in ASSIGNED state)
    state.queue_assignments([Assignment(task_id=task.task_id, worker_id=worker_id)])
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
    state.remove_worker(worker_id)
    state.queue_assignments([Assignment(task_id=task.task_id, worker_id=worker_id)])

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
    state.add_endpoint(ep)

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
    state.add_endpoint(ep)

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
    state.add_endpoint(ep)
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
    state.add_endpoint(ep)
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
    state.queue_assignments([Assignment(task_id=task.task_id, worker_id=worker_id)])
    task = _query_task(state, task.task_id)
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=worker_id,
            worker_resource_snapshot=None,
            updates=[
                TaskUpdate(
                    task_id=task.task_id,
                    attempt_id=task.current_attempt_id,
                    new_state=job_pb2.TASK_STATE_BUILDING,
                )
            ],
        )
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
    state.add_endpoint(ep)
    assert len(_endpoints(state, EndpointQuery(exact_name="ns-1/actor"))) == 1

    # Transition to RUNNING — endpoint should survive
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=worker_id,
            worker_resource_snapshot=None,
            updates=[
                TaskUpdate(
                    task_id=task.task_id,
                    attempt_id=_query_task(state, task.task_id).current_attempt_id,
                    new_state=job_pb2.TASK_STATE_RUNNING,
                )
            ],
        )
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

    state.add_endpoint(
        EndpointRow(
            endpoint_id="ep-1",
            name="ns-1/actor",
            address="10.0.0.1:8080",
            task_id=tasks1[0].task_id,
            metadata={},
            registered_at=Timestamp.now(),
        )
    )
    state.add_endpoint(
        EndpointRow(
            endpoint_id="ep-2",
            name="ns-2/actor",
            address="10.0.0.2:8080",
            task_id=tasks2[0].task_id,
            metadata={},
            registered_at=Timestamp.now(),
        )
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
    parent_wire = JobName.root("test-user", "parent").to_wire()
    with state._db.snapshot() as q:
        children = JOB_DETAIL_PROJECTION.decode(q.fetchall("SELECT * FROM jobs WHERE parent_job_id = ?", (parent_wire,)))
    assert len(children) == 2
    assert {c.job_id for c in children} == {
        JobName.from_string("/test-user/parent/child1"),
        JobName.from_string("/test-user/parent/child2"),
    }

    # No children for leaf nodes
    grandchild_wire = JobName.from_string("/test-user/parent/child1/grandchild").to_wire()
    with state._db.snapshot() as q:
        leaf_children = JOB_DETAIL_PROJECTION.decode(
            q.fetchall("SELECT * FROM jobs WHERE parent_job_id = ?", (grandchild_wire,)),
        )
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
    """E2E: Multiple small tasks can fill a worker's capacity, blocking further assignments.

    This verifies that the scheduler correctly tracks cumulative resource usage across
    multiple running tasks. With round-robin scheduling, each worker gets at most one
    task per cycle, so we run multiple cycles to fill capacity.
    """

    # Worker with 4 CPUs
    register_worker(state, "w1", "host:8080", make_worker_metadata(cpu=4))

    # Submit 3 jobs, each using 2 CPUs
    for i in range(3):
        submit_job(state, f"j{i}", make_job_request(cpu=2))

    scheduler = Scheduler()

    # First scheduling cycle: 1 task assigned (round-robin: 1 per worker per cycle)
    context = _build_scheduling_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 1
    for task_id, worker_id in result.assignments:
        task = _query_task(state, task_id)
        dispatch_task(state, task, worker_id)

    # Second scheduling cycle: 1 more task assigned (worker still has 2 CPUs)
    context = _build_scheduling_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 1
    for task_id, worker_id in result.assignments:
        task = _query_task(state, task_id)
        dispatch_task(state, task, worker_id)

    # Third task should still be pending
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
    txn = transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_FAILED, error="OOM")

    # Task-0 should be FAILED, all other tasks should be WORKER_FAILED
    assert _query_task(state, tasks[0].task_id).state == job_pb2.TASK_STATE_FAILED
    for task in tasks[1:]:
        assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_WORKER_FAILED
        assert task.task_id in txn.tasks_to_kill


def test_coscheduled_cascade_releases_worker_resources(state):
    """Coscheduled sibling cascade must free committed resources on surviving workers.

    Regression test: previously, _cascade_coscheduled_failure marked siblings
    terminal but never called _cleanup_task_resources, leaking committed_cpu_millicores/mem
    on workers and permanently blocking future scheduling.
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
        w = _query_worker(state, WorkerId(f"w{i}"))
        assert w.committed_cpu_millicores == 2000
        assert len(worker_running_tasks(state, WorkerId(f"w{i}"))) == 1

    # Fail task-0 terminally → cascade kills siblings on w1, w2, w3
    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_FAILED, error="OOM")

    # All surviving workers (w1..w3) must have resources fully released
    for i in range(1, 4):
        w = _query_worker(state, WorkerId(f"w{i}"))
        assert w.committed_cpu_millicores == 0, f"w{i} has leaked committed_cpu_millicores={w.committed_cpu_millicores}"
        assert w.committed_mem == 0, f"w{i} has leaked committed_mem={w.committed_mem}"
        assert len(worker_running_tasks(state, WorkerId(f"w{i}"))) == 0

    # w0 should also be clean (task-0 was the trigger, cleaned up by _on_task_state_changed)
    w0 = _query_worker(state, WorkerId("w0"))
    assert w0.committed_cpu_millicores == 0
    assert len(worker_running_tasks(state, WorkerId("w0"))) == 0


def test_coscheduled_task_worker_failure_kills_siblings(state):
    """WORKER_FAILED also triggers sibling kill when retries exhausted."""

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

    # First WORKER_FAILED is retriable (retries remaining)
    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_WORKER_FAILED, error="Worker crashed (first)")

    # Task-0 is retriable, siblings still running
    assert _query_task(state, tasks[0].task_id).preemption_count == 1
    assert check_task_can_be_scheduled(_query_task(state, tasks[0].task_id))
    for task in tasks[1:]:
        assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_RUNNING

    # Re-dispatch task-0
    dispatch_task(state, tasks[0], WorkerId("w0"))

    # Second WORKER_FAILED exhausts retries - now terminal
    txn = transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_WORKER_FAILED, error="Worker crashed (second)")

    assert _query_task(state, tasks[0].task_id).state == job_pb2.TASK_STATE_WORKER_FAILED
    assert check_task_is_finished(_query_task(state, tasks[0].task_id))
    for task in tasks[1:]:
        assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_WORKER_FAILED
        assert task.task_id in txn.tasks_to_kill


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
    txn = transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_SUCCEEDED)

    # Task-0 succeeded, siblings still running
    assert _query_task(state, tasks[0].task_id).state == job_pb2.TASK_STATE_SUCCEEDED
    for task in tasks[1:]:
        assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_RUNNING
    assert len(txn.tasks_to_kill) == 0


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
    txn = transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_FAILED, error="OOM")

    # Task-0 failed, but siblings are still running (no cascade)
    assert _query_task(state, tasks[0].task_id).state == job_pb2.TASK_STATE_FAILED
    for task in tasks[1:]:
        assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_RUNNING

    # No tasks marked to kill from coscheduling cascade
    assert len(txn.tasks_to_kill) == 0


def test_coscheduled_retriable_failure_does_not_kill_siblings(state):
    """When a coscheduled task fails but has retries remaining, siblings are NOT killed."""

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

    # Fail task-0 (first failure, has retry remaining)
    txn = transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_FAILED, error="OOM")

    # Task-0 failed but is retriable, requeued to PENDING
    assert tasks[0].state == job_pb2.TASK_STATE_PENDING
    assert check_task_can_be_scheduled(tasks[0])  # Can retry
    assert not check_task_is_finished(tasks[0])  # Not terminal

    # Siblings should still be running (no cascade for retriable failures)
    for task in tasks[1:]:
        assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_RUNNING

    # No tasks marked for kill
    assert len(txn.tasks_to_kill) == 0


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
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=worker_id,
            worker_resource_snapshot=None,
            updates=[
                TaskUpdate(
                    task_id=task.task_id,
                    attempt_id=old_attempt_id,
                    new_state=job_pb2.TASK_STATE_SUCCEEDED,
                )
            ],
        )
    )

    # Task should still be RUNNING on the new attempt
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_RUNNING
    assert _query_task(state, task.task_id).current_attempt_id == 1


def test_stale_attempt_error_log_for_non_terminal(state, caplog):
    """Stale attempt report logs ERROR when the old attempt is not terminal."""
    import logging

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
    _create_attempt(state._stores, task.task_id, worker_id)
    assert _query_task(state, task.task_id).current_attempt_id == 1
    # The old attempt (0) is still in RUNNING state (non-terminal)
    with state._db.snapshot() as q:
        attempts = ATTEMPT_PROJECTION.decode(
            q.fetchall("SELECT * FROM task_attempts WHERE task_id = ?", (task.task_id.to_wire(),))
        )
    assert not attempt_is_terminal(attempts[0].state)

    with caplog.at_level(logging.ERROR, logger="iris.cluster.controller.transitions"):
        state.apply_task_updates(
            HeartbeatApplyRequest(
                worker_id=worker_id,
                worker_resource_snapshot=None,
                updates=[TaskUpdate(task_id=task.task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_SUCCEEDED)],
            )
        )

    assert any("Stale attempt precondition violation" in r.message for r in caplog.records)


# =============================================================================
# Heartbeat Log Forwarding Tests
# =============================================================================


def test_log_service_direct_push(state, log_service):
    """Log entries pushed via LogService are queryable."""
    from iris.cluster.log_store import task_log_key
    from iris.cluster.types import TaskAttempt

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
    log_service.push_logs(push_req, None)

    fetch_resp = log_service.fetch_logs(logging_pb2.FetchLogsRequest(source=log_key), None)
    assert len(fetch_resp.entries) == 1
    assert fetch_resp.entries[0].data == "hello world"


def test_log_service_accumulates_pushes(state, log_service):
    """Multiple pushes accumulate logs in the service."""

    from iris.cluster.log_store import task_log_key
    from iris.cluster.types import TaskAttempt

    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    tasks = submit_job(state, "j1", make_job_request("job1"))
    task = tasks[0]
    dispatch_task(state, task, worker_id)

    attempt_id = _query_task(state, task.task_id).current_attempt_id
    log_key = task_log_key(TaskAttempt(task_id=task.task_id, attempt_id=attempt_id))

    for i in range(3):
        entry = logging_pb2.LogEntry(source="stdout", data=f"line {i}")
        entry.timestamp.epoch_ms = 1000 + i
        log_service.push_logs(logging_pb2.PushLogsRequest(key=log_key, entries=[entry]), None)

    fetch_resp = log_service.fetch_logs(logging_pb2.FetchLogsRequest(source=log_key), None)
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

    demand = compute_demand_entries(state._stores)
    assert len(demand) == 1
    assert demand[0].normalized.device_type == DeviceType.TPU
    assert demand[0].normalized.device_variants == frozenset({"v5litepod-16"})
    assert demand[0].task_ids == ["/test-user/j1/0", "/test-user/j1/1", "/test-user/j1/2", "/test-user/j1/3"]
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

    demand = compute_demand_entries(state._stores)
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

    demand = compute_demand_entries(state._stores)
    assert len(demand) == 3
    coscheduled = [entry for entry in demand if entry.coschedule_group_id == "/test-user/j1"]
    regular = [entry for entry in demand if entry.coschedule_group_id is None]
    assert len(coscheduled) == 1
    assert len(regular) == 2
    assert coscheduled[0].task_ids == ["/test-user/j1/0", "/test-user/j1/1", "/test-user/j1/2", "/test-user/j1/3"]
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

    demand = compute_demand_entries(state._stores)
    assert len(demand) == 2

    by_preemptible = {d.normalized.preemptible: d for d in demand}
    assert by_preemptible[True].normalized.device_type == DeviceType.TPU
    assert by_preemptible[True].task_ids == ["/test-user/j1/0"]
    assert by_preemptible[False].normalized.device_type == DeviceType.TPU
    assert by_preemptible[False].task_ids == ["/test-user/j2/0"]


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

    demand = compute_demand_entries(state._stores)
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

    demand = compute_demand_entries(state._stores)
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

    demand = compute_demand_entries(state._stores)
    assert len(demand) == 1
    assert demand[0].invalid_reason is not None


# =============================================================================
# Reservation Demand Deduplication Tests
# =============================================================================


def _make_reservation_make_job_request(
    *,
    task_device: job_pb2.DeviceConfig,
    reservation_devices: list[job_pb2.DeviceConfig],
    replicas: int = 1,
) -> controller_pb2.Controller.LaunchJobRequest:
    """Build a LaunchJobRequest with a reservation and task resources.

    Each reservation entry gets auto-generated constraints from its device
    config, mirroring what the service layer does for the top-level request.
    This ensures holder jobs get the correct device constraints from the
    entry, not from the parent.
    """
    req = controller_pb2.Controller.LaunchJobRequest(
        name="reservation-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=task_device,
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=replicas,
    )
    for dev in reservation_devices:
        entry_resources = job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=dev,
        )
        entry_constraints = [c.to_proto() for c in constraints_from_resources(entry_resources)]
        req.reservation.entries.append(
            job_pb2.ReservationEntry(
                resources=entry_resources,
                constraints=entry_constraints,
            )
        )
    return req


def _h100_device() -> job_pb2.DeviceConfig:
    return job_pb2.DeviceConfig(gpu=job_pb2.GpuDevice(variant="H100", count=8))


def _a100_device() -> job_pb2.DeviceConfig:
    return job_pb2.DeviceConfig(gpu=job_pb2.GpuDevice(variant="A100", count=8))


def _is_synthetic_demand(state: ControllerTransitions, demand_entry: DemandEntry) -> bool:
    """Check if a demand entry comes from a holder job task."""
    for tid in demand_entry.task_ids:
        task = _query_task(state, JobName.from_string(tid))
        if task:
            job = _query_job(state, task.job_id)
            if job and job.is_reservation_holder:
                return True
    return False


def test_demand_reservation_all_tasks_generate_demand(state):
    """2 H100 reservation + 2 H100 tasks = 4 total demand (no budget dedup).

    All tasks generate demand through a unified path. Holder tasks and real
    tasks are independent demand sources — preemption during scheduling
    (not demand) handles the dedup.
    """
    req = _make_reservation_make_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device(), _h100_device()],
        replicas=2,
    )
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state._stores)
    synthetic_demand = [d for d in demand if _is_synthetic_demand(state, d)]
    real_demand = [d for d in demand if not _is_synthetic_demand(state, d)]

    assert len(synthetic_demand) == 2
    assert len(real_demand) == 2


def test_demand_reservation_excess_tasks(state):
    """2 H100 reservation + 5 H100 tasks = 2 synthetic + 5 real task demand."""
    req = _make_reservation_make_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device(), _h100_device()],
        replicas=5,
    )
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state._stores)
    synthetic_demand = [d for d in demand if _is_synthetic_demand(state, d)]
    real_demand = [d for d in demand if not _is_synthetic_demand(state, d)]

    assert len(synthetic_demand) == 2
    assert len(real_demand) == 5


def test_demand_reservation_holder_uses_entry_resources(state):
    """Holder tasks use the reservation entry's resource spec, not the parent's.

    Each reservation entry carries its own resources and constraints. The
    holder job uses the entry's resources so the autoscaler provisions the
    correct device type even when the parent job differs.
    """
    # Job tasks request A100, but reservation entries specify H100.
    # Holder job should use the entry's H100 resource spec.
    req = _make_reservation_make_job_request(
        task_device=_a100_device(),
        reservation_devices=[_h100_device(), _h100_device()],
        replicas=2,
    )
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state._stores)
    synthetic_demand = [d for d in demand if _is_synthetic_demand(state, d)]
    real_demand = [d for d in demand if not _is_synthetic_demand(state, d)]

    assert len(synthetic_demand) == 2
    assert len(real_demand) == 2
    # Holder demand uses entry's H100 device, not parent's A100
    for d in synthetic_demand:
        assert d.normalized.device_variants == frozenset({"h100"})


def test_demand_reservation_mixed_jobs(state):
    """Reservation job + regular job: demand is independent per job."""

    # h100-job: 3 H100 tasks + 3 reservation entries
    h100_req = _make_reservation_make_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device(), _h100_device(), _h100_device()],
        replicas=3,
    )
    submit_job(state, "h100-job", h100_req)

    a100_req = controller_pb2.Controller.LaunchJobRequest(
        name="a100-job",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_a100_device(),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=2,
    )
    submit_job(state, "a100-job", a100_req)

    demand = compute_demand_entries(state._stores)
    synthetic_demand = [d for d in demand if _is_synthetic_demand(state, d)]
    real_demand = [d for d in demand if not _is_synthetic_demand(state, d)]

    # 3 synthetic holder tasks from h100-job's reservation
    assert len(synthetic_demand) == 3

    # h100-job: 3 real tasks + a100-job: 2 tasks = 5 real demand
    assert len(real_demand) == 5
    a100_demand = [d for d in real_demand if d.normalized.device_variants == frozenset({"a100"})]
    assert len(a100_demand) == 2


def test_demand_no_reservation_passes_all_tasks(state):
    """Job without reservation emits all task demand entries (no synthetic tasks)."""
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

    demand = compute_demand_entries(state._stores)
    assert len(demand) == 3
    for d in demand:
        assert not _is_synthetic_demand(state, d)


def test_demand_reservation_independent_per_job(state):
    """Each job's demand is independent — no cross-job interference."""

    # Job A: 2 H100 reservation, 2 H100 tasks
    job_a_req = _make_reservation_make_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device(), _h100_device()],
        replicas=2,
    )
    submit_job(state, "job-a", job_a_req)

    # Job B: no reservation, 2 H100 tasks (must all pass through)
    job_b_req = controller_pb2.Controller.LaunchJobRequest(
        name="job-b",
        entrypoint=_make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=2,
    )
    submit_job(state, "job-b", job_b_req)

    demand = compute_demand_entries(state._stores)
    synthetic_demand = [d for d in demand if _is_synthetic_demand(state, d)]
    real_demand = [d for d in demand if not _is_synthetic_demand(state, d)]

    # Job A's 2 synthetic holder tasks
    assert len(synthetic_demand) == 2
    # Job A's 2 real tasks + Job B's 2 tasks = 4 real demand
    assert len(real_demand) == 4


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
    parent_submitted = parent_job.submitted_at

    # Submit child later
    child_req = make_job_request("child")
    submit_job(state, "/test-user/parent/child", child_req, timestamp_ms=2000)
    child_job = _query_job(state, JobName.from_string("/test-user/parent/child"))

    # Child's root_submitted_at should equal parent's
    assert child_job.root_submitted_at == parent_submitted
    assert child_job.root_submitted_at == parent_job.root_submitted_at


def test_requeued_task_maintains_priority_position(state):
    """Requeued task maintains its priority position (deeper job still prioritized)."""

    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    # Submit a deep job and a shallow job
    submit_job(state, "/test-user/tree/deep", make_job_request("deep"), timestamp_ms=1000)
    submit_job(state, "shallow", make_job_request("shallow"), timestamp_ms=2000)

    # Initially: deep job comes first
    pending = _schedulable_tasks(state)
    assert len(pending) == 2
    assert pending[0].job_id == JobName.from_string("/test-user/tree/deep")
    assert pending[1].job_id == JobName.root("test-user", "shallow")

    # Dispatch and fail the deep job's task (with retries enabled)
    deep_req = make_job_request("deep")
    deep_req.max_retries_failure = 1
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


# =============================================================================
# Heartbeat Dispatch Transition Tests
# =============================================================================


def test_fail_heartbeat_clears_dispatch_when_worker_fails(state):
    """Dispatch buffer is cleared when worker hits failure threshold.

    When consecutive heartbeat failures hit the threshold:
    1. Worker is marked unhealthy
    2. Running tasks transition to WORKER_FAILED
    3. Pending dispatch buffer is cleared (not orphaned)
    """
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    # Submit and dispatch a task
    tasks = submit_job(state, "j1", make_job_request("job1"))
    dispatch_task(state, tasks[0], worker_id)

    # Buffer a dispatch for the worker
    fake_request = job_pb2.RunTaskRequest(task_id="/test-user/fake/0")
    state.buffer_dispatch(worker_id, fake_request)

    # Verify dispatch is buffered (assignment already buffered one run request).
    queued_run, queued_kill = _queued_dispatch(state, worker_id)
    assert not queued_kill
    assert len(queued_run) == 2

    # Create a snapshot (simulating begin_heartbeat)
    snapshot = state.begin_heartbeat(worker_id)
    assert snapshot is not None
    assert len(snapshot.tasks_to_run) == 2

    # Verify buffer is now drained
    queued_run, queued_kill = _queued_dispatch(state, worker_id)
    assert not queued_run
    assert not queued_kill

    # Simulate repeated failures up to threshold
    _set_worker_consecutive_failures(state._db, worker_id, HEARTBEAT_FAILURE_THRESHOLD - 1)

    # This fail_heartbeat should trigger worker failure
    state.fail_heartbeat(snapshot, "Connection refused")

    # Verify worker is now unhealthy
    worker = _query_worker(state, worker_id)
    assert worker is None

    # Verify dispatch buffer was NOT repopulated (would be orphaned)
    # The fix clears the dispatch buffer when worker fails
    queued_run, _ = _queued_dispatch(state, worker_id)
    assert not queued_run


def test_fail_heartbeat_requeues_dispatch_for_retry(state):
    """Heartbeat failure re-queues dispatches for the next heartbeat.

    We cannot tell whether the worker received the previous heartbeat (RPC
    timeout ≠ delivery failure), so we re-send the same RunTaskRequests.
    If the worker did receive them, it will reject re-sends as benign
    duplicates. If it did not, it will start them fresh.
    """

    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = make_job_request("job1")
    req.max_retries_preemption = 5
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    # Assign the task (creates attempt, commits resources)
    state.queue_assignments([Assignment(task_id=task.task_id, worker_id=worker_id)])
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_ASSIGNED
    assert _query_task(state, task.task_id).current_attempt_id == 0

    # Take snapshot (drains buffer)
    snapshot = state.begin_heartbeat(worker_id)
    assert snapshot is not None
    assert len(snapshot.tasks_to_run) == 1

    # Fail heartbeat (worker stays healthy - below threshold)
    state.fail_heartbeat(snapshot, "Timeout")

    worker = _query_worker(state, worker_id)
    assert _query_worker(state, worker.worker_id).healthy
    assert worker.consecutive_failures == 1

    # Task stays ASSIGNED — we don't know if the worker received it
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_ASSIGNED
    assert _query_task(state, task.task_id).preemption_count == 0
    assert _query_task(state, task.task_id).failure_count == 0

    # Dispatch re-queued for the next heartbeat (same attempt_id)
    queued_run, queued_kill = _queued_dispatch(state, worker_id)
    assert not queued_kill
    assert len(queued_run) == 1
    assert queued_run[0].attempt_id == 0


def test_complete_heartbeat_processes_task_states(state):
    """complete_heartbeat properly processes task state changes from response."""
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    # Submit and dispatch a task
    tasks = submit_job(state, "j1", make_job_request("job1"))
    dispatch_task(state, tasks[0], worker_id)

    # Take snapshot
    snapshot = state.begin_heartbeat(worker_id)
    assert snapshot is not None

    # Create a mock response with completed task
    response = job_pb2.HeartbeatResponse(
        worker_healthy=True,
        tasks=[
            job_pb2.WorkerTaskStatus(
                task_id=tasks[0].task_id.to_wire(),
                state=job_pb2.TASK_STATE_SUCCEEDED,
                exit_code=0,
                attempt_id=0,
            )
        ],
    )

    # Complete heartbeat
    state.complete_heartbeat(snapshot, response)

    # Verify task is now succeeded
    task = _query_task(state, tasks[0].task_id)
    assert _query_task(state, task.task_id).state == job_pb2.TASK_STATE_SUCCEEDED

    # Verify job is succeeded
    job = _query_job(state, tasks[0].job_id)
    assert _query_job(state, job.job_id).state == job_pb2.JOB_STATE_SUCCEEDED


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
    state.queue_assignments([Assignment(task_id=task.task_id, worker_id=worker_id)])
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
    state.queue_assignments([Assignment(task_id=task.task_id, worker_id=worker_id)])
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

    result = state.fail_workers_batch(["w2"], reason="slice terminated")

    assert len(result.removed_workers) == 1
    assert result.removed_workers[0][0] == w2
    assert result.removed_workers[0][1] == "host2:8080"
    assert len(result.results) == 1
    assert result.results[0].action == HeartbeatAction.WORKER_FAILED

    t2 = _query_task(state, tasks2[0].task_id)
    assert t2.state in (job_pb2.TASK_STATE_WORKER_FAILED, job_pb2.TASK_STATE_PENDING)

    assert _query_task(state, tasks1[0].task_id).state == job_pb2.TASK_STATE_RUNNING
    assert _query_worker(state, w1) is not None
    assert _query_worker(state, w2) is None


def test_fail_workers_batch_skips_unknown(state):
    """fail_workers_batch returns empty for unknown worker IDs."""
    meta = make_worker_metadata()
    register_worker(state, "w1", "host1:8080", meta)

    result = state.fail_workers_batch(["w-unknown"], reason="unknown")
    assert result.removed_workers == []

    w = _query_worker(state, WorkerId("w1"))
    assert w is not None
    assert w.healthy


def test_fail_workers_batch_force_removes_without_threshold(state):
    """fail_workers_batch removes targets immediately instead of incrementing failures."""
    meta = make_worker_metadata()
    worker_id = register_worker(state, "w1", "host1:8080", meta)

    result = state.fail_workers_batch(["w1"], reason="slice terminated")

    assert len(result.results) == 1
    assert result.results[0].action == HeartbeatAction.WORKER_FAILED
    assert _query_worker(state, worker_id) is None


def test_fail_workers_batch_does_not_block_readers(state):
    """fail_workers_batch uses read_snapshot for lookups, so concurrent reads don't block.

    Verifies that read_snapshot() (not write-locked snapshot()) is used for the
    worker lookup query. We hold a write transaction open on a second thread while
    calling fail_workers_batch from the main thread; if the lookup used
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
        """Hold the DB write lock to prove fail_workers_batch doesn't need it for reads."""
        with state._db.transaction():
            barrier.set()
            done.wait(timeout=5)

    t = threading.Thread(target=hold_write_lock, daemon=True)
    t.start()
    barrier.wait(timeout=5)

    # fail_workers_batch should still complete even though the write lock is held,
    # because its lookup query uses read_snapshot (WAL reader).
    # Note: the actual fail_heartbeat_for_worker inside will need the write lock
    # for its transaction, so we test with unknown IDs to isolate the read path.
    result = state.fail_workers_batch(["w-nonexistent"], reason="test")
    assert result.removed_workers == []

    done.set()
    t.join(timeout=5)


def test_fail_heartbeat_kills_requeue_only(state):
    """Kill requests are still requeued on heartbeat failure (idempotent)."""
    worker_id = register_worker(state, "w1", "host:8080", make_worker_metadata())

    tasks = submit_job(state, "j1", make_job_request("job1"))
    dispatch_task(state, tasks[0], worker_id)

    # Buffer a kill
    state.buffer_kill(worker_id, tasks[0].task_id.to_wire())

    snapshot = state.begin_heartbeat(worker_id)
    assert snapshot is not None
    assert len(snapshot.tasks_to_kill) == 1

    # Fail heartbeat
    state.fail_heartbeat(snapshot, "Timeout")

    worker = _query_worker(state, worker_id)
    assert _query_worker(state, worker.worker_id).healthy

    # Kills should be requeued
    _, queued_kill = _queued_dispatch(state, worker_id)
    assert len(queued_kill) == 1


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
    scheduler = Scheduler(max_building_tasks_per_worker=2)

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
    workers = healthy_active_workers(state)
    demand = compute_demand_entries(state._stores, scheduler, workers)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]
    assert len(task_demand) == 0, "Building-limited task should not generate demand"


def test_demand_includes_truly_unschedulable_tasks(state):
    """No worker with matching device type -> demand IS emitted."""
    scheduler = Scheduler()

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

    workers = healthy_active_workers(state)
    demand = compute_demand_entries(state._stores, scheduler, workers)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]
    assert len(task_demand) == 1, "Task with no matching device should generate demand"


def test_demand_includes_resource_exhausted_tasks(state):
    """Worker has right device but insufficient CPU -> demand IS emitted."""
    scheduler = Scheduler()

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

    workers = healthy_active_workers(state)
    demand = compute_demand_entries(state._stores, scheduler, workers)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]
    assert len(task_demand) == 1, "Task exceeding worker CPU should generate demand"


def test_demand_holders_absorbed_by_dry_run(state):
    """Holder tasks participate in the dry-run and are absorbed when workers exist.

    Unlike the old design where holders always generated demand, they now
    participate in the dry-run like normal tasks and are absorbed when matching
    workers have available capacity.
    """
    scheduler = Scheduler()

    # Register a large GPU worker with capacity for 1 task
    register_worker(state, "w1", "10.0.0.1:8080", _gpu_make_worker_metadata(cpu=2, memory_gb=4))

    # Submit a job with reservation (2 entries) and 2 tasks.
    # Worker can fit 1 task — so 1 task absorbed, 3 remain as demand.
    req = _make_reservation_make_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device(), _h100_device()],
        replicas=2,
    )
    submit_job(state, "j1", req)

    workers = healthy_active_workers(state)
    demand = compute_demand_entries(state._stores, scheduler, workers)
    # Worker fits 1 task (holder or real). 3 remaining generate demand.
    assert len(demand) == 3


def test_demand_absorbs_capacity_before_emitting(state):
    """2 workers fit 1 task each, 3 pending tasks -> only 1 demand entry."""
    scheduler = Scheduler()

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

    workers = healthy_active_workers(state)
    demand = compute_demand_entries(state._stores, scheduler, workers)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]
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
    demand = compute_demand_entries(state._stores)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]
    assert len(task_demand) == 3


def test_demand_building_limited_with_multiple_workers(state):
    """All matching workers at building limit -> no demand, even with multiple workers."""
    scheduler = Scheduler(max_building_tasks_per_worker=1)

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

    workers = healthy_active_workers(state)
    demand = compute_demand_entries(state._stores, scheduler, workers)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]
    assert len(task_demand) == 0, "All workers at building limit -> no demand"


def test_demand_mixed_building_limited_and_unschedulable(state):
    """Some tasks building-limited, some truly unschedulable -> only unschedulable emit demand."""
    scheduler = Scheduler(max_building_tasks_per_worker=1)

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

    workers = healthy_active_workers(state)
    demand = compute_demand_entries(state._stores, scheduler, workers)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]

    assert len(task_demand) == 1
    assert "a100-job" in task_demand[0].task_ids[0], "Only A100 task should emit demand"


# =============================================================================
# Holder Task Zero-Resource Tests
# =============================================================================


def test_holder_tasks_consume_zero_resources(state):
    """Holder tasks consume zero resources when assigned to workers."""

    req = _make_reservation_make_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device()],
        replicas=1,
    )
    submit_job(state, "j1", req)

    wid = register_worker(state, "w1", "10.0.0.1:8080", _gpu_make_worker_metadata())
    holder_job_id = JobName.root("test-user", "j1").child(":reservation:")
    holder_tasks = _query_tasks_for_job(state, holder_job_id)
    assert len(holder_tasks) == 1

    worker_before = _query_worker(state, wid)
    gpus_before = worker_before.total_gpu_count - worker_before.committed_gpu

    # Assign holder task
    state.queue_assignments([Assignment(task_id=holder_tasks[0].task_id, worker_id=wid)])

    # Worker's available GPUs should NOT decrease (zero resources)
    worker_after = _query_worker(state, wid)
    assert worker_after.total_gpu_count - worker_after.committed_gpu == gpus_before

    # But the task should be tracked in running_tasks
    assert holder_tasks[0].task_id in worker_running_tasks(state, wid)


def test_holder_task_cleanup_releases_no_resources(state):
    """When a holder task finishes, it doesn't release resources it never committed."""

    req = _make_reservation_make_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device()],
        replicas=1,
    )
    submit_job(state, "j1", req)

    wid = register_worker(state, "w1", "10.0.0.1:8080", _gpu_make_worker_metadata())
    holder_job_id = JobName.root("test-user", "j1").child(":reservation:")
    holder_tasks = _query_tasks_for_job(state, holder_job_id)

    # Assign holder task
    state.queue_assignments([Assignment(task_id=holder_tasks[0].task_id, worker_id=wid)])

    worker_before = _query_worker(state, wid)
    gpus_before = worker_before.total_gpu_count - worker_before.committed_gpu

    # Kill the holder task via parent job cancellation
    parent_job_id = JobName.root("test-user", "j1")
    state.cancel_job(parent_job_id, reason="test")

    # Worker GPUs should be unchanged (nothing to release)
    worker_after = _query_worker(state, wid)
    assert worker_after.total_gpu_count - worker_after.committed_gpu == gpus_before


def test_holder_tasks_excluded_from_building_counts(state):
    """Holder tasks in ASSIGNED state should not consume building slots.

    Without this exclusion, a worker holding only a reservation task would be
    permanently "at building limit" and the real reserved task could never be
    assigned to that otherwise idle worker.
    """

    req = _make_reservation_make_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device()],
        replicas=1,
    )
    submit_job(state, "j1", req)

    wid = register_worker(state, "w1", "10.0.0.1:8080", _gpu_make_worker_metadata())
    holder_job_id = JobName.root("test-user", "j1").child(":reservation:")
    holder_tasks = _query_tasks_for_job(state, holder_job_id)
    assert len(holder_tasks) == 1

    # Assign holder task — it goes to ASSIGNED state
    state.queue_assignments([Assignment(task_id=holder_tasks[0].task_id, worker_id=wid)])
    assert _query_task(state, holder_tasks[0].task_id).state == job_pb2.TASK_STATE_ASSIGNED

    # Building counts should NOT include the holder task
    building_counts = _building_counts(state)
    assert building_counts.get(wid, 0) == 0


def test_holder_tasks_excluded_from_heartbeat_expected_tasks(state):
    """Holder tasks must not appear in heartbeat expected_tasks.

    Holder tasks are virtual — never dispatched to the worker. If included
    in expected_tasks the worker reports "Task not found on worker", causing
    a worker_failed → retry loop (GH-3178).
    """

    req = _make_reservation_make_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device()],
        replicas=1,
    )
    submit_job(state, "j1", req)

    wid = register_worker(state, "w1", "10.0.0.1:8080", _gpu_make_worker_metadata())
    holder_job_id = JobName.root("test-user", "j1").child(":reservation:")
    holder_tasks = _query_tasks_for_job(state, holder_job_id)
    assert len(holder_tasks) == 1

    # Assign holder task to worker
    state.queue_assignments([Assignment(task_id=holder_tasks[0].task_id, worker_id=wid)])

    # Heartbeat snapshot must NOT include the holder task
    snapshot = state.begin_heartbeat(wid)
    assert snapshot is not None
    running_task_ids = {entry.task_id for entry in snapshot.running_tasks}
    assert holder_tasks[0].task_id not in running_task_ids


def test_snapshot_round_trip_preserves_reservation_holder(state):
    """DB checkpoint copy round-trip preserves is_reservation_holder flag."""
    import tempfile
    from pathlib import Path

    req = _make_reservation_make_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device()],
        replicas=1,
    )
    submit_job(state, "j1", req)

    holder_job_id = JobName.root("test-user", "j1").child(":reservation:")
    holder_job = _query_job(state, holder_job_id)
    assert holder_job is not None
    assert holder_job.is_reservation_holder is True

    # Save and restore
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "controller.sqlite3"
        state._db.backup_to(checkpoint_path)
        restored_db = ControllerDB(db_dir=Path(tmpdir))
        from iris.cluster.controller.store import ControllerStores

        restored_stores = ControllerStores.from_db(restored_db)
        restored_state = ControllerTransitions(stores=restored_stores)

        restored_holder = _query_job(restored_state, holder_job_id)
        assert restored_holder is not None
        assert restored_holder.is_reservation_holder is True

        # Parent should not be a holder
        parent_job_id = JobName.root("test-user", "j1")
        restored_parent = _query_job(restored_state, parent_job_id)
        assert restored_parent is not None
        assert restored_parent.is_reservation_holder is False


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
    """apply_task_updates with WORKER_FAILED terminal task cascades children via preemption policy."""
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

    # Report WORKER_FAILED via heartbeat update (goes through apply_task_updates)
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


def test_endpoint_registered_after_task_terminal_is_orphaned(state):
    """Reproduce endpoint leak: register_endpoint succeeds for already-terminal tasks.

    When a task completes, apply_task_updates deletes its endpoints. But
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
    state.add_endpoint(ep)

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
    state._db.execute(
        "UPDATE jobs SET finished_at_ms = 1000 WHERE job_id = ?",
        (old_job_id.to_wire(),),
    )

    # All three jobs exist
    assert _query_job(state, old_job_id) is not None
    assert _query_job(state, recent_job_id) is not None
    assert _query_job(state, active_job_id) is not None

    # Prune with a 1-day retention — old-job finished at ~epoch, recent-job finished just now
    result = state.prune_old_data(
        job_retention=Duration.from_seconds(86400),
        worker_retention=Duration.from_seconds(86400),
        txn_action_retention=Duration.from_seconds(86400),
        profile_retention=Duration.from_seconds(86400),
    )

    assert result.jobs_deleted == 1
    assert _query_job(state, old_job_id) is None  # pruned
    assert _query_job(state, recent_job_id) is not None  # kept (recent)
    assert _query_job(state, active_job_id) is not None  # kept (non-terminal)

    # Tasks for old job should also be gone (CASCADE)
    assert _query_task(state, old_tasks[0].task_id) is None


def test_prune_old_inactive_workers(state):
    """Inactive workers with stale heartbeats are pruned; active workers are kept."""

    # Register two workers: one healthy, one that we'll make inactive
    active_wid = register_worker(state, "active-w", "host:8080", make_worker_metadata())
    stale_wid = register_worker(state, "stale-w", "host:8081", make_worker_metadata())

    # Mark the stale worker as unhealthy with an old heartbeat
    state._db.execute(
        "UPDATE workers SET healthy = 0, last_heartbeat_ms = ? WHERE worker_id = ?",
        (1000, str(stale_wid)),
    )

    assert _query_worker(state, active_wid) is not None
    assert _query_worker(state, stale_wid) is not None

    result = state.prune_old_data(
        job_retention=Duration.from_seconds(86400),
        worker_retention=Duration.from_seconds(86400),
        txn_action_retention=Duration.from_seconds(86400),
        profile_retention=Duration.from_seconds(86400),
    )

    assert result.workers_deleted == 1
    assert _query_worker(state, active_wid) is not None  # kept (healthy+active)
    assert _query_worker(state, stale_wid) is None  # pruned


def test_prune_old_txn_actions(state):
    """Old txn_actions are pruned by the txn_action retention."""
    register_worker(state, "w1", "host:8080", make_worker_metadata())

    # Submit a job to generate txn_actions, then backdate some
    req = make_job_request("txn-test")
    submit_job(state, "txn-test", req)

    # Backdate all existing txn_actions to epoch
    state._db.execute("UPDATE txn_actions SET created_at_ms = 1000")

    old_txn_count = state._db.fetchone("SELECT COUNT(*) as c FROM txn_actions")["c"]

    assert old_txn_count > 0

    result = state.prune_old_data(
        job_retention=Duration.from_seconds(86400),
        worker_retention=Duration.from_seconds(86400),
        txn_action_retention=Duration.from_seconds(86400),
        profile_retention=Duration.from_seconds(86400),
    )

    assert result.txn_actions_deleted == old_txn_count

    remaining_txn_actions = state._db.fetchone("SELECT COUNT(*) as c FROM txn_actions")["c"]

    # Incremental prune deletes old txn_actions in batches; no new aggregate
    # action rows are recorded for txn_action cleanup.
    assert remaining_txn_actions == 0


def test_prune_noop_when_nothing_old(state):
    """Pruning with no old data returns zero counts."""

    result = state.prune_old_data(
        job_retention=Duration.from_seconds(86400),
        worker_retention=Duration.from_seconds(86400),
        txn_action_retention=Duration.from_seconds(86400),
        profile_retention=Duration.from_seconds(86400),
    )

    assert result == PruneResult()
    assert result.total == 0


# =============================================================================
# drain_dispatch_all Tests
# =============================================================================


def test_drain_dispatch_all_excludes_reservation_holders(state):
    """drain_dispatch_all returns running tasks but filters out reservation-holder tasks."""
    wid = register_worker(state, "w1", "host:8080", make_worker_metadata())

    normal_req = make_job_request("normal-job")
    normal_tasks = submit_job(state, "normal-job", normal_req)
    dispatch_task(state, normal_tasks[0], wid)

    holder_req = make_job_request("holder-job")
    holder_tasks = submit_job(state, "holder-job", holder_req)
    holder_job_id = JobName.root("test-user", "holder-job")
    state._db.execute(
        "UPDATE jobs SET is_reservation_holder = 1 WHERE job_id = ?",
        (holder_job_id.to_wire(),),
    )
    dispatch_task(state, holder_tasks[0], wid)

    batches = state.drain_dispatch_all()
    assert len(batches) == 1
    batch = batches[0]
    running_task_ids = {entry.task_id for entry in batch.running_tasks}

    assert normal_tasks[0].task_id in running_task_ids
    assert holder_tasks[0].task_id not in running_task_ids


def test_drain_dispatch_all_drains_dispatch_queue(state):
    """drain_dispatch_all drains queued dispatches and deletes them from the queue."""
    wid = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = make_job_request("j1")
    tasks = submit_job(state, "j1", req)
    state.queue_assignments([Assignment(task_id=tasks[0].task_id, worker_id=wid)])

    rows_before = state._db.fetchall("SELECT * FROM dispatch_queue WHERE worker_id = ?", (str(wid),))
    assert len(rows_before) > 0

    batches = state.drain_dispatch_all()
    assert len(batches) == 1
    assert len(batches[0].tasks_to_run) > 0

    rows_after = state._db.fetchall("SELECT * FROM dispatch_queue WHERE worker_id = ?", (str(wid),))
    assert len(rows_after) == 0


def test_dispatch_propagates_task_image(state):
    """task_image set on the LaunchJobRequest is copied into the dispatched RunTaskRequest."""
    wid = register_worker(state, "w1", "host:8080", make_worker_metadata())

    req = make_job_request("img-job", task_image="custom/swetrace:dev")
    tasks = submit_job(state, "img-job", req)
    state.queue_assignments([Assignment(task_id=tasks[0].task_id, worker_id=wid)])

    batches = state.drain_dispatch_all()
    assert len(batches) == 1
    assert len(batches[0].tasks_to_run) == 1
    assert batches[0].tasks_to_run[0].task_image == "custom/swetrace:dev"


def test_prune_old_data_short_circuits_when_nothing_prunable(state):
    """prune_old_data skips the write lock when a read_snapshot shows nothing to prune."""
    wid = register_worker(state, "w1", "host:8080", make_worker_metadata())
    req = make_job_request("active-job")
    tasks = submit_job(state, "active-job", req)
    dispatch_task(state, tasks[0], wid)

    result = state.prune_old_data(
        job_retention=Duration.from_seconds(86400),
        worker_retention=Duration.from_seconds(86400),
        txn_action_retention=Duration.from_seconds(86400),
        profile_retention=Duration.from_seconds(86400),
    )

    assert result == PruneResult()
    assert result.total == 0


# =============================================================================
# Direct Provider Transition Tests
# =============================================================================


def _submit_job_direct(
    state: ControllerTransitions,
    job_id_str: str,
    *,
    replicas: int = 1,
    max_retries_failure: int = 0,
    max_retries_preemption: int = 0,
) -> list[JobName]:
    job_id = JobName.from_wire(job_id_str)
    request = controller_pb2.Controller.LaunchJobRequest(
        name="test-job",
        replicas=replicas,
        max_retries_failure=max_retries_failure,
        max_retries_preemption=max_retries_preemption,
    )
    result = state.submit_job(job_id, request, Timestamp.now())
    return result.task_ids


def _task_state_direct(state: ControllerTransitions, task_id: JobName) -> int:
    with state._db.snapshot() as q:
        tasks = TASK_DETAIL_PROJECTION.decode(q.fetchall("SELECT * FROM tasks WHERE task_id = ?", (task_id.to_wire(),)))
    assert len(tasks) == 1
    return tasks[0].state


def _task_row_direct(state: ControllerTransitions, task_id: JobName):
    with state._db.snapshot() as q:
        tasks = TASK_DETAIL_PROJECTION.decode(q.fetchall("SELECT * FROM tasks WHERE task_id = ?", (task_id.to_wire(),)))
    assert len(tasks) == 1
    return tasks[0]


def _run_direct_tasks(state: ControllerTransitions, task_ids: list[JobName]) -> None:
    """Drain and transition tasks to RUNNING via direct provider."""
    state.drain_for_direct_provider()
    state.apply_direct_provider_updates(
        [TaskUpdate(task_id=t, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING) for t in task_ids]
    )


def test_drain_pending_creates_attempt_rows(state):
    """drain_for_direct_provider promotes PENDING tasks to ASSIGNED with NULL worker_id."""
    task_ids = _submit_job_direct(state, "/user/job1")
    task_id = task_ids[0]

    batch = state.drain_for_direct_provider()

    assert len(batch.tasks_to_run) == 1
    assert batch.tasks_to_run[0].task_id == task_id.to_wire()
    assert batch.tasks_to_run[0].attempt_id == 0
    assert _task_state_direct(state, task_id) == job_pb2.TASK_STATE_ASSIGNED

    # Verify attempt row was created with NULL worker_id.
    row = state._db.fetchone(
        "SELECT worker_id, state FROM task_attempts WHERE task_id = ? AND attempt_id = 0",
        (task_id.to_wire(),),
    )
    assert row is not None
    assert row["worker_id"] is None
    assert int(row["state"]) == job_pb2.TASK_STATE_ASSIGNED


def test_drain_skips_already_assigned(state):
    """Already ASSIGNED tasks appear in running_tasks, not tasks_to_run."""
    task_ids = _submit_job_direct(state, "/user/job1")
    task_id = task_ids[0]

    # First drain promotes to ASSIGNED.
    batch1 = state.drain_for_direct_provider()
    assert len(batch1.tasks_to_run) == 1

    # Second drain: no new tasks to run, but task appears in running_tasks.
    batch2 = state.drain_for_direct_provider()
    assert len(batch2.tasks_to_run) == 0
    assert len(batch2.running_tasks) == 1
    assert batch2.running_tasks[0].task_id == task_id
    assert batch2.running_tasks[0].attempt_id == 0


def test_drain_caps_promotions_per_cycle(state):
    """Promotions are capped by max_promotions per drain call."""
    _submit_job_direct(state, "/user/big-job", replicas=200)

    batch1 = state.drain_for_direct_provider(max_promotions=128)
    assert len(batch1.tasks_to_run) == 128

    # Remaining tasks promoted with another budget.
    batch2 = state.drain_for_direct_provider(max_promotions=128)
    assert len(batch2.tasks_to_run) == 72


def test_drain_max_promotions_limits_batch(state):
    """max_promotions caps the number of tasks promoted per cycle."""
    _submit_job_direct(state, "/user/cap-job", replicas=250)

    batch1 = state.drain_for_direct_provider(max_promotions=50)
    assert len(batch1.tasks_to_run) == 50

    # Remaining tasks still available with a fresh budget.
    batch2 = state.drain_for_direct_provider(max_promotions=50)
    assert len(batch2.tasks_to_run) == 50


def test_drain_kill_queue(state):
    """Buffered kill entries are drained and deleted."""
    task_ids = _submit_job_direct(state, "/user/job1")
    task_id = task_ids[0]

    state.buffer_direct_kill(task_id.to_wire())

    batch = state.drain_for_direct_provider()
    assert task_id.to_wire() in batch.tasks_to_kill

    # Second drain should be empty (kills were consumed).
    batch2 = state.drain_for_direct_provider()
    assert len(batch2.tasks_to_kill) == 0


def test_apply_running(state):
    """Applying a RUNNING update transitions task from ASSIGNED to RUNNING."""
    task_ids = _submit_job_direct(state, "/user/job1")
    task_id = task_ids[0]
    state.drain_for_direct_provider()

    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING),
        ]
    )

    assert _task_state_direct(state, task_id) == job_pb2.TASK_STATE_RUNNING


def test_apply_succeeded(state):
    """Applying SUCCEEDED transitions task to terminal state with exit_code=0."""
    task_ids = _submit_job_direct(state, "/user/job1")
    task_id = task_ids[0]
    state.drain_for_direct_provider()

    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING),
        ]
    )
    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_SUCCEEDED),
        ]
    )

    task = _task_row_direct(state, task_id)
    assert task.state == job_pb2.TASK_STATE_SUCCEEDED
    assert task.exit_code == 0
    assert task.finished_at is not None


def test_apply_failed_with_retry(state):
    """FAILED with retries remaining returns task to PENDING."""
    task_ids = _submit_job_direct(state, "/user/job1", max_retries_failure=1)
    task_id = task_ids[0]
    state.drain_for_direct_provider()

    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING),
        ]
    )
    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_FAILED, error="boom"),
        ]
    )

    # Task should be PENDING again (1 failure <= 1 max_retries_failure).
    assert _task_state_direct(state, task_id) == job_pb2.TASK_STATE_PENDING

    # The dead attempt 0 must have finished_at_ms stamped even though the task
    # itself rolled back to PENDING. Otherwise the row is indistinguishable from
    # a still-assigned attempt. Regression guard for the terminal_ms conflation.
    with state._db.snapshot() as q:
        attempts = ATTEMPT_PROJECTION.decode(
            q.fetchall("SELECT * FROM task_attempts WHERE task_id = ?", (task_id.to_wire(),))
        )
    assert len(attempts) == 1
    assert attempts[0].state == job_pb2.TASK_STATE_FAILED
    assert attempts[0].finished_at is not None

    # Draining again should promote it for a second attempt.
    batch = state.drain_for_direct_provider()
    assert len(batch.tasks_to_run) == 1
    assert batch.tasks_to_run[0].attempt_id == 1


def test_apply_failed_no_retry(state):
    """FAILED with no retries remaining leaves task in FAILED terminal state."""
    task_ids = _submit_job_direct(state, "/user/job1", max_retries_failure=0)
    task_id = task_ids[0]
    state.drain_for_direct_provider()

    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING),
        ]
    )
    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_FAILED, error="boom"),
        ]
    )

    task = _task_row_direct(state, task_id)
    assert task.state == job_pb2.TASK_STATE_FAILED
    assert task.failure_count == 1
    assert task.finished_at is not None


def test_apply_worker_failed(state):
    """WORKER_FAILED on a RUNNING task increments preemption_count and retries if allowed."""
    task_ids = _submit_job_direct(state, "/user/job1", max_retries_preemption=1)
    task_id = task_ids[0]
    state.drain_for_direct_provider()

    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING),
        ]
    )
    state.apply_direct_provider_updates(
        [
            TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_WORKER_FAILED, error="node died"),
        ]
    )

    # Should be retried (preemption_count=1 <= max_retries_preemption=1).
    assert _task_state_direct(state, task_id) == job_pb2.TASK_STATE_PENDING
    task = _task_row_direct(state, task_id)
    assert task.preemption_count == 1


def test_buffer_direct_kill(state):
    """buffer_direct_kill inserts a NULL-worker_id kill entry in dispatch_queue."""
    state.buffer_direct_kill("/user/job1:task-0")

    row = state._db.fetchone(
        "SELECT worker_id, kind, task_id FROM dispatch_queue WHERE task_id = ?",
        ("/user/job1:task-0",),
    )
    assert row is not None
    assert row["worker_id"] is None
    assert row["kind"] == "kill"


def test_cancel_job_kills_direct_provider_tasks(state):
    """cancel_job includes NULL-worker_id (direct-provider) tasks in tasks_to_kill."""
    task_ids = _submit_job_direct(state, "/user/job1", replicas=2)
    _run_direct_tasks(state, task_ids)

    result = state.cancel_job(JobName.from_wire("/user/job1"), reason="test cancel")

    assert result.tasks_to_kill == set(task_ids)


def test_kill_non_terminal_direct_provider_tasks(state):
    """_kill_non_terminal_tasks includes NULL-worker_id tasks in tasks_to_kill."""
    task_ids = _submit_job_direct(state, "/user/job1")
    _run_direct_tasks(state, task_ids)

    # Trigger via cancel_job which calls _kill_non_terminal_tasks indirectly through
    # cascade, or call it via a job failure path. Use cancel_job for simplicity.
    result = state.cancel_job(JobName.from_wire("/user/job1"), reason="test kill")

    assert task_ids[0] in result.tasks_to_kill


def test_max_failures_kills_direct_provider_tasks(state):
    """When a task fails and triggers kill of siblings, direct-provider tasks appear in tasks_to_kill."""
    task_ids = _submit_job_direct(state, "/user/job1", replicas=2, max_retries_failure=0)
    _run_direct_tasks(state, task_ids)

    # Fail one task — with max_task_failures=0 (default) this should kill the job,
    # triggering _kill_non_terminal_tasks for the sibling.
    result = state.apply_direct_provider_updates(
        [TaskUpdate(task_id=task_ids[0], attempt_id=0, new_state=job_pb2.TASK_STATE_FAILED, error="boom")]
    )

    # The sibling task (task_ids[1]) should be in tasks_to_kill.
    assert task_ids[1] in result.tasks_to_kill


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
    harness.state.mark_task_unschedulable(tasks[0].task_id, reason="no capacity")
    assert harness.query_job(JobName.root("test-user", "unsched")).state == job_pb2.JOB_STATE_UNSCHEDULABLE


def test_job_cancel_marks_job_killed(harness) -> None:
    harness.submit("killed", replicas=2)
    jid = JobName.root("test-user", "killed")
    harness.state.cancel_job(jid, reason="manual")
    assert harness.query_job(jid).state == job_pb2.JOB_STATE_KILLED
