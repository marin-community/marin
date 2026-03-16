# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for controller state management.

These tests exercise end-to-end observable behavior through the event-driven API (handle_event).
They focus on:
- Full workflows (submit job -> dispatch tasks -> complete/fail)
- Important edge cases (retry exhaustion, worker failure cascades, failure domains)
- Final state verification rather than intermediate steps
"""

import tempfile
import threading
from pathlib import Path

import pytest
from iris.cluster.controller.autoscaler import DemandEntry
from iris.cluster.controller.controller import compute_demand_entries
from iris.cluster.controller.db import (
    ACTIVE_TASK_STATES,
    ATTEMPTS,
    ControllerDB,
    JOBS,
    TASKS,
    TERMINAL_TASK_STATES,
    WORKERS,
    WORKER_ATTRIBUTES,
    Endpoint,
    EndpointQuery,
    Job,
    Task,
    Worker,
    _decode_attribute_rows,
    _tasks_with_attempts,
    endpoint_query_predicate,
    ENDPOINTS,
)
from iris.cluster.controller.scheduler import JobRequirements, Scheduler
from iris.cluster.controller.transitions import (
    MAX_REPLICAS_PER_JOB,
    Assignment,
    ControllerTransitions,
    HEARTBEAT_FAILURE_THRESHOLD,
    HeartbeatApplyRequest,
    TaskUpdate,
)
from iris.cluster.constraints import WellKnownAttribute, constraints_from_resources
from iris.cluster.constraints import DeviceType
from iris.cluster.types import JobName, WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import Duration, Timestamp

# =============================================================================
# Test Helpers
# =============================================================================


def _make_state(**kwargs) -> ControllerTransitions:
    """Create a ControllerTransitions with a fresh temp DB."""
    tmp = Path(tempfile.mkdtemp(prefix="iris_test_"))
    db_path = tmp / "controller.sqlite3"
    db = ControllerDB(db_path=db_path)
    return ControllerTransitions(db=db, **kwargs)


def _schedulable_tasks(state: ControllerTransitions):
    with state._db.snapshot() as q:
        tasks = q.select(
            TASKS,
            where=TASKS.c.state.not_null() & ~TASKS.c.state.in_(list(TERMINAL_TASK_STATES)),
            order_by=(
                TASKS.c.priority_neg_depth.asc(),
                TASKS.c.priority_root_submitted_ms.asc(),
                TASKS.c.submitted_at_ms.asc(),
                TASKS.c.task_id.asc(),
            ),
        )
    return [t for t in tasks if t.can_be_scheduled()]


def _worker_running_tasks(state: ControllerTransitions, worker_id: WorkerId):
    with state._db.snapshot() as q:
        rows = q.raw(
            "SELECT t.task_id FROM tasks t "
            "JOIN task_attempts a ON t.task_id = a.task_id AND t.current_attempt_id = a.attempt_id "
            "WHERE a.worker_id = ? AND t.state IN (?, ?, ?)",
            (str(worker_id), *ACTIVE_TASK_STATES),
            decoders={"task_id": JobName.from_wire},
        )
    return frozenset(row.task_id for row in rows)


def _queued_dispatch(
    state: ControllerTransitions, worker_id: WorkerId
) -> tuple[list[cluster_pb2.Worker.RunTaskRequest], list[str]]:
    rows = state._db.fetchall(
        "SELECT kind, payload_proto, task_id FROM dispatch_queue WHERE worker_id = ? ORDER BY id ASC",
        (str(worker_id),),
    )
    tasks_to_run: list[cluster_pb2.Worker.RunTaskRequest] = []
    tasks_to_kill: list[str] = []
    for row in rows:
        if str(row["kind"]) == "run" and row["payload_proto"] is not None:
            req = cluster_pb2.Worker.RunTaskRequest()
            req.ParseFromString(bytes(row["payload_proto"]))
            tasks_to_run.append(req)
        elif row["task_id"] is not None:
            tasks_to_kill.append(str(row["task_id"]))
    return tasks_to_run, tasks_to_kill


def _healthy_active_workers_with_attributes(state: ControllerTransitions) -> list[Worker]:
    from dataclasses import replace as _replace

    with state._db.snapshot() as q:
        workers = q.select(WORKERS, where=(WORKERS.c.healthy == 1) & (WORKERS.c.active == 1))
        if not workers:
            return []
        attrs = q.select(
            WORKER_ATTRIBUTES,
            columns=(
                WORKER_ATTRIBUTES.c.worker_id,
                WORKER_ATTRIBUTES.c.key,
                WORKER_ATTRIBUTES.c.value_type,
                WORKER_ATTRIBUTES.c.str_value,
                WORKER_ATTRIBUTES.c.int_value,
                WORKER_ATTRIBUTES.c.float_value,
            ),
            where=WORKER_ATTRIBUTES.c.worker_id.in_([str(w.worker_id) for w in workers]),
        )
    attrs_by_worker = _decode_attribute_rows(attrs)
    return [_replace(w, attributes=attrs_by_worker.get(w.worker_id, {})) for w in workers]


def _building_counts(state: ControllerTransitions) -> dict[WorkerId, int]:
    with state._db.snapshot() as snapshot:
        rows = snapshot.raw(
            "SELECT a.worker_id, COUNT(*) as c FROM tasks t "
            "JOIN task_attempts a ON t.task_id = a.task_id AND t.current_attempt_id = a.attempt_id "
            "JOIN jobs j ON t.job_id = j.job_id "
            "WHERE t.state IN (?, ?) AND j.is_reservation_holder = 0 "
            "GROUP BY a.worker_id ORDER BY a.worker_id ASC",
            (
                cluster_pb2.TASK_STATE_BUILDING,
                cluster_pb2.TASK_STATE_ASSIGNED,
            ),
            decoders={"worker_id": WorkerId, "c": int},
        )
    return {row.worker_id: row.c for row in rows}


def _task_by_id_with_attempts(state: ControllerTransitions, task_id: JobName) -> Task | None:
    wire = task_id.to_wire()
    with state._db.snapshot() as q:
        tasks = q.select(TASKS, where=TASKS.c.task_id == wire)
        attempts = q.select(
            ATTEMPTS,
            where=ATTEMPTS.c.task_id == wire,
            order_by=(ATTEMPTS.c.attempt_id.asc(),),
        )
    hydrated = _tasks_with_attempts(tasks, attempts)
    return hydrated[0] if hydrated else None


def _endpoints(state: ControllerTransitions, query: EndpointQuery = EndpointQuery()) -> list[Endpoint]:
    joins, where = endpoint_query_predicate(query)
    with state._db.snapshot() as q:
        return q.select(
            ENDPOINTS,
            where=where,
            joins=tuple(joins),
            order_by=(ENDPOINTS.c.registered_at_ms.desc(), ENDPOINTS.c.endpoint_id.asc()),
            limit=query.limit,
        )


def _build_scheduling_context(scheduler: Scheduler, state: ControllerTransitions):
    """Build a SchedulingContext from current state, mirroring what the controller does."""
    pending = _schedulable_tasks(state)
    workers = _healthy_active_workers_with_attributes(state)
    task_ids = [t.task_id for t in pending]
    jobs: dict[JobName, JobRequirements] = {}
    for t in pending:
        job_id = t.task_id.parent
        if job_id and job_id not in jobs:
            job = _query_job(state, job_id)
            if job:
                jobs[job_id] = JobRequirements(
                    resources=job.request.resources,
                    constraints=list(job.request.constraints),
                    is_coscheduled=job.is_coscheduled,
                    coscheduling_group_by=job.coscheduling_group_by,
                )
    return scheduler.create_scheduling_context(
        workers,
        building_counts=_building_counts(state),
        pending_tasks=task_ids,
        jobs=jobs,
    )


def dispatch_task(state: ControllerTransitions, task, worker_id: WorkerId) -> None:
    """Dispatch a task to a worker: assign + mark running."""
    state.queue_assignments([Assignment(task_id=task.task_id, worker_id=worker_id)])
    task = _query_task(state, task.task_id)
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=worker_id,
            worker_resource_snapshot=None,
            updates=[
                TaskUpdate(
                    task_id=task.task_id,
                    attempt_id=_query_task(state, task.task_id).current_attempt_id,
                    new_state=cluster_pb2.TASK_STATE_RUNNING,
                )
            ],
        )
    )


def transition_task(
    state: ControllerTransitions,
    task_id: JobName,
    new_state: int,
    *,
    error: str | None = None,
    exit_code: int | None = None,
) -> object:
    """Transition a task to a new state via heartbeat update or direct cancel."""
    task = _task_by_id_with_attempts(state, task_id)
    assert task is not None
    if new_state == cluster_pb2.TASK_STATE_KILLED:
        return state.cancel_job(task.job_id, reason=error or "killed")
    if task.worker_id is None:
        state.set_task_state_for_test(
            task_id,
            new_state,
            error=error,
            exit_code=exit_code,
        )
        return state
    return state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=task.worker_id,
            worker_resource_snapshot=None,
            updates=[
                TaskUpdate(
                    task_id=task_id,
                    attempt_id=task.current_attempt_id,
                    new_state=new_state,
                    error=error,
                    exit_code=exit_code,
                )
            ],
        )
    )


def fail_worker(state: ControllerTransitions, worker_id: WorkerId, error: str) -> None:
    """Drive worker to failed state through heartbeat-failure path."""
    batch = state.drain_dispatch(worker_id)
    if batch is None:
        return
    for _ in range(HEARTBEAT_FAILURE_THRESHOLD):
        state.record_heartbeat_failure(worker_id, error, batch)


def _make_test_entrypoint() -> cluster_pb2.RuntimeEntrypoint:
    """Create a minimal RuntimeEntrypoint proto for testing."""
    entrypoint = cluster_pb2.RuntimeEntrypoint()
    entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
    return entrypoint


@pytest.fixture
def job_request():
    """Create a minimal LaunchJobRequest for testing."""

    def _make(name: str = "test-job") -> cluster_pb2.Controller.LaunchJobRequest:
        job_name = JobName.root("test-user", name)
        return cluster_pb2.Controller.LaunchJobRequest(
            name=job_name.to_wire(),
            entrypoint=_make_test_entrypoint(),
            resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            environment=cluster_pb2.EnvironmentConfig(),
            replicas=1,
        )

    return _make


@pytest.fixture
def worker_metadata():
    """Create WorkerMetadata for testing."""

    def _make(
        cpu: int = 10,
        memory_bytes: int = 10 * 1024**3,
        disk_bytes: int = 10 * 1024**3,
    ) -> cluster_pb2.WorkerMetadata:
        device = cluster_pb2.DeviceConfig()
        device.cpu.CopyFrom(cluster_pb2.CpuDevice(variant="cpu"))

        return cluster_pb2.WorkerMetadata(
            hostname="test-worker",
            ip_address="127.0.0.1",
            cpu_count=cpu,
            memory_bytes=memory_bytes,
            disk_bytes=disk_bytes,
            device=device,
        )

    return _make


def register_worker(
    state: ControllerTransitions,
    worker_id: str,
    address: str,
    metadata: cluster_pb2.WorkerMetadata,
) -> WorkerId:
    """Register a worker via command API."""
    wid = WorkerId(worker_id)
    state.register_or_refresh_worker(
        worker_id=wid,
        address=address,
        metadata=metadata,
        ts=Timestamp.now(),
    )
    return wid


def _query_job(state: ControllerTransitions, job_id: JobName) -> Job | None:
    with state._db.snapshot() as q:
        return q.one(JOBS, where=JOBS.c.job_id == job_id.to_wire())


def _query_task(state: ControllerTransitions, task_id: JobName) -> Task | None:
    with state._db.snapshot() as q:
        return q.one(TASKS, where=TASKS.c.task_id == task_id.to_wire())


def _query_worker(state: ControllerTransitions, worker_id: WorkerId) -> Worker | None:
    with state._db.snapshot() as q:
        return q.one(WORKERS, where=WORKERS.c.worker_id == str(worker_id))


def _query_tasks_for_job(state: ControllerTransitions, job_id: JobName) -> list[Task]:
    with state._db.snapshot() as q:
        return q.select(TASKS, where=TASKS.c.job_id == job_id.to_wire())


def test_db_snapshot_select_returns_typed_rows(job_request) -> None:
    state = _make_state()
    request = job_request("typed-rows")
    tasks = submit_job(state, "typed-rows", request)

    with state._db.snapshot() as q:
        jobs = q.select(JOBS, where=JOBS.c.job_id == JobName.root("test-user", "typed-rows").to_wire())
        task_count = q.count(TASKS, where=TASKS.c.job_id == JobName.root("test-user", "typed-rows").to_wire())

    assert len(jobs) == 1
    assert jobs[0].submitted_at is not None
    assert jobs[0].job_id == JobName.root("test-user", "typed-rows")
    assert task_count == len(tasks)


def test_db_snapshot_projection_inferrs_typed_values(worker_metadata) -> None:
    state = _make_state()
    wid = register_worker(state, "proj-worker", "addr", worker_metadata())
    request = cluster_pb2.Controller.LaunchJobRequest(
        name=JobName.root("test-user", "projection").to_wire(),
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    [task] = submit_job(state, "projection", request)
    state.queue_assignments([Assignment(task_id=task.task_id, worker_id=wid)])

    running = _worker_running_tasks(state, wid)

    assert len(running) == 1
    assert task.task_id in running


def test_db_snapshot_exists_for_workers(worker_metadata) -> None:
    state = _make_state()
    register_worker(state, "exists-worker", "addr", worker_metadata())

    with state._db.snapshot() as q:
        assert q.exists(WORKERS, where=WORKERS.c.worker_id == "exists-worker")


def submit_job(
    state: ControllerTransitions,
    job_id: str,
    request: cluster_pb2.Controller.LaunchJobRequest,
    timestamp_ms: int | None = None,
) -> list:
    """Submit a job via event and return tasks.

    Auto-injects resource-derived constraints (device-type, device-variant)
    to mirror service-layer behavior. All jobs in production go through
    service.launch_job() which does this injection.
    """
    jid = JobName.from_string(job_id) if job_id.startswith("/") else JobName.root("test-user", job_id)
    request.name = jid.to_wire()

    auto_constraints = constraints_from_resources(request.resources)
    for ac in auto_constraints:
        already = any(c.key == ac.key for c in request.constraints)
        if not already:
            request.constraints.append(
                cluster_pb2.Constraint(
                    key=ac.key,
                    op=cluster_pb2.CONSTRAINT_OP_EQ,
                    value=cluster_pb2.AttributeValue(string_value=ac.value),
                )
            )

    state.submit_job(
        jid,
        request,
        Timestamp.from_ms(timestamp_ms) if timestamp_ms is not None else Timestamp.now(),
    )
    return _query_tasks_for_job(state, jid)


# =============================================================================
# Job/Task Lifecycle Integration Tests
# =============================================================================


def test_job_lifecycle_success(job_request, worker_metadata):
    """E2E: Submit job -> dispatch task -> succeed -> verify final state."""
    state = _make_state()

    # Setup: register worker
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    # Submit job via event
    req = job_request("test-job")
    req.replicas = 2
    tasks = submit_job(state, "j1", req)

    job = _query_job(state, JobName.root("test-user", "j1"))

    assert job is not None
    assert len(tasks) == 2
    assert _query_job(state, job.job_id).state == cluster_pb2.JOB_STATE_PENDING

    # Dispatch and succeed all tasks
    for task in tasks:
        dispatch_task(state, task, worker_id)
        transition_task(state, task.task_id, cluster_pb2.TASK_STATE_SUCCEEDED)

    # Verify final state
    assert _query_job(state, job.job_id).state == cluster_pb2.JOB_STATE_SUCCEEDED
    for task in tasks:
        assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_SUCCEEDED
        worker = _query_worker(state, worker_id)
        assert worker is not None
    assert len(_schedulable_tasks(state)) == 0


def test_job_lifecycle_failure_exhausted_retries(job_request, worker_metadata):
    """E2E: Task failure with no retries -> job fails."""
    state = _make_state()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())
    req = job_request("job1")
    tasks = submit_job(state, "j1", req)
    task = tasks[0]
    job = _query_job(state, JobName.root("test-user", "j1"))

    # Dispatch and fail (default max_retries_failure=0)
    dispatch_task(state, task, worker_id)
    transition_task(state, task.task_id, cluster_pb2.TASK_STATE_FAILED, error="Task failed")

    # Verify final state
    assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_FAILED
    assert _query_task(state, task.task_id).is_finished()
    assert _query_job(state, job.job_id).state == cluster_pb2.JOB_STATE_FAILED
    worker = _query_worker(state, worker_id)
    assert worker is not None


def test_task_failure_with_retry_requeues(job_request, worker_metadata):
    """E2E: Task failure with retries -> task requeued, job stays running."""
    state = _make_state()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("job1")
    req.max_task_failures = 1
    req.max_retries_failure = 1
    tasks = submit_job(state, "j1", req)
    task = tasks[0]
    job = _query_job(state, JobName.root("test-user", "j1"))

    # First attempt fails
    dispatch_task(state, task, worker_id)
    transition_task(state, task.task_id, cluster_pb2.TASK_STATE_FAILED)

    # Verify: task requeued (back to PENDING), job still running
    assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_PENDING
    assert _query_task(state, task.task_id).can_be_scheduled()
    assert _query_job(state, job.job_id).state == cluster_pb2.JOB_STATE_RUNNING
    pending = _schedulable_tasks(state)
    assert len(pending) == 1
    assert pending[0].task_id == task.task_id


def test_unschedulable_task_finalizes_job_with_timeout_error(job_request, worker_metadata):
    """E2E: Task UNSCHEDULABLE propagates timeout-style error to final job state."""
    state = _make_state()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("job1")
    req.scheduling_timeout.CopyFrom(Duration.from_seconds(300).to_proto())
    tasks = submit_job(state, "j1", req)
    task = tasks[0]
    job = _query_job(state, JobName.root("test-user", "j1"))

    dispatch_task(state, task, worker_id)
    transition_task(state, task.task_id, cluster_pb2.TASK_STATE_UNSCHEDULABLE)

    assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_UNSCHEDULABLE
    assert _query_task(state, task.task_id).error == "Scheduling timeout exceeded"
    assert _query_job(state, job.job_id).state == cluster_pb2.JOB_STATE_UNSCHEDULABLE
    assert _query_job(state, job.job_id).error == "Scheduling timeout exceeded"


def test_job_cancellation_kills_all_tasks(job_request, worker_metadata):
    """E2E: Job cancellation -> all tasks killed."""
    state = _make_state()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("test-job")
    req.replicas = 3
    tasks = submit_job(state, "j1", req)
    job = _query_job(state, JobName.root("test-user", "j1"))

    # Dispatch 2 tasks, leave 1 pending
    dispatch_task(state, tasks[0], worker_id)
    dispatch_task(state, tasks[1], worker_id)

    # Cancel job
    state.cancel_job(JobName.root("test-user", "j1"), reason="User cancelled")

    # Verify all tasks killed
    assert _query_job(state, job.job_id).state == cluster_pb2.JOB_STATE_KILLED
    for task in tasks:
        assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_KILLED


def test_cancel_job_releases_committed_worker_resources(job_request, worker_metadata):
    """cancel_job must decommit resources on workers that had active tasks.

    Regression: cancel_job marked tasks KILLED without calling _decommit_worker_resources.
    apply_heartbeat then skipped the update (task already finished), so committed resources
    were never released, permanently blocking scheduling on those workers.
    """
    state = _make_state()

    w1 = register_worker(state, "w1", "host1:8080", worker_metadata())
    w2 = register_worker(state, "w2", "host2:8080", worker_metadata())

    req = job_request("test-job")
    req.replicas = 3
    tasks = submit_job(state, "j1", req)

    # Dispatch 2 tasks to different workers, leave 1 pending
    dispatch_task(state, tasks[0], w1)
    dispatch_task(state, tasks[1], w2)

    # Verify resources are committed
    assert _query_worker(state, w1).committed_cpu_millicores == 1000
    assert _query_worker(state, w1).committed_mem == 1024**3
    assert _query_worker(state, w2).committed_cpu_millicores == 1000

    # Cancel job
    state.cancel_job(JobName.root("test-user", "j1"), reason="User cancelled")

    # Resources must be fully released on both workers
    assert _query_worker(state, w1).committed_cpu_millicores == 0, "w1 leaked committed_cpu_millicores"
    assert _query_worker(state, w1).committed_mem == 0, "w1 leaked committed_mem"
    assert _query_worker(state, w2).committed_cpu_millicores == 0, "w2 leaked committed_cpu_millicores"
    assert _query_worker(state, w2).committed_mem == 0, "w2 leaked committed_mem"

    # No active tasks on either worker
    assert len(_worker_running_tasks(state, w1)) == 0
    assert len(_worker_running_tasks(state, w2)) == 0


def test_cancelled_job_tasks_excluded_from_demand(job_request, worker_metadata):
    """Regression test for issue #2777: Killed tasks with no attempts should not appear in demand entries."""
    state = _make_state()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("test-job")
    req.replicas = 3
    tasks = submit_job(state, "j1", req)
    job = _query_job(state, JobName.root("test-user", "j1"))

    # Dispatch 1 task, leave 2 pending (these will have no attempts when killed)
    dispatch_task(state, tasks[0], worker_id)

    # Cancel job - pending tasks will be killed with no attempts
    state.cancel_job(JobName.root("test-user", "j1"), reason="User cancelled")

    # Verify all tasks are killed
    assert _query_job(state, job.job_id).state == cluster_pb2.JOB_STATE_KILLED
    for task in tasks:
        assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_KILLED

    # Verify killed tasks are not schedulable (bug fix)
    for task in tasks:
        assert not _query_task(
            state, task.task_id
        ).can_be_scheduled(), f"Killed task {task.task_id} should not be schedulable"

    # Verify no pending tasks appear in queue
    pending = _schedulable_tasks(state)
    assert len(pending) == 0, "Killed tasks should not appear in pending queue"

    # Verify no demand entries generated for killed job
    demand_entries = compute_demand_entries(state._db)
    assert len(demand_entries) == 0, "Killed job should not generate demand entries"


# =============================================================================
# Worker Failure Cascade Tests
# =============================================================================


def test_worker_failure_cascades_to_running_tasks(job_request, worker_metadata):
    """E2E: Worker failure -> running tasks transition to WORKER_FAILED and requeue."""
    state = _make_state()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())
    req = job_request("job1")
    req.max_retries_preemption = 1
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    dispatch_task(state, task, worker_id)

    # Worker fails
    fail_worker(state, worker_id, "Connection lost")

    # Verify: worker unhealthy, task requeued (back to PENDING)
    assert _query_worker(state, worker_id) is None
    assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_PENDING
    assert _query_task(state, task.task_id).can_be_scheduled()
    pending = _schedulable_tasks(state)
    assert len(pending) == 1


def test_failed_worker_is_pruned_from_state(job_request, worker_metadata):
    """E2E: Worker failure removes worker from state, preventing dead worker accumulation."""
    state = _make_state()

    w1 = register_worker(state, "w1", "host1:8080", worker_metadata())
    w2 = register_worker(state, "w2", "host2:8080", worker_metadata())

    req = job_request("job1")
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
        all_workers = q.select(WORKERS)
    assert len(all_workers) == 1
    assert all_workers[0].worker_id == w2

    # Task was requeued despite worker removal
    assert tasks[0].state == cluster_pb2.TASK_STATE_PENDING
    assert tasks[0].can_be_scheduled()

    # A re-registering worker creates a fresh entry
    w1_again = register_worker(state, "w1", "host1:8080", worker_metadata())
    assert _query_worker(state, w1_again) is not None
    assert _query_worker(state, w1_again).healthy is True
    with state._db.snapshot() as q:
        assert len(q.select(WORKERS)) == 2


def test_dispatch_failure_marks_worker_failed_and_requeues_task(job_request, worker_metadata):
    """E2E: Dispatch RPC failure (task in PENDING) -> worker failed event cascades to task."""
    state = _make_state()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("job1")
    req.max_retries_preemption = 1
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    # Task gets assigned (creates attempt, puts in ASSIGNED state)
    state.queue_assignments([Assignment(task_id=task.task_id, worker_id=worker_id)])
    assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_ASSIGNED
    assert _query_task(state, task.task_id).current_attempt_id == 0

    # Dispatch RPC fails -> WORKER_FAILED event
    fail_worker(state, worker_id, "Dispatch RPC failed: Connection refused")

    # Verify cascade:
    # 1. Worker marked unhealthy
    assert _query_worker(state, worker_id) is None

    # 2. Task requeued (back to PENDING for retry).
    #    Since the task was still ASSIGNED (never confirmed BUILDING/RUNNING),
    #    this is a delivery failure — no budget consumed at all.
    assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_PENDING
    assert _query_task(state, task.task_id).preemption_count == 0
    assert _query_task(state, task.task_id).failure_count == 0
    assert _query_task(state, task.task_id).can_be_scheduled()

    # 3. Task should be requeued for retry
    pending = _schedulable_tasks(state)
    assert len(pending) == 1
    assert pending[0].task_id == task.task_id

    # 4. Worker no longer has task assigned
    assert _query_worker(state, worker_id) is None


def test_task_assigned_to_missing_worker_is_ignored(job_request, worker_metadata):
    """Stale assignments to pruned workers are skipped without crashing."""
    state = _make_state()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())
    tasks = submit_job(state, "j1", job_request("job1"))
    task = tasks[0]

    # Worker disappears between scheduling and assignment commit.
    state.remove_worker(worker_id)
    state.queue_assignments([Assignment(task_id=task.task_id, worker_id=worker_id)])

    # Task remains schedulable and no attempt/resources are committed.
    assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_PENDING
    assert _query_task(state, task.task_id).current_attempt_id == -1
    assert _query_task(state, task.task_id).can_be_scheduled()
    assert task.task_id in {t.task_id for t in _schedulable_tasks(state)}


# =============================================================================
# Failure Domain Tests (max_task_failures)
# =============================================================================


def test_failure_domain_kills_remaining_tasks(worker_metadata):
    """E2E: One task fails beyond retries -> remaining tasks killed, job fails."""
    state = _make_state()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="multi-task-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        max_task_failures=0,
        replicas=3,
    )
    tasks = submit_job(state, "j1", req)
    job = _query_job(state, JobName.root("test-user", "j1"))

    # Dispatch 2 tasks, leave 1 pending
    dispatch_task(state, tasks[0], worker_id)
    dispatch_task(state, tasks[1], worker_id)

    # Task-0 fails
    transition_task(state, tasks[0].task_id, cluster_pb2.TASK_STATE_FAILED, error="Task failed")

    # Verify final state
    assert _query_job(state, job.job_id).state == cluster_pb2.JOB_STATE_FAILED
    assert _query_task(state, tasks[0].task_id).state == cluster_pb2.TASK_STATE_FAILED
    assert _query_task(state, tasks[1].task_id).state == cluster_pb2.TASK_STATE_KILLED
    assert _query_task(state, tasks[2].task_id).state == cluster_pb2.TASK_STATE_KILLED


def test_max_task_failures_tolerance(worker_metadata):
    """E2E: Job tolerates max_task_failures, then fails on next failure."""
    state = _make_state()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="tolerant-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=3,
        environment=cluster_pb2.EnvironmentConfig(),
        max_task_failures=1,
    )
    tasks = submit_job(state, "j1", req)
    job = _query_job(state, JobName.root("test-user", "j1"))

    for task in tasks:
        dispatch_task(state, task, worker_id)

    # First failure - job should keep running
    transition_task(state, tasks[0].task_id, cluster_pb2.TASK_STATE_FAILED, error="First")
    assert _query_job(state, job.job_id).state == cluster_pb2.JOB_STATE_RUNNING

    # Second task succeeds
    transition_task(state, tasks[1].task_id, cluster_pb2.TASK_STATE_SUCCEEDED)
    assert _query_job(state, job.job_id).state == cluster_pb2.JOB_STATE_RUNNING

    # Third task fails - exceeds threshold, job fails
    transition_task(state, tasks[2].task_id, cluster_pb2.TASK_STATE_FAILED, error="Second")
    assert _query_job(state, job.job_id).state == cluster_pb2.JOB_STATE_FAILED


def test_preemption_does_not_count_toward_max_task_failures(worker_metadata):
    """E2E: Worker failures (preemptions) don't count toward max_task_failures."""
    state = _make_state()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="preemption-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=2,
        environment=cluster_pb2.EnvironmentConfig(),
        max_task_failures=0,
        max_retries_preemption=1,
    )
    tasks = submit_job(state, "j1", req)
    job = _query_job(state, JobName.root("test-user", "j1"))

    dispatch_task(state, tasks[0], worker_id)
    transition_task(state, tasks[0].task_id, cluster_pb2.TASK_STATE_WORKER_FAILED, error="Worker died")

    # Preemption doesn't count toward failure threshold; task requeued to PENDING
    assert tasks[0].state == cluster_pb2.TASK_STATE_PENDING
    assert tasks[0].can_be_scheduled()
    assert _query_job(state, job.job_id).state == cluster_pb2.JOB_STATE_RUNNING


# =============================================================================
# Endpoint Cleanup Tests
# =============================================================================


def test_terminal_states_clean_up_endpoints(job_request, worker_metadata):
    """E2E: Task reaching terminal state removes associated endpoints."""
    state = _make_state()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("job1")
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    dispatch_task(state, task, worker_id)

    ep = Endpoint(
        endpoint_id="ep1",
        name="j1/actor",
        address="a:1",
        job_id=JobName.root("test-user", "j1"),
        metadata={},
        registered_at=Timestamp.now(),
    )
    state.add_endpoint(ep, task.task_id)

    # Verify endpoint visible while running
    assert len(_endpoints(state, EndpointQuery(exact_name="j1/actor"))) == 1

    # Task succeeds
    transition_task(state, task.task_id, cluster_pb2.TASK_STATE_SUCCEEDED)

    # Endpoint removed
    assert _endpoints(state, EndpointQuery(exact_name="j1/actor")) == []


def test_endpoint_visibility_by_job_state(job_request, worker_metadata):
    """Endpoints are visible for all non-terminal job states (PENDING, RUNNING, BUILDING)
    and hidden once the job reaches a terminal state."""
    state = _make_state()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("test")
    tasks = submit_job(state, "ns-1", req)
    job = _query_job(state, JobName.root("test-user", "ns-1"))
    task = tasks[0]

    ep = Endpoint(
        endpoint_id="ep-1",
        name="ns-1/actor",
        address="10.0.0.1:8080",
        job_id=JobName.root("test-user", "ns-1"),
        metadata={},
        registered_at=Timestamp.now(),
    )
    state.add_endpoint(ep)

    # Visible while pending (task may be executing before job state updates)
    assert len(_endpoints(state, EndpointQuery(exact_name="ns-1/actor"))) == 1

    # Still visible after transition to running
    dispatch_task(state, task, worker_id)
    assert _query_job(state, job.job_id).state == cluster_pb2.JOB_STATE_RUNNING
    assert len(_endpoints(state, EndpointQuery(exact_name="ns-1/actor"))) == 1

    # Not visible after completion (terminal state)
    transition_task(state, task.task_id, cluster_pb2.TASK_STATE_SUCCEEDED)
    assert _query_job(state, job.job_id).state == cluster_pb2.JOB_STATE_SUCCEEDED
    assert len(_endpoints(state, EndpointQuery(exact_name="ns-1/actor"))) == 0


def test_namespace_isolation(job_request, worker_metadata):
    """E2E: Endpoints are isolated by namespace prefix."""
    state = _make_state()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req1 = job_request("test1")
    req2 = job_request("test2")

    tasks1 = submit_job(state, "ns-1", req1)
    tasks2 = submit_job(state, "ns-2", req2)

    # Dispatch tasks to transition jobs to RUNNING state
    dispatch_task(state, tasks1[0], worker_id)
    dispatch_task(state, tasks2[0], worker_id)

    state.add_endpoint(
        Endpoint(
            endpoint_id="ep-1",
            name="ns-1/actor",
            address="10.0.0.1:8080",
            job_id=JobName.root("test-user", "ns-1"),
            metadata={},
            registered_at=Timestamp.now(),
        )
    )
    state.add_endpoint(
        Endpoint(
            endpoint_id="ep-2",
            name="ns-2/actor",
            address="10.0.0.2:8080",
            job_id=JobName.root("test-user", "ns-2"),
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


def test_task_queue_fifo_order(job_request):
    """Tasks are returned in FIFO order."""
    state = _make_state()

    req1 = job_request("job1")
    req2 = job_request("job2")
    submit_job(state, "j1", req1)
    submit_job(state, "j2", req2)

    pending = _schedulable_tasks(state)
    assert len(pending) == 2
    assert pending[0].job_id == JobName.root("test-user", "j1")
    assert pending[1].job_id == JobName.root("test-user", "j2")


def test_hierarchical_job_tracking(job_request):
    """Parent-child job relationships are tracked correctly."""
    state = _make_state()

    parent_req = job_request("parent")
    submit_job(state, "parent", parent_req)

    child1_req = job_request("child1")
    submit_job(state, "/test-user/parent/child1", child1_req)

    child2_req = job_request("child2")
    submit_job(state, "/test-user/parent/child2", child2_req)

    grandchild_req = job_request("grandchild")
    submit_job(state, "/test-user/parent/child1/grandchild", grandchild_req)

    # get_children only returns direct children
    with state._db.snapshot() as q:
        children = q.select(JOBS, where=JOBS.c.parent_job_id == JobName.root("test-user", "parent").to_wire())
    assert len(children) == 2
    assert {c.job_id for c in children} == {
        JobName.from_string("/test-user/parent/child1"),
        JobName.from_string("/test-user/parent/child2"),
    }

    # No children for leaf nodes
    with state._db.snapshot() as q:
        leaf_children = q.select(
            JOBS,
            where=JOBS.c.parent_job_id == JobName.from_string("/test-user/parent/child1/grandchild").to_wire(),
        )
    assert leaf_children == []


def test_thread_safety(job_request):
    """Concurrent access doesn't corrupt state."""
    state = _make_state()
    num_threads = 10
    jobs_per_thread = 50
    barrier = threading.Barrier(num_threads)
    errors = []

    def add_jobs(thread_id: int):
        try:
            barrier.wait()
            for i in range(jobs_per_thread):
                job_id = f"t{thread_id}_j{i}"
                req = job_request(f"job-{job_id}")
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


def test_excessive_replicas_fails_job(job_request):
    """E2E: Job with replicas exceeding MAX_REPLICAS_PER_JOB fails immediately."""
    state = _make_state()

    req = job_request("too-many-replicas")
    req.replicas = MAX_REPLICAS_PER_JOB + 1

    tasks = submit_job(state, "j1", req)
    job = _query_job(state, JobName.root("test-user", "j1"))

    assert job is not None
    assert _query_job(state, job.job_id).state == cluster_pb2.JOB_STATE_FAILED
    assert f"exceeds max {MAX_REPLICAS_PER_JOB}" in _query_job(state, job.job_id).error
    assert len(tasks) == 0
    assert len(_schedulable_tasks(state)) == 0


# =============================================================================
# Worker Resource Commitment Tests
# =============================================================================


@pytest.fixture
def make_job_request():
    """Create a LaunchJobRequest with configurable resources."""

    def _make(
        name: str = "test-job",
        cpu: int = 1,
        memory_bytes: int = 1024**3,
    ) -> cluster_pb2.Controller.LaunchJobRequest:
        return cluster_pb2.Controller.LaunchJobRequest(
            name=name,
            entrypoint=_make_test_entrypoint(),
            resources=cluster_pb2.ResourceSpecProto(cpu_millicores=cpu * 1000, memory_bytes=memory_bytes),
            environment=cluster_pb2.EnvironmentConfig(),
            replicas=1,
        )

    return _make


def test_worker_cannot_accept_task_when_resources_committed(make_job_request, worker_metadata):
    """E2E: A worker with committed resources cannot accept tasks that exceed remaining capacity."""
    state = _make_state()

    # Worker with 4 CPUs
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata(cpu=4))

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


def test_worker_can_accept_new_task_after_previous_completes(make_job_request, worker_metadata):
    """E2E: After a task completes, its resources are freed and new tasks can be scheduled.

    This verifies that task completion releases committed resources back to the worker.
    """
    state = _make_state()

    # Worker with 4 CPUs
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata(cpu=4))

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
    transition_task(state, tasks1[0].task_id, cluster_pb2.TASK_STATE_SUCCEEDED)

    # Now the second task can be scheduled
    context = _build_scheduling_context(scheduler, state)
    result = scheduler.find_assignments(context)
    assert len(result.assignments) == 1
    assert result.assignments[0][0].parent == JobName.root("test-user", "j2")


def test_multiple_small_tasks_fill_worker_capacity(make_job_request, worker_metadata):
    """E2E: Multiple small tasks can fill a worker's capacity, blocking further assignments.

    This verifies that the scheduler correctly tracks cumulative resource usage across
    multiple running tasks. With round-robin scheduling, each worker gets at most one
    task per cycle, so we run multiple cycles to fill capacity.
    """
    state = _make_state()

    # Worker with 4 CPUs
    register_worker(state, "w1", "host:8080", worker_metadata(cpu=4))

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


def test_coscheduled_task_failure_kills_siblings(worker_metadata):
    """When one coscheduled task fails terminally, all running siblings are killed."""
    state = _make_state()

    # Register 4 workers (one per task)
    for i in range(4):
        meta = worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    # Create coscheduled job with 4 tasks
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j1", req)

    job = _query_job(state, JobName.root("test-user", "j1"))
    assert job.is_coscheduled

    # Dispatch all tasks
    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Fail task-0 (terminal failure with no retries)
    txn = transition_task(state, tasks[0].task_id, cluster_pb2.TASK_STATE_FAILED, error="OOM")

    # Task-0 should be FAILED, all other tasks should be WORKER_FAILED
    assert _query_task(state, tasks[0].task_id).state == cluster_pb2.TASK_STATE_FAILED
    for task in tasks[1:]:
        assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_WORKER_FAILED
        assert task.task_id in txn.tasks_to_kill


def test_coscheduled_cascade_releases_worker_resources(worker_metadata):
    """Coscheduled sibling cascade must free committed resources on surviving workers.

    Regression test: previously, _cascade_coscheduled_failure marked siblings
    terminal but never called _cleanup_task_resources, leaking committed_cpu_millicores/mem
    on workers and permanently blocking future scheduling.
    """
    state = _make_state()

    for i in range(4):
        meta = worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="leak-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=2000, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j-leak", req)

    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Verify resources are committed before failure
    for i in range(4):
        w = _query_worker(state, WorkerId(f"w{i}"))
        assert w.committed_cpu_millicores == 2000
        assert len(_worker_running_tasks(state, WorkerId(f"w{i}"))) == 1

    # Fail task-0 terminally → cascade kills siblings on w1, w2, w3
    transition_task(state, tasks[0].task_id, cluster_pb2.TASK_STATE_FAILED, error="OOM")

    # All surviving workers (w1..w3) must have resources fully released
    for i in range(1, 4):
        w = _query_worker(state, WorkerId(f"w{i}"))
        assert w.committed_cpu_millicores == 0, f"w{i} has leaked committed_cpu_millicores={w.committed_cpu_millicores}"
        assert w.committed_mem == 0, f"w{i} has leaked committed_mem={w.committed_mem}"
        assert len(_worker_running_tasks(state, WorkerId(f"w{i}"))) == 0

    # w0 should also be clean (task-0 was the trigger, cleaned up by _on_task_state_changed)
    w0 = _query_worker(state, WorkerId("w0"))
    assert w0.committed_cpu_millicores == 0
    assert len(_worker_running_tasks(state, WorkerId("w0"))) == 0


def test_coscheduled_task_worker_failure_kills_siblings(worker_metadata):
    """WORKER_FAILED also triggers sibling kill when retries exhausted."""
    state = _make_state()

    for i in range(4):
        meta = worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    # Use max_retries_preemption=1 so second worker failure is terminal.
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
        max_retries_preemption=1,  # Allow one retry, so second failure is terminal
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j1", req)

    # Dispatch all tasks
    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # First WORKER_FAILED is retriable (retries remaining)
    transition_task(state, tasks[0].task_id, cluster_pb2.TASK_STATE_WORKER_FAILED, error="Worker crashed (first)")

    # Task-0 is retriable, siblings still running
    assert _query_task(state, tasks[0].task_id).preemption_count == 1
    assert _query_task(state, tasks[0].task_id).can_be_scheduled()
    for task in tasks[1:]:
        assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_RUNNING

    # Re-dispatch task-0
    dispatch_task(state, tasks[0], WorkerId("w0"))

    # Second WORKER_FAILED exhausts retries - now terminal
    txn = transition_task(state, tasks[0].task_id, cluster_pb2.TASK_STATE_WORKER_FAILED, error="Worker crashed (second)")

    assert _query_task(state, tasks[0].task_id).state == cluster_pb2.TASK_STATE_WORKER_FAILED
    assert _query_task(state, tasks[0].task_id).is_finished()
    for task in tasks[1:]:
        assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_WORKER_FAILED
        assert task.task_id in txn.tasks_to_kill


def test_coscheduled_task_success_does_not_affect_siblings(worker_metadata):
    """Task success does NOT kill siblings."""
    state = _make_state()

    for i in range(4):
        meta = worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j1", req)

    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Task-0 succeeds
    txn = transition_task(state, tasks[0].task_id, cluster_pb2.TASK_STATE_SUCCEEDED)

    # Task-0 succeeded, siblings still running
    assert _query_task(state, tasks[0].task_id).state == cluster_pb2.TASK_STATE_SUCCEEDED
    for task in tasks[1:]:
        assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_RUNNING
    assert len(txn.tasks_to_kill) == 0


def test_non_coscheduled_task_failure_does_not_kill_siblings(worker_metadata):
    """Regular jobs don't cascade failures to siblings."""
    state = _make_state()

    for i in range(4):
        register_worker(state, f"w{i}", f"addr{i}:8080", worker_metadata())

    # Regular job (no coscheduling)
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="regular-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
        max_task_failures=3,  # Allow failures without killing the job
    )
    tasks = submit_job(state, "j1", req)

    job = _query_job(state, JobName.root("test-user", "j1"))
    assert not job.is_coscheduled

    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Fail task-0
    txn = transition_task(state, tasks[0].task_id, cluster_pb2.TASK_STATE_FAILED, error="OOM")

    # Task-0 failed, but siblings are still running (no cascade)
    assert _query_task(state, tasks[0].task_id).state == cluster_pb2.TASK_STATE_FAILED
    for task in tasks[1:]:
        assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_RUNNING

    # No tasks marked to kill from coscheduling cascade
    assert len(txn.tasks_to_kill) == 0


def test_coscheduled_retriable_failure_does_not_kill_siblings(worker_metadata):
    """When a coscheduled task fails but has retries remaining, siblings are NOT killed."""
    state = _make_state()

    for i in range(4):
        meta = worker_metadata()
        meta.attributes[WellKnownAttribute.TPU_NAME].string_value = "tpu-a"
        meta.attributes[WellKnownAttribute.TPU_WORKER_ID].int_value = i
        register_worker(state, f"w{i}", f"addr{i}:8080", meta)

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        replicas=4,
        environment=cluster_pb2.EnvironmentConfig(),
        max_retries_failure=1,  # Allow one retry
        max_task_failures=4,  # Don't fail job on task failure
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    tasks = submit_job(state, "j1", req)

    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Fail task-0 (first failure, has retry remaining)
    txn = transition_task(state, tasks[0].task_id, cluster_pb2.TASK_STATE_FAILED, error="OOM")

    # Task-0 failed but is retriable, requeued to PENDING
    assert tasks[0].state == cluster_pb2.TASK_STATE_PENDING
    assert tasks[0].can_be_scheduled()  # Can retry
    assert not tasks[0].is_finished()  # Not terminal

    # Siblings should still be running (no cascade for retriable failures)
    for task in tasks[1:]:
        assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_RUNNING

    # No tasks marked for kill
    assert len(txn.tasks_to_kill) == 0


# =============================================================================
# compute_demand_entries Tests
# =============================================================================


# =============================================================================
# Stale Attempt Tracking Tests
# =============================================================================


def test_stale_attempt_ignored(job_request, worker_metadata):
    """Stale attempt report does not change task state."""
    state = _make_state()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("job1")
    req.max_retries_preemption = 2
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    # First attempt: dispatch, then fail via worker failure (retriable)
    dispatch_task(state, task, worker_id)
    old_attempt_id = _query_task(state, task.task_id).current_attempt_id
    assert old_attempt_id == 0

    transition_task(state, task.task_id, cluster_pb2.TASK_STATE_WORKER_FAILED, error="Worker died")

    # Second attempt
    dispatch_task(state, task, worker_id)
    assert _query_task(state, task.task_id).current_attempt_id == 1
    assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_RUNNING

    # Stale report from old attempt should be ignored
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=worker_id,
            worker_resource_snapshot=None,
            updates=[
                TaskUpdate(
                    task_id=task.task_id,
                    attempt_id=old_attempt_id,
                    new_state=cluster_pb2.TASK_STATE_SUCCEEDED,
                )
            ],
        )
    )

    # Task should still be RUNNING on the new attempt
    assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_RUNNING
    assert _query_task(state, task.task_id).current_attempt_id == 1


def test_stale_attempt_error_log_for_non_terminal(caplog, job_request, worker_metadata):
    """Stale attempt report logs ERROR when the old attempt is not terminal."""
    import logging

    state = _make_state()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("job1")
    req.max_retries_preemption = 2
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    # First attempt
    dispatch_task(state, task, worker_id)

    # Manually create a second attempt without properly terminating the first.
    # This simulates a scenario where the controller created a new attempt
    # but the old one is still non-terminal (a precondition violation).
    state.create_attempt_for_test(task.task_id, worker_id)
    assert _query_task(state, task.task_id).current_attempt_id == 1
    # The old attempt (0) is still in RUNNING state (non-terminal)
    with state._db.snapshot() as q:
        attempts = q.select(ATTEMPTS, where=ATTEMPTS.c.task_id == task.task_id.to_wire())
    assert not attempts[0].is_terminal

    with caplog.at_level(logging.ERROR, logger="iris.cluster.controller.transitions"):
        state.apply_task_updates(
            HeartbeatApplyRequest(
                worker_id=worker_id,
                worker_resource_snapshot=None,
                updates=[TaskUpdate(task_id=task.task_id, attempt_id=0, new_state=cluster_pb2.TASK_STATE_SUCCEEDED)],
            )
        )

    assert any("Stale attempt precondition violation" in r.message for r in caplog.records)


# =============================================================================
# Heartbeat Log Forwarding Tests
# =============================================================================


def test_log_entries_accumulated_in_log_store(job_request, worker_metadata):
    """Log entries from heartbeat are stored in the controller's log store."""
    from iris.rpc import logging_pb2

    state = _make_state()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    tasks = submit_job(state, "j1", job_request("job1"))
    task = tasks[0]
    dispatch_task(state, task, worker_id)

    snapshot = state.begin_heartbeat(worker_id)
    assert snapshot is not None

    log_entry = logging_pb2.LogEntry(source="stdout", data="hello world")
    log_entry.timestamp.epoch_ms = 1000

    response = cluster_pb2.HeartbeatResponse(
        worker_healthy=True,
        tasks=[
            cluster_pb2.Controller.WorkerTaskStatus(
                task_id=task.task_id.to_wire(),
                attempt_id=_query_task(state, task.task_id).current_attempt_id,
                state=cluster_pb2.TASK_STATE_RUNNING,
                log_entries=[log_entry],
            )
        ],
    )
    state.complete_heartbeat(snapshot, response)

    from iris.cluster.log_store import task_log_key
    from iris.cluster.types import TaskAttempt

    log_result = state._db.get_logs(
        task_log_key(TaskAttempt(task_id=task.task_id, attempt_id=_query_task(state, task.task_id).current_attempt_id))
    )
    assert len(log_result.entries) == 1
    assert log_result.entries[0].data == "hello world"


def test_log_entries_accumulated_across_heartbeats(job_request, worker_metadata):
    """Multiple heartbeats accumulate logs in the store."""
    from iris.rpc import logging_pb2

    state = _make_state()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    tasks = submit_job(state, "j1", job_request("job1"))
    task = tasks[0]
    dispatch_task(state, task, worker_id)

    for i in range(3):
        snapshot = state.begin_heartbeat(worker_id)
        assert snapshot is not None
        entry = logging_pb2.LogEntry(source="stdout", data=f"line {i}")
        entry.timestamp.epoch_ms = 1000 + i
        response = cluster_pb2.HeartbeatResponse(
            worker_healthy=True,
            tasks=[
                cluster_pb2.Controller.WorkerTaskStatus(
                    task_id=task.task_id.to_wire(),
                    attempt_id=_query_task(state, task.task_id).current_attempt_id,
                    state=cluster_pb2.TASK_STATE_RUNNING,
                    log_entries=[entry],
                )
            ],
        )
        state.complete_heartbeat(snapshot, response)

    from iris.cluster.log_store import task_log_key
    from iris.cluster.types import TaskAttempt

    log_result = state._db.get_logs(
        task_log_key(TaskAttempt(task_id=task.task_id, attempt_id=_query_task(state, task.task_id).current_attempt_id))
    )
    assert len(log_result.entries) == 3
    assert [e.data for e in log_result.entries] == ["line 0", "line 1", "line 2"]


# =============================================================================
# compute_demand_entries Tests
# =============================================================================


def test_compute_demand_entries_counts_coscheduled_job_once():
    """Coscheduled job with 4 tasks should count as 1 slice demand, not 4."""
    state = _make_state()
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=4,
    )
    req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state._db)
    assert len(demand) == 1
    assert demand[0].normalized.device_type == DeviceType.TPU
    assert demand[0].normalized.device_variants == frozenset({"v5litepod-16"})
    assert demand[0].task_ids == ["/test-user/j1/0", "/test-user/j1/1", "/test-user/j1/2", "/test-user/j1/3"]
    assert demand[0].coschedule_group_id == "/test-user/j1"


def test_compute_demand_entries_counts_non_coscheduled_tasks_individually():
    """Non-coscheduled job with 4 tasks should count as 4 slices demand."""
    state = _make_state()
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="regular-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=4,
    )
    # No coscheduling set
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state._db)
    assert len(demand) == 4
    for entry in demand:
        assert entry.normalized.device_type == DeviceType.TPU
        assert entry.normalized.device_variants == frozenset({"v5litepod-16"})
        assert entry.coschedule_group_id is None
        assert len(entry.task_ids) == 1


def test_compute_demand_entries_mixed_coscheduled_and_regular():
    """Mix of coscheduled and regular jobs should count correctly."""
    state = _make_state()

    # Coscheduled job with 4 tasks -> 1 slice
    coscheduled_req = cluster_pb2.Controller.LaunchJobRequest(
        name="coschedule-test",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=4,
    )
    coscheduled_req.coscheduling.group_by = WellKnownAttribute.TPU_NAME
    submit_job(state, "j1", coscheduled_req)

    # Regular job with 2 tasks -> 2 slices
    regular_req = cluster_pb2.Controller.LaunchJobRequest(
        name="regular-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=2,
    )
    submit_job(state, "j2", regular_req)

    demand = compute_demand_entries(state._db)
    assert len(demand) == 3
    coscheduled = [entry for entry in demand if entry.coschedule_group_id == "/test-user/j1"]
    regular = [entry for entry in demand if entry.coschedule_group_id is None]
    assert len(coscheduled) == 1
    assert len(regular) == 2
    assert coscheduled[0].task_ids == ["/test-user/j1/0", "/test-user/j1/1", "/test-user/j1/2", "/test-user/j1/3"]
    for entry in regular:
        assert entry.normalized.device_type == DeviceType.TPU
        assert entry.normalized.device_variants == frozenset({"v5litepod-16"})


def test_compute_demand_entries_separates_by_preemptible_constraint():
    """Jobs with different preemptible constraints produce separate demand entries."""
    state = _make_state()

    # Job requiring preemptible workers
    preemptible_req = cluster_pb2.Controller.LaunchJobRequest(
        name="preemptible-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5p-8")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
        constraints=[
            cluster_pb2.Constraint(
                key=WellKnownAttribute.PREEMPTIBLE,
                op=cluster_pb2.CONSTRAINT_OP_EQ,
                value=cluster_pb2.AttributeValue(string_value="true"),
            )
        ],
    )
    submit_job(state, "j1", preemptible_req)

    # Job requiring non-preemptible workers
    on_demand_req = cluster_pb2.Controller.LaunchJobRequest(
        name="on-demand-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5p-8")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
        constraints=[
            cluster_pb2.Constraint(
                key=WellKnownAttribute.PREEMPTIBLE,
                op=cluster_pb2.CONSTRAINT_OP_EQ,
                value=cluster_pb2.AttributeValue(string_value="false"),
            )
        ],
    )
    submit_job(state, "j2", on_demand_req)

    demand = compute_demand_entries(state._db)
    assert len(demand) == 2

    by_preemptible = {d.normalized.preemptible: d for d in demand}
    assert by_preemptible[True].normalized.device_type == DeviceType.TPU
    assert by_preemptible[True].task_ids == ["/test-user/j1/0"]
    assert by_preemptible[False].normalized.device_type == DeviceType.TPU
    assert by_preemptible[False].task_ids == ["/test-user/j2/0"]


def test_compute_demand_entries_no_preemptible_constraint_gives_none():
    """Job without preemptible constraint produces demand with preemptible=None."""
    state = _make_state()

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="unconstrained-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5p-8")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state._db)
    assert len(demand) == 1
    assert demand[0].normalized.preemptible is None


def test_compute_demand_entries_extracts_required_region():
    state = _make_state()
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="regional-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5p-8")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
        constraints=[
            cluster_pb2.Constraint(
                key=WellKnownAttribute.REGION,
                op=cluster_pb2.CONSTRAINT_OP_EQ,
                value=cluster_pb2.AttributeValue(string_value="us-west4"),
            )
        ],
    )
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state._db)
    assert len(demand) == 1
    assert demand[0].normalized.required_regions == frozenset({"us-west4"})
    assert demand[0].invalid_reason is None


def test_compute_demand_entries_marks_invalid_on_conflicting_region_constraints():
    state = _make_state()
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="invalid-regional-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5p-8")),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
        constraints=[
            cluster_pb2.Constraint(
                key=WellKnownAttribute.REGION,
                op=cluster_pb2.CONSTRAINT_OP_EQ,
                value=cluster_pb2.AttributeValue(string_value="us-west4"),
            ),
            cluster_pb2.Constraint(
                key=WellKnownAttribute.REGION,
                op=cluster_pb2.CONSTRAINT_OP_EQ,
                value=cluster_pb2.AttributeValue(string_value="eu-west4"),
            ),
        ],
    )
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state._db)
    assert len(demand) == 1
    assert demand[0].invalid_reason is not None


# =============================================================================
# Reservation Demand Deduplication Tests
# =============================================================================


def _make_reservation_job_request(
    *,
    task_device: cluster_pb2.DeviceConfig,
    reservation_devices: list[cluster_pb2.DeviceConfig],
    replicas: int = 1,
) -> cluster_pb2.Controller.LaunchJobRequest:
    """Build a LaunchJobRequest with a reservation and task resources.

    Each reservation entry gets auto-generated constraints from its device
    config, mirroring what the service layer does for the top-level request.
    This ensures holder jobs get the correct device constraints from the
    entry, not from the parent.
    """
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="reservation-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=task_device,
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=replicas,
    )
    for dev in reservation_devices:
        entry_resources = cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=dev,
        )
        entry_constraints = [c.to_proto() for c in constraints_from_resources(entry_resources)]
        req.reservation.entries.append(
            cluster_pb2.ReservationEntry(
                resources=entry_resources,
                constraints=entry_constraints,
            )
        )
    return req


def _h100_device() -> cluster_pb2.DeviceConfig:
    return cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(variant="H100", count=8))


def _a100_device() -> cluster_pb2.DeviceConfig:
    return cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(variant="A100", count=8))


def _is_synthetic_demand(state: ControllerTransitions, demand_entry: DemandEntry) -> bool:
    """Check if a demand entry comes from a holder job task."""
    for tid in demand_entry.task_ids:
        task = _query_task(state, JobName.from_string(tid))
        if task:
            job = _query_job(state, task.job_id)
            if job and job.is_reservation_holder:
                return True
    return False


def test_demand_reservation_all_tasks_generate_demand():
    """2 H100 reservation + 2 H100 tasks = 4 total demand (no budget dedup).

    All tasks generate demand through a unified path. Holder tasks and real
    tasks are independent demand sources — preemption during scheduling
    (not demand) handles the dedup.
    """
    state = _make_state()
    req = _make_reservation_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device(), _h100_device()],
        replicas=2,
    )
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state._db)
    synthetic_demand = [d for d in demand if _is_synthetic_demand(state, d)]
    real_demand = [d for d in demand if not _is_synthetic_demand(state, d)]

    assert len(synthetic_demand) == 2
    assert len(real_demand) == 2


def test_demand_reservation_excess_tasks():
    """2 H100 reservation + 5 H100 tasks = 2 synthetic + 5 real task demand."""
    state = _make_state()
    req = _make_reservation_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device(), _h100_device()],
        replicas=5,
    )
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state._db)
    synthetic_demand = [d for d in demand if _is_synthetic_demand(state, d)]
    real_demand = [d for d in demand if not _is_synthetic_demand(state, d)]

    assert len(synthetic_demand) == 2
    assert len(real_demand) == 5


def test_demand_reservation_holder_uses_entry_resources():
    """Holder tasks use the reservation entry's resource spec, not the parent's.

    Each reservation entry carries its own resources and constraints. The
    holder job uses the entry's resources so the autoscaler provisions the
    correct device type even when the parent job differs.
    """
    state = _make_state()
    # Job tasks request A100, but reservation entries specify H100.
    # Holder job should use the entry's H100 resource spec.
    req = _make_reservation_job_request(
        task_device=_a100_device(),
        reservation_devices=[_h100_device(), _h100_device()],
        replicas=2,
    )
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state._db)
    synthetic_demand = [d for d in demand if _is_synthetic_demand(state, d)]
    real_demand = [d for d in demand if not _is_synthetic_demand(state, d)]

    assert len(synthetic_demand) == 2
    assert len(real_demand) == 2
    # Holder demand uses entry's H100 device, not parent's A100
    for d in synthetic_demand:
        assert d.normalized.device_variants == frozenset({"h100"})


def test_demand_reservation_mixed_jobs():
    """Reservation job + regular job: demand is independent per job."""
    state = _make_state()

    # h100-job: 3 H100 tasks + 3 reservation entries
    h100_req = _make_reservation_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device(), _h100_device(), _h100_device()],
        replicas=3,
    )
    submit_job(state, "h100-job", h100_req)

    a100_req = cluster_pb2.Controller.LaunchJobRequest(
        name="a100-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_a100_device(),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=2,
    )
    submit_job(state, "a100-job", a100_req)

    demand = compute_demand_entries(state._db)
    synthetic_demand = [d for d in demand if _is_synthetic_demand(state, d)]
    real_demand = [d for d in demand if not _is_synthetic_demand(state, d)]

    # 3 synthetic holder tasks from h100-job's reservation
    assert len(synthetic_demand) == 3

    # h100-job: 3 real tasks + a100-job: 2 tasks = 5 real demand
    assert len(real_demand) == 5
    a100_demand = [d for d in real_demand if d.normalized.device_variants == frozenset({"a100"})]
    assert len(a100_demand) == 2


def test_demand_no_reservation_passes_all_tasks():
    """Job without reservation emits all task demand entries (no synthetic tasks)."""
    state = _make_state()
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="regular-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=3,
    )
    submit_job(state, "j1", req)

    demand = compute_demand_entries(state._db)
    assert len(demand) == 3
    for d in demand:
        assert not _is_synthetic_demand(state, d)


def test_demand_reservation_independent_per_job():
    """Each job's demand is independent — no cross-job interference."""
    state = _make_state()

    # Job A: 2 H100 reservation, 2 H100 tasks
    job_a_req = _make_reservation_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device(), _h100_device()],
        replicas=2,
    )
    submit_job(state, "job-a", job_a_req)

    # Job B: no reservation, 2 H100 tasks (must all pass through)
    job_b_req = cluster_pb2.Controller.LaunchJobRequest(
        name="job-b",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=2,
    )
    submit_job(state, "job-b", job_b_req)

    demand = compute_demand_entries(state._db)
    synthetic_demand = [d for d in demand if _is_synthetic_demand(state, d)]
    real_demand = [d for d in demand if not _is_synthetic_demand(state, d)]

    # Job A's 2 synthetic holder tasks
    assert len(synthetic_demand) == 2
    # Job A's 2 real tasks + Job B's 2 tasks = 4 real demand
    assert len(real_demand) == 4


# =============================================================================
# Depth-First Scheduling Priority Tests
# =============================================================================


def test_peek_pending_tasks_deeper_job_before_shallow(job_request):
    """Depth-first priority: deeper jobs come before shallow ones in queue order."""
    state = _make_state()

    # Submit root job and child job (both with 1 CPU)
    submit_job(state, "root", job_request("root"), timestamp_ms=1000)
    submit_job(state, "/test-user/root/child", job_request("child"), timestamp_ms=2000)

    pending = _schedulable_tasks(state)
    assert len(pending) == 2
    # Child (depth 2) should come first
    assert pending[0].job_id == JobName.from_string("/test-user/root/child")
    assert pending[1].job_id == JobName.root("test-user", "root")


def test_peek_pending_tasks_older_root_tree_preferred(job_request):
    """At same depth, older root tree is preferred."""
    state = _make_state()

    # Submit two root jobs at different timestamps
    req_a = job_request("user-a-job")
    submit_job(state, "user-a-job", req_a, timestamp_ms=1000)

    req_b = job_request("user-b-job")
    submit_job(state, "user-b-job", req_b, timestamp_ms=2000)

    pending = _schedulable_tasks(state)
    assert len(pending) == 2
    # user-a-job submitted first, should come first
    assert pending[0].job_id == JobName.root("test-user", "user-a-job")
    assert pending[1].job_id == JobName.root("test-user", "user-b-job")


def test_peek_pending_tasks_child_of_older_tree_beats_newer_root(job_request):
    """Child of older tree beats root of newer tree."""
    state = _make_state()

    # Submit old tree
    submit_job(state, "old-tree", job_request("old-tree"), timestamp_ms=1000)

    # Submit new tree
    submit_job(state, "new-tree", job_request("new-tree"), timestamp_ms=2000)

    # Submit child of old tree (depth 2) after new tree
    submit_job(state, "/test-user/old-tree/child", job_request("child"), timestamp_ms=3000)

    pending = _schedulable_tasks(state)
    assert len(pending) == 3

    # Expected order: child (depth 2), old-tree (depth 1, older), new-tree (depth 1, newer)
    assert pending[0].job_id == JobName.from_string("/test-user/old-tree/child")
    assert pending[1].job_id == JobName.root("test-user", "old-tree")
    assert pending[2].job_id == JobName.root("test-user", "new-tree")


def test_peek_pending_tasks_fifo_within_same_depth_and_tree(job_request):
    """FIFO within same depth and tree."""
    state = _make_state()

    # Submit parent first
    submit_job(state, "tree", job_request("tree"), timestamp_ms=1000)

    # Submit two children at different times
    submit_job(state, "/test-user/tree/child-a", job_request("child-a"), timestamp_ms=2000)
    submit_job(state, "/test-user/tree/child-b", job_request("child-b"), timestamp_ms=3000)

    pending = _schedulable_tasks(state)
    assert len(pending) == 3

    # Both children at depth 2, same root tree — child-a submitted first
    child_tasks = [t for t in pending if t.job_id.parent == JobName.root("test-user", "tree")]
    assert len(child_tasks) == 2
    assert child_tasks[0].job_id == JobName.from_string("/test-user/tree/child-a")
    assert child_tasks[1].job_id == JobName.from_string("/test-user/tree/child-b")


def test_child_job_inherits_root_submitted_at(job_request):
    """Child job inherits root_submitted_at from parent."""
    state = _make_state()

    # Submit parent at known time
    parent_req = job_request("parent")
    submit_job(state, "parent", parent_req, timestamp_ms=1000)
    parent_job = _query_job(state, JobName.root("test-user", "parent"))
    parent_submitted = parent_job.submitted_at

    # Submit child later
    child_req = job_request("child")
    submit_job(state, "/test-user/parent/child", child_req, timestamp_ms=2000)
    child_job = _query_job(state, JobName.from_string("/test-user/parent/child"))

    # Child's root_submitted_at should equal parent's
    assert child_job.root_submitted_at == parent_submitted
    assert child_job.root_submitted_at == parent_job.root_submitted_at


def test_requeued_task_maintains_priority_position(job_request, worker_metadata):
    """Requeued task maintains its priority position (deeper job still prioritized)."""
    state = _make_state()

    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    # Submit a deep job and a shallow job
    submit_job(state, "/test-user/tree/deep", job_request("deep"), timestamp_ms=1000)
    submit_job(state, "shallow", job_request("shallow"), timestamp_ms=2000)

    # Initially: deep job comes first
    pending = _schedulable_tasks(state)
    assert len(pending) == 2
    assert pending[0].job_id == JobName.from_string("/test-user/tree/deep")
    assert pending[1].job_id == JobName.root("test-user", "shallow")

    # Dispatch and fail the deep job's task (with retries enabled)
    deep_req = job_request("deep")
    deep_req.max_retries_failure = 1
    deep_tasks = submit_job(state, "/test-user/tree/deep-retry", deep_req, timestamp_ms=3000)
    submit_job(state, "shallow-2", job_request("shallow-2"), timestamp_ms=4000)

    dispatch_task(state, deep_tasks[0], worker_id)
    transition_task(state, deep_tasks[0].task_id, cluster_pb2.TASK_STATE_FAILED, error="Retriable failure")

    # Verify task was requeued
    assert deep_tasks[0].state == cluster_pb2.TASK_STATE_PENDING
    assert deep_tasks[0].can_be_scheduled()

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


def test_fail_heartbeat_clears_dispatch_when_worker_fails(job_request, worker_metadata):
    """Dispatch buffer is cleared when worker hits failure threshold.

    When consecutive heartbeat failures hit the threshold:
    1. Worker is marked unhealthy
    2. Running tasks transition to WORKER_FAILED
    3. Pending dispatch buffer is cleared (not orphaned)
    """
    state = _make_state()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    # Submit and dispatch a task
    tasks = submit_job(state, "j1", job_request("job1"))
    dispatch_task(state, tasks[0], worker_id)

    # Buffer a dispatch for the worker
    fake_request = cluster_pb2.Worker.RunTaskRequest(task_id="/test-user/fake/0")
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
    state.set_worker_consecutive_failures_for_test(worker_id, HEARTBEAT_FAILURE_THRESHOLD - 1)

    # This fail_heartbeat should trigger worker failure
    state.fail_heartbeat(snapshot, "Connection refused")

    # Verify worker is now unhealthy
    worker = _query_worker(state, worker_id)
    assert worker is None

    # Verify dispatch buffer was NOT repopulated (would be orphaned)
    # The fix clears the dispatch buffer when worker fails
    queued_run, _ = _queued_dispatch(state, worker_id)
    assert not queued_run


def test_fail_heartbeat_requeues_dispatch_for_retry(job_request, worker_metadata):
    """Heartbeat failure re-queues dispatches for the next heartbeat.

    We cannot tell whether the worker received the previous heartbeat (RPC
    timeout ≠ delivery failure), so we re-send the same RunTaskRequests.
    If the worker did receive them, it will reject re-sends as benign
    duplicates. If it did not, it will start them fresh.
    """

    state = _make_state()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("job1")
    req.max_retries_preemption = 5
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    # Assign the task (creates attempt, commits resources)
    state.queue_assignments([Assignment(task_id=task.task_id, worker_id=worker_id)])
    assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_ASSIGNED
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
    assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_ASSIGNED
    assert _query_task(state, task.task_id).preemption_count == 0
    assert _query_task(state, task.task_id).failure_count == 0

    # Dispatch re-queued for the next heartbeat (same attempt_id)
    queued_run, queued_kill = _queued_dispatch(state, worker_id)
    assert not queued_kill
    assert len(queued_run) == 1
    assert queued_run[0].attempt_id == 0


def test_complete_heartbeat_processes_task_states(job_request, worker_metadata):
    """complete_heartbeat properly processes task state changes from response."""
    state = _make_state()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    # Submit and dispatch a task
    tasks = submit_job(state, "j1", job_request("job1"))
    dispatch_task(state, tasks[0], worker_id)

    # Take snapshot
    snapshot = state.begin_heartbeat(worker_id)
    assert snapshot is not None

    # Create a mock response with completed task
    response = cluster_pb2.HeartbeatResponse(
        worker_healthy=True,
        tasks=[
            cluster_pb2.Controller.WorkerTaskStatus(
                task_id=tasks[0].task_id.to_wire(),
                state=cluster_pb2.TASK_STATE_SUCCEEDED,
                exit_code=0,
                attempt_id=0,
            )
        ],
    )

    # Complete heartbeat
    state.complete_heartbeat(snapshot, response)

    # Verify task is now succeeded
    task = _query_task(state, tasks[0].task_id)
    assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_SUCCEEDED

    # Verify job is succeeded
    job = _query_job(state, tasks[0].job_id)
    assert _query_job(state, job.job_id).state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_worker_failed_from_assigned_is_delivery_failure(job_request, worker_metadata):
    """WORKER_FAILED on a task still in ASSIGNED state is a delivery failure.

    When a task was assigned but never confirmed running (BUILDING/RUNNING),
    a WORKER_FAILED is a delivery failure — no budget is consumed. This
    prevents preemption count inflation from repeated 'Task not found' reports.
    """
    state = _make_state()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("job1")
    req.max_retries_preemption = 5
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    # Assign but do NOT transition to RUNNING
    state.queue_assignments([Assignment(task_id=task.task_id, worker_id=worker_id)])
    assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_ASSIGNED

    # Worker reports WORKER_FAILED (e.g., "Task not found on worker")
    transition_task(
        state,
        task.task_id,
        cluster_pb2.TASK_STATE_WORKER_FAILED,
        error="Task not found on worker",
    )

    # Delivery failure: no budget consumed at all
    assert _query_task(state, task.task_id).preemption_count == 0
    assert _query_task(state, task.task_id).failure_count == 0
    assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_PENDING
    assert _query_task(state, task.task_id).can_be_scheduled()


def test_worker_failed_from_running_counts_as_preemption(job_request, worker_metadata):
    """WORKER_FAILED on a task in RUNNING state counts as a preemption."""
    state = _make_state()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("job1")
    req.max_retries_preemption = 5
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    # Full lifecycle: assign and transition to RUNNING
    dispatch_task(state, task, worker_id)
    assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_RUNNING

    # Worker dies
    transition_task(
        state,
        task.task_id,
        cluster_pb2.TASK_STATE_WORKER_FAILED,
        error="Worker crashed",
    )

    # Real preemption: counts against preemption budget
    assert _query_task(state, task.task_id).preemption_count == 1
    assert _query_task(state, task.task_id).failure_count == 0
    assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_PENDING
    assert _query_task(state, task.task_id).can_be_scheduled()


def test_worker_failed_from_building_counts_as_preemption(job_request, worker_metadata):
    """WORKER_FAILED on a task in BUILDING state counts as a preemption."""
    state = _make_state()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    req = job_request("job1")
    req.max_retries_preemption = 5
    tasks = submit_job(state, "j1", req)
    task = tasks[0]

    # Assign and transition to BUILDING (worker confirmed it received the task)
    state.queue_assignments([Assignment(task_id=task.task_id, worker_id=worker_id)])
    transition_task(state, task.task_id, cluster_pb2.TASK_STATE_BUILDING)
    assert _query_task(state, task.task_id).state == cluster_pb2.TASK_STATE_BUILDING

    # Worker dies
    transition_task(
        state,
        task.task_id,
        cluster_pb2.TASK_STATE_WORKER_FAILED,
        error="Worker crashed",
    )

    # Real preemption: worker had started processing the task
    assert _query_task(state, task.task_id).preemption_count == 1
    assert _query_task(state, task.task_id).failure_count == 0


def test_fail_workers_by_ids_cascades_tasks(job_request, worker_metadata):
    """fail_workers_by_ids fails sibling workers and cascades their tasks."""
    state = _make_state()

    meta1 = worker_metadata()
    w1 = register_worker(state, "w1", "host1:8080", meta1)

    meta2 = worker_metadata()
    w2 = register_worker(state, "w2", "host2:8080", meta2)

    tasks1 = submit_job(state, "j1", job_request("job1"))
    dispatch_task(state, tasks1[0], w1)

    tasks2 = submit_job(state, "j2", job_request("job2"))
    dispatch_task(state, tasks2[0], w2)

    assert _query_task(state, tasks1[0].task_id).state == cluster_pb2.TASK_STATE_RUNNING
    assert _query_task(state, tasks2[0].task_id).state == cluster_pb2.TASK_STATE_RUNNING

    failed = state.fail_workers_by_ids(["w2"], reason="slice terminated")

    assert len(failed) == 1
    assert failed[0][0] == w2
    assert failed[0][1] == "host2:8080"

    t2 = _query_task(state, tasks2[0].task_id)
    assert t2.state in (cluster_pb2.TASK_STATE_WORKER_FAILED, cluster_pb2.TASK_STATE_PENDING)

    assert _query_task(state, tasks1[0].task_id).state == cluster_pb2.TASK_STATE_RUNNING
    assert _query_worker(state, w1) is not None
    assert _query_worker(state, w2) is None


def test_fail_workers_by_ids_skips_unknown(worker_metadata):
    """fail_workers_by_ids returns empty for unknown worker IDs."""
    state = _make_state()
    meta = worker_metadata()
    register_worker(state, "w1", "host1:8080", meta)

    failed = state.fail_workers_by_ids(["w-unknown"], reason="unknown")
    assert failed == []

    w = _query_worker(state, WorkerId("w1"))
    assert w is not None
    assert w.healthy


def test_fail_heartbeat_kills_requeue_only(job_request, worker_metadata):
    """Kill requests are still requeued on heartbeat failure (idempotent)."""
    state = _make_state()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    tasks = submit_job(state, "j1", job_request("job1"))
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


def _gpu_worker_metadata(
    *,
    cpu: int = 128,
    memory_gb: int = 256,
    variant: str = "H100",
    gpu_count: int = 8,
) -> cluster_pb2.WorkerMetadata:
    """Create worker metadata for a GPU worker with scheduling attributes."""
    return cluster_pb2.WorkerMetadata(
        hostname="gpu-worker",
        ip_address="10.0.0.1",
        cpu_count=cpu,
        memory_bytes=memory_gb * 1024**3,
        disk_bytes=100 * 1024**3,
        device=cluster_pb2.DeviceConfig(
            gpu=cluster_pb2.GpuDevice(variant=variant, count=gpu_count),
        ),
        attributes={
            WellKnownAttribute.DEVICE_TYPE: cluster_pb2.AttributeValue(string_value="gpu"),
            WellKnownAttribute.DEVICE_VARIANT: cluster_pb2.AttributeValue(string_value=variant.lower()),
            WellKnownAttribute.PREEMPTIBLE: cluster_pb2.AttributeValue(string_value="false"),
        },
    )


def _tpu_worker_metadata(
    *,
    cpu: int = 128,
    memory_gb: int = 256,
    variant: str = "v5litepod-16",
    chip_count: int = 8,
) -> cluster_pb2.WorkerMetadata:
    """Create worker metadata for a TPU worker with scheduling attributes."""
    return cluster_pb2.WorkerMetadata(
        hostname="tpu-worker",
        ip_address="10.0.0.1",
        cpu_count=cpu,
        memory_bytes=memory_gb * 1024**3,
        disk_bytes=100 * 1024**3,
        device=cluster_pb2.DeviceConfig(
            tpu=cluster_pb2.TpuDevice(variant=variant, chip_count=chip_count),
        ),
        attributes={
            WellKnownAttribute.DEVICE_TYPE: cluster_pb2.AttributeValue(string_value="tpu"),
            WellKnownAttribute.DEVICE_VARIANT: cluster_pb2.AttributeValue(string_value=variant.lower()),
            WellKnownAttribute.PREEMPTIBLE: cluster_pb2.AttributeValue(string_value="false"),
        },
    )


def _cpu_worker_metadata(
    *,
    cpu: int = 128,
    memory_gb: int = 256,
) -> cluster_pb2.WorkerMetadata:
    return cluster_pb2.WorkerMetadata(
        hostname="cpu-worker",
        ip_address="10.0.0.1",
        cpu_count=cpu,
        memory_bytes=memory_gb * 1024**3,
        disk_bytes=100 * 1024**3,
        device=cluster_pb2.DeviceConfig(
            cpu=cluster_pb2.CpuDevice(variant="cpu"),
        ),
        attributes={
            WellKnownAttribute.DEVICE_TYPE: cluster_pb2.AttributeValue(string_value="cpu"),
            WellKnownAttribute.PREEMPTIBLE: cluster_pb2.AttributeValue(string_value="false"),
        },
    )


def test_demand_excludes_building_limited_tasks():
    """Worker has resources but is at building limit -> no demand emitted."""
    state = _make_state()
    scheduler = Scheduler(max_building_tasks_per_worker=2)

    # Register a CPU worker with plenty of capacity
    wid = register_worker(state, "w1", "10.0.0.1:8080", _cpu_worker_metadata(cpu=128, memory_gb=256))

    # Submit a job with 1 pending CPU task
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="cpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j1", req)

    # Fill the worker with 2 building tasks (at the building limit).
    # These use minimal resources so the worker still has plenty of capacity.
    build_req = cluster_pb2.Controller.LaunchJobRequest(
        name="build-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=100,
            memory_bytes=1024**2,
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=2,
    )
    build_tasks = submit_job(state, "build-job", build_req)
    for bt in build_tasks:
        dispatch_task(state, bt, wid)
        transition_task(state, bt.task_id, cluster_pb2.TASK_STATE_BUILDING)

    # Now w1 has 2 building tasks (at limit), but has plenty of CPU/memory.
    # The pending task from j1 should be building-limited, not truly unschedulable.
    workers = _healthy_active_workers_with_attributes(state)
    demand = compute_demand_entries(state._db, scheduler, workers)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]
    assert len(task_demand) == 0, "Building-limited task should not generate demand"


def test_demand_includes_truly_unschedulable_tasks():
    """No worker with matching device type -> demand IS emitted."""
    state = _make_state()
    scheduler = Scheduler()

    # Register a CPU-only worker
    register_worker(state, "w1", "10.0.0.1:8080", _cpu_worker_metadata())

    # Submit a job requiring H100 GPUs
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="gpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j1", req)

    workers = _healthy_active_workers_with_attributes(state)
    demand = compute_demand_entries(state._db, scheduler, workers)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]
    assert len(task_demand) == 1, "Task with no matching device should generate demand"


def test_demand_includes_resource_exhausted_tasks():
    """Worker has right device but insufficient CPU -> demand IS emitted."""
    state = _make_state()
    scheduler = Scheduler()

    # Register a GPU worker with only 1 CPU core
    register_worker(state, "w1", "10.0.0.1:8080", _gpu_worker_metadata(cpu=1))

    # Submit a job requiring 4 CPU cores
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="gpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=4000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "j1", req)

    workers = _healthy_active_workers_with_attributes(state)
    demand = compute_demand_entries(state._db, scheduler, workers)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]
    assert len(task_demand) == 1, "Task exceeding worker CPU should generate demand"


def test_demand_holders_absorbed_by_dry_run():
    """Holder tasks participate in the dry-run and are absorbed when workers exist.

    Unlike the old design where holders always generated demand, they now
    participate in the dry-run like normal tasks and are absorbed when matching
    workers have available capacity.
    """
    state = _make_state()
    scheduler = Scheduler()

    # Register a large GPU worker with capacity for 1 task
    register_worker(state, "w1", "10.0.0.1:8080", _gpu_worker_metadata(cpu=2, memory_gb=4))

    # Submit a job with reservation (2 entries) and 2 tasks.
    # Worker can fit 1 task — so 1 task absorbed, 3 remain as demand.
    req = _make_reservation_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device(), _h100_device()],
        replicas=2,
    )
    submit_job(state, "j1", req)

    workers = _healthy_active_workers_with_attributes(state)
    demand = compute_demand_entries(state._db, scheduler, workers)
    # Worker fits 1 task (holder or real). 3 remaining generate demand.
    assert len(demand) == 3


def test_demand_absorbs_capacity_before_emitting():
    """2 workers fit 1 task each, 3 pending tasks -> only 1 demand entry."""
    state = _make_state()
    scheduler = Scheduler()

    # Register 2 GPU workers, each with enough capacity for 1 task
    register_worker(state, "w1", "10.0.0.1:8080", _gpu_worker_metadata(cpu=2, memory_gb=4))
    register_worker(state, "w2", "10.0.0.2:8080", _gpu_worker_metadata(cpu=2, memory_gb=4))

    # Submit 3 tasks each needing 2 CPU cores (each worker fits exactly 1)
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="gpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=2000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=3,
    )
    submit_job(state, "j1", req)

    workers = _healthy_active_workers_with_attributes(state)
    demand = compute_demand_entries(state._db, scheduler, workers)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]
    assert len(task_demand) == 1, "Only 1 of 3 tasks should generate demand (2 absorbed)"


def test_demand_no_workers_falls_back_to_all_pending():
    """When no workers provided, all pending tasks generate demand (backward compat)."""
    state = _make_state()

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="gpu-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=3,
    )
    submit_job(state, "j1", req)

    # No scheduler, no workers -> all tasks become demand
    demand = compute_demand_entries(state._db)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]
    assert len(task_demand) == 3


def test_demand_building_limited_with_multiple_workers():
    """All matching workers at building limit -> no demand, even with multiple workers."""
    state = _make_state()
    scheduler = Scheduler(max_building_tasks_per_worker=1)

    # Register 2 CPU workers
    wid1 = register_worker(state, "w1", "10.0.0.1:8080", _cpu_worker_metadata())
    wid2 = register_worker(state, "w2", "10.0.0.2:8080", _cpu_worker_metadata())

    # Fill both workers with 1 building task each (at limit since max=1).
    # Use minimal resources so workers retain plenty of capacity.
    for i, wid in enumerate([wid1, wid2]):
        build_req = cluster_pb2.Controller.LaunchJobRequest(
            name=f"build-{i}",
            entrypoint=_make_test_entrypoint(),
            resources=cluster_pb2.ResourceSpecProto(
                cpu_millicores=100,
                memory_bytes=1024**2,
            ),
            environment=cluster_pb2.EnvironmentConfig(),
            replicas=1,
        )
        build_tasks = submit_job(state, f"build-{i}", build_req)
        dispatch_task(state, build_tasks[0], wid)
        transition_task(state, build_tasks[0].task_id, cluster_pb2.TASK_STATE_BUILDING)

    # Submit a new task
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="pending-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "pending-job", req)

    workers = _healthy_active_workers_with_attributes(state)
    demand = compute_demand_entries(state._db, scheduler, workers)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]
    assert len(task_demand) == 0, "All workers at building limit -> no demand"


def test_demand_mixed_building_limited_and_unschedulable():
    """Some tasks building-limited, some truly unschedulable -> only unschedulable emit demand."""
    state = _make_state()
    scheduler = Scheduler(max_building_tasks_per_worker=1)

    # Register 1 GPU worker at building limit.
    # Use a minimal CPU task to fill the building slot so GPU capacity stays intact.
    wid = register_worker(state, "w1", "10.0.0.1:8080", _gpu_worker_metadata())
    build_req = cluster_pb2.Controller.LaunchJobRequest(
        name="build-0",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=100,
            memory_bytes=1024**2,
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    build_tasks = submit_job(state, "build-0", build_req)
    dispatch_task(state, build_tasks[0], wid)
    transition_task(state, build_tasks[0].task_id, cluster_pb2.TASK_STATE_BUILDING)

    # Task 1: H100 job (building-limited, worker has resources but at limit)
    h100_req = cluster_pb2.Controller.LaunchJobRequest(
        name="h100-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_h100_device(),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "h100-job", h100_req)

    # Task 2: A100 job (truly unschedulable, no A100 workers exist)
    a100_req = cluster_pb2.Controller.LaunchJobRequest(
        name="a100-job",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_a100_device(),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    submit_job(state, "a100-job", a100_req)

    workers = _healthy_active_workers_with_attributes(state)
    demand = compute_demand_entries(state._db, scheduler, workers)
    task_demand = [d for d in demand if not _is_synthetic_demand(state, d)]

    assert len(task_demand) == 1
    assert "a100-job" in task_demand[0].task_ids[0], "Only A100 task should emit demand"


# =============================================================================
# Holder Task Zero-Resource Tests
# =============================================================================


def test_holder_tasks_consume_zero_resources():
    """Holder tasks consume zero resources when assigned to workers."""
    state = _make_state()

    req = _make_reservation_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device()],
        replicas=1,
    )
    submit_job(state, "j1", req)

    wid = register_worker(state, "w1", "10.0.0.1:8080", _gpu_worker_metadata())
    holder_job_id = JobName.root("test-user", "j1").child(":reservation:")
    holder_tasks = _query_tasks_for_job(state, holder_job_id)
    assert len(holder_tasks) == 1

    worker_before = _query_worker(state, wid)
    gpus_before = worker_before.available_gpus

    # Assign holder task
    state.queue_assignments([Assignment(task_id=holder_tasks[0].task_id, worker_id=wid)])

    # Worker's available GPUs should NOT decrease (zero resources)
    worker_after = _query_worker(state, wid)
    assert worker_after.available_gpus == gpus_before

    # But the task should be tracked in running_tasks
    assert holder_tasks[0].task_id in _worker_running_tasks(state, wid)


def test_holder_task_cleanup_releases_no_resources():
    """When a holder task finishes, it doesn't release resources it never committed."""
    state = _make_state()

    req = _make_reservation_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device()],
        replicas=1,
    )
    submit_job(state, "j1", req)

    wid = register_worker(state, "w1", "10.0.0.1:8080", _gpu_worker_metadata())
    holder_job_id = JobName.root("test-user", "j1").child(":reservation:")
    holder_tasks = _query_tasks_for_job(state, holder_job_id)

    # Assign holder task
    state.queue_assignments([Assignment(task_id=holder_tasks[0].task_id, worker_id=wid)])

    worker_before = _query_worker(state, wid)
    gpus_before = worker_before.available_gpus

    # Kill the holder task via parent job cancellation
    parent_job_id = JobName.root("test-user", "j1")
    state.cancel_job(parent_job_id, reason="test")

    # Worker GPUs should be unchanged (nothing to release)
    worker_after = _query_worker(state, wid)
    assert worker_after.available_gpus == gpus_before


def test_holder_tasks_excluded_from_building_counts():
    """Holder tasks in ASSIGNED state should not consume building slots.

    Without this exclusion, a worker holding only a reservation task would be
    permanently "at building limit" and the real reserved task could never be
    assigned to that otherwise idle worker.
    """
    state = _make_state()

    req = _make_reservation_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device()],
        replicas=1,
    )
    submit_job(state, "j1", req)

    wid = register_worker(state, "w1", "10.0.0.1:8080", _gpu_worker_metadata())
    holder_job_id = JobName.root("test-user", "j1").child(":reservation:")
    holder_tasks = _query_tasks_for_job(state, holder_job_id)
    assert len(holder_tasks) == 1

    # Assign holder task — it goes to ASSIGNED state
    state.queue_assignments([Assignment(task_id=holder_tasks[0].task_id, worker_id=wid)])
    assert _query_task(state, holder_tasks[0].task_id).state == cluster_pb2.TASK_STATE_ASSIGNED

    # Building counts should NOT include the holder task
    building_counts = _building_counts(state)
    assert building_counts.get(wid, 0) == 0


def test_holder_tasks_excluded_from_heartbeat_expected_tasks():
    """Holder tasks must not appear in heartbeat expected_tasks.

    Holder tasks are virtual — never dispatched to the worker. If included
    in expected_tasks the worker reports "Task not found on worker", causing
    a worker_failed → retry loop (GH-3178).
    """
    state = _make_state()

    req = _make_reservation_job_request(
        task_device=_h100_device(),
        reservation_devices=[_h100_device()],
        replicas=1,
    )
    submit_job(state, "j1", req)

    wid = register_worker(state, "w1", "10.0.0.1:8080", _gpu_worker_metadata())
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


def test_snapshot_round_trip_preserves_reservation_holder():
    """DB checkpoint copy round-trip preserves is_reservation_holder flag."""
    import tempfile
    from pathlib import Path

    state = _make_state()

    req = _make_reservation_job_request(
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
        checkpoint_path = Path(tmpdir) / "checkpoint.sqlite3"
        state._db.backup_to(checkpoint_path)
        restored_db = ControllerDB(db_path=checkpoint_path)
        restored_state = ControllerTransitions(db=restored_db)

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


def test_worker_death_cascades_children_terminal(job_request, worker_metadata):
    """Single-task parent exhausts preemption retries -> job terminal -> children killed."""
    state = _make_state()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    # Parent job with 0 preemption retries so worker death => WORKER_FAILED (terminal task)
    parent_req = job_request("parent")
    parent_req.max_retries_preemption = 0
    parent_req.max_task_failures = 0
    parent_tasks = submit_job(state, "parent", parent_req)
    dispatch_task(state, parent_tasks[0], worker_id)

    # Child job under parent
    child_req = job_request("child")
    child_tasks = submit_job(state, "/test-user/parent/child", child_req)

    # Register new worker for child and dispatch
    w2 = register_worker(state, "w2", "host2:8080", worker_metadata())
    dispatch_task(state, child_tasks[0], w2)
    assert _query_task(state, child_tasks[0].task_id).state == cluster_pb2.TASK_STATE_RUNNING

    # Worker w1 dies — parent task exhausts preemption retries
    fail_worker(state, worker_id, "Connection lost")

    # Parent task should be terminal (WORKER_FAILED)
    parent_task = _query_task(state, parent_tasks[0].task_id)
    assert parent_task.state == cluster_pb2.TASK_STATE_WORKER_FAILED

    # Child should be killed via cascade
    child_task = _query_task(state, child_tasks[0].task_id)
    assert child_task.state == cluster_pb2.TASK_STATE_KILLED

    child_job = _query_job(state, JobName.from_string("/test-user/parent/child"))
    assert child_job.state == cluster_pb2.JOB_STATE_KILLED


def test_worker_death_preemption_policy_terminate(job_request, worker_metadata):
    """Single-task parent retried after worker death -> children killed (default TERMINATE)."""
    state = _make_state()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    # Parent with retries so task goes back to PENDING
    parent_req = job_request("parent")
    parent_req.max_retries_preemption = 5
    parent_tasks = submit_job(state, "parent", parent_req)
    dispatch_task(state, parent_tasks[0], worker_id)

    # Child job
    child_req = job_request("child")
    child_tasks = submit_job(state, "/test-user/parent/child", child_req)
    w2 = register_worker(state, "w2", "host2:8080", worker_metadata())
    dispatch_task(state, child_tasks[0], w2)
    assert _query_task(state, child_tasks[0].task_id).state == cluster_pb2.TASK_STATE_RUNNING

    # Worker w1 dies — parent task retried (goes to PENDING)
    fail_worker(state, worker_id, "Connection lost")

    # Parent task should be retried
    parent_task = _query_task(state, parent_tasks[0].task_id)
    assert parent_task.state == cluster_pb2.TASK_STATE_PENDING

    # Default policy for single-task job is TERMINATE_CHILDREN: child killed
    child_task = _query_task(state, child_tasks[0].task_id)
    assert child_task.state == cluster_pb2.TASK_STATE_KILLED

    child_job = _query_job(state, JobName.from_string("/test-user/parent/child"))
    assert child_job.state == cluster_pb2.JOB_STATE_KILLED


def test_worker_death_preemption_policy_preserve(job_request, worker_metadata):
    """Parent with PRESERVE_CHILDREN policy -> children survive worker death retry."""
    state = _make_state()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    # Parent with PRESERVE policy
    parent_req = job_request("parent")
    parent_req.max_retries_preemption = 5
    parent_req.preemption_policy = cluster_pb2.JOB_PREEMPTION_POLICY_PRESERVE_CHILDREN
    parent_tasks = submit_job(state, "parent", parent_req)
    dispatch_task(state, parent_tasks[0], worker_id)

    # Child job
    child_req = job_request("child")
    child_tasks = submit_job(state, "/test-user/parent/child", child_req)
    w2 = register_worker(state, "w2", "host2:8080", worker_metadata())
    dispatch_task(state, child_tasks[0], w2)
    assert _query_task(state, child_tasks[0].task_id).state == cluster_pb2.TASK_STATE_RUNNING

    # Worker w1 dies — parent task retried
    fail_worker(state, worker_id, "Connection lost")

    # Parent task goes back to PENDING
    parent_task = _query_task(state, parent_tasks[0].task_id)
    assert parent_task.state == cluster_pb2.TASK_STATE_PENDING

    # PRESERVE_CHILDREN: child stays alive
    child_task = _query_task(state, child_tasks[0].task_id)
    assert child_task.state == cluster_pb2.TASK_STATE_RUNNING

    child_job = _query_job(state, JobName.from_string("/test-user/parent/child"))
    assert child_job.state == cluster_pb2.JOB_STATE_RUNNING


def test_multi_task_parent_preserves_children(job_request, worker_metadata):
    """Multi-task parent (replicas > 1) -> children preserved by default on retry."""
    state = _make_state()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    # Multi-task parent (replicas=2) — default policy is PRESERVE_CHILDREN
    parent_req = cluster_pb2.Controller.LaunchJobRequest(
        name="multi-parent",
        entrypoint=_make_test_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=2,
        max_retries_preemption=5,
    )
    parent_tasks = submit_job(state, "parent", parent_req)
    dispatch_task(state, parent_tasks[0], worker_id)

    # Leave second parent task pending, dispatch child
    child_req = job_request("child")
    child_tasks = submit_job(state, "/test-user/parent/child", child_req)
    w2 = register_worker(state, "w2", "host2:8080", worker_metadata())
    dispatch_task(state, child_tasks[0], w2)
    assert _query_task(state, child_tasks[0].task_id).state == cluster_pb2.TASK_STATE_RUNNING

    # Worker w1 dies — parent task[0] retried
    fail_worker(state, worker_id, "Connection lost")

    parent_task = _query_task(state, parent_tasks[0].task_id)
    assert parent_task.state == cluster_pb2.TASK_STATE_PENDING

    # Multi-task default is PRESERVE_CHILDREN: child stays running
    child_task = _query_task(state, child_tasks[0].task_id)
    assert child_task.state == cluster_pb2.TASK_STATE_RUNNING


def test_task_update_worker_failed_cascades_children(job_request, worker_metadata):
    """apply_task_updates with WORKER_FAILED terminal task cascades children via preemption policy."""
    state = _make_state()
    worker_id = register_worker(state, "w1", "host:8080", worker_metadata())

    # Parent job with 0 preemption retries so WORKER_FAILED makes the task terminal
    parent_req = job_request("parent")
    parent_req.max_retries_preemption = 0
    parent_req.max_task_failures = 0
    parent_tasks = submit_job(state, "parent", parent_req)
    dispatch_task(state, parent_tasks[0], worker_id)

    # Child job under parent
    child_req = job_request("child")
    child_tasks = submit_job(state, "/test-user/parent/child", child_req)
    w2 = register_worker(state, "w2", "host2:8080", worker_metadata())
    dispatch_task(state, child_tasks[0], w2)
    assert _query_task(state, child_tasks[0].task_id).state == cluster_pb2.TASK_STATE_RUNNING

    # Report WORKER_FAILED via heartbeat update (goes through apply_task_updates)
    transition_task(state, parent_tasks[0].task_id, cluster_pb2.TASK_STATE_WORKER_FAILED, error="Worker crashed")

    # Parent task terminal
    parent_task = _query_task(state, parent_tasks[0].task_id)
    assert parent_task.state == cluster_pb2.TASK_STATE_WORKER_FAILED

    # Parent job should be WORKER_FAILED
    parent_job = _query_job(state, JobName.root("test-user", "parent"))
    assert parent_job.state == cluster_pb2.JOB_STATE_WORKER_FAILED

    # Child should be killed via cascade
    child_task = _query_task(state, child_tasks[0].task_id)
    assert child_task.state == cluster_pb2.TASK_STATE_KILLED

    child_job = _query_job(state, JobName.from_string("/test-user/parent/child"))
    assert child_job.state == cluster_pb2.JOB_STATE_KILLED
