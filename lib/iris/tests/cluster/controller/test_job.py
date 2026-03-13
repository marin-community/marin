# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DB-native tests for job/task state behavior and expansion."""


import pytest

from iris.cluster.controller.db import JOBS, TASKS, WORKERS, ControllerDB, Job, Task, Worker
from iris.cluster.controller.transitions import Assignment, ControllerTransitions, HeartbeatApplyRequest, TaskUpdate
from iris.cluster.log_store import LogStore
from iris.cluster.types import JobName, WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import Timestamp


def _query_job(db: ControllerDB, job_id: JobName) -> Job | None:
    with db.snapshot() as q:
        return q.one(JOBS, where=JOBS.c.job_id == job_id.to_wire())


def _query_task(db: ControllerDB, task_id: JobName) -> Task | None:
    with db.snapshot() as q:
        return q.one(TASKS, where=TASKS.c.task_id == task_id.to_wire())


def _query_worker(db: ControllerDB, worker_id: WorkerId) -> Worker | None:
    with db.snapshot() as q:
        return q.one(WORKERS, where=WORKERS.c.worker_id == str(worker_id))


def _query_tasks_for_job(db: ControllerDB, job_id: JobName) -> list[Task]:
    with db.snapshot() as q:
        return q.select(TASKS, where=TASKS.c.job_id == job_id.to_wire())


def _make_test_entrypoint() -> cluster_pb2.RuntimeEntrypoint:
    entrypoint = cluster_pb2.RuntimeEntrypoint()
    entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
    return entrypoint


@pytest.fixture
def state(tmp_path):
    db_path = tmp_path / "controller.sqlite3"
    db = ControllerDB(db_path=db_path)
    log_store = LogStore(db_path=db_path)
    s = ControllerTransitions(db=db, log_store=log_store)
    yield s
    log_store.close()
    db.close()


@pytest.fixture
def make_job_request():
    def _make(name: str = "test-job") -> cluster_pb2.Controller.LaunchJobRequest:
        return cluster_pb2.Controller.LaunchJobRequest(
            name=name,
            entrypoint=_make_test_entrypoint(),
            resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            environment=cluster_pb2.EnvironmentConfig(),
            replicas=1,
        )

    return _make


def _register_worker(state: ControllerTransitions, worker_id: str) -> WorkerId:
    wid = WorkerId(worker_id)
    metadata = cluster_pb2.WorkerMetadata(
        hostname=worker_id,
        ip_address="127.0.0.1",
        cpu_count=8,
        memory_bytes=16 * 1024**3,
        disk_bytes=100 * 1024**3,
    )
    state.register_or_refresh_worker(wid, f"{worker_id}:8080", metadata, Timestamp.now())
    return wid


def _run_task_to_state(state: ControllerTransitions, task_id: JobName, worker_id: WorkerId, new_state: int) -> None:
    state.queue_assignments([Assignment(task_id=task_id, worker_id=worker_id)])
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=worker_id,
            worker_resource_snapshot=None,
            updates=[TaskUpdate(task_id=task_id, attempt_id=0, new_state=cluster_pb2.TASK_STATE_RUNNING)],
        )
    )
    if new_state != cluster_pb2.TASK_STATE_RUNNING:
        state.apply_task_updates(
            HeartbeatApplyRequest(
                worker_id=worker_id,
                worker_resource_snapshot=None,
                updates=[TaskUpdate(task_id=task_id, attempt_id=0, new_state=new_state)],
            )
        )


def test_job_becomes_succeeded_when_all_tasks_succeed(state: ControllerTransitions, make_job_request) -> None:
    request = make_job_request()
    request.replicas = 2
    jid = JobName.root("test-user", "all-succeeded")
    request.name = jid.to_wire()
    state.submit_job(jid, request, Timestamp.now())

    wid = _register_worker(state, "w1")
    for task in _query_tasks_for_job(state._db, jid):
        _run_task_to_state(state, task.task_id, wid, cluster_pb2.TASK_STATE_SUCCEEDED)

    job = _query_job(state._db, jid)
    assert job is not None
    assert job.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_job_failure_threshold_applies(state: ControllerTransitions, make_job_request) -> None:
    request = make_job_request()
    request.replicas = 2
    request.max_task_failures = 0
    jid = JobName.root("test-user", "fail-fast")
    request.name = jid.to_wire()
    state.submit_job(jid, request, Timestamp.now())

    wid = _register_worker(state, "w1")
    first = _query_tasks_for_job(state._db, jid)[0]
    _run_task_to_state(state, first.task_id, wid, cluster_pb2.TASK_STATE_FAILED)

    job = _query_job(state._db, jid)
    assert job is not None
    assert job.state == cluster_pb2.JOB_STATE_FAILED


def test_job_expands_to_replicas_and_retry_limits(state: ControllerTransitions, make_job_request) -> None:
    request = make_job_request()
    request.replicas = 3
    request.max_retries_failure = 3
    request.max_retries_preemption = 7
    jid = JobName.root("test-user", "expand")
    request.name = jid.to_wire()

    state.submit_job(jid, request, Timestamp.now())
    tasks = _query_tasks_for_job(state._db, jid)

    assert len(tasks) == 3
    for idx, task in enumerate(tasks):
        assert task.task_id == jid.task(idx)
        assert task.max_retries_failure == 3
        assert task.max_retries_preemption == 7


def test_job_becomes_unschedulable_when_task_unschedulable(state: ControllerTransitions, make_job_request) -> None:
    request = make_job_request()
    request.replicas = 2
    jid = JobName.root("test-user", "unsched")
    request.name = jid.to_wire()
    state.submit_job(jid, request, Timestamp.now())

    first_task = _query_tasks_for_job(state._db, jid)[0]
    state.mark_task_unschedulable(first_task.task_id, reason="no capacity")

    job = _query_job(state._db, jid)
    assert job is not None
    assert job.state == cluster_pb2.JOB_STATE_UNSCHEDULABLE


def test_job_cancel_marks_job_killed(state: ControllerTransitions, make_job_request) -> None:
    request = make_job_request()
    request.replicas = 2
    jid = JobName.root("test-user", "killed")
    request.name = jid.to_wire()
    state.submit_job(jid, request, Timestamp.now())

    state.cancel_job(jid, reason="manual")
    job = _query_job(state._db, jid)
    assert job is not None
    assert job.state == cluster_pb2.JOB_STATE_KILLED
