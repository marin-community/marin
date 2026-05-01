# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Heartbeats no longer persist task resource usage to the controller DB.

Commit 3 of the controller-stats migration drops every ``insert_resource_usage``
call site; the table itself sticks around until commit 4. This test pins the
new contract: a steady-state heartbeat carrying ``ResourceUsage`` produces zero
rows in ``task_resource_history``.
"""

import pytest
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.stores import ControllerStore
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    HeartbeatApplyRequest,
    TaskUpdate,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Timestamp


@pytest.fixture
def state(tmp_path):
    db = ControllerDB(db_dir=tmp_path)
    s = ControllerTransitions(store=ControllerStore(db))
    yield s
    db.close()


WORKER_META = job_pb2.WorkerMetadata(
    hostname="test-host",
    ip_address="10.0.0.1",
    cpu_count=8,
    memory_bytes=16 * 1024**3,
    disk_bytes=100 * 1024**3,
)


def _setup_running_task(state: ControllerTransitions) -> tuple[JobName, JobName]:
    wid = WorkerId("w1")
    with state._store.transaction() as cur:
        state.register_or_refresh_worker(
            cur, worker_id=wid, address="host:8080", metadata=WORKER_META, ts=Timestamp.now()
        )

    job_id = JobName.from_wire("/user/test-job")
    with state._store.transaction() as cur:
        state.submit_job(
            cur,
            job_id,
            controller_pb2.Controller.LaunchJobRequest(name="/user/test-job", replicas=1),
            Timestamp.now(),
        )
    task_id = job_id.task(0)
    with state._store.transaction() as cur:
        state.queue_assignments(cur, [Assignment(task_id=task_id, worker_id=wid)])
    with state._store.transaction() as cur:
        state.apply_task_updates(
            cur,
            HeartbeatApplyRequest(
                worker_id=wid,
                updates=[TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING)],
            ),
        )
    return job_id, task_id


def _count_history_rows(state: ControllerTransitions, task_id: JobName) -> int:
    with state._db.read_snapshot() as q:
        rows = q.raw(
            "SELECT COUNT(*) as cnt FROM task_resource_history WHERE task_id = ?",
            (task_id.to_wire(),),
        )
    return rows[0].cnt


def test_heartbeat_no_longer_persists_resource_usage(state):
    """Steady-state heartbeats carrying ResourceUsage write zero rows.

    Per-attempt resource samples now flow through the ``iris.task`` stats
    namespace; the controller DB only tracks decisions.
    """
    _, task_id = _setup_running_task(state)
    assert _count_history_rows(state, task_id) == 0

    usage = job_pb2.ResourceUsage(cpu_millicores=1000, memory_mb=512, disk_mb=10)
    with state._store.transaction() as cur:
        state.apply_heartbeats_batch(
            cur,
            [
                HeartbeatApplyRequest(
                    worker_id=WorkerId("w1"),
                    updates=[
                        TaskUpdate(
                            task_id=task_id,
                            attempt_id=0,
                            new_state=job_pb2.TASK_STATE_RUNNING,
                            resource_usage=usage,
                        ),
                    ],
                )
            ],
        )

    assert _count_history_rows(state, task_id) == 0


def test_terminal_transition_no_longer_persists_resource_usage(state):
    """Terminal-transition heartbeats with ResourceUsage also write zero rows."""
    _, task_id = _setup_running_task(state)
    usage = job_pb2.ResourceUsage(cpu_millicores=2000, memory_mb=1024, disk_mb=50)
    with state._store.transaction() as cur:
        state.apply_task_updates(
            cur,
            HeartbeatApplyRequest(
                worker_id=WorkerId("w1"),
                updates=[
                    TaskUpdate(
                        task_id=task_id,
                        attempt_id=0,
                        new_state=job_pb2.TASK_STATE_SUCCEEDED,
                        resource_usage=usage,
                        exit_code=0,
                    ),
                ],
            ),
        )
    assert _count_history_rows(state, task_id) == 0
