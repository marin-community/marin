# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the remote backend's relay as driven through RemoteTaskBackend.reconcile."""

from iris.cluster.backends.remote.backend import RemoteTaskBackend
from iris.cluster.controller.reads import ControlSnapshot
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.types import JobName
from iris.rpc import job_pb2, remote_agent_pb2


def make_req(task_id: str, attempt_id: int, attempt_uid: str) -> job_pb2.RunTaskRequest:
    req = job_pb2.RunTaskRequest()
    req.task_id = task_id
    req.attempt_id = attempt_id
    req.attempt_uid = attempt_uid
    req.entrypoint.run_command.argv.extend(["echo", "hi"])
    return req


def snapshot(tasks_to_run=None, running_tasks=None) -> ControlSnapshot:
    return ControlSnapshot(
        worker_addresses={},
        reconcile_rows=[],
        timeout_rows=[],
        tasks_to_run=tasks_to_run or [],
        running_tasks=running_tasks or [],
    )


def running(req: job_pb2.RunTaskRequest) -> RunningTaskEntry:
    return RunningTaskEntry(task_id=JobName.from_wire(req.task_id), attempt_id=req.attempt_id)


def test_reconcile_publishes_desired_then_returns_update_for_observation():
    backend = RemoteTaskBackend()
    req_a = make_req("/job/a/0", attempt_id=3, attempt_uid="uid-a")
    req_b = make_req("/job/b/0", attempt_id=1, attempt_uid="uid-b")

    first = backend.reconcile(snapshot(tasks_to_run=[req_a, req_b]))
    assert first.updates == []
    assert set(backend.relay.desired) == {"uid-a", "uid-b"}
    sync_after_publish = backend.relay.sync_id
    assert sync_after_publish > 0

    backend.relay.ingest_observations(
        [remote_agent_pb2.AgentObservation(attempt_uid="uid-a", state=job_pb2.TASK_STATE_RUNNING)]
    )

    # Re-publishing the same set keeps both attempts desired (so the observation
    # still resolves) and must not bump sync_id.
    second = backend.reconcile(snapshot(tasks_to_run=[req_a, req_b]))
    assert backend.relay.sync_id == sync_after_publish

    assert len(second.updates) == 1
    update = second.updates[0]
    assert update.task_id == JobName.from_wire("/job/a/0")
    assert update.attempt_id == 3
    assert update.new_state == job_pb2.TASK_STATE_RUNNING


def test_running_task_without_fresh_dispatch_stays_desired():
    backend = RemoteTaskBackend()
    req = make_req("/job/a/0", attempt_id=0, attempt_uid="uid-a")

    backend.reconcile(snapshot(tasks_to_run=[req]))
    # Next tick the task is no longer pending dispatch but is running: the relay
    # resolves it to the cached spec via key_to_uid and keeps it desired.
    backend.reconcile(snapshot(running_tasks=[running(req)]))

    assert set(backend.relay.desired) == {"uid-a"}


def test_dropped_attempt_leaves_desired_set_and_bumps_sync_id():
    backend = RemoteTaskBackend()
    req_a = make_req("/job/a/0", attempt_id=0, attempt_uid="uid-a")
    req_b = make_req("/job/b/0", attempt_id=0, attempt_uid="uid-b")

    backend.reconcile(snapshot(tasks_to_run=[req_a, req_b]))
    sync_with_both = backend.relay.sync_id

    backend.reconcile(snapshot(tasks_to_run=[req_a]))

    assert set(backend.relay.desired) == {"uid-a"}
    assert backend.relay.sync_id > sync_with_both


def test_observation_for_unknown_uid_is_skipped():
    backend = RemoteTaskBackend()
    req_a = make_req("/job/a/0", attempt_id=0, attempt_uid="uid-a")

    backend.reconcile(snapshot(tasks_to_run=[req_a]))
    backend.relay.ingest_observations(
        [
            remote_agent_pb2.AgentObservation(attempt_uid="uid-ghost", state=job_pb2.TASK_STATE_RUNNING),
            remote_agent_pb2.AgentObservation(attempt_uid="uid-a", state=job_pb2.TASK_STATE_RUNNING),
        ]
    )

    result = backend.reconcile(snapshot(tasks_to_run=[req_a]))

    assert [u.task_id for u in result.updates] == [JobName.from_wire("/job/a/0")]
