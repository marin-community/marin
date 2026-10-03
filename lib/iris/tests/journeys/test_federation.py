# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Concise multi-controller federation journeys."""

import pytest
from iris.rpc import controller_pb2, job_pb2
from iris.testing.journeys.federation import PEER_ID, FederationJourney


@pytest.fixture
def federation(tmp_path, monkeypatch):
    journey = FederationJourney(tmp_path, monkeypatch)
    try:
        yield journey
    finally:
        journey.close()


def test_federated_job_runs_on_peer_and_syncs_attempt_to_parent(federation: FederationJourney) -> None:
    job = federation.submit("train", tasks=2)

    federation.promote()
    assert federation.parent_job(job).pending_reason == f"Awaiting acceptance by peer {PEER_ID}"

    federation.sync()
    assert {task.task_id for task in federation.peer_tasks(job)} == {job[0].wire_id, job[1].wire_id}

    federation.run_peer()
    federation.succeed_on_peer(job[0])
    federation.sync()

    parent_tasks = federation.parent_tasks(job)
    assert [task.cluster for task in parent_tasks] == [PEER_ID, PEER_ID]
    assert parent_tasks[0].state == job_pb2.TASK_STATE_SUCCEEDED
    assert [attempt.state for attempt in parent_tasks[0].attempts] == [job_pb2.TASK_STATE_SUCCEEDED]
    assert federation.parent_job(job).peer_status == job_pb2.PEER_STATUS_SYNCED


def test_peer_outage_preserves_last_known_state_until_recovery_sync(federation: FederationJourney) -> None:
    job = federation.submit("recover")
    federation.promote()
    federation.sync()
    federation.run_peer()
    federation.sync()
    assert federation.parent_tasks(job)[0].state == job_pb2.TASK_STATE_RUNNING

    federation.set_peer_reachable(False)
    federation.succeed_on_peer(job[0])
    federation.sync()

    assert not federation.peer_summary().reachable
    assert federation.parent_tasks(job)[0].state == job_pb2.TASK_STATE_RUNNING
    assert federation.parent_job(job).state == job_pb2.JOB_STATE_RUNNING

    federation.set_peer_reachable(True)
    federation.sync()

    assert federation.peer_summary().reachable
    assert federation.parent_tasks(job)[0].state == job_pb2.TASK_STATE_SUCCEEDED
    assert federation.parent_job(job).state == job_pb2.JOB_STATE_SUCCEEDED


def test_unreachable_handoff_reports_awaiting_peer_and_recovers(federation: FederationJourney) -> None:
    job = federation.submit("delayed")
    federation.promote()
    federation.set_peer_reachable(False)

    federation.sync()

    status = federation.parent_job(job)
    assert status.state == job_pb2.JOB_STATE_PENDING
    assert status.pending_reason == f"Awaiting acceptance by peer {PEER_ID}"
    assert federation.peer_tasks(job) == []

    federation.set_peer_reachable(True)
    federation.sync()

    assert [task.task_id for task in federation.peer_tasks(job)] == [job[0].wire_id]
    assert federation.parent_tasks(job)[0].cluster == PEER_ID


def test_parent_cancel_terminates_job_on_peer(federation: FederationJourney) -> None:
    job = federation.submit("cancel")
    federation.promote()
    federation.sync()
    federation.run_peer()

    federation.cancel(job)
    federation.sync()

    assert federation.peer_job(job).state == job_pb2.JOB_STATE_KILLED
    assert federation.parent_job(job).state == job_pb2.JOB_STATE_KILLED


def test_execution_peer_submits_a_child_and_syncs_the_subtree_to_authority_once(federation: FederationJourney) -> None:
    root = federation.submit("tree")
    federation.promote()
    federation.sync()
    child = federation.submit_child_on_peer(root, "child", tasks=2)
    federation.sync()

    assert {task.task_id for task in federation.parent_tasks(child)} == {child[0].wire_id, child[1].wire_id}

    federation.sync()

    assert [task.task_id for task in federation.peer_tasks(root)] == [root[0].wire_id]
    assert {task.task_id for task in federation.parent_tasks(child)} == {child[0].wire_id, child[1].wire_id}


@pytest.mark.parametrize("target_kind", ["task", "attempt", "job"])
def test_parent_kick_retries_child_on_execution_peer(federation: FederationJourney, target_kind) -> None:
    root = federation.submit("kick-tree")
    federation.promote()
    federation.sync()
    child = federation.peer.submit_child(root, "train", preemption_retries=2)
    federation.run_peer()
    # The child has not been mirrored yet; the peer owns target resolution.
    target = child.wire_id if target_kind == "job" else child[0].wire_id
    if target_kind == "attempt":
        target += ":0"
    response = federation.parent.controller.kick_tasks(
        controller_pb2.Controller.KickTasksRequest(
            targets=[target], desired_state=job_pb2.TASK_STATE_PREEMPTED, reason="operator recovery"
        )
    )
    assert response.results[0].queued
    federation.peer.settle()
    federation.sync()
    task = federation.parent_tasks(child)[0]
    assert task.current_attempt_id == 1
    assert task.state == job_pb2.TASK_STATE_RUNNING
    attempt = federation.peer.task(child[0]).attempts[0]
    assert attempt.state == job_pb2.TASK_STATE_PREEMPTED
    assert attempt.error == "operator recovery"
    assert any(
        event.kind == "stopped" and event.task_id == child[0].wire_id and event.attempt_id == 0
        for event in federation.peer.backend.events
    )


def test_parent_kick_rejects_stale_attempt_on_peer(federation: FederationJourney) -> None:
    root = federation.submit("stale-kick")
    federation.promote()
    federation.sync()
    child = federation.peer.submit_child(root, "train", preemption_retries=2)
    federation.run_peer()
    federation.sync()
    assert federation.peer.kick(child[0], attempt_id=0).queued
    federation.peer.settle()
    # Parent still reports attempt 0, but execution has advanced to attempt 1.
    response = federation.parent.kick(child[0], attempt_id=0)
    assert not response.queued
    federation.peer.settle()
    assert federation.peer_tasks(child)[0].current_attempt_id == 1


def test_parent_kick_unreachable_peer_does_not_acknowledge_success(federation: FederationJourney) -> None:
    job = federation.submit("unreachable-kick")
    federation.promote()
    federation.sync()
    federation.run_peer()
    federation.sync()
    federation.set_peer_reachable(False)
    with pytest.raises(ConnectionError):
        federation.parent.kick(job[0])
    assert federation.peer_tasks(job)[0].state == job_pb2.TASK_STATE_RUNNING
