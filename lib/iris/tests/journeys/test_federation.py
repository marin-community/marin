# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Concise multi-controller federation journeys."""

import pytest
from iris.cluster.resources.endpoint import EndpointQuery
from iris.rpc import job_pb2
from tests.journeys.federation import PARENT_CLUSTER_ID, PEER_ID, FederationJourney


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

    assert [task.task_id for task in federation.peer_tasks(root)] == [root[0].wire_id]
    assert {task.task_id for task in federation.parent_tasks(child)} == {child[0].wire_id, child[1].wire_id}


def test_federated_endpoint_keeps_authority_and_execution_coordinates_distinct(
    federation: FederationJourney,
) -> None:
    job = federation.submit("endpoint")
    federation.promote()
    federation.sync()
    federation.run_peer()
    federation.peer.register_endpoint(job[0], "/serve/endpoint", "10.0.0.7:8000", endpoint_id="endpoint-1")

    (received,) = federation.peer.controller.resources.list_endpoints(EndpointQuery()).items
    assert received.key.cluster_id == PARENT_CLUSTER_ID
    assert received.task is not None and received.task.cluster_id == PARENT_CLUSTER_ID
    assert received.execution_cluster_id == PEER_ID

    federation.sync()

    (mirrored,) = federation.parent.controller.resources.list_endpoints(EndpointQuery()).items
    assert mirrored.key.cluster_id == PARENT_CLUSTER_ID
    assert mirrored.task is not None and mirrored.task.cluster_id == PARENT_CLUSTER_ID
    assert mirrored.execution_cluster_id == PEER_ID
