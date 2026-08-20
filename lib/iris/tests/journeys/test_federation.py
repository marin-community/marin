# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Concise multi-controller federation journeys."""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.federation.protocol import PeerCallError
from iris.resources.action import ActionResult, ActionState
from iris.resources.endpoint import EndpointQuery
from iris.resources.job import JobQuery
from iris.resources.source import Freshness, SourceState
from iris.rpc import job_pb2
from iris.testing.journeys.federation import PARENT_CLUSTER_ID, PEER_ID, FederationJourney
from rigging.timing import Duration


@pytest.fixture
def federation(tmp_path, monkeypatch):
    journey = FederationJourney(tmp_path, monkeypatch)
    try:
        yield journey
    finally:
        journey.close()


def test_federated_job_runs_on_peer_and_syncs_attempt_to_parent(federation: FederationJourney) -> None:
    job = federation.submit("train", tasks=2)
    queued_identity = federation.parent_job(job).summary.identity

    federation.promote()
    promoted = federation.parent_job(job).summary
    assert promoted.identity == queued_identity
    assert promoted.state == job_pb2.JOB_STATE_PENDING
    assert promoted.execution_cluster_id == PEER_ID

    federation.sync()
    parent_job = federation.parent_job(job).summary
    peer_job = federation.peer_job(job).summary
    parent_tasks = federation.parent_tasks(job)
    peer_tasks = federation.peer_tasks(job)
    assert parent_job.identity == peer_job.identity == queued_identity
    assert [task.identity for task in parent_tasks] == [task.identity for task in peer_tasks]
    task_identities = [task.identity for task in parent_tasks]
    assert {task.identity.key.resource_id for task in peer_tasks} == {
        job[0].wire_id,
        job[1].wire_id,
    }

    federation.run_peer()
    federation.succeed_on_peer(job[0])
    federation.sync()

    parent_tasks = federation.parent_tasks(job)
    assert [task.identity for task in parent_tasks] == task_identities
    assert [task.execution_cluster_id for task in parent_tasks] == [PEER_ID, PEER_ID]
    assert parent_tasks[0].state == job_pb2.TASK_STATE_SUCCEEDED
    assert [attempt.state for attempt in federation.parent.task(job[0]).attempts] == [job_pb2.TASK_STATE_SUCCEEDED]


def test_peer_outage_and_parent_restart_preserve_last_known_state_until_recovery_sync(
    federation: FederationJourney,
) -> None:
    job = federation.submit("recover")
    federation.promote()
    federation.sync()
    federation.run_peer()
    federation.sync()
    assert federation.parent_tasks(job)[0].state == job_pb2.TASK_STATE_RUNNING

    federation.set_peer_reachable(False)
    federation.succeed_on_peer(job[0])
    federation.sync()
    federation.restart_parent()

    assert not federation.peer_summary().reachable
    cached = federation.parent.task(job[0])
    assert cached.summary.state == job_pb2.TASK_STATE_RUNNING
    assert federation.parent_job(job).summary.state == job_pb2.JOB_STATE_RUNNING
    peer_source = next(status for status in cached.source_statuses if status.source_id == f"federation:{PEER_ID}")
    assert peer_source.state is SourceState.UNAVAILABLE
    assert peer_source.freshness is Freshness.STALE
    jobs_page = federation.parent.controller.controller.list_jobs(JobQuery())
    assert any(status == peer_source for status in jobs_page.source_statuses)

    federation.set_peer_reachable(True)
    federation.sync()

    assert federation.peer_summary().reachable
    assert federation.parent_tasks(job)[0].state == job_pb2.TASK_STATE_SUCCEEDED
    assert federation.parent_job(job).summary.state == job_pb2.JOB_STATE_SUCCEEDED


def test_unreachable_handoff_reports_awaiting_peer_and_recovers(federation: FederationJourney) -> None:
    job = federation.submit("delayed")
    federation.promote()
    federation.set_peer_reachable(False)

    federation.sync()

    status = federation.parent_job(job)
    assert status.summary.state == job_pb2.JOB_STATE_PENDING
    assert status.summary.execution_cluster_id == PEER_ID
    assert federation.peer_tasks(job) == []

    federation.set_peer_reachable(True)
    federation.sync()

    assert [task.identity.key.resource_id for task in federation.peer_tasks(job)] == [job[0].wire_id]
    assert federation.parent_tasks(job)[0].execution_cluster_id == PEER_ID


def test_parent_cancel_terminates_job_on_peer(federation: FederationJourney) -> None:
    job = federation.submit("cancel")
    federation.promote()
    federation.sync()
    federation.run_peer()

    federation.cancel(job)
    federation.sync()

    assert federation.peer_job(job).summary.state == job_pb2.JOB_STATE_KILLED
    assert federation.parent_job(job).summary.state == job_pb2.JOB_STATE_KILLED


def test_federated_cancel_intent_survives_peer_outage_and_parent_restart(federation: FederationJourney) -> None:
    job = federation.submit("cancel-restart")
    federation.promote()
    federation.sync()
    federation.run_peer()
    identity = federation.parent_job(job).summary.identity
    federation.set_peer_reachable(False)

    receipt = federation.parent.cancel_job(identity, idempotency_key="cancel-restart")
    federation.restart_parent()
    duplicate = federation.parent.cancel_job(identity, idempotency_key="cancel-restart")
    federation.set_peer_reachable(True)
    federation.sync()

    assert duplicate.action_id == receipt.action_id
    assert receipt.state is ActionState.SUCCEEDED
    assert receipt.result_code is ActionResult.SATISFIED
    assert federation.peer_job(job).summary.state == job_pb2.JOB_STATE_KILLED
    assert federation.parent_job(job).summary.state == job_pb2.JOB_STATE_KILLED


def test_federated_retry_mutates_execution_peer_then_syncs_one_exact_attempt(federation: FederationJourney) -> None:
    job = federation.submit("retry", preemption_retries=1)
    federation.promote()
    federation.sync()
    federation.run_peer()
    federation.sync()
    before = federation.parent.task(job[0])
    current = before.summary.current_attempt
    assert current is not None

    receipt = federation.parent.retry_task(
        before.summary.identity,
        expected_attempt_uid=current.attempt_uid,
        idempotency_key="federated-retry",
    )
    federation.run_peer()
    federation.sync()

    after = federation.parent.task(job[0])
    assert receipt.state is ActionState.SUCCEEDED
    assert receipt.result_code is ActionResult.TARGET_ABSENT
    assert after.summary.current_attempt is not None
    assert after.summary.current_attempt.attempt_uid != current.attempt_uid
    peer_after = federation.peer.task(job[0])
    assert after.summary == peer_after.summary
    assert [(item.identity, item.state, item.backend_id) for item in after.attempts] == [
        (item.identity, item.state, item.backend_id) for item in peer_after.attempts
    ]


def test_federated_action_outage_has_no_authority_or_execution_side_effect(federation: FederationJourney) -> None:
    job = federation.submit("retry-outage", preemption_retries=1)
    federation.promote()
    federation.sync()
    federation.run_peer()
    federation.sync()
    before_parent = federation.parent.task(job[0])
    before_peer = federation.peer.task(job[0])
    current = before_parent.summary.current_attempt
    assert current is not None
    federation.set_peer_reachable(False)

    with pytest.raises(PeerCallError, match="unreachable"):
        federation.parent.retry_task(
            before_parent.summary.identity,
            expected_attempt_uid=current.attempt_uid,
            idempotency_key="retry-outage",
        )

    after_parent = federation.parent.task(job[0])
    after_peer = federation.peer.task(job[0])
    assert (after_parent.summary, after_parent.attempts, after_parent.root_cause_highlights) == (
        before_parent.summary,
        before_parent.attempts,
        before_parent.root_cause_highlights,
    )
    assert (after_peer.summary, after_peer.attempts, after_peer.root_cause_highlights) == (
        before_peer.summary,
        before_peer.attempts,
        before_peer.root_cause_highlights,
    )


def test_federated_terminate_targets_execution_peers_exact_attempt(federation: FederationJourney) -> None:
    job = federation.submit("terminate")
    federation.promote()
    federation.sync()
    federation.run_peer()
    federation.sync()
    current = federation.parent.task(job[0]).summary.current_attempt
    assert current is not None

    receipt = federation.parent.terminate_attempt(current, idempotency_key="federated-terminate")
    federation.sync()

    assert receipt.state is ActionState.SUCCEEDED
    assert receipt.result_code is ActionResult.SATISFIED
    parent = federation.parent.task(job[0])
    peer = federation.peer.task(job[0])
    assert [(item.identity, item.state, item.backend_id) for item in parent.attempts] == [
        (item.identity, item.state, item.backend_id) for item in peer.attempts
    ]
    assert parent.summary.state == job_pb2.TASK_STATE_KILLED


def test_federated_action_idempotency_is_scoped_per_resource_owner(federation: FederationJourney) -> None:
    alice = federation.submit("same-key", user="alice", preemption_retries=1)
    bob = federation.submit("same-key", user="bob", preemption_retries=1)
    federation.promote()
    federation.sync()
    federation.run_peer()
    federation.sync()

    receipts = []
    for job in (alice, bob):
        task = federation.parent.task(job[0]).summary
        assert task.current_attempt is not None
        receipts.append(
            federation.parent.retry_task(
                task.identity,
                expected_attempt_uid=task.current_attempt.attempt_uid,
                idempotency_key="retry-same-key",
            )
        )

    assert receipts[0].action_id != receipts[1].action_id
    assert all(receipt.state is ActionState.SUCCEEDED for receipt in receipts)


def test_federated_exec_rejects_attempt_replaced_on_peer_before_runtime_call(federation: FederationJourney) -> None:
    job = federation.submit("exec-race", preemption_retries=1)
    federation.promote()
    federation.sync()
    federation.run_peer()
    federation.sync()
    stale = federation.parent.task(job[0]).summary.current_attempt
    assert stale is not None
    peer_task = federation.peer.task(job[0]).summary
    federation.peer.retry_task(
        peer_task.identity,
        expected_attempt_uid=stale.attempt_uid,
        idempotency_key="peer-replaces-attempt",
    )
    federation.run_peer()

    with pytest.raises(ConnectError) as exc_info:
        federation.parent.controller.controller.exec_attempt(
            stale,
            ("echo", "stale"),
            Duration.from_seconds(1),
        )
    assert exc_info.value.code == Code.FAILED_PRECONDITION


def test_execution_peer_submits_a_child_and_syncs_the_subtree_to_authority_once(federation: FederationJourney) -> None:
    root = federation.submit("tree")
    federation.promote()
    federation.sync()
    child = federation.submit_child_on_peer(root, "child", tasks=2)
    federation.sync()

    assert [task.identity.key.resource_id for task in federation.peer_tasks(root)] == [root[0].wire_id]
    assert federation.parent_job(child).summary.identity == federation.peer_job(child).summary.identity
    parent_tasks = federation.parent_tasks(child)
    peer_tasks = federation.peer_tasks(child)
    assert [task.identity for task in parent_tasks] == [task.identity for task in peer_tasks]
    assert {task.identity.key.resource_id for task in parent_tasks} == {
        child[0].wire_id,
        child[1].wire_id,
    }


def test_authority_routes_child_cancel_to_execution_peer_without_killing_root(federation: FederationJourney) -> None:
    root = federation.submit("tree-cancel")
    federation.promote()
    federation.sync()
    child = federation.submit_child_on_peer(root, "child")
    federation.run_peer()
    federation.sync()
    child_identity = federation.parent_job(child).summary.identity

    receipt = federation.parent.cancel_job(child_identity, idempotency_key="cancel-child")
    federation.sync()

    assert receipt.state is ActionState.SUCCEEDED
    assert federation.peer_job(child).summary.state == job_pb2.JOB_STATE_KILLED
    assert federation.parent_job(child).summary.state == job_pb2.JOB_STATE_KILLED
    assert federation.parent_job(root).summary.state == job_pb2.JOB_STATE_RUNNING


def test_federated_endpoint_keeps_authority_and_execution_coordinates_distinct(
    federation: FederationJourney,
) -> None:
    job = federation.submit("endpoint")
    federation.promote()
    federation.sync()
    federation.run_peer()
    federation.peer.register_endpoint(job[0], "/serve/endpoint", "10.0.0.7:8000", endpoint_id="endpoint-1")

    (received,) = federation.peer.controller.controller.list_endpoints(EndpointQuery()).items
    assert received.key.cluster_id == PARENT_CLUSTER_ID
    assert received.task is not None and received.task.cluster_id == PARENT_CLUSTER_ID
    assert received.execution_cluster_id == PEER_ID

    federation.sync()

    parent_page = federation.parent.controller.controller.list_endpoints(EndpointQuery())
    (mirrored,) = parent_page.items
    assert mirrored.key.cluster_id == PARENT_CLUSTER_ID
    assert mirrored.task is not None and mirrored.task.cluster_id == PARENT_CLUSTER_ID
    assert mirrored.execution_cluster_id == PEER_ID

    federation.set_peer_reachable(False)
    federation.sync()

    cached_page = federation.parent.controller.controller.list_endpoints(EndpointQuery())
    assert cached_page.items == (mirrored,)
    peer_source = next(status for status in cached_page.source_statuses if status.source_id == f"federation:{PEER_ID}")
    assert peer_source.state is SourceState.UNAVAILABLE
    assert peer_source.freshness is Freshness.STALE
