# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of AgentCache: desired-set folding and observation lifecycle."""

from iris.cluster.agent.cache import AgentCache
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.types import JobName
from iris.rpc import job_pb2, remote_agent_pb2

from tests.cluster.agent.fakes import make_req, poll_response


def test_apply_response_replaces_desired_with_upserts():
    cache = AgentCache()
    req_a = make_req("/job/a/0", attempt_id=3, attempt_uid="uid-a")
    req_b = make_req("/job/b/0", attempt_id=1, attempt_uid="uid-b")

    cache.apply_response(poll_response([req_a, req_b], sync_id=5))
    assert set(cache.desired) == {"uid-a", "uid-b"}
    # The spec round-trips through the upsert, not just the uid.
    assert cache.desired["uid-a"].task_id == "/job/a/0"
    assert cache.desired["uid-a"].attempt_id == 3
    assert cache.last_sync_id == 5

    # Full-snapshot semantics: an attempt absent from the next response is dropped.
    cache.apply_response(poll_response([req_a], sync_id=6))
    assert set(cache.desired) == {"uid-a"}
    assert cache.last_sync_id == 6


def test_record_update_reports_observation_keyed_by_uid():
    cache = AgentCache()
    req_a = make_req("/job/a/0", attempt_id=3, attempt_uid="uid-a")
    cache.apply_response(poll_response([req_a]))

    cache.record_update(
        TaskUpdate(
            task_id=JobName.from_wire("/job/a/0"),
            attempt_id=3,
            new_state=job_pb2.TASK_STATE_FAILED,
            error="boom",
            exit_code=7,
        )
    )

    pending = cache.pending_observations()
    assert len(pending) == 1
    obs = pending[0]
    assert obs.attempt_uid == "uid-a"
    assert obs.state == job_pb2.TASK_STATE_FAILED
    assert obs.exit_code == 7
    assert obs.message == "boom"
    # The acted epoch carries the root epoch the cache last adopted (1 from the response).
    assert obs.acted_root_epoch == 1


def test_applied_ack_clears_observation():
    cache = AgentCache()
    req_a = make_req("/job/a/0", attempt_id=0, attempt_uid="uid-a")
    cache.apply_response(poll_response([req_a]))
    cache.record_update(
        TaskUpdate(task_id=JobName.from_wire("/job/a/0"), attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING)
    )
    assert [o.attempt_uid for o in cache.pending_observations()] == ["uid-a"]

    cache.apply_response(
        poll_response(
            [req_a],
            acks=[
                remote_agent_pb2.AckObservation(
                    attempt_uid="uid-a", disposition=remote_agent_pb2.ACK_DISPOSITION_APPLIED
                )
            ],
        )
    )
    assert cache.pending_observations() == []


def test_update_for_retired_attempt_is_dropped():
    cache = AgentCache()
    req_a = make_req("/job/a/0", attempt_id=0, attempt_uid="uid-a")
    cache.apply_response(poll_response([req_a]))

    # An update for a task/attempt the root never put in the desired set produces
    # no observation — the root already retired it.
    cache.record_update(
        TaskUpdate(task_id=JobName.from_wire("/job/ghost/0"), attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING)
    )
    assert cache.pending_observations() == []
