# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of AgentLoop: applying the root's desired set and reporting observations."""

from iris.cluster.agent.cache import AgentCache
from iris.cluster.agent.loop import AgentLoop
from iris.rpc import job_pb2

from tests.cluster.agent.fakes import FakeClusterBackend, FakeTransport, make_req, poll_response


def test_tick_applies_root_upsert_to_local_backend():
    req_a = make_req("/job/a/0", attempt_id=0, attempt_uid="uid-a")
    transport = FakeTransport([poll_response([req_a])])
    backend = FakeClusterBackend()
    cache = AgentCache()
    loop = AgentLoop(backend_id="b1", local_backend=backend, transport=transport, cache=cache)

    # First tick: the cache starts empty so the backend sees nothing; the response
    # then installs the upsert into the desired set.
    loop.tick()
    assert set(cache.desired) == {"uid-a"}
    assert backend.last_seen_task_ids == set()

    # Second tick: the backend is now driven to converge the upserted attempt.
    loop.tick()
    assert "/job/a/0" in backend.last_seen_task_ids
    assert transport.requests[-1].backend_id == "b1"


def test_backend_observations_reach_the_root():
    req_a = make_req("/job/a/0", attempt_id=0, attempt_uid="uid-a")
    # The transport keeps re-upserting A, so the agent keeps converging it.
    transport = FakeTransport([poll_response([req_a])])
    backend = FakeClusterBackend()
    loop = AgentLoop(backend_id="b1", local_backend=backend, transport=transport, cache=AgentCache())

    # tick1: install desired; tick2: backend observes A; tick3: backend reports
    # RUNNING; tick4: backend reports SUCCEEDED.
    for _ in range(4):
        loop.tick()

    reported = [(obs.attempt_uid, obs.state) for req in transport.requests for obs in req.observations]
    assert ("uid-a", job_pb2.TASK_STATE_RUNNING) in reported
    assert ("uid-a", job_pb2.TASK_STATE_SUCCEEDED) in reported
