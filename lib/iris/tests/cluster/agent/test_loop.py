# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of AgentLoop: applying the root's desired set and reporting observations."""

import pytest
from iris.cluster.agent.cache import AgentCache
from iris.cluster.agent.loop import AgentLoop
from iris.cluster.controller.backend import BackendCapability
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


def test_rejects_non_cluster_view_backend():
    # The agent drives its backend purely from the root's desired attempts, which
    # a worker-daemon backend ignores (it reconciles from worker snapshots). Such
    # a backend would poll happily while never dispatching, so reject it up front.
    worker_daemon = FakeClusterBackend(capabilities=frozenset({BackendCapability.WORKER_DAEMON}))
    with pytest.raises(ValueError, match="CLUSTER_VIEW"):
        AgentLoop(backend_id="b1", local_backend=worker_daemon, transport=FakeTransport([poll_response([])]))


def test_coscheduling_flag_reaches_local_backend():
    # A vanished Kueue gang pod is classified WORKER_FAILED (preemption budget)
    # only when the running entry is marked coscheduled, so the agent must carry
    # the desired request's group_by through to the local reconcile snapshot.
    req = make_req("/job/gang/0", attempt_id=0, attempt_uid="uid-gang")
    req.coscheduling.group_by = "gang"
    backend = FakeClusterBackend()
    loop = AgentLoop(backend_id="b1", local_backend=backend, transport=FakeTransport([poll_response([req])]))

    loop.tick()  # installs the desired set
    loop.tick()  # drives the backend, which records the running snapshot

    [entry] = backend.last_running_tasks
    assert entry.coscheduled is True
