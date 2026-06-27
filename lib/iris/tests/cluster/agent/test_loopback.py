# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end loopback: a task flows root -> agent -> local backend -> root.

Wires the real root side (RemoteTaskBackend + RelayRegistry + RemoteAgentServer)
to the real agent side (AgentLoop + AgentCache) through an in-process transport
that drives the server's async ``poll`` handler — no HTTP. This proves the relay
composes the two halves: the attempt the root publishes reaches the local
backend, and the RUNNING/SUCCEEDED states the backend observes fold back into the
root's reconcile updates with the right task identity.
"""

import asyncio
from typing import cast

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from iris.cluster.agent.loop import AgentLoop
from iris.cluster.backends.remote.backend import RemoteTaskBackend
from iris.cluster.backends.remote.relay import RelayRegistry
from iris.cluster.backends.remote.server import RemoteAgentServer
from iris.cluster.controller.reads import ControlSnapshot
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.types import JobName
from iris.rpc import job_pb2, remote_agent_pb2

from tests.cluster.agent.fakes import FakeClusterBackend, make_req


class _LoopbackTransport:
    """Drives the root's async poll handler in-process (no network).

    ``RemoteAgentServer.poll`` ignores its ``RequestContext`` (auth and headers
    are interceptor-level), so a placeholder context suffices here.
    """

    def __init__(self, server: RemoteAgentServer) -> None:
        self._server = server
        self._ctx = cast(RequestContext, object())

    def poll(self, request: remote_agent_pb2.PollRequest) -> remote_agent_pb2.PollResponse:
        return asyncio.run(self._server.poll(request, self._ctx))


def _root_tick(root: RemoteTaskBackend, reqs: list[job_pb2.RunTaskRequest]) -> list[TaskUpdate]:
    """One root control tick: publish the desired set, fold drained observations."""
    snapshot = ControlSnapshot(
        worker_addresses={},
        reconcile_rows=[],
        timeout_rows=[],
        tasks_to_run=list(reqs),
        running_tasks=[RunningTaskEntry(JobName.from_wire(r.task_id), r.attempt_id) for r in reqs],
    )
    return root.reconcile(snapshot).updates


def test_task_flows_root_to_agent_and_back():
    backend_id = "remote-1"
    root = RemoteTaskBackend(name=backend_id)
    registry = RelayRegistry()
    registry.register(backend_id, root.relay)
    server = RemoteAgentServer(registry)
    agent = AgentLoop(
        backend_id=backend_id,
        local_backend=FakeClusterBackend(),
        transport=_LoopbackTransport(server),
    )

    req = make_req("/job/demo/0", attempt_id=0, attempt_uid="a" * 16)

    updates: list[TaskUpdate] = []
    for _ in range(6):
        updates.extend(_root_tick(root, [req]))
        agent.tick()

    by_state = {u.new_state: u for u in updates}
    assert job_pb2.TASK_STATE_RUNNING in by_state
    assert job_pb2.TASK_STATE_SUCCEEDED in by_state

    succeeded = by_state[job_pb2.TASK_STATE_SUCCEEDED]
    assert succeeded.task_id == JobName.from_wire("/job/demo/0")
    assert succeeded.attempt_id == 0

    states = [u.new_state for u in updates]
    assert states.index(job_pb2.TASK_STATE_SUCCEEDED) > states.index(job_pb2.TASK_STATE_RUNNING)


def test_unknown_backend_id_is_rejected():
    server = RemoteAgentServer(RelayRegistry())
    transport = _LoopbackTransport(server)
    try:
        transport.poll(remote_agent_pb2.PollRequest(backend_id="absent"))
        raise AssertionError("expected a ConnectError for an unknown backend_id")
    except ConnectError as err:
        assert err.code == Code.NOT_FOUND
