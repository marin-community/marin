# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AgentLoop: drives a local CLUSTER_VIEW backend and reconciles it with the root.

The agent is the client in the agent<->root reconcile: each tick it applies the
cached desired attempt set to its local :class:`TaskBackend`, collects the
backend's observations, and polls the root — reporting observations up and
receiving the next desired set down. :class:`PollTransport` is the network seam,
so the loop runs against an in-process root in tests.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Protocol

from rigging.auth import BearerTokenInjector, StaticTokenProvider

from iris.cluster.agent.cache import AgentCache
from iris.cluster.controller.backend import BackendCapability, TaskBackend
from iris.cluster.controller.reads import ControlSnapshot
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.types import JobName
from iris.rpc import remote_agent_pb2, worker_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.remote_agent_connect import RemoteAgentServiceClient

logger = logging.getLogger(__name__)

AGENT_VERSION = "iris-remote-agent/1"

_DEFAULT_POLL_TIMEOUT_MS = 30_000


class PollTransport(Protocol):
    """The agent's one call home: report observations up, receive desired state down."""

    def poll(self, request: remote_agent_pb2.PollRequest) -> remote_agent_pb2.PollResponse: ...


class ConnectPollTransport(PollTransport):
    """A :class:`PollTransport` backed by the generated Connect client.

    The generated ``RemoteAgentServiceClient.poll`` is async while the agent loop
    is synchronous, so each call is driven to completion on one persistent event
    loop owned by this transport.
    """

    def __init__(self, root_address: str, token: str, *, timeout_ms: int = _DEFAULT_POLL_TIMEOUT_MS) -> None:
        interceptors: tuple[BearerTokenInjector, ...] = ()
        if token:
            interceptors = (BearerTokenInjector(StaticTokenProvider(token), "authorization"),)
        self._client = RemoteAgentServiceClient(
            address=root_address,
            timeout_ms=timeout_ms,
            interceptors=interceptors,
            accept_compression=IRIS_RPC_COMPRESSIONS,
            send_compression=None,
        )
        self._timeout_ms = timeout_ms
        self._loop = asyncio.new_event_loop()

    def poll(self, request: remote_agent_pb2.PollRequest) -> remote_agent_pb2.PollResponse:
        return self._loop.run_until_complete(self._client.poll(request, timeout_ms=self._timeout_ms))

    def close(self) -> None:
        self._loop.close()


class AgentLoop:
    """Reconciles a local CLUSTER_VIEW backend with the root over a poll transport."""

    def __init__(
        self,
        *,
        backend_id: str,
        local_backend: TaskBackend,
        transport: PollTransport,
        cache: AgentCache | None = None,
    ) -> None:
        if BackendCapability.CLUSTER_VIEW not in local_backend.capabilities:
            raise ValueError(
                f"AgentLoop requires a CLUSTER_VIEW backend; {local_backend.name!r} has "
                f"{sorted(c.value for c in local_backend.capabilities)}. The agent drives a backend "
                "purely from the root's desired attempts, which a worker-daemon backend ignores "
                "(it reconciles from worker snapshots, not tasks_to_run)."
            )
        self._backend_id = backend_id
        self._local_backend = local_backend
        self._transport = transport
        self._cache = cache if cache is not None else AgentCache()

    @property
    def cache(self) -> AgentCache:
        """The agent's view of the desired set and pending observations."""
        return self._cache

    def tick(self) -> None:
        """One reconcile round: apply the desired set locally, then poll the root."""
        desired = self._cache.desired_requests()
        snapshot = ControlSnapshot(
            worker_addresses={},
            reconcile_rows=[],
            timeout_rows=[],
            tasks_to_run=desired,
            running_tasks=[
                RunningTaskEntry(
                    JobName.from_wire(req.task_id),
                    req.attempt_id,
                    coscheduled=bool(req.coscheduling.group_by),
                )
                for req in desired
            ],
        )
        result = self._local_backend.reconcile(snapshot)
        for update in result.updates:
            self._cache.record_update(update)

        request = remote_agent_pb2.PollRequest(
            backend_id=self._backend_id,
            root_epoch_seen=self._cache.root_epoch,
            last_sync_id=self._cache.last_sync_id,
            caps=remote_agent_pb2.AgentCapabilities(agent_version=AGENT_VERSION),
            observations=self._cache.pending_observations(),
            rolled_up_health=worker_pb2.Worker.WorkerHealth(),
            capacity=remote_agent_pb2.CapacitySummary(),
        )
        response = self._transport.poll(request)
        self._cache.apply_response(response)

    def run(
        self,
        *,
        poll_interval_seconds: float,
        stop: threading.Event | Callable[[], bool] | None = None,
    ) -> None:
        """Tick until ``stop`` fires, sleeping ``poll_interval_seconds`` between rounds."""
        should_stop = _stop_predicate(stop)
        sleep = stop.wait if isinstance(stop, threading.Event) else None
        while not should_stop():
            self.tick()
            if sleep is not None:
                if sleep(poll_interval_seconds):
                    break
            else:
                time.sleep(poll_interval_seconds)


def _stop_predicate(stop: threading.Event | Callable[[], bool] | None) -> Callable[[], bool]:
    if stop is None:
        return lambda: False
    if isinstance(stop, threading.Event):
        return stop.is_set
    return stop
