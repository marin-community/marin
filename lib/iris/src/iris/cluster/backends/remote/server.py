# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RemoteAgentServer: the root-side handler for the agent's Poll RPC.

The agent dials home and reports observed state up in the request; the root
replies with the desired attempt set down in the response. The server is a thin
adapter over the :class:`RelayRegistry`: it routes each poll to the relay for the
caller's ``backend_id``, ingests the reported observations, and returns the full
desired set (PR3 always sends a snapshot — the agent reconciles to the upserts).
"""

import logging

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from google.protobuf.duration_pb2 import Duration

from iris.cluster.backends.remote.relay import RelayRegistry
from iris.rpc import remote_agent_pb2, worker_pb2

logger = logging.getLogger(__name__)


class RemoteAgentServer:
    """Serves ``RemoteAgentService.Poll`` over the relay registry.

    Identity verification (the caller is ``system:remote-agent``) is handled by
    the server auth interceptor chain; this handler only routes by ``backend_id``.
    """

    def __init__(self, registry: RelayRegistry, *, lease_seconds: int = 30) -> None:
        self._registry = registry
        self._lease_seconds = lease_seconds

    async def poll(
        self,
        request: remote_agent_pb2.PollRequest,
        ctx: RequestContext,
    ) -> remote_agent_pb2.PollResponse:
        relay = self._registry.get(request.backend_id)
        if relay is None:
            raise ConnectError(Code.NOT_FOUND, f"unknown backend_id {request.backend_id!r}")
        # PR4: pin agent identity to backend_id

        ingested = list(request.observations)
        relay.ingest_observations(ingested)
        sync_id, desired = relay.snapshot_desired()

        return remote_agent_pb2.PollResponse(
            root_epoch=1,
            new_sync_id=sync_id,
            snapshot=True,
            lease_duration=Duration(seconds=self._lease_seconds),
            upserts=[
                remote_agent_pb2.DesiredAttempt(
                    attempt_uid=req.attempt_uid,
                    desired_generation=req.attempt_id,
                    spec=worker_pb2.Worker.AttemptSpec(request=req),
                    constraints=list(req.constraints),
                )
                for req in desired
            ],
            removals=[],
            autoscale=[],
            acks=[
                remote_agent_pb2.AckObservation(
                    attempt_uid=obs.attempt_uid,
                    disposition=remote_agent_pb2.ACK_DISPOSITION_APPLIED,
                )
                for obs in ingested
            ],
            pending_commands=[],
        )
