# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AgentCache: the remote agent's recoverable view between polls.

The agent owns a real in-cluster :class:`TaskBackend` and dials home to the root.
This cache holds what the agent learned on the last poll: the desired attempt set
it must converge to, the observations it still owes the root, and the root epoch
and sync id it last saw. It is single-threaded state owned by the agent loop, not
shared across threads, so it carries no lock.

Wire keys: the desired set and the observation inbox are both keyed by
``attempt_uid`` (``RunTaskRequest.attempt_uid``). :meth:`uid_for` resolves a local
backend's ``(task_id, attempt_id)`` update back to its uid through the cached
desired specs.
"""

import logging

from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.rpc import job_pb2, remote_agent_pb2

logger = logging.getLogger(__name__)


class AgentCache:
    """The desired attempt set and observation inbox the agent carries between polls."""

    def __init__(self) -> None:
        # attempt_uid -> the RunTaskRequest the agent must converge to.
        self.desired: dict[str, job_pb2.RunTaskRequest] = {}
        # attempt_uid -> the latest observation the root has not yet acked.
        self.observations: dict[str, remote_agent_pb2.AgentObservation] = {}
        self.last_sync_id: int = 0
        self.root_epoch: int = 0

    def key_to_uid(self) -> dict[tuple[str, int], str]:
        """Build ``{(task_id_wire, attempt_id): attempt_uid}`` from the cached desired specs."""
        return {(req.task_id, req.attempt_id): uid for uid, req in self.desired.items()}

    def uid_for(self, task_id_wire: str, attempt_id: int) -> str | None:
        """Resolve a local ``(task_id, attempt_id)`` to its cached attempt uid, or None."""
        return self.key_to_uid().get((task_id_wire, attempt_id))

    def record_update(self, update: TaskUpdate) -> None:
        """Translate a local-backend ``TaskUpdate`` into a pending observation.

        The update's ``(task_id, attempt_id)`` resolves to its attempt uid through
        the cached desired set. An update for an attempt the root already retired
        (no cached uid) is dropped — the root no longer wants it reported.
        """
        uid = self.uid_for(update.task_id.to_wire(), update.attempt_id)
        if uid is None:
            logger.debug(
                "agent cache: update for %s attempt %d has no desired uid; dropping",
                update.task_id.to_wire(),
                update.attempt_id,
            )
            return
        self.observations[uid] = remote_agent_pb2.AgentObservation(
            attempt_uid=uid,
            acted_root_epoch=self.root_epoch,
            desired_generation=update.attempt_id,
            state=update.new_state,
            observed_worker="",
            exit_code=update.exit_code or 0,
            message=update.error or "",
        )

    def desired_requests(self) -> list[job_pb2.RunTaskRequest]:
        """The cached desired specs the local backend must converge to."""
        return list(self.desired.values())

    def pending_observations(self) -> list[remote_agent_pb2.AgentObservation]:
        """The observations owed to the root, awaiting an APPLIED ack."""
        return list(self.observations.values())

    def apply_response(self, response: remote_agent_pb2.PollResponse) -> None:
        """Fold a poll response into the cache.

        Adopts the response's ``root_epoch`` and ``new_sync_id``, replaces the
        desired set with the response's upserts (full-snapshot semantics: an
        attempt absent from the upserts is dropped from desired), and clears every
        observation the root acked as APPLIED.
        """
        self.root_epoch = response.root_epoch
        self.last_sync_id = response.new_sync_id
        self.desired = {upsert.attempt_uid: upsert.spec.request for upsert in response.upserts}
        for ack in response.acks:
            # PR4: handle ACK_DISPOSITION_RETRY_LATER (retain and re-report) and
            # ACK_DISPOSITION_STALE_DISCARDED (drop the superseded attempt).
            if ack.disposition == remote_agent_pb2.ACK_DISPOSITION_APPLIED:
                self.observations.pop(ack.attempt_uid, None)
