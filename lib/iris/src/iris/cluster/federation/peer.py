# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticated peer operations and last-observed federation availability."""

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace

from rigging.timing import Timestamp

from iris.cluster.config import PeerConfig
from iris.cluster.federation.protocol import (
    FederationBackendObservation,
    FederationSyncBatch,
    HandoffDelivery,
    PeerCallError,
    PeerConnectFactory,
    PeerConnection,
)
from iris.resources.action import ActionReceipt
from iris.resources.endpoint import ExecRequest, ExecResult, ProfileRequest, ProfileResult
from iris.resources.identity import AttemptIdentity, JobIdentity, TaskIdentity
from iris.resources.names import JobName
from iris.resources.system import ProcessInfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PeerHeartbeat:
    """The latest capability-heartbeat observation for one peer."""

    reachable: bool = False
    backends: tuple[FederationBackendObservation, ...] = ()
    last_contact_ms: int = 0


class FederationPeer:
    """Coordinate remote operations and thread-safe capability observations for one peer."""

    def __init__(self, peer_id: str, config: PeerConfig, connection: PeerConnection):
        self.peer_id = peer_id
        self.controller_address = config.controller_address
        self._connection = connection
        self._lock = threading.Lock()
        self._heartbeat = PeerHeartbeat()

    def probe(self) -> None:
        """Refresh the peer's advertised backends via one heartbeat RPC.

        On success, records the peer's backends, marks it reachable, and stamps the
        contact time. On failure, marks it unreachable and keeps the last-known
        backends — staleness is signalled by ``reachable``.
        """
        try:
            backends = self._connection.list_backends()
        except PeerCallError as exc:
            logger.warning("Federation heartbeat to peer %s failed: %s", self.peer_id, exc)
            with self._lock:
                self._heartbeat = replace(self._heartbeat, reachable=False)
            return
        with self._lock:
            self._heartbeat = PeerHeartbeat(
                reachable=True,
                backends=tuple(backends),
                last_contact_ms=Timestamp.now().epoch_ms(),
            )

    def heartbeat(self) -> PeerHeartbeat:
        """The peer's latest heartbeat observation."""
        with self._lock:
            return self._heartbeat

    def launch_job(self, delivery: HandoffDelivery) -> None:
        """Submit a handed-off Job to its remote execution authority."""
        self._connection.launch_job(delivery)

    def terminate_job(self, job_id: JobName) -> None:
        """Cancel a handed-off Job on its execution peer.

        The ``job_id`` is the cluster-invariant local id: the peer runs and
        reports the same id the parent submitted, so there is nothing to rebase.
        """
        self._connection.terminate_job(job_id)

    def federation_sync(self, requester_id: str, cursor: str) -> FederationSyncBatch:
        """Run one delta-sync round against the peer."""
        return self._connection.federation_sync(requester_id, cursor)

    def cancel_job(self, identity: JobIdentity, *, idempotency_key: str, reason: str) -> ActionReceipt:
        """Cancel an exact Job incarnation on the execution peer."""
        return self._connection.cancel_job(identity, idempotency_key=idempotency_key, reason=reason)

    def retry_task(
        self,
        identity: TaskIdentity,
        *,
        expected_attempt_uid: str,
        idempotency_key: str,
        reason: str,
    ) -> ActionReceipt:
        """Retry an exact Task incarnation on the execution peer."""
        return self._connection.retry_task(
            identity,
            expected_attempt_uid=expected_attempt_uid,
            idempotency_key=idempotency_key,
            reason=reason,
        )

    def terminate_attempt(
        self,
        identity: AttemptIdentity,
        *,
        idempotency_key: str,
        reason: str,
    ) -> ActionReceipt:
        """Terminate an exact Attempt incarnation on the execution peer."""
        return self._connection.terminate_attempt(identity, idempotency_key=idempotency_key, reason=reason)

    def fail_attempt(
        self,
        identity: AttemptIdentity,
        *,
        idempotency_key: str,
        reason: str,
    ) -> ActionReceipt:
        """Fail an exact Attempt incarnation on the execution peer."""
        return self._connection.fail_attempt(identity, idempotency_key=idempotency_key, reason=reason)

    def profile_task(self, request: ProfileRequest) -> ProfileResult:
        """Profile an Attempt running on the peer."""
        return self._connection.profile_task(request)

    def exec_in_container(self, request: ExecRequest) -> ExecResult:
        """Execute a command in an Attempt running on the peer."""
        return self._connection.exec_in_container(request)

    def get_process_status(self, target: str) -> ProcessInfo:
        """Read process status for an Attempt running on the peer."""
        return self._connection.get_process_status(target)

    def close(self) -> None:
        """Release the peer connection."""
        self._connection.shutdown()


def build_peers(
    peers: Mapping[str, PeerConfig],
    *,
    connect: PeerConnectFactory,
) -> list[FederationPeer]:
    """Build one :class:`FederationPeer` per configured peer, ordered by peer id.

    ``connect`` is the composition root's transport factory. Tests inject an
    in-memory connection; production injects the Connect adapter from ``iris.rpc``.
    """
    return [FederationPeer(peer_id, config, connect(config)) for peer_id, config in sorted(peers.items())]
