# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The federation manager: peer registry, capability heartbeat, and ListPeers.

The controller composes one manager. It owns the peer registry, runs the
background capability heartbeat, builds the ``ListPeers`` view, and holds the
submit-time :class:`~iris.cluster.federation.router.PeerRouter`. With no peers
configured it is inert: the heartbeat thread is never started and every view is
empty, so a single-cluster deployment is unchanged.
"""

import logging
import threading
from collections.abc import Sequence

from rigging.timing import Duration

from iris.cluster.constraints import Constraint
from iris.cluster.federation.peer import FederationPeer
from iris.cluster.federation.router import PeerRouter, SubmitRouting
from iris.managed_thread import ManagedThread, ThreadContainer
from iris.rpc import controller_pb2

logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_INTERVAL = Duration.from_seconds(30)
_HEARTBEAT_JOIN_TIMEOUT = Duration.from_seconds(5.0)


class FederationManager:
    """Owns the federation peer registry and its background capability heartbeat."""

    def __init__(
        self,
        peers: Sequence[FederationPeer],
        *,
        threads: ThreadContainer,
        heartbeat_interval: Duration = DEFAULT_HEARTBEAT_INTERVAL,
    ):
        self._peers = {peer.peer_id: peer for peer in peers}
        self._threads = threads
        self._heartbeat_interval = heartbeat_interval
        self._router = PeerRouter(peers)
        self._heartbeat_thread: ManagedThread | None = None

    def start(self) -> None:
        """Start the capability heartbeat. A no-op when no peers are configured."""
        if not self._peers:
            return
        self._heartbeat_thread = self._threads.spawn(self._run_heartbeat_loop, name="federation-heartbeat")

    def stop(self) -> None:
        """Stop the heartbeat and release peer connections. Idempotent."""
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.stop()
            self._heartbeat_thread.join(timeout=_HEARTBEAT_JOIN_TIMEOUT)
            self._heartbeat_thread = None
        for peer in self._peers.values():
            peer.close()

    def route_submit(self, constraints: Sequence[Constraint], user: str) -> SubmitRouting:
        """Route a submission to local execution or a peer."""
        return self._router.decide(constraints, user)

    def peer_summaries(self) -> list[controller_pb2.Controller.PeerSummary]:
        """A ``PeerSummary`` for every configured peer, ordered by peer id."""
        return [self._build_summary(peer) for _, peer in sorted(self._peers.items())]

    def _build_summary(self, peer: FederationPeer) -> controller_pb2.Controller.PeerSummary:
        heartbeat = peer.heartbeat()
        return controller_pb2.Controller.PeerSummary(
            peer_id=peer.peer_id,
            controller_address=peer.controller_address,
            dashboard_url=peer.dashboard_url,
            reachable=heartbeat.reachable,
            last_sync_ms=heartbeat.last_contact_ms,
            backends=heartbeat.backends,
        )

    def _run_heartbeat_loop(self, stop_event: threading.Event) -> None:
        interval = self._heartbeat_interval.to_seconds()
        while not stop_event.is_set():
            for peer in self._peers.values():
                peer.probe()
            stop_event.wait(timeout=interval)
