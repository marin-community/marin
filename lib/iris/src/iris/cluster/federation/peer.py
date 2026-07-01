# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One connection per federation peer, plus its capability-heartbeat state.

``RemoteClusterClient`` already encapsulates "one connection to one controller";
:class:`FederationPeer` holds one per peer (keyed by peer id) and caches the
capabilities that peer last advertised. The connection is authenticated with the
credentials this controller presents to the peer, resolved from the peer's
cluster manifest via the shared ``credentials_for`` path — no second credential
system.
"""

import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from connectrpc.errors import ConnectError
from rigging.cluster_manifest import load_manifest
from rigging.credentials import ClientCredentials, credentials_for
from rigging.timing import Timestamp

from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.config import PeerConfig

logger = logging.getLogger(__name__)


class PeerConnection(Protocol):
    """The narrow peer-controller surface the federation heartbeat drives."""

    def get_cluster_capabilities(self) -> list[str]: ...

    def shutdown(self) -> None: ...


PeerConnectFactory = Callable[[PeerConfig], PeerConnection]


@dataclass(frozen=True)
class PeerHeartbeat:
    """The latest capability-heartbeat observation for one peer."""

    reachable: bool = False
    capabilities: tuple[str, ...] = ()
    last_contact_ms: int = 0


def _peer_credentials(peer: PeerConfig) -> ClientCredentials:
    """The client credentials this controller presents to ``peer``.

    Resolved from the peer's cluster manifest via the shared ``credentials_for``
    path. An empty ``cluster`` yields no credentials — loopback/no-auth, for a
    local or same-VPC peer that trusts the connection.
    """
    if not peer.cluster:
        return ClientCredentials()
    manifest = load_manifest(peer.cluster)
    return credentials_for(peer.cluster, manifest.auth, static_token=peer.static_token or None)


def connect_to_peer(peer: PeerConfig) -> PeerConnection:
    """Open one authenticated connection to a peer controller."""
    return RemoteClusterClient(
        controller_address=peer.controller_address,
        interceptors=_peer_credentials(peer).interceptors(),
    )


class FederationPeer:
    """One federation peer: a connection plus its latest heartbeat state.

    Thread-safe: the heartbeat loop writes via :meth:`probe`; RPC handlers read
    via :meth:`heartbeat`.
    """

    def __init__(self, peer_id: str, config: PeerConfig, connection: PeerConnection):
        self.peer_id = peer_id
        self.controller_address = config.controller_address
        self.dashboard_url = config.dashboard_url
        self._connection = connection
        self._lock = threading.Lock()
        self._heartbeat = PeerHeartbeat()

    def probe(self) -> None:
        """Refresh the peer's advertised capabilities via one heartbeat RPC.

        On success, records the peer's markers, marks it reachable, and stamps the
        contact time. On failure, marks it unreachable and keeps the last-known
        markers — staleness is signalled by ``reachable``.
        """
        try:
            capabilities = self._connection.get_cluster_capabilities()
        except (ConnectError, ConnectionError, OSError) as exc:
            logger.warning("Federation heartbeat to peer %s failed: %s", self.peer_id, exc)
            with self._lock:
                self._heartbeat = replace(self._heartbeat, reachable=False)
            return
        with self._lock:
            self._heartbeat = PeerHeartbeat(
                reachable=True,
                capabilities=tuple(capabilities),
                last_contact_ms=Timestamp.now().epoch_ms(),
            )

    def heartbeat(self) -> PeerHeartbeat:
        """The peer's latest heartbeat observation."""
        with self._lock:
            return self._heartbeat

    def close(self) -> None:
        """Release the peer connection."""
        self._connection.shutdown()


def build_peers(
    peers: Mapping[str, PeerConfig],
    *,
    connect: PeerConnectFactory = connect_to_peer,
) -> list[FederationPeer]:
    """Build one :class:`FederationPeer` per configured peer, ordered by peer id.

    ``connect`` builds each peer connection; the default opens a real
    authenticated ``RemoteClusterClient``.
    """
    return [FederationPeer(peer_id, config, connect(config)) for peer_id, config in sorted(peers.items())]
