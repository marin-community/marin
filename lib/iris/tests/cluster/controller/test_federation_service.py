# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Federation is observable through the RPC service, but not targetable.

``ListPeers`` returns the configured peers with their live heartbeat
capabilities, ``GetClusterCapabilities`` reports this controller's own markers,
and — the load-bearing invariant — a job submitted while a reachable peer is
configured still materializes local tasks. Nothing is handed off.
"""

from iris.cluster.config import PeerConfig
from iris.cluster.controller.schema import tasks_table
from iris.cluster.federation.manager import FederationManager
from iris.cluster.federation.peer import FederationPeer
from iris.cluster.types import JobName
from iris.managed_thread import get_thread_container
from iris.rpc import controller_pb2
from rigging.server_auth import VerifiedIdentity, identity_scope
from sqlalchemy import select

from .conftest import make_job_request

_IDENTITY = VerifiedIdentity(user_id="alice", role="user")


class _StubPeerConnection:
    def __init__(self, capabilities: tuple[str, ...]):
        self._capabilities = capabilities

    def get_cluster_capabilities(self) -> list[str]:
        return list(self._capabilities)

    def shutdown(self) -> None:
        pass


def _attach_peer(mock_controller, capabilities: tuple[str, ...] = ("available:tpu",)) -> FederationPeer:
    """Give the controller one reachable peer (already heartbeated once)."""
    peer = FederationPeer(
        "cw-east",
        PeerConfig(controller_address="http://cw:10000", dashboard_url="https://cw.dev"),
        _StubPeerConnection(capabilities),
    )
    peer.probe()
    mock_controller.federation = FederationManager([peer], threads=get_thread_container())
    return peer


def test_list_peers_returns_configured_peer_with_heartbeat_capabilities(controller_service, mock_controller):
    _attach_peer(mock_controller, capabilities=("available:tpu", "available:v5e-4"))
    with identity_scope(_IDENTITY):
        response = controller_service.list_peers(controller_pb2.Controller.ListPeersRequest(), None)
    (peer,) = response.peers
    assert peer.peer_id == "cw-east"
    assert peer.reachable is True
    assert list(peer.advertised_capabilities) == ["available:tpu", "available:v5e-4"]
    assert peer.dashboard_url == "https://cw.dev"


def test_get_cluster_capabilities_reports_backend_devices(controller_service, mock_controller):
    mock_controller.provider.advertised_attributes.return_value = {"device-type": {"tpu"}, "device-variant": {"v5e-4"}}
    with identity_scope(_IDENTITY):
        response = controller_service.get_cluster_capabilities(
            controller_pb2.Controller.GetClusterCapabilitiesRequest(), None
        )
    assert list(response.capabilities) == ["available:tpu", "available:v5e-4"]


def test_launch_job_stays_local_when_a_peer_is_configured(controller_service, mock_controller, state):
    _attach_peer(mock_controller)
    response = controller_service.launch_job(make_job_request("federation-dark", replicas=2), None)
    job_id = JobName.from_wire(response.job_id)

    with state._db.read_snapshot() as tx:
        child_clusters = tx.execute(select(tasks_table.c.child_cluster).where(tasks_table.c.job_id == job_id)).all()
    # The job materialized its tasks locally (child_cluster empty); the reachable
    # peer attracted nothing.
    assert len(child_clusters) == 2
    assert all((row.child_cluster or "") == "" for row in child_clusters)
