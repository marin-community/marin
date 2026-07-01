# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Federation: peer config, capability heartbeat, and the dark submit router.

Covers the observable-but-not-targetable slice: peers parse and validate, a
controller derives the capability markers it advertises, the heartbeat learns a
peer's live capabilities, ListPeers surfaces them, and the submit router routes
every job local even with a reachable peer configured.
"""

import pydantic
import pytest
from iris.cluster.config import PeerConfig, config_to_dict, parse_config
from iris.cluster.federation.capabilities import cluster_capability_markers
from iris.cluster.federation.manager import FederationManager
from iris.cluster.federation.peer import FederationPeer, build_peers
from iris.cluster.federation.router import PeerRouter
from iris.managed_thread import get_thread_container, thread_container_scope
from rigging.timing import Duration, ExponentialBackoff


def _config(**extra) -> dict:
    return {"name": "parent", "platform": {"local": {}}, **extra}


# ---------------------------------------------------------------------------
# peers: config parse + validation
# ---------------------------------------------------------------------------


def test_peers_config_round_trips_through_serialization():
    config = parse_config(
        _config(
            peers={
                "cw-east": {
                    "controller_address": "http://cw:10000",
                    "dashboard_url": "https://cw.dev",
                    "cluster": "cw-east",
                    "static_token": "shhh",
                }
            }
        )
    )
    reparsed = parse_config(config_to_dict(config))
    peer = reparsed.peers["cw-east"]
    assert peer.controller_address == "http://cw:10000"
    assert peer.dashboard_url == "https://cw.dev"
    assert peer.cluster == "cw-east"
    assert peer.static_token == "shhh"


def test_no_peers_configured_is_valid_and_empty():
    assert parse_config(_config()).peers == {}


def test_peers_config_rejects_empty_controller_address():
    with pytest.raises(ValueError, match="controller_address is required"):
        parse_config(_config(peers={"cw": {"controller_address": ""}}))


def test_peers_config_rejects_unknown_field():
    # Capabilities are advertised live, never declared in config; a stray field is
    # a typo we reject rather than silently ignore (extra="forbid").
    with pytest.raises(pydantic.ValidationError):
        parse_config(_config(peers={"cw": {"controller_address": "http://cw", "capabilities": ["H100"]}}))


def test_peers_config_rejects_static_token_without_cluster():
    with pytest.raises(ValueError, match="static_token requires cluster"):
        parse_config(_config(peers={"cw": {"controller_address": "http://cw", "static_token": "shhh"}}))


# ---------------------------------------------------------------------------
# capability markers (the serving side of the heartbeat)
# ---------------------------------------------------------------------------


class _StubBackend:
    def __init__(self, attributes: dict[str, set[str]]):
        self._attributes = attributes

    def advertised_attributes(self) -> dict[str, set[str]]:
        return self._attributes


def test_capability_markers_cover_every_advertised_device():
    backends = {
        "tpu": _StubBackend({"device-type": {"tpu"}, "device-variant": {"v5e-4", "v6e-8"}}),
        "gpu": _StubBackend({"device-type": {"gpu"}, "gpu-variant": {"H100"}}),
    }
    assert cluster_capability_markers(backends) == [
        "available:H100",
        "available:gpu",
        "available:tpu",
        "available:v5e-4",
        "available:v6e-8",
    ]


def test_capability_markers_default_to_cpu_for_attribute_less_backend():
    assert cluster_capability_markers({"default": _StubBackend({})}) == ["available:cpu"]


def test_capability_markers_ignore_placement_attributes():
    # region/zone/preemptible describe placement, not a schedulable device.
    backend = _StubBackend({"region": {"us-east"}, "zone": {"us-east-1a"}, "preemptible": {"true"}})
    assert cluster_capability_markers({"b": backend}) == ["available:cpu"]


# ---------------------------------------------------------------------------
# peer heartbeat + ListPeers view (the parent side)
# ---------------------------------------------------------------------------


class _StubConnection:
    """A peer connection whose capability probe returns a canned answer."""

    def __init__(self, capabilities: tuple[str, ...], *, fail: bool = False):
        self.capabilities = capabilities
        self.fail = fail
        self.probe_count = 0
        self.shutdown_count = 0

    def get_cluster_capabilities(self) -> list[str]:
        self.probe_count += 1
        if self.fail:
            raise ConnectionError("peer unreachable")
        return list(self.capabilities)

    def shutdown(self) -> None:
        self.shutdown_count += 1


def _peer(peer_id: str, connection: _StubConnection, *, dashboard_url: str = "https://cw.dev") -> FederationPeer:
    return FederationPeer(
        peer_id,
        PeerConfig(controller_address="http://cw:10000", dashboard_url=dashboard_url),
        connection,
    )


def test_peer_probe_populates_capabilities_and_reachability():
    peer = _peer("cw", _StubConnection(("available:tpu", "available:v5e-4")))
    peer.probe()
    heartbeat = peer.heartbeat()
    assert heartbeat.reachable is True
    assert heartbeat.capabilities == ("available:tpu", "available:v5e-4")
    assert heartbeat.last_contact_ms > 0


def test_peer_probe_failure_marks_unreachable_and_keeps_last_capabilities():
    connection = _StubConnection(("available:tpu",))
    peer = _peer("cw", connection)
    peer.probe()  # first probe succeeds
    connection.fail = True
    peer.probe()  # second probe fails
    heartbeat = peer.heartbeat()
    assert heartbeat.reachable is False
    assert heartbeat.capabilities == ("available:tpu",)  # last-known markers retained


def test_list_peers_view_surfaces_heartbeat_capabilities():
    peer = _peer("cw-east", _StubConnection(("available:tpu", "available:v5e-4")))
    manager = FederationManager([peer], threads=get_thread_container())
    peer.probe()
    (summary,) = manager.peer_summaries()
    assert summary.peer_id == "cw-east"
    assert summary.controller_address == "http://cw:10000"
    assert summary.dashboard_url == "https://cw.dev"
    assert summary.reachable is True
    assert list(summary.advertised_capabilities) == ["available:tpu", "available:v5e-4"]
    assert summary.last_sync_ms > 0


def test_heartbeat_loop_refreshes_capabilities_and_stop_releases_connections():
    connection = _StubConnection(("available:cpu",))
    peer = _peer("local", connection)
    with thread_container_scope() as threads:
        manager = FederationManager([peer], threads=threads, heartbeat_interval=Duration.from_seconds(0.02))
        manager.start()
        try:
            reached = ExponentialBackoff(initial=0.01, maximum=0.1).wait_until(
                lambda: peer.heartbeat().reachable, timeout=Duration.from_seconds(3.0)
            )
            assert reached
            assert list(manager.peer_summaries()[0].advertised_capabilities) == ["available:cpu"]
        finally:
            manager.stop()
    assert connection.shutdown_count == 1


def test_manager_without_peers_is_inert():
    with thread_container_scope() as threads:
        manager = FederationManager([], threads=threads)
        manager.start()  # nothing to probe; no heartbeat thread
        assert manager.peer_summaries() == []
        assert manager.route_submit([], "alice").is_local is True
        manager.stop()  # idempotent no-op


def test_build_peers_orders_by_id_and_uses_injected_factory():
    created: list[str] = []

    def fake_connect(config: PeerConfig) -> _StubConnection:
        created.append(config.controller_address)
        return _StubConnection(("available:cpu",))

    peers = build_peers(
        {
            "b": PeerConfig(controller_address="http://b"),
            "a": PeerConfig(controller_address="http://a"),
        },
        connect=fake_connect,
    )
    assert [peer.peer_id for peer in peers] == ["a", "b"]
    assert created == ["http://a", "http://b"]


# ---------------------------------------------------------------------------
# router stays dark
# ---------------------------------------------------------------------------


def test_router_selects_local_even_with_a_reachable_peer():
    connection = _StubConnection(("available:tpu",))
    peer = _peer("cw", connection)
    peer.probe()
    decision = PeerRouter([peer]).decide([], "alice")
    assert decision.is_local is True
    assert decision.peer_id == ""
