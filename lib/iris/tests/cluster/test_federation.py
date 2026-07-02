# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Federation: peer config, capability heartbeat, and the submit router.

Covers peer config parse/validation, the capability heartbeat forwarding a peer's
live backends, the ListPeers view, and the submit router's decision matrix
(prefer-local, hand off when locally infeasible, explicit ``cluster`` pin).
"""

import pydantic
import pytest
from iris.cluster.backends.rpc.backend import EXEC_IN_CONTAINER_MAX_TIMEOUT
from iris.cluster.config import PeerConfig, config_to_dict, parse_config
from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute
from iris.cluster.federation import peer as peer_module
from iris.cluster.federation.manager import FederationManager
from iris.cluster.federation.peer import FederationPeer, build_peers
from iris.cluster.federation.router import PeerRouter, RoutingRequest
from iris.managed_thread import get_thread_container, thread_container_scope
from iris.rpc import controller_pb2
from rigging.timing import Duration, ExponentialBackoff


def _device_backend(backend_id: str, device_type: str) -> controller_pb2.Controller.BackendSummary:
    return _backend(
        backend_id,
        advertised_attributes={WellKnownAttribute.DEVICE_TYPE: controller_pb2.StringList(values=[device_type])},
    )


def _device_constraint(device_type: str) -> Constraint:
    return Constraint.create(key=WellKnownAttribute.DEVICE_TYPE, op=ConstraintOp.EQ, value=device_type)


def _config(**extra) -> dict:
    return {"name": "parent", "platform": {"local": {}}, **extra}


def _backend(backend_id: str, **fields) -> controller_pb2.Controller.BackendSummary:
    return controller_pb2.Controller.BackendSummary(backend_id=backend_id, **fields)


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
# peer heartbeat + ListPeers view (the parent side)
# ---------------------------------------------------------------------------


class _StubConnection:
    """A peer connection whose ListBackends probe returns a canned answer."""

    def __init__(self, backends: tuple[controller_pb2.Controller.BackendSummary, ...], *, fail: bool = False):
        self.backends = backends
        self.fail = fail
        self.probe_count = 0
        self.shutdown_count = 0

    def list_backends(self) -> list[controller_pb2.Controller.BackendSummary]:
        self.probe_count += 1
        if self.fail:
            raise ConnectionError("peer unreachable")
        return list(self.backends)

    def shutdown(self) -> None:
        self.shutdown_count += 1


def _peer(peer_id: str, connection: _StubConnection, *, dashboard_url: str = "https://cw.dev") -> FederationPeer:
    return FederationPeer(
        peer_id,
        PeerConfig(controller_address="http://cw:10000", dashboard_url=dashboard_url),
        connection,
    )


def test_peer_probe_populates_backends_and_reachability():
    peer = _peer("cw", _StubConnection((_backend("tpu-fleet", kind="worker-daemon"),)))
    peer.probe()
    heartbeat = peer.heartbeat()
    assert heartbeat.reachable is True
    assert [b.backend_id for b in heartbeat.backends] == ["tpu-fleet"]
    assert heartbeat.last_contact_ms > 0


def test_peer_probe_failure_marks_unreachable_and_keeps_last_backends():
    connection = _StubConnection((_backend("tpu-fleet"),))
    peer = _peer("cw", connection)
    peer.probe()  # first probe succeeds
    connection.fail = True
    peer.probe()  # second probe fails
    heartbeat = peer.heartbeat()
    assert heartbeat.reachable is False
    assert [b.backend_id for b in heartbeat.backends] == ["tpu-fleet"]  # last-known backends retained


def test_list_peers_view_surfaces_heartbeat_backends():
    backend = _backend("tpu-fleet", kind="worker-daemon", worker_count=3)
    peer = _peer("cw-east", _StubConnection((backend,)))
    manager = FederationManager([peer], threads=get_thread_container())
    peer.probe()
    (summary,) = manager.peer_summaries()
    assert summary.peer_id == "cw-east"
    assert summary.controller_address == "http://cw:10000"
    assert summary.dashboard_url == "https://cw.dev"
    assert summary.reachable is True
    (forwarded,) = summary.backends
    assert forwarded.backend_id == "tpu-fleet"
    assert forwarded.kind == "worker-daemon"
    assert forwarded.worker_count == 3
    assert summary.last_sync_ms > 0


class _RecordingStub:
    """A controller stub that records the deadline each on-demand RPC was given."""

    def __init__(self):
        self.exec_timeout_ms = 0

    def exec_in_container(self, request, timeout_ms):
        self.exec_timeout_ms = timeout_ms
        return controller_pb2.Controller.ExecInContainerResponse()

    def close(self):
        pass


def test_exec_proxy_deadline_outlasts_the_peer(monkeypatch):
    """The parent->peer exec hop must give the peer at least its own exec budget.

    A negative ("no caller limit") timeout is capped at EXEC_IN_CONTAINER_MAX_TIMEOUT
    on the peer, so the parent's deadline must clear that cap rather than collapse to
    the proxy margin; a positive timeout carries the caller's budget plus margin.
    """
    stub = _RecordingStub()
    monkeypatch.setattr(peer_module, "ControllerServiceClientSync", lambda **kwargs: stub)
    connection = peer_module._PeerRpcConnection("http://peer:10000", [])

    connection.exec_in_container(controller_pb2.Controller.ExecInContainerRequest(task_id="/u/j/0", timeout_seconds=-1))
    assert stub.exec_timeout_ms >= EXEC_IN_CONTAINER_MAX_TIMEOUT.to_ms()

    connection.exec_in_container(controller_pb2.Controller.ExecInContainerRequest(task_id="/u/j/0", timeout_seconds=30))
    assert stub.exec_timeout_ms > 30 * 1000


def test_heartbeat_loop_refreshes_backends_and_stop_releases_connections():
    connection = _StubConnection((_backend("cpu-fleet"),))
    peer = _peer("local", connection)
    with thread_container_scope() as threads:
        manager = FederationManager([peer], threads=threads, heartbeat_interval=Duration.from_seconds(0.02))
        manager.start()
        try:
            reached = ExponentialBackoff(initial=0.01, maximum=0.1).wait_until(
                lambda: peer.heartbeat().reachable, timeout=Duration.from_seconds(3.0)
            )
            assert reached
            assert [b.backend_id for b in manager.peer_summaries()[0].backends] == ["cpu-fleet"]
        finally:
            manager.stop()
    assert connection.shutdown_count == 1


def test_manager_without_peers_is_inert():
    with thread_container_scope() as threads:
        manager = FederationManager([], threads=threads)
        manager.start()  # nothing to probe; no heartbeat thread
        assert manager.peer_summaries() == []
        request = RoutingRequest(constraints=[], user="alice", local_feasible=True)
        assert manager.route_submit(request).is_local is True
        manager.stop()  # idempotent no-op


def test_build_peers_orders_by_id_and_uses_injected_factory():
    created: list[str] = []

    def fake_connect(config: PeerConfig) -> _StubConnection:
        created.append(config.controller_address)
        return _StubConnection((_backend("cpu-fleet"),))

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
# router decision matrix
# ---------------------------------------------------------------------------


def test_router_prefers_local_when_feasible_even_with_a_reachable_peer():
    peer = _peer("cw", _StubConnection((_device_backend("tpu-fleet", "tpu"),)))
    peer.probe()
    request = RoutingRequest(constraints=[_device_constraint("tpu")], user="alice", local_feasible=True)
    decision = PeerRouter([peer]).decide(request)
    assert decision.is_local is True
    assert decision.peer_id == ""


def test_router_hands_off_when_local_infeasible_and_a_peer_can_host():
    peer = _peer("cw", _StubConnection((_device_backend("tpu-fleet", "tpu"),)))
    peer.probe()
    request = RoutingRequest(constraints=[_device_constraint("tpu")], user="alice", local_feasible=False)
    decision = PeerRouter([peer]).decide(request)
    assert decision.peer_id == "cw"


def test_router_stays_local_when_no_peer_can_host_the_shape():
    # The peer only advertises CPU; a TPU job it cannot host stays local so the
    # caller fails it unschedulable rather than wedging it on an incapable peer.
    peer = _peer("cw", _StubConnection((_device_backend("cpu-fleet", "cpu"),)))
    peer.probe()
    request = RoutingRequest(constraints=[_device_constraint("tpu")], user="alice", local_feasible=False)
    assert PeerRouter([peer]).decide(request).is_local is True


def test_router_skips_an_unreachable_peer():
    connection = _StubConnection((_device_backend("tpu-fleet", "tpu"),))
    peer = _peer("cw", connection)
    peer.probe()
    connection.fail = True
    peer.probe()  # now unreachable; its last-known backends are stale
    request = RoutingRequest(constraints=[_device_constraint("tpu")], user="alice", local_feasible=False)
    assert PeerRouter([peer]).decide(request).is_local is True


def test_router_cluster_pin_forces_the_peer_even_when_locally_feasible():
    peer = _peer("cw", _StubConnection((_device_backend("tpu-fleet", "tpu"),)))
    peer.probe()
    request = RoutingRequest(constraints=[], user="alice", local_feasible=True, cluster_pin="cw")
    assert PeerRouter([peer]).decide(request).peer_id == "cw"
