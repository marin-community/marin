# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Federation: peer config, capability heartbeat, and the submit router.

Covers peer config parse/validation, the capability heartbeat forwarding a peer's
live backends, the ListPeers view, and the submit router's decision matrix
(prefer-local, hand off when locally infeasible, explicit ``cluster`` pin).
"""

from dataclasses import replace
from typing import cast

import pydantic
import pytest
from iris.cluster.config import PeerConfig, config_to_dict, parse_config, user_admitted
from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute
from iris.cluster.federation.availability import AVAILABILITY_METRIC_VERSION
from iris.cluster.federation.manager import FederationManager
from iris.cluster.federation.peer import FederationPeer, build_peers
from iris.cluster.federation.protocol import (
    FederationBackendObservation,
    FederationResourceAvailability,
    FederationStore,
    FederationSyncBatch,
    PeerCallError,
    PeerErrorCode,
)
from iris.cluster.federation.router import PeerRouter, RoutingRequest, SubmitDisposition
from iris.managed_thread import get_thread_container, thread_container_scope
from iris.resources.state import PriorityBand
from rigging.timing import Duration, ExponentialBackoff


def _device_backend(backend_id: str, device_type: str) -> FederationBackendObservation:
    return _backend(
        backend_id,
        advertised_attributes={WellKnownAttribute.DEVICE_TYPE: (device_type,)},
    )


def _device_constraint(device_type: str) -> Constraint:
    return Constraint.create(key=WellKnownAttribute.DEVICE_TYPE, op=ConstraintOp.EQ, value=device_type)


def _gpu_backend(backend_id: str, variant: str) -> FederationBackendObservation:
    """A peer GPU backend advertising both the device type and the variant."""
    return _backend(
        backend_id,
        advertised_attributes={
            WellKnownAttribute.DEVICE_TYPE: ("gpu",),
            WellKnownAttribute.DEVICE_VARIANT: (variant,),
        },
    )


def _gpu_constraints(variant: str) -> list[Constraint]:
    """The routing constraints ``constraints_from_resources`` emits for a GPU request."""
    return [
        Constraint.create(key=WellKnownAttribute.DEVICE_TYPE, op=ConstraintOp.EQ, value="gpu"),
        Constraint.create(key=WellKnownAttribute.DEVICE_VARIANT, op=ConstraintOp.EQ, value=variant),
    ]


def _config(**extra) -> dict:
    return {"name": "parent", "platform": {"local": {}}, **extra}


def _backend(backend_id: str, **fields) -> FederationBackendObservation:
    return FederationBackendObservation(backend_id=backend_id, **fields)


# ---------------------------------------------------------------------------
# peers: config parse + validation
# ---------------------------------------------------------------------------


def test_peers_config_round_trips_through_serialization():
    config = parse_config(
        _config(
            peers={
                "cw-east": {
                    "controller_address": "http://cw:10000",
                    "cluster": "cw-east",
                }
            }
        )
    )
    reparsed = parse_config(config_to_dict(config))
    peer = reparsed.peers["cw-east"]
    assert peer.controller_address == "http://cw:10000"
    assert peer.cluster == "cw-east"


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


# ---------------------------------------------------------------------------
# peer heartbeat + ListPeers view (the parent side)
# ---------------------------------------------------------------------------


class _StubConnection:
    """A peer connection whose ListBackends probe returns a canned answer."""

    def __init__(self, backends: tuple[FederationBackendObservation, ...], *, fail: bool = False):
        self.backends = backends
        self.fail = fail
        self.probe_count = 0
        self.shutdown_count = 0

    def list_backends(self) -> tuple[FederationBackendObservation, ...]:
        self.probe_count += 1
        if self.fail:
            raise PeerCallError(PeerErrorCode.UNAVAILABLE, "peer unreachable")
        return self.backends

    def shutdown(self) -> None:
        self.shutdown_count += 1


class _SyncConnection(_StubConnection):
    def __init__(self, peer_id: str):
        super().__init__(())
        self.peer_id = peer_id
        self.cursors: list[str] = []

    def federation_sync(self, requester_id: str, cursor: str) -> FederationSyncBatch:
        self.cursors.append(cursor)
        return FederationSyncBatch(
            deltas=(),
            next_cursor=f"{self.peer_id}-{len(self.cursors)}",
            cursor_stale=False,
            endpoints=(),
        )


class _TransientUndecodableSyncConnection(_SyncConnection):
    """A peer adapter that cannot decode its first sync response."""

    def federation_sync(self, requester_id: str, cursor: str) -> FederationSyncBatch:
        self.cursors.append(cursor)
        if len(self.cursors) == 1:
            raise ValueError("device has no selected kind")
        if len(self.cursors) > 2:
            raise PeerCallError(PeerErrorCode.UNAVAILABLE, "peer unavailable after recovery")
        return FederationSyncBatch(
            deltas=(),
            next_cursor=f"{self.peer_id}-{len(self.cursors)}",
            cursor_stale=False,
            endpoints=(),
        )


class _RejectingSyncStore:
    def __init__(self) -> None:
        self.cursors: dict[str, str] = {}
        self.reject_peer = "bad"

    def pending_handoffs(self):
        return []

    def pending_cancels(self):
        return []

    def read_cursor(self, peer_id: str) -> str:
        return self.cursors.get(peer_id, "")

    def apply_sync_batch(self, peer_id: str, deltas, *, next_cursor: str, cursor_stale: bool, endpoints=()) -> None:
        if peer_id == self.reject_peer:
            raise ValueError("conflicting retained mirror")
        self.cursors[peer_id] = next_cursor


def _peer(peer_id: str, connection: _StubConnection) -> FederationPeer:
    return FederationPeer(peer_id, PeerConfig(controller_address="http://cw:10000"), connection)


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


def test_peer_observation_surfaces_current_reachability():
    backend = _backend("tpu-fleet", kind="worker-daemon", worker_count=3)
    peer = _peer("cw-east", _StubConnection((backend,)))
    manager = FederationManager([peer], threads=get_thread_container())
    peer.probe()
    (summary,) = manager.peer_observations()
    assert summary.peer_id == "cw-east"
    assert summary.controller_address == "http://cw:10000"
    assert summary.reachable is True
    assert summary.last_contact_ms > 0
    (observed_backend,) = summary.backends
    assert (observed_backend.backend_id, observed_backend.kind, observed_backend.worker_count) == (
        "tpu-fleet",
        "worker-daemon",
        3,
    )


def _availability_backend(backend_id: str, version: int) -> FederationBackendObservation:
    """A GPU backend reporting the capacity metric at ``version``."""
    return replace(
        _gpu_backend(backend_id, "h100"),
        availability=FederationResourceAvailability(
            version=version,
            observation_epoch_ms=1000,
            amounts={"h100": 8},
            total_amounts={"h100": 24},
            held_by_band={PriorityBand.BATCH: {"h100": 16}},
        ),
    )


def test_availability_at_a_known_version_is_read_in_full():
    peer = _peer("cw", _StubConnection((_availability_backend("gpu", AVAILABILITY_METRIC_VERSION),)))
    manager = FederationManager([peer], threads=get_thread_container())
    peer.probe()
    ((backend,),) = [p.backends for p in manager.peer_availability()]
    assert backend.supplies_metric is True
    assert backend.amounts == {"h100": 8}
    assert backend.held_by_band == {PriorityBand.BATCH: {"h100": 16}}


def test_availability_from_an_older_metric_version_is_still_read():
    # v1 reports free amounts and no band split. Its numbers are conservative under v2
    # semantics, so a rolling upgrade keeps gating on them rather than dropping the gate.
    backend = _availability_backend("gpu", 1)
    assert backend.availability is not None
    backend = replace(backend, availability=replace(backend.availability, held_by_band={}))
    peer = _peer("cw", _StubConnection((backend,)))
    manager = FederationManager([peer], threads=get_thread_container())
    peer.probe()
    ((projected,),) = [p.backends for p in manager.peer_availability()]
    assert projected.supplies_metric is True
    assert projected.amounts == {"h100": 8}
    assert projected.held_by_band == {}


def test_availability_from_a_newer_metric_version_is_treated_as_unsupplied():
    # The parent cannot know what a future version's amounts mean, so it falls back to
    # shape-only matching for that backend instead of acting on numbers it misreads.
    peer = _peer("cw", _StubConnection((_availability_backend("gpu", AVAILABILITY_METRIC_VERSION + 1),)))
    manager = FederationManager([peer], threads=get_thread_container())
    peer.probe()
    ((backend,),) = [p.backends for p in manager.peer_availability()]
    assert backend.supplies_metric is False
    assert (backend.amounts, backend.held_by_band, backend.generation) == ({}, {}, 0)


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


def test_rejected_peer_batch_does_not_stop_other_peers_or_future_sync(caplog):
    bad_connection = _SyncConnection("bad")
    good_connection = _SyncConnection("good")
    store = _RejectingSyncStore()
    manager = FederationManager(
        [_peer("bad", bad_connection), _peer("good", good_connection)],
        threads=get_thread_container(),
        store=cast(FederationStore, store),
        cluster_id="parent",
    )

    manager.sync_once()

    assert store.cursors == {"good": "good-1"}
    assert "peer bad at cursor '' was rejected" in caplog.text

    store.reject_peer = ""
    manager.sync_once()

    assert store.cursors == {"bad": "bad-2", "good": "good-2"}
    assert bad_connection.cursors == ["", ""]
    assert good_connection.cursors == ["", "good-1"]


def test_sync_loop_with_undecodable_peer_response_retries_without_losing_cursor(caplog):
    connection = _TransientUndecodableSyncConnection("cw")
    store = _RejectingSyncStore()
    store.reject_peer = ""
    with thread_container_scope() as threads:
        manager = FederationManager(
            [_peer("cw", connection)],
            threads=threads,
            store=cast(FederationStore, store),
            cluster_id="parent",
            heartbeat_interval=Duration.from_seconds(60),
            sync_interval=Duration.from_seconds(0.02),
        )
        manager.start()
        try:
            reached = ExponentialBackoff(initial=0.01, maximum=0.1).wait_until(
                lambda: store.cursors.get("cw") == "cw-2",
                timeout=Duration.from_seconds(3),
            )
            assert reached
        finally:
            manager.stop()

    assert connection.cursors[:2] == ["", ""]
    assert any(record.exc_info is not None and "peer cw" in record.getMessage() for record in caplog.records)


def test_manager_without_peers_is_inert():
    with thread_container_scope() as threads:
        manager = FederationManager([], threads=threads)
        manager.start()  # nothing to probe; no heartbeat thread
        assert manager.peer_summaries() == []
        request = RoutingRequest(constraints=[], local_feasible=True)
        assert manager.classify_submit(request).disposition == SubmitDisposition.LOCAL
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
    request = RoutingRequest(constraints=[_device_constraint("tpu")], local_feasible=True)
    plan = PeerRouter([peer]).classify(request)
    assert plan.disposition == SubmitDisposition.LOCAL
    assert plan.pinned_peer_id == ""


def test_router_queues_when_local_infeasible_and_a_peer_can_host():
    peer = _peer("cw", _StubConnection((_device_backend("tpu-fleet", "tpu"),)))
    peer.probe()
    request = RoutingRequest(constraints=[_device_constraint("tpu")], local_feasible=False)
    plan = PeerRouter([peer]).classify(request)
    # Submit does not pick a peer — it queues; the tick assigns by availability.
    assert plan.disposition == SubmitDisposition.QUEUE
    assert plan.pinned_peer_id == ""


def test_router_rejects_when_no_peer_can_host_the_shape():
    # The peer only advertises CPU; a TPU job it cannot host is rejected now (no queue
    # could help) rather than wedged on an incapable peer.
    peer = _peer("cw", _StubConnection((_device_backend("cpu-fleet", "cpu"),)))
    peer.probe()
    request = RoutingRequest(constraints=[_device_constraint("tpu")], local_feasible=False)
    assert PeerRouter([peer]).classify(request).disposition == SubmitDisposition.REJECT


def test_router_rejects_when_the_only_shape_matching_peer_is_unreachable():
    connection = _StubConnection((_device_backend("tpu-fleet", "tpu"),))
    peer = _peer("cw", connection)
    peer.probe()
    connection.fail = True
    peer.probe()  # now unreachable; its last-known backends are stale
    request = RoutingRequest(constraints=[_device_constraint("tpu")], local_feasible=False)
    assert PeerRouter([peer]).classify(request).disposition == SubmitDisposition.REJECT


def test_router_cluster_pin_queues_to_the_peer_even_when_locally_feasible():
    peer = _peer("cw", _StubConnection((_device_backend("tpu-fleet", "tpu"),)))
    peer.probe()
    request = RoutingRequest(constraints=[], local_feasible=True, cluster_pin="cw")
    plan = PeerRouter([peer]).classify(request)
    assert plan.disposition == SubmitDisposition.QUEUE
    assert plan.pinned_peer_id == "cw"


def test_router_queues_a_gpu_job_to_a_peer_advertising_the_matching_variant():
    # The matching mechanism: a peer whose backend advertises device-type=gpu and the
    # requested device-variant can host a GPU job (both routing constraints satisfied),
    # so an unpinned job queues for peer availability instead of being rejected.
    peer = _peer("cw", _StubConnection((_gpu_backend("h100-fleet", "h100"),)))
    peer.probe()
    request = RoutingRequest(constraints=_gpu_constraints("h100"), local_feasible=False)
    plan = PeerRouter([peer]).classify(request)
    assert plan.disposition == SubmitDisposition.QUEUE
    assert plan.pinned_peer_id == ""


def test_router_rejects_a_gpu_job_when_a_peer_advertises_no_device_attributes():
    # A peer backend that advertises nothing cannot satisfy device-type=gpu, so an
    # auto-match GPU job is rejected — the operational gap that keeps GPU auto-federation
    # off until the backend advertises its device attributes (a pin still bypasses this).
    peer = _peer("cw", _StubConnection((_backend("gpu-fleet"),)))
    peer.probe()
    request = RoutingRequest(constraints=_gpu_constraints("h100"), local_feasible=False)
    assert PeerRouter([peer]).classify(request).disposition == SubmitDisposition.REJECT


def test_router_pin_queues_a_gpu_peer_even_without_advertised_attributes():
    # An explicit pin queues to that peer regardless of advertised capability, so a GPU
    # handoff to CoreWeave works even when auto-match would not (see the case above).
    peer = _peer("cw", _StubConnection((_backend("gpu-fleet"),)))
    peer.probe()
    request = RoutingRequest(constraints=_gpu_constraints("h100"), local_feasible=False, cluster_pin="cw")
    plan = PeerRouter([peer]).classify(request)
    assert plan.disposition == SubmitDisposition.QUEUE
    assert plan.pinned_peer_id == "cw"


# ---------------------------------------------------------------------------
# submitter allowlist (auth.allowed_submitters, enforced by the cluster the job lands on)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "allowed, user, admitted",
    [
        (["*"], "anyone@anywhere.com", True),
        (["*@openathena.ai"], "alice@openathena.ai", True),
        (["*@openathena.ai"], "alice@OPENATHENA.AI", True),  # domain match is case-insensitive
        (["*@openathena.ai"], "mallory@evil.com", False),
        (["*@openathena.ai"], "local_admin", False),  # no @ — never a domain match
        (["alice@openathena.ai"], "alice@openathena.ai", True),  # exact identity
        (["alice@openathena.ai"], "bob@openathena.ai", False),
    ],
)
def test_user_admitted_matches_wildcard_domain_and_exact(allowed, user, admitted):
    assert user_admitted(allowed, user) is admitted
