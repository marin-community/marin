# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for federated availability: the job→gate translation, the
generation-keyed reservation ledger, and the pure queued-assignment pass."""

from iris.cluster.constraints import (
    Constraint,
    ConstraintOp,
    WellKnownAttribute,
    available_key,
    peer_availability_gate,
    required_resource_amounts,
)
from iris.cluster.federation.availability import (
    BackendAvailability,
    PeerAvailability,
    Promotion,
    QueuedCandidate,
    ReservationLedger,
    assign_queued,
)
from iris.cluster.types import JobName
from iris.rpc import job_pb2


def _gpu(variant: str, count: int) -> job_pb2.DeviceConfig:
    return job_pb2.DeviceConfig(gpu=job_pb2.GpuDevice(variant=variant, count=count))


def _gpu_shape(variant: str = "h100") -> list:
    return [
        Constraint.create(key=WellKnownAttribute.DEVICE_TYPE, op=ConstraintOp.EQ, value="gpu"),
        Constraint.create(key=WellKnownAttribute.DEVICE_VARIANT, op=ConstraintOp.EQ, value=variant),
    ]


def _backend(
    backend_id: str,
    free: int,
    variant: str = "h100",
    *,
    supplies: bool = True,
    generation: int = 1000,
    held: dict[int, int] | None = None,
):
    return BackendAvailability(
        backend_id=backend_id,
        supplies_metric=supplies,
        generation=generation if supplies else 0,
        amounts={variant: free} if supplies else {},
        advertised_shape={"device-type": ["gpu"], "device-variant": [variant]},
        held_by_band={band: {variant: amount} for band, amount in (held or {}).items()},
    )


def _peer(peer_id: str, backends: list, *, reachable: bool = True) -> PeerAvailability:
    return PeerAvailability(peer_id=peer_id, reachable=reachable, backends=backends)


def _candidate(name: str, variant: str = "h100", count: int = 8, *, pin: str = "", band: int = 2, ts: int = 0):
    return QueuedCandidate(
        job_id=JobName.from_string(f"/u/{name}"),
        pinned_peer_id=pin,
        priority_band=band,
        submitted_at_ms=ts,
        shape_constraints=_gpu_shape(variant),
        availability_gate=peer_availability_gate(_gpu(variant, count), replicas=1),
    )


# --- translation -----------------------------------------------------------


def test_gpu_request_translates_to_a_ge_available_gate():
    gate = peer_availability_gate(_gpu("H100", 8), replicas=3)
    assert len(gate) == 1
    (constraint,) = gate
    assert constraint.key == available_key("h100")  # lowercased
    assert constraint.op == ConstraintOp.GE
    assert int(constraint.values[0].value) == 24  # 3 replicas * 8 chips


def test_cpu_and_auto_and_tpu_carry_no_gate():
    assert required_resource_amounts(job_pb2.DeviceConfig(cpu=job_pb2.CpuDevice()), 4) == {}
    assert required_resource_amounts(_gpu("auto", 1), 1) == {}  # no concrete token to match
    assert required_resource_amounts(job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5p-8", count=8)), 2) == {}


# --- reservation ledger ----------------------------------------------------


def test_ledger_holds_reservations_within_a_generation_and_resets_on_a_newer_one():
    ledger = ReservationLedger()
    ledger.commit(Promotion(JobName.from_string("/u/a"), "cw", "b", generation=1000, reserved={"h100": 8}))
    assert ledger.reserved_for("cw", "b", 1000) == {"h100": 8}
    # A stale (older or different) generation sees no reservation.
    assert ledger.reserved_for("cw", "b", 2000) == {}
    # Committing at a newer generation resets, not accumulates on top of the old one.
    ledger.commit(Promotion(JobName.from_string("/u/c"), "cw", "b", generation=2000, reserved={"h100": 8}))
    assert ledger.reserved_for("cw", "b", 2000) == {"h100": 8}


# --- assignment pass -------------------------------------------------------


def test_places_a_job_on_a_peer_with_enough_capacity():
    peers = [_peer("cw", [_backend("b", free=8)])]
    [promotion] = assign_queued([_candidate("j", count=8)], peers, ReservationLedger(), max_per_peer_per_cycle=8)
    assert promotion.peer_id == "cw"
    assert promotion.reserved == {"h100": 8}


def test_leaves_a_job_queued_when_no_peer_has_room():
    peers = [_peer("cw", [_backend("b", free=4)])]
    assert assign_queued([_candidate("j", count=8)], peers, ReservationLedger(), max_per_peer_per_cycle=8) == []


def test_never_sums_capacity_across_backends():
    # 6 free on backend a + 4 free on backend b = 10 chips, but a job needs 8 on ONE
    # backend (a job pins to one backend), so neither hosts it: it stays queued.
    peers = [_peer("cw", [_backend("a", free=6), _backend("b", free=4)])]
    assert assign_queued([_candidate("j", count=8)], peers, ReservationLedger(), max_per_peer_per_cycle=8) == []


def test_in_tick_decrement_prevents_double_booking_one_peers_capacity():
    # One peer with room for exactly one 8-GPU job; two queued jobs -> only one promoted.
    peers = [_peer("cw", [_backend("b", free=8)])]
    promotions = assign_queued(
        [_candidate("j1", count=8, ts=1), _candidate("j2", count=8, ts=2)],
        peers,
        ReservationLedger(),
        max_per_peer_per_cycle=8,
    )
    assert [p.job_id.to_wire() for p in promotions] == ["/u/j1"]


def test_spreads_two_jobs_across_two_same_shape_peers():
    peers = [_peer("cw-a", [_backend("b", free=8)]), _peer("cw-b", [_backend("b", free=8)])]
    promotions = assign_queued(
        [_candidate("j1", count=8, ts=1), _candidate("j2", count=8, ts=2)],
        peers,
        ReservationLedger(),
        max_per_peer_per_cycle=8,
    )
    assert {p.peer_id for p in promotions} == {"cw-a", "cw-b"}  # not both onto the first


def test_across_ticks_the_ledger_bounds_assignment_against_one_stale_observation():
    # Same peer/generation across two ticks: after promoting one 8-GPU job (and charging
    # the ledger), a second tick against the SAME heartbeat sees effective 0 and defers.
    peers = [_peer("cw", [_backend("b", free=8, generation=1000)])]
    ledger = ReservationLedger()
    [first] = assign_queued([_candidate("j1", count=8)], peers, ledger, max_per_peer_per_cycle=8)
    ledger.commit(first)
    assert assign_queued([_candidate("j2", count=8)], peers, ledger, max_per_peer_per_cycle=8) == []
    # A newer heartbeat generation resets the ledger, so a fresh 8 free is placeable again.
    fresh = [_peer("cw", [_backend("b", free=8, generation=2000)])]
    [second] = assign_queued([_candidate("j2", count=8)], fresh, ledger, max_per_peer_per_cycle=8)
    assert second.peer_id == "cw"


def test_pin_confines_a_candidate_to_its_peer():
    peers = [_peer("cw-a", [_backend("b", free=8)]), _peer("cw-b", [_backend("b", free=8)])]
    [promotion] = assign_queued(
        [_candidate("j", count=8, pin="cw-b")], peers, ReservationLedger(), max_per_peer_per_cycle=8
    )
    assert promotion.peer_id == "cw-b"


def test_legacy_backend_without_a_metric_matches_on_shape_only():
    # A peer backend that predates the availability field (supplies_metric=False) is
    # matched on shape alone and reserves nothing (its capacity is not tracked).
    peers = [_peer("cw", [_backend("b", free=0, supplies=False)])]
    [promotion] = assign_queued([_candidate("j", count=8)], peers, ReservationLedger(), max_per_peer_per_cycle=8)
    assert promotion.peer_id == "cw"
    assert promotion.reserved == {}


def test_unreachable_peer_hosts_nothing():
    peers = [_peer("cw", [_backend("b", free=8)], reachable=False)]
    assert assign_queued([_candidate("j", count=8)], peers, ReservationLedger(), max_per_peer_per_cycle=8) == []


# --- priority-aware placement ----------------------------------------------

_INTERACTIVE = job_pb2.PRIORITY_BAND_INTERACTIVE
_BATCH = job_pb2.PRIORITY_BAND_BATCH


def test_interactive_job_reaches_a_peer_saturated_by_preemptible_batch_work():
    # The peer advertises nothing free, but every chip is held by batch work the
    # candidate outranks: it is placed so the peer's own scheduler can preempt.
    peers = [_peer("cw", [_backend("b", free=0, held={_BATCH: 64})])]
    [promotion] = assign_queued(
        [_candidate("j", count=32, band=_INTERACTIVE)], peers, ReservationLedger(), max_per_peer_per_cycle=8
    )
    assert promotion.peer_id == "cw"
    assert promotion.reserved == {"h100": 32}


def test_a_job_never_reclaims_capacity_at_its_own_band_or_above():
    # Held by equal-priority (interactive) work: an interactive candidate cannot
    # preempt it, so the job stays queued.
    peers = [_peer("cw", [_backend("b", free=0, held={_INTERACTIVE: 64})])]
    assert (
        assign_queued(
            [_candidate("j", count=32, band=_INTERACTIVE)], peers, ReservationLedger(), max_per_peer_per_cycle=8
        )
        == []
    )
    # And batch work reclaims nothing: no band is below it.
    peers = [_peer("cw", [_backend("b", free=0, held={_BATCH: 64})])]
    assert (
        assign_queued([_candidate("j", count=32, band=_BATCH)], peers, ReservationLedger(), max_per_peer_per_cycle=8)
        == []
    )


def test_idle_capacity_wins_over_preempting_a_busy_peer():
    peers = [
        _peer("cw-busy", [_backend("b", free=0, held={_BATCH: 64})]),
        _peer("cw-idle", [_backend("b", free=64)]),
    ]
    [promotion] = assign_queued(
        [_candidate("j", count=8, band=_INTERACTIVE)], peers, ReservationLedger(), max_per_peer_per_cycle=8
    )
    assert promotion.peer_id == "cw-idle"


def test_a_tracked_preempting_backend_beats_a_shape_only_one():
    # One backend fits only by reclaiming batch work; the other is legacy (no metric).
    # Taking the legacy one would place the job while charging nothing, so successive
    # ticks could keep piling onto the same peer: the tracked fit wins and reserves.
    peers = [_peer("cw", [_backend("metric", free=0, held={_BATCH: 64}), _backend("legacy", free=0, supplies=False)])]
    [promotion] = assign_queued(
        [_candidate("j", count=32, band=_INTERACTIVE)], peers, ReservationLedger(), max_per_peer_per_cycle=8
    )
    assert promotion.backend_id == "metric"
    assert promotion.reserved == {"h100": 32}


def test_a_tracked_preempting_peer_beats_a_shape_only_peer():
    # Same ordering across peers, where the legacy peer also sorts first by id.
    peers = [
        _peer("cw-legacy", [_backend("b", free=0, supplies=False)]),
        _peer("cw-metric", [_backend("b", free=0, held={_BATCH: 64})]),
    ]
    [promotion] = assign_queued(
        [_candidate("j", count=32, band=_INTERACTIVE)], peers, ReservationLedger(), max_per_peer_per_cycle=8
    )
    assert promotion.peer_id == "cw-metric"
    assert promotion.reserved == {"h100": 32}


def test_reclaimed_capacity_is_decremented_within_a_tick():
    # 8 free + 8 reclaimable from batch hosts exactly two 8-GPU jobs, not three.
    peers = [_peer("cw", [_backend("b", free=8, held={_BATCH: 8})])]
    promotions = assign_queued(
        [_candidate(f"j{i}", count=8, band=_INTERACTIVE, ts=i) for i in range(3)],
        peers,
        ReservationLedger(),
        max_per_peer_per_cycle=8,
    )
    assert [p.job_id.to_wire() for p in promotions] == ["/u/j0", "/u/j1"]


def test_a_peer_reporting_no_band_split_reclaims_nothing():
    # A backend that cannot attribute held capacity to a band (worker-daemon, or one
    # that predates the field) is gated on its free amount alone, as before.
    peers = [_peer("cw", [_backend("b", free=4)])]
    assert (
        assign_queued(
            [_candidate("j", count=8, band=_INTERACTIVE)], peers, ReservationLedger(), max_per_peer_per_cycle=8
        )
        == []
    )


def test_per_peer_cap_limits_promotions_per_tick():
    peers = [_peer("cw", [_backend("b", free=100)])]
    promotions = assign_queued(
        [_candidate(f"j{i}", count=1, ts=i) for i in range(5)],
        peers,
        ReservationLedger(),
        max_per_peer_per_cycle=2,
    )
    assert len(promotions) == 2  # capped even though capacity remains
