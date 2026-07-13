# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Federated availability: peer free-capacity snapshots, a generation-keyed
reservation ledger, and the pure queued-assignment pass the control tick runs.

A federation parent holds jobs it cannot place locally in a queue until a peer
reports enough free capacity to host one, then hands it off. This module is the
decision logic for that queue, kept pure (no DB, no proto, no I/O) so the control
tick can call it over a snapshot and so it is unit-testable in isolation:

* :class:`QueuedCandidate` — one queued job, with its shape and its
  ``ge(available:<token>, amount)`` availability gate (built by
  ``constraints.peer_availability_gate``).
* :class:`PeerAvailability` / :class:`BackendAvailability` — a peer's per-backend
  advertised free capacity + shape, as of its last capability heartbeat.
* :class:`ReservationLedger` — capacity the parent has already promoted against a
  peer backend since its last heartbeat, so successive ticks between heartbeats do
  not each re-spend the same advertised number.
* :func:`assign_queued` — the pure pass: choose ``(job, peer)`` promotions.

Why a ledger and not a per-tick decrement: the control tick runs on submit wakes,
far more often than the 30 s heartbeat, so a decrement that evaporates at
end-of-tick would let every tick re-read the same advertised number and promote
the whole queue against one stale observation. The ledger keys reservations on the
heartbeat's ``observation_epoch_ms`` (its *generation*) and holds them until a
strictly newer generation arrives — whose fresh number already reflects the
delivered jobs — so effective availability decreases monotonically between
heartbeats. Over-assignment is bounded to a peer's advertised free capacity per
observation. That residual staleness is acceptable by design — placement need not
be exact; the peer's own scheduler rejects (and the parent requeues) anything that
does not fit, which is the backstop.
"""

import logging
from dataclasses import dataclass, field

from iris.cluster.constraints import AVAILABLE_PREFIX, AttributeValue, Constraint, available_key, evaluate_constraint
from iris.cluster.federation.router import backend_satisfies
from iris.cluster.types import JobName

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendAvailability:
    """One peer backend's advertised shape + free capacity, from a heartbeat.

    ``supplies_metric`` is False for a legacy peer backend that never set the
    availability wrapper (proto3 cannot distinguish an unset map from an empty
    one, so the wrapper's presence is the signal); such a backend is matched on
    shape alone, preserving today's behavior during a rolling upgrade.
    """

    backend_id: str
    supplies_metric: bool
    generation: int  # observation_epoch_ms; 0 when the metric is not supplied
    amounts: dict[str, int]  # free amount per resource token ("h100" -> 8)
    advertised_shape: dict[str, list[str]]  # advertised_attributes, for shape match


@dataclass(frozen=True)
class PeerAvailability:
    """A federation peer's reachability and per-backend availability."""

    peer_id: str
    reachable: bool
    backends: list[BackendAvailability]


@dataclass(frozen=True)
class QueuedCandidate:
    """A queued federated job the tick may place onto a peer.

    ``availability_gate`` is the list of ``ge(available:<token>, amount)``
    constraints the chosen peer backend must satisfy (empty for a job with no
    gated resource, e.g. plain CPU — such a job matches any shape-compatible peer).
    """

    job_id: JobName
    pinned_peer_id: str  # "" when unpinned
    priority_band: int
    submitted_at_ms: int
    shape_constraints: list[Constraint]
    availability_gate: list[Constraint]


@dataclass(frozen=True)
class Promotion:
    """A decision to hand ``job_id`` to ``peer_id``'s ``backend_id``.

    ``reserved`` is the capacity to charge the ledger once the promotion's CAS is
    confirmed at commit; empty for a legacy (shape-only) backend whose capacity the
    parent does not track. ``generation`` ties the reservation to the heartbeat it
    was decided against.
    """

    job_id: JobName
    peer_id: str
    backend_id: str
    generation: int
    reserved: dict[str, int] = field(default_factory=dict)


class ReservationLedger:
    """Capacity promoted against each peer backend since its last heartbeat.

    Keyed ``(peer_id, backend_id) -> (generation, {token: reserved})``. Reset for a
    backend when a strictly newer generation arrives. In-memory only: a controller
    restart forgets it, at worst a burst of re-assignment bounded by the next
    heartbeat — an acceptable one-off, since placement need not be exact and the
    peer's scheduler is the backstop.
    """

    def __init__(self) -> None:
        self._reserved: dict[tuple[str, str], tuple[int, dict[str, int]]] = {}

    def reserved_for(self, peer_id: str, backend_id: str, generation: int) -> dict[str, int]:
        """Reservations still in force for ``(peer, backend)`` at ``generation``.

        A stored reservation from an older generation is stale — the newer heartbeat
        already reflects those handoffs — so it is treated as empty.
        """
        stored_generation, amounts = self._reserved.get((peer_id, backend_id), (0, {}))
        return dict(amounts) if stored_generation == generation and generation != 0 else {}

    def commit(self, promotion: Promotion) -> None:
        """Charge a confirmed promotion's reservation, resetting a stale generation."""
        if not promotion.reserved or promotion.generation == 0:
            return
        key = (promotion.peer_id, promotion.backend_id)
        stored_generation, amounts = self._reserved.get(key, (0, {}))
        merged = dict(amounts) if stored_generation == promotion.generation else {}
        for token, amount in promotion.reserved.items():
            merged[token] = merged.get(token, 0) + amount
        self._reserved[key] = (promotion.generation, merged)

    def drop_peers(self, keep_peer_ids: set[str]) -> None:
        """Forget reservations for peers no longer present (e.g. removed from config)."""
        for key in [k for k in self._reserved if k[0] not in keep_peer_ids]:
            del self._reserved[key]


def _shape_ok(backend: BackendAvailability, constraints: list[Constraint]) -> bool:
    """Whether a peer backend's advertised attributes satisfy every shape constraint."""
    return all(backend_satisfies(backend.advertised_shape, c) for c in constraints)


def _availability_ok(effective: dict[str, int], gate: list[Constraint]) -> bool:
    """Whether ``effective`` free amounts satisfy every ``ge(available:<token>, n)`` gate."""
    attrs = {key: AttributeValue(amount) for key, amount in effective.items()}
    return all(evaluate_constraint(attrs.get(c.key), c) for c in gate)


def _effective(backend: BackendAvailability, reserved: dict[str, int]) -> dict[str, int]:
    """Advertised amounts minus this-generation reservations, keyed by ``available:<token>``.

    Keyed by the ``available:<token>`` constraint key (not the bare token) so a gate
    constraint evaluates directly against it.
    """
    return {available_key(token): max(0, amount - reserved.get(token, 0)) for token, amount in backend.amounts.items()}


def assign_queued(
    candidates: list[QueuedCandidate],
    peers: list[PeerAvailability],
    ledger: ReservationLedger,
    *,
    max_per_peer_per_cycle: int,
) -> list[Promotion]:
    """Choose ``(job, peer, backend)`` promotions for queued federated jobs (pure).

    For each candidate, in the order given (the caller sorts by priority then age),
    find a reachable peer backend that satisfies the job's shape, honors its pin,
    and whose *effective* availability (advertised minus reservations already made
    this generation, minus what earlier candidates in this pass took) meets the
    ``ge`` gate. A legacy backend that supplies no metric is matched on shape alone.
    Tie-break by best fit (least remaining capacity for the gated token after
    placement), then peer id, then backend id, so load spreads and large free blocks
    are preserved. Fit-aware: a candidate that fits nowhere is skipped, not
    head-of-line-blocking the queue.

    Returns the promotions; the caller applies each as a conditional CAS and charges
    the ledger only for confirmed ones. Does not mutate ``ledger``.
    """
    # Per-(peer, backend) working capacity for this pass: advertised - reserved@gen.
    # Only metric-supplying backends appear here — membership marks "has a capacity
    # signal", so a placement onto a key in ``working`` reserves and one onto a legacy
    # or force-routed target does not.
    working: dict[tuple[str, str], dict[str, int]] = {}
    reachable_peers = [peer for peer in peers if peer.reachable]
    generation_of: dict[tuple[str, str], int] = {}
    for peer in reachable_peers:
        for backend in peer.backends:
            key = (peer.peer_id, backend.backend_id)
            generation_of[key] = backend.generation
            if backend.supplies_metric:
                working[key] = _effective(
                    backend, ledger.reserved_for(peer.peer_id, backend.backend_id, backend.generation)
                )

    promoted_per_peer: dict[str, int] = {}
    promotions: list[Promotion] = []

    for candidate in candidates:
        best: tuple[float, str, str] | None = None  # (fit, peer_id, backend_id)
        for peer in reachable_peers:
            if candidate.pinned_peer_id and candidate.pinned_peer_id != peer.peer_id:
                continue
            if promoted_per_peer.get(peer.peer_id, 0) >= max_per_peer_per_cycle:
                continue
            placement = _place_on_peer(candidate, peer, working)
            if placement is None:
                continue
            fit, backend_id = placement
            if best is None or (fit, peer.peer_id, backend_id) < best:
                best = (fit, peer.peer_id, backend_id)

        if best is None:
            continue

        _, peer_id, backend_id = best
        reserved: dict[str, int] = {}
        key = (peer_id, backend_id)
        if key in working:  # a metric backend was chosen: charge and decrement its capacity
            effective = working[key]
            for constraint in candidate.availability_gate:
                token = constraint.key.removeprefix(AVAILABLE_PREFIX)
                need = int(constraint.values[0].value)
                effective[constraint.key] = effective.get(constraint.key, 0) - need
                reserved[token] = reserved.get(token, 0) + need
        promoted_per_peer[peer_id] = promoted_per_peer.get(peer_id, 0) + 1
        promotions.append(
            Promotion(
                job_id=candidate.job_id,
                peer_id=peer_id,
                backend_id=backend_id,
                generation=generation_of.get(key, 0),
                reserved=reserved,
            )
        )

    return promotions


def _place_on_peer(
    candidate: QueuedCandidate, peer: PeerAvailability, working: dict[tuple[str, str], dict[str, int]]
) -> tuple[float, str] | None:
    """Best placement of ``candidate`` on ``peer``: ``(fit, backend_id)`` or ``None``.

    Prefers a shape-matching metric backend with enough effective capacity (best fit),
    then a shape-matching legacy backend (shape only). If no backend matches the shape
    at all, a shapeless candidate (no routing constraints) or a candidate pinned to
    this peer force-routes with ``backend_id=""`` and no reservation: a job with no
    routing constraints can run on any reachable peer, and a pin selects a peer
    regardless of what it advertises. A candidate is NOT force-routed past a
    shape-matching metric backend that is merely full — that waits for the next
    heartbeat.
    """
    shape_matching = [b for b in peer.backends if _shape_ok(b, candidate.shape_constraints)]
    best: tuple[float, str] | None = None
    for backend in shape_matching:
        key = (peer.peer_id, backend.backend_id)
        if key in working:  # metric backend: gate on effective capacity
            if _availability_ok(working[key], candidate.availability_gate):
                candidate_fit = (_remaining_after(working[key], candidate.availability_gate), backend.backend_id)
                if best is None or candidate_fit < best:
                    best = candidate_fit
        elif best is None or (float("inf"), backend.backend_id) < best:  # legacy: shape-only
            best = (float("inf"), backend.backend_id)
    if best is not None:
        return best
    if not shape_matching and (not candidate.shape_constraints or candidate.pinned_peer_id == peer.peer_id):
        return (float("inf"), "")
    return None


def _remaining_after(effective: dict[str, int], gate: list[Constraint]) -> float:
    """Total free capacity left across the gated tokens after placing the job.

    The best-fit key: smaller means a tighter fit, so we prefer the backend that
    ends up most fully packed for the resource the job wants (spreading pressure off
    emptier peers and preserving their large free blocks).
    """
    total = 0
    for constraint in gate:
        need = int(constraint.values[0].value)
        total += max(0, effective.get(constraint.key, 0) - need)
    return float(total)
