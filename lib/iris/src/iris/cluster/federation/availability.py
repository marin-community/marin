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
  advertised free capacity, per-band held capacity, and shape, as of its last
  capability heartbeat.
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

A peer reports its capacity per priority band: a free amount plus what its admitted
work holds at each band. A candidate's effective capacity on a backend is that free
amount plus everything held below its own band (a numerically higher band is lower
priority), which is what the peer's scheduler would preempt to admit the job.
Placement spends idle capacity first, reclaims from the lowest-priority band upward,
and prefers a peer that needs no preemption at all.
"""

import logging
from dataclasses import dataclass, field
from typing import NamedTuple

from iris.cluster.constraints import AVAILABLE_PREFIX, AttributeValue, Constraint, evaluate_constraint
from iris.cluster.federation.router import backend_satisfies
from iris.cluster.types import JobName

logger = logging.getLogger(__name__)

# Semantics version of the capacity metric a peer reports on
# ``BackendSummary.availability``. Bump when the meaning of the amounts (units,
# tokens, aggregation) changes; a parent reading a NEWER version than it knows treats
# the backend as supplying no metric rather than misreading it.
#
# v1: free amounts only, computed from every non-terminal pod.
# v2: free amounts count only capacity admitted work holds (queued, unadmitted work
#     no longer subtracts), plus the ``held_by_band`` split of the held remainder.
AVAILABILITY_METRIC_VERSION = 2


@dataclass(frozen=True)
class BackendAvailability:
    """One peer backend's advertised shape + free capacity, from a heartbeat.

    ``supplies_metric`` is False for a peer backend whose metric the parent cannot
    read: a legacy one that never set the availability wrapper (proto3 cannot
    distinguish an unset map from an empty one, so the wrapper's presence is the
    signal), or one reporting a version newer than ``AVAILABILITY_METRIC_VERSION``.
    Such a backend is matched on shape alone, which preserves pre-metric behavior
    during a rolling upgrade.

    ``held_by_band`` is what admitted work holds there, keyed by the priority band
    holding it; a candidate outranking a band may reclaim that band's amount by
    preemption. Empty for a peer that cannot attribute held capacity to a band (a
    worker-daemon backend, or one that predates the field): nothing is reclaimable
    and the gate reads ``amounts`` alone, as before.
    """

    backend_id: str
    supplies_metric: bool
    generation: int  # observation_epoch_ms; 0 when the metric is not supplied
    amounts: dict[str, int]  # free amount per resource token ("h100" -> 8)
    advertised_shape: dict[str, list[str]]  # advertised_attributes, for shape match
    held_by_band: dict[int, dict[str, int]] = field(default_factory=dict)  # band -> token -> held amount


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

    ``priority_band`` decides both queue order and how much of a peer's held
    capacity the job can reclaim, so the caller resolves UNSPECIFIED to its default
    band before building the candidate (``build_queued_candidates``).
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


ANY_BAND = 0  # PRIORITY_BAND_UNSPECIFIED: outranks every real band, so nothing is off-limits


@dataclass
class _WorkingCapacity:
    """One peer backend's spendable capacity for a single assignment pass.

    ``free`` is idle capacity per token; ``held`` is what admitted work holds, keyed
    by the band holding it then by token. Both are decremented as candidates take
    them, so one pass never spends the same chips twice.
    """

    free: dict[str, int]
    held: dict[int, dict[str, int]]

    def available(self, token: str, band: int) -> int:
        """Idle capacity plus everything held below ``band`` (a higher band number)."""
        return self.free.get(token, 0) + sum(
            amounts.get(token, 0) for held_band, amounts in self.held.items() if held_band > band
        )

    def would_preempt(self, token: str, amount: int) -> bool:
        """Whether taking ``amount`` of ``token`` has to reclaim held capacity."""
        return self.free.get(token, 0) < amount

    def spend(self, token: str, amount: int, band: int) -> None:
        """Take ``amount`` of ``token``: idle capacity first, then the lowest-priority band up.

        Reclaiming the lowest-priority band first leaves the peer's higher-priority
        work alone for as long as possible. ``band`` bounds what may be reclaimed;
        ``ANY_BAND`` lifts the bound (used to replay ledger reservations, which do not
        record the band that took them — an approximation the module's stated
        tolerance for inexact placement absorbs).
        """
        take = min(self.free.get(token, 0), amount)
        self.free[token] = self.free.get(token, 0) - take
        remaining = amount - take
        for held_band in sorted((b for b in self.held if b > band), reverse=True):
            if remaining <= 0:
                return
            amounts = self.held[held_band]
            taken = min(amounts.get(token, 0), remaining)
            amounts[token] = amounts.get(token, 0) - taken
            remaining -= taken


def _shape_ok(backend: BackendAvailability, constraints: list[Constraint]) -> bool:
    """Whether a peer backend's advertised attributes satisfy every shape constraint."""
    return all(backend_satisfies(backend.advertised_shape, c) for c in constraints)


def _availability_ok(working: _WorkingCapacity, gate: list[Constraint], band: int) -> bool:
    """Whether capacity reachable at ``band`` satisfies every ``ge(available:<token>, n)`` gate."""
    attrs = {c.key: AttributeValue(working.available(_token(c), band)) for c in gate}
    return all(evaluate_constraint(attrs.get(c.key), c) for c in gate)


def _token(constraint: Constraint) -> str:
    """The bare resource token an ``available:<token>`` gate constraint names."""
    return constraint.key.removeprefix(AVAILABLE_PREFIX)


def _working_capacity(backend: BackendAvailability, reserved: dict[str, int]) -> _WorkingCapacity:
    """A backend's advertised capacity with this generation's reservations already spent."""
    working = _WorkingCapacity(
        free=dict(backend.amounts),
        held={band: dict(amounts) for band, amounts in backend.held_by_band.items()},
    )
    for token, amount in reserved.items():
        working.spend(token, amount, ANY_BAND)
    return working


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
    and whose *effective* availability (advertised idle capacity plus capacity held
    below the candidate's band, minus reservations already made this generation,
    minus what earlier candidates in this pass took) meets the ``ge`` gate. A legacy
    backend that supplies no metric is matched on shape alone, and ranks behind every
    backend whose capacity the parent can see. Among the rest, prefer a placement that
    needs no preemption; tie-break by best fit (least remaining capacity for the gated
    token after placement), then peer id, then backend id, so load spreads and large
    free blocks are preserved. Fit-aware: a candidate that fits nowhere is skipped, not
    head-of-line-blocking the queue.

    Returns the promotions; the caller applies each as a conditional CAS and charges
    the ledger only for confirmed ones. Does not mutate ``ledger``.
    """
    # Per-(peer, backend) working capacity for this pass: advertised - reserved@gen.
    # Only metric-supplying backends appear here — membership marks "has a capacity
    # signal", so a placement onto a key in ``working`` reserves and one onto a legacy
    # or force-routed target does not.
    working: dict[tuple[str, str], _WorkingCapacity] = {}
    reachable_peers = [peer for peer in peers if peer.reachable]
    generation_of: dict[tuple[str, str], int] = {}
    for peer in reachable_peers:
        for backend in peer.backends:
            key = (peer.peer_id, backend.backend_id)
            generation_of[key] = backend.generation
            if backend.supplies_metric:
                working[key] = _working_capacity(
                    backend, ledger.reserved_for(peer.peer_id, backend.backend_id, backend.generation)
                )

    promoted_per_peer: dict[str, int] = {}
    promotions: list[Promotion] = []

    for candidate in candidates:
        # (shape_only, preempts, fit, peer_id, backend_id): a _Placement widened with the
        # peer id, which breaks ties between equally good backends on different peers.
        best: tuple[bool, bool, float, str, str] | None = None
        for peer in reachable_peers:
            if candidate.pinned_peer_id and candidate.pinned_peer_id != peer.peer_id:
                continue
            if promoted_per_peer.get(peer.peer_id, 0) >= max_per_peer_per_cycle:
                continue
            placement = _place_on_peer(candidate, peer, working)
            if placement is None:
                continue
            ranked = (
                placement.shape_only,
                placement.preempts,
                placement.remaining,
                peer.peer_id,
                placement.backend_id,
            )
            if best is None or ranked < best:
                best = ranked

        if best is None:
            continue

        _, _, _, peer_id, backend_id = best
        reserved: dict[str, int] = {}
        key = (peer_id, backend_id)
        if key in working:  # a metric backend was chosen: charge and decrement its capacity
            for constraint in candidate.availability_gate:
                token = _token(constraint)
                need = int(constraint.values[0].value)
                working[key].spend(token, need, candidate.priority_band)
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


class _Placement(NamedTuple):
    """One backend a candidate could go to, in best-first field order.

    A tuple so it doubles as the sort key: ``shape_only`` first, so a backend whose
    capacity the parent cannot see loses to every backend that verifiably fits, even
    one that fits only by preemption (choosing it would drop the reservation the
    tracked placement would have charged). Then ``preempts``, so an idle backend beats
    one that would have to evict work, then how much capacity is left after the job
    lands (tighter fit first), then the backend id for a stable tie-break.
    """

    shape_only: bool  # True for a backend matched on shape alone: no capacity metric
    preempts: bool  # True when the job fits only by reclaiming held work
    remaining: float  # capacity left across the gated tokens after placement
    backend_id: str  # "" for a force-routed candidate (shapeless or pinned)


def _shape_only_placement(backend_id: str) -> _Placement:
    """A shape-matched backend the parent has no capacity metric for.

    ``preempts`` and ``remaining`` are unknown and never consulted: ``shape_only``
    already ranks this behind every placement whose capacity the parent can see.
    """
    return _Placement(shape_only=True, preempts=False, remaining=0.0, backend_id=backend_id)


def _place_on_peer(
    candidate: QueuedCandidate, peer: PeerAvailability, working: dict[tuple[str, str], _WorkingCapacity]
) -> _Placement | None:
    """Best placement of ``candidate`` on ``peer``, or ``None`` if it fits nowhere there.

    Prefers a shape-matching metric backend with enough effective capacity — idle
    first, then one that only fits by reclaiming lower-priority work — then a
    shape-matching legacy backend (shape only). If no backend matches the shape at
    all, a shapeless candidate (no routing constraints) or a candidate pinned to this
    peer force-routes with ``backend_id=""`` and no reservation: a job with no routing
    constraints can run on any reachable peer, and a pin selects a peer regardless of
    what it advertises. A candidate is NOT force-routed past a shape-matching metric
    backend that is merely full — that waits for the next heartbeat.
    """
    shape_matching = [b for b in peer.backends if _shape_ok(b, candidate.shape_constraints)]
    best: _Placement | None = None
    for backend in shape_matching:
        capacity = working.get((peer.peer_id, backend.backend_id))
        if capacity is None:  # legacy backend: no metric, matched on shape alone
            option = _shape_only_placement(backend.backend_id)
        elif _availability_ok(capacity, candidate.availability_gate, candidate.priority_band):
            option = _Placement(
                shape_only=False,
                preempts=_preempts(capacity, candidate.availability_gate),
                remaining=_remaining_after(capacity, candidate.availability_gate, candidate.priority_band),
                backend_id=backend.backend_id,
            )
        else:  # metric backend that cannot fit the job even by preemption
            continue
        if best is None or option < best:
            best = option
    if best is not None:
        return best
    if not shape_matching and (not candidate.shape_constraints or candidate.pinned_peer_id == peer.peer_id):
        return _shape_only_placement("")
    return None


def _preempts(capacity: _WorkingCapacity, gate: list[Constraint]) -> bool:
    """Whether placing the job has to reclaim held capacity for some gated token.

    Ranked ahead of best fit, so an idle peer always beats one that would have to
    evict work — the parent preempts only where nothing idle fits.
    """
    return any(capacity.would_preempt(_token(c), int(c.values[0].value)) for c in gate)


def _remaining_after(capacity: _WorkingCapacity, gate: list[Constraint], band: int) -> float:
    """Total capacity left across the gated tokens after placing the job.

    The best-fit key: smaller means a tighter fit, so we prefer the backend that
    ends up most fully packed for the resource the job wants (spreading pressure off
    emptier peers and preserving their large free blocks).
    """
    total = 0
    for constraint in gate:
        need = int(constraint.values[0].value)
        total += max(0, capacity.available(_token(constraint), band) - need)
    return float(total)
