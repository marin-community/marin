# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reservation reclaim: free reserved chips held by lower-priority slices.

A reserved pool (scale groups that share one ``quota_pool`` and draw on one cloud
reservation) can be fully held by small, lower-priority slices. Preemption cannot
help a higher-priority job that needs a larger slice: it only replaces one existing
slice of its own variant and at least its size, and idle scale-down never touches busy slices.
The larger slice then waits on the cloud indefinitely.

:func:`plan_reservation_reclaim` picks lower-priority slices in the same pool whose
deletion returns enough chips for the waiting slice. It is a pure decision; the
autoscaler records it and the controller fails the victims' workers and removes
their slices (see ``Controller._remove_worker_capacity``).

Iris cannot see how many chips are free in the cloud reservation, so the planner
assumes none are free and frees the waiting slice's full chip count. It may drain
more than strictly needed, never less.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.cluster.config import ReservationReclaimConfig
from iris.cluster.types import WorkerStatusMap
from iris.rpc.proto_display import PRIORITY_BAND_VALUES, priority_band_rank, priority_band_value


@dataclass(frozen=True)
class PoolSlice:
    """One tracked slice of a reserved scale group, as the planner sees it."""

    slice_id: str
    group: str
    quota_pool: str
    chips: int
    ready: bool
    # True while the slice waits on the cloud for capacity (not yet READY, no failure).
    waiting: bool
    created_at: Timestamp
    worker_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReclaimDecision:
    """Delete ``victims`` so the cloud can fulfil ``waiting_slice_id``."""

    quota_pool: str
    group: str
    waiting_slice_id: str
    preemptor_band: int
    chips_needed: int
    victims: tuple[PoolSlice, ...]

    @property
    def chips_freed(self) -> int:
        return sum(victim.chips for victim in self.victims)

    @property
    def worker_ids(self) -> list[str]:
        return [worker_id for victim in self.victims for worker_id in victim.worker_ids]


# ``ReclaimBlocked.reason`` values.
BLOCKED_TOO_FEW_CHIPS = "too_few_chips"
BLOCKED_AWAITING_CAPACITY = "awaiting_capacity"


@dataclass(frozen=True)
class ReclaimBlocked:
    """A waiting slice that gets no reclaim this pass.

    ``reason`` is ``too_few_chips`` when the pool holds too few lower-priority chips, or
    ``awaiting_capacity`` when an earlier reclaim already freed chips for this slice and it still waits.
    """

    quota_pool: str
    group: str
    waiting_slice_id: str
    preemptor_band: int
    chips_needed: int
    chips_reclaimable: int
    reason: str = BLOCKED_TOO_FEW_CHIPS


@dataclass(frozen=True)
class ReclaimPlan:
    decisions: list[ReclaimDecision]
    blocked: list[ReclaimBlocked]


# Rank of an idle slice: below every band, so idle slices sort first and are always eligible.
_IDLE_RANK = len(PRIORITY_BAND_VALUES)


def _slice_rank(slice_: PoolSlice, worker_status: WorkerStatusMap) -> int | None:
    """Rank of the slice's highest running band, ``_IDLE_RANK`` when idle.

    None (never a victim) when a worker is missing from the snapshot or runs a task of an unknown band.
    """
    ranks: list[int] = []
    for worker_id in slice_.worker_ids:
        status = worker_status.get(worker_id)
        if status is None or any(band not in PRIORITY_BAND_VALUES for band in status.running_bands):
            return None
        ranks.extend(priority_band_rank(band) for band in status.running_bands)
    return min(ranks, default=_IDLE_RANK)


def plan_reservation_reclaim(
    slices: Sequence[PoolSlice],
    waiting_bands: Mapping[str, int],
    worker_status: WorkerStatusMap,
    config: ReservationReclaimConfig,
    last_reclaim: Mapping[str, Timestamp],
    now: Timestamp,
    outstanding: Mapping[str, str] | None = None,
) -> ReclaimPlan:
    """Choose lower-priority slices to delete for each pool's most important waiting slice.

    Args:
        slices: Every tracked slice of the reserved scale groups.
        waiting_bands: Scale group -> highest band of the coscheduled demand routed to it this tick.
        worker_status: The controller's worker snapshot, with each worker's running bands.
        config: Reclaim settings; pools not listed in ``quota_pools`` are never touched.
        last_reclaim: Pool -> time of its last reclaim, for the cooldown.
        now: Current time.
        outstanding: Pool -> waiting slice of its last reclaim, while that slice still waits on the cloud.

    Rules:
        - A slice waits for reclaim once it has waited on the cloud for ``min_wait`` and its group has
          coscheduled demand. Per pool, only the highest-band (then oldest) waiting slice is served.
        - Victims are READY slices of other groups in the same pool. Every task on a victim runs a band
          strictly below the waiting demand's band and no higher than ``highest_victim_band``; a slice
          with one task outside that range is skipped whole, as is a slice with a worker missing from
          the snapshot.
        - Order: idle slices, then the lowest band, then the fewest chips, then the youngest.
        - Stop once the freed chips reach the waiting slice's chips. If the candidates cannot reach
          that, nothing is reclaimed and the shortfall is reported as blocked.
        - At most one reclaim per pool per ``cooldown``.
        - A pool with an outstanding reclaim reclaims nothing more until that slice stops waiting
          (READY, initializing, failed, or gone). The outstanding slice is reported as blocked
          (``awaiting_capacity``), so a reservation the cloud never fills is not drained again and again.
    """
    outstanding = outstanding or {}
    pools = set(config.quota_pools)
    victim_limit = priority_band_rank(priority_band_value(config.highest_victim_band))

    waiting_by_pool: dict[str, list[tuple[PoolSlice, int]]] = {}
    for slice_ in slices:
        band = waiting_bands.get(slice_.group)
        if slice_.quota_pool not in pools or not slice_.waiting or band not in PRIORITY_BAND_VALUES:
            continue
        if now.epoch_ms() - slice_.created_at.epoch_ms() < config.min_wait.to_ms():
            continue
        waiting_by_pool.setdefault(slice_.quota_pool, []).append((slice_, band))

    decisions: list[ReclaimDecision] = []
    blocked: list[ReclaimBlocked] = []
    for pool, waiting in sorted(waiting_by_pool.items()):
        served = outstanding.get(pool)
        if served is not None:
            for slice_, band in waiting:
                if slice_.slice_id == served:
                    blocked.append(
                        ReclaimBlocked(
                            quota_pool=pool,
                            group=slice_.group,
                            waiting_slice_id=slice_.slice_id,
                            preemptor_band=band,
                            chips_needed=slice_.chips,
                            chips_reclaimable=0,
                            reason=BLOCKED_AWAITING_CAPACITY,
                        )
                    )
            continue
        last = last_reclaim.get(pool)
        if last is not None and now.epoch_ms() - last.epoch_ms() < config.cooldown.to_ms():
            continue
        target, band = min(waiting, key=lambda item: (priority_band_rank(item[1]), item[0].created_at.epoch_ms()))
        preemptor_rank = priority_band_rank(band)

        candidates: list[tuple[tuple[int, int, int], PoolSlice]] = []
        for slice_ in slices:
            if slice_.quota_pool != pool or not slice_.ready or slice_.group == target.group:
                continue
            rank = _slice_rank(slice_, worker_status)
            if rank is None or rank <= preemptor_rank or rank < victim_limit:
                continue
            candidates.append(((-rank, slice_.chips, -slice_.created_at.epoch_ms()), slice_))
        candidates.sort(key=lambda item: item[0])

        victims: list[PoolSlice] = []
        freed = 0
        for _key, slice_ in candidates:
            if freed >= target.chips:
                break
            victims.append(slice_)
            freed += slice_.chips

        if freed < target.chips:
            blocked.append(
                ReclaimBlocked(
                    quota_pool=pool,
                    group=target.group,
                    waiting_slice_id=target.slice_id,
                    preemptor_band=band,
                    chips_needed=target.chips,
                    chips_reclaimable=freed,
                )
            )
            continue
        decisions.append(
            ReclaimDecision(
                quota_pool=pool,
                group=target.group,
                waiting_slice_id=target.slice_id,
                preemptor_band=band,
                chips_needed=target.chips,
                victims=tuple(victims),
            )
        )
    return ReclaimPlan(decisions=decisions, blocked=blocked)
