# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reserved-pool chip accounting.

On a fungible reservation, the per-size scale groups sharing a ``quota_pool`` draw
from one physical pool of interchangeable TPU chips. This module builds, once per
control tick, a chip ledger over those pools from live scale-group state.

The ledger buckets every pool's chips into live (READY), in-flight
(REQUESTING/BOOTING/INITIALIZING), and draining (DRAINING) slices, and exposes how
many are free now versus free-or-draining. It is a pure accounting view: building
it changes no scaling decision on its own.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup, SliceLifecycleState
from iris.cluster.tpu_topology import get_tpu_topology

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoolLedger:
    """Chip accounting for one fungible reservation pool, one tick."""

    pool_id: str
    reservation_chips: int
    live_chips: int  # READY slices
    inflight_chips: int  # REQUESTING/BOOTING/INITIALIZING (+ pending_scale_ups), NOT draining
    draining_chips: int  # DRAINING slices (still physically allocated until reaped)
    inflight_slices_by_variant: dict[str, int]  # variant -> count of in-flight (non-draining) slices

    @property
    def allocated_chips(self) -> int:
        """Chips currently held by any slice (live + in-flight + draining)."""
        return self.live_chips + self.inflight_chips + self.draining_chips

    @property
    def free_chips(self) -> int:
        """Chips free against the reservation budget right now (excludes draining)."""
        return self.reservation_chips - self.allocated_chips

    @property
    def incoming_chips(self) -> int:
        """Chips free now or being freed by an in-flight drain (free + draining)."""
        return self.free_chips + self.draining_chips

    @property
    def utilization(self) -> float:
        """Fraction of the reservation consumed (0.0 when the budget is zero)."""
        if self.reservation_chips <= 0:
            return 0.0
        return self.allocated_chips / self.reservation_chips


@dataclass(frozen=True)
class ReservationLedger:
    """Single per-tick chip ledger over all fungible reservation pools.

    Exposes each pool's chip accounting (``free_chips``, ``incoming_chips``,
    ``draining_chips``, ``inflight_slices``) plus lookup maps from worker/variant
    to the pool and physical slice they belong to.
    """

    pools: dict[str, PoolLedger]
    worker_pool: dict[str, str]  # worker_id -> pool_id of the pool that worker's slice draws from
    worker_slice: dict[str, str]  # worker_id -> slice_id of the physical slice the worker belongs to
    variant_pool: dict[str, str]  # device_variant -> pool_id the variant's slices draw from
    chips_per_variant: dict[str, int]  # device_variant -> physical chips one slice of that variant consumes

    def is_empty(self) -> bool:
        """True when no fungible reservation pool participates."""
        return not self.variant_pool

    def free_chips(self, pool_id: str) -> int:
        """Chips free against the reservation budget now (0 if the pool is absent)."""
        pool = self.pools.get(pool_id)
        return pool.free_chips if pool is not None else 0

    def incoming_chips(self, pool_id: str) -> int:
        """Chips free now or being freed by a drain (0 if the pool is absent)."""
        pool = self.pools.get(pool_id)
        return pool.incoming_chips if pool is not None else 0

    def draining_chips(self, pool_id: str) -> int:
        """Chips currently draining in this pool (0 if the pool is absent)."""
        pool = self.pools.get(pool_id)
        return pool.draining_chips if pool is not None else 0

    def inflight_slices(self, pool_id: str, variant: str) -> int:
        """In-flight (non-draining) slice count of ``variant`` in ``pool_id`` (0 if absent)."""
        pool = self.pools.get(pool_id)
        if pool is None:
            return 0
        return pool.inflight_slices_by_variant.get(variant, 0)


_INFLIGHT_LIFECYCLES = (
    SliceLifecycleState.REQUESTING,
    SliceLifecycleState.BOOTING,
    SliceLifecycleState.INITIALIZING,
)


def build_reservation_ledger(groups: Iterable[ScalingGroup]) -> ReservationLedger:
    """Build the per-tick chip ledger from live scaling-group state.

    Only groups with ``reservation_chips > 0`` participate; they are bucketed by
    ``quota_pool``. Each slice consumes ``chip_count(variant)`` physical chips, so a
    v4-16 (8 chips) and two v4-8s (4 chips each) count equally against the pool.
    Every member of a pool carries the same budget; a mismatch (or a member with no
    ``quota_pool``) is a config error and raises.
    """
    budgets: dict[str, int] = {}
    live: dict[str, int] = {}
    inflight: dict[str, int] = {}
    draining: dict[str, int] = {}
    inflight_slices_by_variant: dict[str, dict[str, int]] = {}
    variant_pool: dict[str, str] = {}
    chips_per_variant: dict[str, int] = {}
    worker_pool: dict[str, str] = {}
    worker_slice: dict[str, str] = {}

    for group in groups:
        budget = group.reservation_chips
        if budget <= 0:
            continue
        pool_id = group.config.quota_pool
        if not pool_id:
            raise ValueError(f"scale group {group.name!r} sets reservation_chips but no quota_pool")
        prior = budgets.setdefault(pool_id, budget)
        if prior != budget:
            raise ValueError(f"reservation pool {pool_id!r} has conflicting reservation_chips: {prior} vs {budget}")

        variant = group.accelerator_variant
        chips = get_tpu_topology(variant).chip_count
        variant_pool[variant] = pool_id
        chips_per_variant[variant] = chips

        counts = group.slice_state_counts()
        inflight_count = sum(counts[state] for state in _INFLIGHT_LIFECYCLES)
        live[pool_id] = live.get(pool_id, 0) + counts[SliceLifecycleState.READY] * chips
        inflight[pool_id] = inflight.get(pool_id, 0) + inflight_count * chips
        draining[pool_id] = draining.get(pool_id, 0) + counts[SliceLifecycleState.DRAINING] * chips
        if inflight_count:
            inflight_slices_by_variant.setdefault(pool_id, {})[variant] = inflight_count

        for worker_id in group.all_worker_ids():
            worker_pool[worker_id] = pool_id
        worker_slice.update(group.worker_slice_ids())

    pools = {
        pool_id: PoolLedger(
            pool_id=pool_id,
            reservation_chips=budget,
            live_chips=live.get(pool_id, 0),
            inflight_chips=inflight.get(pool_id, 0),
            draining_chips=draining.get(pool_id, 0),
            inflight_slices_by_variant=inflight_slices_by_variant.get(pool_id, {}),
        )
        for pool_id, budget in budgets.items()
    }
    return ReservationLedger(
        pools=pools,
        worker_pool=worker_pool,
        worker_slice=worker_slice,
        variant_pool=variant_pool,
        chips_per_variant=chips_per_variant,
    )


def log_reservation_ledger(ledger: ReservationLedger) -> None:
    """Emit one utilization line per fungible reservation pool."""
    for pool in ledger.pools.values():
        logger.info(
            "reserved pool %s: res=%d live=%d inflight=%d draining=%d free=%d (%.0f%% utilized)",
            pool.pool_id,
            pool.reservation_chips,
            pool.live_chips,
            pool.inflight_chips,
            pool.draining_chips,
            pool.free_chips,
            pool.utilization * 100,
        )
