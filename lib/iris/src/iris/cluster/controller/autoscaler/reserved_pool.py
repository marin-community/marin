# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reserved-pool chip accounting.

On a fungible reservation, the per-size scale groups sharing a ``quota_pool`` draw
from one physical pool of interchangeable TPU chips. This module computes, per
pool, how many chips live + in-flight slices consume and how many remain free
against the configured budget.

It is read-only accounting: it backs the utilization log line and is the capacity
view the cross-variant preemption pass consults to decide when a full reserved
pool must shed lower-priority slices. It changes no scaling decision on its own.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup
from iris.cluster.tpu_topology import get_tpu_topology

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReservedPoolView:
    """Read-only chip-ledger view of the fungible reservation pools.

    Built once per scheduling tick from live scaling-group state and consumed by
    the cross-variant preemption pass. It maps every fungible pool to its free
    chips and lets the pass translate a victim worker / preemptor variant into
    the pool and chip count it touches. ``pools_on_cooldown`` carries the pools
    whose drain + reprovision is still in flight, so the pass leaves them alone.
    """

    free_chips: dict[str, int]
    """pool_id -> chips currently free against the reservation budget."""
    worker_pool: dict[str, str]
    """worker_id -> pool_id of the pool that worker's slice draws from."""
    worker_slice: dict[str, str]
    """worker_id -> slice_id of the physical slice that worker belongs to.

    The preemption pass groups victims by slice (not by task) because the drain
    tears down a whole physical slice: every task on a slice is evicted together,
    and the slice's chips are freed once."""
    variant_pool: dict[str, str]
    """device_variant -> pool_id the variant's slices draw from."""
    chips_per_variant: dict[str, int]
    """device_variant -> physical chips one slice of that variant consumes."""
    pools_on_cooldown: frozenset[str] = frozenset()
    """Pools recently drained whose reprovision is in flight; skip preemption."""

    def is_empty(self) -> bool:
        """True when no fungible reservation pool participates."""
        return not self.variant_pool


def reserved_pool_view(
    groups: Iterable[ScalingGroup],
    pools_on_cooldown: frozenset[str] = frozenset(),
) -> ReservedPoolView:
    """Build the cross-variant preemption chip-ledger view from scaling groups.

    Only groups with ``reservation_chips > 0`` participate; they are bucketed by
    ``quota_pool``. ``free_chips`` is the reservation budget minus live and in-flight
    slice consumption, and the per-variant and per-worker maps let the preemption
    pass resolve a preemptor's variant and a victim's worker to their pool and chip
    footprint.
    """
    groups = list(groups)
    free_chips = {pool_id: usage.free_chips for pool_id, usage in reserved_pool_usage(groups).items()}
    variant_pool: dict[str, str] = {}
    chips_per_variant: dict[str, int] = {}
    worker_pool: dict[str, str] = {}
    worker_slice: dict[str, str] = {}
    for group in groups:
        if group.reservation_chips <= 0:
            continue
        pool_id = group.config.quota_pool
        variant = group.accelerator_variant
        variant_pool[variant] = pool_id
        chips_per_variant[variant] = get_tpu_topology(variant).chip_count
        for worker_id in group.all_worker_ids():
            worker_pool[worker_id] = pool_id
        worker_slice.update(group.worker_slice_ids())
    return ReservedPoolView(
        free_chips=free_chips,
        worker_pool=worker_pool,
        worker_slice=worker_slice,
        variant_pool=variant_pool,
        chips_per_variant=chips_per_variant,
        pools_on_cooldown=pools_on_cooldown,
    )


@dataclass(frozen=True)
class ReservedPoolUsage:
    """Chip accounting for one fungible reservation pool (one ``quota_pool``)."""

    pool_id: str
    reservation_chips: int
    consumed_chips: int

    @property
    def free_chips(self) -> int:
        return self.reservation_chips - self.consumed_chips

    @property
    def utilization(self) -> float:
        """Fraction of the reservation consumed (0.0 when the budget is zero)."""
        if self.reservation_chips <= 0:
            return 0.0
        return self.consumed_chips / self.reservation_chips


def reserved_pool_usage(groups: Iterable[ScalingGroup]) -> dict[str, ReservedPoolUsage]:
    """Aggregate live + in-flight chip consumption per fungible reservation pool.

    Only groups with ``reservation_chips > 0`` participate; they are bucketed by
    ``quota_pool``. Each slice consumes ``chip_count(variant)`` physical chips, so
    a v4-16 (8 chips) and two v4-8s (4 chips each) count equally against the pool.
    Every member of a pool carries the same budget; a mismatch is a config error
    and raises.
    """
    budgets: dict[str, int] = {}
    consumed: dict[str, int] = {}
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
        chips_per_slice = get_tpu_topology(group.accelerator_variant).chip_count
        consumed[pool_id] = consumed.get(pool_id, 0) + group.slice_count() * chips_per_slice

    return {
        pool_id: ReservedPoolUsage(
            pool_id=pool_id,
            reservation_chips=budget,
            consumed_chips=consumed.get(pool_id, 0),
        )
        for pool_id, budget in budgets.items()
    }


def log_reserved_pool_usage(usage: dict[str, ReservedPoolUsage]) -> None:
    """Emit one utilization line per fungible reservation pool."""
    for pool in usage.values():
        logger.info(
            "reserved pool %s: %d/%d chips used (%d free, %.0f%% utilized)",
            pool.pool_id,
            pool.consumed_chips,
            pool.reservation_chips,
            pool.free_chips,
            pool.utilization * 100,
        )
