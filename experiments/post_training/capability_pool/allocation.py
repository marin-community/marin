# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integer task quotas that realize one capability distribution in every arm of a source-mix comparison.

Every arm gets the same quota per target node, so the capability marginal is identical across arms by construction.
An arm's Taskforge share is applied per node: each node's quota is split so that its Taskforge count is within one
task of ``taskforge_fraction`` times the quota, while the arm's Taskforge total is exactly
``round(taskforge_fraction * pool_size)``. A node's quota must be fillable from TaskTrove alone, which is the
control arm's constraint, and its Taskforge count must be fillable from the Taskforge tasks under that node.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class InfeasiblePolicy(StrEnum):
    """What to do when a source cannot fill a node's quota."""

    ERROR = "error"
    """Raise ``InfeasibleTargetError`` and change nothing."""
    DROP_NODES = "drop_nodes"
    """Remove every node whose ideal share exceeds a source's supply, renormalize the rest and retry."""


@dataclass(frozen=True)
class Supply:
    """Distinct eligible tasks under one target node."""

    tasktrove: int
    taskforge: int


@dataclass(frozen=True)
class Shortfall:
    """One node whose ideal share exceeds what a source can supply."""

    node_id: str
    source: str
    arm: str | None
    ideal: float
    available: int


class InfeasibleTargetError(ValueError):
    def __init__(self, shortfalls: list[Shortfall], max_pool_size: int):
        self.shortfalls = shortfalls
        self.max_pool_size = max_pool_size
        lines = [
            f"  {s.node_id}: {s.source}{f' ({s.arm})' if s.arm else ''} needs {s.ideal:.1f}, has {s.available}"
            for s in shortfalls
        ]
        super().__init__(
            f"{len(shortfalls)} target nodes cannot be filled; the largest feasible pool for this target is "
            f"{max_pool_size}:\n" + "\n".join(lines)
        )


class _ApportionError(ValueError):
    pass


@dataclass(frozen=True)
class Allocation:
    """Per-node quotas shared by all arms, and each arm's Taskforge count per node."""

    weights: dict[str, float]
    quotas: dict[str, int]
    taskforge: dict[str, dict[str, int]]
    dropped: tuple[str, ...]

    def tasktrove(self, arm: str, node_id: str) -> int:
        return self.quotas[node_id] - self.taskforge[arm][node_id]


def apportion(total: int, weights: Mapping[str, float], caps: Mapping[str, int]) -> dict[str, int]:
    """Split ``total`` units in proportion to ``weights`` by largest remainder, never exceeding ``caps``.

    Every node receives the floor or the ceiling of its ideal share. Remainder units go to nodes with the largest
    fractional parts that still have room; ties break by node id.
    """
    weight_sum = sum(weights.values())
    ideal = {node: total * weight / weight_sum for node, weight in weights.items()}
    counts = {node: math.floor(share) for node, share in ideal.items()}
    over = [node for node, count in counts.items() if count > caps[node]]
    if over:
        raise _ApportionError(f"Floor shares exceed supply for {sorted(over)}")
    order = sorted(
        (node for node in weights if ideal[node] > counts[node] and counts[node] < caps[node]),
        key=lambda node: (counts[node] - ideal[node], node),
    )
    remaining = total - sum(counts.values())
    if remaining > len(order):
        raise _ApportionError(f"{remaining} remainder units but only {len(order)} nodes have room")
    for node in order[:remaining]:
        counts[node] += 1
    return counts


def taskforge_total(pool_size: int, fraction: float) -> int:
    return math.floor(fraction * pool_size + 0.5)


def _allocate(
    weights: Mapping[str, float], supply: Mapping[str, Supply], pool_size: int, arms: Mapping[str, float]
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    quotas = apportion(pool_size, weights, {node: supply[node].tasktrove for node in weights})
    taskforge = {
        arm: apportion(
            taskforge_total(pool_size, fraction),
            quotas,
            {node: min(quotas[node], supply[node].taskforge) for node in quotas},
        )
        for arm, fraction in arms.items()
    }
    return quotas, taskforge


def shortfalls(
    weights: Mapping[str, float], supply: Mapping[str, Supply], pool_size: int, arms: Mapping[str, float]
) -> list[Shortfall]:
    """Nodes whose ideal share exceeds a source's supply in some arm."""
    weight_sum = sum(weights.values())
    found = []
    for node in sorted(weights):
        ideal = pool_size * weights[node] / weight_sum
        if ideal > supply[node].tasktrove:
            found.append(Shortfall(node, "tasktrove", None, ideal, supply[node].tasktrove))
        for arm, fraction in sorted(arms.items()):
            needed = ideal * taskforge_total(pool_size, fraction) / pool_size
            if needed > supply[node].taskforge:
                found.append(Shortfall(node, "taskforge", arm, needed, supply[node].taskforge))
    return found


def is_feasible(
    weights: Mapping[str, float], supply: Mapping[str, Supply], pool_size: int, arms: Mapping[str, float]
) -> bool:
    try:
        _allocate(weights, supply, pool_size, arms)
    except _ApportionError:
        return False
    return True


def max_pool_size(weights: Mapping[str, float], supply: Mapping[str, Supply], arms: Mapping[str, float]) -> int:
    """The largest pool size that every arm can fill at these weights, or 0."""
    weight_sum = sum(weights.values())
    largest_fraction = max(arms.values())
    bound = sum(item.tasktrove for item in supply.values())
    for node, weight in weights.items():
        share = weight / weight_sum
        bound = min(bound, math.floor((supply[node].tasktrove + 1) / share))
        if largest_fraction > 0:
            bound = min(bound, math.floor((supply[node].taskforge + 1) / (share * largest_fraction)) + 1)
    for size in range(bound, 0, -1):
        if is_feasible(weights, supply, size, arms):
            return size
    return 0


def allocate(
    weights: Mapping[str, float],
    supply: Mapping[str, Supply],
    pool_size: int,
    arms: Mapping[str, float],
    policy: InfeasiblePolicy,
) -> Allocation:
    """Quotas for every arm, or ``InfeasibleTargetError`` under ``InfeasiblePolicy.ERROR``."""
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    if not weights or any(weight <= 0 for weight in weights.values()):
        raise ValueError("Target weights must be positive and non-empty")
    if any(not 0 <= fraction <= 1 for fraction in arms.values()):
        raise ValueError(f"Taskforge fractions must lie in [0, 1]: {dict(arms)}")
    active = dict(weights)
    dropped: list[str] = []
    while True:
        try:
            quotas, taskforge = _allocate(active, supply, pool_size, arms)
            break
        except _ApportionError as error:
            found = shortfalls(active, supply, pool_size, arms)
            if policy is InfeasiblePolicy.ERROR or not found:
                raise InfeasibleTargetError(found, max_pool_size(active, supply, arms)) from error
            for node in sorted({item.node_id for item in found}):
                del active[node]
                dropped.append(node)
            if not active:
                raise InfeasibleTargetError(found, 0) from error
    weight_sum = sum(active.values())
    return Allocation(
        weights={node: weight / weight_sum for node, weight in active.items()},
        quotas=quotas,
        taskforge=taskforge,
        dropped=tuple(dropped),
    )
