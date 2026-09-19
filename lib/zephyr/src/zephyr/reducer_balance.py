# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Planning for splitting skewed shuffle targets across reduce tasks."""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ReducerBalancePolicy:
    """Thresholds for splitting an oversized reduce target."""

    skew_factor: float = 4.0
    max_slices: int = 16
    min_split_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.skew_factor < 1:
            raise ValueError("skew_factor must be at least 1")
        if self.max_slices < 1:
            raise ValueError("max_slices must be at least 1")
        if self.min_split_bytes < 0:
            raise ValueError("min_split_bytes must be non-negative")


DEFAULT_REDUCER_BALANCE = ReducerBalancePolicy()


@dataclass(frozen=True)
class ReduceTarget:
    """One reduce task's key-disjoint slice of a scatter target."""

    target: int
    slice_index: int = 0
    slice_count: int = 1

    def __post_init__(self) -> None:
        if self.target < 0:
            raise ValueError("target must be non-negative")
        if self.slice_count < 1:
            raise ValueError("slice_count must be at least 1")
        if not 0 <= self.slice_index < self.slice_count:
            raise ValueError("slice_index must be in [0, slice_count)")


def plan_reduce_targets(target_bytes: Sequence[int], policy: ReducerBalancePolicy | None) -> list[ReduceTarget]:
    """Plan ordered reduce tasks from the payload bytes of each scatter target."""
    if any(size < 0 for size in target_bytes):
        raise ValueError("target bytes must be non-negative")

    nonempty = [size for size in target_bytes if size > 0]
    if policy is None or not nonempty:
        return [ReduceTarget(target) for target in range(len(target_bytes))]

    median = statistics.median(nonempty)
    threshold = max(policy.skew_factor * median, policy.min_split_bytes)
    targets: list[ReduceTarget] = []
    for target, size in enumerate(target_bytes):
        slice_count = min(policy.max_slices, math.ceil(size / median)) if size > threshold else 1
        targets.extend(ReduceTarget(target, slice_index, slice_count) for slice_index in range(slice_count))
    return targets
