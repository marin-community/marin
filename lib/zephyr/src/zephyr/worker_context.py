# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker-side execution context exposed to user task code."""

import enum
import logging
from collections.abc import Iterable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Aggregation(enum.StrEnum):
    """How a counter is reduced when aggregating across multiple worker snapshots."""

    SUM = "sum"
    AVERAGE = "average"
    MAX = "max"
    MIN = "min"


@dataclass
class CounterEntry:
    """A single named counter value with its aggregation mode and optional stage scope."""

    value: int | float
    aggregation: Aggregation = field(default=Aggregation.SUM)
    stage: str | None = None
    count: int = 1  # number of observations; used for rolling average via update_counter

    def merge(self, other: "CounterEntry") -> None:
        """Fold *other* into this entry in-place using this entry's aggregation.

        ``other.count`` is respected for AVERAGE so that merging two accumulated
        entries (e.g. across concurrent runners) produces the correct weighted mean.
        All other modes ignore ``other.count``.
        """
        match self.aggregation:
            case Aggregation.SUM:
                self.value += other.value
            case Aggregation.MAX:
                self.value = max(self.value, other.value)
            case Aggregation.MIN:
                self.value = min(self.value, other.value)
            case Aggregation.AVERAGE:
                total = self.count + other.count
                if total > 0:
                    self.value = (self.value * self.count + other.value * other.count) / total
                self.count = total


def merge_counter_entries(entries: Iterable[tuple[str, CounterEntry]]) -> tuple[dict[str, CounterEntry], set[str]]:
    """Fold named counter entries into one accumulator per name.

    The single reducer used everywhere counters are combined: across a
    worker's concurrent runners, and across worker snapshots on the
    coordinator. Each name's first entry seeds a fresh accumulator (the
    inputs are never mutated) and the rest are folded in with
    :meth:`CounterEntry.merge`, so AVERAGE stays weighted by observation
    count no matter how many levels the values pass through.

    Entries that disagree with the accumulator's aggregation mode are
    skipped and their names returned in the second element; a warning is
    logged once per name. That is only reachable when user code calls
    ``set_aggregation`` with different modes for the same counter name.

    Returns:
        ``(merged, conflicted)``. Callers decide whether to drop the
        conflicted names or report the partial fold.
    """
    merged: dict[str, CounterEntry] = {}
    conflicted: set[str] = set()
    for name, entry in entries:
        accumulated = merged.get(name)
        if accumulated is None:
            merged[name] = CounterEntry(entry.value, entry.aggregation, entry.stage, entry.count)
            continue
        if accumulated.aggregation != entry.aggregation:
            if name not in conflicted:
                logger.warning(
                    "Counter %r has conflicting aggregations: %r vs %r; dropping",
                    name,
                    accumulated.aggregation,
                    entry.aggregation,
                )
                conflicted.add(name)
            continue
        accumulated.merge(entry)
    return merged, conflicted


@dataclass
class CounterSnapshot:
    """Bundled counter values and monotonically increasing generation tag.

    The generation increments on every snapshot, so each heartbeat and
    report_result carries a unique tag.  The coordinator uses strict
    ordering (>) to discard stale or out-of-order updates.

    Each entry carries its own aggregation mode and optional stage scope.
    """

    counters: dict[str, CounterEntry]
    generation: int

    @staticmethod
    def empty(generation: int = 0) -> "CounterSnapshot":
        return CounterSnapshot(counters={}, generation=generation)


class WorkerContext(Protocol):
    task_memory_bytes: int

    def get_shared(self, name: str) -> Any: ...
    def set_counter(self, name: str, value: int | float, stage: str | None = None) -> None: ...
    def update_counter(self, name: str, value: int | float, stage: str | None = None) -> None: ...
    def set_aggregation(self, name: str, agg: Aggregation) -> None: ...
    def current_stage_name(self) -> str: ...
    def get_counter_snapshot(self) -> CounterSnapshot: ...


_worker_ctx_var: ContextVar[WorkerContext | None] = ContextVar("zephyr_worker_ctx", default=None)


def zephyr_worker_ctx() -> WorkerContext:
    """Get the current worker's context. Only valid inside a worker task."""
    ctx = _worker_ctx_var.get()
    if ctx is None:
        raise RuntimeError("zephyr_worker_ctx() called outside of a worker task")
    return ctx
