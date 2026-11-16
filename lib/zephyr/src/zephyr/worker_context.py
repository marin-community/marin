# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker-side execution context exposed to user task code."""

import enum
import logging
from collections.abc import Hashable, Iterable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol, TypeVar

logger = logging.getLogger(__name__)


class Aggregation(enum.StrEnum):
    """How a counter is reduced when aggregating across multiple worker snapshots."""

    SUM = "sum"
    AVERAGE = "average"
    MAX = "max"
    MIN = "min"


class CounterKey(NamedTuple):
    """Identifies a counter by scope and name; ``stage`` is None for pipeline-level counters.

    Counters with the same name in different scopes are separate counters.
    """

    stage: str | None
    name: str


@dataclass
class CounterEntry:
    """A single counter value with its aggregation mode."""

    value: int | float
    aggregation: Aggregation = field(default=Aggregation.SUM)
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


_K = TypeVar("_K", bound=Hashable)


def merge_counter_entries(entries: Iterable[tuple[_K, CounterEntry]]) -> tuple[dict[_K, CounterEntry], set[_K]]:
    """Fold counter entries into one accumulator per key.

    The single reducer used everywhere counters are combined: across a
    worker's concurrent runners, and across worker snapshots on the
    coordinator. Keys are :class:`CounterKey` to keep scopes apart, or bare
    names to fold every scope of a name together. Each key's first entry
    seeds a fresh accumulator (the inputs are never mutated) and the rest
    are folded in with :meth:`CounterEntry.merge`, so AVERAGE stays weighted
    by observation count no matter how many levels the values pass through.

    Entries that disagree with the accumulator's aggregation mode are
    skipped and their keys returned in the second element; a warning is
    logged once per key. That is only reachable when user code calls
    ``set_aggregation`` with different modes for the same counter name.

    Returns:
        ``(merged, conflicted)``. Callers decide whether to drop the
        conflicted keys or report the partial fold.
    """
    merged: dict[_K, CounterEntry] = {}
    conflicted: set[_K] = set()
    for key, entry in entries:
        accumulated = merged.get(key)
        if accumulated is None:
            merged[key] = CounterEntry(entry.value, entry.aggregation, entry.count)
            continue
        if accumulated.aggregation != entry.aggregation:
            if key not in conflicted:
                logger.warning(
                    "Counter %r has conflicting aggregations: %r vs %r; dropping",
                    key,
                    accumulated.aggregation,
                    entry.aggregation,
                )
                conflicted.add(key)
            continue
        accumulated.merge(entry)
    return merged, conflicted


def counters_by_name(
    entries: Iterable[tuple[CounterKey, CounterEntry]], stage: str | None = None
) -> dict[str, CounterEntry]:
    """Fold scoped counters into one entry per name.

    With ``stage`` set, only that stage's counters are included. With
    ``stage=None``, a name recorded under several scopes folds into one
    entry. Names whose entries disagree on aggregation are dropped.
    """
    merged, conflicted = merge_counter_entries(
        (key.name, entry) for key, entry in entries if stage is None or key.stage == stage
    )
    return {name: entry for name, entry in merged.items() if name not in conflicted}


@dataclass
class InputProgress:
    """Rows consumed from the current input, with an optional known total."""

    completed: int
    total: int | None


@dataclass(frozen=True)
class RunnerSnapshot:
    """Detached counters and input progress from one runner."""

    counters: dict[CounterKey, CounterEntry] = field(default_factory=dict)
    input_progress: InputProgress | None = None


@dataclass(frozen=True)
class ShardProgress:
    """Live progress for one runner's current shard and input."""

    stage_generation: int
    attempt: int
    elapsed_seconds: float
    records_in: int
    input_rows_completed: int | None
    input_rows_total: int | None


@dataclass
class CounterSnapshot:
    """Counter values, per-shard progress, and a monotonic generation tag.

    The generation increments on every snapshot, so each heartbeat and
    report_result carries a unique tag.  The coordinator uses strict
    ordering (>) to discard stale or out-of-order updates.

    Counters are keyed by scope and name; each entry carries its own
    aggregation mode.
    """

    counters: dict[CounterKey, CounterEntry]
    generation: int
    shards: dict[int, ShardProgress] = field(default_factory=dict)

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
    def set_input_progress(self, progress: InputProgress) -> None: ...


_worker_ctx_var: ContextVar[WorkerContext | None] = ContextVar("zephyr_worker_ctx", default=None)


def zephyr_worker_ctx() -> WorkerContext:
    """Get the current worker's context. Only valid inside a worker task."""
    ctx = _worker_ctx_var.get()
    if ctx is None:
        raise RuntimeError("zephyr_worker_ctx() called outside of a worker task")
    return ctx
