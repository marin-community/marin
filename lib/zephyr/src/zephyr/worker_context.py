# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker-side execution context exposed to user task code."""

import enum
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol


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
    # Number of concurrent worker slots sharing this actor's RAM. Consumers
    # (e.g. the scatter writer's memory budget) divide a per-actor byte budget
    # by this, so it is part of the contract rather than an implementation
    # detail of a particular WorkerContext.
    num_workers: int

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
