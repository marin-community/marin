# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker-side execution context exposed to user task code."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class CounterSnapshot:
    """Bundled counter values and monotonically increasing generation tag.

    The generation increments on every snapshot, so each heartbeat and
    report_result carries a unique tag.  The coordinator uses strict
    ordering (>) to discard stale or out-of-order updates.
    """

    counters: dict[str, int]
    generation: int

    @staticmethod
    def empty(generation: int = 0) -> CounterSnapshot:
        return CounterSnapshot(counters={}, generation=generation)


class WorkerContext(Protocol):
    # Number of concurrent worker slots sharing this actor's RAM. Consumers
    # (e.g. the scatter writer's memory budget) divide a per-actor byte budget
    # by this, so it is part of the contract rather than an implementation
    # detail of a particular WorkerContext.
    num_workers: int

    def get_shared(self, name: str) -> Any: ...
    def increment_counter(self, name: str, value: int = 1) -> None: ...
    def get_counter_snapshot(self) -> CounterSnapshot: ...


_worker_ctx_var: ContextVar[WorkerContext | None] = ContextVar("zephyr_worker_ctx", default=None)


def zephyr_worker_ctx() -> WorkerContext:
    """Get the current worker's context. Only valid inside a worker task."""
    ctx = _worker_ctx_var.get()
    if ctx is None:
        raise RuntimeError("zephyr_worker_ctx() called outside of a worker task")
    return ctx
