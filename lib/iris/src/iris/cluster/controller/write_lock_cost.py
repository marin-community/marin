# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""How long write transactions hold the controller's write lock, and how much of that the observability rows take.

Every write transaction holds the write lock, so time spent there delays every
other writer. The observability rows (:mod:`iris.cluster.controller.observability_rows`)
read their snapshot before commit and enqueue it after commit, both under that
lock. Each ``ControllerDB`` owns one :class:`CostWindow`, touched only while its
lock is held, and emits one summary per window, so a busy controller sends a few
telemetry records a minute instead of one per transaction.
"""

from dataclasses import dataclass

from rigging import telemetry

WINDOW_SECONDS = 60.0

_LOCK_HELD_SECONDS = telemetry.counter("iris_controller_write_lock_held_seconds", unit="s")
_ROW_SECONDS = telemetry.counter("iris_observability_rows_seconds", unit="s")
_ROWS = telemetry.counter("iris_observability_rows_count", unit="{row}")
_ROW_MAX_SECONDS = telemetry.gauge("iris_observability_rows_max_seconds", unit="s")


@dataclass(frozen=True)
class WriteLockCost:
    """One window's totals. ``row_seconds`` is snapshot plus enqueue time."""

    lock_held_seconds: float
    row_seconds: float
    row_max_seconds: float
    rows: int


@dataclass
class CostWindow:
    """Accumulates write-lock and observability-row time until ``WINDOW_SECONDS`` have passed."""

    started: float
    lock_held_seconds: float = 0.0
    row_seconds: float = 0.0
    row_max_seconds: float = 0.0
    rows: int = 0

    def add_row_time(self, seconds: float) -> None:
        """Add time spent on observability rows: a snapshot, or an enqueue."""
        self.row_seconds += seconds

    def add_row_transaction(self, seconds: float, rows: int) -> None:
        """Count one transaction's rows and its total row time (snapshot plus enqueue)."""
        self.row_max_seconds = max(self.row_max_seconds, seconds)
        self.rows += rows

    def add_lock_held(self, seconds: float, now: float) -> WriteLockCost | None:
        """Add one transaction's lock time; return the window's totals and restart it once it is full."""
        self.lock_held_seconds += seconds
        if now - self.started < WINDOW_SECONDS:
            return None
        cost = WriteLockCost(
            lock_held_seconds=self.lock_held_seconds,
            row_seconds=self.row_seconds,
            row_max_seconds=self.row_max_seconds,
            rows=self.rows,
        )
        self.started = now
        self.lock_held_seconds = self.row_seconds = self.row_max_seconds = 0.0
        self.rows = 0
        return cost


def emit(cost: WriteLockCost) -> None:
    """Send one window's totals to telemetry (a no-op when telemetry is not configured)."""
    _LOCK_HELD_SECONDS.add(cost.lock_held_seconds)
    _ROW_SECONDS.add(cost.row_seconds)
    _ROWS.add(cost.rows)
    _ROW_MAX_SECONDS.set(cost.row_max_seconds)
