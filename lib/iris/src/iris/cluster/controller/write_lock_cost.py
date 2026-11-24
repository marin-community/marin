# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""How long write transactions wait for and hold the controller's write lock, and what the observability rows cost.

Every write transaction holds the write lock, so time spent there delays every
other writer. The observability rows (:mod:`iris.cluster.controller.observability_rows`)
read their snapshot before commit and enqueue it after commit, both under that
lock.

Each ``ControllerDB`` owns one :class:`CostWindow`, touched only while its lock is
held. Lock wait and hold time are summed and emitted once per window, because
every write transaction has them. A transaction that copies rows also gets its own
record (:class:`RowTransactionCost`), so the cost of a burst can be read against
its row count. ``write_transaction`` emits both after it releases the lock.
"""

from dataclasses import dataclass, field

from rigging import telemetry
from rigging.telemetry.serialization import EventBody

WINDOW_SECONDS = 60.0
ROW_TRANSACTION_EVENT = "iris_observability_rows_transaction"

_LOCK_HELD_SECONDS = telemetry.counter("iris_controller_write_lock_held_seconds", unit="s")
_LOCK_WAIT_SECONDS = telemetry.counter("iris_controller_write_lock_wait_seconds", unit="s")


@dataclass(frozen=True)
class RowTransactionCost:
    """One transaction's observability rows: how many per table, and their time under the lock."""

    job_rows: int
    job_config_rows: int
    task_rows: int
    attempt_rows: int
    snapshot_seconds: float
    enqueue_seconds: float

    def event_body(self) -> EventBody:
        return EventBody(
            {
                "job_rows": self.job_rows,
                "job_config_rows": self.job_config_rows,
                "task_rows": self.task_rows,
                "attempt_rows": self.attempt_rows,
                "rows": self.job_rows + self.job_config_rows + self.task_rows + self.attempt_rows,
                "snapshot_seconds": self.snapshot_seconds,
                "enqueue_seconds": self.enqueue_seconds,
                "seconds": self.snapshot_seconds + self.enqueue_seconds,
            }
        )


@dataclass(frozen=True)
class LockTime:
    """One window's total write-lock wait and hold time."""

    wait_seconds: float
    held_seconds: float


@dataclass(frozen=True)
class PendingCost:
    """What one write transaction has to emit once it has released the lock."""

    lock_time: LockTime | None
    row_transactions: tuple[RowTransactionCost, ...]


@dataclass
class CostWindow:
    """Lock time summed until ``WINDOW_SECONDS`` have passed, and row records waiting to be emitted."""

    started: float
    wait_seconds: float = 0.0
    held_seconds: float = 0.0
    row_transactions: list[RowTransactionCost] = field(default_factory=list)

    def add_row_transaction(self, cost: RowTransactionCost) -> None:
        self.row_transactions.append(cost)

    def close_transaction(self, wait_seconds: float, held_seconds: float, now: float) -> PendingCost:
        """Add one transaction's lock time and take what is ready to emit.

        The lock totals are returned, and the window restarted, once it is full.
        """
        self.wait_seconds += wait_seconds
        self.held_seconds += held_seconds
        rows = tuple(self.row_transactions)
        self.row_transactions.clear()
        if now - self.started < WINDOW_SECONDS:
            return PendingCost(lock_time=None, row_transactions=rows)
        lock_time = LockTime(wait_seconds=self.wait_seconds, held_seconds=self.held_seconds)
        self.started = now
        self.wait_seconds = self.held_seconds = 0.0
        return PendingCost(lock_time=lock_time, row_transactions=rows)


def emit(pending: PendingCost) -> None:
    """Send a transaction's pending records to telemetry (a no-op when telemetry is not configured)."""
    for row_transaction in pending.row_transactions:
        telemetry.event(ROW_TRANSACTION_EVENT, row_transaction.event_body())
    if pending.lock_time is not None:
        _LOCK_WAIT_SECONDS.add(pending.lock_time.wait_seconds)
        _LOCK_HELD_SECONDS.add(pending.lock_time.held_seconds)
