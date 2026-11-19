# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.controller.write_lock_cost import WINDOW_SECONDS, CostWindow, WriteLockCost


def test_cost_window_sums_a_full_window_then_starts_over():
    window = CostWindow(started=100.0)
    window.add_row_time(0.02)  # snapshot
    window.add_row_time(0.01)  # enqueue
    window.add_row_transaction(0.03, rows=10)
    assert window.add_lock_held(0.05, now=101.0) is None
    window.add_row_time(0.08)
    window.add_row_transaction(0.08, rows=30)

    cost = window.add_lock_held(0.25, now=100.0 + WINDOW_SECONDS)

    assert cost == WriteLockCost(lock_held_seconds=0.30, row_seconds=0.11, row_max_seconds=0.08, rows=40)
    assert window.add_lock_held(0.01, now=100.0 + WINDOW_SECONDS + 1.0) is None
    assert (window.row_seconds, window.rows, window.lock_held_seconds) == (0.0, 0, 0.01)
