# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.controller.write_lock_cost import WINDOW_SECONDS, CostWindow, LockTime, RowTransactionCost

_ROWS = RowTransactionCost(
    job_rows=1, job_config_rows=1, task_rows=5, attempt_rows=5, snapshot_seconds=0.02, enqueue_seconds=0.01
)


def test_each_transaction_hands_over_its_row_records_and_the_lock_time_waits_for_a_full_window():
    window = CostWindow(started=100.0)
    window.add_row_transaction(_ROWS)

    first = window.close_transaction(wait_seconds=0.01, held_seconds=0.05, now=101.0)
    second = window.close_transaction(wait_seconds=0.02, held_seconds=0.25, now=100.0 + WINDOW_SECONDS)
    third = window.close_transaction(wait_seconds=0.0, held_seconds=0.01, now=100.0 + WINDOW_SECONDS + 1.0)

    assert (first.lock_time, first.row_transactions) == (None, (_ROWS,))
    assert second.row_transactions == ()
    assert second.lock_time == LockTime(wait_seconds=0.03, held_seconds=0.30)
    assert third.lock_time is None
    assert (window.wait_seconds, window.held_seconds) == (0.0, 0.01)


def test_a_row_record_holds_only_counts_and_durations():
    fields = _ROWS.event_body().fields

    assert fields["rows"] == 12
    assert fields["seconds"] == 0.03
    assert all(isinstance(value, int | float) for value in fields.values())
