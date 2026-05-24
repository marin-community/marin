# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.rl.run_state import RLRunState, RolloutTransferCounters, RunStatus


def test_run_state_snapshot_tracks_latest_completed_train_step():
    run_state = RLRunState()

    run_state.update_train_step(3)
    run_state.update_train_step(1)

    snapshot = run_state.get_snapshot()

    assert snapshot.status == RunStatus.RUNNING.value
    assert snapshot.train_step == 3
    assert snapshot.failure_message is None


def test_run_state_snapshot_preserves_failure_message():
    run_state = RLRunState()

    run_state.update_train_step(5)
    run_state.mark_failed("trainer crashed")

    snapshot = run_state.get_snapshot()

    assert snapshot.status == RunStatus.FAILED.value
    assert snapshot.train_step == 5
    assert snapshot.failure_message == "trainer crashed"


def test_run_state_accumulates_rollout_transfer_counters_per_worker():
    run_state = RLRunState()

    updated = run_state.add_rollout_transfer_counters(
        worker_index=1,
        total_polls_delta=5,
        successful_receives_delta=3,
        failed_receives_delta=1,
    )
    assert updated == RolloutTransferCounters(total_polls=5, successful_receives=3, failed_receives=1)

    updated = run_state.add_rollout_transfer_counters(
        worker_index=1,
        total_polls_delta=2,
        successful_receives_delta=2,
        failed_receives_delta=0,
    )
    assert updated == RolloutTransferCounters(total_polls=7, successful_receives=5, failed_receives=1)

    assert run_state.get_rollout_transfer_counters(worker_index=0) == RolloutTransferCounters()


def test_run_state_rollout_assignment_reserve_is_idempotent_until_commit():
    run_state = RLRunState()

    first = run_state.reserve_rollout_assignment(
        worker_index=0,
        lesson_id="math_full",
        worker_seed=1042,
        dataset_len=31,
        n_examples=8,
    )
    retry = run_state.reserve_rollout_assignment(
        worker_index=0,
        lesson_id="math_full",
        worker_seed=1042,
        dataset_len=31,
        n_examples=8,
    )

    assert retry == first
    assert run_state.get_rollout_schedule_cursor(worker_index=0, lesson_id="math_full").position == 0
    assert run_state.get_rollout_schedule_stats()["reserved_assignments"] == 1
    assert run_state.get_rollout_schedule_stats()["reused_pending_assignments"] == 1

    cursor = run_state.commit_rollout_assignment(
        worker_index=0,
        lesson_id="math_full",
        assignment_id=first.assignment_id,
    )

    assert cursor.position == 8
    assert run_state.get_rollout_schedule_stats()["committed_assignments"] == 1

    second = run_state.reserve_rollout_assignment(
        worker_index=0,
        lesson_id="math_full",
        worker_seed=1042,
        dataset_len=31,
        n_examples=8,
    )

    assert second.start_position == 8
    assert second.assignment_id != first.assignment_id


def test_run_state_recovers_rollout_assignment_cursor_from_ledger(tmp_path):
    ledger_path = tmp_path / "rollout_schedule_ledger"
    run_state = RLRunState(schedule_ledger_path=str(ledger_path))

    first = run_state.reserve_rollout_assignment(
        worker_index=0,
        lesson_id="math_full",
        worker_seed=1042,
        dataset_len=31,
        n_examples=8,
    )
    run_state.commit_rollout_assignment(
        worker_index=0,
        lesson_id="math_full",
        assignment_id=first.assignment_id,
    )

    recovered = RLRunState(schedule_ledger_path=str(ledger_path))

    assert recovered.get_rollout_schedule_cursor(worker_index=0, lesson_id="math_full").position == 8
    assert recovered.get_rollout_schedule_stats() == {
        "active_cursors": 1,
        "pending_assignments": 0,
        "reserved_assignments": 0,
        "reused_pending_assignments": 0,
        "committed_assignments": 0,
        "ledger_recovered_assignments": 1,
    }

    next_assignment = recovered.reserve_rollout_assignment(
        worker_index=0,
        lesson_id="math_full",
        worker_seed=1042,
        dataset_len=31,
        n_examples=8,
    )

    assert next_assignment.start_position == 8
    assert next_assignment.assignment_id != first.assignment_id


def test_run_state_rollout_assignment_cursors_are_per_worker():
    run_state = RLRunState()

    worker_0 = run_state.reserve_rollout_assignment(
        worker_index=0,
        lesson_id="math_full",
        worker_seed=1042,
        dataset_len=31,
        n_examples=8,
    )
    worker_1 = run_state.reserve_rollout_assignment(
        worker_index=1,
        lesson_id="math_full",
        worker_seed=1043,
        dataset_len=31,
        n_examples=8,
    )

    assert worker_0.start_position == 0
    assert worker_1.start_position == 0
    assert worker_0.indices != worker_1.indices
