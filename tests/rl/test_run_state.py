# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.rl.run_state import RLRunState, RunStatus


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
