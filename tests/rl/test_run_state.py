# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.rl.run_state import RLRunState, RolloutTransferCounters, RunStatus
from marin.rl.telemetry import ArtifactRef, TrackerRunRef, TrackerStream


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


def test_run_state_allocates_monotonic_eval_sequences():
    run_state = RLRunState()

    assert run_state.next_eval_sequence() == 0
    assert run_state.next_eval_sequence() == 1
    assert run_state.next_eval_sequence() == 2


def test_run_state_registers_tracker_and_artifact_refs():
    run_state = RLRunState()
    tracker_ref = TrackerRunRef(
        stream=TrackerStream.ROLLOUT,
        tracker_run_id="wandb-rollout-123",
        project="marin_post_training",
        run_name="rl-test-rollout-0",
        run_url="https://wandb.ai/example/run",
        worker_index=0,
    )
    artifact_ref = ArtifactRef(
        name="train-events",
        path="gs://bucket/metadata/rl-test/events/train-attempt-abc.jsonl",
        artifact_type="event_shard",
        stream=TrackerStream.TRAINER,
    )

    run_state.register_tracker_ref(tracker_ref)
    run_state.register_artifact_ref(artifact_ref)

    assert run_state.get_tracker_ref(TrackerStream.ROLLOUT, worker_index=0) == tracker_ref
    assert run_state.list_tracker_refs() == [tracker_ref]
    assert run_state.get_artifact_ref("train-events") == artifact_ref
    assert run_state.list_artifact_refs() == [artifact_ref]
