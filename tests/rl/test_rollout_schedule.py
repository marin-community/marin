# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.rl.rollout_schedule import FeistelEpochSchedule, rollout_assignment


def test_feistel_epoch_schedule_covers_dataset_once_per_epoch():
    schedule = FeistelEpochSchedule(dataset_len=31, seed=123)

    epoch_0 = schedule.indices_for_positions(start_position=0, count=31)
    epoch_1 = schedule.indices_for_positions(start_position=31, count=31)

    assert sorted(epoch_0) == list(range(31))
    assert sorted(epoch_1) == list(range(31))
    assert epoch_0 != epoch_1


def test_feistel_epoch_schedule_crosses_epoch_boundary():
    schedule = FeistelEpochSchedule(dataset_len=7, seed=123)

    crossed = schedule.indices_for_positions(start_position=5, count=5)

    assert crossed[:2] == schedule.indices_for_positions(start_position=5, count=2)
    assert crossed[2:] == schedule.indices_for_positions(start_position=7, count=3)


def test_rollout_workers_use_different_full_dataset_permutations():
    worker_0 = FeistelEpochSchedule(dataset_len=31, seed=1042).indices_for_positions(0, 31)
    worker_1 = FeistelEpochSchedule(dataset_len=31, seed=1043).indices_for_positions(0, 31)

    assert sorted(worker_0) == list(range(31))
    assert sorted(worker_1) == list(range(31))
    assert worker_0 != worker_1


def test_rollout_assignment_records_deterministic_metadata():
    assignment = rollout_assignment(
        worker_index=2,
        lesson_id="math_full",
        worker_seed=1044,
        dataset_len=31,
        start_position=29,
        n_examples=5,
    )

    assert assignment.assignment_id == "worker-2:lesson-math_full:start-29:count-5"
    assert assignment.worker_index == 2
    assert assignment.worker_seed == 1044
    assert assignment.epoch == 0
    assert assignment.start_position == 29
    assert assignment.end_position == 34
    assert len(assignment.indices) == 5
