# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from levanter.schedule import BatchSchedule, ScheduleStep, value_at_step


@pytest.fixture
def scheduler():
    """
    A pytest fixture that sets up a BatchScheduler with the following schedule:
      - Use batch size 32 until step 1000
      - Then batch size 64 until step 100000
      - Then batch size 128 forever
    """
    schedule = [
        ScheduleStep(start=0, value=32),
        ScheduleStep(start=1000, value=64),
        ScheduleStep(start=100000, value=128),
    ]
    return BatchSchedule(schedule)


@pytest.mark.parametrize(
    "step, expected_bs, expected_offset, expected_indices",
    [
        (0, 32, 0, (0, 32)),
        (500, 32, 500 * 32, (500 * 32, 500 * 32 + 32)),
        (999, 32, 999 * 32, (999 * 32, 999 * 32 + 32)),
        (1000, 64, 1000 * 32, (1000 * 32, 1000 * 32 + 64)),
        (50000, 64, 32000 + (50000 - 1000) * 64, (32000 + (50000 - 1000) * 64, 32000 + (50000 - 1000) * 64 + 64)),
        (
            100000,
            128,
            32000 + (100000 - 1000) * 64,
            (32000 + (100000 - 1000) * 64, 32000 + (100000 - 1000) * 64 + 128),
        ),
        (
            150000,
            128,
            32000 + (100000 - 1000) * 64 + (150000 - 100000) * 128,
            (
                32000 + (100000 - 1000) * 64 + (150000 - 100000) * 128,
                32000 + (100000 - 1000) * 64 + (150000 - 100000) * 128 + 128,
            ),
        ),
    ],
)
def test_batch_scheduler(scheduler, step, expected_bs, expected_offset, expected_indices):
    """
    Parametric test to ensure the batch scheduler returns the correct
    batch size, data offset, and batch indices for given training steps.
    """
    bs = scheduler.batch_size_at_step(step)
    offset = scheduler.global_data_offset_by_step(step)
    # indices = scheduler.batch_indices_at_step(step)

    assert bs == expected_bs, f"Unexpected batch size at step {step}"
    assert offset == expected_offset, f"Unexpected data offset at step {step}"
    # assert indices == expected_indices, f"Unexpected batch indices at step {step}"


def test_value_at_step_scalar():
    assert value_at_step(42, 0) == 42
    assert value_at_step(42, 1000) == 42


@pytest.mark.parametrize(
    "step, expected",
    [
        (0, 32),
        (500, 32),
        (999, 32),
        (1000, 64),
        (50000, 64),
        (99999, 64),
        (100000, 128),
        (250000, 128),
    ],
)
def test_value_at_step_schedule(step, expected):
    schedule = [
        ScheduleStep(start=0, value=32),
        ScheduleStep(start=1000, value=64),
        ScheduleStep(start=100000, value=128),
    ]
    assert value_at_step(schedule, step) == expected


def test_value_at_step_before_first_segment_raises():
    schedule = [ScheduleStep(start=100, value="a")]
    with pytest.raises(ValueError):
        value_at_step(schedule, 50)
