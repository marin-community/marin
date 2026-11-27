"""Stall incidence counts consumed trajectories, including multi-turn episodes."""

import pytest

from skyrl_train.batch_metrics import rollout_metrics
from skyrl_train.rollouts.buffer import RowFacts


@pytest.mark.parametrize("terminal_reason,expected", [("repeat_stall", 0.5), ("stop", 0.0)])
def test_stall_fraction_counts_episodes_once_and_records_zero(terminal_reason, expected):
    batch = {
        "prompt_token_ids": [[1]] * 4,
        "response_ids": [[2]] * 4,
        "loss_masks": [[1]] * 4,
        "rewards": [0.0, 0.0, 0.0, 1.0],
        "is_last_step": [False, False, True, True],
        "stop_reasons": ["stop", "stop", terminal_reason, "stop"],
    }
    metrics = rollout_metrics([RowFacts.from_batch(batch)])
    assert metrics["generate/repeat_stall_fired_fraction"] == expected
    assert metrics["generate/repeat_stall_fired_count"] == 2 * expected
    assert metrics["generate/repeat_stall_stop_reason_coverage"] == 1.0
