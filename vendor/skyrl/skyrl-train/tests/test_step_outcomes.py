# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.utils.advantage_estimators import compute_rloo_n_outcome_advantage, expand_outcome_to_steps


def test_long_earlier_turn_receives_outcome_and_neutral_attempt_stays_excluded():
    # Three GS samples: win, loss, infrastructure failure. Their final turns are short.
    final_mask = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]])
    rewards = torch.tensor([[1., 0, 0, 0], [0., 0, 0, 0], [0., 0, 0, 0]])
    advantage, returns = compute_rloo_n_outcome_advantage(
        rewards, final_mask, np.array(["task"] * 3), exclude_from_baseline=np.array([False, False, True]),
        group_advantage_invariant=GroupAdvantageInvariant.minimum_baseline_eligible(physical_group_size=3, minimum_group_size=2)
    )
    # Each attempt also has an earlier four-token action, including malformed reasoning.
    step_mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0],
                              [1, 1, 1, 1], [1, 0, 0, 0]])
    index = torch.tensor([0, 0, 1, 1, 2, 2])
    expected = torch.tensor([[1., 1, 1, 1], [1., 0, 0, 0], [-1., -1, -1, -1], [-1., -1, 0, 0],
                             [0., 0, 0, 0], [0., 0, 0, 0]])
    torch.testing.assert_close(expand_outcome_to_steps(advantage, final_mask, step_mask, index), expected)
    torch.testing.assert_close(expand_outcome_to_steps(returns, final_mask, step_mask, index), expected)
