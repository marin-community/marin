# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.utils.advantage_estimators import compute_advantages_and_returns


def _group_advantages(estimator, rewards, exclusions=None, *, minimum=2):
    invariant = (
        GroupAdvantageInvariant.minimum_baseline_eligible(physical_group_size=len(rewards), minimum_group_size=minimum)
        if estimator == "rloo_n"
        else GroupAdvantageInvariant.exact_physical(physical_group_size=len(rewards))
    )
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    advantages, _ = compute_advantages_and_returns(
        token_level_rewards=rewards,
        response_mask=torch.ones_like(rewards),
        index=np.array(["g"] * len(rewards)),
        adv_estimator=estimator,
        config=OmegaConf.create({}),
        exclude_from_baseline=np.array(exclusions) if exclusions is not None else None,
        group_advantage_invariant=invariant,
    )
    return advantages.squeeze(-1)


@pytest.mark.parametrize(
    "estimator,minimum,expected",
    [
        ("rloo", 2, [1.0, -0.5, -0.5, 0.0]),
        ("rloo_n", 3, [1.0, -0.5, -0.5, 0.0]),
        ("rloo_n", 4, [0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_group_estimators_exclude_masked_rows(estimator, minimum, expected):
    actual = _group_advantages(estimator, [1.0, 0.0, 0.0, 0.0], [False, False, False, True], minimum=minimum)
    torch.testing.assert_close(actual, torch.tensor(expected), rtol=0, atol=1e-6)


def test_group_estimators_zero_nearly_flat_rewards():
    rewards = [0.7] * 6 + [float(np.nextafter(np.float32(0.7), np.float32(1.0)))]
    assert torch.equal(_group_advantages("grpo", rewards), torch.zeros(7))
