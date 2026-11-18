# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils.advantage_estimators import GROUP_STD_EPSILON

NEAR_FLAT = float(np.nextafter(np.float32(0.7), np.float32(1.0)))


def _advantage_trainer(estimator, invariant, *, step_wise=False):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = OmegaConf.create(
        {
            "trainer": {
                "step_wise_training": step_wise,
                "algorithm": {
                    "advantage_estimator": estimator,
                    "gamma": 1.0,
                    "grpo_norm_by_std": True,
                },
            },
        }
    )
    trainer.group_advantage_invariant = invariant
    trainer.all_metrics = {}
    return trainer


def test_grpo_excludes_masked_rows_and_reports_flat_groups():
    trainer = _advantage_trainer("grpo", GroupAdvantageInvariant.exact_physical(physical_group_size=2))
    data = TrainingInputBatch(
        {
            "rewards": torch.tensor([[1.0], [1.0], [0.0], [2.0], [1.0], [0.0], [1.0], [1.0], [0.7], [NEAR_FLAT]]),
            "response_mask": torch.ones(10, 1),
        }
    )
    data.metadata = {
        "uids": ["easy", "easy", "hard", "hard", "masked", "masked", "excluded-flat", "excluded-flat", "near", "near"],
        "exclude_from_baseline": [False, False, False, False, False, True, False, True, False, False],
        "avg_response_length": 1.0,
    }

    result = trainer.compute_advantages_and_returns(data)

    assert trainer.all_metrics["reward/zero_std_group_fraction"] == pytest.approx(2 / 5)
    scale = 1 / (2**0.5 + GROUP_STD_EPSILON)
    torch.testing.assert_close(result["advantages"].flatten(), torch.tensor([0, 0, -scale, scale, 0, 0, 0, 0, 0, 0]))


def test_step_wise_rows_share_their_trajectory_outcome():
    trainer = _advantage_trainer(
        "rloo_n",
        GroupAdvantageInvariant.minimum_baseline_eligible(physical_group_size=3, minimum_group_size=2),
        step_wise=True,
    )
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]])
    rewards = torch.zeros(4, 6)
    rewards[1, 1] = 1.0
    data = TrainingInputBatch(
        {
            "rewards": rewards,
            "response_mask": mask,
            "is_last_step": torch.tensor([False, True, True, True]),
        }
    )
    data.metadata = {
        "uids": ["g"] * 4,
        "exclude_from_baseline": [False, False, False, True],
        "avg_response_length": 3.5,
    }

    result = trainer.compute_advantages_and_returns(data)

    expected = torch.tensor([1.0, 1.0, -1.0, 0.0]).unsqueeze(-1) * mask
    torch.testing.assert_close(result["advantages"], expected, rtol=0, atol=0)


def test_single_step_advantages_match_token_estimation_bitwise():
    estimator, trial_count, response_length = "grpo", 2, 3
    invariant = GroupAdvantageInvariant.exact_physical(physical_group_size=trial_count)
    rewards = torch.zeros(trial_count, response_length)
    rewards[0, -1] = 1.0
    data = TrainingInputBatch(
        {
            "rewards": rewards,
            "response_mask": torch.ones_like(rewards),
            "is_last_step": torch.ones(trial_count, dtype=torch.bool),
        }
    )
    data.metadata = {"uids": ["g"] * trial_count, "avg_response_length": float(response_length)}
    reference = _advantage_trainer(estimator, invariant).compute_advantages_and_returns(copy.deepcopy(data))

    result = _advantage_trainer(estimator, invariant, step_wise=True).compute_advantages_and_returns(data)

    assert torch.equal(result["advantages"], reference["advantages"])
    assert torch.equal(result["returns"], reference["returns"])


@pytest.mark.parametrize(
    "estimator,minimum,expected",
    [("rloo", 2, [1.0, -0.5, -0.5, 0.0]), ("rloo_n", 3, [1.0, -0.5, -0.5, 0.0]), ("rloo_n", 4, [0.0] * 4)],
)
def test_leave_one_out_estimators_use_eligible_rows(estimator, minimum, expected):
    invariant = (
        GroupAdvantageInvariant.minimum_baseline_eligible(physical_group_size=4, minimum_group_size=minimum)
        if estimator == "rloo_n"
        else GroupAdvantageInvariant.exact_physical(physical_group_size=4)
    )
    data = TrainingInputBatch(
        {"rewards": torch.tensor([[1.0], [0.0], [0.0], [0.0]]), "response_mask": torch.ones(4, 1)}
    )
    data.metadata = {
        "uids": ["g"] * 4,
        "exclude_from_baseline": [False, False, False, True],
        "avg_response_length": 1.0,
    }
    result = _advantage_trainer(estimator, invariant).compute_advantages_and_returns(data)
    torch.testing.assert_close(result["advantages"].flatten(), torch.tensor(expected), rtol=0, atol=1e-6)


@pytest.mark.parametrize("reward_token", [0, 2])
def test_step_wise_reinforce_requires_constant_advantages(reward_token):
    trainer = _advantage_trainer(
        "reinforce++", GroupAdvantageInvariant.no_group_advantage(physical_group_size=2), step_wise=True
    )
    trainer.cfg.trainer.algorithm.gamma = 0.9
    rewards = torch.zeros(2, 3)
    rewards[0, reward_token] = 1.0
    data = TrainingInputBatch(
        {"rewards": rewards, "response_mask": torch.ones_like(rewards), "is_last_step": torch.ones(2, dtype=torch.bool)}
    )
    data.metadata = {"uids": ["g", "g"], "avg_response_length": 3.0}
    if reward_token == 0:
        with pytest.raises(ValueError, match="step-wise training requires an outcome advantage estimator"):
            trainer.compute_advantages_and_returns(data)
    else:
        trainer.cfg.trainer.algorithm.gamma = 1.0
        result = trainer.compute_advantages_and_returns(data)
        assert torch.equal(result["advantages"], result["advantages"][:, :1].expand_as(rewards))
