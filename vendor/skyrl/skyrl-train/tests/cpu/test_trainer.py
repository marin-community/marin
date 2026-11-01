# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import pytest
import torch
from omegaconf import OmegaConf
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils.advantage_estimators import GROUP_STD_EPSILON


def _advantage_trainer(estimator, invariant, *, step_wise=False):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = OmegaConf.create(
        {
            "trainer": {
                "step_wise_training": step_wise,
                "algorithm": {
                    "advantage_estimator": estimator,
                    "gamma": 1.0,
                    "lambd": 1.0,
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
            "rewards": torch.tensor([[1.0], [1.0], [0.0], [2.0], [1.0], [0.0], [1.0], [1.0]]),
            "response_mask": torch.ones(8, 1),
            "values": None,
        }
    )
    data.metadata = {
        "uids": ["easy", "easy", "hard", "hard", "masked", "masked", "excluded-flat", "excluded-flat"],
        "exclude_from_baseline": [False, False, False, False, False, True, False, True],
        "avg_response_length": 1.0,
    }

    result = trainer.compute_advantages_and_returns(data)

    assert trainer.all_metrics["reward/zero_std_group_fraction"] == pytest.approx(1 / 4)
    scale = 1 / (2**0.5 + GROUP_STD_EPSILON)
    torch.testing.assert_close(result["advantages"].flatten(), torch.tensor([0, 0, -scale, scale, 0, 0, 0, 0]))


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
            "values": None,
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
            "values": None,
            "is_last_step": torch.ones(trial_count, dtype=torch.bool),
        }
    )
    data.metadata = {"uids": ["g"] * trial_count, "avg_response_length": float(response_length)}
    reference = _advantage_trainer(estimator, invariant).compute_advantages_and_returns(copy.deepcopy(data))

    result = _advantage_trainer(estimator, invariant, step_wise=True).compute_advantages_and_returns(data)

    assert torch.equal(result["advantages"], reference["advantages"])
    assert torch.equal(result["returns"], reference["returns"])
