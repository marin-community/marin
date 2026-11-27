# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from skyrl_train.dynamic_sampling import DynamicSamplingType, GroupSelectionPolicy, GroupSelectionResult
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.rollouts.buffer import RolloutContentPolicy, RolloutGroup


def test_verdict_samples_eligible_final_outcomes_and_preserves_per_row_rewards():
    batch = {
        "prompt_token_ids": [[1], [1, 2], [1]],
        "response_ids": [[2], [3], [4]],
        "loss_masks": [[1], [1], [1]],
        "rewards": [0.5, 1.0, 0.0],
        "unshaped_rewards": [0.25, -0.25, 0.0],
        "is_last_step": [False, True, True],
        "exclude_from_baseline": [False, False, True],
    }
    policy = RolloutContentPolicy(
        GroupAdmissionPolicy(
            GroupAdvantageInvariant.exact_physical(physical_group_size=2), rollout_logprobs_required=False
        ),
        GroupSelectionPolicy(DynamicSamplingType.FILTER),
    )
    verdict = policy.verdict(RolloutGroup(batch, "one-eligible", 1, {}))
    assert verdict.row_facts.rewards.optimization == (0.5, 1.0, 0.0)
    assert verdict.selection is GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD
    assert verdict.rewards.optimization == (1.0, 0.0)
    assert verdict.rewards.outcome == (-0.25, 0.0)
    assert not verdict.rewards.passed
