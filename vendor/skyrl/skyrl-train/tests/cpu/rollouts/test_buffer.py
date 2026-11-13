# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import pytest
from skyrl_train.dynamic_sampling import DynamicSamplingType, GroupSelectionPolicy
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.rollouts.buffer import (
    BatchPolicy,
    PayloadReference,
    RolloutBuffer,
    RolloutBufferConfig,
    RolloutContentPolicy,
    RolloutGroup,
)
from skyrl_train.rollouts.loader import JudgedGroup


PROGRESS_TIMEOUT = 5.0


def rollout_group_batch(rewards, *, exclude_from_baseline=None, steps_per_trial=None):
    steps_per_trial = steps_per_trial or [1] * len(rewards)
    batch = {key: [] for key in ("response_ids", "prompt_token_ids", "loss_masks", "rewards", "is_last_step")}
    for steps, reward in zip(steps_per_trial, rewards, strict=True):
        for step in range(steps):
            final = step == steps - 1
            batch["response_ids"].append([2])
            batch["prompt_token_ids"].append([1])
            batch["loss_masks"].append([1])
            batch["rewards"].append(reward if final else 0.5)
            batch["is_last_step"].append(final)
    batch.update(
        unshaped_rewards=list(batch["rewards"]),
        stop_reasons=["stop"] * len(batch["rewards"]),
        rollout_metrics={},
        exclude_from_baseline=exclude_from_baseline,
    )
    return batch


@pytest.mark.asyncio
async def test_dynamic_sampling_judges_step_wise_groups_by_their_final_rows():
    content = RolloutContentPolicy(
        GroupAdmissionPolicy(
            GroupAdvantageInvariant.exact_physical(physical_group_size=2), rollout_logprobs_required=False
        ),
        GroupSelectionPolicy(DynamicSamplingType.FILTER),
    )
    buffer = RolloutBuffer(RolloutBufferConfig(2, 4, 1, BatchPolicy.FULL_BATCH, DynamicSamplingType.FILTER, None))
    await buffer.publish(1)
    for group in [
        RolloutGroup(rollout_group_batch([1.0, 0.0], exclude_from_baseline=[False, True]), "one-eligible", 1, {}),
        RolloutGroup(rollout_group_batch([1.0, 0.0], steps_per_trial=[2, 1]), "three-rows", 1, {}),
        RolloutGroup(rollout_group_batch([0.0, 1.0], steps_per_trial=[3, 2]), "five-rows", 1, {}),
    ]:
        lease = await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
        await buffer.commit(lease.lease_id, {}, content.verdict(group), PayloadReference(group.uid))
    while (admission := await buffer.admit(PROGRESS_TIMEOUT)).selection is None:
        pass
    assert admission.selection.judged == [
        JudgedGroup("one-eligible", (1.0, 0.0)),
        JudgedGroup("three-rows", (1.0, 0.0)),
        JudgedGroup("five-rows", (0.0, 1.0)),
    ]
    assert admission.selection.metrics["async/dynamic_sampling/discarded_count"] == 1
    assert admission.selection.metrics["async/dynamic_sampling/candidate_pass_at_2"] == 1.0
    assert admission.selection.metrics["async/dynamic_sampling/candidate_outcome_reward_mean"] == 0.5
