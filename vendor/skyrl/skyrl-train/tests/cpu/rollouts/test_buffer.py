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


@pytest.mark.asyncio
async def test_dynamic_sampling_judges_step_wise_groups_by_their_final_rows(rollout_group_batch_factory):
    content = RolloutContentPolicy(
        GroupAdmissionPolicy(
            GroupAdvantageInvariant.exact_physical(physical_group_size=2), rollout_logprobs_required=False
        ),
        GroupSelectionPolicy(DynamicSamplingType.FILTER),
    )
    buffer = RolloutBuffer(RolloutBufferConfig(2, 4, 1, BatchPolicy.FULL_BATCH, DynamicSamplingType.FILTER, None))
    await buffer.publish(1)
    for group in [
        RolloutGroup(rollout_group_batch_factory([1.0, 0.0], steps_per_trial=[2, 1]), "three-rows", 1, {}),
        RolloutGroup(rollout_group_batch_factory([0.0, 1.0], steps_per_trial=[3, 2]), "five-rows", 1, {}),
    ]:
        lease = await asyncio.wait_for(buffer.acquire_lease(), PROGRESS_TIMEOUT)
        await buffer.commit(lease.lease_id, {}, content.verdict(group), PayloadReference(group.uid))
    while (admission := await buffer.admit(PROGRESS_TIMEOUT)).selection is None:
        pass
    assert admission.selection.judged == [JudgedGroup("three-rows", (1.0, 0.0)), JudgedGroup("five-rows", (0.0, 1.0))]
    assert admission.selection.metrics["async/dynamic_sampling/candidate_pass_at_2"] == 1.0
