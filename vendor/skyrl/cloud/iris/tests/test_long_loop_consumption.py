# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual CPU rollout buffer with the production recipe's quotas."""

from collections import Counter
from pathlib import Path

import pytest
import yaml

from skyrl_train.dynamic_sampling import GroupSelectionResult
from skyrl_train.rollouts.buffer import (
    BatchPolicy,
    GroupRewards,
    PayloadReference,
    RolloutBuffer,
    RolloutBufferConfig,
    RolloutVerdict,
)
from skyrl_train.telemetry import GeneratedWork


@pytest.mark.asyncio
async def test_production_gs8_batch_waits_for_rejected_agentic_replacement():
    recipe = yaml.safe_load((Path(__file__).parents[5] / "experiments/post_training/long_loop/specs/rl.yaml").read_text())
    trainer = recipe["config"]["trainer"]
    config = trainer["rollout_buffer"]
    group_size = recipe["role_plan"]["n_samples_per_prompt"]
    buffer = RolloutBuffer(
        RolloutBufferConfig(
            batch_size=trainer["train_batch_size"],
            max_in_flight=trainer["train_batch_size"],
            max_staleness_steps=config["max_staleness_steps"],
            batch_policy=BatchPolicy(config["batch_policy"]),
            dynamic_sampling=None,
            max_candidate_groups=None,
            consumed_route_quotas=config["consumed_route_quotas"],
        )
    )
    await buffer.publish(1)
    leases = [await buffer.acquire_lease() for _ in range(trainer["train_batch_size"])]
    assert Counter(lease.route for lease in leases) == {"agentic": 32, "nonagentic": 32}
    rejected = next(lease for lease in leases if lease.route == "agentic")

    async def commit(lease, selection):
        rewards = tuple(float(index % 2) for index in range(group_size))
        await buffer.commit(
            lease.lease_id,
            {"uid": lease.lease_id, "group_route": lease.route},
            RolloutVerdict(
                lease.lease_id,
                (),
                selection,
                GroupRewards(rewards, rewards, True),
                GeneratedWork(group_size, group_size, group_size * 32),
            ),
            PayloadReference(lease.lease_id),
        )

    # All fast nonagentic work finishes first; the final agentic result is rejected.
    ordered = sorted(leases, key=lambda lease: lease.route == "agentic")
    for lease in ordered:
        await commit(
            lease,
            GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD if lease == rejected else GroupSelectionResult.KEEP,
        )
    incomplete = await buffer.admit(1)
    assert incomplete.selection is None
    replacement = await buffer.acquire_lease()
    assert replacement.route == "agentic"
    await commit(replacement, GroupSelectionResult.KEEP)
    completed = await buffer.admit(1)
    assert completed.selection.metrics["async/consumed_route/agentic"] == 32
    assert completed.selection.metrics["async/consumed_route/nonagentic"] == 32
