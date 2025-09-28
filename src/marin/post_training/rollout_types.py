# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import equinox as eqx
import haliax as hax
import jax
import numpy as np


class Rollout(eqx.Module):
    env_name: str
    env_example_id: str
    prompt_tokens: jax.Array
    response_tokens: jax.Array
    response_logprobs: jax.Array
    token_rewards: jax.Array
    episode_reward: float


class RolloutGroup(eqx.Module):
    key: str
    rollouts: list[Rollout]

    def compute_rloo_advantages(self) -> np.ndarray:
        """Compute RLOO advantages for this group."""
        rewards = np.array([r.episode_reward for r in self.rollouts])
        n = len(rewards)
        if n <= 1:
            return np.zeros_like(rewards)

        total = rewards.sum()
        leave_one_out_baselines = (total - rewards) / (n - 1)
        advantages = rewards - leave_one_out_baselines

        # Add random noise to avoid failure cases when all rewards are identical
        generator = np.random.default_rng()
        advantages += generator.normal(loc=0.0, scale=1e-6, size=advantages.shape)
        return advantages


@dataclass
class RolloutMetadata:
    worker_id: str
    timestamp: float


class RolloutBatch(eqx.Module):
    groups: list[RolloutGroup]
    metadata: RolloutMetadata


class JaxRolloutBatch(eqx.Module):
    input_ids: jax.Array
    attention_mask: jax.Array
    position_ids: jax.Array
    target_ids: jax.Array
    loss_weights: jax.Array
    loss_masks: jax.Array
    policy_logprobs: jax.Array

    def __len__(self) -> int:
        return len(self.input_ids)

    def as_named(self) -> dict:
        """Convert to dict with NamedArrays for model input."""
        return {
            "input_ids": hax.named(self.input_ids, ("batch", "position")),
            "attention_mask": hax.named(self.attention_mask, ("batch", "position")),
            "position_ids": hax.named(self.position_ids, ("batch", "position")),
            "target_ids": hax.named(self.target_ids, ("batch", "position")),
            "loss_weights": hax.named(self.loss_weights, ("batch", "position")),
            "loss_masks": hax.named(self.loss_masks, ("batch", "position")),
            "policy_logprobs": hax.named(self.policy_logprobs, ("batch", "position")),
        }
