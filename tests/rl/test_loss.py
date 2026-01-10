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

import pytest
import numpy as np
from marin.rl.types import Rollout
from marin.rl.rl_losses import compute_rloo_advantages, compute_ppo_loss_objective


def create_test_rollout(
    prompt_len: int = 8,
    response_len: int = 8,
    env_name: str = "test_env",
    episode_reward: float = 1.0,
    unique_id: int = 12345,
) -> Rollout:
    """Create a test rollout with predictable token values."""
    prompt_tokens = np.full(prompt_len, unique_id, dtype=np.int32)
    response_tokens = np.arange(response_len, dtype=np.int32) + 1000
    response_logprobs = np.full(response_len, -0.5, dtype=np.float32)
    token_rewards = np.full(response_len, 0.1, dtype=np.float32)

    return Rollout(
        env_name=env_name,
        env_example_id=f"example_{unique_id}",
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        response_logprobs=response_logprobs,
        token_rewards=token_rewards,
        episode_reward=episode_reward,
        temperature=1.0,
        top_k=None,
        is_truncated=False,
    )


def test_compute_rloo_advantages():
    rollout_group_rewards = [
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.5, 1.0],
    ]

    rollout_groups = []
    for rewards in rollout_group_rewards:
        rollout_group = []
        for i, reward in enumerate(rewards):
            rollout = create_test_rollout(unique_id=i, episode_reward=reward)
            rollout_group.append(rollout)
        rollout_groups.append(rollout_group)

    expected_advantages = [
        [-0.5, 1.0, -0.5],
        [0.0, 0.0, 0.0],
        [-0.75, 0.0, 0.75],
    ]
    for i, rollout_group in enumerate(rollout_groups):
        advantages = compute_rloo_advantages(rollout_group)
        np.testing.assert_array_equal(advantages, expected_advantages[i])


@pytest.mark.parametrize(
    (
        "importance_sampling_ratio",
        "loss_weights",
        "loss_masks",
        "clip_epsilon",
        "trainer_inference_importance_sampling_ratio",
        "expected_loss",
    ),
    [
        # Simple no padding case
        (
            np.array([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]]),
            np.array([[0.0, 0.0, 0.0, 0.5, 1.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]),
            0.2,
            None,
            -(0.0 + 0.5 + 1.0) / 3.0,
        ),
        # Case with padding and trainer inference importance sampling ratio
        (
            np.array([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
            np.array([[0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]),
            0.2,
            np.array([[0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0]]),
            -(0.5 * 0.5 + 1.0 * 1.0 + 0.0 * 0.0) / 3.0,
        ),
        # Case with negative advantages
        (
            np.array([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
            np.array([[0.0, 0.0, 0.0, -0.5, -1.0, 1.0, 0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]),
            0.2,
            None,
            -(-0.5 - 1.0 + 1.0) / 3.0,
        ),
        # Multi sequence case
        (
            np.array(
                [
                    [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ),
            np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]),
            0.2,
            None,
            -0.0,  # symmetric case should be 0
        ),
    ],
)
def test_ppo_objective(
    importance_sampling_ratio,
    loss_weights,
    loss_masks,
    clip_epsilon,
    trainer_inference_importance_sampling_ratio,
    expected_loss,
):
    loss, _ = compute_ppo_loss_objective(
        importance_sampling_ratio,
        loss_weights,
        loss_masks,
        clip_epsilon_low=clip_epsilon,
        clip_epsilon_high=clip_epsilon,
        max_output_tokens=loss_masks.shape[-1],
        trainer_inference_importance_sampling_ratio=trainer_inference_importance_sampling_ratio,
    )

    np.testing.assert_allclose(loss, expected_loss, atol=1e-6)
