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

"""Test training batch creation utilities."""

import numpy as np
import pytest

from marin.rl import train_batch
from marin.rl.types import Rollout, RolloutWithAdvantage


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
    )


def test_trim_exact_length_no_change():
    """Test that array of exact length is unchanged."""
    ary = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    result = train_batch.trim_and_pad(ary, max_seq_len=5, pad_token_id=999)

    np.testing.assert_array_equal(result, ary)


def test_float_array_padding():
    """Test padding behavior with float arrays."""
    ary = np.array([1.0, 2.0], dtype=np.float32)
    result = train_batch.trim_and_pad(ary, max_seq_len=4, pad_token_id=999)

    expected = np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float32)  # Float arrays pad with 0
    np.testing.assert_array_equal(result, expected)


def test_basic_conversion():
    """Test basic conversion of rollout to training format."""
    rollout = create_test_rollout(prompt_len=4, response_len=3)
    advantage = 2.5

    result = train_batch.convert_rollout_to_training_format(
        rollout, advantage, max_input_length=8, max_output_length=8, pad_token_id=0
    )

    # Check all expected keys are present
    expected_keys = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "target_ids",
        "loss_weights",
        "loss_masks",
        "policy_logprobs",
    }
    assert set(result.keys()) == expected_keys

    # Check shapes - should be padded to max_input_length + max_output_length
    for _, value in result.items():
        assert len(value) == 16  # 8 + 8


def test_loss_mask_correct():
    """Test that loss mask only covers response tokens."""
    rollout = create_test_rollout(prompt_len=4, response_len=3)
    advantage = 1.0

    result = train_batch.convert_rollout_to_training_format(
        rollout, advantage, max_input_length=8, max_output_length=8, pad_token_id=0
    )

    loss_mask = result["loss_masks"]

    # Loss mask should be 0 for prompt tokens (except first which is shifted out)
    # and 1 for response tokens
    # With prompt_len=4, response_len=3, total=7, shifted gives 6 tokens
    # First 3 tokens (4-1) should be 0, next 3 should be 1, rest should be 0 (padding)
    expected_mask = np.array([0, 0, 0, 1, 1, 1] + [0] * 10, dtype=np.float32)
    np.testing.assert_array_equal(loss_mask, expected_mask)


def test_loss_weights_have_advantage():
    """Test that loss weights contain the advantage value for response tokens."""
    rollout = create_test_rollout(prompt_len=4, response_len=3)
    advantage = 2.5

    result = train_batch.convert_rollout_to_training_format(
        rollout, advantage, max_input_length=8, max_output_length=8, pad_token_id=0
    )

    loss_weights = result["loss_weights"]

    # Should be 0 for prompt tokens, advantage for response tokens, 0 for padding
    expected_weights = np.array([0, 0, 0, 2.5, 2.5, 2.5] + [0] * 10, dtype=np.float32)
    np.testing.assert_array_equal(loss_weights, expected_weights)


def test_token_sequence_shifted_correctly():
    """Test that input and target sequences are shifted correctly for next-token prediction."""
    rollout = create_test_rollout(prompt_len=3, response_len=2)

    result = train_batch.convert_rollout_to_training_format(
        rollout, 1.0, max_input_length=8, max_output_length=8, pad_token_id=0
    )

    # Original tokens: prompt=[12345, 12345, 12345], response=[1000, 1001]
    # Full sequence: [12345, 12345, 12345, 1000, 1001]
    # input_ids should be first 4 tokens: [12345, 12345, 12345, 1000]
    # target_ids should be next 4 tokens: [12345, 12345, 1000, 1001]

    input_ids = result["input_ids"]
    target_ids = result["target_ids"]

    assert input_ids[0] == 12345
    assert input_ids[1] == 12345
    assert input_ids[2] == 12345
    assert input_ids[3] == 1000

    assert target_ids[0] == 12345
    assert target_ids[1] == 12345
    assert target_ids[2] == 1000
    assert target_ids[3] == 1001


def test_empty_rollouts_raises_error():
    """Test that empty rollout list raises ValueError."""
    with pytest.raises(ValueError, match="Cannot create batch from empty rollout list"):
        train_batch.create_training_batch_from_rollouts([], 8, 8, 0)


def test_single_rollout_batch_creation():
    """Test creating batch from single rollout."""
    rollout = create_test_rollout()
    individual = RolloutWithAdvantage(rollout=rollout, advantage=1.5)

    batch = train_batch.create_training_batch_from_rollouts([individual], 8, 8, 0)

    # Should have batch size 1
    assert len(batch) == 1
    assert batch.input_ids.axis_size("batch") == 1

    # Check that advantage was applied correctly
    # Loss weights should have the advantage value for response tokens
    expected_advantage_positions = batch.loss_masks.array[0] == 1.0
    advantage_values = batch.loss_weights.array[0][expected_advantage_positions]
    np.testing.assert_array_equal(advantage_values, np.full(advantage_values.shape, 1.5))


def test_multiple_rollouts_batch_creation():
    """Test creating batch from multiple rollouts with different advantages."""
    individual_rollouts = []
    for i in range(3):
        rollout = create_test_rollout(unique_id=i, episode_reward=float(i))
        advantage = float(i) * 0.5
        individual = RolloutWithAdvantage(rollout=rollout, advantage=advantage)
        individual_rollouts.append(individual)

    batch = train_batch.create_training_batch_from_rollouts(individual_rollouts, 8, 8, 0)

    # Should have batch size 3
    assert len(batch) == 3
    assert batch.input_ids.axis_size("batch") == 3

    # Check that each rollout has its own advantage applied
    for i in range(3):
        expected_advantage = float(i) * 0.5
        advantage_positions = batch.loss_masks.array[i] == 1.0
        advantage_values = batch.loss_weights.array[i][advantage_positions]
        np.testing.assert_array_equal(advantage_values, np.full(advantage_values.shape, expected_advantage))


def test_padding_consistency():
    """Test that padding is applied consistently across different rollout lengths."""
    # Create rollouts of different lengths
    rollouts = [
        create_test_rollout(prompt_len=4, response_len=2, unique_id=1),
        create_test_rollout(prompt_len=6, response_len=4, unique_id=2),
        create_test_rollout(prompt_len=3, response_len=1, unique_id=3),
    ]

    individual_rollouts = [RolloutWithAdvantage(rollout=rollout, advantage=1.0) for rollout in rollouts]

    batch = train_batch.create_training_batch_from_rollouts(individual_rollouts, 8, 8, 999)

    # All sequences should have the same length after padding
    assert batch.input_ids.axis_size("position") == 16  # max_input_length + max_output_length
    assert batch.attention_mask.axis_size("position") == 16
    assert batch.target_ids.axis_size("position") == 16

    # Check that padding tokens are present where expected
    # For the shortest rollout (prompt_len=3, response_len=1, total=4, shifted=3)
    # Positions 3 and beyond should be padded
    assert batch.input_ids.array[2, 4] == 999  # Padding should be present


def test_batch_dtypes():
    """Test that training batch arrays have correct dtypes."""
    rollout = create_test_rollout(prompt_len=4, response_len=3)
    individual = RolloutWithAdvantage(rollout=rollout, advantage=1.5)

    batch = train_batch.create_training_batch_from_rollouts([individual], 8, 8, 0)

    # Token IDs and position IDs should be integers
    assert batch.input_ids.array.dtype == np.int32
    assert batch.target_ids.array.dtype == np.int32
    assert batch.position_ids.array.dtype == np.int32
    assert batch.attention_mask.array.dtype == np.int32

    # Loss weights, masks, and logprobs should be floats
    assert batch.loss_weights.array.dtype == np.float32
    assert batch.loss_masks.array.dtype == np.float32
    assert batch.policy_logprobs.array.dtype == np.float32
