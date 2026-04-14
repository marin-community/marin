# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test training batch creation utilities."""

import numpy as np
import pytest

from marin.rl import train_batch
from marin.rl.types import Rollout, RolloutGroup, RolloutGroupMetadata, RolloutMetadata, RolloutWithAdvantage


def create_test_rollout(
    prompt_len: int = 8,
    response_len: int = 8,
    env_name: str = "test_env",
    episode_reward: float = 1.0,
    unique_id: int = 12345,
    metadata: RolloutMetadata | None = None,
    correctness_reward: float | None = None,
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
        metadata=RolloutMetadata() if metadata is None else metadata,
        correctness_reward=correctness_reward,
    )


def test_trim_exact_length_no_change():
    """Test that array of exact length is unchanged."""
    ary = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    result = train_batch.trim_and_pad(ary, max_seq_len=5, pad_to=5, padding_value=999)

    np.testing.assert_array_equal(result, ary)


def test_float_array_padding():
    """Test padding behavior with float arrays."""
    ary = np.array([1.0, 2.0], dtype=np.float32)
    result = train_batch.trim_and_pad(ary, max_seq_len=4, pad_to=4, padding_value=999)

    expected = np.array([1.0, 2.0, 999.0, 999.0], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_basic_conversion():
    """Test basic conversion of rollout to training format."""
    rollout = create_test_rollout(prompt_len=4, response_len=3)
    advantage = 2.5

    result = train_batch.convert_rollout_to_training_format(rollout, advantage, max_tokens=16, pad_token_id=0, pad_to=16)

    # Check all expected keys are present
    expected_keys = {
        "input_ids",
        "position_ids",
        "loss_weights",
        "loss_masks",
        "policy_logprobs",
        "temperature",
        "top_k",
        "truncated",
    }
    assert set(result.keys()) == expected_keys

    # Check shapes - should be padded to max_tokens
    for _, value in result.items():
        if isinstance(value, np.ndarray):
            assert len(value) == 16


def test_loss_mask_correct():
    """Test that loss mask only covers response tokens."""
    rollout = create_test_rollout(prompt_len=4, response_len=3)
    advantage = 1.0

    result = train_batch.convert_rollout_to_training_format(rollout, advantage, max_tokens=16, pad_token_id=0, pad_to=16)

    loss_mask = result["loss_masks"]

    # Loss mask should be 0 for prompt tokens and 1 for response tokens
    # With prompt_len=4, response_len=3, total=7 tokens
    # First 4 tokens (prompt) should be 0, next 3 (response) should be 1, rest should be 0 (padding)
    expected_mask = np.array([0, 0, 0, 0, 1, 1, 1] + [0] * 9, dtype=np.float32)
    np.testing.assert_array_equal(loss_mask, expected_mask)


def test_loss_weights_have_advantage():
    """Test that loss weights contain the advantage value for response tokens."""
    rollout = create_test_rollout(prompt_len=4, response_len=3)
    advantage = 2.5

    result = train_batch.convert_rollout_to_training_format(rollout, advantage, max_tokens=16, pad_token_id=0, pad_to=16)

    loss_weights = result["loss_weights"]

    # Should be 0 for prompt tokens, advantage for response tokens, 0 for padding
    expected_weights = np.array([0, 0, 0, 0, 2.5, 2.5, 2.5] + [0] * 9, dtype=np.float32)
    np.testing.assert_array_equal(loss_weights, expected_weights)


def test_token_sequence_shifted_correctly():
    """Test that input sequence contains full prompt+response (shifting now happens in rl_losses.py)."""
    rollout = create_test_rollout(prompt_len=3, response_len=2)

    result = train_batch.convert_rollout_to_training_format(rollout, 1.0, max_tokens=16, pad_token_id=0, pad_to=16)

    # Original tokens: prompt=[12345, 12345, 12345], response=[1000, 1001]
    # input_ids should contain the full sequence: [12345, 12345, 12345, 1000, 1001]
    # (shifting for next-token prediction now happens in rl_losses.py)

    input_ids = result["input_ids"]

    assert input_ids[0] == 12345
    assert input_ids[1] == 12345
    assert input_ids[2] == 12345
    assert input_ids[3] == 1000
    assert input_ids[4] == 1001


def test_empty_rollouts_raises_error():
    """Test that empty rollout list raises ValueError."""
    with pytest.raises(ValueError, match="Cannot create batch from empty rollout list"):
        train_batch.create_training_batch_from_rollouts([], max_tokens=16, pad_token_id=0)


def test_single_rollout_batch_creation():
    """Test creating batch from single rollout."""
    rollout = create_test_rollout()
    individual = RolloutWithAdvantage(rollout=rollout, advantage=1.5)

    batch = train_batch.create_training_batch_from_rollouts([individual], max_tokens=16, pad_token_id=0)

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

    batch = train_batch.create_training_batch_from_rollouts(individual_rollouts, max_tokens=16, pad_token_id=0)

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

    batch = train_batch.create_training_batch_from_rollouts(individual_rollouts, max_tokens=16, pad_token_id=999)

    # All sequences should have the same length after padding
    assert batch.input_ids.axis_size("position") == 10  # max_tokens (dynamic padding to max in batch)

    # Check that padding tokens are present where expected
    # For the shortest rollout (prompt_len=3, response_len=1, total=4)
    # Positions 4 and beyond should be padded
    assert batch.input_ids.array[2, 4] == 999  # Padding should be present


def test_batch_dtypes():
    """Test that training batch arrays have correct dtypes."""
    rollout = create_test_rollout(prompt_len=4, response_len=3)
    individual = RolloutWithAdvantage(rollout=rollout, advantage=1.5)

    batch = train_batch.create_training_batch_from_rollouts([individual], max_tokens=16, pad_token_id=0)

    # Token IDs and position IDs should be integers
    assert batch.input_ids.array.dtype == np.int32
    assert batch.position_ids.array.dtype == np.int32

    # Loss weights, masks, and logprobs should be floats
    assert batch.loss_weights.array.dtype == np.float32
    assert batch.loss_masks.array.dtype == np.float32
    assert batch.policy_logprobs.array.dtype == np.float32


def test_rollout_to_trajectory_record_preserves_trace_and_verifier_metadata():
    metadata = RolloutMetadata(
        worker_id="worker-7",
        timestamp=12.5,
        weight_step=42,
        run_id="run-1",
        lesson_id="lesson-a",
        group_id="group-3",
        trace_id="trace-9",
        task_name="math",
        task_version="v2",
        verifier_name="grader",
        verifier_version="2026-04-14",
        trace_ref="gs://trace.json",
    )
    rollout = create_test_rollout(metadata=metadata, correctness_reward=0.75)

    record = train_batch.rollout_to_trajectory_record(rollout)

    assert record.trace_id == "trace-9"
    assert record.group_id == "group-3"
    assert record.lesson_id == "lesson-a"
    assert record.task_name == "math"
    assert record.task_version == "v2"
    assert record.verifier_name == "grader"
    assert record.verifier_version == "2026-04-14"
    assert record.trace_ref == "gs://trace.json"
    assert record.rollout_metadata.run_id == "run-1"
    assert record.correctness_reward == 0.75


def test_create_sequence_batch_from_rollouts_produces_neutral_masks_and_info():
    rollout_a = create_test_rollout(
        prompt_len=3,
        response_len=2,
        unique_id=10,
        episode_reward=1.5,
        metadata=RolloutMetadata(
            worker_id="worker-a",
            timestamp=5.0,
            weight_step=11,
            run_id="run-a",
            lesson_id="lesson-1",
            group_id="group-1",
            trace_id="trace-1",
            task_name="task-a",
            task_version="v1",
            verifier_name="verifier-a",
            verifier_version="1.0",
            trace_ref="trace://1",
        ),
        correctness_reward=0.25,
    )
    rollout_b = create_test_rollout(
        prompt_len=2,
        response_len=4,
        unique_id=20,
        episode_reward=2.5,
        metadata=RolloutMetadata(
            worker_id="worker-b",
            timestamp=7.5,
            weight_step=13,
            run_id="run-b",
            lesson_id="lesson-2",
            group_id="group-2",
            trace_id="trace-2",
            task_name="task-b",
            task_version="v3",
        ),
    )

    batch, info = train_batch.create_sequence_batch_from_rollouts([rollout_a, rollout_b], max_tokens=16, pad_token_id=0)

    assert len(batch) == 2
    assert batch.input_ids.axis_size("position") == 6
    np.testing.assert_array_equal(batch.prompt_mask.array[0], np.array([1, 1, 1, 0, 0, 0], dtype=np.float32))
    np.testing.assert_array_equal(batch.response_mask.array[0], np.array([0, 0, 0, 1, 1, 0], dtype=np.float32))
    np.testing.assert_array_equal(
        info.token_rewards.array[0],
        np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(info.prompt_length.array, np.array([3, 2], dtype=np.int32))
    np.testing.assert_array_equal(info.response_length.array, np.array([2, 4], dtype=np.int32))
    np.testing.assert_allclose(info.episode_reward.array, np.array([1.5, 2.5], dtype=np.float32))
    np.testing.assert_allclose(info.correctness_reward.array[0], 0.25)
    assert np.isnan(info.correctness_reward.array[1])
    assert info.group_id == ("group-1", "group-2")
    assert info.lesson_id == ("lesson-1", "lesson-2")
    assert info.task_name == ("task-a", "task-b")
    assert info.verifier_name == ("verifier-a", None)
    assert info.trace_id == ("trace-1", "trace-2")
    assert info.trace_ref == ("trace://1", None)
    assert info.timestamp == (5.0, 7.5)


def test_sequence_batch_lengths_reflect_truncation():
    rollout = create_test_rollout(prompt_len=5, response_len=4, unique_id=33)

    batch, info = train_batch.create_sequence_batch_from_rollouts([rollout], max_tokens=7, pad_token_id=0)

    np.testing.assert_array_equal(batch.prompt_mask.array[0], np.array([1, 1, 1, 1, 1, 0, 0], dtype=np.float32))
    np.testing.assert_array_equal(batch.response_mask.array[0], np.array([0, 0, 0, 0, 0, 1, 1], dtype=np.float32))
    np.testing.assert_array_equal(info.prompt_length.array, np.array([5], dtype=np.int32))
    np.testing.assert_array_equal(info.response_length.array, np.array([2], dtype=np.int32))


def test_batch_info_preserves_timestamp_precision_at_unix_scale():
    base_timestamp = 1_710_000_000.0
    rollout_a = create_test_rollout(
        unique_id=41,
        metadata=RolloutMetadata(timestamp=base_timestamp, weight_step=1),
    )
    rollout_b = create_test_rollout(
        unique_id=42,
        metadata=RolloutMetadata(timestamp=base_timestamp + 1.0, weight_step=2),
    )

    _, info = train_batch.create_sequence_batch_from_rollouts([rollout_a, rollout_b], max_tokens=16, pad_token_id=0)

    assert info.timestamp == (base_timestamp, base_timestamp + 1.0)


def test_rollout_group_to_trajectory_group_record_uses_group_metadata():
    metadata = RolloutMetadata(
        lesson_id="lesson-shared",
        group_id="group-shared",
        trace_id="trace-shared",
        task_name="task-shared",
        task_version="v9",
        verifier_name="verifier-shared",
        verifier_version="v1",
        trace_ref="trace://shared",
    )
    rollout_a = create_test_rollout(prompt_len=3, response_len=2, unique_id=88, metadata=metadata)
    rollout_b = create_test_rollout(prompt_len=3, response_len=2, unique_id=88, metadata=metadata)
    group = RolloutGroup(
        rollouts=[rollout_a, rollout_b],
        metadata=RolloutGroupMetadata(
            group_id="group-override",
            lesson_id="lesson-override",
            trace_id="trace-override",
            task_name="task-override",
            task_version="v10",
            verifier_name="verifier-override",
            verifier_version="v2",
            trace_ref="trace://override",
        ),
    )

    record = train_batch.rollout_group_to_trajectory_group_record(group)

    assert record.group_id == "group-override"
    assert record.lesson_id == "lesson-override"
    assert record.trace_id == "trace-override"
    assert record.task_name == "task-override"
    assert record.task_version == "v10"
    assert record.verifier_name == "verifier-override"
    assert record.verifier_version == "v2"
    assert record.trace_ref == "trace://override"
    assert len(record.trajectories) == 2


def test_rollout_group_to_trajectory_group_record_rejects_mismatched_prompts():
    rollout_a = create_test_rollout(prompt_len=3, response_len=2, unique_id=1)
    rollout_b = create_test_rollout(prompt_len=3, response_len=2, unique_id=2)
    group = RolloutGroup(rollouts=[rollout_a, rollout_b], metadata=RolloutGroupMetadata())

    with pytest.raises(ValueError, match="share the same prompt tokens"):
        train_batch.rollout_group_to_trajectory_group_record(group)
