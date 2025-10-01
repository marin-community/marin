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

"""Test rollout storage and replay buffer functionality."""

import time

import numpy as np
import pytest

try:
    from marin.rl import train_batch
    from marin.rl.replay_buffer import ReplayBuffer, ReplayBufferConfig, ReplayDataLoader
except ImportError:
    pytest.skip("Post training imports unavailable", allow_module_level=True)
from marin.rl.rollout_storage import (
    InMemoryRolloutQueue,
)
from marin.rl.types import (
    Rollout,
    RolloutBatch,
    RolloutGroup,
    RolloutMetadata,
)


def rollouts_to_training_batch(rollouts, max_input_length=16, max_output_length=16, pad_token_id=0):
    """Helper function to convert rollouts to training batch for testing."""
    return train_batch.create_training_batch_from_rollouts(rollouts, max_input_length, max_output_length, pad_token_id)


def create_test_batch(idx: int, batch_size: int = 2, max_seq_len: int = 16, env_name: str | None = None) -> RolloutBatch:
    """Helper to create test batches with identifiable tokens for testing."""
    rng = np.random.default_rng(42 + idx)
    if env_name is None:
        env_name = f"test_env_{idx}"

    # Create individual rollouts with identifiable tokens
    rollouts = []
    for i in range(batch_size):
        # Split sequence into prompt and response
        prompt_len = max_seq_len // 2
        response_len = max_seq_len - prompt_len

        # Create identifiable tokens - first token identifies the rollout
        unique_id = idx * 1000 + i
        prompt_tokens = np.full(prompt_len, unique_id, dtype=np.int32)
        prompt_tokens[1:] = rng.integers(0, 1000, size=prompt_len - 1, dtype=np.int32)

        response_tokens = rng.integers(0, 1000, size=response_len, dtype=np.int32)
        response_logprobs = rng.standard_normal(response_len).astype(np.float32)
        token_rewards = rng.standard_normal(response_len).astype(np.float32)
        episode_reward = float(rng.standard_normal())

        rollout = Rollout(
            env_name=env_name,
            env_example_id=f"example_{idx}_{i}",
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            response_logprobs=response_logprobs,
            token_rewards=token_rewards,
            episode_reward=episode_reward,
        )
        rollouts.append(rollout)

    # Group rollouts
    group = RolloutGroup(rollouts=rollouts)

    # Create batch
    metadata = RolloutMetadata(worker_id="test_worker", timestamp=time.time(), weight_step=0)
    return RolloutBatch(groups=[group], metadata=metadata)


def test_replay_buffer():
    """Test replay buffer functionality."""
    queue = InMemoryRolloutQueue()
    writer = queue.writer()
    reader = queue.reader()

    # Write batches with different environments
    for env in ["env1", "env2"]:
        for i in range(5):
            batch = create_test_batch(i, batch_size=2, max_seq_len=16, env_name=env)
            writer.write_batch(batch)

    replay_buffer = ReplayBuffer(
        config=ReplayBufferConfig(capacity=100, alpha=3.0, max_samples=-1),
        local_batch_size=4,
        process_id=0,
        total_processes=1,
    )

    # Add batches to replay buffer
    batches = reader.read_all_available()
    replay_buffer.add_batches(batches)

    # Test sampling
    sampled_rollouts = replay_buffer.sample_rollouts()
    assert sampled_rollouts is not None
    assert len(sampled_rollouts) == 4

    # Convert to training batch to verify format
    training_batch = rollouts_to_training_batch(sampled_rollouts)
    assert training_batch is not None
    assert training_batch.input_ids.shape == {"batch": 4, "position": 32}

    # Test data loader
    data_loader = ReplayDataLoader(rollout_reader=reader, replay_buffer=replay_buffer, rollout_fetch_interval=0.1)

    # Write more batches for data loader to collect
    for i in range(5, 10):
        batch = create_test_batch(i, batch_size=2, max_seq_len=16, env_name="env1")
        writer.write_batch(batch)

    with data_loader:
        import time

        time.sleep(0.2)  # Let it collect some batches
        rollouts = data_loader.get_rollouts(timeout=1.0)
        assert rollouts is not None


def test_replay_buffer_recency_bias():
    """Test that high recency bias strongly favors recent (higher index) samples."""
    # Create a batch with 100 rollouts across multiple groups for better testing
    batches = []
    rollouts_per_batch = 10
    for batch_idx in range(10):  # 10 batches * 10 rollouts each = 100 total
        batch = create_test_batch(batch_idx, batch_size=rollouts_per_batch, max_seq_len=16, env_name="test_env")
        batches.append(batch)

    replay_buffer = ReplayBuffer(
        config=ReplayBufferConfig(capacity=100, alpha=10.0, max_samples=-1),  # Very strong recency bias
        local_batch_size=50,
        process_id=0,
        total_processes=1,
    )
    replay_buffer.add_batches(batches)

    # Sample many times to see distribution
    all_samples = []
    for _ in range(100):
        sample = replay_buffer.sample_rollouts()
        if sample is not None:
            # Extract the identifiable tokens from the first prompt token of each rollout
            batch_indices = [int(rollout.rollout.prompt_tokens[0]) // 1000 for rollout in sample]
            all_samples.extend(batch_indices)

    # With strong recency bias, we should heavily favor high batch indices (later batches)
    # Split indices into low (0-4) and high (5-9) batch ranges
    low_indices = [idx for idx in all_samples if idx < 5]
    high_indices = [idx for idx in all_samples if idx >= 5]

    # High indices should be much more frequent than low indices
    assert len(high_indices) > len(
        low_indices
    ), f"Expected high indices to be more frequent, got {len(high_indices)} high vs {len(low_indices)} low"


def test_replay_buffer_capacity_eviction():
    """Test that replay buffer respects capacity limits and evicts old data."""
    replay_buffer = ReplayBuffer(
        config=ReplayBufferConfig(capacity=3, alpha=2.0, max_samples=-1),
        local_batch_size=4,
        process_id=0,
        total_processes=1,
    )

    for i in range(5):
        batch = create_test_batch(i, batch_size=2, max_seq_len=16, env_name="test_env")
        replay_buffer.add_batches([batch])

    stats = replay_buffer.get_stats()
    assert stats["total_size"] == 3  # Should be capped at capacity
    assert stats["env_sizes"]["test_env"] == 3
    assert stats["total_batches_added"] == 5

    # Verify most recent data is kept (batches 2, 3, 4 should remain)
    # This is hard to test directly without exposing internals, but capacity enforcement is verified


def test_replay_buffer_max_resamples():
    """Test that examples are retired after max_resamples uses."""
    replay_buffer = ReplayBuffer(
        config=ReplayBufferConfig(capacity=100, alpha=1.0, max_samples=3),  # Uniform sampling for predictable behavior
        local_batch_size=2,
        process_id=0,
        total_processes=1,
    )

    # Create batch with identifiable examples (already has identifiable tokens)
    batch = create_test_batch(0, batch_size=4, max_seq_len=16, env_name="test_env")

    replay_buffer.add_batches([batch])

    # Initial size should be 4
    assert replay_buffer.size() == 4

    # Sample multiple times - with batch size 2, we expect examples to be used multiple times
    samples_taken = 0
    max_iterations = 20

    for _ in range(max_iterations):
        sample = replay_buffer.sample_rollouts()
        if sample is None:
            break
        samples_taken += 1

    # Should have retired all examples due to max_resamples
    final_size = replay_buffer.size()
    assert final_size == 0, f"Expected buffer to shrink due to max_resamples, but size remained {final_size}"
    assert samples_taken > 4, "Should have been able to sample multiple times before retirement"


def test_replay_buffer_max_resamples_disabled():
    """Test that max_resamples=-1 disables retirement."""
    replay_buffer = ReplayBuffer(
        config=ReplayBufferConfig(capacity=100, alpha=1.0, max_samples=-1),  # Disabled
        local_batch_size=2,
        process_id=0,
        total_processes=1,
    )

    # Add small batch
    batch = create_test_batch(0, batch_size=3, max_seq_len=16, env_name="test_env")
    replay_buffer.add_batches([batch])

    initial_size = replay_buffer.size()
    assert initial_size == 3

    # Sample many times - examples should never be retired
    for _ in range(50):
        sample = replay_buffer.sample_rollouts()
        assert sample is not None
        assert len(sample) == 2  # batch_size

        # Size should remain constant
        current_size = replay_buffer.size()
        assert (
            current_size == initial_size
        ), f"Buffer size changed from {initial_size} to {current_size} with disabled max_resamples"


def test_replay_buffer_max_resamples_multiple_envs():
    """Test max_resamples with multiple environments."""
    replay_buffer = ReplayBuffer(
        config=ReplayBufferConfig(capacity=100, alpha=1.0, max_samples=2),
        local_batch_size=3,
        process_id=0,
        total_processes=1,
    )

    # Add batches from different environments (identifiable tokens already set)
    for env_id in range(2):
        env_name = f"env_{env_id}"
        batch = create_test_batch(env_id, batch_size=3, max_seq_len=16, env_name=env_name)
        replay_buffer.add_batches([batch])

    initial_stats = replay_buffer.get_stats()
    assert initial_stats["total_size"] == 6
    assert initial_stats["num_environments"] == 2

    # Sample multiple times
    for _ in range(15):
        sample = replay_buffer.sample_rollouts()
        if sample is None:
            break

    # Both environments should still exist but may have fewer examples
    final_stats = replay_buffer.get_stats()
    assert final_stats["num_environments"] == 2
    assert final_stats["total_size"] < 6, "Expected some examples to be retired"

    # Each environment should have at least some examples remaining
    for env_name in ["env_0", "env_1"]:
        assert env_name in final_stats["env_sizes"]
        # Due to balanced sampling, both environments should have some data


def test_replay_buffer_weight_step_filtering():
    replay_buffer = ReplayBuffer(
        config=ReplayBufferConfig(capacity=100, alpha=2.0, max_samples=-1, max_rollout_delay=30),
        local_batch_size=4,
        process_id=0,
        total_processes=1,
    )

    # Create batches with weight_steps 50, 100, 150
    for weight_step in [50, 100, 150]:
        batch = create_test_batch(weight_step, batch_size=4, max_seq_len=16, env_name="test_env")
        batch = RolloutBatch(
            groups=batch.groups,
            metadata=RolloutMetadata(worker_id="test", timestamp=time.time(), weight_step=weight_step),
        )
        replay_buffer.add_batches([batch])

    assert replay_buffer.size() == 12

    # Set current step to 100, min_step=70, should filter out weight_step=50
    replay_buffer.set_current_step(100)
    assert replay_buffer.size() == 8

    # Set current step to 150, min_step=120, should filter out 50 and 100
    replay_buffer.set_current_step(150)
    assert replay_buffer.size() == 4

    # Add new batch with weight_step=180
    new_batch = create_test_batch(180, batch_size=3, max_seq_len=16, env_name="test_env")
    new_batch = RolloutBatch(
        groups=new_batch.groups, metadata=RolloutMetadata(worker_id="test", timestamp=time.time(), weight_step=180)
    )
    replay_buffer.add_batches([new_batch])

    # Should now have 7 total (4 from weight_step=150 + 3 from weight_step=180)
    assert replay_buffer.size() == 7

    # Set current step to 190, min_step=160, should filter out weight_step=150
    replay_buffer.set_current_step(190)
    assert replay_buffer.size() == 3
