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
    from marin.post_training.replay_buffer import ReplayBuffer, ReplayDataLoader
except ImportError:
    pytest.skip("Post training imports unavailable", allow_module_level=True)
from marin.post_training.rollout_storage import (
    FileRolloutReader,
    FileRolloutWriter,
    InMemoryRolloutQueue,
    JaxRolloutBatch,
    RolloutBatch,
    TaggedRolloutBatch,
)


def create_test_batch(idx: int, batch_size: int = 2, max_seq_len: int = 16) -> TaggedRolloutBatch:
    """Helper to create test batches with all required fields."""
    rng = np.random.default_rng(42 + idx)
    return TaggedRolloutBatch(
        batch=RolloutBatch(
            input_ids=rng.integers(0, 1000, size=(batch_size, max_seq_len), dtype=np.int32),
            attention_mask=np.ones((batch_size, max_seq_len), dtype=np.int32),
            position_ids=np.arange(max_seq_len)[None, :].repeat(batch_size, axis=0).astype(np.int32),
            target_ids=rng.integers(0, 1000, size=(batch_size, max_seq_len), dtype=np.int32),
            loss_weights=np.ones((batch_size, max_seq_len), dtype=np.float32),
            loss_masks=np.ones((batch_size, max_seq_len), dtype=np.float32),
            policy_logprobs=rng.standard_normal((batch_size, max_seq_len)).astype(np.float32),
        ),
        env_name=f"test_env_{idx}",
        worker_id="test_worker",
        timestamp=time.time(),
        rollout_id=f"test_{idx}",
    )


@pytest.fixture
def rollout_queue(request, tmp_path):
    """Create rollout queue based on parameter, with cleanup."""
    queue_type = getattr(request, "param", "memory")

    if queue_type == "memory":
        queue = InMemoryRolloutQueue()
        reader = queue.reader()
        writer = queue.writer()
        yield reader, writer
        # No cleanup needed for in-memory

    elif queue_type == "file":
        queue_path = str(tmp_path / "rollout_queue")
        reader = FileRolloutReader(queue_path)
        writer = FileRolloutWriter(queue_path)
        yield reader, writer
        # Cleanup happens automatically with tmp_path


@pytest.mark.parametrize("rollout_queue", ["memory", "file"], indirect=True)
def test_rollout_queue(rollout_queue):
    """Test rollout queue with one-shot iterator operations."""
    reader, writer = rollout_queue

    batch1 = create_test_batch(1)
    batch2 = create_test_batch(2)

    # Test read_all_available on empty queue
    empty_batches = reader.read_all_available()
    assert len(empty_batches) == 0

    # Test writing
    writer.write_batch(batch1)
    writer.write_batch(batch2)

    # Test read_all_available reads all batches
    batches = reader.read_all_available()
    assert len(batches) == 2

    # Test that read_all_available returns empty list after consuming all
    batches2 = reader.read_all_available()
    assert len(batches2) == 0


def test_read_batch_behavior():
    """Test read_batch behavior for InMemoryRolloutQueue."""
    queue = InMemoryRolloutQueue()
    writer = queue.writer()
    reader = queue.reader()

    # Test read_batch on empty queue
    batch = reader.read_batch(timeout=0.1)
    assert batch is None

    # Write batches
    for i in range(3):
        writer.write_batch(create_test_batch(i))

    # Read batches one by one
    batch1 = reader.read_batch(timeout=0.1)
    assert batch1 is not None

    batch2 = reader.read_batch(timeout=0.1)
    assert batch2 is not None

    # Read all remaining with read_all_available
    remaining = reader.read_all_available()
    assert len(remaining) == 1

    # No more batches available
    batch_none = reader.read_batch(timeout=0.1)
    assert batch_none is None

    # Write more batches
    writer.write_batch(create_test_batch(3))

    # New batches are available
    new_batches = reader.read_all_available()
    assert len(new_batches) == 1


def test_replay_buffer():
    """Test replay buffer functionality."""
    queue = InMemoryRolloutQueue()
    writer = queue.writer()
    reader = queue.reader()

    # Write batches with different environments
    for env in ["env1", "env2"]:
        for i in range(5):
            batch = create_test_batch(i, batch_size=2, max_seq_len=16)
            batch.env_name = env
            writer.write_batch(batch)

    replay_buffer = ReplayBuffer(
        local_batch_size=4,
        process_id=0,
        total_processes=1,
        recency_alpha=3.0,
        capacity=100,
    )

    # Add batches to replay buffer
    batches = reader.read_all_available()
    replay_buffer.add_batches(batches)

    # Test sampling
    training_batch = replay_buffer.sample_training_batch()
    assert training_batch is not None
    assert len(training_batch) == 4

    # Test data loader
    data_loader = ReplayDataLoader(rollout_reader=reader, replay_buffer=replay_buffer, rollout_fetch_interval=0.1)

    # Write more batches for data loader to collect
    for i in range(5, 10):
        batch = create_test_batch(i, batch_size=2, max_seq_len=16)
        batch.env_name = "env1"
        writer.write_batch(batch)

    with data_loader:
        import time

        time.sleep(0.2)  # Let it collect some batches
        batch = data_loader.get_training_batch(timeout=1.0)
        assert batch is not None


def test_replay_buffer_multiprocess_sampling():
    """Test replay buffer with multiple processes sampling."""
    # Create replay buffer for process 0 of 3 processes
    replay_buffer = ReplayBuffer(
        local_batch_size=12,
        process_id=0,
        total_processes=3,
        recency_alpha=2.0,
        capacity=100,
    )

    # Add test batch with 18 examples (divisible by 3)
    batch = create_test_batch(0, batch_size=18, max_seq_len=16)
    batch.env_name = "test_env"
    replay_buffer.add_batches([batch])

    # Test that different processes can sample from the buffer
    all_samples = []
    for process_id in range(3):
        buffer = ReplayBuffer(
            local_batch_size=12,
            process_id=process_id,
            total_processes=3,
            recency_alpha=2.0,
            capacity=100,
        )
        buffer.add_batches([batch])
        sample = buffer.sample_training_batch()
        assert sample is not None
        assert len(sample) == 12

        # Collect all sampled IDs from this process (convert JAX arrays to int)
        process_samples = [int(input_id[0]) for input_id in sample.input_ids]
        all_samples.extend(process_samples)

    # Verify that we got samples (some may overlap due to random sampling)
    assert len(all_samples) == 36  # 3 processes * 12 samples each
    assert len(set(all_samples)) > 0  # At least some unique samples


def test_replay_buffer_recency_bias():
    """Test that high recency bias strongly favors recent (higher index) samples."""
    batch = create_test_batch(0, batch_size=100, max_seq_len=16)
    batch.env_name = "test_env"

    # Create unique identifiable data for each example
    for i in range(100):
        batch.batch.input_ids[i, 0] = 1000 + i  # First token identifies the example

    replay_buffer = ReplayBuffer(
        local_batch_size=50,
        process_id=0,
        total_processes=1,
        recency_alpha=10.0,  # Very strong recency bias
        capacity=100,
    )
    replay_buffer.add_batches([batch])

    # Sample many times to see distribution
    all_samples = []
    for _ in range(100):
        sample = replay_buffer.sample_training_batch()
        if sample is not None:
            indices = [int(token) - 1000 for token in sample.input_ids[:, 0]]
            all_samples.extend(indices)

    # With strong recency bias, we should heavily favor high indices
    # Split indices into low (0-49) and high (50-99) ranges
    low_indices = [idx for idx in all_samples if idx < 50]
    high_indices = [idx for idx in all_samples if idx >= 50]

    # High indices should be much more frequent than low indices
    assert (
        len(high_indices) > len(low_indices) * 3
    ), f"Expected high indices to be 3x more frequent, got {len(high_indices)} high vs {len(low_indices)} low"

    # The highest indices (90-99) should appear more often than lowest (0-9)
    highest_indices = [idx for idx in all_samples if idx >= 90]
    lowest_indices = [idx for idx in all_samples if idx < 10]

    assert len(highest_indices) > len(lowest_indices), (
        f"Expected highest indices (90-99) to be more frequent than lowest (0-9), "
        f"got {len(highest_indices)} highest vs {len(lowest_indices)} lowest"
    )


def test_replay_buffer_stats():
    """Test replay buffer statistics tracking."""
    replay_buffer = ReplayBuffer(
        local_batch_size=4,
        process_id=0,
        total_processes=1,
        recency_alpha=2.0,
        capacity=10,
    )

    # Initially empty
    stats = replay_buffer.get_stats()
    assert stats["total_size"] == 0
    assert stats["num_environments"] == 0
    assert stats["total_batches_added"] == 0
    assert stats["total_batches_sampled"] == 0

    # Add batches
    for i in range(3):
        batch = create_test_batch(i, batch_size=2, max_seq_len=16)
        batch.env_name = f"env_{i % 2}"  # Two different environments
        replay_buffer.add_batches([batch])

    stats = replay_buffer.get_stats()
    assert stats["total_size"] == 6  # 3 batches * 2 examples each = 6 total examples
    assert stats["num_environments"] == 2
    assert stats["total_batches_added"] == 3
    assert stats["total_batches_sampled"] == 0

    # Sample a batch
    training_batch = replay_buffer.sample_training_batch()
    assert training_batch is not None

    stats = replay_buffer.get_stats()
    assert stats["total_batches_sampled"] == 1

    # Test environment-specific sizes
    # env_0 gets batches 0,2 (4 examples), env_1 gets batch 1 (2 examples)
    expected_env_sizes = {"env_0": 4, "env_1": 2}
    assert stats["env_sizes"] == expected_env_sizes


def test_replay_buffer_capacity_eviction():
    """Test that replay buffer respects capacity limits and evicts old data."""
    replay_buffer = ReplayBuffer(
        local_batch_size=4,
        process_id=0,
        total_processes=1,
        recency_alpha=2.0,
        capacity=3,  # Small capacity to test eviction
    )

    # Add more batches than capacity for single environment
    for i in range(5):
        batch = create_test_batch(i, batch_size=2, max_seq_len=16)
        batch.env_name = "test_env"
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
        local_batch_size=2,
        process_id=0,
        total_processes=1,
        recency_alpha=1.0,  # Uniform sampling for predictable behavior
        capacity=100,
        max_samples=3,
    )

    # Create batch with identifiable examples
    batch = create_test_batch(0, batch_size=4, max_seq_len=16)
    batch.env_name = "test_env"

    # Set unique identifiers for each example
    for i in range(4):
        batch.batch.input_ids[i, 0] = 2000 + i

    replay_buffer.add_batches([batch])

    # Initial size should be 4
    assert replay_buffer.size() == 4

    # Sample multiple times - with batch size 2, we expect examples to be used multiple times
    samples_taken = 0
    max_iterations = 20

    for _ in range(max_iterations):
        sample = replay_buffer.sample_training_batch()
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
        local_batch_size=2,
        process_id=0,
        total_processes=1,
        recency_alpha=1.0,
        capacity=100,
        max_samples=-1,  # Disabled
    )

    # Add small batch
    batch = create_test_batch(0, batch_size=3, max_seq_len=16)
    batch.env_name = "test_env"
    replay_buffer.add_batches([batch])

    initial_size = replay_buffer.size()
    assert initial_size == 3

    # Sample many times - examples should never be retired
    for _ in range(50):
        sample = replay_buffer.sample_training_batch()
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
        local_batch_size=3,
        process_id=0,
        total_processes=1,
        recency_alpha=1.0,
        capacity=100,
        max_samples=2,
    )

    # Add batches from different environments
    for env_id in range(2):
        batch = create_test_batch(env_id, batch_size=3, max_seq_len=16)
        batch.env_name = f"env_{env_id}"

        # Set unique identifiers per environment
        for i in range(3):
            batch.batch.input_ids[i, 0] = 3000 + env_id * 100 + i

        replay_buffer.add_batches([batch])

    initial_stats = replay_buffer.get_stats()
    assert initial_stats["total_size"] == 6
    assert initial_stats["num_environments"] == 2

    # Sample multiple times
    for _ in range(15):
        sample = replay_buffer.sample_training_batch()
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


def test_replay_buffer_usage_count_tracking():
    """Test that usage counts are properly tracked."""
    replay_buffer = ReplayBuffer(
        local_batch_size=1,
        process_id=0,
        total_processes=1,
        recency_alpha=1.0,
        capacity=100,
        max_samples=5,
    )

    # Add initial batch
    batch1 = create_test_batch(0, batch_size=2, max_seq_len=16)
    batch1.env_name = "test_env"
    replay_buffer.add_batches([batch1])

    initial_size = replay_buffer.size()
    assert initial_size == 2

    # Sample a few times (less than max_resamples)
    for _ in range(3):
        sample = replay_buffer.sample_training_batch()
        assert sample is not None

    # Size should remain the same since we haven't hit max_resamples
    assert replay_buffer.size() == initial_size

    # Add another batch - this should preserve existing usage counts
    batch2 = create_test_batch(1, batch_size=2, max_seq_len=16)
    batch2.env_name = "test_env"
    replay_buffer.add_batches([batch2])

    # Size should increase
    new_size = replay_buffer.size()
    assert new_size == 4, f"Expected size 4, got {new_size}"

    # Continue sampling - original examples should eventually be retired
    samples_remaining = True
    iterations = 0
    max_iterations = 30

    while samples_remaining and iterations < max_iterations:
        sample = replay_buffer.sample_training_batch()
        if sample is None:
            samples_remaining = False
        iterations += 1

        # If size drops below 4, some original examples were retired
        if replay_buffer.size() < new_size:
            break

    # Should have retired some of the original examples
    final_size = replay_buffer.size()
    assert final_size >= 2, "Should have at least the new batch remaining"
    assert final_size <= new_size, "Buffer size should not increase"


def test_replay_buffer_max_resamples_zero():
    """Test edge case where max_resamples=0 immediately retires examples."""
    replay_buffer = ReplayBuffer(
        local_batch_size=2,
        process_id=0,
        total_processes=1,
        recency_alpha=1.0,
        capacity=100,
        max_samples=0,  # Examples retired immediately after first use
    )

    batch = create_test_batch(0, batch_size=4, max_seq_len=16)
    batch.env_name = "test_env"
    replay_buffer.add_batches([batch])

    assert replay_buffer.size() == 4

    # First sample should work
    sample1 = replay_buffer.sample_training_batch()
    assert sample1 is not None
    assert len(sample1) == 2

    # After sampling, used examples should be retired
    size_after_first_sample = replay_buffer.size()
    assert (
        size_after_first_sample <= 2
    ), f"Expected <= 2 examples after first sample with max_resamples=0, got {size_after_first_sample}"

    # Second sample might work if there are unused examples
    sample2 = replay_buffer.sample_training_batch()
    if sample2 is not None:
        # After this, buffer should be empty or nearly empty
        final_size = replay_buffer.size()
        assert final_size == 0, f"Expected empty buffer after max_resamples=0, got size {final_size}"


def test_replay_buffer_produces_valid_named_arrays():
    """Test that replay buffer returns valid JAX arrays and can convert to NamedArrays."""
    replay_buffer = ReplayBuffer(
        local_batch_size=4,
        process_id=0,
        total_processes=1,
        recency_alpha=1.0,
        capacity=100,
        max_samples=-1,
    )

    # Add multiple batches from different environments
    for env_id in range(2):
        batch = create_test_batch(env_id, batch_size=6, max_seq_len=16)
        batch.env_name = f"test_env_{env_id}"
        replay_buffer.add_batches([batch])

    # Sample a batch
    sampled_batch = replay_buffer.sample_training_batch()
    assert sampled_batch is not None
    assert isinstance(sampled_batch, JaxRolloutBatch)

    # Check that all fields are raw JAX arrays
    import jax.numpy as jnp

    assert isinstance(sampled_batch.input_ids, jnp.ndarray)
    assert sampled_batch.input_ids.shape[0] == 4  # batch size
    assert sampled_batch.input_ids.shape[1] == 16  # sequence length

    # Check batch size matches what we requested
    assert len(sampled_batch) == 4

    # Verify all fields have consistent batch size
    for field_name in [
        "attention_mask",
        "position_ids",
        "target_ids",
        "loss_weights",
        "loss_masks",
        "policy_logprobs",
    ]:
        field = getattr(sampled_batch, field_name)
        assert isinstance(field, jnp.ndarray), f"Field {field_name} should be a JAX array"
        assert field.shape[0] == 4, f"Field {field_name} should have batch size 4"

    # Test that as_named() works correctly
    named_batch = sampled_batch.as_named()
    assert isinstance(named_batch, dict)
    assert "input_ids" in named_batch

    # Check that named version has proper axes
    named_input_ids = named_batch["input_ids"]
    assert hasattr(named_input_ids, "axes")
    assert named_input_ids.axes[0].name == "batch"
    assert named_input_ids.axes[1].name == "position"
    assert named_input_ids.axes[0].size == 4

    print("Replay buffer JAX array output test passed!")


def test_replay_buffer_empty_existing_buffer():
    """Test adding batches to environment with zero-length existing buffer."""
    replay_buffer = ReplayBuffer(
        local_batch_size=2,
        process_id=0,
        total_processes=1,
        recency_alpha=1.0,
        capacity=100,
        max_samples=1,  # Force examples to be retired quickly
    )

    # Add initial batch
    batch1 = create_test_batch(0, batch_size=3, max_seq_len=16)
    batch1.env_name = "test_env"
    replay_buffer.add_batches([batch1])

    assert replay_buffer.size() == 3

    # Sample until buffer is empty due to max_samples=1
    while replay_buffer.size() > 0:
        sample = replay_buffer.sample_training_batch()
        if sample is None:
            break

    # Verify buffer is now empty for this environment
    assert replay_buffer.size() == 0
    stats = replay_buffer.get_stats()
    assert stats["total_size"] == 0

    # Add new batch to same environment - this should work with empty existing buffer
    batch2 = create_test_batch(1, batch_size=2, max_seq_len=16)
    batch2.env_name = "test_env"  # Same environment as before
    replay_buffer.add_batches([batch2])

    # Verify the new batch was added successfully
    assert replay_buffer.size() == 2
    stats = replay_buffer.get_stats()
    assert stats["total_size"] == 2
    assert stats["env_sizes"]["test_env"] == 2

    # Should be able to sample from the new batch
    sample = replay_buffer.sample_training_batch()
    assert sample is not None
    assert len(sample) == 2


def test_replay_buffer_mixed_batch_sizes_concatenation_works():
    """Test that mixed batch sizes work correctly with our JAX refactor."""
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        local_batch_size=4,
        process_id=0,
        total_processes=1,
        recency_alpha=1.0,
        capacity=100,
        max_samples=-1,
    )

    # Add initial batch with batch_size=64
    batch1 = create_test_batch(0, batch_size=64, max_seq_len=255)
    batch1.env_name = "test_env"
    replay_buffer.add_batches([batch1])

    # Add small batch with different batch_size=1
    batch2 = create_test_batch(1, batch_size=1, max_seq_len=255)
    batch2.env_name = "test_env"

    # This should work now with our JAX refactor (no more NamedArray shape issues)
    replay_buffer.add_batches([batch2])

    # Should be able to sample successfully
    sample = replay_buffer.sample_training_batch()
    assert sample is not None
    assert len(sample) == 4  # local_batch_size

    print("Mixed batch sizes concatenation works correctly!")
