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
            reference_logprobs=rng.standard_normal((batch_size, max_seq_len)).astype(np.float32),
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
    assert len(training_batch.input_ids) == 4

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
        assert len(sample.input_ids) == 12

        # Collect all sampled IDs from this process
        process_samples = [input_id[0] for input_id in sample.input_ids]
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
