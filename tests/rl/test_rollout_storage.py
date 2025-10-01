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

"""Test rollout storage functionality with New* types."""

import pickle
import time

import jax.numpy as jnp
import numpy as np
import pytest

from marin.rl.rollout_storage import (
    RolloutStorageConfig,
    StorageType,
)
from marin.rl.types import (
    Rollout,
    RolloutBatch,
    RolloutGroup,
    RolloutMetadata,
)


def create_test_rollout(idx: int) -> Rollout:
    """Create a single test rollout with identifiable data."""
    rng = np.random.default_rng(42 + idx)

    # Create identifiable tokens
    prompt_len = 8
    response_len = 8

    prompt_tokens = jnp.array([1000 + idx, *rng.integers(0, 100, size=prompt_len - 1)], dtype=jnp.int32)
    response_tokens = jnp.array(rng.integers(0, 100, size=response_len), dtype=jnp.int32)
    response_logprobs = jnp.array(rng.standard_normal(response_len), dtype=jnp.float32)
    token_rewards = jnp.array(rng.standard_normal(response_len), dtype=jnp.float32)
    episode_reward = float(rng.standard_normal())

    return Rollout(
        env_name=f"test_env_{idx % 3}",
        env_example_id=f"example_{idx}",
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        response_logprobs=response_logprobs,
        token_rewards=token_rewards,
        episode_reward=episode_reward,
    )


def create_test_rollout_group(key: str, n_rollouts: int, start_idx: int = 0) -> RolloutGroup:
    """Create a test rollout group."""
    rollouts = [create_test_rollout(start_idx + i) for i in range(n_rollouts)]
    return RolloutGroup(rollouts=rollouts)


def create_test_rollout_batch(idx: int, n_groups: int = 2, rollouts_per_group: int = 3) -> RolloutBatch:
    """Create a test rollout batch with multiple groups."""
    groups = []
    for group_idx in range(n_groups):
        start_idx = idx * n_groups * rollouts_per_group + group_idx * rollouts_per_group
        group = create_test_rollout_group(f"group_{idx}_{group_idx}", rollouts_per_group, start_idx)
        groups.append(group)

    metadata = RolloutMetadata(worker_id=f"worker_{idx}", timestamp=time.time(), weight_step=0)
    return RolloutBatch(groups=groups, metadata=metadata)


@pytest.fixture(params=["memory", "file"])
def storage_config(request, tmp_path):
    """Parametrized fixture for different storage types."""
    if request.param == "memory":
        return RolloutStorageConfig(storage_type=StorageType.IN_MEMORY, queue_name="test_queue")
    else:
        return RolloutStorageConfig(storage_type=StorageType.FILE, path=str(tmp_path / "rollout_storage"))


def test_storage_operations(storage_config):
    """Test basic storage read/write operations."""
    reader = storage_config.create_reader()
    writer = storage_config.create_writer()

    # Test empty read
    batches = reader.read_all_available()
    assert len(batches) == 0

    batch = reader.read_batch(timeout=0.1)
    assert batch is None

    # Write test batches
    test_batches = [create_test_rollout_batch(i) for i in range(3)]
    for batch in test_batches:
        writer.write_batch(batch)

    # Read back
    read_batches = reader.read_all_available()
    assert len(read_batches) == 3

    # Verify content
    for original, read_back in zip(test_batches, read_batches, strict=False):
        assert len(read_back.groups) == len(original.groups)
        assert read_back.metadata.worker_id == original.metadata.worker_id

        # Check first rollout of first group
        orig_rollout = original.groups[0].rollouts[0]
        read_rollout = read_back.groups[0].rollouts[0]
        assert read_rollout.env_name == orig_rollout.env_name
        assert np.array_equal(read_rollout.prompt_tokens, orig_rollout.prompt_tokens)


def test_file_storage_timestamp_ordering(tmp_path):
    """Test that file storage maintains timestamp ordering."""
    config = RolloutStorageConfig(storage_type=StorageType.FILE, path=str(tmp_path / "ordered_test"))

    writer = config.create_writer()
    reader = config.create_reader()

    # Write batches with small delays to ensure different timestamps
    batches = []
    for i in range(3):
        batch = create_test_rollout_batch(i)
        writer.write_batch(batch)
        batches.append(batch)
        time.sleep(0.01)  # Small delay

    # Read back - should be in order
    read_batches = reader.read_all_available()
    assert len(read_batches) == 3

    # Verify ordering by checking worker IDs
    for i, batch in enumerate(read_batches):
        assert batch.metadata.worker_id == f"worker_{i}"


def test_large_rollout_batch():
    """Test handling of large rollout batches."""
    # Create batch with many groups and rollouts
    batch = create_test_rollout_batch(0, n_groups=10, rollouts_per_group=20)

    # Test serialization
    serialized = pickle.dumps(batch)
    deserialized = pickle.loads(serialized)

    assert len(deserialized.groups) == 10
    assert len(deserialized.groups[0].rollouts) == 20

    # Test storage round-trip
    config = RolloutStorageConfig(storage_type=StorageType.IN_MEMORY, queue_name="large_test")
    writer = config.create_writer()
    reader = config.create_reader()

    writer.write_batch(batch)
    read_batch = reader.read_batch()

    assert read_batch is not None
    assert len(read_batch.groups) == 10
    assert len(read_batch.groups[0].rollouts) == 20
