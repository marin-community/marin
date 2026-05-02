# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test rollout storage and replay buffer functionality."""

import time

import numpy as np
import pytest

try:
    from marin.rl import train_batch
    from marin.rl.replay_buffer import ReplayBuffer, ReplayDataLoader, StoredTrajectory
    from marin.rl.rl_losses import RLOOLoss
except ImportError:
    pytest.skip("Post training imports unavailable", allow_module_level=True)
from marin.rl.rollout_storage import InMemoryRolloutQueue
from marin.rl.types import Rollout, RolloutBatch, RolloutGroup, RolloutMetadata, RolloutWithAdvantage, TrajectoryRecord


def _trajectory_key(trajectory: TrajectoryRecord) -> tuple[str, str, str]:
    return (trajectory.group_id, trajectory.env_example_id, trajectory.trace_id)


def sampled_trajectories_to_training_batch(
    sampled_trajectories: list[StoredTrajectory],
    max_tokens: int = 32,
    pad_token_id: int = 0,
) -> object:
    """Build the current trainer-compatible batch from neutral replay samples."""
    loss_module = RLOOLoss()
    advantages_by_key: dict[tuple[str, str, str], float] = {}

    seen_groups: set[str] = set()
    for stored_trajectory in sampled_trajectories:
        if stored_trajectory.group.group_id in seen_groups:
            continue
        seen_groups.add(stored_trajectory.group.group_id)
        group_trajectories = stored_trajectory.group.trajectories
        group_rollouts = [train_batch.trajectory_record_to_rollout(trajectory) for trajectory in group_trajectories]
        advantages = loss_module.compute_advantages(group_rollouts)
        for trajectory, advantage in zip(group_trajectories, advantages, strict=True):
            advantages_by_key[_trajectory_key(trajectory)] = float(advantage)

    rollouts_with_advantage = [
        RolloutWithAdvantage(
            rollout=train_batch.trajectory_record_to_rollout(stored_trajectory.trajectory),
            advantage=advantages_by_key[_trajectory_key(stored_trajectory.trajectory)],
        )
        for stored_trajectory in sampled_trajectories
    ]
    return train_batch.create_training_batch_from_rollouts(rollouts_with_advantage, max_tokens, pad_token_id)


def create_test_batch(
    idx: int,
    batch_size: int = 2,
    max_seq_len: int = 16,
    env_name: str | None = None,
    weight_step: int = 0,
    timestamp: float | None = None,
    episode_rewards: list[float] | None = None,
) -> RolloutBatch:
    """Helper to create test batches with identifiable tokens for testing."""
    rng = np.random.default_rng(42 + idx)
    if env_name is None:
        env_name = f"test_env_{idx}"
    if timestamp is None:
        timestamp = time.time()

    batch_metadata = RolloutMetadata(
        worker_id="test_worker",
        timestamp=timestamp,
        weight_step=weight_step,
        lesson_id=f"lesson_{idx}",
        group_id=f"group_{idx}",
        trace_id=f"trace_{idx}",
        task_name="test_task",
        task_version="v1",
    )

    rollouts = []
    prompt_group_id = idx * 1000
    prompt_len = max_seq_len // 2
    response_len = max_seq_len - prompt_len
    shared_prompt_tokens = np.full(prompt_len, prompt_group_id, dtype=np.int32)
    shared_prompt_tokens[1:] = rng.integers(0, 1000, size=prompt_len - 1, dtype=np.int32)
    for i in range(batch_size):
        response_tokens = rng.integers(0, 1000, size=response_len, dtype=np.int32)
        response_logprobs = rng.standard_normal(response_len).astype(np.float32)
        token_rewards = rng.standard_normal(response_len).astype(np.float32)
        if episode_rewards is None:
            episode_reward = float(rng.standard_normal())
        else:
            episode_reward = episode_rewards[i]

        rollout = Rollout(
            env_name=env_name,
            env_example_id=f"example_{idx}_{i}",
            prompt_tokens=shared_prompt_tokens.copy(),
            response_tokens=response_tokens,
            response_logprobs=response_logprobs,
            token_rewards=token_rewards,
            episode_reward=episode_reward,
            temperature=1.0,
            top_k=None,
            is_truncated=False,
            metadata=batch_metadata,
        )
        rollouts.append(rollout)

    group = RolloutGroup(rollouts=rollouts)
    return RolloutBatch(groups=[group], metadata=batch_metadata)


def test_replay_buffer():
    """Test replay buffer functionality."""
    queue = InMemoryRolloutQueue()
    writer = queue.writer()
    reader = queue.reader()

    for env in ["env1", "env2"]:
        for i in range(5):
            batch = create_test_batch(i, batch_size=2, max_seq_len=16, env_name=env)
            writer.write_batch(batch)

    replay_buffer = ReplayBuffer(
        capacity=100,
        local_batch_size=4,
        alpha=3.0,
        total_processes=1,
        max_samples=-1,
        max_rollout_step_delay=1000,
        max_rollout_timestamp_delay=3600.0,
        filter_out_groups_with_no_variance=False,
        seed=42,
    )

    batches = reader.read_all_available()
    replay_buffer.add_batches(batches)

    sampled_trajectories = replay_buffer.sample_sequences()
    assert sampled_trajectories is not None
    assert len(sampled_trajectories) == 4
    assert not hasattr(sampled_trajectories[0], "advantage")

    training_batch = sampled_trajectories_to_training_batch(sampled_trajectories)
    assert training_batch is not None
    assert training_batch.input_ids.shape == {"batch": 4, "position": 16}

    data_loader = ReplayDataLoader(rollout_reader=reader, replay_buffer=replay_buffer, rollout_fetch_interval=0.1)

    for i in range(5, 10):
        batch = create_test_batch(i, batch_size=2, max_seq_len=16, env_name="env1")
        writer.write_batch(batch)

    with data_loader:
        time.sleep(0.2)
        trajectories = data_loader.get_trajectories(timeout=1.0)
        assert trajectories is not None
        assert len(trajectories) == 4


def test_replay_buffer_sample_groups_preserves_full_group_context():
    replay_buffer = ReplayBuffer(
        capacity=100,
        local_batch_size=4,
        alpha=2.0,
        total_processes=1,
        max_samples=-1,
        max_rollout_step_delay=1000,
        max_rollout_timestamp_delay=3600.0,
        filter_out_groups_with_no_variance=False,
        seed=42,
    )

    replay_buffer.add_batches(
        [
            create_test_batch(0, batch_size=2, max_seq_len=16, env_name="env1"),
            create_test_batch(1, batch_size=2, max_seq_len=16, env_name="env1"),
            create_test_batch(2, batch_size=2, max_seq_len=16, env_name="env2"),
        ]
    )

    groups = replay_buffer.sample_groups()
    assert groups is not None
    assert sum(len(group.trajectories) for group in groups) >= 4
    assert all(len(group.trajectories) == 2 for group in groups)
    assert all(group.group_id for group in groups)
    assert all(group.lesson_id.startswith("lesson_") for group in groups)


def test_replay_buffer_recency_bias():
    """Test that high recency bias strongly favors recent (higher index) samples."""
    batches = []
    rollouts_per_batch = 10
    for batch_idx in range(10):
        batch = create_test_batch(batch_idx, batch_size=rollouts_per_batch, max_seq_len=16, env_name="test_env")
        batches.append(batch)

    replay_buffer = ReplayBuffer(
        capacity=100,
        local_batch_size=50,
        alpha=10.0,
        total_processes=1,
        max_samples=-1,
        max_rollout_step_delay=1000,
        max_rollout_timestamp_delay=3600.0,
        filter_out_groups_with_no_variance=False,
        seed=42,
    )
    replay_buffer.add_batches(batches)

    all_samples = []
    for _ in range(100):
        sample = replay_buffer.sample_sequences()
        if sample is not None:
            batch_indices = [int(stored.trajectory.prompt_tokens[0]) // 1000 for stored in sample]
            all_samples.extend(batch_indices)

    low_indices = [idx for idx in all_samples if idx < 5]
    high_indices = [idx for idx in all_samples if idx >= 5]
    assert len(high_indices) > len(
        low_indices
    ), f"Expected high indices to be more frequent, got {len(high_indices)} high vs {len(low_indices)} low"


def test_replay_buffer_capacity_eviction():
    """Test that replay buffer respects capacity limits and evicts old data."""
    replay_buffer = ReplayBuffer(
        capacity=3,
        local_batch_size=4,
        alpha=2.0,
        total_processes=1,
        max_samples=-1,
        max_rollout_step_delay=1000,
        max_rollout_timestamp_delay=3600.0,
        filter_out_groups_with_no_variance=False,
        seed=42,
    )

    for i in range(5):
        replay_buffer.add_batches([create_test_batch(i, batch_size=2, max_seq_len=16, env_name="test_env")])

    stats = replay_buffer.get_stats()
    assert stats["total_size"] == 3
    assert stats["env_sizes"]["test_env"] == 3
    assert stats["total_batches_added"] == 5


def test_replay_buffer_capacity_eviction_removes_evicted_group_members_from_advantages():
    replay_buffer = ReplayBuffer(
        capacity=3,
        local_batch_size=1,
        alpha=1.0,
        total_processes=1,
        max_samples=-1,
        max_rollout_step_delay=1000,
        max_rollout_timestamp_delay=3600.0,
        filter_out_groups_with_no_variance=False,
        seed=42,
    )

    replay_buffer.add_batches(
        [
            create_test_batch(0, batch_size=2, max_seq_len=16, env_name="test_env", episode_rewards=[1.0, 3.0]),
            create_test_batch(1, batch_size=2, max_seq_len=16, env_name="test_env", episode_rewards=[0.0, 0.0]),
        ]
    )

    surviving_group_zero = [
        stored_trajectory
        for stored_trajectory in replay_buffer.rollout_storage["test_env"]
        if stored_trajectory.group_id == "group_0"
    ]
    assert len(surviving_group_zero) == 1

    training_batch = sampled_trajectories_to_training_batch(surviving_group_zero, max_tokens=16, pad_token_id=0)

    assert len(surviving_group_zero[0].group.trajectories) == 1
    np.testing.assert_array_equal(
        training_batch.loss_weights.array[0],
        np.zeros_like(training_batch.loss_weights.array[0]),
    )


def test_replay_buffer_max_resamples():
    """Test that examples are retired after max_resamples uses."""
    replay_buffer = ReplayBuffer(
        capacity=100,
        local_batch_size=2,
        alpha=1.0,
        total_processes=1,
        max_samples=3,
        max_rollout_step_delay=1000,
        max_rollout_timestamp_delay=3600.0,
        filter_out_groups_with_no_variance=False,
        seed=42,
    )

    replay_buffer.add_batches([create_test_batch(0, batch_size=4, max_seq_len=16, env_name="test_env")])

    assert replay_buffer.size() == 4

    samples_taken = 0
    for _ in range(20):
        sample = replay_buffer.sample_sequences()
        if sample is None:
            break
        samples_taken += 1

    final_size = replay_buffer.size()
    assert final_size == 0, f"Expected buffer to shrink due to max_resamples, but size remained {final_size}"
    assert samples_taken > 4


def test_replay_buffer_max_resamples_disabled():
    """Test that max_resamples=-1 disables retirement."""
    replay_buffer = ReplayBuffer(
        capacity=100,
        local_batch_size=2,
        alpha=1.0,
        total_processes=1,
        max_samples=-1,
        max_rollout_step_delay=1000,
        max_rollout_timestamp_delay=3600.0,
        filter_out_groups_with_no_variance=False,
        seed=42,
    )

    replay_buffer.add_batches([create_test_batch(0, batch_size=3, max_seq_len=16, env_name="test_env")])
    initial_size = replay_buffer.size()
    assert initial_size == 3

    for _ in range(50):
        sample = replay_buffer.sample_sequences()
        assert sample is not None
        assert len(sample) == 2
        assert replay_buffer.size() == initial_size


def test_replay_buffer_max_resamples_multiple_envs():
    """Test max_resamples with multiple environments."""
    replay_buffer = ReplayBuffer(
        capacity=100,
        local_batch_size=3,
        alpha=1.0,
        total_processes=1,
        max_samples=2,
        max_rollout_step_delay=1000,
        max_rollout_timestamp_delay=3600.0,
        filter_out_groups_with_no_variance=False,
        seed=42,
    )

    for env_id in range(2):
        env_name = f"env_{env_id}"
        replay_buffer.add_batches([create_test_batch(env_id, batch_size=3, max_seq_len=16, env_name=env_name)])

    initial_stats = replay_buffer.get_stats()
    assert initial_stats["total_size"] == 6
    assert initial_stats["num_environments"] == 2

    for _ in range(15):
        sample = replay_buffer.sample_sequences()
        if sample is None:
            break

    final_stats = replay_buffer.get_stats()
    assert final_stats["num_environments"] == 2
    assert final_stats["total_size"] < 6
    for env_name in ["env_0", "env_1"]:
        assert env_name in final_stats["env_sizes"]


def test_replay_buffer_weight_step_filtering():
    replay_buffer = ReplayBuffer(
        capacity=100,
        local_batch_size=4,
        alpha=2.0,
        total_processes=1,
        max_samples=-1,
        max_rollout_step_delay=30,
        max_rollout_timestamp_delay=3600.0,
        filter_out_groups_with_no_variance=False,
        seed=42,
    )

    for weight_step in [50, 100, 150]:
        replay_buffer.add_batches(
            [create_test_batch(weight_step, batch_size=4, max_seq_len=16, env_name="test_env", weight_step=weight_step)]
        )

    assert replay_buffer.size() == 0

    replay_buffer.set_current_step(100)
    for weight_step in [50, 100, 150]:
        replay_buffer.add_batches(
            [create_test_batch(weight_step, batch_size=4, max_seq_len=16, env_name="test_env", weight_step=weight_step)]
        )

    assert replay_buffer.size() == 4

    batch_95 = create_test_batch(95, batch_size=3, max_seq_len=16, env_name="test_env", weight_step=95)
    replay_buffer.add_batches([batch_95])
    assert replay_buffer.size() == 7

    replay_buffer.set_current_step(150)
    assert replay_buffer.size() == 0

    new_batch = create_test_batch(150, batch_size=2, max_seq_len=16, env_name="test_env", weight_step=150)
    replay_buffer.add_batches([new_batch])
    assert replay_buffer.size() == 2

    batch_130 = create_test_batch(130, batch_size=3, max_seq_len=16, env_name="test_env", weight_step=130)
    replay_buffer.add_batches([batch_130])
    assert replay_buffer.size() == 5

    replay_buffer.set_current_step(160)
    assert replay_buffer.size() == 5


def test_replay_buffer_rollout_delay_progressive():
    """Test rollout delay with progressive step advancement and stale batch rejection."""
    replay_buffer = ReplayBuffer(
        capacity=100,
        local_batch_size=2,
        alpha=2.0,
        total_processes=1,
        max_samples=-1,
        max_rollout_step_delay=10,
        max_rollout_timestamp_delay=3600.0,
        filter_out_groups_with_no_variance=False,
        seed=42,
    )

    replay_buffer.set_current_step(15)

    for step in [5, 10, 15]:
        replay_buffer.add_batches(
            [create_test_batch(step, batch_size=2, max_seq_len=16, env_name="test_env", weight_step=step)]
        )

    assert replay_buffer.size() == 6

    replay_buffer.add_batches([create_test_batch(3, batch_size=2, max_seq_len=16, env_name="test_env", weight_step=3)])
    assert replay_buffer.size() == 6

    replay_buffer.add_batches([create_test_batch(20, batch_size=2, max_seq_len=16, env_name="test_env", weight_step=20)])
    assert replay_buffer.size() == 6

    fresh_batch = create_test_batch(14, batch_size=3, max_seq_len=16, env_name="test_env", weight_step=14)
    replay_buffer.add_batches([fresh_batch])
    assert replay_buffer.size() == 9

    replay_buffer.set_current_step(20)
    assert replay_buffer.size() == 7

    sample = replay_buffer.sample_sequences()
    assert sample is not None
    assert len(sample) == 2


def test_is_rollout_fresh():
    """Test the _is_rollout_fresh helper method handles all filtering conditions."""
    replay_buffer = ReplayBuffer(
        capacity=100,
        local_batch_size=4,
        alpha=2.0,
        total_processes=1,
        max_samples=-1,
        max_rollout_step_delay=10,
        max_rollout_timestamp_delay=100.0,
        filter_out_groups_with_no_variance=False,
        seed=42,
    )

    current_step = 100
    current_time = time.time()

    assert replay_buffer._is_rollout_fresh(90, current_time - 50, current_step, current_time)
    assert replay_buffer._is_rollout_fresh(95, current_time - 50, current_step, current_time)
    assert replay_buffer._is_rollout_fresh(100, current_time - 50, current_step, current_time)

    assert not replay_buffer._is_rollout_fresh(89, current_time - 50, current_step, current_time)

    assert not replay_buffer._is_rollout_fresh(101, current_time - 50, current_step, current_time)
    assert not replay_buffer._is_rollout_fresh(105, current_time - 50, current_step, current_time)
    assert not replay_buffer._is_rollout_fresh(111, current_time - 50, current_step, current_time)

    assert not replay_buffer._is_rollout_fresh(95, current_time - 200, current_step, current_time)
