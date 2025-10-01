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

"""Tests for adaptive curriculum learning system."""

import jax
import jax.numpy as jnp
import numpy as np

from marin.rl.curriculum import (
    Curriculum,
    CurriculumConfig,
    LessonConfig,
    LessonStats,
    get_combined_success_rate,
    sigmoid,
    update_from_rollout,
)
from marin.rl.environments.base import EnvConfig
from marin.rl.types import Rollout


def create_test_rollout(episode_reward: float, lesson_name: str = "test") -> Rollout:
    """Helper to create a minimal rollout for testing."""
    return Rollout(
        env_name=f"mock:{lesson_name}",
        env_example_id="test_example",
        prompt_tokens=jnp.array([1, 2, 3], dtype=jnp.int32),
        response_tokens=jnp.array([4, 5, 6], dtype=jnp.int32),
        response_logprobs=jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32),
        token_rewards=jnp.array([episode_reward] * 3, dtype=jnp.float32),
        episode_reward=episode_reward,
    )


def test_sigmoid():
    """Test sigmoid function produces expected values."""
    # At center, should be 0.5
    assert abs(sigmoid(0.5, center=0.5, steepness=1.0) - 0.5) < 0.01

    # Far below center should be near 0
    assert sigmoid(0.0, center=0.5, steepness=10.0) < 0.01

    # Far above center should be near 1
    assert sigmoid(1.0, center=0.5, steepness=10.0) > 0.99


def test_update_from_rollout():
    """Test statistics updates from rollouts."""
    stats = LessonStats()

    # First rollout with reward
    rollout1 = create_test_rollout(episode_reward=1.0)
    stats = update_from_rollout(stats, rollout1)

    assert stats.total_samples == 1
    assert stats.smoothed_success == 1.0
    assert stats.smoothed_reward == 1.0
    assert len(stats.reward_history) == 1

    # Second rollout with zero reward
    rollout2 = create_test_rollout(episode_reward=0.0)
    stats = update_from_rollout(stats, rollout2)

    assert stats.total_samples == 2
    assert stats.smoothed_success < 1.0  # Should decrease
    assert stats.smoothed_success > 0.0
    assert len(stats.reward_history) == 2


def test_update_from_rollout_history_limit():
    """Test that reward history is limited to 100 samples."""
    stats = LessonStats()

    # Add 150 rollouts
    for i in range(150):
        rollout = create_test_rollout(episode_reward=float(i))
        stats = update_from_rollout(stats, rollout)

    # Should only keep last 100
    assert len(stats.reward_history) == 100
    assert stats.reward_history[0] == 50.0  # First kept reward
    assert stats.reward_history[-1] == 149.0  # Last reward


def test_get_combined_success_rate_no_eval():
    """Test combined success rate when no eval data exists."""
    stats = LessonStats(smoothed_success=0.7)
    rate = get_combined_success_rate(stats, current_step=100)
    assert rate == 0.7


def test_get_combined_success_rate_with_eval():
    """Test combined success rate blends eval and training data."""
    stats = LessonStats(smoothed_success=0.5, eval_success=0.8, eval_step=50)

    # Recent eval should weight heavily
    rate_recent = get_combined_success_rate(stats, current_step=60)
    assert rate_recent > 0.6  # Should be closer to eval

    # Stale eval should weight less
    rate_stale = get_combined_success_rate(stats, current_step=1050)
    assert rate_stale < 0.6  # Should be closer to training


def test_single_lesson_curriculum():
    """Test curriculum with a single lesson."""
    config = CurriculumConfig(
        lessons=[
            LessonConfig(
                lesson_name="only_lesson",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats", "seed": 42},
                ),
            )
        ]
    )

    curriculum = Curriculum(config)

    # Should have one unlocked lesson
    assert len(curriculum.unlocked) == 1
    assert "only_lesson" in curriculum.unlocked
    assert len(curriculum.graduated) == 0

    # Should be able to compute weights
    weights = curriculum.compute_sampling_weights()
    assert len(weights) == 1
    assert abs(weights["only_lesson"] - 1.0) < 0.01

    # Should be able to sample
    key = jax.random.PRNGKey(0)
    lesson_name, env = curriculum.sample_lesson(key)
    assert lesson_name == "only_lesson"
    assert env is not None


def test_weight_computation_at_different_success_rates():
    """Test that weights peak at intermediate success rates."""
    # Create multiple lessons to avoid normalization issues
    config = CurriculumConfig(
        lessons=[
            LessonConfig(
                lesson_name=f"lesson{i}",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats", "seed": i},
                ),
            )
            for i in range(5)
        ]
    )

    curriculum = Curriculum(config)

    # Test at different success rates
    success_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    weights = []

    # Set each lesson to a different success rate
    for i, success_rate in enumerate(success_rates):
        lesson_name = f"lesson{i}"
        curriculum.stats[lesson_name].smoothed_success = success_rate
        curriculum.stats[lesson_name].total_samples = 100

    curriculum.current_step = 1000
    weight_dict = curriculum.compute_sampling_weights()

    # Extract weights in the same order as success rates
    weights = [weight_dict[f"lesson{i}"] for i in range(5)]

    # Weight should peak around 0.5
    max_idx = np.argmax(weights)
    assert success_rates[max_idx] == 0.5

    # Weights should be lower at extremes
    assert weights[0] < weights[2]  # 0.1 < 0.5
    assert weights[-1] < weights[2]  # 0.9 < 0.5


def test_sampling_distribution():
    """Test that sampling respects weight distribution."""
    config = CurriculumConfig(
        lessons=[
            LessonConfig(
                lesson_name="easy",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            ),
            LessonConfig(
                lesson_name="medium",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
            ),
            LessonConfig(
                lesson_name="hard",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "opposites"},
                ),
            ),
        ]
    )

    curriculum = Curriculum(config)

    # Set different success rates to create different weights
    curriculum.stats["easy"].smoothed_success = 0.9  # Too easy - low weight
    curriculum.stats["easy"].total_samples = 100
    curriculum.stats["medium"].smoothed_success = 0.5  # Just right - high weight
    curriculum.stats["medium"].total_samples = 100
    curriculum.stats["hard"].smoothed_success = 0.1  # Too hard - low weight
    curriculum.stats["hard"].total_samples = 100

    # Sample many times and check distribution
    key = jax.random.PRNGKey(42)
    samples = []
    for _ in range(1000):
        key, subkey = jax.random.split(key)
        lesson_name, _ = curriculum.sample_lesson(subkey)
        samples.append(lesson_name)

    # Count samples
    counts = {name: samples.count(name) for name in ["easy", "medium", "hard"]}

    # Medium should be sampled most
    assert counts["medium"] > counts["easy"]
    assert counts["medium"] > counts["hard"]


def test_initial_weights():
    """Test that initial weights are used when no data is available."""
    config = CurriculumConfig(
        lessons=[
            LessonConfig(
                lesson_name="lesson1",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
                initial_weight=0.3,
            ),
            LessonConfig(
                lesson_name="lesson2",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
                initial_weight=0.7,
            ),
        ]
    )

    curriculum = Curriculum(config)

    # With no samples, should use initial weights
    weights = curriculum.compute_sampling_weights()
    assert abs(weights["lesson1"] - 0.3) < 0.01
    assert abs(weights["lesson2"] - 0.7) < 0.01


def test_lesson_stats_serialization():
    """Test that LessonStats can be serialized and deserialized."""
    stats = LessonStats(
        smoothed_success=0.7,
        smoothed_reward=0.8,
        eval_success=0.75,
        eval_reward=0.85,
        eval_step=100,
        total_samples=500,
        reward_history=[0.1, 0.2, 0.3],
    )

    # Serialize
    data = stats.to_dict()

    # Deserialize
    restored = LessonStats.from_dict(data)

    assert restored.smoothed_success == stats.smoothed_success
    assert restored.smoothed_reward == stats.smoothed_reward
    assert restored.eval_success == stats.eval_success
    assert restored.eval_reward == stats.eval_reward
    assert restored.eval_step == stats.eval_step
    assert restored.total_samples == stats.total_samples
    assert restored.reward_history == stats.reward_history
