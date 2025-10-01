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
import pytest

from marin.rl.curriculum import (
    Curriculum,
    CurriculumConfig,
    LessonConfig,
    LessonDependency,
    LessonStats,
    get_combined_success_rate,
    is_plateaued,
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


def test_circular_dependency_detection():
    """Test that circular dependencies are detected during initialization."""
    # Create lessons with circular dependency: A -> B -> A
    config = CurriculumConfig(
        lessons=[
            LessonConfig(
                lesson_name="lessonA",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
                dependencies=[LessonDependency(dependency_name="lessonB")],
            ),
            LessonConfig(
                lesson_name="lessonB",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
                dependencies=[LessonDependency(dependency_name="lessonA")],
            ),
        ]
    )

    with pytest.raises(ValueError, match="Circular dependency"):
        Curriculum(config)


def test_unknown_dependency():
    """Test that unknown dependencies are detected."""
    config = CurriculumConfig(
        lessons=[
            LessonConfig(
                lesson_name="lesson1",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
                dependencies=[LessonDependency(dependency_name="nonexistent")],
            )
        ]
    )

    with pytest.raises(ValueError, match="unknown lesson"):
        Curriculum(config)


def test_progressive_unlocking():
    """Test that lessons unlock progressively as dependencies are met."""
    config = CurriculumConfig(
        lessons=[
            LessonConfig(
                lesson_name="basic",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            ),
            LessonConfig(
                lesson_name="intermediate",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
                dependencies=[LessonDependency(dependency_name="basic", reward_threshold=0.5)],
            ),
        ]
    )

    curriculum = Curriculum(config)

    # Initially, only basic should be unlocked
    assert "basic" in curriculum.unlocked
    assert "intermediate" not in curriculum.unlocked

    # Set basic to have good performance but not plateaued
    for i in range(49):
        curriculum.stats["basic"].reward_history.append(0.6 + i * 0.001)
    curriculum.stats["basic"].smoothed_success = 0.6
    curriculum.stats["basic"].total_samples = 50

    # Intermediate should still not unlock (not plateaued)
    curriculum.update_unlocked_lessons()
    assert "intermediate" not in curriculum.unlocked

    # Add more samples to create a plateau
    for _ in range(51):
        curriculum.stats["basic"].reward_history.append(0.65)
    curriculum.stats["basic"].total_samples = 100

    # Now intermediate should unlock
    curriculum.update_unlocked_lessons()
    assert "intermediate" in curriculum.unlocked


def test_multiple_dependencies():
    """Test lessons with multiple dependencies."""
    config = CurriculumConfig(
        lessons=[
            LessonConfig(
                lesson_name="lesson1",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            ),
            LessonConfig(
                lesson_name="lesson2",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
            ),
            LessonConfig(
                lesson_name="advanced",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "opposites"},
                ),
                dependencies=[
                    LessonDependency(dependency_name="lesson1", reward_threshold=0.6),
                    LessonDependency(dependency_name="lesson2", reward_threshold=0.5),
                ],
            ),
        ]
    )

    curriculum = Curriculum(config)

    # Initially, only lesson1 and lesson2 should be unlocked
    assert "lesson1" in curriculum.unlocked
    assert "lesson2" in curriculum.unlocked
    assert "advanced" not in curriculum.unlocked

    # Set lesson1 to meet threshold and plateau
    for _ in range(50):
        curriculum.stats["lesson1"].reward_history.append(0.7)
    curriculum.stats["lesson1"].smoothed_success = 0.7
    curriculum.stats["lesson1"].total_samples = 50

    # Advanced should still not unlock (lesson2 not ready)
    curriculum.update_unlocked_lessons()
    assert "advanced" not in curriculum.unlocked

    # Set lesson2 to meet threshold and plateau
    for _ in range(50):
        curriculum.stats["lesson2"].reward_history.append(0.6)
    curriculum.stats["lesson2"].smoothed_success = 0.6
    curriculum.stats["lesson2"].total_samples = 50

    # Now advanced should unlock
    curriculum.update_unlocked_lessons()
    assert "advanced" in curriculum.unlocked


def test_plateau_detection():
    """Test plateau detection with different reward patterns."""
    # Flat rewards (plateaued)
    stats_flat = LessonStats(reward_history=[1.0] * 50)
    assert is_plateaued(stats_flat, window=50, threshold=0.01)

    # Increasing rewards (not plateaued)
    stats_increasing = LessonStats(reward_history=list(np.linspace(0.0, 1.0, 50)))
    assert not is_plateaued(stats_increasing, window=50, threshold=0.01)

    # Noisy but flat (plateaued)
    rng = np.random.default_rng(42)
    noisy_rewards = [0.5 + 0.01 * rng.standard_normal() for _ in range(50)]
    stats_noisy = LessonStats(reward_history=noisy_rewards)
    assert is_plateaued(stats_noisy, window=50, threshold=0.01)

    # Insufficient data
    stats_insufficient = LessonStats(reward_history=[1.0] * 20)
    assert not is_plateaued(stats_insufficient, window=50, threshold=0.01)


def test_graduation():
    """Test that lessons graduate when mastered."""
    config = CurriculumConfig(
        lessons=[
            LessonConfig(
                lesson_name="easy_lesson",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
                stop_threshold=0.9,
            )
        ]
    )

    curriculum = Curriculum(config)

    # Initially not graduated
    assert "easy_lesson" not in curriculum.graduated

    # Set high performance but no eval data
    for _ in range(50):
        curriculum.stats["easy_lesson"].reward_history.append(0.95)
    curriculum.stats["easy_lesson"].smoothed_success = 0.95
    curriculum.stats["easy_lesson"].total_samples = 50

    # Should not graduate without eval data
    curriculum.update_graduated_lessons()
    assert "easy_lesson" not in curriculum.graduated

    # Add eval data showing high performance
    curriculum.stats["easy_lesson"].eval_success = 0.95
    curriculum.stats["easy_lesson"].eval_reward = 0.95
    curriculum.stats["easy_lesson"].eval_step = 100
    curriculum.current_step = 100

    # Should graduate now
    curriculum.update_graduated_lessons()
    assert "easy_lesson" in curriculum.graduated


def test_graduation_requires_plateau():
    """Test that lessons don't graduate until plateaued."""
    config = CurriculumConfig(
        lessons=[
            LessonConfig(
                lesson_name="improving_lesson",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
                stop_threshold=0.8,
            )
        ]
    )

    curriculum = Curriculum(config)

    # Set improving performance (not plateaued)
    for i in range(50):
        curriculum.stats["improving_lesson"].reward_history.append(0.5 + i * 0.01)
    curriculum.stats["improving_lesson"].smoothed_success = 0.9
    curriculum.stats["improving_lesson"].eval_success = 0.9
    curriculum.stats["improving_lesson"].eval_step = 50
    curriculum.stats["improving_lesson"].total_samples = 50
    curriculum.current_step = 50

    # Should not graduate (still improving)
    curriculum.update_graduated_lessons()
    assert "improving_lesson" not in curriculum.graduated

    # Add plateau
    for _ in range(50):
        curriculum.stats["improving_lesson"].reward_history.append(0.9)

    # Now should graduate
    curriculum.update_graduated_lessons()
    assert "improving_lesson" in curriculum.graduated


def test_graduated_lessons_excluded_from_weights():
    """Test that graduated lessons are excluded from sampling weights."""
    config = CurriculumConfig(
        lessons=[
            LessonConfig(
                lesson_name="graduated",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            ),
            LessonConfig(
                lesson_name="active",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
            ),
        ]
    )

    curriculum = Curriculum(config)

    # Mark first lesson as graduated
    curriculum.graduated.add("graduated")

    # Get weights
    weights = curriculum.compute_sampling_weights()

    # Only active lesson should have weight
    assert "graduated" not in weights
    assert "active" in weights
    assert abs(weights["active"] - 1.0) < 0.01


def test_exploration_bonus_for_new_lessons():
    """Test that new lessons receive exploration bonus."""
    config = CurriculumConfig(
        lessons=[
            LessonConfig(
                lesson_name="new_lesson",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            ),
            LessonConfig(
                lesson_name="experienced_lesson",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
            ),
        ]
    )

    curriculum = Curriculum(config)

    # Set both lessons to same success rate (0.5) but different sample counts
    curriculum.stats["new_lesson"].smoothed_success = 0.5
    curriculum.stats["new_lesson"].total_samples = 10  # New lesson
    curriculum.stats["experienced_lesson"].smoothed_success = 0.5
    curriculum.stats["experienced_lesson"].total_samples = 150  # Experienced

    weights = curriculum.compute_sampling_weights()

    # New lesson should have higher weight due to exploration bonus
    assert weights["new_lesson"] > weights["experienced_lesson"]


def test_exploration_bonus_decay():
    """Test that exploration bonus decays with increasing samples."""
    config = CurriculumConfig(
        lessons=[
            LessonConfig(
                lesson_name="varying_lesson",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            ),
            LessonConfig(
                lesson_name="reference_lesson",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
            ),
        ]
    )

    curriculum = Curriculum(config)

    # Keep reference lesson constant with many samples
    curriculum.stats["reference_lesson"].smoothed_success = 0.5
    curriculum.stats["reference_lesson"].total_samples = 500

    # Track how varying_lesson's weight changes relative to reference
    sample_counts = [1, 10, 50, 100, 200]
    relative_weights = []

    for count in sample_counts:
        curriculum.stats["varying_lesson"].smoothed_success = 0.5
        curriculum.stats["varying_lesson"].total_samples = count
        weight_dict = curriculum.compute_sampling_weights()
        # Track ratio of varying to reference
        relative_weights.append(weight_dict["varying_lesson"] / weight_dict["reference_lesson"])

    # Relative weight should decrease as samples increase (bonus decays)
    assert relative_weights[0] > relative_weights[1]
    assert relative_weights[1] > relative_weights[2]


def test_exploration_bonus_converges():
    """Test that exploration bonus converges to minimal effect at high sample counts."""
    config = CurriculumConfig(
        lessons=[
            LessonConfig(
                lesson_name="lesson1",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            ),
            LessonConfig(
                lesson_name="lesson2",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
            ),
        ]
    )

    curriculum = Curriculum(config)

    # Both lessons with high sample counts should have similar weights
    curriculum.stats["lesson1"].smoothed_success = 0.5
    curriculum.stats["lesson1"].total_samples = 500
    curriculum.stats["lesson2"].smoothed_success = 0.5
    curriculum.stats["lesson2"].total_samples = 600

    weights = curriculum.compute_sampling_weights()

    # Weights should be very similar (exploration bonus is negligible)
    assert abs(weights["lesson1"] - weights["lesson2"]) < 0.05


def test_checkpoint_save_and_load(tmp_path):
    """Test that curriculum state can be saved and restored."""
    checkpoint_dir = tmp_path / "checkpoints"

    config = CurriculumConfig(
        lessons=[
            LessonConfig(
                lesson_name="lesson1",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            ),
            LessonConfig(
                lesson_name="lesson2",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
                dependencies=[LessonDependency(dependency_name="lesson1", reward_threshold=0.5)],
            ),
        ],
        checkpoint_dir=str(checkpoint_dir),
    )

    # Create curriculum and modify state
    curriculum = Curriculum(config)
    curriculum.current_step = 42
    curriculum.stats["lesson1"].smoothed_success = 0.7
    curriculum.stats["lesson1"].smoothed_reward = 0.8
    curriculum.stats["lesson1"].total_samples = 100
    curriculum.stats["lesson1"].eval_success = 0.75
    curriculum.stats["lesson1"].eval_step = 40
    curriculum.unlocked.add("lesson2")

    # Save checkpoint
    curriculum.save_checkpoint()

    # Create new curriculum and load checkpoint
    restored = Curriculum.load_checkpoint(config)

    # Verify all state was restored
    assert restored.current_step == 42
    assert restored.stats["lesson1"].smoothed_success == 0.7
    assert restored.stats["lesson1"].smoothed_reward == 0.8
    assert restored.stats["lesson1"].total_samples == 100
    assert restored.stats["lesson1"].eval_success == 0.75
    assert restored.stats["lesson1"].eval_step == 40
    assert "lesson1" in restored.unlocked
    assert "lesson2" in restored.unlocked


def test_checkpoint_without_checkpoint_dir():
    """Test that checkpointing fails gracefully when checkpoint_dir is not configured."""
    config = CurriculumConfig(
        lessons=[
            LessonConfig(
                lesson_name="lesson1",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            )
        ],
        checkpoint_dir=None,
    )

    curriculum = Curriculum(config)

    # Should raise ValueError when trying to save
    with pytest.raises(ValueError, match="checkpoint_dir not configured"):
        curriculum.save_checkpoint()

    # Should raise ValueError when trying to load
    with pytest.raises(ValueError, match="checkpoint_dir not configured"):
        Curriculum.load_checkpoint(config)


def test_checkpoint_preserves_reward_history(tmp_path):
    """Test that reward history is preserved through checkpoint."""
    checkpoint_dir = tmp_path / "checkpoints"

    config = CurriculumConfig(
        lessons=[
            LessonConfig(
                lesson_name="lesson",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            )
        ],
        checkpoint_dir=str(checkpoint_dir),
    )

    curriculum = Curriculum(config)

    # Add reward history
    curriculum.stats["lesson"].reward_history = [0.1, 0.2, 0.3, 0.4, 0.5]
    curriculum.stats["lesson"].total_samples = 5

    # Save and restore
    curriculum.save_checkpoint()
    restored = Curriculum.load_checkpoint(config)

    # Verify reward history
    assert restored.stats["lesson"].reward_history == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert restored.stats["lesson"].total_samples == 5


def test_checkpoint_graduated_lessons(tmp_path):
    """Test that graduated lessons are preserved through checkpoint."""
    checkpoint_dir = tmp_path / "checkpoints"

    config = CurriculumConfig(
        lessons=[
            LessonConfig(
                lesson_name="lesson1",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            ),
            LessonConfig(
                lesson_name="lesson2",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
            ),
        ],
        checkpoint_dir=str(checkpoint_dir),
    )

    curriculum = Curriculum(config)
    curriculum.graduated.add("lesson1")

    # Save and restore
    curriculum.save_checkpoint()
    restored = Curriculum.load_checkpoint(config)

    # Verify graduation state
    assert "lesson1" in restored.graduated
    assert "lesson2" not in restored.graduated
