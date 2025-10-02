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
import numpy as np
import pytest

from marin.rl.curriculum import (
    Curriculum,
    CurriculumConfig,
    LessonConfig,
    LessonDependency,
    LessonStats,
    PerformanceStats,
    is_plateaued,
    update_performance_stats,
)
from marin.rl.environments.base import EnvConfig
from marin.rl.types import RolloutStats


def create_test_rollout_stats(episode_reward: float, lesson_id: str = "test") -> RolloutStats:
    """Helper to create rollout stats for testing."""
    return RolloutStats(lesson_id=lesson_id, episode_reward=episode_reward, env_example_id="test_example")


def test_update_performance_stats():
    """Test statistics updates from rollout stats."""
    stats = PerformanceStats()

    # First rollout with reward
    rollout_stats1 = create_test_rollout_stats(episode_reward=1.0)
    stats = update_performance_stats(stats, [rollout_stats1], current_step=1)

    assert stats.total_samples == 1
    assert len(stats.reward_history) == 1
    assert stats.reward_history[0] == 1.0
    assert stats.last_update_step == 1

    # Second rollout with zero reward
    rollout_stats2 = create_test_rollout_stats(episode_reward=0.0)
    stats = update_performance_stats(stats, [rollout_stats2], current_step=2)

    assert stats.total_samples == 2
    assert len(stats.reward_history) == 2
    assert stats.reward_history[1] == 0.0
    assert stats.last_update_step == 2

    # Compute smoothed success on demand
    from marin.rl.curriculum import compute_smoothed_success

    smoothed = compute_smoothed_success(stats.reward_history)
    assert 0.0 < smoothed < 1.0  # Should be between 0 and 1


def test_single_lesson_curriculum():
    """Test curriculum with a single lesson."""
    config = CurriculumConfig(
        lessons={
            "only_lesson": LessonConfig(
                lesson_id="only_lesson",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats", "seed": 42},
                ),
            )
        }
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
    lesson_name = curriculum.sample_lesson(prng_key=jax.random.PRNGKey(0))
    assert lesson_name == "only_lesson"


def test_weight_computation_at_different_success_rates():
    """Test that weights peak at intermediate success rates."""
    # Create multiple lessons to avoid normalization issues
    config = CurriculumConfig(
        lessons={
            f"lesson{i}": LessonConfig(
                lesson_id=f"lesson{i}",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats", "seed": i},
                ),
            )
            for i in range(5)
        }
    )

    curriculum = Curriculum(config)

    # Test at different success rates
    success_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    weights = []

    # Set each lesson to a different success rate by creating appropriate reward histories
    # Interleave successes and failures to get stable EMA
    for i, success_rate in enumerate(success_rates):
        lesson_name = f"lesson{i}"
        n_samples = 100
        # Create interleaved pattern for more stable EMA
        pattern_len = 10
        n_success_per_pattern = int(success_rate * pattern_len)
        pattern = [1.0] * n_success_per_pattern + [0.0] * (pattern_len - n_success_per_pattern)
        reward_history = np.array(pattern * (n_samples // pattern_len))
        curriculum.stats[lesson_name].training_stats = PerformanceStats(
            total_samples=n_samples, reward_history=reward_history, last_update_step=1000
        )

    curriculum.current_step = 1000
    weight_dict = curriculum.compute_sampling_weights()

    # Extract weights in the same order as success rates
    weights = [weight_dict[f"lesson{i}"] for i in range(5)]

    # Weight should peak around intermediate success rates
    # Due to Bayesian prior (0.5) and EMA with alpha=0.1, the smoothed success
    # rates are pulled towards 0.5, so actual 70% gives smoothed ~0.58 which is
    # closest to optimal 0.5
    max_idx = np.argmax(weights)
    assert success_rates[max_idx] in [0.5, 0.7]  # Peak should be at intermediate rates

    # Weights should be lower at extremes
    assert weights[0] < weights[max_idx]  # 0.1 < peak
    assert weights[-1] < weights[max_idx]  # 0.9 < peak


def test_sampling_distribution():
    """Test that sampling respects weight distribution."""
    config = CurriculumConfig(
        lessons={
            "easy": LessonConfig(
                lesson_id="easy",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            ),
            "medium": LessonConfig(
                lesson_id="medium",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
            ),
            "hard": LessonConfig(
                lesson_id="hard",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "opposites"},
                ),
            ),
        }
    )

    curriculum = Curriculum(config)

    # Set different success rates to create different weights
    # Use interleaved patterns for stable EMA
    curriculum.stats["easy"].training_stats = PerformanceStats(
        total_samples=100,
        reward_history=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0] * 10),
        last_update_step=100,
    )  # 90% success - too easy - low weight
    curriculum.stats["medium"].training_stats = PerformanceStats(
        total_samples=100,
        reward_history=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0] * 10),
        last_update_step=100,
    )  # 50% success - just right - high weight
    curriculum.stats["hard"].training_stats = PerformanceStats(
        total_samples=100,
        reward_history=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * 10),
        last_update_step=100,
    )  # 10% success - too hard - low weight

    # Sample many times and check distribution
    samples = []
    for i in range(1000):
        lesson_name = curriculum.sample_lesson(prng_key=jax.random.PRNGKey(i))
        samples.append(lesson_name)

    # Count samples
    counts = {name: samples.count(name) for name in ["easy", "medium", "hard"]}
    print(counts)

    # Medium (50% actual) should have highest weight
    assert counts["medium"] > counts["easy"], counts
    assert counts["medium"] > counts["hard"], counts
    # Due to Bayesian prior pulling towards 0.5, easy (90%) ends up closer to 0.5
    # than hard (10%), so easy gets more samples than hard
    assert counts["easy"] > counts["hard"], counts


def test_lesson_stats_serialization():
    """Test that LessonStats can be serialized and deserialized via JSON."""
    import json

    stats = LessonStats(
        training_stats=PerformanceStats(
            total_samples=500,
            reward_history=np.array([0.1, 0.2, 0.3]),
            last_update_step=100,
        ),
        eval_stats=PerformanceStats(total_samples=50, reward_history=np.array([0.5]), last_update_step=100),
    )

    # Serialize to JSON-compatible dict (like in save_checkpoint)
    data = {
        "training_stats": {
            "total_samples": stats.training_stats.total_samples,
            "reward_history": stats.training_stats.reward_history.tolist(),
            "last_update_step": stats.training_stats.last_update_step,
        },
        "eval_stats": {
            "total_samples": stats.eval_stats.total_samples,
            "reward_history": stats.eval_stats.reward_history.tolist(),
            "last_update_step": stats.eval_stats.last_update_step,
        },
    }

    # Verify JSON serializable
    json_str = json.dumps(data)
    loaded_data = json.loads(json_str)

    # Deserialize (like in restore_checkpoint)
    restored = LessonStats(
        training_stats=PerformanceStats(
            total_samples=loaded_data["training_stats"]["total_samples"],
            reward_history=np.array(loaded_data["training_stats"]["reward_history"]),
            last_update_step=loaded_data["training_stats"]["last_update_step"],
        ),
        eval_stats=PerformanceStats(
            total_samples=loaded_data["eval_stats"]["total_samples"],
            reward_history=np.array(loaded_data["eval_stats"]["reward_history"]),
            last_update_step=loaded_data["eval_stats"]["last_update_step"],
        ),
    )

    assert restored.training_stats.total_samples == stats.training_stats.total_samples
    assert np.array_equal(restored.training_stats.reward_history, stats.training_stats.reward_history)
    assert restored.training_stats.last_update_step == stats.training_stats.last_update_step
    assert np.array_equal(restored.eval_stats.reward_history, stats.eval_stats.reward_history)


def test_circular_dependency_detection():
    """Test that circular dependencies are detected during initialization."""
    # Create lessons with circular dependency: A -> B -> A
    config = CurriculumConfig(
        lessons={
            "lessonA": LessonConfig(
                lesson_id="lessonA",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
                dependencies=[LessonDependency(dependency_id="lessonB")],
            ),
            "lessonB": LessonConfig(
                lesson_id="lessonB",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
                dependencies=[LessonDependency(dependency_id="lessonA")],
            ),
        }
    )

    with pytest.raises(ValueError, match="Circular dependency"):
        Curriculum(config)


def test_unknown_dependency():
    """Test that unknown dependencies are detected."""
    config = CurriculumConfig(
        lessons={
            "lesson1": LessonConfig(
                lesson_id="lesson1",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
                dependencies=[LessonDependency(dependency_id="nonexistent")],
            )
        }
    )

    with pytest.raises(ValueError, match="unknown lesson"):
        Curriculum(config)


def test_progressive_unlocking():
    """Test that lessons unlock progressively as dependencies are met."""
    config = CurriculumConfig(
        lessons={
            "basic": LessonConfig(
                lesson_id="basic",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            ),
            "intermediate": LessonConfig(
                lesson_id="intermediate",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
                dependencies=[LessonDependency(dependency_id="basic", reward_threshold=0.5)],
            ),
        }
    )

    curriculum = Curriculum(config)

    # Initially, only basic should be unlocked
    assert "basic" in curriculum.unlocked
    assert "intermediate" not in curriculum.unlocked

    # Simulate good performance but not plateaued (increasing rewards)
    rollout_stats = []
    for i in range(49):
        reward = 0.6 + i * 0.001
        rollout_stats.append(create_test_rollout_stats(episode_reward=reward, lesson_id="basic"))
    curriculum.update_lesson_stats(rollout_stats, mode="training", current_step=49)

    # Intermediate should still not unlock (not plateaued)
    assert "intermediate" not in curriculum.unlocked

    # Add more samples to create a plateau (constant rewards)
    rollout_stats = []
    for _ in range(51):
        rollout_stats.append(create_test_rollout_stats(episode_reward=0.65, lesson_id="basic"))
    curriculum.update_lesson_stats(rollout_stats, mode="training", current_step=100)

    # Now intermediate should unlock
    assert "intermediate" in curriculum.unlocked


def test_multiple_dependencies():
    """Test lessons with multiple dependencies."""
    config = CurriculumConfig(
        lessons={
            "lesson1": LessonConfig(
                lesson_id="lesson1",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            ),
            "lesson2": LessonConfig(
                lesson_id="lesson2",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
            ),
            "advanced": LessonConfig(
                lesson_id="advanced",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "opposites"},
                ),
                dependencies=[
                    LessonDependency(dependency_id="lesson1", reward_threshold=0.6),
                    LessonDependency(dependency_id="lesson2", reward_threshold=0.5),
                ],
            ),
        }
    )

    curriculum = Curriculum(config)

    # Initially, only lesson1 and lesson2 should be unlocked
    assert "lesson1" in curriculum.unlocked
    assert "lesson2" in curriculum.unlocked
    assert "advanced" not in curriculum.unlocked

    # Set lesson1 to meet threshold and plateau
    rollout_stats = []
    for _ in range(50):
        rollout_stats.append(create_test_rollout_stats(episode_reward=0.7, lesson_id="lesson1"))
    curriculum.update_lesson_stats(rollout_stats, mode="training", current_step=50)

    # Advanced should still not unlock (lesson2 not ready)
    assert "advanced" not in curriculum.unlocked

    # Set lesson2 to meet threshold and plateau
    rollout_stats = []
    for _ in range(50):
        rollout_stats.append(create_test_rollout_stats(episode_reward=0.6, lesson_id="lesson2"))
    curriculum.update_lesson_stats(rollout_stats, mode="training", current_step=100)

    # Now advanced should unlock
    assert "advanced" in curriculum.unlocked


def test_plateau_detection():
    """Test plateau detection with different reward patterns."""
    # Flat rewards (plateaued)
    stats_flat = LessonStats(training_stats=PerformanceStats(reward_history=[1.0] * 50))
    assert is_plateaued(stats_flat, window=50, threshold=0.01)

    # Increasing rewards (not plateaued)
    stats_increasing = LessonStats(training_stats=PerformanceStats(reward_history=list(np.linspace(0.0, 1.0, 50))))
    assert not is_plateaued(stats_increasing, window=50, threshold=0.01)

    # Noisy but flat (plateaued)
    rng = np.random.default_rng(42)
    noisy_rewards = [0.5 + 0.01 * rng.standard_normal() for _ in range(50)]
    stats_noisy = LessonStats(training_stats=PerformanceStats(reward_history=noisy_rewards))
    assert is_plateaued(stats_noisy, window=50, threshold=0.01)

    # Insufficient data
    stats_insufficient = LessonStats(training_stats=PerformanceStats(reward_history=[1.0] * 20))
    assert not is_plateaued(stats_insufficient, window=50, threshold=0.01)


def test_plateau_detection_improving_trend():
    """Test that improving trends are NOT detected as plateaus.

    This mirrors the 'Improving Rewards (No Plateau)' pattern from
    visualize_curriculum.ipynb where performance steadily improves.
    Conservative plateau detection should NOT trigger for improving performance.
    """
    # Linear improvement from 0.1 to 0.9 over 50 steps
    improving_rewards = [0.1 + (0.9 - 0.1) * (i / 49) for i in range(50)]
    stats_improving = LessonStats(training_stats=PerformanceStats(reward_history=improving_rewards))

    # Should NOT plateau - there's a clear upward trend
    assert not is_plateaued(stats_improving, window=50, threshold=0.01)

    # Even with some noise, strong trend should prevent plateau detection
    rng = np.random.default_rng(42)
    noisy_improving = [0.5 + i * 0.01 + 0.02 * rng.standard_normal() for i in range(50)]
    stats_noisy_improving = LessonStats(training_stats=PerformanceStats(reward_history=noisy_improving))
    assert not is_plateaued(stats_noisy_improving, window=50, threshold=0.01)

    # Gradual improvement (slower but still trending)
    slow_improving = [0.7 + i * 0.003 for i in range(50)]
    stats_slow = LessonStats(training_stats=PerformanceStats(reward_history=slow_improving))
    # Conservative: even slow improvement should not plateau
    assert not is_plateaued(stats_slow, window=50, threshold=0.01)


def test_graduation():
    """Test that lessons graduate when mastered."""
    config = CurriculumConfig(
        lessons={
            "easy_lesson": LessonConfig(
                lesson_id="easy_lesson",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
                stop_threshold=0.9,
            )
        }
    )

    curriculum = Curriculum(config)

    # Initially not graduated
    assert "easy_lesson" not in curriculum.graduated

    # Set high performance but no eval data
    rollout_stats = []
    for _ in range(50):
        rollout_stats.append(create_test_rollout_stats(episode_reward=0.95, lesson_id="easy_lesson"))
    curriculum.update_lesson_stats(rollout_stats, mode="training", current_step=50)

    # Should not graduate without eval data
    assert "easy_lesson" not in curriculum.graduated

    # Add eval data showing high performance
    eval_stats = []
    for _ in range(50):
        eval_stats.append(create_test_rollout_stats(episode_reward=0.95, lesson_id="easy_lesson"))
    curriculum.update_lesson_stats(eval_stats, mode="eval", current_step=100)

    # Should graduate now
    assert "easy_lesson" in curriculum.graduated, curriculum.stats


def test_graduation_requires_plateau():
    """Test that lessons don't graduate until plateaued."""
    config = CurriculumConfig(
        lessons={
            "improving_lesson": LessonConfig(
                lesson_id="improving_lesson",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
                stop_threshold=0.8,
            )
        }
    )

    curriculum = Curriculum(config)

    # Set improving performance (not plateaued) in TRAINING stats
    rollout_stats = []
    for i in range(50):
        reward = 0.5 + i * 0.01
        rollout_stats.append(create_test_rollout_stats(episode_reward=reward, lesson_id="improving_lesson"))
    curriculum.update_lesson_stats(rollout_stats, mode="training", current_step=50)

    # Add eval data
    eval_stats = []
    for _ in range(10):
        eval_stats.append(create_test_rollout_stats(episode_reward=0.9, lesson_id="improving_lesson"))
    curriculum.update_lesson_stats(eval_stats, mode="eval", current_step=50)

    # Should not graduate (still improving)
    assert "improving_lesson" not in curriculum.graduated

    # Add plateau to TRAINING stats (constant rewards)
    rollout_stats = []
    for _ in range(50):
        rollout_stats.append(create_test_rollout_stats(episode_reward=0.9, lesson_id="improving_lesson"))
    curriculum.update_lesson_stats(rollout_stats, mode="training", current_step=100)

    # Now should graduate
    assert "improving_lesson" in curriculum.graduated, curriculum


def test_graduated_lessons_excluded_from_weights():
    """Test that graduated lessons are excluded from sampling weights."""
    config = CurriculumConfig(
        lessons={
            "graduated": LessonConfig(
                lesson_id="graduated",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            ),
            "active": LessonConfig(
                lesson_id="active",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
            ),
        }
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
        lessons={
            "new_lesson": LessonConfig(
                lesson_id="new_lesson",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            ),
            "experienced_lesson": LessonConfig(
                lesson_id="experienced_lesson",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
            ),
        }
    )

    curriculum = Curriculum(config)

    # Set both lessons to same success rate (0.5) but different sample counts
    curriculum.stats["new_lesson"].training_stats.smoothed_success = 0.5
    curriculum.stats["new_lesson"].training_stats.total_samples = 10  # New lesson
    curriculum.stats["experienced_lesson"].training_stats.smoothed_success = 0.5
    curriculum.stats["experienced_lesson"].training_stats.total_samples = 150  # Experienced

    weights = curriculum.compute_sampling_weights()

    # New lesson should have higher weight due to exploration bonus
    assert weights["new_lesson"] > weights["experienced_lesson"]


def test_exploration_bonus_decay():
    """Test that exploration bonus decays with increasing samples."""
    config = CurriculumConfig(
        lessons={
            "varying_lesson": LessonConfig(
                lesson_id="varying_lesson",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            ),
            "reference_lesson": LessonConfig(
                lesson_id="reference_lesson",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
            ),
        }
    )

    curriculum = Curriculum(config)

    # Keep reference lesson constant with many samples
    curriculum.stats["reference_lesson"].training_stats.smoothed_success = 0.5
    curriculum.stats["reference_lesson"].training_stats.total_samples = 500

    # Track how varying_lesson's weight changes relative to reference
    sample_counts = [1, 10, 50, 100, 200]
    relative_weights = []

    for count in sample_counts:
        curriculum.stats["varying_lesson"].training_stats.smoothed_success = 0.5
        curriculum.stats["varying_lesson"].training_stats.total_samples = count
        weight_dict = curriculum.compute_sampling_weights()
        # Track ratio of varying to reference
        relative_weights.append(weight_dict["varying_lesson"] / weight_dict["reference_lesson"])

    # Relative weight should decrease as samples increase (bonus decays)
    assert relative_weights[0] > relative_weights[1]
    assert relative_weights[1] > relative_weights[2]


def test_exploration_bonus_converges():
    """Test that exploration bonus converges to minimal effect at high sample counts."""
    config = CurriculumConfig(
        lessons={
            "lesson1": LessonConfig(
                lesson_id="lesson1",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            ),
            "lesson2": LessonConfig(
                lesson_id="lesson2",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
            ),
        }
    )

    curriculum = Curriculum(config)

    # Both lessons with high sample counts should have similar weights
    curriculum.stats["lesson1"].training_stats.smoothed_success = 0.5
    curriculum.stats["lesson1"].training_stats.total_samples = 500
    curriculum.stats["lesson2"].training_stats.smoothed_success = 0.5
    curriculum.stats["lesson2"].training_stats.total_samples = 600

    weights = curriculum.compute_sampling_weights()

    # Weights should be very similar (exploration bonus is negligible)
    assert abs(weights["lesson1"] - weights["lesson2"]) < 0.05


def test_checkpoint_save_and_load(tmp_path):
    """Test that curriculum state can be saved and restored."""
    checkpoint_dir = tmp_path / "checkpoints"

    config = CurriculumConfig(
        lessons={
            "lesson1": LessonConfig(
                lesson_id="lesson1",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            ),
            "lesson2": LessonConfig(
                lesson_id="lesson2",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
                dependencies=[LessonDependency(dependency_id="lesson1", reward_threshold=0.5)],
            ),
        },
    )

    # Create curriculum and modify state
    curriculum = Curriculum(config)
    curriculum.current_step = 42
    curriculum.stats["lesson1"].training_stats = PerformanceStats(
        total_samples=100,
        reward_history=np.array([0.7, 0.8, 0.9]),
        last_update_step=42,
    )
    curriculum.stats["lesson1"].eval_stats = PerformanceStats(
        total_samples=10,
        reward_history=np.array([0.75, 0.80]),
        last_update_step=40,
    )
    curriculum.unlocked.add("lesson2")

    # Save checkpoint
    curriculum.save_checkpoint(str(checkpoint_dir))

    # Create new curriculum and restore from checkpoint
    restored = Curriculum(config)
    restored.restore_checkpoint(str(checkpoint_dir))

    # Verify all state was restored
    assert restored.current_step == 42
    assert restored.stats["lesson1"].training_stats.total_samples == 100
    assert np.array_equal(restored.stats["lesson1"].training_stats.reward_history, np.array([0.7, 0.8, 0.9]))
    assert restored.stats["lesson1"].training_stats.last_update_step == 42
    assert restored.stats["lesson1"].eval_stats.total_samples == 10
    assert np.array_equal(restored.stats["lesson1"].eval_stats.reward_history, np.array([0.75, 0.80]))
    assert restored.stats["lesson1"].eval_stats.last_update_step == 40
    assert "lesson1" in restored.unlocked
    assert "lesson2" in restored.unlocked


def test_checkpoint_without_checkpoint_dir(tmp_path):
    """Test that restore_checkpoint returns early when no checkpoint exists."""
    config = CurriculumConfig(
        lessons={
            "lesson1": LessonConfig(
                lesson_id="lesson1",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            )
        },
    )

    curriculum = Curriculum(config)
    checkpoint_dir = str(tmp_path / "nonexistent")

    # Should not raise when checkpoint doesn't exist, just log and return
    curriculum.restore_checkpoint(checkpoint_dir)

    # State should remain unchanged (at step 0)
    assert curriculum.current_step == 0


def test_checkpoint_preserves_reward_history(tmp_path):
    """Test that reward history is preserved through checkpoint."""
    checkpoint_dir = tmp_path / "checkpoints"

    config = CurriculumConfig(
        lessons={
            "lesson": LessonConfig(
                lesson_id="lesson",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            )
        },
    )

    curriculum = Curriculum(config)

    # Add reward history
    curriculum.stats["lesson"].training_stats = PerformanceStats(
        total_samples=5,
        reward_history=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        last_update_step=5,
    )

    # Save and restore
    curriculum.save_checkpoint(str(checkpoint_dir))
    restored = Curriculum(config)
    restored.restore_checkpoint(str(checkpoint_dir))

    # Verify reward history
    assert np.array_equal(restored.stats["lesson"].training_stats.reward_history, np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert restored.stats["lesson"].training_stats.total_samples == 5


def test_checkpoint_graduated_lessons(tmp_path):
    """Test that graduated lessons are preserved through checkpoint."""
    checkpoint_dir = tmp_path / "checkpoints"

    config = CurriculumConfig(
        lessons={
            "lesson1": LessonConfig(
                lesson_id="lesson1",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats"},
                ),
            ),
            "lesson2": LessonConfig(
                lesson_id="lesson2",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "addition"},
                ),
            ),
        },
    )

    curriculum = Curriculum(config)
    curriculum.graduated.add("lesson1")

    # Save and restore
    curriculum.save_checkpoint(str(checkpoint_dir))
    restored = Curriculum(config)
    restored.restore_checkpoint(str(checkpoint_dir))

    # Verify graduation state
    assert "lesson1" in restored.graduated
    assert "lesson2" not in restored.graduated


def test_rollout_stats_dataclass():
    """Test RolloutStats dataclass creation and serialization."""
    rollout_stats = RolloutStats(lesson_id="test_lesson", episode_reward=1.5, env_example_id="example_123")

    assert rollout_stats.lesson_id == "test_lesson"
    assert rollout_stats.episode_reward == 1.5
    assert rollout_stats.env_example_id == "example_123"

    # Test that it's serializable (important for Ray)
    from dataclasses import asdict

    stats_dict = asdict(rollout_stats)
    assert stats_dict["lesson_id"] == "test_lesson"
    assert stats_dict["episode_reward"] == 1.5

    # Test reconstruction
    reconstructed = RolloutStats(**stats_dict)
    assert reconstructed.lesson_id == rollout_stats.lesson_id
    assert reconstructed.episode_reward == rollout_stats.episode_reward


def test_curriculum_update_lesson_stats():
    """Test update_lesson_stats method."""
    config = CurriculumConfig(
        lessons={
            "test_lesson": LessonConfig(
                lesson_id="test_lesson",
                env_config=EnvConfig(env_class="marin.rl.environments.mock_env.MockEnv", env_args={"task_type": "cats"}),
            )
        }
    )

    curriculum = Curriculum(config)

    # Update stats via method (training mode)
    rollout_stats = create_test_rollout_stats(episode_reward=1.0, lesson_id="test_lesson")
    curriculum.update_lesson_stats([rollout_stats], mode="training", current_step=1)

    assert curriculum.stats["test_lesson"].training_stats.total_samples == 1
    assert len(curriculum.stats["test_lesson"].training_stats.reward_history) == 1
    assert curriculum.stats["test_lesson"].training_stats.reward_history[0] == 1.0

    # Update eval stats
    eval_stats = create_test_rollout_stats(episode_reward=0.9, lesson_id="test_lesson")
    curriculum.update_lesson_stats([eval_stats], mode="eval", current_step=100)

    assert curriculum.stats["test_lesson"].eval_stats.total_samples == 1
    assert len(curriculum.stats["test_lesson"].eval_stats.reward_history) == 1
    assert curriculum.stats["test_lesson"].eval_stats.reward_history[0] == 0.9
    assert curriculum.stats["test_lesson"].eval_stats.last_update_step == 100


def test_curriculum_get_metrics():
    """Test get_metrics method."""
    config = CurriculumConfig(
        lessons={
            "lesson1": LessonConfig(
                lesson_id="lesson1",
                env_config=EnvConfig(env_class="marin.rl.environments.mock_env.MockEnv", env_args={"task_type": "cats"}),
            ),
            "lesson2": LessonConfig(
                lesson_id="lesson2",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv", env_args={"task_type": "addition"}
                ),
            ),
        }
    )

    curriculum = Curriculum(config)
    curriculum.current_step = 42

    metrics = curriculum.get_metrics()

    assert metrics["step"] == 42
    assert metrics["total_lessons"] == 2
    assert metrics["unlocked_lessons"] == 2  # Both unlocked by default
    assert metrics["active_lessons"] == 2
    assert metrics["graduated_lessons"] == 0
    assert "sampling_entropy" in metrics
    assert "effective_lessons" in metrics
    assert "sampling_weights" in metrics
