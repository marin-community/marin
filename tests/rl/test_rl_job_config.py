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

"""Tests for RLJob configuration and validation."""

import pytest

from marin.rl.curriculum import CurriculumConfig, LessonConfig, LessonDependency
from marin.rl.environments import EnvConfig
from marin.rl.rl_job import RLJobConfig, RLOOLoss, SamplingParams, TrainParams
from tests.rl.integration_test_config import (
    create_nano_llama_config,
    create_nano_optimizer_config,
    create_nano_trainer_config,
)


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return str(tmp_path)


@pytest.fixture
def minimal_curriculum():
    """Create a minimal curriculum with one lesson."""
    return CurriculumConfig(
        lessons={
            "test_lesson": LessonConfig(
                lesson_id="test_lesson",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats", "seed": 42},
                ),
            )
        },
        eval_frequency=100,
    )


@pytest.fixture
def curriculum_with_dependencies():
    """Create a curriculum with lesson dependencies."""
    return CurriculumConfig(
        lessons={
            "easy": LessonConfig(
                lesson_id="easy",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats", "seed": 42},
                ),
            ),
            "hard": LessonConfig(
                lesson_id="hard",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "cats", "seed": 42},
                ),
                dependencies=[LessonDependency(dependency_id="easy", reward_threshold=0.8)],
            ),
        },
        eval_frequency=100,
    )


def test_rl_job_config_with_custom_loss(temp_dir, minimal_curriculum):
    """Test RLJobConfig with custom RL loss module."""
    custom_loss = RLOOLoss(kl_coef=0.2, clip_epsilon=0.3)

    config = RLJobConfig(
        model=create_nano_llama_config(),
        trainer=create_nano_trainer_config(temp_dir),
        train_params=TrainParams(
            optimizer=create_nano_optimizer_config(),
            num_train_steps=100,
            batch_size=32,
        ),
        curriculum=minimal_curriculum,
        rl_loss=custom_loss,
    )

    assert config.rl_loss.kl_coef == 0.2
    assert config.rl_loss.clip_epsilon == 0.3


def test_per_lesson_sampling_params(temp_dir):
    """Test that per-lesson sampling parameters override global defaults."""
    # Create lessons with different sampling params
    lesson_with_override = LessonConfig(
        lesson_id="custom",
        env_config=EnvConfig(
            env_class="marin.rl.environments.mock_env.MockEnv",
            env_args={"task_type": "cats", "seed": 42},
        ),
        sampling_params=SamplingParams(
            temperature=0.5,
            n_prompts=16,
            n_generations_per_prompt=8,
        ),
    )

    lesson_without_override = LessonConfig(
        lesson_id="default",
        env_config=EnvConfig(
            env_class="marin.rl.environments.mock_env.MockEnv",
            env_args={"task_type": "cats", "seed": 42},
        ),
    )

    curriculum = CurriculumConfig(
        lessons={
            "custom": lesson_with_override,
            "default": lesson_without_override,
        },
        eval_frequency=100,
    )

    # Verify lesson has sampling params
    assert curriculum.lessons["custom"].sampling_params is not None
    assert curriculum.lessons["custom"].sampling_params.temperature == 0.5
    assert curriculum.lessons["custom"].sampling_params.n_prompts == 16
    assert curriculum.lessons["custom"].sampling_params.n_generations_per_prompt == 8

    # Verify other lesson has no override
    assert curriculum.lessons["default"].sampling_params is None


def test_to_worker_configs_produces_valid_configs(temp_dir, minimal_curriculum):
    """Test that to_worker_configs() produces valid TrainWorkerConfig and RolloutWorkerConfig."""
    from marin.rl.rl_job import RLJob

    config = RLJobConfig(
        model=create_nano_llama_config(),
        trainer=create_nano_trainer_config(temp_dir),
        train_params=TrainParams(
            optimizer=create_nano_optimizer_config(),
            num_train_steps=100,
            batch_size=32,
            replay_buffer_capacity=2048,
            replay_buffer_alpha=3.0,
            max_samples_per_rollout=4,
            max_batch_latency=1000,
        ),
        curriculum=minimal_curriculum,
        max_input_length=128,
        max_output_length=128,
    )

    job = RLJob(config)
    train_config, rollout_config = job.to_worker_configs()

    # Verify train worker config
    assert train_config.model == config.model
    assert train_config.trainer == config.trainer
    assert train_config.optimizer == config.train_params.optimizer
    assert train_config.max_input_length == 128
    assert train_config.max_output_length == 128
    assert train_config.replay_buffer.capacity == 2048
    assert train_config.replay_buffer.alpha == 3.0
    assert train_config.replay_buffer.max_samples == 4
    assert train_config.run_id == config.run_id
    assert train_config.curriculum_config == config.curriculum

    # Verify rollout worker config
    assert rollout_config.model == config.model
    assert rollout_config.trainer == config.trainer
    assert rollout_config.max_input_length == 128
    assert rollout_config.max_output_length == 128
    assert rollout_config.n_prompts_per_step == config.eval_sampling_params.n_prompts
    assert rollout_config.n_generations == config.eval_sampling_params.n_generations_per_prompt
    assert rollout_config.temperature == config.eval_sampling_params.temperature
    assert rollout_config.run_id == config.run_id
    assert rollout_config.curriculum_config == config.curriculum

    # Verify shared configs
    assert train_config.rollout_storage == rollout_config.rollout_storage
    assert train_config.weight_transfer == rollout_config.weight_transfer
