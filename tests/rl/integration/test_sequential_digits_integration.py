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

"""End-to-end integration test for sequential digits task."""

import logging
import os
import time
import uuid

import numpy as np
import pytest

from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
from marin.rl.environments import EnvConfig
from marin.rl.rl_job import RLJob, RLJobConfig, TrainParams
from marin.rl.rl_losses import RLOOLoss
from tests.rl.integration.config import (
    DummyTokenizer,
    RolloutBatchFeeder,
    RolloutWorkerRunner,
    TrainWorkerRunner,
    create_nano_llama_config,
    create_nano_optimizer_config,
    create_nano_trainer_config,
    create_test_rollout_storage_config,
)
from tests.rl.integration.tasks import create_sequential_digits_rollout_batch, validate_sequential_digits_model

pytestmark = pytest.mark.skipif(os.environ.get("CI"), reason="Skipping integration tests on CI environment")

logger = logging.getLogger(__name__)


@pytest.mark.slow("Integration test with training loop")
def test_train_worker_with_sequential_digits(ray_tpu_cluster, tmp_path):
    """Test training worker learns to generate sequential digits.

    This test validates that the training worker can learn a simple pattern
    requiring the model to produce digits in sequential order (0,1,2,3...).
    """
    rollout_storage_config = create_test_rollout_storage_config()
    target_steps = 500
    tokenizer = DummyTokenizer()

    # Create curriculum with sequential digits task
    test_id = uuid.uuid4().hex[:8]
    curriculum_config = CurriculumConfig(
        lessons={
            "sequential_digits": LessonConfig(
                lesson_id="sequential_digits",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "sequential_digits", "seed": 42, "difficulty": "medium"},
                ),
                sampling_params=SamplingParams(temperature=1.0, n_prompts=8, n_generations_per_prompt=4, max_tokens=64),
            )
        },
        eval_frequency=100,
        actor_name=f"test_curriculum_seq_{test_id}",
    )

    # Create trainer config with target steps
    trainer_config = create_nano_trainer_config(tmp_path)
    trainer_config.num_train_steps = target_steps

    # Create RLJobConfig and get worker configs
    job_config = RLJobConfig(
        model=create_nano_llama_config(),
        trainer=trainer_config,
        train_params=TrainParams(
            optimizer=create_nano_optimizer_config(),
            rl_loss=RLOOLoss(kl_coef=0.0, clip_epsilon=0.2),
            replay_buffer_capacity=2048,
            replay_buffer_alpha=3.0,
            max_samples_per_rollout=1,
            max_rollout_delay=1,
        ),
        curriculum=curriculum_config,
        tokenizer=tokenizer,
        rollout_storage=rollout_storage_config,
    )

    job = RLJob(job_config)

    with TrainWorkerRunner.from_job(job) as runner:
        queue_writer = runner.training_worker_config.rollout_storage.create_writer()

        with RolloutBatchFeeder(
            runner=runner,
            batch_generator=create_sequential_digits_rollout_batch,
            queue_writer=queue_writer,
            tokenizer=tokenizer,
        ):
            runner.done.wait()

    # Validate training completed successfully
    assert all(not np.isnan(loss) for loss in runner.losses), "Loss should not be NaN"
    assert all(loss < 10.0 for loss in runner.losses), f"Loss should be reasonable, got {runner.losses}"

    print(f"\n{'=' * 60}")
    print("Training Statistics:")
    print(f"  Steps completed: {runner.steps_completed}")
    print(f"  Initial loss: {runner.losses[0]:.4f}")
    print(f"  Final loss: {runner.losses[-1]:.4f}")
    print(f"  Loss improvement: {runner.losses[0] - runner.losses[-1]:.4f}")
    print(f"{'=' * 60}")

    # Test the trained model
    validate_sequential_digits_model(runner.trained_model, tokenizer)


@pytest.mark.slow("Long-running integration test.")
def test_full_integration_sequential_digits(ray_tpu_cluster, tmp_path):
    """Full integration test with rollout and train workers for sequential digits task."""
    rollout_storage_config = create_test_rollout_storage_config()
    target_steps = 100

    # Create curriculum with sequential digits task
    test_id = uuid.uuid4().hex[:8]
    curriculum_config = CurriculumConfig(
        lessons={
            "sequential_digits": LessonConfig(
                lesson_id="sequential_digits",
                env_config=EnvConfig(
                    env_class="marin.rl.environments.mock_env.MockEnv",
                    env_args={"task_type": "sequential_digits", "seed": 42, "difficulty": "medium"},
                ),
                sampling_params=SamplingParams(temperature=1.0, n_prompts=8, n_generations_per_prompt=4, max_tokens=64),
            )
        },
        eval_frequency=100,
        actor_name=f"test_curriculum_seq_{test_id}",
    )

    # Create trainer config with target steps
    trainer_config = create_nano_trainer_config(tmp_path)
    trainer_config.num_train_steps = target_steps

    # Create RLJobConfig
    job_config = RLJobConfig(
        model=create_nano_llama_config(),
        trainer=trainer_config,
        train_params=TrainParams(
            optimizer=create_nano_optimizer_config(),
            rl_loss=RLOOLoss(kl_coef=0.0, clip_epsilon=0.2),
            max_samples_per_rollout=1,
            max_rollout_delay=1,
        ),
        curriculum=curriculum_config,
        tokenizer=DummyTokenizer(),
        rollout_storage=rollout_storage_config,
    )

    job = RLJob(job_config)

    training_runner = TrainWorkerRunner.from_job(job)
    inference_runner = RolloutWorkerRunner.from_job(job)

    # Apply test-specific overrides
    inference_runner.rollout_worker_config.weight_transfer.sync_interval_steps = 1
    inference_runner.rollout_worker_config.max_rollouts = 200

    with training_runner, inference_runner:
        while not training_runner.done.is_set():
            training_runner.done.wait(timeout=1)

    assert inference_runner.rollouts_generated >= 5, (
        f"Expected at least 5 rollouts, got {inference_runner.rollouts_generated}"
    )
    assert training_runner.steps_completed >= 2, (
        f"Expected at least 2 training steps, got {training_runner.steps_completed}"
    )

    assert inference_runner.weight_transfers >= 1, "Should have at least one weight transfer during long run"

    validate_sequential_digits_model(training_runner.trained_model, DummyTokenizer())
