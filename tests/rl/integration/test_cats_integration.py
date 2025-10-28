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

"""End-to-end integration tests for cats task."""

import logging
import os
import time
from pathlib import Path

import numpy as np
import pytest

from marin.rl.replay_buffer import ReplayBufferConfig
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
    create_test_curriculum_config,
    create_test_rollout_storage_config,
    create_vllm_inference_config,
    create_qwen_config,
    create_qwen_tokenizer,
)
from tests.rl.integration.tasks import create_cats_rollout_batch, validate_cats_model

pytestmark = pytest.mark.skipif(os.environ.get("CI"), reason="Skipping integration tests on CI environment")

logger = logging.getLogger(__name__)


@pytest.mark.slow("Integration test with training loop")
def test_train_worker_with_manual_cats_rollout(ray_tpu_cluster, tmp_path):
    """Test training worker with manually constructed cat-themed rollout batches.

    This test validates that the training worker can process rollout batches
    with varying rewards and learn to prefer high-reward (cat-heavy) responses.
    """
    rollout_storage_config = create_test_rollout_storage_config()
    target_steps = 20
    tokenizer = DummyTokenizer()

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
            replay_buffer=ReplayBufferConfig(
                capacity=2048,
                alpha=3.0,
                max_samples=4,
                max_rollout_step_delay=1,
            ),
        ),
        curriculum=create_test_curriculum_config(),
        tokenizer=tokenizer,
        rollout_storage=rollout_storage_config,
        inference_type="levanter",
    )

    job = RLJob(job_config)

    with TrainWorkerRunner.from_job(job) as runner:
        queue_writer = runner.training_worker_config.rollout_storage.create_writer()

        with RolloutBatchFeeder(
            runner=runner,
            batch_generator=create_cats_rollout_batch,
            queue_writer=queue_writer,
            tokenizer=tokenizer,
        ):
            runner.done.wait()

        assert all(not np.isnan(loss) for loss in runner.losses), "Loss should not be NaN"
        assert all(loss < 10.0 for loss in runner.losses), f"Loss should be reasonable, got {runner.losses}"

        checkpoint_base = Path(runner.training_worker_config.trainer.checkpointer.base_path)
        checkpoint_dirs = list(checkpoint_base.glob("*/*"))
        assert len(checkpoint_dirs) >= 1, f"Expected at least 1 checkpoint, got {len(checkpoint_dirs)}"

    print(f"  - Steps completed: {runner.steps_completed}")
    print(f"  - Loss progression: {runner.losses}")
    print(f"  - Initial loss: {runner.losses[0]:.4f}")
    print(f"  - Final loss: {runner.losses[-1]:.4f}")

    # Test the trained model with example prompts
    validate_cats_model(runner.trained_model, DummyTokenizer())


@pytest.mark.slow("Long-running integration test.")
def test_full_integration_moar_cats(ray_tpu_cluster, tmp_path):
    """Long-running test to validate environment objective improves over time."""
    rollout_storage_config = create_test_rollout_storage_config()
    target_steps = 20

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
            replay_buffer=ReplayBufferConfig(
                capacity=4096,
                alpha=3.0,
                max_samples=1,
                max_rollout_step_delay=1,
            ),
        ),
        curriculum=create_test_curriculum_config(),
        tokenizer=DummyTokenizer(),
        rollout_storage=rollout_storage_config,
        inference_type="levanter",
    )

    job = RLJob(job_config)

    training_runner = TrainWorkerRunner.from_job(job)
    inference_runner = RolloutWorkerRunner.from_job(job)

    # Apply test-specific overrides
    inference_runner.rollout_worker_config.weight_transfer.sync_interval_steps = 1

    with training_runner:
        while training_runner.reference_model is None:
            time.sleep(0.1)
        with inference_runner:
            training_runner.done.wait()

    assert (
        inference_runner.rollouts_generated >= 5
    ), f"Expected at least 5 rollouts, got {inference_runner.rollouts_generated}"
    assert (
        training_runner.steps_completed >= 2
    ), f"Expected at least 2 training steps, got {training_runner.steps_completed}"

    assert inference_runner.weight_transfers >= 1, "Should have at least one weight transfer during long run"

    validate_cats_model(training_runner.trained_model, DummyTokenizer())


@pytest.mark.slow("Long-running integration test.")
def test_full_integration_moar_cats_vllm(ray_tpu_cluster, tmp_path):
    """Long-running test to validate environment objective improves over time."""
    rollout_storage_config = create_test_rollout_storage_config()
    target_steps = 20

    # Create trainer config with target steps
    trainer_config = create_nano_trainer_config(tmp_path)
    trainer_config.num_train_steps = target_steps

    # Create RLJobConfig and get worker configs
    job_config = RLJobConfig(
        model=create_qwen_config(),
        trainer=trainer_config,
        train_params=TrainParams(
            optimizer=create_nano_optimizer_config(),
            rl_loss=RLOOLoss(kl_coef=0.0, clip_epsilon=0.2),
            replay_buffer=ReplayBufferConfig(
                capacity=4096,
                alpha=3.0,
                max_samples=1,
                max_rollout_step_delay=1,
            ),
        ),
        curriculum=create_test_curriculum_config(),
        tokenizer=create_qwen_tokenizer(),
        rollout_storage=rollout_storage_config,
        inference_type="vllm",
        inference_config=create_vllm_inference_config(),
    )

    job = RLJob(job_config)

    training_runner = TrainWorkerRunner.from_job(job)
    inference_runner = RolloutWorkerRunner.from_job(job)

    # Apply test-specific overrides
    inference_runner.rollout_worker_config.weight_transfer.sync_interval_steps = 1

    with training_runner:
        while training_runner.reference_model is None:
            time.sleep(0.1)
        with inference_runner:
            training_runner.done.wait()

    assert (
        inference_runner.rollouts_generated >= 5
    ), f"Expected at least 5 rollouts, got {inference_runner.rollouts_generated}"
    assert (
        training_runner.steps_completed >= 2
    ), f"Expected at least 2 training steps, got {training_runner.steps_completed}"

    assert inference_runner.weight_transfers >= 1, "Should have at least one weight transfer during long run"

    validate_cats_model(training_runner.trained_model, create_qwen_tokenizer())
