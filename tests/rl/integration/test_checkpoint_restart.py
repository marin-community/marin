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

"""Test checkpoint saving and resuming."""

import os
import time
from datetime import timedelta
from pathlib import Path

import pytest

from marin.rl.rl_job import RLJob, RLJobConfig, TrainParams
from marin.rl.rl_losses import RLOOLoss
from tests.rl.integration.config import (
    DummyTokenizer,
    TrainWorkerRunner,
    create_nano_llama_config,
    create_nano_optimizer_config,
    create_nano_trainer_config,
    create_test_curriculum_config,
    create_test_rollout_storage_config,
)
from tests.rl.integration.tasks import create_cats_rollout_batch

pytestmark = pytest.mark.skipif(os.environ.get("CI"), reason="Skipping integration tests on CI environment")


@pytest.mark.slow("Integration test with checkpoint restart")
def test_train_worker_checkpoint_restart(ray_tpu_cluster, tmp_path):
    """Test that training worker correctly restarts from checkpoint without repeating steps."""
    rollout_storage_config = create_test_rollout_storage_config()
    queue_writer = rollout_storage_config.create_writer()

    # Phase 1: Initial training run - small number of steps
    initial_target_steps = 5
    trainer_config = create_nano_trainer_config(tmp_path)
    trainer_config.num_train_steps = initial_target_steps
    trainer_config.checkpointer.save_interval = timedelta(milliseconds=100)

    job_config = RLJobConfig(
        model=create_nano_llama_config(),
        trainer=trainer_config,
        train_params=TrainParams(
            optimizer=create_nano_optimizer_config(),
            rl_loss=RLOOLoss(kl_coef=0.0, clip_epsilon=0.2),
        ),
        curriculum=create_test_curriculum_config(),
        tokenizer=DummyTokenizer(),
        rollout_storage=rollout_storage_config,
        run_id="test-0",
    )

    job = RLJob(job_config)

    with TrainWorkerRunner.from_job(job) as runner:
        tokenizer = DummyTokenizer()
        batch_size = runner.training_worker_config.trainer.train_batch_size
        # Wait for worker to initialize
        while not runner.worker:
            time.sleep(0.1)

        # Add some training data
        for _ in range(5):
            batch = create_cats_rollout_batch(
                policy_model=runner.reference_model,
                batch_size=batch_size,
                tokenizer=tokenizer,
            )
            queue_writer.write_batch(batch)

        # Wait for completion or timeout
        start_time = time.time()
        while runner.alive() and not runner.done.is_set() and time.time() - start_time < 30:
            time.sleep(0.5)

        first_run_steps = runner.all_steps_seen.copy()
        last_step_first_run = runner.steps_completed

        # Verify we trained and created checkpoint
        assert (
            last_step_first_run >= initial_target_steps
        ), f"Expected >= {initial_target_steps} steps, got {last_step_first_run}"
        checkpoint_dir = Path(runner.training_worker_config.trainer.checkpointer.expanded_path("test-0-train"))
        assert checkpoint_dir.exists(), f"Checkpoint directory {checkpoint_dir} does not exist"
        checkpoints = list(checkpoint_dir.glob("*"))
        assert len(checkpoints) > 0, f"No checkpoints found in {checkpoint_dir}"

        print(f"First run completed {last_step_first_run} steps, found {len(checkpoints)} checkpoints")

    # Phase 2: Restart training - should auto-load checkpoint
    trainer_config.num_train_steps = 10  # Continue to step 10

    job_config2 = RLJobConfig(
        model=create_nano_llama_config(),
        trainer=trainer_config,
        train_params=TrainParams(
            optimizer=create_nano_optimizer_config(),
            rl_loss=RLOOLoss(kl_coef=0.0, clip_epsilon=0.2),
        ),
        curriculum=create_test_curriculum_config(),
        tokenizer=DummyTokenizer(),
        rollout_storage=rollout_storage_config,
        run_id="test-0",
    )

    job2 = RLJob(job_config2)

    with TrainWorkerRunner.from_job(job2) as runner:
        # Wait for worker to initialize
        while not runner.worker:
            time.sleep(0.1)

        # Add more training data
        for _ in range(5):
            batch = create_cats_rollout_batch(
                policy_model=runner.reference_model,
                batch_size=batch_size,
                tokenizer=tokenizer,
            )
            queue_writer.write_batch(batch)

        # Wait for completion or timeout
        start_time = time.time()
        while runner.alive() and not runner.done.is_set() and time.time() - start_time < 30:
            time.sleep(0.5)

    second_run_steps = runner.all_steps_seen

    # We should never see step 0 in the second run
    assert 0 not in second_run_steps, f"Step 0 seen in second run! Steps: {second_run_steps}"

    # Second run should start from a checkpoint (step > 1)
    min_step_second_run = min(second_run_steps)
    assert min_step_second_run > 1, f"Second run should restart from checkpoint (step > 1), got {min_step_second_run}"

    # Some overlap is expected when resuming from checkpoint, but verify proper restart
    max_step_second_run = max(second_run_steps)
    max_step_first_run = max(first_run_steps) if first_run_steps else 0
    assert (
        max_step_second_run > max_step_first_run
    ), f"Second run should progress beyond first run: first max={max_step_first_run}, second max={max_step_second_run}"
