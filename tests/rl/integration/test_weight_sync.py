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

"""Test rollout and training workers with weight synchronization."""

import os
import time

import pytest

from marin.rl.rl_job import RLJob, RLJobConfig, TrainParams
from marin.rl.rl_losses import RLOOLoss
from tests.rl.integration.config import (
    DummyTokenizer,
    RolloutWorkerRunner,
    TrainWorkerRunner,
    create_nano_llama_config,
    create_nano_optimizer_config,
    create_nano_trainer_config,
    create_test_curriculum_config,
    create_test_rollout_storage_config,
)

pytestmark = pytest.mark.skipif(os.environ.get("CI"), reason="Skipping integration tests on CI environment")


@pytest.mark.slow("Integration test.")
def test_rollout_and_train_workers(ray_tpu_cluster, tmp_path):
    """Test inference & training workers running together with checkpoint updates."""
    rollout_storage_config = create_test_rollout_storage_config()

    trainer_config = create_nano_trainer_config(tmp_path)
    trainer_config.num_train_steps = 100

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
    )

    job = RLJob(job_config)

    training_runner = TrainWorkerRunner.from_job(job)
    rollout_runner = RolloutWorkerRunner.from_job(job)

    # Apply test-specific overrides
    rollout_runner.rollout_worker_config.weight_transfer.sync_interval_steps = 1
    rollout_runner.rollout_worker_config.max_rollouts = 100

    with training_runner:
        time.sleep(1)
        with rollout_runner:
            start_time = time.time()
            while time.time() - start_time < 60:
                if training_runner.done.is_set() and not rollout_runner.done.is_set():
                    rollout_runner.stop()
                    break

                if rollout_runner.done.is_set() and training_runner.done.is_set():
                    training_runner.stop()
                    break

                time.sleep(1)

    assert (
        rollout_runner.rollouts_generated >= 1
    ), f"Expected at least 1 rollouts, got {rollout_runner.rollouts_generated}"
    assert (
        training_runner.steps_completed >= 0
    ), f"Expected at least 0 training steps, got {training_runner.steps_completed}"

    print(f"Weight transfers detected: {rollout_runner.weight_transfers}")
    assert rollout_runner.weight_transfers >= 1, "Expected at least 1 weight transfer"
    assert rollout_runner.rollouts_generated > 0, "Should have generated at least one rollout"
