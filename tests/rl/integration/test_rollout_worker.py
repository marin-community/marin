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

"""Test rollout worker in isolation."""

import os
import time

import pytest

from marin.rl.rl_job import RLJob, RLJobConfig, TrainParams
from marin.rl.rl_losses import RLOOLoss
from tests.rl.integration.config import (
    DummyTokenizer,
    RolloutWorkerRunner,
    create_nano_llama_config,
    create_nano_optimizer_config,
    create_nano_trainer_config,
    create_test_curriculum_config,
    create_test_rollout_storage_config,
)

pytestmark = pytest.mark.skipif(os.environ.get("CI"), reason="Skipping integration tests on CI environment")


@pytest.mark.slow("Integration test.")
def test_rollout_worker(tmp_path):
    """Test inference worker generates rollouts to in-memory queue."""
    # Use unbounded queue since we're reading all at the end
    rollout_storage_config = create_test_rollout_storage_config()
    rollout_storage_config.queue_maxlen = None
    queue_reader = rollout_storage_config.create_reader()

    trainer_config = create_nano_trainer_config(tmp_path)

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
    runner = RolloutWorkerRunner.from_job(job)
    runner.rollout_worker_config.max_rollouts = 10

    with runner:
        while runner.alive() and not runner.done.is_set():
            time.sleep(0.5)

        # Give a moment for final writes
        time.sleep(0.5)

        batches = queue_reader.read_all_available()
        assert len(batches) > 0, "Should be able to read batches from queue"
        assert runner.rollouts_generated >= 1, f"Expected at least 1 rollout, got {runner.rollouts_generated}"

    print("Rollout worker generated rollout batch successfully")
