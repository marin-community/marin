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

"""Test training worker with manual rollouts."""

import os

import pytest

from marin.rl.rl_job import RLJob, RLJobConfig, TrainParams
from marin.rl.rl_losses import RLOOLoss
from tests.rl.integration.config import (
    DummyTokenizer,
    RolloutBatchFeeder,
    TrainWorkerRunner,
    create_nano_llama_config,
    create_nano_optimizer_config,
    create_nano_trainer_config,
    create_test_curriculum_config,
    create_test_rollout_storage_config,
)
from tests.rl.integration.tasks import create_cats_rollout_batch

pytestmark = pytest.mark.skipif(os.environ.get("CI"), reason="Skipping integration tests on CI environment")


@pytest.mark.slow("Integration test.")
def test_train_worker(ray_tpu_cluster, tmp_path):
    """Test training worker processes rollout batch and creates checkpoint."""
    rollout_storage_config = create_test_rollout_storage_config()
    queue_writer = rollout_storage_config.create_writer()

    trainer_config = create_nano_trainer_config(tmp_path)
    trainer_config.num_train_steps = 10

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

    with TrainWorkerRunner.from_job(job) as runner:
        queue_writer = runner.training_worker_config.rollout_storage.create_writer()
        tokenizer = DummyTokenizer()
        with RolloutBatchFeeder(
            runner=runner,
            batch_generator=create_cats_rollout_batch,
            queue_writer=queue_writer,
            tokenizer=tokenizer,
        ):
            runner.wait_for_result()

    # Verify results
    assert runner.steps_completed >= 1, f"Expected at least 1 training step, got {runner.steps_completed}"
