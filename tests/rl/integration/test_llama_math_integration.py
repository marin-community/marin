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

"""Integration test for Llama-3.2-1B with math curriculum."""

import dataclasses
import datetime
import logging
import os

import jmp
from levanter.utils.mesh import MeshConfig
import pytest
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import RayConfig
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig
from transformers import AutoConfig, AutoTokenizer

from marin.rl.curriculum import CurriculumConfig, LessonConfig, SamplingParams
from marin.rl.environments import EnvConfig
from marin.rl.replay_buffer import ReplayBufferConfig
from marin.rl.rl_job import RLJob, RLJobConfig, TrainParams
from marin.rl.rl_losses import RLOOLoss
from marin.rl.weight_transfer import WeightTransferConfig, WeightTransferMode
from tests.rl.integration.config import (
    RolloutWorkerRunner,
    create_test_rollout_storage_config,
)

pytestmark = pytest.mark.skipif(os.environ.get("CI"), reason="Skipping integration tests on CI environment")

logger = logging.getLogger(__name__)

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MAX_TOKENS = 16


def get_stop_tokens(tokenizer_name: str):
    """Get stop tokens from tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return [tokenizer.eos_token_id]


def create_simple_math_curriculum(run_id: str) -> CurriculumConfig:
    """Create simplified math curriculum for testing."""
    default_sampling = SamplingParams(
        temperature=1.0,
        n_prompts=1,
        n_generations_per_prompt=1,
        max_tokens=MAX_TOKENS,
        stop_tokens=get_stop_tokens(MODEL_NAME),
    )

    lessons = {
        "number_comparison": LessonConfig(
            lesson_id="number_comparison",
            env_config=EnvConfig(
                env_class="marin.rl.environments.mock_env.MockEnv",
                env_args={"task_type": "number_comparison", "seed": 42},
            ),
            dependencies=[],
            sampling_params=default_sampling,
        ),
    }

    return CurriculumConfig(
        lessons=lessons,
        eval_frequency=100,
        eval_n_examples=4,
        micro_eval_frequency=10,
        micro_eval_n_examples=1,
        actor_name=f"test-curriculum-{run_id}",
    )


@pytest.mark.slow("Integration test with real model")
def test_llama_math_integration(tmp_path):
    """Test full integration with Llama-3.2-1B and math curriculum.

    Runs both training and rollout workers for 10 training steps to validate
    the complete pipeline with a real model.
    """
    rollout_storage_config = create_test_rollout_storage_config()
    target_steps = 10

    # Load model config from HuggingFace
    hf_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config = LlamaConfig.from_hf_config(hf_config)
    model_config = dataclasses.replace(model_config, max_seq_len=MAX_TOKENS, tokenizer=MODEL_NAME)

    # Create trainer config
    trainer_config = TrainerConfig(
        tracker=JsonLoggerConfig(),
        log_xla_hlo=False,
        log_jaxprs=False,
        mp=jmp.get_policy("p=bfloat16,c=bfloat16"),
        train_batch_size=16,
        per_device_parallelism=4,
        num_train_steps=target_steps,
        steps_per_eval=100,
        checkpointer=CheckpointerConfig(
            base_path=tmp_path / "checkpoints",
            save_interval=datetime.timedelta(seconds=600),
        ),
        mesh=MeshConfig(axes={"model": 2}),
        ray=RayConfig(auto_start_cluster=False),
    )

    # Create optimizer config
    opt_config = AdamConfig(
        learning_rate=1e-7,
        weight_decay=1e-2,
        warmup=0,
        lr_schedule="constant",
    )

    # Weight transfer config
    weight_transfer = WeightTransferConfig(
        mode=WeightTransferMode.ARROW_FLIGHT,
        sync_interval_steps=1,
        max_weight_transfer_wait_time=0.0,
    )

    # Create curriculum
    curriculum_config = create_simple_math_curriculum("llama-test")

    # Create RLJobConfig
    job_config = RLJobConfig(
        model=model_config,
        trainer=trainer_config,
        train_params=TrainParams(
            optimizer=opt_config,
            rl_loss=RLOOLoss(kl_coef=0.01, clip_epsilon=0.2),
            replay_buffer=ReplayBufferConfig(
                capacity=2048,
                alpha=3.0,
                max_samples=1,
                max_rollout_step_delay=1,
            ),
        ),
        curriculum=curriculum_config,
        tokenizer=MODEL_NAME,
        initial_checkpoint=MODEL_NAME,
        rollout_storage=rollout_storage_config,
        weight_transfer=weight_transfer,
        run_id="llama-test-integration",
        log_freq=1,
        inference_type="levanter",
    )

    job = RLJob(job_config)

    # training_runner = TrainWorkerRunner.from_job(job)
    inference_runner = RolloutWorkerRunner.from_job(job)

    with inference_runner:
        inference_runner.wait_for_result()
