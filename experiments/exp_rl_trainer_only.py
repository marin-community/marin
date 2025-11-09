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

"""
Training-Only Benchmark Experiment

This experiment benchmarks pure training throughput without the overhead of:
- Rollout/inference workers
- Weight transfer/synchronization
- Replay buffer
- Curriculum management
- Rollout storage

The worker generates random tokens of fixed length (default 1024) and measures
training throughput (tokens/second) in isolation.

Metrics logged:
- train.step_duration_sec: Total training step duration
- train.tokens_per_second: Training throughput
- train.total_tokens_trained: Cumulative tokens processed
"""

import dataclasses
import datetime
import logging
import os

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import HFCompatConfig
from levanter.distributed import RayConfig
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from transformers import AutoConfig

from marin.execution.executor import (
    ExecutorStep,
    OutputName,
    executor_main,
)
from marin.rl.rl_losses import RLOOLoss
from marin.rl.train_only_job import TrainOnlyJob, TrainOnlyJobConfig, TrainOnlyParams, TrainOnlyRunConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelConfig:
    name: str
    type: str
    tokenizer: str
    checkpoint: str
    config_class: type[HFCompatConfig]

    @property
    def safe_name(self) -> str:
        return self.name.replace("/", "-").lower()


qwen4b = ModelConfig(
    name="Qwen/Qwen3-4B-Instruct-2507",
    type="qwen",
    tokenizer="Qwen/Qwen3-4B-Instruct-2507",
    checkpoint="Qwen/Qwen3-4B-Instruct-2507",
    config_class=Qwen3Config,
)
llama1b = ModelConfig(
    name="meta-llama/Llama-3.2-1B-Instruct",
    type="llama",
    tokenizer="meta-llama/Llama-3.2-1B-Instruct",
    checkpoint="meta-llama/Llama-3.2-1B-Instruct",
    config_class=LlamaConfig,
)
qwen3_1_7b = ModelConfig(
    name="Qwen/Qwen3-1.7B",
    type="qwen",
    tokenizer="Qwen/Qwen3-1.7B",
    checkpoint="Qwen/Qwen3-1.7B",
    config_class=Qwen3Config,
)
qwen3_0_6b = ModelConfig(
    name="Qwen/Qwen3-0.6B",
    type="qwen",
    tokenizer="Qwen/Qwen3-0.6B",
    checkpoint="Qwen/Qwen3-0.6B",
    config_class=Qwen3Config,
)
llama_3_1_8b = ModelConfig(
    name="meta-llama/Llama-3.1-8B-Instruct",
    type="llama",
    tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    checkpoint="meta-llama/Llama-3.1-8B-Instruct",
    config_class=LlamaConfig,
)

# Default configuration
MODEL = llama1b
MAX_TOKENS = 1024


def train_only_benchmark(name: str, experiment_config: ModelConfig) -> ExecutorStep:
    """Create a training-only benchmark executor step."""
    hf_config = AutoConfig.from_pretrained(experiment_config.name)
    config = experiment_config.config_class.from_hf_config(hf_config)

    # Adjust the max sequence length of the model to reduce memory usage.
    model_config = dataclasses.replace(config, seq_len=MAX_TOKENS, tokenizer=experiment_config.tokenizer)

    trainer_config = TrainerConfig(
        tracker=WandbConfig(
            project="rl-train-only-benchmark",
            name=name,
            tags=["train-only", "benchmark", experiment_config.name.split("/")[-1]],
        ),
        log_xla_hlo=False,
        log_jaxprs=False,
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        # Set the train batch size to match the full RL setup for fair comparison
        train_batch_size=64,
        # Microbatch to avoid OOM
        per_device_parallelism=16,
        num_train_steps=50000,
        steps_per_eval=100,
        checkpointer=CheckpointerConfig(
            base_path=OutputName("checkpoints"),
            save_interval=datetime.timedelta(seconds=600),
        ),
        fsdp_axis="embed",
        batch_axis="batch",
        ray=RayConfig(auto_start_cluster=False),
    )

    opt_config = AdamConfig(
        learning_rate=1e-7,
        weight_decay=1e-2,
        warmup=0,
        lr_schedule="constant",
    )

    # Create TrainOnlyJobConfig
    config = TrainOnlyJobConfig(
        model=model_config,
        trainer=trainer_config,
        train_params=TrainOnlyParams(
            optimizer=opt_config,
            rl_loss=RLOOLoss(kl_coef=0.01, clip_epsilon=0.2),
        ),
        tokenizer=experiment_config.tokenizer,
        initial_checkpoint=experiment_config.checkpoint,
        run_id=name,
        log_freq=10,
        sequence_length=1024,  # Fixed length for random tokens
        run_config=TrainOnlyRunConfig(
            train_tpu_type="v5p-8",
            num_train_slices=1,
        ),
    )

    return ExecutorStep(
        name=f"train_only/{name}",
        description=f"Training-only benchmark: {name}",
        fn=TrainOnlyJob.make_step_fn(),
        config=config,
        pip_dependency_groups=["post_training", "vllm"],
    )


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    datestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Use same model as expxxx_vllm_rl.py for fair comparison
    experiment_configs = [llama1b]
    experiments = []
    for experiment_config in experiment_configs:
        model_base_name = experiment_config.name.split("/")[-1].lower()
        experiments.append(
            train_only_benchmark(
                name=f"{model_base_name}-train-only-bsz64-tok1024-{datestamp}",
                experiment_config=experiment_config,
            ),
        )

    executor_main(
        steps=experiments,
        description="Training-only benchmark experiments",
    )


if __name__ == "__main__":
    main()
