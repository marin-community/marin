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

"""Launch script for training worker."""

import json
import logging
from pathlib import Path

import tyro

from .training_config import (
    DistributedConfig,
    EnvironmentConfig,
    GenerationConfig,
    LoggingConfig,
    ModelConfig,
    TrainingConfig,
    TrainingHyperparameters,
    WorkerConfig,
)
from .training_worker import TrainingWorker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(
    # Model configuration
    load_model: str,
    # Hyperparameters
    num_train_steps: int,
    max_input_length: int,
    max_output_length: int,
    train_bsize: int,
    decode_bsize: int,
    prefill_bsize: int,
    reference_logprobs_bsize: int,
    n_prompts_per_step: int,
    # Distributed configuration
    sharding: str,
    # Worker configuration
    rollout_queue_bucket: str,
    # Logging configuration
    wandb_project: str,
    output_dir: str | None = None,
    # Optional model configuration
    inference_param_dtype: str = "bf16",
    inference_activation_dtype: str = "bf16",
    training_param_dtype: str = "fp32",
    training_activation_dtype: str = "fp32",
    model_config_override: str = "{}",
    tokenizer_override: str = "{}",
    train_attention_kernel_config: str = "splash:{}",
    prefill_attention_kernel_config: str = "splash:{}",
    generate_attention_kernel_config: str = "paged:{}",
    # Optional hyperparameters
    optim_config: str = "adamw:{}",
    pad_token_id: int = 128002,
    kl_coef: float = 0.0,
    # Optional logging configuration
    log_freq: int = 1,
    num_eval_examples: int = 0,
    save_model_freq: int = 100,
    logger_config: str = "{}",
    save_initial_checkpoint: bool = False,
    log_initial_step: bool = True,
    max_checkpoints: int | None = None,
    # Optional environment configuration
    train_environments_path: str = "environments.json",
    test_environments_path: str = "environments.json",
    # Optional distributed configuration
    physical_axis_splitting: bool = False,
    jax_distributed_initalize_config: str = "{}",
    # Optional generation configuration
    generation_config: str = "{}",
    test_generation_config: str = "{}",
    # Optional checkpointer configuration
    checkpointer_config: str = "{}",
    # Worker-specific configuration
    rollout_queue_path: str = "rollout_queue",
    checkpoint_sync_interval: int = 100,
    batch_timeout: float = 60.0,
    max_idle_time: float = 300.0,
    checkpoint_bucket: str | None = None,
    checkpoint_path: str = "checkpoints",
):
    """Launch training worker with configuration."""
    # Parse configurations
    sharding_list: list[int] = list(map(lambda x: int(x.strip()), sharding.split(",")))
    model_config_override_dict = json.loads(model_config_override)
    tokenizer_override_dict = json.loads(tokenizer_override)
    logger_config_dict = json.loads(logger_config)
    jax_distributed_initalize_config_dict = json.loads(jax_distributed_initalize_config)
    generation_config_dict = json.loads(generation_config)
    test_generation_config_dict = json.loads(test_generation_config)
    checkpointer_config_dict = json.loads(checkpointer_config)

    # Create training configuration
    training_config = TrainingConfig(
        model=ModelConfig(
            load_model=load_model,
            inference_param_dtype=inference_param_dtype,
            inference_activation_dtype=inference_activation_dtype,
            training_param_dtype=training_param_dtype,
            training_activation_dtype=training_activation_dtype,
            model_config_override=model_config_override_dict,
            tokenizer_override=tokenizer_override_dict,
            train_attention_kernel_config=train_attention_kernel_config,
            prefill_attention_kernel_config=prefill_attention_kernel_config,
            generate_attention_kernel_config=generate_attention_kernel_config,
        ),
        hyperparameters=TrainingHyperparameters(
            num_train_steps=num_train_steps,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            train_bsize=train_bsize,
            decode_bsize=decode_bsize,
            prefill_bsize=prefill_bsize,
            reference_logprobs_bsize=reference_logprobs_bsize,
            n_prompts_per_step=n_prompts_per_step,
            optim_config=optim_config,
            pad_token_id=pad_token_id,
            kl_coef=kl_coef,
        ),
        logging=LoggingConfig(
            log_freq=log_freq,
            num_eval_examples=num_eval_examples,
            save_model_freq=save_model_freq,
            wandb_project=wandb_project,
            logger_config=logger_config_dict,
            save_initial_checkpoint=save_initial_checkpoint,
            log_initial_step=log_initial_step,
            max_checkpoints=max_checkpoints,
        ),
        environment=EnvironmentConfig(
            train_environments_path=train_environments_path,
            test_environments_path=test_environments_path,
        ),
        distributed=DistributedConfig(
            sharding=sharding_list,
            physical_axis_splitting=physical_axis_splitting,
            jax_distributed_initalize_config=jax_distributed_initalize_config_dict,
        ),
        generation=GenerationConfig(
            generation_config=generation_config_dict,
            test_generation_config=test_generation_config_dict,
        ),
        output_dir=output_dir,
        checkpointer_config=checkpointer_config_dict,
    )

    # Create worker configuration
    worker_config = WorkerConfig(
        rollout_queue_bucket=rollout_queue_bucket,
        rollout_queue_path=rollout_queue_path,
        checkpoint_sync_interval=checkpoint_sync_interval,
        batch_timeout=batch_timeout,
        max_idle_time=max_idle_time,
        checkpoint_bucket=checkpoint_bucket,
        checkpoint_path=checkpoint_path,
    )

    logger.info("Starting training worker with configuration:")
    logger.info(f"  Rollout queue: gs://{rollout_queue_bucket}/{rollout_queue_path}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Training steps: {num_train_steps}")
    logger.info(f"  Batch timeout: {batch_timeout}s")
    logger.info(f"  Max idle time: {max_idle_time}s")

    # Initialize and run training worker
    worker = TrainingWorker(training_config, worker_config)
    worker.train()


def main_from_config(
    config_path: Path,
    rollout_queue_bucket: str,
    rollout_queue_path: str = "rollout_queue",
    checkpoint_sync_interval: int = 100,
    batch_timeout: float = 60.0,
    max_idle_time: float = 300.0,
    checkpoint_bucket: str | None = None,
    checkpoint_path: str = "checkpoints",
):
    """Launch training worker from configuration file."""
    # Load training config from file
    training_config = TrainingConfig.from_file(config_path)

    # Create worker config
    worker_config = WorkerConfig(
        rollout_queue_bucket=rollout_queue_bucket,
        rollout_queue_path=rollout_queue_path,
        checkpoint_sync_interval=checkpoint_sync_interval,
        batch_timeout=batch_timeout,
        max_idle_time=max_idle_time,
        checkpoint_bucket=checkpoint_bucket,
        checkpoint_path=checkpoint_path,
    )

    logger.info(f"Loading training configuration from {config_path}")
    logger.info("Starting training worker with configuration:")
    logger.info(f"  Rollout queue: gs://{rollout_queue_bucket}/{rollout_queue_path}")
    logger.info(f"  Output dir: {training_config.output_dir}")
    logger.info(f"  Training steps: {training_config.hyperparameters.num_train_steps}")
    logger.info(f"  Batch timeout: {batch_timeout}s")
    logger.info(f"  Max idle time: {max_idle_time}s")

    # Initialize and run training worker
    worker = TrainingWorker(training_config, worker_config)
    worker.train()


if __name__ == "__main__":
    tyro.cli(main)