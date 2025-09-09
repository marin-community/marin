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
Trying out RL training with separate trainer & rollout components.
"""

import logging
import os
from dataclasses import dataclass

import ray
from levanter.infra.ray_tpu import run_on_pod_ray

from marin.execution.executor import ExecutorStep, InputName, executor_main, this_output_path
from marin.post_training.training_config import (
    CheckpointerConfigData,
    DistributedConfig,
    EnvironmentConfig,
    GenerationConfig,
    InferenceWorkerConfig,
    LoggingConfig,
    ModelConfig,
    ModelOverrideConfig,
    ModelPathsConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingHyperparameters,
    TrainWorkerConfig,
)
from marin.resources import TpuPodConfig
from marin.training.training import (
    _add_run_env_variables,
)
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)

WANDB_PROJECT = "async_rl_testing"
ENVIRONMENT_SPEC = "olym_math:difficulty=hard"

STOP_TOKENS = [
    [524, 9399],
    [694, 9399],
    [4005, 9399],
    [6199, 9399],
    [8217, 9399],
    [9169, 9399],
    [12817, 9399],
    [19203, 9399],
    [20264, 9399],
    [22246, 9399],
    [27147, 9399],
    [128001],
]


@dataclass(frozen=True)
class RLTrainConfig:
    """Configuration for RL training on a pod, using draccus TrainingConfig."""

    training_config: TrainingConfig
    inference_tpu_type: str
    train_tpu_type: str
    num_inference_workers: int = 1
    num_train_slices: int = 1


@ray.remote
def run_rl_training_on_pod(config: RLTrainConfig):
    """
    Run RL training with both inference and training workers on a Ray cluster.

    This function launches both workers that communicate through a shared rollout queue.
    """
    from marin.post_training.inference_worker import InferenceWorker
    from marin.post_training.rollout_storage import FileRolloutReader, FileRolloutWriter
    from marin.post_training.train_worker import TrainingWorker

    rollout_queue_path = config.training_config.output_dir + "/rollout_queue"
    checkpoint_dir = config.training_config.output_dir + "/checkpoints"

    train_worker_config = TrainWorkerConfig(
        rollout_queue_path=rollout_queue_path,
        checkpoint_sync_interval=60,  # Save checkpoint every 60 steps (~1 minute)
    )

    inference_worker_config = InferenceWorkerConfig(
        environment_spec=ENVIRONMENT_SPEC,
        checkpoint_source_path=checkpoint_dir,
        rollout_output_path=rollout_queue_path,
        checkpoint_poll_interval=30.0,  # Check for new checkpoints every 30 seconds
        rollout_batch_size=8,
        max_rollouts=None,  # Generate rollouts continuously
        checkpoint_timeout=300.0,
    )

    env = {}
    env = _add_run_env_variables(env)

    if "JAX_COMPILATION_CACHE_DIR" not in env:
        marin_prefix = os.environ.get("MARIN_PREFIX")
        if marin_prefix:
            env["JAX_COMPILATION_CACHE_DIR"] = os.path.join(marin_prefix, "compilation-cache")
            logger.info(f"JAX compilation cache enabled at: {env['JAX_COMPILATION_CACHE_DIR']}")
        else:
            logger.warning("MARIN_PREFIX environment variable not set. JAX compilation cache will not be configured.")

    train_pod_config = TpuPodConfig(tpu_type=config.inference_tpu_type)
    train_hw_config = train_pod_config.with_env_vars(env)

    @ray.remote(runtime_env=train_hw_config.runtime_env, max_calls=1)
    def train_worker_task():
        with remove_tpu_lockfile_on_exit():
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
            rollout_reader = FileRolloutReader(rollout_queue_path)
            worker = TrainingWorker(config.training_config, train_worker_config, rollout_reader)
            worker.train()

    inference_pod_config = TpuPodConfig(tpu_type=config.train_tpu_type)
    inference_hw_config = inference_pod_config.with_env_vars(env)

    @ray.remote(runtime_env=inference_hw_config.runtime_env, max_calls=1)
    def inference_worker_task():
        with remove_tpu_lockfile_on_exit():
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
            rollout_writer = FileRolloutWriter(rollout_queue_path)
            worker = InferenceWorker(config.training_config, inference_worker_config, rollout_writer)
            worker.run()

    train_tasks = []
    train_tasks.append(
        run_on_pod_ray.remote(
            train_worker_task,
            config.train_tpu_type,
            num_slices=config.num_train_slices,
            max_retries_failure=1,
            max_retries_preemption=1,
        )
    )
    inference_tasks = []
    for _ in range(config.num_inference_workers):
        inference_tasks.append(
            run_on_pod_ray.remote(
                inference_worker_task,
                config.inference_tpu_type,
                num_slices=1,
                max_retries_failure=1,
                max_retries_preemption=1,
            )
        )

    return ray.get(inference_tasks + train_tasks)


def default_rl_train(
    name: str,
    model_paths: dict[str, str],
    train_bsize: int,
    inference_tpu_type: str,
    train_tpu_type: str,
    kl_coef: float = 1e-3,
    learning_rate: float = 5e-7,
    num_train_steps: int = 16,
    **kwargs,
) -> ExecutorStep:
    """
    Create an RL training experiment following marin's default_train pattern.

    Args:
        name: The name of the training run
        model_paths: Dictionary with 'params', 'tokenizer', and 'config' paths
        tpu_type: TPU type to use
        train_bsize: Training batch size
        kl_coef: KL coefficient
        learning_rate: Learning rate
        num_train_steps: Number of training steps
        **kwargs: Additional arguments
    """

    model_paths_config = ModelPathsConfig(
        params=model_paths["params"],
        tokenizer=model_paths["tokenizer"],
        config=model_paths["config"],
    )

    optim_config = OptimizerConfig(
        init_lr=learning_rate,
        end_lr=learning_rate,
        lr=learning_rate,
        lr_warmup_steps=0,
        lr_decay_steps=num_train_steps,
        b1=0.9,
        b2=0.95,
        clip_gradient=1.0,
        weight_decay=0.0,
        bf16_momentum=False,
        multiply_by_parameter_scale=False,
        weight_decay_exclusions=(),
        schedule="cos",
        grad_accum_steps=16,
    )

    generation_config = GenerationConfig(
        max_output_length=513,
        temperature=1.0,
        stop_tokens=STOP_TOKENS,
        n_generations=8,
    )

    test_generation_config = GenerationConfig(
        max_output_length=513,
        temperature=0.0,
        stop_tokens=STOP_TOKENS,
        n_generations=1,
    )

    model_config_override = ModelOverrideConfig(
        bos_token_id=128000,
        eos_token_id=128001,
        pad_token_id=128002,
        max_sequence_length=2048,
        remat_block="nothing_saveable",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )

    model_config_override = ModelOverrideConfig(
        bos_token_id=128000,
        eos_token_id=128001,
        pad_token_id=128002,
        max_sequence_length=2048,
        remat_block="nothing_saveable",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )

    checkpointer_config = CheckpointerConfigData(
        save_optimizer_state=False,
        save_float_dtype="bf16",
        save_model_freq=1,
    )

    # Shared paths for worker coordination
    output_dir = this_output_path()

    training_config = TrainingConfig(
        model=ModelConfig(
            model_paths=model_paths_config,
            inference_param_dtype="bf16",
            inference_activation_dtype="bf16",
            training_param_dtype="fp32",
            training_activation_dtype="bf16",
            model_config_override=model_config_override,
            train_attention_kernel_config='splash:{"block_size": 256}',
            prefill_attention_kernel_config='splash:{"block_size": 256}',
            generate_attention_kernel_config=(
                'paged:{"page_size": 256, "pages_per_compute_block": 1, "inline_seq_dim": true, "use_int8": false}'
            ),
        ),
        hyperparameters=TrainingHyperparameters(
            num_train_steps=num_train_steps,
            max_input_length=256,
            max_output_length=513,
            train_bsize=train_bsize,
            decode_bsize=8,
            prefill_bsize=8,
            reference_logprobs_bsize=8,
            n_prompts_per_step=16,
            optim_config=optim_config,
            pad_token_id=128002,
            kl_coef=kl_coef,
        ),
        logging=LoggingConfig(
            log_freq=100,
            num_eval_examples=0,
            wandb_project=WANDB_PROJECT,
            save_initial_checkpoint=True,
            log_initial_step=True,
            max_checkpoints=None,
            online=True,
            prefix=name,
            prefix_to_id=True,
        ),
        environment=EnvironmentConfig(),
        distributed=DistributedConfig(
            sharding=[1, 4, 1, -1],
            physical_axis_splitting=False,
            jax_distributed_initalize_config={},
        ),
        output_dir=output_dir,
        checkpoint=checkpointer_config,
        generation_config=generation_config,
        test_generation_config=test_generation_config,
    )

    config = RLTrainConfig(
        training_config=training_config,
        inference_tpu_type=inference_tpu_type,
        train_tpu_type=train_tpu_type,
    )

    return ExecutorStep(
        name=name,
        description=f"RL training experiment: {name} for {num_train_steps} steps",
        fn=run_rl_training_on_pod,
        config=config,
        pip_dependency_groups=["post_training"],
    )


def main():
    """Main function to run RL training experiments."""

    model_paths = {
        "params": InputName.hardcoded("checkpoints/Llama-3.1-8B-Instruct-converted/params.msgpack"),
        "tokenizer": "meta-llama/Meta-Llama-3-8B-Instruct",
        "config": InputName.hardcoded("checkpoints/Llama-3.1-8B-Instruct-converted/config.json"),
    }

    experiments = [
        default_rl_train(
            name="rl_training_rollout_experiment",
            model_paths=model_paths,
            train_bsize=32,
            kl_coef=1e-3,
            learning_rate=5e-7,
            num_train_steps=10000,
            inference_tpu_type="v4-8",
            train_tpu_type="v4-64",
        ),
    ]

    executor_main(
        steps=experiments,
        description="RL math training experiments on Llama 3.1 8B",
    )


if __name__ == "__main__":
    main()
