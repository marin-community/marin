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
RL training experiment following marin patterns.
"""

import json
import logging
import os
from dataclasses import dataclass

import ray

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.resources import ResourceConfig, TpuPodConfig
from marin.training.training import (
    _add_default_env_variables,
    _add_run_env_variables,
    _check_for_wandb_key,
)

from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RlTrainOnPodConfig:
    """Configuration for RL training on a pod, following marin's TrainLmOnPodConfig pattern."""

    resources: ResourceConfig

    load_model: str
    sharding: str = "1,4,1,-1"
    num_train_steps: int = 2048
    max_input_length: int = 256
    max_output_length: int = 1025
    train_bsize: int = 64
    decode_bsize: int = 1024
    prefill_bsize: int = 16
    reference_logprobs_bsize: int = 256
    n_prompts_per_step: int = 16
    log_freq: int = 10
    num_eval_examples: int = 1024
    save_model_freq: int = 10
    wandb_project: str = "marin_post_training"

    inference_param_dtype: str = "bf16"
    inference_activation_dtype: str = "bf16"
    training_param_dtype: str = "fp32"
    training_activation_dtype: str = "bf16"

    optim_config: str = (
        '{"init_lr": 5e-7, "end_lr": 5e-7, "lr": 5e-7, "lr_warmup_steps": 0, "lr_decay_steps": 2048, '
        '"b1": 0.9, "b2": 0.95, "clip_gradient": 1.0, "weight_decay": 0.0, "bf16_momentum": false, '
        '"multiply_by_parameter_scale": false, "weight_decay_exclusions": [], "schedule": "cos", '
        '"grad_accum_steps": 16}'
    )
    logger_config: str = '{"online": true, "prefix": "rl_math_experiment", "prefix_to_id": true}'
    checkpointer_config: str = '{"save_optimizer_state": false, "save_float_dtype": "bf16"}'
    generation_config: str = (
        '{"max_output_length": 1025, "temperature": 1.0, "stop_tokens": [[524, 9399], [694, 9399], '
        "[4005, 9399], [6199, 9399], [8217, 9399], [9169, 9399], [12817, 9399], [19203, 9399], "
        '[20264, 9399], [22246, 9399], [27147, 9399], [128001]], "n_generations": 64}'
    )
    test_generation_config: str = (
        '{"max_output_length": 1025, "temperature": 0.0, "stop_tokens": [[524, 9399], [694, 9399], '
        "[4005, 9399], [6199, 9399], [8217, 9399], [9169, 9399], [12817, 9399], [19203, 9399], [20264, 9399], "
        '[22246, 9399], [27147, 9399], [128001]], "n_generations": 1}'
    )
    model_config_override: str = (
        '{"bos_token_id": 128000, "eos_token_id": 128001, "pad_token_id": 128002, "max_sequence_length": 2048, '
        '"remat_block": "nothing_saveable", "resid_pdrop": 0.0, "embd_pdrop": 0.0, "attn_pdrop": 0.0}'
    )
    tokenizer_override: str = "{}"
    train_attention_kernel_config: str = 'splash:{"block_size": 256}'
    prefill_attention_kernel_config: str = 'splash:{"block_size": 256}'
    generate_attention_kernel_config: str = (
        'paged:{"page_size": 256, "pages_per_compute_block": 1, "inline_seq_dim": true, "use_int8": false}'
    )
    jax_distributed_initalize_config: str = "{}"

    save_initial_checkpoint: bool = False
    log_initial_step: bool = True
    max_checkpoints: int | None = None
    physical_axis_splitting: bool = False
    pad_token_id: int = 128002
    kl_coef: float = 1e-3

    output_path: str | None = None


@remove_tpu_lockfile_on_exit
def run_rl_training_on_pod(config: RlTrainOnPodConfig):
    """
    Run RL training on a Ray cluster, following marin's execution pattern.

    This function follows the same pattern as run_levanter_train_lm but adapted for RL training.
    """

    import levanter.infra.cli_helpers

    from marin.post_training.train import main as rl_training_main

    default_launch_config = levanter.infra.cli_helpers.load_config()

    env = _add_default_env_variables(
        config.resources.runtime_env.get("env_vars", {}),
        default_launch_config.env_for_accel(config.resources.accelerator_descriptor() or ""),
    )

    if isinstance(config.resources, TpuPodConfig):
        _check_for_wandb_key(env)

    env = _add_run_env_variables(env)

    if "JAX_COMPILATION_CACHE_DIR" not in env:
        marin_prefix = os.environ.get("MARIN_PREFIX")
        if marin_prefix:
            env["JAX_COMPILATION_CACHE_DIR"] = os.path.join(marin_prefix, "compilation-cache")
            logger.info(f"JAX compilation cache enabled at: {env['JAX_COMPILATION_CACHE_DIR']}")
        else:
            logger.warning("MARIN_PREFIX environment variable not set. JAX compilation cache will not be configured.")

    hw_config = config.resources.with_env_vars(env)

    @ray.remote(**hw_config.as_remote_kwargs(), max_calls=1, max_retries=10)
    def rl_train_task():

        training_kwargs = {
            "load_model": config.load_model,
            "output_dir": config.output_path,
            "sharding": config.sharding,
            "num_train_steps": config.num_train_steps,
            "max_input_length": config.max_input_length,
            "max_output_length": config.max_output_length,
            "train_bsize": config.train_bsize,
            "decode_bsize": config.decode_bsize,
            "prefill_bsize": config.prefill_bsize,
            "reference_logprobs_bsize": config.reference_logprobs_bsize,
            "n_prompts_per_step": config.n_prompts_per_step,
            "log_freq": config.log_freq,
            "num_eval_examples": config.num_eval_examples,
            "save_model_freq": config.save_model_freq,
            "wandb_project": config.wandb_project,
            "inference_param_dtype": config.inference_param_dtype,
            "inference_activation_dtype": config.inference_activation_dtype,
            "training_param_dtype": config.training_param_dtype,
            "training_activation_dtype": config.training_activation_dtype,
            "optim_config": config.optim_config,
            "logger_config": config.logger_config,
            "checkpointer_config": config.checkpointer_config,
            "generation_config": config.generation_config,
            "test_generation_config": config.test_generation_config,
            "model_config_override": config.model_config_override,
            "tokenizer_override": config.tokenizer_override,
            "train_attention_kernel_config": config.train_attention_kernel_config,
            "prefill_attention_kernel_config": config.prefill_attention_kernel_config,
            "generate_attention_kernel_config": config.generate_attention_kernel_config,
            "jax_distributed_initalize_config": config.jax_distributed_initalize_config,
            "save_initial_checkpoint": config.save_initial_checkpoint,
            "log_initial_step": config.log_initial_step,
            "max_checkpoints": config.max_checkpoints,
            "physical_axis_splitting": config.physical_axis_splitting,
            "pad_token_id": config.pad_token_id,
            "kl_coef": config.kl_coef,
        }

        rl_training_main(**training_kwargs)

    if isinstance(hw_config, TpuPodConfig):
        from levanter.infra.ray_tpu import run_on_pod_multislice_resumable, run_on_pod_resumable

        if hw_config.slice_count == 1:
            return run_on_pod_resumable(rl_train_task, config.resources.accelerator_descriptor(), max_retries_failure=10)
        else:
            return run_on_pod_multislice_resumable(
                rl_train_task,
                config.resources.accelerator_descriptor(),
                hw_config.slice_count,
                max_retries_failure=10,
            )
    else:
        return ray.get(rl_train_task.remote())


def default_rl_train(
    name: str,
    model_paths: dict[str, str],
    tpu_type: str = "v4-64",
    train_bsize: int = 64,
    kl_coef: float = 1e-3,
    learning_rate: float = 5e-7,
    num_train_steps: int = 2048,
    wandb_project: str = "marin_post_training",
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
        wandb_project: Wandb project name
        **kwargs: Additional arguments
    """

    load_model = f"paths:{json.dumps(model_paths)}"

    optim_config = {
        "init_lr": learning_rate,
        "end_lr": learning_rate,
        "lr": learning_rate,
        "lr_warmup_steps": 0,
        "lr_decay_steps": num_train_steps,
        "b1": 0.9,
        "b2": 0.95,
        "clip_gradient": 1.0,
        "weight_decay": 0.0,
        "bf16_momentum": False,
        "multiply_by_parameter_scale": False,
        "weight_decay_exclusions": [],
        "schedule": "cos",
        "grad_accum_steps": 16,
    }

    logger_config = {
        "online": True,
        "prefix": name,
        "prefix_to_id": True,
        "experiment_id": name,
    }

    resources = TpuPodConfig(tpu_type=tpu_type)

    config = RlTrainOnPodConfig(
        load_model=load_model,
        train_bsize=train_bsize,
        kl_coef=kl_coef,
        num_train_steps=num_train_steps,
        wandb_project=wandb_project,
        optim_config=f"adamw:{json.dumps(optim_config)}",
        logger_config=json.dumps(logger_config),
        resources=resources,
        output_path=this_output_path(),
        **kwargs,
    )

    return ExecutorStep(
        name=os.path.join("rl_checkpoints", name),
        description=f"RL training experiment: {name} for {num_train_steps} steps",
        fn=run_rl_training_on_pod,
        config=config,
        pip_dependency_groups=["post_training"],
    )


def main():
    """Main function to run RL training experiments."""

    model_paths = {
        "params": "gs://marin-us-central2/checkpoints/Llama-3.1-8B-Instruct-converted/params.msgpack",
        "tokenizer": "meta-llama/Meta-Llama-3-8B-Instruct",
        "config": "gs://marin-us-central2/checkpoints/Llama-3.1-8B-Instruct-converted/config.json",
    }

    experiments = [
        default_rl_train(
            name="all-math500-v4-64",
            model_paths=model_paths,
            tpu_type="v4-64",
            train_bsize=64,
            kl_coef=1e-3,
            learning_rate=5e-7,
            num_train_steps=2048,
        ),
    ]

    executor_main(
        steps=experiments,
        description="RL math training experiments on Llama 3.1 8B",
    )


if __name__ == "__main__":
    main()
