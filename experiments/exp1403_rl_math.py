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

import logging
import os
from dataclasses import dataclass
from datetime import datetime

import ray

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.post_training.flax.training_config import (
    CheckpointerConfigData,
    DistributedConfig,
    EnvironmentConfig,
    GenerationConfig,
    LoggingConfig,
    ModelConfig,
    ModelOverrideConfig,
    ModelPathsConfig,
    OptimizerConfig,
    TokenizerOverrideConfig,
    TrainingConfig,
    TrainingHyperparameters,
)
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
    """Configuration for RL training on a pod, using draccus TrainingConfig."""

    resources: ResourceConfig
    training_config: TrainingConfig


@remove_tpu_lockfile_on_exit
def run_rl_training_on_pod(config: RlTrainOnPodConfig):
    """
    Run RL training on a Ray cluster, following marin's execution pattern.

    This function follows the same pattern as run_levanter_train_lm but adapted for RL training.
    """

    import levanter.infra.cli_helpers

    from marin.post_training.flax.train import main as rl_training_main

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
        rl_training_main(config.training_config)

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
    end_of_message_token: int,
    tpu_type: str = "v4-64",
    train_bsize: int = 64,
    kl_coef: float = 1e-3,
    learning_rate: float = 5e-7,
    num_train_steps: int = 2048,
    wandb_project: str = "marin_post_training",
    max_output_length: int = 512,
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
        weight_decay_exclusions=[],
        schedule="constant",
        grad_accum_steps=1,
    )

    generation_config = GenerationConfig(
        max_output_length=max_output_length,
        temperature=1.0,
        stop_tokens=[
            [end_of_message_token],
        ],
        n_generations=64,
    )

    test_generation_config = GenerationConfig(
        max_output_length=max_output_length,
        temperature=1.0,
        stop_tokens=[
            [end_of_message_token],
        ],
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
        attn_pdrop=0.0,
    )

    checkpointer_config = CheckpointerConfigData(
        save_optimizer_state=False,
        save_float_dtype="bf16",
        save_model_freq=10,
    )

    resources = TpuPodConfig(tpu_type=tpu_type)

    # Generate unique experiment ID with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_id = f"{timestamp}"

    training_config = TrainingConfig(
        model=ModelConfig(
            model_paths=model_paths_config,
            inference_param_dtype="bf16",
            inference_activation_dtype="bf16",
            training_param_dtype="fp32",
            training_activation_dtype="bf16",
            model_config_override=model_config_override,
            tokenizer_override=TokenizerOverrideConfig(),
            train_attention_kernel_config='splash:{"block_size": 256}',
            prefill_attention_kernel_config='splash:{"block_size": 256}',
            generate_attention_kernel_config=(
                'paged:{"page_size": 256, "pages_per_compute_block": 1, "inline_seq_dim": true, "use_int8": false}'
            ),
        ),
        hyperparameters=TrainingHyperparameters(
            num_train_steps=num_train_steps,
            max_input_length=256,
            max_output_length=max_output_length,
            train_bsize=train_bsize,
            decode_bsize=1024,
            prefill_bsize=16,
            reference_logprobs_bsize=256,
            n_prompts_per_step=16,
            optim_config=optim_config,
            pad_token_id=128002,
            kl_coef=kl_coef,
        ),
        logging=LoggingConfig(
            log_freq=1,
            num_eval_examples=500,
            wandb_project=wandb_project,
            save_initial_checkpoint=False,
            log_initial_step=True,
            max_checkpoints=None,
            online=True,
            prefix=name,
            prefix_to_id=True,
            experiment_id=experiment_id,
        ),
        environment=EnvironmentConfig(),
        distributed=DistributedConfig(
            train_sharding=[1, -1, 1, 1],
            inference_sharding=[1, -1, 1, 1],
            physical_axis_splitting=False,
            jax_distributed_initialize_config={},
        ),
        generation_config=generation_config,
        test_generation_config=test_generation_config,
        output_dir=this_output_path(),
        checkpoint=checkpointer_config,
    )

    config = RlTrainOnPodConfig(
        resources=resources,
        training_config=training_config,
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
    # TODO (Kevin): Update end_of_message_token if using non-Llama models
    # <|eot_id|>
    end_of_message_token = 128009

    experiments = [
        default_rl_train(
            name="math500",
            model_paths=model_paths,
            end_of_message_token=end_of_message_token,
            tpu_type="v5p-16",
            train_bsize=16,
            kl_coef=0.0,
            learning_rate=2e-06,
            num_train_steps=2048,
            # Splash attention requires (max_input_length + max_output_length - 1) to be a multiple of 128.
            max_output_length=513,
        ),
    ]

    executor_main(
        steps=experiments,
        description="RL math training experiments on Llama 3.1 8B",
    )


if __name__ == "__main__":
    main()