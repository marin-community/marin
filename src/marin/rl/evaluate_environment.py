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

"""Utility functions for evaluating RL environments."""

import jmp
import dataclasses
import ray
import logging
import levanter
import haliax as hax
import jax.random as jrandom
import fsspec
import json

from typing import Any
from dataclasses import dataclass
from levanter.trainer import TrainerConfig
from levanter.models.llama import LlamaConfig
from ray.runtime_env import RuntimeEnv
from transformers import AutoConfig, AutoTokenizer
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.inference.engine import InferenceEngineConfig

from marin.resources import TpuPodConfig
from marin.training.training import _add_run_env_variables
from marin.execution import ExecutorStep
from marin.rl.environments.base import MarinEnv, EnvConfig, load_environment_from_spec
from marin.rl.model_utils import load_model_from_checkpoint
from marin.rl.rollout_worker import InferenceContext
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger("ray")


@dataclass
class EnvironmentEvalConfig:
    """Configuration for environment evaluation."""

    model_checkpoint: str
    """Path to model checkpoint to evaluate."""

    env_config: EnvConfig
    """Environment configuration."""

    output_path: str | None = None
    """Path to save evaluation results (local or GCS)."""

    n_examples: int = 100
    """Number of examples to evaluate."""

    n_generations: int = 1
    """Number of generations per prompt."""

    temperature: float = 0.0
    """Sampling temperature."""

    max_input_length: int = 2048
    """Maximum input sequence length."""

    max_output_length: int = 2048
    """Maximum output sequence length."""

    stop_tokens: list[str] | None = None
    """Stop tokens for generation."""

    seed: int = 42
    """Random seed for evaluation."""

    tpu_type: str = "v5litepod-128"
    """TPU type to use for evaluation."""


@ray.remote(max_retries=3)
def _run_evaluation(config: EnvironmentEvalConfig) -> dict[str, Any]:
    """Run environment evaluation."""

    # Get the actual device count from the TPU configuration first
    runtime_env_temp = RuntimeEnv()
    rollout_pod_config_temp = TpuPodConfig(tpu_type=config.tpu_type, runtime_env=runtime_env_temp)
    num_devices_for_config = rollout_pod_config_temp.total_device_count()
    model_axis_size_config = min(4, num_devices_for_config)

    # Initialize Levanter with minimal trainer config
    trainer_config = TrainerConfig(
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        tensor_parallel_axes=["mlp", "heads"],
        fsdp_axis="embed",
        batch_axis="batch",
        ray=levanter.distributed.RayConfig(auto_start_cluster=False),
        model_axis_size=model_axis_size_config,
    )

    env = {}
    env = _add_run_env_variables(env)
    env["EQX_ON_ERROR"] = "nan"

    runtime_env = RuntimeEnv()
    rollout_pod_config = TpuPodConfig(tpu_type=config.tpu_type, runtime_env=runtime_env)
    rollout_hw_config = rollout_pod_config.with_env_vars(env)

    # Get the actual device count from the TPU configuration
    num_devices = rollout_pod_config.total_device_count()
    model_axis_size = min(4, num_devices)  # Conservative choice for model parallelism
    logger.info(f"Using TPU type {config.tpu_type} with {num_devices} devices, model_axis_size={model_axis_size}")

    # Properly request TPU resources for Ray remote
    rollout_kwargs = dict(
        max_calls=1,
        runtime_env=rollout_hw_config.runtime_env,
        num_cpus=8,
        resources={"TPU": num_devices, f"{config.tpu_type}-head": 1},
    )

    inference_server_config = InferenceServerConfig(
        # Turn on tensor parallelism for inference
        trainer=dataclasses.replace(
            trainer_config, tensor_parallel_axes=["mlp", "kv_head"], model_axis_size=model_axis_size
        ),
        tokenizer=config.model_checkpoint,
        temperature=1.0,
        service=InferenceEngineConfig(
            max_seqs=16,
            max_seq_len=config.max_input_length + config.max_output_length,
            page_size=32,
            max_pages=16 * 32,
            max_seqs_in_prefill=16,
        ),
    )

    # Load tokenizer
    logger.info("Loading tokenizer for evaluation")
    tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)

    @ray.remote(**rollout_kwargs)
    def inference_worker_task():
        with remove_tpu_lockfile_on_exit():
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

            # Initialize Levanter
            env_name = config.env_config.env_class.split(".")[-1]
            trainer_config.id = f"eval-rollout-{env_name}"
            levanter.initialize(trainer_config)

            hf_config = AutoConfig.from_pretrained(config.model_checkpoint)
            model_config = LlamaConfig.from_hf_config(hf_config)
            logger.info(f"Model config: {model_config}")

            # Adjust the max sequence length of the model to reduce memory usage.
            model_config = dataclasses.replace(
                model_config,
                seq_len=config.max_input_length + config.max_output_length,
                tokenizer=config.model_checkpoint,
            )

            key = jrandom.PRNGKey(42)
            vocab_size = tokenizer.vocab_size
            Vocab = hax.Axis("vocab", vocab_size)
            logger.info(f"Vocab size: {vocab_size}")

            policy_model = load_model_from_checkpoint(
                checkpoint=config.model_checkpoint,
                model_config=model_config,
                trainer_config=trainer_config,
                mesh=trainer_config.device_mesh,
                # use the compute axis mapping for inference
                axis_mapping=trainer_config.compute_axis_mapping,
                vocab_axis=Vocab,
                tokenizer=tokenizer,
                key=key,
            )
            logger.info(f"Policy model: {policy_model}")

            inference_server = InferenceServer.create(
                inference_server_config,
                model=policy_model,
                tokenizer=tokenizer,
            )

            env = load_environment_from_spec(config.env_config)
            logger.info(f"Loaded environment: {env}")

            policy_ctx = InferenceContext(
                tokenizer=tokenizer,
                inference_server=inference_server,
                max_tokens=config.max_input_length + config.max_output_length,
                stop_tokens=config.stop_tokens,
            )

            with (
                trainer_config.device_mesh,
                hax.axis_mapping(trainer_config.compute_axis_mapping),
            ):
                # Sample examples, generate responses, and create rollouts from selected lesson
                rollout_groups, metrics = env.sample(
                    inference_ctx=policy_ctx,
                    n_examples=config.n_examples,
                    n_generations=config.n_generations,
                    temperature=config.temperature,
                    prng_key=jrandom.PRNGKey(config.seed),
                    mode="eval",
                )

            if len(rollout_groups) == 0:
                logger.warning("No valid rollouts generated in this batch...")
                return None, None

            logger.info("Evaluation completed")
            logger.info(f"Rollout groups: {rollout_groups}")
            logger.info(f"Metrics: {metrics}")

            # Save rollout groups as JSON
            rollout_file = f"{config.output_path}/rollout_groups.json"
            with fsspec.open(rollout_file, "w") as f:
                json.dump([g.model_dump() for g in rollout_groups], f, indent=2)
            logger.info(f"Saved rollout groups to {rollout_file}")

            # Save metrics as JSON
            metrics_file = f"{config.output_path}/metrics.json"
            with fsspec.open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved metrics to {metrics_file}")

            return metrics

    inference_task = inference_worker_task.remote()
    # Wait for the task to complete and return the actual results
    return ray.get(inference_task)


def evaluate_environment(
    model: str,
    env_config: EnvConfig | None = None,
    env: MarinEnv | None = None,
    name: str | None = None,
    output_path: str | None = None,
    tpu_type: str = "v5litepod-128",
) -> ExecutorStep:
    """Create an executor step for evaluating a model on an environment.

    Args:
        model: Path to model checkpoint or ExecutorStep producing a model
        env_config: Environment configuration (use this or env, not both)
        env: MarinEnv instance for backward compatibility (will be converted to config)
        name: Name of the evaluation
        output_path: Path to save evaluation results (local or GCS)
        tpu_type: TPU type to use for evaluation

    Returns:
        ExecutorStep that runs the evaluation
    """
    if env_config is None and env is None:
        raise ValueError("Either env_config or env must be provided")

    if env_config is None:
        # Convert the MarinEnv instance to an EnvConfig for serialization
        env_class_path = f"{env.__class__.__module__}.{env.__class__.__name__}"
        env_config = EnvConfig(
            env_class=env_class_path,
            env_args={
                "env_id": getattr(env, "env_id", ""),
                "env_args": getattr(env, "env_args", {}),
                "max_tokens": getattr(env, "max_tokens", 1024),
                "max_concurrent": getattr(env, "max_concurrent", 32),
            },
        )
        env_name = env.__class__.__name__
        env_id = getattr(env, "env_id", "unknown")
    else:
        env_name = env_config.env_class.split(".")[-1]
        env_id = env_config.env_args.get("env_id", "unknown")

    config = EnvironmentEvalConfig(
        model_checkpoint=model,
        env_config=env_config,
        output_path=output_path,
        tpu_type=tpu_type,
    )

    return ExecutorStep(
        name=name or f"evaluate-{env_name}-{model}-{env_id}",
        fn=_run_evaluation,
        config=config,
        description=f"Evaluate model on {env_name}",
        pip_dependency_groups=["post_training", "rl"],
    )
