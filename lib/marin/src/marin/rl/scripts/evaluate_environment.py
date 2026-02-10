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

import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import fsspec
import haliax as hax
import jax
import jax.random as jrandom
import jmp
import levanter
import numpy
from fray.v1.cluster import (
    CpuConfig,
    EnvironmentConfig,
    Entrypoint,
    JobRequest,
    ResourceConfig,
    current_cluster,
)
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.models.lm_model import LmConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution import ExecutorStep
from marin.execution.executor import executor_main
from marin.rl.environments.base import EnvConfig, load_environment_from_spec
from marin.rl.model_utils import load_model_from_checkpoint
from marin.rl.rollout_worker import create_inference_context
from marin.rl.types import RolloutGroup
from marin.training.training import _add_run_env_variables
from marin.utils import remove_tpu_lockfile_on_exit
from transformers import AutoTokenizer

logger = logging.getLogger("ray")


def _to_list(arr) -> list:
    """Convert array-like object to list, handling both JAX arrays and Python lists."""
    if isinstance(arr, list):
        return arr
    elif hasattr(arr, "tolist"):
        return arr.tolist()
    else:
        # Fallback for other array types
        return list(arr)


def rollout_group_to_dict(group: "RolloutGroup") -> dict[str, Any]:
    """Convert a RolloutGroup to a JSON-serializable dictionary.

    Uses tree mapping to automatically handle all fields including arrays,
    making it robust to schema changes.
    """
    return dataclasses.asdict(
        jax.tree.map(lambda v: _to_list(v) if isinstance(v, list | jax.Array | numpy.ndarray) else v, group)
    )


@dataclass
class EnvironmentEvalConfig:
    """Configuration for environment evaluation."""

    checkpoint: str
    """Path to model checkpoint (HuggingFace repo or local path)."""

    env_config: EnvConfig

    model_config: LmConfig | None = None
    """Model configuration. If None, auto-detected from checkpoint."""

    temperature: float = 0.0
    max_input_length: int = 2048
    max_output_length: int = 8192
    stop_tokens: list[str] | None = None
    seed: int = 42
    tpu_type: str | None = None
    output_path: str | None = None

    n_examples: int = 100
    """Number of examples to evaluate."""

    n_generations: int = 1
    """Number of generations per prompt."""

    vocab_size: int | None = None
    """Vocab size for model construction. Should match the checkpoint's vocab dimension.
    If None, falls back to len(tokenizer)."""


def _run_evaluation(config: EnvironmentEvalConfig) -> None:
    """Run environment evaluation."""

    if config.output_path is None:
        raise ValueError("output_path is required for evaluation results")

    # Calculate TPU device count
    if config.tpu_type is None:
        model_axis_size = 1
    else:
        num_devices = ResourceConfig.with_tpu(config.tpu_type).chip_count()
        model_axis_size = min(4, num_devices)
        logger.info(f"Using TPU type {config.tpu_type} with {num_devices} devices, model_axis_size={model_axis_size}")

    # Initialize Levanter with minimal trainer config
    trainer_config = TrainerConfig(
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        ray=levanter.distributed.RayConfig(auto_start_cluster=False),
        mesh=MeshConfig(axes={"model": model_axis_size}),
    )

    # Setup environment variables
    env_vars = _add_run_env_variables({})
    env_vars["EQX_ON_ERROR"] = "nan"

    checkpoint_path = config.checkpoint

    def _run_inference():
        logger.info("Loading tokenizer for evaluation")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        with remove_tpu_lockfile_on_exit():
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

            env_name = config.env_config.env_class.split(".")[-1]
            trainer_config.id = f"eval-rollout-{env_name}"
            levanter.initialize(trainer_config)

            # Auto-detect model config from checkpoint if not provided
            if config.model_config is None:
                logger.info(f"Auto-detecting model config from checkpoint: {checkpoint_path}")
                converter = HFCheckpointConverter.from_hf(checkpoint_path)
                model_config = converter.default_config
            else:
                model_config = config.model_config

            # Update seq_len for inference
            model_config = dataclasses.replace(
                model_config,
                max_seq_len=config.max_input_length + config.max_output_length,
            )
            logger.info(f"Model config: {model_config}")

            key = jrandom.PRNGKey(42)
            vocab_size = config.vocab_size if config.vocab_size is not None else len(tokenizer)
            Vocab = hax.Axis("vocab", vocab_size)
            logger.info(f"Vocab size: {vocab_size}")

            policy_model = load_model_from_checkpoint(
                checkpoint=checkpoint_path,
                model_config=model_config,
                trainer_config=trainer_config,
                mesh=trainer_config.device_mesh,  # Now this is concrete after levanter.initialize
                # use the compute axis mapping for inference
                axis_mapping=trainer_config.compute_axis_mapping,
                vocab_axis=Vocab,
                tokenizer=tokenizer,
                key=key,
            )
            logger.info(f"Policy model: {policy_model}")

            with (
                trainer_config.use_device_mesh(),
                hax.axis_mapping(trainer_config.compute_axis_mapping),
            ):
                inference_server_config = InferenceServerConfig(
                    trainer=dataclasses.replace(
                        trainer_config,
                        mesh=dataclasses.replace(trainer_config.mesh, shared_mapping={"kv_head": "model"}),
                    ),
                    tokenizer=checkpoint_path,
                    temperature=1.0,
                    service=InferenceEngineConfig(
                        max_seqs=16,
                        max_seq_len=config.max_input_length + config.max_output_length,
                        page_size=32,
                        max_seqs_in_prefill=16,
                        hbm_utilization=0.3,  # Reduced from default 0.9 to prevent OOM on v4-8
                    ),
                )

                inference_server = InferenceServer.create(
                    inference_server_config,
                    model=policy_model,
                    tokenizer=tokenizer,
                )

                import threading

                threading.Thread(target=inference_server.serve, daemon=True).start()
                time.sleep(2)

                env = load_environment_from_spec(config.env_config)
                logger.info(f"Loaded environment: {env}")

                policy_ctx = create_inference_context(
                    inference_type="levanter",
                    inference_config=inference_server_config,
                )

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
                json.dump([rollout_group_to_dict(g) for g in rollout_groups], f, indent=2)
            logger.info(f"Saved rollout groups to {rollout_file}")

            # Save metrics as JSON
            metrics_file = f"{config.output_path}/metrics.json"
            with fsspec.open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved metrics to {metrics_file}")

    if config.tpu_type is None:
        resources = ResourceConfig(device=CpuConfig(), replicas=1)
    else:
        resources = ResourceConfig.with_tpu(config.tpu_type)

    job_request = JobRequest(
        name=f"evaluate-{config.env_config.env_class}",
        entrypoint=Entrypoint.from_callable(_run_inference),
        resources=resources,
        environment=EnvironmentConfig.create(
            extras=["math", "rl"],
            env_vars=env_vars,
        ),
    )

    cluster = current_cluster()
    job_id = cluster.launch(job_request)
    logger.info(f"Launched evaluation job: {job_id}")
    job_info = cluster.monitor(job_id)
    logger.info(f"Evaluation job completed with status: {job_info.status}")
    if job_info.status != "succeeded":
        raise RuntimeError(f"Evaluation job failed with status: {job_info.status}")

    logger.info("Evaluation completed successfully")
    return None


def evaluate_environment(
    checkpoint: str,
    env_config: EnvConfig,
    output_path: str,
    model_config: LmConfig | None = None,
    tpu_type: str | None = None,  # "v5litepod-128"
) -> ExecutorStep:
    """Create an executor step for evaluating a model on an environment.

    Args:
        checkpoint: Path to model checkpoint (HuggingFace repo or local path)
        env_config: Environment configuration
        output_path: Path to save evaluation results (local or GCS)
        model_config: Model configuration. If None, auto-detected from checkpoint.
        tpu_type: TPU type to use for evaluation

    Returns:
        ExecutorStep that runs the evaluation
    """
    env_name = env_config.env_class.split(".")[-1]
    env_id = env_config.env_args.get("env_id", "unknown")

    # Get model identifier from checkpoint path for naming
    model_identifier = checkpoint.split("/")[-1] if "/" in checkpoint else checkpoint

    config = EnvironmentEvalConfig(
        checkpoint=checkpoint,
        env_config=env_config,
        model_config=model_config,
        output_path=output_path,
        tpu_type=tpu_type,
    )

    return ExecutorStep(
        name=f"evaluate-{env_name}-{model_identifier}-{env_id}",
        fn=_run_evaluation,
        config=config,
        description=f"Evaluate model on {env_name}",
        pip_dependency_groups=["rl"],
    )


if __name__ == "__main__":
    step = evaluate_environment(
        checkpoint="HuggingFaceTB/SmolLM2-135M",
        env_config=EnvConfig(
            env_class="marin.rl.environments.mock_env.MockEnv",
            env_args={"task_type": "addition", "difficulty": "easy", "seed": 42},
        ),
        output_path="/tmp/evals",
    )
    executor_main([step])
