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

import ray
import logging
import levanter
import threading
import haliax as hax
import jax.random as jrandom

from typing import Any
from dataclasses import dataclass
from levanter.trainer import TrainerConfig
from levanter.models.llama import LlamaConfig
from transformers import AutoConfig, AutoTokenizer
from levanter.inference.openai import InferenceServer, InferenceServerConfig

from marin.execution import ExecutorStep
from marin.rl.environments.base import MarinEnv
from marin.rl.model_utils import load_model_from_checkpoint
from marin.rl.rollout_worker import LevanterInferenceContext

logger = logging.getLogger("ray")


@dataclass
class EnvironmentEvalConfig:
    """Configuration for environment evaluation."""

    model_checkpoint: str
    """Path to model checkpoint to evaluate."""

    env: MarinEnv
    """Environment to evaluate on."""

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


@ray.remote(max_retries=3)
def _run_evaluation(config: EnvironmentEvalConfig) -> dict[str, Any]:
    """Run environment evaluation."""

    # Initialize Levanter with minimal trainer config
    trainer_config = TrainerConfig(
        ray=levanter.distributed.RayConfig(auto_start_cluster=False),
    )
    trainer_config.id = "evaluation"
    levanter.initialize(trainer_config)

    # Load tokenizer
    logger.info("Loading tokenizer for evaluation")
    if hasattr(config.env, "tokenizer"):
        tokenizer = config.env.tokenizer
    else:
        # Infer tokenizer from checkpoint or use default
        tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)

    # Load model
    logger.info(f"Loading model from {config.model_checkpoint}")
    hf_config = AutoConfig.from_pretrained(config.model_checkpoint)
    model_config = LlamaConfig.from_hf_config(hf_config)

    vocab_size = tokenizer.vocab_size
    Vocab = hax.Axis("vocab", vocab_size)

    key = jrandom.PRNGKey(config.seed)
    model = load_model_from_checkpoint(
        checkpoint=config.model_checkpoint,
        model_config=model_config,
        trainer_config=trainer_config,
        mesh=trainer_config.device_mesh,
        axis_mapping=trainer_config.compute_axis_mapping,
        vocab_axis=Vocab,
        tokenizer=tokenizer,
        key=key,
    )

    # Create inference server
    logger.info("Starting inference server")
    inference_server_config = InferenceServerConfig(
        trainer=trainer_config,
        tokenizer=tokenizer if isinstance(tokenizer, str) else config.model_checkpoint,
        temperature=config.temperature,
    )
    inference_server = InferenceServer.create(
        inference_server_config,
        model=model,
        tokenizer=tokenizer,
    )
    inference_thread = threading.Thread(target=lambda: inference_server.serve(), daemon=True)
    inference_thread.start()

    # Create inference context
    inference_ctx = LevanterInferenceContext(
        tokenizer=tokenizer,
        inference_server=inference_server,
        max_tokens=config.max_output_length,
        stop_tokens=config.stop_tokens,
    )

    # Run evaluation
    logger.info(f"Running evaluation on {config.n_examples} examples")
    rng = jrandom.PRNGKey(config.seed)

    # Sample from environment
    with (
        trainer_config.device_mesh,
        hax.axis_mapping(trainer_config.compute_axis_mapping),
    ):
        rollout_groups, metrics = config.env.sample(
            inference_ctx=inference_ctx,
            n_examples=config.n_examples,
            n_generations=config.n_generations,
            temperature=config.temperature,
            prng_key=rng,
            mode="eval",
        )

    logger.info("Evaluation completed")

    # Compute summary statistics
    total_reward = sum(rollout.episode_reward for group in rollout_groups for rollout in group.rollouts)
    n_rollouts = sum(len(group.rollouts) for group in rollout_groups)
    avg_reward = total_reward / n_rollouts if n_rollouts > 0 else 0.0

    # Shutdown inference server
    inference_server.shutdown()

    # Return metrics
    return {
        "status": "completed",
        "metrics": metrics,
        "n_rollouts": n_rollouts,
        "avg_reward": avg_reward,
    }


def evaluate_environment(model: str, env: MarinEnv, name: str = None) -> ExecutorStep:
    """Create an executor step for evaluating a model on an environment.

    Args:
        model: Path to model checkpoint or ExecutorStep producing a model
        env: Environment to evaluate on

    Returns:
        ExecutorStep that runs the evaluation
    """
    config = EnvironmentEvalConfig(
        model_checkpoint=model,
        env=env,
    )

    return ExecutorStep(
        name=name or f"evaluate-{env.__class__.__name__}-{model}-{env.env_id}",
        fn=_run_evaluation,
        config=config,
        description=f"Evaluate model on {env.__class__.__name__}",
        pip_dependency_groups=["post_training", "rl"],
    )
