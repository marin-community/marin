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
Environment Wrapper for the Environments Hub by Prime-Intellect, which contains a collection of environments.
https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars
"""
import logging
import os
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import verifiers as vf

from marin.rl.types import InferenceContext, Rollout, RolloutGroup

from .base import MarinEnv

logger = logging.getLogger("ray")


class PrimeIntellectEnv(MarinEnv):
    """
    Environment Wrapper for the Environments Hub by Prime-Intellect, which contains a collection of environments.
    """

    ENVS: ClassVar[dict[str, vf.Environment]] = {}

    def __init__(self, tokenizer, env_id: str, output_dir_path: str | None = None, **kwargs):
        self.tokenizer = tokenizer
        self.env_id = env_id
        self.env_args = kwargs
        self._output_dir_path: str | None = None
        if output_dir_path:
            self._output_dir_path = os.path.join(output_dir_path)
            os.makedirs(self._output_dir_path, exist_ok=True)

    def load_prime_intellect_env(self, env_id: str, env_args: dict) -> vf.Environment:
        """
        Get the Verifiers environment for the environment ID.
        """
        logger.debug(f"Loading Verifiers environment for {env_id} with arguments: {env_args}")

        if env_id not in self.ENVS:
            self.ENVS[env_id] = vf.load_environment(env_id=env_id, **env_args)

        return self.ENVS[env_id]

    def sample(
        self,
        inference_ctx: InferenceContext,
        n_examples: int,
        n_generations: int,
        temperature: float,
        prng_key,
        mode: str = "train",
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        """Sample examples, generate responses, and create rollouts using verifiers library.

        This uses the Prime Intellect verifiers library which handles the full pipeline
        internally (sampling, generation, and scoring).
        """
        # Load the verifiers environment
        vf_env = self.load_prime_intellect_env(self.env_id, self.env_args)

        # Get OpenAI client from inference context
        client = inference_ctx.openai_client()

        # Use evaluate() to sample from dataset and generate/score responses
        # Note: verifiers uses "model" as a string identifier, we'll pass "marin-model"
        sampling_args = {"temperature": temperature}

        logger.info(f"Evaluating {n_examples} examples with {n_generations} generations per example")
        result = vf_env.evaluate(
            client=client,
            model="marin-model",  # Model identifier for the OpenAI client
            sampling_args=sampling_args,
            num_examples=n_examples,
            rollouts_per_example=n_generations,
            score_rollouts=True,
        )

        # Convert verifiers GenerateOutputs to our RolloutGroup format
        rollout_groups = []

        # Group results by prompt (each prompt has multiple rollouts)
        for prompt_idx in range(len(result.prompt)):
            rollouts = []

            # Each prompt has n_generations completions
            for gen_idx in range(n_generations):
                overall_idx = prompt_idx * n_generations + gen_idx
                if overall_idx >= len(result.completion):
                    break

                completion = result.completion[overall_idx]
                reward = result.reward[overall_idx] if overall_idx < len(result.reward) else 0.0

                # Tokenize prompt and completion
                prompt_tokens = self.tokenizer.encode(result.prompt[prompt_idx])
                response_tokens = self.tokenizer.encode(completion)

                # Create uniform reward and logprobs (verifiers doesn't give us token-level data)
                token_rewards = jnp.full(len(response_tokens), reward, dtype=jnp.float32)
                # Use zero logprobs since verifiers doesn't provide them
                response_logprobs = jnp.zeros(len(response_tokens), dtype=jnp.float32)

                rollout = Rollout(
                    env_name=f"prime_intellect:{self.env_id}",
                    env_example_id=f"{self.env_id}_example_{prompt_idx}",
                    prompt_tokens=jnp.array(prompt_tokens, dtype=jnp.int32),
                    response_tokens=jnp.array(response_tokens, dtype=jnp.int32),
                    response_logprobs=response_logprobs,
                    token_rewards=token_rewards,
                    episode_reward=float(reward),
                )
                rollouts.append(rollout)

            if rollouts:
                rollout_groups.append(RolloutGroup(rollouts=rollouts))

        # Extract metrics from verifiers result
        metrics = {}
        if hasattr(result, "metrics") and result.metrics:
            metrics.update(result.metrics)

        # Add basic statistics
        if result.reward:
            metrics[f"{self.env_id}.mean_reward"] = float(np.mean(result.reward))
            metrics[f"{self.env_id}.total_rollouts"] = len(result.reward)

        logger.info(f"Generated {len(rollout_groups)} rollout groups with metrics: {metrics}")

        return rollout_groups, metrics
