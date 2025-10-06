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

    def __init__(
        self,
        env_id: str,
        env_args: dict,
        max_tokens: int = 1024,
        max_concurrent: int = 32,
    ):
        """Initialize PrimeIntellect environment.

        Args:
            env_id: Environment ID like "primeintellect/gsm8k"
            env_args: Dict with verifier-specific args (num_train_examples, etc.)
            max_tokens: Maximum tokens for generation
            max_concurrent: Maximum concurrent requests
        """
        self.env_id = env_id
        self.env_args = env_args
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent

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
        """Sample problems and generate responses using the model."""
        import subprocess

        # Download/install the environment
        subprocess.run(["prime", "env", "install", self.env_id], check=True)

        # Extract just the env name after slash
        env_id = self.env_id.split("/", 1)[-1]
        vf_env = self.load_prime_intellect_env(env_id, self.env_args)

        # Prepare sampling arguments
        sampling_args = {
            "max_tokens": self.max_tokens,
            "temperature": temperature,
        }

        logger.info(
            f"Starting evaluation: n_examples={n_examples}, "
            f"n_generations={n_generations}, max_concurrent={self.max_concurrent}"
        )

        # Get dataset based on mode
        if mode == "train":
            if vf_env.dataset is None:
                raise ValueError(f"Train dataset missing for {env_id}")
            inputs = vf_env.get_dataset(n=n_examples)
        else:
            if vf_env.eval_dataset is None:
                raise ValueError(f"Eval dataset missing for {env_id}")
            inputs = vf_env.get_eval_dataset(n=n_examples)

        # Repeat inputs for multiple generations
        if n_generations > 1:
            inputs = inputs.repeat(n_generations)

        # Generate using verifiers
        result = vf_env.generate(
            dataset=inputs,
            client=inference_ctx.openai_client(),
            model="marin-model",
            sampling_args=sampling_args,
            max_concurrent=self.max_concurrent,
        )

        # Access tokenizer from inference context
        tokenizer = inference_ctx.tokenizer

        # Convert to RolloutGroups
        rollout_groups = []
        for prompt_idx in range(len(result.prompt)):
            rollouts = []
            for gen_idx in range(n_generations):
                overall_idx = prompt_idx * n_generations + gen_idx
                if overall_idx >= len(result.completion):
                    break

                completion = result.completion[overall_idx]
                reward = result.reward[overall_idx] if overall_idx < len(result.reward) else 0.0

                # Use tokenizer from inference context
                prompt_tokens = tokenizer.encode(result.prompt[prompt_idx])
                response_tokens = tokenizer.encode(completion)

                token_rewards = jnp.full(len(response_tokens), reward, dtype=jnp.float32)
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

        # Extract metrics
        metrics = {}
        if hasattr(result, "metrics") and result.metrics:
            metrics.update(result.metrics)

        if result.reward:
            metrics[f"{self.env_id}.mean_reward"] = float(np.mean(result.reward))
            metrics[f"{self.env_id}.total_rollouts"] = len(result.reward)

        logger.info(f"Generated {len(rollout_groups)} rollout groups")
        return rollout_groups, metrics
