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
from typing import TYPE_CHECKING, Any, ClassVar, cast

import jax.numpy as jnp
import numpy as np

from marin.rl.environments import MarinEnv
from marin.rl.environments.process_vllm_results import process_vllm_chat_results
from marin.rl.inference_ctx import InferenceContext
from marin.rl.types import Rollout, RolloutGroup

# Lazy import for optional dependencies
if TYPE_CHECKING:
    pass

logger = logging.getLogger("ray")


class PrimeIntellectEnv(MarinEnv):
    """
    Environment Wrapper for the Environments Hub by Prime-Intellect, which contains a collection of environments.
    """

    ENVS: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        env_id: str,
        env_args: dict = {},  # noqa: B006
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

    def _ensure_verifiers_installed(self):
        """Ensure verifiers package is installed."""
        try:
            import verifiers  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The 'verifiers' package is required to use PrimeIntellectEnv. "
                "Please install it with: uv pip install 'marin[rl]' or uv pip install verifiers"
            ) from e

    def load_prime_intellect_env(self, env_id: str, env_args: dict) -> Any:
        """
        Get the Verifiers environment for the environment ID.
        """
        self._ensure_verifiers_installed()
        import verifiers as vf

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
        self._ensure_verifiers_installed()
        from verifiers.types import GenerateOutputs
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
            "logprobs": True,
            # Note: return_tokens_as_token_ids is not supported by current vLLM version
            # We use convert_tokens_to_ids() in process_vllm_results.py instead
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

        result = cast(
            GenerateOutputs,
            vf_env.generate(
                inputs=inputs,
                client=inference_ctx.openai_client(),
                model="marin-model",
                sampling_args=sampling_args,
                max_concurrent=self.max_concurrent,
            ),
        )

        logger.info("Result:")
        logger.info(f"Prompt: {result.prompt[0]}")
        logger.info(f"Completion: {result.completion[0]}")
        logger.info(f"State: {result.state[0]}")
        logger.info(f"Reward: {result.reward[0]}")

        # Use custom processing function to handle vLLM output format correctly
        processed_outputs = process_vllm_chat_results(
            result.prompt, result.completion, result.state, result.reward, inference_ctx.tokenizer
        )

        # Convert to RolloutGroups
        rollout_groups = []
        for prompt_idx in range(len(processed_outputs.prompt_ids)):
            rollouts = []
            for gen_idx in range(n_generations):
                overall_idx = prompt_idx * n_generations + gen_idx
                if overall_idx >= len(processed_outputs.completion_ids):
                    break

                reward = processed_outputs.rewards[overall_idx] if overall_idx < len(processed_outputs.rewards) else 0.0

                # Use tokenizer from inference context
                prompt_tokens = processed_outputs.prompt_ids[prompt_idx]
                response_tokens = processed_outputs.completion_ids[overall_idx]

                token_rewards = jnp.full(len(response_tokens), reward, dtype=jnp.float32)
                response_logprobs = processed_outputs.completion_logprobs[overall_idx]

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
