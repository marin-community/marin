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
import subprocess
import time
from typing import ClassVar

import verifiers as vf

from .marin_env import EnvStep, InferenceContext, MarinEnv

logger = logging.getLogger("ray")


class PrimeIntellectEnv(MarinEnv):
    """
    Environment Wrapper for the Environments Hub by Prime-Intellect, which contains a collection of environments.
    """

    ENVS: ClassVar[dict[str, vf.Environment]] = {}

    def __init__(self, tokenizer, output_dir_path: str, **kwargs):
        self.tokenizer = tokenizer
        self._output_dir_path: str = os.path.join(output_dir_path)
        os.makedirs(self._output_dir_path, exist_ok=True)

    def load_prime_intellect_env(self, env_id: str, env_args: dict) -> vf.Environment:
        """
        Get the Verifiers environment for the environment ID.
        """
        logger.debug(f"Loading Verifiers environment for {env_id} with arguments: {env_args}")

        if env_id not in self.ENVS:
            self.ENVS[env_id] = vf.load_environment(env_id=env_id, **env_args)

        return self.ENVS[env_id]

    def step(
        self,
        env_id: str,
        env_args: dict,
        num_examples: int,
        rollouts_per_example: int,
        inference_ctx: InferenceContext | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_concurrent: int = 32,
    ) -> EnvStep:
        """
        Sample problems and generate responses using the model.

        Args:
            env_id: The ID of the environment to evaluate
            env_args: The arguments to use for the environment
            inference_ctx: The inference context
            temperature: The temperature to use for the model
            max_tokens: The maximum number of tokens to use for the model
            num_examples: The number of examples to use for the model
            rollouts_per_example: The number of rollouts to use for the model
            max_concurrent: The maximum number of concurrent requests to use for the model
        """
        # Download the environment
        subprocess.run(["prime", "env", "install", env_id])
        env_id = env_id.split("/", 1)[-1]

        vf_env = self.load_prime_intellect_env(env_id, env_args)

        sampling_args: dict = {}
        if max_tokens is not None:
            sampling_args["max_tokens"] = max_tokens
        if temperature is not None:
            sampling_args["temperature"] = temperature

        logger.info(f"Starting evaluation with model: {inference_ctx.model}")
        logger.info(
            f"Configuration: num_examples={num_examples}, \
                rollouts_per_example={rollouts_per_example}, \
                max_concurrent={max_concurrent}, \
                max_tokens={max_tokens}, \
                temperature={temperature}"
        )

        start_time = time.time()
        result = vf_env.evaluate(
            client=inference_ctx.openai_client(),
            model=inference_ctx.model,
            sampling_args=sampling_args,
            num_examples=num_examples,
            rollouts_per_example=rollouts_per_example,
            max_concurrent=max_concurrent,
        )

        end_time = time.time()
        logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")

        return EnvStep(
            examples=result.prompt, responses=result.completion, rewards=result.reward, metrics=result.metrics
        )
