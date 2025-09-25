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
import os
import time
import logging
import subprocess

import verifiers as vf

from .marin_env import EnvStep, InferenceContext, MarinEnv

logger = logging.getLogger("ray")


class PrimeIntellectEnv(MarinEnv):
    """
    Environment Wrapper for the Environments Hub by Prime-Intellect, which contains a collection of environments.
    """

    def __init__(self, tokenizer, output_dir_path: str, **kwargs):
        self.tokenizer = tokenizer
        self._output_dir_path: str = os.path.join(output_dir_path)
        os.makedirs(self._output_dir_path, exist_ok=True)

    def get_prime_intellect_env(self, env_id: str, env_args: dict) -> vf.Environment:
        """
        Get the Verifiers environment for the environment ID.
        """
        logger.debug(f"Loading Verifiers environment for {env_id} with arguments: {env_args}")
        return vf.load_environment(env_id=env_id, **env_args)

    def step(
        self,
        env_id: str,
        env_args: dict,
        inference_ctx: InferenceContext | None = None,
        mode: str = "train",
        temperature: float | None = None,
        max_tokens: int | None = None,
        sampling_args: dict | None = None,
        num_examples: int = 20,
        rollouts_per_example: int = 3,
        max_concurrent: int = 32,
        **kwargs,
    ) -> EnvStep:
        """
        Sample problems and generate responses using the model.

        Args:
            env_id: The ID of the environment to evaluate
            env_args: The arguments to use for the environment
            inference_ctx: The inference context
            mode: The mode to run the environment
            temperature: The temperature to use for the model
            max_tokens: The maximum number of tokens to use for the model
            sampling_args: The sampling arguments to use for the model
            num_examples: The number of examples to use for the model
            rollouts_per_example: The number of rollouts to use for the model
            max_concurrent: The maximum number of concurrent requests to use for the model
            kwargs: The keyword arguments
        """
        # Download the environment
        subprocess.run(["prime", "env", "install", env_id])
        env_id = env_id.split("/", 1)[-1]

        vf_env = self.get_prime_intellect_env(env_id, env_args)

        merged_sampling_args: dict = {}
        if sampling_args is not None:
            merged_sampling_args.update(sampling_args)
        if "max_tokens" not in merged_sampling_args:
            merged_sampling_args["max_tokens"] = max_tokens
        if temperature is not None and "temperature" not in merged_sampling_args:
            merged_sampling_args["temperature"] = temperature

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
            client=inference_ctx.client,
            model=inference_ctx.model,
            sampling_args=merged_sampling_args,
            num_examples=num_examples,
            rollouts_per_example=rollouts_per_example,
            max_concurrent=max_concurrent,
        )

        end_time = time.time()
        logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")

        return EnvStep(
            examples=result.prompt, responses=result.completion, rewards=result.reward, metrics=result.metrics
        )
