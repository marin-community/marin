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
import json
from typing import ClassVar

import verifiers as vf

from .marin_env import EnvStep, InferenceContext, MarinEnv

logger = logging.getLogger("ray")


class PrimeIntellectEnv(MarinEnv):
    """
    Environment Wrapper for the Environments Hub by Prime-Intellect, which contains a collection of environments.
    """

    ENVS: ClassVar[dict[str, vf.Environment]] = {}

    def __init__(self, tokenizer, output_dir_path: str, max_tokens: int = 1024, max_concurrent: int = 32, **kwargs):
        self.tokenizer = tokenizer
        self._output_dir_path: str = os.path.join(output_dir_path)
        os.makedirs(self._output_dir_path, exist_ok=True)

        self.env_id = kwargs.get("env_id", None)
        self.env_args = kwargs.get("env_args", None)

        assert (
            self.env_id is not None
        ), (
            "env_id is required for PrimeIntellectEnv, pass it as an keyword argument or in the environment spec like: "
            "prime_intellect:env_id=primeintellect/gsm8k,env_args={num_train_examples=-1,num_eval_examples=-1}"
        )
        assert (
            self.env_args is not None
        ), (
            "env_args is required for PrimeIntellectEnv, pass it as an keyword argument or in the environment spec like: "
            "prime_intellect:env_id=primeintellect/gsm8k,env_args={num_train_examples=-1,num_eval_examples=-1}"
        )

        self.env_args = json.loads(self.env_args)

        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.max_concurrent = kwargs.get("max_concurrent", 32)

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
        inference_ctx: InferenceContext,
        n_examples: int,
        prng_key,
        mode: str = "train",
        n_generations: int = 1,
        temperature: float = 1.0,
    ) -> EnvStep:
        """
        Sample problems and generate responses using the model.

        Args:
            inference_ctx: The inference context
            n_examples: The number of examples to use for the model
            prng_key: The PRNG key to use for the model
            mode: The mode to use for the environment
            n_generations: The number of generations to use for the model
            temperature: The temperature to use for the model
        """

        # Download the environment
        subprocess.run(["prime", "env", "install", self.env_id])
        env_id = self.env_id.split("/", 1)[-1]

        vf_env = self.load_prime_intellect_env(env_id, self.env_args)

        sampling_args: dict = {}
        if self.max_tokens is not None:
            sampling_args["max_tokens"] = self.max_tokens
        if temperature is not None:
            sampling_args["temperature"] = temperature

        logger.info(f"Starting evaluation with model: {inference_ctx.model}")
        logger.info(
            f"Configuration: num_examples={n_examples}, \
                n_generations={n_generations}, \
                max_concurrent={self.max_concurrent}, \
                max_tokens={self.max_tokens}, \
                temperature={temperature}"
        )

        if mode == "train":
            assert vf_env.dataset is not None, f"Train Dataset is not set for environment {env_id}"
            vf_env.eval_dataset = None
        else:
            assert vf_env.eval_dataset is not None, f"Eval dataset is not set for environment {env_id}"
            vf_env.dataset = None

        start_time = time.time()
        result = vf_env.evaluate(
            client=inference_ctx.client,
            model=inference_ctx.model,
            sampling_args=sampling_args,
            num_examples=n_examples,
            rollouts_per_example=n_generations,
            max_concurrent=self.max_concurrent,
        )

        end_time = time.time()
        logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")

        return EnvStep(
            examples=result.prompt, responses=result.completion, rewards=result.reward, metrics=result.metrics
        )
