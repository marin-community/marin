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

import verifiers as vf

from marin.rl.types import EnvExample, InferenceResponse, RolloutGroup

from .base import MarinEnv

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

    def sample(
        self,
        n_examples: int,
        prng_key,
        mode: str = "train",
    ) -> list[EnvExample]:
        """Sample examples from the environment dataset.

        TODO: This environment needs to be updated to match the new API.
        The Prime Intellect environments use the verifiers library which has its own
        sampling/evaluation flow that doesn't match our current MarinEnv interface.
        """
        raise NotImplementedError("PrimeIntellectEnv needs to be updated to match the new API")

    def evaluate(
        self,
        examples: list[EnvExample],
        responses: list[InferenceResponse],
        max_input_length: int,
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        """Evaluate model responses and create rollouts.

        TODO: This environment needs to be updated to match the new API.
        The Prime Intellect environments use the verifiers library which has its own
        sampling/evaluation flow that doesn't match our current MarinEnv interface.
        """
        raise NotImplementedError("PrimeIntellectEnv needs to be updated to match the new API")
