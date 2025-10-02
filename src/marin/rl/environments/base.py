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

from abc import ABC, abstractmethod
from dataclasses import dataclass

from marin.rl.types import InferenceContext, RolloutGroup


class MarinEnv(ABC):
    """Abstract base class for RL environments.

    Environments manage datasets, generate responses, and evaluate them.
    Subclasses must implement sample() method.
    """

    @abstractmethod
    def sample(
        self,
        inference_ctx: InferenceContext,
        n_examples: int,
        n_generations: int,
        temperature: float,
        prng_key,
        mode: str = "train",
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        """Sample examples, generate responses, and create rollouts.

        Args:
            inference_ctx: Context for generating responses from the model
            n_examples: Number of examples to sample
            n_generations: Number of generations per example
            temperature: Sampling temperature for generation
            prng_key: JAX random key for sampling
            mode: "train" or "eval" - which dataset to sample from

        Returns:
            Tuple of (rollout_groups, metrics)
        """
        ...


@dataclass
class EnvConfig:
    """Configuration for an environment."""

    env_class: str
    """Fully qualified class name of the environment, e.g. 'marin.rl.environments.math.MathEnvironment'."""

    env_args: dict
    """Arguments to pass to the environment constructor."""


def load_environment_from_spec(config: EnvConfig) -> MarinEnv:
    """Load an environment from the given configuration."""
    env_class = config.env_class
    env_args = config.env_args
    # Dynamically import the environment class
    module_name, class_name = env_class.rsplit(".", 1)
    env_module = __import__(module_name, fromlist=[class_name])
    env_class = getattr(env_module, class_name)
    return env_class(**env_args)
