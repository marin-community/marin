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


def load_environment_from_spec(env_spec: str) -> MarinEnv:
    """Load environment from spec string."""
    print("Environment spec:", env_spec)
    env_name = env_spec.split(":")[0]
    env_args = {}
    if ":" in env_spec:
        env_arg_str = env_spec.split(":")[1]
        for arg in env_arg_str.split(","):
            key, value = arg.split("=")
            env_args[key] = value

    # hash hostname for seeding mock environment
    import socket

    host_name = socket.gethostname()
    seed = abs(hash(host_name))

    if env_name == "math":
        from .math import MathEnvironment

        return MathEnvironment(**env_args)
    elif env_name == "mock":
        from .mock_env import MockEnv

        return MockEnv(seed=seed, **env_args)
    elif env_name == "prime_intellect":
        from .prime_intellect_env import PrimeIntellectEnv

        return PrimeIntellectEnv(**env_args)
    else:
        raise ValueError(f"Unknown environment spec: {env_spec}")
