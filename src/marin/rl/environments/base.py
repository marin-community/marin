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

from levanter.compat.hf_checkpoints import HfTokenizer

from marin.rl.types import EnvStep, InferenceContext


class MarinEnv:
    """Abstract base class for RL environments.

    This class defines the interface that all environment implementations must follow.
    Environments are responsible for:
    1. Managing datasets of problems/tasks
    2. Sampling problems for training or evaluation
    3. Running inference to generate responses
    4. Computing rewards based on responses
    5. Collecting metrics for monitoring training progress

    Subclasses must implement the `step` method to define environment-specific
    behavior for problem sampling, inference, and reward computation.

    Example:
        >>> class MyEnv(MarinEnv):
        ...     def __init__(self, dataset_path, **kwargs):
        ...         super().__init__(**kwargs)
        ...         self.dataset = load_dataset(dataset_path)
        ...
        ...     def step(self, sampler, params, n_examples, prng_key, **kwargs):
        ...         # Sample problems, run inference, compute rewards
        ...         return EnvStep(examples, responses, rewards, metrics)
    """

    def __init__(self, **kwargs):
        """Initialize the environment with environment-specific configuration.

        This method should be overridden by subclasses to perform any necessary
        setup such as:
        - Loading datasets and perform preprocessing
        - Configuring tokenizers (used for reward computation)
        """
        pass

    def step(
        self,
        inference_ctx: InferenceContext,
        n_examples: int,
        prng_key,
        mode: str = "train",
        n_generations: int = 1,
        temperature: float = 1.0,
        **kwargs,
    ) -> EnvStep:
        """Execute one step of environment interaction.

        This is the main interface method that subclasses must implement. It should:
        1. Sample a batch of problems from the dataset
        2. Generate model responses using the provided inference context
        3. Compute rewards by comparing responses to ground truth
        4. Collect metrics for monitoring and logging
        5. Return all data packaged in an EnvStep container

        Args:
            inference_ctx: Context for generating responses
            n_examples: Number of examples to sample
            prng_key: JAX random key
            mode: "train" or "eval"
            n_generations: Number of generations per example
            temperature: Generation temperature
            **kwargs: Additional environment-specific parameters

        Returns:
            EnvStep: A container with the sampled examples, generated responses,
                computed rewards, and collected metrics from this environment step.
        """
        raise NotImplementedError("Subclasses must implement the step method")


def load_environment_from_spec(env_spec: str, tokenizer: HfTokenizer) -> MarinEnv:
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

        return MathEnvironment(tokenizer=tokenizer, **env_args)
    elif env_name == "mock":
        from .mock_env import MockEnv

        return MockEnv(tokenizer=tokenizer, seed=seed, **env_args)
    elif env_name == "prime_intellect":
        from .prime_intellect_env import PrimeIntellectEnv

        return PrimeIntellectEnv(tokenizer=tokenizer, **env_args)
    else:
        raise ValueError(f"Unknown environment spec: {env_spec}")
