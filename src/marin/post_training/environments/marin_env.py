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

from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol

import numpy as np


@dataclass
class DataExample:
    """Single data example with transformations for debugging."""

    raw_prompt: str
    raw_answer: str
    processed_prompt: str
    processed_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


class EnvStep(NamedTuple):
    """Container for a single interactive environment step.

    This class encapsulates all the data generated during one step of interaction
    with an environment, including the input problems (prompts), model responses,
    rewards computed, and additional metrics collected.

    Attributes:
        examples (list[dict[str, Any]]): A list of problem instances sampled from
            the dataset. Each instance contains data with keys:
                - 'prompt': Problem description
                - 'answer': Ground truth solution used for grading

        responses (list[list[dict[str, np.ndarray]]]): A nested list structure where
            responses[i][j] contains the j-th generated sample for the i-th problem
            in the batch. Each inner dict contains:
            - 'tokens': numpy array of generated token IDs
            - 'logprobs': numpy array of log probabilities for each generated token
            - Other generation-specific metadata

        rewards (np.ndarray): A 2D numpy array with shape
            (number of examples, number of generations per example) containing
            the computed reward for each generated response. Rewards are typically
            binary (0.0 or 1.0) indicating correctness, but can be continuous values.

        metrics (dict[str, float]): Additional scalar metrics computed during this
            environment step, such as:
            - Average reward across all responses
            - Format validation success rate
            - Average response length
            - Problem-specific evaluation metrics

    Example:
        >>> env_step = EnvStep(
        ...     examples=[{'prompt': 'What is 2+2?', 'answer': '4'}],
        ...     responses=[[[{'tokens': np.array([1, 2, 3]), 'logprobs': np.array([0.1, 0.2, 0.3])}]]],
        ...     rewards=np.array([[1.0]]),
        ...     metrics={'avg_reward': 1.0, 'avg_length': 3.0}
        ... )
    """

    examples: list[dict[str, Any]]
    responses: list[list[dict[str, np.ndarray]]]
    rewards: np.ndarray
    metrics: dict[str, float]


class InferenceContext(Protocol):
    """Protocol for inference providers that generate text from prompts.

    This decouples the backend (Flax vs Levanter) during our transition period.
    """

    @property
    def tokenizer(self):
        """Return the tokenizer."""
        ...

    def generate(
        self,
        prompts: list[str],
        temperature: float = 1.0,
        n_generations: int = 1,
    ) -> list[list[dict]]:
        """Generate responses for a batch of prompts.

        Returns:
            List of lists where outer list corresponds to prompts and
            inner list contains n_generations responses per prompt.
            Each response is a dict with 'tokens' and 'logprobs' arrays.
        """
        ...

    def compute_logprobs(
        self,
        input_tokens: np.ndarray,
        input_attention_mask: np.ndarray,
        target_tokens: np.ndarray,
        target_attention_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute log probabilities for given input/target pairs.

        Returns:
            Log probabilities for target tokens
        """
        ...


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
