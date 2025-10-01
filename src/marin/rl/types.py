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
Consolidated type definitions for RL/post-training.

This module contains all shared type definitions used across the RL system:
- Environment types (EnvResponse, EnvExample, EnvStep)
- Inference context protocol
- Rollout types (Rollout, RolloutGroup, RolloutBatch, etc.)
- Training types (TrainingBatch, RolloutWithAdvantage)
"""

from dataclasses import dataclass
from typing import NamedTuple, Protocol, TypedDict

import equinox as eqx
import haliax.haxtyping as ht
import jax
import numpy as np
from haliax import NamedArray


class EnvResponse(TypedDict):
    """Response from model generation."""

    tokens: np.ndarray
    logprobs: np.ndarray


class EnvExample(TypedDict):
    """A single environment example."""

    prompt: str
    answer: str


class EnvStep(NamedTuple):
    """Container for a single interactive environment step.

    This class encapsulates all the data generated during one step of interaction
    with an environment, including the input problems (prompts), model responses,
    rewards computed, and additional metrics collected.

    Attributes:
        examples (list[EnvExample]): A list of problem instances sampled from
            the dataset. Each instance contains data with keys:
                - 'prompt': Problem description
                - 'answer': Ground truth solution used for grading

        responses (list[list[EnvResponse]]): A nested list structure where
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

    examples: list[EnvExample]
    responses: list[list[EnvResponse]]
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

    def generate(self, prompts: list[str], temperature: float, n_generations: int) -> list[list[EnvResponse]]:
        """Generate responses for a batch of prompts.

        Returns:
            List of lists where outer list corresponds to prompts and
            inner list contains n_generations responses per prompt.
            Each response is an EnvResponse.
        """
        ...


class Rollout(eqx.Module):
    """A single rollout: one prompt + one generated response + rewards."""

    env_name: str
    env_example_id: str
    prompt_tokens: jax.Array
    response_tokens: jax.Array
    response_logprobs: jax.Array
    token_rewards: jax.Array
    episode_reward: float


class RolloutGroup(eqx.Module):
    """Multiple rollouts for the same prompt (e.g., n_generations samples)."""

    key: str
    rollouts: list[Rollout]


@dataclass
class RolloutMetadata:
    """Metadata about when/where rollouts were generated."""

    worker_id: str
    timestamp: float

    """The step at which the model weights were used to generate this rollout."""
    weight_step: int


class RolloutBatch(eqx.Module):
    """A batch of rollout groups with metadata."""

    groups: list[RolloutGroup]
    metadata: RolloutMetadata


@dataclass
class RolloutWithAdvantage:
    """A rollout paired with its computed advantage."""

    rollout: Rollout
    advantage: float


class TrainingBatch(eqx.Module):
    """A batch ready for training with Haliax named arrays."""

    input_ids: ht.Int[NamedArray, "batch position"]
    attention_mask: ht.Int[NamedArray, "batch position"]
    position_ids: ht.Int[NamedArray, "batch position"]
    target_ids: ht.Int[NamedArray, "batch position"]
    loss_weights: ht.Float[NamedArray, "batch position"]
    loss_masks: ht.Int[NamedArray, "batch position"]
    policy_logprobs: ht.Float[NamedArray, "batch position"]

    def __len__(self) -> int:
        return self.input_ids.axis_size("batch")
