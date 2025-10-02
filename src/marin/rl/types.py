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
- Inference types (InferenceChoice, InferenceResponse, InferenceContext)
- Rollout types (Rollout, RolloutGroup, RolloutBatch, etc.)
- Training types (TrainingBatch, RolloutWithAdvantage)
"""

from dataclasses import dataclass
from typing import Protocol

import equinox as eqx
import haliax.haxtyping as ht
import jax
import numpy as np
from haliax import NamedArray


@dataclass
class InferenceChoice:
    """A single choice from the inference provider."""

    response_text: str
    response_tokens: np.ndarray  # Shape: (sequence_length,)
    logprobs: np.ndarray  # Shape: (sequence_length,)


@dataclass
class InferenceResponse:
    """A single response from the inference provider."""

    prompt: str
    prompt_tokens: np.ndarray  # Shape: (prompt_length,)
    choices: list[InferenceChoice]


class InferenceContext(Protocol):
    """Protocol for inference providers that generate text from prompts.

    This decouples the backend (Flax vs Levanter) during our transition period.
    """

    @property
    def tokenizer(self):
        """Return the tokenizer."""
        ...

    def generate(self, prompts: list[str], temperature: float, n_generations: int) -> list[InferenceResponse]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of text prompts to generate from
            temperature: Sampling temperature
            n_generations: Number of generations per prompt

        Returns:
            List of InferenceResponse objects, one per input prompt.
            Each InferenceResponse contains n_generations choices.
        """
        ...

    def openai_client(self):
        """Return an OpenAI-compatible client for environments that need it.

        Returns:
            AsyncOpenAI or OpenAI client instance
        """
        ...


class Rollout(eqx.Module):
    """A single rollout: one prompt + one generated response + rewards."""

    env_name: str
    """The name of the environment used to generate this rollout."""

    env_example_id: str
    """An identifier for the example used to initialize the environment."""

    prompt_tokens: jax.Array
    """Array of (prompt_length,) token IDs representing the input prompt."""

    response_tokens: jax.Array
    """Array of (response_length,) token IDs representing the generated response."""

    response_logprobs: jax.Array
    """Array of (response_length,) log probabilities for each generated token."""

    token_rewards: jax.Array
    """The reward assigned to each generated token."""

    episode_reward: float
    """The overall reward for the episode."""


class RolloutGroup(eqx.Module):
    """Multiple rollouts for the same prompt (e.g., n_generations samples)."""

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
