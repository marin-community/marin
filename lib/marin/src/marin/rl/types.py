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
Type definitions for RL/post-training.

This module contains training-focused type definitions:
- Rollout types (Rollout, RolloutGroup, RolloutBatch, etc.)
- Training types (TrainingBatch, RolloutWithAdvantage)

For inference-related types, see marin.rl.inference_ctx
"""

from dataclasses import dataclass

import equinox as eqx
import haliax.haxtyping as ht
import jax
import numpy as np
from haliax import NamedArray


@dataclass
class RolloutStats:
    """Statistics from a Rollout. Used for curriculum feedback."""

    episode_reward: float
    env_example_id: str
    lesson_id: str
    temperature: float
    top_k: int | None


@dataclass(frozen=True)
class RolloutMetadata:
    """Metadata about when/where rollouts were generated."""

    worker_id: str = ""
    """Worker that generated the rollout."""

    timestamp: float = 0.0
    """The timestamp at which the rollout was generated."""

    weight_step: int = -1
    """The step at which the model weights were used to generate this rollout."""


class Rollout(eqx.Module):
    """A single rollout: one prompt + one generated response + rewards."""

    env_name: str
    """The name of the environment used to generate this rollout."""

    env_example_id: str
    """An identifier for the example used to initialize the environment."""

    prompt_tokens: np.ndarray
    """Array of (prompt_length,) token IDs representing the input prompt."""

    response_tokens: np.ndarray
    """Array of (response_length,) token IDs representing the generated response."""

    response_logprobs: np.ndarray
    """Array of (response_length,) log probabilities for each generated token."""

    token_rewards: np.ndarray
    """The reward assigned to each generated token."""

    episode_reward: float
    """The overall reward for the episode."""

    temperature: float
    """The temperature used to sample the response."""

    top_k: int | None
    """The top_k used to sample the response."""

    is_truncated: bool
    """True if the rollout was truncated due to length. False otherwise."""

    metadata: RolloutMetadata = RolloutMetadata()
    """Metadata about when/where this rollout was generated."""

    correctness_reward: float | None = None
    """The reward for the correctness of the response."""


class RolloutGroup(eqx.Module):
    """Multiple rollouts for the same prompt (e.g., n_generations samples)."""

    rollouts: list[Rollout]


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
    position_ids: ht.Int[NamedArray, "batch position"]
    loss_weights: ht.Float[NamedArray, "batch position"]
    loss_masks: ht.Int[NamedArray, "batch position"]
    policy_logprobs: ht.Float[NamedArray, "batch position"]
    temperature: ht.Float[NamedArray, "batch"]  # noqa: F821
    top_k: ht.Int[NamedArray, "batch"]  # noqa: F821
    truncated: jax.Array  # [batch] # Make this haxtyped array?
    max_output_tokens: int

    def __len__(self) -> int:
        return self.input_ids.axis_size("batch")
