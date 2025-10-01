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
Training batch creation utilities for RL training.

This module provides functions to convert rollout data into training-ready
batches with proper padding, masking, and advantage weighting.
"""

import logging

import haliax as hax
import jax.numpy as jnp
import numpy as np

from .types import Rollout, RolloutWithAdvantage, TrainingBatch

logger = logging.getLogger(__name__)


def trim_and_pad(ary: np.ndarray, max_seq_len: int, pad_token_id: int) -> np.ndarray:
    """Trim and pad array to max sequence length."""
    ary = ary[:max_seq_len]
    ary = np.pad(
        ary,
        (0, max_seq_len - len(ary)),
        mode="constant",
        constant_values=pad_token_id if ary.dtype == np.int32 else 0,
    )
    return ary


def convert_rollout_to_training_format(
    rollout: Rollout, advantage: float, max_input_length: int, max_output_length: int, pad_token_id: int
) -> dict:
    """Convert a single rollout to training format with advantage.

    Args:
        rollout: The rollout data to convert
        advantage: Precomputed advantage value for this rollout
        max_input_length: Maximum input sequence length for padding
        max_output_length: Maximum output sequence length for padding
        pad_token_id: Token ID to use for padding

    Returns:
        Dictionary containing training-ready arrays for the rollout
    """
    full_tokens = np.concatenate([rollout.prompt_tokens, rollout.response_tokens])
    full_mask = np.ones(len(full_tokens), dtype=np.int32)
    full_position_ids = np.maximum(np.cumsum(full_mask) - 1, 0).astype(np.int32)

    # Shifted for next-token prediction
    input_tokens = full_tokens[:-1]
    input_attention_mask = full_mask[:-1]
    target_tokens = full_tokens[1:]
    position_ids = full_position_ids[:-1]

    # Loss mask (only on response tokens)
    loss_mask = np.concatenate(
        [
            np.zeros(len(rollout.prompt_tokens) - 1, dtype=np.float32),
            np.ones(len(rollout.response_tokens), dtype=np.float32),
        ]
    )

    # Loss weights (advantage for all response tokens)
    loss_weight = np.concatenate(
        [
            np.zeros(len(rollout.prompt_tokens) - 1, dtype=np.float32),
            np.full(len(rollout.response_tokens), advantage, dtype=np.float32),
        ]
    )

    policy_logprob = np.concatenate(
        [np.zeros(len(rollout.prompt_tokens) - 1, dtype=np.float32), rollout.response_logprobs.astype(np.float32)]
    )

    max_seq_len = max_input_length + max_output_length

    return {
        "input_ids": trim_and_pad(input_tokens, max_seq_len, pad_token_id),
        "attention_mask": trim_and_pad(input_attention_mask, max_seq_len, pad_token_id),
        "position_ids": trim_and_pad(position_ids, max_seq_len, pad_token_id),
        "target_ids": trim_and_pad(target_tokens, max_seq_len, pad_token_id),
        "loss_weights": trim_and_pad(loss_weight, max_seq_len, pad_token_id),
        "loss_masks": trim_and_pad(loss_mask, max_seq_len, pad_token_id),
        "policy_logprobs": trim_and_pad(policy_logprob, max_seq_len, pad_token_id),
    }


def create_training_batch_from_rollouts(
    individual_rollouts: list[RolloutWithAdvantage], max_input_length: int, max_output_length: int, pad_token_id: int
) -> TrainingBatch:
    """Create a training batch from a list of individual rollouts.

    Args:
        individual_rollouts: List of RolloutWithAdvantage objects with precomputed advantages
        max_input_length: Maximum input sequence length for padding
        max_output_length: Maximum output sequence length for padding
        pad_token_id: Token ID to use for padding

    Returns:
        TrainingBatch ready for training
    """
    if not individual_rollouts:
        raise ValueError("Cannot create batch from empty rollout list")

    training_examples = []
    for individual in individual_rollouts:
        training_example = convert_rollout_to_training_format(
            individual.rollout,
            individual.advantage,
            max_input_length,
            max_output_length,
            pad_token_id,
        )
        training_examples.append(training_example)

    # Stack the examples into a single batch with proper 2D arrays
    stacked = {}
    for key in training_examples[0].keys():
        stacked[key] = jnp.stack([ex[key] for ex in training_examples], axis=0)

    return TrainingBatch(
        input_ids=hax.named(stacked["input_ids"], ["batch", "position"]),
        attention_mask=hax.named(stacked["attention_mask"], ["batch", "position"]),
        position_ids=hax.named(stacked["position_ids"], ["batch", "position"]),
        target_ids=hax.named(stacked["target_ids"], ["batch", "position"]),
        loss_weights=hax.named(stacked["loss_weights"], ["batch", "position"]),
        loss_masks=hax.named(stacked["loss_masks"], ["batch", "position"]),
        policy_logprobs=hax.named(stacked["policy_logprobs"], ["batch", "position"]),
    )
