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


def trim_and_pad(ary: np.ndarray, max_seq_len: int, pad_to: int, padding_value: int | float) -> np.ndarray:
    """Trim to max_seq_len and pad to pad_to."""
    ary = ary[:max_seq_len]
    if pad_to < len(ary):
        raise ValueError(f"pad_to ({pad_to}) must be >= trimmed length ({len(ary)})")
    ary = np.pad(ary, (0, pad_to - len(ary)), mode="constant", constant_values=padding_value)
    return ary


def convert_rollout_to_training_format(
    rollout: Rollout,
    advantage: float,
    max_tokens: int,
    pad_to: int,
    pad_token_id: int,
) -> dict:
    """Convert a single rollout to training format with advantage.

    Args:
        rollout: The rollout data to convert
        advantage: Precomputed advantage value for this rollout
        max_tokens: Maximum sequence length for truncation
        pad_to: Target length to pad to after truncation
        pad_token_id: Token ID to use for padding

    Returns:
        Dictionary containing training-ready arrays for the rollout
    """
    input_tokens = np.concatenate([rollout.prompt_tokens, rollout.response_tokens])
    position_ids = np.arange(len(input_tokens), dtype=np.int32)

    # Loss mask (only on response tokens)
    loss_mask = np.concatenate(
        [
            np.zeros(len(rollout.prompt_tokens), dtype=np.float32),
            np.ones(len(rollout.response_tokens), dtype=np.float32),
        ]
    )

    # Loss weights (advantage for all response tokens)
    loss_weight = np.concatenate(
        [
            np.zeros(len(rollout.prompt_tokens), dtype=np.float32),
            np.full(len(rollout.response_tokens), advantage, dtype=np.float32),
        ]
    )

    policy_logprob = np.concatenate(
        [np.zeros(len(rollout.prompt_tokens), dtype=np.float32), rollout.response_logprobs.astype(np.float32)]
    )

    max_seq_len = max_tokens

    input_ids = trim_and_pad(input_tokens, max_seq_len, pad_to, padding_value=pad_token_id)
    position_ids = trim_and_pad(position_ids, max_seq_len, pad_to, padding_value=0)
    loss_weight = trim_and_pad(loss_weight, max_seq_len, pad_to, padding_value=0)
    loss_mask = trim_and_pad(loss_mask, max_seq_len, pad_to, padding_value=0)
    policy_logprob = trim_and_pad(policy_logprob, max_seq_len, pad_to, padding_value=0)

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "loss_weights": loss_weight,
        "loss_masks": loss_mask,
        "policy_logprobs": policy_logprob,
        "temperature": rollout.temperature,
        "truncated": rollout.is_truncated,
    }


def create_training_batch_from_rollouts(
    individual_rollouts: list[RolloutWithAdvantage],
    max_tokens: int,
    pad_token_id: int,
    pad_to_multiple: int | None = None,
) -> TrainingBatch:
    """Create a training batch from a list of individual rollouts.

    Args:
        individual_rollouts: List of RolloutWithAdvantage objects with precomputed advantages
        max_tokens: Maximum sequence length for truncation
        pad_token_id: Token ID to use for padding
        pad_to_multiple: Optional multiple to pad sequence length to for FlashAttention

    Returns:
        TrainingBatch ready for training
    """
    if not individual_rollouts:
        raise ValueError("Cannot create batch from empty rollout list")

    max_seq_len = max_tokens
    batch_max_len = max(
        min(len(individual.rollout.prompt_tokens) + len(individual.rollout.response_tokens), max_seq_len)
        for individual in individual_rollouts
    )
    pad_to = batch_max_len
    if pad_to_multiple is not None and pad_to_multiple > 0:
        pad_to = ((batch_max_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        if pad_to > max_seq_len:
            raise ValueError(
                "Rounded batch padding length exceeds max_tokens. "
                f"batch_max_len={batch_max_len}, pad_to_multiple={pad_to_multiple}, max_tokens={max_seq_len}. "
                "Increase max_seq_len or reduce flash_attention_block_size."
            )

    training_examples = []
    for individual in individual_rollouts:
        training_example = convert_rollout_to_training_format(
            individual.rollout,
            individual.advantage,
            max_tokens,
            pad_to,
            pad_token_id,
        )
        training_examples.append(training_example)

    # Stack the examples into a single batch with proper 2D arrays
    stacked = {}
    for key in training_examples[0].keys():
        if isinstance(training_examples[0][key], float):
            stacked[key] = jnp.array([ex[key] for ex in training_examples], dtype=np.float32)
        else:
            stacked[key] = jnp.stack([ex[key] for ex in training_examples], axis=0)

    # Ensure each row has at least one non-zero loss mask (otherwise division by zero -> NaN)
    per_row_mask_sum = stacked["loss_masks"].sum(axis=1)
    zero_mask_rows = int((per_row_mask_sum == 0).sum())
    assert zero_mask_rows == 0, (
        f"Found {zero_mask_rows} rollouts with all-zero loss masks. "
        "This happens when prompt_tokens >= max_seq_len, leaving no room for response tokens. "
        "Increase max_seq_len in CurriculumConfig."
    )

    batch = TrainingBatch(
        input_ids=hax.named(stacked["input_ids"], ["batch", "position"]),
        position_ids=hax.named(stacked["position_ids"], ["batch", "position"]),
        loss_weights=hax.named(stacked["loss_weights"], ["batch", "position"]),
        loss_masks=hax.named(stacked["loss_masks"], ["batch", "position"]),
        policy_logprobs=hax.named(stacked["policy_logprobs"], ["batch", "position"]),
        temperature=hax.named(stacked["temperature"], ["batch"]),
        truncated=stacked["truncated"],
        max_output_tokens=max_tokens,
    )

    return batch
