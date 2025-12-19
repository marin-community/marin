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
from dataclasses import dataclass, field

import haliax as hax
import jax.numpy as jnp
import numpy as np

from .types import Rollout, RolloutWithAdvantage, TrainingBatch

logger = logging.getLogger(__name__)


def trim_and_pad(ary: np.ndarray, max_seq_len: int, padding_value: int | float) -> np.ndarray:
    """Trim and pad array to max sequence length."""
    ary = ary[:max_seq_len]
    ary = np.pad(ary, (0, max_seq_len - len(ary)), mode="constant", constant_values=padding_value)
    return ary


def convert_rollout_to_training_segment(rollout: Rollout, advantage: float, max_tokens: int) -> dict:
    """Convert a single rollout into an unpadded training segment with advantage.

    Args:
        rollout: The rollout data to convert
        advantage: Precomputed advantage value for this rollout
        max_tokens: Maximum sequence length before truncation

    Returns:
        Dictionary containing training-ready arrays for the rollout segment
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

    max_seq_len = min(max_tokens, len(input_tokens))

    input_ids = input_tokens[:max_seq_len]
    position_ids = position_ids[:max_seq_len]
    loss_weight = loss_weight[:max_seq_len]
    loss_mask = loss_mask[:max_seq_len]
    policy_logprob = policy_logprob[:max_seq_len]

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "loss_weights": loss_weight,
        "loss_masks": loss_mask,
        "policy_logprobs": policy_logprob,
        "temperature": rollout.temperature,
        "truncated": rollout.is_truncated,
    }


def convert_rollout_to_training_format(rollout: Rollout, advantage: float, max_tokens: int, pad_token_id: int) -> dict:
    """Convert a single rollout to training format with advantage.

    Args:
        rollout: The rollout data to convert
        advantage: Precomputed advantage value for this rollout
        max_tokens: Maximum sequence length for padding
        pad_token_id: Token ID to use for padding

    Returns:
        Dictionary containing training-ready arrays for the rollout
    """
    segment = convert_rollout_to_training_segment(rollout, advantage, max_tokens)
    max_seq_len = max_tokens

    input_ids = trim_and_pad(segment["input_ids"], max_seq_len, padding_value=pad_token_id)
    position_ids = trim_and_pad(segment["position_ids"], max_seq_len, padding_value=0)
    loss_weight = trim_and_pad(segment["loss_weights"], max_seq_len, padding_value=0)
    loss_mask = trim_and_pad(segment["loss_masks"], max_seq_len, padding_value=0)
    policy_logprob = trim_and_pad(segment["policy_logprobs"], max_seq_len, padding_value=0)

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "loss_weights": loss_weight,
        "loss_masks": loss_mask,
        "policy_logprobs": policy_logprob,
        "temperature": segment["temperature"],
        "truncated": segment["truncated"],
    }


@dataclass
class _PackedExample:
    input_ids: list[int] = field(default_factory=list)
    position_ids: list[int] = field(default_factory=list)
    loss_weights: list[float] = field(default_factory=list)
    loss_masks: list[float] = field(default_factory=list)
    policy_logprobs: list[float] = field(default_factory=list)
    segment_ids: list[int] = field(default_factory=list)
    truncated_mask: list[float] = field(default_factory=list)
    temperature: float | None = None
    num_segments: int = 0

    def can_pack(self, length: int, temperature: float, max_tokens: int, max_segments_per_example: int) -> bool:
        if (len(self.input_ids) + length) > max_tokens:
            return False
        if self.num_segments >= max_segments_per_example:
            return False
        if self.temperature is not None and not np.isclose(self.temperature, temperature):
            return False
        return True

    def add_segment(self, segment: dict) -> None:
        segment_id = self.num_segments
        segment_length = len(segment["input_ids"])

        self.input_ids.extend(segment["input_ids"].tolist())
        self.position_ids.extend(segment["position_ids"].tolist())
        self.loss_weights.extend(segment["loss_weights"].tolist())
        self.loss_masks.extend(segment["loss_masks"].tolist())
        self.policy_logprobs.extend(segment["policy_logprobs"].tolist())
        self.segment_ids.extend([segment_id] * segment_length)
        if segment["truncated"]:
            self.truncated_mask.extend([1.0] * segment_length)
        else:
            self.truncated_mask.extend([0.0] * segment_length)

        if self.temperature is None:
            self.temperature = float(segment["temperature"])

        self.num_segments += 1

    def pad(self, max_tokens: int, pad_token_id: int) -> dict:
        input_ids = trim_and_pad(np.array(self.input_ids, dtype=np.int32), max_tokens, padding_value=pad_token_id)
        position_ids = trim_and_pad(np.array(self.position_ids, dtype=np.int32), max_tokens, padding_value=0)
        loss_weights = trim_and_pad(np.array(self.loss_weights, dtype=np.float32), max_tokens, padding_value=0)
        loss_masks = trim_and_pad(np.array(self.loss_masks, dtype=np.float32), max_tokens, padding_value=0)
        policy_logprobs = trim_and_pad(np.array(self.policy_logprobs, dtype=np.float32), max_tokens, padding_value=0)
        segment_ids = trim_and_pad(np.array(self.segment_ids, dtype=np.int32), max_tokens, padding_value=-1)
        truncated_mask = trim_and_pad(np.array(self.truncated_mask, dtype=np.float32), max_tokens, padding_value=0)

        if self.temperature is None:
            raise ValueError("Packed example missing temperature assignment")

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_weights": loss_weights,
            "loss_masks": loss_masks,
            "policy_logprobs": policy_logprobs,
            "segment_ids": segment_ids,
            "truncated": truncated_mask,
            "temperature": float(self.temperature),
        }


def create_training_batch_from_rollouts(
    individual_rollouts: list[RolloutWithAdvantage], max_tokens: int, pad_token_id: int
) -> TrainingBatch:
    """Create a training batch from a list of individual rollouts.

    Args:
        individual_rollouts: List of RolloutWithAdvantage objects with precomputed advantages
        max_tokens: Maximum sequence length for padding
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
            max_tokens,
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
        segment_ids=None,
        loss_weights=hax.named(stacked["loss_weights"], ["batch", "position"]),
        loss_masks=hax.named(stacked["loss_masks"], ["batch", "position"]),
        policy_logprobs=hax.named(stacked["policy_logprobs"], ["batch", "position"]),
        temperature=hax.named(stacked["temperature"], ["batch"]),
        truncated=stacked["truncated"],
        max_output_tokens=max_tokens,
    )

    return batch


def create_packed_training_batch_from_rollouts(
    rollouts: list[RolloutWithAdvantage],
    max_tokens: int,
    pad_token_id: int,
    target_batch_size: int,
    max_segments_per_example: int = 64,
) -> tuple[TrainingBatch | None, int]:
    """Pack multiple rollouts into a fixed-size training batch.

    Returns:
        Tuple of (TrainingBatch or None, number of rollouts consumed).
    """
    if not rollouts:
        return None, 0

    packers: list[_PackedExample] = []
    used_rollouts = 0

    for rollout_with_adv in rollouts:
        segment = convert_rollout_to_training_segment(rollout_with_adv.rollout, rollout_with_adv.advantage, max_tokens)
        placed = False
        for packer in packers:
            if packer.can_pack(len(segment["input_ids"]), segment["temperature"], max_tokens, max_segments_per_example):
                packer.add_segment(segment)
                placed = True
                break

        if not placed:
            if len(packers) >= target_batch_size:
                break
            new_packer = _PackedExample()
            new_packer.add_segment(segment)
            packers.append(new_packer)

        used_rollouts += 1

    if len(packers) < target_batch_size:
        return None, 0

    packed_examples = [packer.pad(max_tokens, pad_token_id) for packer in packers]

    stacked = {}
    for key in packed_examples[0].keys():
        if key == "temperature":
            stacked[key] = jnp.array([ex[key] for ex in packed_examples], dtype=np.float32)
        else:
            stacked[key] = jnp.stack([ex[key] for ex in packed_examples], axis=0)

    per_row_mask_sum = stacked["loss_masks"].sum(axis=1)
    zero_mask_rows = int((per_row_mask_sum == 0).sum())
    assert zero_mask_rows == 0, (
        f"Found {zero_mask_rows} packed examples with all-zero loss masks. "
        "This indicates that packed rollouts have no response tokens."
    )

    batch = TrainingBatch(
        input_ids=hax.named(stacked["input_ids"], ["batch", "position"]),
        position_ids=hax.named(stacked["position_ids"], ["batch", "position"]),
        segment_ids=hax.named(stacked["segment_ids"], ["batch", "position"]),
        loss_weights=hax.named(stacked["loss_weights"], ["batch", "position"]),
        loss_masks=hax.named(stacked["loss_masks"], ["batch", "position"]),
        policy_logprobs=hax.named(stacked["policy_logprobs"], ["batch", "position"]),
        temperature=hax.named(stacked["temperature"], ["batch"]),
        truncated=stacked["truncated"],
        max_output_tokens=max_tokens,
    )

    return batch, used_rollouts
