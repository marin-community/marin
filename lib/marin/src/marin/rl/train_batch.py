# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Training batch creation utilities for RL training.

This module owns the neutral trajectory-to-batch transform used by the RL
training stack. The current TrainingBatch remains available as a compatibility
adapter until replay and objective runtime refactors land.
"""

import logging
from typing import Any

import haliax as hax
import jax.numpy as jnp
import numpy as np

from .types import (
    BatchInfo,
    Rollout,
    RolloutGroup,
    RolloutWithAdvantage,
    SequenceBatch,
    TrainingBatch,
    TrajectoryGroupRecord,
    TrajectoryRecord,
)

logger = logging.getLogger(__name__)

TOP_K_NONE_SENTINEL = -1
UNKNOWN_TRAIN_STEP = -1
UNKNOWN_CORRECTNESS_REWARD = np.nan


def rollout_to_trajectory_record(rollout: Rollout) -> TrajectoryRecord:
    """Convert a rollout into the canonical neutral training record."""
    metadata = rollout.metadata
    return TrajectoryRecord(
        trace_id=metadata.trace_id,
        env_name=rollout.env_name,
        task_name=metadata.task_name,
        task_version=metadata.task_version,
        lesson_id=metadata.lesson_id,
        env_example_id=rollout.env_example_id,
        group_id=metadata.group_id,
        verifier_name=metadata.verifier_name,
        verifier_version=metadata.verifier_version,
        prompt_tokens=rollout.prompt_tokens,
        response_tokens=rollout.response_tokens,
        behavior_logprobs=rollout.response_logprobs,
        token_rewards=rollout.token_rewards,
        episode_reward=rollout.episode_reward,
        correctness_reward=rollout.correctness_reward,
        is_truncated=rollout.is_truncated,
        sampling_temperature=rollout.temperature,
        sampling_top_k=rollout.top_k,
        rollout_metadata=metadata,
        trace_ref=metadata.trace_ref,
    )


def rollout_group_to_trajectory_group_record(group: RolloutGroup) -> TrajectoryGroupRecord:
    """Convert a rollout group into a prompt-level grouped record."""
    if not group.rollouts:
        raise ValueError("Cannot convert an empty rollout group")

    prompt_tokens = group.rollouts[0].prompt_tokens
    for rollout in group.rollouts[1:]:
        if not np.array_equal(rollout.prompt_tokens, prompt_tokens):
            raise ValueError("All rollouts in a group must share the same prompt tokens")

    trajectories = tuple(rollout_to_trajectory_record(rollout) for rollout in group.rollouts)
    metadata = group.metadata
    first_trajectory = trajectories[0]
    return TrajectoryGroupRecord(
        group_id=metadata.group_id or first_trajectory.group_id,
        lesson_id=metadata.lesson_id or first_trajectory.lesson_id,
        trace_id=metadata.trace_id or first_trajectory.trace_id,
        task_name=metadata.task_name or first_trajectory.task_name,
        task_version=metadata.task_version or first_trajectory.task_version,
        verifier_name=metadata.verifier_name if metadata.verifier_name is not None else first_trajectory.verifier_name,
        verifier_version=(
            metadata.verifier_version if metadata.verifier_version is not None else first_trajectory.verifier_version
        ),
        prompt_tokens=prompt_tokens,
        trajectories=trajectories,
        trace_ref=metadata.trace_ref if metadata.trace_ref is not None else first_trajectory.trace_ref,
    )


def trim_and_pad(ary: np.ndarray, max_seq_len: int, pad_to: int, padding_value: int | float) -> np.ndarray:
    """Trim to max_seq_len and pad to pad_to."""
    ary = ary[:max_seq_len]
    if pad_to < len(ary):
        raise ValueError(f"pad_to ({pad_to}) must be >= trimmed length ({len(ary)})")
    ary = np.pad(ary, (0, pad_to - len(ary)), mode="constant", constant_values=padding_value)
    return ary


def convert_trajectory_to_sequence_format(
    trajectory: TrajectoryRecord,
    max_tokens: int,
    pad_to: int,
    pad_token_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert a trajectory into neutral sequence tensors and metadata."""
    input_tokens = np.concatenate([trajectory.prompt_tokens, trajectory.response_tokens])
    position_ids = np.arange(len(input_tokens), dtype=np.int32)

    prompt_mask = np.concatenate(
        [
            np.ones(len(trajectory.prompt_tokens), dtype=np.float32),
            np.zeros(len(trajectory.response_tokens), dtype=np.float32),
        ]
    )
    response_mask = np.concatenate(
        [
            np.zeros(len(trajectory.prompt_tokens), dtype=np.float32),
            np.ones(len(trajectory.response_tokens), dtype=np.float32),
        ]
    )
    behavior_logprobs = np.concatenate(
        [
            np.zeros(len(trajectory.prompt_tokens), dtype=np.float32),
            trajectory.behavior_logprobs.astype(np.float32),
        ]
    )
    token_rewards = np.concatenate(
        [
            np.zeros(len(trajectory.prompt_tokens), dtype=np.float32),
            trajectory.token_rewards.astype(np.float32),
        ]
    )

    input_ids = trim_and_pad(input_tokens, max_tokens, pad_to, padding_value=pad_token_id)
    position_ids = trim_and_pad(position_ids, max_tokens, pad_to, padding_value=0)
    prompt_mask = trim_and_pad(prompt_mask, max_tokens, pad_to, padding_value=0)
    response_mask = trim_and_pad(response_mask, max_tokens, pad_to, padding_value=0)
    behavior_logprobs = trim_and_pad(behavior_logprobs, max_tokens, pad_to, padding_value=0)
    token_rewards = trim_and_pad(token_rewards, max_tokens, pad_to, padding_value=0)

    metadata = trajectory.rollout_metadata
    sequence_example = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "prompt_mask": prompt_mask,
        "response_mask": response_mask,
        "behavior_logprobs": behavior_logprobs,
        "sampling_temperature": trajectory.sampling_temperature,
        "sampling_top_k": trajectory.sampling_top_k if trajectory.sampling_top_k is not None else TOP_K_NONE_SENTINEL,
        "truncated": trajectory.is_truncated,
    }
    info_example = {
        "group_id": trajectory.group_id,
        "lesson_id": trajectory.lesson_id,
        "env_name": trajectory.env_name,
        "env_example_id": trajectory.env_example_id,
        "task_name": trajectory.task_name,
        "task_version": trajectory.task_version,
        "verifier_name": trajectory.verifier_name,
        "verifier_version": trajectory.verifier_version,
        "worker_id": metadata.worker_id,
        "run_id": metadata.run_id,
        "trace_id": trajectory.trace_id,
        "trace_ref": trajectory.trace_ref,
        "prompt_length": int(prompt_mask.sum()),
        "response_length": int(response_mask.sum()),
        "episode_reward": trajectory.episode_reward,
        "token_rewards": token_rewards,
        "correctness_reward": (
            UNKNOWN_CORRECTNESS_REWARD
            if trajectory.correctness_reward is None
            else np.float32(trajectory.correctness_reward)
        ),
        "weight_step": metadata.weight_step,
        "train_step": UNKNOWN_TRAIN_STEP,
        "timestamp": metadata.timestamp,
    }
    return sequence_example, info_example


def convert_rollout_to_training_format(
    rollout: Rollout,
    advantage: float,
    max_tokens: int,
    pad_to: int,
    pad_token_id: int,
) -> dict[str, Any]:
    """Compatibility wrapper that materializes the legacy training example."""
    sequence_example, _ = convert_trajectory_to_sequence_format(
        rollout_to_trajectory_record(rollout),
        max_tokens=max_tokens,
        pad_to=pad_to,
        pad_token_id=pad_token_id,
    )
    return {
        "input_ids": sequence_example["input_ids"],
        "position_ids": sequence_example["position_ids"],
        "loss_weights": sequence_example["response_mask"] * np.float32(advantage),
        "loss_masks": sequence_example["response_mask"],
        "policy_logprobs": sequence_example["behavior_logprobs"],
        "temperature": sequence_example["sampling_temperature"],
        "top_k": sequence_example["sampling_top_k"],
        "truncated": sequence_example["truncated"],
    }


def _compute_pad_to(sequence_lengths: list[int], max_tokens: int, pad_to_multiple: int | None) -> int:
    batch_max_len = max(min(sequence_length, max_tokens) for sequence_length in sequence_lengths)
    pad_to = batch_max_len
    if pad_to_multiple is not None and pad_to_multiple > 0:
        pad_to = ((batch_max_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        if pad_to > max_tokens:
            raise ValueError(
                "Rounded batch padding length exceeds max_tokens. "
                f"batch_max_len={batch_max_len}, pad_to_multiple={pad_to_multiple}, max_tokens={max_tokens}. "
                "Increase max_seq_len or reduce flash_attention_block_size."
            )
    return pad_to


def create_sequence_batch_from_trajectories(
    trajectories: list[TrajectoryRecord],
    max_tokens: int,
    pad_token_id: int,
    pad_to_multiple: int | None = None,
) -> tuple[SequenceBatch, BatchInfo]:
    """Create a neutral sequence batch and sidecar metadata from trajectories."""
    if not trajectories:
        raise ValueError("Cannot create batch from empty trajectory list")

    pad_to = _compute_pad_to(
        [len(trajectory.prompt_tokens) + len(trajectory.response_tokens) for trajectory in trajectories],
        max_tokens=max_tokens,
        pad_to_multiple=pad_to_multiple,
    )

    sequence_examples = []
    info_examples = []
    for trajectory in trajectories:
        sequence_example, info_example = convert_trajectory_to_sequence_format(
            trajectory,
            max_tokens=max_tokens,
            pad_to=pad_to,
            pad_token_id=pad_token_id,
        )
        sequence_examples.append(sequence_example)
        info_examples.append(info_example)

    sequence_tensors = {
        "input_ids": jnp.stack([example["input_ids"] for example in sequence_examples], axis=0),
        "position_ids": jnp.stack([example["position_ids"] for example in sequence_examples], axis=0),
        "prompt_mask": jnp.stack([example["prompt_mask"] for example in sequence_examples], axis=0),
        "response_mask": jnp.stack([example["response_mask"] for example in sequence_examples], axis=0),
        "behavior_logprobs": jnp.stack([example["behavior_logprobs"] for example in sequence_examples], axis=0),
        "sampling_temperature": jnp.asarray(
            [example["sampling_temperature"] for example in sequence_examples],
            dtype=np.float32,
        ),
        "sampling_top_k": jnp.asarray(
            [example["sampling_top_k"] for example in sequence_examples],
            dtype=np.int32,
        ),
        "truncated": jnp.asarray([example["truncated"] for example in sequence_examples], dtype=np.bool_),
    }
    info_tensors = {
        "prompt_length": jnp.asarray([example["prompt_length"] for example in info_examples], dtype=np.int32),
        "response_length": jnp.asarray([example["response_length"] for example in info_examples], dtype=np.int32),
        "episode_reward": jnp.asarray([example["episode_reward"] for example in info_examples], dtype=np.float32),
        "token_rewards": jnp.stack([example["token_rewards"] for example in info_examples], axis=0),
        "correctness_reward": jnp.asarray(
            [example["correctness_reward"] for example in info_examples],
            dtype=np.float32,
        ),
        "weight_step": jnp.asarray([example["weight_step"] for example in info_examples], dtype=np.int32),
        "train_step": jnp.asarray([example["train_step"] for example in info_examples], dtype=np.int32),
    }

    per_row_mask_sum = sequence_tensors["response_mask"].sum(axis=1)
    zero_mask_rows = int((per_row_mask_sum == 0).sum())
    assert zero_mask_rows == 0, (
        f"Found {zero_mask_rows} rollouts with all-zero response masks. "
        "This happens when prompt_tokens >= max_seq_len, leaving no room for response tokens. "
        "Increase max_seq_len in CurriculumConfig."
    )

    sequence_batch = SequenceBatch(
        input_ids=hax.named(sequence_tensors["input_ids"], ["batch", "position"]),
        position_ids=hax.named(sequence_tensors["position_ids"], ["batch", "position"]),
        prompt_mask=hax.named(sequence_tensors["prompt_mask"], ["batch", "position"]),
        response_mask=hax.named(sequence_tensors["response_mask"], ["batch", "position"]),
        behavior_logprobs=hax.named(sequence_tensors["behavior_logprobs"], ["batch", "position"]),
        sampling_temperature=hax.named(sequence_tensors["sampling_temperature"], ["batch"]),
        sampling_top_k=hax.named(sequence_tensors["sampling_top_k"], ["batch"]),
        truncated=sequence_tensors["truncated"],
        max_output_tokens=max_tokens,
    )
    batch_info = BatchInfo(
        group_id=tuple(example["group_id"] for example in info_examples),
        lesson_id=tuple(example["lesson_id"] for example in info_examples),
        env_name=tuple(example["env_name"] for example in info_examples),
        env_example_id=tuple(example["env_example_id"] for example in info_examples),
        task_name=tuple(example["task_name"] for example in info_examples),
        task_version=tuple(example["task_version"] for example in info_examples),
        verifier_name=tuple(example["verifier_name"] for example in info_examples),
        verifier_version=tuple(example["verifier_version"] for example in info_examples),
        worker_id=tuple(example["worker_id"] for example in info_examples),
        run_id=tuple(example["run_id"] for example in info_examples),
        trace_id=tuple(example["trace_id"] for example in info_examples),
        trace_ref=tuple(example["trace_ref"] for example in info_examples),
        prompt_length=hax.named(info_tensors["prompt_length"], ["batch"]),
        response_length=hax.named(info_tensors["response_length"], ["batch"]),
        episode_reward=hax.named(info_tensors["episode_reward"], ["batch"]),
        token_rewards=hax.named(info_tensors["token_rewards"], ["batch", "position"]),
        correctness_reward=hax.named(info_tensors["correctness_reward"], ["batch"]),
        weight_step=hax.named(info_tensors["weight_step"], ["batch"]),
        train_step=hax.named(info_tensors["train_step"], ["batch"]),
        timestamp=tuple(example["timestamp"] for example in info_examples),
    )
    return sequence_batch, batch_info


def create_sequence_batch_from_rollouts(
    rollouts: list[Rollout],
    max_tokens: int,
    pad_token_id: int,
    pad_to_multiple: int | None = None,
) -> tuple[SequenceBatch, BatchInfo]:
    """Create a neutral batch directly from rollout objects."""
    trajectories = [rollout_to_trajectory_record(rollout) for rollout in rollouts]
    return create_sequence_batch_from_trajectories(
        trajectories,
        max_tokens=max_tokens,
        pad_token_id=pad_token_id,
        pad_to_multiple=pad_to_multiple,
    )


def create_training_batch_from_rollouts(
    individual_rollouts: list[RolloutWithAdvantage],
    max_tokens: int,
    pad_token_id: int,
    pad_to_multiple: int | None = None,
) -> TrainingBatch:
    """Create the legacy TrainingBatch from the neutral sequence batch path."""
    if not individual_rollouts:
        raise ValueError("Cannot create batch from empty rollout list")

    sequence_batch, _ = create_sequence_batch_from_rollouts(
        [individual.rollout for individual in individual_rollouts],
        max_tokens=max_tokens,
        pad_token_id=pad_token_id,
        pad_to_multiple=pad_to_multiple,
    )
    advantages = jnp.asarray([individual.advantage for individual in individual_rollouts], dtype=np.float32)
    loss_weights = sequence_batch.response_mask.array * advantages[:, None]

    return TrainingBatch(
        input_ids=sequence_batch.input_ids,
        position_ids=sequence_batch.position_ids,
        loss_weights=hax.named(loss_weights, ["batch", "position"]),
        loss_masks=sequence_batch.response_mask,
        policy_logprobs=sequence_batch.behavior_logprobs,
        temperature=sequence_batch.sampling_temperature,
        top_k=sequence_batch.sampling_top_k,
        truncated=sequence_batch.truncated,
        max_output_tokens=sequence_batch.max_output_tokens,
    )
