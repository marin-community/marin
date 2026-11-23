# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare learner batches from exact-token binary rollout responses."""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import haliax as hax
import jax.numpy as jnp
import numpy as np

from levanter.inference.binary import (
    LOGPROB_MODE,
    POLICY_ID,
    POLICY_LOGPROBS,
    TOKEN_IDS,
    choice_expert_routes,
    choice_tensor_key,
    decode_frame,
)
from levanter.layers.sampler import LogprobMode
from levanter.models.snowball import SnowballLMHeadModel
from levanter.policy_model import PolicyBatch, prepare_policy_route_replay


@dataclass(frozen=True)
class BinaryRolloutChoice:
    """A binary generation frame and the choice to replay."""

    frame: bytes
    choice: int


def prepare_binary_policy_batch(
    model: SnowballLMHeadModel,
    batch: PolicyBatch,
    rollouts: Sequence[BinaryRolloutChoice],
    *,
    policy_id: str,
) -> PolicyBatch:
    """Validate serving tokens against the batch and attach old policy scores and routes."""
    if not policy_id or batch.segment_ids is not None:
        raise ValueError("Binary rollout replay requires a policy ID and unpacked trajectories")
    batch_axis, position = batch.tokens.resolve_axis(("batch", "position"))
    response = batch.response_mask.resolve_axis("response")
    if len(rollouts) != batch_axis.size or not 0 < response.size < position.size:
        raise ValueError("Provide one binary choice per learner row with room for a prompt")
    tokens = np.asarray(batch.tokens.rearrange((batch_axis, position)).array)
    attended = np.asarray(batch.attention_mask.rearrange((batch_axis, position)).array)
    if np.any((attended != 0) & (attended != 1)):
        raise ValueError("Binary rollout replay requires a binary attention mask")
    if model.config.replay_precision == "zero_kl":
        position_ids = np.asarray(batch.position_ids.rearrange((batch_axis, position)).array)
        sequence_positions = np.arange(position.size)
        prefix = sequence_positions < attended.sum(axis=1, keepdims=True)
        if not np.array_equal(attended, prefix) or np.any(prefix & (position_ids != sequence_positions)):
            raise ValueError("Snowball zero-KL replay requires right padding and positions starting from zero")
    response_mask = np.asarray(batch.response_mask.rearrange((batch_axis, response)).array).astype(bool)
    response_start = position.size - response.size
    expert_ids = np.zeros((*tokens.shape, model.config.num_layers, model.config.num_experts_per_token), dtype=np.int32)
    valid = np.zeros(tokens.shape, dtype=np.bool_)
    old_logprobs = np.zeros((batch_axis.size, response.size), dtype=np.float32)

    for row, rollout in enumerate(rollouts):
        metadata, tensors = decode_frame(rollout.frame)
        choices = metadata.get("choices")
        if (
            not isinstance(choices, list)
            or not 0 <= rollout.choice < len(choices)
            or not isinstance(choices[rollout.choice], dict)
            or choices[rollout.choice].get(POLICY_ID) != policy_id
        ):
            raise ValueError("Binary rollout does not belong to the expected policy")
        if metadata.get(LOGPROB_MODE) != LogprobMode.POLICY:
            raise ValueError("Learner replay requires raw policy log probabilities")
        positions = np.flatnonzero(attended[row])
        ids, route_valid = choice_expert_routes(tensors, rollout.choice, tokens[row, positions])
        if ids.shape != (model.config.num_layers, positions.size, model.config.num_experts_per_token):
            raise ValueError("Captured expert routes do not match the Snowball model dimensions")
        generated = tensors[choice_tensor_key(rollout.choice, TOKEN_IDS)]
        scores = tensors.get(choice_tensor_key(rollout.choice, POLICY_LOGPROBS))
        if (
            scores is None
            or scores.shape != generated.shape
            or scores.dtype != np.float32
            or not 0 < generated.size < positions.size
            or not np.all(np.isfinite(scores))
            or not route_valid[:-1].all()
        ):
            raise ValueError("Binary rollout is missing finite scores or predictor routes")
        targets = positions[-generated.size :] - response_start
        if np.any(targets < 0):
            raise ValueError("Binary rollout targets do not align with the learner response")
        learned_targets = np.flatnonzero(response_mask[row])
        if np.any(~np.isin(learned_targets, targets)):
            raise ValueError("Every learner response target must come from the binary rollout")
        expert_ids[row, positions] = ids.transpose(1, 0, 2)
        valid[row, positions] = route_valid
        old_logprobs[row, targets] = scores

    replay = prepare_policy_route_replay(
        model,
        batch.tokens,
        hax.named(jnp.asarray(expert_ids), (batch_axis, position, "layer", "top_k")),
        hax.named(jnp.asarray(valid), (batch_axis, position)),
    )
    return dataclasses.replace(
        batch, old_logprobs=hax.named(jnp.asarray(old_logprobs), (batch_axis, response)), route_replay=replay
    )
