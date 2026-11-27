# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare learner batches from exact-token gRPC rollout responses."""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import haliax as hax
import jax.numpy as jnp
import numpy as np

from levanter.inference import inference_pb2
from levanter.inference.router_replay import read_router_replay
from levanter.models.snowball import SnowballLMHeadModel
from levanter.policy_model import PolicyBatch, prepare_policy_route_replay


@dataclass(frozen=True)
class RolloutChoice:
    """A generation response and the choice to replay."""

    response: inference_pb2.GenerateResponse
    choice: int


def prepare_rollout_policy_batch(
    model: SnowballLMHeadModel,
    batch: PolicyBatch,
    rollouts: Sequence[RolloutChoice],
    *,
    policy_id: str,
) -> PolicyBatch:
    """Validate serving tokens against the batch and attach old policy scores and routes."""
    if not policy_id or batch.segment_ids is not None:
        raise ValueError("Rollout replay requires a policy ID and unpacked trajectories")
    batch_axis, position = batch.tokens.resolve_axis(("batch", "position"))
    response = batch.response_mask.resolve_axis("response")
    if len(rollouts) != batch_axis.size or not 0 < response.size < position.size:
        raise ValueError("Provide one rollout choice per learner row with room for a prompt")
    tokens = np.asarray(batch.tokens.rearrange((batch_axis, position)).array)
    attended = np.asarray(batch.attention_mask.rearrange((batch_axis, position)).array)
    if np.any((attended != 0) & (attended != 1)):
        raise ValueError("Rollout replay requires a binary attention mask")
    position_ids = np.asarray(batch.position_ids.rearrange((batch_axis, position)).array)
    response_mask = np.asarray(batch.response_mask.rearrange((batch_axis, response)).array).astype(bool)
    response_start = position.size - response.size
    expert_ids = np.zeros((*tokens.shape, model.config.num_layers, model.config.num_experts_per_token), dtype=np.int32)
    valid = np.zeros(tokens.shape, dtype=np.bool_)
    old_logprobs = np.zeros((batch_axis.size, response.size), dtype=np.float32)

    for row, rollout in enumerate(rollouts):
        result = rollout.response
        if not 0 <= rollout.choice < len(result.choices):
            raise ValueError("Missing choice in generation response")
        choice = result.choices[rollout.choice]
        if choice.policy_id != policy_id:
            raise ValueError("Rollout does not belong to the expected policy")
        if result.logprob_mode != inference_pb2.LOGPROB_MODE_POLICY:
            raise ValueError("Learner replay requires raw policy log probabilities")
        positions = np.flatnonzero(attended[row])
        if np.any(np.diff(positions) != 1) or not np.array_equal(
            position_ids[row, positions], np.arange(positions.size)
        ):
            raise ValueError("Rollout replay requires contiguous tokens with position IDs starting from zero")
        ids, route_valid = read_router_replay(result, rollout.choice, tokens[row, positions])
        if ids.shape != (model.config.num_layers, positions.size, model.config.num_experts_per_token):
            raise ValueError("Captured expert routes do not match the Snowball model dimensions")
        generated = len(choice.token_ids)
        scores = np.asarray(choice.policy_logprobs, dtype=np.float32)
        if (
            scores.size != generated
            or not 0 < generated < positions.size
            or not np.all(np.isfinite(scores))
            or not route_valid[:-1].all()
        ):
            raise ValueError("Rollout is missing finite scores or predictor routes")
        targets = positions[-generated:] - response_start
        if np.any(targets < 0):
            raise ValueError("Rollout targets do not align with the learner response")
        learned_targets = np.flatnonzero(response_mask[row])
        if np.any(~np.isin(learned_targets, targets)):
            raise ValueError("Every learner response target must come from the rollout")
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
