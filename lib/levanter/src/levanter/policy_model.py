# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Model-facing scoring and loss for persisted synchronous policy examples."""

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import NamedArray

from levanter.data.packing import pack_documents
from levanter.policy import ClippedPolicyConfig, clipped_policy_loss
from levanter.layers.attention import AttentionMask
from levanter.metrics import Metric
from levanter.models.lm_model import LmHeadModel, split_activations
from levanter.models.loss import fused_cross_entropy_loss_and_logsumexp_penalty
from levanter.models.snowball import SnowballLMHeadModel, SnowballRouteReplay
from levanter.utils.tree_utils import inference_mode


class PolicyRouteReplay(eqx.Module):
    """Prepared expert IDs and their validity along the batch’s token axes."""

    expert_ids: NamedArray
    token_valid: NamedArray


class PolicyBatch(eqx.Module):
    """A learner batch with complete sequences and aligned response objectives.

    Full sequences use axes (batch, position); objective arrays use
    (batch, response). The final response.size sequence positions are the
    response, including any masked trailing padding. Position IDs count real
    tokens from zero, independently of left padding. Objective weights and
    advantages must be computed before slicing execution microbatches.
    Packed rows carry explicit trajectory segment IDs; unpacked examples may
    omit them to separate real tokens from padding with the attention mask.
    """

    tokens: NamedArray
    attention_mask: NamedArray
    position_ids: NamedArray
    response_mask: NamedArray
    advantages: NamedArray
    policy_weights: NamedArray
    kl_weights: NamedArray
    old_logprobs: NamedArray
    reference_logprobs: NamedArray | None
    segment_ids: NamedArray | None = None
    route_replay: PolicyRouteReplay | None = None


def prepare_policy_route_replay(
    model: SnowballLMHeadModel, tokens: NamedArray, expert_ids: NamedArray, token_valid: NamedArray
) -> PolicyRouteReplay:
    """Validate aligned Snowball routes before packing or learner JIT execution."""
    Batch, Position = tokens.resolve_axis(("batch", "position"))
    Layer, TopK = expert_ids.resolve_axis(("layer", "top_k"))
    valid = token_valid.rearrange((Batch, Position))
    ids = expert_ids.rearrange((Layer, Batch, Position, TopK))
    prepared = model.prepare_route_replay(ids.array, valid)
    return PolicyRouteReplay(hax.named(prepared.expert_ids, ids.axes), prepared.token_valid)


def pack_policy_batch(
    example: PolicyBatch,
    *,
    sequence_length: int,
    pad_token_id: int,
    batch_size_multiple: int = 1,
    minimum_rows: int = 1,
) -> PolicyBatch:
    """Pack complete trajectories on the host while preserving their objectives.

    Masked input padding is removed, source position IDs are retained, and
    response fields move to the predictor of their original target token.
    Prompt, cross-segment, and padding targets receive zero response masks and
    objective weights. Trajectories longer than the requested row are rejected.
    """
    if sequence_length < 2 or batch_size_multiple < 1 or minimum_rows < 1:
        raise ValueError("Packing requires sequence_length >= 2 and positive batch dimensions")
    if example.segment_ids is not None:
        raise ValueError("Packing requires unpacked trajectories without explicit segment IDs")
    source = {
        name: np.asarray(getattr(example, name).rearrange(("batch", "position")).array)
        for name in ("tokens", "attention_mask", "position_ids")
    }
    response_names = ("response_mask", "advantages", "policy_weights", "kl_weights", "old_logprobs")
    if example.reference_logprobs is not None:
        response_names += ("reference_logprobs",)
    source_objective = {
        name: np.asarray(getattr(example, name).rearrange(("batch", "response")).array) for name in response_names
    }
    if np.any((source["attention_mask"] != 0) & (source["attention_mask"] != 1)):
        raise ValueError("Packing requires a binary input attention mask")
    valid = source["attention_mask"].astype(bool)
    packs = pack_documents(valid.sum(axis=1), sequence_length, slice_strategy="raise")
    rows = ((max(minimum_rows, len(packs)) + batch_size_multiple - 1) // batch_size_multiple) * batch_size_multiple
    tokens = np.full((rows, sequence_length), pad_token_id, dtype=np.int32)
    segments = np.full((rows, sequence_length), -1, dtype=np.int32)
    positions = np.zeros((rows, sequence_length), dtype=np.int32)
    source_replay = example.route_replay
    if source_replay is not None:
        source_experts = np.asarray(source_replay.expert_ids.rearrange(("batch", "position", "layer", "top_k")).array)
        source_route_valid = np.asarray(source_replay.token_valid.rearrange(("batch", "position")).array)
        if np.any(source_route_valid & ~valid):
            raise ValueError("Expert routes must be within attended token positions")
        packed_experts = np.empty((rows, sequence_length, *source_experts.shape[2:]), dtype=np.int32)
        packed_experts[...] = np.arange(source_experts.shape[-1], dtype=np.int32)
        packed_route_valid = np.zeros((rows, sequence_length), dtype=np.bool_)
    objective = {
        name: np.zeros((rows, sequence_length - 1), dtype=value.dtype) for name, value in source_objective.items()
    }
    response_start = source["tokens"].shape[1] - source_objective["old_logprobs"].shape[1]
    if response_start < 1:
        raise ValueError("Each response needs a preceding prompt token")
    response_mask = source_objective["response_mask"]
    if source_replay is not None and np.any(response_mask > source_route_valid[:, response_start - 1 : -1]):
        raise ValueError("Every response predictor must have a captured expert route")
    if np.any(response_mask > valid[:, response_start:]) or np.any(response_mask > valid[:, response_start - 1 : -1]):
        raise ValueError("Each response target and its predecessor must be attended")
    if np.any(
        (response_mask == 0) & ((source_objective["policy_weights"] != 0) | (source_objective["kl_weights"] != 0))
    ):
        raise ValueError("Objective weights must be zero outside the response mask")
    for row, documents in enumerate(packs):
        offset = 0
        for document in documents:
            indices = np.flatnonzero(valid[document])
            count = len(indices)
            tokens[row, offset : offset + count] = source["tokens"][document, indices]
            segments[row, offset : offset + count] = document
            positions[row, offset : offset + count] = source["position_ids"][document, indices]
            if source_replay is not None:
                packed_experts[row, offset : offset + count] = source_experts[document, indices]
                packed_route_valid[row, offset : offset + count] = source_route_valid[document, indices]
            for compact_target, target in enumerate(indices):
                if target < response_start or compact_target == 0 or not valid[document, target - 1]:
                    continue
                response_index = target - response_start
                predictor = offset + compact_target - 1
                for name, value in source_objective.items():
                    objective[name][row, predictor] = value[document, response_index]
            offset += count
    Batch, Position, Response = (
        hax.Axis("batch", rows),
        hax.Axis("position", sequence_length),
        hax.Axis("response", sequence_length - 1),
    )

    def response(name: str) -> NamedArray:
        return hax.named(jnp.asarray(objective[name]), (Batch, Response))

    packed_replay = None
    if source_replay is not None:
        packed_replay = PolicyRouteReplay(
            hax.named(jnp.asarray(packed_experts), (Batch, Position, "layer", "top_k")),
            hax.named(jnp.asarray(packed_route_valid), (Batch, Position)),
        )
    return PolicyBatch(
        tokens=hax.named(jnp.asarray(tokens), (Batch, Position)),
        attention_mask=hax.named(jnp.asarray((segments >= 0).astype(np.int32)), (Batch, Position)),
        position_ids=hax.named(jnp.asarray(positions), (Batch, Position)),
        response_mask=response("response_mask"),
        advantages=response("advantages"),
        policy_weights=response("policy_weights"),
        kl_weights=response("kl_weights"),
        old_logprobs=response("old_logprobs"),
        reference_logprobs=response("reference_logprobs") if example.reference_logprobs is not None else None,
        segment_ids=hax.named(jnp.asarray(segments), (Batch, Position)),
        route_replay=packed_replay,
    )


def response_logprobs(
    model: LmHeadModel,
    batch: PolicyBatch,
    *,
    key: jax.Array | None = None,
    block_size: int = 1024,
) -> NamedArray:
    """Score raw-policy response tokens before applying the objective weights.

    Both scoring and training disable dropout. Padding forms a separate
    attention segment in standard scoring. Zero-KL scoring uses an unpacked,
    right-padded causal layout. Returned scores at masked positions have no
    effect on the objective.
    """
    Position = batch.tokens.resolve_axis("position")
    Response = batch.response_mask.resolve_axis("response")
    if Response.size >= Position.size:
        raise ValueError("Every response needs at least one preceding prompt token")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    model = inference_mode(model, True)
    replay = batch.route_replay
    mask = AttentionMask.causal()
    exact_model = (
        model if isinstance(model, SnowballLMHeadModel) and model.config.replay_precision == "zero_kl" else None
    )
    if exact_model is not None:
        if batch.segment_ids is not None or replay is None:
            raise ValueError(
                "Snowball zero-KL scoring requires unpacked, right-padded trajectories with captured routes"
            )
    else:
        segments = batch.segment_ids if batch.segment_ids is not None else batch.attention_mask.astype(jnp.int32)
        mask = mask.with_segment_ids(segments)
    if replay is None:
        activations, _ = split_activations(model.activations(batch.tokens, mask, pos_ids=batch.position_ids, key=key))
    else:
        if not isinstance(model, SnowballLMHeadModel):
            raise ValueError("Prepared expert route replay requires a Snowball learner")
        axes = tuple(axis.name for axis in batch.tokens.axes)
        routes = SnowballRouteReplay(
            replay.expert_ids.rearrange(("layer", *axes, "top_k")).array,
            replay.token_valid.rearrange(axes),
        )
        activations = model.activations_with_replay(batch.tokens, mask, routes)
    # A token at i is scored by the activation at i-1. Slice before the
    # vocabulary projection so prompt positions do not incur that cost.
    start = Position.size - Response.size
    predictions = activations[Position, hax.ds(start - 1, Response.size)].rename({Position.name: Response.name})
    targets = batch.tokens[Position, hax.ds(start, Response.size)].rename({Position.name: Response.name})
    if exact_model is not None:
        return exact_model.fixed_policy_logprobs(predictions, targets).rearrange(("batch", "response"))
    loss = fused_cross_entropy_loss_and_logsumexp_penalty(
        predictions.astype(jnp.float32),
        model.get_lm_head().astype(jnp.float32),
        Contract=model.Embed,
        Label=model.Vocab,
        target_y=targets,
        reduction=None,
        logsumexp_weight=0.0,
        block_size=block_size,
        dtype=jnp.float32,
    )
    return -loss.rearrange(("batch", "response"))


def policy_model_loss(
    model: LmHeadModel,
    batch: PolicyBatch,
    *,
    key: jax.Array | None,
    config: ClippedPolicyConfig,
    accumulation_steps: int,
    block_size: int,
) -> tuple[jax.Array, dict[str, Metric]]:
    """Return policy loss and metrics for a complete model batch."""
    current = response_logprobs(model, batch, key=key, block_size=block_size)
    return clipped_policy_loss(
        current,
        batch.old_logprobs,
        batch.reference_logprobs,
        batch.advantages,
        batch.policy_weights,
        batch.kl_weights,
        config=config,
        accumulation_steps=accumulation_steps,
    )
