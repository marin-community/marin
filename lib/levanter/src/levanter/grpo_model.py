# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Model-facing scoring and loss for persisted synchronous GRPO examples."""

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
from haliax import NamedArray

from levanter.grpo import GrpoConfig, grpo_loss
from levanter.layers.attention import AttentionMask
from levanter.metrics import Metric
from levanter.models.lm_model import LmHeadModel, split_activations
from levanter.models.loss import fused_cross_entropy_loss_and_logsumexp_penalty
from levanter.utils.tree_utils import inference_mode


class GrpoExample(eqx.Module):
    """A learner batch with complete sequences and aligned response objectives.

    Full sequences use axes (batch, position); objective arrays use
    (batch, response). The final response.size sequence positions are the
    response, including any masked trailing padding. Position IDs count real
    tokens from zero, independently of left padding. Objective weights and
    advantages must be computed before slicing execution microbatches.
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


def response_logprobs(
    model: LmHeadModel,
    batch: GrpoExample,
    *,
    key: jax.Array | None = None,
    block_size: int = 1024,
) -> NamedArray:
    """Score raw-policy response tokens without materializing full-vocab logits.

    Both scoring and training disable dropout. Padding forms a separate
    attention segment, so valid queries cannot attend to padded keys. Returned
    values include masked response positions; their objective weights are zero.
    """
    Position = batch.tokens.resolve_axis("position")
    Response = batch.response_mask.resolve_axis("response")
    if Response.size >= Position.size:
        raise ValueError("Every response needs at least one preceding prompt token")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    model = inference_mode(model, True)
    mask = AttentionMask.causal().with_segment_ids(batch.attention_mask.astype(jnp.int32))
    activations, _ = split_activations(model.activations(batch.tokens, mask, pos_ids=batch.position_ids, key=key))
    # A token at i is scored by the activation at i-1. Slice before the
    # vocabulary projection so prompt positions do not incur that cost.
    start = Position.size - Response.size
    predictions = activations[Position, hax.ds(start - 1, Response.size)].rename({Position.name: Response.name})
    targets = batch.tokens[Position, hax.ds(start, Response.size)].rename({Position.name: Response.name})
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


def grpo_model_loss(
    model: LmHeadModel,
    batch: GrpoExample,
    *,
    key: jax.Array | None,
    config: GrpoConfig,
    accumulation_steps: int,
    block_size: int,
) -> tuple[jax.Array, dict[str, Metric]]:
    """Compute the same inference-mode raw-policy scores and GRPO objective."""
    current = response_logprobs(model, batch, key=key, block_size=block_size)
    return grpo_loss(
        current,
        batch.old_logprobs,
        batch.reference_logprobs,
        batch.advantages,
        batch.policy_weights,
        batch.kl_weights,
        config=config,
        accumulation_steps=accumulation_steps,
    )
