# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent full-vocabulary policy scoring for numerical tests."""

import haliax as hax
import jax
import jax.numpy as jnp
from haliax import NamedArray

from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmHeadModel
from levanter.policy_model import PolicyBatch
from levanter.utils.tree_utils import inference_mode


def full_softmax_response_logprobs(model: LmHeadModel, batch: PolicyBatch) -> NamedArray:
    """Score response targets with a full-vocabulary softmax reference."""
    model = inference_mode(model, True)
    segments = batch.segment_ids if batch.segment_ids is not None else batch.attention_mask.astype(jnp.int32)
    mask = AttentionMask.causal().with_segment_ids(segments)
    logits = model(batch.tokens, mask, pos_ids=batch.position_ids).rearrange(("batch", "position", "vocab"))
    tokens = batch.tokens.rearrange(("batch", "position"))
    width = batch.response_mask.axis_size("response")
    # Explicit shifted full-vocabulary reference, independent of fused CE slicing.
    logits = logits.array.astype(jnp.float32)[:, :-1]
    shifted = logits - jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
    # TPU's default log approximation is not a full-precision reference.
    accuracy = jax.lax.AccuracyMode.HIGHEST if jax.default_backend() == "tpu" else None
    logprobs = shifted - jax.lax.log(
        jnp.sum(jax.lax.exp(shifted, accuracy=accuracy), axis=-1, keepdims=True), accuracy=accuracy
    )
    selected = jnp.take_along_axis(logprobs, tokens.array[:, 1:, None], axis=-1)[..., 0]
    return hax.named(selected[:, -width:], batch.response_mask.axes)

