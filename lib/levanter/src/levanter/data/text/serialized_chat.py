# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Loss masks for complete, already tokenized chat transcripts."""

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp

from levanter.data.text.examples import GrugLmExample


@partial(jax.jit, static_argnames=("message_start_id", "assistant_prefix"))
def mask_serialized_chat(
    example: GrugLmExample, *, message_start_id: int, assistant_prefix: tuple[int, ...]
) -> GrugLmExample:
    """Retain next-token loss only inside assistant message bodies."""
    if not assistant_prefix or assistant_prefix[0] != message_start_id:
        raise ValueError("assistant_prefix must begin with message_start_id")
    tokens = example.tokens
    positions = jnp.arange(tokens.shape[-1])
    header_end = positions >= len(assistant_prefix) - 1
    for offset, token in enumerate(assistant_prefix):
        header_end &= jnp.roll(tokens, len(assistant_prefix) - 1 - offset, axis=-1) == token
    segments = None
    if example.attn_mask.segment_ids is not None:
        segments = example.attn_mask.segment_ids[0]
        header_end &= segments == jnp.roll(segments, len(assistant_prefix) - 1)
    latest_message = jax.lax.associative_scan(
        jnp.maximum, jnp.where(tokens == message_start_id, positions, -1), axis=-1
    )
    latest_assistant = jax.lax.associative_scan(jnp.maximum, jnp.where(header_end, positions, -1), axis=-1)
    if segments is not None:
        boundary = (positions == 0) | (segments != jnp.roll(segments, 1))
        latest_boundary = jax.lax.associative_scan(jnp.maximum, jnp.where(boundary, positions, -1), axis=-1)
        latest_message = jnp.maximum(latest_message, latest_boundary)
    assistant = (latest_assistant >= latest_message) & (positions > latest_assistant) & (latest_assistant >= 0)
    # GrugLmExample weights index the prediction of the following token.
    weights = example.loss_weight * jnp.roll(assistant, -1, axis=-1)
    if segments is not None:
        weights *= segments == jnp.roll(segments, -1)
    return replace(example, loss_weight=weights.at[-1].set(0))
