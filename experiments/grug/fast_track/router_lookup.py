# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact FP32 router selection for unique expert IDs within each token row."""

from functools import partial

import jax
import jax.numpy as jnp


def _select_bits(logits, selected, implementation):
    bits = jax.lax.bitcast_convert_type(logits, jnp.uint32)
    expert_ids = jnp.arange(logits.shape[-1])
    if implementation == "broadcast":
        mask = selected[..., None] == expert_ids
        result = jnp.bitwise_or.reduce(jnp.where(mask, bits[:, None, :], 0), axis=-1)
    else:
        result = jnp.stack(
            [
                jnp.bitwise_or.reduce(jnp.where(expert_ids == selected[:, i, None], bits, 0), axis=-1)
                for i in range(selected.shape[-1])
            ],
            axis=-1,
        )
    return jax.lax.bitcast_convert_type(result, jnp.float32)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def selected_logits(logits, selected, experts, implementation):
    """Select FP32 logits for unique, in-range expert IDs in each token row."""
    del experts
    return _select_bits(logits, selected, implementation)


def _selected_fwd(logits, selected, experts, implementation):
    del experts
    return _select_bits(logits, selected, implementation), selected


def _selected_bwd(experts, implementation, selected, cotangent):
    # Match the reference scatter-add into a positive-zero initialized buffer.
    cotangent = jnp.where(cotangent == 0, jnp.float32(0), cotangent)
    bits = jax.lax.bitcast_convert_type(cotangent, jnp.uint32)
    expert_ids = jnp.arange(experts)
    if implementation == "broadcast":
        mask = selected[..., None] == expert_ids
        result = jnp.bitwise_or.reduce(jnp.where(mask, bits[..., None], 0), axis=1)
    else:
        result = jnp.zeros((selected.shape[0], experts), jnp.uint32)
        for i in range(selected.shape[-1]):
            result |= jnp.where(expert_ids == selected[:, i, None], bits[:, i, None], 0)
    return jax.lax.bitcast_convert_type(result, jnp.float32), None


selected_logits.defvjp(_selected_fwd, _selected_bwd)


def unrolled_router_logits(logits, selected):
    """Select logits and transpose cotangents without a general gather/scatter."""
    return selected_logits(logits, selected, logits.shape[-1], "unrolled")
