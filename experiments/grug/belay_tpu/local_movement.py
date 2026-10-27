# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combine local expert outputs with a pullback in expert row order."""

import jax
import jax.numpy as jnp


@jax.custom_vjp
def combine_expert_order(rows, order, inverse, weights):
    """Reduce top-k rows per token while retaining expert-ordered residuals."""
    del order
    slots = rows[inverse].reshape(*weights.shape, rows.shape[-1])
    return (slots.astype(jnp.float32) * weights[..., None]).sum(axis=1).astype(rows.dtype)


def _combine_fwd(rows, order, inverse, weights):
    return combine_expert_order(rows, order, inverse, weights), (rows, order, inverse, weights)


def _combine_bwd(residuals, gradient):
    rows, order, inverse, weights = residuals
    topk = weights.shape[1]
    # Gather from the small token buffer before expanding or multiplying.
    # Both row gradients and route-weight gradients are formed in expert order.
    token_gradient = gradient[order // topk].astype(jnp.float32)
    row_weights = weights.reshape(-1)[order]
    d_rows = (token_gradient * row_weights[:, None]).astype(rows.dtype)
    d_weights = (token_gradient * rows.astype(jnp.float32)).sum(axis=-1)
    d_weights = d_weights[inverse].reshape(weights.shape).astype(weights.dtype)
    return d_rows, None, None, d_weights


combine_expert_order.defvjp(_combine_fwd, _combine_bwd)
