# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Custom-VJP down/gather implementation for local Grug MoE."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from haliax.nn.ragged_dot import ragged_dot
from levanter.grug.grug_moe import _gather_sum_reference


def _custom_vjp_down_bwd(
    dout: jax.Array,
    h_interleaved: jax.Array,
    w_down: jax.Array,
    combine_weights: jax.Array,
    token_ids_sort: jax.Array,
    sorted_assignment_ids: jax.Array,
    expert_frequency_offset: jax.Array,
    dispatch_output: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    group_sizes = jnp.diff(expert_frequency_offset)
    assignments = h_interleaved.shape[0]
    sorted_scores = combine_weights.reshape(assignments)[sorted_assignment_ids].astype(jnp.float32)
    dout_sorted = dout[token_ids_sort]

    def activation_forward(h: jax.Array) -> jax.Array:
        gate = h[:, 0::2]
        up = h[:, 1::2]
        return jax.nn.silu(gate) * up

    hidden, activation_pullback = jax.vjp(activation_forward, h_interleaved)
    weighted_dout = (dout_sorted.astype(jnp.float32) * sorted_scores[:, None]).astype(dispatch_output.dtype)
    _, down_pullback = jax.vjp(lambda h, w: ragged_dot(h, w, group_sizes), hidden, w_down)
    d_hidden, d_w_down = down_pullback(weighted_dout)
    (d_h_interleaved,) = activation_pullback(d_hidden)
    d_scores_sorted = jnp.sum(dout_sorted.astype(jnp.float32) * dispatch_output.astype(jnp.float32), axis=-1)
    d_scores = jnp.zeros_like(sorted_scores).at[sorted_assignment_ids].set(d_scores_sorted)
    return d_h_interleaved, d_scores, d_w_down


@jax.custom_vjp
def custom_vjp_interleaved_down_gather_sum(
    w13_out_interleaved: jax.Array,
    combine_weights: jax.Array,
    w_down: jax.Array,
    token_ids_sort: jax.Array,
    sorted_assignment_ids: jax.Array,
    dispatch_positions: jax.Array,
    group_sizes: jax.Array,
) -> jax.Array:
    out, _ = _custom_vjp_interleaved_down_gather_sum_forward(
        w13_out_interleaved,
        combine_weights,
        w_down,
        token_ids_sort,
        sorted_assignment_ids,
        dispatch_positions,
        group_sizes,
    )
    return out


def _custom_vjp_interleaved_down_gather_sum_forward(
    w13_out_interleaved: jax.Array,
    combine_weights: jax.Array,
    w_down: jax.Array,
    token_ids_sort: jax.Array,
    sorted_assignment_ids: jax.Array,
    dispatch_positions: jax.Array,
    group_sizes: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    del sorted_assignment_ids
    hidden = jax.nn.silu(w13_out_interleaved[:, 0::2]) * w13_out_interleaved[:, 1::2]
    dispatch_output = ragged_dot(hidden, w_down, group_sizes)
    out = _gather_sum_reference(dispatch_output, dispatch_positions, combine_weights)
    expert_frequency_offset = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(group_sizes, dtype=jnp.int32)]
    )
    return out, (
        w13_out_interleaved,
        combine_weights,
        w_down,
        token_ids_sort,
        expert_frequency_offset,
        dispatch_output,
    )


def _custom_vjp_interleaved_down_gather_sum_fwd(
    w13_out_interleaved: jax.Array,
    combine_weights: jax.Array,
    w_down: jax.Array,
    token_ids_sort: jax.Array,
    sorted_assignment_ids: jax.Array,
    dispatch_positions: jax.Array,
    group_sizes: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    out, residuals = _custom_vjp_interleaved_down_gather_sum_forward(
        w13_out_interleaved,
        combine_weights,
        w_down,
        token_ids_sort,
        sorted_assignment_ids,
        dispatch_positions,
        group_sizes,
    )
    return out, (*residuals, sorted_assignment_ids)


def _custom_vjp_interleaved_down_gather_sum_bwd(
    residuals: tuple[jax.Array, ...],
    dout: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, None, None, None, None]:
    (
        w13_out_interleaved,
        combine_weights,
        w_down,
        token_ids_sort,
        expert_frequency_offset,
        dispatch_output,
        sorted_assignment_ids,
    ) = residuals
    d_h_interleaved, d_scores_flat, d_w_down = _custom_vjp_down_bwd(
        dout,
        w13_out_interleaved,
        w_down,
        combine_weights,
        token_ids_sort,
        sorted_assignment_ids,
        expert_frequency_offset,
        dispatch_output,
    )
    d_combine_weights = d_scores_flat.reshape(combine_weights.shape).astype(combine_weights.dtype)
    return d_h_interleaved, d_combine_weights, d_w_down.astype(w_down.dtype), None, None, None, None


custom_vjp_interleaved_down_gather_sum.defvjp(
    _custom_vjp_interleaved_down_gather_sum_fwd,
    _custom_vjp_interleaved_down_gather_sum_bwd,
)
