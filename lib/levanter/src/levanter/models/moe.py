# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for mixture-of-experts blocks.

The DenseMixer straight-through estimator lives here because both
[`levanter.models.qwen3_moe`][] and [`levanter.models.mixtral`][] need it and
neither model may import the other.
"""

from typing import Callable

import jax

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray


def dense_router_delta(
    x_flat: NamedArray,
    router_logits: NamedArray,
    gate_weight: NamedArray,
    up_weight: NamedArray,
    down_weight: NamedArray,
    act: Callable[[NamedArray], NamedArray],
    *,
    Experts: Axis,
    Embed: Axis,
    Mlp: Axis,
) -> NamedArray:
    """DenseMixer straight-through term for the router.

    Returns a ``[Token, Embed]`` array whose *value* is exactly zero but whose
    gradient is the dense, all-experts router gradient: the gradient the router
    logits would receive from a full-softmax dense forward ``sum_e E_e(x) * softmax(logits)_e``.

    Expert outputs are wrapped in ``stop_gradient`` so the delta's gradient
    reaches only ``router_logits`` (through the full softmax) and never the
    expert weights or ``x_flat``; the caller keeps the conventional sparse
    forward for the expert-parameter gradient. Adding this delta to the sparse
    output leaves the forward value unchanged.

    Args:
        x_flat: Token-major inputs ``[Token, Embed]``.
        router_logits: Gate logits ``[Token, Experts]``.
        gate_weight: ``gate_proj`` expert weights ``[Experts, Embed, Mlp]``.
        up_weight: ``up_proj`` expert weights ``[Experts, Embed, Mlp]``.
        down_weight: ``down_proj`` expert weights ``[Experts, Mlp, Embed]``.
        act: Elementwise activation applied to the gate projection.
    """
    dense_weights = hnn.softmax(router_logits.astype(jax.numpy.float32), axis=Experts).astype(x_flat.dtype)

    def add_expert(acc: NamedArray, expert) -> NamedArray:
        gate_w, up_w, down_w, weight = expert
        hidden = act(x_flat.dot(gate_w, axis=Embed)) * x_flat.dot(up_w, axis=Embed)
        hidden = hax.auto_sharded(hidden)
        expert_out = hidden.dot(down_w, axis=Mlp)
        return acc + jax.lax.stop_gradient(expert_out) * weight

    acc0 = hax.zeros(x_flat.axes, dtype=x_flat.dtype)
    # remat the fold so peak memory stays O(Token * Mlp): recompute each expert in the backward pass
    # instead of stacking Token x Experts x Mlp dense intermediates across the scan.
    dense = hax.fold(add_expert, Experts, remat=True)(acc0, (gate_weight, up_weight, down_weight, dense_weights))
    return dense - jax.lax.stop_gradient(dense)
