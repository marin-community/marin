# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter-add local Grug MoE backend."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from jaxtyping import Array, Float, Int

from haliax._src.fp8_ragged_current import fp8_current_scaled_ragged_dot
from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _prepare_moe_dispatch,
    _zero_dropped_assignments,
    split_moe_w13_output,
)

# A grouped-matmul drop-in: (lhs[T,D], rhs[E,D,N], group_sizes[E]) -> [T,N]. The bf16 path
# uses ``haliax.nn.ragged_dot.ragged_dot``; the f8 validation path swaps in the all-E4M3 op.
RaggedDotFn = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


def _fp8_current_ragged_dot(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    """All-E4M3, current/per-step-scaled grouped GEMM (the GFP8-035 E4M3-range validation knob).

    Accumulates in float32 and uses the ``auto`` backend -- the same per-platform default the bf16
    ``ragged_dot`` resolves to (triton/mosaic on GPU). All-E4M3 operands share one f8 dtype, so the
    mixed-f8 backend walls don't apply; the E4M3 quantization is identical on any backend. (Forcing
    ``xla`` instead tripped an XLA-GPU layout-normalization RET_CHECK.) Result is cast back to the
    operand dtype so the surrounding bf16 graph is unchanged.
    """
    # preferred_element_type and implementation are custom_vjp nondiff_argnums -> pass positionally.
    out = fp8_current_scaled_ragged_dot(lhs, rhs, group_sizes, jnp.float32, "auto")
    return out.astype(lhs.dtype)


def _moe_mlp_local_scatter(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    moe_w13: Float[Array, "E D I2"],
    moe_w2: Float[Array, "E I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    ragged_dot_fn: RaggedDotFn = ragged_dot,
) -> tuple[Float[Array, "T D"], Int[Array, ""]]:
    """Local fallback MoE path: sorted grouped GMM then scatter-add combine.

    ``ragged_dot_fn`` selects the grouped-matmul kernel for both expert GEMMs: the default bf16
    ``ragged_dot`` (the ``scatter`` implementation), or :func:`_fp8_current_ragged_dot` for the
    all-E4M3 ``scatter_f8`` validation path.
    """
    x_dispatch, w_dispatch, token_dispatch, group_sizes = _prepare_moe_dispatch(
        x,
        selected_experts,
        combine_weights,
        num_experts=num_experts,
    )
    x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)

    with jax.named_scope("moe_up_down"):
        w13_out = tree_checkpoint_name(ragged_dot_fn(x_dispatch, moe_w13, group_sizes), _CHECKPOINT_EXPERT_HIDDEN)
        moe_dim = moe_w2.shape[1]
        gate, up = split_moe_w13_output(w13_out, intermediate_dim=moe_dim, interleaved=False)
        out_dispatch = tree_checkpoint_name(
            ragged_dot_fn(activation_fn(gate) * up, moe_w2, group_sizes),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("scatter"):
        out = jnp.zeros_like(x).at[token_dispatch].add(out_dispatch * w_dispatch[:, None], mode="drop")
    return out, _zero_dropped_assignments()
