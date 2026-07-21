# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Local Grug MoE backend using Tri Dao's QuACK SM100 kernels (SonicMoE) on B200.

Dispatch/combine as in ``scatter``, but the expert MLP GEMMs run on QuACK's
``GemmGatedSm100`` / ``GemmDefaultSm100`` via the vendored ``cutlass.jax.cutlass_call``
shim. QuACK does all four activation-path grouped GEMMs (gate/up fwd fused with
SwiGLU, down fwd, and the ``dh``/``dx`` backward matmuls); the SwiGLU backward is
elementwise in JAX; the two weight-gradient GEMMs (``dw13``/``dw2``) stay on XLA
``ragged_dot`` (a different varlen-k grouping). QuACK covers ~2/3 of the MoE FLOPs.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from jaxtyping import Array, Float, Int

from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _prepare_moe_dispatch,
    _zero_dropped_assignments,
)
from levanter.grug._moe.quack_moe_cute import quack_gated_grouped_gemm, quack_grouped_gemm


def _interleave_gate_up(moe_w13: jax.Array, moe_dim: int) -> jax.Array:
    """grug w13 [E,H,2I] gate=[:I], up=[I:] -> interleaved [g0,u0,g1,u1,...] (QuACK layout)."""
    gate = moe_w13[..., :moe_dim]
    up = moe_w13[..., moe_dim:]
    return jnp.stack([gate, up], axis=-1).reshape(moe_w13.shape)


@jax.custom_vjp
def _expert_mlp(x_dispatch, w13_il, moe_w2, group_sizes, cu):
    """y = down( swiglu( x @ w13_il ) ), grouped by experts. Activation-path GEMMs on QuACK.

    ``group_sizes``/``cu`` are traced int arrays passed as explicit args (not closed
    over — that leaks under shard_map; not nondiff_argnums — that rejects tracers).
    """
    _gu, h = quack_gated_grouped_gemm(x_dispatch, w13_il, cu, return_preact=True)
    return quack_grouped_gemm(h, moe_w2, cu, b_major="n")


def _expert_mlp_fwd(x_dispatch, w13_il, moe_w2, group_sizes, cu):
    gu, h = quack_gated_grouped_gemm(x_dispatch, w13_il, cu, return_preact=True)
    y = quack_grouped_gemm(h, moe_w2, cu, b_major="n")
    return y, (x_dispatch, w13_il, moe_w2, gu, h, group_sizes, cu)


def _expert_mlp_bwd(res, dy):
    x_dispatch, w13_il, moe_w2, gu, h, group_sizes, cu = res
    # down backward: dh via QuACK (transposed contraction), dw2 via XLA weight-grad
    dh = quack_grouped_gemm(dy, moe_w2, cu, b_major="k")
    (dw2,) = jax.vjp(lambda w: ragged_dot(h, w, group_sizes), moe_w2)[1](dy)
    # SwiGLU backward (interleaved gate/up), elementwise
    gate, up = gu[:, 0::2], gu[:, 1::2]
    sg = jax.nn.sigmoid(gate)
    silu = gate * sg
    dgate = dh * up * (sg + silu * (1.0 - sg))
    dup = dh * silu
    d_gu = jnp.stack([dgate, dup], axis=-1).reshape(gu.shape)
    # gate/up backward: dx via QuACK, dw13 via XLA weight-grad
    dx = quack_grouped_gemm(d_gu, w13_il, cu, b_major="k")
    (dw13_il,) = jax.vjp(lambda w: ragged_dot(x_dispatch, w, group_sizes), w13_il)[1](d_gu)
    # int-typed routing args get float0 zero cotangents
    gs_ct = np.zeros(group_sizes.shape, dtype=jax.dtypes.float0)
    cu_ct = np.zeros(cu.shape, dtype=jax.dtypes.float0)
    return dx, dw13_il, dw2, gs_ct, cu_ct


_expert_mlp.defvjp(_expert_mlp_fwd, _expert_mlp_bwd)


def _moe_mlp_local_sonic_cute(
    x: Float[Array, "T H"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    moe_w13: Float[Array, "E H I2"],
    moe_w2: Float[Array, "E I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
) -> tuple[Float[Array, "T H"], Int[Array, ""]]:
    x_dispatch, w_dispatch, token_dispatch, group_sizes = _prepare_moe_dispatch(
        x, selected_experts, combine_weights, num_experts=num_experts
    )
    x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
    moe_dim = moe_w2.shape[1]
    w13_il = _interleave_gate_up(moe_w13, moe_dim)
    cu = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes).astype(jnp.int32)])

    with jax.named_scope("moe_up_down_quack"):
        out_dispatch = tree_checkpoint_name(
            _expert_mlp(x_dispatch, w13_il, moe_w2, group_sizes, cu), _CHECKPOINT_DISPATCH_OUTPUT
        )

    with jax.named_scope("scatter"):
        out = jnp.zeros_like(x).at[token_dispatch].add(out_dispatch * w_dispatch[:, None], mode="drop")
    return out, _zero_dropped_assignments()


def _moe_mlp_local_sonic_cute_chunked(
    x: Float[Array, "T H"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    moe_w13_local: Float[Array, "E Hlocal I2"],
    moe_w2_local: Float[Array, "E I Hlocal"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    chunks: int,
    data_axis_name: str,
) -> tuple[Float[Array, "T H"], Int[Array, ""]]:
    """Chunked variant that gathers only ``1/chunks`` of the expert weights at a time.

    The FSDP weights arrive H-sharded over ``data_axis_name`` (``moe_w13_local`` is [E, H/data, 2I],
    ``moe_w2_local`` is [E, I, H/data]). Dispatch runs ONCE (``_prepare_moe_dispatch`` sorts every
    local (token, expert) assignment by expert, so experts ``[lo, hi)`` are a contiguous segment of
    the sorted buffer). For each static chunk we all-gather only that chunk's expert weights (a
    ``1/chunks``-size collective, so chunk k+1's gather fits the scheduler's overlap-memory budget and
    can hide under chunk k's GEMM), slice the matching token segment, run the QuACK grouped GEMM over
    just that segment, and scatter-accumulate the outputs.

    Segment handling (see the module docstring on ``_expert_mlp``): the QuACK kernel and the XLA
    ``ragged_dot`` weight-grad path both index rows relative to row 0 of the buffer they are given, so
    each chunk gets a segment-relative ``x_dispatch`` (sliced to start at ``cu[lo]``) with a rebased
    ``cu_c``. Segment length is a STATIC ``capacity = (T*K) // chunks`` (1x balanced, drop overflow).
    Rows past the chunk's real assignments are folded into the last expert group (so the kernel never
    leaves ungrouped garbage rows) but weight-masked to zero, so they contribute nothing to the
    forward output and route a zero cotangent back to the router in the combine backward.
    """
    if num_experts % chunks != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by SCALE_MOE_EXPERT_CHUNKS={chunks}")

    x_dispatch, w_dispatch, token_dispatch, group_sizes = _prepare_moe_dispatch(
        x, selected_experts, combine_weights, num_experts=num_experts
    )
    x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
    moe_dim = moe_w2_local.shape[1]
    total_assignments, hidden = x_dispatch.shape
    per = num_experts // chunks
    capacity = total_assignments // chunks
    cu = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes).astype(jnp.int32)])

    # Pad the sorted buffers by ``capacity`` rows so ``dynamic_slice(start=cu[lo], size=capacity)``
    # never clamps its start index (which would silently shift the window and misgroup rows). Padding
    # carries zero combine weight, so it never contributes to the output or its gradient.
    x_pad = jnp.pad(x_dispatch, ((0, capacity), (0, 0)))
    w_pad = jnp.pad(w_dispatch, (0, capacity))
    token_pad = jnp.pad(token_dispatch, (0, capacity))

    # w_down was already async/overlapped in the unchunked baseline (only w_up_gate's gather was
    # exposed), so gather it ONCE as a single collective and slice it locally per chunk. Only
    # w_up_gate is chunked. This drops the per-layer MoE collective count from 2*chunks to chunks+1,
    # removing the w_down over-chunking overhead while still splitting the one gather that was exposed.
    with jax.named_scope("gather_down"):
        w2_full = jax.lax.all_gather(moe_w2_local, data_axis_name, axis=2, tiled=True)

    out = jnp.zeros_like(x)
    for c in range(chunks):
        lo = c * per
        hi = lo + per
        with jax.named_scope("gather_chunk"):
            w13_chunk = jax.lax.all_gather(moe_w13_local[lo:hi], data_axis_name, axis=1, tiled=True)
        w2_chunk = w2_full[lo:hi]
        w13_il = _interleave_gate_up(w13_chunk, moe_dim)

        start = cu[lo]
        x_seg = jax.lax.dynamic_slice(x_pad, (start, 0), (capacity, hidden))
        token_seg = jax.lax.dynamic_slice(token_pad, (start,), (capacity,))
        w_seg = jax.lax.dynamic_slice(w_pad, (start,), (capacity,))

        # Real assignments for this chunk occupy segment rows [0, count); the rest are padding or
        # rows belonging to later chunks. Mask their combine weight to zero.
        count = cu[hi] - start
        valid = jnp.arange(capacity, dtype=jnp.int32) < jnp.minimum(count, capacity)
        w_seg = jnp.where(valid, w_seg, jnp.zeros_like(w_seg))

        # Segment-relative group boundaries. Fold the leftover capacity into the last expert so the
        # kernel writes every row (no ungrouped garbage); those extra rows are weight-masked above.
        raw = jnp.clip(cu[lo : hi + 1] - start, 0, capacity)
        group_sizes_c = jnp.diff(raw)
        group_sizes_c = group_sizes_c.at[-1].add(capacity - raw[-1])
        cu_c = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes_c).astype(jnp.int32)])

        with jax.named_scope("moe_up_down_quack_chunk"):
            out_dispatch = tree_checkpoint_name(
                _expert_mlp(x_seg, w13_il, w2_chunk, group_sizes_c, cu_c), _CHECKPOINT_DISPATCH_OUTPUT
            )
        with jax.named_scope("scatter_chunk"):
            out = out.at[token_seg].add(out_dispatch * w_seg[:, None], mode="drop")
    return out, _zero_dropped_assignments()
