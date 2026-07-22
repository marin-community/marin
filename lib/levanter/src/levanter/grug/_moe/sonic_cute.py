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
    w13_pre0: Float[Array, "per H I2"] | None = None,
    w2_pre0: Float[Array, "per I H"] | None = None,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    chunk_sizes: tuple[int, ...],
    data_axis_name: str,
) -> tuple[Float[Array, "T H"], Int[Array, ""]]:
    """Chunked variant that gathers only one chunk of the expert weights at a time.

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
    ``cu_c``. Each chunk's segment length is a STATIC ``capacity`` that scales with its expert count
    (``total_assignments * size / num_experts``, 1x balanced, drop overflow), so the per-chunk
    capacities sum to ``total_assignments`` exactly as the uniform case does. ``chunk_sizes`` need not
    be equal: e.g. ``(16, 16, 96)`` runs two small gathers (which start the expert GEMMs quickly, and
    let chunk 0 prefetch under attention via ``w13_pre0``) then one large gather that overlaps them.
    Rows past the chunk's real assignments are folded into the last expert group (so the kernel never
    leaves ungrouped garbage rows) but weight-masked to zero, so they contribute nothing to the
    forward output and route a zero cotangent back to the router in the combine backward.
    """
    if sum(chunk_sizes) != num_experts:
        raise ValueError(f"chunk_sizes={chunk_sizes} must sum to num_experts={num_experts}")

    x_dispatch, w_dispatch, token_dispatch, group_sizes = _prepare_moe_dispatch(
        x, selected_experts, combine_weights, num_experts=num_experts
    )
    x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
    moe_dim = moe_w2_local.shape[1]
    total_assignments, hidden = x_dispatch.shape

    # Expert-group boundaries and a per-chunk static capacity proportional to the chunk's expert
    # count. The capacities sum to total_assignments (as with equal chunks); a larger chunk holds
    # proportionally more tokens.
    bounds = [0]
    for size in chunk_sizes:
        bounds.append(bounds[-1] + size)
    caps = [total_assignments * size // num_experts for size in chunk_sizes]
    max_cap = max(caps)
    cu = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes).astype(jnp.int32)])

    # Pad the sorted buffers by the LARGEST chunk capacity so every chunk's
    # ``dynamic_slice(start=cu[lo], size=cap)`` never clamps its start index (which would silently
    # shift the window and misgroup rows). Padding carries zero combine weight, so it never
    # contributes to the output or its gradient.
    x_pad = jnp.pad(x_dispatch, ((0, max_cap), (0, 0)))
    w_pad = jnp.pad(w_dispatch, (0, max_cap))
    token_pad = jnp.pad(token_dispatch, (0, max_cap))

    out = jnp.zeros_like(x)
    for c, cap in enumerate(caps):
        lo = bounds[c]
        hi = bounds[c + 1]
        with jax.named_scope("gather_chunk"):
            if c == 0 and w13_pre0 is not None:
                # Chunk 0 was gathered OUTSIDE the shard_map (operand-gated on the weights only, so it
                # can overlap attention instead of being pinned to the region's x_flat input).
                w13_chunk, w2_chunk = w13_pre0, w2_pre0
            else:
                w13_chunk = jax.lax.all_gather(moe_w13_local[lo:hi], data_axis_name, axis=1, tiled=True)
                w2_chunk = jax.lax.all_gather(moe_w2_local[lo:hi], data_axis_name, axis=2, tiled=True)
        w13_il = _interleave_gate_up(w13_chunk, moe_dim)

        start = cu[lo]
        x_seg = jax.lax.dynamic_slice(x_pad, (start, 0), (cap, hidden))
        token_seg = jax.lax.dynamic_slice(token_pad, (start,), (cap,))
        w_seg = jax.lax.dynamic_slice(w_pad, (start,), (cap,))

        # Real assignments for this chunk occupy segment rows [0, count); the rest are padding or
        # rows belonging to later chunks. Mask their combine weight to zero.
        count = cu[hi] - start
        valid = jnp.arange(cap, dtype=jnp.int32) < jnp.minimum(count, cap)
        w_seg = jnp.where(valid, w_seg, jnp.zeros_like(w_seg))

        # Segment-relative group boundaries. Fold the leftover capacity into the last expert so the
        # kernel writes every row (no ungrouped garbage); those extra rows are weight-masked above.
        raw = jnp.clip(cu[lo : hi + 1] - start, 0, cap)
        group_sizes_c = jnp.diff(raw)
        group_sizes_c = group_sizes_c.at[-1].add(cap - raw[-1])
        cu_c = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes_c).astype(jnp.int32)])

        with jax.named_scope("moe_up_down_quack_chunk"):
            out_dispatch = tree_checkpoint_name(
                _expert_mlp(x_seg, w13_il, w2_chunk, group_sizes_c, cu_c), _CHECKPOINT_DISPATCH_OUTPUT
            )
        with jax.named_scope("scatter_chunk"):
            out = out.at[token_seg].add(out_dispatch * w_seg[:, None], mode="drop")
    return out, _zero_dropped_assignments()


def _moe_mlp_local_sonic_cute_intermediate_chunked(
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
    """Dropless chunked variant that partitions the expert MLP's INTERMEDIATE dim into ``chunks``.

    The expert-dim chunker (:func:`_moe_mlp_local_sonic_cute_chunked`) caps each expert-group at a
    static ``capacity`` and drops overflow. This variant keeps every ``(token, expert)`` assignment:
    dispatch runs once over the full sorted buffer, and for each static intermediate slice
    ``[ilo, ihi)`` we all-gather only that slice's ``w13`` columns (gate ``[ilo:ihi]`` + up
    ``[I+ilo:I+ihi]`` -> ``[E, H, 2I/chunks]``) and ``w2`` rows (``[E, I/chunks, H]``) — a
    ``1/chunks``-size collective that hides under the previous slice's GEMM — run the full token
    buffer through the QuACK up/SwiGLU/down over just that slice, and accumulate the partial
    down-projection outputs. The down GEMM sums over the intermediate dim, so the full expert output
    is the sum over slices; there is no capacity and nothing is dropped (exact up to float summation
    order). Total FLOPs are unchanged; the only shape cost is a thinner down-GEMM contraction
    (``K = I/chunks``).

    The intermediate dim is not sharded here (the ``model`` axis is absent under pure FSDP), so
    ``moe_w13_local`` is ``[E, H/data, 2I]`` and ``moe_w2_local`` is ``[E, I, H/data]`` with the full
    intermediate present locally; only ``H`` is gathered over ``data_axis_name``.
    """
    moe_dim = moe_w2_local.shape[1]
    if moe_dim % chunks != 0:
        raise ValueError(f"intermediate_dim={moe_dim} must be divisible by SCALE_MOE_EXPERT_CHUNKS={chunks}")

    x_dispatch, w_dispatch, token_dispatch, group_sizes = _prepare_moe_dispatch(
        x, selected_experts, combine_weights, num_experts=num_experts
    )
    x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
    cu = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes).astype(jnp.int32)])
    per_i = moe_dim // chunks

    out_dispatch = jnp.zeros_like(x_dispatch)
    for c in range(chunks):
        ilo = c * per_i
        hi = ilo + per_i
        with jax.named_scope("gather_ichunk"):
            # gate columns [ilo:hi] and up columns [I+ilo:I+hi] -> [E, H/data, 2*per_i], then gather H.
            w13_slice = jnp.concatenate(
                [moe_w13_local[:, :, ilo:hi], moe_w13_local[:, :, moe_dim + ilo : moe_dim + hi]], axis=-1
            )
            w13_chunk = jax.lax.all_gather(w13_slice, data_axis_name, axis=1, tiled=True)
            w2_chunk = jax.lax.all_gather(moe_w2_local[:, ilo:hi, :], data_axis_name, axis=2, tiled=True)
        w13_il = _interleave_gate_up(w13_chunk, per_i)
        with jax.named_scope("moe_up_down_quack_ichunk"):
            out_dispatch = out_dispatch + _expert_mlp(x_dispatch, w13_il, w2_chunk, group_sizes, cu)

    out_dispatch = tree_checkpoint_name(out_dispatch, _CHECKPOINT_DISPATCH_OUTPUT)
    with jax.named_scope("scatter"):
        out = jnp.zeros_like(x).at[token_dispatch].add(out_dispatch * w_dispatch[:, None], mode="drop")
    return out, _zero_dropped_assignments()
