# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Local Grug MoE backend using Tri Dao's QuACK SM100 kernels (SonicMoE) on B200.

Dispatch/combine as in ``scatter``, but the expert MLP GEMMs run on QuACK's
``GemmGatedSm100`` / ``GemmDefaultSm100`` via the vendored ``cutlass.jax.cutlass_call``
shim. QuACK does all four activation-path grouped GEMMs (gate/up fwd fused with
SwiGLU, down fwd, and the ``dh``/``dx`` backward matmuls); the SwiGLU backward is
elementwise in JAX; the two weight-gradient GEMMs (``dw13``/``dw2``) stay on XLA
``ragged_dot`` (a different varlen-k grouping). QuACK covers ~2/3 of the MoE FLOPs.

``_expert_mlp_cudnn`` is the same forward with those two weight gradients moved onto cuDNN
Frontend grouped Wgrad kernels, which is faster than ``ragged_dot`` at the hero shapes.
"""

import jax
import jax.numpy as jnp
import numpy as np
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from jaxtyping import Array, Float, Int

from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _chunk_capacity_drops,
    _interleave_gate_up,
    _prepare_moe_dispatch,
    _swiglu_gate_up_backward,
    _zero_dropped_assignments,
    _zero_inactive_grouped_rows,
)
from levanter.grug._moe.cudnn_wgrad_cute import cudnn_grouped_wgrad, cudnn_grouped_wgrad_prealigned
from levanter.grug._moe.quack_moe_cute import quack_gated_grouped_gemm, quack_grouped_gemm

# QuACK activation-path GEMM configuration, tuned at the i3072 hero shapes on one GB200.
# Tile (256, 256) beats the (256, 128) default by 1.235x on the gated GEMM and 1.094x on the
# down GEMM. CLC persistence adds a further 1.049x / 1.120x at that tile. Under CLC the two
# GEMMs prefer different clusters: the gated gate/up GEMM stays at (2, 1, 1), while the plain
# grouped GEMMs -- down forward plus the backward dh/dx matmuls -- gain 1.055x at (2, 2, 1).
# All of this is scheduling, so none of it changes the computed function.
#
# These reach `_expert_mlp_cudnn` only. `_expert_mlp` -- the local FSDP path, used by the
# `fsdp-nodrop` and `fsdp-chunk4` ablation arms -- still calls the GEMMs at their defaults, as it
# did before this tuning existed, so nothing regressed. It is untuned rather than deliberately
# tuned differently: the measurements above were taken at the i3072 hero shapes and the FSDP arms
# run d768, so the numbers do not transfer without re-measuring.
# TODO: re-measure these at the FSDP ablation shapes and either extend the tuning to
# `_expert_mlp` or record why the defaults win there.
_QUACK_TILE_MN = (256, 256)
_QUACK_USE_CLC = True
_QUACK_GATED_KW = dict(tile_mn=_QUACK_TILE_MN, cluster_mnk=(2, 1, 1), use_clc_persistence=_QUACK_USE_CLC)
_QUACK_GROUPED_KW = dict(tile_mn=_QUACK_TILE_MN, cluster_mnk=(2, 2, 1), use_clc_persistence=_QUACK_USE_CLC)


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
    d_gu = _swiglu_gate_up_backward(gu, dh)
    # gate/up backward: dx via QuACK, dw13 via XLA weight-grad
    dx = quack_grouped_gemm(d_gu, w13_il, cu, b_major="k")
    (dw13_il,) = jax.vjp(lambda w: ragged_dot(x_dispatch, w, group_sizes), w13_il)[1](d_gu)
    # int-typed routing args get float0 zero cotangents
    gs_ct = np.zeros(group_sizes.shape, dtype=jax.dtypes.float0)
    cu_ct = np.zeros(cu.shape, dtype=jax.dtypes.float0)
    return dx, dw13_il, dw2, gs_ct, cu_ct


_expert_mlp.defvjp(_expert_mlp_fwd, _expert_mlp_bwd)


def _padded_operand_rows(values: jax.Array, cu: jax.Array) -> jax.Array:
    """`pad_grouped_rows` rebuilds the operand anyway, so the padding path masks nothing."""
    del cu
    return values


def _build_expert_mlp_cudnn(grouped_wgrad, wgrad_operand_rows, name: str):
    """``_expert_mlp`` with the two weight-gradient GEMMs on ``grouped_wgrad``.

    The forward output is masked past the last expert group: the grouped GEMMs write only
    the rows inside ``cu``, and those trailing rows flow on through the unpermute and
    combine, so they have to be zero rather than whatever the buffer held.

    ``grouped_wgrad`` is which cuDNN entry point the weight gradients take, which is fixed per
    caller: `cudnn_grouped_wgrad` pads the operands into the kernel's aligned layout, and
    `cudnn_grouped_wgrad_prealigned` skips that copy for a caller whose ``group_sizes`` already
    describe an aligned layout. Both compute the same function; the second is the same kernel
    call on the same bytes without the pre-pass.

    ``wgrad_operand_rows`` is what has to happen to a weight-gradient operand first, and it is the
    price of skipping the pad. The pre-aligned kernel call must cover every row of its operands
    (its last group absorbs the buffer's leftover), so it reads rows past ``cu[-1]`` -- rows the
    QuACK GEMMs never write, which hold whatever the freshly allocated buffer held. Of the four
    operands, ``dy`` and ``x_dispatch`` are already zero there, but ``h`` and ``d_gu`` are not: a
    stale NaN in either would survive multiplication by its zero partner. So the pre-aligned build
    extends the existing `_zero_inactive_grouped_rows` masking to exactly those two, and the
    padding build passes them straight through because `pad_grouped_rows` re-zeroes them itself.

    ``h`` is masked in the forward rather than the backward: the mask has to materialize either
    way (its consumer is an opaque kernel call), and doing it where ``h`` is produced keeps one
    copy of it live instead of two. ``d_gu``'s mask is free -- it fuses into the elementwise
    SwiGLU backward that already reads and writes that whole buffer.
    """

    @jax.custom_vjp
    def _expert_mlp_cudnn(x_dispatch, w13_il, moe_w2, group_sizes, cu):
        _gu, h = quack_gated_grouped_gemm(x_dispatch, w13_il, cu, return_preact=True, **_QUACK_GATED_KW)
        h = wgrad_operand_rows(h, cu)
        y = quack_grouped_gemm(h, moe_w2, cu, b_major="n", **_QUACK_GROUPED_KW)
        return _zero_inactive_grouped_rows(y, cu)

    def _expert_mlp_cudnn_fwd(x_dispatch, w13_il, moe_w2, group_sizes, cu):
        gu, h = quack_gated_grouped_gemm(x_dispatch, w13_il, cu, return_preact=True, **_QUACK_GATED_KW)
        h = wgrad_operand_rows(h, cu)
        y = quack_grouped_gemm(h, moe_w2, cu, b_major="n", **_QUACK_GROUPED_KW)
        return _zero_inactive_grouped_rows(y, cu), (x_dispatch, w13_il, moe_w2, gu, h, group_sizes, cu)

    def _expert_mlp_cudnn_bwd(res, dy):
        x_dispatch, w13_il, moe_w2, gu, h, group_sizes, cu = res
        # The cotangent's trailing rows are whatever the caller's buffer held; the grouped GEMMs
        # contract every row they are handed, so they have to be cleared here too.
        dy = _zero_inactive_grouped_rows(dy, cu)
        dh = quack_grouped_gemm(dy, moe_w2, cu, b_major="k", **_QUACK_GROUPED_KW)
        dw2 = grouped_wgrad(h, dy, group_sizes)
        # Masked before the dx GEMM, not between it and the weight gradient, so one d_gu is built.
        # The GEMM reads only rows inside `cu`, so the mask cannot change dx.
        d_gu = wgrad_operand_rows(_swiglu_gate_up_backward(gu, dh), cu)
        dx = quack_grouped_gemm(d_gu, w13_il, cu, b_major="k", **_QUACK_GROUPED_KW)
        dx = _zero_inactive_grouped_rows(dx, cu)
        dw13_il = grouped_wgrad(x_dispatch, d_gu, group_sizes)
        gs_ct = np.zeros(group_sizes.shape, dtype=jax.dtypes.float0)
        cu_ct = np.zeros(cu.shape, dtype=jax.dtypes.float0)
        return dx, dw13_il, dw2, gs_ct, cu_ct

    _expert_mlp_cudnn.defvjp(_expert_mlp_cudnn_fwd, _expert_mlp_cudnn_bwd)
    # Distinct names so a jaxpr, an HLO metadata line, or a profile says which build ran.
    _expert_mlp_cudnn.__name__ = name
    _expert_mlp_cudnn.__qualname__ = name
    return _expert_mlp_cudnn


_expert_mlp_cudnn = _build_expert_mlp_cudnn(cudnn_grouped_wgrad, _padded_operand_rows, "_expert_mlp_cudnn")
# For a receiver buffer whose expert groups already start on the kernel's aligned rows, with the
# slack between them zero. The transport builds that layout, so the pad pass is pure copy traffic.
_expert_mlp_cudnn_prealigned = _build_expert_mlp_cudnn(
    cudnn_grouped_wgrad_prealigned, _zero_inactive_grouped_rows, "_expert_mlp_cudnn_prealigned"
)


def _moe_mlp_local_sonic_cute(
    x: Float[Array, "T H"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    moe_w13: Float[Array, "E H I2"],
    moe_w2: Float[Array, "E I H"],
    *,
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
    be equal: e.g. ``(16, 16, 96)`` runs two small gathers, which start the expert GEMMs quickly,
    followed by one large gather that overlaps them.
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
            # Interleave on the local shard: it rewrites the last axis and the gather is along H, so
            # the two commute, and this does 1/data-th of the elementwise work.
            w13_local = _interleave_gate_up(moe_w13_local[lo:hi], moe_dim)
            w13_il = jax.lax.all_gather(w13_local, data_axis_name, axis=1, tiled=True)
            w2_chunk = jax.lax.all_gather(moe_w2_local[lo:hi], data_axis_name, axis=2, tiled=True)

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
    return out, _chunk_capacity_drops(cu, bounds, caps)
