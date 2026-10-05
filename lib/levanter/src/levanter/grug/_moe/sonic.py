# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw Sonic Triton gather/combine backend for local Grug MoE.

The Triton gather kernel is adapted from SonicMoE commit
cfbd65f39b980b85b878b3cccdacb09191e24993:
https://github.com/Dao-AILab/sonic-moe/blob/cfbd65f39b980b85b878b3cccdacb09191e24993/sonicmoe/functional/reduction_over_k_gather.py.
SonicMoE is also Apache-2.0.
"""

from collections.abc import Callable
import os

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from jax.typing import DTypeLike
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _CHECKPOINT_MOE_OUTPUT,
    _prepare_moe_dispatch_indices_with_assignment_ids,
    _zero_dropped_assignments,
    _zero_inactive_grouped_rows,
    split_moe_w13_output,
)

try:
    import jax_triton as jt
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    jt = None
    triton = None
    tl = None


_DEFAULT_TRITON_CACHE_DIR = "/tmp/marin-triton-cache"


if triton is not None and tl is not None:

    @triton.jit
    def _sonic_token_gather_sum_kernel(
        x_ptr,  # (Mtotal, H)
        w_ptr,  # (Mtotal,)
        m_perm_ptr,  # (Mtotal,) int32
        m_offset_ptr,  # unused for fixed-K, kept to match Sonic metadata
        out_ptr,  # (T, H)
        t: tl.constexpr,
        h: tl.constexpr,
        max_k: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xh: tl.constexpr,
        stride_outt: tl.constexpr,
        stride_outh: tl.constexpr,
        block_h: tl.constexpr,
        block_k: tl.constexpr,
        w_is_none: tl.constexpr,
        is_varlen_k: tl.constexpr,
    ):
        pid_t = tl.program_id(axis=0)
        t_idx = pid_t.to(tl.int64)

        if is_varlen_k:
            ms = tl.load(m_offset_ptr + t_idx).to(tl.int64)
            me = tl.load(m_offset_ptr + t_idx + 1).to(tl.int64)
            k_this_token = me - ms
        else:
            ms = max_k * t_idx
            k_this_token: tl.constexpr = max_k

        for h_tile in tl.static_range(triton.cdiv(h, block_h)):
            h_idx = (h_tile * block_h + tl.arange(0, block_h)).to(tl.int64)
            h_mask = h_idx < h
            acc = tl.zeros([block_h], dtype=tl.float32)

            for k_tile in tl.range(tl.cdiv(k_this_token, block_k)):
                k_offset = k_tile * block_k
                k_idx = (k_offset + tl.arange(0, block_k)).to(tl.int64)
                k_mask = k_idx < k_this_token
                m_abs = ms + k_idx
                perm_idx = tl.load(m_perm_ptr + m_abs, mask=k_mask, other=0).to(tl.int64)
                x_ptrs = x_ptr + perm_idx[:, None] * stride_xm + h_idx[None, :] * stride_xh

                if w_is_none:
                    x_mask = k_mask[:, None] & h_mask[None, :]
                    x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
                    acc += tl.sum(x_vals, axis=0)
                else:
                    w_vals = tl.load(w_ptr + m_abs, mask=k_mask, other=0.0).to(tl.float32)
                    # A zero-weight row contributes nothing, so it is never read and may hold
                    # anything, including non-finite values.
                    x_mask = (k_mask & (w_vals != 0.0))[:, None] & h_mask[None, :]
                    x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
                    acc += tl.sum(x_vals * w_vals[:, None], axis=0)

            out_ptrs = out_ptr + t_idx * stride_outt + h_idx * stride_outh
            tl.store(out_ptrs, acc, mask=h_mask)

    @triton.jit
    def _sonic_token_gather_sum_bwd_kernel(
        dout_ptr,  # (T, H)
        x_ptr,  # (Mtotal, H)
        w_ptr,  # (Mtotal,)
        m_perm_ptr,  # (Mtotal,) int32
        m_offset_ptr,  # unused for fixed-K, kept to match Sonic metadata
        dx_ptr,  # (Mtotal, H)
        dw_ptr,  # (Mtotal,)
        t: tl.constexpr,
        h: tl.constexpr,
        max_k: tl.constexpr,
        stride_doutt: tl.constexpr,
        stride_douth: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xh: tl.constexpr,
        block_h: tl.constexpr,
    ):
        assignment = tl.program_id(axis=0)
        token = assignment // max_k
        h_idx = tl.arange(0, block_h).to(tl.int64)
        h_mask = h_idx < h

        perm_idx = tl.load(m_perm_ptr + assignment).to(tl.int64)
        weight = tl.load(w_ptr + assignment).to(tl.float32)

        dout = tl.load(
            dout_ptr + token * stride_doutt + h_idx * stride_douth,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        x = tl.load(
            x_ptr + perm_idx * stride_xm + h_idx * stride_xh,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)

        tl.store(dx_ptr + perm_idx * stride_xm + h_idx * stride_xh, dout * weight, mask=h_mask)
        dw = tl.sum(dout * x, axis=0)
        tl.store(dw_ptr + assignment, dw)

    @triton.jit
    def _write_nothing_kernel(marker_ptr, out_ptr):
        pass

    @triton.jit
    def _token_scatter_rows_kernel(
        x_ptr,  # (T, H)
        w_ptr,  # (T * K,) float32; read only when scaled
        keep_ptr,  # (T * K,) int8
        m_pos_ptr,  # (T * K,) int32
        out_ptr,  # (M, H)
        h: tl.constexpr,
        topk: tl.constexpr,
        block_h: tl.constexpr,
        scaled: tl.constexpr,
    ):
        token = tl.program_id(axis=0).to(tl.int64)
        for h_tile in tl.static_range(triton.cdiv(h, block_h)):
            h_idx = (h_tile * block_h + tl.arange(0, block_h)).to(tl.int64)
            h_mask = h_idx < h
            x = tl.load(x_ptr + token * h + h_idx, mask=h_mask, other=0.0)
            for k in tl.static_range(topk):
                assignment = token * topk + k
                keep = tl.load(keep_ptr + assignment) != 0
                position = tl.load(m_pos_ptr + assignment).to(tl.int64)
                if scaled:
                    weight = tl.load(w_ptr + assignment)
                    row = (x.to(tl.float32) * weight).to(out_ptr.dtype.element_ty)
                else:
                    row = x
                tl.store(out_ptr + position * h + h_idx, row, mask=h_mask & keep)

else:
    _sonic_token_gather_sum_kernel = None
    _sonic_token_gather_sum_bwd_kernel = None
    _write_nothing_kernel = None
    _token_scatter_rows_kernel = None


def _require_sonic_deps() -> None:
    if jt is None or _sonic_token_gather_sum_kernel is None or _sonic_token_gather_sum_bwd_kernel is None:
        raise ImportError(
            "implementation='sonic' requires jax-triton and triton; install the gpu extra for marin-levanter "
            "or marin."
        )
    if not os.environ.get("TRITON_CACHE_DIR"):
        os.environ["TRITON_CACHE_DIR"] = _DEFAULT_TRITON_CACHE_DIR


def sonic_gather_sum_available() -> bool:
    """Whether the fused gather-and-weighted-sum kernel can run in this process.

    The kernel is Triton, so a non-GPU backend rules it out however the imports went.
    """
    return (
        jt is not None
        and _sonic_token_gather_sum_kernel is not None
        and _sonic_token_gather_sum_bwd_kernel is not None
        and jax.default_backend() == "gpu"
    )


def unwritten_buffer(shape: tuple[int, ...], dtype: DTypeLike, marker: Int[Array, ""]) -> Array:
    """A buffer with unspecified contents, made by a GPU kernel that writes nothing.

    The kernel takes ``marker`` as an operand without reading it, so the buffer depends on the
    marker: a loop-carried marker keeps the buffer inside the loop body, and two call sites with
    different markers get two buffers.
    """
    _require_sonic_deps()
    return jt.triton_call(
        marker,
        kernel=_write_nothing_kernel,
        out_shape=jax.ShapeDtypeStruct(shape, dtype),
        grid=(1,),
    )


def sonic_scatter_rows(
    x: Float[Array, "T H"],
    dispatch_positions: Int[Array, "T K"],
    keep: Bool[Array, "T K"],
    *,
    rows: int,
    weights: Float[Array, "T K"] | None = None,
) -> Float[Array, "M H"]:
    """Copy each token's row to its ``K`` dispatch positions, reading the row once.

    Builds the same buffer as gathering ``x[sorted_token_ids]`` row by row, which reads every
    source row ``K`` times, in a little over half the memory traffic. Only the positions whose
    ``keep`` is set are written; the other rows of the ``[rows, H]`` result are unspecified. With
    ``weights``, each written row is ``x`` times its fp32 weight, rounded once to ``x``'s dtype.
    """
    _require_sonic_deps()
    tokens, topk = dispatch_positions.shape
    hidden_dim = x.shape[1]
    # Best measured on GB200 at TK = 524288, H = 3072: 0.535 ms, 6.6 TB/s.
    block_h = min(_next_power_of_2(hidden_dim), 2048)
    weights_flat = (
        jnp.zeros((1,), jnp.float32) if weights is None else weights.reshape(tokens * topk).astype(jnp.float32)
    )
    return jt.triton_call(
        x,
        weights_flat,
        keep.reshape(tokens * topk).astype(jnp.int8),
        dispatch_positions.reshape(tokens * topk).astype(jnp.int32),
        kernel=_token_scatter_rows_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, hidden_dim), x.dtype),
        grid=(tokens,),
        num_warps=16,
        num_stages=1,
        h=hidden_dim,
        topk=topk,
        block_h=block_h,
        scaled=weights is not None,
    )


def _next_power_of_2(value: int) -> int:
    if value < 1:
        raise ValueError(f"value must be positive, got {value}")
    return 1 << (value - 1).bit_length()


def _sonic_kernel_config(hidden_dim: int) -> tuple[int, int, int]:
    block_h = min(max(256, _next_power_of_2(hidden_dim)), 4096)
    block_k = 1
    num_warps = 8 if block_h >= 1024 else 4
    return block_h, block_k, num_warps


def _sonic_fixed_k_offsets(*, tokens: int, topk: int) -> Int[Array, "Tp1"]:
    return jnp.arange(0, tokens * topk + 1, topk, dtype=jnp.int32)


def _sonic_gather_sum_impl(
    dispatch_output: Float[Array, "M H"],
    weights_flat: Float[Array, "M"],
    positions_flat: Int[Array, "M"],
    offsets: Int[Array, "Tp1"],
    *,
    tokens: int,
    topk: int,
    output_dtype: DTypeLike | None = None,
) -> Float[Array, "T H"]:
    _require_sonic_deps()
    hidden_dim = dispatch_output.shape[1]
    widest_block_h, block_k, num_warps = _sonic_kernel_config(hidden_dim)
    # The forward kernel loops over hidden tiles, and narrower tiles than the backward's single
    # tile keep more programs resident: on GB200 at TK = 524288, H = 3072, 1024-wide tiles take
    # 0.508 ms against 0.574 ms for one 4096-wide tile. Each element still sums its K rows in order.
    block_h = min(widest_block_h, 1024)
    out_shape = jax.ShapeDtypeStruct(
        (tokens, hidden_dim),
        dispatch_output.dtype if output_dtype is None else output_dtype,
    )
    return jt.triton_call(
        dispatch_output,
        weights_flat,
        positions_flat,
        offsets,
        kernel=_sonic_token_gather_sum_kernel,
        out_shape=out_shape,
        grid=(tokens,),
        num_warps=num_warps,
        num_stages=4,
        t=tokens,
        h=hidden_dim,
        max_k=topk,
        stride_xm=hidden_dim,
        stride_xh=1,
        stride_outt=hidden_dim,
        stride_outh=1,
        block_h=block_h,
        block_k=block_k,
        w_is_none=False,
        is_varlen_k=False,
    )


def sonic_gather_sum_masked(
    dispatch_output: Float[Array, "M H"],
    dispatch_positions: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    *,
    output_dtype: DTypeLike | None = None,
) -> Float[Array, "T H"]:
    """Gather weighted rows and ignore positions outside ``dispatch_output``."""
    tokens, topk = combine_weights.shape
    valid = (dispatch_positions >= 0) & (dispatch_positions < dispatch_output.shape[0])
    positions_flat = jnp.clip(dispatch_positions, 0, dispatch_output.shape[0] - 1).reshape(tokens * topk)
    weights_flat = jnp.where(valid, combine_weights, 0).reshape(tokens * topk).astype(jnp.float32)
    offsets = _sonic_fixed_k_offsets(tokens=tokens, topk=topk)
    return _sonic_gather_sum_impl(
        dispatch_output,
        weights_flat,
        positions_flat,
        offsets,
        tokens=tokens,
        topk=topk,
        output_dtype=output_dtype,
    )


def _sonic_gather_sum_bwd_impl(
    dout: Float[Array, "T H"],
    dispatch_output: Float[Array, "M H"],
    weights_flat: Float[Array, "M"],
    positions_flat: Int[Array, "M"],
    offsets: Int[Array, "Tp1"],
    *,
    tokens: int,
    topk: int,
) -> tuple[Float[Array, "M H"], Float[Array, "M"]]:
    _require_sonic_deps()
    hidden_dim = dispatch_output.shape[1]
    block_h, _block_k, num_warps = _sonic_kernel_config(hidden_dim)
    dx_shape = jax.ShapeDtypeStruct(dispatch_output.shape, dispatch_output.dtype)
    dw_shape = jax.ShapeDtypeStruct(weights_flat.shape, jnp.float32)
    return jt.triton_call(
        dout,
        dispatch_output,
        weights_flat,
        positions_flat,
        offsets,
        kernel=_sonic_token_gather_sum_bwd_kernel,
        out_shape=(dx_shape, dw_shape),
        grid=(tokens * topk,),
        num_warps=num_warps,
        num_stages=4,
        t=tokens,
        h=hidden_dim,
        max_k=topk,
        stride_doutt=hidden_dim,
        stride_douth=1,
        stride_xm=hidden_dim,
        stride_xh=1,
        block_h=block_h,
    )


@jax.custom_vjp
def sonic_gather_sum(
    dispatch_output: Float[Array, "M H"],
    dispatch_positions: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
) -> Float[Array, "T H"]:
    """Sum each token's ``K`` rows of ``dispatch_output``, gathered at its positions and weighted.

    A row with zero weight is not read, so it may hold unspecified values. The backward still
    reads every row for the weight gradient. A caller that passes unspecified rows zeroes their
    weights with ``jnp.where``, whose transpose discards those gradients.
    """
    tokens, topk = combine_weights.shape
    weights_flat = combine_weights.reshape(tokens * topk).astype(jnp.float32)
    positions_flat = dispatch_positions.reshape(tokens * topk).astype(jnp.int32)
    offsets = _sonic_fixed_k_offsets(tokens=tokens, topk=topk)
    return _sonic_gather_sum_impl(
        dispatch_output,
        weights_flat,
        positions_flat,
        offsets,
        tokens=tokens,
        topk=topk,
    )


def _sonic_gather_sum_fwd(
    dispatch_output: Float[Array, "M H"],
    dispatch_positions: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
) -> tuple[Float[Array, "T H"], tuple[Float[Array, "M H"], Int[Array, "T K"], Float[Array, "T K"]]]:
    tokens, topk = combine_weights.shape
    weights_flat = combine_weights.reshape(tokens * topk).astype(jnp.float32)
    positions_flat = dispatch_positions.reshape(tokens * topk).astype(jnp.int32)
    offsets = _sonic_fixed_k_offsets(tokens=tokens, topk=topk)
    out = _sonic_gather_sum_impl(
        dispatch_output,
        weights_flat,
        positions_flat,
        offsets,
        tokens=tokens,
        topk=topk,
    )
    return out, (dispatch_output, dispatch_positions, combine_weights)


def _sonic_gather_sum_bwd(
    residuals: tuple[Float[Array, "M H"], Int[Array, "T K"], Float[Array, "T K"]],
    dout: Float[Array, "T H"],
) -> tuple[Float[Array, "M H"], None, Float[Array, "T K"]]:
    dispatch_output, dispatch_positions, combine_weights = residuals
    tokens, topk = combine_weights.shape
    weights_flat = combine_weights.reshape(tokens * topk).astype(jnp.float32)
    positions_flat = dispatch_positions.reshape(tokens * topk).astype(jnp.int32)
    offsets = _sonic_fixed_k_offsets(tokens=tokens, topk=topk)
    d_dispatch_output, d_weights_flat = _sonic_gather_sum_bwd_impl(
        dout,
        dispatch_output,
        weights_flat,
        positions_flat,
        offsets,
        tokens=tokens,
        topk=topk,
    )
    d_combine_weights = d_weights_flat.reshape(combine_weights.shape).astype(combine_weights.dtype)
    return d_dispatch_output, None, d_combine_weights


sonic_gather_sum.defvjp(_sonic_gather_sum_fwd, _sonic_gather_sum_bwd)


def _moe_mlp_local_sonic(
    x: Float[Array, "T H"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    token_valid: Bool[Array, "T"],
    moe_w13: Float[Array, "E H I2"],
    moe_w2: Float[Array, "E I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
) -> tuple[Float[Array, "T H"], Int[Array, ""]]:
    """Local raw-Sonic path: JAX grouped GEMMs plus Sonic Triton gather/combine."""
    token_ids_sort, dispatch_positions, group_sizes, _sorted_assignment_ids = (
        _prepare_moe_dispatch_indices_with_assignment_ids(
            selected_experts,
            token_valid,
            num_experts=num_experts,
        )
    )
    cumulative_group_sizes = jnp.cumsum(group_sizes).astype(jnp.int32)
    x_dispatch = _zero_inactive_grouped_rows(x[token_ids_sort], cumulative_group_sizes)
    x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)

    with jax.named_scope("moe_up_down"):
        # Rows past the last group are unspecified kernel output; every consumer before the
        # gather-sum is row-local or group-bounded, so only `out_dispatch` needs zeroing.
        w13_out = tree_checkpoint_name(ragged_dot(x_dispatch, moe_w13, group_sizes), _CHECKPOINT_EXPERT_HIDDEN)
        moe_dim = moe_w2.shape[1]
        gate, up = split_moe_w13_output(w13_out, intermediate_dim=moe_dim, interleaved=False)
        hidden = activation_fn(gate) * up
        out_dispatch = _zero_inactive_grouped_rows(ragged_dot(hidden, moe_w2, group_sizes), cumulative_group_sizes)
        out = tree_checkpoint_name(
            sonic_gather_sum(out_dispatch, dispatch_positions, jnp.where(token_valid[:, None], combine_weights, 0)),
            _CHECKPOINT_MOE_OUTPUT,
        )

    return out, _zero_dropped_assignments()
