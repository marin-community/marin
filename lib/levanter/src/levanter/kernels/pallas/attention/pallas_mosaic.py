# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""GPU attention kernel used by Grug with optional sliding-window masking."""

from __future__ import annotations

import math
import functools
from typing import cast

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import jax.numpy as jnp
import numpy as np

from jax.experimental.pallas.ops.gpu.attention import BlockSizes as BlockSizes


DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


def segment_mask(q_segment_ids: jax.Array, kv_segment_ids: jax.Array) -> jax.Array:
    # Pallas block views may materialize singleton trailing dims (e.g. [Q, 1] / [K, 1]).
    # Normalize those to 1D so blockwise backward masking remains shape-stable.
    if q_segment_ids.ndim == 2 and 1 in q_segment_ids.shape:
        q_segment_ids = jnp.reshape(q_segment_ids, (q_segment_ids.size,))
    if kv_segment_ids.ndim == 2 and 1 in kv_segment_ids.shape:
        kv_segment_ids = jnp.reshape(kv_segment_ids, (kv_segment_ids.size,))

    if q_segment_ids.ndim == 1 and kv_segment_ids.ndim == 1:
        return jnp.equal(q_segment_ids[:, None], kv_segment_ids[None, :]).astype(jnp.bool_)

    if q_segment_ids.ndim == 2 and kv_segment_ids.ndim == 2:
        if q_segment_ids.shape[0] != kv_segment_ids.shape[0]:
            raise ValueError(
                "segment-id batch mismatch in segment_mask: " f"q={q_segment_ids.shape}, kv={kv_segment_ids.shape}"
            )
        return jnp.equal(q_segment_ids[:, :, None], kv_segment_ids[:, None, :]).astype(jnp.bool_)

    raise ValueError(f"segment ids must be rank-1 or rank-2, got q={q_segment_ids.ndim}, kv={kv_segment_ids.ndim}")


def _apply_window_mask(
    mask: jax.Array | None,
    span_q: jax.Array,
    span_k: jax.Array,
    *,
    sliding_window: int | None,
) -> jax.Array | None:
    if sliding_window is None:
        return mask
    window_mask = span_k[None, :] >= (span_q[:, None] - (sliding_window - 1))
    if mask is None:
        return window_mask
    return jnp.logical_and(mask, window_mask)


def _validate_block_sizes(block_sizes: BlockSizes) -> None:
    if block_sizes.block_q <= 0 or block_sizes.block_k <= 0:
        raise ValueError(f"block sizes must be positive: block_q={block_sizes.block_q}, block_k={block_sizes.block_k}")
    if block_sizes.block_q_dkv is not None and block_sizes.block_q_dkv <= 0:
        raise ValueError(f"block_q_dkv must be positive, got {block_sizes.block_q_dkv}.")
    if block_sizes.block_kv_dkv is not None and block_sizes.block_kv_dkv <= 0:
        raise ValueError(f"block_kv_dkv must be positive, got {block_sizes.block_kv_dkv}.")
    if block_sizes.block_q_dq is not None and block_sizes.block_q_dq <= 0:
        raise ValueError(f"block_q_dq must be positive, got {block_sizes.block_q_dq}.")
    if block_sizes.block_kv_dq is not None and block_sizes.block_kv_dq <= 0:
        raise ValueError(f"block_kv_dq must be positive, got {block_sizes.block_kv_dq}.")


def _validate_sliding_window(sliding_window: int | None) -> None:
    if sliding_window is None:
        return
    if sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}.")


def _with_optional_grid_dim(
    total_q: int,
    total_k: int,
    batch_size: int,
    num_heads: int,
    block_q: int,
    block_k: int,
    grid: tuple[int, ...] | None,
) -> tuple[int, int, int]:
    if grid is not None:
        if len(grid) != 3:
            raise ValueError(f"grid must have length 3, got {grid}")
        return grid
    return (pl.cdiv(total_q, block_q), batch_size, num_heads)


def _normalize_block_size(seq_len: int, requested: int, *, name: str) -> int:
    block = min(requested, seq_len)
    if block <= 0:
        raise ValueError(f"{name} must be positive, got {requested}")
    while block > 1 and seq_len % block != 0:
        block -= 1
    return block


def _resolve_backward_block_sizes(
    q_seq_len: int,
    kv_seq_len: int,
    block_sizes: BlockSizes,
    forward_block_q: int,
    forward_block_k: int,
) -> tuple[int, int, int, int]:
    block_q_dkv = _normalize_block_size(
        q_seq_len,
        block_sizes.block_q_dkv if block_sizes.block_q_dkv is not None else forward_block_q,
        name="block_q_dkv",
    )
    block_kv_dkv = _normalize_block_size(
        kv_seq_len,
        block_sizes.block_kv_dkv if block_sizes.block_kv_dkv is not None else forward_block_k,
        name="block_kv_dkv",
    )
    block_q_dq = _normalize_block_size(
        q_seq_len,
        block_sizes.block_q_dq if block_sizes.block_q_dq is not None else forward_block_q,
        name="block_q_dq",
    )
    block_kv_dq = _normalize_block_size(
        kv_seq_len,
        block_sizes.block_kv_dq if block_sizes.block_kv_dq is not None else forward_block_k,
        name="block_kv_dq",
    )
    return block_q_dkv, block_kv_dkv, block_q_dq, block_kv_dq


def mha_forward_kernel(
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    o_ref,
    lse_ref: jax.Array | None = None,
    *,
    sm_scale: float,
    causal: bool,
    sliding_window: int | None,
    block_q: int,
    block_k: int,
    head_dim: int,
) -> None:
    seq_len = k_ref.shape[0]
    start_q = pl.program_id(0)
    head_dim_padded = q_ref.shape[-1]

    m_i = jnp.zeros(block_q, dtype=jnp.float32) - float("inf")
    l_i = jnp.zeros(block_q, dtype=jnp.float32)
    o = jnp.zeros((block_q, head_dim_padded), dtype=jnp.float32)

    curr_q_slice = pl.dslice(start_q * block_q, block_q)
    head_mask = (jnp.arange(head_dim_padded) < head_dim)[None, :]
    q = plgpu.load(q_ref, mask=head_mask, other=0.0)
    q_segment_ids = None if q_segment_ids_ref is None else q_segment_ids_ref[curr_q_slice]

    def body(start_k, carry):
        o_prev, m_prev, l_prev = carry
        curr_k_slice = pl.dslice(start_k * block_k, block_k)
        kv_segment_ids = None if kv_segment_ids_ref is None else kv_segment_ids_ref[curr_k_slice]
        span_q = start_q * block_q + jnp.arange(block_q)
        span_k = start_k * block_k + jnp.arange(block_k)

        k = plgpu.load(k_ref.at[curr_k_slice, :], mask=head_mask, other=0.0)
        qk = pl.dot(q, k.T)
        qk_scale = math.log2(math.e)
        if sm_scale != 1.0:
            qk_scale *= sm_scale
        qk *= qk_scale

        if causal or q_segment_ids_ref is not None or sliding_window is not None:
            mask = None
            if q_segment_ids_ref is not None:
                mask = segment_mask(q_segment_ids, kv_segment_ids)
            if causal:
                causal_mask = span_q[:, None] >= span_k[None, :]
                mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
            if sliding_window is not None:
                window_mask = _apply_window_mask(mask, span_q, span_k, sliding_window=sliding_window)
                mask = window_mask
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

        m_curr = jnp.max(qk, axis=-1)
        m_next = jnp.maximum(m_prev, m_curr)
        correction = jnp.exp2(m_prev - m_next)
        l_prev_corr = correction * l_prev
        s_curr = jnp.exp2(qk - m_next[:, None])
        l_curr = s_curr.sum(axis=-1)
        l_next = l_prev_corr + l_curr
        o_prev_corr = correction[:, None] * o_prev
        v = plgpu.load(v_ref.at[curr_k_slice, :], mask=head_mask)
        o_curr = pl.dot(s_curr.astype(v.dtype), v)
        o_next = o_prev_corr + o_curr
        return o_next, m_next, l_next

    upper_bound = pl.cdiv(seq_len, block_k)
    if causal:
        upper_bound = (block_q * (start_q + 1) + block_k - 1) // block_k

    o, m_i, l_i = lax.fori_loop(0, upper_bound, body, (o, m_i, l_i))
    o /= l_i[:, None]
    if lse_ref is not None:
        lse_ref[...] = m_i + jnp.log2(l_i)
    plgpu.store(o_ref.at[:, : o.shape[-1]], o.astype(o_ref.dtype), mask=head_mask)


def _mha_call(
    q,
    k,
    v,
    segment_ids: tuple[jax.Array, jax.Array] | jax.Array | None,
    sm_scale: float = 1.0,
    causal: bool = False,
    sliding_window: int | None = None,
    block_sizes: BlockSizes = BlockSizes.get_default(),
    backward_pass_impl: str = "triton",
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
    return_residuals: bool = False,
):
    _validate_block_sizes(block_sizes)
    _validate_sliding_window(sliding_window)

    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    head_dim_padded = pl.next_power_of_2(head_dim)
    block_q = min(block_sizes.block_q, q_seq_len)
    block_k = min(block_sizes.block_k, kv_seq_len)

    if (q.shape[-1] != k.shape[-1]) or (q.shape[-1] != v.shape[-1]):
        raise ValueError(f"q, k, and v must share head dim, got q={q.shape}, k={k.shape}, v={v.shape}.")
    if q_seq_len % block_q != 0:
        raise ValueError(f"{q_seq_len=}, must be divisible by {block_q=}")
    if kv_seq_len % block_k != 0:
        raise ValueError(f"{kv_seq_len=}, must be divisible by {block_k=}")

    grid_ = _with_optional_grid_dim(
        q_seq_len,
        kv_seq_len,
        batch_size,
        num_heads,
        block_q,
        block_k,
        grid,
    )
    num_warps_ = num_warps
    if num_warps_ is None:
        num_warps_ = 4 if head_dim <= 64 else 8

    if segment_ids is None:
        q_segment_ids = None
        kv_segment_ids = None
    else:
        if isinstance(segment_ids, tuple):
            q_segment_ids, kv_segment_ids = segment_ids
        else:
            q_segment_ids = segment_ids
            kv_segment_ids = segment_ids

        if q_segment_ids.ndim != 2:
            raise ValueError(
                f"query segment ids must be 2D (shared across batch as [1, S] or per-batch as [B, S]), "
                f"got shape {q_segment_ids.shape}"
            )
        if q_segment_ids.shape[1] != q.shape[1]:
            raise ValueError(
                f"query segment ids length must match q sequence length {q.shape[1]}, got {q_segment_ids.shape[1]}"
            )
        if q_segment_ids.shape[0] not in (1, batch_size):
            raise ValueError(
                f"query segment ids batch dimension must be 1 or {batch_size}, got {q_segment_ids.shape[0]}"
            )
        if q_segment_ids.shape[0] == 1 and batch_size > 1:
            q_segment_ids = jnp.tile(q_segment_ids, (batch_size, 1))

        if kv_segment_ids.ndim != 2:
            raise ValueError(
                "key segment ids must be 2D (shared across batch as [1, S] or per-batch as [B, S]), "
                f"got shape {kv_segment_ids.shape}"
            )
        if kv_segment_ids.shape[1] != k.shape[1]:
            raise ValueError(
                f"key segment ids length must match k sequence length {k.shape[1]}, got {kv_segment_ids.shape[1]}"
            )
        if kv_segment_ids.shape[0] not in (1, batch_size):
            raise ValueError(
                f"key segment ids batch dimension must be 1 or {batch_size}, got {kv_segment_ids.shape[0]}"
            )
        if kv_segment_ids.shape[0] == 1 and batch_size > 1:
            kv_segment_ids = jnp.tile(kv_segment_ids, (batch_size, 1))

    out_shape: list[jax.ShapeDtypeStruct | jax.Array] = [q]
    out_specs = [
        pl.BlockSpec(
            (None, block_q, None, head_dim_padded),
            lambda i, j, k: (j, i, k, 0),
        )
    ]
    if return_residuals:
        out_shape.append(jax.ShapeDtypeStruct(shape=(batch_size, num_heads, q_seq_len), dtype=jnp.float32))
        out_specs.append(pl.BlockSpec((None, None, block_q), lambda i, j, k: (j, k, i)))

    out = pl.pallas_call(
        functools.partial(
            mha_forward_kernel,
            sm_scale=sm_scale,
            causal=causal,
            sliding_window=sliding_window,
            block_q=block_q,
            block_k=block_k,
            head_dim=head_dim,
        ),
        grid=grid_,
        in_specs=[
            pl.BlockSpec((None, block_q, None, head_dim_padded), lambda i, j, k: (j, i, k, 0)),
            pl.BlockSpec((None, kv_seq_len, None, head_dim_padded), lambda _, j, k: (j, 0, k, 0)),
            pl.BlockSpec((None, kv_seq_len, None, head_dim_padded), lambda _, j, k: (j, 0, k, 0)),
            None if segment_ids is None else pl.BlockSpec((None, q_seq_len), lambda _, j, k: (j, 0)),
            None if segment_ids is None else pl.BlockSpec((None, kv_seq_len), lambda _, j, k: (j, 0)),
        ],
        out_shape=out_shape,
        out_specs=out_specs,
        debug=debug,
        interpret=interpret,
        compiler_params=plgpu.CompilerParams(num_warps=num_warps_, num_stages=num_stages),
        name="mha_forward_sliding",
    )(q, k, v, q_segment_ids, kv_segment_ids)
    if return_residuals:
        return out
    return out[0] if isinstance(out, tuple) else out


def _mha_forward(
    q,
    k,
    v,
    segment_ids,
    sm_scale,
    causal,
    sliding_window,
    block_sizes,
    backward_pass_impl,
    num_warps,
    num_stages,
    grid,
    interpret,
    debug,
    return_residuals,
):
    out, lse = _mha_call(
        q,
        k,
        v,
        segment_ids=segment_ids,
        sm_scale=sm_scale,
        causal=causal,
        sliding_window=sliding_window,
        block_sizes=block_sizes,
        backward_pass_impl=backward_pass_impl,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=grid,
        interpret=interpret,
        debug=debug,
        return_residuals=True,
    )
    residuals = (q, k, v, segment_ids, out, lse)
    ret = (out, lse) if return_residuals else out
    return ret, residuals


@functools.partial(
    jax.custom_vjp,
    nondiff_argnums=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
)
@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "causal",
        "sliding_window",
        "block_sizes",
        "backward_pass_impl",
        "num_warps",
        "num_stages",
        "grid",
        "interpret",
        "debug",
        "return_residuals",
    ],
)
def mha(
    q,
    k,
    v,
    segment_ids: tuple[jax.Array, jax.Array] | jax.Array | None,
    sm_scale: float = 1.0,
    causal: bool = False,
    sliding_window: int | None = None,
    block_sizes: BlockSizes = BlockSizes.get_default(),
    backward_pass_impl: str = "triton",
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
    return_residuals: bool = False,
):
    return _mha_call(
        q,
        k,
        v,
        segment_ids=segment_ids,
        sm_scale=sm_scale,
        causal=causal,
        sliding_window=sliding_window,
        block_sizes=block_sizes,
        backward_pass_impl=backward_pass_impl,
        num_warps=num_warps,
        num_stages=num_stages,
        grid=grid,
        interpret=interpret,
        debug=debug,
        return_residuals=return_residuals,
    )


def _mha_backward(
    sm_scale,
    causal,
    sliding_window,
    block_sizes,
    backward_pass_impl,
    num_warps,
    num_stages,
    grid,
    interpret,
    debug,
    return_residuals,
    res,
    do,
):
    if return_residuals:
        raise NotImplementedError("return_residuals not supported.")
    q, k, v, segment_ids, out, lse = res
    del num_stages, grid

    if segment_ids is None:
        q_segment_ids = None
        kv_segment_ids = None
    elif isinstance(segment_ids, tuple):
        q_segment_ids, kv_segment_ids = segment_ids
    else:
        q_segment_ids = segment_ids
        kv_segment_ids = segment_ids

    if backward_pass_impl == "xla":
        _, vjp = jax.vjp(
            functools.partial(
                _mha_reference,
                sm_scale=sm_scale,
                causal=causal,
                sliding_window=sliding_window,
            ),
            q,
            k,
            v,
            segment_ids,
        )
        dq, dk, dv, _ = vjp(do)
        return dq, dk, dv, None

    if backward_pass_impl == "triton":
        batch_size, q_seq_len, num_heads, head_dim = q.shape
        kv_seq_len = k.shape[1]
        block_q = min(block_sizes.block_q, q_seq_len)
        block_q_dkv, block_kv_dkv, block_q_dq, block_kv_dq = _resolve_backward_block_sizes(
            q_seq_len,
            kv_seq_len,
            block_sizes,
            block_q,
            min(block_sizes.block_k, kv_seq_len),
        )
        head_dim_padded = pl.next_power_of_2(head_dim)

        if q_seq_len // block_q_dq != kv_seq_len // block_kv_dkv:
            raise ValueError(
                "q_seq_len and kv_seq_len must be divided into the same "
                "number of blocks for the fused backward pass."
            )

        delta = _preprocess_backward(out, do, lse, block_q, debug, interpret)
        out_shapes = [
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct(k.shape, k.dtype),
            jax.ShapeDtypeStruct(v.shape, v.dtype),
        ]

        in_specs = [
            pl.BlockSpec((None, q_seq_len, None, head_dim_padded), lambda i, j, _: (i, 0, j, 0)),
            pl.BlockSpec((None, kv_seq_len, None, head_dim_padded), lambda i, j, _: (i, 0, j, 0)),
            pl.BlockSpec((None, kv_seq_len, None, head_dim_padded), lambda i, j, _: (i, 0, j, 0)),
            None if q_segment_ids is None else pl.BlockSpec((None, q_seq_len), lambda i, j, _: (i, 0)),
            None if kv_segment_ids is None else pl.BlockSpec((None, kv_seq_len), lambda i, j, _: (i, 0)),
            pl.BlockSpec((None, q_seq_len, None, head_dim_padded), lambda i, j, _: (i, 0, j, 0)),
            pl.BlockSpec((None, q_seq_len, None, head_dim_padded), lambda i, j, _: (i, 0, j, 0)),
            pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
            pl.BlockSpec((None, None, q_seq_len), lambda i, j, _: (i, j, 0)),
        ]

        grid = (batch_size, num_heads, pl.cdiv(kv_seq_len, block_kv_dkv))
        num_warps_ = num_warps
        if num_warps_ is None:
            if block_q_dkv * block_kv_dkv < 128 * 128 or block_q_dq * block_kv_dq < 128 * 128:
                num_warps_ = 4
            else:
                num_warps_ = 8

        dq, dk, dv = pl.pallas_call(
            functools.partial(
                mha_backward_kernel,
                sm_scale=sm_scale,
                causal=causal,
                block_q_dkv=block_q_dkv,
                block_kv_dkv=block_kv_dkv,
                block_q_dq=block_q_dq,
                block_kv_dq=block_kv_dq,
                head_dim=head_dim,
                sliding_window=sliding_window,
            ),
            out_shape=out_shapes,
            in_specs=in_specs,
            grid=grid,
            out_specs=[
                pl.BlockSpec(
                    (None, block_q_dq, None, head_dim_padded),
                    lambda i, j, k: (i, k, j, 0),
                ),
                pl.BlockSpec(
                    (None, block_kv_dkv, None, head_dim_padded),
                    lambda i, j, k: (i, k, j, 0),
                ),
                pl.BlockSpec(
                    (None, block_kv_dkv, None, head_dim_padded),
                    lambda i, j, k: (i, k, j, 0),
                ),
            ],
            name="mha_backward",
            debug=debug,
            interpret=interpret,
            compiler_params=plgpu.CompilerParams(num_warps=num_warps_, num_stages=2),
        )(q, k, v, q_segment_ids, kv_segment_ids, out, do, lse, delta)
        return dq.astype(q.dtype), dk, dv, None

    raise ValueError(f"Invalid backward pass implementation: {backward_pass_impl}")


def _preprocess_backward_kernel(out_ref, dout_ref, delta_ref, head_dim: int):
    # load
    head_mask = (jnp.arange(out_ref.shape[-1]) < head_dim)[None, :]
    o = plgpu.load(out_ref, mask=head_mask, other=0.0)
    do = plgpu.load(dout_ref, mask=head_mask, other=0.0)
    # compute
    delta = jnp.sum(o * do, axis=1)
    # write-back
    delta_ref[...] = delta.astype(delta_ref.dtype)


@jax.named_scope("preprocess_backward")
def _preprocess_backward(out, do, lse, block_q: int, debug: bool, interpret: bool):
    batch_size, seq_len, num_heads, head_dim = out.shape
    head_dim_padded = pl.next_power_of_2(head_dim)
    out_shape = jax.ShapeDtypeStruct(lse.shape, lse.dtype)
    delta = pl.pallas_call(
        functools.partial(_preprocess_backward_kernel, head_dim=head_dim),
        grid=(pl.cdiv(seq_len, block_q), batch_size, num_heads),
        in_specs=[
            pl.BlockSpec((None, block_q, None, head_dim_padded), lambda i, j, k: (j, i, k, 0)),
            pl.BlockSpec((None, block_q, None, head_dim_padded), lambda i, j, k: (j, i, k, 0)),
        ],
        out_specs=pl.BlockSpec((None, None, block_q), lambda i, j, k: (j, k, i)),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=3),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_preprocess_backward",
    )(out, do)
    return delta


def mha_backward_kernel(
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref: jax.Array | None,
    kv_segment_ids_ref: jax.Array | None,
    out_ref,
    do_scaled_ref,
    lse_ref,
    delta_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    *,
    sm_scale: float,
    causal: bool,
    block_q_dkv: int,
    block_kv_dkv: int,
    block_q_dq: int,
    block_kv_dq: int,
    head_dim: int,
    sliding_window: int | None,
) -> None:
    del out_ref
    q_seq_len = q_ref.shape[0]
    kv_seq_len = k_ref.shape[0]

    start_k = pl.program_id(2)
    curr_k_slice = pl.dslice(start_k * block_kv_dkv, block_kv_dkv)

    head_dim_padded = q_ref.shape[-1]
    dv = jnp.zeros((block_kv_dkv, head_dim_padded), dtype=jnp.float32)
    dk = jnp.zeros((block_kv_dkv, head_dim_padded), dtype=jnp.float32)

    head_mask = (jnp.arange(head_dim_padded) < head_dim)[None, :]
    v = plgpu.load(v_ref.at[curr_k_slice, :], mask=head_mask, other=0.0)
    k = plgpu.load(k_ref.at[curr_k_slice, :], mask=head_mask, other=0.0)
    span_k = start_k * block_kv_dkv + jnp.arange(block_kv_dkv)
    kv_segment_ids = None if kv_segment_ids_ref is None else kv_segment_ids_ref[curr_k_slice]

    def inner_loop_dkdv(start_q, carry):
        dv_, dk_ = carry
        curr_q_slice = pl.dslice(start_q * block_q_dkv, block_q_dkv)
        q = plgpu.load(q_ref.at[curr_q_slice, :], mask=head_mask, other=0.0)
        qk = pl.dot(q, k.T)
        qk_scale = math.log2(math.e)
        if sm_scale != 1.0:
            qk_scale *= sm_scale
        qk *= qk_scale

        if causal or q_segment_ids_ref is not None or sliding_window is not None:
            mask = None
            if q_segment_ids_ref is not None:
                q_segment_ids = q_segment_ids_ref[curr_q_slice]
                mask = segment_mask(q_segment_ids, kv_segment_ids)
            if causal:
                span_q = start_q * block_q_dkv + jnp.arange(block_q_dkv)
                causal_mask = span_q[:, None] >= span_k[None, :]
                mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
            if sliding_window is not None:
                span_q = start_q * block_q_dkv + jnp.arange(block_q_dkv)
                window_mask = _apply_window_mask(mask, span_q, span_k, sliding_window=sliding_window)
                mask = window_mask
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

        lse = lse_ref[curr_q_slice]
        di = delta_ref[curr_q_slice]
        do = plgpu.load(do_scaled_ref.at[curr_q_slice, :], mask=head_mask, other=0.0)

        p = jnp.exp2(qk - lse[:, None])
        dv_ = dv_ + pl.dot(p.astype(do.dtype).T, do)
        dp = jnp.zeros((block_q_dkv, block_kv_dkv), dtype=jnp.float32) - di[:, None]
        dp = dp + pl.dot(do, v.T)
        ds = p * dp
        if sm_scale != 1.0:
            ds = ds * sm_scale
        dk_ = dk_ + pl.dot(ds.astype(q_ref.dtype).T, q)
        return dv_, dk_

    lower_bound = lax.div(start_k * block_kv_dkv, block_q_dkv) if causal else 0
    dv, dk = lax.fori_loop(lower_bound, pl.cdiv(q_seq_len, block_q_dkv), inner_loop_dkdv, (dv, dk))
    plgpu.store(
        dv_ref.at[:, : dv.shape[-1]],
        dv.astype(dv_ref.dtype),
        mask=head_mask,
    )
    plgpu.store(
        dk_ref.at[:, : dk.shape[-1]],
        dk.astype(dk_ref.dtype),
        mask=head_mask,
    )

    start_q = pl.program_id(2)
    curr_q_slice = pl.dslice(start_q * block_q_dq, block_q_dq)
    span_q = start_q * block_q_dq + jnp.arange(block_q_dq)
    dq = jnp.zeros((block_q_dq, head_dim_padded), dtype=jnp.float32)

    q = plgpu.load(q_ref.at[curr_q_slice, :], mask=head_mask, other=0.0)
    q_segment_ids = None if q_segment_ids_ref is None else q_segment_ids_ref[curr_q_slice]
    lse = lse_ref[curr_q_slice]
    do = plgpu.load(do_scaled_ref.at[curr_q_slice, :], mask=head_mask, other=0.0)
    di = delta_ref[curr_q_slice]
    if q_segment_ids_ref is not None and kv_segment_ids_ref is None:
        raise ValueError("q and kv segment ids must be provided together.")

    def inner_loop_dq(start_k, dq_):
        curr_k_slice = pl.dslice(start_k * block_kv_dq, block_kv_dq)
        k = plgpu.load(k_ref.at[curr_k_slice, :], mask=head_mask, other=0.0)
        v = plgpu.load(v_ref.at[curr_k_slice, :], mask=head_mask, other=0.0)

        qk = pl.dot(q, k.T)
        qk_scale = math.log2(math.e)
        if sm_scale != 1.0:
            qk_scale *= sm_scale
        qk *= qk_scale

        if causal or q_segment_ids_ref is not None or sliding_window is not None:
            mask = None
            if q_segment_ids_ref is not None:
                kv_segment_ids = cast(jax.Array, kv_segment_ids_ref)[curr_k_slice]
                mask = segment_mask(q_segment_ids, kv_segment_ids)
            if causal:
                span_k = start_k * block_kv_dq + jnp.arange(block_kv_dq)
                causal_mask = span_q[:, None] >= span_k[None, :]
                mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
            if sliding_window is not None:
                span_k = start_k * block_kv_dq + jnp.arange(block_kv_dq)
                window_mask = _apply_window_mask(mask, span_q, span_k, sliding_window=sliding_window)
                mask = window_mask
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

        p = jnp.exp2(qk - lse[:, None])
        dp = jnp.zeros((block_q_dq, block_kv_dq), dtype=jnp.float32) - di[:, None]
        dp = dp + pl.dot(do, v.T)
        ds = p * dp
        if sm_scale != 1.0:
            ds = ds * sm_scale

        dq_ = dq_ + pl.dot(ds.astype(k.dtype), k).astype(dq_.dtype)
        return dq_

    if causal:
        upper_bound = pl.cdiv((start_q + 1) * block_q_dq, block_kv_dq)
    else:
        upper_bound = pl.cdiv(kv_seq_len, block_kv_dq)

    dq = lax.fori_loop(0, upper_bound, inner_loop_dq, (dq))
    plgpu.store(dq_ref.at[:, : dq.shape[-1]], dq.astype(dq_ref.dtype), mask=head_mask)


def _mha_reference(
    q,
    k,
    v,
    segment_ids: tuple[jax.Array, jax.Array] | jax.Array | None,
    *,
    sm_scale: float,
    causal: bool,
    sliding_window: int | None,
) -> jax.Array:
    q_seq_len = q.shape[1]
    kv_seq_len = k.shape[1]
    logits = jnp.einsum("bqhc,bkhc->bhqk", q, k, preferred_element_type=jnp.float32)
    mask = None
    if segment_ids is not None:
        if isinstance(segment_ids, tuple):
            q_ids, kv_ids = segment_ids
            mask = segment_mask(q_ids, kv_ids)
        else:
            mask = segment_mask(segment_ids, segment_ids)
        mask = jnp.expand_dims(mask, axis=1)
    if causal:
        causal_mask = jnp.tril(jnp.ones((1, 1, q_seq_len, kv_seq_len), dtype=bool))
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
    if sliding_window is not None:
        q_idx = jnp.arange(q_seq_len)
        k_idx = jnp.arange(kv_seq_len)
        window_mask = k_idx[None, :] >= (q_idx[:, None] - (sliding_window - 1))
        mask = window_mask if mask is None else jnp.logical_and(mask, window_mask)
        mask = jnp.broadcast_to(mask, logits.shape)
    logits = logits if mask is None else jnp.where(mask, logits, float("-inf"))
    logits = logits * sm_scale
    weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    return jnp.einsum("bhqk,bkhc->bqhc", weights, v)


mha.defvjp(_mha_forward, _mha_backward)


__all__ = ["BlockSizes", "DEFAULT_MASK_VALUE", "mha"]
