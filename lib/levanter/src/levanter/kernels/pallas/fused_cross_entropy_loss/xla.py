# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import Optional, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .config import BlockSizes
from .reference import linear_softmax_cross_entropy_loss_reference, linear_softmax_cross_entropy_loss_streaming
from .tuned_block_sizes import (
    _largest_divisor_at_most,
    infer_block_sizes_with_tuned_match,
    infer_xla_b_block_size,
    infer_xla_v_block_size,
)

_XLA_MAX_TILE_WORDS = 2**31 - 1


def _materialize_cotangent(
    cotangent: jax.Array | jax.custom_derivatives.SymbolicZero, reference: jax.Array
) -> jax.Array:
    if isinstance(cotangent, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros_like(reference)
    return jnp.asarray(cotangent, dtype=reference.dtype)


def _resolve_xla_batch_block_size(
    b_dim: int,
    v_block_size: int,
    requested_b_block_size: int | None,
    *,
    strict: bool,
) -> int:
    inferred = infer_xla_b_block_size(b_dim, v_block_size)
    if requested_b_block_size is None:
        return inferred

    requested = requested_b_block_size
    if requested <= 0:
        if strict:
            raise ValueError(f"XLA batch block size must be positive, got {requested}.")
        return inferred

    if v_block_size <= 0:
        if strict:
            raise ValueError(f"XLA vocab block size must be positive, got {v_block_size}.")
        return inferred

    max_b_block_size = _XLA_MAX_TILE_WORDS // v_block_size
    if max_b_block_size <= 0:
        if strict:
            raise ValueError(
                "XLA batch/vocab tile exceeds TPU/XLA int32 word-count limit: "
                f"b_block_size={requested}, v_block_size={v_block_size}."
            )
        return 1

    if requested * v_block_size > _XLA_MAX_TILE_WORDS:
        if strict:
            raise ValueError(
                "XLA batch/vocab tile exceeds TPU/XLA int32 word-count limit: "
                f"b_block_size={requested}, v_block_size={v_block_size}."
            )
        requested = max_b_block_size

    if requested <= b_dim and b_dim % requested == 0:
        return requested

    # Default/tuned block sizes often encode a preferred upper bound like 1024.
    # Preserve that intent by shrinking to the largest valid divisor of B instead
    # of rejecting smaller local batches outright.
    return _largest_divisor_at_most(b_dim, min(requested, inferred))


def _infer_tuned_xla_batch_block_size(
    b_dim: int,
    h_dim: int,
    v_dim: int,
    *,
    dtype: Optional[jnp.dtype],
    x_dtype: jnp.dtype,
    w_dtype: jnp.dtype,
) -> int | None:
    tuned_block_sizes, has_tuned_match = infer_block_sizes_with_tuned_match(
        b_dim,
        h_dim,
        v_dim,
        dtype=dtype,
        x_dtype=x_dtype,
        w_dtype=w_dtype,
    )
    if not has_tuned_match:
        return None
    return tuned_block_sizes.b_block_size


def _linear_softmax_cross_entropy_loss_streaming_fwd(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    *,
    block_size: int,
    dtype: Optional[jnp.dtype],
    batch_block_size: int,
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
) -> tuple[Float[Array, "B"], Float[Array, "B"]]:
    if batch_block_size <= 0:
        raise ValueError(f"batch_block_size must be positive, got {batch_block_size}.")

    b_dim, h_dim = x.shape
    if b_dim % batch_block_size != 0:
        raise ValueError(
            f"batch_block_size must divide batch dimension, got B={b_dim}, batch_block_size={batch_block_size}."
        )

    if batch_block_size >= b_dim:
        return cast(
            tuple[jax.Array, jax.Array],
            linear_softmax_cross_entropy_loss_streaming(
                x,
                labels,
                w,
                block_size=block_size,
                dtype=dtype,
                logit_soft_cap=logit_soft_cap,
                precision=precision,
                return_argmax=False,
            ),
        )

    out_dtype = jnp.dtype(dtype) if dtype is not None else x.dtype
    loss_init = jnp.zeros((b_dim,), dtype=out_dtype)
    lse_init = jnp.zeros((b_dim,), dtype=out_dtype)
    num_b_blocks = b_dim // batch_block_size

    def body(block_idx, state):
        loss, lse = state
        start = block_idx * batch_block_size
        x_block = jax.lax.dynamic_slice(x, (start, 0), (batch_block_size, h_dim))
        labels_block = jax.lax.dynamic_slice(labels, (start,), (batch_block_size,))

        loss_block, lse_block = cast(
            tuple[jax.Array, jax.Array],
            linear_softmax_cross_entropy_loss_streaming(
                x_block,
                labels_block,
                w,
                block_size=block_size,
                dtype=dtype,
                logit_soft_cap=logit_soft_cap,
                precision=precision,
                return_argmax=False,
            ),
        )
        loss = jax.lax.dynamic_update_slice(loss, loss_block, (start,))
        lse = jax.lax.dynamic_update_slice(lse, lse_block, (start,))
        return loss, lse

    return jax.lax.fori_loop(0, num_b_blocks, body, (loss_init, lse_init))


def _linear_softmax_cross_entropy_loss_streaming_fwd_with_argmax(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    *,
    block_size: int,
    dtype: Optional[jnp.dtype],
    batch_block_size: int,
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
) -> tuple[Float[Array, "B"], Float[Array, "B"], Int[Array, "B"]]:
    if batch_block_size <= 0:
        raise ValueError(f"batch_block_size must be positive, got {batch_block_size}.")

    b_dim, h_dim = x.shape
    if b_dim % batch_block_size != 0:
        raise ValueError(
            f"batch_block_size must divide batch dimension, got B={b_dim}, batch_block_size={batch_block_size}."
        )

    if batch_block_size >= b_dim:
        return cast(
            tuple[jax.Array, jax.Array, jax.Array],
            linear_softmax_cross_entropy_loss_streaming(
                x,
                labels,
                w,
                block_size=block_size,
                dtype=dtype,
                logit_soft_cap=logit_soft_cap,
                precision=precision,
                return_argmax=True,
            ),
        )

    out_dtype = jnp.dtype(dtype) if dtype is not None else x.dtype
    loss_init = jnp.zeros((b_dim,), dtype=out_dtype)
    lse_init = jnp.zeros((b_dim,), dtype=out_dtype)
    argmax_init = jnp.zeros((b_dim,), dtype=jnp.int32)
    num_b_blocks = b_dim // batch_block_size

    def body(block_idx, state):
        loss, lse, argmax = state
        start = block_idx * batch_block_size
        x_block = jax.lax.dynamic_slice(x, (start, 0), (batch_block_size, h_dim))
        labels_block = jax.lax.dynamic_slice(labels, (start,), (batch_block_size,))

        loss_block, lse_block, argmax_block = cast(
            tuple[jax.Array, jax.Array, jax.Array],
            linear_softmax_cross_entropy_loss_streaming(
                x_block,
                labels_block,
                w,
                block_size=block_size,
                dtype=dtype,
                logit_soft_cap=logit_soft_cap,
                precision=precision,
                return_argmax=True,
            ),
        )
        loss = jax.lax.dynamic_update_slice(loss, loss_block, (start,))
        lse = jax.lax.dynamic_update_slice(lse, lse_block, (start,))
        argmax = jax.lax.dynamic_update_slice(argmax, argmax_block, (start,))
        return loss, lse, argmax

    return jax.lax.fori_loop(0, num_b_blocks, body, (loss_init, lse_init, argmax_init))


def _linear_softmax_cross_entropy_loss_streaming_bwd(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    lse: Float[Array, "B"],
    dout_loss: Float[Array, "B"],
    dout_lse: Float[Array, "B"],
    *,
    block_size: int,
    dtype: Optional[jnp.dtype],
    batch_block_size: int,
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
) -> tuple[Float[Array, "B H"], Float[Array, "H V"]]:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    b_dim, h_dim = x.shape
    if batch_block_size <= 0:
        raise ValueError(f"batch_block_size must be positive, got {batch_block_size}.")
    if b_dim % batch_block_size != 0:
        raise ValueError(
            f"batch_block_size must divide batch dimension, got B={b_dim}, batch_block_size={batch_block_size}."
        )

    v_dim = w.shape[1]
    row_indices = jnp.arange(batch_block_size, dtype=labels.dtype)

    pad = (-v_dim) % block_size
    if pad:
        w_padded = jnp.pad(w, ((0, 0), (0, pad)), mode="constant", constant_values=0)
    else:
        w_padded = w

    v_padded = v_dim + pad
    num_blocks = v_padded // block_size
    lse_dtype = lse.dtype
    dout_loss = dout_loss.astype(lse_dtype)
    dout_lse = dout_lse.astype(lse_dtype)
    gx_init = jnp.zeros_like(x)
    gw_init = jnp.zeros((h_dim, v_padded), dtype=w.dtype)
    num_b_blocks = b_dim // batch_block_size

    def body(block_idx, state):
        gx, gw = state
        start = block_idx * block_size

        w_block = jax.lax.dynamic_slice(w_padded, (0, start), (h_dim, block_size))
        valid = (start + jnp.arange(block_size, dtype=labels.dtype)) < v_dim

        def batch_body(batch_block_idx, batch_state):
            gx_inner, gw_block = batch_state
            batch_start = batch_block_idx * batch_block_size
            x_block = jax.lax.dynamic_slice(x, (batch_start, 0), (batch_block_size, h_dim))
            labels_block = jax.lax.dynamic_slice(labels, (batch_start,), (batch_block_size,))
            lse_block = jax.lax.dynamic_slice(lse, (batch_start,), (batch_block_size,))
            dout_loss_block = jax.lax.dynamic_slice(dout_loss, (batch_start,), (batch_block_size,))
            dout_lse_block = jax.lax.dynamic_slice(dout_lse, (batch_start,), (batch_block_size,))

            logits = jax.lax.dot_general(
                x_block,
                w_block,
                (((1,), (0,)), ((), ())),
                precision=precision,
                preferred_element_type=jnp.float32,
            )
            if dtype is not None:
                logits = logits.astype(dtype)

            cap_deriv = jnp.asarray(1.0, dtype=logits.dtype)
            if logit_soft_cap is not None:
                tanh_arg = logits / logit_soft_cap
                tanh_val = jnp.tanh(tanh_arg)
                logits = tanh_val * logit_soft_cap
                cap_deriv = (1.0 - tanh_val**2).astype(logits.dtype)

            logits = jnp.where(valid[None, :], logits, -jnp.inf)
            probs = jnp.exp(logits - lse_block[:, None].astype(logits.dtype))
            delta = (
                dout_loss_block[:, None].astype(logits.dtype) + dout_lse_block[:, None].astype(logits.dtype)
            ) * probs

            in_block = (labels_block >= start) & (labels_block < start + block_size)
            label_idx = labels_block - start
            safe_idx = jnp.where(in_block, label_idx, 0)
            delta = delta.at[row_indices, safe_idx].add(
                jnp.where(in_block, -dout_loss_block.astype(logits.dtype), 0.0)
            )
            delta = (delta * cap_deriv).astype(logits.dtype)

            gx_block = jax.lax.dot_general(
                delta,
                w_block,
                (((1,), (1,)), ((), ())),
                precision=precision,
                preferred_element_type=jnp.float32,
            ).astype(gx.dtype)
            gw_block_update = jax.lax.dot_general(
                x_block,
                delta,
                (((0,), (0,)), ((), ())),
                precision=precision,
                preferred_element_type=jnp.float32,
            ).astype(gw.dtype)

            current_gx_block = jax.lax.dynamic_slice(gx_inner, (batch_start, 0), (batch_block_size, h_dim))
            gx_inner = jax.lax.dynamic_update_slice(gx_inner, current_gx_block + gx_block, (batch_start, 0))
            return gx_inner, gw_block + gw_block_update

        gw_block_init = jnp.zeros((h_dim, block_size), dtype=gw.dtype)
        gx, gw_block = jax.lax.fori_loop(0, num_b_blocks, batch_body, (gx, gw_block_init))
        gw = jax.lax.dynamic_update_slice(gw, gw_block, (0, start))
        return gx, gw

    gx, gw = jax.lax.fori_loop(0, num_blocks, body, (gx_init, gw_init))
    return gx, gw[:, :v_dim]


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _linear_softmax_cross_entropy_loss_streaming_custom_vjp(
    block_size: int,
    batch_block_size: int,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
) -> tuple[Float[Array, "B"], Float[Array, "B"]]:
    loss, lse = _linear_softmax_cross_entropy_loss_streaming_fwd(
        x,
        labels,
        w,
        block_size=block_size,
        dtype=dtype,
        batch_block_size=batch_block_size,
        logit_soft_cap=logit_soft_cap,
        precision=precision,
    )
    return loss, lse


def _linear_softmax_cross_entropy_loss_streaming_custom_vjp_fwd(
    block_size: int,
    batch_block_size: int,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
) -> tuple[tuple[Float[Array, "B"], Float[Array, "B"]], tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    loss, lse = _linear_softmax_cross_entropy_loss_streaming_fwd(
        x,
        labels,
        w,
        block_size=block_size,
        dtype=dtype,
        batch_block_size=batch_block_size,
        logit_soft_cap=logit_soft_cap,
        precision=precision,
    )
    return (loss, lse), (x, labels, w, lse)


def _linear_softmax_cross_entropy_loss_streaming_custom_vjp_bwd(
    block_size: int,
    batch_block_size: int,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
    residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    cotangents: tuple[
        jax.Array | jax.custom_derivatives.SymbolicZero, jax.Array | jax.custom_derivatives.SymbolicZero
    ],
) -> tuple[jax.Array, None, jax.Array]:
    x, labels, w, lse = residuals
    dout_loss, dout_lse = cotangents
    dout_loss_arr = _materialize_cotangent(dout_loss, lse)
    dout_lse_arr = _materialize_cotangent(dout_lse, lse)
    gx, gw = _linear_softmax_cross_entropy_loss_streaming_bwd(
        x,
        labels,
        w,
        lse,
        dout_loss_arr,
        dout_lse_arr,
        block_size=block_size,
        dtype=dtype,
        batch_block_size=batch_block_size,
        logit_soft_cap=logit_soft_cap,
        precision=precision,
    )
    return gx, None, gw


_linear_softmax_cross_entropy_loss_streaming_custom_vjp.defvjp(
    _linear_softmax_cross_entropy_loss_streaming_custom_vjp_fwd,
    _linear_softmax_cross_entropy_loss_streaming_custom_vjp_bwd,
)


def linear_softmax_cross_entropy_loss_xla(
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
    *,
    block_sizes: BlockSizes | None = None,
    dtype: Optional[jnp.dtype] = jnp.float32,
    logit_soft_cap: Optional[float] = None,
    precision: jax.lax.PrecisionLike = None,
    return_argmax: bool = False,
) -> tuple[Float[Array, "B"], Float[Array, "B"]] | tuple[Float[Array, "B"], Float[Array, "B"], Int[Array, "B"]]:
    if block_sizes is None:
        v_block_size = infer_xla_v_block_size(x.shape[0], x.shape[1], w.shape[1], dtype=dtype)
        b_block_size = _resolve_xla_batch_block_size(
            x.shape[0],
            v_block_size,
            _infer_tuned_xla_batch_block_size(
                x.shape[0],
                x.shape[1],
                w.shape[1],
                dtype=dtype,
                x_dtype=x.dtype,
                w_dtype=w.dtype,
            ),
            strict=False,
        )
    else:
        v_block_size = block_sizes.v_block_size
        b_block_size = _resolve_xla_batch_block_size(x.shape[0], v_block_size, block_sizes.b_block_size, strict=True)
    if v_block_size >= w.shape[1]:
        return linear_softmax_cross_entropy_loss_reference(
            x,
            labels,
            w,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
            return_argmax=return_argmax,
        )

    if return_argmax:
        return _linear_softmax_cross_entropy_loss_streaming_fwd_with_argmax(
            x,
            labels,
            w,
            block_size=v_block_size,
            dtype=dtype,
            batch_block_size=b_block_size,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
        )

    return _linear_softmax_cross_entropy_loss_streaming_custom_vjp(
        v_block_size,
        b_block_size,
        dtype,
        logit_soft_cap,
        precision,
        x,
        labels,
        w,
    )


__all__ = ["linear_softmax_cross_entropy_loss_xla"]
