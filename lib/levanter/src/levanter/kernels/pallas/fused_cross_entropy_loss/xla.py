# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .config import BlockSizes
from .reference import linear_softmax_cross_entropy_loss_reference, linear_softmax_cross_entropy_loss_streaming
from .tuned_block_sizes import infer_xla_v_block_size


def _materialize_cotangent(
    cotangent: jax.Array | jax.custom_derivatives.SymbolicZero, reference: jax.Array
) -> jax.Array:
    if isinstance(cotangent, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros_like(reference)
    return jnp.asarray(cotangent, dtype=reference.dtype)


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
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
) -> tuple[Float[Array, "B H"], Float[Array, "H V"]]:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    b_dim, h_dim = x.shape
    v_dim = w.shape[1]
    row_indices = jnp.arange(b_dim, dtype=labels.dtype)

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

    def body(block_idx, state):
        gx, gw = state
        start = block_idx * block_size

        w_block = jax.lax.dynamic_slice(w_padded, (0, start), (h_dim, block_size))
        logits = jax.lax.dot_general(
            x,
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

        valid = (start + jnp.arange(block_size, dtype=labels.dtype)) < v_dim
        logits = jnp.where(valid[None, :], logits, -jnp.inf)

        probs = jnp.exp(logits - lse[:, None].astype(logits.dtype))
        delta = (dout_loss[:, None].astype(logits.dtype) + dout_lse[:, None].astype(logits.dtype)) * probs

        in_block = (labels >= start) & (labels < start + block_size)
        label_idx = labels - start
        safe_idx = jnp.where(in_block, label_idx, 0)
        delta = delta.at[row_indices, safe_idx].add(jnp.where(in_block, -dout_loss.astype(logits.dtype), 0.0))
        delta = (delta * cap_deriv).astype(logits.dtype)

        gx_block = jax.lax.dot_general(
            delta,
            w_block,
            (((1,), (1,)), ((), ())),
            precision=precision,
            preferred_element_type=jnp.float32,
        ).astype(gx.dtype)
        gw_block = jax.lax.dot_general(
            x,
            delta,
            (((0,), (0,)), ((), ())),
            precision=precision,
            preferred_element_type=jnp.float32,
        ).astype(gw.dtype)
        gx = gx + gx_block
        gw = jax.lax.dynamic_update_slice(gw, gw_block, (0, start))
        return gx, gw

    gx, gw = jax.lax.fori_loop(0, num_blocks, body, (gx_init, gw_init))
    return gx, gw[:, :v_dim]


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _linear_softmax_cross_entropy_loss_streaming_custom_vjp(
    block_size: int,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
) -> tuple[Float[Array, "B"], Float[Array, "B"]]:
    loss, lse, *_ = linear_softmax_cross_entropy_loss_streaming(
        x,
        labels,
        w,
        block_size=block_size,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
        precision=precision,
    )
    return loss, lse


def _linear_softmax_cross_entropy_loss_streaming_custom_vjp_fwd(
    block_size: int,
    dtype: Optional[jnp.dtype],
    logit_soft_cap: Optional[float],
    precision: jax.lax.PrecisionLike,
    x: Float[Array, "B H"],
    labels: Int[Array, "B"],
    w: Float[Array, "H V"],
) -> tuple[tuple[Float[Array, "B"], Float[Array, "B"]], tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    loss, lse, *_ = linear_softmax_cross_entropy_loss_streaming(
        x,
        labels,
        w,
        block_size=block_size,
        dtype=dtype,
        logit_soft_cap=logit_soft_cap,
        precision=precision,
    )
    return (loss, lse), (x, labels, w, lse)


def _linear_softmax_cross_entropy_loss_streaming_custom_vjp_bwd(
    block_size: int,
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
    else:
        v_block_size = block_sizes.v_block_size
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
        return linear_softmax_cross_entropy_loss_streaming(
            x,
            labels,
            w,
            block_size=v_block_size,
            dtype=dtype,
            logit_soft_cap=logit_soft_cap,
            precision=precision,
            return_argmax=True,
        )

    return _linear_softmax_cross_entropy_loss_streaming_custom_vjp(
        v_block_size,
        dtype,
        logit_soft_cap,
        precision,
        x,
        labels,
        w,
    )


__all__ = ["linear_softmax_cross_entropy_loss_xla"]
