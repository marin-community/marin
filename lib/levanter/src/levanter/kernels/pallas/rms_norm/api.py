# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Public dispatch and custom backward for the RMSNorm prototype."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import math
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .pallas_tpu import batch_invariant_rms_norm_pallas_tpu
from .reference import MAX_FLATTENED_ROWS, SUPPORTED_HIDDEN_DIMS, _sum_hidden_tiles, batch_invariant_rms_norm_reference

Implementation: TypeAlias = Literal["reference", "pallas_tpu"]
ArrayImpl = Callable[..., jax.Array]


def _reference_impl(x: jax.Array, weight: jax.Array, *, eps: float) -> jax.Array:
    return batch_invariant_rms_norm_reference(x, weight, eps=eps)


def _pallas_tpu_impl(x: jax.Array, weight: jax.Array, *, eps: float) -> jax.Array:
    return batch_invariant_rms_norm_pallas_tpu(x, weight, eps=eps)


IMPLEMENTATIONS: dict[str, ArrayImpl] = {
    "reference": _reference_impl,
    "pallas_tpu": _pallas_tpu_impl,
}


def _validate_public_inputs(x: jax.Array, weight: jax.Array, eps: float) -> None:
    if x.ndim < 1:
        raise ValueError(f"x must have at least one dimension, got shape {x.shape}.")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError(f"weight must have shape ({x.shape[-1]},), got {weight.shape}.")
    if x.dtype not in (jnp.float32, jnp.bfloat16) or weight.dtype not in (jnp.float32, jnp.bfloat16):
        raise TypeError(f"batch-invariant rms_norm supports float32/bfloat16, got x={x.dtype}, weight={weight.dtype}.")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be a positive finite scalar, got {eps}.")


def _forward(x: jax.Array, weight: jax.Array, *, eps: float, implementation: Implementation) -> jax.Array:
    _validate_public_inputs(x, weight, eps)
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"Unsupported RMSNorm implementation: {implementation!r}.")
    hidden = x.shape[-1]
    if hidden not in SUPPORTED_HIDDEN_DIMS:
        raise ValueError(
            f"batch-invariant rms_norm supports hidden dimensions {sorted(SUPPORTED_HIDDEN_DIMS)}, got {hidden}."
        )
    rows = math.prod(x.shape[:-1]) if x.ndim > 1 else 1
    if not 1 <= rows <= MAX_FLATTENED_ROWS:
        raise ValueError(f"batch-invariant rms_norm supports 1..{MAX_FLATTENED_ROWS} flattened rows, got {rows}.")
    flat_x = x.reshape((-1, hidden))
    out = IMPLEMENTATIONS[implementation](flat_x, weight, eps=eps)
    return out.reshape(x.shape)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def batch_invariant_rms_norm(
    x: Float[Array, "... D"],
    weight: Float[Array, "D"],
    eps: float,
    implementation: Implementation = "reference",
) -> jax.Array:
    """Opt-in weighted RMSNorm with row arithmetic independent of surrounding batch.

    The TPU implementation is local to each visible chip and supports only a
    replicated weight and row-sharded activation. No implicit all-gather occurs.
    """
    return _forward(x, weight, eps=eps, implementation=implementation)


def _batch_invariant_rms_norm_fwd(x, weight, eps, implementation):
    y = _forward(x, weight, eps=eps, implementation=implementation)
    return y, (x, weight)


def _batch_invariant_rms_norm_bwd(eps, implementation, residual, dy):
    del implementation
    x, weight = residual
    x32 = x.astype(jnp.float32)
    weight32 = weight.astype(jnp.float32)
    dy32 = dy.astype(jnp.float32)
    tile_sums = _sum_hidden_tiles(x32)
    variance = tile_sums[..., None] / x32.shape[-1]
    inv = jax.lax.rsqrt(variance + eps)
    normalized = x32 * inv
    weighted_grad = dy32 * weight32
    dx32 = weighted_grad * inv - x32 * (inv * inv * inv) * jnp.mean(weighted_grad * x32, axis=-1, keepdims=True)
    dw32 = jnp.sum(dy32 * normalized, axis=tuple(range(dy32.ndim - 1)))
    return dx32.astype(x.dtype), dw32.astype(weight.dtype)


batch_invariant_rms_norm.defvjp(_batch_invariant_rms_norm_fwd, _batch_invariant_rms_norm_bwd)
