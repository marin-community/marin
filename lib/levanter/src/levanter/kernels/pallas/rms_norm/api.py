# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Public dispatch and custom backward for the RMSNorm prototype."""

from __future__ import annotations

from functools import partial
from types import MappingProxyType
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jax.sharding import NamedSharding

from levanter.kernels.pallas.autotune_utils import sharding_of

from .pallas_tpu import batch_invariant_rms_norm_pallas_tpu
from .reference import _reference, _sum_hidden_tiles, _validate_inputs

Implementation: TypeAlias = Literal["reference", "pallas_tpu"]


IMPLEMENTATIONS = MappingProxyType({"reference": _reference, "pallas_tpu": batch_invariant_rms_norm_pallas_tpu})


def _reshard_to_nonempty_named_mesh(value: jax.Array, target_sharding) -> jax.Array:
    if isinstance(target_sharding, NamedSharding) and not target_sharding.mesh.empty:
        return jax.sharding.reshard(value, target_sharding)
    return value


def _forward(x: jax.Array, weight: jax.Array, *, eps: float, implementation: Implementation) -> jax.Array:
    _validate_inputs(x, weight, eps)
    apply = IMPLEMENTATIONS.get(implementation)
    if apply is None:
        raise ValueError(f"Unsupported RMSNorm implementation: {implementation!r}.")
    if implementation == "reference":
        return apply(x, weight, eps=eps)

    flat_x = x.reshape((-1, x.shape[-1]))
    return apply(flat_x, weight, eps=eps).reshape(x.shape)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def batch_invariant_rms_norm(
    x: Float[Array, "... D"],
    weight: Float[Array, "D"],
    eps: float,
    implementation: Implementation = "reference",
) -> jax.Array:
    """Opt-in weighted RMSNorm with row arithmetic independent of surrounding batch.

    Accepts positive hidden widths divisible by 128. On a multi-device TPU
    mesh, shard activations by row and replicate the weight. The TPU path
    computes gradients in FP32 and returns them in the input dtypes.
    """
    return _forward(x, weight, eps=eps, implementation=implementation)


def _batch_invariant_rms_norm_fwd(x, weight, eps, implementation):
    y = _forward(x, weight, eps=eps, implementation=implementation)
    return y, (x, weight)


def _parallel_gradients(x: jax.Array, weight: jax.Array, dy: jax.Array, eps: float) -> tuple[jax.Array, jax.Array]:
    x32 = x.astype(jnp.float32)
    dy32 = dy.astype(jnp.float32)
    weighted_dy = dy32 * weight.astype(jnp.float32)
    width = x.shape[-1]
    # Keep the forward's hidden reduction order. Compute all row gradients
    # together so the scale gradient needs one reduction over the batch.
    inv_rms = jax.lax.rsqrt(_sum_hidden_tiles(x32)[..., None] / width + eps)
    normalized = x32 * inv_rms
    mean_x_dy = jnp.sum(x32 * weighted_dy, axis=-1, keepdims=True) / width
    dx = (weighted_dy - normalized * mean_x_dy * inv_rms) * inv_rms
    dw = jnp.sum(dy32 * normalized, axis=tuple(range(x.ndim - 1)))
    return dx.astype(x.dtype), dw.astype(weight.dtype)


def _batch_invariant_rms_norm_bwd(eps, implementation, residual, dy):
    x, weight = residual
    x_sharding = sharding_of(x)
    dy_aligned = _reshard_to_nonempty_named_mesh(dy, x_sharding)
    if implementation == "pallas_tpu":
        dx, dw = _parallel_gradients(x, weight, dy_aligned, eps)
        return _reshard_to_nonempty_named_mesh(dx, x_sharding), _reshard_to_nonempty_named_mesh(
            dw, sharding_of(weight)
        )

    def reference(value, scale):
        output = _reference(value, scale, eps=eps)
        return _reshard_to_nonempty_named_mesh(output, x_sharding)

    _, pullback = jax.vjp(reference, x, weight)
    dx, dw = pullback(dy_aligned)
    return dx.astype(x.dtype), dw.astype(weight.dtype)


batch_invariant_rms_norm.defvjp(_batch_invariant_rms_norm_fwd, _batch_invariant_rms_norm_bwd)
