# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Public dispatch and custom backward for the RMSNorm prototype."""

from __future__ import annotations

from functools import partial
from types import MappingProxyType
from typing import Literal, TypeAlias

import jax
from jaxtyping import Array, Float
from jax.sharding import NamedSharding

from levanter.kernels.pallas.autotune_utils import sharding_of

from .pallas_tpu import batch_invariant_rms_norm_pallas_tpu
from .reference import _reference, _validate_inputs

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
    mesh, shard activations by row and replicate the weight.
    """
    return _forward(x, weight, eps=eps, implementation=implementation)


def _batch_invariant_rms_norm_fwd(x, weight, eps, implementation):
    y = _forward(x, weight, eps=eps, implementation=implementation)
    return y, (x, weight)


def _batch_invariant_rms_norm_bwd(eps, implementation, residual, dy):
    del implementation
    x, weight = residual
    x_sharding = sharding_of(x)
    dy_aligned = _reshard_to_nonempty_named_mesh(dy, x_sharding)

    def reference(value, scale):
        output = _reference(value, scale, eps=eps)
        return _reshard_to_nonempty_named_mesh(output, x_sharding)

    _, pullback = jax.vjp(reference, x, weight)
    dx, dw = pullback(dy_aligned)
    return dx.astype(x.dtype), dw.astype(weight.dtype)


batch_invariant_rms_norm.defvjp(_batch_invariant_rms_norm_fwd, _batch_invariant_rms_norm_bwd)
