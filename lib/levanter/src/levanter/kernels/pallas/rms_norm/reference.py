# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX reference for batch-invariant weighted RMSNorm."""

from functools import partial
import math

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P, reshard
from jaxtyping import Array, Float

from levanter.kernels.pallas.autotune_utils import named_sharding_of

_HIDDEN_TILE_SIZE = 128
_ROW_TILE_SIZE = 512


def _sum_hidden_tiles(x32: jax.Array) -> jax.Array:
    """Sum squares in fixed-width hidden tiles without changing the row layout."""
    tile_sums = jnp.zeros(x32.shape[:-1], dtype=jnp.float32)
    for start in range(0, x32.shape[-1], _HIDDEN_TILE_SIZE):
        tile = x32[..., start : start + _HIDDEN_TILE_SIZE]
        tile_sums = tile_sums + jnp.sum(jnp.square(tile), axis=-1)
    return tile_sums


def _validate_inputs(x: jax.Array, weight: jax.Array, eps: float) -> None:
    if x.ndim < 1:
        raise ValueError(f"x must have at least one dimension, got shape {x.shape}.")
    hidden = x.shape[-1]
    if hidden == 0 or hidden % _HIDDEN_TILE_SIZE:
        raise ValueError(f"RMSNorm hidden width must be a positive multiple of {_HIDDEN_TILE_SIZE}, got {hidden}.")
    if weight.shape != (hidden,):
        raise ValueError(f"weight must have shape ({hidden},), got {weight.shape}.")
    if 0 in x.shape[:-1]:
        raise ValueError("Batch-invariant RMSNorm requires at least one row.")
    if x.dtype not in (jnp.float32, jnp.bfloat16) or weight.dtype not in (jnp.float32, jnp.bfloat16):
        raise TypeError(f"Batch-invariant RMSNorm supports float32/bfloat16, got x={x.dtype}, weight={weight.dtype}.")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be a positive finite scalar, got {eps}.")


def _reference_tile(x: jax.Array, weight: jax.Array, *, eps: float) -> jax.Array:
    x32 = x.astype(jnp.float32)
    mean_square = _sum_hidden_tiles(x32)[..., None] / x.shape[-1]
    return (x32 * jax.lax.rsqrt(mean_square + eps) * weight).astype(x.dtype)


def _reference_local(x: jax.Array, weight: jax.Array, *, eps: float) -> jax.Array:
    hidden = x.shape[-1]
    rows = math.prod(x.shape[:-1])
    flat = x.reshape(rows, hidden)
    padded = jnp.pad(flat, ((0, -rows % _ROW_TILE_SIZE), (0, 0)))
    weight = weight.astype(jnp.float32)
    tile = partial(_reference_tile, weight=weight, eps=eps)
    if padded.shape[0] == _ROW_TILE_SIZE:
        return tile(padded)[:rows].reshape(x.shape)
    output = jax.lax.map(tile, padded.reshape(-1, _ROW_TILE_SIZE, hidden))
    return output.reshape(-1, hidden)[:rows].reshape(x.shape)


def _reference(x: jax.Array, weight: jax.Array, *, eps: float) -> jax.Array:
    sharding = named_sharding_of(x)
    if sharding is None or sharding.mesh.empty or AxisType.Auto in sharding.mesh.axis_types:
        return _reference_local(x, weight, eps=eps)

    # Tile within each shard so batches whose local row count is smaller than
    # 512 do not gather other shards to fill a reference tile.
    spec = tuple(sharding.spec) + (None,) * (x.ndim - len(sharding.spec))
    local_sharding = NamedSharding(sharding.mesh, P(*spec[:-1], None))
    local_weight = reshard(weight, NamedSharding(sharding.mesh, P()))
    output = jax.shard_map(
        partial(_reference_local, eps=eps),
        mesh=sharding.mesh,
        in_specs=(local_sharding.spec, P()),
        out_specs=local_sharding.spec,
    )(reshard(x, local_sharding), local_weight)
    return reshard(output, sharding)


def batch_invariant_rms_norm_reference(
    x: Float[Array, "... D"],
    weight: Float[Array, "D"],
    *,
    eps: float,
) -> jax.Array:
    """Apply weighted RMSNorm with a fixed per-row float32 reduction order."""
    _validate_inputs(x, weight, eps)
    return _reference(x, weight, eps=eps)
