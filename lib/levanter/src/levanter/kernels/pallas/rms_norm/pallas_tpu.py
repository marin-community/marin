# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU Pallas implementation of row-local RMSNorm."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jaxtyping import Array, Float

from levanter.kernels.pallas.autotune_utils import sharding_of
from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

from .reference import _sum_hidden_tiles


class PallasRmsNormUnsupportedError(NotImplementedError):
    """Raised when the TPU implementation cannot honor its contract."""


# Mosaic TPU block mappings require the penultimate row dimension to equal the
# full extent or be divisible by 8, and the minor hidden dimension to equal the
# full extent or be divisible by 128. Both fixed 128-sized tiles satisfy these
# rules; padding keeps the per-row arithmetic unchanged for every local count.
_ROW_BLOCK_SIZE = 128


def _validate_tpu_inputs(x: jax.Array) -> None:
    if x.ndim != 2:
        raise PallasRmsNormUnsupportedError(f"pallas_tpu expects rank-2 local rows, got {x.shape}.")
    if jax.default_backend() != "tpu":
        raise PallasRmsNormUnsupportedError("rms_norm pallas_tpu requires a TPU backend.")


def _rms_norm_cost_reference(x: jax.Array, weight: jax.Array, *, eps: float) -> jax.Array:
    x32 = x.astype(jnp.float32)
    weight32 = weight.astype(jnp.float32)
    tile_sums = _sum_hidden_tiles(x32)
    mean_square = tile_sums[..., None] / x32.shape[-1]
    return (x32 * jax.lax.rsqrt(mean_square + eps) * weight32).astype(x.dtype)


def _cost_estimate(x: jax.Array, weight: jax.Array, *, eps: float, out_shape) -> pl.CostEstimate | None:
    return with_io_bytes_accessed(
        pl.estimate_cost(_rms_norm_cost_reference, x, weight, eps=eps),
        kernel_inputs_specs=(x, weight),
        kernel_outputs_specs=(out_shape,),
    )


def _rms_norm_kernel(x_ref, weight_ref, out_ref, *, eps: float):
    x32 = x_ref[...].astype(jnp.float32)
    weight32 = weight_ref[...].astype(jnp.float32)
    tile_sums = _sum_hidden_tiles(x32)
    mean_square = tile_sums[..., None] / x32.shape[1]
    normalized = x32 * jax.lax.rsqrt(mean_square + eps)
    out_ref[...] = (normalized * weight32).astype(out_ref.dtype)


def _local_pallas(x: Float[Array, "M D"], weight: Float[Array, "D"], *, eps: float) -> jax.Array:
    rows, hidden = x.shape
    padded_rows = math.ceil(rows / _ROW_BLOCK_SIZE) * _ROW_BLOCK_SIZE
    if padded_rows != rows:
        padded_x = jnp.pad(x, ((0, padded_rows - rows), (0, 0)))
    else:
        padded_x = x
    out_shape = jax.ShapeDtypeStruct(padded_x.shape, padded_x.dtype)
    padded_output = pl.pallas_call(
        partial(_rms_norm_kernel, eps=eps),
        out_shape=out_shape,
        in_specs=(
            pl.BlockSpec(((_ROW_BLOCK_SIZE), hidden), lambda row: (row, 0), memory_space=pltpu.VMEM),
            pl.BlockSpec((hidden,), lambda row: (0,), memory_space=pltpu.VMEM),
        ),
        out_specs=pl.BlockSpec(((_ROW_BLOCK_SIZE), hidden), lambda row: (row, 0), memory_space=pltpu.VMEM),
        grid=(padded_rows // _ROW_BLOCK_SIZE,),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        cost_estimate=_cost_estimate(padded_x, weight, eps=eps, out_shape=out_shape),
    )(padded_x, weight)
    return padded_output[:rows]


def _normalize_partition_spec(spec: jax.sharding.PartitionSpec, rank: int) -> jax.sharding.PartitionSpec:
    if len(spec) > rank:
        raise PallasRmsNormUnsupportedError(f"partition spec rank {len(spec)} exceeds operand rank {rank}: {spec}.")
    return jax.sharding.PartitionSpec(*spec, *(None for _ in range(rank - len(spec))))


def _mesh_descriptor(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> tuple:
    return (
        tuple(mesh.axis_names),
        tuple((axis, int(mesh.shape[axis])) for axis in mesh.axis_names),
    )


def _meshes_match(
    left: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    right: jax.sharding.Mesh | jax.sharding.AbstractMesh,
) -> bool:
    """Compare concrete topology when available, logical shape while tracing.

    Tracers expose an ``AbstractMesh`` without devices, so a concrete mesh
    captured by a jit closure must be compared by its logical descriptor. Two
    concrete meshes, however, must agree on the actual ordered device
    topology; axis names and sizes alone are not enough to prove shard_map
    locality.
    """
    if isinstance(left, jax.sharding.Mesh) and isinstance(right, jax.sharding.Mesh):
        return left == right
    return _mesh_descriptor(left) == _mesh_descriptor(right)


def _mesh_size(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> int:
    return math.prod(int(mesh.shape[axis]) for axis in mesh.axis_names)


def _sharded_local_pallas(x: Float[Array, "M D"], weight: Float[Array, "D"], *, eps: float) -> jax.Array:
    """Run only explicitly proven local or row-sharded arrays.

    Both arguments must carry explicit NamedSharding. A one-device named mesh
    may be replicated after JAX tracing; a multi-device mesh must shard rows.
    """
    x_sharding = sharding_of(x)
    weight_sharding = sharding_of(weight)
    if not isinstance(x_sharding, jax.sharding.NamedSharding) or not isinstance(
        weight_sharding, jax.sharding.NamedSharding
    ):
        raise PallasRmsNormUnsupportedError(
            "pallas_tpu requires explicit NamedSharding inputs; SingleDeviceSharding is unsupported."
        )
    if not _meshes_match(x_sharding.mesh, weight_sharding.mesh):
        raise PallasRmsNormUnsupportedError("x and weight must use the same mesh.")
    x_spec = _normalize_partition_spec(x_sharding.spec, x.ndim)
    weight_spec = _normalize_partition_spec(weight_sharding.spec, weight.ndim)
    if x_spec[0] is None or x_spec[1] is not None or any(axis is not None for axis in weight_spec):
        if _mesh_size(x_sharding.mesh) != 1 or x_spec[1] is not None or any(axis is not None for axis in weight_spec):
            raise PallasRmsNormUnsupportedError("only row-sharded x and replicated weight are supported.")
    body: Callable[[jax.Array, jax.Array], jax.Array] = lambda local_x, local_weight: _local_pallas(
        local_x, local_weight, eps=eps
    )
    return jax.shard_map(
        body,
        mesh=x_sharding.mesh,
        in_specs=(x_spec, weight_spec),
        out_specs=x_spec,
        check_vma=False,
    )(x, weight)


def batch_invariant_rms_norm_pallas_tpu(x: Float[Array, "M D"], weight: Float[Array, "D"], *, eps: float) -> jax.Array:
    """Apply the row-local TPU kernel; callers normalize leading axes first."""
    _validate_tpu_inputs(x)
    return _sharded_local_pallas(x, weight, eps=eps)
