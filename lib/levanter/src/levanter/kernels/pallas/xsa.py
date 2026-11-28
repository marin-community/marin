# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded, batch-invariant exclusive self-attention projection."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import math
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxtyping import Array, Float

from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

Implementation: TypeAlias = Literal["reference", "pallas_tpu"]

_HEAD_WIDTH = 128
_ROW_BLOCK_SIZE = 128
_MAX_FLATTENED_ROWS = 2048
_EPSILON = 1e-6
_SUPPORTED_MESH = {"replica_dcn": 1, "data": 4, "context": 1, "expert": 1, "model": 1}


class XsaUnsupportedError(NotImplementedError):
    """Raised when the bounded TPU XSA implementation cannot honor its contract."""


def _validate_shape(y: jax.Array, v: jax.Array) -> tuple[int, int]:
    if y.ndim < 2 or v.shape != y.shape:
        raise ValueError(f"y and v must have the same shape (..., heads, width), got {y.shape} and {v.shape}")
    if y.shape[-1] < 1:
        raise ValueError(f"XSA requires a non-empty head width, got {y.shape[-1]}")
    rows = math.prod(y.shape[:-1])
    if not 1 <= rows <= _MAX_FLATTENED_ROWS:
        raise ValueError(f"XSA supports 1..{_MAX_FLATTENED_ROWS} flattened rows, got {rows}")
    if y.dtype not in (jnp.float32, jnp.bfloat16) or v.dtype not in (jnp.float32, jnp.bfloat16):
        raise TypeError(f"XSA supports float32/bfloat16 storage, got y={y.dtype}, v={v.dtype}")
    return rows, y.shape[-1]


def _xsa_reference_flat(y: jax.Array, v: jax.Array) -> jax.Array:
    """Reference arithmetic: one float32 dot and norm reduction per flattened row."""
    y32 = y.astype(jnp.float32)
    v32 = v.astype(jnp.float32)
    dot = jnp.sum(y32 * v32, axis=-1, keepdims=True)
    norm_sq = jnp.sum(v32 * v32, axis=-1, keepdims=True)
    return (y32 - (dot / (norm_sq + _EPSILON)) * v32).astype(y.dtype)


def xsa_reference(y: Float[Array, "... H 128"], v: Float[Array, "... H 128"]) -> jax.Array:
    """Apply XSA using the obvious row-local JAX reference.

    Reference arithmetic accepts arbitrary positive head widths so portable model tests can
    exercise small configurations; the bounded TPU implementation below is fixed at width 128.
    """
    _validate_shape(y, v)
    shape = y.shape
    return _xsa_reference_flat(y.reshape((-1, shape[-1])), v.reshape((-1, shape[-1]))).reshape(shape)


def _xsa_cost_reference(y: jax.Array, v: jax.Array) -> jax.Array:
    return _xsa_reference_flat(y, v)


def _xsa_kernel(y_ref, v_ref, out_ref):
    y32 = y_ref[...].astype(jnp.float32)
    v32 = v_ref[...].astype(jnp.float32)
    dot = jnp.sum(y32 * v32, axis=1, keepdims=True)
    norm_sq = jnp.sum(v32 * v32, axis=1, keepdims=True)
    out_ref[...] = (y32 - (dot / (norm_sq + _EPSILON)) * v32).astype(out_ref.dtype)


def _array_sharding(value: jax.Array):
    return jax.typeof(value).sharding if isinstance(value, jax.core.Tracer) else value.sharding


def _restore_layout(value: jax.Array, original: jax.Array) -> jax.Array:
    sharding = _array_sharding(original)
    if isinstance(sharding, NamedSharding) and not sharding.mesh.empty:
        return jax.sharding.reshard(value, sharding)
    return value


def _normalize_spec(spec: P, rank: int) -> P:
    if len(spec) > rank:
        raise XsaUnsupportedError(f"partition spec rank {len(spec)} exceeds rank {rank}: {spec}")
    return P(*spec, *(None for _ in range(rank - len(spec))))


def _canonicalize_spec(spec: P, mesh) -> P:
    """Remove singleton mesh axes while retaining the physical layout contract."""
    entries = []
    for entry in spec:
        axes = (entry,) if isinstance(entry, str) else tuple(axis for axis in entry or () if axis is not None)
        non_singleton = tuple(axis for axis in axes if int(mesh.shape[axis]) > 1)
        entries.append(None if not non_singleton else non_singleton[0] if len(non_singleton) == 1 else non_singleton)
    return P(*entries)


def _mesh_descriptor(mesh) -> tuple:
    return tuple(mesh.axis_names), tuple((axis, int(mesh.shape[axis])) for axis in mesh.axis_names)


def _meshes_match(left, right) -> bool:
    if isinstance(left, jax.sharding.Mesh) and isinstance(right, jax.sharding.Mesh):
        return left == right
    return _mesh_descriptor(left) == _mesh_descriptor(right)


def _local_pallas(y: jax.Array, v: jax.Array) -> jax.Array:
    rows, width = y.shape
    padded_rows = math.ceil(rows / _ROW_BLOCK_SIZE) * _ROW_BLOCK_SIZE
    padded_y = jnp.pad(y, ((0, padded_rows - rows), (0, 0)))
    padded_v = jnp.pad(v, ((0, padded_rows - rows), (0, 0)))
    output_shape = jax.ShapeDtypeStruct(padded_y.shape, padded_y.dtype)
    cost = with_io_bytes_accessed(
        pl.estimate_cost(_xsa_cost_reference, padded_y, padded_v),
        kernel_inputs_specs=(padded_y, padded_v),
        kernel_outputs_specs=(output_shape,),
    )
    output = pl.pallas_call(
        _xsa_kernel,
        out_shape=output_shape,
        in_specs=(
            pl.BlockSpec((_ROW_BLOCK_SIZE, width), lambda row: (row, 0), memory_space=pltpu.VMEM),
            pl.BlockSpec((_ROW_BLOCK_SIZE, width), lambda row: (row, 0), memory_space=pltpu.VMEM),
        ),
        out_specs=pl.BlockSpec((_ROW_BLOCK_SIZE, width), lambda row: (row, 0), memory_space=pltpu.VMEM),
        grid=(padded_rows // _ROW_BLOCK_SIZE,),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        cost_estimate=cost,
    )(padded_y, padded_v)
    return output[:rows]


def _validate_physical_layout(y: jax.Array, v: jax.Array) -> P | None:
    """Validate the unflattened activation layout before it can hide invalid partitions."""
    y_sharding = _array_sharding(y)
    v_sharding = _array_sharding(v)
    if not isinstance(y_sharding, NamedSharding) or not isinstance(v_sharding, NamedSharding):
        return None
    if not _meshes_match(y_sharding.mesh, v_sharding.mesh):
        raise XsaUnsupportedError("y and v must use the same mesh")
    if dict(y_sharding.mesh.shape) != _SUPPORTED_MESH:
        raise XsaUnsupportedError("XSA requires the bounded data4/context1/expert1/model1 mesh")
    y_spec = _canonicalize_spec(_normalize_spec(y_sharding.spec, y.ndim), y_sharding.mesh)
    v_spec = _canonicalize_spec(_normalize_spec(v_sharding.spec, v.ndim), v_sharding.mesh)
    if y_spec != v_spec:
        raise XsaUnsupportedError("y and v must use the same canonical physical layout")
    if y_spec[-1] is not None or (y.ndim > 2 and y_spec[-2] is not None):
        raise XsaUnsupportedError("XSA cannot partition head features or genuine head axes")
    if any(entry is not None for entry in y_spec[1:-2]):
        raise XsaUnsupportedError("XSA supports only the leading physical row dimension")
    return y_spec


def xsa_pallas_tpu(y: Float[Array, "M 128"], v: Float[Array, "M 128"]) -> jax.Array:
    """Run XSA on explicitly replicated or row-sharded TPU inputs."""
    rows, width = _validate_shape(y, v)
    if width != _HEAD_WIDTH:
        raise XsaUnsupportedError(f"pallas_tpu requires head width {_HEAD_WIDTH}, got {width}")
    if y.ndim != 2 or rows != y.shape[0] or jax.default_backend() != "tpu":
        raise XsaUnsupportedError("pallas_tpu requires rank-2 rows on a TPU backend")
    spec = _validate_physical_layout(y, v)
    if spec is None:
        raise XsaUnsupportedError("pallas_tpu requires explicit NamedSharding inputs")
    body: Callable[[jax.Array, jax.Array], jax.Array] = _local_pallas
    output = jax.shard_map(body, mesh=_array_sharding(y).mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False)(
        y, v
    )
    return _restore_layout(output, y)


def _forward(y: jax.Array, v: jax.Array, implementation: Implementation) -> jax.Array:
    _, width = _validate_shape(y, v)
    if implementation == "reference":
        return xsa_reference(y, v)
    if implementation == "pallas_tpu":
        if width != _HEAD_WIDTH:
            raise XsaUnsupportedError(f"pallas_tpu requires head width {_HEAD_WIDTH}, got {width}")
        spec = _validate_physical_layout(y, v)
        flat_y = y.reshape((-1, width))
        flat_v = v.reshape((-1, width))
        if spec is not None:
            flat_y = jax.sharding.reshard(flat_y, P(spec[0], None))
            flat_v = jax.sharding.reshard(flat_v, P(spec[0], None))
        output = xsa_pallas_tpu(flat_y, flat_v).reshape(y.shape)
        return _restore_layout(output, y)
    raise ValueError(f"unsupported XSA implementation: {implementation!r}")


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def xsa_forward(
    y: Float[Array, "... H 128"], v: Float[Array, "... H 128"], implementation: Implementation = "reference"
) -> jax.Array:
    """Project each attention row orthogonally away from its aligned value row."""
    return _forward(y, v, implementation)


def _xsa_fwd(y, v, implementation):
    return _forward(y, v, implementation), (y, v)


def _xsa_bwd(implementation, residual, dy):
    del implementation
    y, v = residual

    def reference(value, aligned_value):
        return _restore_layout(xsa_reference(value, _restore_layout(aligned_value, value)), y)

    _, pullback = jax.vjp(reference, y, v)
    dy, dv = pullback(_restore_layout(dy, y))
    return _restore_layout(dy, y).astype(y.dtype), _restore_layout(dv, v).astype(v.dtype)


xsa_forward.defvjp(_xsa_fwd, _xsa_bwd)
