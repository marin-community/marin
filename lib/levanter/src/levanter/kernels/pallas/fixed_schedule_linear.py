# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""TPUv4 fixed-schedule dense projection used by the opt-in Snowball path."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxtyping import Array, Float

from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

_BLOCK_SIZE = 128


class FixedScheduleLinearUnsupportedError(NotImplementedError):
    """The requested fixed-schedule projection is outside its supported contract."""


def _cost_reference(x: jax.Array, w: jax.Array) -> jax.Array:
    return jax.lax.dot_general(
        x,
        w,
        (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
    )


def _make_fixed_schedule_kernel(k_steps: int):
    def fixed_schedule_kernel(x_ref, w_ref, out_ref, acc_ref):
        k_index = pl.program_id(2)

        @pl.when(k_index == 0)
        def initialize() -> None:
            acc_ref[...] = jnp.zeros_like(acc_ref)

        acc_ref[...] += jax.lax.dot_general(
            x_ref[...],
            w_ref[...],
            (((1,), (0,)), ((), ())),
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )

        @pl.when(k_index == k_steps - 1)
        def finalize() -> None:
            out_ref[...] = acc_ref[...].astype(out_ref.dtype)

    return fixed_schedule_kernel


def _array_sharding(value: jax.Array):
    if isinstance(value, jax.core.Tracer):
        return jax.typeof(value).sharding
    return value.sharding


def _normalize_partition_spec(spec: P, rank: int) -> P:
    if len(spec) > rank:
        raise FixedScheduleLinearUnsupportedError(
            f"partition spec rank {len(spec)} exceeds operand rank {rank}: {spec}"
        )
    return P(*spec, *(None for _ in range(rank - len(spec))))


def _mesh_descriptor(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> tuple:
    return tuple(mesh.axis_names), tuple((axis, int(mesh.shape[axis])) for axis in mesh.axis_names)


def _meshes_match(left, right) -> bool:
    if isinstance(left, jax.sharding.Mesh) and isinstance(right, jax.sharding.Mesh):
        return left == right
    return _mesh_descriptor(left) == _mesh_descriptor(right)


def _partition_axes(entry: object) -> tuple[str, ...]:
    if entry is None:
        return ()
    if isinstance(entry, str):
        return (entry,)
    if isinstance(entry, tuple):
        if any(axis is not None and not isinstance(axis, str) for axis in entry):
            raise FixedScheduleLinearUnsupportedError(f"partition spec contains a non-axis entry: {entry}")
        return tuple(axis for axis in entry if axis is not None)
    raise FixedScheduleLinearUnsupportedError(f"partition spec contains an invalid entry: {entry}")


def _canonicalize_partition_spec(spec: P, mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> P:
    """Remove only known singleton mesh axes from a spec used by the kernel."""
    canonical_entries = []
    mesh_axes = set(mesh.axis_names)
    for entry in spec:
        axes = _partition_axes(entry)
        unknown_axes = tuple(axis for axis in axes if axis not in mesh_axes)
        if unknown_axes:
            raise FixedScheduleLinearUnsupportedError(f"partition spec names unknown mesh axes: {unknown_axes}")
        effective_axes = tuple(axis for axis in axes if int(mesh.shape[axis]) > 1)
        if not effective_axes:
            canonical_entries.append(None)
        elif len(effective_axes) == 1:
            canonical_entries.append(effective_axes[0])
        else:
            canonical_entries.append(effective_axes)
    return P(*canonical_entries)


def _validate_four_device_mesh(mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh) -> None:
    if jax.device_count() != 4 or jax.local_device_count() != 4:
        raise FixedScheduleLinearUnsupportedError(
            "multi-device fixed-schedule projections require exactly four same-process TPUv4 devices; "
            f"got global={jax.device_count()}, local={jax.local_device_count()}"
        )
    devices = jax.devices()
    if any("tpu v4" not in device.device_kind.lower() for device in devices):
        raise FixedScheduleLinearUnsupportedError("the four-device fixed-schedule mesh requires TPUv4 devices")
    if tuple(size for size in mesh.shape.values() if size > 1) != (4,):
        raise FixedScheduleLinearUnsupportedError(
            "the four-device fixed-schedule mesh must have a single axis spanning all four devices"
        )
    axis_types = tuple(mesh.axis_types)
    if axis_types and any(str(axis_type).lower().endswith("explicit") is False for axis_type in axis_types):
        raise FixedScheduleLinearUnsupportedError("the four-device fixed-schedule mesh must use Explicit axes")
    if isinstance(mesh, jax.sharding.Mesh):
        mesh_devices = tuple(mesh.devices.flat)
        if len(mesh_devices) != 4 or set(mesh_devices) != set(devices):
            raise FixedScheduleLinearUnsupportedError("the four-device mesh must contain all four local TPUv4 devices")


def _validate_named_sharding(x: jax.Array, w: jax.Array) -> tuple[NamedSharding, P, P, P, P] | None:
    x_sharding = _array_sharding(x)
    w_sharding = _array_sharding(w)
    x_named = isinstance(x_sharding, NamedSharding)
    w_named = isinstance(w_sharding, NamedSharding)
    if not x_named or not w_named:
        if x_named != w_named:
            raise FixedScheduleLinearUnsupportedError("x and w must both use explicit NamedSharding or neither")
        if jax.device_count() != 1 or jax.local_device_count() != 1:
            raise FixedScheduleLinearUnsupportedError(
                "multi-device fixed-schedule projections require explicit NamedSharding for x and w"
            )
        return None

    assert isinstance(x_sharding, NamedSharding)
    assert isinstance(w_sharding, NamedSharding)
    if not _meshes_match(x_sharding.mesh, w_sharding.mesh):
        raise FixedScheduleLinearUnsupportedError("x and w must use the same explicit mesh")
    x_public_spec = _normalize_partition_spec(x_sharding.spec, x.ndim)
    w_public_spec = _normalize_partition_spec(w_sharding.spec, w.ndim)
    x_spec = _canonicalize_partition_spec(x_public_spec, x_sharding.mesh)
    w_spec = _canonicalize_partition_spec(w_public_spec, x_sharding.mesh)
    if x_sharding.mesh.empty or w_sharding.mesh.empty:
        if jax.device_count() == 1 and jax.local_device_count() == 1:
            return None
        raise FixedScheduleLinearUnsupportedError("fixed-schedule NamedSharding requires a nonempty mesh")
    mesh_size = math.prod(int(x_sharding.mesh.shape[axis]) for axis in x_sharding.mesh.axis_names)
    if mesh_size == 1:
        if any(_partition_axes(entry) for entry in x_spec + w_spec):
            raise FixedScheduleLinearUnsupportedError("single-device NamedSharding must be fully replicated")
        return x_sharding, x_spec, w_spec, x_public_spec, w_public_spec
    _validate_four_device_mesh(x_sharding.mesh)
    if isinstance(x_sharding.mesh, jax.sharding.Mesh) and (
        not x_sharding.is_fully_addressable or not w_sharding.is_fully_addressable
    ):
        raise FixedScheduleLinearUnsupportedError("four-device fixed-schedule inputs must be fully addressable")
    if any(_partition_axes(entry) for entry in w_spec):
        raise FixedScheduleLinearUnsupportedError(
            "w must be fully replicated; contraction and output features cannot shard"
        )
    row_axes = _partition_axes(x_spec[0])
    if _partition_axes(x_spec[1]):
        raise FixedScheduleLinearUnsupportedError(
            "x may shard only its row dimension; contraction features cannot shard"
        )
    if any(axis not in x_sharding.mesh.axis_names for axis in row_axes):
        raise FixedScheduleLinearUnsupportedError(f"x row sharding names unknown mesh axes: {row_axes}")
    non_singleton_axes = tuple(axis for axis in x_sharding.mesh.axis_names if int(x_sharding.mesh.shape[axis]) > 1)
    if any(axis not in non_singleton_axes for axis in row_axes):
        raise FixedScheduleLinearUnsupportedError("x may not shard across singleton mesh axes")
    if len(row_axes) > 1:
        raise FixedScheduleLinearUnsupportedError("only one row data-parallel mesh axis is supported")
    row_factor = math.prod(int(x_sharding.mesh.shape[axis]) for axis in row_axes) if row_axes else 1
    if x.shape[0] % row_factor:
        raise FixedScheduleLinearUnsupportedError(
            f"row-sharded x requires M divisible by the row mesh size {row_factor}, got M={x.shape[0]}"
        )
    if row_axes and len(non_singleton_axes) != 1:
        raise FixedScheduleLinearUnsupportedError(
            "fixed-schedule supports only one non-singleton data-parallel mesh axis; TP/EP/model sharding is unsupported"
        )
    return x_sharding, x_spec, w_spec, x_public_spec, w_public_spec


def _validate(x: jax.Array, w: jax.Array) -> tuple[int, int, int]:
    if jax.default_backend() != "tpu":
        raise FixedScheduleLinearUnsupportedError("fixed-schedule projections require the TPU backend")
    device_kind = jax.devices()[0].device_kind.lower()
    if "tpu v4" not in device_kind:
        raise FixedScheduleLinearUnsupportedError(
            "fixed-schedule projections require a TPU v4 device; got " + jax.devices()[0].device_kind
        )
    if x.ndim != 2 or w.ndim != 2:
        raise FixedScheduleLinearUnsupportedError(f"expected x=[M,D], w=[D,N], got {x.shape} and {w.shape}")
    if x.shape[1] != w.shape[0]:
        raise FixedScheduleLinearUnsupportedError(f"incompatible projection shapes: {x.shape} and {w.shape}")
    supported_dtypes = (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16))
    if x.dtype not in supported_dtypes or w.dtype not in supported_dtypes:
        raise FixedScheduleLinearUnsupportedError("x and w must use float32 or bfloat16 storage")
    if x.shape[1] % _BLOCK_SIZE or w.shape[1] % _BLOCK_SIZE:
        raise FixedScheduleLinearUnsupportedError(
            f"projection dimensions must be multiples of {_BLOCK_SIZE}: {x.shape} and {w.shape}"
        )
    return x.shape[0], x.shape[1], w.shape[1]


def _pallas_forward_local(x: jax.Array, w: jax.Array) -> jax.Array:
    m, d, n = _validate(x, w)
    x_compute = x.astype(w.dtype)
    m_padded = (m + _BLOCK_SIZE - 1) // _BLOCK_SIZE * _BLOCK_SIZE
    x_padded = jnp.pad(x_compute, ((0, m_padded - m), (0, 0)))
    k_steps = d // _BLOCK_SIZE
    kernel = _make_fixed_schedule_kernel(k_steps)
    output_spec = jax.ShapeDtypeStruct((m_padded, n), x.dtype)
    input_specs = [
        pl.BlockSpec((_BLOCK_SIZE, _BLOCK_SIZE), lambda i, j, k: (i, k)),
        pl.BlockSpec((_BLOCK_SIZE, _BLOCK_SIZE), lambda i, j, k: (k, j)),
    ]
    body_cost = pl.estimate_cost(_cost_reference, x_compute, w)
    cost_estimate = with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=(x_padded, w),
        kernel_outputs_specs=output_spec,
    )
    output = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=input_specs,
            out_specs=pl.BlockSpec((_BLOCK_SIZE, _BLOCK_SIZE), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((_BLOCK_SIZE, _BLOCK_SIZE), jnp.float32)],
            grid=(m_padded // _BLOCK_SIZE, n // _BLOCK_SIZE, k_steps),
        ),
        out_shape=output_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        cost_estimate=cost_estimate,
    )(x_padded, w)
    return output[:m].astype(x.dtype)


def _pallas_forward(x: jax.Array, w: jax.Array) -> jax.Array:
    sharding = _validate_named_sharding(x, w)
    if sharding is None:
        for name, value in (("x", x), ("w", w)):
            if not isinstance(value, jax.core.Tracer) and (
                not value.sharding.is_fully_addressable or len(value.sharding.device_set) != 1
            ):
                raise FixedScheduleLinearUnsupportedError(
                    f"{name} must be fully addressable on the single visible device"
                )
        return _pallas_forward_local(x, w)

    x_sharding, _, _, x_public_spec, w_public_spec = sharding
    body: Callable[[jax.Array, jax.Array], jax.Array] = _pallas_forward_local
    local_output = jax.shard_map(
        body,
        mesh=x_sharding.mesh,
        in_specs=(x_public_spec, w_public_spec),
        out_specs=x_public_spec,
        check_vma=False,
    )(x, w)
    return jax.sharding.reshard(local_output, NamedSharding(x_sharding.mesh, x_public_spec))


@jax.custom_vjp
def fixed_schedule_linear(x: Float[Array, "M D"], w: Float[Array, "D N"]) -> Float[Array, "M N"]:
    """Apply the measured fixed BM=BK=BN=128 TPU projection schedule.

    The forward is fixed over the token dimension and uses DEFAULT products with FP32
    accumulation. Activations are cast to the weight storage dtype for the contraction and
    outputs are restored to the activation storage dtype. The custom backward uses HIGHEST-
    precision FP32 GEMMs.
    """
    return _pallas_forward(x, w)


def _fixed_schedule_linear_fwd(x: jax.Array, w: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    return _pallas_forward(x, w), (x, w)


def _backward_local(
    x: jax.Array, w: jax.Array, dy: jax.Array, row_axes: tuple[str, ...]
) -> tuple[jax.Array, jax.Array]:
    x_compute = x.astype(w.dtype)
    dx = jax.lax.dot_general(
        dy.astype(jnp.float32),
        w.astype(jnp.float32),
        (((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    # Preserve weight-dtype cotangent rounding before widening.
    dx = jax.lax.optimization_barrier(dx.astype(w.dtype)).astype(x.dtype)
    dw = jax.lax.dot_general(
        x_compute.astype(jnp.float32),
        dy.astype(jnp.float32),
        (((0,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    if row_axes:
        dw = jax.lax.psum(dw, row_axes)
    return dx, dw.astype(w.dtype)


def _fixed_schedule_linear_bwd(residual: tuple[jax.Array, jax.Array], dy: jax.Array) -> tuple[jax.Array, jax.Array]:
    x, w = residual
    sharding = _validate_named_sharding(x, w)
    if sharding is not None:
        x_sharding, x_spec, _, x_public_spec, w_public_spec = sharding
        dy_sharding = _array_sharding(dy)
        if not isinstance(dy_sharding, NamedSharding):
            raise FixedScheduleLinearUnsupportedError("dy must use explicit NamedSharding on a multi-device mesh")
        if not _meshes_match(dy_sharding.mesh, x_sharding.mesh):
            raise FixedScheduleLinearUnsupportedError("dy must use the same explicit mesh as x")
        dy_public_spec = _normalize_partition_spec(dy_sharding.spec, dy.ndim)
        dy_spec = _canonicalize_partition_spec(dy_public_spec, x_sharding.mesh)
        if dy.dtype != x.dtype or dy.shape != (x.shape[0], w.shape[1]):
            raise FixedScheduleLinearUnsupportedError(
                f"dy must have x storage dtype and output shape {(x.shape[0], w.shape[1])}, got {dy.dtype} and {dy.shape}"
            )
        if _partition_axes(dy_spec[1]) or any(
            axis not in x_sharding.mesh.axis_names for axis in _partition_axes(dy_spec[0])
        ):
            raise FixedScheduleLinearUnsupportedError("dy may shard only rows on the fixed-schedule data axis")
        dy_row_axes = _partition_axes(dy_spec[0])
        x_row_axes = _partition_axes(x_spec[0])
        if dy_row_axes not in ((), x_row_axes):
            raise FixedScheduleLinearUnsupportedError("dy must be replicated or use the same row sharding as x")
        dy_aligned = jax.sharding.reshard(dy, NamedSharding(x_sharding.mesh, x_public_spec))

        local_dx, local_dw = jax.shard_map(
            partial(_backward_local, row_axes=x_row_axes),
            mesh=x_sharding.mesh,
            in_specs=(x_public_spec, w_public_spec, x_public_spec),
            out_specs=(x_public_spec, w_public_spec),
            check_vma=False,
        )(x, w, dy_aligned)
        return (
            jax.sharding.reshard(local_dx, NamedSharding(x_sharding.mesh, x_public_spec)),
            jax.sharding.reshard(local_dw, NamedSharding(x_sharding.mesh, w_public_spec)),
        )
    return _backward_local(x, w, dy, row_axes=())


fixed_schedule_linear.defvjp(_fixed_schedule_linear_fwd, _fixed_schedule_linear_bwd)
