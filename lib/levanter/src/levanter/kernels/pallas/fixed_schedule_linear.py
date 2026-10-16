# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""TPUv4 fixed-schedule dense projection used by the opt-in Snowball path."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
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


def _validate(x: jax.Array, w: jax.Array) -> tuple[int, int, int]:
    if jax.default_backend() != "tpu":
        raise FixedScheduleLinearUnsupportedError("fixed-schedule projections require the TPU backend")
    device_kind = jax.devices()[0].device_kind.lower()
    if "tpu v4" not in device_kind:
        raise FixedScheduleLinearUnsupportedError(
            "fixed-schedule projections require a TPU v4 device; got " + jax.devices()[0].device_kind
        )
    if jax.device_count() != 1 or jax.local_device_count() != 1:
        raise FixedScheduleLinearUnsupportedError(
            "fixed-schedule projections currently require one global and local visible TPU device; "
            f"got global={jax.device_count()}, local={jax.local_device_count()}"
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
    if not isinstance(x, jax.core.Tracer) and not isinstance(w, jax.core.Tracer):
        for name, value in (("x", x), ("w", w)):
            sharding = value.sharding
            if not sharding.is_fully_addressable or len(sharding.device_set) != 1:
                raise FixedScheduleLinearUnsupportedError(
                    f"{name} must be fully addressable on the single visible device"
                )
    return x.shape[0], x.shape[1], w.shape[1]


def _pallas_forward(x: jax.Array, w: jax.Array) -> jax.Array:
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


def _fixed_schedule_linear_bwd(residual: tuple[jax.Array, jax.Array], dy: jax.Array) -> tuple[jax.Array, jax.Array]:
    x, w = residual
    x_compute = x.astype(w.dtype)
    dx = (
        jax.lax.dot_general(
            dy.astype(jnp.float32),
            w.astype(jnp.float32),
            (((1,), (1,)), ((), ())),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        .astype(w.dtype)
        .astype(x.dtype)
    )
    dw = jax.lax.dot_general(
        x_compute.astype(jnp.float32),
        dy.astype(jnp.float32),
        (((0,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    ).astype(w.dtype)
    return dx, dw


fixed_schedule_linear.defvjp(_fixed_schedule_linear_fwd, _fixed_schedule_linear_bwd)
