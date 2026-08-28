# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""cuDNN Frontend grouped weight gradients through the CuTeDSL JAX bridge."""

import functools
import importlib
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call

# Row count each expert group is padded up to, so every group starts where the kernel's
# tile loads expect it to.
_GROUP_ALIGNMENT = 8
# Vector width the feature (non-grouped) dimensions must divide, which is what the tensor
# specs below declare to the kernel.
_FEATURE_ALIGNMENT = 8
_MMA_TILER_MN = (256, 256)
_CLUSTER_SHAPE_MN = (2, 1)


class _CudnnModules(NamedTuple):
    """CuTeDSL and cuDNN Frontend entry points, imported only where the GPU extra exists."""

    cutlass: Any
    cute: Any
    cutlass_jax: Any
    kernel_type: Any
    weight_mode: Any
    input_order: Any


@functools.cache
def _cudnn_modules() -> _CudnnModules:
    kernel_module = importlib.import_module("cudnn.gemm.cutedsl.grouped.wgrad.moe_grouped_gemm_wgrad")
    utility_module = importlib.import_module("cudnn.gemm.cutedsl.grouped.moe_utils")
    return _CudnnModules(
        cutlass=importlib.import_module("cutlass"),
        cute=importlib.import_module("cutlass.cute"),
        cutlass_jax=importlib.import_module("cutlass.jax"),
        kernel_type=kernel_module.MoEGroupedGemmWgradBF16Kernel,
        weight_mode=utility_module.MoEWeightMode,
        input_order=utility_module.WGradInputOrder,
    )


@cute_launcher_factory
def _build_launcher(
    modules,
    *,
    expert_count: int,
    max_active_clusters: int,
):
    @modules.cute.jit
    def launcher(stream, mat_a, mat_b, offsets, output, workspace):
        kernel = modules.kernel_type(
            acc_dtype=modules.cutlass.Float32,
            use_2cta_instrs=True,
            mma_tiler_mn=_MMA_TILER_MN,
            cluster_shape_mn=_CLUSTER_SHAPE_MN,
            accumulate_on_output=False,
            expert_cnt=expert_count,
            weight_mode=modules.weight_mode.DENSE,
            input_order=modules.input_order.Tensor2D,
        )
        kernel(mat_a, mat_b, output, offsets, workspace, max_active_clusters, stream, None)

    return launcher


def pad_grouped_rows(values: jax.Array, group_sizes: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Insert zero rows so each contiguous expert group ends on an eight-row boundary."""
    if values.ndim != 2:
        raise ValueError(f"values must be rank 2, got shape={values.shape}")
    if values.shape[0] == 0:
        raise ValueError("values must contain at least one physical row")
    if group_sizes.ndim != 1 or group_sizes.shape[0] == 0:
        raise ValueError(f"group_sizes must be a nonempty vector, got shape={group_sizes.shape}")

    group_sizes = group_sizes.astype(jnp.int32)
    padded_group_sizes = ((group_sizes + _GROUP_ALIGNMENT - 1) // _GROUP_ALIGNMENT) * _GROUP_ALIGNMENT
    active_offsets = jnp.cumsum(group_sizes, dtype=jnp.int32)
    padded_offsets = jnp.cumsum(padded_group_sizes, dtype=jnp.int32)
    active_starts = jnp.concatenate([jnp.zeros((1,), jnp.int32), active_offsets[:-1]])
    padded_starts = jnp.concatenate([jnp.zeros((1,), jnp.int32), padded_offsets[:-1]])

    # At most seven rows are inserted per group. Keep a static worst-case tail so the output shape
    # does not depend on the runtime routing counts.
    padded_capacity = values.shape[0] + (_GROUP_ALIGNMENT - 1) * group_sizes.shape[0]
    padded_rows = jnp.arange(padded_capacity, dtype=jnp.int32)
    expert_ids = jnp.sum(padded_rows[:, None] >= padded_offsets[None, :], axis=1, dtype=jnp.int32)
    safe_expert_ids = jnp.minimum(expert_ids, group_sizes.shape[0] - 1)
    rows_within_group = padded_rows - padded_starts[safe_expert_ids]
    source_rows = active_starts[safe_expert_ids] + rows_within_group
    valid = (expert_ids < group_sizes.shape[0]) & (rows_within_group < group_sizes[safe_expert_ids])
    source_rows = jnp.clip(source_rows, 0, values.shape[0] - 1)
    padded = jnp.where(valid[:, None], values[source_rows], jnp.zeros((), dtype=values.dtype))
    return padded, padded_offsets


def cudnn_grouped_wgrad(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    """Compute per-expert ``lhs.T @ rhs`` for contiguous ragged row groups."""
    if lhs.ndim != 2 or rhs.ndim != 2:
        raise ValueError(f"lhs and rhs must be rank 2, got lhs={lhs.shape}, rhs={rhs.shape}")
    if lhs.shape[0] != rhs.shape[0]:
        raise ValueError(f"lhs and rhs row counts must match, got lhs={lhs.shape}, rhs={rhs.shape}")
    if lhs.dtype != jnp.bfloat16 or rhs.dtype != jnp.bfloat16:
        raise ValueError(f"cuDNN grouped Wgrad requires BF16 inputs, got lhs={lhs.dtype}, rhs={rhs.dtype}")
    if lhs.shape[1] % _FEATURE_ALIGNMENT != 0 or rhs.shape[1] % _FEATURE_ALIGNMENT != 0:
        raise ValueError(
            f"cuDNN grouped Wgrad feature dimensions must divide {_FEATURE_ALIGNMENT}, "
            f"got lhs={lhs.shape}, rhs={rhs.shape}"
        )

    lhs_padded, padded_offsets = pad_grouped_rows(lhs, group_sizes)
    rhs_padded, _ = pad_grouped_rows(rhs, group_sizes)
    modules = _cudnn_modules()
    max_active_clusters = modules.cutlass.utils.HardwareInfo().get_max_active_clusters(
        _CLUSTER_SHAPE_MN[0] * _CLUSTER_SHAPE_MN[1]
    )
    launcher = _build_launcher(
        modules,
        expert_count=group_sizes.shape[0],
        max_active_clusters=max_active_clusters,
    )
    tensor_spec = modules.cutlass_jax.TensorSpec
    call = cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((group_sizes.shape[0], lhs.shape[1], rhs.shape[1]), lhs.dtype),
            jax.ShapeDtypeStruct((1,), jnp.uint8),
        ),
        input_spec=(
            tensor_spec(mode=(1, 0), divisibility=(1, _FEATURE_ALIGNMENT), static=False),
            tensor_spec(divisibility=(1, _FEATURE_ALIGNMENT), static=False),
            tensor_spec(static=False),
        ),
        output_spec=(
            tensor_spec(divisibility=(1, 1, _FEATURE_ALIGNMENT), static=False),
            tensor_spec(static=False),
        ),
        use_static_tensors=False,
    )
    output, _workspace = call(lhs_padded, rhs_padded, padded_offsets)
    return output
