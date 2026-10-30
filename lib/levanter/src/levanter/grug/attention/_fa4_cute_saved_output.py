# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Native SM100 attention whose backward uses the pre-cast FP32 output.

The backward normalization term ``D = rowsum(dO * O)`` is normally formed from the
BF16 output the forward returns. When the softmax is sharply peaked, rounding O to
BF16 loses most of ``O - V_max``, and D then cancels badly against ``dO . V``
(marin-logbooks issue 12). Here the native forward writes its normalized FP32
accumulator, the custom VJP keeps that FP32 O as the residual, and D is reduced from
it in FP32. The returned forward output is the same FP32 accumulator cast to BF16.

Q/K/V, dO and the returned gradients stay BF16; only the saved O, LSE and the
backward accumulators are FP32. The backward mainloop and postprocess are the
unchanged production kernels.
"""

import math
from dataclasses import replace
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from levanter.cutlass_kernel_cache import cutlass_call
from levanter.grug.attention._fa4_cute_backend import (
    _cutlass_attention_forward_specs,
    _import_cutlass_cute,
    _native_backward_accum_output_shapes,
    _native_backward_accum_specs,
    _native_backward_gradients,
    _packed_segment_backward_block_sparse_indices_with_full,
    _packed_segment_forward_block_sparse_indices_with_full,
    _validate_forward_inputs,
)
from levanter.grug.attention._fa4_cute_config import Flash4CuteKernelConfig, Flash4CuteSm100ForwardConfig
from levanter.grug.attention._fa4_cute_kernels import (
    segmented_flash_attention_backward_sm100_launcher,
    segmented_flash_attention_forward_sm100_launcher,
)

LOG2_E = math.log2(math.e)
# Upstream FA4 SM100 forward shared-memory budget for Q, O and the KV pipeline.
SM100_FORWARD_SHARED_MEMORY_BUDGET = 224 * 1024


class _SavedOutputResiduals(NamedTuple):
    q: jax.Array
    k: jax.Array
    v: jax.Array
    output_fp32: jax.Array
    lse: jax.Array
    lower_bounds: jax.Array
    valid: jax.Array
    q_offset: jax.Array


def fp32_output_forward_config(config: Flash4CuteSm100ForwardConfig) -> Flash4CuteSm100ForwardConfig:
    """Fit an FP32 output into the forward's shared memory.

    Two 128x128 query stages with a separate FP32 output buffer leave one KV slot,
    which deadlocks the multi-key loop. Aliasing the output buffer with the query
    buffer keeps both query stages and three KV slots; one query stage is the
    fallback the logbook measured.
    """
    if config.tile != (128, 128):
        raise ValueError(f"FP32 saved output requires 128x128 forward tiles, got {config.tile}")
    if config.overlap_output_with_query:
        query_and_output_bytes = config.q_stage * config.tile[0] * 128 * 4
    else:
        query_and_output_bytes = config.q_stage * config.tile[0] * 128 * (2 + 4)
    kv_stages = (SM100_FORWARD_SHARED_MEMORY_BUDGET - query_and_output_bytes) // (config.tile[1] * 128 * 2)
    if kv_stages < 2:
        raise ValueError(f"FP32 saved-output forward leaves {kv_stages} KV pipeline slots for {config}")
    return config


def _fp32_output_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    sm_scale: float,
    config: Flash4CuteSm100ForwardConfig,
    q_offset: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    _validate_forward_inputs(q, k, v, lower_bounds, valid, softmax_scale=sm_scale, q_offset=q_offset)
    config = fp32_output_forward_config(config)
    modules = _import_cutlass_cute()
    sparse = _packed_segment_forward_block_sparse_indices_with_full(
        lower_bounds,
        valid,
        key_sequence_length=k.shape[1],
        q_offset=q_offset,
        tile_m=config.tile[0] * config.q_stage,
        tile_n=config.tile[1],
    )
    launcher = segmented_flash_attention_forward_sm100_launcher(
        modules,
        head_dim=q.shape[-1],
        head_dim_v=v.shape[-1],
        qhead_per_kvhead=q.shape[2] // k.shape[2],
        config=config,
    )
    inputs, outputs = _cutlass_attention_forward_specs(modules, vector_elems=8)
    metadata_spec = modules.cjax.TensorSpec(static=True)
    call = cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((*q.shape[:3], v.shape[-1]), jnp.float32),
            jax.ShapeDtypeStruct((q.shape[0], q.shape[2], q.shape[1]), jnp.float32),
        ),
        input_spec=(*inputs, metadata_spec, metadata_spec, metadata_spec, metadata_spec),
        output_spec=outputs,
        use_static_tensors=True,
        softmax_scale=sm_scale,
    )
    return call(
        q,
        k,
        v,
        lower_bounds,
        valid.astype(jnp.int32),
        q_offset,
        sparse.partial_block_cnt,
        sparse.partial_block_idx,
        sparse.full_block_cnt,
        sparse.full_block_idx,
    )


def _saved_output_preprocess(
    output_fp32: jax.Array, dout: jax.Array, lse: jax.Array, *, tile_m: int
) -> tuple[jax.Array, jax.Array]:
    """Return the padded native-layout D and log2 LSE, with D reduced from FP32 O in FP32."""
    d = jnp.sum(output_fp32 * dout.astype(jnp.float32), axis=-1, dtype=jnp.float32)
    d = jnp.transpose(d, (0, 2, 1))
    lse_log2 = jnp.where(jnp.isneginf(lse), jnp.float32(0), lse * jnp.float32(LOG2_E))
    pads = ((0, 0), (0, 0), (0, (-output_fp32.shape[1]) % tile_m))
    return jnp.pad(d, pads), jnp.pad(lse_log2, pads)


def _saved_output_backward(
    residuals: _SavedOutputResiduals,
    dout: jax.Array,
    *,
    sm_scale: float,
    kernel_config: Flash4CuteKernelConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    config = kernel_config.sm100_backward
    if config is None:
        raise ValueError("FP32 saved-output backward requires the native SM100 backward config")
    q, k, v = residuals.q, residuals.k, residuals.v
    modules = _import_cutlass_cute()
    sparse = _packed_segment_backward_block_sparse_indices_with_full(
        residuals.lower_bounds,
        residuals.valid,
        key_sequence_length=k.shape[1],
        q_offset=residuals.q_offset,
        tile_m=config.tile[0],
        tile_n=config.tile[1],
    )
    dpsum, lse_log2 = _saved_output_preprocess(residuals.output_fp32, dout, residuals.lse, tile_m=config.tile[0])
    inputs, outputs = _native_backward_accum_specs(modules, vector_elems=8)
    backward = cutlass_call(
        segmented_flash_attention_backward_sm100_launcher(
            modules,
            head_dim=q.shape[-1],
            head_dim_v=v.shape[-1],
            qhead_per_kvhead=q.shape[2] // k.shape[2],
            config=config,
        ),
        output_shape_dtype=_native_backward_accum_output_shapes(q, k, v, config.tile),
        input_spec=inputs,
        output_spec=outputs,
        use_static_tensors=True,
        softmax_scale=sm_scale,
    )
    accumulators = backward(
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        residuals.lower_bounds,
        residuals.valid.astype(jnp.int32),
        residuals.q_offset,
        sparse.partial_block_cnt,
        sparse.partial_block_idx,
        sparse.full_block_cnt,
        sparse.full_block_idx,
    )
    return _native_backward_gradients(
        modules,
        (q, k, v),
        accumulators,
        tile_rows=(config.tile[0], config.tile[1], config.tile[1]),
        arch=100,
        num_threads=config.postprocess_threads,
        softmax_scale=sm_scale,
    )


def _forward_config(kernel_config: Flash4CuteKernelConfig) -> Flash4CuteSm100ForwardConfig:
    config = kernel_config.sm100_forward
    if config is None or kernel_config.sm100_backward is None:
        raise ValueError("FP32 saved output requires the native SM100 forward and backward configs")
    return config


@partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _saved_output_attention(q, k, v, lower_bounds, valid, q_offset, sm_scale, kernel_config):
    output_fp32, _ = _fp32_output_forward(
        q, k, v, lower_bounds, valid, sm_scale=sm_scale, config=_forward_config(kernel_config), q_offset=q_offset
    )
    return output_fp32.astype(q.dtype)


def _saved_output_attention_fwd(q, k, v, lower_bounds, valid, q_offset, sm_scale, kernel_config):
    output_fp32, lse = _fp32_output_forward(
        q, k, v, lower_bounds, valid, sm_scale=sm_scale, config=_forward_config(kernel_config), q_offset=q_offset
    )
    residuals = _SavedOutputResiduals(q, k, v, output_fp32, lse, lower_bounds, valid, q_offset)
    return output_fp32.astype(q.dtype), residuals


def _saved_output_attention_bwd(sm_scale, kernel_config, residuals, cotangent):
    if isinstance(cotangent, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros_like(residuals.q), jnp.zeros_like(residuals.k), jnp.zeros_like(residuals.v), None, None, None
    dq, dk, dv = _saved_output_backward(
        residuals, cotangent.astype(residuals.q.dtype), sm_scale=sm_scale, kernel_config=kernel_config
    )
    return dq, dk, dv, None, None, None


_saved_output_attention.defvjp(_saved_output_attention_fwd, _saved_output_attention_bwd)


def fp32_saved_output_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    sm_scale: float,
    kernel_config: Flash4CuteKernelConfig,
    q_offset: jax.Array,
) -> jax.Array:
    """Drop-in for ``fa4_cute_attention_forward`` whose VJP keeps the FP32 output for D."""
    return _saved_output_attention(q, k, v, lower_bounds, valid, q_offset, sm_scale, kernel_config)


def fp32_saved_output_kernel_config(kernel_config: Flash4CuteKernelConfig) -> Flash4CuteKernelConfig:
    """Return ``kernel_config`` with the FP32 saved output and the output/query buffer alias enabled."""
    forward = _forward_config(kernel_config)
    return replace(
        kernel_config,
        sm100_forward=replace(forward, fp32_saved_output=True, overlap_output_with_query=True),
    )
