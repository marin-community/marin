# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX/CuTe backend boundary for Grug packed-segment attention.

The production attention kernel is intentionally isolated here so the high-level Grug
attention code stays independent of optional CUDA-only dependencies. The first kernel
target is BF16/FP16 BSHD causal self-attention with dynamic per-token lower bounds:

    valid[b, q] and lower_bounds[b, q] <= k <= q + q_offset

This avoids both THD compaction and materialized [B, S, S] masks.
"""

import functools
import importlib
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from levanter.cutlass_kernel_cache import cutlass_call
from levanter.grug.attention._fa4_cute_kernels import (
    flash_attention_backward_postprocess_launcher,
    segmented_flash_attention_backward_launcher,
    segmented_flash_attention_backward_sm100_launcher,
    segmented_flash_attention_backward_sm90_launcher,
    native_flash_attention_backward_preprocess_launcher,
    segmented_flash_attention_forward_launcher,
    segmented_flash_attention_forward_sm100_launcher,
)
from levanter.grug.attention._fa4_cute_config import (
    SM100_GQA_RATIOS,
    SM100_HEAD_DIM,
    Flash4CuteKernelConfig,
    Flash4CuteSm100BackwardConfig,
)


@dataclass(frozen=True)
class _CutlassCuteModules:
    cute: Any
    cjax: Any
    cuda: Any


@dataclass(frozen=True)
class _BlockSparseMetadata:
    partial_block_cnt: jax.Array
    partial_block_idx: jax.Array
    full_block_cnt: jax.Array
    full_block_idx: jax.Array


@functools.lru_cache(maxsize=1)
def _import_cutlass_cute() -> _CutlassCuteModules:
    """Return the CuTe/CUTLASS module bundle.

    The launcher factories are keyed on this bundle, so it has to be a singleton.
    """
    cute = importlib.import_module("cutlass.cute")
    cjax = importlib.import_module("cutlass.jax")
    cuda = importlib.import_module("cuda.bindings.driver")
    return _CutlassCuteModules(cute=cute, cjax=cjax, cuda=cuda)


def _optional_dependency_error() -> RuntimeError:
    return RuntimeError(
        "gpu_fa4_cute_attention requires nvidia-cutlass-dsl with JAX support, and backward requires "
        "flash-attn-4. Install the CUDA 13 CUTLASS DSL extra, for example "
        "`nvidia-cutlass-dsl[cu13]>=4.4`, plus `flash-attn-4`."
    )


def segmented_flash_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    softmax_scale: float,
    kernel_config: Flash4CuteKernelConfig,
    q_offset: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """FA4/CuTe segmented attention forward entry point.

    ``Sq`` is the local query sequence length; ``Sk`` is the full key/value sequence length.

    Args:
        q: Query tensor with shape [B, Sq, Hq, D].
        k: Key tensor with shape [B, Sk, Hkv, D].
        v: Value tensor with shape [B, Sk, Hkv, Dv].
        lower_bounds: Inclusive per-token key lower bound, shape [B, Sq].
        valid: Per-token query validity mask, shape [B, Sq].
        softmax_scale: QK softmax scale.
        kernel_config: Architecture-specific tile/config object selected by attention.py.
        q_offset: Context-parallel shard offset, shape [1] int32. Local query ``i`` sits at
            global position ``i + q_offset``, which is the causal upper bound the kernel
            applies. The query slice must fit within K/V; unpartitioned queries use zero.

    Returns:
        ``(out, lse)`` where ``out`` has shape [B, Sq, Hq, Dv] and ``lse`` has
        shape [B, Hq, Sq]. The backward kernel consumes both tensors.
    """
    _validate_forward_inputs(q, k, v, lower_bounds, valid, softmax_scale=softmax_scale, q_offset=q_offset)
    if kernel_config.sm100_forward is not None:
        validate_sm100_layout(q, k, v)
    try:
        modules = _import_cutlass_cute()
    except Exception as exc:
        raise _optional_dependency_error() from exc

    if kernel_config.sm100_forward is not None:
        config = kernel_config.sm100_forward
        # One sparse query block spans every Q stage in the upstream schedule.
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
        input_spec, output_spec = _cutlass_attention_forward_specs(modules, vector_elems=8)
        metadata_spec = modules.cjax.TensorSpec(static=True)
        input_spec = (*input_spec, metadata_spec, metadata_spec, metadata_spec, metadata_spec)
        call = cutlass_call(
            launcher,
            output_shape_dtype=(
                jax.ShapeDtypeStruct((*q.shape[:3], v.shape[-1]), q.dtype),
                jax.ShapeDtypeStruct((q.shape[0], q.shape[2], q.shape[1]), jnp.float32),
            ),
            input_spec=input_spec,
            output_spec=output_spec,
            use_static_tensors=True,
            softmax_scale=softmax_scale,
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

    forward_tile = kernel_config.forward_tile
    num_threads = kernel_config.num_threads
    launcher = segmented_flash_attention_forward_launcher(
        modules,
        head_dim=q.shape[-1],
        head_dim_v=v.shape[-1],
        qhead_per_kvhead=q.shape[2] // k.shape[2],
        tile_m=forward_tile[0],
        tile_n=forward_tile[1],
        num_threads=num_threads,
    )
    input_spec, output_spec = _cutlass_attention_forward_specs(
        modules,
        vector_elems=8,
    )
    out_shape_dtype = jax.ShapeDtypeStruct((*q.shape[:3], v.shape[-1]), q.dtype)
    lse_shape_dtype = jax.ShapeDtypeStruct((q.shape[0], q.shape[2], q.shape[1]), jnp.float32)
    call = cutlass_call(
        launcher,
        output_shape_dtype=(out_shape_dtype, lse_shape_dtype),
        input_spec=input_spec,
        output_spec=output_spec,
        use_static_tensors=True,
        softmax_scale=softmax_scale,
    )
    return call(q, k, v, lower_bounds, valid.astype(jnp.int32), q_offset)


def segmented_flash_attention_backward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    dout: jax.Array,
    lse: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    softmax_scale: float,
    kernel_config: Flash4CuteKernelConfig,
    q_offset: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return gradients for FA4/CuTe packed-segment attention."""
    _validate_forward_inputs(q, k, v, lower_bounds, valid, softmax_scale=softmax_scale, q_offset=q_offset)
    _validate_backward_inputs(q, k, v, out, dout, lse)
    if kernel_config.sm100_backward is not None:
        validate_sm100_layout(q, k, v)
    try:
        modules = _import_cutlass_cute()
    except Exception as exc:
        raise _optional_dependency_error() from exc

    qhead_per_kvhead = q.shape[2] // k.shape[2]
    if kernel_config.sm100_backward is not None:
        return _segmented_flash_attention_backward_sm100(
            q,
            k,
            v,
            out,
            dout,
            lse,
            lower_bounds,
            valid,
            softmax_scale=softmax_scale,
            config=kernel_config.sm100_backward,
            q_offset=q_offset,
        )

    if kernel_config.sm90_backward is not None and qhead_per_kvhead > 1 and q.shape[-1] == 128:
        sm90_config = kernel_config.sm90_backward
        sparse_metadata = _packed_segment_backward_block_sparse_indices_with_full(
            lower_bounds,
            valid,
            tile_m=sm90_config.tile[0],
            tile_n=sm90_config.tile[1],
            key_sequence_length=k.shape[1],
            q_offset=q_offset,
        )
        return segmented_flash_attention_backward_sm90_native(
            q,
            k,
            v,
            out,
            dout,
            lse,
            lower_bounds,
            valid,
            sparse_metadata.partial_block_cnt,
            sparse_metadata.partial_block_idx,
            sparse_metadata.full_block_cnt,
            sparse_metadata.full_block_idx,
            softmax_scale=softmax_scale,
            kernel_config=kernel_config,
            q_offset=q_offset,
            window_size_left=None,
        )

    backward_tile = kernel_config.backward_tile
    num_threads = kernel_config.num_threads
    launcher = segmented_flash_attention_backward_launcher(
        modules,
        dtype=q.dtype,
        head_dim=q.shape[-1],
        head_dim_v=v.shape[-1],
        qhead_per_kvhead=qhead_per_kvhead,
        tile_m=backward_tile[0],
        tile_n=backward_tile[1],
        num_threads=num_threads,
        compute_arch=kernel_config.backward_arch,
    )
    input_spec, output_spec = _cutlass_attention_backward_specs(
        modules,
        vector_elems=8,
        qhead_per_kvhead=qhead_per_kvhead,
    )
    output_shape_dtype = _cutlass_attention_backward_output_shapes(q, k, v, backward_tile)
    call = cutlass_call(
        launcher,
        output_shape_dtype=output_shape_dtype,
        input_spec=input_spec,
        output_spec=output_spec,
        use_static_tensors=True,
        softmax_scale=softmax_scale,
    )
    dq, dk, dv, *_scratch = call(q, k, v, out, dout, lse, lower_bounds, valid.astype(jnp.int32), q_offset)
    return dq, dk, dv


def _segmented_flash_attention_backward_sm100(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    dout: jax.Array,
    lse: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    softmax_scale: float,
    config: Flash4CuteSm100BackwardConfig,
    q_offset: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Match the segmented backend contract using native one-CTA SM100."""
    ratio = q.shape[2] // k.shape[2]
    modules = _import_cutlass_cute()
    tile = config.tile
    sparse = _packed_segment_backward_block_sparse_indices_with_full(
        lower_bounds,
        valid,
        key_sequence_length=k.shape[1],
        q_offset=q_offset,
        tile_m=tile[0],
        tile_n=tile[1],
    )
    dpsum, lse_log2 = _native_backward_preprocess(modules, q, out, dout, lse, tile=tile, softmax_scale=softmax_scale)
    accum_inputs, accum_outputs = _native_backward_accum_specs(modules, vector_elems=8)
    backward = cutlass_call(
        segmented_flash_attention_backward_sm100_launcher(
            modules, head_dim=q.shape[-1], head_dim_v=v.shape[-1], qhead_per_kvhead=ratio, config=config
        ),
        output_shape_dtype=_native_backward_accum_output_shapes(q, k, v, tile),
        input_spec=accum_inputs,
        output_spec=accum_outputs,
        use_static_tensors=True,
        softmax_scale=softmax_scale,
    )
    accumulators = backward(
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        lower_bounds,
        valid.astype(jnp.int32),
        q_offset,
        sparse.partial_block_cnt,
        sparse.partial_block_idx,
        sparse.full_block_cnt,
        sparse.full_block_idx,
    )
    return _native_backward_gradients(
        modules,
        (q, k, v),
        accumulators,
        tile_rows=(tile[0], tile[1], tile[1]),
        arch=100,
        num_threads=config.postprocess_threads,
        softmax_scale=softmax_scale,
    )


def segmented_flash_attention_backward_sm90_native(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    dout: jax.Array,
    lse: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    mask_block_cnt: jax.Array,
    mask_block_idx: jax.Array,
    full_block_cnt: jax.Array | None = None,
    full_block_idx: jax.Array | None = None,
    *,
    softmax_scale: float,
    kernel_config: Flash4CuteKernelConfig,
    q_offset: jax.Array,
    window_size_left: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run the native SM90 segmented backward path for D128 GQA kernels.

    ``q_offset`` is an int32[1] array holding the global position of local query 0 within the
    full key sequence. It is zero unless context parallelism slices the queries.
    """
    _validate_forward_inputs(q, k, v, lower_bounds, valid, softmax_scale=softmax_scale, q_offset=q_offset)
    _validate_backward_inputs(q, k, v, out, dout, lse)
    sm90_config = kernel_config.sm90_backward
    if sm90_config is None:
        raise NotImplementedError("native SM90 backward requires kernel_config.sm90_backward.")
    if sm90_config.tile[0] != 64:
        raise NotImplementedError(f"native SM90 postprocess requires tile_m=64, got {sm90_config.tile}.")
    _validate_backward_block_sparse_metadata(
        q,
        k,
        mask_block_cnt,
        mask_block_idx,
        tile_m=sm90_config.tile[0],
        tile_n=sm90_config.tile[1],
    )
    if full_block_cnt is None:
        full_block_cnt = jnp.zeros_like(mask_block_cnt)
    if full_block_idx is None:
        full_block_idx = jnp.zeros_like(mask_block_idx)
    _validate_backward_block_sparse_metadata(
        q,
        k,
        full_block_cnt,
        full_block_idx,
        tile_m=sm90_config.tile[0],
        tile_n=sm90_config.tile[1],
    )
    mask_block_cnt, mask_block_idx = _broadcast_backward_block_sparse_metadata(q, mask_block_cnt, mask_block_idx)
    full_block_cnt, full_block_idx = _broadcast_backward_block_sparse_metadata(q, full_block_cnt, full_block_idx)
    try:
        modules = _import_cutlass_cute()
    except Exception as exc:
        raise _optional_dependency_error() from exc

    # Upstream SM90 backward is not exposed as a single JAX custom call in this
    # integration. Its mainloop consumes dPsum and log2 LSE from preprocess, and
    # its postprocess expects gmem-backed accumulator buffers with the SM90
    # accumulator layout. Keeping these as separate cutlass_call boundaries
    # preserves that ABI and avoids decoding SM90 accumulators with the older
    # segmented fallback postprocess contract.
    backward_launcher = segmented_flash_attention_backward_sm90_launcher(
        modules,
        dtype=q.dtype,
        head_dim=q.shape[-1],
        head_dim_v=v.shape[-1],
        qhead_per_kvhead=q.shape[2] // k.shape[2],
        config=sm90_config,
        window_size_left=window_size_left,
    )
    qhead_per_kvhead = q.shape[2] // k.shape[2]
    if qhead_per_kvhead == 1:
        raise NotImplementedError("native SM90 backward currently expects GQA so dK/dV accumulators are present.")
    dpsum, lse_log2 = _native_backward_preprocess(
        modules, q, out, dout, lse, tile=sm90_config.tile, softmax_scale=softmax_scale
    )

    backward_input_spec, backward_output_spec = _native_backward_accum_specs(modules, vector_elems=8)
    backward_output_shape_dtype = _native_backward_accum_output_shapes(q, k, v, sm90_config.tile)
    backward_call = cutlass_call(
        backward_launcher,
        output_shape_dtype=backward_output_shape_dtype,
        input_spec=backward_input_spec,
        output_spec=backward_output_spec,
        use_static_tensors=True,
        softmax_scale=softmax_scale,
    )
    dq_accum, dk_accum, dv_accum = backward_call(
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        lower_bounds,
        valid.astype(jnp.int32),
        q_offset,
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
    )
    return _native_backward_gradients(
        modules,
        (q, k, v),
        (dq_accum, dk_accum, dv_accum),
        tile_rows=(sm90_config.tile[0],) * 3,
        arch=90,
        num_threads=128,
        softmax_scale=softmax_scale,
    )


def _native_backward_preprocess(
    modules: _CutlassCuteModules,
    q: jax.Array,
    out: jax.Array,
    dout: jax.Array,
    lse: jax.Array,
    *,
    tile: tuple[int, int],
    softmax_scale: float,
) -> tuple[jax.Array, jax.Array]:
    inputs, outputs = _native_backward_preprocess_specs(modules, vector_elems=8)
    preprocess = cutlass_call(
        native_flash_attention_backward_preprocess_launcher(
            modules, dtype=q.dtype, head_dim=q.shape[-1], head_dim_v=out.shape[-1], tile_m=tile[0]
        ),
        output_shape_dtype=_native_backward_preprocess_output_shapes(q, tile),
        input_spec=inputs,
        output_spec=outputs,
        use_static_tensors=True,
        softmax_scale=softmax_scale,
    )
    dpsum, lse_log2 = preprocess(out, dout, lse)
    return dpsum, lse_log2


def _native_backward_gradients(
    modules: _CutlassCuteModules,
    qkv: tuple[jax.Array, jax.Array, jax.Array],
    accumulators: tuple[jax.Array, jax.Array, jax.Array],
    *,
    tile_rows: tuple[int, int, int],
    arch: int,
    num_threads: int,
    softmax_scale: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    inputs, outputs = _native_backward_postprocess_specs(modules, vector_elems=8)
    gradients = []
    for tensor, accum, scale, rows in zip(
        qkv, accumulators, (softmax_scale, softmax_scale, 1.0), tile_rows, strict=True
    ):
        postprocess = cutlass_call(
            flash_attention_backward_postprocess_launcher(
                modules,
                dtype=tensor.dtype,
                head_dim=tensor.shape[-1],
                tile_m=rows,
                atom_layout_m=1,
                arch=arch,
                num_threads=num_threads,
                cluster_size=1,
                use_2cta_instrs=False,
                accum_is_gmem=True,
            ),
            output_shape_dtype=(jax.ShapeDtypeStruct(tensor.shape, tensor.dtype),),
            input_spec=inputs,
            output_spec=outputs,
            use_static_tensors=True,
            softmax_scale=scale,
        )
        gradients.append(postprocess(accum)[0])
    return gradients[0], gradients[1], gradients[2]


def _cutlass_attention_forward_specs(
    modules: _CutlassCuteModules, *, vector_elems: int
) -> tuple[tuple[Any, ...], Any]:
    tensor_spec = modules.cjax.TensorSpec
    qkv_spec = tensor_spec(mode=(1, 3, 2, 0), divisibility=(1, 1, 1, vector_elems), static=True)
    lse_spec = tensor_spec(divisibility=(1, 1, 1), static=True)
    metadata_spec = tensor_spec(static=True)
    input_spec = (qkv_spec, qkv_spec, qkv_spec, metadata_spec, metadata_spec, tensor_spec(mode=(0,), static=True))
    return input_spec, (qkv_spec, lse_spec)


def _cutlass_attention_backward_specs(
    modules: _CutlassCuteModules, *, vector_elems: int, qhead_per_kvhead: int
) -> tuple[tuple[Any, ...], Any]:
    tensor_spec = modules.cjax.TensorSpec
    qkv_spec = tensor_spec(mode=(0, 1, 2, 3), divisibility=(1, 1, 1, vector_elems), static=True)
    lse_spec = tensor_spec(mode=(0, 1, 2), divisibility=(1, 1, 1), static=True)
    metadata_spec = tensor_spec(mode=(0, 1), static=True)
    scratch_spec = tensor_spec(mode=(0, 1, 2), static=True)
    input_spec = (
        qkv_spec,
        qkv_spec,
        qkv_spec,
        qkv_spec,
        qkv_spec,
        lse_spec,
        metadata_spec,
        metadata_spec,
        tensor_spec(mode=(0,), static=True),
    )
    dkv_accum_spec = scratch_spec if qhead_per_kvhead > 1 else qkv_spec
    return input_spec, (
        qkv_spec,
        qkv_spec,
        qkv_spec,
        scratch_spec,
        scratch_spec,
        scratch_spec,
        dkv_accum_spec,
        dkv_accum_spec,
    )


def _native_backward_accum_specs(modules: _CutlassCuteModules, *, vector_elems: int) -> tuple[tuple[Any, ...], Any]:
    tensor_spec = modules.cjax.TensorSpec
    qkv_spec = tensor_spec(mode=(0, 1, 2, 3), divisibility=(1, 1, 1, vector_elems), static=True)
    scratch_spec = tensor_spec(mode=(0, 1, 2), static=True)
    metadata_spec = tensor_spec(mode=(0, 1), static=True)
    sparse_cnt_spec = tensor_spec(mode=(0, 1, 2), static=True)
    sparse_idx_spec = tensor_spec(mode=(0, 1, 2, 3), static=True)
    input_spec = (
        qkv_spec,
        qkv_spec,
        qkv_spec,
        qkv_spec,
        scratch_spec,
        scratch_spec,
        metadata_spec,
        metadata_spec,
        tensor_spec(mode=(0,), static=True),
        sparse_cnt_spec,
        sparse_idx_spec,
        sparse_cnt_spec,
        sparse_idx_spec,
    )
    return input_spec, (scratch_spec, scratch_spec, scratch_spec)


def _native_backward_preprocess_specs(
    modules: _CutlassCuteModules, *, vector_elems: int
) -> tuple[tuple[Any, ...], Any]:
    tensor_spec = modules.cjax.TensorSpec
    qkv_spec = tensor_spec(mode=(0, 1, 2, 3), divisibility=(1, 1, 1, vector_elems), static=True)
    lse_spec = tensor_spec(mode=(0, 1, 2), divisibility=(1, 1, 1), static=True)
    scratch_spec = tensor_spec(mode=(0, 1, 2), static=True)
    return (qkv_spec, qkv_spec, lse_spec), (scratch_spec, scratch_spec)


def _native_backward_postprocess_specs(
    modules: _CutlassCuteModules, *, vector_elems: int
) -> tuple[tuple[Any, ...], Any]:
    tensor_spec = modules.cjax.TensorSpec
    scratch_spec = tensor_spec(mode=(0, 1, 2), static=True)
    qkv_spec = tensor_spec(mode=(0, 1, 2, 3), divisibility=(1, 1, 1, vector_elems), static=True)
    return (scratch_spec,), (qkv_spec,)


def _packed_segment_backward_block_sparse_indices_with_full(
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    key_sequence_length: int,
    q_offset: jax.Array,
    tile_m: int,
    tile_n: int,
) -> _BlockSparseMetadata:
    partial, full = _packed_segment_block_masks(
        lower_bounds, valid, key_sequence_length=key_sequence_length, q_offset=q_offset, tile_m=tile_m, tile_n=tile_n
    )
    return _block_sparse_indices(partial, full)


def _packed_segment_forward_block_sparse_indices_with_full(
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    key_sequence_length: int,
    q_offset: jax.Array,
    tile_m: int,
    tile_n: int,
) -> _BlockSparseMetadata:
    partial, full = _packed_segment_block_masks(
        lower_bounds, valid, key_sequence_length=key_sequence_length, q_offset=q_offset, tile_m=tile_m, tile_n=tile_n
    )
    return _block_sparse_indices(jnp.swapaxes(partial, 1, 2), jnp.swapaxes(full, 1, 2))


def _packed_segment_block_masks(
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    key_sequence_length: int,
    q_offset: jax.Array,
    tile_m: int,
    tile_n: int,
) -> tuple[jax.Array, jax.Array]:
    """Classify tiles as partial/full in [batch, key block, query block] order."""
    if tile_m <= 0 or tile_n <= 0:
        raise ValueError(f"tile_m and tile_n must be positive, got {tile_m=} {tile_n=}")
    if lower_bounds.ndim != 2 or valid.ndim != 2:
        raise ValueError(f"lower_bounds and valid must have shape [B, S], got {lower_bounds.shape=} {valid.shape=}")
    if lower_bounds.shape != valid.shape:
        raise ValueError(f"lower_bounds and valid must have matching shape, got {lower_bounds.shape=} {valid.shape=}")

    if key_sequence_length <= 0:
        raise ValueError(f"key_sequence_length must be positive, got {key_sequence_length}")
    if q_offset.shape != (1,) or q_offset.dtype != jnp.int32:
        raise ValueError(f"q_offset must be int32[1], got {q_offset.shape} {q_offset.dtype}")
    batch_size, seq_len = lower_bounds.shape
    num_m_blocks = (seq_len + tile_m - 1) // tile_m
    num_n_blocks = (key_sequence_length + tile_n - 1) // tile_n
    padded_q_len = num_m_blocks * tile_m
    q_positions = jnp.arange(padded_q_len, dtype=jnp.int32).reshape(num_m_blocks, tile_m) + q_offset[0]
    lower_padded = jnp.pad(
        lower_bounds,
        ((0, 0), (0, padded_q_len - seq_len)),
        mode="constant",
        constant_values=key_sequence_length,
    ).reshape(batch_size, num_m_blocks, tile_m)
    valid_padded = jnp.pad(
        valid,
        ((0, 0), (0, padded_q_len - seq_len)),
        mode="constant",
        constant_values=False,
    ).reshape(batch_size, num_m_blocks, tile_m)

    n_starts = jnp.arange(num_n_blocks, dtype=jnp.int32) * tile_n
    n_ends = jnp.minimum(n_starts + tile_n, key_sequence_length) - 1
    has_contributor = jnp.any(
        valid_padded[:, None, :, :]
        & (q_positions[None, None, :, :] >= n_starts[None, :, None, None])
        & (lower_padded[:, None, :, :] <= n_ends[None, :, None, None]),
        axis=-1,
    )
    all_queries_valid = jnp.all(valid_padded, axis=-1)
    tile_starts = q_positions[:, 0]
    tile_lower_bounds = jnp.max(lower_padded, axis=-1)
    is_full = (
        has_contributor
        & all_queries_valid[:, None, :]
        & (n_ends[None, :, None] <= tile_starts[None, None, :])
        & (n_starts[None, :, None] >= tile_lower_bounds[:, None, :])
    )
    is_partial = has_contributor & ~is_full

    return is_partial, is_full


def _block_sparse_indices(is_partial: jax.Array, is_full: jax.Array) -> _BlockSparseMetadata:
    """Compact the last block axis into upstream FA4 sparse lists."""
    num_blocks = is_partial.shape[-1]
    block_indices = jnp.arange(num_blocks, dtype=jnp.int32)
    partial_indices = jnp.where(is_partial, block_indices[None, None, :], num_blocks)
    full_indices = jnp.where(is_full, block_indices[None, None, :], num_blocks)
    sorted_partial_indices = jnp.sort(partial_indices, axis=-1)
    sorted_full_indices = jnp.sort(full_indices, axis=-1)
    mask_block_cnt = jnp.sum(is_partial.astype(jnp.int32), axis=-1)[:, None, :]
    full_block_cnt = jnp.sum(is_full.astype(jnp.int32), axis=-1)[:, None, :]
    mask_block_idx = jnp.where(sorted_partial_indices < num_blocks, sorted_partial_indices, 0)[:, None, :, :]
    full_block_idx = jnp.where(sorted_full_indices < num_blocks, sorted_full_indices, 0)[:, None, :, :]
    return _BlockSparseMetadata(
        partial_block_cnt=mask_block_cnt,
        partial_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
    )


def _cutlass_attention_backward_output_shapes(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    backward_tile: tuple[int, int],
) -> tuple[jax.ShapeDtypeStruct, ...]:
    batch, seq_len, q_heads, head_dim = q.shape
    kv_heads = k.shape[2]
    tile_m, tile_n = backward_tile
    seq_q_rounded = ((seq_len + tile_m - 1) // tile_m) * tile_m
    seq_k_rounded = ((k.shape[1] + tile_n - 1) // tile_n) * tile_n
    head_dim_rounded = ((head_dim + 31) // 32) * 32
    head_dim_v_rounded = ((v.shape[-1] + 31) // 32) * 32
    qhead_per_kvhead = q_heads // kv_heads
    dk_accum = (
        jax.ShapeDtypeStruct((batch, kv_heads, seq_k_rounded * head_dim_rounded), jnp.float32)
        if qhead_per_kvhead > 1
        else jax.ShapeDtypeStruct(k.shape, k.dtype)
    )
    dv_accum = (
        jax.ShapeDtypeStruct((batch, kv_heads, seq_k_rounded * head_dim_v_rounded), jnp.float32)
        if qhead_per_kvhead > 1
        else jax.ShapeDtypeStruct(v.shape, v.dtype)
    )
    return (
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(v.shape, v.dtype),
        jax.ShapeDtypeStruct((batch, q_heads, seq_q_rounded), jnp.float32),
        jax.ShapeDtypeStruct((batch, q_heads, seq_q_rounded), jnp.float32),
        jax.ShapeDtypeStruct((batch, q_heads, seq_q_rounded * head_dim_rounded), jnp.float32),
        dk_accum,
        dv_accum,
    )


def _native_backward_preprocess_output_shapes(
    q: jax.Array,
    backward_tile: tuple[int, int],
) -> tuple[jax.ShapeDtypeStruct, ...]:
    batch, seq_len, q_heads, _head_dim = q.shape
    tile_m, _tile_n = backward_tile
    seq_q_rounded = ((seq_len + tile_m - 1) // tile_m) * tile_m
    scratch_q = jax.ShapeDtypeStruct((batch, q_heads, seq_q_rounded), jnp.float32)
    return scratch_q, scratch_q


def _native_backward_accum_output_shapes(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    backward_tile: tuple[int, int],
) -> tuple[jax.ShapeDtypeStruct, ...]:
    batch, seq_len, q_heads, head_dim = q.shape
    kv_heads = k.shape[2]
    tile_m, tile_n = backward_tile
    seq_q_rounded = ((seq_len + tile_m - 1) // tile_m) * tile_m
    seq_k_rounded = ((k.shape[1] + tile_n - 1) // tile_n) * tile_n
    head_dim_rounded = ((head_dim + 31) // 32) * 32
    head_dim_v_rounded = ((v.shape[-1] + 31) // 32) * 32
    dq_accum = jax.ShapeDtypeStruct((batch, q_heads, seq_q_rounded * head_dim_rounded), jnp.float32)
    dk_accum = jax.ShapeDtypeStruct((batch, kv_heads, seq_k_rounded * head_dim_rounded), jnp.float32)
    dv_accum = jax.ShapeDtypeStruct((batch, kv_heads, seq_k_rounded * head_dim_v_rounded), jnp.float32)
    return dq_accum, dk_accum, dv_accum


def fa4_cute_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    sm_scale: float | None = None,
    kernel_config: Flash4CuteKernelConfig,
    q_offset: jax.Array,
) -> jax.Array:
    """FA4/CuTe attention boundary with packed causal metadata.

    Forward uses the CUTLASS/CuTe JAX FFI path. Backward is routed through a custom VJP so JAX does not
    attempt to autodiff through ``cutlass_call``. ``q_offset`` is the context-parallel shard offset
    described in :func:`segmented_flash_attention_forward`.
    """
    if sm_scale is None:
        sm_scale = float(q.shape[-1] ** -0.5)
    return _segmented_flash_attention_custom_vjp(
        q,
        k,
        v,
        lower_bounds,
        valid,
        q_offset,
        sm_scale,
        kernel_config,
    )


@partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _segmented_flash_attention_custom_vjp(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    q_offset: jax.Array,
    softmax_scale: float,
    kernel_config: Flash4CuteKernelConfig,
) -> jax.Array:
    out, _ = segmented_flash_attention_forward(
        q,
        k,
        v,
        lower_bounds,
        valid,
        softmax_scale=softmax_scale,
        kernel_config=kernel_config,
        q_offset=q_offset,
    )
    return out


class _SegmentedAttentionResiduals(NamedTuple):
    q: jax.Array
    k: jax.Array
    v: jax.Array
    out: jax.Array
    lse: jax.Array
    lower_bounds: jax.Array
    valid: jax.Array
    q_offset: jax.Array


def _segmented_flash_attention_custom_vjp_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    q_offset: jax.Array,
    softmax_scale: float,
    kernel_config: Flash4CuteKernelConfig,
) -> tuple[jax.Array, _SegmentedAttentionResiduals]:
    out, lse = segmented_flash_attention_forward(
        q,
        k,
        v,
        lower_bounds,
        valid,
        softmax_scale=softmax_scale,
        kernel_config=kernel_config,
        q_offset=q_offset,
    )
    return out, _SegmentedAttentionResiduals(
        q=q,
        k=k,
        v=v,
        out=out,
        lse=lse,
        lower_bounds=lower_bounds,
        valid=valid,
        q_offset=q_offset,
    )


def _segmented_flash_attention_custom_vjp_bwd(
    softmax_scale: float,
    kernel_config: Flash4CuteKernelConfig,
    residuals: _SegmentedAttentionResiduals,
    cotangent: jax.Array | jax.custom_derivatives.SymbolicZero,
) -> tuple[jax.Array | None, jax.Array | None, jax.Array | None, None, None, None]:
    if isinstance(cotangent, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros_like(residuals.q), jnp.zeros_like(residuals.k), jnp.zeros_like(residuals.v), None, None, None
    dq, dk, dv = segmented_flash_attention_backward(
        residuals.q,
        residuals.k,
        residuals.v,
        residuals.out,
        cotangent.astype(residuals.q.dtype),
        residuals.lse,
        residuals.lower_bounds,
        residuals.valid,
        softmax_scale=softmax_scale,
        kernel_config=kernel_config,
        q_offset=residuals.q_offset,
    )
    return dq, dk, dv, None, None, None


_segmented_flash_attention_custom_vjp.defvjp(
    _segmented_flash_attention_custom_vjp_fwd,
    _segmented_flash_attention_custom_vjp_bwd,
)


def validate_sm100_layout(q: jax.Array, k: jax.Array, v: jax.Array) -> None:
    """Reject layouts outside the BF16 D128 GQA shapes the native SM100 kernels were validated on."""
    if q.dtype != jnp.bfloat16 or q.shape[-1] != SM100_HEAD_DIM or v.shape[-1] != SM100_HEAD_DIM:
        raise ValueError(
            f"gpu_fa4_cute_sm100 requires BF16 with D == Dv == {SM100_HEAD_DIM}, "
            f"got {q.dtype} D={q.shape[-1]} Dv={v.shape[-1]}."
        )
    if q.shape[2] % k.shape[2] or q.shape[2] // k.shape[2] not in SM100_GQA_RATIOS:
        raise ValueError(
            f"gpu_fa4_cute_sm100 requires a GQA ratio in {SM100_GQA_RATIOS}, got {q.shape[2]}/{k.shape[2]} heads."
        )


def _validate_forward_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    softmax_scale: float,
    q_offset: jax.Array,
) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"q/k/v must be BSHD tensors, got q={q.shape}, k={k.shape}, v={v.shape}")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(f"q/k/v batch sizes must match, got q={q.shape}, k={k.shape}, v={v.shape}")
    if k.shape[1] != v.shape[1]:
        raise ValueError(f"k/v sequence lengths must match, got k={k.shape}, v={v.shape}")
    if q.shape[1] > k.shape[1]:
        raise ValueError(f"q sequence length must not exceed the key sequence length, got q={q.shape}, k={k.shape}")
    if q_offset.shape != (1,) or q_offset.dtype != jnp.int32:
        raise ValueError(f"q_offset must be an int32 array of shape [1], got {q_offset.shape} {q_offset.dtype}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q/k head dimensions must match, got q={q.shape}, k={k.shape}")
    if k.shape[2] != v.shape[2]:
        raise ValueError(f"k/v head counts must match, got k={k.shape}, v={v.shape}")
    if v.shape[-1] != q.shape[-1]:
        raise NotImplementedError(f"gpu_fa4_cute_attention currently requires Dv == D, got q={q.shape}, v={v.shape}")
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError(f"Hq must be divisible by Hkv for GQA, got q={q.shape}, k={k.shape}")
    if lower_bounds.shape != q.shape[:2]:
        raise ValueError(f"lower_bounds must have shape [B, S]={q.shape[:2]}, got {lower_bounds.shape}")
    if valid.shape != q.shape[:2]:
        raise ValueError(f"valid must have shape [B, S]={q.shape[:2]}, got {valid.shape}")
    if lower_bounds.dtype != jnp.int32:
        raise ValueError(f"lower_bounds must be int32, got {lower_bounds.dtype}")
    if valid.dtype != jnp.bool_:
        raise ValueError(f"valid must be bool, got {valid.dtype}")
    if q.dtype not in (jnp.bfloat16, jnp.float16):
        raise TypeError(f"gpu_fa4_cute_attention currently supports only bf16/fp16, got {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError(f"q/k/v dtypes must match, got q={q.dtype}, k={k.dtype}, v={v.dtype}")
    if not isinstance(softmax_scale, float):
        raise TypeError(f"softmax_scale must be a Python float, got {type(softmax_scale).__name__}")
    if softmax_scale <= 0.0:
        raise ValueError(f"softmax_scale must be positive, got {softmax_scale}")


def _validate_backward_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    dout: jax.Array,
    lse: jax.Array,
) -> None:
    expected_out_shape = (*q.shape[:3], v.shape[-1])
    if out.shape != expected_out_shape:
        raise ValueError(f"out must have shape {expected_out_shape}, got {out.shape}")
    if dout.shape != expected_out_shape:
        raise ValueError(f"dout must have shape {expected_out_shape}, got {dout.shape}")
    if out.dtype != q.dtype or dout.dtype != q.dtype:
        raise TypeError(f"out/dout dtypes must match q dtype {q.dtype}, got out={out.dtype}, dout={dout.dtype}")
    expected_lse_shape = (q.shape[0], q.shape[2], q.shape[1])
    if lse.shape != expected_lse_shape:
        raise ValueError(f"lse must have shape [B, Hq, S]={expected_lse_shape}, got {lse.shape}")
    if lse.dtype != jnp.float32:
        raise TypeError(f"lse must be float32, got {lse.dtype}")


def _validate_backward_block_sparse_metadata(
    q: jax.Array,
    k: jax.Array,
    mask_block_cnt: jax.Array,
    mask_block_idx: jax.Array,
    *,
    tile_m: int,
    tile_n: int,
) -> None:
    batch, q_len, q_heads, _ = q.shape
    k_len, kv_heads = k.shape[1], k.shape[2]
    if q_heads % kv_heads != 0:
        raise ValueError(f"Hq must be divisible by Hkv for GQA, got q={q.shape}, k={k.shape}")
    expected_n_blocks = (k_len + tile_n - 1) // tile_n
    expected_m_blocks = (q_len + tile_m - 1) // tile_m
    if mask_block_cnt.dtype != jnp.int32:
        raise ValueError(f"mask_block_cnt must be int32, got {mask_block_cnt.dtype}")
    if mask_block_idx.dtype != jnp.int32:
        raise ValueError(f"mask_block_idx must be int32, got {mask_block_idx.dtype}")
    if mask_block_cnt.ndim != 3:
        raise ValueError(f"mask_block_cnt must have shape [B, H|1, N], got {mask_block_cnt.shape}")
    if mask_block_idx.ndim != 4:
        raise ValueError(f"mask_block_idx must have shape [B, H|1, N, M], got {mask_block_idx.shape}")
    if mask_block_cnt.shape[0] != batch or mask_block_idx.shape[0] != batch:
        raise ValueError(
            f"block sparse batch dim must be {batch}, got {mask_block_cnt.shape=} {mask_block_idx.shape=}"
        )
    if mask_block_cnt.shape[1] not in (1, q_heads):
        raise ValueError(f"mask_block_cnt head dim must be 1 or {q_heads}, got {mask_block_cnt.shape}")
    if mask_block_idx.shape[1] not in (1, q_heads):
        raise ValueError(f"mask_block_idx head dim must be 1 or {q_heads}, got {mask_block_idx.shape}")
    if mask_block_cnt.shape[2] != expected_n_blocks:
        raise ValueError(f"mask_block_cnt N dim must be {expected_n_blocks}, got {mask_block_cnt.shape}")
    if mask_block_idx.shape[2] != expected_n_blocks:
        raise ValueError(f"mask_block_idx N dim must be {expected_n_blocks}, got {mask_block_idx.shape}")
    if mask_block_idx.shape[3] > expected_m_blocks:
        raise ValueError(f"mask_block_idx M dim must be <= {expected_m_blocks}, got {mask_block_idx.shape}")


def _broadcast_backward_block_sparse_metadata(
    q: jax.Array,
    mask_block_cnt: jax.Array,
    mask_block_idx: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    q_heads = q.shape[2]
    if mask_block_cnt.shape[1] == q_heads and mask_block_idx.shape[1] == q_heads:
        return mask_block_cnt, mask_block_idx
    if mask_block_cnt.shape[1] != 1 or mask_block_idx.shape[1] != 1:
        raise ValueError(f"block sparse head dims must both be 1 or Hq={q_heads}.")
    return (
        jnp.broadcast_to(mask_block_cnt, (mask_block_cnt.shape[0], q_heads, mask_block_cnt.shape[2])),
        jnp.broadcast_to(
            mask_block_idx,
            (mask_block_idx.shape[0], q_heads, mask_block_idx.shape[2], mask_block_idx.shape[3]),
        ),
    )


__all__ = [
    "fa4_cute_attention_forward",
    "segmented_flash_attention_backward",
    "segmented_flash_attention_backward_sm90_native",
    "segmented_flash_attention_forward",
]
