# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX/CuTe backend boundary for Grug packed-segment attention.

The production attention kernel is intentionally isolated here so the high-level Grug
attention code stays independent of optional CUDA-only dependencies. The first kernel
target is BF16/FP16 BSHD causal self-attention with dynamic per-token lower bounds:

    valid[b, q] and lower_bounds[b, q] <= k <= q

This avoids both THD compaction and materialized [B, S, S] masks.
"""

import importlib
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from levanter.grug.attention._fa4_cute_kernels import (
    flash_attention_backward_postprocess_launcher,
    segmented_flash_attention_backward_launcher,
    segmented_flash_attention_backward_sm90_launcher,
    segmented_flash_attention_backward_sm90_preprocess_launcher,
    segmented_flash_attention_forward_launcher,
)
from levanter.grug.attention._fa4_cute_config import Flash4CuteKernelConfig


@dataclass(frozen=True)
class _CutlassCuteModules:
    cute: Any
    cjax: Any
    cuda: Any


@dataclass(frozen=True)
class _BackwardBlockSparseMetadata:
    partial_block_cnt: jax.Array
    partial_block_idx: jax.Array
    full_block_cnt: jax.Array
    full_block_idx: jax.Array


def _import_cutlass_cute() -> _CutlassCuteModules:
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


def cutlass_cute_available() -> bool:
    """Return whether the optional CuTe/JAX CUTLASS modules are importable."""
    try:
        _import_cutlass_cute()
    except Exception:
        return False
    return True


def require_cutlass_cute() -> None:
    """Raise a clear error if nvidia-cutlass-dsl with JAX support is unavailable."""
    try:
        _import_cutlass_cute()
    except Exception as exc:
        raise _optional_dependency_error() from exc


def segmented_flash_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    softmax_scale: float,
    kernel_config: Flash4CuteKernelConfig,
) -> tuple[jax.Array, jax.Array]:
    """FA4/CuTe segmented attention forward entry point.

    Args:
        q: Query tensor with shape [B, S, Hq, D].
        k: Key tensor with shape [B, S, Hkv, D].
        v: Value tensor with shape [B, S, Hkv, Dv].
        lower_bounds: Inclusive per-token key lower bound, shape [B, S].
        valid: Per-token query validity mask, shape [B, S].
        softmax_scale: QK softmax scale.
        kernel_config: Architecture-specific tile/config object selected by attention.py.

    Returns:
        ``(out, lse)`` where ``out`` has shape [B, S, Hq, Dv] and ``lse`` has
        shape [B, Hq, S]. The backward kernel consumes both tensors.
    """
    _validate_forward_inputs(q, k, v, lower_bounds, valid, softmax_scale=softmax_scale)
    try:
        modules = _import_cutlass_cute()
    except Exception as exc:
        raise _optional_dependency_error() from exc

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
    input_spec, output_spec = _cutlass_attention_forward_specs(modules, vector_elems=8)
    out_shape_dtype = jax.ShapeDtypeStruct((*q.shape[:3], v.shape[-1]), q.dtype)
    lse_shape_dtype = jax.ShapeDtypeStruct((q.shape[0], q.shape[2], q.shape[1]), jnp.float32)
    call = modules.cjax.cutlass_call(
        launcher,
        output_shape_dtype=(out_shape_dtype, lse_shape_dtype),
        input_spec=input_spec,
        output_spec=output_spec,
        use_static_tensors=True,
        softmax_scale=softmax_scale,
    )
    return call(q, k, v, lower_bounds, valid.astype(jnp.int32))


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
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FA4/CuTe segmented attention backward boundary.

    This is intentionally a separate CUTLASS call from the forward path because
    upstream FA4 backward is a preprocess + main-kernel + postprocess pipeline.
    SM90 D128 GQA routes to the native Hopper path; other cases use the
    SM120-compatible segmented launcher.
    """
    _validate_forward_inputs(q, k, v, lower_bounds, valid, softmax_scale=softmax_scale)
    _validate_backward_inputs(q, k, v, out, dout, lse)
    try:
        modules = _import_cutlass_cute()
    except Exception as exc:
        raise _optional_dependency_error() from exc

    qhead_per_kvhead = q.shape[2] // k.shape[2]
    if kernel_config.sm90_backward is not None and qhead_per_kvhead > 1 and q.shape[-1] == 128:
        sm90_config = kernel_config.sm90_backward
        sparse_metadata = _packed_segment_backward_block_sparse_indices_with_full(
            lower_bounds,
            valid,
            tile_m=sm90_config.tile[0],
            tile_n=sm90_config.tile[1],
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
    call = modules.cjax.cutlass_call(
        launcher,
        output_shape_dtype=output_shape_dtype,
        input_spec=input_spec,
        output_spec=output_spec,
        use_static_tensors=True,
        softmax_scale=softmax_scale,
    )
    dq, dk, dv, *_scratch = call(q, k, v, out, dout, lse, lower_bounds, valid.astype(jnp.int32))
    return dq, dk, dv


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
    window_size_left: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run the native SM90 segmented backward path for D128 GQA kernels."""
    _validate_forward_inputs(q, k, v, lower_bounds, valid, softmax_scale=softmax_scale)
    _validate_backward_inputs(q, k, v, out, dout, lse)
    sm90_config = kernel_config.sm90_backward
    if sm90_config is None:
        raise NotImplementedError("native SM90 backward requires kernel_config.sm90_backward.")
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

    preprocess_launcher = segmented_flash_attention_backward_sm90_preprocess_launcher(
        modules,
        dtype=q.dtype,
        head_dim=q.shape[-1],
        head_dim_v=v.shape[-1],
        tile_m=sm90_config.tile[0],
    )
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
    preprocess_input_spec, preprocess_output_spec = _cutlass_attention_backward_sm90_preprocess_specs(
        modules,
        vector_elems=8,
    )
    preprocess_output_shape_dtype = _cutlass_attention_backward_sm90_preprocess_output_shapes(q, sm90_config.tile)
    preprocess_call = modules.cjax.cutlass_call(
        preprocess_launcher,
        output_shape_dtype=preprocess_output_shape_dtype,
        input_spec=preprocess_input_spec,
        output_spec=preprocess_output_spec,
        use_static_tensors=True,
    )
    dpsum, lse_log2, _dq_accum = preprocess_call(out, dout, lse)

    backward_input_spec, backward_output_spec = _cutlass_attention_backward_sm90_accum_specs(modules, vector_elems=8)
    backward_output_shape_dtype = _cutlass_attention_backward_sm90_backward_output_shapes(q, k, v, sm90_config.tile)
    backward_call = modules.cjax.cutlass_call(
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
        mask_block_cnt,
        mask_block_idx,
        full_block_cnt,
        full_block_idx,
    )
    postprocess_input_spec, postprocess_output_spec = _cutlass_attention_backward_sm90_postprocess_specs(
        modules,
        vector_elems=8,
    )
    postprocess_arch = 90
    postprocess_tile_m = 64
    postprocess_atom_layout_m = 1
    dq_postprocess = modules.cjax.cutlass_call(
        flash_attention_backward_postprocess_launcher(
            modules,
            dtype=q.dtype,
            head_dim=q.shape[-1],
            tile_m=postprocess_tile_m,
            atom_layout_m=postprocess_atom_layout_m,
            arch=postprocess_arch,
            cluster_size=1,
            use_2cta_instrs=False,
            accum_is_gmem=True,
        ),
        output_shape_dtype=(jax.ShapeDtypeStruct(q.shape, q.dtype),),
        input_spec=postprocess_input_spec,
        output_spec=postprocess_output_spec,
        use_static_tensors=True,
        softmax_scale=softmax_scale,
    )
    dk_postprocess = modules.cjax.cutlass_call(
        flash_attention_backward_postprocess_launcher(
            modules,
            dtype=k.dtype,
            head_dim=k.shape[-1],
            tile_m=postprocess_tile_m,
            atom_layout_m=postprocess_atom_layout_m,
            arch=postprocess_arch,
            cluster_size=1,
            accum_is_gmem=True,
        ),
        output_shape_dtype=(jax.ShapeDtypeStruct(k.shape, k.dtype),),
        input_spec=postprocess_input_spec,
        output_spec=postprocess_output_spec,
        use_static_tensors=True,
        softmax_scale=softmax_scale,
    )
    dv_postprocess = modules.cjax.cutlass_call(
        flash_attention_backward_postprocess_launcher(
            modules,
            dtype=v.dtype,
            head_dim=v.shape[-1],
            tile_m=postprocess_tile_m,
            atom_layout_m=postprocess_atom_layout_m,
            arch=postprocess_arch,
            cluster_size=1,
            accum_is_gmem=True,
        ),
        output_shape_dtype=(jax.ShapeDtypeStruct(v.shape, v.dtype),),
        input_spec=postprocess_input_spec,
        output_spec=postprocess_output_spec,
        use_static_tensors=True,
        softmax_scale=1.0,
    )
    (dq,) = dq_postprocess(dq_accum)
    (dk,) = dk_postprocess(dk_accum)
    (dv,) = dv_postprocess(dv_accum)
    return dq, dk, dv


def _cutlass_attention_forward_specs(
    modules: _CutlassCuteModules, *, vector_elems: int
) -> tuple[tuple[Any, ...], Any]:
    tensor_spec = modules.cjax.TensorSpec
    qkv_spec = tensor_spec(mode=(1, 3, 2, 0), divisibility=(1, 1, 1, vector_elems), static=True)
    lse_spec = tensor_spec(divisibility=(1, 1, 1), static=True)
    metadata_spec = tensor_spec(static=True)
    return (qkv_spec, qkv_spec, qkv_spec, metadata_spec, metadata_spec), (qkv_spec, lse_spec)


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


def _cutlass_attention_backward_sm90_accum_specs(
    modules: _CutlassCuteModules, *, vector_elems: int
) -> tuple[tuple[Any, ...], Any]:
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
        sparse_cnt_spec,
        sparse_idx_spec,
        sparse_cnt_spec,
        sparse_idx_spec,
    )
    return input_spec, (scratch_spec, scratch_spec, scratch_spec)


def _cutlass_attention_backward_sm90_preprocess_specs(
    modules: _CutlassCuteModules, *, vector_elems: int
) -> tuple[tuple[Any, ...], Any]:
    tensor_spec = modules.cjax.TensorSpec
    qkv_spec = tensor_spec(mode=(0, 1, 2, 3), divisibility=(1, 1, 1, vector_elems), static=True)
    lse_spec = tensor_spec(mode=(0, 1, 2), divisibility=(1, 1, 1), static=True)
    scratch_spec = tensor_spec(mode=(0, 1, 2), static=True)
    return (qkv_spec, qkv_spec, lse_spec), (scratch_spec, scratch_spec, scratch_spec)


def _cutlass_attention_backward_sm90_postprocess_specs(
    modules: _CutlassCuteModules, *, vector_elems: int
) -> tuple[tuple[Any, ...], Any]:
    tensor_spec = modules.cjax.TensorSpec
    scratch_spec = tensor_spec(mode=(0, 1, 2), static=True)
    qkv_spec = tensor_spec(mode=(0, 1, 2, 3), divisibility=(1, 1, 1, vector_elems), static=True)
    return (scratch_spec,), (qkv_spec,)


def _packed_segment_backward_block_sparse_indices(
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    tile_m: int,
    tile_n: int,
) -> tuple[jax.Array, jax.Array]:
    """Build upstream-style backward Q-block sparse metadata for Grug masks."""
    sparse_metadata = _packed_segment_backward_block_sparse_indices_with_full(
        lower_bounds,
        valid,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    partial_block_cnt = sparse_metadata.partial_block_cnt
    mask_block_cnt = partial_block_cnt + sparse_metadata.full_block_cnt
    max_count = sparse_metadata.partial_block_idx.shape[-1]
    positions = jnp.arange(max_count, dtype=jnp.int32)
    partial_idx = jnp.where(
        positions[None, None, None, :] < partial_block_cnt[..., None],
        sparse_metadata.partial_block_idx,
        max_count,
    )
    full_idx = jnp.where(
        positions[None, None, None, :] < sparse_metadata.full_block_cnt[..., None],
        sparse_metadata.full_block_idx,
        max_count,
    )
    combined = jnp.sort(jnp.concatenate([partial_idx, full_idx], axis=-1), axis=-1)
    mask_block_idx = jnp.where(combined[..., :max_count] < max_count, combined[..., :max_count], 0)
    return mask_block_cnt, mask_block_idx


def _packed_segment_backward_block_sparse_indices_with_full(
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    tile_m: int,
    tile_n: int,
) -> _BackwardBlockSparseMetadata:
    """Build partial and full upstream-style backward Q-block sparse metadata."""
    if tile_m <= 0 or tile_n <= 0:
        raise ValueError(f"tile_m and tile_n must be positive, got {tile_m=} {tile_n=}")
    if lower_bounds.ndim != 2 or valid.ndim != 2:
        raise ValueError(f"lower_bounds and valid must have shape [B, S], got {lower_bounds.shape=} {valid.shape=}")
    if lower_bounds.shape != valid.shape:
        raise ValueError(f"lower_bounds and valid must have matching shape, got {lower_bounds.shape=} {valid.shape=}")

    batch_size, seq_len = lower_bounds.shape
    num_m_blocks = (seq_len + tile_m - 1) // tile_m
    num_n_blocks = (seq_len + tile_n - 1) // tile_n
    padded_q_len = num_m_blocks * tile_m
    q_positions = jnp.arange(padded_q_len, dtype=jnp.int32).reshape(num_m_blocks, tile_m)
    lower_padded = jnp.pad(
        lower_bounds,
        ((0, 0), (0, padded_q_len - seq_len)),
        mode="constant",
        constant_values=seq_len,
    ).reshape(batch_size, num_m_blocks, tile_m)
    valid_padded = jnp.pad(
        valid,
        ((0, 0), (0, padded_q_len - seq_len)),
        mode="constant",
        constant_values=False,
    ).reshape(batch_size, num_m_blocks, tile_m)

    n_starts = jnp.arange(num_n_blocks, dtype=jnp.int32) * tile_n
    n_ends = jnp.minimum(n_starts + tile_n, seq_len) - 1
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

    block_indices = jnp.arange(num_m_blocks, dtype=jnp.int32)
    partial_indices = jnp.where(is_partial, block_indices[None, None, :], num_m_blocks)
    full_indices = jnp.where(is_full, block_indices[None, None, :], num_m_blocks)
    sorted_partial_indices = jnp.sort(partial_indices, axis=-1)
    sorted_full_indices = jnp.sort(full_indices, axis=-1)
    mask_block_cnt = jnp.sum(is_partial.astype(jnp.int32), axis=-1)[:, None, :]
    full_block_cnt = jnp.sum(is_full.astype(jnp.int32), axis=-1)[:, None, :]
    mask_block_idx = jnp.where(sorted_partial_indices < num_m_blocks, sorted_partial_indices, 0)[:, None, :, :]
    full_block_idx = jnp.where(sorted_full_indices < num_m_blocks, sorted_full_indices, 0)[:, None, :, :]
    return _BackwardBlockSparseMetadata(
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
    seq_k_rounded = ((seq_len + tile_n - 1) // tile_n) * tile_n
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


def _cutlass_attention_backward_sm90_preprocess_output_shapes(
    q: jax.Array,
    backward_tile: tuple[int, int],
) -> tuple[jax.ShapeDtypeStruct, ...]:
    batch, seq_len, q_heads, head_dim = q.shape
    tile_m, _tile_n = backward_tile
    seq_q_rounded = ((seq_len + tile_m - 1) // tile_m) * tile_m
    head_dim_rounded = ((head_dim + 31) // 32) * 32
    scratch_q = jax.ShapeDtypeStruct((batch, q_heads, seq_q_rounded), jnp.float32)
    dq_accum = jax.ShapeDtypeStruct((batch, q_heads, seq_q_rounded * head_dim_rounded), jnp.float32)
    return scratch_q, scratch_q, dq_accum


def _cutlass_attention_backward_sm90_backward_output_shapes(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    backward_tile: tuple[int, int],
) -> tuple[jax.ShapeDtypeStruct, ...]:
    batch, seq_len, q_heads, head_dim = q.shape
    kv_heads = k.shape[2]
    tile_m, tile_n = backward_tile
    seq_q_rounded = ((seq_len + tile_m - 1) // tile_m) * tile_m
    seq_k_rounded = ((seq_len + tile_n - 1) // tile_n) * tile_n
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
) -> jax.Array:
    """FA4/CuTe attention boundary with packed causal metadata.

    Forward uses the CUTLASS/CuTe JAX FFI path. Backward is routed through a custom VJP so JAX does not
    attempt to autodiff through ``cutlass_call``.
    """
    if sm_scale is None:
        sm_scale = float(q.shape[-1] ** -0.5)
    return _segmented_flash_attention_custom_vjp(
        q,
        k,
        v,
        lower_bounds,
        valid,
        sm_scale,
        kernel_config,
    )


@partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _segmented_flash_attention_custom_vjp(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
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
    )
    return out


def _segmented_flash_attention_custom_vjp_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    softmax_scale: float,
    kernel_config: Flash4CuteKernelConfig,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
    out, lse = segmented_flash_attention_forward(
        q,
        k,
        v,
        lower_bounds,
        valid,
        softmax_scale=softmax_scale,
        kernel_config=kernel_config,
    )
    return out, (q, k, v, out, lse, lower_bounds, valid)


def _segmented_flash_attention_custom_vjp_bwd(
    softmax_scale: float,
    kernel_config: Flash4CuteKernelConfig,
    residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    cotangent: jax.Array | jax.custom_derivatives.SymbolicZero,
) -> tuple[jax.Array | None, jax.Array | None, jax.Array | None, None, None]:
    q, k, v, out, lse, lower_bounds, valid = residuals
    if isinstance(cotangent, jax.custom_derivatives.SymbolicZero):
        return jnp.zeros_like(q), jnp.zeros_like(k), jnp.zeros_like(v), None, None
    dq, dk, dv = segmented_flash_attention_backward(
        q,
        k,
        v,
        out,
        cotangent.astype(q.dtype),
        lse,
        lower_bounds,
        valid,
        softmax_scale=softmax_scale,
        kernel_config=kernel_config,
    )
    return dq, dk, dv, None, None


_segmented_flash_attention_custom_vjp.defvjp(
    _segmented_flash_attention_custom_vjp_fwd,
    _segmented_flash_attention_custom_vjp_bwd,
)


def _validate_forward_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    softmax_scale: float,
) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"q/k/v must be BSHD tensors, got q={q.shape}, k={k.shape}, v={v.shape}")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(f"q/k/v batch sizes must match, got q={q.shape}, k={k.shape}, v={v.shape}")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError(f"q/k/v sequence lengths must match, got q={q.shape}, k={k.shape}, v={v.shape}")
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
    batch, seq_len, q_heads, _ = q.shape
    kv_heads = k.shape[2]
    if q_heads % kv_heads != 0:
        raise ValueError(f"Hq must be divisible by Hkv for GQA, got q={q.shape}, k={k.shape}")
    expected_n_blocks = (seq_len + tile_n - 1) // tile_n
    expected_m_blocks = (seq_len + tile_m - 1) // tile_m
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
    "cutlass_cute_available",
    "fa4_cute_attention_forward",
    "require_cutlass_cute",
    "segmented_flash_attention_backward",
    "segmented_flash_attention_backward_sm90_native",
    "segmented_flash_attention_forward",
]
