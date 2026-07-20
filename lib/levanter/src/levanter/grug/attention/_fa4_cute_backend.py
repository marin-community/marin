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

from levanter.grug.attention._core import align_kv_heads
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
    bias: jax.Array | None = None,
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
        bias: Optional learned relative-position band, shape [B, S, Hq, window]. When
            given, ``bias[b, i, h, i-j]`` is added to the scaled logit of every in-window
            causal pair. When ``None`` the kernel is byte-for-byte the bias-free path.

    Returns:
        ``(out, lse)`` where ``out`` has shape [B, S, Hq, Dv] and ``lse`` has
        shape [B, Hq, S]. The backward kernel consumes both tensors.
    """
    _validate_forward_inputs(q, k, v, lower_bounds, valid, softmax_scale=softmax_scale)
    rel_pos_window = _validate_bias(q, bias)
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
        rel_pos_window=rel_pos_window,
    )
    input_spec, output_spec = _cutlass_attention_forward_specs(modules, vector_elems=8, include_bias=bias is not None)
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
    if bias is None:
        return call(q, k, v, lower_bounds, valid.astype(jnp.int32))
    return call(q, k, v, lower_bounds, valid.astype(jnp.int32), bias)


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
    bias: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return dq/dk/dv for FA4/CuTe packed-segment attention.

    When ``bias`` (the ``[B, S, Hq, window]`` relative-position band) is given, the recomputed
    score tile is biased so dq/dk/dv match the biased forward. The band's own gradient is not
    produced here (the custom VJP forms it in JAX from the forward residuals).
    """
    _validate_forward_inputs(q, k, v, lower_bounds, valid, softmax_scale=softmax_scale)
    _validate_backward_inputs(q, k, v, out, dout, lse)
    rel_pos_window = _validate_bias(q, bias)
    try:
        modules = _import_cutlass_cute()
    except Exception as exc:
        raise _optional_dependency_error() from exc

    qhead_per_kvhead = q.shape[2] // k.shape[2]
    # Route every SM90 head-dim the native warp-specialized backward supports through it: GQA D128
    # (ratio > 1) and MLA qk=192 / v=128 MHA (ratio == 1). The segmented Sm120 fallback is ~40x
    # slower per call and dominated (~34%) the MLA step time in profiling.
    if kernel_config.sm90_backward is not None and q.shape[-1] in (128, 192):
        if bias is not None:
            raise NotImplementedError(
                "Relative-position bias in the FA4/CuTe backward is only wired for the segmented "
                "(SM80/SM120) path; the native SM90 warp-specialized backward is out of scope."
            )
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
        rel_pos_window=rel_pos_window,
    )
    input_spec, output_spec = _cutlass_attention_backward_specs(
        modules,
        vector_elems=8,
        qhead_per_kvhead=qhead_per_kvhead,
        include_bias=bias is not None,
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
    if bias is None:
        dq, dk, dv, *_scratch = call(q, k, v, out, dout, lse, lower_bounds, valid.astype(jnp.int32))
    else:
        dq, dk, dv, *_scratch = call(q, k, v, out, dout, lse, lower_bounds, valid.astype(jnp.int32), bias)
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
    # MHA (ratio == 1, MLA) is the degenerate GQA group of size 1: dK/dV accumulators are per-head,
    # so the native accumulation path is correct without a grouped reduction.
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

    mha = q.shape[2] == k.shape[2]
    backward_input_spec, backward_output_spec = _cutlass_attention_backward_sm90_accum_specs(
        modules, vector_elems=8, mha=mha
    )
    backward_output_shape_dtype = _cutlass_attention_backward_sm90_backward_output_shapes(
        q, k, v, sm90_config.tile, mha=mha
    )
    backward_call = modules.cjax.cutlass_call(
        backward_launcher,
        output_shape_dtype=backward_output_shape_dtype,
        input_spec=backward_input_spec,
        output_spec=backward_output_spec,
        use_static_tensors=True,
        softmax_scale=softmax_scale,
    )
    dq_accum, dk_out, dv_out = backward_call(
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
    postprocess_tile_m = sm90_config.tile[0]
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
    if mha:
        # MHA: dK/dV came out of the kernel epilogue as final softmax-scaled bf16 — no postprocess.
        return dq, dk_out, dv_out
    (dk,) = dk_postprocess(dk_out)
    (dv,) = dv_postprocess(dv_out)
    return dq, dk, dv


def _cutlass_attention_forward_specs(
    modules: _CutlassCuteModules, *, vector_elems: int, include_bias: bool = False
) -> tuple[tuple[Any, ...], Any]:
    tensor_spec = modules.cjax.TensorSpec
    qkv_spec = tensor_spec(mode=(1, 3, 2, 0), divisibility=(1, 1, 1, vector_elems), static=True)
    lse_spec = tensor_spec(divisibility=(1, 1, 1), static=True)
    metadata_spec = tensor_spec(static=True)
    input_spec = (qkv_spec, qkv_spec, qkv_spec, metadata_spec, metadata_spec)
    if include_bias:
        # Keep the bias band in its native [B, S, Hq, window] layout (identity mode) so the
        # kernel can index mBias[b, i, h, offset] directly.
        bias_spec = tensor_spec(static=True)
        input_spec = (*input_spec, bias_spec)
    return input_spec, (qkv_spec, lse_spec)


def _validate_bias(q: jax.Array, bias: jax.Array | None) -> int | None:
    """Validate the optional relative-position bias band and return its window width."""
    if bias is None:
        return None
    expected = (q.shape[0], q.shape[1], q.shape[2])
    if bias.ndim != 4 or bias.shape[:3] != expected:
        raise ValueError(f"bias must have shape [B, S, Hq, window]={expected}+(window,), got {bias.shape}")
    if bias.dtype != q.dtype:
        raise TypeError(f"bias dtype must match q dtype {q.dtype}, got {bias.dtype}")
    window = bias.shape[3]
    if window <= 0:
        raise ValueError(f"bias window must be positive, got {window}")
    return window


def _cutlass_attention_backward_specs(
    modules: _CutlassCuteModules, *, vector_elems: int, qhead_per_kvhead: int, include_bias: bool = False
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
    if include_bias:
        # Bias band [B, S, Hq, window] in native layout (matches the backward's identity BSHD mode).
        bias_spec = tensor_spec(mode=(0, 1, 2, 3), static=True)
        input_spec = (*input_spec, bias_spec)
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
    modules: _CutlassCuteModules, *, vector_elems: int, mha: bool = False
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
    # For MHA (qhead_per_kvhead == 1) the kernel writes final bf16 dK/dV directly in [B, S, H, D]
    # layout (already softmax-scaled), so their outputs are 4D qkv_spec instead of the fp32
    # accumulator scratch tensors used by the GQA (grouped-reduction) path.
    output_spec = (scratch_spec, qkv_spec, qkv_spec) if mha else (scratch_spec, scratch_spec, scratch_spec)
    return input_spec, output_spec


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
    *,
    mha: bool = False,
) -> tuple[jax.ShapeDtypeStruct, ...]:
    batch, seq_len, q_heads, head_dim = q.shape
    kv_heads = k.shape[2]
    tile_m, tile_n = backward_tile
    seq_q_rounded = ((seq_len + tile_m - 1) // tile_m) * tile_m
    seq_k_rounded = ((seq_len + tile_n - 1) // tile_n) * tile_n
    head_dim_rounded = ((head_dim + 31) // 32) * 32
    head_dim_v_rounded = ((v.shape[-1] + 31) // 32) * 32
    dq_accum = jax.ShapeDtypeStruct((batch, q_heads, seq_q_rounded * head_dim_rounded), jnp.float32)
    if mha:
        # MHA: the kernel emits final bf16 dK/dV in the input [B, S, H, D] layout (no grouped
        # accumulation, no dK/dV postprocess). dQ still uses the fp32 atomic accumulator.
        dk = jax.ShapeDtypeStruct(k.shape, k.dtype)
        dv = jax.ShapeDtypeStruct(v.shape, v.dtype)
        return dq_accum, dk, dv
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
    bias: jax.Array | None = None,
) -> jax.Array:
    """FA4/CuTe attention boundary with packed causal metadata.

    Forward uses the CUTLASS/CuTe JAX FFI path. Backward is routed through a custom VJP so JAX does not
    attempt to autodiff through ``cutlass_call``. When ``bias`` (the ``[B, S, Hq, window]``
    relative-position band) is given it is threaded through both the biased forward/backward kernels
    and a JAX-side band-gradient path so ``d_bias`` composes with the model's einsum VJP.
    """
    if sm_scale is None:
        sm_scale = float(q.shape[-1] ** -0.5)
    if bias is None:
        return _segmented_flash_attention_custom_vjp(
            q,
            k,
            v,
            lower_bounds,
            valid,
            sm_scale,
            kernel_config,
        )
    return _segmented_flash_attention_bias_custom_vjp(
        q,
        k,
        v,
        lower_bounds,
        valid,
        bias,
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


@partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _segmented_flash_attention_bias_custom_vjp(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    bias: jax.Array,
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
        bias=bias,
    )
    return out


def _segmented_flash_attention_bias_custom_vjp_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    bias: jax.Array,
    softmax_scale: float,
    kernel_config: Flash4CuteKernelConfig,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    out, lse = segmented_flash_attention_forward(
        q,
        k,
        v,
        lower_bounds,
        valid,
        softmax_scale=softmax_scale,
        kernel_config=kernel_config,
        bias=bias,
    )
    return out, (q, k, v, out, lse, lower_bounds, valid, bias)


def _segmented_flash_attention_bias_custom_vjp_bwd(
    softmax_scale: float,
    kernel_config: Flash4CuteKernelConfig,
    residuals: tuple[jax.Array, ...],
    cotangent: jax.Array | jax.custom_derivatives.SymbolicZero,
) -> tuple[jax.Array | None, ...]:
    q, k, v, out, lse, lower_bounds, valid, bias = residuals
    if isinstance(cotangent, jax.custom_derivatives.SymbolicZero):
        zeros = (jnp.zeros_like(q), jnp.zeros_like(k), jnp.zeros_like(v), None, None, jnp.zeros_like(bias))
        return zeros
    dout = cotangent.astype(q.dtype)
    dq, dk, dv = segmented_flash_attention_backward(
        q,
        k,
        v,
        out,
        dout,
        lse,
        lower_bounds,
        valid,
        softmax_scale=softmax_scale,
        kernel_config=kernel_config,
        bias=bias,
    )
    d_bias = _rel_pos_bias_band_grad(
        q,
        k,
        v,
        out,
        dout,
        lse,
        lower_bounds,
        valid,
        bias,
        softmax_scale=softmax_scale,
    )
    return dq, dk, dv, None, None, d_bias


_segmented_flash_attention_bias_custom_vjp.defvjp(
    _segmented_flash_attention_bias_custom_vjp_fwd,
    _segmented_flash_attention_bias_custom_vjp_bwd,
)


def _rel_pos_bias_band_grad(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    dout: jax.Array,
    lse: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    bias: jax.Array,
    *,
    softmax_scale: float,
) -> jax.Array:
    """Gradient of the loss w.r.t. the relative-position bias band ``[B, S, Hq, window]``.

    The band adds 1:1 to the scaled logit, so ``d_bias[b, i, h, r] == dL/ds_{i, j=i-r}`` where
    ``s`` is the (biased, masked) scaled score. Using the flash residuals this is the exact softmax
    input gradient restricted to the causal in-window band::

        P_ij  = exp(scale * q_i·k_j + bias[b,i,h,i-j] - lse_i)
        dP_ij = dO_i · v_j
        D_i   = sum_d dO_i,d * O_i,d
        g_ij  = P_ij * (dP_ij - D_i)

    computed only for ``0 <= i-j < window`` and the same causal/segment predicate the kernel uses.
    This mirrors the kernel's own dQ/dK/dV recompute (which uses the biased P), so the band gradient
    is consistent with dq/dk/dv without scattering out of the fused CuTe kernel.
    """
    batch, seq_len, q_heads, head_dim = q.shape
    window = bias.shape[3]
    compute_dtype = q.dtype

    k_f = align_kv_heads(k, num_q_heads=q_heads).astype(jnp.float32)  # [B, S, Hq, D]
    v_f = align_kv_heads(v, num_q_heads=q_heads).astype(jnp.float32)  # [B, S, Hq, Dv]
    q_f = q.astype(jnp.float32)
    o_f = out.astype(jnp.float32)
    do_f = dout.astype(jnp.float32)
    bias_f = bias.astype(jnp.float32)  # [B, S, Hq, window]

    # D_i = rowsum(dO ∘ O) over head_dim -> [B, S, Hq]; lse arrives as [B, Hq, S].
    d_row = jnp.sum(do_f * o_f, axis=-1)  # [B, S, Hq]
    lse_bshq = jnp.transpose(lse, (0, 2, 1))  # [B, S, Hq]

    # A dense per-(i, offset) gather would materialize [B, S, window, Hq, D] -- hundreds of GB at
    # seq 4096 / window 1024. Bound peak memory by scanning over blocks of query rows; only the query
    # axis is chunked (keys stay full), so the validated per-row math runs unchanged per block.
    block = min(64, seq_len)
    num_blocks = (seq_len + block - 1) // block
    padded = num_blocks * block
    offsets = jnp.arange(window)

    def _pad_q(x: jax.Array) -> jax.Array:
        pad = [(0, 0)] * x.ndim
        pad[1] = (0, padded - seq_len)
        return jnp.pad(x, pad)

    q_p = _pad_q(q_f).reshape(batch, num_blocks, block, q_heads, head_dim)
    do_p = _pad_q(do_f).reshape(batch, num_blocks, block, q_heads, v_f.shape[-1])
    d_row_p = _pad_q(d_row).reshape(batch, num_blocks, block, q_heads)
    lse_p = _pad_q(lse_bshq).reshape(batch, num_blocks, block, q_heads)
    bias_p = _pad_q(bias_f).reshape(batch, num_blocks, block, q_heads, window)
    lb_p = _pad_q(lower_bounds).reshape(batch, num_blocks, block)
    valid_p = _pad_q(valid).reshape(batch, num_blocks, block)

    def _one_block(block_idx: jax.Array) -> jax.Array:
        qi = block_idx * block + jnp.arange(block)  # [block] global query indices
        q_blk = q_p[:, block_idx]  # [B, block, Hq, D]
        do_blk = do_p[:, block_idx]
        d_row_blk = d_row_p[:, block_idx]
        lse_blk = lse_p[:, block_idx]
        bias_blk = bias_p[:, block_idx]  # [B, block, Hq, window]
        lb_blk = lb_p[:, block_idx]  # [B, block]
        valid_blk = valid_p[:, block_idx]  # [B, block]

        j = qi[:, None] - offsets[None, :]  # [block, window]
        j_cl = jnp.clip(j, 0, seq_len - 1)
        k_band = jnp.take(k_f, j_cl, axis=1)  # [B, block, window, Hq, D]
        v_band = jnp.take(v_f, j_cl, axis=1)  # [B, block, window, Hq, Dv]

        qk = jnp.einsum("bihd,biwhd->biwh", q_blk, k_band) * softmax_scale
        dp = jnp.einsum("bihd,biwhd->biwh", do_blk, v_band)
        scores = qk + jnp.transpose(bias_blk, (0, 1, 3, 2))  # [B, block, window, Hq]
        p = jnp.exp(scores - lse_blk[:, :, None, :])
        g = p * (dp - d_row_blk[:, :, None, :])  # [B, block, window, Hq]

        in_band = (
            (j[None] >= 0)
            & (j[None] >= lb_blk[:, :, None])
            & (j[None] <= qi[None, :, None])
            & valid_blk[:, :, None]
            & (qi[None, :, None] < seq_len)
        )  # [B, block, window]
        g = jnp.where(in_band[:, :, :, None], g, 0.0)
        return jnp.transpose(g, (0, 1, 3, 2))  # [B, block, Hq, window]

    d_bias_blocks = jax.lax.map(_one_block, jnp.arange(num_blocks))  # [num_blocks, B, block, Hq, window]
    d_bias = jnp.transpose(d_bias_blocks, (1, 0, 2, 3, 4)).reshape(batch, padded, q_heads, window)
    return d_bias[:, :seq_len].astype(compute_dtype)


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
    # Asymmetric V head dim (MLA qk=192 / v=128) is supported by the kernel; V/O layouts are
    # sized on head_dim_v independently of the q/k head_dim. Both must be multiples of 8.
    if v.shape[-1] % 8 != 0:
        raise NotImplementedError(f"gpu_fa4_cute_attention requires Dv % 8 == 0, got v={v.shape}")
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
