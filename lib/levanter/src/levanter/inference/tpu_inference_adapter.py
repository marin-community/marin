# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the `tpu-inference` ragged paged attention kernel."""

from __future__ import annotations

import importlib.util

import dataclasses
import jax
import jax.numpy as jnp
from haliax import NamedArray
from haliax.partitioning import shard_map
from jax.sharding import PartitionSpec

from levanter.inference.page_table import PageBatchInfo
from levanter.layers.kv_cache import KvPageCache


def is_available() -> bool:
    """Return whether the `tpu_inference` package is importable."""

    return importlib.util.find_spec("tpu_inference") is not None


def _request_distribution(cu_q_lens: jax.Array, num_seqs: jax.Array) -> jax.Array:
    """Return the v3 RPA request distribution for Levanter's current sequence order."""

    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    valid = jnp.arange(q_lens.shape[0], dtype=jnp.int32) < num_seqs
    all_decode = jnp.all(jnp.where(valid, q_lens == 1, True))
    decode_distribution = jnp.stack([num_seqs, num_seqs, num_seqs]).astype(jnp.int32)
    mixed_distribution = jnp.stack(
        [
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            num_seqs.astype(jnp.int32),
        ]
    )
    return jnp.where(all_decode, decode_distribution, mixed_distribution)


def _pad_head_dim(array: jax.Array) -> tuple[jax.Array, int]:
    head_dim = array.shape[-1]
    padded_head_dim = ((head_dim + 127) // 128) * 128
    if padded_head_dim == head_dim:
        return array, head_dim
    return jnp.pad(array, [(0, 0)] * (array.ndim - 1) + [(0, padded_head_dim - head_dim)]), head_dim


def _pack_kv_cache_for_tpu_inference(kv_pages: NamedArray) -> tuple[jax.Array, int, int]:
    from tpu_inference.kernels.ragged_paged_attention.v3.util import align_to, get_dtype_packing

    cache = kv_pages.array
    original_kv_heads_x2 = cache.shape[2]
    original_head_dim = cache.shape[3]
    packing = get_dtype_packing(cache.dtype)
    padded_kv_heads_x2 = align_to(original_kv_heads_x2, packing)
    padded_head_dim = align_to(original_head_dim, 128)

    if padded_kv_heads_x2 != original_kv_heads_x2 or padded_head_dim != original_head_dim:
        cache = jnp.pad(
            cache,
            (
                (0, 0),
                (0, 0),
                (0, padded_kv_heads_x2 - original_kv_heads_x2),
                (0, padded_head_dim - original_head_dim),
            ),
        )

    return (
        cache.reshape(
            cache.shape[0],
            cache.shape[1],
            padded_kv_heads_x2 // packing,
            packing,
            padded_head_dim,
        ),
        original_kv_heads_x2,
        original_head_dim,
    )


def _unpack_kv_cache_from_tpu_inference(
    kv_cache: jax.Array,
    axes,
    original_kv_heads_x2: int,
    original_head_dim: int,
) -> NamedArray:
    unpacked = kv_cache.reshape(
        kv_cache.shape[0], kv_cache.shape[1], kv_cache.shape[2] * kv_cache.shape[3], kv_cache.shape[4]
    )
    unpacked = unpacked[:, :, :original_kv_heads_x2, :original_head_dim]
    return NamedArray(unpacked, axes)


def _flatten_q(q: NamedArray) -> tuple[jax.Array, tuple]:
    q_flat = q.flatten_axes(("kv_head", "q_heads_per_group"), "heads")
    q_padded, _ = _pad_head_dim(q_flat.array)
    return q_padded, q.axes


def _query_array_for_out_dtype(q_array: jax.Array, out_dtype: jnp.dtype | None) -> jax.Array:
    if out_dtype is None:
        return q_array
    out_dtype = jnp.dtype(out_dtype)
    if q_array.dtype == out_dtype:
        return q_array
    # tpu-inference v3 ties its output buffer shape/packing to Q dtype, so f32
    # output requires f32 Q even when the KV cache remains bf16.
    return q_array.astype(out_dtype)


def _kv_array_for_cache_dtype(kv_array: jax.Array, kv_cache: NamedArray) -> jax.Array:
    cache_dtype = kv_cache.array.dtype
    if kv_array.dtype == cache_dtype:
        return kv_array
    return kv_array.astype(cache_dtype)


def _unflatten_output(
    output: jax.Array, q_axes: tuple, num_kv_heads: int, q_heads_per_group: int, head_dim: int
) -> NamedArray:
    output = output[:, :, :head_dim]
    output = output.reshape(output.shape[0], num_kv_heads, q_heads_per_group, head_dim)
    return NamedArray(output, q_axes)


def _ragged_paged_attention_partition_specs() -> tuple[tuple[PartitionSpec, ...], tuple[PartitionSpec, PartitionSpec]]:
    replicated = PartitionSpec()
    qkv_spec = PartitionSpec(None, "model", None)
    kv_cache_spec = PartitionSpec(None, None, "model", None, None)
    return (
        (
            qkv_spec,
            qkv_spec,
            qkv_spec,
            kv_cache_spec,
            replicated,
            replicated,
            replicated,
            replicated,
        ),
        (qkv_spec, kv_cache_spec),
    )


def _sharded_ragged_paged_attention(
    q_array: jax.Array,
    k_array: jax.Array,
    v_array: jax.Array,
    packed_cache: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float | jax.Array,
    soft_cap: float | None,
    out_dtype: jnp.dtype | None,
    vmem_limit_bytes: int | None,
) -> tuple[jax.Array, jax.Array]:
    from tpu_inference.kernels.ragged_paged_attention.v3.kernel import ragged_paged_attention

    def call_kernel(
        q_array: jax.Array,
        k_array: jax.Array,
        v_array: jax.Array,
        packed_cache: jax.Array,
        seq_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return ragged_paged_attention(
            q_array,
            k_array,
            v_array,
            packed_cache,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            soft_cap=soft_cap,
            out_dtype=out_dtype,
            vmem_limit_bytes=vmem_limit_bytes,
        )

    in_specs, out_specs = _ragged_paged_attention_partition_specs()
    return shard_map(
        call_kernel,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=False,
    )(q_array, k_array, v_array, packed_cache, seq_lens, page_indices, cu_q_lens, distribution)


def paged_attention_with_kv_update(
    q: NamedArray,
    new_k: NamedArray,
    new_v: NamedArray,
    kv_cache: KvPageCache,
    batch_info: PageBatchInfo,
    *,
    sm_scale: float | jax.Array,
    soft_cap: float | None,
    out_dtype: jnp.dtype | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[NamedArray, KvPageCache]:
    """Run `tpu-inference` v3 ragged paged attention with fused KV cache update."""

    q_array, q_axes = _flatten_q(q)
    q_array = _query_array_for_out_dtype(q_array, out_dtype)
    k_array, original_head_dim = _pad_head_dim(_kv_array_for_cache_dtype(new_k.array, kv_cache.kv_pages))
    v_array, _ = _pad_head_dim(_kv_array_for_cache_dtype(new_v.array, kv_cache.kv_pages))
    packed_cache, original_kv_heads_x2, original_cache_head_dim = _pack_kv_cache_for_tpu_inference(kv_cache.kv_pages)

    if original_head_dim != original_cache_head_dim:
        raise ValueError(
            f"new key/value head dimension {original_head_dim} does not match KV cache head dimension "
            f"{original_cache_head_dim}"
        )

    output, updated_cache = _sharded_ragged_paged_attention(
        q_array,
        k_array,
        v_array,
        packed_cache,
        batch_info.seq_lens.array.astype(jnp.int32),
        batch_info.page_indices.array.reshape(-1).astype(jnp.int32),
        batch_info.cu_q_lens.array.astype(jnp.int32),
        _request_distribution(batch_info.cu_q_lens.array.astype(jnp.int32), batch_info.num_seqs.astype(jnp.int32)),
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        out_dtype=out_dtype,
        vmem_limit_bytes=vmem_limit_bytes,
    )

    attn = _unflatten_output(
        output,
        q_axes,
        q.axis_size("kv_head"),
        q.axis_size("q_heads_per_group"),
        original_head_dim,
    )
    updated_pages = _unpack_kv_cache_from_tpu_inference(
        updated_cache,
        kv_cache.kv_pages.axes,
        original_kv_heads_x2,
        original_cache_head_dim,
    )
    return attn, dataclasses.replace(kv_cache, kv_pages=updated_pages)
