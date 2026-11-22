# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Array-first attention over Levanter's interleaved paged KV cache."""

from functools import partial
from math import gcd
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.ragged_paged_attention import ragged_paged_attention as tpu_ragged_paged_attention
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.kernel import get_min_heads_per_blk
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

from levanter.kernels.pallas.autotune_utils import named_sharding_of
from levanter.kernels.pallas.ragged_paged_attention_v3 import ragged_paged_attention as tpu_ragged_paged_attention_v3

TPU_HEAD_ALIGNMENT = 128
FUSED_KV_APPEND_MAX_TOKENS = 8
PagedAttentionImplementation = Literal["reference", "tpu", "tpu_fp32", "tpu_v3"]
_TPU_KV_HEAD_GROUP = 8
_FP32_QUERIES_PER_BLOCK = 128
_FP32_KV_PAGES_PER_BLOCK = 8
_RPA_V3_VMEM_LIMIT_BYTES = {
    pltpu.ChipVersion.TPU_V5P: 50_000 * 1024,
    pltpu.ChipVersion.TPU_V6E: 98_304 * 1024,
}


def ragged_paged_attention(
    q: jax.Array,
    kv_pages: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    implementation: PagedAttentionImplementation | None = None,
) -> jax.Array:
    """Attend to cached prefixes and new tokens in a mixed prefill/decode batch.

    Args:
        q: Queries [tokens, kv_heads, query_heads_per_group, head_dim].
        kv_pages: Interleaved K/V cache [pages, slots, 2 * kv_heads, head_dim].
        kv_lens: Total cached lengths [sequences], including the new queries.
        page_indices: Physical page IDs [sequences, pages_per_sequence].
        cu_q_lens: Cumulative new-query counts [sequences + 1].
        num_seqs: Scalar count of active sequences; unused rows may contain -1.
        sm_scale: Query/key logit multiplier.
        sliding_window: Number of visible tokens, including the query itself.
        soft_cap: Optional tanh logit cap, applied before masking.
        implementation: TPU Pallas or portable reference; defaults to TPU on TPU.
            `tpu_fp32` uses FP32 operands in the kernel and preserves the query dtype on output.
            `tpu_v3` uses BF16 KV DMA and FP32 value accumulation on v5p and v6e. Small sliding-window
            batches use the compact FP32 kernel.

    Query positions start at ``kv_lens - diff(cu_q_lens)`` for each sequence.
    Padding queries produce zero. Default TPU dispatch uses the tiled kernel for
    FP32 inputs and the reference for lower-precision inputs.
    GPU and CPU default to the reference.
    Only the KV-head axis is partitioned; sequence metadata and pages are replicated.
    """
    if implementation is None:
        implementation = "tpu" if jax.default_backend() == "tpu" else "reference"
    if implementation not in ("tpu", "reference", "tpu_fp32", "tpu_v3"):
        raise ValueError(f"Unknown paged attention implementation: {implementation}")
    if implementation in ("tpu_fp32", "tpu_v3") and jax.default_backend() != "tpu":
        raise ValueError(f"{implementation} paged attention requires a TPU")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be positive")
    if soft_cap is not None and soft_cap <= 0:
        raise ValueError("soft_cap must be positive")
    if q.ndim != 4 or kv_pages.ndim != 4 or kv_pages.shape[2:] != (2 * q.shape[1], q.shape[3]):
        raise ValueError("Expected grouped queries and interleaved KV pages with matching heads and head_dim")

    fp32 = partial(
        _tpu_kernel_attention,
        num_queries_per_block=min(_FP32_QUERIES_PER_BLOCK, q.shape[0]),
        num_kv_pages_per_block=min(_FP32_KV_PAGES_PER_BLOCK, page_indices.shape[-1]),
    )
    if implementation == "tpu_v3":
        if pltpu.get_tpu_info().chip_version not in _RPA_V3_VMEM_LIMIT_BYTES:
            raise ValueError("tpu_v3 paged attention requires TPU v5p or v6e")
        short_window = sliding_window is not None and q.shape[0] <= _FP32_QUERIES_PER_BLOCK
        supports_v3 = (
            not short_window
            and q.dtype == kv_pages.dtype == jnp.bfloat16
            and q.shape[-1] == kv_pages.shape[1] == TPU_HEAD_ALIGNMENT
            and isinstance(sm_scale, (float, int))
        )
        backend = _tpu_v3_attention if supports_v3 else fp32
    else:
        backend = {"tpu": _tpu_attention, "tpu_fp32": fp32, "reference": _reference_attention}[implementation]
    fn = partial(
        backend,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    q_sharding = _attention_sharding(q)
    if q_sharding is not None:
        q_spec = tuple(q_sharding.spec) + (None,) * (q.ndim - len(q_sharding.spec))
        heads = q_spec[1]
        # Cache updates can erase a size-one head-axis annotation.
        kv_spec = P(None, None, heads, None)
        kv_pages = reshard(kv_pages, NamedSharding(q_sharding.mesh, kv_spec))
        fn = jax.shard_map(
            fn,
            mesh=q_sharding.mesh,
            in_specs=(P(None, heads, None, None), kv_spec, P(), P(), P(), P()),
            out_specs=P(None, heads, None, None),
            check_vma=False,
        )
    return fn(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs)


def _attention_sharding(q: jax.Array) -> NamedSharding | None:
    sharding = named_sharding_of(q)
    if sharding is None or sharding.mesh.empty:
        return None
    if any(axis is not None for index, axis in enumerate(sharding.spec) if index != 1):
        raise ValueError("Paged attention supports sharding only the KV-head axis of q")
    return sharding


def _tpu_v3_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, *, sm_scale, sliding_window, soft_cap):
    output, _ = _tpu_v3(
        q,
        kv_pages,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_seqs,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    return output


def ragged_paged_attention_append(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_pages: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int | None,
) -> tuple[jax.Array, jax.Array]:
    """Append BF16 keys and values during a small RPA v3 decode batch."""
    if jax.default_backend() != "tpu" or pltpu.get_tpu_info().chip_version not in _RPA_V3_VMEM_LIMIT_BYTES:
        raise ValueError("Fused paged KV append requires TPU v5p or v6e")
    if q.shape[0] > FUSED_KV_APPEND_MAX_TOKENS:
        raise ValueError("Fused paged KV append is only supported for small decode batches")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype != kv_pages.dtype or q.dtype != jnp.bfloat16:
        raise ValueError("Fused paged KV append requires BF16 queries, keys, values, and cache pages")
    if q.shape[-1] != TPU_HEAD_ALIGNMENT or kv_pages.shape[1] != TPU_HEAD_ALIGNMENT:
        raise ValueError("Fused paged KV append requires 128-wide heads and 128-token pages")

    def attend(q, k, v, pages, lengths, indices, offsets, num_seqs):
        return _tpu_v3(
            q,
            pages,
            lengths,
            indices,
            offsets,
            num_seqs,
            k=k,
            v=v,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=None,
        )

    sharding = _attention_sharding(q)
    if sharding is not None:
        heads = sharding.spec[1] if len(sharding.spec) > 1 else None
        q_spec, update_spec, page_spec = P(None, heads, None, None), P(None, heads, None), P(None, None, heads, None)
        kv_pages = reshard(kv_pages, NamedSharding(sharding.mesh, page_spec))
        attend = jax.shard_map(
            attend,
            mesh=sharding.mesh,
            in_specs=(q_spec, update_spec, update_spec, page_spec, P(), P(), P(), P()),
            out_specs=(q_spec, page_spec),
            check_vma=False,
        )
    return attend(q, k, v, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs)


def _tpu_v3(
    q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, *, sm_scale, sliding_window, soft_cap, k=None, v=None
):
    append = k is not None
    dummy_kv = jnp.zeros((q.shape[0], q.shape[1], q.shape[-1]), dtype=q.dtype)
    active = jnp.arange(kv_lens.shape[0]) < num_seqs
    decode = jnp.where(jnp.all((jnp.diff(cu_q_lens) == 1) | ~active), num_seqs, 0)
    pages = jnp.where((page_indices >= 0) & (page_indices < kv_pages.shape[0]), page_indices, 0)
    decode_block = (512 if sliding_window is not None else 1024) if append else 2048
    output, updated = tpu_ragged_paged_attention_v3(
        q.reshape(q.shape[0], -1, q.shape[-1]),
        dummy_kv if k is None else k,
        dummy_kv if v is None else v,
        kv_pages.reshape(*kv_pages.shape[:2], q.shape[1], 2, q.shape[-1]),
        jnp.maximum(kv_lens, 0),
        pages.reshape(-1),
        cu_q_lens,
        jnp.stack((decode, decode, num_seqs)),
        update_kv_cache=append,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        sliding_window=sliding_window,
        out_dtype=jnp.float32,
        d_block_sizes=(1, decode_block, 1, 512),
        m_block_sizes=(min(q.shape[0], 128), 1024, min(q.shape[0], 64), 512),
        vmem_limit_bytes=_RPA_V3_VMEM_LIMIT_BYTES[pltpu.get_tpu_info().chip_version],
    )
    valid = jnp.arange(q.shape[0]) < cu_q_lens[num_seqs]
    return jnp.where(valid[:, None, None, None], output.reshape(q.shape), 0), updated.reshape(kv_pages.shape)


class _QueryMetadata(NamedTuple):
    sequence: jax.Array
    position: jax.Array
    valid: jax.Array


def _query_metadata(q, kv_lens, cu_q_lens, num_seqs) -> _QueryMetadata:
    token = jnp.arange(q.shape[0])
    active = jnp.arange(kv_lens.shape[0]) < num_seqs.reshape(())
    belongs = active[:, None] & (token >= cu_q_lens[:-1, None]) & (token < cu_q_lens[1:, None])
    seq = jnp.argmax(belongs, axis=0)
    valid = jnp.any(belongs, axis=0)
    position = kv_lens[seq] - (cu_q_lens[seq + 1] - cu_q_lens[seq]) + token - cu_q_lens[seq]
    return _QueryMetadata(seq, position, valid)


class _ReferenceState(NamedTuple):
    output: jax.Array
    denominator: jax.Array
    maximum: jax.Array


def _reference_attention(
    q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, *, sm_scale, sliding_window, soft_cap
):
    # Stream pages so decode memory does not scale as tokens * maximum context * head_dim.
    metadata = _query_metadata(q, kv_lens, cu_q_lens, num_seqs)
    seq, position, valid = metadata.sequence, metadata.position, metadata.valid
    page_size = kv_pages.shape[1]
    initial = _ReferenceState(
        jnp.zeros(q.shape, jnp.float32),
        jnp.zeros(q.shape[:-1], jnp.float32),
        jnp.full(q.shape[:-1], -jnp.inf, jnp.float32),
    )

    def attend_page(page, state):
        output, denominator, maximum = state
        physical = page_indices[seq, page]
        cached = kv_pages[jnp.maximum(physical, 0)].astype(jnp.float32)
        k, v = cached[:, :, 0::2], cached[:, :, 1::2]
        key_position = page * page_size + jnp.arange(page_size)
        allowed = valid[:, None] & (physical[:, None] >= 0) & (key_position <= position[:, None])
        allowed &= key_position < kv_lens[seq, None]
        if sliding_window is not None:
            allowed &= key_position > position[:, None] - sliding_window
        scores = jnp.einsum("thgd,tshd->thgs", q.astype(jnp.float32), k, precision=jax.lax.Precision.HIGHEST)
        scores = scores * sm_scale
        if soft_cap is not None:
            scores = soft_cap * jax.lax.tanh(scores / soft_cap, accuracy=jax.lax.AccuracyMode.HIGHEST)
        scores = jnp.where(allowed[:, None, None, :], scores, -jnp.inf)
        next_maximum = jnp.maximum(maximum, jnp.max(scores, axis=-1))
        safe_maximum = jnp.where(jnp.isfinite(next_maximum), next_maximum, 0)
        correction = jax.lax.exp(maximum - safe_maximum, accuracy=jax.lax.AccuracyMode.HIGHEST)
        probabilities = jax.lax.exp(scores - safe_maximum[..., None], accuracy=jax.lax.AccuracyMode.HIGHEST)
        # Unallocated slots can contain NaNs; masked values must not leak through 0 * NaN.
        v = jnp.where(allowed[:, :, None, None], v, 0)
        output = output * correction[..., None] + jnp.einsum(
            "thgs,tshd->thgd", probabilities, v, precision=jax.lax.Precision.HIGHEST
        )
        denominator = denominator * correction + jnp.sum(probabilities, axis=-1)
        return _ReferenceState(output, denominator, next_maximum)

    output, denominator, _ = jax.lax.fori_loop(0, page_indices.shape[1], attend_page, initial)
    return (output / jnp.where(denominator > 0, denominator, 1)[..., None]).astype(q.dtype)


def _tpu_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, *, sm_scale, sliding_window, soft_cap):
    # JAX's TPU kernel applies one precision to QK and AV. Default precision rounds
    # the FP32 softmax weights to BF16; highest precision rejects BF16 operands.
    # Stream reference pages instead of materializing an FP32 copy of the full cache.
    if q.dtype != jnp.float32 or kv_pages.dtype != jnp.float32:
        return _reference_attention(
            q,
            kv_pages,
            kv_lens,
            page_indices,
            cu_q_lens,
            num_seqs,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
        )
    return _tpu_kernel_attention(
        q,
        kv_pages,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_seqs,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )


def _tpu_kernel_attention(
    q,
    kv_pages,
    kv_lens,
    page_indices,
    cu_q_lens,
    num_seqs,
    *,
    sm_scale,
    sliding_window,
    soft_cap,
    num_queries_per_block: int | None = None,
    num_kv_pages_per_block: int | None = None,
):
    if sliding_window is not None and num_kv_pages_per_block is not None:
        # The Pallas kernel masks the window but reads every page in its page table.
        # Drop only whole tiles to keep the softmax accumulation order.
        page_size = kv_pages.shape[1]
        block_tokens = page_size * num_kv_pages_per_block
        window_pages = (sliding_window + q.shape[0] + block_tokens + page_size - 3) // page_size
        if window_pages < page_indices.shape[1] and window_pages * kv_lens.shape[0] < kv_pages.shape[0]:
            query_lengths = jnp.maximum(jnp.diff(cu_q_lens), 0)
            first_keys = jnp.maximum(kv_lens - query_lengths - sliding_window + 1, 0)
            first_pages = first_keys // block_tokens * num_kv_pages_per_block
            page_offsets = first_pages[:, None] + jnp.arange(window_pages)
            physical_pages = jnp.take_along_axis(page_indices, page_offsets, axis=1, mode="clip")
            kv_pages = kv_pages[jnp.maximum(physical_pages, 0).reshape(-1)]
            page_indices = jnp.arange(physical_pages.size, dtype=jnp.int32).reshape(physical_pages.shape)
            kv_lens = kv_lens - first_pages * page_size

    original_dim = q.shape[-1]
    padding = (-original_dim) % TPU_HEAD_ALIGNMENT
    q_padded = jnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, padding)))
    pages_padded = jnp.pad(kv_pages.astype(jnp.float32), ((0, 0), (0, 0), (0, 0), (0, padding)))
    q_flat = q_padded.astype(jnp.float32).reshape(q.shape[0], -1, q_padded.shape[-1])
    if isinstance(sm_scale, (float, int)):
        kernel_scale = sm_scale
    else:
        # Runtime scales cannot be static kernel arguments.
        q_flat = q_flat * sm_scale
        kernel_scale = 1.0
    heads = q.shape[1]
    try:
        get_min_heads_per_blk(q_flat.shape[1], 2 * heads, q_flat.dtype, pages_padded.dtype)
        heads_per_group = heads
    except ValueError:
        # The upstream kernel rejects packed head counts such as 3, 5, 6, and 12.
        # Batch aligned head groups while keeping metadata shared.
        heads_per_group = gcd(heads, _TPU_KV_HEAD_GROUP)
    groups = heads // heads_per_group
    kernel = partial(
        tpu_ragged_paged_attention,
        sm_scale=kernel_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        num_queries_per_block=num_queries_per_block,
        num_kv_pages_per_block=num_kv_pages_per_block,
    )
    metadata = (
        jnp.maximum(kv_lens, 0),
        jnp.maximum(page_indices, 0),
        cu_q_lens,
        jnp.maximum(num_seqs, 0).reshape(1),
    )
    with jax.default_matmul_precision("highest"):
        if groups == 1:
            output = kernel(q_flat, pages_padded, *metadata)
        else:
            q_flat = q_flat.reshape(q.shape[0], groups, heads_per_group * q.shape[2], q_flat.shape[-1])
            pages_padded = pages_padded.reshape(
                *pages_padded.shape[:2], groups, 2 * heads_per_group, pages_padded.shape[-1]
            )

            def attend_group(group):
                # ANY-memory Pallas operands require a whole compact array. vmap
                # instead adds a sliced block mapping that TPU lowering rejects.
                return kernel(q_flat[:, group], pages_padded[:, :, group], *metadata)

            output = jax.lax.map(attend_group, jnp.arange(groups)).swapaxes(0, 1)
    output = output.reshape(q_padded.shape)[..., :original_dim].astype(q.dtype)
    valid = _query_metadata(q, kv_lens, cu_q_lens, num_seqs).valid
    return jnp.where(valid[:, None, None, None], output, 0)
