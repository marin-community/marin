# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Array-first attention over Levanter's interleaved paged KV cache."""

from functools import partial
from types import MappingProxyType
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

from levanter.kernels.pallas.autotune_utils import named_sharding_of
from levanter.kernels.pallas.paged_attention import tpu_paged_attention, tpu_paged_attention_available

from levanter.kernels.pallas.ragged_paged_attention_v3 import ragged_paged_attention as tpu_ragged_paged_attention_v3

PagedAttentionImplementation = Literal["reference", "tpu", "tpu_v3"]
_TPU_QUERIES_PER_BLOCK = 128
_TPU_KV_PAGES_PER_BLOCK = 8
_RPA_V3_VMEM_LIMIT_BYTES = MappingProxyType(
    {
        pltpu.ChipVersion.TPU_V5P: 50_000 * 1024,
        pltpu.ChipVersion.TPU_V6E: 98_304 * 1024,
    }
)
_RPA_V3_WIDE_DECODE_TILES = (4096, 2048)
FUSED_KV_APPEND_MAX_TOKENS = 32
FUSED_KV_PREFILL_MAX_TOKENS = 8192
_RPA_V3_ALIGNMENT = 128


def ragged_paged_attention(
    q: jax.Array,
    kv_pages: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    *,
    sm_scale: float | jax.Array,
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
        implementation: Select the reference, the upstream FP32 TPU kernel,
            or BF16 paged attention v3 on v5p and v6e.
            The v3 selection uses the FP32 TPU kernel for small sliding-window
            batches or inputs outside the BF16 kernel's layout.

    Query positions start at ``kv_lens - diff(cu_q_lens)`` for each sequence.
    Padding queries produce zero. By default, FP32 inputs on supported TPUs use Pallas
    and other inputs use the reference. The KV-head axis can be sharded.
    Page and slot axes and sequence metadata must be replicated.
    """
    if implementation not in (None, "tpu", "reference", "tpu_v3"):
        raise ValueError(f"Unknown paged attention implementation: {implementation}")
    if implementation in (None, "tpu"):
        supports_tpu = q.dtype == kv_pages.dtype == jnp.float32 and tpu_paged_attention_available()
        if implementation == "tpu" and not supports_tpu:
            raise ValueError("The Pallas paged attention kernel requires a supported TPU and FP32 queries and cache")
        implementation = "tpu" if supports_tpu else "reference"
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be positive")
    if soft_cap is not None and soft_cap <= 0:
        raise ValueError("soft_cap must be positive")
    if q.ndim != 4 or kv_pages.ndim != 4 or kv_pages.shape[2:] != (2 * q.shape[1], q.shape[3]):
        raise ValueError("Expected grouped queries and interleaved KV pages with matching heads and head_dim")

    if implementation == "tpu_v3":
        if not _tpu_v3_available():
            raise ValueError("tpu_v3 paged attention requires TPU v5p or v6e")
        supports_v3 = (
            (sliding_window is None or q.shape[0] > _TPU_QUERIES_PER_BLOCK)
            and q.dtype == kv_pages.dtype == jnp.bfloat16
            and q.shape[-1] == kv_pages.shape[1] == _RPA_V3_ALIGNMENT
            and isinstance(sm_scale, (float, int))
        )
        backend = _tpu_v3_attention if supports_v3 else _tpu_attention
    else:
        backend = _tpu_attention if implementation == "tpu" else _reference_attention
    fn = partial(
        backend,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    fn, kv_pages = _partition_heads(fn, q, kv_pages)
    return fn(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs)


def _tpu_v3_available() -> bool:
    return jax.default_backend() == "tpu" and pltpu.get_tpu_info().chip_version in _RPA_V3_VMEM_LIMIT_BYTES


def _attention_sharding(q: jax.Array) -> NamedSharding | None:
    q_sharding = named_sharding_of(q)
    if q_sharding is None or q_sharding.mesh.empty:
        return None
    q_spec = tuple(q_sharding.spec) + (None,) * (q.ndim - len(q_sharding.spec))
    if any(q_spec[axis] is not None for axis in (0, 2, 3)):
        raise ValueError("Paged attention supports sharding only the KV-head axis of q")
    return q_sharding


def _partition_heads(fn, q: jax.Array, kv_pages: jax.Array):
    q_sharding = _attention_sharding(q)
    if q_sharding is None:
        return fn, kv_pages
    heads = q_sharding.spec[1] if len(q_sharding.spec) > 1 else None
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
    return fn, kv_pages


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
    """Append BF16 keys and values during RPA v3 attention on a bounded batch.

    Sequences may share read-only prefix pages. Each sequence must append to
    a private page.
    """
    if not _tpu_v3_available():
        raise ValueError("Fused paged KV append requires TPU v5p or v6e")
    if q.shape[0] > FUSED_KV_PREFILL_MAX_TOKENS:
        raise ValueError(f"Fused paged KV append supports batches of up to {FUSED_KV_PREFILL_MAX_TOKENS} tokens")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype != kv_pages.dtype or q.dtype != jnp.bfloat16:
        raise ValueError("Fused paged KV append requires BF16 queries, keys, values, and cache pages")
    if q.shape[-1] != _RPA_V3_ALIGNMENT or kv_pages.shape[1] != _RPA_V3_ALIGNMENT:
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
    chip = pltpu.get_tpu_info().chip_version
    decode_block = (512 if sliding_window is not None else 1024) if append else 2048
    decode_compute_block = 512
    decode_cores = 1
    if append and sliding_window is None and q.shape[0] <= FUSED_KV_APPEND_MAX_TOKENS and q.shape[2] == 4:
        if chip == pltpu.ChipVersion.TPU_V6E and q.shape[1] == 1:
            decode_block, decode_compute_block = _RPA_V3_WIDE_DECODE_TILES
        elif chip == pltpu.ChipVersion.TPU_V5P and q.shape[1] == 2:
            decode_cores = 2
            decode_block, decode_compute_block = _RPA_V3_WIDE_DECODE_TILES
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
        decode_cores=decode_cores,
        d_block_sizes=(1, decode_block, 1, decode_compute_block),
        m_block_sizes=(min(q.shape[0], 128), 1024, min(q.shape[0], 64), 512),
        vmem_limit_bytes=_RPA_V3_VMEM_LIMIT_BYTES[chip],
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
        weights = jax.lax.exp(scores - safe_maximum[..., None], accuracy=jax.lax.AccuracyMode.HIGHEST)
        # Unallocated slots can contain NaNs; masked values must not leak through 0 * NaN.
        v = jnp.where(allowed[:, :, None, None], v, 0)
        output = output * correction[..., None] + jnp.einsum(
            "thgs,tshd->thgd", weights, v, precision=jax.lax.Precision.HIGHEST
        )
        denominator = denominator * correction + jnp.sum(weights, axis=-1)
        return _ReferenceState(output, denominator, next_maximum)

    output, denominator, _ = jax.lax.fori_loop(0, page_indices.shape[1], attend_page, initial)
    return (output / jnp.where(denominator > 0, denominator, 1)[..., None]).astype(q.dtype)


def _tpu_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, *, sm_scale, sliding_window, soft_cap):
    kv_pages_per_block = min(_TPU_KV_PAGES_PER_BLOCK, page_indices.shape[1])
    if sliding_window is not None:
        # Drop whole tiles only so the surviving softmax accumulation order stays fixed.
        page_size = kv_pages.shape[1]
        block_tokens = page_size * kv_pages_per_block
        window_pages = (sliding_window + q.shape[0] + block_tokens + page_size - 3) // page_size
        if window_pages < page_indices.shape[1] and window_pages * kv_lens.shape[0] < kv_pages.shape[0]:
            query_lengths = jnp.maximum(jnp.diff(cu_q_lens), 0)
            first_keys = jnp.maximum(kv_lens - query_lengths - sliding_window + 1, 0)
            first_pages = first_keys // block_tokens * kv_pages_per_block
            page_offsets = first_pages[:, None] + jnp.arange(window_pages)
            physical_pages = jnp.take_along_axis(page_indices, page_offsets, axis=1, mode="clip")
            kv_pages = kv_pages[jnp.maximum(physical_pages, 0).reshape(-1)]
            page_indices = jnp.arange(physical_pages.size, dtype=jnp.int32).reshape(physical_pages.shape)
            kv_lens = kv_lens - first_pages * page_size

    # Keep both scores and softmax weights in FP32, including the AV contraction.
    with jax.default_matmul_precision("highest"):
        # The upstream long-context tiles can exceed scoped VMEM with FP32 KV.
        output = tpu_paged_attention(
            q.astype(jnp.float32),
            kv_pages.astype(jnp.float32),
            kv_lens,
            page_indices,
            cu_q_lens,
            num_seqs,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            num_queries_per_block=min(_TPU_QUERIES_PER_BLOCK, q.shape[0]),
            num_kv_pages_per_block=kv_pages_per_block,
        )
    valid = _query_metadata(q, kv_lens, cu_q_lens, num_seqs).valid
    return jnp.where(valid[:, None, None, None], output.astype(q.dtype), 0)
