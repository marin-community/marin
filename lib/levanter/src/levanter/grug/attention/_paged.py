# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Array-first attention over Levanter's interleaved paged KV cache."""

from functools import partial
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

from levanter.grug.attention._paged_gpu import gpu_paged_attention, gpu_paged_prefill
from levanter.kernels.pallas.autotune_utils import named_sharding_of
from levanter.kernels.pallas.paged_attention import tpu_paged_attention, tpu_paged_attention_available

PagedAttentionImplementation = Literal["reference", "tpu", "gpu"]
_GPU_PAGED_QUERY_LIMIT = 8
_GPU_FULL_DECODE_SPLITS = 128
_GPU_WINDOWED_DECODE_SPLITS = 16


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
        implementation: Reference, FP32 TPU, or BF16 CUDA split-K decode and FA4 prefill.

    Query positions start at ``kv_lens - diff(cu_q_lens)`` for each sequence.
    Padding queries produce zero. By default, FP32 inputs on supported TPUs use Pallas
    and other inputs use the reference. The KV-head axis can be sharded.
    Page and slot axes and sequence metadata must be replicated.
    """
    if implementation not in (None, "tpu", "reference", "gpu"):
        raise ValueError(f"Unknown paged attention implementation: {implementation}")
    if implementation == "gpu":
        if jax.default_backend() != "gpu" or q.dtype != jnp.bfloat16 or kv_pages.dtype != jnp.bfloat16:
            raise ValueError("GPU paged attention requires CUDA with BF16 queries and cache")
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

    fn = partial(
        {"gpu": _gpu_attention, "tpu": _tpu_attention, "reference": _reference_attention}[implementation],
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    fn, kv_pages = _partition_heads(fn, q, kv_pages)
    return fn(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs)


def _partition_heads(fn, q: jax.Array, kv_pages: jax.Array):
    q_sharding = named_sharding_of(q)
    if q_sharding is None or q_sharding.mesh.empty:
        return fn, kv_pages
    q_spec = tuple(q_sharding.spec) + (None,) * (q.ndim - len(q_sharding.spec))
    if any(q_spec[axis] is not None for axis in (0, 2, 3)):
        raise ValueError("Paged attention supports sharding only the KV-head axis of q")
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
    return fn, kv_pages


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


def _gpu_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, *, sm_scale, sliding_window, soft_cap):
    if q.shape[0] <= _GPU_PAGED_QUERY_LIMIT:
        metadata = _query_metadata(q, kv_lens, cu_q_lens, num_seqs)
        upper = jnp.where(metadata.valid, metadata.position + 1, 0)
        lower = jnp.zeros_like(upper) if sliding_window is None else jnp.maximum(0, upper - sliding_window)
        return gpu_paged_attention(
            q,
            kv_pages,
            jnp.maximum(page_indices[metadata.sequence], 0),
            jnp.stack((lower, upper), axis=-1),
            sm_scale,
            max_splits=_GPU_FULL_DECODE_SPLITS if sliding_window is None else _GPU_WINDOWED_DECODE_SPLITS,
            soft_cap=soft_cap,
        )
    if soft_cap is not None or not isinstance(sm_scale, (float, int)):
        raise ValueError("GPU prefill requires a static scale and does not support soft caps")

    # The scheduler pads the page table to its slot capacity. Only active
    # sequences enter FA4, each with its own query span and reordered KV pages.
    tokens = jnp.arange(q.shape[0])

    def attend_sequence(seq, output):
        start, stop = cu_q_lens[seq], cu_q_lens[seq + 1]
        indices = start + tokens
        query = q[jnp.minimum(indices, q.shape[0] - 1)]
        attended = gpu_paged_prefill(
            query,
            kv_pages,
            jax.lax.dynamic_slice_in_dim(kv_lens, seq, 1),
            page_indices[seq],
            stop - start,
            sm_scale=float(sm_scale),
            sliding_window=sliding_window,
        )
        return output.at[jnp.where(indices < stop, indices, q.shape[0])].set(attended, mode="drop")

    return jax.lax.fori_loop(0, num_seqs.reshape(()), attend_sequence, jnp.zeros_like(q))


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
    # Keep both scores and softmax weights in FP32, including the AV contraction.
    with jax.default_matmul_precision("highest"):
        output = tpu_paged_attention(
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
    valid = _query_metadata(q, kv_lens, cu_q_lens, num_seqs).valid
    return jnp.where(valid[:, None, None, None], output, 0)
