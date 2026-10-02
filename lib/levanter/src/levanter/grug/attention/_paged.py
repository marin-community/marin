# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Array-first attention over Levanter's interleaved paged KV cache."""

from functools import partial
from math import gcd
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.ragged_paged_attention import ragged_paged_attention as tpu_ragged_paged_attention
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.kernel import get_min_heads_per_blk
from jax.sharding import PartitionSpec as P

from levanter.grug.attention._paged_tpu import TPU_HEAD_ALIGNMENT, TPU_PAGE_ALIGNMENT, tpu_paged_decode
from levanter.kernels.pallas.autotune_utils import named_sharding_of

PagedAttentionImplementation = Literal["reference", "tpu", "tpu_fp32_tiles"]
_TPU_KV_HEAD_GROUP = 8


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
    tpu_dma_buffers: int = 1,
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
        tpu_dma_buffers: One serial or two overlapping page DMA buffers for TPU decode.
        implementation: TPU Pallas or portable reference; defaults to TPU on TPU.

    Query positions start at ``kv_lens - diff(cu_q_lens)`` for each sequence.
    Padding queries produce zero. The TPU path uses JAX's existing ragged kernel
    for FP32 inputs; lower-precision inputs use the reference to preserve accuracy.
    ``tpu_fp32_tiles`` also promotes BF16 cache tiles to FP32 inside the TPU kernel,
    preserving the cache's storage dtype. GPU and CPU default to the reference.
    Only the KV-head axis is partitioned; sequence metadata and pages are replicated.
    """
    if implementation is None:
        implementation = "tpu" if jax.default_backend() == "tpu" else "reference"
    if implementation not in ("tpu", "tpu_fp32_tiles", "reference"):
        raise ValueError(f"Unknown paged attention implementation: {implementation}")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be positive")
    if soft_cap is not None and soft_cap <= 0:
        raise ValueError("soft_cap must be positive")
    if q.ndim != 4 or kv_pages.ndim != 4 or kv_pages.shape[2:] != (2 * q.shape[1], q.shape[3]):
        raise ValueError("Expected grouped queries and interleaved KV pages with matching heads and head_dim")

    backend = {
        "tpu": _tpu_attention,
        "tpu_fp32_tiles": partial(_tpu_decode_attention, dma_buffers=tpu_dma_buffers),
        "reference": _reference_attention,
    }
    fn = partial(
        backend[implementation],
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    q_sharding = named_sharding_of(q)
    if q_sharding is not None and not q_sharding.mesh.empty:
        q_spec = tuple(q_sharding.spec) + (None,) * (q.ndim - len(q_sharding.spec))
        if any(q_spec[axis] is not None for axis in (0, 2, 3)):
            raise ValueError("Paged attention supports sharding only the KV-head axis of q")
        heads = q_spec[1]
        fn = jax.shard_map(
            fn,
            mesh=q_sharding.mesh,
            in_specs=(P(None, heads, None, None), P(None, None, heads, None), P(), P(), P(), P()),
            out_specs=P(None, heads, None, None),
            check_vma=False,
        )
    return fn(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs)


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
    q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, *, sm_scale, sliding_window, soft_cap
):
    original_dim = q.shape[-1]
    padding = (-original_dim) % 128
    q_padded = jnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, padding)))
    pages_padded = jnp.pad(kv_pages, ((0, 0), (0, 0), (0, 0), (0, padding)))
    q_flat = q_padded.astype(jnp.float32).reshape(q.shape[0], -1, q_padded.shape[-1])
    if isinstance(sm_scale, (float, int)):
        kernel_scale = sm_scale
    else:
        # Runtime scales cannot be static kernel arguments.
        q_flat = q_flat * sm_scale
        kernel_scale = 1.0
    heads = q.shape[1]
    try:
        get_min_heads_per_blk(q_flat.shape[1], 2 * heads, q_flat.dtype, kv_pages.dtype)
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


def _tpu_decode_attention(
    q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, *, sm_scale, sliding_window, soft_cap, dma_buffers
):
    reference = partial(
        _reference_attention,
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
    if (
        soft_cap is not None
        or kv_pages.shape[1] < TPU_PAGE_ALIGNMENT
        or kv_pages.shape[1] % TPU_PAGE_ALIGNMENT
        or q.shape[-1] % TPU_HEAD_ALIGNMENT
    ):
        return reference()
    metadata = _query_metadata(q, kv_lens, cu_q_lens, num_seqs)
    upper = jnp.where(metadata.valid, metadata.position + 1, 0)
    lower = jnp.zeros_like(upper) if sliding_window is None else jnp.maximum(0, upper - sliding_window)
    bounds = jnp.stack((lower, upper), axis=-1)
    token_pages = jnp.maximum(page_indices[metadata.sequence], 0)
    active = jnp.arange(kv_lens.shape[0]) < num_seqs.reshape(())
    decode_only = jnp.all(jnp.where(active, jnp.diff(cu_q_lens) <= 1, True))
    return jax.lax.cond(
        decode_only,
        lambda: tpu_paged_decode(q, kv_pages, token_pages, bounds, sm_scale, dma_buffers=dma_buffers),
        reference,
    )
