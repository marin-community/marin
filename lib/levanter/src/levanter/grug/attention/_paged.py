# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Array-first attention over Levanter's interleaved paged KV cache."""

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.ragged_paged_attention import ragged_paged_attention as tpu_ragged_paged_attention
from jax.sharding import PartitionSpec as P

from levanter.kernels.pallas.autotune_utils import named_sharding_of

PagedAttentionImplementation = Literal["reference", "tpu"]


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

    Query positions start at ``kv_lens - diff(cu_q_lens)`` for each sequence.
    Padding queries produce zero. The TPU path uses JAX's existing ragged kernel;
    GPU and CPU currently use a reference implementation, not an optimized kernel.
    Only the KV-head axis is partitioned; sequence metadata and pages are replicated.
    """
    if implementation is None:
        implementation = "tpu" if jax.default_backend() == "tpu" else "reference"
    if implementation not in ("tpu", "reference"):
        raise ValueError(f"Unknown paged attention implementation: {implementation}")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be positive")
    if soft_cap is not None and soft_cap <= 0:
        raise ValueError("soft_cap must be positive")
    if q.ndim != 4 or kv_pages.ndim != 4 or kv_pages.shape[2:] != (2 * q.shape[1], q.shape[3]):
        raise ValueError("Expected grouped queries and interleaved KV pages with matching heads and head_dim")

    fn = partial(
        _tpu_attention if implementation == "tpu" else _reference_attention,
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


def _query_metadata(q, kv_lens, cu_q_lens, num_seqs):
    token = jnp.arange(q.shape[0])
    active = jnp.arange(kv_lens.shape[0]) < num_seqs.reshape(())
    belongs = active[:, None] & (token >= cu_q_lens[:-1, None]) & (token < cu_q_lens[1:, None])
    seq = jnp.argmax(belongs, axis=0)
    valid = jnp.any(belongs, axis=0)
    position = kv_lens[seq] - (cu_q_lens[seq + 1] - cu_q_lens[seq]) + token - cu_q_lens[seq]
    return seq, position, valid


def _reference_attention(
    q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, *, sm_scale, sliding_window, soft_cap
):
    # Stream pages so decode memory does not scale as tokens * maximum context * head_dim.
    seq, position, valid = _query_metadata(q, kv_lens, cu_q_lens, num_seqs)
    page_size = kv_pages.shape[1]
    initial = (
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
        scores = jnp.einsum("thgd,tshd->thgs", q.astype(jnp.float32) * sm_scale, k)
        if soft_cap is not None:
            scores = soft_cap * jnp.tanh(scores / soft_cap)
        scores = jnp.where(allowed[:, None, None, :], scores, -jnp.inf)
        next_maximum = jnp.maximum(maximum, jnp.max(scores, axis=-1))
        safe_maximum = jnp.where(jnp.isfinite(next_maximum), next_maximum, 0)
        correction = jnp.exp(maximum - safe_maximum)
        probabilities = jnp.exp(scores - safe_maximum[..., None])
        # Unallocated slots can contain NaNs; masked values must not leak through 0 * NaN.
        v = jnp.where(allowed[:, :, None, None], v, 0)
        output = output * correction[..., None] + jnp.einsum("thgs,tshd->thgd", probabilities, v)
        denominator = denominator * correction + jnp.sum(probabilities, axis=-1)
        return output, denominator, next_maximum

    output, denominator, _ = jax.lax.fori_loop(0, page_indices.shape[1], attend_page, initial)
    return (output / jnp.where(denominator > 0, denominator, 1)[..., None]).astype(q.dtype)


def _tpu_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, *, sm_scale, sliding_window, soft_cap):
    original_dim = q.shape[-1]
    padding = (-original_dim) % 128
    q_padded = jnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, padding)))
    pages_padded = jnp.pad(kv_pages, ((0, 0), (0, 0), (0, 0), (0, padding)))
    # Scaling q keeps runtime scales out of the kernel's static argument list.
    q_flat = (q_padded * sm_scale).reshape(q.shape[0], -1, q_padded.shape[-1])
    output = tpu_ragged_paged_attention(
        q_flat,
        pages_padded,
        jnp.maximum(kv_lens, 0),
        jnp.maximum(page_indices, 0),
        cu_q_lens,
        jnp.maximum(num_seqs, 0).reshape(1),
        sm_scale=1.0,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    output = output.reshape(q_padded.shape)[..., :original_dim]
    _, _, valid = _query_metadata(q, kv_lens, cu_q_lens, num_seqs)
    return jnp.where(valid[:, None, None, None], output, 0)
