# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt Levanter's grouped query and interleaved cache layouts to JAX's TPU kernel."""

from functools import partial
from math import gcd

import jax
import jax.numpy as jnp

try:
    from jax.experimental.pallas.ops.tpu.ragged_paged_attention import (
        ragged_paged_attention as tpu_ragged_paged_attention,
    )
    from jax.experimental.pallas.ops.tpu.ragged_paged_attention.kernel import get_min_heads_per_blk
except ImportError:  # pragma: no cover - optional JAX kernel
    tpu_ragged_paged_attention = None
    get_min_heads_per_blk = None

_TPU_HEAD_ALIGNMENT = 128
_TPU_KV_HEAD_GROUP = 8


def tpu_paged_attention_available() -> bool:
    """Whether JAX exposes paged attention for the active TPU generation."""
    if tpu_ragged_paged_attention is None or get_min_heads_per_blk is None or jax.default_backend() != "tpu":
        return False
    kind = jax.devices()[0].device_kind.lower()
    return "tpu v2" not in kind and "tpu v3" not in kind


def tpu_paged_attention(
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
    num_queries_per_block: int | None = None,
    num_kv_pages_per_block: int | None = None,
) -> jax.Array:
    """Run upstream ragged attention on local grouped queries and interleaved pages."""
    if tpu_ragged_paged_attention is None or get_min_heads_per_blk is None:
        raise RuntimeError("JAX's TPU ragged paged attention kernel is unavailable")
    original_dim = q.shape[-1]
    padding = (-original_dim) % _TPU_HEAD_ALIGNMENT
    q_padded, pages_padded = q, kv_pages
    if padding:
        q_padded = jnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, padding)))
        pages_padded = jnp.pad(kv_pages, ((0, 0), (0, 0), (0, 0), (0, padding)))
    q_flat = q_padded.reshape(q.shape[0], -1, q_padded.shape[-1])
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
        # The upstream kernel requires the interleaved KV axis to tile in XLA.
        heads_per_group = gcd(heads, _TPU_KV_HEAD_GROUP)
        get_min_heads_per_blk(heads_per_group * q.shape[2], 2 * heads_per_group, q_flat.dtype, kv_pages.dtype)
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
    return output.reshape(q_padded.shape)[..., :original_dim]
