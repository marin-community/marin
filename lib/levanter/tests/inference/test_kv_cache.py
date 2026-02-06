# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import numpy as np

from levanter.layers.kv_cache import kv_update_unified_prefix


def _reference_kv_update(kv_pages_np, t_pages, t_slots, new_k, new_v, k_valid: int):
    kv_ev = jnp.stack([new_k, new_v], axis=2).reshape(new_k.shape[0], 2 * new_k.shape[1], new_k.shape[2])
    expected = kv_pages_np.copy()
    for i in range(k_valid):
        expected[int(t_pages[i]), int(t_slots[i]), :, :] = np.asarray(kv_ev[i])
    return expected


def test_kv_update_unified_prefix_updates_only_valid_prefix():
    pages, slots, kv_heads, head_dim = 5, 4, 2, 3
    total_tokens = 6

    kv_pages = jnp.arange(pages * slots * (2 * kv_heads) * head_dim, dtype=jnp.float32).reshape(
        pages, slots, 2 * kv_heads, head_dim
    )
    new_k = (jnp.arange(total_tokens * kv_heads * head_dim, dtype=jnp.float32) + 1000).reshape(
        total_tokens, kv_heads, head_dim
    )
    new_v = (jnp.arange(total_tokens * kv_heads * head_dim, dtype=jnp.float32) + 2000).reshape(
        total_tokens, kv_heads, head_dim
    )
    t_pages = jnp.array([1, 2, 3, -1, -1, -1], dtype=jnp.int32)
    t_slots = jnp.array([0, 1, 2, -1, -1, -1], dtype=jnp.int32)
    k_valid = 3

    kv_pages_np = np.asarray(kv_pages).copy()
    updated = kv_update_unified_prefix(kv_pages, t_pages, t_slots, new_k, new_v, jnp.array(k_valid, dtype=jnp.int32))
    expected = _reference_kv_update(kv_pages_np, t_pages, t_slots, new_k, new_v, k_valid)
    np.testing.assert_allclose(np.asarray(updated), expected)


def test_kv_update_unified_prefix_noop_when_k_zero():
    pages, slots, kv_heads, head_dim = 3, 2, 1, 2
    total_tokens = 4
    kv_pages = jnp.arange(pages * slots * (2 * kv_heads) * head_dim, dtype=jnp.float32).reshape(
        pages, slots, 2 * kv_heads, head_dim
    )
    new_k = jnp.ones((total_tokens, kv_heads, head_dim), dtype=jnp.float32)
    new_v = 2 * jnp.ones((total_tokens, kv_heads, head_dim), dtype=jnp.float32)
    t_pages = jnp.array([-1, -1, -1, -1], dtype=jnp.int32)
    t_slots = jnp.array([-1, -1, -1, -1], dtype=jnp.int32)

    kv_pages_np = np.asarray(kv_pages).copy()
    updated = kv_update_unified_prefix(kv_pages, t_pages, t_slots, new_k, new_v, jnp.array(0, dtype=jnp.int32))
    np.testing.assert_allclose(np.asarray(updated), kv_pages_np)


def test_kv_update_unified_prefix_clips_k_to_token_count():
    pages, slots, kv_heads, head_dim = 4, 4, 1, 2
    total_tokens = 5
    kv_pages = jnp.zeros((pages, slots, 2 * kv_heads, head_dim), dtype=jnp.float32)
    new_k = jnp.arange(total_tokens * kv_heads * head_dim, dtype=jnp.float32).reshape(total_tokens, kv_heads, head_dim)
    new_v = (new_k + 100).astype(jnp.float32)
    t_pages = jnp.array([0, 1, 2, 3, 0], dtype=jnp.int32)
    t_slots = jnp.array([0, 1, 2, 3, 1], dtype=jnp.int32)

    kv_pages_np = np.asarray(kv_pages).copy()
    updated = kv_update_unified_prefix(kv_pages, t_pages, t_slots, new_k, new_v, jnp.array(999, dtype=jnp.int32))
    expected = _reference_kv_update(kv_pages_np, t_pages, t_slots, new_k, new_v, total_tokens)
    np.testing.assert_allclose(np.asarray(updated), expected)
