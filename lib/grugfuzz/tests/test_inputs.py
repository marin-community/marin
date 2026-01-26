"""Tests for inputs module."""

import jax.numpy as jnp
import numpy as np
import pytest

from grugfuzz import random_attention_mask, random_hidden, random_kv_cache, random_qkv, random_tokens


class TestRandomTokens:
    def test_shape(self):
        tokens = random_tokens(vocab_size=1000, batch=4, seq=64)
        assert tokens.shape == (4, 64)

    def test_dtype(self):
        tokens = random_tokens(vocab_size=1000, batch=2, seq=32)
        assert tokens.dtype == jnp.int32

    def test_range(self):
        vocab_size = 100
        tokens = random_tokens(vocab_size=vocab_size, batch=10, seq=100)
        assert jnp.all(tokens >= 0)
        assert jnp.all(tokens < vocab_size)

    def test_reproducible(self):
        t1 = random_tokens(vocab_size=1000, batch=2, seq=32, seed=42)
        t2 = random_tokens(vocab_size=1000, batch=2, seq=32, seed=42)
        np.testing.assert_array_equal(np.array(t1), np.array(t2))

    def test_different_seeds(self):
        t1 = random_tokens(vocab_size=1000, batch=2, seq=32, seed=1)
        t2 = random_tokens(vocab_size=1000, batch=2, seq=32, seed=2)
        assert not np.array_equal(np.array(t1), np.array(t2))


class TestRandomHidden:
    def test_shape(self):
        hidden = random_hidden(batch=2, seq=32, dim=128)
        assert hidden.shape == (2, 32, 128)

    def test_dtype_default(self):
        hidden = random_hidden(batch=2, seq=32, dim=128)
        assert hidden.dtype == jnp.float32

    def test_dtype_specified(self):
        hidden = random_hidden(batch=2, seq=32, dim=128, dtype=jnp.float16)
        assert hidden.dtype == jnp.float16

    def test_scale(self):
        # With default scale=0.02, values should be small
        hidden = random_hidden(batch=2, seq=32, dim=128, scale=0.02)
        assert np.abs(hidden).max() < 1.0  # Very unlikely to exceed 1 with scale=0.02

    def test_reproducible(self):
        h1 = random_hidden(batch=2, seq=32, dim=64, seed=42)
        h2 = random_hidden(batch=2, seq=32, dim=64, seed=42)
        np.testing.assert_array_equal(np.array(h1), np.array(h2))


class TestRandomQkv:
    def test_shapes(self):
        q, k, v = random_qkv(batch=2, seq=16, num_heads=4, head_dim=32)
        expected_shape = (2, 4, 16, 32)
        assert q.shape == expected_shape
        assert k.shape == expected_shape
        assert v.shape == expected_shape

    def test_different_values(self):
        q, k, v = random_qkv(batch=2, seq=16, num_heads=4, head_dim=32)
        # Q, K, V should be different
        assert not np.allclose(np.array(q), np.array(k))
        assert not np.allclose(np.array(k), np.array(v))

    def test_dtype(self):
        q, k, v = random_qkv(batch=2, seq=16, num_heads=4, head_dim=32, dtype=jnp.float16)
        assert q.dtype == jnp.float16
        assert k.dtype == jnp.float16
        assert v.dtype == jnp.float16

    def test_reproducible(self):
        qkv1 = random_qkv(batch=2, seq=16, num_heads=4, head_dim=32, seed=42)
        qkv2 = random_qkv(batch=2, seq=16, num_heads=4, head_dim=32, seed=42)
        for a, b in zip(qkv1, qkv2):
            np.testing.assert_array_equal(np.array(a), np.array(b))


class TestRandomAttentionMask:
    def test_shape(self):
        mask = random_attention_mask(batch=4, seq=64)
        assert mask.shape == (4, 64)

    def test_dtype(self):
        mask = random_attention_mask(batch=2, seq=32)
        assert mask.dtype == jnp.bool_

    def test_pad_ratio(self):
        # With pad_ratio=0.5, roughly half should be masked
        mask = random_attention_mask(batch=10, seq=100, pad_ratio=0.5)
        ratio = mask.sum() / mask.size
        # Should be roughly 0.5 (True values)
        assert 0.3 < ratio < 0.7

    def test_no_padding(self):
        mask = random_attention_mask(batch=2, seq=32, pad_ratio=0.0)
        assert jnp.all(mask)  # All True

    def test_reproducible(self):
        m1 = random_attention_mask(batch=2, seq=32, seed=42)
        m2 = random_attention_mask(batch=2, seq=32, seed=42)
        np.testing.assert_array_equal(np.array(m1), np.array(m2))


class TestRandomKvCache:
    def test_shapes(self):
        k_cache, v_cache = random_kv_cache(batch=2, cache_len=64, num_kv_heads=4, head_dim=32)
        expected_shape = (2, 4, 64, 32)
        assert k_cache.shape == expected_shape
        assert v_cache.shape == expected_shape

    def test_different_values(self):
        k_cache, v_cache = random_kv_cache(batch=2, cache_len=64, num_kv_heads=4, head_dim=32)
        assert not np.allclose(np.array(k_cache), np.array(v_cache))

    def test_dtype(self):
        k_cache, v_cache = random_kv_cache(
            batch=2, cache_len=64, num_kv_heads=4, head_dim=32, dtype=jnp.bfloat16
        )
        assert k_cache.dtype == jnp.bfloat16
        assert v_cache.dtype == jnp.bfloat16
