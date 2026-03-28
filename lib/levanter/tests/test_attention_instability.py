# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

import haliax as hax
from haliax import Axis

from levanter.callbacks.attention_instability import (
    AttentionInstabilityCallback,
    AttentionInstabilityConfig,
    compute_attention_instability_stats,
)
from levanter.layers.attention import (
    Attention,
    AttentionConfig,
    get_max_attn_logit,
    reset_max_attn_logit,
    set_attn_logit_tracking,
    simple_attention_with_dropout,
)


Embed = Axis("embed", 32)
Pos = Axis("position", 8)
KPos = Axis("key_position", 8)
HeadSize = Axis("head_size", 16)


def _make_attention(*, key, num_heads=2, num_kv_heads=2):
    config = AttentionConfig(
        Embed=Embed,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    return Attention.init(config, key=key)


class _TwoLayerModel(eqx.Module):
    layer0: Attention
    layer1: Attention

    @staticmethod
    def init(*, key):
        k0, k1 = jrandom.split(key)
        return _TwoLayerModel(
            layer0=_make_attention(key=k0),
            layer1=_make_attention(key=k1),
        )


def test_compute_attention_instability_stats_returns_metrics():
    model = _TwoLayerModel.init(key=jrandom.PRNGKey(0))
    stats = compute_attention_instability_stats(model)

    assert "attention/qk_norm_product_max" in stats
    assert "attention/qk_norm_product/0" in stats
    assert "attention/qk_norm_product/1" in stats

    # All values should be finite positive
    for key, val in stats.items():
        v = float(val)
        assert math.isfinite(v), f"{key} = {v}"
        assert v > 0, f"{key} should be positive"


def test_qk_norm_product_scales_with_weight_magnitude():
    """Scaling Q weights up should increase the norm product."""
    key = jrandom.PRNGKey(1)
    attn = _make_attention(key=key)

    model_normal = _TwoLayerModel(layer0=attn, layer1=attn)
    stats_normal = compute_attention_instability_stats(model_normal)

    # Scale Q weights by 10x
    big_q = eqx.tree_at(
        lambda m: m.q_proj.weight,
        attn,
        attn.q_proj.weight * 10.0,
    )
    model_big = _TwoLayerModel(layer0=big_q, layer1=big_q)
    stats_big = compute_attention_instability_stats(model_big)

    assert float(stats_big["attention/qk_norm_product_max"]) > float(
        stats_normal["attention/qk_norm_product_max"]
    ) * 5  # should be ~10x larger


def test_no_attention_modules_returns_empty():
    class _NoAttn(eqx.Module):
        w: jax.Array

    model = _NoAttn(w=jnp.ones(5))
    stats = compute_attention_instability_stats(model)
    assert stats == {}


def test_max_attn_logit_tracking_vanilla():
    """Verify that vanilla attention captures max pre-softmax logit."""
    reset_max_attn_logit()
    set_attn_logit_tracking(True)

    try:
        key = jrandom.PRNGKey(42)
        q = hax.random.normal(key, (Pos, HeadSize))
        k = hax.random.normal(jrandom.PRNGKey(43), (KPos, HeadSize))
        v = hax.random.normal(jrandom.PRNGKey(44), (KPos, HeadSize))

        # Run vanilla attention
        simple_attention_with_dropout(
            Pos, KPos, HeadSize, q, k, v,
        )

        max_logit = get_max_attn_logit()
        assert max_logit > float("-inf"), "max logit should have been captured"
        assert math.isfinite(max_logit), f"max logit should be finite, got {max_logit}"

        # Verify the captured value is consistent: compute QK manually
        scale = 1.0 / jnp.sqrt(HeadSize.size)
        qk = hax.dot(q * scale, k, axis=HeadSize)
        expected_max = float(jnp.max(qk.array))
        assert abs(max_logit - expected_max) < 1e-5, (
            f"captured {max_logit} vs expected {expected_max}"
        )
    finally:
        set_attn_logit_tracking(False)
        reset_max_attn_logit()


def test_max_attn_logit_tracking_disabled_by_default():
    """When tracking is off, accumulator stays at -inf."""
    reset_max_attn_logit()
    set_attn_logit_tracking(False)

    q = hax.random.normal(jrandom.PRNGKey(0), (Pos, HeadSize))
    k = hax.random.normal(jrandom.PRNGKey(1), (KPos, HeadSize))
    v = hax.random.normal(jrandom.PRNGKey(2), (KPos, HeadSize))

    simple_attention_with_dropout(Pos, KPos, HeadSize, q, k, v)

    assert get_max_attn_logit() == float("-inf")


def test_config_disabled_by_default():
    config = AttentionInstabilityConfig()
    assert not config.is_enabled


def test_config_enabled_with_interval():
    config = AttentionInstabilityConfig(interval=10)
    assert config.is_enabled
    cb = config.build()
    assert isinstance(cb, AttentionInstabilityCallback)


def test_max_logit_accumulates_across_calls():
    """Multiple attention calls should keep the running maximum."""
    reset_max_attn_logit()
    set_attn_logit_tracking(True)

    try:
        key = jrandom.PRNGKey(99)
        # First call with small values
        q1 = hax.random.normal(key, (Pos, HeadSize)) * 0.1
        k1 = hax.random.normal(jrandom.PRNGKey(100), (KPos, HeadSize)) * 0.1
        v1 = hax.random.normal(jrandom.PRNGKey(101), (KPos, HeadSize))
        simple_attention_with_dropout(Pos, KPos, HeadSize, q1, k1, v1)
        first_max = get_max_attn_logit()

        # Second call with larger values
        q2 = hax.random.normal(jrandom.PRNGKey(102), (Pos, HeadSize)) * 10.0
        k2 = hax.random.normal(jrandom.PRNGKey(103), (KPos, HeadSize)) * 10.0
        v2 = hax.random.normal(jrandom.PRNGKey(104), (KPos, HeadSize))
        simple_attention_with_dropout(Pos, KPos, HeadSize, q2, k2, v2)
        second_max = get_max_attn_logit()

        assert second_max >= first_max, "running max should be non-decreasing"
        assert second_max > first_max, "larger inputs should produce larger max logit"
    finally:
        set_attn_logit_tracking(False)
        reset_max_attn_logit()
