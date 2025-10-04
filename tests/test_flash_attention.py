# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import functools
import math

import equinox
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from chex import assert_trees_all_close

import haliax as hax
import haliax.nn as hnn

import levanter.layers.attention
from levanter.layers.attention import AttentionMask, simple_attention_with_dropout
from levanter.models.flash_attention import flash_attention
from test_utils import use_test_mesh


BLOCK_SIZE = 64


def test_flash_attention_acausal():
    Key = hax.Axis("Key", 8)
    QPos = hax.Axis("QPos", BLOCK_SIZE * 2)
    KPos = hax.Axis("KPos", BLOCK_SIZE * 2)

    q = hax.random.normal(jrandom.PRNGKey(0), (QPos, Key)) * 0.2
    k = hax.random.normal(jrandom.PRNGKey(1), (KPos, Key)) * 0.2
    v = hax.random.normal(jrandom.PRNGKey(2), (KPos, Key)) * 0.2

    flash_out = flash_attention(QPos, KPos, Key, q, k, v, inference=True, block_size=BLOCK_SIZE)
    hax_out = hnn.attention.dot_product_attention(KPos, Key, q, k, v)

    assert hax_out.axes == flash_out.axes
    assert_trees_all_close(hax_out.array, flash_out.array, atol=1e-3, rtol=1e-3)


def test_flash_attention_causal_mask():
    Key = hax.Axis("Key", 8)
    QPos = hax.Axis("QPos", BLOCK_SIZE * 4)
    KPos = hax.Axis("KPos", BLOCK_SIZE * 4)

    mask = AttentionMask.causal()

    q = hax.random.normal(jrandom.PRNGKey(0), (QPos, Key)) * 0.02
    k = hax.random.normal(jrandom.PRNGKey(1), (KPos, Key)) * 0.02
    v = hax.random.normal(jrandom.PRNGKey(2), (KPos, Key)) * 0.02

    flash_out = flash_attention(
        QPos, KPos, Key, q, k, v, inference=True, mask=mask, block_size=BLOCK_SIZE, precision="highest"
    )
    hax_out = hnn.attention.dot_product_attention(
        KPos, Key, q, k, v, mask=mask.materialize(QPos, KPos), precision="highest"
    )

    assert hax_out.axes == flash_out.axes
    assert_trees_all_close(hax_out.array, flash_out.array, atol=1e-3, rtol=1e-3)


def test_grad_attention():
    Key = hax.Axis("Key", 8)
    QPos = hax.Axis("QPos", BLOCK_SIZE * 4)
    KPos = hax.Axis("KPos", BLOCK_SIZE * 4)

    mask = AttentionMask.causal()

    q = hax.random.normal(jrandom.PRNGKey(0), (QPos, Key))
    k = hax.random.normal(jrandom.PRNGKey(1), (KPos, Key))
    v = hax.random.normal(jrandom.PRNGKey(2), (KPos, Key))

    @equinox.filter_value_and_grad
    def d_attn(qkv, fn):
        q, k, v = qkv
        x_out = fn(QPos, KPos, Key, q, k, v, mask=mask)
        return (x_out * x_out).mean().scalar()

    hax_val, (hax_dq, hax_dk, hax_dv) = d_attn((q, k, v), simple_attention_with_dropout)
    fa_val, (fa_dq, fa_dk, fa_dv) = d_attn(
        (q, k, v), functools.partial(flash_attention, inference=True, block_size=BLOCK_SIZE)
    )

    assert_trees_all_close(hax_val, fa_val, atol=1e-3, rtol=1e-3)
    assert hax_dq.axes == fa_dq.axes
    assert hax_dk.axes == fa_dk.axes
    assert hax_dv.axes == fa_dv.axes

    assert_trees_all_close(hax_dq.array, fa_dq.array, atol=1e-3, rtol=1e-3)
    assert_trees_all_close(hax_dk.array, fa_dk.array, atol=1e-3, rtol=1e-3)
    assert_trees_all_close(hax_dv.array, fa_dv.array, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_grad_group_query_attention(num_kv_heads):
    Batch = hax.Axis("batch", 2)
    KVHeads = hax.Axis("kv_heads", num_kv_heads)
    QHeadsPerGroup = hax.Axis("q_heads_per_group", 4 // num_kv_heads)
    Key = hax.Axis("Key", 8)
    QPos = hax.Axis("QPos", BLOCK_SIZE * 2)
    KPos = hax.Axis("KPos", BLOCK_SIZE * 2)

    mask = AttentionMask.causal()

    q = hax.random.normal(jrandom.PRNGKey(0), (Batch, KVHeads, QHeadsPerGroup, QPos, Key))
    k = hax.random.normal(jrandom.PRNGKey(1), (Batch, KVHeads, KPos, Key))
    v = hax.random.normal(jrandom.PRNGKey(2), (Batch, KVHeads, KPos, Key))

    @equinox.filter_value_and_grad
    def d_attn(qkv, fn):
        q, k, v = qkv
        x_out = fn(QPos, KPos, Key, q, k, v, mask=mask)
        return (x_out * x_out).mean().scalar()

    hax_val, (hax_dq, hax_dk, hax_dv) = d_attn((q, k, v), simple_attention_with_dropout)
    fa_val, (fa_dq, fa_dk, fa_dv) = d_attn(
        (q, k, v), functools.partial(flash_attention, inference=True, block_size=BLOCK_SIZE, mask=mask)
    )

    assert_trees_all_close(hax_val, fa_val, atol=1e-3, rtol=1e-3)
    assert hax_dq.axes == fa_dq.axes
    assert hax_dk.axes == fa_dk.axes
    assert hax_dv.axes == fa_dv.axes

    assert_trees_all_close(hax_dq.array, fa_dq.array, atol=1e-3, rtol=1e-3)
    assert_trees_all_close(hax_dk.array, fa_dk.array, atol=1e-3, rtol=1e-3)
    assert_trees_all_close(hax_dv.array, fa_dv.array, atol=1e-3, rtol=1e-3)


def test_fa_dropout_does_something():
    Key = hax.Axis("Key", 8)
    QPos = hax.Axis("QPos", BLOCK_SIZE * 2)
    KPos = hax.Axis("KPos", BLOCK_SIZE * 2)

    mask = hax.nn.attention.causal_mask(QPos, KPos)

    q = hax.random.normal(jrandom.PRNGKey(0), (QPos, Key))
    k = hax.random.normal(jrandom.PRNGKey(1), (KPos, Key))
    v = hax.random.normal(jrandom.PRNGKey(2), (KPos, Key))

    p_drop = 0.5

    fa_with_dropout = functools.partial(
        flash_attention, inference=False, dropout=p_drop, key=jrandom.PRNGKey(3), block_size=BLOCK_SIZE
    )
    fa_without_dropout = functools.partial(flash_attention, inference=True, block_size=BLOCK_SIZE)

    without_o = fa_without_dropout(QPos, KPos, Key, q, k, v, mask=mask)
    with_o = fa_with_dropout(QPos, KPos, Key, q, k, v, mask=mask)

    assert with_o.axes == without_o.axes
    mean = jnp.mean(jnp.isclose(with_o.array, without_o.array, atol=1e-3, rtol=1e-3))
    assert mean < 1e-2


@pytest.mark.parametrize("soft_cap", [True, False])
def test_tpu_flash_attention(soft_cap):
    if jax.devices()[0].device_kind != "tpu":
        pytest.skip("TPU-only test")

    if soft_cap:
        cap = 4
    else:
        cap = None

    Key = hax.Axis("Key", 128)
    QPos = hax.Axis("QPos", BLOCK_SIZE * 4)
    KPos = hax.Axis("KPos", BLOCK_SIZE * 4)
    with use_test_mesh():
        mask = AttentionMask.causal()

        q = hax.random.normal(jrandom.PRNGKey(0), (QPos, Key))
        k = hax.random.normal(jrandom.PRNGKey(1), (KPos, Key))
        v = hax.random.normal(jrandom.PRNGKey(2), (KPos, Key))

        flash_out = levanter.layers.attention._tpu_splash_attention(
            QPos,
            KPos,
            Key,
            q,
            k,
            v,
            inference=True,
            mask=mask,
            block_size=BLOCK_SIZE,
            scaling_factor=1 / math.sqrt(Key.size),  # Use the default scaling factor
            logits_soft_cap=cap,
        )
        hax_out = simple_attention_with_dropout(
            KPos,
            Key,
            q,
            k,
            v,
            mask=mask.materialize(QPos, KPos),
            dropout=0.0,
            scaling_factor=1 / math.sqrt(Key.size),
            logits_soft_cap=cap,
        )

        assert hax_out.axes == flash_out.axes
        assert_trees_all_close(hax_out.array, flash_out.array, atol=1e-3, rtol=1e-3)
