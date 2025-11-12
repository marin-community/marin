# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import math

import equinox
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest
import equinox as eqx
from chex import assert_trees_all_close
from jax.lax import Precision
from jax.sharding import NamedSharding, PartitionSpec

import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis

from levanter.layers.attention import (
    AttentionBackend,
    AttentionConfig,
    AttentionMask,
    _bin_and_group_axes_by_function,
    _te_flash_attention,
    _tpu_splash_attention,
    AttentionWithSink,
    dot_product_attention,
)
from test_utils import skip_if_module_missing, skip_if_no_torch, use_test_mesh


@pytest.mark.skip
def test_causal_mask_blocking():
    pos = hax.Axis("pos", 128)
    key_pos = pos.alias("key_pos")

    mask = AttentionMask.causal()

    blocked_mask = mask.blocked(pos, 16).blocked(key_pos, 16)
    assert blocked_mask.Pos.size == 128 // 16
    assert blocked_mask.KeyPos.size == 128 // 16

    mat_blocked = blocked_mask.materialize()

    assert hax.all(mat_blocked == hax.nn.attention.causal_mask(pos.resize(8), key_pos.resize(8)))

    mat_mask = mask.materialize()

    for i in range(8):
        for j in range(8):
            assert mat_blocked.array[i, j] == jnp.any(mat_mask.array[i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16])


def test_causal_mask_slicing():
    pos = hax.Axis("pos", 128)
    key_pos = pos.alias("key_pos")

    mask = AttentionMask.causal()

    mat_mask = mask.materialize(pos, key_pos)
    mat_sliced = mask.materialize(pos, key_pos, q_slice=hax.dslice(7, 16), k_slice=hax.dslice(24, 16))

    for i in range(16):
        for j in range(16):
            assert mat_sliced.array[i, j] == mat_mask.array[7 + i, 24 + j]


def test_sliding_window_mask():
    Pos = hax.Axis("pos", 16)
    KeyPos = Pos.alias("key_pos")
    window = 4
    mask = AttentionMask.causal(sliding_window=window)
    mat = mask.materialize(Pos, KeyPos)
    q_pos = hax.arange(Pos)
    k_pos = hax.arange(KeyPos)
    diff = q_pos.broadcast_axis(KeyPos) - k_pos.broadcast_axis(Pos)
    expected = (diff >= 0) & (diff < window)
    assert hax.all(mat == expected)


def test_attention_sink():
    Pos = hax.Axis("position", 2)
    KeyPos = Pos.alias("key_pos")
    Head = hax.Axis("kv_heads", 1)
    QHead = hax.Axis("q_heads_per_group", 1)
    D = hax.Axis("head_size", 1)

    q = hax.zeros((Head, QHead, Pos, D))
    k = hax.zeros((Head, KeyPos, D))
    v = hax.ones((Head, KeyPos, D))
    sink = hax.zeros((Head, QHead))

    out = dot_product_attention(
        Pos.name,
        KeyPos.name,
        D.name,
        q,
        k,
        v,
        attn_sink=sink,
    )

    expected = np.full((1, 1, 2, 1), 2.0 / 3)
    assert_trees_all_close(out.array, expected)


def test_attention_with_sink_module():
    Pos = hax.Axis("position", 2)
    Embed = hax.Axis("embed", 1)

    config = AttentionConfig(Embed=Embed, num_heads=1, num_kv_heads=1, use_bias=True)
    attn = AttentionWithSink.init(config, key=jrandom.PRNGKey(0))

    attn = eqx.tree_at(lambda a: a.q_proj.weight, attn, hax.zeros(attn.q_proj.weight.axes))
    attn = eqx.tree_at(lambda a: a.q_proj.bias, attn, hax.zeros(attn.q_proj.bias.axes))
    attn = eqx.tree_at(lambda a: a.k_proj.weight, attn, hax.zeros(attn.k_proj.weight.axes))
    attn = eqx.tree_at(lambda a: a.k_proj.bias, attn, hax.zeros(attn.k_proj.bias.axes))
    attn = eqx.tree_at(lambda a: a.v_proj.weight, attn, hax.zeros(attn.v_proj.weight.axes))
    attn = eqx.tree_at(lambda a: a.v_proj.bias, attn, hax.ones(attn.v_proj.bias.axes))
    attn = eqx.tree_at(lambda a: a.o_proj.weight, attn, hax.ones(attn.o_proj.weight.axes))
    attn = eqx.tree_at(lambda a: a.o_proj.bias, attn, hax.zeros(attn.o_proj.bias.axes))

    x = hax.zeros((Pos, Embed))
    out = attn(x, None)

    expected = np.full((2, 1), 2.0 / 3)
    assert_trees_all_close(out.array, expected)


def test_te_bin_and_group_axes_by_function():
    QPos = hax.Axis("QPos", 128)
    KPos = hax.Axis("KPos", 128)
    D = hax.Axis("D", 64)
    H = hax.Axis("H", 8)
    B = hax.Axis("B", 32)
    G = hax.Axis("G", 4)

    q = hax.zeros((B, QPos, H, D))
    k = hax.zeros((B, KPos, H, D))
    v = hax.zeros((B, KPos, H, D))

    q_c, k_c, v_c = _bin_and_group_axes_by_function(q, k, v, "QPos", "KPos", "D")
    assert q_c["B"] == [B]
    assert k_c["B"] == [B]
    assert v_c["B"] == [B]

    assert q_c["S"] == [QPos]
    assert k_c["S"] == [KPos]
    assert v_c["S"] == [KPos]

    assert q_c["H"] == [H]
    assert k_c["H"] == [H]
    assert v_c["H"] == [H]

    assert q_c["D"] == [D]
    assert k_c["D"] == [D]
    assert v_c["D"] == [D]

    gq = hax.zeros((B, QPos, H, G, D))
    q_c, k_c, v_c = _bin_and_group_axes_by_function(gq, k, v, "QPos", "KPos", "D")
    assert q_c["H"] == [H, G]
    assert k_c["H"] == [H]
    assert v_c["H"] == [H]

    gk = hax.zeros((B, KPos, G, H, D))
    with pytest.raises(ValueError):
        _bin_and_group_axes_by_function(q, gk, v, "QPos", "KPos", "D")

    with pytest.raises(ValueError):
        _bin_and_group_axes_by_function(gq, gk, v, "QPos", "KPos", "D")

    for gk_axes in [(B, KPos, G, H, D), (B, KPos, G, H, D), (G, B, KPos, H, D)]:
        gk = hax.zeros(gk_axes)
        q_c, k_c, v_c = _bin_and_group_axes_by_function(gq, gk, gk, "QPos", "KPos", "D")
        assert q_c["H"] == [H, G]
        assert k_c["H"] == [H, G]
        assert v_c["H"] == [H, G]

    # axes that come before QPos are treated as batch (if shared)
    gq = hax.zeros((G, B, QPos, H, D))
    for gk_axes in [(B, KPos, H, G, D), (B, KPos, G, H, D), (G, B, KPos, H, D)]:
        gk = hax.zeros(gk_axes)
        q_c, k_c, v_c = _bin_and_group_axes_by_function(gq, gk, gk, "QPos", "KPos", "D")
        assert q_c["H"] == [H]
        assert k_c["H"] == [H]
        assert v_c["H"] == [H]
        assert q_c["B"] == [G, B]
        assert k_c["B"] == [G, B]
        assert v_c["B"] == [G, B]


def test_mqa_te_bin_and_group_axes_by_function():
    B = hax.Axis("B", 32)
    QPos = hax.Axis("QPos", 128)
    KPos = hax.Axis("KPos", 128)
    D = hax.Axis("D", 64)
    G = hax.Axis("G", 4)

    # MQA
    gq = hax.zeros((B, QPos, G, D))
    k = hax.zeros((B, KPos, D))
    v = hax.zeros((B, KPos, D))

    q_c, k_c, v_c = _bin_and_group_axes_by_function(gq, k, v, "QPos", "KPos", "D")
    assert q_c["H"] == [G]
    assert k_c["H"] == []
    assert v_c["H"] == []

    gk = hax.zeros((B, KPos, G, D))
    with pytest.raises(ValueError):
        # don't currently handle dim in Q and K but not V
        _bin_and_group_axes_by_function(gq, gk, v, "QPos", "KPos", "D")


@skip_if_module_missing("transformer_engine")
@pytest.mark.parametrize("q_heads", [1, 2, 4])
def test_llama_attention_uses_te(q_heads):
    QPos = hax.Axis("position", 128)
    KPos = hax.Axis("key_position", 128)
    B = hax.Axis("batch", 8)
    Head = hax.Axis("kv_heads", 8)
    D = hax.Axis("head_size", 64)
    Q_Head = hax.Axis("q_heads_per_group", q_heads)
    mask = AttentionMask.causal()

    q = hax.zeros((B, Head, Q_Head, QPos, D), dtype=jnp.bfloat16)
    k = hax.zeros((B, Head, KPos, D), dtype=jnp.bfloat16)
    v = hax.zeros((B, Head, KPos, D), dtype=jnp.bfloat16)

    # mostly testing that this doesn't crash
    out = _te_flash_attention(
        "position",
        "key_position",
        "head_size",
        q,
        k,
        v,
        mask,
        attention_dtype=jnp.bfloat16,
        scaling_factor=1 / math.sqrt(D.size),
    )

    assert_trees_all_close(out.array, 0.0)


@skip_if_module_missing("transformer_engine")
def test_gpt2_attention_uses_te():
    QPos = hax.Axis("position", 128)
    KPos = hax.Axis("key_position", 128)
    B = hax.Axis("batch", 8)
    Head = hax.Axis("heads", 8)
    D = hax.Axis("head_size", 64)
    mask = AttentionMask.causal()

    q = hax.zeros((B, Head, QPos, D), dtype=jnp.bfloat16)
    k = hax.zeros((B, Head, KPos, D), dtype=jnp.bfloat16)
    v = hax.zeros((B, Head, KPos, D), dtype=jnp.bfloat16)

    out = _te_flash_attention(
        "position",
        "key_position",
        "head_size",
        q,
        k,
        v,
        mask,
        attention_dtype=jnp.bfloat16,
        scaling_factor=1 / math.sqrt(D.size),
    )
    assert_trees_all_close(out.array, 0.0)


@skip_if_module_missing("transformer_engine")
def test_te_flash_attention_non_causal_mask_raises():
    QPos = hax.Axis("position", 16)
    KPos = hax.Axis("key_position", 16)
    B = hax.Axis("batch", 2)
    Head = hax.Axis("heads", 2)
    D = hax.Axis("head_size", 8)

    q = hax.zeros((B, Head, QPos, D), dtype=jnp.bfloat16)
    k = hax.zeros((B, Head, KPos, D), dtype=jnp.bfloat16)
    v = hax.zeros((B, Head, KPos, D), dtype=jnp.bfloat16)

    explicit = hax.ones((QPos, KPos))
    mask = AttentionMask.explicit(explicit)

    with pytest.raises(NotImplementedError):
        _te_flash_attention(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask,
            attention_dtype=jnp.bfloat16,
            scaling_factor=1 / math.sqrt(D.size),
        )


def test_tpu_splash_attention():
    if jax.default_backend() != "tpu":
        pytest.skip("TPU only")

    BLOCK_SIZE = 512

    Head = hax.Axis("Head", 8)
    Key = hax.Axis("Key", 128)  # splash only supports 128
    QPos = hax.Axis("QPos", BLOCK_SIZE * 2)
    KPos = hax.Axis("KPos", BLOCK_SIZE * 2)

    q = hax.random.normal(jrandom.PRNGKey(0), (QPos, Head, Key)) * 0.02
    k = hax.random.normal(jrandom.PRNGKey(1), (KPos, Head, Key)) * 0.02
    v = hax.random.normal(jrandom.PRNGKey(2), (KPos, Head, Key)) * 0.02

    mask = AttentionMask.causal()

    with use_test_mesh():
        flash_out = _tpu_splash_attention(
            QPos,
            KPos,
            Key,
            q,
            k,
            v,
            inference=True,
            mask=mask,
            block_size=BLOCK_SIZE,
            scaling_factor=1 / math.sqrt(Head.size),
        )
        hax_out = hax.nn.attention.dot_product_attention(KPos, Key, q, k, v, mask=mask.materialize(QPos, KPos))
        assert hax_out.axes == flash_out.axes
        assert_trees_all_close(hax_out.array, flash_out.array, atol=1e-3, rtol=1e-3)


def test_tpu_splash_attention_sliding_window():
    if jax.default_backend() != "tpu":
        pytest.skip("TPU only")

    BLOCK_SIZE = 512

    Head = hax.Axis("Head", 8)
    Key = hax.Axis("Key", 128)  # splash only supports 128
    QPos = hax.Axis("QPos", BLOCK_SIZE * 2)
    KPos = hax.Axis("KPos", BLOCK_SIZE * 2)

    q = hax.random.normal(jrandom.PRNGKey(0), (QPos, Head, Key)) * 0.02
    k = hax.random.normal(jrandom.PRNGKey(1), (KPos, Head, Key)) * 0.02
    v = hax.random.normal(jrandom.PRNGKey(2), (KPos, Head, Key)) * 0.02

    mask = AttentionMask.causal(sliding_window=BLOCK_SIZE)

    with use_test_mesh():
        flash_out = _tpu_splash_attention(
            QPos,
            KPos,
            Key,
            q,
            k,
            v,
            inference=True,
            mask=mask,
            block_size=BLOCK_SIZE,
            scaling_factor=1 / math.sqrt(Key.size),
        )
        hax_out = hax.nn.attention.dot_product_attention(KPos, Key, q, k, v, mask=mask.materialize(QPos, KPos))
        assert hax_out.axes == flash_out.axes
        assert_trees_all_close(hax_out.array, flash_out.array, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("impl", ["default", "jax_flash", "vanilla"])
def test_segment_ids_are_respected(impl):
    # test that we can't attend to something outside of the range
    # splash needs 128
    D = 128 if impl == "default" else 2
    L = 256
    Pos = Axis("Pos", L)
    Head = Axis("Head", D)

    keys = np.zeros((L, D), dtype=np.float32)
    keys[0, 0] = 100.0  # really want to attend to this
    values = np.zeros((L, D), dtype=np.float32)
    values[0, 1] = 300.0  # check if we did attend
    KPos = Pos.alias("KPos")

    query = np.ones((L, D), dtype=np.float32)

    query = hax.named(query, (Pos, Head))
    keys = hax.named(keys, (KPos, Head))
    values = hax.named(values, (KPos, Head))

    with use_test_mesh() as dp_mesh:
        query, keys, values = jax.device_put(
            [query, keys, values],
            NamedSharding(dp_mesh, PartitionSpec(ResourceAxis.DATA, None)),
        )

        segment_ids = np.array([0, 0, 0] + [1] * (L - 3), dtype=np.int32)
        segment_ids = jax.device_put(segment_ids, NamedSharding(dp_mesh, PartitionSpec(ResourceAxis.DATA)))
        segment_ids = hax.named(segment_ids, (Pos,))
        mask = AttentionMask(causal_offset=0, segment_ids=segment_ids)

        result = jit_dpa(
            Pos,
            KPos,
            Head,
            query,
            keys,
            values,
            attn_backend=AttentionBackend(impl),
            mask=mask,
            flash_block_size=128,
        )

    # the first 3 positions should all have a value of 300.0
    assert_trees_all_close(result.array[0:3, 1], 300.0, atol=1e-3, rtol=1e-3)
    # the rest should be 0
    assert_trees_all_close(result.array[3:, 1], 0.0, atol=1e-3, rtol=1e-3)


# TODO: fix flash attention for offsets
@pytest.mark.parametrize("impl", ["vanilla"])
def test_causal_offset_cross_attention(impl):
    """Verify that a positive causal *offset* relaxes the masking during cross-attention.

    We compare the output of ``dot_product_attention`` when provided the structured
    ``AttentionMask`` with *offset* against the output obtained when passing the
    *materialised* boolean mask explicitly – they should be identical.
    """

    offset = 2
    FullPos = Axis("pos", 6)
    Pos = FullPos.resize(offset)
    KeyPos = Axis("key_pos", 6)
    Head = Axis("head", 2)
    KeyDim = Axis("embed", 4)

    k = hax.random.normal(jrandom.PRNGKey(1), (KeyPos, Head, KeyDim))
    v = hax.random.normal(jrandom.PRNGKey(2), (KeyPos, Head, KeyDim))
    q = hax.random.normal(jrandom.PRNGKey(0), (FullPos, Head, KeyDim))
    q_sub = q["pos", 4:6]

    struct_mask = AttentionMask.causal(offset=FullPos.size - offset)

    offset_out = jit_dpa(
        Pos,
        KeyPos,
        KeyDim,
        q_sub,
        k,
        v,
        mask=struct_mask,
        inference=True,
        attn_backend=AttentionBackend(impl),
        flash_block_size=1,
        precision=Precision.HIGHEST,
    )

    mask = AttentionMask.causal()

    full_out = jit_dpa(
        FullPos,
        KeyPos,
        KeyDim,
        q,
        k,
        v,
        mask=mask,
        flash_block_size=1,
        precision=Precision.HIGHEST,
    )

    # The output should be the same, since the mask is relaxed by the offset
    assert_trees_all_close(offset_out.array, full_out.array[4:6, :], atol=3e-3, rtol=3e-3)

    # sanity check: output should be wrong if we don't use the offset
    wrong_out = jit_dpa(
        Pos,
        KeyPos,
        KeyDim,
        q_sub,
        k,
        v,
        mask=mask,
        inference=True,
        attn_backend=AttentionBackend(impl),
        flash_block_size=1,
        precision=Precision.HIGHEST,
    )

    assert not jnp.allclose(
        offset_out.array, wrong_out.array, atol=2e-3, rtol=2e-3
    ), "Output should differ without offset"


# This is a bottleneck in tests
jit_dpa = equinox.filter_jit(dot_product_attention)


# Reference implementation of Attention Sink with Sliding Window from https://github.com/openai/gpt-oss/blob/main/gpt_oss/triton/attention.py
def sink_attention_ref_gpt_oss(
    query,
    key,
    value,
    sinks,
    sm_scale: float = 0.125,
    sliding_window: int | None = None,
    start_q=0,
):
    import torch

    batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim = query.shape
    batch_size, num_keys, num_key_value_heads, head_dim = key.shape

    sinks = sinks.view(1, num_key_value_heads, num_key_value_groups, 1, 1).float()
    key = key.unsqueeze(3)
    value = value.unsqueeze(3)

    pos_keys = torch.arange(num_keys, device=query.device)
    pos_queries = torch.arange(num_queries, device=query.device) + start_q
    mask = pos_keys[None, :] > pos_queries[:, None]
    mask = mask.float().masked_fill(mask, float("-inf"))

    if sliding_window:
        too_old = pos_keys[None, :] < (pos_queries[:, None] - sliding_window + 1)
        mask.masked_fill_(too_old, float("-inf"))

    logits = torch.einsum("bqhmd,bkhmd->bhmqk", query.float(), key.float()) * sm_scale
    logits = logits + mask[None, None, None, :, :]

    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    logits_or_sinks_max = torch.maximum(sinks, logits_max)
    sinks = torch.exp(sinks - logits_or_sinks_max)
    unnormalized_scores = torch.exp(logits - logits_or_sinks_max)
    normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks
    scores = unnormalized_scores / normalizer

    output = torch.einsum("bhmqk,bkhmd->bqhmd", scores, value.float())

    output = output.reshape(batch_size, num_queries, num_key_value_heads * num_key_value_groups * head_dim).bfloat16()
    return output


def sink_attention(
    query,
    key,
    value,
    sinks,
    sm_scale: float = 0.125,
    sliding_window: int | None = None,
    start_q: int = 0,
    *,
    attn_backend: AttentionBackend | None = None,
    block_size: int | None = None,
    inference: bool = True,
):
    import torch

    batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim = query.shape
    _, num_keys, _, _ = key.shape

    # Convert torch tensors to JAX NamedArrays
    q_jax = jnp.array(query.to(torch.float32).cpu().numpy(), dtype=jnp.float32)
    k_jax = jnp.array(key.to(torch.float32).cpu().numpy(), dtype=jnp.float32)
    v_jax = jnp.array(value.to(torch.float32).cpu().numpy(), dtype=jnp.float32)
    sink_jax = jnp.array(
        sinks.view(num_key_value_heads, num_key_value_groups).to(torch.float32).cpu().numpy(),
        dtype=jnp.float32,
    )

    Batch = Axis("batch", batch_size)
    QPos = Axis("q_pos", num_queries)
    KPos = Axis("k_pos", num_keys)
    KVHead = Axis("kv_heads", num_key_value_heads)
    KVGroup = Axis("kv_groups", num_key_value_groups)
    D = Axis("head_dim", head_dim)

    q = hax.named(q_jax, (Batch, QPos, KVHead, KVGroup, D))
    k = hax.named(k_jax, (Batch, KPos, KVHead, D))
    v = hax.named(v_jax, (Batch, KPos, KVHead, D))
    sink = hax.named(sink_jax, (KVHead, KVGroup))

    pos_queries = jnp.arange(num_queries, dtype=jnp.int32) + int(start_q)
    pos_keys = jnp.arange(num_keys, dtype=jnp.int32)
    mask_arr = pos_queries[:, None] >= pos_keys[None, :]
    if sliding_window is not None:
        mask_arr &= pos_queries[:, None] - sliding_window + 1 <= pos_keys[None, :]
    mask = hax.named(mask_arr, (QPos, KPos))

    out = dot_product_attention(
        QPos,
        KPos,
        D,
        q,
        k,
        v,
        mask=mask,
        scaling_factor=sm_scale,
        attn_backend=attn_backend,
        flash_block_size=block_size,
        inference=inference,
        attn_sink=sink,
    )

    out_np = np.asarray(out.array, dtype=np.float32)
    out_torch = torch.from_numpy(out_np).to(query.device)
    out_torch = out_torch.view(batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim)
    return out_torch.reshape(batch_size, num_queries, -1).bfloat16()


def sink_attention_vanilla(
    query,
    key,
    value,
    sinks,
    sm_scale: float = 0.125,
    sliding_window: int | None = None,
    start_q=0,
):
    return sink_attention(
        query,
        key,
        value,
        sinks,
        sm_scale,
        sliding_window,
        start_q,
        attn_backend=None,
        block_size=None,
        inference=True,
    )


def sink_attention_jax_flash(
    query,
    key,
    value,
    sinks,
    sm_scale: float = 0.125,
    sliding_window: int | None = None,
    start_q: int = 0,
    *,
    block_size: int = 64,
):
    return sink_attention(
        query,
        key,
        value,
        sinks,
        sm_scale,
        sliding_window,
        start_q,
        attn_backend=AttentionBackend.JAX_FLASH,
        block_size=block_size,
        inference=True,
    )


@skip_if_no_torch
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_queries", [1, 128])
@pytest.mark.parametrize("num_keys", [128, 32])
@pytest.mark.parametrize("num_key_value_heads", [8])
@pytest.mark.parametrize("num_key_value_groups", [8])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("sm_scale", [0.125])
@pytest.mark.parametrize("sliding_window", [None, 128])
@pytest.mark.parametrize("start_q", [0, 5])
def test_attention_equivalence(
    batch_size,
    num_queries,
    num_keys,
    num_key_value_heads,
    num_key_value_groups,
    head_dim,
    sm_scale,
    sliding_window,
    start_q,
):
    import torch

    if num_queries > num_keys:
        pytest.skip("too many queries")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = torch.randn(
        batch_size,
        num_queries,
        num_key_value_heads,
        num_key_value_groups,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        batch_size,
        num_keys,
        num_key_value_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        batch_size,
        num_keys,
        num_key_value_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    sinks = torch.randn(num_key_value_heads * num_key_value_groups, device=device, dtype=torch.bfloat16)

    o1 = sink_attention_vanilla(q, k, v, sinks, sm_scale, sliding_window, start_q)
    o2 = sink_attention_ref_gpt_oss(q, k, v, sinks, sm_scale, sliding_window, start_q)

    torch.testing.assert_close(o1, o2)


@skip_if_no_torch
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_queries", [128])
@pytest.mark.parametrize("num_keys", [128])
@pytest.mark.parametrize("num_key_value_heads", [8])
@pytest.mark.parametrize("num_key_value_groups", [8])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("sm_scale", [0.125])
@pytest.mark.parametrize("sliding_window", [None, 64])
@pytest.mark.parametrize("start_q", [0, 5])
@pytest.mark.parametrize("block_size", [64, 128])
def test_attention_equivalence_jax_flash(
    batch_size,
    num_queries,
    num_keys,
    num_key_value_heads,
    num_key_value_groups,
    head_dim,
    sm_scale,
    sliding_window,
    start_q,
    block_size,
):
    """Make sure the JAX backend is tested"""
    import torch

    if num_queries > num_keys:
        pytest.skip("too many queries")
    if (num_queries % block_size) != 0 or (num_keys % block_size) != 0:
        pytest.skip("block size must divide sequence lengths for JAX Flash path")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = torch.randn(
        batch_size,
        num_queries,
        num_key_value_heads,
        num_key_value_groups,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        batch_size,
        num_keys,
        num_key_value_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        batch_size,
        num_keys,
        num_key_value_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    sinks = torch.randn(num_key_value_heads * num_key_value_groups, device=device, dtype=torch.bfloat16)

    o1 = sink_attention_jax_flash(q, k, v, sinks, sm_scale, sliding_window, start_q, block_size=block_size)
    o2 = sink_attention_ref_gpt_oss(q, k, v, sinks, sm_scale, sliding_window, start_q)

    torch.testing.assert_close(o1, o2)
