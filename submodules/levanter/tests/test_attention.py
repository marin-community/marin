import math

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest
from chex import assert_trees_all_close
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import haliax as hax
from haliax import Axis

from levanter.layers.attention import (
    AttentionBackend,
    AttentionMask,
    _bin_and_group_axes_by_function,
    _te_flash_attention,
    _tpu_splash_attention,
    dot_product_attention,
)
from test_utils import skip_if_module_missing


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

    with jax.sharding.Mesh(jax.devices(), ("dp",)):
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

    dp_mesh = Mesh(jax.devices(), ("dp",))
    query, keys, values = jax.device_put(
        [query, keys, values], NamedSharding(dp_mesh, PartitionSpec("dp", None))
    )

    segment_ids = np.array([0, 0, 0] + [1] * (L - 3), dtype=np.int32)
    segment_ids = jax.device_put(segment_ids, NamedSharding(dp_mesh, PartitionSpec("dp")))
    segment_ids = hax.named(segment_ids, (Pos,))
    mask = AttentionMask(is_causal=True, segment_ids=segment_ids)

    with dp_mesh:
        result = hax.named_jit(dot_product_attention)(
            Pos, KPos, Head, query, keys, values, attn_backend=AttentionBackend(impl), mask=mask, flash_block_size=128
        )

    # the first 3 positions should all have a value of 300.0
    assert_trees_all_close(result.array[0:3, 1], 300.0, atol=1e-3, rtol=1e-3)
    # the rest should be 0
    assert_trees_all_close(result.array[3:, 1], 0.0, atol=1e-3, rtol=1e-3)
