# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, set_mesh
from levanter.grug.attention import AttentionMask

from experiments.grug.moe_hero_ep import model


def _explicit_mesh(*axis_sizes):
    return Mesh(
        np.asarray(jax.devices()).reshape(*axis_sizes),
        ("replica_dcn", "data", "expert", "context", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )


def _tiny_config():
    return model.GrugModelConfig(
        vocab_size=128,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=1,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=2,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        global_every=2,
        capacity_factor=1.0,
        initializer_std=0.5 / math.sqrt(32),
        qk_mult=1.3,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
        report_capacity_overflow=True,
    )


def _attention_output(cfg, x, is_global):
    # Same key and shapes for every config, so each attention module gets identical weights. The
    # scan passes the layer flag traced, as both disable_rope and is_global; jit with it as an
    # argument to exercise that path.
    attn = model.CausalSelfAttention.init(cfg, key=jax.random.key(61))
    return eqx.filter_jit(lambda h, flag: attn(h, AttentionMask.causal(), disable_rope=flag, is_global=flag))(
        x, jnp.asarray(is_global)
    )


def test_qk_mult_long_scale_scales_global_layers_and_leaves_local_layers_alone():
    mesh = _explicit_mesh(1, 1, 1, 1, 1)
    long_scale = 1.3
    base = dataclasses.replace(_tiny_config(), qk_mult=1.3)
    scaled = dataclasses.replace(base, qk_mult_long_scale=long_scale)
    # A global layer under the long scale must see the temperature a uniform qk_mult=1.3*s gives.
    uniform = dataclasses.replace(base, qk_mult=1.3 * long_scale)
    x = jax.random.normal(jax.random.key(62), (2, 8, base.hidden_dim))

    with set_mesh(mesh):
        local_base = _attention_output(base, x, is_global=False)
        local_scaled = _attention_output(scaled, x, is_global=False)
        global_base = _attention_output(base, x, is_global=True)
        global_scaled = _attention_output(scaled, x, is_global=True)
        global_uniform = _attention_output(uniform, x, is_global=True)

    # Two sequential multiplies and one product multiplier round differently, and a selected
    # scalar compiles to a different fusion, so the match is to float32 rounding rather than
    # bitwise; the outputs are O(0.1).
    np.testing.assert_allclose(local_scaled, local_base, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(global_scaled, global_uniform, rtol=1e-6, atol=1e-6)
    assert np.max(np.abs(np.asarray(global_scaled) - np.asarray(global_base))) > 1e-4


def test_qk_mult_long_scale_reaches_the_logits_through_the_layer_scan():
    mesh = _explicit_mesh(1, 1, 1, 1, 1)
    base = dataclasses.replace(_tiny_config(), num_layers=2, rope_fused=True)
    tokens = jax.random.randint(jax.random.key(63), (2, 8), 0, base.vocab_size)

    def logits(cfg):
        transformer = model.Transformer.init(cfg, key=jax.random.key(64))
        return eqx.filter_jit(transformer.logits)(tokens)

    with set_mesh(mesh):
        default = logits(base)
        scaled = logits(dataclasses.replace(base, qk_mult_long_scale=1.3))

    # Layer 1 is global (global_every=2). The logits change only if the layer scan passes the
    # per-layer is_global flag through to attention.
    assert np.max(np.abs(np.asarray(scaled) - np.asarray(default))) > 1e-4


def _log_scaling_logits(cfg, tokens, mask=None):
    with set_mesh(_explicit_mesh(1, 1, 1, 1, 1)):
        transformer = model.Transformer.init(cfg, key=jax.random.key(64))
        return np.asarray(eqx.filter_jit(transformer.logits)(tokens, mask))


def _log_scaling_config():
    # Layer 0 is local and layer 1 global. The capacity keeps every token, so a row's MoE output
    # depends only on that row and the causal structure of the logits is exact.
    return dataclasses.replace(_tiny_config(), num_layers=2, rope_fused=True, capacity_factor=4.0)


def _log_scaling_tokens(cfg):
    return jax.random.randint(jax.random.key(63), (1, cfg.max_seq_len), 0, cfg.vocab_size)


def test_log_scaling_scales_only_rows_beyond_the_floor_and_matches_a_uniform_scale():
    base = _log_scaling_config()
    tokens = _log_scaling_tokens(base)
    off = _log_scaling_logits(base, tokens)
    scaled = _log_scaling_logits(dataclasses.replace(base, log_scaling_n_floor=4), tokens)
    # The last row sees 8 keys, twice the floor.
    uniform = _log_scaling_logits(dataclasses.replace(base, qk_mult_long_scale=1 + 0.1 * math.log(2)), tokens)

    # Rows 0-3 see at most 4 keys. Their factor is exactly 1, and an extra multiply only moves the
    # fusion, so they match to float32 rounding; the logits are O(0.1).
    np.testing.assert_allclose(scaled[:, :4], off[:, :4], rtol=1e-5, atol=1e-5)
    assert np.max(np.abs(scaled[:, 4:] - off[:, 4:])) > 1e-4
    np.testing.assert_allclose(scaled[:, -1], uniform[:, -1], rtol=1e-5, atol=1e-5)


def test_log_scaling_counts_keys_from_the_document_start_and_skips_padding():
    base = _log_scaling_config()
    tokens = _log_scaling_tokens(base)
    # Left padding, a three-token document, a four-token document that restarts the count.
    segment_ids = jnp.asarray([[-1, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.int32)
    mask = AttentionMask.causal().with_segment_ids(segment_ids)
    off = _log_scaling_logits(base, tokens, mask)
    scaled = _log_scaling_logits(dataclasses.replace(base, log_scaling_n_floor=3), tokens, mask)
    # Row 7 is the fourth token of its document, not the eighth of the sequence.
    uniform = _log_scaling_logits(dataclasses.replace(base, qk_mult_long_scale=1 + 0.1 * math.log(4 / 3)), tokens, mask)

    # The padded row and every token that sees at most 3 keys in its document stay unchanged, including
    # rows 4-6, whose sequence position alone would put them past the floor.
    np.testing.assert_allclose(scaled[:, :7], off[:, :7], rtol=1e-5, atol=1e-5)
    assert np.max(np.abs(scaled[:, 7] - off[:, 7])) > 1e-4
    np.testing.assert_allclose(scaled[:, 7], uniform[:, 7], rtol=1e-5, atol=1e-5)


def test_log_scaling_keeps_a_bf16_query_bf16(monkeypatch):
    # XLA:CPU would run a promoted Q without complaint; FA4 on GPU takes the bf16 kernels only if Q stays bf16.
    cfg = dataclasses.replace(_tiny_config(), log_scaling_n_floor=2)
    x = jax.random.normal(jax.random.key(62), (2, 8, cfg.hidden_dim), dtype=jnp.bfloat16)
    # A strong float32, like the real factor. A weakly typed scale would leave Q bf16 even without the cast.
    scale = jnp.full((2, 8), 1.125, dtype=jnp.float32)
    query_dtypes = []
    real_attention = model.attention

    def recording_attention(q, *args, **kwargs):
        query_dtypes.append(q.dtype)
        return real_attention(q, *args, **kwargs)

    monkeypatch.setattr(model, "attention", recording_attention)
    with set_mesh(_explicit_mesh(1, 1, 1, 1, 1)):
        attn = model.CausalSelfAttention.init(cfg, key=jax.random.key(61))
        attn = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf, attn)
        attn(x, AttentionMask.causal(), is_global=jnp.asarray(True), global_query_scale=scale)

    assert query_dtypes == [jnp.bfloat16]
