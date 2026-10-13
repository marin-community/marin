# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, set_mesh
from levanter.grug.attention import AttentionMask

from experiments.grug.moe_hero_ep import model


def _explicit_mesh(*axis_sizes):
    return Mesh(
        np.asarray(jax.devices()).reshape(*axis_sizes),
        ("replica_dcn", "data", "context", "expert", "model"),
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


@pytest.mark.parametrize("rope_fused", [False, True])
def test_qk_mult_long_scale_scales_global_layers_and_leaves_local_layers_alone(rope_fused):
    mesh = _explicit_mesh(1, 1, 1, 1, 1)
    long_scale = 1.3
    base = dataclasses.replace(_tiny_config(), qk_mult=1.3, rope_fused=rope_fused)
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


def test_qk_mult_long_scale_of_one_keeps_the_model_output():
    mesh = _explicit_mesh(1, 1, 1, 1, 1)
    base = dataclasses.replace(_tiny_config(), num_layers=2, rope_fused=True)
    tokens = jax.random.randint(jax.random.key(63), (2, 8), 0, base.vocab_size)

    def logits(cfg):
        transformer = model.Transformer.init(cfg, key=jax.random.key(64))
        return eqx.filter_jit(transformer.logits)(tokens)

    with set_mesh(mesh):
        default = logits(base)
        one = logits(dataclasses.replace(base, qk_mult_long_scale=1.0))
        scaled = logits(dataclasses.replace(base, qk_mult_long_scale=1.3))

    np.testing.assert_array_equal(one, default)
    # Layer 1 is global (global_every=2), so a long scale other than 1.0 must reach the logits.
    assert np.max(np.abs(np.asarray(scaled) - np.asarray(default))) > 1e-4


def test_qk_mult_long_scale_hf_config_roundtrips_and_legacy_configs_load_with_one():
    cfg = dataclasses.replace(_tiny_config(), qk_mult=1.3, qk_mult_long_scale=1.2154)

    serialized = cfg.to_hf_config(cfg.vocab_size).to_dict()
    roundtripped = model.GrugModelConfig.from_hf_config(model.GrugMoeHfConfig(**serialized))
    serialized.pop("qk_mult_long_scale")
    legacy = model.GrugModelConfig.from_hf_config(model.GrugMoeHfConfig(**serialized))

    assert roundtripped.qk_mult == 1.3
    assert roundtripped.qk_mult_long_scale == 1.2154
    assert legacy.qk_mult_long_scale == 1.0
