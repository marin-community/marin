# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the native Hero snapshot against the independent production experiment."""

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax import Axis
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.grug.sharding import compact_grug_mesh
from levanter.layers.attention import AttentionMask
from levanter.models.hero import HeroConfig
from levanter.models.hero_model import HeroLMHeadModel

from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer


@pytest.mark.parametrize("rope_fused", [False, True])
@pytest.mark.parametrize("expert_layout", ["bank", "individual"])
@pytest.mark.parametrize("segmented", [False, True])
def test_hero_export_load_matches_experiment_logits(rope_fused, expert_layout, segmented):
    exp_cfg = GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=12,
        shared_expert_intermediate_dim=12,
        num_shared_experts=2,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=8,
        num_layers=3,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=1,
        head_dim=8,
        max_seq_len=16,
        sliding_window=3,
        global_every=2,
        initializer_std=0.15,
        qk_mult=1.37,
        sconv=True,
        sconv_kernel=3,
        sconv_sites=("k", "attn", "mlp"),
        rope_fused=rope_fused,
        attention_implementation="reference",
        moe_implementation="ring",
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        experiment = Transformer.init(exp_cfg, key=jax.random.key(13))
        block = experiment.stacked_blocks.stacked
        # Identity-initialized convolutions and zero head gates would hide omitted weights.
        conv_weights = (block.attn.sconv_k.weight, block.sconv_attn.weight, block.sconv_mlp.weight)
        experiment = eqx.tree_at(
            lambda m: (
                m.stacked_blocks.stacked.attn.sconv_k.weight,
                m.stacked_blocks.stacked.sconv_attn.weight,
                m.stacked_blocks.stacked.sconv_mlp.weight,
                m.stacked_blocks.stacked.attn.attn_gate,
                m.stacked_blocks.stacked.mlp.router_bias,
            ),
            experiment,
            (
                *(jax.random.normal(jax.random.key(i + 1), x.shape) * 0.2 for i, x in enumerate(conv_weights)),
                jax.random.normal(jax.random.key(7), block.attn.attn_gate.shape) * 0.3,
                jax.random.normal(jax.random.key(8), block.mlp.router_bias.shape) * 0.2,
            ),
        )
        canonical = experiment.to_state_dict()
        exported = canonical
        if expert_layout == "individual":
            exported = {}
            for name, value in canonical.items():
                if ".mlp.experts." in name:
                    stem, projection = name.split(".mlp.experts.")
                    exported.update(
                        {f"{stem}.mlp.experts.{i}.{projection}": value[i] for i in range(exp_cfg.num_experts)}
                    )
                else:
                    exported[name] = value
        cfg = HeroConfig.from_hf_config(exp_cfg.to_hf_config(exp_cfg.vocab_size))
        cfg = dataclasses.replace(cfg, attention_implementation="reference", moe_implementation="ring")
        native = HeroLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(99))
        native = hax.named_jit(lambda m, state: m.from_state_dict(state))(native, exported)
        actual_weights = native.to_state_dict()
        assert actual_weights.keys() == canonical.keys()
        for name, value in canonical.items():
            np.testing.assert_array_equal(np.asarray(actual_weights[name]), np.asarray(value), err_msg=name)
        tokens = jnp.broadcast_to(jnp.arange(7, dtype=jnp.int32) * 3 + 1, (2 * jax.device_count(), 7))
        ids = hax.named(tokens, (Axis("batch", 2 * jax.device_count()), Axis("position", 7)))
        segments = jnp.broadcast_to(jnp.array([0, 0, 0, 1, 1, 1, 1]), tokens.shape)
        experiment_mask = GrugAttentionMask(is_causal=True, segment_ids=(segments, segments)) if segmented else None
        native_mask = AttentionMask.causal().with_segment_ids(hax.named(segments, ids.axes)) if segmented else None
        expected = np.asarray(eqx.filter_jit(lambda m, t: m.logits(t, mask=experiment_mask))(experiment, tokens))
        actual = np.asarray(hax.named_jit(lambda m, t: m(t, native_mask))(native, ids).array)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(actual.argmax(axis=-1), expected.argmax(axis=-1))
