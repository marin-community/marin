# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax import Axis
from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.models.hero import HeroConfig
from levanter.models.hero_model import HeroLMHeadModel
from levanter.testing.inference import assert_mixed_paged_decode_matches_forward


@pytest.mark.parametrize("sliding_window", [2, 16])
@pytest.mark.parametrize("rope_fused", [False, True])
@jax.default_matmul_precision("highest")
def test_hero_paged_decode_matches_full_forward(sliding_window, rope_fused):
    """Mixed chunked prefill/decode preserves per-request positions and Hero attention and convolution history."""
    cfg = HeroConfig(
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
        sliding_window=sliding_window,
        global_every=2,
        sconv_kernel=3,
        rope_fused=rope_fused,
        initializer_std=0.2,
        attention_implementation="reference",
        inference_attention_implementation="reference",
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = HeroLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(17))
        # Exercise the learned head gates, whose initializer is identically zero.
        blocks = model.transformer.stacked_blocks.stacked
        for site in [
            lambda b: b.attn.attn_gate,
            lambda b: b.attn.sconv_k.weight,
            lambda b: b.sconv_attn.weight,
            lambda b: b.sconv_mlp.weight,
        ]:
            array = site(blocks)
            blocks = eqx.tree_at(site, blocks, jax.random.normal(jax.random.key(42), array.shape) * 0.2)
        model = eqx.tree_at(lambda m: m.transformer.stacked_blocks.stacked, model, blocks)
        assert_mixed_paged_decode_matches_forward(model)


@pytest.mark.parametrize("prompt_length", [3, 4])
@jax.default_matmul_precision("highest")
def test_hero_engine_clones_preserve_convolution_history(prompt_length):
    """Cloning partial and complete KV pages preserves all emitted tokens and scores."""
    cfg = HeroConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=12,
        shared_expert_intermediate_dim=12,
        num_shared_experts=2,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=8,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=1,
        head_dim=8,
        max_seq_len=16,
        sliding_window=3,
        global_every=2,
        sconv_kernel=3,
        initializer_std=0.2,
        attention_implementation="reference",
        inference_attention_implementation="reference",
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = HeroLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(29))
        blocks = model.transformer.stacked_blocks.stacked
        for site in [lambda b: b.attn.sconv_k.weight, lambda b: b.sconv_attn.weight, lambda b: b.sconv_mlp.weight]:
            weight = site(blocks)
            blocks = eqx.tree_at(site, blocks, jax.random.normal(jax.random.key(31), weight.shape) * 0.2)
        model = eqx.tree_at(lambda m: m.transformer.stacked_blocks.stacked, model, blocks)
        config = InferenceEngineConfig(
            max_seq_len=16,
            max_pages=16,
            max_seqs=2,
            page_size=4,
            max_rounds=1,
            compute_dtype=jnp.float32,
            max_queued_tokens=16,
            max_seqs_in_prefill=2,
            max_prefill_size=8 * jax.device_count(),
            max_tokens_per_round=2 * jax.device_count(),
        )
        engine = InferenceEngine.from_model_with_config(model, None, config)
        params = dataclasses.replace(SeqDecodingParams.default(), max_num_tokens=jnp.asarray(prompt_length + 4))
        request = Request([2, 7, 9, 3][:prompt_length], 0, params, n_generations=1)
        reference = engine.generate([request])
        next_request = Request([6, 1], 1, dataclasses.replace(params, max_num_tokens=jnp.asarray(6)), n_generations=1)
        next_reference = engine.generate([next_request])
        cloned = engine.generate([dataclasses.replace(request, n_generations=2), next_request])
        assert len(reference.tokens[0]) == 4
        assert cloned.tokens == reference.tokens * 2 + next_reference.tokens
        np.testing.assert_allclose(
            cloned.logprobs, reference.logprobs * 2 + next_reference.logprobs, rtol=1e-4, atol=1e-4
        )
