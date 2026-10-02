# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.models.hero import HeroConfig
from levanter.models.hero_model import HeroLMHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

pytest.importorskip("fastapi")
pytest.importorskip("openai")
from levanter.inference.openai import InferenceContext, InferenceServerConfig


@jax.default_matmul_precision("highest")
def test_hero_weight_publication_resets_populated_convolution_and_kv_state():
    devices = jax.device_count()
    trainer = TrainerConfig(
        mesh=MeshConfig(
            axes={"data": -1, "model": 1, "expert": 1, "context": 1},
            dcn_axes={"replica_dcn": -1},
            shared_mapping={"batch": ["replica_dcn", "data"], "heads": "model", "kv_head": "model"},
            param_mapping={},
        ),
        use_explicit_mesh_axes=True,
    )
    config = HeroConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=12,
        shared_expert_intermediate_dim=12,
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
        sliding_window=2,
        global_every=2,
        sconv_kernel=3,
        initializer_std=0.2,
        attention_implementation="reference",
        inference_attention_implementation="reference",
    )
    service = InferenceEngineConfig(
        max_seq_len=16,
        page_size=2,
        max_pages=8 * devices,
        max_seqs=devices,
        max_seqs_in_prefill=devices,
        max_prefill_size=4 * devices,
        max_queued_tokens=devices,
        max_tokens_per_round=devices,
        max_rounds=1,
        max_stop_seqs=0,
        max_stop_tokens=0,
        compute_dtype=jnp.float32,
    )
    requests = [
        Request(
            [2, 8, 3, 7],
            index,
            dataclasses.replace(SeqDecodingParams.default(), max_num_tokens=jnp.array(8, dtype=jnp.int32)),
            1,
        )
        for index in range(devices)
    ]
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        model = HeroLMHeadModel.init(hax.Axis("vocab", config.vocab_size), config, key=jax.random.key(17))
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
        engine = InferenceEngine.from_model_with_config(model, None, service)
        context = InferenceContext(model, None, engine, InferenceServerConfig(trainer=trainer, service=service))
        before = engine.generate(requests)
        for layer in engine.gen_state.cache:
            assert np.any(np.asarray(layer.kv.kv_pages.array))
            for history in (layer.k_history, layer.attn_history, layer.mlp_history):
                assert history is not None
                assert np.any(np.asarray(history.history.array))
        populated = [np.asarray(array).copy() for array in jax.tree.leaves(engine.gen_state.cache)]

        def failed_stage(current):
            raise ValueError("candidate checkpoint incomplete")

        with pytest.raises(ValueError, match="candidate checkpoint incomplete"):
            context.reload(failed_stage, expected_version=0)
        assert context.model_version == 0
        assert not context.pause_event.is_set()
        for actual, expected in zip(jax.tree.leaves(engine.gen_state.cache), populated, strict=True):
            np.testing.assert_array_equal(actual, expected)
        unchanged = engine.generate(requests)
        assert unchanged.tokens == before.tokens
        np.testing.assert_array_equal(unchanged.logprobs, before.logprobs)

        replacement = eqx.tree_at(
            lambda m: m.transformer.output_proj,
            model,
            jax.device_put(
                np.roll(np.asarray(model.transformer.output_proj), 1, axis=-1), model.transformer.output_proj.sharding
            ),
        )
        assert context.reload(lambda current: replacement, expected_version=0) == 1
        assert context.model_version == 1
        for array in jax.tree.leaves(engine.gen_state.cache):
            assert not np.any(np.asarray(array))
        installed = engine.generate(requests)
        fresh = InferenceEngine.from_model_with_config(replacement, None, service).generate(requests)
        assert installed.tokens != before.tokens
        assert installed.tokens == fresh.tokens
        np.testing.assert_array_equal(installed.logprobs, fresh.logprobs)
