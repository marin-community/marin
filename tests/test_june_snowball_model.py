# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.grug.sharding import compact_grug_mesh
from levanter.layers.attention import AttentionMask
from levanter.models.snowball import SnowballConfig

from experiments.june_tpu_67b_a2b.moe.model import RMSNorm
from experiments.june_tpu_67b_a2b.moe.rl_model import JuneSnowballConfig, split_june_pipeline_model


def test_bf16_rms_norm_scoring_matches_differentiated_forward():
    # This size exercises reduction fusion that differs when AD retains intermediates.
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        inputs = jax.random.normal(jax.random.PRNGKey(10), (2, 2048, 2560)).astype(jnp.bfloat16)
        weight = (jax.random.normal(jax.random.PRNGKey(11), (2560,)) * 0.3 + 1).astype(jnp.bfloat16)
        norm = RMSNorm(weight, eps=1e-6)

        def forward(module, values):
            return module(values)

        def objective(module, values):
            output = forward(module, values)
            return jnp.mean(jnp.sin(output.astype(jnp.float32))), output

        scored = jax.jit(forward)(norm, inputs)
        (_, trained), gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))(norm, inputs)
        jax.block_until_ready(gradients)
        np.testing.assert_array_equal(scored, trained)


def _config():
    return JuneSnowballConfig(
        vocab_size=24,
        hidden_dim=12,
        intermediate_dim=16,
        shared_expert_intermediate_dim=12,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        initializer_std=0.02,
        attention_implementation="reference",
        moe_implementation="ring",
    )


def _inputs():
    Batch, Position = hax.Axis("batch", jax.device_count()), hax.Axis("position", 6)
    tokens = hax.named(jnp.broadcast_to(jnp.array([0, 0, 2, 3, 4, 5]), (Batch.size, 6)), (Batch, Position))
    segments = hax.named(jnp.broadcast_to(jnp.array([0, 0, 1, 1, 1, 1]), (Batch.size, 6)), (Batch, Position))
    positions = hax.named(jnp.broadcast_to(jnp.array([0, 0, 0, 2, 4, 6]), (Batch.size, 6)), (Batch, Position))
    return tokens, segments, positions


def _arrays(tree):
    return [np.asarray(x) for x in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


@pytest.mark.timeout(180)
def test_june_hf_mapping_roundtrip_preserves_weights_and_forward():
    config = _config()
    decoded = JuneSnowballConfig.from_hf_config(config.to_hf_config(config.vocab_size))
    assert isinstance(decoded, JuneSnowballConfig)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = config.build(hax.Axis("vocab", config.vocab_size), key=jax.random.PRNGKey(5))
        tensors = model.to_state_dict()
        # A non-square Q projection catches transpose errors independently of roundtrip symmetry.
        np.testing.assert_array_equal(
            tensors["model.layers.0.self_attn.q_proj.weight"], model.transformer.blocks[0].attn.w_q.T
        )
        loaded = config.build(model.Vocab, key=jax.random.PRNGKey(9)).from_state_dict(tensors)
        for source, restored in zip(_arrays(model), _arrays(loaded), strict=True):
            np.testing.assert_array_equal(source, restored)
        serving_config = SnowballConfig.from_hf_config(config.to_hf_config(config.vocab_size))
        serving = serving_config.build(model.Vocab, key=jax.random.PRNGKey(3)).from_state_dict(tensors)
        tokens, _, _ = _inputs()
        # Unpadded inputs exercise shared HF semantics without the serving snapshot's mask limitation.
        tokens = tokens["position", hax.ds(2, 4)]
        actual = hax.named_jit(lambda m, t: m.activations(t))(loaded, tokens)
        reference = hax.named_jit(lambda m, t: m.activations(t))(serving, tokens)
        np.testing.assert_allclose(actual.array, reference.array, atol=1e-5, rtol=1e-5)


@pytest.mark.timeout(180)
def test_june_adapter_preserves_mask_positions_and_transformer_gradient():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = _config().build(hax.Axis("vocab", 24), key=jax.random.PRNGKey(7))
        tokens, segments, positions = _inputs()
        mask = AttentionMask.causal().with_segment_ids(segments)

        def wrapped(m):
            return m.activations(tokens, mask, pos_ids=positions).array[:, 2:].sum()

        def direct(m):
            hidden, _ = m.transformer(
                tokens.array, GrugAttentionMask.causal().with_segment_ids(segments.array), position_ids=positions.array
            )
            return hidden[:, 2:].sum()

        value, gradient = eqx.filter_jit(eqx.filter_value_and_grad(wrapped))(model)
        expected, expected_gradient = eqx.filter_jit(eqx.filter_value_and_grad(direct))(model)
        np.testing.assert_array_equal(value, expected)
        for actual, reference in zip(_arrays(gradient), _arrays(expected_gradient), strict=True):
            np.testing.assert_array_equal(actual, reference)
        padded = hax.named_jit(lambda m, t, a, p: m.activations(t, a, pos_ids=p))(model, tokens, mask, positions)
        unpadded_tokens = tokens["position", hax.ds(2, 4)]
        unpadded_positions = positions["position", hax.ds(2, 4)]
        unpadded = hax.named_jit(lambda m, t, p: m.activations(t, pos_ids=p))(model, unpadded_tokens, unpadded_positions)
        np.testing.assert_allclose(padded.array[:, 2:], unpadded.array, atol=1e-5, rtol=1e-5)


@pytest.mark.timeout(180)
@pytest.mark.parametrize("num_layers", [2, 5])
def test_june_pipeline_stage_split_preserves_masked_forward(num_layers):
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = dataclasses.replace(_config(), num_layers=num_layers).build(
            hax.Axis("vocab", 24), key=jax.random.PRNGKey(21)
        )
        tokens, segments, positions = _inputs()
        stages = split_june_pipeline_model(model, 2)

        def pipeline(parts):
            hidden = parts[0].embed(tokens.array)
            counts = []
            for stage in parts:
                hidden, stats = stage.run_blocks_with_stats(hidden, segments.array, positions.array)
                counts.append(stats)
            hidden = parts[-1].finish(hidden)
            return hidden @ parts[-1].get_lm_head(), counts

        actual, counts = eqx.filter_jit(pipeline)(stages)
        for stage, stats in zip(stages, counts, strict=True):
            # Padding is routed too and consumes capacity; count the complete input width.
            assert int(stats["routing_assignments"]) == tokens.array.size * 2 * len(stage.blocks)
            assert int(stats["routing_sender_drops"]) == 0
            assert int(stats["routing_receiver_drops"]) == 0
            assert int(stats["routing_max_layer_drops"]) == 0
            assert int(stats["routing_drop_layer"]) == -1
        expected = hax.named_jit(
            lambda m: m.activations(
                tokens,
                AttentionMask.causal().with_segment_ids(segments),
                pos_ids=positions,
            ).array
            @ m.get_lm_head().array
        )(model)
        np.testing.assert_array_equal(actual, expected)
        # Every parameter is owned by exactly one stage, including both boundary modules.
        assert sum(array.size for array in _arrays(stages)) == sum(array.size for array in _arrays(model))


@pytest.mark.timeout(180)
def test_june_adapter_mask_axis_order_does_not_change_attention():
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = _config().build(hax.Axis("vocab", 24), key=jax.random.PRNGKey(23))
        tokens, segments, positions = _inputs()
        Batch = hax.Axis("batch", 2 * jax.device_count())
        tokens, segments, positions = (hax.concatenate(Batch, [value, value]) for value in (tokens, segments, positions))
        canonical = AttentionMask.causal().with_segment_ids(segments)
        transposed = AttentionMask.causal().with_segment_ids(segments.rearrange(("position", "batch")))
        score = hax.named_jit(lambda mask: model.activations(tokens, mask, pos_ids=positions))
        expected = score(canonical)
        actual = score(transposed)
        np.testing.assert_array_equal(actual.array, expected.array)


@pytest.mark.parametrize("num_stages", [1, 2, 3])
def test_june_stage_hf_mapping_loads_only_owned_tensors(num_stages):
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = dataclasses.replace(_config(), num_layers=5).build(hax.Axis("vocab", 24), key=jax.random.PRNGKey(31))
        full = model.to_state_dict(prefix="policy")
        stage_tensors = {}
        for stage in split_june_pipeline_model(model, num_stages):
            owned = stage.to_state_dict(prefix="policy")
            template = eqx.filter_eval_shape(lambda stage: stage, stage)
            expected_shapes = eqx.filter_eval_shape(lambda stage: stage.to_state_dict(prefix="policy"), template)
            assert owned.keys() == expected_shapes.keys()
            assert not (stage_tensors.keys() & owned.keys())
            stage_tensors.update(owned)
            # The loader only receives this stage's key subset and abstract shape leaves.
            restored = template.from_state_dict(owned, prefix="policy")
            for expected, actual in zip(_arrays(stage), _arrays(restored), strict=True):
                np.testing.assert_array_equal(expected, actual)
        assert stage_tensors.keys() == full.keys()
        for key, value in full.items():
            np.testing.assert_array_equal(stage_tensors[key], value)
