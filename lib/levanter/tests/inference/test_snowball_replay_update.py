# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Focused CPU tests for the public Snowball replay learner-update diagnostic."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import haliax as hax
from haliax import Axis
from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.snowball_replay_update import run_snowball_replay_update
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel


def _example() -> tuple[SnowballLMHeadModel, LmExample]:
    config = SnowballConfig(
        vocab_size=32,
        hidden_dim=32,
        intermediate_dim=48,
        shared_expert_intermediate_dim=32,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=16,
        max_seq_len=8,
        sliding_window=4,
        attention_implementation="reference",
        inference_attention_implementation="reference",
        moe_implementation="scatter",
    )
    position = Axis("position", 8)
    tokens = hax.named(
        jnp.asarray([2, 5, 7, 11, 13, 17, 19, 23], dtype=jnp.int32),
        position,
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", config.vocab_size), config, key=jax.random.key(4))
    return model, LmExample.causal(tokens)


def test_replay_update_keeps_captured_routes_and_updates_live_parameters():
    model, example = _example()
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        with hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"), "position": None, "vocab": "model"}):
            sharded_example = hax.auto_sharded(example)
            result = run_snowball_replay_update(model, sharded_example)
            perturbed_router = result.model.transformer.blocks[0].mlp.router.at[:, 0].add(10.0)
            perturbed_block = eqx.tree_at(
                lambda block: block.mlp.router, result.model.transformer.blocks[0], perturbed_router
            )
            perturbed_model = eqx.tree_at(lambda current: current.transformer.blocks[0], result.model, perturbed_block)
            fresh = perturbed_model(sharded_example.tokens, sharded_example.attn_mask)
            replay = perturbed_model(
                sharded_example.tokens, sharded_example.attn_mask, route_captures=result.route_captures
            )

    assert result.used_jit is False
    assert not np.array_equal(np.asarray(result.pre_update_logits.array), np.asarray(result.post_update_logits.array))
    assert not np.array_equal(np.asarray(fresh.array), np.asarray(replay.array))
    assert float(jnp.max(jnp.abs(result.gradients.transformer.blocks[0].mlp.router))) > 0.0
    assert float(jnp.max(jnp.abs(result.gradients.transformer.blocks[0].mlp.expert_mlp.w_gate))) > 0.0
    assert np.all(np.isfinite(np.asarray(result.post_update_replay_logits.array)))


def test_replay_update_jit_matches_eager_loss_and_rejects_mismatched_capture():
    model, example = _example()
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        with hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"), "position": None, "vocab": "model"}):
            sharded_example = hax.auto_sharded(example)
            eager = run_snowball_replay_update(model, sharded_example, use_jit=False)
            jitted = run_snowball_replay_update(model, sharded_example, use_jit=True)
            changed_tokens = hax.named(sharded_example.tokens.array.at[0].set(3), sharded_example.tokens.axes)
            changed = LmExample.causal(changed_tokens)
            with pytest.raises(ValueError, match="different token IDs"):
                model.compute_next_token_loss(changed, route_captures=eager.route_captures)

            segment_ids = hax.named(jnp.zeros_like(sharded_example.tokens.array), sharded_example.tokens.axes)
            segmented = LmExample(
                tokens=sharded_example.tokens,
                loss_weight=sharded_example.loss_weight,
                attn_mask=AttentionMask.causal().with_segment_ids(segment_ids),
            )
            with pytest.raises(ValueError, match="different packed segment IDs"):
                model.compute_next_token_loss(segmented, route_captures=eager.route_captures)

    np.testing.assert_allclose(
        np.asarray(eager.pre_update_loss), np.asarray(jitted.pre_update_loss), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(eager.post_update_loss), np.asarray(jitted.post_update_loss), rtol=1e-5, atol=1e-5
    )
    assert jitted.used_jit is True
