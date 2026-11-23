# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage for June Snowball learner route replay."""

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from haliax import Axis
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel

from experiments.june_tpu_67b_a2b.moe.rl_model import JuneSnowballConfig, JuneSnowballLMHeadModel


def test_june_snowball_route_replay_reaches_weighted_learner_update():
    config = JuneSnowballConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=24,
        shared_expert_intermediate_dim=16,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        max_seq_len=8,
        sliding_window=4,
        attention_implementation="reference",
        moe_implementation="scatter",
    )
    batch = Axis("batch", 1)
    position = Axis("position", 6)
    ids = hax.named(jnp.asarray([[2, 5, 7, 11, 0, 0]], dtype=jnp.int32), (batch, position))
    segments = hax.named(jnp.asarray([[0, 0, 0, 0, -1, -1]], dtype=jnp.int32), (batch, position))
    weights = hax.named(jnp.asarray([[1.0, 0.5, 0.25, 0.75, 0.0, 0.0]], dtype=jnp.float32), (batch, position))
    mask = AttentionMask.causal().with_segment_ids(segments)
    example = LmExample(tokens=ids, loss_weight=weights, attn_mask=mask)

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = JuneSnowballLMHeadModel.init(Axis("vocab", config.vocab_size), config, key=jax.random.key(0))
        with hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"), "position": None, "vocab": "model"}):
            captures = model.capture_route_captures(ids, mask)
            assert not bool(np.asarray(captures.route_captures[0].token_valid)[-1])

            changed_router = model.transformer.blocks[0].mlp.router.at[:, 0].add(10.0)
            changed_block = eqx.tree_at(lambda block: block.mlp.router, model.transformer.blocks[0], changed_router)
            changed = eqx.tree_at(lambda current: current.transformer.blocks[0], model, changed_block)
            changed_captures = changed.capture_route_captures(ids, mask)
            assert not np.array_equal(
                np.asarray(captures.route_captures[0].selected_experts),
                np.asarray(changed_captures.route_captures[0].selected_experts),
            )

            grug_mask = GrugAttentionMask(is_causal=True, segment_ids=(segments.array, segments.array))
            replay_loss = changed.transformer.next_token_loss(
                ids.array,
                weights.array,
                mask=grug_mask,
                route_captures=captures.route_captures,
            )
            ordinary_loss = changed.transformer.next_token_loss(ids.array, weights.array, mask=grug_mask)

            def loss(current):
                return current.transformer.next_token_loss(
                    ids.array,
                    weights.array,
                    mask=grug_mask,
                    route_captures=captures.route_captures,
                )

            loss_value, gradients = eqx.filter_value_and_grad(loss)(changed)
            optimizer = optax.sgd(0.01)
            opt_state = optimizer.init(eqx.filter(changed, eqx.is_array))
            updates, _ = optimizer.update(eqx.filter(gradients, eqx.is_array), opt_state)
            updated = eqx.apply_updates(changed, updates)

            mismatch_segments = hax.named(segments.array.at[0, -1].set(0), segments.axes)
            mismatch = dataclasses.replace(example, attn_mask=AttentionMask.causal().with_segment_ids(mismatch_segments))
            with pytest.raises(ValueError, match="packed segment IDs"):
                changed.activations(mismatch.tokens, mismatch.attn_mask, route_captures=captures)

    assert bool(jnp.isfinite(replay_loss))
    assert bool(jnp.isfinite(ordinary_loss))
    assert float(replay_loss) != float(ordinary_loss)
    assert bool(jnp.isfinite(loss_value))
    assert float(jnp.max(jnp.abs(gradients.transformer.blocks[0].mlp.router))) > 0.0
    assert float(jnp.max(jnp.abs(gradients.transformer.blocks[0].mlp.expert_mlp.w_gate))) > 0.0
    assert (
        float(jnp.max(jnp.abs(updated.transformer.blocks[0].mlp.router - changed.transformer.blocks[0].mlp.router)))
        > 0.0
    )


def test_june_public_loss_replays_cached_snowball_routes_under_jit():
    config = JuneSnowballConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=24,
        shared_expert_intermediate_dim=16,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        max_seq_len=8,
        sliding_window=4,
        attention_implementation="reference",
        moe_implementation="scatter",
    )
    serving_config_values = {field.name: getattr(config, field.name) for field in dataclasses.fields(SnowballConfig)}
    serving_config = dataclasses.replace(
        SnowballConfig(**serving_config_values), inference_attention_implementation="reference"
    )
    batch = Axis("batch", 1)
    position = Axis("position", 4)
    serving_ids = hax.named(jnp.asarray([2, 5, 7, 11], dtype=jnp.int32), "position")
    serving_positions = hax.named(jnp.arange(4, dtype=jnp.int32), "position")
    serving_info = PageBatchInfo(
        slot_ids=hax.named(jnp.asarray([0], dtype=jnp.int32), "seq"),
        page_indices=hax.named(jnp.asarray([[0, 1]], dtype=jnp.int32), ("seq", "page")),
        seq_lens=hax.named(jnp.asarray([4], dtype=jnp.int32), "seq"),
        cu_q_lens=hax.named(jnp.asarray([0, 4], dtype=jnp.int32), "seq"),
        num_seqs=jnp.asarray(1, dtype=jnp.int32),
        new_token_dests=hax.named(jnp.arange(4, dtype=jnp.int32), "position"),
        page_size=2,
    )
    teacher_ids = hax.named(serving_ids.array[None, :], (batch, position))
    loss_weight = hax.named(jnp.asarray([[1.0, 0.25, 0.75, 0.5]], dtype=jnp.float32), (batch, position))
    example = LmExample(tokens=teacher_ids, loss_weight=loss_weight, attn_mask=AttentionMask.causal())

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        serving_model = SnowballLMHeadModel.init(
            Axis("vocab", serving_config.vocab_size), serving_config, key=jax.random.key(0)
        )
        cache = serving_model.initial_cache(PageTableSpec(num_pages=4, page_size=2, max_seqs=1), dtype=jnp.float32)
        _, _, decode_capture = serving_model.decode_with_route_captures(
            serving_ids, cache, serving_info, serving_positions
        )
        replay_capture = decode_capture.as_teacher_forced(
            teacher_ids,
            jnp.arange(4, dtype=jnp.int32)[None, :],
            jnp.ones((4,), dtype=jnp.bool_),
            batch_info=serving_info,
        )
        later_layer_bad_routes = list(replay_capture.route_captures)
        valid = later_layer_bad_routes[1].token_valid
        later_layer_bad_routes[1] = later_layer_bad_routes[1]._replace(
            token_valid=valid.at[0].set(False, out_sharding=valid.sharding)
        )
        mismatched_validity_capture = replay_capture._replace(route_captures=tuple(later_layer_bad_routes))
        june_model = JuneSnowballLMHeadModel.init(
            Axis("vocab", config.vocab_size), config, key=jax.random.key(1)
        ).from_state_dict(serving_model.to_state_dict())
        changed_router = june_model.transformer.blocks[0].mlp.router.at[:, 0].add(100.0)
        changed_block = eqx.tree_at(lambda block: block.mlp.router, june_model.transformer.blocks[0], changed_router)
        changed = eqx.tree_at(lambda current: current.transformer.blocks[0], june_model, changed_block)
        changed_capture = changed.capture_route_captures(teacher_ids, AttentionMask.causal())
        assert not np.array_equal(
            np.asarray(replay_capture.route_captures[0].selected_experts),
            np.asarray(changed_capture.route_captures[0].selected_experts),
        )

        with hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"), "position": None, "vocab": "model"}):

            def loss(current, token_array, capture):
                dynamic_example = dataclasses.replace(example, tokens=hax.named(token_array, teacher_ids.axes))
                return current.compute_next_token_loss(dynamic_example, route_captures=capture)

            def ordinary_loss(current, token_array):
                return current.transformer.next_token_loss(
                    token_array,
                    loss_weight.array,
                    mask=GrugAttentionMask(is_causal=True),
                )

            jitted_loss = eqx.filter_jit(eqx.filter_value_and_grad(loss))
            replay_loss, gradients = jitted_loss(changed, teacher_ids.array, replay_capture)
            replay_loss = jax.block_until_ready(replay_loss)
            ordinary = jax.block_until_ready(ordinary_loss(changed, teacher_ids.array))

            wrong_ids = teacher_ids.array.at[0, 1].set(13)
            with pytest.raises(Exception, match="route captures"):
                bad_loss, _ = jitted_loss(changed, wrong_ids, replay_capture)
                jax.block_until_ready(bad_loss)
            with pytest.raises(Exception, match="token_valid"):
                bad_validity_loss, _ = jitted_loss(changed, teacher_ids.array, mismatched_validity_capture)
                jax.block_until_ready(bad_validity_loss)

            replay_none = changed.compute_next_token_loss(example, reduction=None, route_captures=replay_capture)
            replay_sum = changed.compute_next_token_loss(example, reduction=hax.sum, route_captures=replay_capture)
            replay_mean = changed.compute_next_token_loss(example, reduction=hax.mean, route_captures=replay_capture)
            logits = changed(teacher_ids, route_captures=replay_capture)
            logits_host = np.asarray(jax.device_get(logits.array))
            targets_host = np.roll(np.asarray(teacher_ids.array), -1, axis=1)
            log_probs = logits_host - jax.nn.logsumexp(logits_host, axis=-1, keepdims=True)
            per_token = -np.asarray(log_probs)[np.arange(1)[:, None], np.arange(4)[None, :], targets_host]
            effective_weight = np.asarray(loss_weight.array).copy()
            effective_weight[:, -1] = 0.0
            manual_none = per_token * effective_weight
            manual_sum = float(np.sum(manual_none))
            manual_mean = manual_sum / float(np.sum(effective_weight))
            replay_none_array = np.asarray(replay_none.array if hasattr(replay_none, "array") else replay_none)

            optimizer = optax.sgd(0.01)
            opt_state = optimizer.init(eqx.filter(changed, eqx.is_array))
            updates, _ = optimizer.update(eqx.filter(gradients, eqx.is_array), opt_state)
            updated = eqx.apply_updates(changed, updates)

    assert bool(jnp.isfinite(replay_loss))
    assert bool(jnp.isfinite(ordinary))
    gradient_leaves = [leaf for leaf in jax.tree.leaves(gradients) if eqx.is_array(leaf)]
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in gradient_leaves)
    assert float(jnp.max(jnp.abs(gradients.transformer.blocks[0].mlp.router))) > 0.0
    assert float(jnp.max(jnp.abs(gradients.transformer.blocks[0].mlp.expert_mlp.w_gate))) > 0.0
    assert (
        float(jnp.max(jnp.abs(updated.transformer.blocks[0].mlp.router - changed.transformer.blocks[0].mlp.router)))
        > 0.0
    )
    assert np.allclose(replay_none_array, manual_none, rtol=1e-5, atol=1e-6)
    assert np.isclose(float(replay_sum), manual_sum, rtol=1e-5, atol=1e-6)
    assert np.isclose(float(replay_mean), manual_mean, rtol=1e-5, atol=1e-6)
