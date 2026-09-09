# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from levanter.grpo import GrpoConfig, KlGradient, grpo_loss
from levanter.grpo_model import GrpoExample
from levanter.grpo_pipeline import (
    pipeline_batch,
    packed_pipeline_batch,
    PipelineBatch,
    RoutingDropPolicy,
    validate_routing_drops,
    _pipeline_value_and_grad,
    _scorer_repeatability_details,
    pipeline_loss,
)
from levanter.grug.sharding import compact_grug_mesh


class Stage(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def embed(self, tokens):
        return jax.nn.one_hot(tokens, self.weight.shape[0])

    def run_blocks(self, hidden, attention_mask, position_ids):
        return jnp.tanh(hidden @ self.weight + self.bias + position_ids[..., None] * 0.01) * attention_mask[..., None]

    def finish(self, hidden):
        return hidden

    def get_lm_head(self):
        return self.weight.T


@pytest.mark.parametrize("compute_dtype", ["float32", "bfloat16"])
def test_pipeline_objective_preserves_response_alignment_masks_and_full_batch_weights(compute_dtype):
    rows = 2 * jax.device_count()
    Batch, Response = hax.Axis("batch", rows), hax.Axis("response", 2)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        stages = tuple(
            Stage(jax.random.normal(jax.random.PRNGKey(i), (8, 8)) * 0.1, jnp.arange(8) * 0.001) for i in range(2)
        )
        selected, frozen = zip(*(eqx.partition(stage, eqx.is_inexact_array) for stage in stages), strict=True)
        tokens = jnp.tile(jnp.array([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]]), (rows // 2, 1))
        mask = (tokens != 0).astype(jnp.int32)
        positions = jnp.maximum(jnp.cumsum(mask, -1) - 1, 0)
        advantages = jnp.tile(jnp.array([[1.0, 0.0], [-1.0, -1.0]]), (rows // 2, 1))
        weights = jnp.tile(jnp.array([[1.0, 0.0], [0.5, 0.5]]), (rows // 2, 1)) / rows
        batch = PipelineBatch(
            tokens,
            mask,
            mask,
            positions,
            advantages,
            weights,
            weights,
            jnp.full((rows, 2), -2.0),
            jnp.full((rows, 2), -2.1),
        )
        config = GrpoConfig(0.2, 0.2, 0.001, KlGradient.DETACHED)
        policy = jmp.get_policy(f"params=float32,compute={compute_dtype},output=float32")

        def actual(params):
            return pipeline_loss(params, frozen, batch, config=config, mp_policy=policy)[0]

        def expected(params):
            combined = eqx.combine(policy.cast_to_compute(params), frozen)
            hidden = combined[0].embed(tokens)
            for stage in combined:
                hidden = stage.run_blocks(hidden, mask, positions)
            logits = hidden @ combined[-1].get_lm_head()
            logprobs = jax.nn.log_softmax(logits[:, -3:-1], axis=-1)
            selected_probs = jnp.take_along_axis(logprobs, tokens[:, -2:, None], axis=-1)[..., 0]
            inputs = (selected_probs, batch.old_logprobs, batch.reference_logprobs, advantages, weights, weights)
            return grpo_loss(
                *(hax.named(value, (Batch, Response)) for value in inputs), config=config, accumulation_steps=1
            )[0]

        value, grads = eqx.filter_jit(eqx.filter_value_and_grad(actual))(selected)
        reference, reference_grads = eqx.filter_jit(eqx.filter_value_and_grad(expected))(selected)
        np.testing.assert_allclose(value, reference, atol=1e-7, rtol=1e-6)
        for actual_grad, expected_grad in zip(jax.tree.leaves(grads), jax.tree.leaves(reference_grads), strict=True):
            np.testing.assert_allclose(actual_grad, expected_grad, atol=1e-7, rtol=1e-5)


def test_hoisted_bf16_weights_preserve_per_microbatch_fp32_gradient_accumulation():
    rows = 2 * jax.device_count()
    policy = jmp.get_policy("params=float32,compute=bfloat16,output=float32")
    config = GrpoConfig(0.2, 0.2, 0.0, KlGradient.DETACHED)
    stage = Stage(jax.random.normal(jax.random.PRNGKey(9), (8, 8)) * 0.2, jnp.arange(8) * 0.001)
    selected = (eqx.tree_at(lambda value: value.bias, stage, None),)
    frozen = (eqx.tree_at(lambda value: value.weight, stage, None),)
    compute = policy.cast_to_compute(selected)
    actual_gradients, expected_gradients, bf16_gradients = [], [], []
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        for index in range(3):
            tokens = (jnp.tile(jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]]), (rows // 2, 1)) + index) % 8
            mask = jnp.ones_like(tokens)
            positions = jnp.broadcast_to(jnp.arange(4), tokens.shape)
            advantages = jnp.tile(jnp.array([[0.37 + index, -0.71], [1.13, 0.19 - index]]), (rows // 2, 1))
            weights = jnp.ones_like(advantages) / (6 * rows)
            batch = PipelineBatch(
                tokens,
                mask,
                mask,
                positions,
                advantages,
                weights,
                weights,
                jnp.full((rows, 2), -2.0),
                jnp.zeros((rows, 2)),
            )

            def reference(masters):
                return pipeline_loss(masters, frozen, batch, config=config, mp_policy=policy)[0]

            expected_value, expected = jax.value_and_grad(reference)(selected)
            (actual_value, _), actual = _pipeline_value_and_grad(
                compute, frozen, batch, config=config, mp_policy=policy, mark=lambda value: value
            )
            np.testing.assert_array_equal(actual_value, expected_value)
            for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
                assert result.dtype == jnp.float32
                np.testing.assert_array_equal(result, wanted)
            (bf16_value, _), bf16_gradient = _pipeline_value_and_grad(
                compute,
                frozen,
                batch,
                config=config,
                mp_policy=policy,
                mark=lambda value: value,
                gradient_accum_dtype="bfloat16",
            )
            np.testing.assert_array_equal(bf16_value, expected_value)
            for result, wanted in zip(jax.tree.leaves(bf16_gradient), jax.tree.leaves(expected), strict=True):
                assert result.dtype == jnp.bfloat16
                np.testing.assert_array_equal(result, wanted.astype(jnp.bfloat16))
            bf16_gradients.append(bf16_gradient)
            actual_gradients.append(actual)
            expected_gradients.append(expected)
        actual_total = jax.tree.map(lambda *values: sum(values), *actual_gradients)
        expected_total = jax.tree.map(lambda *values: sum(values), *expected_gradients)
        for result, wanted in zip(jax.tree.leaves(actual_total), jax.tree.leaves(expected_total), strict=True):
            np.testing.assert_array_equal(result, wanted)
        # This example detects moving the FP32 cast after the temporal reduction.
        bf16_total = jax.tree.map(
            lambda *values: sum(value.astype(jnp.bfloat16) for value in values), *actual_gradients
        )
        actual_bf16_total = jax.tree.map(lambda *values: sum(values).astype(jnp.float32), *bf16_gradients)
        for result, wanted in zip(jax.tree.leaves(actual_bf16_total), jax.tree.leaves(bf16_total), strict=True):
            assert result.dtype == jnp.float32
            np.testing.assert_array_equal(result, wanted.astype(jnp.float32))
        assert any(
            np.any(np.asarray(correct) != np.asarray(rounded.astype(jnp.float32)))
            for correct, rounded in zip(jax.tree.leaves(actual_total), jax.tree.leaves(bf16_total), strict=True)
        )


def test_scorer_repeatability_reports_routing_separately_from_logprobs():
    first = jnp.array([[-2.0, -3.0, -4.0]])
    repeated = jnp.array([[-2.0, -2.0, -2.0]])
    weights = jnp.array([[0.5, 0.5, 0.0]])
    routing = {"routing_sender_drops": jnp.array(0)}
    repeated_routing = {"routing_sender_drops": jnp.array(395)}
    details = _scorer_repeatability_details(first, repeated, weights, routing, repeated_routing)
    assert details["active_logprob_max_abs"] == 1.0
    assert details["inactive_logprob_max_abs"] == 2.0
    assert details["active_logprob_changed"] == 1
    assert details["inactive_logprob_changed"] == 1
    assert details["routing"]["routing_sender_drops"] == {"first": 0, "repeated": 395}
    counters_only = _scorer_repeatability_details(first, first, weights, routing, repeated_routing)
    assert counters_only["active_logprob_max_abs"] == 0.0
    assert counters_only["inactive_logprob_max_abs"] == 0.0


@pytest.mark.parametrize("counter", ["routing_sender_drops", "routing_receiver_drops"])
def test_routing_drop_experiment_is_explicit_and_reports_unchanged_counts(counter, caplog):
    metrics = {"routing_assignments": 48, "routing_sender_drops": 0, "routing_receiver_drops": 0}
    metrics[counter] = 40
    with pytest.raises(FloatingPointError, match="capacity dropped assignments"):
        validate_routing_drops(metrics, context="test scorer")
    validate_routing_drops(metrics, context="test scorer", policy=RoutingDropPolicy.REPORT)
    assert "test scorer" in caplog.text
    assert "padding included" in caplog.text
    assert str(metrics) in caplog.text
    assert metrics[counter] == 40
    assert metrics["routing_assignments"] == 48


def test_packing_preserves_response_objectives_positions_and_segment_boundaries():
    Batch, Position, Response = hax.Axis("batch", 4), hax.Axis("position", 8), hax.Axis("response", 4)
    tokens = jnp.array(
        [
            [0, 0, 1, 2, 3, 4, 0, 0],
            [0, 5, 6, 7, 8, 0, 0, 0],
            [0, 0, 0, 9, 10, 11, 12, 0],
            [0, 13, 14, 15, 16, 17, 0, 0],
        ]
    )
    attention = (tokens != 0).astype(jnp.int32)
    response_mask = attention[:, 4:].astype(jnp.float32)
    policy_weights = response_mask / (response_mask.sum(axis=1, keepdims=True) * 4)
    positions = jnp.maximum(attention.cumsum(axis=1) - 1, 0)
    # Nonzero offsets ensure packing preserves source positions instead of regenerating them.
    positions = positions + jnp.arange(4)[:, None] * attention

    def seq(x):
        return hax.named(x, (Batch, Position))

    def resp(x):
        return hax.named(x, (Batch, Response))

    example = GrpoExample(
        seq(tokens),
        seq(attention),
        seq(positions),
        resp(response_mask),
        resp(response_mask * jnp.array([1.0, -0.6, 0.3, -1.0])[:, None]),
        resp(policy_weights),
        resp(policy_weights * 0.7),
        resp(jnp.full((4, 4), -3.2)),
        resp(jnp.full((4, 4), -3.1)),
    )
    unpacked = pipeline_batch(example, microbatches=2)
    packed = packed_pipeline_batch(
        example, sequence_length=10, rows_per_microbatch=1, pad_token_id=0, minimum_microbatches=3
    )
    assert packed.tokens.shape == (3, 1, 10)
    assert np.all(np.asarray(packed.segment_ids[-1]) == -1)
    for name in ("advantages", "policy_weights", "kl_weights", "old_logprobs", "reference_logprobs"):
        before = np.asarray(getattr(unpacked, name))[np.asarray(unpacked.policy_weights) > 0]
        after = np.asarray(getattr(packed, name))[np.asarray(packed.policy_weights) > 0]
        np.testing.assert_array_equal(after, before)
    for document in range(4):
        locations = np.asarray(packed.segment_ids) == document
        np.testing.assert_array_equal(
            np.asarray(packed.tokens)[locations], np.asarray(tokens[document])[attention[document] > 0]
        )
        np.testing.assert_array_equal(
            np.asarray(packed.position_ids)[locations], np.asarray(positions[document])[attention[document] > 0]
        )
    segments = np.asarray(packed.segment_ids)
    boundary = (segments[..., 1:] != segments[..., :-1]) | (segments[..., 1:] < 0)
    assert np.all(np.asarray(packed.policy_weights)[boundary] == 0)
    assert np.all(np.asarray(packed.kl_weights)[boundary] == 0)
    with pytest.raises(ValueError, match="exceeds"):
        packed_pipeline_batch(example, sequence_length=4, rows_per_microbatch=1, pad_token_id=0)
