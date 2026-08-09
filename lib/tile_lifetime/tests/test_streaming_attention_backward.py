# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime import (
    DType,
    StreamingAttentionBackwardDomainTraversal,
    StreamingAttentionBackwardFoldOrder,
    StreamingAttentionBackwardProvenance,
    StreamingTileSchedule,
    apply_causal_score_mask,
    apply_tanh_softcap,
    build_attention_tensor_program,
    derive_streaming_attention,
    derive_streaming_attention_backward,
    derive_streaming_attention_backward_tile_schedule,
    estimate_streaming_attention_backward_work,
    execute_streaming_attention_backward,
    execute_streaming_attention_with_state,
    scaled_score_map,
)
from tile_lifetime.tensor_program import ScalarExpressionKind, scalar_binary, scalar_constant


def _program(*, causal: bool, softcap: float | None = None, output_scale: float = 1.0):
    score_map = scaled_score_map(0.7)
    if softcap is not None:
        score_map = apply_tanh_softcap(score_map, softcap)
    if causal:
        score_map = apply_causal_score_mask(score_map)
    source = build_attention_tensor_program(
        batch_size=1,
        query_length=5,
        key_length=5,
        query_heads=4,
        key_value_heads=2,
        key_dimension=3,
        value_dimension=3,
        score_map=score_map,
        input_dtype=DType.FP32,
    )
    forward = derive_streaming_attention(
        source,
        schedule=StreamingTileSchedule(query_tile_size=2, key_value_tile_size=3, pipeline_depth=2),
    )
    if output_scale != 1.0:
        finalize = replace(
            forward.finalize,
            expression=scalar_binary(
                ScalarExpressionKind.MULTIPLY,
                scalar_constant(output_scale),
                forward.finalize.expression,
            ),
        )
        forward = replace(forward, finalize=finalize)
    return forward


def _jax_reference(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    causal: bool,
    softcap: float | None,
    output_scale: float,
) -> jax.Array:
    expanded_key = jnp.repeat(key, 2, axis=2)
    expanded_value = jnp.repeat(value, 2, axis=2)
    score = jnp.einsum("bqhd,bkhd->bhqk", query, expanded_key) * 0.7
    if softcap is not None:
        score = softcap * jnp.tanh(score / softcap)
    if causal:
        query_position = jnp.arange(query.shape[1])
        key_position = jnp.arange(key.shape[1])
        score = jnp.where(key_position[None, None, None, :] <= query_position[None, None, :, None], score, -jnp.inf)
    probability = jax.nn.softmax(score, axis=-1)
    return output_scale * jnp.einsum("bhqk,bkhv->bqhv", probability, expanded_value)


@pytest.mark.parametrize(
    ("causal", "softcap", "output_scale"),
    ((True, None, 1.0), (False, 1.6, 0.75)),
)
def test_streaming_attention_backward_matches_jax_autodiff(
    causal: bool,
    softcap: float | None,
    output_scale: float,
) -> None:
    forward = _program(causal=causal, softcap=softcap, output_scale=output_scale)
    backward = derive_streaming_attention_backward(forward)
    rng = np.random.default_rng(91)
    query = rng.normal(size=(1, 5, 4, 3)).astype(np.float32)
    key = rng.normal(size=(1, 5, 2, 3)).astype(np.float32)
    value = rng.normal(size=(1, 5, 2, 3)).astype(np.float32)
    output_cotangent = rng.normal(size=(1, 5, 4, 3)).astype(np.float32)
    inputs = {"query": query, "key": key, "value": value}
    if causal:
        inputs.update(
            {
                "query.position": np.arange(5, dtype=np.int32),
                "key.position": np.arange(5, dtype=np.int32),
            }
        )
    execution = execute_streaming_attention_with_state(forward, inputs)
    actual = execute_streaming_attention_backward(backward, inputs, execution, output_cotangent)

    def loss(q, k, v):
        output = _jax_reference(
            q,
            k,
            v,
            causal=causal,
            softcap=softcap,
            output_scale=output_scale,
        )
        return jnp.sum(output * output_cotangent)

    expected_query, expected_key, expected_value = jax.grad(loss, argnums=(0, 1, 2))(query, key, value)

    np.testing.assert_allclose(actual.query_cotangent, expected_query, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual.key_cotangent, expected_key, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual.value_cotangent, expected_value, rtol=2e-5, atol=2e-5)


def test_score_map_mutation_reuses_one_backward_stage_family() -> None:
    baseline = derive_streaming_attention_backward(_program(causal=True))
    mutated = derive_streaming_attention_backward(_program(causal=False, softcap=1.3))

    assert mutated.stages == baseline.stages
    assert mutated.materialized_values == baseline.materialized_values
    assert mutated.score_map_vjp.expression != baseline.score_map_vjp.expression
    assert tuple(value.name for value in mutated.materialized_values) == (
        "cotangent.query",
        "cotangent.key",
        "cotangent.value",
    )


def test_grouped_key_value_schedule_is_derived_from_contract_index_relation() -> None:
    backward = derive_streaming_attention_backward(_program(causal=True))
    recovered = replace(backward, provenance=StreamingAttentionBackwardProvenance.JAX_VJP_HLO_RECOVERY)
    schedule = derive_streaming_attention_backward_tile_schedule(
        recovered,
        query_tile_size=1,
        key_value_tile_size=1,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    work = estimate_streaming_attention_backward_work(recovered, schedule)

    assert schedule.query_heads_per_key_value_tile == 2
    assert schedule.key_value_fold_order is StreamingAttentionBackwardFoldOrder.QUERY_ROW_MAJOR_MAPPED_HEAD_MINOR_TREE
    assert work.logical_query_key_tile_pairs == 60
    assert work.fully_restricted_tile_pairs == 40
    assert work.query_gradient_contract_invocations == 180
    assert work.full_domain_query_gradient_contract_invocations == 300
    assert work.key_value_gradient_contract_invocations == 120
    assert work.scalar_head_key_value_contract_invocations == 240
    assert work.full_domain_scalar_head_key_value_contract_invocations == 400
    assert work.key_value_contract_invocation_reduction == 2.0
    assert work.key_value_contract_invocation_reduction_from_full_scalar == 10 / 3
    assert work.packed_query_rows == 2
    assert work.peak_score_tile_elements == 2
    assert work.peak_query_tile_elements == 6
    assert work.key_value_gradient_accumulator_elements == 6


def test_score_map_mutation_changes_domain_work_without_changing_grouped_schedule() -> None:
    causal = derive_streaming_attention_backward(_program(causal=True))
    softcap = derive_streaming_attention_backward(_program(causal=False, softcap=1.3))
    causal_schedule = derive_streaming_attention_backward_tile_schedule(
        causal,
        query_tile_size=1,
        key_value_tile_size=1,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    softcap_schedule = derive_streaming_attention_backward_tile_schedule(
        softcap,
        query_tile_size=1,
        key_value_tile_size=1,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.FULL,
    )

    causal_work = estimate_streaming_attention_backward_work(causal, causal_schedule)
    softcap_work = estimate_streaming_attention_backward_work(softcap, softcap_schedule)
    assert causal_schedule.query_heads_per_key_value_tile == softcap_schedule.query_heads_per_key_value_tile
    assert causal_work.logical_query_key_tile_pairs == 60
    assert softcap_work.logical_query_key_tile_pairs == 100
    assert softcap_work.fully_restricted_tile_pairs == 0
    assert causal.provenance is StreamingAttentionBackwardProvenance.REFERENCE_SYMBOLIC_VJP
    assert softcap.provenance is StreamingAttentionBackwardProvenance.REFERENCE_SYMBOLIC_VJP
