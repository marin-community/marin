# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime import (
    DType,
    OwnerComputeTraversal,
    SharedReverseFusionDisposition,
    StreamingAttentionBackwardDomainTraversal,
    StreamingAttentionBackwardFoldOrder,
    StreamingAttentionBackwardProvenance,
    StreamingTileSchedule,
    apply_causal_score_mask,
    apply_tanh_softcap,
    build_attention_tensor_program,
    derive_streaming_attention,
    derive_streaming_attention_backward,
    derive_streaming_attention_backward_fusion_plan,
    derive_streaming_attention_backward_tile_schedule,
    estimate_streaming_attention_backward_work,
    execute_streaming_attention_backward,
    execute_streaming_attention_with_state,
    scaled_score_map,
    verify_streaming_attention_backward_score_map_vjp,
)
from tile_lifetime.tensor_program import ScalarExpressionKind, scalar_binary, scalar_constant


def _program(
    *,
    causal: bool,
    softcap: float | None = None,
    output_scale: float = 1.0,
    query_heads: int = 4,
    key_value_heads: int = 2,
    query_length: int = 5,
    key_length: int = 5,
    head_dimension: int = 3,
):
    score_map = scaled_score_map(0.7)
    if softcap is not None:
        score_map = apply_tanh_softcap(score_map, softcap)
    if causal:
        score_map = apply_causal_score_mask(score_map)
    source = build_attention_tensor_program(
        batch_size=1,
        query_length=query_length,
        key_length=key_length,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        key_dimension=head_dimension,
        value_dimension=head_dimension,
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
    verify_streaming_attention_backward_score_map_vjp(baseline)
    verify_streaming_attention_backward_score_map_vjp(mutated)


def test_score_map_vjp_verifier_rejects_a_fixed_backend_formula() -> None:
    backward = derive_streaming_attention_backward(_program(causal=False, softcap=1.3))
    forged_vjp = replace(
        backward.score_map_vjp,
        expression=scalar_constant(0.7),
        inputs=(),
    )

    with pytest.raises(ValueError, match="not the derivative"):
        verify_streaming_attention_backward_score_map_vjp(replace(backward, score_map_vjp=forged_vjp))


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
    assert work.query_gradient_contract_invocations == 90
    assert work.scalar_head_query_gradient_contract_invocations == 180
    assert work.full_domain_query_gradient_contract_invocations == 150
    assert work.full_domain_scalar_head_query_gradient_contract_invocations == 300
    assert work.query_gradient_contract_invocation_reduction == 2.0
    assert work.query_gradient_contract_invocation_reduction_from_full_scalar == 10 / 3
    assert work.key_value_gradient_contract_invocations == 120
    assert work.scalar_head_key_value_contract_invocations == 240
    assert work.full_domain_scalar_head_key_value_contract_invocations == 400
    assert work.key_value_contract_invocation_reduction == 2.0
    assert work.key_value_contract_invocation_reduction_from_full_scalar == 10 / 3
    assert work.packed_query_rows == 2
    assert work.peak_score_tile_elements == 2
    assert work.peak_query_tile_elements == 6
    assert work.key_value_gradient_accumulator_elements == 6


@pytest.mark.parametrize(
    ("query_heads", "key_value_heads", "expected_group_size"),
    ((4, 4, 1), (4, 2, 2), (8, 2, 4)),
)
def test_query_gradient_packing_tracks_contract_head_index_relation(
    query_heads: int,
    key_value_heads: int,
    expected_group_size: int,
) -> None:
    backward = derive_streaming_attention_backward(
        _program(
            causal=False,
            query_heads=query_heads,
            key_value_heads=key_value_heads,
        )
    )
    schedule = derive_streaming_attention_backward_tile_schedule(
        backward,
        query_tile_size=1,
        key_value_tile_size=1,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.FULL,
    )
    work = estimate_streaming_attention_backward_work(backward, schedule)

    assert schedule.query_heads_per_key_value_tile == expected_group_size
    assert work.query_gradient_contract_invocation_reduction == expected_group_size
    assert work.key_value_contract_invocation_reduction == expected_group_size


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


def test_fused_reverse_relation_tracks_causal_domain_restriction() -> None:
    causal = derive_streaming_attention_backward(_program(causal=True))
    unrestricted = derive_streaming_attention_backward(_program(causal=False))
    causal_schedule = derive_streaming_attention_backward_tile_schedule(
        causal,
        query_tile_size=1,
        key_value_tile_size=1,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    unrestricted_schedule = derive_streaming_attention_backward_tile_schedule(
        unrestricted,
        query_tile_size=1,
        key_value_tile_size=1,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.FULL,
    )

    causal_fusion = derive_streaming_attention_backward_fusion_plan(
        causal,
        causal_schedule,
        local_capacity_bytes=176,
    )
    unrestricted_fusion = derive_streaming_attention_backward_fusion_plan(
        unrestricted,
        unrestricted_schedule,
        local_capacity_bytes=176,
    )

    assert causal_fusion.disposition is SharedReverseFusionDisposition.FUSED_LOCAL
    assert unrestricted_fusion.disposition is SharedReverseFusionDisposition.FUSED_LOCAL
    assert len(causal_fusion.relation.pairs) == 30
    assert len(unrestricted_fusion.relation.pairs) == 50
    assert causal_fusion.baseline_contract_invocations == 210
    assert causal_fusion.fused_contract_invocations == 150
    assert unrestricted_fusion.baseline_contract_invocations == 350
    assert unrestricted_fusion.fused_contract_invocations == 250


def test_fused_reverse_ownership_is_independent_of_score_map_vjp() -> None:
    baseline = derive_streaming_attention_backward(_program(causal=True))
    mutated = derive_streaming_attention_backward(_program(causal=True, softcap=1.3))
    schedule = derive_streaming_attention_backward_tile_schedule(
        baseline,
        query_tile_size=1,
        key_value_tile_size=1,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    mutated_schedule = derive_streaming_attention_backward_tile_schedule(
        mutated,
        query_tile_size=1,
        key_value_tile_size=1,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )

    baseline_fusion = derive_streaming_attention_backward_fusion_plan(
        baseline,
        schedule,
        local_capacity_bytes=176,
    )
    mutated_fusion = derive_streaming_attention_backward_fusion_plan(
        mutated,
        mutated_schedule,
        local_capacity_bytes=176,
    )

    assert baseline.score_map_vjp.expression != mutated.score_map_vjp.expression
    assert baseline_fusion.components == mutated_fusion.components
    assert baseline_fusion.baseline_contract_invocations == mutated_fusion.baseline_contract_invocations
    assert baseline_fusion.fused_contract_invocations == mutated_fusion.fused_contract_invocations


def test_primary_fused_reverse_rejects_large_owner_compute_frontier() -> None:
    backward = derive_streaming_attention_backward(
        _program(
            causal=True,
            query_heads=32,
            key_value_heads=8,
            query_length=2048,
            key_length=2048,
            head_dimension=128,
        )
    )
    schedule = derive_streaming_attention_backward_tile_schedule(
        backward,
        query_tile_size=32,
        key_value_tile_size=32,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    fusion = derive_streaming_attention_backward_fusion_plan(
        backward,
        schedule,
        local_capacity_bytes=227 * 1024,
    )

    assert fusion.disposition is SharedReverseFusionDisposition.REJECTED_LOCAL_CAPACITY
    assert len(fusion.components) == 8
    assert all(component.selected_traversal is OwnerComputeTraversal.SOURCE_MAJOR for component in fusion.components)
    assert fusion.required_local_bytes == 2_195_456
    assert fusion.baseline_contract_invocations == 116_480
    assert fusion.fused_contract_invocations == 83_200
    assert fusion.physical_contract_reduction == 1.4
    assert fusion.source_accumulator_elements == 16_384
    assert fusion.target_accumulator_elements == 8_192
    assert fusion.transient_edge_elements == 16_384
