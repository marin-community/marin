# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from enum import StrEnum
from pathlib import Path

import numpy as np
import pytest

from tile_lifetime import (
    AttentionScoreAxis,
    AxisIndexMap,
    ContractPrimitive,
    DType,
    FoldPrimitive,
    MapPrimitive,
    ScalarExpressionKind,
    StreamingAttentionStage,
    StreamingTileSchedule,
    TensorProgram,
    add_score_bias,
    apply_arbitrary_score_mask,
    apply_causal_score_mask,
    apply_tanh_softcap,
    build_attention_tensor_program,
    derive_streaming_attention,
    execute_streaming_attention,
    execute_tensor_program,
    scaled_score_map,
)
from tile_lifetime.h100_streaming_lowering import lower_h100_streaming_program
from tile_lifetime.tensor_program import scalar_binary, scalar_constant


class ScoreMutation(StrEnum):
    """Semantically distinct score programs exercised through one generator."""

    CAUSAL = "causal"
    BIAS_AND_MASK = "bias_and_mask"
    TANH_SOFTCAP = "tanh_softcap"


def _program_and_inputs(
    mutation: ScoreMutation,
    *,
    query_heads: int = 4,
    key_value_heads: int = 2,
    score_scale: float = 0.5,
) -> tuple[TensorProgram, dict[str, np.ndarray]]:
    batch, query_length, key_length = 2, 7, 9
    rng = np.random.default_rng(17)
    score_map = scaled_score_map(score_scale)
    inputs: dict[str, np.ndarray] = {
        "query": rng.normal(size=(batch, query_length, query_heads, 4)).astype(np.float32),
        "key": rng.normal(size=(batch, key_length, key_value_heads, 4)).astype(np.float32),
        "value": rng.normal(size=(batch, key_length, key_value_heads, 3)).astype(np.float32),
    }
    if mutation is ScoreMutation.CAUSAL:
        score_map = apply_causal_score_mask(score_map)
        inputs["query.position"] = np.arange(query_length, dtype=np.int32)
        inputs["key.position"] = np.arange(key_length, dtype=np.int32)
    else:
        if mutation is ScoreMutation.TANH_SOFTCAP:
            score_map = apply_tanh_softcap(score_map, cap=0.35)
        else:
            score_map = add_score_bias(
                score_map,
                axes=(
                    AttentionScoreAxis.HEAD,
                    AttentionScoreAxis.QUERY,
                    AttentionScoreAxis.KEY,
                ),
            )
            inputs["score.bias"] = rng.normal(size=(query_heads, query_length, key_length)).astype(np.float32) * 0.2
        score_map = apply_arbitrary_score_mask(score_map)
        mask = rng.random(size=(batch, query_heads, query_length, key_length)) > 0.45
        mask[..., 0] = True
        inputs["score.mask"] = mask

    source = build_attention_tensor_program(
        batch_size=batch,
        query_length=query_length,
        key_length=key_length,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        key_dimension=4,
        value_dimension=3,
        score_map=score_map,
        input_dtype=DType.FP32,
    )
    return source, inputs


def test_attention_source_is_only_contract_map_fold_and_derives_bounded_state() -> None:
    source, _ = _program_and_inputs(ScoreMutation.CAUSAL)
    schedule = StreamingTileSchedule(query_tile_size=3, key_value_tile_size=4, pipeline_depth=2)

    generated = derive_streaming_attention(source, schedule=schedule)

    assert [type(operation) for operation in source.operations] == [
        ContractPrimitive,
        MapPrimitive,
        FoldPrimitive,
        MapPrimitive,
        MapPrimitive,
        FoldPrimitive,
        ContractPrimitive,
        MapPrimitive,
    ]
    assert generated.state.row_max.shape == (2, 4, 7)
    assert generated.state.row_sum_exp.shape == (2, 4, 7)
    assert generated.state.weighted_value_accumulator.shape == (2, 7, 4, 3)
    assert generated.materialized_values == source.outputs
    assert generated.schedule.stages == tuple(StreamingAttentionStage)
    assert not hasattr(generated, "backend")


@pytest.mark.parametrize("mutation", tuple(ScoreMutation))
def test_score_semantic_mutations_use_same_generator_and_match_materialized_program(
    mutation: ScoreMutation,
) -> None:
    source, inputs = _program_and_inputs(mutation)
    generated = derive_streaming_attention(
        source,
        schedule=StreamingTileSchedule(query_tile_size=3, key_value_tile_size=4, pipeline_depth=2),
    )

    materialized = execute_tensor_program(source, inputs)["attention.output"]
    streamed = execute_streaming_attention(generated, inputs)
    difference = np.abs(streamed - materialized)

    assert float(np.max(difference)) < 2e-6
    assert float(np.mean(difference)) < 2e-7


def test_changed_score_map_changes_results_without_changing_streaming_structure() -> None:
    _, inputs = _program_and_inputs(ScoreMutation.BIAS_AND_MASK)
    softcap_source, softcap_inputs = _program_and_inputs(ScoreMutation.TANH_SOFTCAP)
    shared_inputs = {
        "query": inputs["query"],
        "key": inputs["key"],
        "value": inputs["value"],
        "score.mask": inputs["score.mask"],
    }
    softcap_inputs.update(shared_inputs)
    plain_score_map = apply_arbitrary_score_mask(scaled_score_map(0.5))
    plain_source = build_attention_tensor_program(
        batch_size=2,
        query_length=7,
        key_length=9,
        query_heads=4,
        key_value_heads=2,
        key_dimension=4,
        value_dimension=3,
        score_map=plain_score_map,
        input_dtype=DType.FP32,
    )
    schedule = StreamingTileSchedule(query_tile_size=3, key_value_tile_size=4, pipeline_depth=2)

    plain = derive_streaming_attention(plain_source, schedule=schedule)
    softcap = derive_streaming_attention(softcap_source, schedule=schedule)
    plain_output = execute_streaming_attention(plain, shared_inputs)
    softcap_output = execute_streaming_attention(softcap, softcap_inputs)

    assert plain.schedule == softcap.schedule
    assert plain.state == softcap.state
    assert not np.allclose(plain_output, softcap_output, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("score_scale", [0.125, 0.5, 1.25])
def test_score_scale_mutation_uses_same_generator_and_matches_materialized_program(score_scale: float) -> None:
    source, inputs = _program_and_inputs(ScoreMutation.CAUSAL, score_scale=score_scale)
    generated = derive_streaming_attention(
        source,
        schedule=StreamingTileSchedule(query_tile_size=3, key_value_tile_size=4, pipeline_depth=2),
    )

    materialized = execute_tensor_program(source, inputs)["attention.output"]
    streamed = execute_streaming_attention(generated, inputs)

    assert float(np.max(np.abs(streamed - materialized))) < 2e-6


def test_finalization_mutation_reuses_the_derived_fold_schedule() -> None:
    source, inputs = _program_and_inputs(ScoreMutation.CAUSAL)
    generated = derive_streaming_attention(
        source,
        schedule=StreamingTileSchedule(query_tile_size=3, key_value_tile_size=4, pipeline_depth=2),
    )
    output_scale = 0.375
    scaled_finalize = replace(
        generated.finalize,
        expression=scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            generated.finalize.expression,
            scalar_constant(output_scale),
        ),
    )
    scaled_source = replace(source, operations=(*source.operations[:-1], scaled_finalize))
    scaled_generated = replace(generated, source=scaled_source, finalize=scaled_finalize)

    materialized = execute_tensor_program(scaled_source, inputs)["attention.output"]
    streamed = execute_streaming_attention(scaled_generated, inputs)
    baseline = execute_streaming_attention(generated, inputs)

    assert scaled_generated.qk == generated.qk
    assert scaled_generated.score_map == generated.score_map
    assert scaled_generated.pv == generated.pv
    assert scaled_generated.state == generated.state
    assert scaled_generated.schedule == generated.schedule
    assert float(np.max(np.abs(streamed - materialized))) < 2e-6
    assert float(np.max(np.abs(streamed - baseline * output_scale))) < 2e-6


@pytest.mark.parametrize("query_heads,key_value_heads", [(8, 2), (8, 4)])
def test_grouped_head_index_map_is_explicit_and_matches_expanded_reference(
    query_heads: int,
    key_value_heads: int,
) -> None:
    source, inputs = _program_and_inputs(
        ScoreMutation.CAUSAL,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
    )
    generated = derive_streaming_attention(
        source,
        schedule=StreamingTileSchedule(query_tile_size=3, key_value_tile_size=4, pipeline_depth=2),
    )

    key_mapping = generated.qk.index_maps_for_input(1)
    value_mapping = generated.pv.index_maps_for_input(1)
    expected_mapping = AxisIndexMap(
        domain_axis=generated.qk.output.axes[1],
        operand_axis=generated.qk.inputs[1].axes[2],
        divisor=query_heads // key_value_heads,
    )
    assert key_mapping == (expected_mapping,)
    assert value_mapping[0].divisor == query_heads // key_value_heads

    ratio = query_heads // key_value_heads
    expanded_inputs = dict(inputs)
    expanded_inputs["key"] = np.repeat(inputs["key"], ratio, axis=2)
    expanded_inputs["value"] = np.repeat(inputs["value"], ratio, axis=2)
    query = inputs["query"].astype(np.float32)
    key = expanded_inputs["key"].astype(np.float32)
    value = expanded_inputs["value"].astype(np.float32)
    scores = np.einsum("bqhd,bkhd->bhqk", query, key) * 0.5
    query_positions = np.arange(query.shape[1])[:, None]
    key_positions = np.arange(key.shape[1])[None, :]
    scores = np.where(key_positions <= query_positions, scores, -np.inf)
    probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    reference = np.einsum("bhqk,bkhd->bqhd", probabilities, value)

    streamed = execute_streaming_attention(generated, inputs)
    assert float(np.max(np.abs(streamed - reference))) < 2e-6


def _h100_program(score_map):
    source = build_attention_tensor_program(
        batch_size=1,
        query_length=2048,
        key_length=2048,
        query_heads=32,
        key_value_heads=8,
        key_dimension=128,
        value_dimension=128,
        score_map=score_map,
        input_dtype=DType.BF16,
    )
    return derive_streaming_attention(
        source,
        schedule=StreamingTileSchedule(query_tile_size=128, key_value_tile_size=128, pipeline_depth=2),
    )


def test_h100_lowering_recovers_causal_scale_schedule_and_gqa_without_cuda() -> None:
    program = _h100_program(apply_causal_score_mask(scaled_score_map(0.125)))

    lowering = lower_h100_streaming_program(program)

    assert lowering.score_map.scale == 0.125
    assert lowering.score_map.causal
    assert lowering.score_map.softcap is None
    assert lowering.head_group_size == 4
    assert (lowering.schedule.tile_m, lowering.schedule.tile_n, lowering.schedule.stages) == (128, 128, 2)


def test_h100_lowering_accepts_softcap_mutation_without_named_attention_dispatch() -> None:
    score_map = apply_tanh_softcap(apply_causal_score_mask(scaled_score_map(0.0625)), cap=16.0)
    lowering = lower_h100_streaming_program(_h100_program(score_map))

    assert lowering.score_map.scale == 0.0625
    assert lowering.score_map.causal
    assert lowering.score_map.softcap == 16.0


def test_h100_lowering_changes_only_finalization_scale_for_output_map_mutation() -> None:
    program = _h100_program(apply_causal_score_mask(scaled_score_map(0.125)))
    output_scale = 0.375
    scaled_finalize = replace(
        program.finalize,
        expression=scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            program.finalize.expression,
            scalar_constant(output_scale),
        ),
    )
    scaled_program = replace(program, finalize=scaled_finalize)

    baseline = lower_h100_streaming_program(program)
    scaled = lower_h100_streaming_program(scaled_program)

    assert baseline.output_scale == 1.0
    assert scaled.output_scale == output_scale
    assert scaled.score_map == baseline.score_map
    assert scaled.schedule == baseline.schedule
    assert scaled.head_group_size == baseline.head_group_size


def test_h100_lowering_reports_tensor_mask_as_missing_auxiliary_emitter() -> None:
    program = _h100_program(apply_arbitrary_score_mask(scaled_score_map(0.125)))

    with pytest.raises(ValueError, match="auxiliary-tensor emitter"):
        lower_h100_streaming_program(program)


def test_sm90_streaming_skeleton_owns_fold_map_and_domain_semantics() -> None:
    backend = Path(__file__).parents[1] / "backends" / "h100"
    physical_sources = tuple(
        (backend / filename).read_text() for filename in ("cute_streaming_base.py", "cute_streaming_sm90.py")
    )
    forbidden_dependencies = (
        "flash_attn.cute.softmax",
        "flash_attn.cute.mask",
        "apply_score_mod_inner",
        "AttentionMask",
        "Softmax.create",
    )

    for source in physical_sources:
        assert not any(dependency in source for dependency in forbidden_dependencies)
        assert "NormalizedExpFoldState" in source
        assert "DomainRestriction" in source
        assert "apply_score_map_inner" in source
        assert "finalize(output_scale=self.output_scale" in source


def test_normalized_exp_finalize_binds_register_state_before_child_region() -> None:
    source = (Path(__file__).parents[1] / "backends" / "h100" / "cute_normalized_exp.py").read_text()

    assert "row_sum = self.row_sum" in source
    assert "row_max = self.row_max" in source
    assert "row_sum.store(utils.warp_reduce(row_sum.load()" in source
    assert "self.row_sum.store(utils.warp_reduce(self.row_sum.load()" not in source
