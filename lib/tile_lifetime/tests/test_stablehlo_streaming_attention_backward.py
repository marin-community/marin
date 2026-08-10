# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime import (
    StreamingAttentionBackwardMaximumVJP,
    StreamingAttentionBackwardProvenance,
    StreamingTileSchedule,
    execute_streaming_attention_backward,
    execute_streaming_attention_with_state,
    recover_stablehlo_streaming_attention_backward,
)
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.stablehlo_import import CompareAttributes, import_stablehlo
from tile_lifetime.stablehlo_streaming_attention_backward import StableHLOStreamingAttentionBackwardError
from tile_lifetime.streaming_attention_backward import eliminate_normalized_exp_maximum_vjp
from tile_lifetime.streaming_attention_backward_reference import (
    STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    StreamingAttentionBackwardDebugConfig,
    causal_gqa_attention_training,
    causal_gqa_attention_vjp,
    export_debug_streaming_attention_backward,
    export_debug_streaming_attention_training,
)

FIXTURE = Path(__file__).parent / "fixtures" / "stablehlo" / "causal_gqa_attention_vjp_v1_17_0.mlir.bc.b64"
SCHEDULE = StreamingTileSchedule(query_tile_size=2, key_value_tile_size=2, pipeline_depth=2)


def _fixture_graph():
    return import_stablehlo(
        base64.b64decode(FIXTURE.read_text()),
        input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    )


def test_frozen_jax_vjp_recovers_visible_generic_reverse_algebra() -> None:
    graph = _fixture_graph()
    recovered = recover_stablehlo_streaming_attention_backward(graph, schedule=SCHEDULE)

    assert recovered.program.provenance is StreamingAttentionBackwardProvenance.JAX_VJP_HLO_RECOVERY
    assert recovered.program.maximum_vjp is StreamingAttentionBackwardMaximumVJP.JAX_EQUAL_SPLIT
    assert recovered.score_scale == 0.5
    assert len(recovered.contract_operation_ids) == 5
    assert len(recovered.normalized_exponential_fold_operation_ids) == 2
    assert len(recovered.broadcast_vjp_fold_operation_ids) == 2
    assert recovered.maximum_vjp_tie_fold_operation_id not in recovered.normalized_exponential_fold_operation_ids
    assert len(recovered.source_operation_ids) == len(graph.operations)
    assert tuple(stage.value for stage in recovered.program.stages) == (
        "load_query_and_state",
        "load_key_value",
        "recompute_qk",
        "recompute_probability",
        "dv_contract",
        "dp_contract",
        "score_map_vjp",
        "dq_contract",
        "dk_contract",
    )


@pytest.mark.parametrize("scale", (0.5, 0.375))
def test_live_jax_vjp_recovery_executes_with_bf16_gradient_parity(scale: float) -> None:
    config = StreamingAttentionBackwardDebugConfig(scale=scale)
    reverse = causal_gqa_attention_vjp(config)
    graph = import_stablehlo(
        export_debug_streaming_attention_backward(config),
        input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    )
    recovered = recover_stablehlo_streaming_attention_backward(graph, schedule=SCHEDULE)
    rng = np.random.default_rng(71)
    arguments = (
        jnp.asarray(rng.normal(size=(1, 4, 4, 4)), dtype=jnp.bfloat16),
        jnp.asarray(rng.normal(size=(1, 4, 2, 4)), dtype=jnp.bfloat16),
        jnp.asarray(rng.normal(size=(1, 4, 2, 4)), dtype=jnp.bfloat16),
        jnp.asarray(rng.normal(size=(1, 4, 4, 4)), dtype=jnp.bfloat16),
    )
    expected = reverse(*arguments)
    inputs = {
        "query": np.asarray(arguments[0], dtype=np.float32),
        "key": np.asarray(arguments[1], dtype=np.float32),
        "value": np.asarray(arguments[2], dtype=np.float32),
        "query.position": np.arange(4, dtype=np.int32),
        "key.position": np.arange(4, dtype=np.int32),
    }
    forward = execute_streaming_attention_with_state(recovered.program.forward, inputs)
    actual = execute_streaming_attention_backward(
        recovered.program,
        inputs,
        forward,
        np.asarray(arguments[3], dtype=np.float32),
    )

    assert recovered.score_scale == scale
    for generated, reference in zip(
        (actual.query_cotangent, actual.key_cotangent, actual.value_cotangent),
        expected,
        strict=True,
    ):
        generated_bf16 = np.asarray(jnp.asarray(generated, dtype=jnp.bfloat16), dtype=np.float32)
        reference_bf16 = np.asarray(reference, dtype=np.float32)
        error = np.abs(generated_bf16 - reference_bf16)
        assert float(error.max()) == 0.0
        assert float(error.mean()) == 0.0


def test_natural_training_boundary_recovers_forward_output_by_data_dependencies() -> None:
    config = StreamingAttentionBackwardDebugConfig(scale=0.375)
    graph = import_stablehlo(
        export_debug_streaming_attention_training(config),
        input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    )
    recovered = recover_stablehlo_streaming_attention_backward(graph, schedule=SCHEDULE)
    rng = np.random.default_rng(72)
    arguments = (
        jnp.asarray(rng.normal(size=(1, 4, 4, 4)), dtype=jnp.bfloat16),
        jnp.asarray(rng.normal(size=(1, 4, 2, 4)), dtype=jnp.bfloat16),
        jnp.asarray(rng.normal(size=(1, 4, 2, 4)), dtype=jnp.bfloat16),
        jnp.asarray(rng.normal(size=(1, 4, 4, 4)), dtype=jnp.bfloat16),
    )

    expected = causal_gqa_attention_training(config)(*arguments)
    inputs = {
        "query": np.asarray(arguments[0], dtype=np.float32),
        "key": np.asarray(arguments[1], dtype=np.float32),
        "value": np.asarray(arguments[2], dtype=np.float32),
        "query.position": np.arange(4, dtype=np.int32),
        "key.position": np.arange(4, dtype=np.int32),
    }
    generated_forward = execute_streaming_attention_with_state(recovered.program.forward, inputs)
    generated_reverse = execute_streaming_attention_backward(
        recovered.program,
        inputs,
        generated_forward,
        np.asarray(arguments[3], dtype=np.float32),
    )
    generated_outputs = (
        generated_forward.output,
        generated_reverse.query_cotangent,
        generated_reverse.key_cotangent,
        generated_reverse.value_cotangent,
    )

    assert recovered.forward_output in graph.outputs
    assert recovered.query_cotangent in graph.outputs
    assert recovered.forward_output != recovered.query_cotangent
    assert len(recovered.source_operation_ids) == len(graph.operations)
    assert recovered.score_scale == 0.375
    for generated, reference in zip(generated_outputs, expected, strict=True):
        generated_bf16 = np.asarray(jnp.asarray(generated, dtype=jnp.bfloat16), dtype=np.float32)
        np.testing.assert_array_equal(generated_bf16, np.asarray(reference, dtype=np.float32))


def test_maximum_vjp_invariant_rewrite_requires_rounding_reorder_policy() -> None:
    recovered = recover_stablehlo_streaming_attention_backward(_fixture_graph(), schedule=SCHEDULE)

    with pytest.raises(ValueError, match="bitwise policy"):
        eliminate_normalized_exp_maximum_vjp(
            recovered.program,
            numerical_policy=NumericalPolicy.BITWISE_EXACT,
        )

    lowered = eliminate_normalized_exp_maximum_vjp(
        recovered.program,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    assert recovered.program.maximum_vjp is StreamingAttentionBackwardMaximumVJP.JAX_EQUAL_SPLIT
    assert lowered.maximum_vjp is StreamingAttentionBackwardMaximumVJP.NORMALIZED_EXP_INVARIANT
    assert lowered.provenance is StreamingAttentionBackwardProvenance.JAX_VJP_HLO_RECOVERY
    assert lowered.forward == recovered.program.forward
    assert lowered.score_map_vjp == recovered.program.score_map_vjp


def test_recovery_rejects_noncausal_domain_predicate_before_assigning_provenance() -> None:
    graph = _fixture_graph()
    causal = next(
        operation
        for operation in graph.operations
        if operation.kind == "compare"
        and operation.attributes == CompareAttributes(direction="LE", compare_type="SIGNED")
    )
    changed_operations = tuple(
        (
            replace(
                operation,
                attributes=CompareAttributes(direction="GE", compare_type="SIGNED"),
            )
            if operation.id == causal.id
            else operation
        )
        for operation in graph.operations
    )

    with pytest.raises(StableHLOStreamingAttentionBackwardError) as error:
        recover_stablehlo_streaming_attention_backward(
            replace(graph, operations=changed_operations),
            schedule=SCHEDULE,
        )

    assert error.value.stage == "domain_restriction"
    assert error.value.operation_ids == (causal.id,)
