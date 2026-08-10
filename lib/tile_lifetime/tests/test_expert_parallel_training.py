# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tile_lifetime import DType, ExpertParallelConfig, NumericalPolicy, compile_stablehlo_expert_parallel_region
from tile_lifetime.expert_parallel_training import (
    ExpertParallelTrainingStageKind,
    derive_expert_parallel_training_plan,
)
from tile_lifetime.moe_reference import MoEDebugConfig, moe_region
from tile_lifetime.moe_training_reference import (
    PRIMARY_MOK_BF16_TRAINING_CONFIG,
    MoETrainingBoundaryConfig,
    export_debug_moe_training_boundary_text,
    moe_training_boundary,
)
from tile_lifetime.tensor_program import ScalarExpressionKind, scalar_binary, scalar_input, serialize_scalar_expression

_PRIMARY_FIXTURE = (
    Path(__file__).parent / "fixtures" / "stablehlo" / "moe_primary_t2048_h7168_i3072_e384_k6_v1_14_1.mlir.bc.b64"
)


def _primary_forward_plan():
    return compile_stablehlo_expert_parallel_region(
        base64.b64decode(_PRIMARY_FIXTURE.read_text()),
        input_names=(
            "x",
            "router_weight",
            "shared_gate_weight",
            "shared_up_weight",
            "shared_down_weight",
            "routed_gate_weight",
            "routed_up_weight",
            "routed_down_weight",
        ),
        gemm_accumulation_dtype=DType.FP32,
        config=ExpertParallelConfig(expert_parallel_size=4),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )


def test_natural_jax_training_boundary_matches_direct_whole_program_gradient() -> None:
    config = MoEDebugConfig(tokens=4, hidden=4, intermediate=6, experts=3, top_k=2)
    boundary_config = MoETrainingBoundaryConfig(
        tokens=config.tokens,
        hidden=config.hidden,
        intermediate=config.intermediate,
        experts=config.experts,
        top_k=config.top_k,
        expert_parallel_size=1,
    )
    key = jax.random.key(17)
    shapes = (
        (config.tokens, config.hidden),
        (config.hidden, config.experts),
        (config.intermediate, config.hidden),
        (config.intermediate, config.hidden),
        (config.hidden, config.intermediate),
        (config.experts, config.intermediate, config.hidden),
        (config.experts, config.intermediate, config.hidden),
        (config.experts, config.hidden, config.intermediate),
    )
    keys = jax.random.split(key, len(shapes) + 1)
    values = tuple(
        jax.random.normal(array_key, shape, dtype=jnp.bfloat16) / 4
        for array_key, shape in zip(keys[:-1], shapes, strict=True)
    )
    output_cotangent = jax.random.normal(keys[-1], (config.tokens, config.hidden), dtype=jnp.bfloat16)

    actual = moe_training_boundary(boundary_config)(*values, output_cotangent)
    forward = moe_region(config)

    def loss(*inputs):
        output, _, _ = forward(*inputs)
        return jnp.sum(output.astype(jnp.float32) * output_cotangent.astype(jnp.float32))

    expected_gradients = jax.grad(loss, argnums=tuple(range(len(values))))(*values)
    expected_output, expected_indices, expected_route_weights = forward(*values)

    np.testing.assert_array_equal(actual[0], expected_output)
    np.testing.assert_array_equal(actual[1], expected_indices)
    np.testing.assert_array_equal(actual[2], expected_route_weights)
    np.testing.assert_allclose(actual[3], expected_gradients[0], rtol=0, atol=0)
    np.testing.assert_allclose(actual[5], expected_gradients[1], rtol=0, atol=0)
    for generated, expected in zip(actual[6:], expected_gradients[2:], strict=True):
        np.testing.assert_allclose(generated, expected, rtol=0, atol=0)
    assert actual[4].shape == (config.tokens, config.top_k)


def test_exported_training_boundary_contains_only_ordinary_stablehlo_algebra() -> None:
    config = MoEDebugConfig(tokens=4, hidden=4, intermediate=6, experts=3, top_k=2)
    stablehlo = export_debug_moe_training_boundary_text(config)

    assert "stablehlo.custom_call" not in stablehlo
    assert "stablehlo.composite" in stablehlo
    assert stablehlo.count("stablehlo.dot_general") >= 9
    assert stablehlo.count("stablehlo.scatter") >= 3


def test_primary_training_shape_matches_pinned_mok_bf16_boundary() -> None:
    config = PRIMARY_MOK_BF16_TRAINING_CONFIG

    assert (config.tokens, config.hidden, config.intermediate) == (2_048, 7_168, 3_072)
    assert (config.experts, config.local_experts, config.top_k, config.expert_parallel_size) == (384, 96, 6, 4)


def test_training_plan_derives_payload_only_forward_and_reverse_structure() -> None:
    forward = _primary_forward_plan()
    plan = derive_expert_parallel_training_plan(forward)

    assert plan.stage(ExpertParallelTrainingStageKind.DOWN_INPUT_ADJOINT).primitive == "SegmentedContract"
    assert plan.stage(ExpertParallelTrainingStageKind.DOWN_WEIGHT_ADJOINT).primitive.startswith("SegmentedContract")
    assert plan.stage(ExpertParallelTrainingStageKind.PAIR_MAP_ADJOINT).primitive == "generated Map VJP"
    assert plan.stage(ExpertParallelTrainingStageKind.SOURCE_INPUT_ADJOINT_FOLD).primitive == (
        "deterministic source-slot Fold"
    )
    assert plan.stage(ExpertParallelTrainingStageKind.ROUTER_VJP).primitive == "JAX-owned Map/Fold/Contract reverse"
    assert all("combine" not in transport for transport in plan.payload_transports)
    assert all("mok" not in boundary.lower() for boundary in plan.external_implementation_boundaries)


def test_pair_map_mutation_regenerates_vjp_without_changing_training_stage_family() -> None:
    forward = _primary_forward_plan()
    baseline = derive_expert_parallel_training_plan(forward)
    mutated_semantics = replace(
        forward.map_fold_semantics,
        pair_map=scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_input("left"),
            scalar_input("right"),
        ),
    )
    mutated = derive_expert_parallel_training_plan(replace(forward, map_fold_semantics=mutated_semantics))

    assert serialize_scalar_expression(baseline.pair_map_left_vjp) != serialize_scalar_expression(
        mutated.pair_map_left_vjp
    )
    assert tuple(stage.kind for stage in baseline.stages) == tuple(stage.kind for stage in mutated.stages)
