# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from shuttle.ir import DType
from tile_lifetime import ExpertParallelConfig, NumericalPolicy
from tile_lifetime.cuda_expert_parallel_training_codegen import (
    expert_parallel_training_scalar_program,
    render_cuda_expert_parallel_training_include,
)
from tile_lifetime.expert_parallel_training import (
    ExpertParallelTrainingStageKind,
    derive_expert_parallel_training_plan,
)
from tile_lifetime.expert_parallel_training_runtime import (
    derive_distributed_expert_backward_abi,
    execute_distributed_expert_backward_reference,
)
from tile_lifetime.moe_reference import MoEDebugConfig, moe_region
from tile_lifetime.moe_training_reference import (
    PRIMARY_MOK_BF16_TRAINING_CONFIG,
    MoETrainingBoundaryConfig,
    export_debug_moe_training_boundary_text,
    moe_training_boundary,
)
from tile_lifetime.reference_pipeline import compile_reference_stablehlo_expert_parallel_region
from tile_lifetime.relation import build_relation_plan
from tile_lifetime.tensor_program import ScalarExpressionKind, scalar_binary, scalar_input, serialize_scalar_expression

_PRIMARY_FIXTURE = (
    Path(__file__).parent / "fixtures" / "stablehlo" / "moe_primary_t2048_h7168_i3072_e384_k6_v1_14_1.mlir.bc.b64"
)


def _primary_forward_plan():
    return compile_reference_stablehlo_expert_parallel_region(
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


def _four_rank_relation():
    expert_indices = np.asarray(
        (
            (0, 3),
            (3, 4),
            (7, 3),
            (0, 7),
            (4, 3),
            (7, 0),
        ),
        dtype=np.int32,
    )
    route_weights = np.asarray(
        (
            (0.6, 0.4),
            (0.25, 0.75),
            (0.1, 0.9),
            (0.3, 0.7),
            (0.8, 0.2),
            (0.45, 0.55),
        ),
        dtype=np.float32,
    )
    return build_relation_plan(
        expert_indices,
        route_weights,
        destination_rank_by_item=np.arange(8, dtype=np.int32) // 2,
        destination_local_item_by_item=np.arange(8, dtype=np.int32) % 2,
        padding_quantum=2,
    )


def test_generated_training_scalar_program_is_mutation_driven() -> None:
    forward = _primary_forward_plan()
    baseline_plan = derive_expert_parallel_training_plan(forward)
    baseline = expert_parallel_training_scalar_program(baseline_plan)
    mutated_forward = replace(
        forward,
        map_fold_semantics=replace(
            forward.map_fold_semantics,
            pair_map=scalar_binary(
                ScalarExpressionKind.MULTIPLY,
                scalar_input("left"),
                scalar_input("right"),
            ),
        ),
    )
    mutated = expert_parallel_training_scalar_program(derive_expert_parallel_training_plan(mutated_forward))
    rendered = render_cuda_expert_parallel_training_include(baseline_plan)

    assert baseline.fingerprint != mutated.fingerprint
    assert "generated_pair_left_vjp" in rendered
    assert "generated_pair_right_vjp" in rendered
    assert "generated_route_weight_fold_update" in rendered
    assert "swiglu" not in rendered.lower()
    assert "moe" not in rendered.lower()


def test_four_rank_reference_backward_matches_jax_vjp_and_is_deterministic() -> None:
    relation = _four_rank_relation()
    training_plan = derive_expert_parallel_training_plan(_primary_forward_plan())
    key = jax.random.key(29)
    source_key, gate_up_key, down_key, cotangent_key = jax.random.split(key, 4)
    source = jax.random.normal(source_key, (6, 4), dtype=jnp.float32) / 5
    gate_up_weight = jax.random.normal(gate_up_key, (8, 6, 4), dtype=jnp.float32) / 5
    down_weight = jax.random.normal(down_key, (8, 4, 3), dtype=jnp.float32) / 5
    output_cotangent = jax.random.normal(cotangent_key, (6, 4), dtype=jnp.float32) / 5
    expert_indices = jnp.asarray(relation.destination_item.reshape(6, 2))
    route_weights = jnp.asarray(relation.weight)

    def selected_program(source_value, route_weight_value, gate_up_value, down_value):
        selected_gate_up = gate_up_value[expert_indices]
        pair_input = jnp.einsum("sh,skih->ski", source_value, selected_gate_up)
        left, right = jnp.split(pair_input, 2, axis=-1)
        hidden = jax.nn.silu(left) * right
        selected_down = down_value[expert_indices]
        edge_output = jnp.einsum("ski,skhi->skh", hidden, selected_down)
        return jnp.sum(edge_output * route_weight_value[..., None], axis=1)

    expected_output, pullback = jax.vjp(
        selected_program,
        source,
        route_weights,
        gate_up_weight,
        down_weight,
    )
    expected = pullback(output_cotangent)
    first = execute_distributed_expert_backward_reference(
        relation,
        np.asarray(source),
        np.asarray(gate_up_weight),
        np.asarray(down_weight),
        np.asarray(output_cotangent),
        training_plan,
    )
    second = execute_distributed_expert_backward_reference(
        relation,
        np.asarray(source),
        np.asarray(gate_up_weight),
        np.asarray(down_weight),
        np.asarray(output_cotangent),
        training_plan,
    )

    np.testing.assert_allclose(first.output, expected_output, rtol=2e-5, atol=2e-6)
    for actual, target in zip(
        (
            first.input_cotangent,
            first.route_weight_cotangent,
            first.gate_up_weight_cotangent,
            first.down_weight_cotangent,
        ),
        expected,
        strict=True,
    ):
        np.testing.assert_allclose(actual, target, rtol=3e-5, atol=3e-6)
    for first_value, second_value in zip(first.__dict__.values(), second.__dict__.values(), strict=True):
        np.testing.assert_array_equal(first_value, second_value)


def test_primary_four_rank_backward_abi_is_derived_without_payload_allocation() -> None:
    config = PRIMARY_MOK_BF16_TRAINING_CONFIG
    global_sources = config.tokens * config.expert_parallel_size
    source_rows = np.arange(global_sources, dtype=np.int32)[:, None]
    route_slots = np.arange(config.top_k, dtype=np.int32)[None, :]
    expert_indices = (source_rows * 17 + route_slots * 29) % config.experts
    relation = build_relation_plan(
        expert_indices,
        np.full(expert_indices.shape, 1.0 / config.top_k, dtype=np.float32),
        destination_rank_by_item=np.arange(config.experts, dtype=np.int32) // config.local_experts,
        destination_local_item_by_item=np.arange(config.experts, dtype=np.int32) % config.local_experts,
        padding_quantum=256,
    )
    abi = derive_distributed_expert_backward_abi(
        relation,
        hidden=config.hidden,
        intermediate=config.intermediate,
    )

    assert len(abi.ranks) == 4
    assert abi.transport_semantics == "payload_permutation_only"
    assert sum(rank.valid_destination_rows.size for rank in abi.ranks) == global_sources * config.top_k
    assert all(rank.local_expert_count == config.local_experts for rank in abi.ranks)
    assert all(rank.total_buffer_bytes > 0 for rank in abi.ranks)
    assert all(
        buffer.shape[0] == config.local_experts
        for rank in abi.ranks
        for buffer in rank.buffers
        if buffer.name in {"gate_up_weight_cotangent", "down_weight_cotangent"}
    )
