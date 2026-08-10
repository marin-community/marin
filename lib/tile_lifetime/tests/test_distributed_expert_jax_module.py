# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import gzip
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime import DType, ExpertParallelConfig, NumericalPolicy, compile_stablehlo_expert_parallel_region
from tile_lifetime.distributed_expert_jax_module import (
    DistributedExpertJaxModuleConfig,
    audit_handler_module_stablehlo,
    build_natural_router_relation,
    evaluate_decomposed_training_reference,
    evaluate_natural_jax_training,
    lower_handler_module_stablehlo,
    plan_distributed_expert_jax_module,
    prepare_input_adjoint_weights,
)
from tile_lifetime.expert_parallel_training import derive_expert_parallel_training_plan
from tile_lifetime.relation import RelationPlanError, build_fixed_capacity_relation_plan
from tile_lifetime.xla_routed_shared_map_training_ffi import plan_routed_shared_map_training_typed_ffi

_ROOT = Path(__file__).parents[1]
_NATURAL_HLO = _ROOT / "benchmarks/artifacts/xla_grug_routed_combined_gpu_gb200_v0/original-gpu-pre-scheduler-hlo.txt.gz"
_PRIMARY_FIXTURE = _ROOT / "tests/fixtures/stablehlo/moe_primary_t2048_h7168_i3072_e384_k6_v1_14_1.mlir.bc.b64"


@lru_cache(maxsize=1)
def _templates():
    hlo = gzip.decompress(_NATURAL_HLO.read_bytes()).decode()
    return plan_routed_shared_map_training_typed_ffi(hlo)


@lru_cache(maxsize=1)
def _training_plan():
    forward = compile_stablehlo_expert_parallel_region(
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
    return derive_expert_parallel_training_plan(forward)


def _values():
    keys = jax.random.split(jax.random.key(41), 7)
    source = jax.random.normal(keys[0], (8, 32), dtype=jnp.bfloat16) / 5
    router = jax.random.normal(keys[1], (32, 8), dtype=jnp.bfloat16) / 5
    gate = jax.random.normal(keys[2], (8, 32, 32), dtype=jnp.bfloat16) / 5
    up = jax.random.normal(keys[3], (8, 32, 32), dtype=jnp.bfloat16) / 5
    down = jax.random.normal(keys[4], (8, 32, 32), dtype=jnp.bfloat16) / 5
    cotangent = jax.random.normal(keys[5], (8, 32), dtype=jnp.bfloat16) / 5
    return source, router, gate, up, down, cotangent


def _module_plan(router_scale: float = 1.0):
    source, router, *_ = _values()
    relation = build_natural_router_relation(
        source,
        router * router_scale,
        route_slots=2,
        destination_rank_by_item=np.arange(8, dtype=np.int32) // 2,
        destination_local_item_by_item=np.arange(8, dtype=np.int32) % 2,
        destination_capacity=5,
    )
    templates = _templates()
    return plan_distributed_expert_jax_module(
        relation,
        config=DistributedExpertJaxModuleConfig(source_items_per_rank=2, hidden=32, intermediate=32),
        input_adjoint_template=templates.recovered_input_adjoint,
        weight_gradient_templates=templates.weight_gradients,
        source_fold_template=templates.source_fold,
        target_prefix="shuttle.distributed_expert_test",
    )


def test_natural_jax_training_matches_decomposed_generated_stage_reference() -> None:
    source, router, gate, up, down, cotangent = _values()
    plan = _module_plan()
    natural = evaluate_natural_jax_training(
        source,
        router,
        gate,
        up,
        down,
        cotangent,
        route_slots=2,
    )
    first = evaluate_decomposed_training_reference(
        plan,
        _training_plan(),
        source,
        router,
        gate,
        up,
        down,
        cotangent,
    )
    second = evaluate_decomposed_training_reference(
        plan,
        _training_plan(),
        source,
        router,
        gate,
        up,
        down,
        cotangent,
    )

    maximum_errors = {}
    mean_errors = {}
    for name in natural.__dataclass_fields__:
        expected = np.asarray(getattr(natural, name), dtype=np.float32)
        actual = np.asarray(getattr(first, name), dtype=np.float32)
        repeated = np.asarray(getattr(second, name))
        error = np.abs(actual - expected)
        maximum_errors[name] = float(np.max(error))
        mean_errors[name] = float(np.mean(error))
        np.testing.assert_array_equal(actual, repeated)

    assert max(maximum_errors.values()) < 1e-3
    assert max(mean_errors.values()) < 2e-4


def test_transformed_hlo_contains_each_generic_handler_once_and_jax_router_vjp() -> None:
    plan = _module_plan()
    stablehlo = lower_handler_module_stablehlo(plan)
    occurrences = audit_handler_module_stablehlo(plan, stablehlo)

    assert len(occurrences) == 5
    assert set(occurrences.values()) == {1}
    assert stablehlo.count("stablehlo.custom_call") == 5
    assert stablehlo.count("stablehlo.dot_general") >= 2
    assert "stablehlo.gather" in stablehlo
    assert "torch" not in stablehlo.lower()


def test_fixed_relation_mutation_changes_index_plane_not_handler_programs() -> None:
    baseline = _module_plan(router_scale=1.0)
    source, router, *_ = _values()
    permutation = jnp.asarray((3, 1, 6, 0, 7, 4, 2, 5), dtype=jnp.int32)
    mutated_relation = build_natural_router_relation(
        source,
        router[:, permutation],
        route_slots=2,
        destination_rank_by_item=np.arange(8, dtype=np.int32) // 2,
        destination_local_item_by_item=np.arange(8, dtype=np.int32) % 2,
        destination_capacity=5,
    )
    templates = _templates()
    mutated = plan_distributed_expert_jax_module(
        mutated_relation,
        config=baseline.config,
        input_adjoint_template=templates.recovered_input_adjoint,
        weight_gradient_templates=templates.weight_gradients,
        source_fold_template=templates.source_fold,
        target_prefix="shuttle.distributed_expert_test",
    )

    assert baseline.composition.relation_digest != mutated.composition.relation_digest
    assert baseline.relation.destination_row_count == mutated.relation.destination_row_count == 40
    assert baseline.relation.exchange_source_item.shape == mutated.relation.exchange_source_item.shape == (32,)
    assert tuple(rank.edge_reverse.source_digest for rank in baseline.composition.ranks) == tuple(
        rank.edge_reverse.source_digest for rank in mutated.composition.ranks
    )
    assert baseline.handlers.input_adjoint.source_digest == mutated.handlers.input_adjoint.source_digest
    assert tuple(handler.source_digest for handler in baseline.handlers.weight_gradients) == tuple(
        handler.source_digest for handler in mutated.handlers.weight_gradients
    )
    assert baseline.handlers.source_fold.source_digest == mutated.handlers.source_fold.source_digest


def test_relation_return_identity_and_payload_collective_boundaries_are_explicit() -> None:
    plan = _module_plan()
    observed_edges = []
    for rank in plan.composition.ranks:
        metadata = rank.metadata
        valid = metadata.route_valid
        observed_edges.extend(metadata.route_edge_identity[valid].tolist())
        assert np.array_equal(
            metadata.route_edge_identity[valid],
            metadata.route_source_item[valid] * plan.relation.route_slots + metadata.route_source_slot[valid],
        )

    assert np.array_equal(np.sort(observed_edges), np.arange(plan.relation.route_count))
    assert tuple(boundary.name for boundary in plan.collectives) == (
        "output_adjoint_transport",
        "input_adjoint_return_transport",
        "route_weight_return_transport",
    )
    assert all(boundary.mechanism == "jax.lax.all_to_all" for boundary in plan.collectives)
    assert all(boundary.semantics == "payload_permutation_only" for boundary in plan.collectives)
    assert plan.ad_owner.startswith("JAX VJP")


def test_input_adjoint_weight_abi_transposes_forward_layouts() -> None:
    down = jnp.arange(2 * 3 * 4, dtype=jnp.bfloat16).reshape(2, 3, 4)
    gate = jnp.arange(2 * 4 * 3, dtype=jnp.bfloat16).reshape(2, 4, 3)
    up = gate + jnp.bfloat16(64)
    down_rhs, gate_up_rhs = prepare_input_adjoint_weights(down, gate, up)

    np.testing.assert_array_equal(down_rhs, jnp.swapaxes(down, -1, -2).reshape(8, 3))
    np.testing.assert_array_equal(
        gate_up_rhs,
        jnp.concatenate((jnp.swapaxes(gate, -1, -2), jnp.swapaxes(up, -1, -2)), axis=1).reshape(12, 4),
    )


def test_fixed_relation_rejects_destination_overflow() -> None:
    destination = np.zeros((4, 2), dtype=np.int32)
    with pytest.raises(RelationPlanError, match="fixed destination capacity exceeded"):
        build_fixed_capacity_relation_plan(
            destination,
            np.full(destination.shape, 0.5, dtype=np.float32),
            destination_rank_by_item=np.arange(4, dtype=np.int32) // 2,
            destination_local_item_by_item=np.arange(4, dtype=np.int32) % 2,
            destination_capacity=4,
        )
