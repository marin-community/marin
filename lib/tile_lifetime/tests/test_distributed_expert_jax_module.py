# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import gzip
import json
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
    audit_shard_mapped_handler_module_stablehlo,
    build_natural_router_relation,
    build_relation_return_metadata,
    compare_numerical_arrays,
    evaluate_decomposed_training_reference,
    evaluate_natural_jax_training,
    lower_handler_module_stablehlo,
    plan_distributed_expert_jax_module,
    prepare_input_adjoint_weights,
    rebind_pair_weight_cotangent,
)
from tile_lifetime.expert_parallel_training import derive_expert_parallel_training_plan
from tile_lifetime.jax_routed_reverse_ffi import segmented_input_adjoint_cuda_compile_plan
from tile_lifetime.relation import RelationPlanError, build_fixed_capacity_relation_plan
from tile_lifetime.xla_routed_shared_map_training_ffi import plan_routed_shared_map_training_typed_ffi
from tile_lifetime.xla_segmented_input_adjoint_ffi import (
    audit_segmented_input_adjoint_resources,
    evaluate_segmented_input_adjoint_plan,
    generate_cuda_segmented_input_adjoint_ffi,
    plan_segmented_input_adjoint_ffi,
)

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


def _values(*, hidden: int = 32, intermediate: int = 32):
    keys = jax.random.split(jax.random.key(41), 7)
    source = jax.random.normal(keys[0], (8, hidden), dtype=jnp.bfloat16) / 5
    router = jax.random.normal(keys[1], (hidden, 8), dtype=jnp.bfloat16) / 5
    gate = jax.random.normal(keys[2], (8, hidden, intermediate), dtype=jnp.bfloat16) / 5
    up = jax.random.normal(keys[3], (8, hidden, intermediate), dtype=jnp.bfloat16) / 5
    down = jax.random.normal(keys[4], (8, intermediate, hidden), dtype=jnp.bfloat16) / 5
    cotangent = jax.random.normal(keys[5], (8, hidden), dtype=jnp.bfloat16) / 5
    return source, router, gate, up, down, cotangent


def _module_plan(
    router_scale: float = 1.0,
    target_prefix: str = "shuttle.distributed_expert_test",
    hidden: int = 32,
    intermediate: int = 32,
):
    source, router, *_ = _values(hidden=hidden, intermediate=intermediate)
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
        config=DistributedExpertJaxModuleConfig(
            source_items_per_rank=2,
            hidden=hidden,
            intermediate=intermediate,
        ),
        input_adjoint_template=templates.recovered_input_adjoint,
        weight_gradient_templates=templates.weight_gradients,
        source_fold_template=templates.source_fold,
        target_prefix=target_prefix,
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


def test_nonsquare_training_reports_abs_relative_and_bf16_ulp_error() -> None:
    hidden = 48
    intermediate = 32
    source, router, gate, up, down, cotangent = _values(hidden=hidden, intermediate=intermediate)
    plan = _module_plan(hidden=hidden, intermediate=intermediate)
    natural = evaluate_natural_jax_training(
        source,
        router,
        gate,
        up,
        down,
        cotangent,
        route_slots=2,
    )
    decomposed = evaluate_decomposed_training_reference(
        plan,
        _training_plan(),
        source,
        router,
        gate,
        up,
        down,
        cotangent,
    )

    metrics = {
        name: compare_numerical_arrays(getattr(natural, name), getattr(decomposed, name))
        for name in natural.__dataclass_fields__
    }

    assert max(metric.maximum_absolute for metric in metrics.values()) < 1.1e-3
    assert max(metric.mean_absolute for metric in metrics.values()) < 2.3e-4
    assert all(np.isfinite(metric.maximum_relative) for metric in metrics.values())


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


def test_sealed_four_rank_shard_mapped_hlo_integrates_handlers_and_collectives() -> None:
    artifact = _ROOT / "benchmarks/artifacts/distributed_expert_jax_module_cpu_v0"
    stablehlo = (artifact / "integrated-stablehlo.mlir").read_text()
    summary = json.loads((artifact / "summary.json").read_text())
    plan = _module_plan(target_prefix="shuttle.distributed_expert_cpu")

    audit = audit_shard_mapped_handler_module_stablehlo(plan, stablehlo)

    assert audit == summary["integrated_graph"]
    assert stablehlo.count("stablehlo.custom_call") == 5
    assert audit["all_to_all_count"] == 3
    assert audit["all_reduce_count"] == 1


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
    returned = build_relation_return_metadata(plan)
    observed_edges = []
    for rank in plan.composition.ranks:
        metadata = rank.metadata
        valid = metadata.route_valid
        observed_edges.extend(metadata.route_edge_identity[valid].tolist())
        assert np.array_equal(
            metadata.route_edge_identity[valid],
            metadata.route_source_item[valid] * plan.relation.route_slots + metadata.route_source_slot[valid],
        )
        padded_rows = metadata.route_padded_rows[valid]
        returned_edges = (
            returned.source_item[metadata.rank, padded_rows] * plan.relation.route_slots
            + returned.route_slot[metadata.rank, padded_rows]
        )
        np.testing.assert_array_equal(returned_edges, metadata.route_edge_identity[valid])
        np.testing.assert_array_equal(returned.valid[metadata.rank, padded_rows], True)

    assert np.array_equal(np.sort(observed_edges), np.arange(plan.relation.route_count))
    assert tuple(boundary.name for boundary in plan.collectives) == (
        "output_adjoint_transport",
        "input_adjoint_return_transport",
        "route_weight_return_transport",
    )
    assert all(boundary.mechanism == "jax.lax.all_to_all" for boundary in plan.collectives)
    assert all(boundary.semantics == "payload_permutation_only" for boundary in plan.collectives)
    assert plan.collectives[1].payload_shape_per_rank == (4, 4, 32)
    assert plan.ad_owner.startswith("JAX VJP")


def test_input_adjoint_weight_abi_transposes_forward_layouts() -> None:
    down = jnp.arange(2 * 3 * 4, dtype=jnp.bfloat16).reshape(2, 3, 4)
    gate = jnp.arange(2 * 4 * 3, dtype=jnp.bfloat16).reshape(2, 4, 3)
    up = gate + jnp.bfloat16(64)
    down_rhs, gate_up_rhs = prepare_input_adjoint_weights(down, gate, up)

    np.testing.assert_array_equal(down_rhs, jnp.swapaxes(down, -1, -2))
    np.testing.assert_array_equal(
        gate_up_rhs,
        jnp.concatenate((jnp.swapaxes(gate, -1, -2), jnp.swapaxes(up, -1, -2)), axis=1),
    )


def test_concatenated_pair_weight_cotangent_rebinds_to_natural_storage() -> None:
    concatenated = jnp.arange(2 * 4 * 6, dtype=jnp.bfloat16).reshape(2, 4, 6)

    gate, up = rebind_pair_weight_cotangent(concatenated, intermediate_features=3)

    np.testing.assert_array_equal(gate, concatenated[..., :3])
    np.testing.assert_array_equal(up, concatenated[..., 3:])


def test_segmented_input_adjoint_matches_nonsquare_reference_with_empty_segment() -> None:
    groups = 4
    capacity = 3
    hidden = 48
    intermediate = 32
    plan = plan_segmented_input_adjoint_ffi(
        _templates().recovered_input_adjoint,
        segment_count=groups,
        capacity=capacity,
        input_features=hidden,
        intermediate_features=intermediate,
    )
    keys = jax.random.split(jax.random.key(53), 4)
    padded_cotangent = jax.random.normal(keys[0], (groups, capacity, hidden), dtype=jnp.bfloat16) / 5
    saved_pair = jax.random.normal(keys[1], (groups, capacity, 2 * intermediate), dtype=jnp.bfloat16) / 5
    down_rhs = jax.random.normal(keys[2], (groups, hidden, intermediate), dtype=jnp.bfloat16) / 5
    gate_up_rhs = jax.random.normal(keys[3], (groups, 2 * intermediate, hidden), dtype=jnp.bfloat16) / 5
    validity = jnp.asarray(
        (
            (True, True, True),
            (False, False, False),
            (True, False, False),
            (True, True, False),
        )
    )

    pair_cotangent, input_cotangent = evaluate_segmented_input_adjoint_plan(
        plan,
        np.asarray(padded_cotangent),
        np.asarray(saved_pair),
        np.asarray(validity),
        np.asarray(down_rhs),
        np.asarray(gate_up_rhs),
    )
    projection = (padded_cotangent @ down_rhs).astype(jnp.bfloat16)
    gate, up = jnp.split(saved_pair, 2, axis=-1)
    _, pullback = jax.vjp(
        lambda gate_value, up_value: (jax.nn.silu(gate_value) * up_value).astype(jnp.bfloat16), gate, up
    )
    gate_cotangent, up_cotangent = pullback(projection)
    reference_pair = jnp.concatenate((gate_cotangent, up_cotangent), axis=-1) * validity[..., None]
    reference_input = (reference_pair.astype(jnp.bfloat16) @ gate_up_rhs).astype(jnp.bfloat16)

    np.testing.assert_array_equal(pair_cotangent, np.asarray(reference_pair, dtype=np.float32))
    np.testing.assert_array_equal(input_cotangent, np.asarray(reference_input, dtype=np.float32))
    np.testing.assert_array_equal(pair_cotangent[1], 0.0)
    np.testing.assert_array_equal(input_cotangent[1], 0.0)


def test_primary_segmented_input_adjoint_avoids_dense_expert_axis() -> None:
    plan = plan_segmented_input_adjoint_ffi(
        _templates().recovered_input_adjoint,
        segment_count=96,
        capacity=256,
        input_features=7168,
        intermediate_features=3072,
    )
    audit = audit_segmented_input_adjoint_resources(plan)
    generated = generate_cuda_segmented_input_adjoint_ffi(plan, target="shuttle.segmented_input_adjoint.test")

    assert audit.map_items == 150_994_944
    assert audit.rejected_dense_map_items == 14_495_514_624
    assert audit.rejected_dense_map_items > np.iinfo(np.int32).max
    assert audit.total_generated_bytes == 805_306_368
    assert audit.rejected_dense_total_intermediate_bytes == 106_803_757_056
    assert audit.dense_contract_work_ratio == 96
    assert "std::uint64_t kMapItems" in generated.source
    assert "cublasGemmStridedBatchedEx" in generated.source
    assert "kPhysicalMapFeatures" not in generated.source

    seams = plan.fusion_seams
    assert seams.consumed_edge_tile.axes == ("segment", "row_within_segment", "feature")
    assert seams.consumed_edge_tile.feature_extent == 7168
    assert seams.pair_state_tile.feature_extent == 6144
    assert seams.produced_edge_tile.feature_extent == 7168
    assert seams.maximum_logical_edges == 96 * 256
    assert seams.standalone_ffi_materializes_full_buffers
    assert not seams.fused_candidate_requires_full_materialization
    assert seams.buffer_elision.produced_edge_payload
    assert seams.buffer_elision.exact_edge_identity_required
    assert seams.buffer_elision.all_pair_consumers_must_share_tile_lifetime
    assert seams.readiness.produced_edge_ready_after == "second_contract"


def test_segmented_input_adjoint_compile_plan_links_only_generic_cuda_dependencies(tmp_path: Path) -> None:
    plan = plan_segmented_input_adjoint_ffi(
        _templates().recovered_input_adjoint,
        segment_count=4,
        capacity=3,
        input_features=48,
        intermediate_features=32,
    )
    generated = generate_cuda_segmented_input_adjoint_ffi(plan, target="shuttle.segmented_input_adjoint.test")
    toolkit = tmp_path / "cuda"
    nvcc = toolkit / "bin" / "nvcc"
    library_directory = toolkit / "lib64"
    include_directory = tmp_path / "jaxlib-include"
    nvcc.parent.mkdir(parents=True)
    library_directory.mkdir()
    include_directory.mkdir()
    nvcc.touch()
    (library_directory / "libcudart.so").touch()
    (library_directory / "libcublas.so").touch()

    compile_plan = segmented_input_adjoint_cuda_compile_plan(
        generated,
        directory=tmp_path / "build",
        nvcc=nvcc,
        architecture="sm_100a",
        jaxlib_include=include_directory,
    )

    assert generated.handler_symbol == "shuttle_segmented_input_adjoint_test"
    assert "-arch=sm_100a" in compile_plan.argv
    assert str(library_directory / "libcudart.so") in compile_plan.argv
    assert str(library_directory / "libcublas.so") in compile_plan.argv
    assert all("torch" not in argument.lower() for argument in compile_plan.argv)
    assert all("pybind" not in argument.lower() for argument in compile_plan.argv)
    assert all("mok" not in argument.lower() for argument in compile_plan.argv)


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
