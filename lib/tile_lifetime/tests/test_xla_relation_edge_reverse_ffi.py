# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tile_lifetime.distributed_expert_backward_ffi import (
    DistributedBackwardHandlerFamily,
    compose_distributed_expert_backward_typed_ffi,
    deterministic_source_slot_fold,
    restore_source_route_payload,
    source_indexed_fold_operands,
)
from tile_lifetime.jax_relation_edge_reverse_ffi import (
    call_cuda_relation_edge_reverse_ffi,
    evaluate_relation_edge_reverse_jax,
    relation_edge_reverse_cuda_compile_plan,
)
from tile_lifetime.relation import build_relation_plan
from tile_lifetime.xla_relation_edge_reverse_ffi import (
    RelationEdgeReverseFfiPlan,
    evaluate_relation_edge_reverse,
    generate_cuda_relation_edge_reverse_ffi,
)


def _fixture():
    plan = RelationEdgeReverseFfiPlan(received_rows=4, route_slots=2, padded_rows=8, features=8)
    received = np.arange(32, dtype=np.float32).reshape(4, 8) / 8
    route_rows = np.asarray(((0, 2), (4, -1), (1, 6), (3, 5)), dtype=np.int32)
    route_weights = np.asarray(((0.25, 0.5), (0.75, 0.0), (0.5, 0.25), (1.0, 0.5)), dtype=np.float32)
    saved = np.arange(64, dtype=np.float32).reshape(8, 8) / 4
    return plan, received, route_rows, route_weights, saved


def test_relation_edge_reverse_cpu_reference_owns_map_and_ordered_fold() -> None:
    plan, received, route_rows, route_weights, saved = _fixture()
    padded, route_cotangent = evaluate_relation_edge_reverse(
        plan,
        received,
        route_rows,
        route_weights,
        saved,
    )

    expected_padded = np.zeros_like(saved)
    expected_route = np.zeros_like(route_weights)
    for received_row in range(plan.received_rows):
        for route_slot in range(plan.route_slots):
            padded_row = route_rows[received_row, route_slot]
            if padded_row < 0:
                continue
            expected_padded[padded_row] = received[received_row] * route_weights[received_row, route_slot]
            expected_route[received_row, route_slot] = np.dot(saved[padded_row], received[received_row])

    np.testing.assert_array_equal(padded, expected_padded)
    np.testing.assert_array_equal(route_cotangent, expected_route)


def test_relation_edge_reverse_typed_ffi_is_torch_free_and_body_generic() -> None:
    plan, *_ = _fixture()
    generated = generate_cuda_relation_edge_reverse_ffi(plan, target="shuttle.relation_edge_reverse")
    lowered = generated.source.lower()

    assert "xla_ffi_define_handler_symbol" in lowered
    assert "ffi::buffer<ffi::bf16, 2>" in lowered
    assert "generated_edge_cotangent_map" in lowered
    assert "generated_route_weight_fold_update" in lowered
    assert "torch" not in lowered
    assert "pybind" not in lowered
    assert "at::tensor" not in lowered
    assert "atomic" not in lowered
    assert "deep_ep" not in lowered
    assert "mok" not in lowered
    assert "swiglu" not in lowered
    assert "attention" not in lowered


def test_relation_mutation_changes_data_without_changing_generated_source() -> None:
    plan, received, route_rows, route_weights, saved = _fixture()
    mutated_rows = route_rows.copy()
    mutated_rows[0] = (7, 2)
    baseline = evaluate_relation_edge_reverse(plan, received, route_rows, route_weights, saved)
    mutated = evaluate_relation_edge_reverse(plan, received, mutated_rows, route_weights, saved)
    generated = generate_cuda_relation_edge_reverse_ffi(plan, target="shuttle.relation_edge_reverse")

    assert not np.array_equal(baseline[0], mutated[0])
    assert not np.array_equal(baseline[1], mutated[1])
    assert (
        generated.source_digest
        == generate_cuda_relation_edge_reverse_ffi(plan, target="shuttle.relation_edge_reverse").source_digest
    )


def test_relation_edge_reverse_rejects_aliasing_padded_rows() -> None:
    plan, received, route_rows, route_weights, saved = _fixture()
    invalid = route_rows.copy()
    invalid[1, 0] = invalid[0, 0]

    with pytest.raises(ValueError, match="distinct padded row"):
        evaluate_relation_edge_reverse(plan, received, invalid, route_weights, saved)


def test_relation_edge_reverse_dimensions_are_part_of_physical_source() -> None:
    small = generate_cuda_relation_edge_reverse_ffi(
        RelationEdgeReverseFfiPlan(received_rows=4, route_slots=2, padded_rows=8, features=8),
        target="shuttle.relation_edge_reverse",
    )
    larger = generate_cuda_relation_edge_reverse_ffi(
        RelationEdgeReverseFfiPlan(received_rows=8, route_slots=2, padded_rows=16, features=8),
        target="shuttle.relation_edge_reverse",
    )

    assert small.semantic_digest != larger.semantic_digest
    assert small.source_digest != larger.source_digest


def _four_rank_relation():
    destination_indices = np.asarray(
        ((0, 3), (3, 4), (7, 3), (0, 7), (4, 3), (7, 0)),
        dtype=np.int32,
    )
    weights = np.asarray(
        ((0.6, 0.4), (0.25, 0.75), (0.1, 0.9), (0.3, 0.7), (0.8, 0.2), (0.45, 0.55)),
        dtype=np.float32,
    )
    return build_relation_plan(
        destination_indices,
        weights,
        destination_rank_by_item=np.arange(8, dtype=np.int32) // 2,
        destination_local_item_by_item=np.arange(8, dtype=np.int32) % 2,
        padding_quantum=2,
    )


def test_exact_relation_metadata_composes_torch_free_edge_and_existing_handler_families() -> None:
    relation = _four_rank_relation()
    composition = compose_distributed_expert_backward_typed_ffi(
        relation,
        hidden=8,
        intermediate=4,
        target_prefix="shuttle.distributed_reverse",
    )
    source_cotangent = np.arange(48, dtype=np.float32).reshape(6, 8) / 8
    saved_edge_output = np.arange(relation.destination_row_count * 8, dtype=np.float32).reshape(-1, 8) / 4
    rank_route_cotangents = []
    for rank_plan in composition.ranks:
        metadata = rank_plan.metadata
        received = source_cotangent[metadata.exchange_source_item]
        _, route_cotangent = evaluate_relation_edge_reverse(
            rank_plan.edge_reverse_plan,
            received,
            metadata.route_padded_rows,
            metadata.route_weights,
            saved_edge_output[metadata.global_padded_rows],
        )
        rank_route_cotangents.append(route_cotangent)
    restored_route_cotangent = restore_source_route_payload(
        relation,
        composition,
        tuple(rank_route_cotangents),
    )

    expected_route_cotangent = np.zeros((relation.source_item_count, relation.route_slots), dtype=np.float32)
    for source_item in range(relation.source_item_count):
        for route_slot in range(relation.route_slots):
            flat_route = source_item * relation.route_slots + route_slot
            padded_row = relation.route_to_destination_row[flat_route]
            expected_route_cotangent[source_item, route_slot] = np.dot(
                saved_edge_output[padded_row],
                source_cotangent[source_item],
            )
    np.testing.assert_array_equal(restored_route_cotangent, expected_route_cotangent)
    assert composition.runtime_dependencies == ("JAX/XLA typed FFI", "CUDA runtime", "cuBLAS")
    assert composition.router_vjp.startswith("JAX-owned")
    assert all("torch" not in rank.edge_reverse.source.lower() for rank in composition.ranks)
    for rank in composition.ranks:
        assert rank.input_adjoint.family is DistributedBackwardHandlerFamily.ROUTED_INPUT_ADJOINT
        assert tuple(binding.family for binding in rank.weight_adjoint) == (
            DistributedBackwardHandlerFamily.GROUP_BATCHED_CONTRACT,
            DistributedBackwardHandlerFamily.GROUP_BATCHED_CONTRACT,
        )
        assert rank.source_fold.family is DistributedBackwardHandlerFamily.SOURCE_INDEXED_FOLD
        assert rank.input_adjoint.generator.endswith(".generate_cuda_routed_input_adjoint_ffi")
        assert all(
            binding.generator.endswith(".generate_cuda_group_batched_contract_ffi") for binding in rank.weight_adjoint
        )
        assert rank.source_fold.generator.endswith(".generate_cuda_source_indexed_fold_ffi")


def test_exact_relation_return_mapping_preserves_source_slot_fold_order() -> None:
    relation = _four_rank_relation()
    composition = compose_distributed_expert_backward_typed_ffi(
        relation,
        hidden=8,
        intermediate=4,
        target_prefix="shuttle.distributed_reverse",
    )
    padded_input_cotangent = np.arange(relation.destination_row_count * 8, dtype=np.float32).reshape(-1, 8) / 8
    rank_payloads = []
    for rank_plan in composition.ranks:
        metadata = rank_plan.metadata
        local = padded_input_cotangent[metadata.global_padded_rows]
        payload = np.zeros((*metadata.route_valid.shape, 8), dtype=np.float32)
        payload[metadata.route_valid] = local[metadata.route_padded_rows[metadata.route_valid]]
        rank_payloads.append(payload)
    restored = restore_source_route_payload(relation, composition, tuple(rank_payloads))
    actual = deterministic_source_slot_fold(relation, restored)
    expected = relation.inverse_dispatch(padded_input_cotangent).sum(axis=1, dtype=np.float32)
    source_indices, contributions = source_indexed_fold_operands(relation, restored)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, deterministic_source_slot_fold(relation, restored))
    np.testing.assert_array_equal(
        source_indices[:, 0],
        np.repeat(np.arange(relation.source_item_count, dtype=np.int32), relation.route_slots),
    )
    np.testing.assert_array_equal(contributions.reshape(relation.source_item_count, relation.route_slots, -1), restored)


def test_one_device_jax_edge_reverse_matches_natural_weighted_fold_vjp_and_repeats() -> None:
    relation = _four_rank_relation()
    composition = compose_distributed_expert_backward_typed_ffi(
        relation,
        hidden=8,
        intermediate=4,
        target_prefix="shuttle.distributed_reverse",
    )
    saved_edge_output = (jnp.arange(relation.destination_row_count * 8).reshape(-1, 8) / 64).astype(jnp.bfloat16)
    output_cotangent = (jnp.arange(relation.source_item_count * 8).reshape(-1, 8) / 32).astype(jnp.bfloat16)
    destination_rows = jnp.asarray(relation.route_to_destination_row).reshape(
        relation.source_item_count,
        relation.route_slots,
    )

    def natural_weighted_fold(edge_output: jax.Array, route_weights: jax.Array) -> jax.Array:
        restored = edge_output[destination_rows]
        state = jnp.zeros((relation.source_item_count, 8), dtype=jnp.float32)
        for route_slot in range(relation.route_slots):
            state = state + restored[:, route_slot].astype(jnp.float32) * route_weights[:, route_slot, None]
        return state.astype(jnp.bfloat16)

    _, pullback = jax.vjp(natural_weighted_fold, saved_edge_output, jnp.asarray(relation.weight))
    expected_edge_cotangent, expected_route_cotangent = pullback(output_cotangent)

    def execute_adapter() -> tuple[np.ndarray, np.ndarray]:
        padded_edge_cotangent = np.zeros(expected_edge_cotangent.shape, dtype=np.float32)
        rank_route_cotangents = []
        for rank in composition.ranks:
            metadata = rank.metadata
            local_padded, local_route = evaluate_relation_edge_reverse_jax(
                rank.edge_reverse_plan,
                output_cotangent[jnp.asarray(metadata.exchange_source_item)],
                jnp.asarray(metadata.route_padded_rows),
                jnp.asarray(metadata.route_weights),
                saved_edge_output[jnp.asarray(metadata.global_padded_rows)],
            )
            padded_edge_cotangent[metadata.global_padded_rows] = np.asarray(local_padded, dtype=np.float32)
            rank_route_cotangents.append(np.asarray(local_route))
        route_cotangent = restore_source_route_payload(
            relation,
            composition,
            tuple(rank_route_cotangents),
        )
        return padded_edge_cotangent, route_cotangent

    first = execute_adapter()
    second = execute_adapter()

    np.testing.assert_array_equal(first[0], np.asarray(expected_edge_cotangent, dtype=np.float32))
    np.testing.assert_array_equal(first[1], np.asarray(expected_route_cotangent))
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_relation_weights_change_metadata_not_generic_handler_source() -> None:
    relation = _four_rank_relation()
    mutated = build_relation_plan(
        relation.destination_item.reshape(relation.source_item_count, relation.route_slots),
        relation.weight * np.float32(0.5),
        destination_rank_by_item=np.arange(8, dtype=np.int32) // 2,
        destination_local_item_by_item=np.arange(8, dtype=np.int32) % 2,
        padding_quantum=2,
    )
    baseline_composition = compose_distributed_expert_backward_typed_ffi(
        relation, hidden=8, intermediate=4, target_prefix="shuttle.distributed_reverse"
    )
    mutated_composition = compose_distributed_expert_backward_typed_ffi(
        mutated, hidden=8, intermediate=4, target_prefix="shuttle.distributed_reverse"
    )

    assert baseline_composition.relation_digest != mutated_composition.relation_digest
    assert tuple(rank.edge_reverse.source_digest for rank in baseline_composition.ranks) == tuple(
        rank.edge_reverse.source_digest for rank in mutated_composition.ranks
    )


def test_jax_boundary_rejects_wrong_shape_before_ffi_dispatch() -> None:
    plan, *_ = _fixture()
    generated = generate_cuda_relation_edge_reverse_ffi(plan, target="shuttle.relation_edge_reverse")

    with pytest.raises(ValueError, match=r"received_cotangent.*shape"):
        call_cuda_relation_edge_reverse_ffi(
            generated,
            plan,
            jnp.zeros((3, 8), dtype=jnp.bfloat16),
            jnp.zeros((4, 2), dtype=jnp.int32),
            jnp.zeros((4, 2), dtype=jnp.float32),
            jnp.zeros((8, 8), dtype=jnp.bfloat16),
        )


def test_torch_free_compile_plan_links_only_jax_ffi_and_cuda(tmp_path: Path) -> None:
    plan, *_ = _fixture()
    generated = generate_cuda_relation_edge_reverse_ffi(plan, target="shuttle.relation_edge_reverse")
    toolkit = tmp_path / "cuda"
    binary = toolkit / "bin" / "nvcc"
    library = toolkit / "lib64" / "libcudart.so"
    include = tmp_path / "jaxlib" / "include"
    binary.parent.mkdir(parents=True)
    library.parent.mkdir(parents=True)
    include.mkdir(parents=True)
    binary.touch()
    library.touch()
    compile_plan = relation_edge_reverse_cuda_compile_plan(
        generated,
        directory=tmp_path / "build",
        nvcc=binary,
        architecture="sm_100a",
        jaxlib_include=include,
    )
    command = " ".join(compile_plan.argv).lower()

    assert "sm_100a" in command
    assert "libcudart.so" in command
    argument_names = tuple(Path(argument).name.lower() for argument in compile_plan.argv)
    assert all("torch" not in name for name in argument_names)
    assert all("pybind" not in name for name in argument_names)
    assert all("mok" not in name for name in argument_names)
