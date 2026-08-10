# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tile_lifetime.cast_scalar_program import (
    CastScalarKind,
    CastScalarProgram,
    generate_cuda_scalar_body,
)
from tile_lifetime.ffi_command_buffer import DirectLaunchFfiPhysicalCandidate, audit_ffi_command_buffer_eligibility
from tile_lifetime.xla_hlo_recovery import EntryRegionValue, parse_hlo_module_text
from tile_lifetime.xla_rank_two_contract_ffi import (
    audit_rank_two_contract_replacement,
    evaluate_rank_two_contract_plan,
    generate_cuda_rank_two_contract_ffi,
    plan_rank_two_bf16_contract_typed_ffi,
    replace_rank_two_contract_with_custom_call,
)
from tile_lifetime.xla_relation_program_recovery import plan_routed_input_adjoint_typed_ffi
from tile_lifetime.xla_routed_shared_map_training_ffi import (
    RoutedSharedMapTrainingFfiTargets,
    audit_routed_shared_map_training_replacement,
    plan_routed_shared_map_training_typed_ffi,
    replace_routed_shared_map_training_regions_with_custom_calls,
)
from tile_lifetime.xla_source_indexed_fold_ffi import (
    audit_source_indexed_fold_replacement,
    evaluate_source_indexed_fold_plan,
    generate_cuda_source_indexed_fold_ffi,
    plan_source_indexed_fold_typed_ffi,
    replace_source_indexed_fold_with_custom_call,
)

_ARTIFACT = (
    Path(__file__).parents[1]
    / "benchmarks/artifacts/xla_grug_routed_forward_gpu_gb200_v0/original-gpu-pre-scheduler-hlo.txt.gz"
)


def _hlo() -> str:
    return gzip.decompress(_ARTIFACT.read_bytes()).decode()


def _bf16(value: np.ndarray) -> np.ndarray:
    return np.asarray(jnp.asarray(value, dtype=jnp.bfloat16).astype(jnp.float32))


def test_rank_two_contract_recovery_generation_and_replacement_preserve_generic_boundary() -> None:
    hlo = _hlo()
    input_adjoint = plan_routed_input_adjoint_typed_ffi(hlo)
    plan = plan_rank_two_bf16_contract_typed_ffi(hlo, input_adjoint.contracts[0])

    assert (plan.lhs.instruction, plan.lhs.shape) == ("reshape.357", "bf16[512,128]{1,0}")
    assert (plan.rhs.instruction, plan.rhs.shape) == ("reshape.358", "bf16[128,32]{1,0}")
    assert plan.output_shape == "bf16[512,32]{1,0}"
    assert plan.external_users == ("slice.54",)

    generated = generate_cuda_rank_two_contract_ffi(plan, target="shuttle.contract.test")
    transformed = replace_rank_two_contract_with_custom_call(hlo, plan, target=generated.target)
    audit = audit_rank_two_contract_replacement(hlo, transformed, plan, target=generated.target)

    assert audit.call_instruction == "dot.67"
    assert audit.operands == ("reshape.357", "reshape.358")
    assert audit.external_users == ("slice.54",)
    assert "atomicAdd(" not in generated.source


def test_rank_two_contract_cpu_semantics_match_independent_numpy_reference() -> None:
    input_adjoint = plan_routed_input_adjoint_typed_ffi(_hlo())
    plan = plan_rank_two_bf16_contract_typed_ffi(_hlo(), input_adjoint.contracts[0])
    rng = np.random.default_rng(41)
    lhs = _bf16(rng.normal(scale=0.2, size=(512, 128)).astype(np.float32))
    rhs = _bf16(rng.normal(scale=0.2, size=(128, 32)).astype(np.float32))

    observed = evaluate_rank_two_contract_plan(plan, lhs, rhs)
    expected = _bf16(lhs.astype(np.float32) @ rhs.astype(np.float32))

    assert np.array_equal(observed, expected)


def test_rank_two_contract_shape_mutation_regenerates_source_and_semantic_digest() -> None:
    input_adjoint = plan_routed_input_adjoint_typed_ffi(_hlo())
    baseline = plan_rank_two_bf16_contract_typed_ffi(_hlo(), input_adjoint.contracts[0])
    mutated = replace(
        baseline,
        lhs=replace(baseline.lhs, shape="bf16[256,128]{1,0}"),
        output_shape="bf16[256,32]{1,0}",
    )

    baseline_generated = generate_cuda_rank_two_contract_ffi(baseline, target="shuttle.contract.baseline")
    mutated_generated = generate_cuda_rank_two_contract_ffi(mutated, target="shuttle.contract.mutated")

    assert baseline_generated.semantic_digest != mutated_generated.semantic_digest
    assert "constexpr int kRows = 512;" in baseline_generated.source
    assert "constexpr int kRows = 256;" in mutated_generated.source


def test_source_indexed_fold_recovery_generation_and_replacement_preserve_ordering_inputs() -> None:
    hlo = _hlo()
    input_adjoint = plan_routed_input_adjoint_typed_ffi(hlo)
    plan = plan_source_indexed_fold_typed_ffi(hlo, input_adjoint, input_adjoint.contracts[1])

    assert plan.instruction == "scatter-add.42"
    assert plan.initial == EntryRegionValue("broadcast.113", "bf16[8,32]{1,0}")
    assert plan.source_indices == EntryRegionValue("broadcast_in_dim.428", "s32[16,1]{1,0}")
    assert plan.contributions == EntryRegionValue("reshape.408", "bf16[16,1,32]{2,1,0}")
    assert plan.contribution_wrappers == ("slice.58", "reshape.408")
    assert plan.external_users == ("psum.50",)
    assert not plan.numerical_contract.atomic_accumulation

    generated = generate_cuda_source_indexed_fold_ffi(plan, target="shuttle.fold.test")
    transformed = replace_source_indexed_fold_with_custom_call(hlo, plan, target=generated.target)
    audit = audit_source_indexed_fold_replacement(hlo, transformed, plan, target=generated.target)

    assert audit.operands == ("broadcast.113", "broadcast_in_dim.428", "reshape.408")
    assert audit.external_users == ("psum.50",)
    assert "atomicAdd(" not in generated.source


def test_source_indexed_fold_capture_safe_variant_passes_static_audit() -> None:
    input_adjoint = plan_routed_input_adjoint_typed_ffi(_hlo())
    plan = plan_source_indexed_fold_typed_ffi(_hlo(), input_adjoint, input_adjoint.contracts[1])
    launch_checked = generate_cuda_source_indexed_fold_ffi(plan, target="shuttle.fold.checked")
    capture_safe = generate_cuda_source_indexed_fold_ffi(
        plan,
        target="shuttle.fold.capture_safe",
        physical_candidate=DirectLaunchFfiPhysicalCandidate.COMMAND_BUFFER_CAPTURE_SAFE,
    )

    assert not launch_checked.command_buffer_compatible
    assert not audit_ffi_command_buffer_eligibility(launch_checked.source).eligible
    assert capture_safe.command_buffer_compatible
    assert audit_ffi_command_buffer_eligibility(capture_safe.source).eligible
    assert "cudaPeekAtLastError(" not in capture_safe.source


def test_source_indexed_fold_cpu_semantics_are_source_ordered_and_deterministic() -> None:
    input_adjoint = plan_routed_input_adjoint_typed_ffi(_hlo())
    plan = plan_source_indexed_fold_typed_ffi(_hlo(), input_adjoint, input_adjoint.contracts[1])
    rng = np.random.default_rng(53)
    initial = _bf16(rng.normal(scale=0.2, size=(8, 32)).astype(np.float32))
    source_indices = np.asarray([3, 0, 3, 7, 1, 3, 0, 2, 7, 7, 4, 6, 3, 5, 0, 1], dtype=np.int32)[:, None]
    contributions = _bf16(rng.normal(scale=0.2, size=(16, 1, 32)).astype(np.float32))

    observed = evaluate_source_indexed_fold_plan(plan, initial, source_indices, contributions)
    expected = initial.copy()
    for source in range(8):
        for feature in range(32):
            accumulator = expected[source, feature]
            for edge in range(16):
                if source_indices[edge, 0] == source:
                    accumulator = _bf16(np.asarray([accumulator + contributions[edge, 0, feature]], dtype=np.float32))[0]
            expected[source, feature] = accumulator

    assert np.array_equal(observed, expected)
    assert np.array_equal(
        observed,
        evaluate_source_indexed_fold_plan(plan, initial, source_indices, contributions),
    )


def test_source_indexed_fold_shape_and_reducer_mutations_regenerate_generic_body() -> None:
    input_adjoint = plan_routed_input_adjoint_typed_ffi(_hlo())
    baseline = plan_source_indexed_fold_typed_ffi(_hlo(), input_adjoint, input_adjoint.contracts[1])
    shape_mutation = replace(
        baseline,
        initial=replace(baseline.initial, shape="bf16[4,16]{1,0}"),
        source_indices=replace(baseline.source_indices, shape="s32[6,1]{1,0}"),
        contributions=replace(baseline.contributions, shape="bf16[6,1,16]{2,1,0}"),
        output_shape="bf16[4,16]{1,0}",
    )
    reducer_expression = replace(baseline.reducer_program.expression, kind=CastScalarKind.SUBTRACT)
    reducer_program = CastScalarProgram(reducer_expression)
    reducer_mutation = replace(
        baseline,
        reducer_program=reducer_program,
        generated_reducer_cuda=generate_cuda_scalar_body(reducer_program, symbol="generated_fold_update"),
    )

    baseline_generated = generate_cuda_source_indexed_fold_ffi(baseline, target="shuttle.fold.baseline")
    shape_generated = generate_cuda_source_indexed_fold_ffi(shape_mutation, target="shuttle.fold.shape")
    reducer_generated = generate_cuda_source_indexed_fold_ffi(reducer_mutation, target="shuttle.fold.reducer")

    assert (
        len({baseline_generated.semantic_digest, shape_generated.semantic_digest, reducer_generated.semantic_digest})
        == 3
    )
    assert "constexpr int kSources = 4;" in shape_generated.source
    assert "constexpr int kEdges = 6;" in shape_generated.source
    assert "constexpr int kFeatures = 16;" in shape_generated.source
    assert "__fsub_rn" in reducer_generated.source


def test_composed_replacement_leaves_only_view_wrappers_in_input_adjoint() -> None:
    hlo = _hlo()
    targets = RoutedSharedMapTrainingFfiTargets(
        forward="shuttle.composed.forward",
        input_contracts=("shuttle.composed.contract.0", "shuttle.composed.contract.1"),
        shared_contract_multi_map="shuttle.composed.maps",
        source_fold="shuttle.composed.fold",
        weight_gradients=("shuttle.composed.weight.0", "shuttle.composed.weight.1"),
    )
    plan = plan_routed_shared_map_training_typed_ffi(hlo)
    transformed = replace_routed_shared_map_training_regions_with_custom_calls(hlo, plan, targets=targets)
    audit = audit_routed_shared_map_training_replacement(hlo, transformed, plan, targets=targets)
    module = parse_hlo_module_text(transformed)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}

    assert tuple(instructions[name].opcode for name in ("dot.67", "dot.68", "scatter-add.42")) == (
        "custom-call",
        "custom-call",
        "custom-call",
    )
    assert audit.retained_input_adjoint_wrappers == (
        "slice.54",
        "transpose.167",
        "reshape.360",
        "slice.58",
        "reshape.408",
    )
    assert audit.source_fold_collective == "psum.50"
    assert {instructions[name].opcode for name in audit.retained_input_adjoint_wrappers} <= {
        "bitcast",
        "copy",
        "reshape",
        "slice",
        "transpose",
    }
