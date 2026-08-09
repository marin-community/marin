# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import re
from dataclasses import replace
from pathlib import Path

from tile_lifetime.cast_scalar_program import (
    CastScalarExpression,
    CastScalarKind,
    CastScalarProgram,
    generate_cuda_scalar_body,
)
from tile_lifetime.contract_map_chain import BoundCastScalarMap, ContractMapChainValue
from tile_lifetime.cuda_contract_map_chain_codegen import (
    audit_cuda_contract_map_chain_source,
    generate_cuda_contract_map_chain_ffi,
)
from tile_lifetime.xla_low_rank_gated_product import recover_low_rank_gated_product_training
from tile_lifetime.xla_low_rank_gated_product_ffi import (
    audit_generated_low_rank_contract_map_training,
    audit_low_rank_contract_map_training_hlo_replacement,
    mutate_forward_hidden_scalar_program,
    plan_generated_low_rank_contract_map_training,
    plan_low_rank_contract_map_training_hlo_replacements,
    replace_generated_low_rank_contract_map_training,
    replace_low_rank_contract_map_training_hlo_regions_with_custom_calls,
)

_ARTIFACT = (
    Path(__file__).parents[1] / "benchmarks/artifacts/xla_grug_shared_map_h100_fused_reverses_unaccepted_e3411679_v0/"
    "transformed-gpu-pre-scheduler-hlo.txt.gz"
)
_FORWARD_TARGET = "shuttle.generic.low_rank_contract_map.forward.test"
_REVERSE_TARGET = "shuttle.generic.low_rank_contract_map.reverse.test"
_GENERATED_FORWARD_TARGET = "shuttle.generic.low_rank_contract_map.generated.forward.test"
_GENERATED_REVERSE_TARGET = "shuttle.generic.low_rank_contract_map.generated.reverse.test"


def _current_training_hlo() -> str:
    with gzip.open(_ARTIFACT, "rt") as handle:
        return handle.read()


def test_natural_training_hlo_recovers_repeated_low_rank_gated_product_families() -> None:
    report = recover_low_rank_gated_product_training(_current_training_hlo())

    group_counts = sorted(
        len(tuple(plan for plan in report.forward_realizations if plan.parameter_origins == origins))
        for origins in {plan.parameter_origins for plan in report.forward_realizations}
    )
    assert len(report.forward_realizations) == 6
    assert len(report.reverse_families) == 4
    assert group_counts == [1, 1, 2, 2]
    assert len({plan.hidden_map.digest for plan in report.forward_realizations}) == 1
    assert len({plan.output_map.digest for plan in report.forward_realizations}) == 1
    assert len({plan.up_input_map.digest for plan in report.reverse_families}) == 1
    assert len({plan.hidden_vjp_map.digest for plan in report.reverse_families}) == 1
    assert len({plan.residual_vjp_map.digest for plan in report.reverse_families}) == 1
    assert report.live_contract_count == 52
    assert report.live_contract_flops == 2_232_320
    assert report.owned_contract_count == 28
    assert report.owned_contract_flops == 1_835_008


def test_low_rank_gated_product_recovery_ignores_frontend_metadata() -> None:
    hlo = _current_training_hlo()
    without_metadata = re.sub(r", metadata=\{.*$", "", hlo, flags=re.MULTILINE)

    assert recover_low_rank_gated_product_training(without_metadata) == recover_low_rank_gated_product_training(hlo)


def test_low_rank_gated_product_plan_preserves_bf16_and_collective_boundaries() -> None:
    report = recover_low_rank_gated_product_training(_current_training_hlo())

    for forward in report.forward_realizations:
        assert forward.input.shape.startswith("bf16[")
        assert forward.down_contract.output.shape.startswith("bf16[")
        assert forward.hidden.shape.startswith("bf16[")
        assert forward.up_contract.output.shape.startswith("bf16[")
        assert forward.output.shape.startswith("bf16[")
        assert forward.hidden_map.numerical_policy.value == "source_ordered"
        assert forward.output_map.numerical_policy.value == "source_ordered"
    for reverse in report.reverse_families:
        assert reverse.up_input_adjoint.output.shape.startswith("bf16[")
        assert reverse.down_input_adjoint.output.shape.startswith("bf16[")
        assert reverse.input_adjoint.shape.startswith("bf16[")
        assert reverse.down_weight_adjoint.output.shape.startswith("bf16[")
        assert reverse.up_weight_adjoint.output.shape.startswith("bf16[")
        assert reverse.upstream_collectives


def test_low_rank_gated_product_hidden_map_mutates_through_same_scalar_generator() -> None:
    plan = recover_low_rank_gated_product_training(_current_training_hlo()).forward_realizations[0]
    hidden_input = plan.hidden_map.inputs[0]
    mutated_map = CastScalarProgram(
        CastScalarExpression(
            kind=CastScalarKind.TANH,
            dtype=hidden_input.dtype,
            operands=(hidden_input,),
        )
    )
    mutated_plan = replace(plan, hidden_map=mutated_map)

    original_source = generate_cuda_scalar_body(plan.hidden_map, symbol="generated_hidden_map")
    mutated_source = generate_cuda_scalar_body(mutated_plan.hidden_map, symbol="generated_hidden_map")
    assert mutated_plan.down_contract == plan.down_contract
    assert mutated_plan.up_contract == plan.up_contract
    assert mutated_source.semantic_digest != original_source.semantic_digest
    assert "tanhf" in mutated_source.source


def test_low_rank_contract_map_plans_form_exact_forward_remat_and_jax_reverse_abis() -> None:
    plan = plan_low_rank_contract_map_training_hlo_replacements(_current_training_hlo())

    assert len(plan.forward) == 6
    assert len(plan.reverse) == 4
    assert tuple(len(boundary.outputs) for boundary in plan.forward) == (5, 1, 1, 5, 5, 5)
    assert all(len(boundary.outputs) == 3 for boundary in plan.reverse)
    assert all(len(boundary.inputs) == 4 for boundary in plan.forward)
    assert all(len(boundary.inputs) == 9 for boundary in plan.reverse)
    assert all(
        value.shape.startswith("bf16[") for boundary in (*plan.forward, *plan.reverse) for value in boundary.inputs
    )
    assert all(
        value.shape.startswith("bf16[") for boundary in (*plan.forward, *plan.reverse) for value in boundary.outputs
    )
    assert plan.replaced_dot_count == 28
    assert plan.replaced_dot_flops == 1_835_008
    assert plan.original_live_dot_count == 52
    assert plan.original_live_dot_flops == 2_232_320
    assert tuple(value.instruction for value in plan.reverse[0].cotangent_inputs) == ("psum.48",)
    assert tuple(value.instruction for value in plan.reverse[1].cotangent_inputs) == ("reshape.932",)


def test_low_rank_contract_map_replacement_kills_old_work_and_preserves_collectives() -> None:
    hlo = _current_training_hlo()
    plan = plan_low_rank_contract_map_training_hlo_replacements(hlo)
    rewritten = replace_low_rank_contract_map_training_hlo_regions_with_custom_calls(
        hlo,
        plan,
        forward_target=_FORWARD_TARGET,
        reverse_target=_REVERSE_TARGET,
    )
    audit = audit_low_rank_contract_map_training_hlo_replacement(
        hlo,
        rewritten,
        plan,
        forward_target=_FORWARD_TARGET,
        reverse_target=_REVERSE_TARGET,
    )

    assert audit.generated_call_count == 10
    assert audit.removed_dot_count == 28
    assert audit.removed_dot_flops == 1_835_008
    assert not audit.live_old_arithmetic
    assert len(audit.collective_instructions) == 10
    assert audit.forward[0].outputs == ("reshape.22", "div.10", "reshape.23", "reshape.952", "mul.693")
    assert audit.forward[0].output_users[:4] == (
        ("reshape.22", ("shuttle.generated.low_rank_contract_map.reverse.3",)),
        ("div.10", ("shuttle.generated.low_rank_contract_map.reverse.3",)),
        ("reshape.23", ("shuttle.generated.low_rank_contract_map.reverse.3",)),
        ("reshape.952", ("shuttle.generated.low_rank_contract_map.reverse.3",)),
    )
    assert audit.reverse[0].outputs == ("add_any.68", "dot.85", "dot.86")
    assert audit.reverse[0].output_users == (
        ("add_any.68", ("convert_element_type.402",)),
        ("dot.85", ("convert_element_type.498",)),
        ("dot.86", ("convert_element_type.496",)),
    )
    assert audit.upstream_collective_paths[:4] == (
        ("shuttle.generated.low_rank_contract_map.reverse.0", ("psum.48",), "psum.48"),
        ("shuttle.generated.low_rank_contract_map.reverse.1", ("reshape.932",), "psum.48"),
        ("shuttle.generated.low_rank_contract_map.reverse.1", ("reshape.932",), "psum.50"),
        ("shuttle.generated.low_rank_contract_map.reverse.1", ("reshape.932",), "psum.51"),
    )
    assert rewritten.count(f'custom_call_target="{_FORWARD_TARGET}"') == 6
    assert rewritten.count(f'custom_call_target="{_REVERSE_TARGET}"') == 4
    assert rewritten.count("custom_call_target=") == 23


def test_low_rank_contract_map_mutation_retains_boundary_family_and_target() -> None:
    hlo = _current_training_hlo()
    plan = plan_low_rank_contract_map_training_hlo_replacements(hlo)
    original = plan.forward[0]
    hidden_input = original.scalar_programs[0].inputs[0]
    tanh_map = CastScalarProgram(
        CastScalarExpression(
            kind=CastScalarKind.TANH,
            dtype=hidden_input.dtype,
            operands=(hidden_input,),
        )
    )
    mutated = mutate_forward_hidden_scalar_program(original, tanh_map)

    assert mutated.boundary_family_digest == original.boundary_family_digest
    assert mutated.semantic_digest != original.semantic_digest
    assert mutated.inputs == original.inputs
    assert mutated.outputs == original.outputs
    assert mutated.call_name == original.call_name
    assert "tanhf" in generate_cuda_scalar_body(mutated.scalar_programs[0], symbol="generated_map").source
    mutated_training = replace(plan, forward=(mutated, *plan.forward[1:]))
    rewritten = replace_low_rank_contract_map_training_hlo_regions_with_custom_calls(
        hlo,
        mutated_training,
        forward_target=_FORWARD_TARGET,
        reverse_target=_REVERSE_TARGET,
    )
    assert rewritten.count(f'custom_call_target="{_FORWARD_TARGET}"') == 6


def test_low_rank_contract_map_boundary_is_independent_of_hlo_metadata() -> None:
    hlo = _current_training_hlo()
    stripped = re.sub(r", metadata=\{.*$", "", hlo, flags=re.MULTILINE)

    original = plan_low_rank_contract_map_training_hlo_replacements(hlo)
    without_metadata = plan_low_rank_contract_map_training_hlo_replacements(stripped)
    assert tuple(plan.boundary_family_digest for plan in (*original.forward, *original.reverse)) == tuple(
        plan.boundary_family_digest for plan in (*without_metadata.forward, *without_metadata.reverse)
    )
    assert tuple((plan.inputs, plan.outputs) for plan in (*original.forward, *original.reverse)) == tuple(
        (plan.inputs, plan.outputs) for plan in (*without_metadata.forward, *without_metadata.reverse)
    )


def test_generated_low_rank_contract_map_normalizes_ten_calls_to_one_physical_family() -> None:
    hlo = _current_training_hlo()
    plan = plan_generated_low_rank_contract_map_training(
        hlo,
        forward_target_prefix=_GENERATED_FORWARD_TARGET,
        reverse_target_prefix=_GENERATED_REVERSE_TARGET,
    )
    rewritten = replace_generated_low_rank_contract_map_training(hlo, plan)
    audit = audit_generated_low_rank_contract_map_training(hlo, rewritten, plan)

    assert len(plan.families) == 1
    assert plan.expected_target_occurrences == ((_GENERATED_FORWARD_TARGET, 6), (_GENERATED_REVERSE_TARGET, 4))
    assert plan.families[0].program.first_weight_adjoint_minor_to_major == (0, 1)
    assert plan.families[0].program.second_weight_adjoint_minor_to_major == (0, 1)
    assert audit.generated_call_count == 10
    assert audit.generated_target_count == 2
    assert audit.target_occurrences == plan.expected_target_occurrences
    assert audit.removed_dot_count == 28
    assert audit.removed_dot_flops == 1_835_008
    assert not audit.live_old_arithmetic
    assert len(audit.collective_instructions) == 10
    assert all(len(call.inputs) == 3 and len(call.outputs) == 4 for call in audit.forward)
    assert all(len(call.inputs) == 7 and len(call.outputs) == 3 for call in audit.reverse)
    assert all(call.outputs[1].shape.endswith("{0,1}") for call in audit.reverse)
    assert rewritten.count(f'custom_call_target="{_GENERATED_FORWARD_TARGET}"') == 6
    assert rewritten.count(f'custom_call_target="{_GENERATED_REVERSE_TARGET}"') == 4


def test_generated_low_rank_contract_map_mutation_reuses_targets_and_physical_abi() -> None:
    plan = plan_generated_low_rank_contract_map_training(
        _current_training_hlo(),
        forward_target_prefix=_GENERATED_FORWARD_TARGET,
        reverse_target_prefix=_GENERATED_REVERSE_TARGET,
    )
    family = plan.families[0]
    hidden_input = family.program.hidden_map.program.inputs[0]
    tanh_program = CastScalarProgram(
        CastScalarExpression(
            kind=CastScalarKind.TANH,
            dtype=hidden_input.dtype,
            operands=(hidden_input,),
        )
    )
    mutated_program = replace(
        family.program,
        hidden_map=BoundCastScalarMap(tanh_program, (ContractMapChainValue.FIRST_CONTRACT_OUTPUT,)),
    )
    original = generate_cuda_contract_map_chain_ffi(
        family.program,
        forward_target=family.forward_target,
        reverse_target=family.reverse_target,
    )
    mutated = generate_cuda_contract_map_chain_ffi(
        mutated_program,
        forward_target=family.forward_target,
        reverse_target=family.reverse_target,
    )
    audit = audit_cuda_contract_map_chain_source(mutated)

    assert mutated.forward_target == original.forward_target
    assert mutated.reverse_target == original.reverse_target
    assert mutated.forward_handler_symbol == original.forward_handler_symbol
    assert mutated.reverse_handler_symbol == original.reverse_handler_symbol
    assert mutated.semantic_digest != original.semantic_digest
    assert mutated.source_digest != original.source_digest
    assert "tanhf" in mutated.source
    assert not audit.has_atomics
    assert not audit.opaque_semantic_dependencies
