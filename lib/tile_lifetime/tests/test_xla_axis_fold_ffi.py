# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
from pathlib import Path

import pytest

from tile_lifetime.cuda_axis_fold_codegen import generate_cuda_axis_fold_ffi
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.tensor_program import serialize_scalar_expression
from tile_lifetime.xla_axis_fold_ffi import (
    audit_axis_fold_hlo_region_replacement,
    plan_axis_fold_hlo_region_replacement,
    recover_axis_fold_hlo_region_candidates,
    replace_axis_fold_hlo_region_with_custom_call,
)
from tile_lifetime.xla_hlo_recovery import parse_hlo_module_text

_GRUG_HLO = (
    Path(__file__).parents[1]
    / "benchmarks/artifacts/grug_moe_train_step_pre_scheduler_jax011_v0/pre-scheduler-hlo.txt.gz"
)
_GRUG_GB200_HLO = (
    Path(__file__).parents[1]
    / "benchmarks/artifacts/xla_grug_routed_combined_gpu_gb200_v0/original-gpu-pre-scheduler-hlo.txt.gz"
)
_TARGET = "shuttle.generic_axis_fold.test"


def _hlo() -> str:
    return gzip.decompress(_GRUG_HLO.read_bytes()).decode()


def _gb200_hlo() -> str:
    return gzip.decompress(_GRUG_GB200_HLO.read_bytes()).decode()


def _generated_and_plan(hlo: str):
    report = recover_axis_fold_hlo_region_candidates(
        hlo,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    assert len(report.plans) == 1
    generated = generate_cuda_axis_fold_ffi((report.plans[0].program,), target_name=_TARGET)
    plan = plan_axis_fold_hlo_region_replacement(
        hlo,
        generated,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    return generated, plan


def test_natural_grug_hlo_recovers_one_generic_axis_fold_without_metadata_names() -> None:
    hlo = _hlo()
    report = recover_axis_fold_hlo_region_candidates(
        hlo,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    renamed = hlo.replace("RMSNorm", "AnonymousRowProgram").replace("GatedNorm", "AnonymousContract")
    renamed_report = recover_axis_fold_hlo_region_candidates(
        renamed,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    assert len(report.plans) == 1
    plan = report.plans[0]
    assert plan.program.rows == 8
    assert plan.program.columns == 32
    assert plan.program.semantic_fingerprint == renamed_report.plans[0].program.semantic_fingerprint
    assert plan.output_physical_shape == "bf16[2,4,32]{2,1,0}"
    assert tuple(value.ffi_shape for value in plan.inputs) == (
        "f32[8,32]{1,0}",
        "bf16[8,32]{1,0}",
        "bf16[8,32]{1,0}",
        "bf16[32]{0}",
        "f32[8]{0}",
        "f32[8]{0}",
    )
    assert len(plan.external_users) == 3
    assert "fold_sum" in serialize_scalar_expression(plan.provenance.output_expression)


def test_axis_fold_replacement_requires_explicit_rounding_reorder_policy() -> None:
    with pytest.raises(ValueError, match="rounding-reorder numerical policy"):
        recover_axis_fold_hlo_region_candidates(
            _hlo(),
            numerical_policy=NumericalPolicy.BITWISE_EXACT,
        )


def test_axis_fold_recovery_fails_closed_on_ambiguous_supported_regions() -> None:
    hlo = _hlo()
    fold = next(line for line in hlo.splitlines() if line.startswith("  %multiply_reduce_fusion.15 = "))
    final = next(line for line in hlo.splitlines() if line.startswith("  %add_convert_fusion.12 = "))
    cloned_fold = fold.replace("%multiply_reduce_fusion.15", "%cloned.fold", 1)
    cloned_final = final.replace("%add_convert_fusion.12", "%cloned.final", 1).replace(
        "%multiply_reduce_fusion.15", "%cloned.fold"
    )
    cloned_user = "  %cloned.user = bf16[2,4,32]{2,1,0} copy(%cloned.final)"
    ambiguous = hlo.replace(final, final + "\n" + cloned_fold + "\n" + cloned_final + "\n" + cloned_user, 1)
    report = recover_axis_fold_hlo_region_candidates(
        ambiguous,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    assert len(report.plans) == 2
    generated = generate_cuda_axis_fold_ffi((report.plans[0].program,), target_name=_TARGET)

    with pytest.raises(ValueError, match="matching the generated program, found 2"):
        plan_axis_fold_hlo_region_replacement(
            ambiguous,
            generated,
            numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        )


def test_axis_fold_recovery_rejects_control_crossing_and_extra_fold_user() -> None:
    hlo = _hlo()
    final_definition = (
        "  %add_convert_fusion.12 = bf16[2,4,32]{2,1,0} fusion(%bitcast_add_fusion.1, "
        "%copy_divide_fusion.5, %multiply_reduce_fusion.15, %wrapped_convert.59, "
        "%multiply_convert_fusion.20, /*index=5*/%wrapped_convert.55, %copy_rsqrt_fusion.4), "
        "kind=kLoop, calls=%fused_computation.294"
    )
    assert hlo.count(final_definition) == 1
    controlled = hlo.replace(
        final_definition, final_definition + ", control-predecessors={%multiply_reduce_fusion.15}", 1
    )
    controlled_report = recover_axis_fold_hlo_region_candidates(
        controlled,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    assert not controlled_report.plans
    assert any("control predecessor" in reason for _, reason in controlled_report.rejected)

    fold_definition = next(line for line in hlo.splitlines() if line.startswith("  %multiply_reduce_fusion.15 = "))
    extra_user = "  %unexpected.fold.user = f32[2,4]{1,0} copy(%multiply_reduce_fusion.15)"
    widened = hlo.replace(fold_definition, fold_definition + "\n" + extra_user, 1)
    widened_report = recover_axis_fold_hlo_region_candidates(
        widened,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    assert not widened_report.plans
    assert any("one finalizer" in reason for _, reason in widened_report.rejected)


def test_axis_fold_replacement_rewires_every_external_user_without_copy_or_transpose() -> None:
    hlo = _hlo()
    _, plan = _generated_and_plan(hlo)
    transformed = replace_axis_fold_hlo_region_with_custom_call(hlo, plan, target=_TARGET)
    audit = audit_axis_fold_hlo_region_replacement(hlo, transformed, plan, target=_TARGET)
    entry = parse_hlo_module_text(transformed).computation("main.149")
    instructions = {instruction.name: instruction for instruction in entry.instructions}

    assert audit.call_instruction == "shuttle.generated.axis_fold.region.shuttle.generic_axis_fold.test"
    assert audit.rewired_external_users == plan.external_users
    assert audit.copy_count[0] == audit.copy_count[1]
    assert audit.transpose_count[0] == audit.transpose_count[1]
    for user in plan.external_users:
        assert plan.output_instruction not in instructions[user].operands
        assert "shuttle.axis_fold.output.physical.shuttle.generic_axis_fold.test" in instructions[user].operands
    assert transformed.count(f'custom_call_target="{_TARGET}"') == 1


def test_axis_fold_roundtrip_audit_fails_closed_on_target_signature_and_consumer_changes() -> None:
    hlo = _hlo()
    _, plan = _generated_and_plan(hlo)
    transformed = replace_axis_fold_hlo_region_with_custom_call(hlo, plan, target=_TARGET)

    wrong_target = transformed.replace(
        f'custom_call_target="{_TARGET}"',
        'custom_call_target="shuttle.wrong.target"',
        1,
    )
    with pytest.raises(ValueError, match="expected one post-roundtrip axis-Fold call"):
        audit_axis_fold_hlo_region_replacement(hlo, wrong_target, plan, target=_TARGET)

    expected_constraints = f"operand_layout_constraints={{{', '.join(value.ffi_shape for value in plan.inputs)}}}"
    wrong_signature = transformed.replace(expected_constraints, "operand_layout_constraints={}", 1)
    with pytest.raises(ValueError, match="operand layout signature changed"):
        audit_axis_fold_hlo_region_replacement(hlo, wrong_signature, plan, target=_TARGET)

    replacement_output = "shuttle.axis_fold.output.physical.shuttle.generic_axis_fold.test"
    user = plan.external_users[0]
    user_line = next(line for line in transformed.splitlines() if line.startswith(f"  %{user} = "))
    wrong_consumer_line = user_line.replace(f"%{replacement_output}", f"%{plan.inputs[0].instruction}", 1)
    assert wrong_consumer_line != user_line
    wrong_consumer = transformed.replace(user_line, wrong_consumer_line, 1)
    with pytest.raises(ValueError, match="does not consume replacement Fold output"):
        audit_axis_fold_hlo_region_replacement(hlo, wrong_consumer, plan, target=_TARGET)

    old_consumer_line = user_line.replace(f"%{replacement_output}", f"%{plan.output_instruction}", 1)
    still_live = transformed.replace(user_line, old_consumer_line, 1)
    with pytest.raises(ValueError, match="remains externally live"):
        audit_axis_fold_hlo_region_replacement(hlo, still_live, plan, target=_TARGET)


def test_axis_fold_roundtrip_audit_allows_dead_layout_elimination_but_rejects_new_layout_work() -> None:
    hlo = _hlo()
    _, plan = _generated_and_plan(hlo)
    transformed = replace_axis_fold_hlo_region_with_custom_call(hlo, plan, target=_TARGET)
    module = parse_hlo_module_text(hlo)
    entry = module.computation(module.entry)
    copy_instruction = next(instruction for instruction in entry.instructions if instruction.opcode == "copy")
    copy_line = next(line for line in hlo.splitlines() if line.startswith(f"  %{copy_instruction.name} = "))
    copy_name = copy_line.split(" = ", 1)[0]
    dead_copy = copy_line.replace(copy_name, "  %audit.dead.copy", 1)

    original_with_dead_copy = hlo.replace(copy_line, copy_line + "\n" + dead_copy, 1)
    audit = audit_axis_fold_hlo_region_replacement(
        original_with_dead_copy,
        transformed,
        plan,
        target=_TARGET,
    )
    assert audit.copy_count[1] < audit.copy_count[0]

    transformed_with_new_copy = transformed.replace(copy_line, copy_line + "\n" + dead_copy, 1)
    with pytest.raises(ValueError, match="increased copy count"):
        audit_axis_fold_hlo_region_replacement(
            hlo,
            transformed_with_new_copy,
            plan,
            target=_TARGET,
        )


def test_preserved_gb200_hlo_recovers_two_anonymous_row_fold_programs() -> None:
    hlo = _gb200_hlo()
    report = recover_axis_fold_hlo_region_candidates(
        hlo,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    selected = tuple(
        plan
        for plan in report.plans
        if plan.program.rows == 8 and plan.program.columns == 32 and plan.output_ffi_shape.startswith("bf16[")
    )
    renamed_report = recover_axis_fold_hlo_region_candidates(
        hlo.replace("RMSNorm", "AnonymousFold").replace("GatedNorm", "AnonymousMap"),
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    renamed_selected = tuple(
        plan
        for plan in renamed_report.plans
        if plan.program.rows == 8 and plan.program.columns == 32 and plan.output_ffi_shape.startswith("bf16[")
    )

    assert tuple(plan.provenance.fold_instruction for plan in selected) == ("reduce_sum.688", "reduce_sum.770")
    assert tuple(plan.provenance.final_map_instruction for plan in selected) == (
        "convert_element_type.408",
        "convert_element_type.496",
    )
    assert tuple(plan.program.semantic_fingerprint for plan in selected) == tuple(
        plan.program.semantic_fingerprint for plan in renamed_selected
    )
    assert all(len(plan.program.reductions) == 1 for plan in selected)
    assert all(plan.program.output_dtype.value == "bf16" for plan in selected)
    assert all(plan.output_ffi_shape == "bf16[8,32]{1,0}" for plan in selected)


def test_preserved_gb200_row_fold_programs_rewrite_without_layout_adapters() -> None:
    hlo = _gb200_hlo()
    report = recover_axis_fold_hlo_region_candidates(
        hlo,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    selected = tuple(
        plan
        for plan in report.plans
        if plan.program.rows == 8 and plan.program.columns == 32 and plan.output_ffi_shape.startswith("bf16[")
    )
    assert len(selected) == 2

    rewritten = hlo
    for index, recovered in enumerate(selected):
        target = f"{_TARGET}.{index}"
        generated = generate_cuda_axis_fold_ffi((recovered.program,), target_name=target)
        plan = plan_axis_fold_hlo_region_replacement(
            rewritten,
            generated,
            numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        )
        next_hlo = replace_axis_fold_hlo_region_with_custom_call(rewritten, plan, target=target)
        audit = audit_axis_fold_hlo_region_replacement(rewritten, next_hlo, plan, target=target)

        assert audit.copy_count[0] == audit.copy_count[1]
        assert audit.transpose_count[0] == audit.transpose_count[1]
        rewritten = next_hlo

    assert rewritten.count('custom_call_target="shuttle.generic_axis_fold.test.') == 2
