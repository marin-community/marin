# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import functools
import gzip
import re
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from lib.tile_lifetime.benchmarks.xla_grug_routed_combined_gpu_custom_call import (
    _TARGETS as _SEVEN_CALL_TARGETS,
)
from lib.tile_lifetime.benchmarks.xla_grug_routed_combined_gpu_custom_call import (
    _custom_call_target_occurrences,
    _generate_axis_fold_programs,
)
from tile_lifetime.cuda_axis_fold_codegen import generate_cuda_axis_fold_ffi
from tile_lifetime.jax_streaming_attention_backward_ffi import (
    StreamingAttentionBackwardFfiBufferLayout,
    StreamingAttentionBackwardStatePolicy,
    generate_streaming_attention_backward_ffi,
)
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.stablehlo_import import import_stablehlo
from tile_lifetime.stablehlo_streaming_attention_backward import (
    recover_experimental_whole_pattern_streaming_attention_backward,
)
from tile_lifetime.streaming_attention import StreamingTileSchedule
from tile_lifetime.streaming_attention_backward import (
    StreamingAttentionBackwardDomainTraversal,
    derive_streaming_attention_backward_tile_schedule,
    eliminate_normalized_exp_maximum_vjp,
)
from tile_lifetime.streaming_attention_backward_reference import (
    STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    StreamingAttentionBackwardDebugConfig,
    causal_gqa_attention_vjp,
    export_debug_streaming_attention_backward,
)
from tile_lifetime.xla_axis_fold_ffi import recover_axis_fold_hlo_region_candidates
from tile_lifetime.xla_hlo_recovery import parse_hlo_module_text
from tile_lifetime.xla_routed_training_ffi import (
    RoutedTrainingAndAttentionFfiTargets,
    RoutedTrainingAttentionAndAxisFoldFfiTargets,
    RoutedTrainingFfiTargets,
    audit_routed_training_and_attention_replacement,
    audit_routed_training_attention_and_axis_fold_replacement,
    plan_routed_training_and_attention_typed_ffi,
    plan_routed_training_attention_and_axis_fold_typed_ffi,
    replace_routed_training_and_attention_regions_with_custom_calls,
    replace_routed_training_attention_and_axis_fold_regions_with_custom_calls,
    replace_routed_training_regions_with_custom_calls,
)
from tile_lifetime.xla_streaming_attention_backward_ffi import (
    StreamingReverseHloRole,
    audit_streaming_attention_backward_region_replacement,
    derive_streaming_attention_backward_ffi_output_layouts,
    plan_streaming_attention_backward_hlo_region_replacement,
    plan_streaming_attention_backward_hlo_replacement,
    replace_streaming_attention_backward_entry_with_custom_call,
    replace_streaming_attention_backward_region_with_custom_call,
)

_GRUG_GPU_HLO = (
    Path(__file__).parents[1]
    / "benchmarks/artifacts/xla_grug_routed_combined_gpu_gb200_v0/original-gpu-pre-scheduler-hlo.txt.gz"
)
_GRUG_CPU_HLO = (
    Path(__file__).parents[1]
    / "benchmarks/artifacts/grug_moe_train_step_pre_scheduler_jax011_v0/pre-scheduler-hlo.txt.gz"
)
_COMBINED_TARGETS = RoutedTrainingAndAttentionFfiTargets(
    routed=RoutedTrainingFfiTargets(
        forward="shuttle.combined.forward.test",
        input_adjoint="shuttle.combined.input_adjoint.test",
        weight_gradients=(
            "shuttle.combined.weight_gradient.0.test",
            "shuttle.combined.weight_gradient.1.test",
        ),
    ),
    attention_backward="shuttle.combined.attention_backward.test",
)
_SIX_TARGETS = RoutedTrainingAttentionAndAxisFoldFfiTargets(
    routed_attention=_COMBINED_TARGETS,
    axis_folds=(
        "shuttle.combined.axis_fold.0.test",
        "shuttle.combined.axis_fold.1.test",
    ),
)


@functools.cache
def _program_generated_and_hlo(scale: float = 0.125):
    config = StreamingAttentionBackwardDebugConfig(
        batch=1,
        query_length=16,
        key_length=16,
        query_heads=4,
        key_value_heads=2,
        head_dimension=64,
        scale=scale,
    )
    graph = import_stablehlo(
        export_debug_streaming_attention_backward(config),
        input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    )
    recovered = recover_experimental_whole_pattern_streaming_attention_backward(
        graph,
        schedule=StreamingTileSchedule(query_tile_size=8, key_value_tile_size=8, pipeline_depth=3),
    )
    program = eliminate_normalized_exp_maximum_vjp(
        recovered.program,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    schedule = derive_streaming_attention_backward_tile_schedule(
        program,
        query_tile_size=8,
        key_value_tile_size=8,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    generated = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name=f"shuttle.streaming_reverse.hlo_scale_{str(scale).replace('.', '_')}",
    )
    shapes = tuple(specification.shape for specification in generated.inputs)
    lowered = jax.jit(causal_gqa_attention_vjp(config)).lower(
        *(jax.ShapeDtypeStruct(shape, jnp.bfloat16) for shape in shapes)
    )
    return program, generated, lowered.compiler_ir("hlo").as_hlo_text()


@functools.cache
def _grug_region_inputs(scale: float = 0.32421875):
    config = StreamingAttentionBackwardDebugConfig(
        batch=2,
        query_length=4,
        key_length=4,
        query_heads=2,
        key_value_heads=1,
        head_dimension=16,
        scale=scale,
    )
    graph = import_stablehlo(
        export_debug_streaming_attention_backward(config),
        input_names=STREAMING_ATTENTION_BACKWARD_INPUT_NAMES,
    )
    recovered = recover_experimental_whole_pattern_streaming_attention_backward(
        graph,
        schedule=StreamingTileSchedule(query_tile_size=4, key_value_tile_size=4, pipeline_depth=2),
    )
    program = eliminate_normalized_exp_maximum_vjp(
        recovered.program,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    schedule = derive_streaming_attention_backward_tile_schedule(
        program,
        query_tile_size=4,
        key_value_tile_size=4,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    generated = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name=f"shuttle.streaming_reverse.grug_region_test_{str(scale).replace('.', '_')}",
    )
    hlo = gzip.decompress(_GRUG_GPU_HLO.read_bytes()).decode()
    return program, generated, hlo


def test_natural_jax_vjp_hlo_derives_generic_reverse_boundary_and_roles() -> None:
    program, generated, hlo = _program_generated_and_hlo()

    plan = plan_streaming_attention_backward_hlo_replacement(hlo, program, generated)

    assert tuple(value.role for value in plan.inputs) == (
        StreamingReverseHloRole.QUERY,
        StreamingReverseHloRole.KEY,
        StreamingReverseHloRole.VALUE,
        StreamingReverseHloRole.OUTPUT_COTANGENT,
    )
    assert tuple(value.role for value in plan.outputs) == (
        StreamingReverseHloRole.QUERY_COTANGENT,
        StreamingReverseHloRole.KEY_COTANGENT,
        StreamingReverseHloRole.VALUE_COTANGENT,
    )
    assert len(plan.provenance.reverse_contracts) == 4
    assert len(plan.provenance.additive_folds) >= 2
    assert plan.provenance.domain_restriction is not None
    assert plan.provenance.score_scale == 0.125
    assert plan.state_policy is StreamingAttentionBackwardStatePolicy.RECOMPUTE
    assert plan.maximum_vjp == "normalized_exp_invariant"


def test_rewrite_preserves_entry_layouts_around_default_ffi_buffers() -> None:
    program, generated, hlo = _program_generated_and_hlo()
    plan = plan_streaming_attention_backward_hlo_replacement(hlo, program, generated)

    rewritten = replace_streaming_attention_backward_entry_with_custom_call(
        hlo,
        plan,
        target=generated.target_name,
    )
    module = parse_hlo_module_text(rewritten)
    entry = module.computation(module.entry)
    call = next(instruction for instruction in entry.instructions if instruction.opcode == "custom-call")
    instructions = {instruction.name: instruction for instruction in entry.instructions}

    assert generated.target_name in call.attributes
    assert entry.root.shape == plan.root_shape
    assert tuple(entry.root.operands) == (
        "shuttle.query_cotangent.ffi",
        "shuttle.key_cotangent.physical",
        "shuttle.value_cotangent.physical",
    )
    assert instructions["shuttle.query_cotangent.ffi"].shape == plan.outputs[0].ffi_shape
    assert instructions["shuttle.key_cotangent.physical"].shape == plan.outputs[1].physical_shape
    assert instructions["shuttle.value_cotangent.physical"].shape == plan.outputs[2].physical_shape


def test_hlo_derived_ffi_output_layouts_erase_all_output_copies() -> None:
    program, default_generated, hlo = _program_generated_and_hlo()
    default_plan = plan_streaming_attention_backward_hlo_replacement(hlo, program, default_generated)
    schedule = derive_streaming_attention_backward_tile_schedule(
        program,
        query_tile_size=8,
        key_value_tile_size=8,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )

    generated = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name="shuttle.streaming_reverse.physical_outputs",
        output_layouts=derive_streaming_attention_backward_ffi_output_layouts(default_plan),
    )
    plan = plan_streaming_attention_backward_hlo_replacement(hlo, program, generated)
    rewritten = replace_streaming_attention_backward_entry_with_custom_call(
        hlo,
        plan,
        target=generated.target_name,
    )
    entry = parse_hlo_module_text(rewritten).computation(parse_hlo_module_text(rewritten).entry)
    shuttle_instructions = tuple(
        instruction for instruction in entry.instructions if instruction.name.startswith("shuttle.")
    )

    assert all(value.physical_shape == value.ffi_shape for value in plan.outputs)
    assert not any(instruction.opcode == "copy" for instruction in shuttle_instructions)
    assert tuple(entry.root.operands) == (
        "shuttle.query_cotangent.ffi",
        "shuttle.key_cotangent.ffi",
        "shuttle.value_cotangent.ffi",
    )
    assert tuple(output.layout for output in generated.outputs) == (
        (3, 2, 1, 0),
        (3, 1, 2, 0),
        (1, 3, 2, 0),
    )
    assert tuple(output.strides for output in generated.outputs) == (
        (4096, 256, 64, 1),
        (2048, 64, 1024, 1),
        (2048, 1, 1024, 16),
    )
    assert tuple(output.jax_layout for output in generated.outputs) == (
        (0, 1, 2, 3),
        (0, 2, 1, 3),
        (0, 2, 3, 1),
    )
    dkdv = next(kernel for kernel in generated.aot_kernels if kernel.output_name == "shuttle_streaming_dkdv")
    assert dkdv.signature[31:35] == ("2048", "64", "1024", "1")
    assert dkdv.signature[35:39] == ("2048", "1", "1024", "16")


def test_physical_output_layout_mutation_changes_strides_without_semantic_dispatch() -> None:
    program, _, _ = _program_generated_and_hlo()
    schedule = derive_streaming_attention_backward_tile_schedule(
        program,
        query_tile_size=8,
        key_value_tile_size=8,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    layouts = (
        StreamingAttentionBackwardFfiBufferLayout("query_cotangent", (3, 2, 1, 0)),
        StreamingAttentionBackwardFfiBufferLayout("key_cotangent", (3, 2, 1, 0)),
        StreamingAttentionBackwardFfiBufferLayout("value_cotangent", (3, 1, 2, 0)),
    )

    generated = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name="shuttle.streaming_reverse.layout_mutation",
        output_layouts=layouts,
    )

    assert generated.outputs[1].strides == (2048, 128, 64, 1)
    assert generated.outputs[2].strides == (2048, 64, 1024, 1)


def test_invalid_or_incomplete_output_layouts_fail_closed() -> None:
    program, _, _ = _program_generated_and_hlo()
    schedule = derive_streaming_attention_backward_tile_schedule(
        program,
        query_tile_size=8,
        key_value_tile_size=8,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )

    with pytest.raises(ValueError, match="must cover exactly"):
        generate_streaming_attention_backward_ffi(
            program,
            schedule,
            target_name="shuttle.streaming_reverse.missing_layout",
            output_layouts=(StreamingAttentionBackwardFfiBufferLayout("query_cotangent", (3, 2, 1, 0)),),
        )
    with pytest.raises(ValueError, match="must be a permutation"):
        generate_streaming_attention_backward_ffi(
            program,
            schedule,
            target_name="shuttle.streaming_reverse.invalid_layout",
            output_layouts=(
                StreamingAttentionBackwardFfiBufferLayout("query_cotangent", (3, 2, 1, 0)),
                StreamingAttentionBackwardFfiBufferLayout("key_cotangent", (3, 1, 1, 0)),
                StreamingAttentionBackwardFfiBufferLayout("value_cotangent", (1, 3, 2, 0)),
            ),
        )


def test_hlo_output_without_explicit_layout_fails_closed() -> None:
    program, generated, hlo = _program_generated_and_hlo()
    plan = plan_streaming_attention_backward_hlo_replacement(hlo, program, generated)
    output = replace(plan.outputs[1], physical_shape="bf16[1,16,2,64]")

    with pytest.raises(ValueError, match="requires one explicit dense layout"):
        derive_streaming_attention_backward_ffi_output_layouts(
            replace(plan, outputs=(plan.outputs[0], output, plan.outputs[2]))
        )


def test_parameter_spelling_does_not_select_reverse_roles() -> None:
    program, generated, hlo = _program_generated_and_hlo()
    renamed = hlo
    for original, replacement in (
        ("query.1", "natural_parameter_2"),
        ("key.1", "natural_parameter_0"),
        ("value.1", "natural_parameter_3"),
        ("output_cotangent.1", "natural_parameter_1"),
    ):
        renamed = renamed.replace(original, replacement)

    plan = plan_streaming_attention_backward_hlo_replacement(renamed, program, generated)

    assert tuple(value.instruction for value in plan.inputs) == (
        "natural_parameter_2",
        "natural_parameter_0",
        "natural_parameter_3",
        "natural_parameter_1",
    )


def test_mismatched_recovered_score_scale_is_rejected_before_rewrite() -> None:
    _, _, hlo = _program_generated_and_hlo(0.125)
    mutated_program, mutated_generated, _ = _program_generated_and_hlo(0.25)

    with pytest.raises(ValueError, match=r"physical score scale .* does not match recovered scale"):
        plan_streaming_attention_backward_hlo_replacement(hlo, mutated_program, mutated_generated)


def test_saved_state_policy_cannot_replace_natural_recompute_vjp_boundary() -> None:
    program, _, hlo = _program_generated_and_hlo()
    schedule = derive_streaming_attention_backward_tile_schedule(
        program,
        query_tile_size=8,
        key_value_tile_size=8,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    generated = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name="shuttle.streaming_reverse.saved_hlo",
        state_policy=StreamingAttentionBackwardStatePolicy.SAVED_OUTPUT_AND_LOG_SUM_EXP,
    )

    with pytest.raises(ValueError, match="requires explicit recompute state policy"):
        plan_streaming_attention_backward_hlo_replacement(hlo, program, generated)


def test_natural_grug_gpu_hlo_proves_one_entry_local_reverse_region() -> None:
    program, generated, hlo = _grug_region_inputs()

    plan = plan_streaming_attention_backward_hlo_region_replacement(hlo, program, generated)

    assert tuple(value.role for value in plan.inputs) == (
        StreamingReverseHloRole.QUERY,
        StreamingReverseHloRole.KEY,
        StreamingReverseHloRole.VALUE,
        StreamingReverseHloRole.OUTPUT_COTANGENT,
    )
    assert tuple(value.role for value in plan.outputs) == (
        StreamingReverseHloRole.QUERY_COTANGENT,
        StreamingReverseHloRole.KEY_COTANGENT,
        StreamingReverseHloRole.VALUE_COTANGENT,
    )
    assert plan.insertion_instruction == plan.inputs[-1].instruction
    assert len(plan.internal_instructions) == 20
    assert len(plan.provenance.reverse_contracts) == 4
    assert len(plan.provenance.additive_folds) >= 2
    assert plan.provenance.domain_restriction is not None
    assert plan.provenance.score_scale == pytest.approx(0.32421875, rel=5e-4)
    assert all(users for _, users in plan.external_users)
    assert not ({value.instruction for value in plan.inputs} & set(plan.internal_instructions))


def test_natural_grug_output_layout_restores_singleton_elided_value_axis() -> None:
    program, generated, hlo = _grug_region_inputs()
    plan = plan_streaming_attention_backward_hlo_region_replacement(hlo, program, generated)

    layouts = derive_streaming_attention_backward_ffi_output_layouts(plan)

    assert tuple(layout.minor_to_major for layout in layouts) == (
        (3, 2, 1, 0),
        (3, 2, 1, 0),
        (1, 3, 2, 0),
    )
    schedule = derive_streaming_attention_backward_tile_schedule(
        program,
        query_tile_size=4,
        key_value_tile_size=4,
        domain_traversal=StreamingAttentionBackwardDomainTraversal.LOWER_TRIANGULAR,
    )
    regenerated = generate_streaming_attention_backward_ffi(
        program,
        schedule,
        target_name="shuttle.streaming_reverse.grug_singleton_layout",
        output_layouts=layouts,
    )
    regenerated_plan = plan_streaming_attention_backward_hlo_region_replacement(hlo, program, regenerated)
    rewritten = replace_streaming_attention_backward_region_with_custom_call(
        hlo,
        regenerated_plan,
        target=regenerated.target_name,
    )
    entry = parse_hlo_module_text(rewritten).computation("main.165")
    shuttle_instructions = tuple(
        instruction for instruction in entry.instructions if instruction.name.startswith("shuttle.")
    )

    assert not any(instruction.opcode == "copy" for instruction in shuttle_instructions)


def test_natural_grug_region_rewrite_redirects_only_cotangent_users() -> None:
    program, generated, hlo = _grug_region_inputs()
    plan = plan_streaming_attention_backward_hlo_region_replacement(hlo, program, generated)

    rewritten = replace_streaming_attention_backward_region_with_custom_call(
        hlo,
        plan,
        target=generated.target_name,
    )
    entry = parse_hlo_module_text(rewritten).computation("main.165")
    audit = audit_streaming_attention_backward_region_replacement(
        hlo,
        rewritten,
        plan,
        target=generated.target_name,
    )
    calls = tuple(
        instruction
        for instruction in entry.instructions
        if instruction.opcode == "custom-call" and generated.target_name in instruction.attributes
    )
    instructions = {instruction.name: instruction for instruction in entry.instructions}

    assert len(calls) == 1
    assert set(audit.dead_reverse_closure) == set(plan.internal_instructions) | {
        value.instruction for value in plan.outputs
    }
    assert audit.preserved_shared_users
    assert calls[0].operands[-1] == plan.inputs[-1].instruction
    assert instructions["shuttle.region.value.canonical"].opcode == "reshape"
    assert instructions["shuttle.region.value_cotangent.physical"].opcode == "reshape"
    assert sum(instruction.opcode == "transpose" for instruction in entry.instructions) == hlo.count(" transpose(")
    for old_output, users in plan.external_users:
        assert all(old_output not in instructions[user].operands for user in users)
    assert all(name in instructions for name in plan.internal_instructions)


def test_outer_contract_renaming_does_not_change_local_reverse_proof() -> None:
    program, generated, hlo = _grug_region_inputs()
    baseline = plan_streaming_attention_backward_hlo_region_replacement(hlo, program, generated)
    mutated = hlo.replace("%dot.89", "%outer.contract.changed")

    changed = plan_streaming_attention_backward_hlo_region_replacement(mutated, program, generated)

    assert tuple(value.role for value in changed.inputs) == tuple(value.role for value in baseline.inputs)
    assert changed.provenance == baseline.provenance
    assert changed.internal_instructions == baseline.internal_instructions


def test_local_reverse_rejects_an_internal_value_with_an_extra_external_user() -> None:
    program, generated, hlo = _grug_region_inputs()
    plan = plan_streaming_attention_backward_hlo_region_replacement(hlo, program, generated)
    internal = plan.internal_instructions[0]
    instruction = parse_hlo_module_text(hlo).computation("main.165")
    shape = next(value.shape for value in instruction.instructions if value.name == internal)
    root_pattern = re.compile(r"^(?P<indent>\s*)ROOT %tuple\.27 = ", re.MULTILINE)
    root = root_pattern.search(hlo)
    assert root is not None
    mutated = hlo[: root.start()] + f"  %extra.reverse.user = {shape} copy(%{internal})\n" + hlo[root.start() :]

    with pytest.raises(ValueError, match="has external users"):
        plan_streaming_attention_backward_hlo_region_replacement(mutated, program, generated)


def test_local_reverse_rejects_a_cross_region_control_dependency() -> None:
    program, generated, hlo = _grug_region_inputs()
    plan = plan_streaming_attention_backward_hlo_region_replacement(hlo, program, generated)
    _, users = plan.external_users[0]
    user = users[0]
    line_pattern = re.compile(rf"^(?P<line>\s*%?{re.escape(user)} = .*?)$", re.MULTILINE)
    match = line_pattern.search(hlo)
    assert match is not None
    mutated_line = match.group("line") + f", control-predecessors={{%{plan.internal_instructions[0]}}}"
    mutated = hlo[: match.start()] + mutated_line + hlo[match.end() :]

    with pytest.raises(ValueError, match="crosses an explicit control dependency"):
        plan_streaming_attention_backward_hlo_region_replacement(mutated, program, generated)


def test_local_reverse_preserves_a_literal_zero_score_scale() -> None:
    program, generated, hlo = _grug_region_inputs(0.0)
    original = "%constant.331 = bf16[] constant(0.3242)"
    assert hlo.count(original) == 1
    mutated = hlo.replace(original, "%constant.331 = bf16[] constant(0)", 1)

    plan = plan_streaming_attention_backward_hlo_region_replacement(mutated, program, generated)

    assert plan.provenance.score_scale == 0.0


def test_local_reverse_rejects_ambiguous_same_depth_output_boundaries() -> None:
    program, generated, hlo = _grug_region_inputs()
    definition = "  %dot.41 = bf16[2,2,4,4]{3,2,1,0} dot(%transpose.113, %transpose.95)"
    assert hlo.count(definition) == 1
    ambiguous = "\n".join(
        (
            "  %ambiguous.output.copy.0 = bf16[2,4,2,16]{3,2,1,0} copy(%add_any.96)",
            "  %ambiguous.output.copy.1 = bf16[2,4,2,16]{3,2,1,0} copy(%add_any.96)",
            "  %ambiguous.output.transpose.0 = bf16[2,2,4,16]{2,1,3,0} "
            "transpose(%ambiguous.output.copy.0), dimensions={0,2,1,3}",
            "  %ambiguous.output.transpose.1 = bf16[2,2,4,16]{2,1,3,0} "
            "transpose(%ambiguous.output.copy.1), dimensions={0,2,1,3}",
            "  %ambiguous.output.path = bf16[2,2,4,16]{2,1,3,0} "
            "multiply(%ambiguous.output.transpose.0, %ambiguous.output.transpose.1)",
            definition.replace("%transpose.113", "%ambiguous.output.path"),
        )
    )
    mutated = hlo.replace(definition, ambiguous, 1)

    with pytest.raises(ValueError, match="ambiguous compatible ancestors at depth"):
        plan_streaming_attention_backward_hlo_region_replacement(mutated, program, generated)


def test_older_cpu_grug_hlo_fails_closed_when_reverse_boundaries_are_fused() -> None:
    program, generated, _ = _grug_region_inputs()
    hlo = gzip.decompress(_GRUG_CPU_HLO.read_bytes()).decode()

    with pytest.raises(ValueError, match="expected one region-local streaming reverse candidate, found 0"):
        plan_streaming_attention_backward_hlo_region_replacement(hlo, program, generated)


def test_natural_grug_combined_plan_rewrites_routed_and_attention_regions() -> None:
    program, generated, hlo = _grug_region_inputs()
    plan = plan_routed_training_and_attention_typed_ffi(hlo, program, generated)

    rewritten = replace_routed_training_and_attention_regions_with_custom_calls(
        hlo,
        plan,
        targets=_COMBINED_TARGETS,
    )
    audit = audit_routed_training_and_attention_replacement(
        hlo,
        rewritten,
        plan,
        targets=_COMBINED_TARGETS,
    )

    assert audit.attention_backward_instruction == "shuttle.generated.streaming_reverse.region"
    assert len(audit.routed.target_instructions) == 4
    assert rewritten.count('custom_call_target="shuttle.combined.') == 5
    parse_hlo_module_text(rewritten)


def test_natural_grug_combined_plan_adds_generic_axis_fold_regions() -> None:
    program, generated_attention, hlo = _grug_region_inputs()
    fold_report = recover_axis_fold_hlo_region_candidates(
        hlo,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    selected_folds = tuple(
        plan
        for plan in fold_report.plans
        if plan.program.rows == 8 and plan.program.columns == 32 and plan.output_ffi_shape.startswith("bf16[")
    )
    assert len(selected_folds) == 2
    generated_folds = tuple(
        generate_cuda_axis_fold_ffi((fold.program,), target_name=target)
        for fold, target in zip(selected_folds, _SIX_TARGETS.axis_folds, strict=True)
    )
    plan = plan_routed_training_attention_and_axis_fold_typed_ffi(
        hlo,
        program,
        generated_attention,
        generated_folds,
        axis_fold_numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )

    rewritten = replace_routed_training_attention_and_axis_fold_regions_with_custom_calls(
        hlo,
        plan,
        targets=_SIX_TARGETS,
    )
    audit = audit_routed_training_attention_and_axis_fold_replacement(
        hlo,
        rewritten,
        plan,
        targets=_SIX_TARGETS,
    )

    assert len(audit.routed_attention.routed.target_instructions) == 4
    assert audit.routed_attention.attention_backward_instruction == "shuttle.generated.streaming_reverse.region"
    assert len({value.call_instruction for value in audit.axis_folds}) == 2
    assert all(value.call_instruction.startswith("shuttle.generated.axis_fold.region.") for value in audit.axis_folds)
    assert rewritten.count('custom_call_target="shuttle.combined.') == 7
    selected_targets = (
        _SIX_TARGETS.routed_attention.routed.forward,
        _SIX_TARGETS.routed_attention.routed.input_adjoint,
        *_SIX_TARGETS.routed_attention.routed.weight_gradients,
        _SIX_TARGETS.routed_attention.attention_backward,
        *_SIX_TARGETS.axis_folds,
    )
    expected_occurrences = dict.fromkeys(selected_targets, 1)
    exact_occurrences = _custom_call_target_occurrences(rewritten, expected_occurrences)
    assert set(exact_occurrences.values()) == {1}
    assert all(rewritten.count(target) > exact_occurrences[target] for target in _SIX_TARGETS.axis_folds)

    axis_target = _SIX_TARGETS.axis_folds[0]
    axis_call_line = next(line for line in rewritten.splitlines() if f'custom_call_target="{axis_target}"' in line)
    missing_attribute_line = axis_call_line.replace(f', custom_call_target="{axis_target}"', "", 1)
    with pytest.raises(RuntimeError, match="has 0 exact custom_call_target attributes"):
        _custom_call_target_occurrences(
            rewritten.replace(axis_call_line, missing_attribute_line, 1),
            expected_occurrences,
        )
    duplicate_attribute_line = axis_call_line.replace(
        f'custom_call_target="{axis_target}"',
        f'custom_call_target="{axis_target}", custom_call_target="{axis_target}"',
        1,
    )
    with pytest.raises(RuntimeError, match="has 2 exact custom_call_target attributes"):
        _custom_call_target_occurrences(
            rewritten.replace(axis_call_line, duplicate_attribute_line, 1),
            expected_occurrences,
        )
    parse_hlo_module_text(rewritten)


def test_combined_roundtrip_audit_accepts_xla_text_canonicalization() -> None:
    program, generated_attention, hlo = _grug_region_inputs()
    fold_report = recover_axis_fold_hlo_region_candidates(
        hlo,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    selected_folds = tuple(
        plan
        for plan in fold_report.plans
        if plan.program.rows == 8 and plan.program.columns == 32 and plan.output_ffi_shape.startswith("bf16[")
    )
    generated_folds = tuple(
        generate_cuda_axis_fold_ffi((fold.program,), target_name=target)
        for fold, target in zip(selected_folds, _SIX_TARGETS.axis_folds, strict=True)
    )
    plan = plan_routed_training_attention_and_axis_fold_typed_ffi(
        hlo,
        program,
        generated_attention,
        generated_folds,
        axis_fold_numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    rewritten = replace_routed_training_attention_and_axis_fold_regions_with_custom_calls(
        hlo,
        plan,
        targets=_SIX_TARGETS,
    )

    computations = parse_hlo_module_text(rewritten).computations
    first_name, second_name = computations[0].name, computations[1].name

    def computation_span(text: str, name: str) -> tuple[int, int]:
        header = re.search(rf"(?m)^(?:ENTRY )?%?{re.escape(name)}.*\{{$", text)
        assert header is not None
        end = text.find("\n}\n", header.start())
        assert end >= 0
        return header.start(), end + len("\n}\n")

    first_start, first_end = computation_span(rewritten, first_name)
    second_start, second_end = computation_span(rewritten, second_name)
    assert first_end <= second_start
    canonicalized = (
        rewritten[:first_start]
        + rewritten[second_start:second_end]
        + rewritten[first_end:second_start]
        + rewritten[first_start:first_end]
        + rewritten[second_end:]
    )
    canonicalized = re.sub(
        r"stack_frame_id=(\d+)",
        lambda match: f"stack_frame_id={int(match.group(1)) + 1000}",
        canonicalized,
    )
    assert "constant(-0)" in canonicalized
    canonicalized = canonicalized.replace("constant(-0)", "constant(0)", 1)
    axis_call_line = next(
        line for line in canonicalized.splitlines() if f'custom_call_target="{_SIX_TARGETS.axis_folds[0]}"' in line
    )
    commented_axis_call = axis_call_line.replace(", %", ", /*index=1*/%", 1)
    assert commented_axis_call != axis_call_line
    canonicalized = canonicalized.replace(axis_call_line, commented_axis_call, 1)

    audit = audit_routed_training_attention_and_axis_fold_replacement(
        hlo,
        canonicalized,
        plan,
        targets=_SIX_TARGETS,
    )

    assert len(audit.axis_folds) == 2
    assert any(value.transpose_count[1] < value.transpose_count[0] for value in audit.axis_folds)


def test_seven_call_harness_generates_two_self_contained_axis_fold_targets() -> None:
    _, _, hlo = _grug_region_inputs()

    generated = _generate_axis_fold_programs(hlo)

    assert tuple(value.target_name for value in generated) == _SEVEN_CALL_TARGETS.axis_folds
    assert all(len(value.outputs) == 1 for value in generated)
    assert all("shuttle_axis_fold_ffi_call_count" in value.source for value in generated)
    assert all("torch" not in value.source.lower() and "triton" not in value.source.lower() for value in generated)


def test_routed_and_attention_replacements_are_order_independent() -> None:
    program, generated, hlo = _grug_region_inputs()
    plan = plan_routed_training_and_attention_typed_ffi(hlo, program, generated)
    attention_first = replace_routed_training_and_attention_regions_with_custom_calls(
        hlo,
        plan,
        targets=_COMBINED_TARGETS,
    )
    routed_first = replace_routed_training_regions_with_custom_calls(
        hlo,
        plan.routed,
        targets=_COMBINED_TARGETS.routed,
    )
    routed_first = replace_streaming_attention_backward_region_with_custom_call(
        routed_first,
        plan.attention_backward,
        target=_COMBINED_TARGETS.attention_backward,
    )

    first_audit = audit_routed_training_and_attention_replacement(
        hlo,
        attention_first,
        plan,
        targets=_COMBINED_TARGETS,
    )
    second_audit = audit_routed_training_and_attention_replacement(
        hlo,
        routed_first,
        plan,
        targets=_COMBINED_TARGETS,
    )

    assert first_audit == second_audit
    assert attention_first.count('custom_call_target="shuttle.combined.') == 5
    assert routed_first.count('custom_call_target="shuttle.combined.') == 5
