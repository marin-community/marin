# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import functools
import gzip
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from tile_lifetime.ir import DType
from tile_lifetime.jax_streaming_attention_backward_ffi import (
    GeneratedStreamingAttentionBackwardFfi,
    StreamingAttentionBackwardFfiBuffer,
    StreamingAttentionBackwardStatePolicy,
    generate_streaming_attention_backward_ffi,
)
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.stablehlo_import import import_stablehlo
from tile_lifetime.stablehlo_streaming_attention_backward import recover_stablehlo_streaming_attention_backward
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
from tile_lifetime.xla_hlo_recovery import parse_hlo_module_text
from tile_lifetime.xla_routed_training_ffi import (
    RoutedTrainingAndAttentionFfiTargets,
    RoutedTrainingFfiTargets,
    audit_routed_training_and_attention_replacement,
    plan_routed_training_and_attention_typed_ffi,
    replace_routed_training_and_attention_regions_with_custom_calls,
    replace_routed_training_regions_with_custom_calls,
)
from tile_lifetime.xla_streaming_attention_backward_ffi import (
    StreamingReverseHloRole,
    audit_streaming_attention_backward_region_replacement,
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
    recovered = recover_stablehlo_streaming_attention_backward(
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
    recovered = recover_stablehlo_streaming_attention_backward(
        graph,
        schedule=StreamingTileSchedule(query_tile_size=4, key_value_tile_size=4, pipeline_depth=2),
    )
    program = eliminate_normalized_exp_maximum_vjp(
        recovered.program,
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
    )
    query_shape = (2, 4, 2, 16)
    key_value_shape = (2, 4, 1, 16)

    def buffer(name: str, shape: tuple[int, ...]) -> StreamingAttentionBackwardFfiBuffer:
        return StreamingAttentionBackwardFfiBuffer(name, DType.BF16, shape)

    # The preserved natural Grug fixture uses D16, below the current AOT
    # backend's D64/D128 gate. This is a real typed-buffer boundary used only
    # for CPU HLO planning; the test does not claim or simulate GPU execution.
    generated = GeneratedStreamingAttentionBackwardFfi(
        target_name="shuttle.streaming_reverse.grug_region_test",
        handler_symbol="shuttle_streaming_reverse_grug_region_test",
        state_policy=StreamingAttentionBackwardStatePolicy.RECOMPUTE,
        inputs=(
            buffer("query", query_shape),
            buffer("key", key_value_shape),
            buffer("value", key_value_shape),
            buffer("output_cotangent", query_shape),
        ),
        outputs=(
            buffer("query_cotangent", query_shape),
            buffer("key_cotangent", key_value_shape),
            buffer("value_cotangent", key_value_shape),
        ),
        aot_kernels=(),
        handler_template="",
        semantic_fingerprint=f"grug-region-structural-test-{scale}",
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


def test_rewrite_preserves_entry_layouts_around_canonical_ffi_buffers() -> None:
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
        "shuttle.query_cotangent.canonical",
        "shuttle.key_cotangent.physical",
        "shuttle.value_cotangent.physical",
    )
    assert instructions["shuttle.query_cotangent.canonical"].shape == plan.outputs[0].canonical_shape
    assert instructions["shuttle.key_cotangent.physical"].shape == plan.outputs[1].physical_shape
    assert instructions["shuttle.value_cotangent.physical"].shape == plan.outputs[2].physical_shape


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
