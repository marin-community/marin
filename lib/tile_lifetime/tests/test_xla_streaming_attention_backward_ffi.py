# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import functools

import jax
import jax.numpy as jnp
import pytest

from tile_lifetime.jax_streaming_attention_backward_ffi import (
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
from tile_lifetime.xla_streaming_attention_backward_ffi import (
    StreamingReverseHloRole,
    plan_streaming_attention_backward_hlo_replacement,
    replace_streaming_attention_backward_entry_with_custom_call,
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
