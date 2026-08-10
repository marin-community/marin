# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan a legal early-forward/later-reverse attention ownership split.

The split is derived from ordinary JAX HLO.  The early region emits the natural
output and a log-normalizer coordinate of the generic normalized-exponential
Fold state.  The later region consumes that state after JAX has produced the
output cotangent.  No call spans the intervening train-step dataflow.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from enum import StrEnum

from tile_lifetime.ir import DType
from tile_lifetime.jax_streaming_attention_backward_ffi import (
    GeneratedStreamingAttentionBackwardFfi,
    StreamingAttentionBackwardResultPolicy,
    StreamingAttentionBackwardStatePolicy,
)
from tile_lifetime.jax_streaming_attention_forward_ffi import GeneratedStreamingAttentionForwardFfi
from tile_lifetime.streaming_attention_backward import StreamingAttentionBackwardProgram
from tile_lifetime.xla_hlo_recovery import HloInstruction, parse_hlo_module_text
from tile_lifetime.xla_streaming_attention_backward_ffi import (
    StreamingReverseHloRegionReplacementPlan,
    StreamingReverseHloRole,
    _bind_region_score_inputs,
    _boundary_adapter_opcode,
    _closest_compatible_ancestor,
    _emit_boundary_adapter,
    _entry_ancestor_slice,
    _entry_ancestors,
    _entry_descendants,
    _entry_fold_reducer,
    _entry_users,
    _first_entry_opcode_users,
    _nearest_entry_opcode_ancestor_function,
    _replace_entry_operand,
    _shape_signature,
    _unique_compatible_descendant,
    _validate_region_effect_and_control_safety,
    plan_streaming_attention_backward_hlo_region_replacement,
)

_FORWARD_REGION_OPCODES = frozenset(
    {
        "add",
        "broadcast",
        "compare",
        "constant",
        "convert",
        "copy",
        "divide",
        "dot",
        "exponential",
        "multiply",
        "negate",
        "reduce",
        "reshape",
        "select",
        "subtract",
        "transpose",
    }
)
_COLLECTIVE_OPCODES = frozenset({"all-gather", "all-reduce", "all-to-all", "collective-permute", "reduce-scatter"})


class StreamingForwardHloRole(StrEnum):
    """Generic roles at the early attention boundary."""

    QUERY = "query"
    KEY = "key"
    VALUE = "value"
    OUTPUT = "output"
    LOG_SUM_EXP = "log_sum_exp"


@dataclass(frozen=True)
class StreamingForwardHloValue:
    """One recovered or introduced value in the early typed-FFI boundary."""

    role: StreamingForwardHloRole
    instruction: str
    physical_shape: str
    ffi_shape: str


@dataclass(frozen=True)
class StreamingForwardHloProvenance:
    """Generic algebra proving one forward region."""

    score_contract: str
    value_contract: str
    maximum_fold: str
    sum_fold: str
    domain_restriction: str | None
    score_scale: float


@dataclass(frozen=True)
class StreamingForwardHloRegionReplacementPlan:
    """An early output region that may legally publish saved Fold state."""

    inputs: tuple[StreamingForwardHloValue, ...]
    output: StreamingForwardHloValue
    saved_state: StreamingForwardHloValue
    insertion_instruction: str
    internal_instructions: tuple[str, ...]
    external_output_users: tuple[str, ...]
    provenance: StreamingForwardHloProvenance
    semantic_fingerprint: str


@dataclass(frozen=True)
class StreamingAttentionTrainingRegionPlan:
    """Two generated calls separated by ordinary JAX train-step dataflow."""

    forward: StreamingForwardHloRegionReplacementPlan
    rematerialized_forward: StreamingForwardHloRegionReplacementPlan
    reverse: StreamingReverseHloRegionReplacementPlan
    forward_target: str
    reverse_target: str
    saved_state_policy: StreamingAttentionBackwardStatePolicy
    collectives_inside_regions: tuple[str, ...]
    numerical_contract: str


@dataclass(frozen=True)
class StreamingAttentionTrainingRegionAudit:
    """Post-rewrite liveness and state-link evidence."""

    forward_call: str
    reverse_call: str
    saved_state_producer: str
    reverse_saved_state_operands: tuple[str, str]
    dead_forward_closure: tuple[str, ...]
    dead_rematerialized_forward_closure: tuple[str, ...]
    dead_reverse_closure: tuple[str, ...]
    external_collectives: tuple[str, ...]


def plan_streaming_attention_training_regions(
    hlo_text: str,
    program: StreamingAttentionBackwardProgram,
    generated_forward: GeneratedStreamingAttentionForwardFfi,
    generated_saved_reverse: GeneratedStreamingAttentionBackwardFfi,
) -> StreamingAttentionTrainingRegionPlan:
    """Derive an early O/LSE call and later saved-state reverse call."""
    _validate_pair_signatures(generated_forward, generated_saved_reverse)
    proof_reverse = _recompute_proof_boundary(generated_saved_reverse)
    reverse = plan_streaming_attention_backward_hlo_region_replacement(hlo_text, program, proof_reverse)
    forward = _plan_forward_region(
        hlo_text,
        generated_forward,
        reverse=reverse,
    )
    rematerialized_forward = _plan_rematerialized_forward_region(
        hlo_text,
        generated_forward,
        reverse=reverse,
    )
    if not math.isclose(
        forward.provenance.score_scale,
        reverse.provenance.score_scale,
        rel_tol=5e-4,
        abs_tol=1e-12,
    ):
        raise ValueError("forward and reverse regions implement different score Maps")
    if generated_forward.reverse_state_policy is not generated_saved_reverse.state_policy:
        raise ValueError("forward saved-state coordinate does not match the reverse state policy")
    if not math.isclose(
        forward.provenance.score_scale,
        rematerialized_forward.provenance.score_scale,
        rel_tol=5e-4,
        abs_tol=1e-12,
    ):
        raise ValueError("early and rematerialized forward regions implement different score Maps")

    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    owned = (
        set(forward.internal_instructions)
        | set(rematerialized_forward.internal_instructions)
        | set(reverse.internal_instructions)
    )
    collectives = tuple(
        instruction.name
        for instruction in entry.instructions
        if instruction.name in owned and instruction.opcode in _COLLECTIVE_OPCODES
    )
    if collectives:
        raise ValueError(f"attention regions cross placement collectives: {collectives}")
    return StreamingAttentionTrainingRegionPlan(
        forward=forward,
        rematerialized_forward=rematerialized_forward,
        reverse=reverse,
        forward_target=generated_forward.target_name,
        reverse_target=generated_saved_reverse.target_name,
        saved_state_policy=generated_saved_reverse.state_policy,
        collectives_inside_regions=collectives,
        numerical_contract="bf16_contract_boundaries_fp32_online_fold_allow_rounding_reorder",
    )


def replace_streaming_attention_training_regions_with_custom_calls(
    hlo_text: str,
    plan: StreamingAttentionTrainingRegionPlan,
) -> str:
    """Insert two calls without spanning the output-cotangent readiness gap."""
    transformed = _replace_forward_region(
        hlo_text,
        plan.forward,
        rematerialized=plan.rematerialized_forward,
        target=plan.forward_target,
    )
    transformed = _replace_saved_reverse_region(transformed, plan.reverse, target=plan.reverse_target)
    parse_hlo_module_text(transformed)
    return transformed


def audit_streaming_attention_training_region_replacement(
    original_hlo: str,
    transformed_hlo: str,
    plan: StreamingAttentionTrainingRegionPlan,
) -> StreamingAttentionTrainingRegionAudit:
    """Verify state linkage, old-region liveness, and collective exclusion."""
    original = parse_hlo_module_text(original_hlo).computation(parse_hlo_module_text(original_hlo).entry)
    transformed = parse_hlo_module_text(transformed_hlo).computation(parse_hlo_module_text(transformed_hlo).entry)
    original_users = _entry_users(original.instructions)
    transformed_users = _entry_users(transformed.instructions)
    transformed_instructions = {instruction.name: instruction for instruction in transformed.instructions}
    live = _entry_ancestors(transformed.root.name, transformed_instructions)
    forward_calls = tuple(
        instruction
        for instruction in transformed.instructions
        if instruction.opcode == "custom-call" and plan.forward_target in instruction.attributes
    )
    reverse_calls = tuple(
        instruction
        for instruction in transformed.instructions
        if instruction.opcode == "custom-call" and plan.reverse_target in instruction.attributes
    )
    if len(forward_calls) != 1 or len(reverse_calls) != 1:
        raise ValueError("expected exactly one early forward and one later reverse call")
    forward_call = forward_calls[0]
    reverse_call = reverse_calls[0]
    source_order = {instruction.name: index for index, instruction in enumerate(transformed.instructions)}
    if source_order[forward_call.name] >= source_order[reverse_call.name]:
        raise ValueError("saved-state producer does not precede its reverse consumer")
    expected_saved_operands = (
        "shuttle.streaming_forward.output.ffi",
        "shuttle.streaming_forward.log_sum_exp.ffi",
    )
    if not all(name in reverse_call.operands for name in expected_saved_operands):
        raise ValueError("later reverse does not consume both generated forward-state coordinates")

    forward_closure = set(plan.forward.internal_instructions) | {plan.forward.output.instruction}
    for name in forward_closure:
        crossing = tuple(user for user in transformed_users[name] if user not in forward_closure)
        if crossing:
            raise ValueError(f"old forward value %{name} remains externally live through {crossing}")
    reverse_closure = set(plan.reverse.internal_instructions) | {value.instruction for value in plan.reverse.outputs}
    for name in reverse_closure:
        crossing = tuple(user for user in transformed_users[name] if user not in reverse_closure)
        if crossing:
            raise ValueError(f"old reverse value %{name} remains externally live through {crossing}")
    rematerialized_closure = set(plan.rematerialized_forward.internal_instructions) | {
        plan.rematerialized_forward.output.instruction
    }
    stale_rematerialized = tuple(name for name in rematerialized_closure if name in live)
    if stale_rematerialized:
        raise ValueError(f"old rematerialized forward remains root-live: {stale_rematerialized}")

    external_collectives = tuple(
        instruction.name
        for instruction in transformed.instructions
        if instruction.opcode in _COLLECTIVE_OPCODES
        and instruction.name not in forward_closure
        and instruction.name not in reverse_closure
    )
    if any(name not in original_users for name in external_collectives):
        raise ValueError("replacement introduced a collective")
    return StreamingAttentionTrainingRegionAudit(
        forward_call=forward_call.name,
        reverse_call=reverse_call.name,
        saved_state_producer="shuttle.streaming_forward.log_sum_exp.ffi",
        reverse_saved_state_operands=expected_saved_operands,
        dead_forward_closure=tuple((*plan.forward.internal_instructions, plan.forward.output.instruction)),
        dead_rematerialized_forward_closure=tuple(
            (*plan.rematerialized_forward.internal_instructions, plan.rematerialized_forward.output.instruction)
        ),
        dead_reverse_closure=tuple(
            (*plan.reverse.internal_instructions, *(value.instruction for value in plan.reverse.outputs))
        ),
        external_collectives=external_collectives,
    )


def _plan_forward_region(
    hlo_text: str,
    generated: GeneratedStreamingAttentionForwardFfi,
    *,
    reverse: StreamingReverseHloRegionReplacementPlan,
) -> StreamingForwardHloRegionReplacementPlan:
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    users = _entry_users(entry.instructions)
    source_order = {instruction.name: index for index, instruction in enumerate(entry.instructions)}
    specifications = {value.name: value for value in (*generated.inputs, *generated.outputs)}
    reverse_start = source_order[reverse.insertion_instruction]
    candidates: list[StreamingForwardHloRegionReplacementPlan] = []
    errors: list[str] = []
    for maximum in entry.instructions:
        if maximum.opcode != "reduce" or _entry_fold_reducer(module, maximum) != "maximum":
            continue
        if source_order[maximum.name] >= reverse_start or maximum.name == reverse.provenance.maximum_fold:
            continue
        try:
            candidates.append(
                _plan_forward_from_maximum(
                    entry.instructions,
                    instructions,
                    users,
                    source_order,
                    module,
                    maximum,
                    specifications,
                    generated=generated,
                )
            )
        except ValueError as error:
            errors.append(f"%{maximum.name}: {error}")
    if len(candidates) != 1:
        detail = "; ".join(errors)
        raise ValueError(
            f"expected one early streaming forward candidate, found {len(candidates)}"
            + (f" ({detail})" if detail else "")
        )
    return candidates[0]


def _plan_rematerialized_forward_region(
    hlo_text: str,
    generated: GeneratedStreamingAttentionForwardFfi,
    *,
    reverse: StreamingReverseHloRegionReplacementPlan,
) -> StreamingForwardHloRegionReplacementPlan:
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    users = _entry_users(entry.instructions)
    source_order = {instruction.name: index for index, instruction in enumerate(entry.instructions)}
    maximum = instructions[reverse.provenance.maximum_fold]
    specifications = {value.name: value for value in (*generated.inputs, *generated.outputs)}
    plan = _plan_forward_from_maximum(
        entry.instructions,
        instructions,
        users,
        source_order,
        module,
        maximum,
        specifications,
        generated=generated,
        # The rematerialized probability and Fold state feed JAX's expanded
        # reverse algebra outside the narrow cotangent closure.  The paired
        # saved-state reverse removes those users together, so they are legal
        # here and verified root-dead after the combined rewrite.
        allowed_external_users=frozenset(instructions),
    )
    if plan.provenance.score_contract != reverse.provenance.score_contract:
        raise ValueError("later reverse is not attached to the recovered rematerialized forward")
    return plan


def _plan_forward_from_maximum(
    ordered: tuple[HloInstruction, ...],
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
    source_order: dict[str, int],
    module,
    maximum: HloInstruction,
    specifications,
    *,
    generated: GeneratedStreamingAttentionForwardFfi,
    allowed_external_users: frozenset[str] = frozenset(),
) -> StreamingForwardHloRegionReplacementPlan:
    nearest = _nearest_entry_opcode_ancestor_function(instructions)
    score_contracts = nearest(maximum.operands[0], "dot")
    if len(score_contracts) != 1:
        raise ValueError("maximum Fold must have one nearest score Contract")
    score_contract = instructions[next(iter(score_contracts))]
    query_name, key_name, scale = _bind_region_score_inputs(
        score_contract,
        instructions,
        specifications["query"],
        specifications["key"],
    )
    score_slice = _entry_ancestor_slice(maximum.operands[0], instructions, stop={score_contract.name})
    restrictions = tuple(instructions[name] for name in score_slice if instructions[name].opcode == "select")
    comparisons = tuple(instructions[name] for name in score_slice if instructions[name].opcode == "compare")
    if len(restrictions) != 1 or len(comparisons) != 1 or "direction=LE" not in comparisons[0].attributes:
        raise ValueError("expected one less-equal DomainRestriction")

    exponentials = tuple(
        instructions[name]
        for name in _first_entry_opcode_users(maximum.name, instructions, users, opcode="exponential")
        if score_contract.name in _entry_ancestors(name, instructions)
    )
    if len(exponentials) != 1:
        raise ValueError("expected one normalized-exponential Map")
    exponential = exponentials[0]
    probabilities = tuple(
        instruction
        for instruction in ordered
        if instruction.opcode == "divide"
        and _shape_signature(instruction.shape)[1] == _shape_signature(score_contract.shape)[1]
        and exponential.name in _entry_ancestors(instruction.name, instructions)
        and nearest(instruction.name, "dot") == frozenset({score_contract.name})
    )
    if len(probabilities) != 1:
        raise ValueError("expected one normalized probability value")
    probability = probabilities[0]
    sum_folds = tuple(
        instruction
        for instruction in ordered
        if instruction.opcode == "reduce"
        and _entry_fold_reducer(module, instruction) == "add"
        and exponential.name in _entry_ancestors(instruction.name, instructions)
        and instruction.name in _entry_ancestors(probability.name, instructions)
    )
    if len(sum_folds) != 1:
        raise ValueError("expected one normalized-exponential sum Fold")

    value_contracts: list[tuple[HloInstruction, str]] = []
    for name in _first_entry_opcode_users(probability.name, instructions, users, opcode="dot"):
        contract = instructions[name]
        for probability_operand, value_operand in (contract.operands, tuple(reversed(contract.operands))):
            if probability.name not in _entry_ancestors(probability_operand, instructions):
                continue
            value_name = _closest_compatible_ancestor(
                value_operand,
                instructions,
                specifications["value"],
                allow_singleton_elision=True,
            )
            if value_name is not None:
                value_contracts.append((contract, value_name))
    if len(value_contracts) != 1:
        raise ValueError(f"expected one probability/value Contract, found {len(value_contracts)}")
    value_contract, value_name = value_contracts[0]
    output_name = _unique_compatible_descendant(
        value_contract.name,
        instructions,
        users,
        specifications["output"],
        allow_singleton_elision=False,
    )
    output_ancestors = _entry_ancestors(output_name, instructions)
    score_descendants = _entry_descendants(score_contract.name, users)
    internal = (output_ancestors & score_descendants) - {output_name}
    if score_contract.name not in internal or value_contract.name not in internal:
        raise ValueError("forward closure omits a required Contract")
    unsupported = tuple(name for name in internal if instructions[name].opcode not in _FORWARD_REGION_OPCODES)
    if unsupported:
        raise ValueError(f"forward closure contains unsupported operations: {sorted(unsupported)}")
    _validate_region_effect_and_control_safety(internal | {output_name}, instructions)
    for name in sorted(internal, key=source_order.__getitem__):
        crossing = tuple(
            user
            for user in users[name]
            if user not in internal and user != output_name and user not in allowed_external_users
        )
        if crossing:
            raise ValueError(f"forward internal value %{name} has external users {crossing}")
    external_users = tuple(user for user in users[output_name] if user not in internal)
    if not external_users:
        raise ValueError("forward output has no later train-step consumer")

    role_names = {
        StreamingForwardHloRole.QUERY: query_name,
        StreamingForwardHloRole.KEY: key_name,
        StreamingForwardHloRole.VALUE: value_name,
    }
    inputs = tuple(
        StreamingForwardHloValue(
            role=role,
            instruction=role_names[role],
            physical_shape=instructions[role_names[role]].shape,
            ffi_shape=_hlo_ffi_shape(specifications[role.value]),
        )
        for role in (StreamingForwardHloRole.QUERY, StreamingForwardHloRole.KEY, StreamingForwardHloRole.VALUE)
    )
    output = StreamingForwardHloValue(
        role=StreamingForwardHloRole.OUTPUT,
        instruction=output_name,
        physical_shape=instructions[output_name].shape,
        ffi_shape=_hlo_ffi_shape(specifications["output"]),
    )
    saved_state = StreamingForwardHloValue(
        role=StreamingForwardHloRole.LOG_SUM_EXP,
        instruction="shuttle.streaming_forward.log_sum_exp.ffi",
        physical_shape=_hlo_ffi_shape(specifications["log_sum_exp"]),
        ffi_shape=_hlo_ffi_shape(specifications["log_sum_exp"]),
    )
    for value in (*inputs, output):
        _boundary_adapter_opcode(value.physical_shape, value.ffi_shape)
    insertion = max((value.instruction for value in inputs), key=source_order.__getitem__)
    if source_order[insertion] >= min(source_order[name] for name in internal | {output_name}):
        raise ValueError("all generated forward inputs must dominate the forward closure")
    return StreamingForwardHloRegionReplacementPlan(
        inputs=inputs,
        output=output,
        saved_state=saved_state,
        insertion_instruction=insertion,
        internal_instructions=tuple(sorted(internal, key=source_order.__getitem__)),
        external_output_users=external_users,
        provenance=StreamingForwardHloProvenance(
            score_contract=score_contract.name,
            value_contract=value_contract.name,
            maximum_fold=maximum.name,
            sum_fold=sum_folds[0].name,
            domain_restriction=restrictions[0].name,
            score_scale=scale,
        ),
        semantic_fingerprint=generated.semantic_fingerprint,
    )


def _replace_forward_region(
    hlo_text: str,
    plan: StreamingForwardHloRegionReplacementPlan,
    *,
    rematerialized: StreamingForwardHloRegionReplacementPlan,
    target: str,
) -> str:
    indent, position = _insertion_point(hlo_text, plan.insertion_instruction)
    lines: list[str] = []
    input_names: dict[StreamingForwardHloRole, str] = {}
    for value in plan.inputs:
        input_names[value.role] = _emit_boundary_adapter(
            lines,
            indent=indent,
            source=value.instruction,
            source_shape=value.physical_shape,
            target_shape=value.ffi_shape,
            name=f"shuttle.streaming_forward.{value.role.value}.canonical",
        )
    output_shapes = f"{plan.output.ffi_shape}, {plan.saved_state.ffi_shape}"
    operands = ", ".join(
        f"%{input_names[role]}"
        for role in (StreamingForwardHloRole.QUERY, StreamingForwardHloRole.KEY, StreamingForwardHloRole.VALUE)
    )
    constraints = ", ".join(value.ffi_shape for value in plan.inputs)
    call_name = "shuttle.generated.streaming_forward.region"
    lines.append(
        f"{indent}%{call_name} = ({output_shapes}) custom-call({operands}), "
        f'custom_call_target="{target}", operand_layout_constraints={{{constraints}}}, '
        "api_version=API_VERSION_TYPED_FFI, backend_config={}"
    )
    lines.append(
        f"{indent}%shuttle.streaming_forward.output.ffi = {plan.output.ffi_shape} "
        f"get-tuple-element(%{call_name}), index=0"
    )
    lines.append(
        f"{indent}%shuttle.streaming_forward.log_sum_exp.ffi = {plan.saved_state.ffi_shape} "
        f"get-tuple-element(%{call_name}), index=1"
    )
    physical_output = _emit_boundary_adapter(
        lines,
        indent=indent,
        source="shuttle.streaming_forward.output.ffi",
        source_shape=plan.output.ffi_shape,
        target_shape=plan.output.physical_shape,
        name="shuttle.streaming_forward.output.physical",
    )
    rematerialized_output = _emit_boundary_adapter(
        lines,
        indent=indent,
        source="shuttle.streaming_forward.output.ffi",
        source_shape=plan.output.ffi_shape,
        target_shape=rematerialized.output.physical_shape,
        name="shuttle.streaming_forward.output.rematerialized_physical",
    )
    transformed = hlo_text[:position] + "\n" + "\n".join(lines) + hlo_text[position:]
    for user in plan.external_output_users:
        transformed = _replace_entry_operand(
            transformed,
            user=user,
            old=plan.output.instruction,
            new=physical_output,
        )
    for user in rematerialized.external_output_users:
        transformed = _replace_entry_operand(
            transformed,
            user=user,
            old=rematerialized.output.instruction,
            new=rematerialized_output,
        )
    return transformed


def _replace_saved_reverse_region(
    hlo_text: str,
    plan: StreamingReverseHloRegionReplacementPlan,
    *,
    target: str,
) -> str:
    indent, position = _insertion_point(hlo_text, plan.insertion_instruction)
    lines: list[str] = []
    input_by_role = {value.role: value for value in plan.inputs}
    input_names: dict[StreamingReverseHloRole, str] = {}
    for role in (StreamingReverseHloRole.QUERY, StreamingReverseHloRole.KEY, StreamingReverseHloRole.VALUE):
        value = input_by_role[role]
        input_names[role] = _emit_boundary_adapter(
            lines,
            indent=indent,
            source=value.instruction,
            source_shape=value.physical_shape,
            target_shape=value.ffi_shape,
            name=f"shuttle.streaming_saved_reverse.{role.value}.canonical",
        )
    cotangent = input_by_role[StreamingReverseHloRole.OUTPUT_COTANGENT]
    input_names[StreamingReverseHloRole.OUTPUT_COTANGENT] = _emit_boundary_adapter(
        lines,
        indent=indent,
        source=cotangent.instruction,
        source_shape=cotangent.physical_shape,
        target_shape=cotangent.ffi_shape,
        name="shuttle.streaming_saved_reverse.output_cotangent.canonical",
    )
    output_shapes = ", ".join(value.ffi_shape for value in plan.outputs)
    operands = ", ".join(
        (
            f"%{input_names[StreamingReverseHloRole.QUERY]}",
            f"%{input_names[StreamingReverseHloRole.KEY]}",
            f"%{input_names[StreamingReverseHloRole.VALUE]}",
            "%shuttle.streaming_forward.output.ffi",
            "%shuttle.streaming_forward.log_sum_exp.ffi",
            f"%{input_names[StreamingReverseHloRole.OUTPUT_COTANGENT]}",
        )
    )
    constraints = ", ".join(
        (
            input_by_role[StreamingReverseHloRole.QUERY].ffi_shape,
            input_by_role[StreamingReverseHloRole.KEY].ffi_shape,
            input_by_role[StreamingReverseHloRole.VALUE].ffi_shape,
            _shape_with_default_layout(input_by_role[StreamingReverseHloRole.QUERY].ffi_shape),
            _log_sum_exp_shape(input_by_role[StreamingReverseHloRole.QUERY].ffi_shape),
            input_by_role[StreamingReverseHloRole.OUTPUT_COTANGENT].ffi_shape,
        )
    )
    call_name = "shuttle.generated.streaming_saved_reverse.region"
    lines.append(
        f"{indent}%{call_name} = ({output_shapes}) custom-call({operands}), "
        f'custom_call_target="{target}", operand_layout_constraints={{{constraints}}}, '
        "api_version=API_VERSION_TYPED_FFI, backend_config={}"
    )
    replacements: dict[str, str] = {}
    for index, value in enumerate(plan.outputs):
        canonical = f"shuttle.streaming_saved_reverse.{value.role.value}.canonical"
        lines.append(f"{indent}%{canonical} = {value.ffi_shape} get-tuple-element(%{call_name}), index={index}")
        replacements[value.instruction] = _emit_boundary_adapter(
            lines,
            indent=indent,
            source=canonical,
            source_shape=value.ffi_shape,
            target_shape=value.physical_shape,
            name=f"shuttle.streaming_saved_reverse.{value.role.value}.physical",
        )
    transformed = hlo_text[:position] + "\n" + "\n".join(lines) + hlo_text[position:]
    for old, users in plan.external_users:
        for user in users:
            transformed = _replace_entry_operand(transformed, user=user, old=old, new=replacements[old])
    return transformed


def _validate_pair_signatures(
    forward: GeneratedStreamingAttentionForwardFfi,
    reverse: GeneratedStreamingAttentionBackwardFfi,
) -> None:
    if tuple(value.name for value in forward.inputs) != ("query", "key", "value"):
        raise ValueError("early forward must consume Q/K/V")
    if tuple(value.name for value in forward.outputs) != ("output", "log_sum_exp"):
        raise ValueError("early forward must emit O plus log-normalizer state")
    if reverse.state_policy is not StreamingAttentionBackwardStatePolicy.SAVED_OUTPUT_AND_LOG_SUM_EXP:
        raise ValueError("later reverse must explicitly consume saved output and log-normalizer state")
    if reverse.result_policy is not StreamingAttentionBackwardResultPolicy.GRADIENTS_ONLY:
        raise ValueError("later reverse may emit only dQ/dK/dV")
    if tuple(value.name for value in reverse.inputs) != (
        "query",
        "key",
        "value",
        "output",
        "log_sum_exp",
        "output_cotangent",
    ):
        raise ValueError("saved reverse has an unexpected input ABI")
    if tuple(value.name for value in reverse.outputs) != (
        "query_cotangent",
        "key_cotangent",
        "value_cotangent",
    ):
        raise ValueError("saved reverse has an unexpected output ABI")
    forward_by_name = {value.name: value for value in (*forward.inputs, *forward.outputs)}
    reverse_by_name = {value.name: value for value in reverse.inputs}
    for name in ("query", "key", "value", "output", "log_sum_exp"):
        if (forward_by_name[name].dtype, forward_by_name[name].shape, forward_by_name[name].layout) != (
            reverse_by_name[name].dtype,
            reverse_by_name[name].shape,
            reverse_by_name[name].layout,
        ):
            raise ValueError(f"forward/reverse state ABI differs for {name}")


def _recompute_proof_boundary(
    saved: GeneratedStreamingAttentionBackwardFfi,
) -> GeneratedStreamingAttentionBackwardFfi:
    names = {"query", "key", "value", "output_cotangent"}
    return replace(
        saved,
        state_policy=StreamingAttentionBackwardStatePolicy.RECOMPUTE,
        inputs=tuple(value for value in saved.inputs if value.name in names),
    )


def _insertion_point(hlo_text: str, instruction: str) -> tuple[str, int]:
    pattern = re.compile(
        rf"^(?P<indent>\s*)(?:ROOT\s+)?%?{re.escape(instruction)} = .*?$",
        re.MULTILINE,
    )
    matches = tuple(pattern.finditer(hlo_text))
    if len(matches) != 1:
        raise ValueError(f"expected one insertion definition for {instruction!r}")
    return matches[0].group("indent"), matches[0].end()


def _shape_with_default_layout(shape: str) -> str:
    dtype, dimensions = _shape_signature(shape)
    layout = ",".join(str(index) for index in reversed(range(len(dimensions))))
    return f"{dtype}[{','.join(str(value) for value in dimensions)}]{{{layout}}}"


def _hlo_ffi_shape(specification) -> str:
    dtype = {
        DType.BF16: "bf16",
        DType.FP32: "f32",
    }.get(specification.dtype)
    if dtype is None:
        raise ValueError(f"unsupported attention FFI dtype {specification.dtype.value}")
    dimensions = ",".join(str(value) for value in specification.shape)
    layout = ",".join(str(value) for value in specification.layout)
    return f"{dtype}[{dimensions}]{{{layout}}}"


def _log_sum_exp_shape(query_shape: str) -> str:
    _, dimensions = _shape_signature(query_shape)
    batch, sequence, heads, _ = dimensions
    return f"f32[{batch},{heads},{sequence}]{{2,1,0}}"
