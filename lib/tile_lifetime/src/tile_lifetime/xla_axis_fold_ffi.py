# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover and replace generic entry-local Fold/final-Map regions in XLA HLO.

The recovery consumes only physical shapes, scalar dataflow, a sum reducer,
and entry-computation uses.  Frontend metadata and workload names do not
participate.  Supported scalar bodies lower directly to :class:`AxisFoldProgram`
and therefore use the same generated Map/Fold CUDA family as standalone
Shuttle programs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import reduce
from operator import mul

from shuttle.ir import DType
from tile_lifetime.cuda_axis_fold_codegen import (
    AxisFoldDirection,
    AxisFoldInput,
    AxisFoldInputLayout,
    AxisFoldOutputKind,
    AxisFoldProgram,
    AxisFoldReassociation,
    AxisFoldReduction,
    CudaAxisFoldFfiBuffer,
    GeneratedCudaAxisFoldFfi,
)
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.tensor_program import (
    ScalarExpression,
    ScalarExpressionKind,
    scalar_binary,
    scalar_constant,
    scalar_expression_inputs,
    scalar_input,
)
from tile_lifetime.xla_hlo_recovery import HloComputation, HloInstruction, HloModuleGraph, parse_hlo_module_text

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\](?:\{(?P<layout>[0-9,]+)\})?")
_CALLED_COMPUTATION = re.compile(r"(?:calls|to_apply)=%?(?P<name>[A-Za-z0-9_.-]+)")
_CONSTANT = re.compile(r"constant\((?P<value>-?inf|[-+0-9.eE]+)\)")
_CONTROL_PREDECESSORS = re.compile(r"control-predecessors=\{(?P<values>[^}]*)\}")
_DIMENSIONS = re.compile(r"dimensions=\{(?P<values>[0-9,]*)\}")
_PARAMETER_NUMBER = re.compile(r"parameter\((?P<number>[0-9]+)\)")


@dataclass(frozen=True)
class AxisFoldHloInput:
    """One physical HLO value adapted to a generated Fold input."""

    name: str
    instruction: str
    physical_shape: str
    ffi_shape: str


@dataclass(frozen=True)
class AxisFoldHloProvenance:
    """Generic HLO algebra proving one Fold/final-Map replacement."""

    fold_instruction: str
    final_map_instruction: str
    reduction_dimension: int
    reducer: str
    contribution_expression: ScalarExpression
    output_expression: ScalarExpression


@dataclass(frozen=True)
class AxisFoldHloRegionReplacementPlan:
    """A generated axis Fold plus its exact entry-computation boundary."""

    program: AxisFoldProgram
    inputs: tuple[AxisFoldHloInput, ...]
    output_instruction: str
    output_physical_shape: str
    output_ffi_shape: str
    insertion_instruction: str
    internal_instructions: tuple[str, ...]
    external_users: tuple[str, ...]
    provenance: AxisFoldHloProvenance
    numerical_policy: NumericalPolicy


@dataclass(frozen=True)
class AxisFoldHloRecoveryReport:
    """All supported candidates and structured near-miss diagnostics."""

    plans: tuple[AxisFoldHloRegionReplacementPlan, ...]
    rejected: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class AxisFoldHloRegionReplacementAudit:
    """Post-roundtrip liveness and layout evidence for one replacement."""

    call_instruction: str
    dead_internal_instructions: tuple[str, ...]
    rewired_external_users: tuple[str, ...]
    copy_count: tuple[int, int]
    transpose_count: tuple[int, int]


def recover_axis_fold_hlo_region_candidates(
    hlo_text: str,
    *,
    numerical_policy: NumericalPolicy,
    threads: int = 256,
) -> AxisFoldHloRecoveryReport:
    """Derive every supported sum-Fold/final-Map region from physical HLO."""
    if numerical_policy is NumericalPolicy.BITWISE_EXACT:
        raise ValueError("parallel axis-Fold replacement requires an explicit rounding-reorder numerical policy")
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    computations = {computation.name: computation for computation in module.computations}
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    users = _entry_users(entry)
    source_order = {instruction.name: index for index, instruction in enumerate(entry.instructions)}
    plans: list[AxisFoldHloRegionReplacementPlan] = []
    rejected: list[tuple[str, str]] = []
    for fold_instruction in entry.instructions:
        called: HloComputation | None = None
        if fold_instruction.opcode == "fusion":
            called_name = _called_computation_name(fold_instruction.attributes)
            if called_name is not None:
                called = computations[called_name]
            if called is None or called.root.opcode != "reduce":
                continue
        elif fold_instruction.opcode != "reduce":
            continue
        try:
            if called is not None:
                plan = _recover_fusion_candidate(
                    module,
                    entry,
                    instructions,
                    users,
                    source_order,
                    fold_instruction,
                    called,
                    numerical_policy=numerical_policy,
                    threads=threads,
                )
            else:
                plan = _recover_flat_candidate(
                    module,
                    entry,
                    instructions,
                    users,
                    source_order,
                    fold_instruction,
                    numerical_policy=numerical_policy,
                    threads=threads,
                )
            plans.append(plan)
        except ValueError as error:
            rejected.append((fold_instruction.name, str(error)))
    return AxisFoldHloRecoveryReport(plans=tuple(plans), rejected=tuple(rejected))


def plan_axis_fold_hlo_region_replacement(
    hlo_text: str,
    generated: GeneratedCudaAxisFoldFfi,
    *,
    numerical_policy: NumericalPolicy,
    threads: int = 256,
) -> AxisFoldHloRegionReplacementPlan:
    """Require one unambiguous candidate matching a generated typed FFI."""
    report = recover_axis_fold_hlo_region_candidates(
        hlo_text,
        numerical_policy=numerical_policy,
        threads=threads,
    )
    matching = tuple(
        plan for plan in report.plans if generated.semantic_fingerprints == (plan.program.semantic_fingerprint,)
    )
    if len(matching) != 1:
        details = "; ".join(f"%{name}: {reason}" for name, reason in report.rejected)
        raise ValueError(
            f"expected one generic axis-Fold/final-Map region matching the generated program, found {len(matching)}"
            + (f" ({details})" if details else "")
        )
    plan = matching[0]
    _validate_generated_signature(plan, generated)
    return plan


def replace_axis_fold_hlo_region_with_custom_call(
    hlo_text: str,
    plan: AxisFoldHloRegionReplacementPlan,
    *,
    target: str,
) -> str:
    """Insert one generated Fold call and redirect only proven live users."""
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    reserved = {instruction.name for instruction in entry.instructions}
    suffix = re.sub(r"[^A-Za-z0-9_.-]", "_", target)
    call_name = f"shuttle.generated.axis_fold.region.{suffix}"
    canonical_output = f"shuttle.axis_fold.output.canonical.{suffix}"
    physical_output = f"shuttle.axis_fold.output.physical.{suffix}"
    generated_names = {
        call_name,
        canonical_output,
        physical_output,
        *(f"shuttle.axis_fold.{value.name}.canonical.{suffix}" for value in plan.inputs),
    }
    collision = reserved & generated_names
    if collision:
        raise ValueError(f"axis-Fold replacement names already exist: {sorted(collision)}")
    insertion_pattern = re.compile(
        rf"^(?P<indent>\s*)(?:ROOT\s+)?%?{re.escape(plan.insertion_instruction)} = .*?$",
        re.MULTILINE,
    )
    matches = tuple(insertion_pattern.finditer(hlo_text))
    if len(matches) != 1:
        raise ValueError(f"expected one insertion definition for {plan.insertion_instruction!r}")
    match = matches[0]
    indent = match.group("indent")
    lines: list[str] = []
    operands: list[str] = []
    for value in plan.inputs:
        adapted = _emit_reshape_adapter(
            lines,
            indent=indent,
            source=value.instruction,
            source_shape=value.physical_shape,
            target_shape=value.ffi_shape,
            name=f"shuttle.axis_fold.{value.name}.canonical.{suffix}",
        )
        operands.append(adapted)
    operand_text = ", ".join(f"%{name}" for name in operands)
    constraints = ", ".join(value.ffi_shape for value in plan.inputs)
    lines.append(
        f"{indent}%{call_name} = ({plan.output_ffi_shape}) custom-call({operand_text}), "
        f'custom_call_target="{target}", operand_layout_constraints={{{constraints}}}, '
        "api_version=API_VERSION_TYPED_FFI, backend_config={}"
    )
    canonical = canonical_output
    lines.append(f"{indent}%{canonical} = {plan.output_ffi_shape} get-tuple-element(%{call_name}), index=0")
    physical = _emit_reshape_adapter(
        lines,
        indent=indent,
        source=canonical,
        source_shape=plan.output_ffi_shape,
        target_shape=plan.output_physical_shape,
        name=physical_output,
    )
    insertion = "\n" + "\n".join(lines)
    rewritten = hlo_text[: match.end()] + insertion + hlo_text[match.end() :]
    for user in plan.external_users:
        rewritten = _replace_entry_operand(
            rewritten,
            user=user,
            old=plan.output_instruction,
            new=physical,
        )
    parse_hlo_module_text(rewritten)
    return rewritten


def audit_axis_fold_hlo_region_replacement(
    original_hlo: str,
    transformed_hlo: str,
    plan: AxisFoldHloRegionReplacementPlan,
    *,
    target: str,
) -> AxisFoldHloRegionReplacementAudit:
    """Verify the old region is dead externally and no layout work was added."""
    original_module = parse_hlo_module_text(original_hlo)
    transformed_module = parse_hlo_module_text(transformed_hlo)
    original_entry = original_module.computation(original_module.entry)
    transformed_entry = transformed_module.computation(transformed_module.entry)
    transformed_instructions = {instruction.name: instruction for instruction in transformed_entry.instructions}
    transformed_users = _entry_users(transformed_entry)
    target_attribute = f'custom_call_target="{target}"'
    calls = tuple(
        instruction
        for instruction in transformed_entry.instructions
        if instruction.opcode == "custom-call" and target_attribute in instruction.attributes
    )
    if len(calls) != 1:
        raise ValueError(f"expected one post-roundtrip axis-Fold call for {target!r}, found {len(calls)}")
    call = calls[0]
    suffix = re.sub(r"[^A-Za-z0-9_.-]", "_", target)
    expected_call_shape = f"({plan.output_ffi_shape})"
    if call.shape != expected_call_shape:
        raise ValueError(f"axis-Fold call output signature changed: {call.shape} != {expected_call_shape}")
    expected_operands: list[str] = []
    for value in plan.inputs:
        if value.physical_shape == value.ffi_shape:
            expected_operands.append(value.instruction)
            continue
        adapter_name = f"shuttle.axis_fold.{value.name}.canonical.{suffix}"
        adapter = transformed_instructions.get(adapter_name)
        if adapter is None:
            raise ValueError(f"axis-Fold input adapter %{adapter_name} is missing")
        if adapter.opcode != "reshape" or adapter.shape != value.ffi_shape or adapter.operands != (value.instruction,):
            raise ValueError(f"axis-Fold input adapter %{adapter_name} changed signature or source")
        expected_operands.append(adapter_name)
    if call.operands != tuple(expected_operands):
        raise ValueError(f"axis-Fold call operands changed: {call.operands} != {tuple(expected_operands)}")
    expected_constraints = f"operand_layout_constraints={{{', '.join(value.ffi_shape for value in plan.inputs)}}}"
    if expected_constraints not in call.attributes:
        raise ValueError("axis-Fold call operand layout signature changed")
    if "api_version=API_VERSION_TYPED_FFI" not in call.attributes:
        raise ValueError("axis-Fold call no longer uses the typed FFI API")

    canonical_output_name = f"shuttle.axis_fold.output.canonical.{suffix}"
    canonical_output = transformed_instructions.get(canonical_output_name)
    if canonical_output is None:
        raise ValueError(f"axis-Fold canonical output %{canonical_output_name} is missing")
    if (
        canonical_output.opcode != "get-tuple-element"
        or canonical_output.shape != plan.output_ffi_shape
        or canonical_output.operands != (call.name,)
        or not canonical_output.attributes.endswith(", index=0")
    ):
        raise ValueError("axis-Fold canonical output changed signature or source")
    replacement_output = canonical_output_name
    if plan.output_ffi_shape != plan.output_physical_shape:
        physical_output_name = f"shuttle.axis_fold.output.physical.{suffix}"
        physical_output = transformed_instructions.get(physical_output_name)
        if physical_output is None:
            raise ValueError(f"axis-Fold physical output %{physical_output_name} is missing")
        if (
            physical_output.opcode != "reshape"
            or physical_output.shape != plan.output_physical_shape
            or physical_output.operands != (canonical_output_name,)
        ):
            raise ValueError("axis-Fold physical output changed signature or source")
        replacement_output = physical_output_name

    internal = set(plan.internal_instructions)
    for name in internal:
        crossing = tuple(user for user in transformed_users.get(name, ()) if user not in internal)
        if crossing:
            raise ValueError(f"old axis-Fold value %{name} remains externally live through {crossing}")
    for user in plan.external_users:
        transformed_user = transformed_instructions.get(user)
        if transformed_user is None:
            raise ValueError(f"external axis-Fold user %{user} is missing")
        if plan.output_instruction in transformed_user.operands:
            raise ValueError(f"external user %{user} still consumes old Fold output %{plan.output_instruction}")
        if replacement_output not in transformed_user.operands:
            raise ValueError(f"external user %{user} does not consume replacement Fold output %{replacement_output}")

    def count(entry: HloComputation, opcode: str) -> int:
        return sum(instruction.opcode == opcode for instruction in entry.instructions)

    copy_count = (count(original_entry, "copy"), count(transformed_entry, "copy"))
    transpose_count = (count(original_entry, "transpose"), count(transformed_entry, "transpose"))
    if copy_count[1] > copy_count[0]:
        raise ValueError(f"axis-Fold replacement increased copy count: {copy_count[0]} -> {copy_count[1]}")
    if transpose_count[1] > transpose_count[0]:
        raise ValueError(
            f"axis-Fold replacement increased transpose count: {transpose_count[0]} -> {transpose_count[1]}"
        )
    return AxisFoldHloRegionReplacementAudit(
        call_instruction=calls[0].name,
        dead_internal_instructions=plan.internal_instructions,
        rewired_external_users=plan.external_users,
        copy_count=copy_count,
        transpose_count=transpose_count,
    )


def _recover_fusion_candidate(
    module: HloModuleGraph,
    entry: HloComputation,
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
    source_order: dict[str, int],
    fold_instruction: HloInstruction,
    fold_computation: HloComputation,
    *,
    numerical_policy: NumericalPolicy,
    threads: int,
) -> AxisFoldHloRegionReplacementPlan:
    fold_root = fold_computation.root
    if len(fold_root.operands) != 2:
        raise ValueError("Fold root must have one value and one initializer")
    fold_instructions = {value.name: value for value in fold_computation.instructions}
    input_instruction = fold_instructions[fold_root.operands[0]]
    input_dtype, input_dimensions, input_layout = _shape_signature(input_instruction.shape)
    output_dtype, output_dimensions, _ = _shape_signature(fold_root.shape)
    _validate_minor_sum_fold(
        module,
        fold_root,
        fold_instructions[fold_root.operands[1]],
        input_dtype,
        input_dimensions,
        input_layout,
        output_dtype,
        output_dimensions,
    )
    direct_users = users[fold_instruction.name]
    if len(direct_users) != 1:
        raise ValueError(f"Fold result must have one finalizer, found {direct_users}")
    final_instruction = instructions[direct_users[0]]
    if final_instruction.opcode != "fusion" or final_instruction.operands.count(fold_instruction.name) != 1:
        raise ValueError("Fold result must feed one scalar final-Map fusion")
    final_name = _called_computation_name(final_instruction.attributes)
    if final_name is None:
        raise ValueError("final-Map fusion has no called computation")
    final_computation = module.computation(final_name)
    final_dtype, final_dimensions, final_layout = _shape_signature(final_computation.root.shape)
    if final_dimensions != input_dimensions or final_layout != input_layout or final_dtype not in {"bf16", "f32"}:
        raise ValueError("final Map must restore the Fold input shape and dense layout")
    external_users = users[final_instruction.name]
    if not external_users:
        raise ValueError("final Map has no live external user")
    _verify_no_control_crossing(entry, frozenset({fold_instruction.name, final_instruction.name}))
    if any(
        "custom_call_has_side_effect=true" in value.attributes or "sharding=" in value.attributes
        for value in (fold_instruction, final_instruction)
    ):
        raise ValueError("Fold/final-Map region crosses side-effect or explicit-sharding semantics")

    rows = math.prod(output_dimensions)
    columns = input_dimensions[-1]
    entry_input_names = tuple(
        dict.fromkeys(
            (
                *fold_instruction.operands,
                *(value for value in final_instruction.operands if value != fold_instruction.name),
            )
        )
    )
    input_aliases = {name: f"input{index}" for index, name in enumerate(entry_input_names)}
    contribution = _scalar_expression(
        fold_computation,
        fold_root.operands[0],
        _parameter_bindings(fold_computation, fold_instruction),
        input_aliases,
        fold_instruction=None,
        allow_output_narrowing=False,
    )
    final_root = final_computation.root
    expression_root = final_root.name
    if final_root.opcode == "convert":
        source = {value.name: value for value in final_computation.instructions}[final_root.operands[0]]
        if source.dtype != "f32" or final_root.dtype not in {"bf16", "f32"}:
            raise ValueError("final output conversion must be FP32 to BF16/FP32")
        expression_root = source.name
    output_expression = _scalar_expression(
        final_computation,
        expression_root,
        _parameter_bindings(final_computation, final_instruction),
        input_aliases,
        fold_instruction=fold_instruction.name,
        allow_output_narrowing=False,
    )
    used_inputs = scalar_expression_inputs(contribution) | scalar_expression_inputs(output_expression)
    ordered_entries = tuple(name for name in entry_input_names if input_aliases[name] in used_inputs)
    axis_inputs, hlo_inputs = _axis_fold_inputs(
        ordered_entries,
        instructions,
        input_aliases,
        rows=rows,
        columns=columns,
    )
    dimensions = _reduction_dimensions(fold_root.attributes)
    program = _axis_fold_program(
        rows,
        columns,
        axis_inputs,
        contribution,
        output_expression,
        _dtype(final_dtype),
        threads,
    )
    return AxisFoldHloRegionReplacementPlan(
        program=program,
        inputs=hlo_inputs,
        output_instruction=final_instruction.name,
        output_physical_shape=final_instruction.shape,
        output_ffi_shape=_hlo_shape(final_dtype, (rows, columns)),
        insertion_instruction=final_instruction.name,
        internal_instructions=(fold_instruction.name, final_instruction.name),
        external_users=tuple(sorted(external_users, key=source_order.__getitem__)),
        provenance=AxisFoldHloProvenance(
            fold_instruction=fold_instruction.name,
            final_map_instruction=final_instruction.name,
            reduction_dimension=dimensions[0],
            reducer="add",
            contribution_expression=contribution,
            output_expression=output_expression,
        ),
        numerical_policy=numerical_policy,
    )


def _recover_flat_candidate(
    module: HloModuleGraph,
    entry: HloComputation,
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
    source_order: dict[str, int],
    fold_instruction: HloInstruction,
    *,
    numerical_policy: NumericalPolicy,
    threads: int,
) -> AxisFoldHloRegionReplacementPlan:
    if len(fold_instruction.operands) != 2:
        raise ValueError("Fold root must have one value and one initializer")
    input_instruction = instructions[fold_instruction.operands[0]]
    input_dtype, input_dimensions, input_layout = _shape_signature(input_instruction.shape)
    output_dtype, output_dimensions, _ = _shape_signature(fold_instruction.shape)
    _validate_minor_sum_fold(
        module,
        fold_instruction,
        instructions[fold_instruction.operands[1]],
        input_dtype,
        input_dimensions,
        input_layout,
        output_dtype,
        output_dimensions,
    )
    supported = frozenset(
        {"add", "bitcast", "broadcast", "convert", "copy", "divide", "multiply", "negate", "reshape", "subtract"}
    )
    reachable = _entry_descendants(fold_instruction.name, users, supported, instructions)
    compatible = tuple(
        name
        for name in reachable
        if name != fold_instruction.name
        and _shape_signature(instructions[name].shape)[1] == input_dimensions
        and _shape_signature(instructions[name].shape)[0] in {"bf16", "f32"}
    )
    terminal = tuple(
        name
        for name in compatible
        if not any(
            other != name and _entry_reachable(name, other, users, supported, instructions) for other in compatible
        )
    )
    if len(terminal) != 1:
        raise ValueError(f"Fold must reach one terminal full-shape final Map, found {terminal}")
    final_name = terminal[0]
    internal = _fold_dependent_ancestors(final_name, fold_instruction.name, instructions)
    for name in internal - {final_name}:
        crossing = tuple(user for user in users[name] if user not in internal)
        if crossing:
            raise ValueError(f"internal Fold value %{name} has external users {crossing}")
    external_users = users[final_name]
    if not external_users:
        raise ValueError("final Map has no live external user")
    _verify_no_control_crossing(entry, frozenset(internal))
    if any(
        "custom_call_has_side_effect=true" in instructions[name].attributes
        or "sharding=" in instructions[name].attributes
        for name in internal
    ):
        raise ValueError("Fold/final-Map region crosses side-effect or explicit-sharding semantics")

    final_instruction = instructions[final_name]
    final_dtype, final_dimensions, final_layout = _shape_signature(final_instruction.shape)
    if final_dimensions != input_dimensions or final_layout != input_layout:
        raise ValueError("final Map must restore the Fold input shape and dense layout")
    rows = math.prod(output_dimensions)
    columns = input_dimensions[-1]
    boundary_names = {fold_instruction.operands[0]}
    boundary_names.update(
        operand
        for name in internal
        for operand in instructions[name].operands
        if operand not in internal and operand != fold_instruction.operands[1]
    )
    ordered_entries = tuple(sorted(boundary_names, key=source_order.__getitem__))
    input_aliases = {name: f"input{index}" for index, name in enumerate(ordered_entries)}
    contribution = scalar_input(input_aliases[fold_instruction.operands[0]])
    expression_root = final_name
    if final_instruction.opcode == "convert":
        source = instructions[final_instruction.operands[0]]
        if source.dtype != "f32" or final_instruction.dtype not in {"bf16", "f32"}:
            raise ValueError("final output conversion must be FP32 to BF16/FP32")
        expression_root = source.name
    output_expression = _entry_scalar_expression(
        instructions,
        expression_root,
        internal,
        input_aliases,
        fold_instruction=fold_instruction.name,
    )
    used_inputs = scalar_expression_inputs(contribution) | scalar_expression_inputs(output_expression)
    ordered_entries = tuple(name for name in ordered_entries if input_aliases[name] in used_inputs)
    axis_inputs, hlo_inputs = _axis_fold_inputs(
        ordered_entries,
        instructions,
        input_aliases,
        rows=rows,
        columns=columns,
    )
    dimensions = _reduction_dimensions(fold_instruction.attributes)
    program = _axis_fold_program(
        rows,
        columns,
        axis_inputs,
        contribution,
        output_expression,
        _dtype(final_dtype),
        threads,
    )
    ordered_internal = tuple(sorted(internal, key=source_order.__getitem__))
    return AxisFoldHloRegionReplacementPlan(
        program=program,
        inputs=hlo_inputs,
        output_instruction=final_name,
        output_physical_shape=final_instruction.shape,
        output_ffi_shape=_hlo_shape(final_dtype, (rows, columns)),
        insertion_instruction=final_name,
        internal_instructions=ordered_internal,
        external_users=tuple(sorted(external_users, key=source_order.__getitem__)),
        provenance=AxisFoldHloProvenance(
            fold_instruction=fold_instruction.name,
            final_map_instruction=final_name,
            reduction_dimension=dimensions[0],
            reducer="add",
            contribution_expression=contribution,
            output_expression=output_expression,
        ),
        numerical_policy=numerical_policy,
    )


def _validate_minor_sum_fold(
    module: HloModuleGraph,
    fold: HloInstruction,
    initializer: HloInstruction,
    input_dtype: str,
    input_dimensions: tuple[int, ...],
    input_layout: tuple[int, ...],
    output_dtype: str,
    output_dimensions: tuple[int, ...],
) -> None:
    if input_dtype != "f32" or output_dtype != "f32" or len(input_dimensions) < 2:
        raise ValueError("only FP32 accumulation from a rank-two-or-higher input is supported")
    dimensions = _reduction_dimensions(fold.attributes)
    if dimensions != (len(input_dimensions) - 1,) or output_dimensions != input_dimensions[:-1]:
        raise ValueError("Fold must reduce exactly the minor logical axis")
    if input_layout != tuple(reversed(range(len(input_dimensions)))):
        raise ValueError("Fold input must use a dense minor-axis layout")
    if initializer.opcode != "constant" or _constant_value(initializer) != 0.0:
        raise ValueError("Fold initializer must be scalar zero")
    reducer_name = _called_computation_name(fold.attributes)
    if reducer_name is None or module.computation(reducer_name).root.opcode != "add":
        raise ValueError("Fold reducer must be generic addition")


def _axis_fold_program(
    rows: int,
    columns: int,
    inputs: tuple[AxisFoldInput, ...],
    contribution: ScalarExpression,
    output_expression: ScalarExpression,
    output_dtype: DType,
    threads: int,
) -> AxisFoldProgram:
    return AxisFoldProgram(
        rows=rows,
        columns=columns,
        inputs=inputs,
        reductions=(AxisFoldReduction("fold_sum", contribution),),
        reduction_axis=AxisFoldDirection.COLUMNS,
        output_kind=AxisFoldOutputKind.ELEMENT,
        output_expression=output_expression,
        output_dtype=output_dtype,
        threads=threads,
        reassociation=AxisFoldReassociation.DETERMINISTIC_TREE,
    )


def _axis_fold_inputs(
    ordered_entries: tuple[str, ...],
    instructions: dict[str, HloInstruction],
    input_aliases: dict[str, str],
    *,
    rows: int,
    columns: int,
) -> tuple[tuple[AxisFoldInput, ...], tuple[AxisFoldHloInput, ...]]:
    axis_inputs: list[AxisFoldInput] = []
    hlo_inputs: list[AxisFoldHloInput] = []
    for name in ordered_entries:
        instruction = instructions[name]
        dtype, physical_dimensions, physical_layout = _shape_signature(instruction.shape)
        layout = _classify_input_layout(physical_dimensions, rows=rows, columns=columns)
        canonical_dimensions = {
            AxisFoldInputLayout.ELEMENT: (rows, columns),
            AxisFoldInputLayout.ROW: (rows,),
            AxisFoldInputLayout.COLUMN: (columns,),
            AxisFoldInputLayout.SCALAR: (),
        }[layout]
        if physical_layout != tuple(reversed(range(len(physical_dimensions)))):
            raise ValueError(f"input %{name} does not have a dense minor-axis layout")
        alias = input_aliases[name]
        axis_inputs.append(AxisFoldInput(alias, _dtype(dtype), layout))
        hlo_inputs.append(
            AxisFoldHloInput(
                name=alias,
                instruction=name,
                physical_shape=instruction.shape,
                ffi_shape=_hlo_shape(dtype, canonical_dimensions),
            )
        )
    return tuple(axis_inputs), tuple(hlo_inputs)


def _entry_descendants(
    root: str,
    users: dict[str, tuple[str, ...]],
    supported: frozenset[str],
    instructions: dict[str, HloInstruction],
) -> frozenset[str]:
    reachable = {root}
    pending = [root]
    while pending:
        producer = pending.pop()
        for user in users[producer]:
            if instructions[user].opcode in supported and user not in reachable:
                reachable.add(user)
                pending.append(user)
    return frozenset(reachable)


def _entry_reachable(
    source: str,
    target: str,
    users: dict[str, tuple[str, ...]],
    supported: frozenset[str],
    instructions: dict[str, HloInstruction],
) -> bool:
    return target in _entry_descendants(source, users, supported, instructions)


def _fold_dependent_ancestors(
    root: str,
    fold: str,
    instructions: dict[str, HloInstruction],
) -> set[str]:
    dependency_cache: dict[str, bool] = {}

    def depends(name: str) -> bool:
        if name in dependency_cache:
            return dependency_cache[name]
        result = name == fold or any(depends(operand) for operand in instructions[name].operands)
        dependency_cache[name] = result
        return result

    internal: set[str] = set()
    pending = [root]
    while pending:
        name = pending.pop()
        if name in internal or not depends(name):
            continue
        internal.add(name)
        pending.extend(instructions[name].operands)
    if fold not in internal:
        raise ValueError("final Map does not depend on the candidate Fold")
    return internal


def _entry_scalar_expression(
    instructions: dict[str, HloInstruction],
    root: str,
    internal: set[str],
    input_aliases: dict[str, str],
    *,
    fold_instruction: str,
) -> ScalarExpression:
    cache: dict[str, ScalarExpression] = {}

    def visit(name: str) -> ScalarExpression:
        if name in cache:
            return cache[name]
        if name == fold_instruction:
            result = scalar_input("fold_sum")
        elif name not in internal:
            result = scalar_input(input_aliases[name])
        else:
            instruction = instructions[name]
            if instruction.opcode == "constant":
                result = scalar_constant(_constant_value(instruction))
            elif instruction.opcode in {"bitcast", "broadcast", "copy", "reshape"}:
                result = visit(instruction.operands[0])
            elif instruction.opcode == "convert":
                source = instructions[instruction.operands[0]]
                if source.dtype == "f32" and instruction.dtype == "bf16":
                    raise ValueError("internal FP32-to-BF16 rounding boundary is not representable by the scalar AST")
                if {source.dtype, instruction.dtype} - {"f32", "bf16"}:
                    raise ValueError(f"unsupported scalar conversion {source.dtype} -> {instruction.dtype}")
                result = visit(source.name)
            elif instruction.opcode in {"add", "subtract", "multiply", "divide"}:
                kind = {
                    "add": ScalarExpressionKind.ADD,
                    "subtract": ScalarExpressionKind.SUBTRACT,
                    "multiply": ScalarExpressionKind.MULTIPLY,
                    "divide": ScalarExpressionKind.DIVIDE,
                }[instruction.opcode]
                result = scalar_binary(kind, visit(instruction.operands[0]), visit(instruction.operands[1]))
            elif instruction.opcode == "negate":
                result = scalar_binary(
                    ScalarExpressionKind.MULTIPLY,
                    scalar_constant(-1.0),
                    visit(instruction.operands[0]),
                )
            else:
                raise ValueError(f"unsupported scalar-Map opcode {instruction.opcode!r}")
        cache[name] = result
        return result

    return visit(root)


def _scalar_expression(
    computation: HloComputation,
    root: str,
    bindings: dict[str, str],
    input_aliases: dict[str, str],
    *,
    fold_instruction: str | None,
    allow_output_narrowing: bool,
) -> ScalarExpression:
    instructions = {instruction.name: instruction for instruction in computation.instructions}
    cache: dict[str, ScalarExpression] = {}

    def visit(name: str) -> ScalarExpression:
        if name in cache:
            return cache[name]
        instruction = instructions[name]
        if instruction.opcode == "parameter":
            entry_name = bindings[name]
            result = scalar_input("fold_sum" if entry_name == fold_instruction else input_aliases[entry_name])
        elif instruction.opcode == "constant":
            result = scalar_constant(_constant_value(instruction))
        elif instruction.opcode in {"bitcast", "broadcast", "copy", "reshape"}:
            if len(instruction.operands) != 1:
                raise ValueError(f"scalar wrapper %{name} is not unary")
            result = visit(instruction.operands[0])
        elif instruction.opcode == "convert":
            if len(instruction.operands) != 1:
                raise ValueError(f"scalar convert %{name} is not unary")
            source = instructions[instruction.operands[0]]
            if source.dtype == "f32" and instruction.dtype == "bf16" and not allow_output_narrowing:
                raise ValueError("internal FP32-to-BF16 rounding boundary is not representable by the scalar AST")
            if {source.dtype, instruction.dtype} - {"f32", "bf16"}:
                raise ValueError(f"unsupported scalar conversion {source.dtype} -> {instruction.dtype}")
            result = visit(source.name)
        elif instruction.opcode in {"add", "subtract", "multiply", "divide"}:
            if len(instruction.operands) != 2:
                raise ValueError(f"scalar {instruction.opcode} %{name} is not binary")
            kind = {
                "add": ScalarExpressionKind.ADD,
                "subtract": ScalarExpressionKind.SUBTRACT,
                "multiply": ScalarExpressionKind.MULTIPLY,
                "divide": ScalarExpressionKind.DIVIDE,
            }[instruction.opcode]
            result = scalar_binary(kind, visit(instruction.operands[0]), visit(instruction.operands[1]))
        elif instruction.opcode == "negate":
            if len(instruction.operands) != 1:
                raise ValueError(f"scalar negate %{name} is not unary")
            result = scalar_binary(
                ScalarExpressionKind.MULTIPLY,
                scalar_constant(-1.0),
                visit(instruction.operands[0]),
            )
        else:
            raise ValueError(f"unsupported scalar-Map opcode {instruction.opcode!r}")
        cache[name] = result
        return result

    return visit(root)


def _parameter_bindings(computation: HloComputation, call: HloInstruction) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for instruction in computation.instructions:
        if instruction.opcode != "parameter":
            continue
        match = _PARAMETER_NUMBER.search(instruction.attributes)
        if match is None:
            raise ValueError(f"parameter %{instruction.name} lacks a parameter number")
        number = int(match.group("number"))
        if number >= len(call.operands):
            raise ValueError(f"parameter %{instruction.name} exceeds call arity")
        bindings[instruction.name] = call.operands[number]
    return bindings


def _validate_generated_signature(
    plan: AxisFoldHloRegionReplacementPlan,
    generated: GeneratedCudaAxisFoldFfi,
) -> None:
    if generated.semantic_fingerprints != (plan.program.semantic_fingerprint,):
        raise ValueError("generated axis-Fold semantic fingerprint does not match recovered HLO algebra")
    expected_inputs = tuple(
        CudaAxisFoldFfiBuffer(value.name, input_value.dtype, _shape_signature(value.ffi_shape)[1])
        for value, input_value in zip(plan.inputs, plan.program.inputs, strict=True)
    )
    if generated.inputs != expected_inputs:
        raise ValueError(f"generated axis-Fold inputs differ from recovered boundary: {generated.inputs}")
    expected_output = CudaAxisFoldFfiBuffer(
        "output0",
        plan.program.output_dtype,
        _shape_signature(plan.output_ffi_shape)[1],
    )
    if generated.outputs != (expected_output,):
        raise ValueError(f"generated axis-Fold output differs from recovered boundary: {generated.outputs}")


def _entry_users(entry: HloComputation) -> dict[str, tuple[str, ...]]:
    users: dict[str, list[str]] = {instruction.name: [] for instruction in entry.instructions}
    for instruction in entry.instructions:
        for operand in instruction.operands:
            users.setdefault(operand, []).append(instruction.name)
    return {name: tuple(values) for name, values in users.items()}


def _verify_no_control_crossing(entry: HloComputation, region: frozenset[str]) -> None:
    for instruction in entry.instructions:
        match = _CONTROL_PREDECESSORS.search(instruction.attributes)
        predecessors = frozenset(re.findall(r"%?([A-Za-z0-9_.-]+)", match.group("values"))) if match else frozenset()
        if instruction.name in region and predecessors:
            raise ValueError("Fold/final-Map region has an explicit control predecessor")
        if instruction.name not in region and predecessors & region:
            raise ValueError("Fold/final-Map region has an explicit control successor")


def _classify_input_layout(
    dimensions: tuple[int, ...],
    *,
    rows: int,
    columns: int,
) -> AxisFoldInputLayout:
    size = reduce(mul, dimensions, 1)
    candidates: list[AxisFoldInputLayout] = []
    if size == rows * columns and dimensions and dimensions[-1] == columns:
        candidates.append(AxisFoldInputLayout.ELEMENT)
    if size == rows and dimensions != (columns,):
        candidates.append(AxisFoldInputLayout.ROW)
    if dimensions == (columns,):
        candidates.append(AxisFoldInputLayout.COLUMN)
    if dimensions == ():
        candidates.append(AxisFoldInputLayout.SCALAR)
    if len(candidates) != 1:
        raise ValueError(f"cannot uniquely classify input shape {dimensions} over ({rows}, {columns})")
    return candidates[0]


def _emit_reshape_adapter(
    lines: list[str],
    *,
    indent: str,
    source: str,
    source_shape: str,
    target_shape: str,
    name: str,
) -> str:
    if source_shape == target_shape:
        return source
    source_dtype, source_dimensions, source_layout = _shape_signature(source_shape)
    target_dtype, target_dimensions, target_layout = _shape_signature(target_shape)
    if source_dtype != target_dtype or math.prod(source_dimensions) != math.prod(target_dimensions):
        raise ValueError(f"boundary reshape cannot change dtype or element count: {source_shape} -> {target_shape}")
    if source_layout != tuple(reversed(range(len(source_dimensions)))) or target_layout != tuple(
        reversed(range(len(target_dimensions)))
    ):
        raise ValueError(f"boundary reshape requires dense minor-axis layouts: {source_shape} -> {target_shape}")
    lines.append(f"{indent}%{name} = {target_shape} reshape(%{source})")
    return name


def _replace_entry_operand(hlo_text: str, *, user: str, old: str, new: str) -> str:
    pattern = re.compile(
        rf"^(?P<prefix>\s*(?:ROOT\s+)?%?{re.escape(user)} = )(?P<body>.*?)$",
        re.MULTILINE,
    )
    matches = tuple(pattern.finditer(hlo_text))
    if len(matches) != 1:
        raise ValueError(f"expected one external user definition for {user!r}")
    match = matches[0]
    body = match.group("body")
    replaced, count = re.subn(rf"%{re.escape(old)}(?![A-Za-z0-9_.-])", f"%{new}", body)
    if count == 0:
        raise ValueError(f"external user %{user} does not consume %{old}")
    return hlo_text[: match.start("body")] + replaced + hlo_text[match.end("body") :]


def _called_computation_name(attributes: str) -> str | None:
    match = _CALLED_COMPUTATION.search(attributes)
    return match.group("name") if match else None


def _reduction_dimensions(attributes: str) -> tuple[int, ...]:
    match = _DIMENSIONS.search(attributes)
    if match is None:
        raise ValueError("Fold has no explicit reduction dimensions")
    values = match.group("values")
    return tuple(int(value) for value in values.split(",") if value)


def _constant_value(instruction: HloInstruction) -> float:
    match = _CONSTANT.search(instruction.attributes)
    if match is None:
        raise ValueError(f"constant %{instruction.name} has no scalar literal")
    return float(match.group("value"))


def _shape_signature(shape: str) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    match = _ARRAY_SHAPE.fullmatch(shape.strip())
    if match is None or match.group("layout") is None:
        raise ValueError(f"unsupported physical HLO array shape: {shape!r}")
    dimensions = tuple(int(value) for value in match.group("dims").split(",") if value)
    layout = tuple(int(value) for value in match.group("layout").split(",") if value)
    if tuple(sorted(layout)) != tuple(range(len(dimensions))):
        raise ValueError(f"physical layout is not a rank-{len(dimensions)} permutation: {layout}")
    return match.group("dtype"), dimensions, layout


def _hlo_shape(dtype: str, dimensions: tuple[int, ...]) -> str:
    dims = ",".join(str(value) for value in dimensions)
    layout = ",".join(str(value) for value in reversed(range(len(dimensions))))
    return f"{dtype}[{dims}]{{{layout}}}"


def _dtype(dtype: str) -> DType:
    try:
        return {"bf16": DType.BF16, "f32": DType.FP32}[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported axis-Fold dtype {dtype!r}") from error
