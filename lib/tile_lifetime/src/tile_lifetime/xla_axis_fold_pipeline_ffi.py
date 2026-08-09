# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace a natural whole-entry HLO computation by generated Fold dataflow."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tile_lifetime.cuda_axis_fold_codegen import CudaAxisFoldFfiBuffer, GeneratedCudaAxisFoldFfi
from tile_lifetime.plan import NumericalPolicy
from tile_lifetime.xla_hlo_recovery import HloComputation, HloInstruction, parse_hlo_module_text

_PARAMETER_NUMBER = re.compile(r"parameter\((?P<number>\d+)\)")


@dataclass(frozen=True)
class AxisFoldPipelineHloInput:
    """One entry parameter bound by position to a generated pipeline input."""

    buffer_name: str
    instruction: str
    physical_shape: str


@dataclass(frozen=True)
class AxisFoldPipelineHloReplacementPlan:
    """Exact whole-entry boundary for one generated typed-FFI pipeline."""

    entry: str
    inputs: tuple[AxisFoldPipelineHloInput, ...]
    root_instruction: str
    output_shapes: tuple[str, ...]
    replaced_instructions: tuple[str, ...]
    numerical_policy: NumericalPolicy


@dataclass(frozen=True)
class AxisFoldPipelineHloReplacementAudit:
    """Post-rewrite proof that only the generated call remains live."""

    call_instruction: str
    dead_internal_instructions: tuple[str, ...]
    root_outputs: tuple[str, ...]
    copy_count: tuple[int, int]
    transpose_count: tuple[int, int]


def plan_axis_fold_pipeline_hlo_replacement(
    hlo_text: str,
    generated: GeneratedCudaAxisFoldFfi,
    *,
    numerical_policy: NumericalPolicy,
) -> AxisFoldPipelineHloReplacementPlan:
    """Validate a natural whole-entry boundary against generated FFI buffers."""
    if numerical_policy is NumericalPolicy.BITWISE_EXACT:
        raise ValueError("parallel axis-Fold replacement requires an explicit rounding-reorder numerical policy")
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    parameters = sorted(
        (instruction for instruction in entry.instructions if instruction.opcode == "parameter"),
        key=_parameter_number,
    )
    if len(parameters) != len(generated.inputs):
        raise ValueError(
            f"whole-entry axis-Fold replacement expected {len(generated.inputs)} parameters, " f"found {len(parameters)}"
        )
    inputs: list[AxisFoldPipelineHloInput] = []
    for parameter, buffer in zip(parameters, generated.inputs, strict=True):
        expected = _hlo_shape(buffer)
        if parameter.shape != expected:
            raise ValueError(
                f"axis-Fold input {buffer.name!r} requires physical shape {expected}, " f"found {parameter.shape}"
            )
        inputs.append(
            AxisFoldPipelineHloInput(
                buffer_name=buffer.name,
                instruction=parameter.name,
                physical_shape=parameter.shape,
            )
        )
    root = entry.root
    if root.opcode != "tuple" or len(root.operands) != len(generated.outputs):
        raise ValueError("whole-entry axis-Fold replacement requires a tuple root matching generated outputs")
    output_shapes = tuple(_hlo_shape(buffer) for buffer in generated.outputs)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    for operand, expected in zip(root.operands, output_shapes, strict=True):
        actual = instructions[operand].shape
        if actual != expected:
            raise ValueError(f"axis-Fold output requires physical shape {expected}, found {actual}")
    return AxisFoldPipelineHloReplacementPlan(
        entry=entry.name,
        inputs=tuple(inputs),
        root_instruction=root.name,
        output_shapes=output_shapes,
        replaced_instructions=tuple(
            instruction.name
            for instruction in entry.instructions
            if instruction.opcode != "parameter" and instruction.name != root.name
        ),
        numerical_policy=numerical_policy,
    )


def replace_axis_fold_pipeline_hlo_with_custom_call(
    hlo_text: str,
    plan: AxisFoldPipelineHloReplacementPlan,
    *,
    target: str,
) -> str:
    """Replace the entry root's complete dataflow with one generated call."""
    suffix = re.sub(r"[^A-Za-z0-9_.-]", "_", target)
    call_name = f"shuttle.generated.axis_fold.pipeline.{suffix}"
    output_names = tuple(
        f"shuttle.axis_fold.pipeline.output.{index}.{suffix}" for index in range(len(plan.output_shapes))
    )
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(plan.entry)
    reserved = {instruction.name for instruction in entry.instructions}
    collision = reserved & {call_name, *output_names}
    if collision:
        raise ValueError(f"axis-Fold pipeline replacement names already exist: {sorted(collision)}")
    root_pattern = re.compile(
        rf"^(?P<indent>\s*)ROOT\s+%?{re.escape(plan.root_instruction)} = .*?$",
        re.MULTILINE,
    )
    matches = tuple(root_pattern.finditer(hlo_text))
    if len(matches) != 1:
        raise ValueError(f"expected one root definition for {plan.root_instruction!r}")
    indent = matches[0].group("indent")
    result_shape = f"({', '.join(plan.output_shapes)})"
    operands = ", ".join(f"%{value.instruction}" for value in plan.inputs)
    constraints = ", ".join(value.physical_shape for value in plan.inputs)
    lines = [
        f"{indent}%{call_name} = {result_shape} custom-call({operands}), "
        f'custom_call_target="{target}", operand_layout_constraints={{{constraints}}}, '
        "api_version=API_VERSION_TYPED_FFI, backend_config={}",
    ]
    for index, (name, shape) in enumerate(zip(output_names, plan.output_shapes, strict=True)):
        lines.append(f"{indent}%{name} = {shape} get-tuple-element(%{call_name}), index={index}")
    root_operands = ", ".join(f"%{name}" for name in output_names)
    lines.append(f"{indent}ROOT %{plan.root_instruction} = {result_shape} tuple({root_operands})")
    rewritten = hlo_text[: matches[0].start()] + "\n".join(lines) + hlo_text[matches[0].end() :]
    parse_hlo_module_text(rewritten)
    return rewritten


def audit_axis_fold_pipeline_hlo_replacement(
    original_hlo: str,
    transformed_hlo: str,
    plan: AxisFoldPipelineHloReplacementPlan,
    *,
    target: str,
) -> AxisFoldPipelineHloReplacementAudit:
    """Prove old entry dataflow is unreachable and layout op counts are stable."""
    original_entry = parse_hlo_module_text(original_hlo).computation(plan.entry)
    transformed_entry = parse_hlo_module_text(transformed_hlo).computation(plan.entry)
    target_attribute = f'custom_call_target="{target}"'
    calls = tuple(
        instruction
        for instruction in transformed_entry.instructions
        if instruction.opcode == "custom-call" and target_attribute in instruction.attributes
    )
    if len(calls) != 1:
        raise ValueError(f"expected one post-roundtrip axis-Fold pipeline call for {target!r}, found {len(calls)}")
    call = calls[0]
    expected_operands = tuple(value.instruction for value in plan.inputs)
    if call.operands != expected_operands:
        raise ValueError(f"axis-Fold pipeline operands changed: expected {expected_operands}, found {call.operands}")
    expected_shape = f"({', '.join(plan.output_shapes)})"
    if call.shape != expected_shape:
        raise ValueError(f"axis-Fold pipeline result shape changed: expected {expected_shape}, found {call.shape}")
    expected_constraints = f"operand_layout_constraints={{{', '.join(value.physical_shape for value in plan.inputs)}}}"
    if expected_constraints not in call.attributes:
        raise ValueError("axis-Fold pipeline operand layout constraints changed")
    if "api_version=API_VERSION_TYPED_FFI" not in call.attributes:
        raise ValueError("axis-Fold pipeline call no longer uses the typed-FFI API")
    live = _live_instructions(transformed_entry)
    remaining = tuple(name for name in plan.replaced_instructions if name in live)
    if remaining:
        raise ValueError(f"old axis-Fold pipeline values remain live: {remaining}")
    root_outputs = transformed_entry.root.operands
    if len(root_outputs) != len(plan.output_shapes):
        raise ValueError("rewritten axis-Fold pipeline root has the wrong arity")
    transformed_instructions = {instruction.name: instruction for instruction in transformed_entry.instructions}
    for index, (output_name, expected_output_shape) in enumerate(zip(root_outputs, plan.output_shapes, strict=True)):
        output = transformed_instructions[output_name]
        if output.shape != expected_output_shape:
            raise ValueError(
                f"axis-Fold pipeline output {index} shape changed: "
                f"expected {expected_output_shape}, found {output.shape}"
            )
        if output.opcode != "get-tuple-element" or output.operands != (call.name,):
            raise ValueError(f"axis-Fold pipeline output {index} is not extracted from the generated call")
        if f"index={index}" not in output.attributes:
            raise ValueError(f"axis-Fold pipeline output {index} has the wrong tuple index")

    def count(entry: HloComputation, opcode: str) -> int:
        return sum(instruction.opcode == opcode for instruction in entry.instructions)

    copy_count = (count(original_entry, "copy"), count(transformed_entry, "copy"))
    transpose_count = (count(original_entry, "transpose"), count(transformed_entry, "transpose"))
    if copy_count[1] > copy_count[0]:
        raise ValueError(f"axis-Fold replacement added copies: {copy_count[0]} -> {copy_count[1]}")
    if transpose_count[1] > transpose_count[0]:
        raise ValueError(f"axis-Fold replacement added transposes: {transpose_count[0]} -> {transpose_count[1]}")
    return AxisFoldPipelineHloReplacementAudit(
        call_instruction=call.name,
        dead_internal_instructions=plan.replaced_instructions,
        root_outputs=root_outputs,
        copy_count=copy_count,
        transpose_count=transpose_count,
    )


def _parameter_number(instruction: HloInstruction) -> int:
    match = _PARAMETER_NUMBER.search(instruction.attributes)
    if match is None:
        raise ValueError(f"parameter %{instruction.name} lacks a parameter number")
    return int(match.group("number"))


def _hlo_shape(buffer: CudaAxisFoldFfiBuffer) -> str:
    dtype = {"bf16": "bf16", "fp32": "f32"}[buffer.dtype.value]
    dimensions = ",".join(str(dimension) for dimension in buffer.shape)
    layout = ",".join(str(index) for index in reversed(range(buffer.rank)))
    return f"{dtype}[{dimensions}]{{{layout}}}" if buffer.rank else f"{dtype}[]"


def _live_instructions(entry: HloComputation) -> frozenset[str]:
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    live: set[str] = set()
    pending = [entry.root.name]
    while pending:
        name = pending.pop()
        if name in live:
            continue
        live.add(name)
        pending.extend(instructions[name].operands)
    return frozenset(live)
