# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference execution and HLO replacement for a shared Contract/multi-Map plan."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from tile_lifetime.cast_scalar_program import evaluate_cast_scalar_program
from tile_lifetime.xla_hlo_recovery import HloComputation, parse_hlo_module_text
from tile_lifetime.xla_relation_program_recovery import (
    SharedContractMapDependence,
    SharedContractMultiMapOperandRole,
    SharedContractMultiMapRegionRecord,
)

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]\{(?P<layout>[0-9,]+)\}")


@dataclass(frozen=True)
class SharedContractMultiMapReplacementAudit:
    """Post-roundtrip liveness and wiring evidence for one replacement."""

    custom_call_instruction: str
    operands: tuple[str, ...]
    outputs: tuple[str, ...]
    external_users: tuple[tuple[str, tuple[str, ...]], ...]
    removed_internal_instructions: tuple[str, ...]
    copy_count: tuple[int, int]
    transpose_count: tuple[int, int]


def replace_shared_contract_multi_map_region_with_custom_call(
    hlo_text: str,
    plan: SharedContractMultiMapRegionRecord,
    *,
    target: str,
) -> str:
    """Replace a shared Contract and both generated Maps with one tuple call."""
    if not plan.convex or not plan.topologically_insertable:
        raise ValueError("shared Contract/multi-Map region is not convex and topologically insertable")
    if len(plan.outputs) < 2:
        raise ValueError("shared Contract/multi-Map replacement requires at least two live outputs")
    rewritten = hlo_text
    for instruction in plan.boundary.internal_instructions:
        if instruction == plan.insertion_instruction:
            continue
        pattern = re.compile(rf"^\s*%{re.escape(instruction)} = .*?\n", re.MULTILINE)
        matches = tuple(pattern.finditer(rewritten))
        if len(matches) != 1:
            raise ValueError(f"expected one physical definition for internal instruction %{instruction}")
        rewritten = rewritten[: matches[0].start()] + rewritten[matches[0].end() :]
    insertion_pattern = re.compile(
        rf"^(?P<indent>\s*)%{re.escape(plan.insertion_instruction)} = .*?$",
        re.MULTILINE,
    )
    insertion_matches = tuple(insertion_pattern.finditer(rewritten))
    if len(insertion_matches) != 1:
        raise ValueError("expected one shared Contract insertion instruction")
    insertion = insertion_matches[0]
    operands = ", ".join(f"%{operand.value.instruction}" for operand in plan.operands)
    constraints = ", ".join(operand.value.shape for operand in plan.operands)
    output_shapes = ", ".join(output.value.shape for output in plan.outputs)
    call_name = "shuttle_generated_shared_contract_multi_map"
    lines = [
        f"{insertion.group('indent')}%{call_name} = ({output_shapes}) custom-call({operands}), "
        f'custom_call_target="{target}", operand_layout_constraints={{{constraints}}}, '
        "api_version=API_VERSION_TYPED_FFI, backend_config={}",
    ]
    lines.extend(
        f"{insertion.group('indent')}%{output.value.instruction} = {output.value.shape} "
        f"get-tuple-element(%{call_name}), index={index}"
        for index, output in enumerate(plan.outputs)
    )
    replacement = "\n".join(lines)
    transformed = rewritten[: insertion.start()] + replacement + rewritten[insertion.end() :]
    parse_hlo_module_text(transformed)
    return transformed


def audit_shared_contract_multi_map_replacement(
    original_hlo: str,
    transformed_hlo: str,
    plan: SharedContractMultiMapRegionRecord,
    *,
    target: str,
) -> SharedContractMultiMapReplacementAudit:
    """Verify exact operands, output consumers, and deletion of the old region."""
    original = parse_hlo_module_text(original_hlo).computation(parse_hlo_module_text(original_hlo).entry)
    transformed_module = parse_hlo_module_text(transformed_hlo)
    transformed = transformed_module.computation(transformed_module.entry)
    transformed_by_name = {instruction.name: instruction for instruction in transformed.instructions}
    users = _users(transformed)
    target_attribute = f'custom_call_target="{target}"'
    calls = tuple(
        instruction
        for instruction in transformed.instructions
        if instruction.opcode == "custom-call" and target_attribute in instruction.attributes
    )
    if len(calls) != 1:
        raise ValueError(f"expected one generated shared Contract/multi-Map call, found {len(calls)}")
    call = calls[0]
    expected_operands = tuple(operand.value.instruction for operand in plan.operands)
    if call.operands != expected_operands:
        raise ValueError(f"generated call operands changed: expected {expected_operands}, found {call.operands}")
    expected_users = dict(plan.boundary.external_users)
    output_names = tuple(output.value.instruction for output in plan.outputs)
    for index, output_name in enumerate(output_names):
        output = transformed_by_name[output_name]
        if output.opcode != "get-tuple-element" or output.operands != (call.name,):
            raise ValueError(f"generated output %{output_name} is not extracted from %{call.name}")
        if f"index={index}" not in output.attributes:
            raise ValueError(f"generated output %{output_name} has the wrong tuple index")
        if users[output_name] != expected_users[output_name]:
            raise ValueError(
                f"generated output %{output_name} users changed: "
                f"expected {expected_users[output_name]}, found {users[output_name]}"
            )
    removed = tuple(
        instruction for instruction in plan.boundary.internal_instructions if instruction not in output_names
    )
    survivors = tuple(instruction for instruction in removed if instruction in transformed_by_name)
    if survivors:
        raise ValueError(f"old shared Contract/multi-Map instructions remain: {survivors}")
    return SharedContractMultiMapReplacementAudit(
        custom_call_instruction=call.name,
        operands=call.operands,
        outputs=output_names,
        external_users=tuple((name, users[name]) for name in output_names),
        removed_internal_instructions=removed,
        copy_count=(_opcode_count(original, "copy"), _opcode_count(transformed, "copy")),
        transpose_count=(_opcode_count(original, "transpose"), _opcode_count(transformed, "transpose")),
    )


def evaluate_shared_contract_multi_map_plan(
    plan: SharedContractMultiMapRegionRecord,
    operands: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    """Execute the generic shared Contract and scalar Maps in reference order."""
    if len(operands) != len(plan.operands):
        raise ValueError("runtime operand count does not match the shared Contract/multi-Map plan")
    by_role = {binding.role: np.asarray(operands[index]) for index, binding in enumerate(plan.operands)}
    lhs = by_role[SharedContractMultiMapOperandRole.CONTRACT_LHS]
    rhs = by_role[SharedContractMultiMapOperandRole.CONTRACT_RHS]
    auxiliary = by_role[SharedContractMultiMapOperandRole.MAP_AUXILIARY]
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[1] != rhs.shape[0]:
        raise ValueError("shared Contract operands do not form one rank-two contraction")
    projection = _round_bf16_array(np.matmul(lhs.astype(np.float32), rhs.astype(np.float32)))
    results: list[np.ndarray] = []
    for output in plan.outputs:
        _, physical_shape, _ = _parse_shape(output.value.shape)
        validity = by_role[output.validity_role]
        if tuple(validity.shape) != physical_shape:
            raise ValueError(f"validity for {output.dependence.value} does not match its physical output")
        logical = np.zeros((output.logical_row_extent, output.logical_feature_extent), dtype=np.float32)
        for scalar_output in output.scalar_outputs:
            for row in range(output.logical_row_extent):
                for local_feature in range(scalar_output.feature_extent):
                    feature = scalar_output.feature_offset + local_feature
                    scalar_inputs: dict[str, float] = {}
                    for scalar_input in scalar_output.scalar_program.inputs:
                        if scalar_input.input_name is None or scalar_input.input_index is None:
                            raise ValueError("generated scalar Map input lacks a concrete index relation")
                        source_row = row + scalar_input.input_index.row_offset
                        source_feature = local_feature + scalar_input.input_index.feature_offset
                        if output.dependence is SharedContractMapDependence.CONTRACT_ONLY:
                            source = projection
                        elif scalar_input.input_name.startswith("input0"):
                            source = auxiliary
                        elif scalar_input.input_name.startswith("input1"):
                            source = projection
                        else:
                            raise ValueError(f"unknown auxiliary Map scalar source {scalar_input.input_name!r}")
                        scalar_inputs[scalar_input.input_name] = float(source[source_row, source_feature])
                    logical[row, feature] = evaluate_cast_scalar_program(
                        scalar_output.scalar_program,
                        scalar_inputs,
                    )
        segments, padded_rows, physical_features = physical_shape
        if physical_features != output.logical_feature_extent or padded_rows < output.logical_row_extent:
            raise ValueError("physical Map output does not preserve its logical feature domain")
        padded = np.zeros((padded_rows, physical_features), dtype=np.float32)
        padded[: output.logical_row_extent] = logical
        broadcast = np.broadcast_to(padded, (segments, padded_rows, physical_features))
        results.append(np.where(validity, broadcast, 0.0).astype(np.float32))
    return tuple(results)


def _parse_shape(shape: str) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    match = _ARRAY_SHAPE.fullmatch(shape)
    if match is None:
        raise ValueError(f"unsupported physical array shape {shape!r}")
    return (
        match.group("dtype"),
        tuple(int(value) for value in match.group("dims").split(",") if value),
        tuple(int(value) for value in match.group("layout").split(",")),
    )


def _round_bf16_array(value: np.ndarray) -> np.ndarray:
    values = np.asarray(value, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _users(entry: HloComputation) -> dict[str, tuple[str, ...]]:
    mutable: dict[str, list[str]] = {instruction.name: [] for instruction in entry.instructions}
    for instruction in entry.instructions:
        for operand in instruction.operands:
            mutable.setdefault(operand, []).append(instruction.name)
    return {name: tuple(values) for name, values in mutable.items()}


def _opcode_count(entry: HloComputation, opcode: str) -> int:
    return sum(instruction.opcode == opcode for instruction in entry.instructions)
