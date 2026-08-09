# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import a pointwise HLO dataflow slice into a cast-aware scalar AST."""

from __future__ import annotations

import re

from tile_lifetime.cast_scalar_program import (
    CastScalarDType,
    CastScalarExpression,
    CastScalarKind,
    CastScalarProgram,
    ScalarIndexRelation,
)
from tile_lifetime.xla_hlo_recovery import HloComputation, InlinedHloGraph

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\]")
_SLICE_RANGES = re.compile(r"slice=\{(?P<ranges>[^}]*)\}")
_SLICE_RANGE = re.compile(r"\[(?P<start>-?[0-9]+):(?P<limit>-?[0-9]+)(?::(?P<stride>[0-9]+))?\]")
_CONSTANT = re.compile(r"constant\((?P<value>true|false|[-+0-9.eE]+)\)")
_PARAMETER_NUMBER = re.compile(r"parameter\((?P<number>[0-9]+)\)")


def import_hlo_scalar_map(
    graph: InlinedHloGraph,
    *,
    source_nodes: tuple[str, ...],
    target_node: str,
    concatenate_choices: dict[str, int] | None = None,
) -> CastScalarProgram:
    """Import scalar semantics between one Contract output and a Map output.

    Slice starts become affine source-coordinate offsets. Shape-only operations
    are accepted only when they preserve rank-two scalar coordinates. Every HLO
    convert remains an explicit AST node, including F32-to-BF16-to-F32 round
    trips.
    """
    nodes = {node.id: node for node in graph.nodes}
    if not source_nodes or len(set(source_nodes)) != len(source_nodes):
        raise ValueError("scalar Map sources must be nonempty and unique")
    source_indices = {node_id: index for index, node_id in enumerate(source_nodes)}
    for source_node in source_nodes:
        source_shape = _array_shape(nodes[source_node].shape)
        if source_shape is None or len(source_shape[1]) != 2:
            raise ValueError(f"scalar Map source must be a rank-two array: {nodes[source_node].shape}")
    selected_concatenation_operands = concatenate_choices or {}
    memo: dict[tuple[str, int, int], CastScalarExpression] = {}

    def import_node(node_id: str, row_offset: int, feature_offset: int) -> CastScalarExpression:
        key = (node_id, row_offset, feature_offset)
        if key in memo:
            return memo[key]
        node = nodes[node_id]
        dtype = _scalar_dtype(node.shape)
        if node_id in source_indices:
            source_index = source_indices[node_id]
            prefix = "input" if len(source_nodes) == 1 else f"input{source_index}"
            name = f"{prefix}_r{row_offset}_f{feature_offset}"
            result = CastScalarExpression(
                kind=CastScalarKind.INPUT,
                dtype=dtype,
                input_name=name,
                input_index=ScalarIndexRelation(row_offset=row_offset, feature_offset=feature_offset),
            )
        elif node.opcode == "constant":
            result = CastScalarExpression(
                kind=CastScalarKind.CONSTANT,
                dtype=dtype,
                constant=_constant_value(node.attributes),
            )
        elif node.opcode == "slice":
            if len(node.operands) != 1:
                raise ValueError(f"HLO slice {node.id!r} has {len(node.operands)} operands")
            starts = _slice_starts(node.attributes)
            if len(starts) != 2:
                raise ValueError(f"scalar Map slice {node.id!r} must have rank two")
            result = import_node(node.operands[0], row_offset + starts[0], feature_offset + starts[1])
        elif node.opcode == "concatenate":
            try:
                operand_index = selected_concatenation_operands[node.id]
            except KeyError as error:
                raise ValueError(f"concatenate {node.id!r} requires an output-segment choice") from error
            if operand_index < 0 or operand_index >= len(node.operands):
                raise ValueError(f"concatenate {node.id!r} has no operand {operand_index}")
            result = import_node(node.operands[operand_index], row_offset, feature_offset)
        elif node.opcode == "broadcast":
            source_shape = _array_shape(nodes[node.operands[0]].shape) if len(node.operands) == 1 else None
            if source_shape is None or source_shape[1]:
                raise ValueError(f"only scalar broadcasts are accepted in a scalar Map: {node.id!r}")
            result = import_node(node.operands[0], row_offset, feature_offset)
        elif node.opcode in {"bitcast", "copy", "reshape"}:
            if len(node.operands) != 1:
                raise ValueError(f"shape wrapper {node.id!r} must have one operand")
            source = _array_shape(nodes[node.operands[0]].shape)
            target = _array_shape(node.shape)
            if source is None or target is None or source[1] != target[1]:
                raise ValueError(f"shape-changing wrapper {node.id!r} needs an explicit index map")
            result = import_node(node.operands[0], row_offset, feature_offset)
        elif node.opcode == "convert":
            result = CastScalarExpression(
                kind=CastScalarKind.CONVERT,
                dtype=dtype,
                operands=(import_node(node.operands[0], row_offset, feature_offset),),
            )
        elif node.opcode in {"negate", "exponential", "tanh"}:
            kind = {
                "negate": CastScalarKind.NEGATE,
                "exponential": CastScalarKind.EXP,
                "tanh": CastScalarKind.TANH,
            }[node.opcode]
            result = CastScalarExpression(
                kind=kind,
                dtype=dtype,
                operands=(import_node(node.operands[0], row_offset, feature_offset),),
            )
        elif node.opcode in {"add", "subtract", "multiply", "divide"}:
            kind = {
                "add": CastScalarKind.ADD,
                "subtract": CastScalarKind.SUBTRACT,
                "multiply": CastScalarKind.MULTIPLY,
                "divide": CastScalarKind.DIVIDE,
            }[node.opcode]
            result = CastScalarExpression(
                kind=kind,
                dtype=dtype,
                operands=tuple(import_node(operand, row_offset, feature_offset) for operand in node.operands),
            )
        elif node.opcode == "select":
            result = CastScalarExpression(
                kind=CastScalarKind.SELECT,
                dtype=dtype,
                operands=tuple(import_node(operand, row_offset, feature_offset) for operand in node.operands),
            )
        else:
            raise ValueError(f"unsupported scalar Map HLO opcode {node.opcode!r} at {node.id!r}")
        memo[key] = result
        return result

    return CastScalarProgram(import_node(target_node, 0, 0))


def import_hlo_scalar_computation(computation: HloComputation) -> CastScalarProgram:
    """Import a scalar HLO computation such as a Fold reducer.

    Parameter names are replaced with stable positional names. This keeps the
    generated program independent of frontend or HLO instruction spelling
    while preserving every explicit conversion in the reducer body.
    """
    nodes = {instruction.name: instruction for instruction in computation.instructions}
    parameters = tuple(instruction for instruction in computation.instructions if instruction.opcode == "parameter")
    parameter_indices = {instruction.name: _parameter_number(instruction.attributes) for instruction in parameters}
    if set(parameter_indices.values()) != set(range(len(parameters))):
        raise ValueError(f"scalar computation {computation.name!r} parameters are not contiguous from zero")
    memo: dict[str, CastScalarExpression] = {}

    def import_node(node_id: str) -> CastScalarExpression:
        if node_id in memo:
            return memo[node_id]
        node = nodes[node_id]
        dtype = _scalar_dtype(node.shape)
        if node.opcode == "parameter":
            index = parameter_indices[node.name]
            result = CastScalarExpression(
                kind=CastScalarKind.INPUT,
                dtype=dtype,
                input_name=f"input{index}",
                input_index=ScalarIndexRelation(row_offset=0, feature_offset=0),
            )
        elif node.opcode == "constant":
            result = CastScalarExpression(
                kind=CastScalarKind.CONSTANT,
                dtype=dtype,
                constant=_constant_value(node.attributes),
            )
        elif node.opcode in {"bitcast", "copy", "reshape", "convert"}:
            if len(node.operands) != 1:
                raise ValueError(f"scalar wrapper {node.name!r} must have one operand")
            operand = import_node(node.operands[0])
            result = (
                CastScalarExpression(kind=CastScalarKind.CONVERT, dtype=dtype, operands=(operand,))
                if node.opcode == "convert"
                else operand
            )
        elif node.opcode in {"negate", "exponential", "tanh"}:
            kind = {
                "negate": CastScalarKind.NEGATE,
                "exponential": CastScalarKind.EXP,
                "tanh": CastScalarKind.TANH,
            }[node.opcode]
            result = CastScalarExpression(kind=kind, dtype=dtype, operands=(import_node(node.operands[0]),))
        elif node.opcode in {"add", "subtract", "multiply", "divide"}:
            kind = {
                "add": CastScalarKind.ADD,
                "subtract": CastScalarKind.SUBTRACT,
                "multiply": CastScalarKind.MULTIPLY,
                "divide": CastScalarKind.DIVIDE,
            }[node.opcode]
            result = CastScalarExpression(
                kind=kind,
                dtype=dtype,
                operands=tuple(import_node(operand) for operand in node.operands),
            )
        elif node.opcode == "select":
            result = CastScalarExpression(
                kind=CastScalarKind.SELECT,
                dtype=dtype,
                operands=tuple(import_node(operand) for operand in node.operands),
            )
        else:
            raise ValueError(f"unsupported scalar computation HLO opcode {node.opcode!r} at {node.name!r}")
        memo[node_id] = result
        return result

    return CastScalarProgram(import_node(computation.root.name))


def _array_shape(shape: str) -> tuple[str, tuple[int, ...]] | None:
    match = _ARRAY_SHAPE.match(shape.lstrip("("))
    if match is None:
        return None
    return match.group("dtype"), tuple(int(value) for value in match.group("dims").split(",") if value)


def _shape_dtype(shape: str) -> str:
    parsed = _array_shape(shape)
    if parsed is not None:
        return parsed[0]
    scalar = re.match(r"(?P<dtype>[A-Za-z0-9]+)\[\]", shape.lstrip("("))
    if scalar is None:
        raise ValueError(f"unsupported scalar shape {shape!r}")
    return scalar.group("dtype")


def _scalar_dtype(shape: str) -> CastScalarDType:
    dtype = _shape_dtype(shape)
    try:
        return CastScalarDType(dtype)
    except ValueError as error:
        raise ValueError(f"unsupported scalar Map dtype {dtype!r}") from error


def _slice_starts(attributes: str) -> tuple[int, ...]:
    match = _SLICE_RANGES.search(attributes)
    if match is None:
        raise ValueError(f"HLO slice has no index ranges: {attributes!r}")
    ranges = tuple(_SLICE_RANGE.finditer(match.group("ranges")))
    if any(value.group("stride") not in {None, "1"} for value in ranges):
        raise ValueError("strided scalar Map slices are unsupported")
    return tuple(int(value.group("start")) for value in ranges)


def _constant_value(attributes: str) -> float | bool:
    match = _CONSTANT.search(attributes)
    if match is None:
        raise ValueError(f"unsupported scalar constant: {attributes!r}")
    value = match.group("value")
    if value == "true":
        return True
    if value == "false":
        return False
    return float(value)


def _parameter_number(attributes: str) -> int:
    match = _PARAMETER_NUMBER.search(attributes)
    if match is None:
        raise ValueError(f"scalar computation parameter has no index: {attributes!r}")
    return int(match.group("number"))
