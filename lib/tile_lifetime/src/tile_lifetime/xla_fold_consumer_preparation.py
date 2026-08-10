# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover generated scalar preparation between a Fold and a consumer Contract."""

from __future__ import annotations

import re
from dataclasses import dataclass

from tile_lifetime.cast_scalar_program import (
    CastScalarDType,
    CastScalarExpression,
    CastScalarKind,
    CastScalarProgram,
    ScalarIndexRelation,
)
from tile_lifetime.xla_hlo_recovery import HloComputation, HloInstruction, parse_hlo_module_text
from tile_lifetime.xla_partitioned_contract_fold import AttachedPartitionFoldPlan

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\](?P<layout>\{[^}]+\})?")
_CONSTANT = re.compile(r"constant\((?P<value>true|false|[-+0-9.eE]+)\)")
_POST_PREPARATION_OPCODES = frozenset({"bitcast", "broadcast", "copy", "multiply", "reshape", "transpose"})


@dataclass(frozen=True)
class IndexTransformStep:
    """One physical shape/index transform retained around a scalar preparation."""

    instruction: str
    opcode: str
    input_shapes: tuple[str, ...]
    output_shape: str
    attributes: str


@dataclass(frozen=True)
class FoldConsumerPreparation:
    """One raw value plus Fold state prepared for a downstream Contract operand."""

    raw_partition: str
    fold_output: str
    prepared_value: str
    consumer_contract: str
    consumer_operand: int
    scalar_program: CastScalarProgram
    preparation_steps: tuple[IndexTransformStep, ...]
    consumer_steps: tuple[IndexTransformStep, ...]


@dataclass(frozen=True)
class FoldConsumerPreparationPlan:
    """All unambiguous Fold-to-Contract preparation attachments in an HLO module."""

    attachments: tuple[FoldConsumerPreparation, ...]


def plan_fold_consumer_preparations(
    hlo_text: str,
    fold_plan: AttachedPartitionFoldPlan,
) -> FoldConsumerPreparationPlan:
    """Recover scalar Fold finalization and its physical consumer index path.

    This pass is deliberately driven by data dependence. It accepts a BF16
    prepared value only when its scalar expression depends on exactly one raw
    partition and its attached FP32 Fold, and when a unique shape/local-map
    path reaches a Contract operand.
    """
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    users = _users(entry)
    source_order = {instruction.name: index for index, instruction in enumerate(entry.instructions)}
    attachments: list[FoldConsumerPreparation] = []
    for call in fold_plan.calls:
        for recovered, output in zip(
            fold_plan.families[call.family_index].program.auxiliary_folds,
            call.fold_outputs,
            strict=True,
        ):
            raw = call.base.outputs[recovered.source_partition].instruction
            candidates = _preparation_candidates(raw, output.instruction, instructions, users, source_order)
            if len(candidates) == 1:
                attachments.append(candidates[0])
    return FoldConsumerPreparationPlan(attachments=tuple(attachments))


def _preparation_candidates(
    raw: str,
    fold: str,
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
    source_order: dict[str, int],
) -> tuple[FoldConsumerPreparation, ...]:
    depends_raw = _dependency_predicate(raw, instructions)
    depends_fold = _dependency_predicate(fold, instructions)
    candidates: list[FoldConsumerPreparation] = []
    for prepared in instructions.values():
        shape = _array_shape(prepared.shape)
        if prepared.opcode != "convert" or shape is None or shape[0] != "bf16":
            continue
        if not depends_raw(prepared.name) or not depends_fold(prepared.name):
            continue
        try:
            program, preparation_names = _import_scalar_preparation(prepared.name, raw, fold, instructions)
        except ValueError:
            continue
        consumers = _contract_paths(prepared.name, instructions, users)
        if len(consumers) != 1:
            continue
        contract, operand, consumer_names = consumers[0]
        candidates.append(
            FoldConsumerPreparation(
                raw_partition=raw,
                fold_output=fold,
                prepared_value=prepared.name,
                consumer_contract=contract,
                consumer_operand=operand,
                scalar_program=program,
                preparation_steps=_steps(preparation_names, instructions, source_order),
                consumer_steps=_steps(consumer_names, instructions, source_order),
            )
        )
    return tuple(candidates)


def _import_scalar_preparation(
    target: str,
    raw: str,
    fold: str,
    instructions: dict[str, HloInstruction],
) -> tuple[CastScalarProgram, frozenset[str]]:
    sources = {raw: ("raw", CastScalarDType.BF16), fold: ("fold", CastScalarDType.F32)}
    memo: dict[str, CastScalarExpression] = {}
    visited: set[str] = set()

    def import_value(name: str) -> CastScalarExpression:
        if name in memo:
            return memo[name]
        instruction = instructions[name]
        visited.add(name)
        if name in sources:
            input_name, dtype = sources[name]
            result = CastScalarExpression(
                CastScalarKind.INPUT,
                dtype,
                input_name=input_name,
                input_index=ScalarIndexRelation(0, 0),
            )
        elif instruction.opcode == "constant":
            result = CastScalarExpression(
                CastScalarKind.CONSTANT,
                _dtype(instruction.shape),
                constant=_constant_value(instruction.attributes),
            )
        elif instruction.opcode in {"bitcast", "broadcast", "copy", "reshape"}:
            if len(instruction.operands) != 1:
                raise ValueError("shape/index preparation step must be unary")
            result = import_value(instruction.operands[0])
        elif instruction.opcode == "convert":
            result = CastScalarExpression(
                CastScalarKind.CONVERT,
                _dtype(instruction.shape),
                operands=(import_value(instruction.operands[0]),),
            )
        elif instruction.opcode == "rsqrt":
            result = CastScalarExpression(
                CastScalarKind.RSQRT,
                _dtype(instruction.shape),
                operands=(import_value(instruction.operands[0]),),
            )
        elif instruction.opcode in {"add", "subtract", "multiply", "divide"}:
            kind = {
                "add": CastScalarKind.ADD,
                "subtract": CastScalarKind.SUBTRACT,
                "multiply": CastScalarKind.MULTIPLY,
                "divide": CastScalarKind.DIVIDE,
            }[instruction.opcode]
            result = CastScalarExpression(
                kind,
                _dtype(instruction.shape),
                operands=tuple(import_value(operand) for operand in instruction.operands),
            )
        else:
            raise ValueError(f"unsupported preparation opcode {instruction.opcode!r}")
        memo[name] = result
        return result

    program = CastScalarProgram(import_value(target))
    leaves = {(leaf.input_name, leaf.dtype) for leaf in program.inputs}
    if leaves != {("raw", CastScalarDType.BF16), ("fold", CastScalarDType.F32)}:
        raise ValueError("preparation must consume exactly the raw partition and Fold output")
    if not _contains(program.expression, CastScalarKind.RSQRT):
        raise ValueError("Fold finalization candidate must contain a generated rsqrt")
    return program, frozenset(visited - {raw, fold})


def _contract_paths(
    source: str,
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
) -> tuple[tuple[str, int, frozenset[str]], ...]:
    results: list[tuple[str, int, frozenset[str]]] = []

    def visit(name: str, path: frozenset[str]) -> None:
        for user_name in users.get(name, ()):
            user = instructions[user_name]
            if user.opcode == "dot":
                for operand, operand_name in enumerate(user.operands):
                    if operand_name == name:
                        results.append((user.name, operand, path))
                continue
            if user.opcode not in _POST_PREPARATION_OPCODES:
                continue
            if user.opcode == "multiply" and any(
                operand != name and not _constant_derived(operand, instructions) for operand in user.operands
            ):
                continue
            visit(user.name, path | {user.name})

    visit(source, frozenset())
    unique = {(contract, operand, names) for contract, operand, names in results}
    return tuple(sorted(unique, key=lambda value: (value[0], value[1], tuple(sorted(value[2])))))


def _dependency_predicate(source: str, instructions: dict[str, HloInstruction]):
    memo: dict[str, bool] = {}

    def depends(name: str) -> bool:
        if name == source:
            return True
        if name in memo:
            return memo[name]
        memo[name] = False
        memo[name] = any(depends(operand) for operand in instructions[name].operands if operand in instructions)
        return memo[name]

    return depends


def _constant_derived(name: str, instructions: dict[str, HloInstruction]) -> bool:
    instruction = instructions[name]
    if instruction.opcode == "constant":
        return True
    if instruction.opcode not in {"bitcast", "broadcast", "convert", "copy", "reshape"}:
        return False
    return bool(instruction.operands) and all(
        _constant_derived(operand, instructions) for operand in instruction.operands
    )


def _steps(
    names: frozenset[str],
    instructions: dict[str, HloInstruction],
    source_order: dict[str, int],
) -> tuple[IndexTransformStep, ...]:
    return tuple(
        IndexTransformStep(
            instruction=name,
            opcode=instructions[name].opcode,
            input_shapes=tuple(instructions[operand].shape for operand in instructions[name].operands),
            output_shape=instructions[name].shape,
            attributes=instructions[name].attributes,
        )
        for name in sorted(names, key=source_order.__getitem__)
    )


def _users(entry: HloComputation) -> dict[str, tuple[str, ...]]:
    mutable: dict[str, list[str]] = {instruction.name: [] for instruction in entry.instructions}
    for instruction in entry.instructions:
        for operand in instruction.operands:
            mutable.setdefault(operand, []).append(instruction.name)
    return {name: tuple(dict.fromkeys(values)) for name, values in mutable.items()}


def _array_shape(shape: str) -> tuple[str, tuple[int, ...], str | None] | None:
    match = _ARRAY_SHAPE.fullmatch(shape)
    if match is None:
        return None
    return (
        match.group("dtype"),
        tuple(int(value) for value in match.group("dims").split(",") if value),
        match.group("layout"),
    )


def _dtype(shape: str) -> CastScalarDType:
    parsed = _array_shape(shape)
    scalar = re.match(r"(?P<dtype>[A-Za-z0-9]+)\[\]", shape) if parsed is None else None
    name = parsed[0] if parsed is not None else scalar.group("dtype") if scalar is not None else None
    if name is None:
        raise ValueError(f"unsupported preparation shape {shape!r}")
    return CastScalarDType(name)


def _constant_value(attributes: str) -> float | bool:
    match = _CONSTANT.search(attributes)
    if match is None:
        raise ValueError(f"unsupported preparation constant {attributes!r}")
    value = match.group("value")
    if value == "true":
        return True
    if value == "false":
        return False
    return float(value)


def _contains(expression: CastScalarExpression, kind: CastScalarKind) -> bool:
    return expression.kind is kind or any(_contains(operand, kind) for operand in expression.operands)
