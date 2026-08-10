# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attach generic scalar-contribution Folds to partitioned Contracts."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from tile_lifetime.cast_scalar_program import CastScalarDType, CastScalarKind
from tile_lifetime.partitioned_gemm_program import (
    AccumulatorPartition,
    AuxiliaryPartitionFold,
    GeneratedPartitionedGemmFinalization,
    PartitionedGemmProgram,
    PartitionFoldReassociation,
    PassthroughPartitionFinalization,
    generate_partitioned_gemm_finalization,
)
from tile_lifetime.xla_demand_sliced_contract import DemandSlicedContractFfiCall, plan_demand_sliced_contract_ffi
from tile_lifetime.xla_hlo_recovery import (
    EntryRegionValue,
    HloComputation,
    HloInstruction,
    HloModuleGraph,
    InlinedHloGraph,
    inline_elementwise_fusions,
    parse_hlo_module_text,
)
from tile_lifetime.xla_scalar_map_import import import_hlo_scalar_computation, import_hlo_scalar_map

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\](?P<layout>\{[^}]+\})?")
_CALLED_COMPUTATION = re.compile(r"to_apply=%?(?P<name>[A-Za-z0-9_.-]+)")
_CONSTANT = re.compile(r"constant\((?P<value>[-+0-9.eE]+)\)")
_DIMENSIONS = re.compile(r"dimensions=\{(?P<values>[0-9,]*)\}")
_INSTRUCTION_DEFINITION = re.compile(r"^(?P<indent>\s*)%(?P<name>[^ ]+) = ")
_COLLECTIVE_OPCODES = frozenset(
    {"all-gather", "all-reduce", "all-to-all", "collective-broadcast", "collective-permute", "reduce-scatter"}
)


@dataclass(frozen=True)
class AttachedPartitionFoldFamily:
    """One source-independent Contract/Fold program and typed-FFI target."""

    target: str
    program: PartitionedGemmProgram
    generated: GeneratedPartitionedGemmFinalization
    call_names: tuple[str, ...]


@dataclass(frozen=True)
class AttachedPartitionFoldCall:
    """One exact natural-HLO Contract plus auxiliary Fold boundary."""

    call_name: str
    base: DemandSlicedContractFfiCall
    family_index: int
    fold_outputs: tuple[EntryRegionValue, ...]
    internal_instructions: tuple[str, ...]

    @property
    def outputs(self) -> tuple[EntryRegionValue, ...]:
        """Return retained partition outputs followed by Fold outputs."""
        return (*self.base.outputs, *self.fold_outputs)


@dataclass(frozen=True)
class AttachedPartitionFoldPlan:
    """All unambiguous auxiliary Fold attachments in one HLO module."""

    families: tuple[AttachedPartitionFoldFamily, ...]
    calls: tuple[AttachedPartitionFoldCall, ...]

    @property
    def target_occurrences(self) -> tuple[tuple[str, int], ...]:
        """Return deterministic generated target multiplicities."""
        counts = Counter(self.families[call.family_index].target for call in self.calls)
        return tuple((family.target, counts[family.target]) for family in self.families)


@dataclass(frozen=True)
class AttachedPartitionFoldAudit:
    """Exact liveness evidence for a Contract/Fold HLO replacement."""

    call_names: tuple[str, ...]
    target_occurrences: tuple[tuple[str, int], ...]
    removed_contract_count: int
    retained_partition_users: tuple[tuple[str, tuple[str, ...]], ...]
    fold_output_users: tuple[tuple[str, tuple[str, ...]], ...]
    collective_instructions: tuple[str, ...]


@dataclass(frozen=True)
class _RecoveredFold:
    output: EntryRegionValue
    fold: AuxiliaryPartitionFold
    removable_instructions: frozenset[str]


@dataclass(frozen=True)
class _Candidate:
    base: DemandSlicedContractFfiCall
    folds: tuple[_RecoveredFold, ...]
    program: PartitionedGemmProgram
    internal_instructions: tuple[str, ...]


def plan_attached_partition_folds(hlo_text: str, *, target_prefix: str) -> AttachedPartitionFoldPlan:
    """Recover unambiguous self-product Folds from partition outputs.

    The self-product restriction is a bounded candidate policy, not a physical
    workload dispatch key. Contribution and reducer bodies are imported as
    scalar ASTs, and partitions with multiple Fold frontiers are left intact.
    """
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    users = _users(entry)
    source_order = {instruction.name: index for index, instruction in enumerate(entry.instructions)}
    graph = inline_elementwise_fusions(module)
    base = plan_demand_sliced_contract_ffi(hlo_text, target_prefix=f"{target_prefix}.structural")
    candidates = tuple(
        candidate
        for call in base.calls
        for candidate in (_candidate(call, module, graph, instructions, users, source_order),)
        if candidate is not None
    )
    programs: list[PartitionedGemmProgram] = []
    family_indices: list[int] = []
    for candidate in candidates:
        try:
            family_index = programs.index(candidate.program)
        except ValueError:
            family_index = len(programs)
            programs.append(candidate.program)
        family_indices.append(family_index)
    calls = tuple(
        AttachedPartitionFoldCall(
            call_name=f"shuttle.generated.partitioned_contract_fold.{index}",
            base=candidate.base,
            family_index=family_index,
            fold_outputs=tuple(fold.output for fold in candidate.folds),
            internal_instructions=candidate.internal_instructions,
        )
        for index, (candidate, family_index) in enumerate(zip(candidates, family_indices, strict=True))
    )
    families = tuple(
        AttachedPartitionFoldFamily(
            target=f"{target_prefix}.{program.semantic_digest[:16]}",
            program=program,
            generated=generate_partitioned_gemm_finalization(program),
            call_names=tuple(call.call_name for call in calls if call.family_index == index),
        )
        for index, program in enumerate(programs)
    )
    return AttachedPartitionFoldPlan(families=families, calls=calls)


def replace_attached_partition_folds(hlo_text: str, plan: AttachedPartitionFoldPlan) -> str:
    """Replace selected Contract/Map/Fold regions with multi-result calls."""
    family_by_call = {call_name: family for family in plan.families for call_name in family.call_names}
    all_internal = set().union(*(set(call.internal_instructions) for call in plan.calls))
    replacement_blocks = {
        call.base.recovered.entry_instruction: _replacement_lines(call, family_by_call[call.call_name])
        for call in plan.calls
    }
    rewritten: list[str] = []
    emitted: set[str] = set()
    for line in hlo_text.splitlines(keepends=True):
        match = _INSTRUCTION_DEFINITION.match(line)
        name = match.group("name") if match is not None else None
        if name not in all_internal:
            rewritten.append(line)
            continue
        block = replacement_blocks.get(name)
        if block is None:
            continue
        indent = match.group("indent") if match is not None else ""
        rewritten.extend(f"{indent}{generated}\n" for generated in block)
        emitted.add(name)
    if emitted != set(replacement_blocks):
        raise ValueError(f"failed to emit attached Fold calls at {sorted(set(replacement_blocks) - emitted)}")
    transformed = "".join(rewritten)
    parse_hlo_module_text(transformed)
    return transformed


def audit_attached_partition_folds(
    original_hlo: str,
    transformed_hlo: str,
    plan: AttachedPartitionFoldPlan,
) -> AttachedPartitionFoldAudit:
    """Verify generated ABI, retained users, removed arithmetic, and collectives."""
    original_module = parse_hlo_module_text(original_hlo)
    original = original_module.computation(original_module.entry)
    transformed_module = parse_hlo_module_text(transformed_hlo)
    transformed = transformed_module.computation(transformed_module.entry)
    original_users = _users(original)
    transformed_users = _users(transformed)
    transformed_instructions = {instruction.name: instruction for instruction in transformed.instructions}
    family_by_call = {call_name: family for family in plan.families for call_name in family.call_names}
    retained_users: list[tuple[str, tuple[str, ...]]] = []
    fold_users: list[tuple[str, tuple[str, ...]]] = []
    for call in plan.calls:
        family = family_by_call[call.call_name]
        physical = transformed_instructions.get(call.call_name)
        if physical is None or physical.opcode != "custom-call":
            raise ValueError(f"missing attached partition Fold call %{call.call_name}")
        if f'custom_call_target="{family.target}"' not in physical.attributes:
            raise ValueError(f"attached partition Fold call %{call.call_name} has the wrong target")
        if physical.operands != tuple(value.instruction for value in call.base.inputs):
            raise ValueError(f"attached partition Fold call %{call.call_name} changed its inputs")
        logical_outputs = {output.instruction for output in call.outputs}
        for internal in call.internal_instructions:
            if internal in transformed_instructions and internal not in logical_outputs:
                raise ValueError(f"old attached partition/Fold arithmetic %{internal} remains live")
        for output_index, output in enumerate(call.outputs):
            actual = transformed_instructions.get(output.instruction)
            if actual is None or actual.opcode != "get-tuple-element" or actual.operands != (call.call_name,):
                raise ValueError(f"generated output %{output.instruction} is not extracted from its call")
            if f"index={output_index}" not in actual.attributes or actual.shape != output.shape:
                raise ValueError(f"generated output %{output.instruction} changed index or layout")
            if transformed_users[output.instruction] != original_users[output.instruction]:
                raise ValueError(f"generated output %{output.instruction} changed downstream users")
            record = (output.instruction, transformed_users[output.instruction])
            if output_index < len(call.base.outputs):
                retained_users.append(record)
            else:
                fold_users.append(record)
    original_collectives = _collectives(original)
    transformed_collectives = _collectives(transformed)
    if transformed_collectives != original_collectives:
        raise ValueError("attached partition Fold replacement changed an external collective")
    occurrences = Counter(
        family.target
        for instruction in transformed.instructions
        for family in plan.families
        if instruction.opcode == "custom-call" and f'custom_call_target="{family.target}"' in instruction.attributes
    )
    target_occurrences = tuple((family.target, occurrences[family.target]) for family in plan.families)
    if target_occurrences != plan.target_occurrences:
        raise ValueError("attached partition Fold target multiplicity changed")
    return AttachedPartitionFoldAudit(
        call_names=tuple(call.call_name for call in plan.calls),
        target_occurrences=target_occurrences,
        removed_contract_count=len(plan.calls),
        retained_partition_users=tuple(retained_users),
        fold_output_users=tuple(fold_users),
        collective_instructions=tuple(name for name, _, _ in transformed_collectives),
    )


def _candidate(
    call: DemandSlicedContractFfiCall,
    module: HloModuleGraph,
    graph: InlinedHloGraph,
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
    source_order: dict[str, int],
) -> _Candidate | None:
    fold_counts = Counter(index for opportunity in call.folds for index in opportunity.source_partitions)
    recovered: list[_RecoveredFold] = []
    for opportunity in call.folds:
        if len(opportunity.source_partitions) != 1:
            continue
        partition_index = opportunity.source_partitions[0]
        if fold_counts[partition_index] != 1:
            continue
        candidate = _recover_fold(
            call,
            partition_index,
            opportunity.instruction,
            module,
            graph,
            instructions,
            users,
        )
        if candidate is not None and candidate.fold.contribution.expression.kind is CastScalarKind.MULTIPLY:
            recovered.append(candidate)
    if not recovered or len({fold.fold.source_partition for fold in recovered}) != len(recovered):
        return None
    program = _physical_program(call, tuple(recovered))
    concatenation = instructions[
        instructions[call.recovered.entry_instruction].operands[call.recovered.concatenated_operand]
    ]
    internal = {
        concatenation.name,
        call.recovered.entry_instruction,
        *(output.instruction for output in call.outputs),
        *(name for fold in recovered for name in fold.removable_instructions),
        *(fold.output.instruction for fold in recovered),
    }
    return _Candidate(
        base=call,
        folds=tuple(recovered),
        program=program,
        internal_instructions=tuple(sorted(internal, key=source_order.__getitem__)),
    )


def _recover_fold(
    call: DemandSlicedContractFfiCall,
    partition_index: int,
    fold_name: str,
    module: HloModuleGraph,
    graph: InlinedHloGraph,
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
) -> _RecoveredFold | None:
    fold_instruction = instructions[fold_name]
    if len(fold_instruction.operands) != 2:
        return None
    contribution_name, initializer_name = fold_instruction.operands
    contribution_shape = _array_shape(instructions[contribution_name].shape)
    output_shape = _array_shape(fold_instruction.shape)
    if contribution_shape is None or output_shape is None:
        return None
    dimensions = _reduction_dimensions(fold_instruction.attributes)
    if dimensions != (len(contribution_shape[1]) - 1,) or contribution_shape[1][:-1] != output_shape[1]:
        return None
    source_name = call.outputs[partition_index].instruction
    view_source = _bf16_view_source(source_name, contribution_name, contribution_shape[1], instructions)
    if view_source is None:
        return None
    try:
        contribution = import_hlo_scalar_map(
            graph,
            source_nodes=(graph.entry_value(view_source),),
            target_node=graph.entry_value(contribution_name),
        )
        reducer_name = _called_computation_name(fold_instruction.attributes)
        if reducer_name is None:
            return None
        reducer = import_hlo_scalar_computation(module.computation(reducer_name))
    except (KeyError, ValueError):
        return None
    if contribution.expression.dtype is not CastScalarDType.F32 or reducer.expression.dtype is not CastScalarDType.F32:
        return None
    initializer = _constant_value(instructions[initializer_name])
    if initializer is None:
        return None
    closure = _dependent_ancestors(contribution_name, view_source, instructions)
    removable = frozenset(
        name for name in closure - {view_source} if set(users.get(name, ())).issubset(closure | {fold_name})
    )
    return _RecoveredFold(
        output=EntryRegionValue(fold_name, fold_instruction.shape),
        fold=AuxiliaryPartitionFold(
            source_partition=partition_index,
            input_shape=instructions[view_source].shape,
            contribution=contribution,
            reducer=reducer,
            initializer=initializer,
            output_shape=fold_instruction.shape,
            accumulator_dtype="f32",
            output_dtype="f32",
            reassociation=PartitionFoldReassociation.ALLOW_ROUNDING_REORDER,
        ),
        removable_instructions=removable,
    )


def _physical_program(
    call: DemandSlicedContractFfiCall,
    folds: tuple[_RecoveredFold, ...],
) -> PartitionedGemmProgram:
    region = call.recovered
    output = _array_shape(region.contract.output_shape)
    if output is None or call.family.dimensions.lhs_batch or call.family.dimensions.rhs_batch:
        raise ValueError("attached partition Fold requires an unbatched array Contract")
    output_dimensions = output[1]
    n = output_dimensions[region.output_partition_axis]
    m = math.prod(extent for axis, extent in enumerate(output_dimensions) if axis != region.output_partition_axis)
    if region.concatenated_operand == 1:
        other_shape = _array_shape(call.inputs[0].shape)
        contracting_axis = call.family.dimensions.lhs_contracting[0]
    else:
        other_shape = _array_shape(call.inputs[-1].shape)
        contracting_axis = call.family.dimensions.rhs_contracting[0]
    if other_shape is None:
        raise ValueError("attached partition Fold shared operand is not an array")
    partitions = tuple(
        AccumulatorPartition(partition.output_start, partition.output_limit, partition.output.shape)
        for partition in region.partitions
    )
    return PartitionedGemmProgram(
        shape=(m, n, other_shape[1][contracting_axis]),
        partitioned_operand=region.concatenated_operand,
        operand_shapes=tuple(value.shape for value in call.inputs),
        partitions=partitions,
        scalar_finalizations=(),
        passthrough_finalizations=tuple(
            PassthroughPartitionFinalization(index, output.shape) for index, output in enumerate(call.outputs)
        ),
        input_dtype="bf16",
        accumulation_dtype="f32",
        partition_dtype="bf16",
        output_dtype="bf16",
        output_rounding="round_to_nearest_even",
        auxiliary_folds=tuple(fold.fold for fold in folds),
    )


def _bf16_view_source(
    source: str,
    target: str,
    target_dimensions: tuple[int, ...],
    instructions: dict[str, HloInstruction],
) -> str | None:
    candidates = tuple(
        name
        for name in _dependent_ancestors(target, source, instructions)
        if (shape := _array_shape(instructions[name].shape)) is not None
        and shape[0] == "bf16"
        and shape[1] == target_dimensions
    )
    return candidates[0] if len(candidates) == 1 else None


def _dependent_ancestors(target: str, source: str, instructions: dict[str, HloInstruction]) -> frozenset[str]:
    memo: dict[str, bool] = {}

    def depends(name: str) -> bool:
        if name == source:
            return True
        if name in memo:
            return memo[name]
        memo[name] = False
        memo[name] = any(depends(operand) for operand in instructions[name].operands if operand in instructions)
        return memo[name]

    result: set[str] = set()

    def visit(name: str) -> None:
        if name in result or not depends(name):
            return
        result.add(name)
        for operand in instructions[name].operands:
            if operand in instructions:
                visit(operand)

    visit(target)
    return frozenset(result)


def _replacement_lines(
    call: AttachedPartitionFoldCall,
    family: AttachedPartitionFoldFamily,
) -> tuple[str, ...]:
    output_shapes = ", ".join(output.shape for output in call.outputs)
    operands = ", ".join(f"%{value.instruction}" for value in call.base.inputs)
    constraints = ", ".join(value.shape for value in call.base.inputs)
    return (
        f"%{call.call_name} = ({output_shapes}) custom-call({operands}), "
        f'custom_call_target="{family.target}", operand_layout_constraints={{{constraints}}}, '
        "api_version=API_VERSION_TYPED_FFI, backend_config={}",
        *(
            f"%{output.instruction} = {output.shape} get-tuple-element(%{call.call_name}), index={index}"
            for index, output in enumerate(call.outputs)
        ),
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


def _called_computation_name(attributes: str) -> str | None:
    match = _CALLED_COMPUTATION.search(attributes)
    return match.group("name") if match is not None else None


def _reduction_dimensions(attributes: str) -> tuple[int, ...]:
    match = _DIMENSIONS.search(attributes)
    if match is None:
        return ()
    return tuple(int(value) for value in match.group("values").split(",") if value)


def _constant_value(instruction: HloInstruction) -> float | None:
    if instruction.opcode != "constant":
        return None
    match = _CONSTANT.search(instruction.attributes)
    return float(match.group("value")) if match is not None else None


def _collectives(entry: HloComputation) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (instruction.name, instruction.shape, instruction.attributes)
        for instruction in entry.instructions
        if instruction.opcode in _COLLECTIVE_OPCODES
    )
