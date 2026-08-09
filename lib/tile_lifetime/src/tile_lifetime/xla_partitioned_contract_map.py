# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover generated scalar finalizations on demand-sliced XLA Contracts."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from tile_lifetime.cast_scalar_program import CastScalarProgram
from tile_lifetime.partitioned_gemm_program import (
    AccumulatorPartition,
    GeneratedPartitionedGemmFinalization,
    PartitionedGemmProgram,
    PassthroughPartitionFinalization,
    ScalarPartitionFinalization,
    generate_partitioned_gemm_finalization,
)
from tile_lifetime.xla_demand_sliced_contract import (
    DemandSlicedContractFfiCall,
    plan_demand_sliced_contract_ffi,
)
from tile_lifetime.xla_hlo_recovery import (
    EntryRegionValue,
    HloComputation,
    HloInstruction,
    InlinedHloGraph,
    inline_elementwise_fusions,
    parse_hlo_module_text,
)
from tile_lifetime.xla_scalar_map_import import import_hlo_scalar_map

_ARRAY_SHAPE = re.compile(r"(?P<dtype>[A-Za-z0-9]+)\[(?P<dims>[0-9,]*)\](?P<layout>\{[^}]+\})?")
_INSTRUCTION_DEFINITION = re.compile(r"^(?P<indent>\s*)%(?P<name>[^ ]+) = ")
_LOCAL_MAP_OPCODES = frozenset(
    {
        "abs",
        "add",
        "bitcast",
        "broadcast",
        "compare",
        "convert",
        "copy",
        "divide",
        "exponential",
        "maximum",
        "minimum",
        "multiply",
        "negate",
        "reshape",
        "rsqrt",
        "select",
        "subtract",
        "tanh",
        "transpose",
    }
)
_COLLECTIVE_OPCODES = frozenset(
    {
        "all-gather",
        "all-reduce",
        "all-to-all",
        "collective-broadcast",
        "collective-permute",
        "reduce-scatter",
    }
)


@dataclass(frozen=True)
class AttachedPartitionedContractFamily:
    """One source-independent physical program and its static typed-FFI target."""

    target: str
    program: PartitionedGemmProgram
    generated: GeneratedPartitionedGemmFinalization
    call_names: tuple[str, ...]


@dataclass(frozen=True)
class AttachedPartitionedContractCall:
    """One exact natural-HLO Contract/Map boundary."""

    call_name: str
    base: DemandSlicedContractFfiCall
    family_index: int
    map_output: EntryRegionValue
    mapped_partitions: tuple[int, ...]
    passthrough_outputs: tuple[EntryRegionValue, ...]
    internal_instructions: tuple[str, ...]

    @property
    def outputs(self) -> tuple[EntryRegionValue, ...]:
        """Return generated Map output followed by independent partitions."""
        return (self.map_output, *self.passthrough_outputs)


@dataclass(frozen=True)
class AttachedPartitionedContractPlan:
    """All safely attachable partition Maps in one natural HLO module."""

    families: tuple[AttachedPartitionedContractFamily, ...]
    calls: tuple[AttachedPartitionedContractCall, ...]

    @property
    def target_occurrences(self) -> tuple[tuple[str, int], ...]:
        """Return deterministic static target multiplicities."""
        counts = Counter(self.families[call.family_index].target for call in self.calls)
        return tuple((family.target, counts[family.target]) for family in self.families)


@dataclass(frozen=True)
class AttachedPartitionedContractAudit:
    """Evidence that exact users and placement boundaries survive replacement."""

    call_names: tuple[str, ...]
    target_occurrences: tuple[tuple[str, int], ...]
    removed_contract_count: int
    removed_contract_flops: int
    output_users: tuple[tuple[str, tuple[str, ...]], ...]
    collective_instructions: tuple[str, ...]
    copy_count: tuple[int, int]
    transpose_count: tuple[int, int]


@dataclass(frozen=True)
class _Candidate:
    base: DemandSlicedContractFfiCall
    map_output: EntryRegionValue
    mapped_partitions: tuple[int, ...]
    passthrough_outputs: tuple[EntryRegionValue, ...]
    internal_instructions: tuple[str, ...]
    program: PartitionedGemmProgram


def plan_attached_partitioned_contract_maps(
    hlo_text: str,
    *,
    target_prefix: str,
) -> AttachedPartitionedContractPlan:
    """Recover exclusive scalar Maps and form generic partitioned GEMM families."""
    module = parse_hlo_module_text(hlo_text)
    entry = module.computation(module.entry)
    graph = inline_elementwise_fusions(module)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    users = _users(entry)
    source_order = {instruction.name: index for index, instruction in enumerate(entry.instructions)}
    base = plan_demand_sliced_contract_ffi(hlo_text, target_prefix=f"{target_prefix}.structural")
    candidates = tuple(
        candidate
        for call in base.calls
        for candidate in (_attached_candidate(call, graph, instructions, users, source_order),)
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
        AttachedPartitionedContractCall(
            call_name=f"shuttle.generated.partitioned_contract_map.{index}",
            base=candidate.base,
            family_index=family_index,
            map_output=candidate.map_output,
            mapped_partitions=candidate.mapped_partitions,
            passthrough_outputs=candidate.passthrough_outputs,
            internal_instructions=candidate.internal_instructions,
        )
        for index, (candidate, family_index) in enumerate(zip(candidates, family_indices, strict=True))
    )
    families = tuple(
        AttachedPartitionedContractFamily(
            target=f"{target_prefix}.{program.semantic_digest[:16]}",
            program=program,
            generated=generate_partitioned_gemm_finalization(program),
            call_names=tuple(call.call_name for call in calls if call.family_index == index),
        )
        for index, program in enumerate(programs)
    )
    return AttachedPartitionedContractPlan(families=families, calls=calls)


def replace_attached_partitioned_contract_maps(
    hlo_text: str,
    plan: AttachedPartitionedContractPlan,
) -> str:
    """Replace exact Contract/Map regions with structural typed-FFI calls.

    The result is not an accepted compute path until a generic backend embeds
    the generated scalar bodies in one measured partition-aware mainloop.
    """
    family_by_call = {call_name: family for family in plan.families for call_name in family.call_names}
    all_internal = set().union(*(set(call.internal_instructions) for call in plan.calls))
    replacement_blocks = {
        call.base.recovered.entry_instruction: _replacement_lines(call, family_by_call[call.call_name])
        for call in plan.calls
    }
    if len(replacement_blocks) != len(plan.calls):
        raise ValueError("two attached partition Maps selected the same physical Contract")
    emitted: set[str] = set()
    rewritten: list[str] = []
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
        raise ValueError(f"failed to emit attached partition calls at {sorted(set(replacement_blocks) - emitted)}")
    transformed = "".join(rewritten)
    parse_hlo_module_text(transformed)
    return transformed


def audit_attached_partitioned_contract_maps(
    original_hlo: str,
    transformed_hlo: str,
    plan: AttachedPartitionedContractPlan,
) -> AttachedPartitionedContractAudit:
    """Verify generated targets, dead old arithmetic, users, and collectives."""
    original = _entry(original_hlo)
    transformed = _entry(transformed_hlo)
    original_users = _users(original)
    transformed_users = _users(transformed)
    transformed_instructions = {instruction.name: instruction for instruction in transformed.instructions}
    family_by_call = {call_name: family for family in plan.families for call_name in family.call_names}
    output_users: list[tuple[str, tuple[str, ...]]] = []
    for call in plan.calls:
        family = family_by_call[call.call_name]
        physical_call = transformed_instructions.get(call.call_name)
        if physical_call is None or physical_call.opcode != "custom-call":
            raise ValueError(f"missing attached partition call %{call.call_name}")
        if f'custom_call_target="{family.target}"' not in physical_call.attributes:
            raise ValueError(f"attached partition call %{call.call_name} has the wrong target")
        if physical_call.operands != tuple(value.instruction for value in call.base.inputs):
            raise ValueError(f"attached partition call %{call.call_name} changed its inputs")
        logical_outputs = {output.instruction for output in call.outputs}
        for internal in call.internal_instructions:
            if internal in transformed_instructions and internal not in logical_outputs:
                raise ValueError(f"old attached partition arithmetic %{internal} remains live")
        for output_index, output in enumerate(call.outputs):
            actual = transformed_instructions.get(output.instruction)
            if actual is None or actual.opcode != "get-tuple-element" or actual.operands != (call.call_name,):
                raise ValueError(f"generated output %{output.instruction} is not extracted from its call")
            if f"index={output_index}" not in actual.attributes or actual.shape != output.shape:
                raise ValueError(f"generated output %{output.instruction} changed index or layout")
            if transformed_users[output.instruction] != original_users[output.instruction]:
                raise ValueError(f"generated output %{output.instruction} changed downstream users")
            output_users.append((output.instruction, transformed_users[output.instruction]))
    original_collectives = _collectives(original)
    transformed_collectives = _collectives(transformed)
    if transformed_collectives != original_collectives:
        raise ValueError("attached partition replacement changed an external collective")
    occurrences = Counter(
        family.target
        for instruction in transformed.instructions
        for family in plan.families
        if instruction.opcode == "custom-call" and f'custom_call_target="{family.target}"' in instruction.attributes
    )
    target_occurrences = tuple((family.target, occurrences[family.target]) for family in plan.families)
    if target_occurrences != plan.target_occurrences:
        raise ValueError("attached partition target multiplicity changed")
    return AttachedPartitionedContractAudit(
        call_names=tuple(call.call_name for call in plan.calls),
        target_occurrences=target_occurrences,
        removed_contract_count=len(plan.calls),
        removed_contract_flops=sum(call.base.recovered.flops for call in plan.calls),
        output_users=tuple(output_users),
        collective_instructions=tuple(name for name, _, _ in transformed_collectives),
        copy_count=(_opcode_count(original, "copy"), _opcode_count(transformed, "copy")),
        transpose_count=(_opcode_count(original, "transpose"), _opcode_count(transformed, "transpose")),
    )


def _attached_candidate(
    call: DemandSlicedContractFfiCall,
    graph: InlinedHloGraph,
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
    source_order: dict[str, int],
) -> _Candidate | None:
    candidates: list[_Candidate] = []
    for opportunity in call.cross_partition_maps:
        mapped = opportunity.source_partitions
        if len(mapped) < 2 or not _has_nonlocal_user(opportunity.instruction, instructions, users):
            continue
        source_names = tuple(call.outputs[index].instruction for index in mapped)
        closure = _source_dependent_closure(opportunity.instruction, frozenset(source_names), instructions)
        if frozenset(source_names) - closure:
            continue
        if any(instructions[name].opcode not in _LOCAL_MAP_OPCODES | {"slice"} for name in closure):
            continue
        if any(user not in closure for name in closure - {opportunity.instruction} for user in users.get(name, ())):
            continue
        source_shapes = tuple(_array_shape(call.outputs[index].shape) for index in mapped)
        target_shape = _array_shape(instructions[opportunity.instruction].shape)
        if any(shape is None for shape in source_shapes) or target_shape is None:
            continue
        assert all(shape is not None for shape in source_shapes)
        source_domains = {_flattened_domain(shape[1]) for shape in source_shapes if shape is not None}
        if len(source_domains) != 1 or _flattened_domain(target_shape[1]) not in source_domains:
            continue
        try:
            scalar_program = import_hlo_scalar_map(
                graph,
                source_nodes=tuple(graph.entry_value(name) for name in source_names),
                target_node=graph.entry_value(opportunity.instruction),
            )
        except (KeyError, ValueError):
            continue
        if len(scalar_program.inputs) != len(mapped):
            continue
        passthrough_indices = tuple(index for index in range(len(call.outputs)) if index not in mapped)
        passthroughs = tuple(call.outputs[index] for index in passthrough_indices)
        program = _physical_program(
            call,
            mapped=mapped,
            passthrough_indices=passthrough_indices,
            scalar_program=scalar_program,
            scalar_output_shape=instructions[opportunity.instruction].shape,
        )
        concatenation = instructions[
            instructions[call.recovered.entry_instruction].operands[call.recovered.concatenated_operand]
        ]
        internal = {
            concatenation.name,
            call.recovered.entry_instruction,
            *(output.instruction for output in call.outputs),
            *closure,
        }
        candidates.append(
            _Candidate(
                base=call,
                map_output=EntryRegionValue(opportunity.instruction, instructions[opportunity.instruction].shape),
                mapped_partitions=mapped,
                passthrough_outputs=passthroughs,
                internal_instructions=tuple(sorted(internal, key=source_order.__getitem__)),
                program=program,
            )
        )
    if len(candidates) > 1:
        raise ValueError(
            f"demand-sliced Contract %{call.recovered.entry_instruction} has multiple exclusive scalar finalizations"
        )
    return candidates[0] if candidates else None


def _physical_program(
    call: DemandSlicedContractFfiCall,
    *,
    mapped: tuple[int, ...],
    passthrough_indices: tuple[int, ...],
    scalar_program: CastScalarProgram,
    scalar_output_shape: str,
) -> PartitionedGemmProgram:
    region = call.recovered
    output = _array_shape(region.contract.output_shape)
    if output is None or call.family.dimensions.lhs_batch or call.family.dimensions.rhs_batch:
        raise ValueError("the first partitioned GEMM finalization requires an unbatched array Contract")
    output_dimensions = output[1]
    n = output_dimensions[region.output_partition_axis]
    m = math.prod(extent for axis, extent in enumerate(output_dimensions) if axis != region.output_partition_axis)
    if len(call.family.dimensions.lhs_contracting) != 1 or len(call.family.dimensions.rhs_contracting) != 1:
        raise ValueError("the first partitioned GEMM finalization requires one contracting dimension")
    if region.concatenated_operand == 1:
        other_shape = _array_shape(call.inputs[0].shape)
        contracting_axis = call.family.dimensions.lhs_contracting[0]
    else:
        other_shape = _array_shape(call.inputs[-1].shape)
        contracting_axis = call.family.dimensions.rhs_contracting[0]
    if other_shape is None:
        raise ValueError("partitioned GEMM shared operand is not an array")
    k = other_shape[1][contracting_axis]
    partitions = tuple(
        AccumulatorPartition(
            start=partition.output_start,
            limit=partition.output_limit,
            result_shape=partition.output.shape,
        )
        for partition in region.partitions
    )
    return PartitionedGemmProgram(
        shape=(m, n, k),
        partitioned_operand=region.concatenated_operand,
        operand_shapes=tuple(value.shape for value in call.inputs),
        partitions=partitions,
        scalar_finalizations=(
            ScalarPartitionFinalization(
                source_partitions=mapped,
                program=scalar_program,
                output_shape=scalar_output_shape,
            ),
        ),
        passthrough_finalizations=tuple(
            PassthroughPartitionFinalization(
                source_partition=index,
                output_shape=call.outputs[index].shape,
            )
            for index in passthrough_indices
        ),
        input_dtype="bf16",
        accumulation_dtype="f32",
        partition_dtype="bf16",
        output_dtype="bf16",
        output_rounding="round_to_nearest_even",
    )


def _source_dependent_closure(
    target: str,
    sources: frozenset[str],
    instructions: dict[str, HloInstruction],
) -> frozenset[str]:
    memo: dict[str, bool] = {}

    def depends(name: str) -> bool:
        if name in memo:
            return memo[name]
        if name in sources:
            memo[name] = True
            return True
        memo[name] = False
        memo[name] = any(depends(operand) for operand in instructions[name].operands if operand in instructions)
        return memo[name]

    closure: set[str] = set()

    def visit(name: str) -> None:
        if name in closure or not depends(name):
            return
        closure.add(name)
        for operand in instructions[name].operands:
            if operand in instructions:
                visit(operand)

    visit(target)
    return frozenset(closure)


def _has_nonlocal_user(
    instruction: str,
    instructions: dict[str, HloInstruction],
    users: dict[str, tuple[str, ...]],
) -> bool:
    return any(instructions[user].opcode not in _LOCAL_MAP_OPCODES for user in users.get(instruction, ()))


def _replacement_lines(
    call: AttachedPartitionedContractCall,
    family: AttachedPartitionedContractFamily,
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


def _array_shape(shape: str) -> tuple[str, tuple[int, ...], str | None] | None:
    match = _ARRAY_SHAPE.fullmatch(shape)
    if match is None:
        return None
    return (
        match.group("dtype"),
        tuple(int(value) for value in match.group("dims").split(",") if value),
        match.group("layout"),
    )


def _flattened_domain(dimensions: tuple[int, ...]) -> tuple[int, int]:
    if not dimensions:
        raise ValueError("scalar partition Map requires at least one dimension")
    return math.prod(dimensions[:-1]), dimensions[-1]


def _entry(hlo_text: str) -> HloComputation:
    module = parse_hlo_module_text(hlo_text)
    return module.computation(module.entry)


def _users(entry: HloComputation) -> dict[str, tuple[str, ...]]:
    mutable: dict[str, list[str]] = {instruction.name: [] for instruction in entry.instructions}
    for instruction in entry.instructions:
        for operand in instruction.operands:
            mutable.setdefault(operand, []).append(instruction.name)
    return {name: tuple(dict.fromkeys(values)) for name, values in mutable.items()}


def _collectives(entry: HloComputation) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (instruction.name, instruction.shape, instruction.attributes)
        for instruction in entry.instructions
        if instruction.opcode in _COLLECTIVE_OPCODES
    )


def _opcode_count(entry: HloComputation, opcode: str) -> int:
    return sum(instruction.opcode == opcode for instruction in entry.instructions)
