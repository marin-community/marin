# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover generic Contract/Map structure from post-SPMD XLA HLO text.

This module deliberately parses the stable, human-readable HLO dump instead of
depending on private ``jaxlib`` Python bindings.  It is an inspection path: it
does not rewrite the module and it does not use frontend metadata or model
names when identifying regions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_COMPUTATION_HEADER = re.compile(r"^(?P<entry>ENTRY )?%(?P<name>[^ ]+) .*\{$")
_INSTRUCTION_NAME = re.compile(r"^\s*(?P<root>ROOT )?%(?P<name>[^ ]+) = (?P<body>.*)$")
_VALUE_REFERENCE = re.compile(r"%([A-Za-z0-9_.-]+)")
_CALL_REFERENCE = re.compile(r"(?:calls|to_apply)=%([A-Za-z0-9_.-]+)")
_PARAMETER_NUMBER = re.compile(r"parameter\((\d+)\)")

_POINTWISE_OPCODES = frozenset(
    {
        "abs",
        "add",
        "broadcast",
        "compare",
        "constant",
        "convert",
        "copy",
        "divide",
        "exponential",
        "maximum",
        "minimum",
        "multiply",
        "negate",
        "rsqrt",
        "select",
        "subtract",
        "tanh",
    }
)
_WRAPPER_OPCODES = frozenset({"bitcast", "copy", "reshape"})


@dataclass(frozen=True)
class HloInstruction:
    """One parsed instruction from an HLO computation."""

    name: str
    opcode: str
    shape: str
    operands: tuple[str, ...]
    attributes: str
    is_root: bool = False

    @property
    def dtype(self) -> str:
        """Return the element type prefix of this instruction's shape."""
        shape = self.shape.lstrip("(")
        match = re.match(r"([A-Za-z0-9]+)(?:\[|\])", shape)
        return match.group(1) if match else "tuple"


@dataclass(frozen=True)
class HloComputation:
    """One parsed HLO computation in source order."""

    name: str
    instructions: tuple[HloInstruction, ...]
    is_entry: bool = False

    @property
    def root(self) -> HloInstruction:
        """Return the explicitly marked computation root."""
        roots = tuple(instruction for instruction in self.instructions if instruction.is_root)
        if len(roots) != 1:
            raise ValueError(f"HLO computation {self.name!r} has {len(roots)} roots")
        return roots[0]


@dataclass(frozen=True)
class HloModuleGraph:
    """Parsed computations and the selected entry computation."""

    computations: tuple[HloComputation, ...]
    entry: str

    def computation(self, name: str) -> HloComputation:
        """Look up one computation by exact HLO name."""
        for computation in self.computations:
            if computation.name == name:
                return computation
        raise KeyError(name)


@dataclass(frozen=True)
class InlinedHloNode:
    """One logical node after replacing fusion calls with their bodies."""

    id: str
    opcode: str
    shape: str
    operands: tuple[str, ...]
    attributes: str
    source_computation: str
    source_instruction: str

    @property
    def dtype(self) -> str:
        """Return the element type prefix of this node's shape."""
        shape = self.shape.lstrip("(")
        match = re.match(r"([A-Za-z0-9]+)(?:\[|\])", shape)
        return match.group(1) if match else "tuple"


@dataclass(frozen=True)
class WrapperBoundary:
    """A physical wrapper crossed while exposing a logical producer."""

    opcode: str
    source_shape: str
    result_shape: str

    @property
    def changes_dtype(self) -> bool:
        """Whether the wrapper crosses a finite-precision element boundary."""
        return _shape_dtype(self.source_shape) != _shape_dtype(self.result_shape)


@dataclass(frozen=True)
class StrippedHloValue:
    """A wrapper-free base value plus all wrappers in producer-to-user order."""

    base: str
    wrappers: tuple[WrapperBoundary, ...]


@dataclass(frozen=True)
class InlinedHloGraph:
    """Entry graph with ordinary fusion computations inlined."""

    nodes: tuple[InlinedHloNode, ...]
    entry_values: tuple[tuple[str, str], ...]

    def node(self, node_id: str) -> InlinedHloNode:
        """Look up one inlined node."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def entry_value(self, instruction_name: str) -> str:
        """Return the inlined value for one entry-computation instruction."""
        try:
            return dict(self.entry_values)[instruction_name]
        except KeyError as error:
            raise KeyError(instruction_name) from error

    def strip_wrappers(self, node_id: str) -> StrippedHloValue:
        """Expose a producer while retaining cast and layout boundaries."""
        wrappers: list[WrapperBoundary] = []
        current = self.node(node_id)
        while current.opcode in _WRAPPER_OPCODES | {"convert"} and len(current.operands) == 1:
            source = self.node(current.operands[0])
            wrappers.append(
                WrapperBoundary(
                    opcode=current.opcode,
                    source_shape=source.shape,
                    result_shape=current.shape,
                )
            )
            current = source
        return StrippedHloValue(base=current.id, wrappers=tuple(reversed(wrappers)))


@dataclass(frozen=True)
class ContractRecord:
    """One generic contraction found after inlining physical fusion wrappers."""

    node: str
    lhs: StrippedHloValue
    rhs: StrippedHloValue
    output_shape: str


@dataclass(frozen=True)
class RecoveredPairMapRegion:
    """Two shared-input Contracts feeding a generic pointwise Map."""

    left_contract: ContractRecord
    right_contract: ContractRecord
    shared_input: str
    map_root: str
    map_opcodes: tuple[str, ...]
    map_cast_boundaries: tuple[WrapperBoundary, ...]
    consumer_contracts: tuple[str, ...]


@dataclass(frozen=True)
class PairMapRecoveryReport:
    """Inspectable result of generic post-SPMD pair-Map analysis."""

    computation_count: int
    inlined_node_count: int
    contract_count: int
    regions: tuple[RecoveredPairMapRegion, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Encode the report using only JSON-compatible values."""

        def stripped(value: StrippedHloValue) -> dict[str, object]:
            return {
                "base": value.base,
                "wrappers": [boundary(boundary_value) for boundary_value in value.wrappers],
            }

        def boundary(value: WrapperBoundary) -> dict[str, object]:
            return {
                "opcode": value.opcode,
                "source_shape": value.source_shape,
                "result_shape": value.result_shape,
                "changes_dtype": value.changes_dtype,
            }

        def contract(value: ContractRecord) -> dict[str, object]:
            return {
                "node": value.node,
                "lhs": stripped(value.lhs),
                "rhs": stripped(value.rhs),
                "output_shape": value.output_shape,
            }

        return {
            "computation_count": self.computation_count,
            "inlined_node_count": self.inlined_node_count,
            "contract_count": self.contract_count,
            "regions": [
                {
                    "left_contract": contract(region.left_contract),
                    "right_contract": contract(region.right_contract),
                    "shared_input": region.shared_input,
                    "map_root": region.map_root,
                    "map_opcodes": list(region.map_opcodes),
                    "map_cast_boundaries": [boundary(value) for value in region.map_cast_boundaries],
                    "consumer_contracts": list(region.consumer_contracts),
                }
                for region in self.regions
            ],
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class RecoveredMultiOutputContractMapRegion:
    """One Contract whose pointwise users expose several boundary values."""

    contract: ContractRecord
    map_opcodes: tuple[str, ...]
    boundary: RecoveredEntryRegionBoundary
    consumer_contracts: tuple[str, ...]


@dataclass(frozen=True)
class MultiOutputContractMapRecoveryReport:
    """Generic post-SPMD Contract-to-multi-output-Map candidates."""

    computation_count: int
    inlined_node_count: int
    contract_count: int
    regions: tuple[RecoveredMultiOutputContractMapRegion, ...]


@dataclass(frozen=True)
class EntryRegionValue:
    """One physical entry-computation value on a recovered region boundary."""

    instruction: str
    shape: str


@dataclass(frozen=True)
class RecoveredEntryRegionBoundary:
    """Maximal pointwise region grown from a recovered Contract pair."""

    internal_instructions: tuple[str, ...]
    inputs: tuple[EntryRegionValue, ...]
    outputs: tuple[EntryRegionValue, ...]
    external_users: tuple[tuple[str, tuple[str, ...]], ...]
    has_explicit_sharding: bool
    has_side_effect: bool


def form_pair_map_entry_region(
    hlo_text: str,
    region: RecoveredPairMapRegion,
) -> RecoveredEntryRegionBoundary:
    """Grow a maximal entry-local pointwise region from two Contracts.

    Growth follows only physical dataflow and pointwise/wrapper opcodes. Other
    Contracts and reductions remain outside and become users of the region's
    outputs. Additional operands, including saved values and cotangents, become
    explicit region inputs.
    """
    module = parse_hlo_module_text(hlo_text)
    graph = inline_elementwise_fusions(module)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}

    def entry_instruction_for(node_id: str) -> str:
        prefix = f"{module.entry}/"
        if not node_id.startswith(prefix):
            raise ValueError(f"node {node_id!r} is outside the entry computation")
        name = node_id.removeprefix(prefix).split("/", 1)[0]
        if name not in instructions:
            raise ValueError(f"node {node_id!r} has no entry instruction boundary")
        return name

    seeds = {
        entry_instruction_for(region.left_contract.node),
        entry_instruction_for(region.right_contract.node),
    }
    return _form_entry_region(module, graph, entry, seeds)


def recover_multi_output_contract_map_regions(hlo_text: str) -> MultiOutputContractMapRecoveryReport:
    """Recover Contracts whose scalar descendants produce several live values.

    The recovery does not assume a particular activation or reverse rule. Side
    inputs to the Map remain explicit boundary values. A candidate must expose
    at least two same-shaped results, each consumed by a downstream Contract.
    """
    module = parse_hlo_module_text(hlo_text)
    graph = inline_elementwise_fusions(module)
    entry = module.computation(module.entry)
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    candidates: list[RecoveredMultiOutputContractMapRegion] = []
    contract_count = sum(node.opcode == "dot" for node in graph.nodes)
    for instruction in entry.instructions:
        root = graph.node(graph.entry_value(instruction.name))
        stripped = graph.strip_wrappers(root.id)
        contract_node = graph.node(stripped.base)
        if contract_node.opcode != "dot" or len(contract_node.operands) != 2:
            continue
        boundary = _form_entry_region(module, graph, entry, {instruction.name})
        if len(boundary.internal_instructions) == 1 or len(boundary.outputs) < 2:
            continue
        output_shapes = {output.shape for output in boundary.outputs}
        if len(output_shapes) != 1:
            continue
        consumer_contracts: set[str] = set()
        every_output_reaches_contract = True
        for output in boundary.outputs:
            reached = _first_entry_contract_users(output.instruction, instructions)
            every_output_reaches_contract &= bool(reached)
            consumer_contracts.update(reached)
        if not every_output_reaches_contract:
            continue
        map_opcodes = tuple(
            graph.node(graph.entry_value(name)).opcode
            for name in boundary.internal_instructions
            if name != instruction.name
        )
        candidates.append(
            RecoveredMultiOutputContractMapRegion(
                contract=ContractRecord(
                    node=contract_node.id,
                    lhs=graph.strip_wrappers(contract_node.operands[0]),
                    rhs=graph.strip_wrappers(contract_node.operands[1]),
                    output_shape=contract_node.shape,
                ),
                map_opcodes=map_opcodes,
                boundary=boundary,
                consumer_contracts=tuple(sorted(consumer_contracts)),
            )
        )
    return MultiOutputContractMapRecoveryReport(
        computation_count=len(module.computations),
        inlined_node_count=len(graph.nodes),
        contract_count=contract_count,
        regions=tuple(candidates),
    )


def _form_entry_region(
    module: HloModuleGraph,
    graph: InlinedHloGraph,
    entry: HloComputation,
    seeds: set[str],
) -> RecoveredEntryRegionBoundary:
    instructions = {instruction.name: instruction for instruction in entry.instructions}
    source_order = {instruction.name: index for index, instruction in enumerate(entry.instructions)}
    users: dict[str, list[str]] = {instruction.name: [] for instruction in entry.instructions}
    for instruction in entry.instructions:
        for operand in instruction.operands:
            users.setdefault(operand, []).append(instruction.name)

    internal = set(seeds)
    pending = list(seeds)
    while pending:
        producer = pending.pop()
        for user in users.get(producer, ()):
            root = graph.node(graph.entry_value(user))
            if root.opcode not in _POINTWISE_OPCODES | _WRAPPER_OPCODES:
                continue
            if user not in internal:
                internal.add(user)
                pending.append(user)

    ordered_internal = tuple(sorted(internal, key=source_order.__getitem__))
    input_names = {operand for name in internal for operand in instructions[name].operands if operand not in internal}
    output_names = {name for name in internal if any(user not in internal for user in users.get(name, ()))}
    ordered_inputs = tuple(sorted(input_names, key=source_order.__getitem__))
    ordered_outputs = tuple(sorted(output_names, key=source_order.__getitem__))
    return RecoveredEntryRegionBoundary(
        internal_instructions=ordered_internal,
        inputs=tuple(EntryRegionValue(name, instructions[name].shape) for name in ordered_inputs),
        outputs=tuple(EntryRegionValue(name, instructions[name].shape) for name in ordered_outputs),
        external_users=tuple(
            (name, tuple(user for user in users[name] if user not in internal)) for name in ordered_outputs
        ),
        has_explicit_sharding=any("sharding=" in instructions[name].attributes for name in internal),
        has_side_effect=any("custom_call_has_side_effect=true" in instructions[name].attributes for name in internal),
    )


def _first_entry_contract_users(
    instruction_name: str,
    instructions: dict[str, HloInstruction],
) -> tuple[str, ...]:
    users: dict[str, list[str]] = {name: [] for name in instructions}
    for instruction in instructions.values():
        for operand in instruction.operands:
            users.setdefault(operand, []).append(instruction.name)
    pending = list(users.get(instruction_name, ()))
    visited: set[str] = set()
    contracts: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        instruction = instructions[name]
        if instruction.opcode == "dot":
            contracts.add(name)
            continue
        if instruction.opcode in _WRAPPER_OPCODES | {"transpose"}:
            pending.extend(users.get(name, ()))
    return tuple(sorted(contracts))


def parse_hlo_module_text(hlo_text: str) -> HloModuleGraph:
    """Parse computations and data operands from an XLA HLO text dump."""
    computations: list[HloComputation] = []
    current_name: str | None = None
    current_entry = False
    instructions: list[HloInstruction] = []
    for line in hlo_text.splitlines():
        if current_name is None:
            header = _COMPUTATION_HEADER.match(line)
            if header is None:
                continue
            current_name = header.group("name")
            current_entry = header.group("entry") is not None
            instructions = []
            continue
        if line == "}":
            computations.append(
                HloComputation(
                    name=current_name,
                    instructions=tuple(instructions),
                    is_entry=current_entry,
                )
            )
            current_name = None
            current_entry = False
            instructions = []
            continue
        match = _INSTRUCTION_NAME.match(line)
        if match is None:
            continue
        instructions.append(
            _parse_instruction(match.group("name"), match.group("body"), match.group("root") is not None)
        )
    if current_name is not None:
        raise ValueError(f"unterminated HLO computation {current_name!r}")
    entries = tuple(computation.name for computation in computations if computation.is_entry)
    if len(entries) != 1:
        raise ValueError(f"expected exactly one HLO entry computation, found {len(entries)}")
    return HloModuleGraph(computations=tuple(computations), entry=entries[0])


def inline_elementwise_fusions(module: HloModuleGraph) -> InlinedHloGraph:
    """Inline every fusion computation while preserving non-fusion operations."""
    nodes: dict[str, InlinedHloNode] = {}
    entry_values: dict[str, str] = {}
    computations = {computation.name: computation for computation in module.computations}
    instruction_maps = {
        computation.name: {instruction.name: instruction for instruction in computation.instructions}
        for computation in module.computations
    }
    entry = module.computation(module.entry)

    def expand_called(
        computation_name: str,
        instruction_name: str,
        *,
        namespace: str,
        parameter_bindings: dict[int, str],
    ) -> str:
        instruction = instruction_maps[computation_name][instruction_name]
        if instruction.opcode == "parameter" and parameter_bindings:
            number_match = _PARAMETER_NUMBER.search(instruction.attributes)
            if number_match is None:
                raise ValueError(f"parameter {instruction_name!r} has no parameter number")
            return parameter_bindings[int(number_match.group(1))]
        node_id = f"{namespace}/{instruction_name}"
        if node_id in nodes:
            return node_id
        operands = tuple(
            expand_called(
                computation_name,
                operand,
                namespace=namespace,
                parameter_bindings=parameter_bindings,
            )
            for operand in instruction.operands
        )
        if instruction.opcode == "fusion":
            call_match = _CALL_REFERENCE.search(instruction.attributes)
            if call_match is None:
                raise ValueError(f"fusion {instruction.name!r} has no called computation")
            called_name = call_match.group(1)
            called = computations[called_name]
            bindings = {index: operand for index, operand in enumerate(operands)}
            return expand_called(
                called_name,
                called.root.name,
                namespace=node_id,
                parameter_bindings=bindings,
            )
        nodes[node_id] = InlinedHloNode(
            id=node_id,
            opcode=instruction.opcode,
            shape=instruction.shape,
            operands=operands,
            attributes=instruction.attributes,
            source_computation=computation_name,
            source_instruction=instruction.name,
        )
        return node_id

    for instruction in entry.instructions:
        try:
            operands = tuple(entry_values[operand] for operand in instruction.operands)
        except KeyError as error:
            raise ValueError(
                f"entry instruction {instruction.name!r} refers to a value defined after it: {error.args[0]!r}"
            ) from error
        node_id = f"{entry.name}/{instruction.name}"
        if instruction.opcode == "fusion":
            call_match = _CALL_REFERENCE.search(instruction.attributes)
            if call_match is None:
                raise ValueError(f"fusion {instruction.name!r} has no called computation")
            called_name = call_match.group(1)
            called = computations[called_name]
            entry_values[instruction.name] = expand_called(
                called_name,
                called.root.name,
                namespace=node_id,
                parameter_bindings={index: operand for index, operand in enumerate(operands)},
            )
            continue
        nodes[node_id] = InlinedHloNode(
            id=node_id,
            opcode=instruction.opcode,
            shape=instruction.shape,
            operands=operands,
            attributes=instruction.attributes,
            source_computation=entry.name,
            source_instruction=instruction.name,
        )
        entry_values[instruction.name] = node_id
    return InlinedHloGraph(nodes=tuple(nodes.values()), entry_values=tuple(entry_values.items()))


def recover_pair_map_regions(hlo_text: str) -> PairMapRecoveryReport:
    """Find generic shared-input Contract pairs followed by scalar Maps.

    The matcher uses only HLO opcodes, shapes, and data dependencies.  Metadata,
    stack-frame names, and instruction spelling do not participate.
    """
    module = parse_hlo_module_text(hlo_text)
    graph = inline_elementwise_fusions(module)
    nodes = {node.id: node for node in graph.nodes}
    users: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for node in graph.nodes:
        for operand in node.operands:
            users.setdefault(operand, []).append(node.id)
    contracts = {
        node.id: ContractRecord(
            node=node.id,
            lhs=graph.strip_wrappers(node.operands[0]),
            rhs=graph.strip_wrappers(node.operands[1]),
            output_shape=node.shape,
        )
        for node in graph.nodes
        if node.opcode == "dot" and len(node.operands) == 2
    }

    ancestor_cache: dict[str, frozenset[str]] = {}

    def contract_ancestors(node_id: str) -> frozenset[str]:
        if node_id in ancestor_cache:
            return ancestor_cache[node_id]
        node = nodes[node_id]
        if node.opcode == "dot":
            result = frozenset({node_id})
        elif node.opcode in _POINTWISE_OPCODES | _WRAPPER_OPCODES:
            result = frozenset().union(*(contract_ancestors(operand) for operand in node.operands))
        else:
            result = frozenset()
        ancestor_cache[node_id] = result
        return result

    candidates: list[RecoveredPairMapRegion] = []
    for node in graph.nodes:
        ancestors = tuple(sorted(contract_ancestors(node.id)))
        if len(ancestors) != 2 or node.opcode not in _POINTWISE_OPCODES | _WRAPPER_OPCODES:
            continue
        left = contracts[ancestors[0]]
        right = contracts[ancestors[1]]
        if left.output_shape != right.output_shape:
            continue
        shared = _shared_contract_input(left, right)
        if shared is None:
            continue
        if any(
            contract_ancestors(user) == frozenset(ancestors)
            for user in users[node.id]
            if nodes[user].opcode in _POINTWISE_OPCODES | _WRAPPER_OPCODES
        ):
            continue
        map_nodes = _pointwise_subgraph(nodes, node.id, frozenset(ancestors))
        map_casts = tuple(
            WrapperBoundary(
                opcode="convert",
                source_shape=nodes[value.operands[0]].shape,
                result_shape=value.shape,
            )
            for value in map_nodes
            if value.opcode == "convert" and len(value.operands) == 1
        )
        consumer_contracts = tuple(sorted(_first_contract_users(nodes, users, node.id)))
        candidates.append(
            RecoveredPairMapRegion(
                left_contract=left,
                right_contract=right,
                shared_input=shared,
                map_root=node.id,
                map_opcodes=tuple(value.opcode for value in map_nodes),
                map_cast_boundaries=map_casts,
                consumer_contracts=consumer_contracts,
            )
        )
    regions = tuple(_deduplicate_regions(candidates))
    return PairMapRecoveryReport(
        computation_count=len(module.computations),
        inlined_node_count=len(graph.nodes),
        contract_count=len(contracts),
        regions=regions,
        limitations=(
            "This read-only pass proves structural recovery; it does not yet replace HLO instructions.",
            "Convert boundaries are retained explicitly, so recovery does not imply source-ordered fusion legality.",
            "Backward roles are represented as ordinary downstream Contracts; "
            "saved-value and cotangent role assignment remains open.",
        ),
    )


def _parse_instruction(name: str, body: str, is_root: bool) -> HloInstruction:
    opcode_match = re.search(r" (?P<opcode>[a-z][a-z0-9-]*)\(", body)
    if opcode_match is None:
        raise ValueError(f"HLO instruction {name!r} has no operand list")
    shape = body[: opcode_match.start()]
    opcode = opcode_match.group("opcode")
    open_parenthesis = opcode_match.end() - 1
    close_parenthesis = _matching_parenthesis(body, open_parenthesis)
    operands_text = body[open_parenthesis + 1 : close_parenthesis]
    attributes = body[close_parenthesis + 1 :]
    operands = tuple(_VALUE_REFERENCE.findall(operands_text))
    return HloInstruction(
        name=name,
        opcode=opcode,
        shape=shape,
        operands=operands,
        attributes=f"{opcode}({operands_text}){attributes}",
        is_root=is_root,
    )


def _matching_parenthesis(text: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(text)):
        character = text[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unterminated HLO operand list")


def _shape_dtype(shape: str) -> str:
    shape = shape.lstrip("(")
    match = re.match(r"([A-Za-z0-9]+)(?:\[|\])", shape)
    return match.group(1) if match else "tuple"


def _shared_contract_input(left: ContractRecord, right: ContractRecord) -> str | None:
    left_inputs = {left.lhs.base, left.rhs.base}
    right_inputs = {right.lhs.base, right.rhs.base}
    shared = left_inputs & right_inputs
    return next(iter(shared)) if len(shared) == 1 else None


def _pointwise_subgraph(
    nodes: dict[str, InlinedHloNode],
    root: str,
    contract_leaves: frozenset[str],
) -> tuple[InlinedHloNode, ...]:
    ordered: list[InlinedHloNode] = []
    seen: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in seen or node_id in contract_leaves:
            return
        seen.add(node_id)
        node = nodes[node_id]
        for operand in node.operands:
            visit(operand)
        if node.opcode in _POINTWISE_OPCODES | _WRAPPER_OPCODES:
            ordered.append(node)

    visit(root)
    return tuple(ordered)


def _first_contract_users(
    nodes: dict[str, InlinedHloNode],
    users: dict[str, list[str]],
    root: str,
) -> set[str]:
    contracts: set[str] = set()
    pending = list(users[root])
    seen: set[str] = set()
    while pending:
        node_id = pending.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        node = nodes[node_id]
        if node.opcode == "dot":
            contracts.add(node_id)
        elif node.opcode in _POINTWISE_OPCODES | _WRAPPER_OPCODES:
            pending.extend(users[node_id])
    return contracts


def _deduplicate_regions(regions: list[RecoveredPairMapRegion]) -> list[RecoveredPairMapRegion]:
    unique: dict[tuple[str, str, str], RecoveredPairMapRegion] = {}
    for region in regions:
        left_contract, right_contract = sorted((region.left_contract.node, region.right_contract.node))
        unique[(left_contract, right_contract, region.map_root)] = region
    return list(unique.values())
