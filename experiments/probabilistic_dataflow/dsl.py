# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Self


class AxisKind(StrEnum):
    ORDERED = "ordered"
    SET = "set"
    MESH = "mesh"
    CATEGORICAL = "categorical"
    UNORDERED_PAIR = "unordered_pair"


@dataclass(frozen=True)
class Axis:
    name: str
    size: int

    kind: ClassVar[AxisKind]

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"Axis {self.name!r} must have positive size, got {self.size}")


@dataclass(frozen=True)
class OrderedAxis(Axis):
    kind: ClassVar[AxisKind] = AxisKind.ORDERED


@dataclass(frozen=True)
class SetAxis(Axis):
    kind: ClassVar[AxisKind] = AxisKind.SET


@dataclass(frozen=True)
class MeshAxis(Axis):
    coordinates: tuple[tuple[float, ...], ...] = ()
    kind: ClassVar[AxisKind] = AxisKind.MESH

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.coordinates and len(self.coordinates) != self.size:
            raise ValueError(f"Mesh axis {self.name!r} has size {self.size} but {len(self.coordinates)} coordinates")


@dataclass(frozen=True)
class CategoricalAxis(Axis):
    labels: tuple[str, ...] = ()
    kind: ClassVar[AxisKind] = AxisKind.CATEGORICAL

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.labels and len(self.labels) != self.size:
            raise ValueError(f"Categorical axis {self.name!r} has size {self.size} but {len(self.labels)} labels")


@dataclass(frozen=True)
class UnorderedPairAxis(Axis):
    source: SetAxis = field(default_factory=lambda: SetAxis("item", 1))
    kind: ClassVar[AxisKind] = AxisKind.UNORDERED_PAIR

    @classmethod
    def of(cls, source: SetAxis, *, name: str | None = None) -> Self:
        return cls(name or f"unordered_pair({source.name})", math.comb(source.size, 2), source)

    def canonical_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple((left, right) for left in range(self.source.size) for right in range(left + 1, self.source.size))


@dataclass(frozen=True)
class FieldType:
    name: str
    axes: tuple[Axis, ...] = ()
    bins: int = 16

    def __post_init__(self) -> None:
        if self.bins <= 1:
            raise ValueError(f"Field {self.name!r} must have at least two discrete bins, got {self.bins}")

    @property
    def token_count(self) -> int:
        return math.prod(axis.size for axis in self.axes)


@dataclass(frozen=True)
class Source:
    name: str
    environments: frozenset[str]
    split_keys: frozenset[str]


@dataclass(frozen=True)
class FlowInfo:
    provenance: frozenset[str] = frozenset()
    environments: frozenset[str] = frozenset()
    split_keys: frozenset[str] = frozenset()
    random_ancestors: frozenset[str] = frozenset()


class NodeKind(StrEnum):
    VARIABLE = "variable"
    DETERMINISTIC = "deterministic"
    SAMPLE = "sample"


@dataclass(frozen=True)
class FactorSpec:
    name: str
    inputs: tuple[Value, ...]
    implementation: str = "shared_transformer"


def learned_joint(*inputs: Value, name: str = "learned_joint") -> FactorSpec:
    return FactorSpec(name=name, inputs=inputs)


@dataclass(frozen=True)
class Node:
    id: int
    name: str
    value_type: FieldType
    kind: NodeKind
    inputs: tuple[int, ...]
    flow: FlowInfo
    operation: str | None = None
    factor_id: str | None = None
    factor_name: str | None = None
    allows_cross_split: bool = False


@dataclass(frozen=True)
class Value:
    program_name: str
    node_id: int
    name: str
    value_type: FieldType
    _program: Program = field(compare=False, hash=False, repr=False)

    @property
    def flow(self) -> FlowInfo:
        return self._program.node(self).flow


class Program:
    """Mutable builder whose completed nodes form the probabilistic dataflow IR."""

    def __init__(self, name: str):
        self.name = name
        self._nodes: list[Node] = []
        self._names: set[str] = set()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        return None

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes)

    def node(self, value: Value | int) -> Node:
        node_id = value if isinstance(value, int) else value.node_id
        return self._nodes[node_id]

    def value(self, name: str) -> Value:
        for node in self._nodes:
            if node.name == name:
                return self._value(node)
        raise KeyError(name)

    def variable(self, name: str, value_type: FieldType, *, source: Source | None = None) -> Value:
        if source is None:
            flow = FlowInfo()
        else:
            flow = FlowInfo(
                provenance=frozenset({source.name}),
                environments=source.environments,
                split_keys=source.split_keys,
            )
        return self._append(name, value_type, NodeKind.VARIABLE, (), flow)

    def sample(self, name: str, value_type: FieldType, factor: FactorSpec) -> Value:
        self._check_values(factor.inputs)
        factor_id = f"{self.name}:{name}:{len(self._nodes)}"
        flow = _merge_flow(factor.inputs, random_ancestor=factor_id)
        return self._append(
            name,
            value_type,
            NodeKind.SAMPLE,
            tuple(value.node_id for value in factor.inputs),
            flow,
            factor_id=factor_id,
            factor_name=factor.name,
        )

    def deterministic(
        self,
        name: str,
        operation: str,
        *inputs: Value,
        value_type: FieldType | None = None,
        allows_cross_split: bool = False,
    ) -> Value:
        if not inputs:
            raise ValueError("Deterministic operations require at least one input")
        self._check_values(inputs)
        return self._append(
            name,
            value_type or inputs[0].value_type,
            NodeKind.DETERMINISTIC,
            tuple(value.node_id for value in inputs),
            _merge_flow(inputs),
            operation=operation,
            allows_cross_split=allows_cross_split,
        )

    def map(self, name: str, value: Value, *, operation: str, value_type: FieldType | None = None) -> Value:
        return self.deterministic(name, f"map:{operation}", value, value_type=value_type)

    def join(self, name: str, *values: Value, value_type: FieldType) -> Value:
        return self.deterministic(name, "join", *values, value_type=value_type)

    def select(self, name: str, value: Value, *, selector: str, value_type: FieldType | None = None) -> Value:
        return self.deterministic(name, f"select:{selector}", value, value_type=value_type)

    def reduce(
        self,
        name: str,
        value: Value,
        *,
        reduction: str,
        value_type: FieldType,
        allows_cross_split: bool = False,
    ) -> Value:
        return self.deterministic(
            name,
            f"reduce:{reduction}",
            value,
            value_type=value_type,
            allows_cross_split=allows_cross_split,
        )

    def _append(
        self,
        name: str,
        value_type: FieldType,
        kind: NodeKind,
        inputs: tuple[int, ...],
        flow: FlowInfo,
        *,
        operation: str | None = None,
        factor_id: str | None = None,
        factor_name: str | None = None,
        allows_cross_split: bool = False,
    ) -> Value:
        if name in self._names:
            raise ValueError(f"Program {self.name!r} already contains a value named {name!r}")
        node = Node(
            id=len(self._nodes),
            name=name,
            value_type=value_type,
            kind=kind,
            inputs=inputs,
            flow=flow,
            operation=operation,
            factor_id=factor_id,
            factor_name=factor_name,
            allows_cross_split=allows_cross_split,
        )
        self._nodes.append(node)
        self._names.add(name)
        return self._value(node)

    def _value(self, node: Node) -> Value:
        return Value(self.name, node.id, node.name, node.value_type, self)

    def _check_values(self, values: tuple[Value, ...]) -> None:
        for value in values:
            if value._program is not self:
                raise ValueError(f"Value {value.name!r} belongs to program {value.program_name!r}, not {self.name!r}")


def _merge_flow(values: tuple[Value, ...], *, random_ancestor: str | None = None) -> FlowInfo:
    environments = [value.flow.environments for value in values if value.flow.environments]
    available_environments = frozenset.intersection(*environments) if environments else frozenset()
    random_ancestors = frozenset().union(*(value.flow.random_ancestors for value in values))
    if random_ancestor is not None:
        random_ancestors |= {random_ancestor}
    return FlowInfo(
        provenance=frozenset().union(*(value.flow.provenance for value in values)),
        environments=available_environments,
        split_keys=frozenset().union(*(value.flow.split_keys for value in values)),
        random_ancestors=random_ancestors,
    )


@dataclass(frozen=True)
class Budget:
    model_calls: int
    generated_tokens: int


@dataclass(frozen=True)
class Query:
    program: Program
    given: tuple[Value, ...]
    targets: tuple[Value, ...]
    environment: str
    budget: Budget = Budget(model_calls=4, generated_tokens=100_000)

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("A query requires at least one target")
        if len(set(self.given)) != len(self.given):
            raise ValueError("A query cannot contain duplicate given values")
        for value in (*self.given, *self.targets):
            if value._program is not self.program:
                raise ValueError(f"Value {value.name!r} does not belong to query program {self.program.name!r}")


@dataclass(frozen=True)
class QueryRoleDifference:
    value_name: str
    training_role: str
    deployment_role: str


def training_deployment_differences(
    program: Program,
    *,
    training_given: tuple[Value, ...],
    deployment_given: tuple[Value, ...],
) -> tuple[QueryRoleDifference, ...]:
    differences = []
    for node in program.nodes:
        value = program.value(node.name)
        training_role = "given" if value in training_given else "generated"
        deployment_role = "given" if value in deployment_given else "generated"
        if training_role != deployment_role:
            differences.append(QueryRoleDifference(node.name, training_role, deployment_role))
    return tuple(differences)
