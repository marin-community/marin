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
    split_keys: frozenset[str]


@dataclass(frozen=True)
class FlowInfo:
    provenance: frozenset[str] = frozenset()
    split_keys: frozenset[str] = frozenset()
    random_ancestors: frozenset[str] = frozenset()


class NodeKind(StrEnum):
    INPUT = "input"
    DETERMINISTIC = "deterministic"
    SAMPLE = "sample"


@dataclass(frozen=True)
class Budget:
    model_calls: int
    generated_tokens: int


class AttentionPattern(StrEnum):
    FULL = "full"
    CAUSAL = "causal"


class PositionMode(StrEnum):
    SCIENTIFIC = "scientific"
    SEQUENCE = "sequence"


@dataclass(frozen=True)
class DocumentSpec:
    attention: AttentionPattern
    positions: PositionMode


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
class InferenceCall:
    id: int
    target_ids: tuple[int, ...]
    context_ids: tuple[int, ...]
    dependency_call_ids: tuple[int, ...]
    operator: str
    document: DocumentSpec
    iteration: int = 0
    resample_fraction: float | None = None


@dataclass(frozen=True)
class Value:
    program_name: str
    node_id: int
    name: str
    value_type: FieldType
    _program: InferenceProgram = field(compare=False, hash=False, repr=False)

    @property
    def flow(self) -> FlowInfo:
        return self._program.node(self).flow


class InferenceProgram:
    """Staged scientific program whose generated values form a model-call DAG."""

    def __init__(self, name: str, *, budget: Budget):
        self.name = name
        self.budget = budget
        self._nodes: list[Node] = []
        self._names: set[str] = set()
        self._calls: list[InferenceCall] = []
        self._latest_call_by_value: dict[int, int] = {}
        self._output_ids: tuple[int, ...] = ()
        self._finished = False

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes)

    @property
    def calls(self) -> tuple[InferenceCall, ...]:
        return tuple(self._calls)

    @property
    def outputs(self) -> tuple[Value, ...]:
        return tuple(self._value(self.node(node_id)) for node_id in self._output_ids)

    @property
    def external_inputs(self) -> tuple[Value, ...]:
        context_ids = {node_id for call in self._calls for node_id in call.context_ids}
        external_ids = {node_id for node_id in context_ids if not self._has_generated_ancestor(node_id)}
        return tuple(self._value(node) for node in self._nodes if node.id in external_ids)

    def node(self, value: Value | int) -> Node:
        node_id = value if isinstance(value, int) else value.node_id
        return self._nodes[node_id]

    def value(self, name: str) -> Value:
        for node in self._nodes:
            if node.name == name:
                return self._value(node)
        raise KeyError(name)

    def input_value(self, name: str, value_type: FieldType, *, source: Source | None = None) -> Value:
        if source is None:
            flow = FlowInfo()
        else:
            flow = FlowInfo(
                provenance=frozenset({source.name}),
                split_keys=source.split_keys,
            )
        return self._append(name, value_type, NodeKind.INPUT, (), flow)

    def generate(
        self,
        name: str,
        value_type: FieldType,
        *,
        context: tuple[Value, ...],
        document: DocumentSpec,
        factor_name: str,
    ) -> Value:
        self._check_open()
        if len(set(context)) != len(context):
            raise ValueError(f"Generation context for {name!r} contains duplicate values")
        self._check_values(context)
        factor_id = f"{self.name}:{name}:{len(self._nodes)}"
        value = self._append(
            name,
            value_type,
            NodeKind.SAMPLE,
            tuple(value.node_id for value in context),
            _merge_flow(context, random_ancestor=factor_id),
            factor_id=factor_id,
            factor_name=factor_name,
        )
        call = InferenceCall(
            id=len(self._calls),
            target_ids=(value.node_id,),
            context_ids=tuple(item.node_id for item in context),
            dependency_call_ids=self._dependencies(context),
            operator="generate",
            document=document,
        )
        self._calls.append(call)
        self._latest_call_by_value[value.node_id] = call.id
        return value

    def refine(
        self,
        target: Value,
        *,
        context: tuple[Value, ...],
        document: DocumentSpec,
        resample_fraction: float,
    ) -> Value:
        self._check_open()
        if not 0 < resample_fraction <= 1:
            raise ValueError(f"resample_fraction must be in (0, 1], got {resample_fraction}")
        self._check_values((target, *context))
        if target.node_id not in self._latest_call_by_value:
            raise ValueError(f"Cannot refine {target.name!r} before it has been generated")
        if target in context:
            raise ValueError(f"Refinement context for {target.name!r} already contains the target")
        full_context = (*context, target)
        previous_call = self._calls[self._latest_call_by_value[target.node_id]]
        call = InferenceCall(
            id=len(self._calls),
            target_ids=(target.node_id,),
            context_ids=tuple(value.node_id for value in full_context),
            dependency_call_ids=self._dependencies(full_context),
            operator="refine",
            document=document,
            iteration=previous_call.iteration + 1,
            resample_fraction=resample_fraction,
        )
        self._calls.append(call)
        self._latest_call_by_value[target.node_id] = call.id
        return target

    def finish(self, *outputs: Value) -> Self:
        self._check_open()
        if not outputs:
            raise ValueError("An inference program requires at least one output")
        if len(set(outputs)) != len(outputs):
            raise ValueError("An inference program cannot repeat outputs")
        self._check_values(outputs)
        for output in outputs:
            if self.node(output).kind != NodeKind.SAMPLE:
                raise ValueError(f"Inference output {output.name!r} is not generated")
        self._output_ids = tuple(output.node_id for output in outputs)
        self._validate()
        self._finished = True
        return self

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
        self._check_open()
        if name in self._names:
            raise ValueError(f"Inference program {self.name!r} already contains a value named {name!r}")
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

    def _check_open(self) -> None:
        if self._finished:
            raise ValueError(f"Inference program {self.name!r} is already finished")

    def _sample_ancestors(self, node_id: int) -> set[int]:
        node = self.node(node_id)
        ancestors = {node_id} if node.kind == NodeKind.SAMPLE else set()
        for input_id in node.inputs:
            ancestors.update(self._sample_ancestors(input_id))
        return ancestors

    def _has_generated_ancestor(self, node_id: int) -> bool:
        return bool(self._sample_ancestors(node_id))

    def _dependencies(self, context: tuple[Value, ...]) -> tuple[int, ...]:
        sample_ids = set().union(*(self._sample_ancestors(value.node_id) for value in context))
        missing = sample_ids - self._latest_call_by_value.keys()
        if missing:
            names = [self.node(node_id).name for node_id in sorted(missing)]
            raise ValueError(f"Context uses generated values before their model calls: {names}")
        return tuple(sorted({self._latest_call_by_value[node_id] for node_id in sample_ids}))

    def _validate(self) -> None:
        generated_tokens = sum(
            self.node(target_id).value_type.token_count for call in self._calls for target_id in call.target_ids
        )
        if len(self._calls) > self.budget.model_calls:
            raise ValueError(
                f"Program requires {len(self._calls)} model calls but budget allows {self.budget.model_calls}"
            )
        if generated_tokens > self.budget.generated_tokens:
            raise ValueError(
                f"Program generates {generated_tokens} tokens but budget allows {self.budget.generated_tokens}"
            )


def _merge_flow(values: tuple[Value, ...], *, random_ancestor: str | None = None) -> FlowInfo:
    random_ancestors = frozenset().union(*(value.flow.random_ancestors for value in values))
    if random_ancestor is not None:
        random_ancestors |= {random_ancestor}
    return FlowInfo(
        provenance=frozenset().union(*(value.flow.provenance for value in values)),
        split_keys=frozenset().union(*(value.flow.split_keys for value in values)),
        random_ancestors=random_ancestors,
    )


@dataclass(frozen=True)
class InferenceRoleDifference:
    value_name: str
    training_role: str
    deployment_role: str


def training_deployment_differences(
    program: InferenceProgram,
    *,
    training_given: tuple[Value, ...],
) -> tuple[InferenceRoleDifference, ...]:
    deployment_given = set(program.external_inputs)
    differences = []
    for node in program.nodes:
        value = program.value(node.name)
        training_role = "given" if value in training_given else "generated"
        deployment_role = "given" if value in deployment_given else "generated"
        if training_role != deployment_role:
            differences.append(InferenceRoleDifference(node.name, training_role, deployment_role))
    return tuple(differences)
