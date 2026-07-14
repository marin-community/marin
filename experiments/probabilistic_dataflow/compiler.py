# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from experiments.probabilistic_dataflow.dsl import (
    CategoricalAxis,
    MeshAxis,
    Node,
    NodeKind,
    Program,
    Query,
    UnorderedPairAxis,
    Value,
)


class CompilationError(ValueError):
    pass


class AvailabilityError(CompilationError):
    pass


class SplitLeakageError(CompilationError):
    pass


class FactorizationError(CompilationError):
    pass


@dataclass(frozen=True)
class ParallelQuery:
    targets: tuple[Value, ...] = ()


@dataclass(frozen=True)
class Autoregressive:
    order: tuple[Value, ...] = ()


@dataclass(frozen=True)
class Refine:
    proposal: ParallelQuery | Autoregressive
    steps: int
    resample_fraction: float

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError(f"Refinement steps must be positive, got {self.steps}")
        if not 0 < self.resample_fraction <= 1:
            raise ValueError(f"resample_fraction must be in (0, 1], got {self.resample_fraction}")


InferenceOperator = ParallelQuery | Autoregressive | Refine


@dataclass(frozen=True)
class ConditionalQueryIR:
    program_name: str
    conditioned_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    required_factor_ids: tuple[str, ...]
    deployment_environment: str
    execution_time: int
    model_call_budget: int
    generated_token_budget: int


@dataclass(frozen=True)
class ModelCallIR:
    id: int
    target_ids: tuple[int, ...]
    context_ids: tuple[int, ...]
    dependency_call_ids: tuple[int, ...]
    operator: str
    iteration: int = 0
    approximation_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class InferencePlanIR:
    query: ConditionalQueryIR
    calls: tuple[ModelCallIR, ...]


def compile_query(query: Query, plan: InferenceOperator | None = None) -> InferencePlanIR:
    _check_availability(query)
    _check_split_isolation(query.program)
    _check_plan_values(query, plan)
    conditional_ir = _conditional_query_ir(query)
    required_nodes = _required_generated_nodes(query)

    if plan is None:
        calls = _default_calls(query, required_nodes)
    elif isinstance(plan, ParallelQuery):
        calls = _parallel_calls(query, required_nodes, plan)
    elif isinstance(plan, Autoregressive):
        calls = _autoregressive_calls(query, required_nodes, plan)
    else:
        calls = _refinement_calls(query, required_nodes, plan)

    generated_tokens = sum(
        query.program.node(node_id).value_type.token_count for call in calls for node_id in call.target_ids
    )
    if len(calls) > query.budget.model_calls:
        raise CompilationError(f"Plan requires {len(calls)} model calls but budget allows {query.budget.model_calls}")
    if generated_tokens > query.budget.generated_tokens:
        raise CompilationError(
            f"Plan generates {generated_tokens} tokens but budget allows {query.budget.generated_tokens}"
        )
    return InferencePlanIR(conditional_ir, tuple(calls))


def _check_plan_values(query: Query, plan: InferenceOperator | None) -> None:
    if plan is None:
        return
    proposal = plan.proposal if isinstance(plan, Refine) else plan
    values = proposal.targets if isinstance(proposal, ParallelQuery) else proposal.order
    for value in values:
        if value._program is not query.program:
            raise FactorizationError(
                f"Inference plan value {value.name!r} belongs to {value.program_name!r}, not {query.program.name!r}"
            )


def _check_availability(query: Query) -> None:
    for binding in query.evidence.bindings:
        value = binding.value
        if binding.environment != query.environment.name:
            raise AvailabilityError(
                f"Evidence for {value.name!r} is bound in {binding.environment!r}, not {query.environment.name!r}"
            )
        if binding.available_at > query.environment.execution_time:
            raise AvailabilityError(
                f"Evidence {value.name!r} is available at t={binding.available_at}, after deployment time "
                f"t={query.environment.execution_time}; provenance={sorted(value.flow.provenance)}"
            )
        allowed_environments = value.flow.deployment_environments
        if allowed_environments and query.environment.name not in allowed_environments:
            raise AvailabilityError(
                f"Evidence {value.name!r} is unavailable in deployment environment {query.environment.name!r}; "
                f"allowed={sorted(allowed_environments)}"
            )


def _check_split_isolation(program: Program) -> None:
    for node in program.nodes:
        if node.kind != NodeKind.DETERMINISTIC or not node.operation or not node.operation.startswith("reduce:"):
            continue
        if len(node.flow.split_keys) > 1 and not node.allows_cross_split:
            raise SplitLeakageError(
                f"Reduction {node.name!r} combines split keys {sorted(node.flow.split_keys)} without explicit permission"
            )


def _conditional_query_ir(query: Query) -> ConditionalQueryIR:
    factor_ids = []
    for node in query.program.nodes:
        if node.factor_id is not None and _is_ancestor_of_any(query.program, node.id, query.targets):
            factor_ids.append(node.factor_id)
    return ConditionalQueryIR(
        program_name=query.program.name,
        conditioned_ids=tuple(value.node_id for value in query.evidence.values),
        target_ids=tuple(value.node_id for value in query.targets),
        required_factor_ids=tuple(factor_ids),
        deployment_environment=query.environment.name,
        execution_time=query.environment.execution_time,
        model_call_budget=query.budget.model_calls,
        generated_token_budget=query.budget.generated_tokens,
    )


def _required_generated_nodes(query: Query) -> tuple[int, ...]:
    conditioned = {value.node_id for value in query.evidence.values}
    required: set[int] = set()

    def visit(node_id: int) -> None:
        if node_id in conditioned:
            return
        node = query.program.node(node_id)
        if node.kind == NodeKind.VARIABLE:
            raise CompilationError(f"Required input {node.name!r} is not bound by query evidence")
        for input_id in node.inputs:
            visit(input_id)
        if node.kind == NodeKind.SAMPLE:
            required.add(node_id)

    for target in query.targets:
        visit(target.node_id)
        if target.node_id not in conditioned and query.program.node(target).kind == NodeKind.SAMPLE:
            required.add(target.node_id)
    return tuple(node.id for node in query.program.nodes if node.id in required)


def _default_calls(query: Query, required_nodes: tuple[int, ...]) -> list[ModelCallIR]:
    remaining = set(required_nodes)
    produced: dict[int, int] = {}
    calls: list[ModelCallIR] = []
    conditioned = {value.node_id for value in query.evidence.values}
    while remaining:
        ready = tuple(
            node_id
            for node_id in required_nodes
            if node_id in remaining
            and all(ancestor not in remaining for ancestor in _sample_ancestors(query.program, node_id))
        )
        if not ready:
            raise FactorizationError("Probabilistic factor graph contains a cycle")
        call = _make_call(query, len(calls), ready, conditioned, produced, operator="parallel")
        calls.append(call)
        for node_id in ready:
            remaining.remove(node_id)
            produced[node_id] = call.id
    return calls


def _parallel_calls(
    query: Query,
    required_nodes: tuple[int, ...],
    plan: ParallelQuery,
) -> list[ModelCallIR]:
    requested = tuple(value.node_id for value in plan.targets) or tuple(value.node_id for value in query.targets)
    missing_latents = set(required_nodes) - set(requested)
    if missing_latents:
        names = [query.program.node(node_id).name for node_id in sorted(missing_latents)]
        raise FactorizationError(
            f"ParallelQuery would skip required latent factors {names}; use the default planner or Autoregressive"
        )
    for left in requested:
        for right in requested:
            if left != right and left in _sample_ancestors(query.program, right):
                raise FactorizationError(
                    f"ParallelQuery cannot generate {query.program.node(left).name!r} and "
                    f"{query.program.node(right).name!r} together because the latter factor depends on the former"
                )
    return [_make_call(query, 0, requested, {value.node_id for value in query.evidence.values}, {}, "parallel")]


def _autoregressive_calls(
    query: Query,
    required_nodes: tuple[int, ...],
    plan: Autoregressive,
) -> list[ModelCallIR]:
    order = tuple(value.node_id for value in plan.order) or required_nodes
    if set(order) != set(required_nodes) or len(order) != len(required_nodes):
        required_names = [query.program.node(node_id).name for node_id in required_nodes]
        raise FactorizationError(
            f"Autoregressive order must contain each required generated value once: {required_names}"
        )

    produced: dict[int, int] = {}
    conditioned = {value.node_id for value in query.evidence.values}
    calls = []
    for node_id in order:
        unresolved = _sample_ancestors(query.program, node_id) - produced.keys() - conditioned
        if unresolved:
            names = [query.program.node(ancestor).name for ancestor in sorted(unresolved)]
            raise FactorizationError(
                f"Autoregressive order generates {query.program.node(node_id).name!r} before ancestors {names}"
            )
        call = _make_call(query, len(calls), (node_id,), conditioned, produced, "autoregressive")
        calls.append(call)
        produced[node_id] = call.id
    return calls


def _refinement_calls(
    query: Query,
    required_nodes: tuple[int, ...],
    plan: Refine,
) -> list[ModelCallIR]:
    if isinstance(plan.proposal, ParallelQuery):
        calls = _parallel_calls(query, required_nodes, plan.proposal)
    else:
        calls = _autoregressive_calls(query, required_nodes, plan.proposal)

    target_ids = tuple(value.node_id for value in query.targets)
    latest_call_by_target = {target_id: call.id for call in calls for target_id in call.target_ids}
    base_context = {value.node_id for value in query.evidence.values}
    for iteration in range(1, plan.steps):
        call_id = len(calls)
        dependencies = tuple(sorted(set(latest_call_by_target.values())))
        context = tuple(sorted(base_context | set(target_ids)))
        notes = (f"resample_low_confidence_fraction={plan.resample_fraction}",)
        calls.append(ModelCallIR(call_id, target_ids, context, dependencies, "refine", iteration, notes))
        latest_call_by_target = dict.fromkeys(target_ids, call_id)
    return calls


def _make_call(
    query: Query,
    call_id: int,
    target_ids: tuple[int, ...],
    conditioned: set[int],
    produced: dict[int, int],
    operator: str,
) -> ModelCallIR:
    context = set(conditioned)
    dependencies = set()
    notes = []
    for target_id in target_ids:
        node = query.program.node(target_id)
        for input_id in node.inputs:
            if input_id in produced:
                context.add(input_id)
                dependencies.add(produced[input_id])
            elif query.program.node(input_id).kind == NodeKind.SAMPLE:
                raise FactorizationError(
                    f"Factor for {node.name!r} consumes ungenerated sample {query.program.node(input_id).name!r}"
                )
            else:
                context.add(input_id)
        if operator == "parallel" and node.value_type.token_count > 1:
            notes.append(
                f"factor {node.factor_id} is approximated as {node.value_type.token_count} parallel token marginals"
            )
    return ModelCallIR(
        id=call_id,
        target_ids=target_ids,
        context_ids=tuple(sorted(context)),
        dependency_call_ids=tuple(sorted(dependencies)),
        operator=operator,
        approximation_notes=tuple(notes),
    )


def _sample_ancestors(program: Program, node_id: int) -> set[int]:
    ancestors: set[int] = set()

    def visit(current: int) -> None:
        for input_id in program.node(current).inputs:
            input_node = program.node(input_id)
            if input_node.kind == NodeKind.SAMPLE:
                ancestors.add(input_id)
            visit(input_id)

    visit(node_id)
    return ancestors


def _is_ancestor_of_any(program: Program, candidate: int, values: tuple[Value, ...]) -> bool:
    def contains(node_id: int) -> bool:
        if node_id == candidate:
            return True
        return any(contains(input_id) for input_id in program.node(node_id).inputs)

    return any(contains(value.node_id) for value in values)


@dataclass(frozen=True)
class ConcreteExample:
    id: str
    program_name: str
    values: dict[str, tuple[int, ...]]

    def value(self, node: Node) -> tuple[int, ...]:
        try:
            result = self.values[node.name]
        except KeyError as exc:
            raise CompilationError(f"Example {self.id!r} has no value for {node.name!r}") from exc
        if len(result) != node.value_type.token_count:
            raise CompilationError(
                f"Example {self.id!r} gives {len(result)} tokens for {node.name!r}, "
                f"expected {node.value_type.token_count}"
            )
        if any(token < 0 or token >= node.value_type.bins for token in result):
            raise CompilationError(
                f"Example {self.id!r} has an out-of-range token for {node.name!r}; bins={node.value_type.bins}"
            )
        return result


@dataclass
class TokenCodec:
    max_scientific_positions: int = 64
    max_data_bins: int = 32
    _scientific_positions: dict[str, int] = field(default_factory=dict)
    _tokens: dict[str, int] = field(default_factory=dict)

    PAD_ID: int = 0
    QUERY_ID: int = 1
    TOKEN_OFFSET: int = 2
    DATA_OFFSET: int = 32

    @property
    def vocab_size(self) -> int:
        return self.DATA_OFFSET + self.max_data_bins

    @property
    def scientific_position_count(self) -> int:
        return len(self._scientific_positions)

    def scientific_position_name(self, position_id: int) -> str:
        for name, candidate_id in self._scientific_positions.items():
            if candidate_id == position_id:
                return name
        raise KeyError(position_id)

    def token_label(self, token_id: int) -> str:
        if token_id == self.PAD_ID:
            return "<pad>"
        if token_id == self.QUERY_ID:
            return "<query>"
        for name, candidate_id in self._tokens.items():
            if candidate_id == token_id:
                return f"<text:{name}>"
        value = token_id - self.DATA_OFFSET
        if 0 <= value < self.max_data_bins:
            return f"value:{value}"
        return f"token:{token_id}"

    def token(self, name: str) -> int:
        if name not in self._tokens:
            token_id = self.TOKEN_OFFSET + len(self._tokens)
            if token_id >= self.DATA_OFFSET:
                raise CompilationError(f"Shared token table exceeded {self.DATA_OFFSET - self.TOKEN_OFFSET} entries")
            self._tokens[name] = token_id
        return self._tokens[name]

    def scientific_position(self, key: str) -> int:
        if key not in self._scientific_positions:
            if len(self._scientific_positions) >= self.max_scientific_positions:
                raise CompilationError(f"Scientific position table exceeded {self.max_scientific_positions} entries")
            self._scientific_positions[key] = len(self._scientific_positions)
        return self._scientific_positions[key]

    def data(self, value: int) -> int:
        if not 0 <= value < self.max_data_bins:
            raise CompilationError(f"Data token {value} is outside codec range [0, {self.max_data_bins})")
        return self.DATA_OFFSET + value


class AttentionLayout(StrEnum):
    FULL = "full_segment"
    CAUSAL = "causal_segment"


@dataclass(frozen=True)
class ExecutionSequenceIR:
    example_id: str
    call_id: int
    sequence_id: int
    token_ids: tuple[int, ...]
    scientific_position_ids: tuple[int, ...]
    rotary_position_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    loss_weights: tuple[float, ...]

    def reordered(self, order: tuple[int, ...]) -> ExecutionSequenceIR:
        """Return the same scientific records in a different physical order."""
        if tuple(sorted(order)) != tuple(range(len(self.token_ids))):
            raise ValueError("Record order must be a permutation of all sequence positions")
        return ExecutionSequenceIR(
            self.example_id,
            self.call_id,
            self.sequence_id,
            tuple(self.token_ids[index] for index in order),
            tuple(self.scientific_position_ids[index] for index in order),
            tuple(self.rotary_position_ids[index] for index in order),
            tuple(self.target_ids[index] for index in order),
            tuple(self.loss_weights[index] for index in order),
        )


@dataclass(frozen=True)
class TransformerCallExecutionIR:
    call_id: int
    operator: str
    dependency_call_ids: tuple[int, ...]
    sequences: tuple[ExecutionSequenceIR, ...]
    attention_layout: AttentionLayout = AttentionLayout.FULL


@dataclass(frozen=True)
class TransformerExecutionIR:
    example_id: str
    calls: tuple[TransformerCallExecutionIR, ...]


def lower_to_transformer(
    program: Program,
    plan: InferencePlanIR,
    example: ConcreteExample,
    codec: TokenCodec,
) -> TransformerExecutionIR:
    if example.program_name != program.name or plan.query.program_name != program.name:
        raise CompilationError(
            f"Program mismatch: program={program.name!r}, plan={plan.query.program_name!r}, "
            f"example={example.program_name!r}"
        )
    calls = []
    for call in plan.calls:
        sequence = _scientific_record_sequence(program, call, example, codec)
        calls.append(
            TransformerCallExecutionIR(
                call_id=call.id,
                operator=call.operator,
                dependency_call_ids=call.dependency_call_ids,
                sequences=(sequence,),
            )
        )
    return TransformerExecutionIR(example.id, tuple(calls))


def _scientific_record_sequence(
    program: Program,
    call: ModelCallIR,
    example: ConcreteExample,
    codec: TokenCodec,
) -> ExecutionSequenceIR:
    tokens: list[int] = []
    scientific_positions: list[int] = []
    target_ids: list[int] = []
    weights: list[float] = []
    for node_id in call.context_ids:
        node = program.node(node_id)
        for index, value in enumerate(example.value(node)):
            tokens.append(codec.data(value))
            scientific_positions.append(codec.scientific_position(_scientific_position_key(program, node, index)))
            target_ids.append(-1)
            weights.append(0.0)
    for target_id in call.target_ids:
        node = program.node(target_id)
        for index, value in enumerate(example.value(node)):
            tokens.append(codec.QUERY_ID)
            scientific_positions.append(codec.scientific_position(_scientific_position_key(program, node, index)))
            target_ids.append(codec.data(value))
            weights.append(1.0)
    return ExecutionSequenceIR(
        example.id,
        call.id,
        0,
        tuple(tokens),
        tuple(scientific_positions),
        (0,) * len(tokens),
        tuple(target_ids),
        tuple(weights),
    )


def _scientific_position_key(program: Program, node: Node, index: int) -> str:
    axes = node.value_type.axes
    if not axes:
        return f"{program.name}.{node.name}[scalar]"
    coordinates = np.unravel_index(index, tuple(axis.size for axis in axes))
    components = []
    for axis, coordinate in zip(axes, coordinates, strict=True):
        coordinate = int(coordinate)
        if isinstance(axis, MeshAxis) and axis.coordinates:
            components.append(f"{axis.name}={axis.coordinates[coordinate]}")
        elif isinstance(axis, CategoricalAxis) and axis.labels:
            components.append(f"{axis.name}={axis.labels[coordinate]}")
        elif isinstance(axis, UnorderedPairAxis):
            left, right = axis.canonical_pairs()[coordinate]
            components.append(f"{axis.source.name}={{{left},{right}}}")
        else:
            components.append(f"{axis.name}={coordinate}")
    return f"{program.name}.{node.name}[{','.join(components)}]"


@dataclass(frozen=True)
class PackedSequenceLocation:
    example_id: str
    call_id: int
    sequence_id: int
    row: int
    start: int
    end: int


@dataclass(frozen=True)
class PackedBatch:
    token_ids: np.ndarray
    scientific_position_ids: np.ndarray
    rotary_position_ids: np.ndarray
    target_ids: np.ndarray
    loss_weights: np.ndarray
    segment_ids: np.ndarray
    locations: tuple[PackedSequenceLocation, ...]


def pack_transformer_calls(executions: tuple[TransformerExecutionIR, ...], *, max_seq_len: int) -> PackedBatch:
    sequences = [sequence for execution in executions for call in execution.calls for sequence in call.sequences]
    if not sequences:
        raise CompilationError("Cannot pack an empty execution")
    if any(len(sequence.token_ids) > max_seq_len for sequence in sequences):
        longest = max(len(sequence.token_ids) for sequence in sequences)
        raise CompilationError(f"Execution sequence length {longest} exceeds max_seq_len={max_seq_len}")

    rows: list[list[ExecutionSequenceIR]] = [[]]
    row_lengths = [0]
    for sequence in sequences:
        if row_lengths[-1] + len(sequence.token_ids) > max_seq_len:
            rows.append([])
            row_lengths.append(0)
        rows[-1].append(sequence)
        row_lengths[-1] += len(sequence.token_ids)

    shape = (len(rows), max_seq_len)
    token_ids = np.zeros(shape, dtype=np.int32)
    scientific_position_ids = np.full(shape, -1, dtype=np.int32)
    rotary_position_ids = np.zeros(shape, dtype=np.int32)
    target_ids = np.full(shape, -1, dtype=np.int32)
    loss_weights = np.zeros(shape, dtype=np.float32)
    segment_ids = np.full(shape, -1, dtype=np.int32)
    locations = []
    for row_index, row in enumerate(rows):
        offset = 0
        for segment_id, sequence in enumerate(row):
            end = offset + len(sequence.token_ids)
            token_ids[row_index, offset:end] = sequence.token_ids
            scientific_position_ids[row_index, offset:end] = sequence.scientific_position_ids
            rotary_position_ids[row_index, offset:end] = sequence.rotary_position_ids
            target_ids[row_index, offset:end] = sequence.target_ids
            loss_weights[row_index, offset:end] = sequence.loss_weights
            segment_ids[row_index, offset:end] = segment_id
            locations.append(
                PackedSequenceLocation(
                    sequence.example_id,
                    sequence.call_id,
                    sequence.sequence_id,
                    row_index,
                    offset,
                    end,
                )
            )
            offset = end
    return PackedBatch(
        token_ids,
        scientific_position_ids,
        rotary_position_ids,
        target_ids,
        loss_weights,
        segment_ids,
        tuple(locations),
    )
