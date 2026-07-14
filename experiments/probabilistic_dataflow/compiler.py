# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from experiments.probabilistic_dataflow.dsl import (
    AttentionPattern,
    CategoricalAxis,
    InferenceProgram,
    MeshAxis,
    Node,
    NodeKind,
    PositionMode,
    UnorderedPairAxis,
)


class CompilationError(ValueError):
    pass


class SplitLeakageError(CompilationError):
    pass


@dataclass(frozen=True)
class ModelCallIR:
    id: int
    target_ids: tuple[int, ...]
    context_ids: tuple[int, ...]
    dependency_call_ids: tuple[int, ...]
    operator: str
    attention_layout: AttentionLayout
    position_mode: PositionMode
    iteration: int = 0
    approximation_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class InferencePlanIR:
    program_name: str
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    factor_ids: tuple[str, ...]
    model_call_budget: int
    generated_token_budget: int
    calls: tuple[ModelCallIR, ...]


def inference_plan_ir(program: InferenceProgram) -> InferencePlanIR:
    """Validate and expose the model-call plan recorded by an inference program."""
    if not program.outputs:
        raise CompilationError(f"Inference program {program.name!r} has not been finished")
    _check_split_isolation(program)
    calls = []
    for call in program.calls:
        notes = []
        if call.operator == "generate":
            for target_id in call.target_ids:
                node = program.node(target_id)
                if node.value_type.token_count > 1:
                    notes.append(
                        f"factor {node.factor_id} is approximated as "
                        f"{node.value_type.token_count} parallel token marginals"
                    )
        if call.resample_fraction is not None:
            notes.append(f"resample_low_confidence_fraction={call.resample_fraction}")
        calls.append(
            ModelCallIR(
                id=call.id,
                target_ids=call.target_ids,
                context_ids=call.context_ids,
                dependency_call_ids=call.dependency_call_ids,
                operator=call.operator,
                attention_layout=(
                    AttentionLayout.FULL if call.document.attention == AttentionPattern.FULL else AttentionLayout.CAUSAL
                ),
                position_mode=call.document.positions,
                iteration=call.iteration,
                approximation_notes=tuple(notes),
            )
        )
    return InferencePlanIR(
        program_name=program.name,
        input_ids=tuple(value.node_id for value in program.external_inputs),
        output_ids=tuple(value.node_id for value in program.outputs),
        factor_ids=tuple(node.factor_id for node in program.nodes if node.factor_id is not None),
        model_call_budget=program.budget.model_calls,
        generated_token_budget=program.budget.generated_tokens,
        calls=tuple(calls),
    )


def _check_split_isolation(program: InferenceProgram) -> None:
    for node in program.nodes:
        if node.kind != NodeKind.DETERMINISTIC or not node.operation or not node.operation.startswith("reduce:"):
            continue
        if len(node.flow.split_keys) > 1 and not node.allows_cross_split:
            raise SplitLeakageError(
                f"Reduction {node.name!r} combines split keys {sorted(node.flow.split_keys)} without explicit permission"
            )


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
    attention_layout: AttentionLayout
    position_mode: PositionMode


@dataclass(frozen=True)
class TransformerExecutionIR:
    example_id: str
    calls: tuple[TransformerCallExecutionIR, ...]


def lower_to_transformer(
    program: InferenceProgram,
    example: ConcreteExample,
    codec: TokenCodec,
) -> TransformerExecutionIR:
    if example.program_name != program.name:
        raise CompilationError(f"Program mismatch: program={program.name!r}, example={example.program_name!r}")
    plan = inference_plan_ir(program)
    calls = []
    for call in plan.calls:
        sequence = _scientific_record_sequence(program, call, example, codec)
        calls.append(
            TransformerCallExecutionIR(
                call_id=call.id,
                operator=call.operator,
                dependency_call_ids=call.dependency_call_ids,
                sequences=(sequence,),
                attention_layout=call.attention_layout,
                position_mode=call.position_mode,
            )
        )
    return TransformerExecutionIR(example.id, tuple(calls))


def _scientific_record_sequence(
    program: InferenceProgram,
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
            scientific_positions.append(_position_id(program, node, index, call.position_mode, codec))
            target_ids.append(-1)
            weights.append(0.0)
    for target_id in call.target_ids:
        node = program.node(target_id)
        for index, value in enumerate(example.value(node)):
            tokens.append(codec.QUERY_ID)
            scientific_positions.append(_position_id(program, node, index, call.position_mode, codec))
            target_ids.append(codec.data(value))
            weights.append(1.0)
    return ExecutionSequenceIR(
        example.id,
        call.id,
        0,
        tuple(tokens),
        tuple(scientific_positions),
        (0,) * len(tokens) if call.position_mode == PositionMode.SCIENTIFIC else tuple(range(len(tokens))),
        tuple(target_ids),
        tuple(weights),
    )


def _position_id(
    program: InferenceProgram,
    node: Node,
    index: int,
    mode: PositionMode,
    codec: TokenCodec,
) -> int:
    if mode == PositionMode.SEQUENCE:
        return -1
    return codec.scientific_position(_scientific_position_key(program, node, index))


def _scientific_position_key(program: InferenceProgram, node: Node, index: int) -> str:
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
    attention_layout: AttentionLayout
    locations: tuple[PackedSequenceLocation, ...]


def pack_transformer_calls(executions: tuple[TransformerExecutionIR, ...], *, max_seq_len: int) -> PackedBatch:
    attention_layouts = {call.attention_layout for execution in executions for call in execution.calls}
    if len(attention_layouts) != 1:
        raise CompilationError(f"Packed calls must share one attention layout, got {sorted(attention_layouts)}")
    attention_layout = attention_layouts.pop()
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
        attention_layout,
        tuple(locations),
    )
