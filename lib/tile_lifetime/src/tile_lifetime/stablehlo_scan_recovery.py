# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover a generic affine ``StatefulScan`` from exported StableHLO ``while``."""

import hashlib
from dataclasses import dataclass, replace
from enum import StrEnum

from shuttle.experimental.stablehlo_import import (
    BroadcastAttributes,
    CallAttributes,
    DotAttributes,
    StableHLOGraph,
    StableHLOOperation,
    StableHLOProgram,
    TransposeAttributes,
    WhileAttributes,
    import_stablehlo_program,
)
from tile_lifetime.plan import (
    ScanNumericalContract,
    SemanticErasureReport,
    SemanticLoweringStep,
    StatefulScanSkeleton,
)
from tile_lifetime.semantic_erasure import SemanticErasureError, semantic_erasure_errors
from tile_lifetime.stateful_scan import (
    LogicalAxis,
    ScanPrimitive,
    ScanPrimitiveKind,
    ScanValue,
    ScanValueRole,
    StatefulScan,
    TensorExpression,
    TensorExpressionKind,
    binary_expression,
    contract_expression,
    input_expression,
)
from tile_lifetime.stateful_scan_planner import compile_affine_scan_candidates
from tile_lifetime.stateful_scan_recovery import (
    RecoveredAffineStateUpdate,
    recover_affine_state_update,
)
from tile_lifetime.stateful_scan_reference import (
    NATURAL_AFFINE_SCAN_INPUT_NAMES,
    NaturalAffineScanConfig,
    export_natural_affine_scan,
)


class StableHLOScanRecoveryError(ValueError):
    """Raised when structured StableHLO is not a supported ordered affine scan."""


class StatefulScanSourceKind(StrEnum):
    """Verified source boundary for an accepted StatefulScan compilation."""

    JAX_EXPORT_STABLEHLO_WHILE = "jax_export_stablehlo_while"
    STABLEHLO_WHILE = "stablehlo_while"
    REFERENCE_TENSOR_EXPRESSION = "reference_tensor_expression"


@dataclass(frozen=True)
class StatefulScanProvenance:
    """Immutable evidence that candidate generation began at structured StableHLO."""

    source_kind: StatefulScanSourceKind
    artifact_sha256: str
    structured_while_count: int


@dataclass(frozen=True)
class StableHLOStatefulScanCompilation:
    """Natural StableHLO scan, recovered affine structure, and physical candidates."""

    program: StatefulScan
    recovered_update: RecoveredAffineStateUpdate
    candidates: tuple[StatefulScanSkeleton, ...]
    source_operation_count: int
    semantic_erasure_report: SemanticErasureReport
    provenance: StatefulScanProvenance


@dataclass(frozen=True)
class _ScanStep:
    entry_graph: StableHLOGraph
    while_operation: StableHLOOperation
    body_call: StableHLOOperation
    step_graph: StableHLOGraph
    state_input_index: int
    state_output_index: int
    output_index: int
    input_names: tuple[str, ...]
    sequence_length: int


def compile_stablehlo_stateful_scan(
    artifact: bytes,
    *,
    input_names: tuple[str, ...] | None = None,
    chunk_sizes: tuple[int, ...] = (32, 64),
) -> StableHLOStatefulScanCompilation:
    """Compile one natural ``jax.lax.scan`` export without matching a model name."""
    source = import_stablehlo_program(artifact, input_names=input_names)
    step = _recover_scan_step(source)
    axes = _infer_logical_axes(step.step_graph)
    expressions = _build_tensor_expressions(step.step_graph, axes, step.input_names)
    state_output_id = step.step_graph.outputs[step.state_output_index]
    output_id = step.step_graph.outputs[step.output_index]
    state_input_id = step.step_graph.inputs[step.state_input_index]
    state_name = step.input_names[step.state_input_index]
    recovered = recover_affine_state_update(expressions[state_output_id], state_name)
    program = _build_stateful_scan(step, axes, state_output_id=state_output_id, output_id=output_id)
    state_value = step.step_graph.value(state_input_id)
    report = _stateful_scan_erasure_report(program, recovered)
    _validate_stateful_scan_report(program, recovered, report)
    candidates = compile_affine_scan_candidates(
        recovered,
        ordered_axis=program.ordered_axis.label or str(program.ordered_axis.id),
        length=step.sequence_length,
        state=program.state_input,
        state_shape=state_value.shape,
        state_dtype=state_value.dtype,
        output=program.scan_outputs[0],
        state_layout="logical_axes_" + "_".join(str(axis.id) for axis in recovered.state_axes),
        chunk_sizes=chunk_sizes,
    )
    compilation = StableHLOStatefulScanCompilation(
        program=program,
        recovered_update=recovered,
        candidates=candidates,
        source_operation_count=len(step.step_graph.operations),
        semantic_erasure_report=report,
        provenance=StatefulScanProvenance(
            source_kind=StatefulScanSourceKind.STABLEHLO_WHILE,
            artifact_sha256=hashlib.sha256(artifact).hexdigest(),
            structured_while_count=1,
        ),
    )
    validate_stateful_scan_semantic_erasure(compilation)
    return compilation


def compile_natural_affine_scan(
    config: NaturalAffineScanConfig,
    *,
    chunk_sizes: tuple[int, ...] = (32, 64),
) -> StableHLOStatefulScanCompilation:
    """Compile ordinary JAX ``lax.scan`` math through exported StableHLO."""
    compilation = compile_stablehlo_stateful_scan(
        export_natural_affine_scan(config),
        input_names=NATURAL_AFFINE_SCAN_INPUT_NAMES,
        chunk_sizes=chunk_sizes,
    )
    natural = replace(
        compilation,
        provenance=replace(
            compilation.provenance,
            source_kind=StatefulScanSourceKind.JAX_EXPORT_STABLEHLO_WHILE,
        ),
    )
    validate_stateful_scan_semantic_erasure(natural)
    return natural


def stateful_scan_scheduling_keys(
    program: StatefulScan,
    recovered: RecoveredAffineStateUpdate,
) -> tuple[str, ...]:
    """Derive candidate-selection keys from generic scan structure only."""
    state = program.value(program.state_input)
    primitive_keys = tuple(
        f"{primitive.kind.value}:inputs={len(primitive.inputs)}:reduction_rank={len(primitive.reduction_axes)}"
        for primitive in (*program.update, *program.read)
    )
    return (
        "scan:"
        f"ordered_extent={program.ordered_axis.extent}:"
        f"state_rank={len(state.axes)}:"
        f"inputs={len(program.scan_inputs)}:"
        f"outputs={len(program.scan_outputs)}:"
        f"numerical={program.numerical_contract.value}",
        "transition:"
        f"structure={recovered.transition_structure.value}:"
        f"diagonal_rank={len(recovered.diagonal_scale_axes)}:"
        f"maximum_update_rank={recovered.maximum_low_rank}",
        *primitive_keys,
    )


def validate_stateful_scan_semantic_erasure(
    compilation: StableHLOStatefulScanCompilation,
) -> None:
    """Reject a stale or named scheduling report before scan candidates execute."""
    if compilation.provenance.source_kind not in (
        StatefulScanSourceKind.JAX_EXPORT_STABLEHLO_WHILE,
        StatefulScanSourceKind.STABLEHLO_WHILE,
    ):
        raise SemanticErasureError("accepted StatefulScan candidates must originate from structured StableHLO while")
    if compilation.provenance.structured_while_count != 1:
        raise SemanticErasureError("accepted StatefulScan candidates require exactly one structured StableHLO while")
    _validate_stateful_scan_report(
        compilation.program,
        compilation.recovered_update,
        compilation.semantic_erasure_report,
    )


def _validate_stateful_scan_report(
    program: StatefulScan,
    recovered: RecoveredAffineStateUpdate,
    report: SemanticErasureReport,
) -> None:
    expected = stateful_scan_scheduling_keys(program, recovered)
    errors = list(semantic_erasure_errors(report))
    if report.scheduling_keys != expected:
        errors.append("scheduling keys do not match the recovered generic Scan program")
    if errors:
        raise SemanticErasureError("; ".join(dict.fromkeys(errors)))


def _stateful_scan_erasure_report(
    program: StatefulScan,
    recovered: RecoveredAffineStateUpdate,
) -> SemanticErasureReport:
    provisional = SemanticErasureReport(
        source_semantics=("stablehlo.while", "stablehlo.tensor_expression_body"),
        lowering_steps=(
            SemanticLoweringStep(source_semantic="stablehlo.while", generic_primitives=("Scan",)),
            SemanticLoweringStep(
                source_semantic="stablehlo.tensor_expression_body",
                generic_primitives=("Map", "Contract"),
            ),
        ),
        scheduling_keys=stateful_scan_scheduling_keys(program, recovered),
    )
    return replace(provisional, validation_errors=semantic_erasure_errors(provisional))


def _recover_scan_step(source: StableHLOProgram) -> _ScanStep:
    entry = source.function(source.entry).graph
    while_operations = tuple(
        operation for operation in entry.operations if isinstance(operation.attributes, WhileAttributes)
    )
    if len(while_operations) != 1:
        raise StableHLOScanRecoveryError(f"expected one structured while, found {len(while_operations)}")
    while_operation = while_operations[0]
    attributes = while_operation.attributes
    if not isinstance(attributes, WhileAttributes):
        raise AssertionError("while attribute narrowing failed")
    body = attributes.body
    candidates: list[tuple[StableHLOOperation, int, int]] = []
    for operation in body.operations:
        if not isinstance(operation.attributes, CallAttributes):
            continue
        callee = source.function(operation.attributes.callee).graph
        if len(operation.outputs) < 2 or len(callee.outputs) != len(operation.outputs):
            continue
        for result_index, result in enumerate(operation.outputs):
            if result not in body.outputs:
                continue
            carry_index = body.outputs.index(result)
            carry_argument = body.arguments[carry_index]
            matching_inputs = tuple(index for index, value in enumerate(operation.inputs) if value == carry_argument)
            if len(matching_inputs) == 1:
                candidates.append((operation, matching_inputs[0], result_index))
    if len(candidates) != 1:
        raise StableHLOScanRecoveryError(f"expected one state-carrying scan body call, found {len(candidates)}")
    body_call, state_input_index, state_output_index = candidates[0]
    call_attributes = body_call.attributes
    if not isinstance(call_attributes, CallAttributes):
        raise AssertionError("call attribute narrowing failed")
    step_graph = source.function(call_attributes.callee).graph
    nonstate_outputs = tuple(index for index in range(len(step_graph.outputs)) if index != state_output_index)
    if len(nonstate_outputs) != 1:
        raise StableHLOScanRecoveryError("the first scan importer requires one emitted value")
    output_index = nonstate_outputs[0]
    input_names = _recover_step_input_names(entry, while_operation, body, body_call, state_input_index)
    sequence_lengths = tuple(
        entry.value(while_operation.inputs[_body_argument_index(body, value)]).shape[0]
        for index, value in enumerate(body_call.inputs)
        if index != state_input_index and value in body.arguments
    )
    indexed_lengths = tuple(
        _scan_major_length(entry, while_operation, body, body_call, index)
        for index in range(len(body_call.inputs))
        if index != state_input_index
    )
    lengths = tuple(length for length in (*sequence_lengths, *indexed_lengths) if length is not None)
    if not lengths or len(set(lengths)) != 1:
        raise StableHLOScanRecoveryError(f"scan inputs have inconsistent ordered extents {lengths}")
    return _ScanStep(
        entry_graph=entry,
        while_operation=while_operation,
        body_call=body_call,
        step_graph=step_graph,
        state_input_index=state_input_index,
        state_output_index=state_output_index,
        output_index=output_index,
        input_names=input_names,
        sequence_length=lengths[0],
    )


def _recover_step_input_names(
    entry: StableHLOGraph,
    while_operation: StableHLOOperation,
    body,
    body_call: StableHLOOperation,
    state_input_index: int,
) -> tuple[str, ...]:
    names: list[str] = []
    for index, value in enumerate(body_call.inputs):
        if index == state_input_index:
            names.append("state_prev")
            continue
        producer = next(
            (operation for operation in body.operations if value in operation.outputs),
            None,
        )
        if producer is None or not isinstance(producer.attributes, CallAttributes) or not producer.inputs:
            names.append(f"scan_input_{index}")
            continue
        source_argument = producer.inputs[0]
        if source_argument not in body.arguments:
            names.append(f"scan_input_{index}")
            continue
        carry_index = _body_argument_index(body, source_argument)
        names.append(_source_input_name(entry, while_operation.inputs[carry_index]))
    return tuple(names)


def _source_input_name(graph: StableHLOGraph, value_id: int) -> str:
    current = value_id
    while current not in graph.inputs:
        producer = graph.producer(current)
        if producer is None or producer.kind not in ("transpose", "reshape", "convert") or len(producer.inputs) != 1:
            return f"source_v{value_id}"
        current = producer.inputs[0]
    return graph.value(current).name


def _scan_major_length(
    entry: StableHLOGraph,
    while_operation: StableHLOOperation,
    body,
    body_call: StableHLOOperation,
    input_index: int,
) -> int | None:
    value = body_call.inputs[input_index]
    producer = next((operation for operation in body.operations if value in operation.outputs), None)
    if producer is None or not isinstance(producer.attributes, CallAttributes) or not producer.inputs:
        return None
    source_argument = producer.inputs[0]
    if source_argument not in body.arguments:
        return None
    carry_index = _body_argument_index(body, source_argument)
    shape = entry.value(while_operation.inputs[carry_index]).shape
    return shape[0] if shape else None


def _body_argument_index(body, value_id: int) -> int:
    try:
        return body.arguments.index(value_id)
    except ValueError as error:
        raise StableHLOScanRecoveryError(f"value {value_id} is not a while-body argument") from error


class _DisjointAxes:
    def __init__(self, graph: StableHLOGraph):
        self.parent = {
            (value.id, dimension): (value.id, dimension)
            for value in graph.values
            for dimension in range(len(value.shape))
        }
        self.extent = {
            (value.id, dimension): value.shape[dimension]
            for value in graph.values
            for dimension in range(len(value.shape))
        }

    def find(self, node: tuple[int, int]) -> tuple[int, int]:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: tuple[int, int], right: tuple[int, int]) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.extent[left_root] != self.extent[right_root]:
            raise StableHLOScanRecoveryError(
                f"logical-axis constraint joins extents {self.extent[left_root]} and {self.extent[right_root]}"
            )
        self.parent[right_root] = left_root


def _infer_logical_axes(graph: StableHLOGraph) -> dict[tuple[int, int], LogicalAxis]:
    axes = _DisjointAxes(graph)
    for operation in graph.operations:
        if not operation.outputs:
            continue
        output = operation.outputs[0]
        if operation.kind in ("convert", "exponential", "negate"):
            _union_corresponding_axes(axes, operation.inputs[0], output, graph)
        elif operation.kind in ("add", "subtract", "multiply", "divide", "maximum"):
            for value in operation.inputs:
                _union_corresponding_axes(axes, value, output, graph)
        elif operation.kind == "broadcast_in_dim":
            attributes = operation.attributes
            if not isinstance(attributes, BroadcastAttributes):
                raise AssertionError("broadcast attribute narrowing failed")
            for input_dimension, output_dimension in enumerate(attributes.dimensions):
                input_extent = graph.value(operation.inputs[0]).shape[input_dimension]
                output_extent = graph.value(output).shape[output_dimension]
                if input_extent == output_extent:
                    axes.union(
                        (operation.inputs[0], input_dimension),
                        (output, output_dimension),
                    )
        elif operation.kind == "transpose":
            attributes = operation.attributes
            if not isinstance(attributes, TransposeAttributes):
                raise AssertionError("transpose attribute narrowing failed")
            for output_dimension, input_dimension in enumerate(attributes.permutation):
                axes.union((operation.inputs[0], input_dimension), (output, output_dimension))
        elif operation.kind == "reshape":
            _union_nonunit_reshape_axes(axes, operation.inputs[0], output, graph)
        elif operation.kind == "dot_general":
            _constrain_dot_axes(axes, graph, operation)
    roots = sorted({axes.find(node) for node in axes.parent})
    logical_by_root = {
        root: LogicalAxis(id=index, extent=axes.extent[root], label=f"axis_{index}") for index, root in enumerate(roots)
    }
    return {node: logical_by_root[axes.find(node)] for node in axes.parent}


def _union_corresponding_axes(
    axes: _DisjointAxes,
    input_id: int,
    output_id: int,
    graph: StableHLOGraph,
) -> None:
    input_shape = graph.value(input_id).shape
    output_shape = graph.value(output_id).shape
    if input_shape != output_shape:
        raise StableHLOScanRecoveryError(f"pointwise operation changed shape {input_shape} -> {output_shape}")
    for dimension in range(len(input_shape)):
        axes.union((input_id, dimension), (output_id, dimension))


def _union_nonunit_reshape_axes(
    axes: _DisjointAxes,
    input_id: int,
    output_id: int,
    graph: StableHLOGraph,
) -> None:
    input_dimensions = tuple(index for index, extent in enumerate(graph.value(input_id).shape) if extent != 1)
    output_dimensions = tuple(index for index, extent in enumerate(graph.value(output_id).shape) if extent != 1)
    if tuple(graph.value(input_id).shape[index] for index in input_dimensions) != tuple(
        graph.value(output_id).shape[index] for index in output_dimensions
    ):
        raise StableHLOScanRecoveryError("scan-step reshape merges or splits non-unit logical axes")
    for input_dimension, output_dimension in zip(input_dimensions, output_dimensions, strict=True):
        axes.union((input_id, input_dimension), (output_id, output_dimension))


def _constrain_dot_axes(
    axes: _DisjointAxes,
    graph: StableHLOGraph,
    operation: StableHLOOperation,
) -> None:
    attributes = operation.attributes
    if not isinstance(attributes, DotAttributes):
        raise AssertionError("dot attribute narrowing failed")
    lhs, rhs = operation.inputs
    output = operation.outputs[0]
    for lhs_dimension, rhs_dimension in zip(
        attributes.lhs_batching_dimensions,
        attributes.rhs_batching_dimensions,
        strict=True,
    ):
        axes.union((lhs, lhs_dimension), (rhs, rhs_dimension))
    for lhs_dimension, rhs_dimension in zip(
        attributes.lhs_contracting_dimensions,
        attributes.rhs_contracting_dimensions,
        strict=True,
    ):
        axes.union((lhs, lhs_dimension), (rhs, rhs_dimension))
    lhs_excluded = set(attributes.lhs_batching_dimensions) | set(attributes.lhs_contracting_dimensions)
    rhs_excluded = set(attributes.rhs_batching_dimensions) | set(attributes.rhs_contracting_dimensions)
    lhs_free = tuple(index for index in range(len(graph.value(lhs).shape)) if index not in lhs_excluded)
    rhs_free = tuple(index for index in range(len(graph.value(rhs).shape)) if index not in rhs_excluded)
    output_dimensions = (
        tuple((lhs, index) for index in attributes.lhs_batching_dimensions)
        + tuple((lhs, index) for index in lhs_free)
        + tuple((rhs, index) for index in rhs_free)
    )
    if len(output_dimensions) != len(graph.value(output).shape):
        raise StableHLOScanRecoveryError("dot output rank does not match StableHLO dimension numbers")
    for output_dimension, source in enumerate(output_dimensions):
        axes.union(source, (output, output_dimension))


def _value_axes(
    value_id: int,
    graph: StableHLOGraph,
    axes: dict[tuple[int, int], LogicalAxis],
) -> tuple[LogicalAxis, ...]:
    return tuple(axes[(value_id, dimension)] for dimension in range(len(graph.value(value_id).shape)))


def _build_tensor_expressions(
    graph: StableHLOGraph,
    axes: dict[tuple[int, int], LogicalAxis],
    input_names: tuple[str, ...],
) -> dict[int, TensorExpression]:
    expressions = {
        value_id: input_expression(input_names[index], _value_axes(value_id, graph, axes))
        for index, value_id in enumerate(graph.inputs)
    }
    for operation in graph.operations:
        if len(operation.outputs) != 1:
            raise StableHLOScanRecoveryError(f"scan-step operation {operation.kind} must have one result")
        output = operation.outputs[0]
        output_axes = _value_axes(output, graph, axes)
        inputs = tuple(expressions[value] for value in operation.inputs)
        if operation.kind in ("add", "subtract", "multiply"):
            kind = TensorExpressionKind(operation.kind)
            expression = binary_expression(kind, inputs[0], inputs[1], output_axes)
        elif operation.kind == "dot_general":
            attributes = operation.attributes
            if not isinstance(attributes, DotAttributes):
                raise AssertionError("dot attribute narrowing failed")
            reduction_axes = tuple(
                axes[(operation.inputs[0], dimension)] for dimension in attributes.lhs_contracting_dimensions
            )
            expression = contract_expression(*inputs, axes=output_axes, reduction_axes=reduction_axes)
        elif operation.kind == "broadcast_in_dim":
            expression = inputs[0]
        elif operation.kind in (
            "convert",
            "exponential",
            "negate",
            "reshape",
            "transpose",
        ):
            operation_name = "exp" if operation.kind == "exponential" else operation.kind
            semantic_axes = inputs[0].axes if operation.kind in ("convert", "exponential", "negate") else output_axes
            expression = TensorExpression(
                kind=TensorExpressionKind.UNARY,
                axes=semantic_axes,
                inputs=(inputs[0],),
                operation=operation_name,
            )
        else:
            raise StableHLOScanRecoveryError(f"unsupported scan-step expression {operation.kind}")
        expressions[output] = expression
    return expressions


def _build_stateful_scan(
    step: _ScanStep,
    axes: dict[tuple[int, int], LogicalAxis],
    *,
    state_output_id: int,
    output_id: int,
) -> StatefulScan:
    graph = step.step_graph
    state_input_id = graph.inputs[step.state_input_index]
    ordered_axis = LogicalAxis(
        id=max((axis.id for axis in axes.values()), default=-1) + 1,
        extent=step.sequence_length,
        label="position",
    )
    names = {value.id: f"v{value.id}" for value in graph.values}
    for index, value_id in enumerate(graph.inputs):
        names[value_id] = step.input_names[index]
    names[state_input_id] = "state_prev"
    names[state_output_id] = "state_next"
    names[output_id] = "output"

    update_operations = _ancestor_operations(graph, state_output_id)
    output_operations = _ancestor_operations(graph, output_id)
    read_operations = tuple(operation for operation in output_operations if operation not in update_operations)
    used_operations = (*update_operations, *read_operations)
    used_values = {value for operation in used_operations for value in (*operation.inputs, *operation.outputs)} | {
        state_input_id,
        state_output_id,
        output_id,
    }
    values: list[ScanValue] = []
    for value_id in sorted(used_values):
        value = graph.value(value_id)
        logical_axes = _value_axes(value_id, graph, axes)
        if value_id == state_input_id or value_id == state_output_id:
            role = ScanValueRole.STATE
        elif value_id == output_id:
            role = ScanValueRole.OUTPUT
            logical_axes = _insert_ordered_axis(logical_axes, ordered_axis)
        elif value_id in graph.inputs:
            role = ScanValueRole.INPUT
            logical_axes = _insert_ordered_axis(logical_axes, ordered_axis)
        else:
            role = ScanValueRole.TEMPORARY
        values.append(ScanValue(names[value_id], logical_axes, value.dtype, role))

    update = tuple(_scan_primitive(operation, graph, axes, names) for operation in update_operations)
    read = tuple(_scan_primitive(operation, graph, axes, names) for operation in read_operations)
    scan_inputs = tuple(
        names[value_id]
        for index, value_id in enumerate(graph.inputs)
        if index != step.state_input_index and value_id in used_values
    )
    return StatefulScan(
        name="stablehlo_affine_scan",
        ordered_axis=ordered_axis,
        values=tuple(values),
        state_input="state_prev",
        state_output="state_next",
        scan_inputs=scan_inputs,
        scan_outputs=("output",),
        update=update,
        read=read,
        numerical_contract=ScanNumericalContract.SOURCE_ORDERED,
    )


def _insert_ordered_axis(
    axes: tuple[LogicalAxis, ...],
    ordered_axis: LogicalAxis,
) -> tuple[LogicalAxis, ...]:
    insertion = 1 if axes else 0
    return (*axes[:insertion], ordered_axis, *axes[insertion:])


def _ancestor_operations(graph: StableHLOGraph, value_id: int) -> tuple[StableHLOOperation, ...]:
    needed: set[int] = set()

    def visit(value: int) -> None:
        producer = graph.producer(value)
        if producer is None or producer.id in needed:
            return
        needed.add(producer.id)
        for input_value in producer.inputs:
            visit(input_value)

    visit(value_id)
    return tuple(operation for operation in graph.operations if operation.id in needed)


def _scan_primitive(
    operation: StableHLOOperation,
    graph: StableHLOGraph,
    axes: dict[tuple[int, int], LogicalAxis],
    names: dict[int, str],
) -> ScanPrimitive:
    kind = ScanPrimitiveKind.CONTRACT if operation.kind == "dot_general" else ScanPrimitiveKind.MAP
    reduction_axes: tuple[str, ...] = ()
    if isinstance(operation.attributes, DotAttributes):
        reduction_axes = tuple(
            axes[(operation.inputs[0], dimension)].label or str(axes[(operation.inputs[0], dimension)].id)
            for dimension in operation.attributes.lhs_contracting_dimensions
        )
    return ScanPrimitive(
        name=f"source_op_{operation.id}_{operation.kind}",
        kind=kind,
        inputs=tuple(names[value] for value in operation.inputs),
        output=names[operation.outputs[0]],
        equation=operation.kind,
        reduction_axes=reduction_axes,
    )
