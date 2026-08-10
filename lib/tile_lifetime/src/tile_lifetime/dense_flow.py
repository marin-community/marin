# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generic Flow erasure for the connected dense Transformer prototype."""

from dataclasses import dataclass, replace
from enum import StrEnum

from shuttle.ir import DType
from tile_lifetime.ir import (
    LinearOp,
    PairwiseSwiGLUOp,
    QKVProjectionOp,
    ResidualAddOp,
    RMSNormOp,
    RoPEOp,
    ScaledDotProductAttentionOp,
    SemanticOp,
    TensorGraph,
    TensorValue,
    ViewOp,
)
from tile_lifetime.plan import SemanticErasureReport, SemanticLoweringStep
from tile_lifetime.semantic_erasure import SemanticErasureError, semantic_erasure_errors
from tile_lifetime.tensor_program import (
    FoldReducer,
    ScalarExpression,
    ScalarExpressionKind,
    scalar_binary,
    scalar_constant,
    scalar_input,
    scalar_unary,
)


@dataclass(frozen=True)
class FlowValue:
    """A logical tensor value retained after frontend operation names erase."""

    name: str
    shape: tuple[int, ...]
    dtype: DType


@dataclass(frozen=True)
class FlowContract:
    """A generic two-input contraction."""

    name: str
    inputs: tuple[FlowValue, FlowValue]
    output: FlowValue
    reduction_dimensions: tuple[int, ...]
    accumulation_dtype: DType


class FlowMapIteration(StrEnum):
    """Indexing family used by a generic Map."""

    POINTWISE = "pointwise"
    BROADCAST = "broadcast"
    ADJACENT_PAIR = "adjacent_pair"
    VIEW = "view"
    PARTITION = "partition"


@dataclass(frozen=True)
class FlowMap:
    """One or more scalar expressions evaluated over a generic index map."""

    name: str
    inputs: tuple[FlowValue, ...]
    outputs: tuple[FlowValue, ...]
    iteration: FlowMapIteration
    expressions: tuple[ScalarExpression, ...]
    attributes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class FlowFold:
    """An associative Fold over one logical tensor dimension."""

    name: str
    input: FlowValue
    output: FlowValue
    reduction_dimensions: tuple[int, ...]
    reducer: FoldReducer
    accumulation_dtype: DType


@dataclass(frozen=True)
class FlowDomainRestriction:
    """An index predicate limiting the domain of subsequent computation."""

    name: str
    input: FlowValue
    output: FlowValue
    predicate: ScalarExpression


DenseFlowOperation = FlowContract | FlowMap | FlowFold | FlowDomainRestriction


@dataclass(frozen=True)
class ErasedDenseFlowProgram:
    """Connected dense program after every frontend operation name has erased."""

    operations: tuple[DenseFlowOperation, ...]
    report: SemanticErasureReport

    def with_operations(self, operations: tuple[DenseFlowOperation, ...]) -> "ErasedDenseFlowProgram":
        """Replace generic semantics and regenerate candidate-selection signatures."""
        report = replace(self.report, scheduling_keys=dense_flow_scheduling_keys(operations), validation_errors=())
        errors = semantic_erasure_errors(report)
        return ErasedDenseFlowProgram(operations, replace(report, validation_errors=errors))


def erase_dense_transformer_semantics(graph: TensorGraph) -> ErasedDenseFlowProgram:
    """Canonicalize named dense operations into generic Flow algebra."""
    operations: list[DenseFlowOperation] = []
    lowering_steps: list[SemanticLoweringStep] = []

    def value(tensor: TensorValue) -> FlowValue:
        return FlowValue(tensor.name, tensor.shape, tensor.dtype)

    def synthetic(name: str, shape: tuple[int, ...], dtype: DType) -> FlowValue:
        return FlowValue(name, shape, dtype)

    def operation_name(kind: str) -> str:
        return f"{kind}.{len(operations)}"

    for semantic in graph.operations:
        if isinstance(semantic, ViewOp):
            operations.append(
                FlowMap(
                    operation_name("map"),
                    (value(semantic.input),),
                    (value(semantic.output),),
                    FlowMapIteration.VIEW,
                    (),
                    (("shape", repr(semantic.output.shape)),),
                )
            )
            lowering_steps.append(SemanticLoweringStep(type(semantic).__name__, ("Map",)))
        elif isinstance(semantic, LinearOp):
            operations.append(
                FlowContract(
                    operation_name("contract"),
                    (value(semantic.input), value(semantic.weight)),
                    value(semantic.output),
                    (1,),
                    semantic.accumulation_dtype,
                )
            )
            lowering_steps.append(SemanticLoweringStep(type(semantic).__name__, ("Contract",)))
        elif isinstance(semantic, QKVProjectionOp):
            packed = value(semantic.output)
            operations.append(
                FlowContract(
                    operation_name("contract"),
                    (value(semantic.input), value(semantic.weight)),
                    packed,
                    (len(semantic.input.shape) - 1,),
                    semantic.accumulation_dtype,
                )
            )
            segments = (semantic.query.shape[-2] * semantic.query.shape[-1],)
            segments += (semantic.key.shape[-2] * semantic.key.shape[-1],)
            segments += (semantic.value.shape[-2] * semantic.value.shape[-1],)
            operations.append(
                FlowMap(
                    operation_name("map"),
                    (packed,),
                    (value(semantic.query), value(semantic.key), value(semantic.value)),
                    FlowMapIteration.PARTITION,
                    (),
                    (("segment_extents", repr(segments)),),
                )
            )
            lowering_steps.append(SemanticLoweringStep(type(semantic).__name__, ("Contract", "Map")))
        elif isinstance(semantic, RoPEOp):
            expression = _pairwise_linear_rotation()
            for tensor_input, tensor_output in (
                (semantic.query, semantic.output),
                (semantic.key, semantic.key_output),
            ):
                operations.append(
                    FlowMap(
                        operation_name("map"),
                        (value(tensor_input), value(semantic.sine), value(semantic.cosine)),
                        (value(tensor_output),),
                        FlowMapIteration.ADJACENT_PAIR,
                        expression,
                        (("rotary_extent", str(semantic.rotary_dimension)),),
                    )
                )
            lowering_steps.append(SemanticLoweringStep(type(semantic).__name__, ("Map",)))
        elif isinstance(semantic, ScaledDotProductAttentionOp):
            _lower_normalized_weighted_fold(semantic, operations, value, synthetic, operation_name)
            lowering_steps.append(
                SemanticLoweringStep(
                    type(semantic).__name__,
                    ("Contract", "Map", "DomainRestriction", "Fold", "Contract", "Map"),
                )
            )
        elif isinstance(semantic, ResidualAddOp):
            left = value(semantic.left)
            right = value(semantic.right)
            operations.append(
                FlowMap(
                    operation_name("map"),
                    (left, right),
                    (value(semantic.output),),
                    FlowMapIteration.POINTWISE,
                    (
                        scalar_binary(
                            ScalarExpressionKind.ADD,
                            scalar_input(left.name),
                            scalar_input(right.name),
                        ),
                    ),
                )
            )
            lowering_steps.append(SemanticLoweringStep(type(semantic).__name__, ("Map",)))
        elif isinstance(semantic, RMSNormOp):
            _lower_fold_row_scale(semantic, operations, value, synthetic, operation_name)
            lowering_steps.append(SemanticLoweringStep(type(semantic).__name__, ("Map", "Fold", "Map")))
        elif isinstance(semantic, PairwiseSwiGLUOp):
            operations.append(
                FlowMap(
                    operation_name("map"),
                    (value(semantic.input),),
                    (value(semantic.output),),
                    FlowMapIteration.ADJACENT_PAIR,
                    (pairwise_silu_product_expression(),),
                )
            )
            lowering_steps.append(SemanticLoweringStep(type(semantic).__name__, ("Map",)))
        else:
            _unsupported_semantic(semantic)

    source_semantics = tuple(dict.fromkeys(type(operation).__name__ for operation in graph.operations))
    report = SemanticErasureReport(
        source_semantics,
        tuple(lowering_steps),
        dense_flow_scheduling_keys(tuple(operations)),
    )
    report = replace(report, validation_errors=semantic_erasure_errors(report))
    erased = ErasedDenseFlowProgram(tuple(operations), report)
    validate_erased_dense_flow(erased)
    return erased


def dense_flow_scheduling_keys(operations: tuple[DenseFlowOperation, ...]) -> tuple[str, ...]:
    """Derive workload-name-free scheduling keys from Flow structure."""
    keys: list[str] = []
    for operation in operations:
        if isinstance(operation, FlowContract):
            keys.append(
                f"contract:input_ranks={tuple(len(value.shape) for value in operation.inputs)}:"
                f"output_rank={len(operation.output.shape)}:reduction_rank={len(operation.reduction_dimensions)}"
            )
        elif isinstance(operation, FlowFold):
            keys.append(f"fold:{operation.reducer.value}:dimensions={operation.reduction_dimensions}")
        elif isinstance(operation, FlowDomainRestriction):
            keys.append(f"domain_restriction:{_expression_signature(operation.predicate)}")
        else:
            signatures = tuple(_expression_signature(expression) for expression in operation.expressions)
            keys.append(f"map:{operation.iteration.value}:expressions={signatures}")
    return tuple(keys)


def validate_erased_dense_flow(erased: ErasedDenseFlowProgram) -> None:
    """Machine-check the full region before any physical candidate is selected."""
    expected = dense_flow_scheduling_keys(erased.operations)
    errors = list(semantic_erasure_errors(erased.report))
    if erased.report.scheduling_keys != expected:
        errors.append("dense Flow scheduling keys do not match the supplied generic operations")
    if errors:
        raise SemanticErasureError("; ".join(dict.fromkeys(errors)))


def _lower_fold_row_scale(
    semantic: RMSNormOp,
    operations: list[DenseFlowOperation],
    value,
    synthetic,
    operation_name,
) -> None:
    source = value(semantic.input)
    square = synthetic(f"value.{semantic.id}.square", source.shape, semantic.reduction_dtype)
    row_shape = tuple(extent for index, extent in enumerate(source.shape) if index != semantic.axis)
    summed = synthetic(f"value.{semantic.id}.fold", row_shape, semantic.reduction_dtype)
    row_scalar = synthetic(f"value.{semantic.id}.row_scalar", row_shape, semantic.reduction_dtype)
    feature_scaled = synthetic(f"value.{semantic.id}.feature_scaled", source.shape, source.dtype)
    operations.extend(
        (
            FlowMap(
                operation_name("map"),
                (source,),
                (square,),
                FlowMapIteration.POINTWISE,
                (
                    scalar_binary(
                        ScalarExpressionKind.MULTIPLY,
                        scalar_input(source.name),
                        scalar_input(source.name),
                    ),
                ),
            ),
            FlowFold(
                f"fold.{len(operations) + 1}",
                square,
                summed,
                (semantic.axis,),
                FoldReducer.SUM,
                semantic.reduction_dtype,
            ),
            FlowMap(
                f"map.{len(operations) + 2}",
                (summed,),
                (row_scalar,),
                FlowMapIteration.BROADCAST,
                (
                    scalar_unary(
                        ScalarExpressionKind.RSQRT,
                        scalar_binary(
                            ScalarExpressionKind.ADD,
                            scalar_binary(
                                ScalarExpressionKind.DIVIDE,
                                scalar_input(summed.name),
                                scalar_constant(source.shape[semantic.axis]),
                            ),
                            scalar_constant(semantic.epsilon),
                        ),
                    ),
                ),
            ),
            FlowMap(
                f"map.{len(operations) + 3}",
                (source, value(semantic.gamma)),
                (feature_scaled,),
                FlowMapIteration.BROADCAST,
                (
                    scalar_binary(
                        ScalarExpressionKind.MULTIPLY,
                        scalar_input(source.name),
                        scalar_input(semantic.gamma.name),
                    ),
                ),
            ),
            FlowMap(
                f"map.{len(operations) + 4}",
                (feature_scaled, row_scalar),
                (value(semantic.output),),
                FlowMapIteration.BROADCAST,
                (
                    scalar_binary(
                        ScalarExpressionKind.MULTIPLY,
                        scalar_input(feature_scaled.name),
                        scalar_input(row_scalar.name),
                    ),
                ),
            ),
        )
    )


def _lower_normalized_weighted_fold(
    semantic: ScaledDotProductAttentionOp,
    operations: list[DenseFlowOperation],
    value,
    synthetic,
    operation_name,
) -> None:
    batch, query_length, query_heads, _ = semantic.query.shape
    key_length = semantic.key.shape[1]
    score_shape = (batch, query_heads, query_length, key_length)
    row_shape = score_shape[:-1]
    raw = synthetic(f"value.{semantic.id}.contract0", score_shape, semantic.accumulation_dtype)
    scaled = synthetic(f"value.{semantic.id}.scaled", score_shape, semantic.accumulation_dtype)
    restricted = synthetic(f"value.{semantic.id}.restricted", score_shape, semantic.accumulation_dtype)
    row_max = synthetic(f"value.{semantic.id}.max", row_shape, semantic.accumulation_dtype)
    centered = synthetic(f"value.{semantic.id}.centered", score_shape, semantic.accumulation_dtype)
    exponentials = synthetic(f"value.{semantic.id}.exp", score_shape, semantic.accumulation_dtype)
    row_sum = synthetic(f"value.{semantic.id}.sum", row_shape, semantic.accumulation_dtype)
    weighted = synthetic(f"value.{semantic.id}.contract1", semantic.output.shape, semantic.accumulation_dtype)
    operations.extend(
        (
            FlowContract(
                operation_name("contract"),
                (value(semantic.query), value(semantic.key)),
                raw,
                (3,),
                semantic.accumulation_dtype,
            ),
            FlowMap(
                f"map.{len(operations) + 1}",
                (raw,),
                (scaled,),
                FlowMapIteration.POINTWISE,
                (
                    scalar_binary(
                        ScalarExpressionKind.MULTIPLY,
                        scalar_input(raw.name),
                        scalar_constant(semantic.scale),
                    ),
                ),
            ),
            FlowDomainRestriction(
                f"domain_restriction.{len(operations) + 2}",
                scaled,
                restricted,
                (
                    scalar_binary(
                        ScalarExpressionKind.LESS_EQUAL,
                        scalar_input("key.position"),
                        scalar_input("query.position"),
                    )
                    if semantic.causal
                    else scalar_constant(True)
                ),
            ),
            FlowFold(
                f"fold.{len(operations) + 3}",
                restricted,
                row_max,
                (3,),
                FoldReducer.MAXIMUM,
                semantic.accumulation_dtype,
            ),
            FlowMap(
                f"map.{len(operations) + 4}",
                (restricted, row_max),
                (centered,),
                FlowMapIteration.BROADCAST,
                (
                    scalar_binary(
                        ScalarExpressionKind.SUBTRACT,
                        scalar_input(restricted.name),
                        scalar_input(row_max.name),
                    ),
                ),
            ),
            FlowMap(
                f"map.{len(operations) + 5}",
                (centered,),
                (exponentials,),
                FlowMapIteration.POINTWISE,
                (scalar_unary(ScalarExpressionKind.EXP, scalar_input(centered.name)),),
            ),
            FlowFold(
                f"fold.{len(operations) + 6}",
                exponentials,
                row_sum,
                (3,),
                FoldReducer.SUM,
                semantic.accumulation_dtype,
            ),
            FlowContract(
                f"contract.{len(operations) + 7}",
                (exponentials, value(semantic.value)),
                weighted,
                (3,),
                semantic.accumulation_dtype,
            ),
            FlowMap(
                f"map.{len(operations) + 8}",
                (weighted, row_sum),
                (value(semantic.output),),
                FlowMapIteration.BROADCAST,
                (
                    scalar_binary(
                        ScalarExpressionKind.DIVIDE,
                        scalar_input(weighted.name),
                        scalar_input(row_sum.name),
                    ),
                ),
            ),
        )
    )


def pairwise_silu_product_expression() -> ScalarExpression:
    """Return the canonical adjacent-pair SiLU(left) times right expression."""
    left = scalar_input("pair.left")
    right = scalar_input("pair.right")
    sigmoid = scalar_binary(
        ScalarExpressionKind.DIVIDE,
        scalar_constant(1.0),
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_constant(1.0),
            scalar_unary(
                ScalarExpressionKind.EXP,
                scalar_binary(ScalarExpressionKind.MULTIPLY, scalar_constant(-1.0), left),
            ),
        ),
    )
    return scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        scalar_binary(ScalarExpressionKind.MULTIPLY, left, sigmoid),
        right,
    )


def pairwise_product_expression() -> ScalarExpression:
    """Return a mutation-friendly adjacent-pair product expression."""
    return scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        scalar_input("pair.left"),
        scalar_input("pair.right"),
    )


def _pairwise_linear_rotation() -> tuple[ScalarExpression, ScalarExpression]:
    left = scalar_input("pair.left")
    right = scalar_input("pair.right")
    cosine = scalar_input("coefficient.0")
    sine = scalar_input("coefficient.1")
    return (
        scalar_binary(
            ScalarExpressionKind.SUBTRACT,
            scalar_binary(ScalarExpressionKind.MULTIPLY, left, cosine),
            scalar_binary(ScalarExpressionKind.MULTIPLY, right, sine),
        ),
        scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_binary(ScalarExpressionKind.MULTIPLY, left, sine),
            scalar_binary(ScalarExpressionKind.MULTIPLY, right, cosine),
        ),
    )


def _expression_signature(expression: ScalarExpression) -> str:
    if expression.kind is ScalarExpressionKind.INPUT:
        return "input"
    if expression.kind is ScalarExpressionKind.CONSTANT:
        return "constant"
    return f"{expression.kind.value}({','.join(_expression_signature(item) for item in expression.operands)})"


def _unsupported_semantic(operation: SemanticOp) -> None:
    raise SemanticErasureError(f"dense Flow erasure does not support frontend operation {type(operation).__name__}")
