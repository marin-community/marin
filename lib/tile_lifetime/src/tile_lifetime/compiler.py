# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic planning from erased Contract/Map/Fold semantics."""

from dataclasses import dataclass, replace
from enum import StrEnum

from tile_lifetime.dense_algebra import DenseSemanticErasureError, erase_dense_semantics
from tile_lifetime.gemm_program import GENERIC_H100_GEMM_BACKEND
from tile_lifetime.ir import (
    DType,
    LinearOp,
    ResidualAddOp,
    RMSNormOp,
    ScaledDotProductAttentionOp,
    SemanticOp,
    TensorGraph,
)
from tile_lifetime.plan import (
    Attachment,
    AttachmentSite,
    GemmSkeleton,
    MaterializationDisposition,
    MaterializationRecord,
    NumericalEquivalence,
    NumericalPolicy,
    ReductionSkeleton,
    RegionPlan,
    RewriteExplanation,
    TransformSkeleton,
)
from tile_lifetime.semantic_erasure import (
    ErasedTensorProgram,
    validate_erased_tensor_program,
    validate_plan_semantic_erasure,
)
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    ProgramValue,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
)


@dataclass(frozen=True)
class _FoldScaleContractRegion:
    producer_contract: ContractPrimitive
    tile_add: MapPrimitive
    forwarded_input: ProgramValue
    squared: MapPrimitive
    partial_fold: FoldPrimitive
    row_finalize: MapPrimitive
    feature_scale: MapPrimitive
    feature_vector: ProgramValue
    normalized_scale: MapPrimitive
    consumer_contract: ContractPrimitive


class GenericDensePlanningError(ValueError):
    """No legal generic Fold/Map/Contract placement was found."""

    def __init__(self, reasons: tuple[str, ...]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


class RowScalePlacement(StrEnum):
    """Physical placement of a row scalar around a right contraction."""

    CONSUMER_EPILOGUE = "consumer_epilogue"
    CONSUMER_PROLOGUE = "consumer_prologue"


RMSScalePlacement = RowScalePlacement


def compile_region(
    graph: TensorGraph,
    *,
    numerical_policy: NumericalPolicy,
    rms_scale_placement: RowScalePlacement = RowScalePlacement.CONSUMER_EPILOGUE,
) -> RegionPlan:
    """Erase frontend names, then compile generic algebra or retain a fallback."""
    erased: ErasedTensorProgram | None = None
    try:
        erased = erase_dense_semantics(graph)
        return compile_erased_dense_program(
            erased,
            numerical_policy=numerical_policy,
            scale_placement=rms_scale_placement,
        )
    except DenseSemanticErasureError as error:
        return _materialized_fallback(graph, rejection_reasons=(str(error),))
    except GenericDensePlanningError as error:
        assert erased is not None
        return replace(
            _materialized_fallback(graph, rejection_reasons=error.reasons),
            semantic_erasure_report=erased.report,
        )


def compile_erased_dense_program(
    erased: ErasedTensorProgram,
    *,
    numerical_policy: NumericalPolicy,
    scale_placement: RowScalePlacement = RowScalePlacement.CONSUMER_EPILOGUE,
) -> RegionPlan:
    """Select placements using only generic dataflow and algebraic properties."""
    validate_erased_tensor_program(erased)
    region, reasons = _find_fold_scale_contract_region(
        erased.program.operations,
        numerical_policy=numerical_policy,
    )
    if region is None:
        raise GenericDensePlanningError(reasons)
    plan = replace(
        _fold_scale_contract_plan(region, scale_placement=scale_placement),
        semantic_erasure_report=erased.report,
    )
    validate_plan_semantic_erasure(plan)
    return plan


def _find_fold_scale_contract_region(
    operations: tuple[ContractPrimitive | MapPrimitive | FoldPrimitive, ...],
    *,
    numerical_policy: NumericalPolicy,
) -> tuple[_FoldScaleContractRegion | None, tuple[str, ...]]:
    producer_by_value = {operation.output.name: operation for operation in operations}
    consumers_by_value: dict[str, list[ContractPrimitive | MapPrimitive | FoldPrimitive]] = {}
    for operation in operations:
        operation_inputs = operation.inputs if not isinstance(operation, FoldPrimitive) else (operation.input,)
        for value in operation_inputs:
            consumers_by_value.setdefault(value.name, []).append(operation)

    reasons: list[str] = []
    for consumer in (operation for operation in operations if isinstance(operation, ContractPrimitive)):
        if len(consumer.inputs) != 2 or len(consumer.reduction_axes) != 1:
            continue
        activation = consumer.inputs[0]
        normalized_scale = producer_by_value.get(activation.name)
        if not _is_binary_map(normalized_scale, ScalarExpressionKind.MULTIPLY):
            continue
        assert isinstance(normalized_scale, MapPrimitive)
        reduction_axis = consumer.reduction_axes[0]
        row_axes = tuple(axis for axis in activation.axes if axis != reduction_axis)
        scale_operands = _operands_by_axes(normalized_scale.inputs, activation.axes, row_axes)
        if scale_operands is None:
            reasons.append(
                "row-scalar axes do not equal the unreduced consumer axes; the Fold reduction axis is incompatible"
            )
            continue
        feature_scaled_value, row_scalar_value = scale_operands

        row_finalize = producer_by_value.get(row_scalar_value.name)
        if not isinstance(row_finalize, MapPrimitive) or row_finalize.output.axes != row_axes:
            reasons.append("row scalar is not produced by a generic Map")
            continue
        if len(row_finalize.inputs) != 1:
            reasons.append("row-scalar Map requires unavailable non-fold inputs")
            continue
        partial_fold = producer_by_value.get(row_finalize.inputs[0].name)
        if not isinstance(partial_fold, FoldPrimitive) or partial_fold.reducer is not FoldReducer.SUM:
            reasons.append("row-scalar Map is not fed by an associative sum Fold")
            continue
        if partial_fold.reduction_axes != (reduction_axis,):
            reasons.append("Fold axis is not the consumer Contract reduction axis")
            continue
        if partial_fold.accumulation_dtype not in {DType.FP32, DType.FP64}:
            reasons.append("partial Fold must accumulate in FP32 or FP64")
            continue

        squared = producer_by_value.get(partial_fold.input.name)
        if not _is_square_map(squared):
            reasons.append("sum Fold input is not a generic pointwise square")
            continue
        assert isinstance(squared, MapPrimitive)
        fold_source = squared.inputs[0]

        feature_scale = producer_by_value.get(feature_scaled_value.name)
        if not _is_binary_map(feature_scale, ScalarExpressionKind.MULTIPLY):
            reasons.append("full activation is not produced by a generic feature-scale Map")
            continue
        assert isinstance(feature_scale, MapPrimitive)
        feature_operands = _operands_by_axes(feature_scale.inputs, fold_source.axes, (reduction_axis,))
        if feature_operands is None or feature_operands[0] != fold_source:
            reasons.append("feature-scale Map does not multiply the Fold source by a reduction-axis vector")
            continue
        feature_vector = feature_operands[1]

        tile_add = producer_by_value.get(fold_source.name)
        if not _is_binary_map(tile_add, ScalarExpressionKind.ADD):
            reasons.append("Fold source is not produced by a tile-local binary add")
            continue
        assert isinstance(tile_add, MapPrimitive)
        producer_candidates = tuple((value, producer_by_value.get(value.name)) for value in tile_add.inputs)
        contract_inputs = tuple(
            (value, producer)
            for value, producer in producer_candidates
            if isinstance(producer, ContractPrimitive) and producer.output == value
        )
        if len(contract_inputs) != 1:
            reasons.append("tile-local add does not combine exactly one Contract result with one forwarded input")
            continue
        producer_contract = contract_inputs[0][1]
        assert isinstance(producer_contract, ContractPrimitive)
        forwarded_input = next(value for value in tile_add.inputs if value != producer_contract.output)

        activation_consumers = consumers_by_value.get(normalized_scale.output.name, [])
        if activation_consumers != [consumer]:
            reasons.append(
                f"scaled activation has {len(activation_consumers)} consumers; placement requires exactly one"
            )
            continue
        if numerical_policy is NumericalPolicy.BITWISE_EXACT:
            reasons.append("tile-lifetime placement changes finite-precision rounding under bitwise-exact policy")
            continue
        return (
            _FoldScaleContractRegion(
                producer_contract=producer_contract,
                tile_add=tile_add,
                forwarded_input=forwarded_input,
                squared=squared,
                partial_fold=partial_fold,
                row_finalize=row_finalize,
                feature_scale=feature_scale,
                feature_vector=feature_vector,
                normalized_scale=normalized_scale,
                consumer_contract=consumer,
            ),
            (),
        )
    if not reasons:
        reasons.append("no Contract/Map/Fold subgraph exposes a legal row-scalar placement")
    return None, tuple(dict.fromkeys(reasons))


def _fold_scale_contract_plan(
    region: _FoldScaleContractRegion,
    *,
    scale_placement: RowScalePlacement,
) -> RegionPlan:
    first = region.producer_contract
    second = region.consumer_contract
    fold_source = region.squared.inputs[0]
    scaled_value = region.feature_scale.output
    partials = f"{fold_source.name}_fold_partials"
    row_scalar = region.row_finalize.output
    normalized_for_gemm = f"{scaled_value.name}_times_{row_scalar.name}"

    gemm_0 = GemmSkeleton(
        name="contract_with_maps_and_fold_partials",
        input=first.inputs[0].name,
        weight=first.inputs[1].name,
        output=scaled_value.name,
        shape=(first.output.shape[0], first.output.shape[1], first.inputs[0].shape[1]),
        accumulation_dtype=first.accumulation_dtype,
        backend=GENERIC_H100_GEMM_BACKEND,
        input_layout="row_major_mk",
        output_layout="row_major_mn",
        epilogue=(
            Attachment(
                operation="add",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(first.output.name, region.forwarded_input.name),
                outputs=(region.tile_add.output.name,),
            ),
            Attachment(
                operation="multiply",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(region.tile_add.output.name, region.feature_vector.name),
                outputs=(scaled_value.name,),
            ),
            Attachment(
                operation="partial_sum_square",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(fold_source.name,),
                outputs=(partials,),
            ),
        ),
    )
    reduction = ReductionSkeleton(
        name="combine_fold_partials",
        input=partials,
        output=row_scalar.name,
        operator=_render_scalar_expression(region.row_finalize.expression, {region.partial_fold.output.name: "sum"}),
        reduction_dtype=region.partial_fold.accumulation_dtype,
    )
    consumer_prologue = ()
    consumer_epilogue = ()
    consumer_name = "contract_delayed_row_scale"
    prologue_cluster_shape = (1, 2 if second.output.shape[1] >= 16_384 else 1, 1)
    if scale_placement is RowScalePlacement.CONSUMER_PROLOGUE:
        consumer_name = "contract_prepared_row_scale"
        consumer_prologue = (
            Attachment(
                operation="scale_row",
                site=AttachmentSite.GEMM_PROLOGUE,
                inputs=(scaled_value.name, row_scalar.name),
                outputs=(normalized_for_gemm,),
            ),
        )
    else:
        consumer_epilogue = (
            Attachment(
                operation="scale_row",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(second.output.name, row_scalar.name),
                outputs=(second.output.name,),
            ),
        )

    gemm_1 = GemmSkeleton(
        name=consumer_name,
        input=scaled_value.name,
        weight=second.inputs[1].name,
        output=second.output.name,
        shape=(second.output.shape[0], second.output.shape[1], second.inputs[0].shape[1]),
        accumulation_dtype=second.accumulation_dtype,
        backend=GENERIC_H100_GEMM_BACKEND,
        input_layout="row_major_mk",
        output_layout="row_major_mn",
        physical_tile_shape=(128, 256, 64) if scale_placement is RowScalePlacement.CONSUMER_PROLOGUE else None,
        cluster_shape=prologue_cluster_shape if scale_placement is RowScalePlacement.CONSUMER_PROLOGUE else None,
        pingpong=False if scale_placement is RowScalePlacement.CONSUMER_PROLOGUE else None,
        prologue=consumer_prologue,
        epilogue=consumer_epilogue,
    )

    materializations = (
        MaterializationRecord(
            value=first.output.name,
            shape=first.output.shape,
            dtype=first.output.dtype,
            disposition=MaterializationDisposition.EPILOGUE_ONLY,
            reason="consumed by residual and RMS-partial attachments before the first GEMM tile is stored",
        ),
        MaterializationRecord(
            value=region.tile_add.output.name,
            shape=region.tile_add.output.shape,
            dtype=region.tile_add.output.dtype,
            disposition=MaterializationDisposition.EPILOGUE_ONLY,
            reason="replaced by the gamma-scaled representation consumed by the next GEMM",
        ),
        MaterializationRecord(
            value=region.normalized_scale.output.name,
            shape=region.normalized_scale.output.shape,
            dtype=region.normalized_scale.output.dtype,
            disposition=(
                MaterializationDisposition.PROLOGUE_ONLY
                if scale_placement is RowScalePlacement.CONSUMER_PROLOGUE
                else MaterializationDisposition.EPILOGUE_ONLY
            ),
            reason=(
                "represented by an on-chip BF16 row-scale transform before the consumer WGMMA"
                if scale_placement is RowScalePlacement.CONSUMER_PROLOGUE
                else "inverse-RMS scaling is delayed through the following right-multiplication"
            ),
        ),
        MaterializationRecord(
            value=scaled_value.name,
            shape=scaled_value.shape,
            dtype=scaled_value.dtype,
            disposition=MaterializationDisposition.MATERIALIZE,
            reason="cross-skeleton activation consumed by the second GEMM mainloop",
        ),
        MaterializationRecord(
            value=partials,
            shape=region.partial_fold.output.shape,
            dtype=region.partial_fold.accumulation_dtype,
            disposition=MaterializationDisposition.PARTIAL_REDUCTION_ONLY,
            reason="small per-row partial Fold buffer",
        ),
        MaterializationRecord(
            value=second.output.name,
            shape=second.output.shape,
            dtype=second.output.dtype,
            disposition=MaterializationDisposition.MATERIALIZE,
            reason="region output",
        ),
    )
    if scale_placement is RowScalePlacement.CONSUMER_PROLOGUE:
        rewrite_name = "place_row_scalar_in_consumer_contract_preparation"
        transformed_consumer = (
            f"{normalized_for_gemm} = bf16({scaled_value.name} * {row_scalar.name}) inside Contract preparation",
            f"{second.output.name} = contract({normalized_for_gemm}, {second.inputs[1].name})",
        )
        semantic_properties = (
            "binary add Map is tile-local",
            "feature-vector multiply Map is tile-local",
            "sum Fold over pointwise squares decomposes into tile partials",
            "Fold finalization produces a row scalar available before the consumer mainloop",
        )
        estimated_benefit = (
            "eliminates the materialized normalized activation while retaining source-like pre-GEMM scaling; "
            "the input transform adds shared-memory traffic, synchronization, and repeated scale work per N tile"
        )
        numerical_effect = (
            "places inverse-RMS scaling and BF16 conversion before WGMMA, matching a source GEMM that consumes "
            "a BF16 normalized activation more closely than delayed epilogue scaling; storing u*gamma before the "
            "prologue can still introduce an additional BF16 rounding"
        )
    else:
        rewrite_name = "move_row_scalar_through_right_contract"
        transformed_consumer = (
            f"{second.output.name} = scale_row("
            f"contract({scaled_value.name}, {second.inputs[1].name}), {row_scalar.name})",
        )
        semantic_properties = (
            "binary add Map is tile-local",
            "feature-vector multiply Map is tile-local",
            "sum Fold over pointwise squares decomposes into tile partials",
            "Fold finalization produces a row scalar",
            "row scaling commutes with right multiplication over real arithmetic",
        )
        estimated_benefit = "eliminates the materialized normalized activation and its full-tensor normalization kernel"
        numerical_effect = (
            "moves inverse-RMS scaling after the second GEMM, changing where BF16 rounding occurs; "
            "the generated plan requires differential validation against the declared reference"
        )

    explanation = RewriteExplanation(
        name=rewrite_name,
        applied=True,
        original_fragment=(
            f"{first.output.name} = contract({first.inputs[0].name}, {first.inputs[1].name})",
            f"{region.tile_add.output.name} = map_add({first.output.name}, {region.forwarded_input.name})",
            f"{region.partial_fold.output.name} = fold_sum(map_square({fold_source.name}))",
            f"{row_scalar.name} = map({_render_scalar_expression(region.row_finalize.expression)})",
            f"{second.output.name} = contract({region.normalized_scale.output.name}, {second.inputs[1].name})",
        ),
        transformed_fragment=(
            f"{scaled_value.name}, {partials} = contract_finalization_maps_and_partial_fold(...) ",
            f"{row_scalar.name} = fold_finalize({partials})",
            *transformed_consumer,
        ),
        semantic_properties=semantic_properties,
        legality_checks=(
            "sum Fold covers the consumer Contract reduction dimension",
            "scaled activation has exactly one consumer",
            "consumer is a two-input right contraction",
            "partial Fold accumulates in FP32 or FP64",
            "numerical policy permits reordered finite-precision rounding",
        ),
        estimated_benefit=estimated_benefit,
        numerical_equivalence=NumericalEquivalence.ALGEBRAICALLY_EXACT,
        numerical_effect=numerical_effect,
    )
    return RegionPlan(skeletons=(gemm_0, reduction, gemm_1), materializations=materializations, rewrites=(explanation,))


def _is_binary_map(
    operation: ContractPrimitive | MapPrimitive | FoldPrimitive | None,
    kind: ScalarExpressionKind,
) -> bool:
    return (
        isinstance(operation, MapPrimitive)
        and operation.expression.kind is kind
        and len(operation.expression.operands) == 2
        and all(operand.kind is ScalarExpressionKind.INPUT for operand in operation.expression.operands)
    )


def _is_square_map(operation: ContractPrimitive | MapPrimitive | FoldPrimitive | None) -> bool:
    if not _is_binary_map(operation, ScalarExpressionKind.MULTIPLY):
        return False
    assert isinstance(operation, MapPrimitive)
    left, right = operation.expression.operands
    return left.input_name == right.input_name and len(operation.inputs) == 1


def _operands_by_axes(
    operands: tuple[ProgramValue, ...],
    full_axes: tuple[TensorAxis, ...],
    broadcast_axes: tuple[TensorAxis, ...],
) -> tuple[ProgramValue, ProgramValue] | None:
    full = tuple(value for value in operands if value.axes == full_axes)
    broadcast = tuple(value for value in operands if value.axes == broadcast_axes)
    if len(full) != 1 or len(broadcast) != 1:
        return None
    return full[0], broadcast[0]


def _render_scalar_expression(
    expression: ScalarExpression,
    input_aliases: dict[str, str] | None = None,
    parent_precedence: int = 0,
) -> str:
    aliases = input_aliases or {}
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return aliases.get(expression.input_name, expression.input_name)
    if expression.kind is ScalarExpressionKind.CONSTANT:
        assert expression.constant is not None
        return str(expression.constant)
    if expression.kind is ScalarExpressionKind.RSQRT:
        return f"rsqrt({_render_scalar_expression(expression.operands[0], aliases)})"
    if expression.kind in {ScalarExpressionKind.EXP, ScalarExpressionKind.TANH}:
        return f"{expression.kind.value}({_render_scalar_expression(expression.operands[0], aliases)})"
    operator = {
        ScalarExpressionKind.ADD: ("+", 10),
        ScalarExpressionKind.SUBTRACT: ("-", 10),
        ScalarExpressionKind.MULTIPLY: ("*", 20),
        ScalarExpressionKind.DIVIDE: ("/", 20),
        ScalarExpressionKind.LESS_EQUAL: ("<=", 5),
    }.get(expression.kind)
    if operator is not None:
        symbol, precedence = operator
        left = _render_scalar_expression(expression.operands[0], aliases, precedence)
        right = _render_scalar_expression(expression.operands[1], aliases, precedence + 1)
        rendered = f"{left} {symbol} {right}"
        return f"({rendered})" if precedence < parent_precedence else rendered
    operands = tuple(_render_scalar_expression(operand, aliases) for operand in expression.operands)
    return f"select({', '.join(operands)})"


def _materialized_fallback(graph: TensorGraph, *, rejection_reasons: tuple[str, ...]) -> RegionPlan:
    skeletons = tuple(_fallback_skeleton(operation) for operation in graph.operations)
    materializations = tuple(
        MaterializationRecord(
            value=operation.output.name,
            shape=operation.output.shape,
            dtype=operation.output.dtype,
            disposition=MaterializationDisposition.MATERIALIZE,
            reason="standalone fallback operation boundary",
        )
        for operation in graph.operations
    )
    explanation = RewriteExplanation(
        name="delay_rms_row_scale_through_gemm",
        applied=False,
        original_fragment=tuple(type(operation).__name__ for operation in graph.operations),
        transformed_fragment=(),
        semantic_properties=(),
        legality_checks=(),
        estimated_benefit="none; the materialized reference path is retained",
        numerical_equivalence=NumericalEquivalence.BITWISE_EXACT,
        numerical_effect="none",
        rejection_reasons=rejection_reasons,
    )
    return RegionPlan(skeletons=skeletons, materializations=materializations, rewrites=(explanation,))


def _fallback_skeleton(operation: SemanticOp) -> GemmSkeleton | TransformSkeleton:
    if isinstance(operation, LinearOp):
        return GemmSkeleton(
            name=f"materialized_{operation.output.name}",
            input=operation.input.name,
            weight=operation.weight.name,
            output=operation.output.name,
            shape=(operation.output.shape[0], operation.output.shape[1], operation.input.shape[1]),
            accumulation_dtype=operation.accumulation_dtype,
        )
    if isinstance(operation, ResidualAddOp):
        return TransformSkeleton(
            name=f"materialized_{operation.output.name}",
            operation="residual_add",
            inputs=(operation.left.name, operation.right.name),
            output=operation.output.name,
        )
    if isinstance(operation, RMSNormOp):
        return TransformSkeleton(
            name=f"materialized_{operation.output.name}",
            operation="rms_norm",
            inputs=(operation.input.name, operation.gamma.name),
            output=operation.output.name,
        )
    assert isinstance(operation, ScaledDotProductAttentionOp)
    return TransformSkeleton(
        name=f"materialized_{operation.output.name}",
        operation="scaled_dot_product_attention",
        inputs=(operation.query.name, operation.key.name, operation.value.name),
        output=operation.output.name,
    )
