# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate training plans for row-statistic Maps, Folds, and Contracts.

The planner in this module is deliberately structural.  It consumes ordinary
``Map -> Fold -> Map -> Contract`` tensor algebra, derives its reverse program,
and lowers the resulting primitives into generic physical skeletons.  Neither
the physical plan nor its execution steps contain a named normalization kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from shuttle.ir import DType
from tile_lifetime.autodiff import DifferentiatedTensorProgram, differentiate_tensor_program
from tile_lifetime.cuda_axis_fold_codegen import (
    AxisFoldDirection,
    AxisFoldInput,
    AxisFoldInputLayout,
    AxisFoldOutputKind,
    AxisFoldProgram,
    AxisFoldReassociation,
    AxisFoldReduction,
)
from tile_lifetime.gemm_program import GENERIC_H100_GEMM_BACKEND
from tile_lifetime.plan import Attachment, AttachmentSite, GemmSkeleton, NumericalEquivalence
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    ProgramValue,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    TensorPrimitive,
    TensorProgram,
    scalar_binary,
    scalar_constant,
    scalar_input,
    scalar_unary,
    serialize_scalar_expression,
)


class RowStatisticKind(StrEnum):
    """Second-moment statistic represented by the source Fold graph."""

    UNCENTERED_SECOND_MOMENT = "uncentered_second_moment"
    CENTERED_SECOND_MOMENT = "centered_second_moment"


class RowNormalizationSavePolicy(StrEnum):
    """Activation/statistic values retained for the generated reverse pass."""

    SAVE_NORMALIZED = "save_normalized"
    SAVE_INPUT_AND_INVERSE = "save_input_and_inverse"
    RECOMPUTE_STATISTIC = "recompute_statistic"


class RowStatisticScalePlacement(StrEnum):
    """Physical placement of a row scalar around a following Contract."""

    SOURCE_ORDERED_PREPARATION = "source_ordered_preparation"
    REAL_ALGEBRA_EQUIVALENT_FINALIZATION = "real_algebra_equivalent_finalization"


@dataclass(frozen=True)
class GeneratedMapSkeleton:
    """A scalar AST applied over one logical tile domain."""

    name: str
    inputs: tuple[str, ...]
    output: str
    output_shape: tuple[int, ...]
    expression: ScalarExpression
    tile_shape: tuple[int, ...]


@dataclass(frozen=True)
class GeneratedFoldSkeleton:
    """A generic tiled reduction with a visible reducer and domain."""

    name: str
    input: str
    output: str
    input_shape: tuple[int, ...]
    reduction_axes: tuple[str, ...]
    reducer: FoldReducer
    accumulation_dtype: DType
    tile_extent: int


@dataclass(frozen=True)
class GeneratedContractSkeleton:
    """A semantic Contract and its legalized generic GEMM view."""

    name: str
    operands: tuple[str, str]
    operand_views: tuple[str, str]
    output: str
    reduction_axis: str
    skeleton: GemmSkeleton


GeneratedTrainingSkeleton = GeneratedMapSkeleton | GeneratedFoldSkeleton | GeneratedContractSkeleton


@dataclass(frozen=True)
class RowNormalizationTrainingPlan:
    """Inspectable generated forward placement and reverse physical program."""

    source: TensorProgram
    automatic_adjoint: DifferentiatedTensorProgram
    backward: TensorProgram
    statistic_kind: RowStatisticKind
    save_policy: RowNormalizationSavePolicy
    scale_placement: RowStatisticScalePlacement
    numerical_equivalence: NumericalEquivalence
    saved_values: tuple[str, ...]
    recomputed_values: tuple[str, ...]
    forward_contract: GemmSkeleton
    backward_steps: tuple[GeneratedTrainingSkeleton, ...]

    @property
    def maps(self) -> tuple[GeneratedMapSkeleton, ...]:
        """Pointwise skeletons in reverse execution order."""
        return tuple(step for step in self.backward_steps if isinstance(step, GeneratedMapSkeleton))

    @property
    def folds(self) -> tuple[GeneratedFoldSkeleton, ...]:
        """Reduction skeletons in reverse execution order."""
        return tuple(step for step in self.backward_steps if isinstance(step, GeneratedFoldSkeleton))

    @property
    def contracts(self) -> tuple[GeneratedContractSkeleton, ...]:
        """Contraction skeletons in reverse execution order."""
        return tuple(step for step in self.backward_steps if isinstance(step, GeneratedContractSkeleton))


@dataclass(frozen=True)
class RowNormalizationAxisFoldPrograms:
    """Executable generic axis-Fold programs extracted from one reverse graph."""

    input_cotangent: AxisFoldProgram
    feature_scale_cotangent: AxisFoldProgram


def build_row_normalized_contract_program(
    *,
    rows: int,
    hidden: int,
    features: int,
    statistic_kind: RowStatisticKind,
    epsilon: float = 1e-5,
    dtype: DType = DType.FP32,
) -> TensorProgram:
    """Build ordinary row-statistic algebra followed by a dense Contract."""
    if min(rows, hidden, features) <= 0:
        raise ValueError("row-normalized Contract dimensions must be positive")
    if epsilon < 0:
        raise ValueError("row-statistic epsilon must be non-negative")
    row = TensorAxis(0, rows, "row")
    reduction = TensorAxis(1, hidden, "reduction")
    feature = TensorAxis(2, features, "feature")
    x = ProgramValue("input", (row, reduction), dtype)
    gamma = ProgramValue("feature_scale", (reduction,), dtype)
    weight = ProgramValue("weight", (reduction, feature), dtype)
    operations: list[TensorPrimitive] = []

    centered = x
    if statistic_kind is RowStatisticKind.CENTERED_SECOND_MOMENT:
        row_sum = ProgramValue("row_sum", (row,), DType.FP32)
        row_mean = ProgramValue("row_mean", (row,), DType.FP32)
        centered = ProgramValue("centered", (row, reduction), dtype)
        operations.extend(
            (
                FoldPrimitive("Fold input", x, row_sum, (reduction,), FoldReducer.SUM, DType.FP32),
                MapPrimitive(
                    "scale row statistic",
                    (row_sum,),
                    row_mean,
                    _divide(scalar_input(row_sum.name), scalar_constant(float(hidden))),
                ),
                MapPrimitive(
                    "subtract row statistic",
                    (x, row_mean),
                    centered,
                    _subtract(scalar_input(x.name), scalar_input(row_mean.name)),
                ),
            )
        )

    squared = ProgramValue("squared", (row, reduction), DType.FP32)
    sum_square = ProgramValue("sum_square", (row,), DType.FP32)
    mean_square = ProgramValue("mean_square", (row,), DType.FP32)
    inverse = ProgramValue("inverse_scale", (row,), DType.FP32)
    standardized = ProgramValue("standardized", (row, reduction), dtype)
    scaled = ProgramValue("scaled", (row, reduction), dtype)
    output = ProgramValue("output", (row, feature), dtype)
    operations.extend(
        (
            MapPrimitive(
                "square local value",
                (centered,),
                squared,
                _multiply(scalar_input(centered.name), scalar_input(centered.name)),
            ),
            FoldPrimitive("Fold squared values", squared, sum_square, (reduction,), FoldReducer.SUM, DType.FP32),
            MapPrimitive(
                "scale second moment",
                (sum_square,),
                mean_square,
                _divide(scalar_input(sum_square.name), scalar_constant(float(hidden))),
            ),
            MapPrimitive(
                "invert second moment",
                (mean_square,),
                inverse,
                scalar_unary(
                    ScalarExpressionKind.RSQRT,
                    _add(scalar_input(mean_square.name), scalar_constant(epsilon)),
                ),
            ),
            MapPrimitive(
                "standardize local value",
                (centered, inverse),
                standardized,
                _multiply(scalar_input(centered.name), scalar_input(inverse.name)),
            ),
            MapPrimitive(
                "apply feature scale",
                (standardized, gamma),
                scaled,
                _multiply(scalar_input(standardized.name), scalar_input(gamma.name)),
            ),
            ContractPrimitive(
                "project scaled value",
                (scaled, weight),
                output,
                (reduction,),
                DType.FP32,
            ),
        )
    )
    return TensorProgram(inputs=(x, gamma, weight), operations=tuple(operations), outputs=(output,))


def compile_row_normalization_training(
    source: TensorProgram,
    *,
    save_policy: RowNormalizationSavePolicy,
    scale_placement: RowStatisticScalePlacement,
    physical_tile_shape: tuple[int, int, int] = (128, 128, 64),
    map_tile_elements: int = 256,
    fold_tile_extent: int = 256,
) -> RowNormalizationTrainingPlan:
    """Derive a generic reverse program and legalize it to physical skeletons."""
    recovered = _recover_row_statistic_contract(source)
    automatic_adjoint = differentiate_tensor_program(
        source,
        with_respect_to=(recovered.input.name, recovered.gamma.name, recovered.weight.name),
    )
    backward, saved_values, recomputed_values = _build_efficient_backward(recovered, save_policy)
    backward_steps = tuple(
        _lower_training_operation(
            operation,
            physical_tile_shape=physical_tile_shape,
            map_tile_elements=map_tile_elements,
            fold_tile_extent=fold_tile_extent,
        )
        for operation in backward.operations
    )
    forward_contract = _forward_contract_skeleton(recovered, scale_placement, physical_tile_shape)
    numerical_equivalence = (
        NumericalEquivalence.BITWISE_EXACT
        if scale_placement is RowStatisticScalePlacement.SOURCE_ORDERED_PREPARATION
        else NumericalEquivalence.ALGEBRAICALLY_EXACT
    )
    return RowNormalizationTrainingPlan(
        source=source,
        automatic_adjoint=automatic_adjoint,
        backward=backward,
        statistic_kind=recovered.statistic_kind,
        save_policy=save_policy,
        scale_placement=scale_placement,
        numerical_equivalence=numerical_equivalence,
        saved_values=saved_values,
        recomputed_values=recomputed_values,
        forward_contract=forward_contract,
        backward_steps=backward_steps,
    )


def lower_row_normalization_axis_folds(
    plan: RowNormalizationTrainingPlan,
    *,
    threads: int = 256,
) -> RowNormalizationAxisFoldPrograms:
    """Fuse the reverse graph's scalar Maps into two generic axis Folds.

    The first program reduces row-local correlation state and applies the final
    input-cotangent Map.  The second reduces the feature-scale cotangent over
    rows.  A centered statistic adds one reduction state to the first program;
    the physical generator and schedule remain unchanged.
    """
    rows, hidden = plan.source.inputs[0].shape
    return build_row_normalization_axis_fold_programs(
        rows=rows,
        hidden=hidden,
        source_dtype=plan.source.inputs[0].dtype,
        statistic_kind=plan.statistic_kind,
        threads=threads,
    )


def build_row_normalization_axis_fold_programs(
    *,
    rows: int,
    hidden: int,
    source_dtype: DType,
    statistic_kind: RowStatisticKind,
    threads: int = 256,
) -> RowNormalizationAxisFoldPrograms:
    """Build fused axis Folds from recovered generic reverse algebra."""
    if rows <= 0 or hidden <= 0:
        raise ValueError("row-normalization reverse extents must be positive")
    projected = scalar_input("projected")
    feature_scale = scalar_input("feature_scale")
    standardized = scalar_input("standardized")
    inverse_scale = scalar_input("inverse_scale")
    local = _multiply(projected, feature_scale)
    correlation_name = "correlation_sum"
    reductions = [
        AxisFoldReduction(
            correlation_name,
            _multiply(local, standardized),
        )
    ]
    centered_term = _subtract(
        local,
        _multiply(
            standardized,
            _divide(scalar_input(correlation_name), scalar_constant(float(hidden))),
        ),
    )
    if statistic_kind is RowStatisticKind.CENTERED_SECOND_MOMENT:
        reductions.append(AxisFoldReduction("local_sum", local))
        centered_term = _subtract(
            centered_term,
            _divide(scalar_input("local_sum"), scalar_constant(float(hidden))),
        )
    shared_inputs = (
        AxisFoldInput("projected", DType.FP32, AxisFoldInputLayout.ELEMENT),
        AxisFoldInput("feature_scale", source_dtype, AxisFoldInputLayout.COLUMN),
        AxisFoldInput("standardized", source_dtype, AxisFoldInputLayout.ELEMENT),
        AxisFoldInput("inverse_scale", DType.FP32, AxisFoldInputLayout.ROW),
    )
    input_cotangent = AxisFoldProgram(
        rows=rows,
        columns=hidden,
        inputs=shared_inputs,
        reductions=tuple(reductions),
        reduction_axis=AxisFoldDirection.COLUMNS,
        output_kind=AxisFoldOutputKind.ELEMENT,
        output_expression=_multiply(inverse_scale, centered_term),
        output_dtype=DType.FP32,
        threads=threads,
        reassociation=AxisFoldReassociation.DETERMINISTIC_TREE,
    )
    feature_scale_cotangent = AxisFoldProgram(
        rows=rows,
        columns=hidden,
        inputs=(
            AxisFoldInput("projected", DType.FP32, AxisFoldInputLayout.ELEMENT),
            AxisFoldInput("standardized", source_dtype, AxisFoldInputLayout.ELEMENT),
        ),
        reductions=(
            AxisFoldReduction(
                "feature_scale_sum",
                _multiply(projected, standardized),
            ),
        ),
        reduction_axis=AxisFoldDirection.ROWS,
        output_kind=AxisFoldOutputKind.REDUCED,
        output_expression=scalar_input("feature_scale_sum"),
        output_dtype=DType.FP32,
        threads=threads,
        reassociation=AxisFoldReassociation.DETERMINISTIC_TREE,
    )
    return RowNormalizationAxisFoldPrograms(
        input_cotangent=input_cotangent,
        feature_scale_cotangent=feature_scale_cotangent,
    )


@dataclass(frozen=True)
class _RecoveredRowStatisticContract:
    input: ProgramValue
    gamma: ProgramValue
    weight: ProgramValue
    output: ProgramValue
    row_axis: TensorAxis
    reduction_axis: TensorAxis
    feature_axis: TensorAxis
    statistic_kind: RowStatisticKind
    epsilon: float


def _recover_row_statistic_contract(source: TensorProgram) -> _RecoveredRowStatisticContract:
    if len(source.inputs) != 3 or len(source.outputs) != 1:
        raise ValueError("row-statistic Contract recovery expects three inputs and one output")
    contract = source.operations[-1]
    if not isinstance(contract, ContractPrimitive) or len(contract.inputs) != 2 or len(contract.reduction_axes) != 1:
        raise ValueError("row-statistic program must terminate in a two-input Contract")
    scaled_map = source.operations[-2]
    if not isinstance(scaled_map, MapPrimitive) or scaled_map.output != contract.inputs[0]:
        raise ValueError("terminal Contract input must be produced by a scalar Map")
    standardized_map = source.operations[-3]
    if not isinstance(standardized_map, MapPrimitive) or standardized_map.output != scaled_map.inputs[0]:
        raise ValueError("feature-scale Map must consume a standardized scalar Map")
    inverse_map = source.operations[-4]
    if not isinstance(inverse_map, MapPrimitive) or inverse_map.output != standardized_map.inputs[1]:
        raise ValueError("standardization must consume an inverse row statistic")
    epsilon = _inverse_statistic_epsilon(inverse_map.expression)
    second_moment_fold = source.operations[-6]
    if not isinstance(second_moment_fold, FoldPrimitive) or second_moment_fold.reducer is not FoldReducer.SUM:
        raise ValueError("inverse row statistic must originate from a sum Fold")
    x, gamma, weight = source.inputs
    reduction_axis = contract.reduction_axes[0]
    if x.axes != (contract.output.axes[0], reduction_axis):
        raise ValueError("source input axes do not form row-by-reduction matrix")
    if gamma.axes != (reduction_axis,) or weight.axes != (reduction_axis, contract.output.axes[1]):
        raise ValueError("feature scale or Contract weight axes are incompatible")
    standardized_input = standardized_map.inputs[0]
    statistic_kind = (
        RowStatisticKind.UNCENTERED_SECOND_MOMENT if standardized_input == x else RowStatisticKind.CENTERED_SECOND_MOMENT
    )
    if statistic_kind is RowStatisticKind.CENTERED_SECOND_MOMENT:
        centered_producer = source.operations[-8]
        mean_fold = source.operations[-10]
        if (
            not isinstance(centered_producer, MapPrimitive)
            or centered_producer.output != standardized_input
            or not isinstance(mean_fold, FoldPrimitive)
            or mean_fold.reducer is not FoldReducer.SUM
        ):
            raise ValueError("centered statistic requires a preceding sum Fold and subtraction Map")
    return _RecoveredRowStatisticContract(
        input=x,
        gamma=gamma,
        weight=weight,
        output=contract.output,
        row_axis=x.axes[0],
        reduction_axis=reduction_axis,
        feature_axis=contract.output.axes[1],
        statistic_kind=statistic_kind,
        epsilon=epsilon,
    )


def _build_efficient_backward(
    recovered: _RecoveredRowStatisticContract,
    save_policy: RowNormalizationSavePolicy,
) -> tuple[TensorProgram, tuple[str, ...], tuple[str, ...]]:
    x = recovered.input
    gamma = recovered.gamma
    weight = recovered.weight
    row = recovered.row_axis
    reduction = recovered.reduction_axis
    feature = recovered.feature_axis
    inverse = ProgramValue("inverse_scale", (row,), DType.FP32)
    standardized = ProgramValue("standardized", (row, reduction), x.dtype)
    output_cotangent = ProgramValue(f"cotangent.{recovered.output.name}", (row, feature), recovered.output.dtype)
    operations: list[TensorPrimitive] = []
    saved_values: tuple[str, ...]
    recomputed_values: list[str] = []

    if save_policy is RowNormalizationSavePolicy.SAVE_NORMALIZED:
        inputs = (standardized, inverse, gamma, weight, output_cotangent)
        saved_values = (standardized.name, inverse.name)
    else:
        inputs = (x, inverse, gamma, weight, output_cotangent)
        saved_values = (x.name, inverse.name)
        centered = x
        if recovered.statistic_kind is RowStatisticKind.CENTERED_SECOND_MOMENT:
            centered = _append_centering(operations, x, row, reduction)
            recomputed_values.extend(("row_sum", "row_mean", centered.name))
        operations.append(
            MapPrimitive(
                "reconstruct standardized value",
                (centered, inverse),
                standardized,
                _multiply(scalar_input(centered.name), scalar_input(inverse.name)),
            )
        )
        recomputed_values.append(standardized.name)
        if save_policy is RowNormalizationSavePolicy.RECOMPUTE_STATISTIC:
            inputs = (x, gamma, weight, output_cotangent)
            saved_values = (x.name,)
            operations.clear()
            centered = x
            if recovered.statistic_kind is RowStatisticKind.CENTERED_SECOND_MOMENT:
                centered = _append_centering(operations, x, row, reduction)
            _append_second_moment(operations, centered, inverse, row, reduction, recovered.epsilon)
            operations.append(
                MapPrimitive(
                    "reconstruct standardized value",
                    (centered, inverse),
                    standardized,
                    _multiply(scalar_input(centered.name), scalar_input(inverse.name)),
                )
            )
            recomputed_values = [operation.output.name for operation in operations]

    scaled = ProgramValue("backward.scaled", (row, reduction), x.dtype)
    projected = ProgramValue("backward.projected", (row, reduction), DType.FP32)
    weight_gradient = ProgramValue("cotangent.weight", (reduction, feature), DType.FP32)
    operations.extend(
        (
            MapPrimitive(
                "prepare weight-gradient operand",
                (standardized, gamma),
                scaled,
                _multiply(scalar_input(standardized.name), scalar_input(gamma.name)),
            ),
            ContractPrimitive(
                "Contract output cotangent with weight",
                (output_cotangent, weight),
                projected,
                (feature,),
                DType.FP32,
            ),
            ContractPrimitive(
                "Contract activation with output cotangent",
                (scaled, output_cotangent),
                weight_gradient,
                (row,),
                DType.FP32,
            ),
        )
    )
    u = ProgramValue("backward.local", (row, reduction), DType.FP32)
    gamma_terms = ProgramValue("backward.scale_terms", (row, reduction), DType.FP32)
    gamma_gradient = ProgramValue("cotangent.feature_scale", (reduction,), DType.FP32)
    correlation_terms = ProgramValue("backward.correlation_terms", (row, reduction), DType.FP32)
    correlation_sum = ProgramValue("backward.correlation_sum", (row,), DType.FP32)
    correlation_mean = ProgramValue("backward.correlation_mean", (row,), DType.FP32)
    operations.extend(
        (
            MapPrimitive(
                "apply feature scale to local cotangent",
                (projected, gamma),
                u,
                _multiply(scalar_input(projected.name), scalar_input(gamma.name)),
            ),
            MapPrimitive(
                "form feature-scale cotangent terms",
                (projected, standardized),
                gamma_terms,
                _multiply(scalar_input(projected.name), scalar_input(standardized.name)),
            ),
            FoldPrimitive(
                "Fold feature-scale cotangent terms",
                gamma_terms,
                gamma_gradient,
                (row,),
                FoldReducer.SUM,
                DType.FP32,
            ),
            MapPrimitive(
                "form row correlation terms",
                (u, standardized),
                correlation_terms,
                _multiply(scalar_input(u.name), scalar_input(standardized.name)),
            ),
            FoldPrimitive(
                "Fold row correlation terms",
                correlation_terms,
                correlation_sum,
                (reduction,),
                FoldReducer.SUM,
                DType.FP32,
            ),
            MapPrimitive(
                "average row correlation",
                (correlation_sum,),
                correlation_mean,
                _divide(scalar_input(correlation_sum.name), scalar_constant(float(reduction.extent))),
            ),
        )
    )
    mean_u: ProgramValue | None = None
    if recovered.statistic_kind is RowStatisticKind.CENTERED_SECOND_MOMENT:
        sum_u = ProgramValue("backward.local_sum", (row,), DType.FP32)
        mean_u = ProgramValue("backward.local_mean", (row,), DType.FP32)
        operations.extend(
            (
                FoldPrimitive("Fold local cotangent", u, sum_u, (reduction,), FoldReducer.SUM, DType.FP32),
                MapPrimitive(
                    "average local cotangent",
                    (sum_u,),
                    mean_u,
                    _divide(scalar_input(sum_u.name), scalar_constant(float(reduction.extent))),
                ),
            )
        )
    input_gradient = ProgramValue("cotangent.input", (row, reduction), DType.FP32)
    centered_term = _subtract(
        scalar_input(u.name),
        _multiply(scalar_input(standardized.name), scalar_input(correlation_mean.name)),
    )
    gradient_inputs = [u, standardized, correlation_mean, inverse]
    if mean_u is not None:
        centered_term = _subtract(centered_term, scalar_input(mean_u.name))
        gradient_inputs.append(mean_u)
    operations.append(
        MapPrimitive(
            "finalize input cotangent",
            tuple(gradient_inputs),
            input_gradient,
            _multiply(scalar_input(inverse.name), centered_term),
        )
    )
    return (
        TensorProgram(
            inputs=inputs,
            operations=tuple(operations),
            outputs=(input_gradient, gamma_gradient, weight_gradient),
        ),
        saved_values,
        tuple(recomputed_values),
    )


def _append_centering(
    operations: list[TensorPrimitive],
    x: ProgramValue,
    row: TensorAxis,
    reduction: TensorAxis,
) -> ProgramValue:
    row_sum = ProgramValue("row_sum", (row,), DType.FP32)
    row_mean = ProgramValue("row_mean", (row,), DType.FP32)
    centered = ProgramValue("centered", (row, reduction), x.dtype)
    operations.extend(
        (
            FoldPrimitive("Fold input", x, row_sum, (reduction,), FoldReducer.SUM, DType.FP32),
            MapPrimitive(
                "scale row statistic",
                (row_sum,),
                row_mean,
                _divide(scalar_input(row_sum.name), scalar_constant(float(reduction.extent))),
            ),
            MapPrimitive(
                "subtract row statistic",
                (x, row_mean),
                centered,
                _subtract(scalar_input(x.name), scalar_input(row_mean.name)),
            ),
        )
    )
    return centered


def _append_second_moment(
    operations: list[TensorPrimitive],
    centered: ProgramValue,
    inverse: ProgramValue,
    row: TensorAxis,
    reduction: TensorAxis,
    epsilon: float = 1e-5,
) -> None:
    squared = ProgramValue("squared", (row, reduction), DType.FP32)
    sum_square = ProgramValue("sum_square", (row,), DType.FP32)
    mean_square = ProgramValue("mean_square", (row,), DType.FP32)
    operations.extend(
        (
            MapPrimitive(
                "square local value",
                (centered,),
                squared,
                _multiply(scalar_input(centered.name), scalar_input(centered.name)),
            ),
            FoldPrimitive("Fold squared values", squared, sum_square, (reduction,), FoldReducer.SUM, DType.FP32),
            MapPrimitive(
                "scale second moment",
                (sum_square,),
                mean_square,
                _divide(scalar_input(sum_square.name), scalar_constant(float(reduction.extent))),
            ),
            MapPrimitive(
                "invert second moment",
                (mean_square,),
                inverse,
                scalar_unary(
                    ScalarExpressionKind.RSQRT,
                    _add(scalar_input(mean_square.name), scalar_constant(epsilon)),
                ),
            ),
        )
    )


def _forward_contract_skeleton(
    recovered: _RecoveredRowStatisticContract,
    placement: RowStatisticScalePlacement,
    tile_shape: tuple[int, int, int],
) -> GemmSkeleton:
    source = (
        recovered.input.name if recovered.statistic_kind is RowStatisticKind.UNCENTERED_SECOND_MOMENT else "centered"
    )
    if placement is RowStatisticScalePlacement.SOURCE_ORDERED_PREPARATION:
        standardized = "prepared.standardized"
        mainloop_input = "prepared.scaled"
        prologue = (
            Attachment(
                operation="scale_row",
                site=AttachmentSite.GEMM_PROLOGUE,
                inputs=(source, "inverse_scale"),
                outputs=(standardized,),
                attributes=(("expression_ast", _binary_ast(source, "inverse_scale")),),
            ),
            Attachment(
                operation="multiply",
                site=AttachmentSite.GEMM_PROLOGUE,
                inputs=(standardized, recovered.gamma.name),
                outputs=(mainloop_input,),
                attributes=(("expression_ast", _binary_ast(standardized, recovered.gamma.name)),),
            ),
        )
        epilogue: tuple[Attachment, ...] = ()
    else:
        mainloop_input = "prepared.feature_scaled"
        prologue = (
            Attachment(
                operation="multiply",
                site=AttachmentSite.GEMM_PROLOGUE,
                inputs=(source, recovered.gamma.name),
                outputs=(mainloop_input,),
                attributes=(("expression_ast", _binary_ast(source, recovered.gamma.name)),),
            ),
        )
        epilogue = (
            Attachment(
                operation="scale_row",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(recovered.output.name, "inverse_scale"),
                outputs=(recovered.output.name,),
            ),
        )
    return GemmSkeleton(
        name="generated row-statistic Contract",
        input=source,
        weight=recovered.weight.name,
        output=recovered.output.name,
        shape=(recovered.row_axis.extent, recovered.feature_axis.extent, recovered.reduction_axis.extent),
        accumulation_dtype=DType.FP32,
        backend=GENERIC_H100_GEMM_BACKEND,
        input_layout="row_major_mk",
        output_layout="row_major_mn",
        physical_tile_shape=tile_shape,
        cluster_shape=(1, 1, 1),
        pingpong=True,
        prologue=prologue,
        epilogue=epilogue,
    )


def _lower_training_operation(
    operation: TensorPrimitive,
    *,
    physical_tile_shape: tuple[int, int, int],
    map_tile_elements: int,
    fold_tile_extent: int,
) -> GeneratedTrainingSkeleton:
    if isinstance(operation, MapPrimitive):
        return GeneratedMapSkeleton(
            name=operation.name,
            inputs=tuple(value.name for value in operation.inputs),
            output=operation.output.name,
            output_shape=operation.output.shape,
            expression=operation.expression,
            tile_shape=tuple(min(extent, map_tile_elements) for extent in operation.output.shape),
        )
    if isinstance(operation, FoldPrimitive):
        return GeneratedFoldSkeleton(
            name=operation.name,
            input=operation.input.name,
            output=operation.output.name,
            input_shape=operation.input.shape,
            reduction_axes=tuple(axis.label or f"axis_{axis.id}" for axis in operation.reduction_axes),
            reducer=operation.reducer,
            accumulation_dtype=operation.accumulation_dtype,
            tile_extent=min(fold_tile_extent, max(axis.extent for axis in operation.reduction_axes)),
        )
    return _lower_contract(operation, physical_tile_shape)


def _lower_contract(operation: ContractPrimitive, tile_shape: tuple[int, int, int]) -> GeneratedContractSkeleton:
    if len(operation.inputs) != 2 or len(operation.output.axes) != 2 or len(operation.reduction_axes) != 1:
        raise ValueError("generic GEMM legalization requires a two-input rank-two Contract with one reduction axis")
    m_axis, n_axis = operation.output.axes
    k_axis = operation.reduction_axes[0]
    left, right = operation.inputs
    left_view = _matrix_view(left, (m_axis, k_axis))
    right_view = _matrix_view(right, (k_axis, n_axis))
    skeleton = GemmSkeleton(
        name=operation.name,
        input=left_view,
        weight=right_view,
        output=operation.output.name,
        shape=(m_axis.extent, n_axis.extent, k_axis.extent),
        accumulation_dtype=operation.accumulation_dtype,
        backend=GENERIC_H100_GEMM_BACKEND,
        input_layout="row_major_mk",
        output_layout="row_major_mn",
        physical_tile_shape=tile_shape,
        cluster_shape=(1, 1, 1),
        pingpong=True,
    )
    return GeneratedContractSkeleton(
        name=operation.name,
        operands=(left.name, right.name),
        operand_views=(left_view, right_view),
        output=operation.output.name,
        reduction_axis=k_axis.label or f"axis_{k_axis.id}",
        skeleton=skeleton,
    )


def _matrix_view(value: ProgramValue, axes: tuple[TensorAxis, TensorAxis]) -> str:
    if value.axes == axes:
        return value.name
    if value.axes == tuple(reversed(axes)):
        return f"transpose.{value.name}"
    raise ValueError(f"value {value.name!r} cannot legalize to matrix axes {[axis.label for axis in axes]}")


def _inverse_statistic_epsilon(expression: ScalarExpression) -> float:
    if expression.kind is not ScalarExpressionKind.RSQRT:
        raise ValueError("inverse row statistic must be an rsqrt Map")
    radicand = expression.operands[0]
    if radicand.kind is not ScalarExpressionKind.ADD:
        raise ValueError("inverse row statistic must add an explicit epsilon")
    constants = tuple(operand.constant for operand in radicand.operands if operand.kind is ScalarExpressionKind.CONSTANT)
    if len(constants) != 1 or isinstance(constants[0], bool):
        raise ValueError("inverse row statistic must contain one numeric epsilon")
    return float(constants[0])


def _binary_ast(left: str, right: str) -> str:
    return serialize_scalar_expression(_multiply(scalar_input(left), scalar_input(right)))


def _add(left: ScalarExpression, right: ScalarExpression) -> ScalarExpression:
    return scalar_binary(ScalarExpressionKind.ADD, left, right)


def _subtract(left: ScalarExpression, right: ScalarExpression) -> ScalarExpression:
    return scalar_binary(ScalarExpressionKind.SUBTRACT, left, right)


def _multiply(left: ScalarExpression, right: ScalarExpression) -> ScalarExpression:
    return scalar_binary(ScalarExpressionKind.MULTIPLY, left, right)


def _divide(left: ScalarExpression, right: ScalarExpression) -> ScalarExpression:
    return scalar_binary(ScalarExpressionKind.DIVIDE, left, right)
