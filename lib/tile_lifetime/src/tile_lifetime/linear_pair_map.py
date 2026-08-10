# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile a natural pair-gated linear map into generated training skeletons."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from shuttle.ir import DType
from tile_lifetime.autodiff import scalar_expression_vjp
from tile_lifetime.gemm_program import GENERIC_H100_GEMM_BACKEND
from tile_lifetime.plan import Attachment, AttachmentSite, GemmSkeleton
from tile_lifetime.tensor_program import (
    ContractPrimitive,
    MapPrimitive,
    ProgramValue,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
    scalar_binary,
    scalar_constant,
    scalar_input,
    scalar_unary,
    serialize_scalar_expression,
)


class PairMapSavePolicy(StrEnum):
    """Forward value retained for a generated pair-Map reverse pass."""

    SAVE_PREACTIVATION = "save_preactivation"
    RECOMPUTE_PREACTIVATION = "recompute_preactivation"


@dataclass(frozen=True)
class PairMapVjpProgram:
    """Two adjacent-lane VJPs evaluated by one generic pointwise skeleton."""

    left: ScalarExpression
    right: ScalarExpression
    input_names: tuple[str, str, str] = ("pair.left", "pair.right", "cotangent")

    def __post_init__(self) -> None:
        allowed = set(self.input_names)
        for expression in (self.left, self.right):
            inputs = _expression_inputs(expression)
            if "cotangent" not in inputs or not inputs <= allowed:
                raise ValueError("pair-Map VJP expressions must consume the cotangent and only declared pair inputs")


@dataclass(frozen=True)
class LinearPairMapTrainingProgram:
    """Generated forward/Map-adjoint/Contract-adjoint physical programs."""

    source: TensorProgram
    save_policy: PairMapSavePolicy
    forward: GemmSkeleton
    recompute: GemmSkeleton | None
    pair_vjp: PairMapVjpProgram
    input_gradient: GemmSkeleton
    weight_gradient: GemmSkeleton
    activation: ProgramValue
    left_weight: ProgramValue
    right_weight: ProgramValue
    output: ProgramValue
    physical_interleaved_weight: str
    physical_interleaved_weight_transpose: str
    preactivation: str
    output_cotangent: str
    preactivation_cotangent: str


def pair_silu_product_expression() -> ScalarExpression:
    """Return ordinary scalar SiLU(left) multiplied by right."""
    left = scalar_input("pair.left")
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
        scalar_input("pair.right"),
    )


def pair_tanh_product_expression() -> ScalarExpression:
    """Return a semantic mutation used to audit generator independence."""
    return scalar_binary(
        ScalarExpressionKind.MULTIPLY,
        scalar_unary(ScalarExpressionKind.TANH, scalar_input("pair.left")),
        scalar_input("pair.right"),
    )


def build_linear_pair_map_program(
    *,
    rows: int,
    reduction: int,
    features: int,
    pair_expression: ScalarExpression,
    dtype: DType = DType.BF16,
) -> TensorProgram:
    """Build natural two-projection plus scalar-Map tensor algebra.

    The source does not request a combined GEMM or an epilogue. Those are
    physical choices made by :func:`compile_linear_pair_map_training`.
    """
    row = TensorAxis(0, rows, "row")
    reduction_axis = TensorAxis(1, reduction, "reduction")
    feature = TensorAxis(2, features, "feature")
    activation = ProgramValue("activation", (row, reduction_axis), dtype)
    left_weight = ProgramValue("left_weight", (reduction_axis, feature), dtype)
    right_weight = ProgramValue("right_weight", (reduction_axis, feature), dtype)
    left = ProgramValue("left_projection", (row, feature), dtype)
    right = ProgramValue("right_projection", (row, feature), dtype)
    output = ProgramValue("pair_map_output", (row, feature), dtype)
    source_expression = _rename_expression(
        pair_expression,
        {"pair.left": left.name, "pair.right": right.name},
    )
    return TensorProgram(
        inputs=(activation, left_weight, right_weight),
        operations=(
            ContractPrimitive(
                name="left projection",
                inputs=(activation, left_weight),
                output=left,
                reduction_axes=(reduction_axis,),
                accumulation_dtype=DType.FP32,
            ),
            ContractPrimitive(
                name="right projection",
                inputs=(activation, right_weight),
                output=right,
                reduction_axes=(reduction_axis,),
                accumulation_dtype=DType.FP32,
            ),
            MapPrimitive(
                name="pair map",
                inputs=(left, right),
                output=output,
                expression=source_expression,
            ),
        ),
        outputs=(output,),
    )


def compile_linear_pair_map_training(
    source: TensorProgram,
    *,
    save_policy: PairMapSavePolicy,
    physical_tile_shape: tuple[int, int, int] = (128, 256, 64),
    cluster_shape: tuple[int, int, int] = (1, 2, 1),
    pingpong: bool = True,
) -> LinearPairMapTrainingProgram:
    """Recover a generic shared-input Contract pair followed by one Map."""
    left_contract, right_contract, map_operation = _recover_pair_map(source)
    activation, left_weight, right_weight = _contract_operands(left_contract, right_contract)
    row_axis, reduction_axis = activation.axes
    feature_axis = map_operation.output.axes[-1]
    rows = row_axis.extent
    reduction = reduction_axis.extent
    features = feature_axis.extent
    physical_weight = "physical.interleaved_weight"
    physical_weight_transpose = "physical.interleaved_weight_transpose"
    preactivation = "physical.preactivation_pairs"
    output_cotangent = f"cotangent.{map_operation.output.name}"
    preactivation_cotangent = f"cotangent.{preactivation}"
    pair_expression = _rename_expression(
        map_operation.expression,
        {
            map_operation.inputs[0].name: "pair.left",
            map_operation.inputs[1].name: "pair.right",
        },
    )
    pair_attachment = Attachment(
        operation="pairwise_map",
        site=AttachmentSite.GEMM_EPILOGUE,
        inputs=(preactivation,),
        outputs=(map_operation.output.name,),
        attributes=(("expression_ast", serialize_scalar_expression(pair_expression)),),
    )
    if save_policy is PairMapSavePolicy.SAVE_PREACTIVATION:
        output_name = preactivation
        output_layout = "row_major_mn"
        epilogue = (
            pair_attachment,
            Attachment(
                operation="store_tile",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(preactivation,),
                outputs=(),
                attributes=(("destination", preactivation), ("layout", "row_major_mn")),
            ),
        )
        recompute = None
    else:
        output_name = map_operation.output.name
        output_layout = "row_major_mn_pair_reduced"
        epilogue = (pair_attachment,)
        recompute = _plain_gemm(
            name="recompute paired preactivation",
            input_name=activation.name,
            weight_name=physical_weight_transpose,
            output_name=preactivation,
            shape=(rows, 2 * features, reduction),
            tile_shape=physical_tile_shape,
            cluster_shape=cluster_shape,
            pingpong=pingpong,
        )
    forward = GemmSkeleton(
        name="combined Contract pair with generated finalization Map",
        input=activation.name,
        weight=physical_weight_transpose,
        output=output_name,
        shape=(rows, 2 * features, reduction),
        accumulation_dtype=left_contract.accumulation_dtype,
        backend=GENERIC_H100_GEMM_BACKEND,
        input_layout="row_major_mk",
        output_layout=output_layout,
        physical_tile_shape=physical_tile_shape,
        cluster_shape=cluster_shape,
        pingpong=pingpong,
        epilogue=epilogue,
    )
    source_left = map_operation.inputs[0].name
    source_right = map_operation.inputs[1].name
    left_vjp = scalar_expression_vjp(
        map_operation.expression,
        input_name=source_left,
        cotangent_name=output_cotangent,
    )
    right_vjp = scalar_expression_vjp(
        map_operation.expression,
        input_name=source_right,
        cotangent_name=output_cotangent,
    )
    vjp_aliases = {
        source_left: "pair.left",
        source_right: "pair.right",
        output_cotangent: "cotangent",
    }
    pair_vjp = PairMapVjpProgram(
        left=_rename_expression(left_vjp, vjp_aliases),
        right=_rename_expression(right_vjp, vjp_aliases),
    )
    input_gradient = _plain_gemm(
        name="Contract adjoint for shared activation",
        input_name=preactivation_cotangent,
        weight_name=physical_weight,
        output_name=f"cotangent.{activation.name}",
        shape=(rows, reduction, 2 * features),
        tile_shape=physical_tile_shape,
        cluster_shape=(1, 1, 1),
        pingpong=pingpong,
    )
    weight_gradient = _plain_gemm(
        name="Contract adjoint for combined weight",
        input_name=f"transpose.{preactivation_cotangent}",
        weight_name=activation.name,
        output_name=f"cotangent.{physical_weight}",
        shape=(2 * features, reduction, rows),
        tile_shape=physical_tile_shape,
        cluster_shape=(1, 1, 1),
        pingpong=pingpong,
    )
    return LinearPairMapTrainingProgram(
        source=source,
        save_policy=save_policy,
        forward=forward,
        recompute=recompute,
        pair_vjp=pair_vjp,
        input_gradient=input_gradient,
        weight_gradient=weight_gradient,
        activation=activation,
        left_weight=left_weight,
        right_weight=right_weight,
        output=map_operation.output,
        physical_interleaved_weight=physical_weight,
        physical_interleaved_weight_transpose=physical_weight_transpose,
        preactivation=preactivation,
        output_cotangent=output_cotangent,
        preactivation_cotangent=preactivation_cotangent,
    )


def _recover_pair_map(
    source: TensorProgram,
) -> tuple[ContractPrimitive, ContractPrimitive, MapPrimitive]:
    if len(source.outputs) != 1:
        raise ValueError("linear pair-Map recovery requires one program output")
    producer_by_output = {operation.output.name: operation for operation in source.operations}
    output_producer = producer_by_output.get(source.outputs[0].name)
    if not isinstance(output_producer, MapPrimitive) or len(output_producer.inputs) != 2:
        raise ValueError("program output must be a two-input scalar Map")
    contract_producers = tuple(producer_by_output.get(value.name) for value in output_producer.inputs)
    if not all(isinstance(operation, ContractPrimitive) for operation in contract_producers):
        raise ValueError("both pair-Map inputs must come from Contracts")
    left, right = contract_producers
    assert isinstance(left, ContractPrimitive) and isinstance(right, ContractPrimitive)
    if left.output.axes != right.output.axes or left.reduction_axes != right.reduction_axes:
        raise ValueError("pair Contracts must have matching output and reduction axes")
    if left.accumulation_dtype is not DType.FP32 or right.accumulation_dtype is not DType.FP32:
        raise ValueError("generated H100 pair Contracts require FP32 accumulation")
    return left, right, output_producer


def _contract_operands(
    left: ContractPrimitive,
    right: ContractPrimitive,
) -> tuple[ProgramValue, ProgramValue, ProgramValue]:
    if len(left.inputs) != 2 or len(right.inputs) != 2:
        raise ValueError("linear pair-Map recovery requires binary Contracts")
    shared = tuple(value for value in left.inputs if value in right.inputs)
    if len(shared) != 1:
        raise ValueError("pair Contracts must share exactly one activation input")
    activation = shared[0]
    left_weight = next(value for value in left.inputs if value != activation)
    right_weight = next(value for value in right.inputs if value != activation)
    if len(activation.axes) != 2 or len(left_weight.axes) != 2 or len(right_weight.axes) != 2:
        raise ValueError("first generated physical proof supports matrix-shaped pair Contracts")
    if left_weight.axes != right_weight.axes:
        raise ValueError("pair Contract weights must have matching logical layouts")
    return activation, left_weight, right_weight


def _plain_gemm(
    *,
    name: str,
    input_name: str,
    weight_name: str,
    output_name: str,
    shape: tuple[int, int, int],
    tile_shape: tuple[int, int, int],
    cluster_shape: tuple[int, int, int],
    pingpong: bool,
) -> GemmSkeleton:
    return GemmSkeleton(
        name=name,
        input=input_name,
        weight=weight_name,
        output=output_name,
        shape=shape,
        accumulation_dtype=DType.FP32,
        backend=GENERIC_H100_GEMM_BACKEND,
        input_layout="row_major_mk",
        output_layout="row_major_mn",
        physical_tile_shape=tile_shape,
        cluster_shape=cluster_shape,
        pingpong=pingpong,
    )


def _rename_expression(expression: ScalarExpression, names: dict[str, str]) -> ScalarExpression:
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return scalar_input(names.get(expression.input_name, expression.input_name))
    if expression.kind is ScalarExpressionKind.CONSTANT:
        assert expression.constant is not None
        return scalar_constant(expression.constant)
    return ScalarExpression(
        kind=expression.kind,
        operands=tuple(_rename_expression(operand, names) for operand in expression.operands),
    )


def _expression_inputs(expression: ScalarExpression) -> set[str]:
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return {expression.input_name}
    result: set[str] = set()
    for operand in expression.operands:
        result.update(_expression_inputs(operand))
    return result
