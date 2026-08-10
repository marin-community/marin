# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile semantic GEMM attachments into reusable tile programs."""

from dataclasses import dataclass

from tile_lifetime.ir import DType
from tile_lifetime.plan import Attachment, AttachmentSite, GemmSkeleton
from tile_lifetime.tile_program import (
    TileOp,
    TilePrimitive,
    TileProgram,
    TileProgramError,
    TileProgramStage,
    optimize_tile_program,
)

GENERIC_H100_GEMM_BACKEND = "h100_gemm_tile_program"


@dataclass(frozen=True)
class GemmProgram:
    """One fixed GEMM mainloop surrounded by optimized tile programs."""

    input: str
    weight: str
    output: str
    shape: tuple[int, int, int]
    accumulation_dtype: DType
    input_layout: str
    output_layout: str
    tile_program: TileProgram
    mainloop_input: str
    stored_values: tuple[str, ...]

    @property
    def preparation(self) -> tuple[TileOp, ...]:
        """Operations applied before the GEMM mainloop consumes its A tile."""
        return self.tile_program.operations_at(TileProgramStage.PREPARATION)

    @property
    def finalization(self) -> tuple[TileOp, ...]:
        """Operations applied while the GEMM accumulator tile is resident."""
        return self.tile_program.operations_at(TileProgramStage.FINALIZATION)


_ATTACHMENT_PRIMITIVES = {
    "scale_row": TilePrimitive.SCALE_ROW,
    "residual_add": TilePrimitive.RESIDUAL_ADD,
    "multiply_gamma": TilePrimitive.MULTIPLY_GAMMA,
    "partial_sum_square": TilePrimitive.PARTIAL_SUM_SQUARE,
    "partial_sum": TilePrimitive.PARTIAL_SUM,
    "pairwise_map": TilePrimitive.PAIRWISE_MAP,
    "pairwise_linear_map": TilePrimitive.PAIRWISE_LINEAR_MAP,
    "pairwise_swiglu": TilePrimitive.PAIRWISE_SWIGLU,
    "pairwise_rope_q": TilePrimitive.PAIRWISE_ROPE,
    "pairwise_rope_k": TilePrimitive.PAIRWISE_ROPE,
    "alias_reshape_bsh": TilePrimitive.VIEW,
    "partition_qkv_segment_views_bshd": TilePrimitive.PARTITION,
    "partition": TilePrimitive.PARTITION,
    "view": TilePrimitive.VIEW,
    "add": TilePrimitive.ADD,
    "subtract": TilePrimitive.SUBTRACT,
    "multiply": TilePrimitive.MULTIPLY,
    "convert": TilePrimitive.CONVERT,
    "load_tile": TilePrimitive.LOAD_TILE,
    "load_row": TilePrimitive.LOAD_ROW,
    "load_edge_weight": TilePrimitive.LOAD_EDGE_WEIGHT,
    "load_state": TilePrimitive.LOAD_STATE,
    "store_tile": TilePrimitive.STORE,
}

_SUPPORTED_INPUT_LAYOUTS = frozenset({"row_major_mk", "bsh_contiguous"})
_SUPPORTED_OUTPUT_LAYOUTS = frozenset(
    {"row_major_mn", "row_major_mn_pair_reduced", "fa3_bshd_last_dimension_contiguous"}
)


def compile_gemm_program(skeleton: GemmSkeleton) -> GemmProgram:
    """Lower natural attachment dataflow without inspecting a workload name."""
    if skeleton.backend != GENERIC_H100_GEMM_BACKEND:
        raise TileProgramError(f"GEMM backend {skeleton.backend!r} is not the generic tile-program backend")
    if skeleton.input_layout not in _SUPPORTED_INPUT_LAYOUTS:
        raise TileProgramError(f"unsupported GEMM input layout {skeleton.input_layout!r}")
    if skeleton.output_layout not in _SUPPORTED_OUTPUT_LAYOUTS:
        raise TileProgramError(f"unsupported GEMM output layout {skeleton.output_layout!r}")
    if skeleton.accumulation_dtype is not DType.FP32:
        raise TileProgramError("H100 GEMM tile programs require FP32 accumulation")

    preparation = tuple(_lower_attachment(attachment) for attachment in skeleton.prologue)
    finalization = tuple(_lower_attachment(attachment) for attachment in skeleton.epilogue)
    _validate_attachment_sites(skeleton)
    _validate_composition(skeleton, preparation, finalization)

    mainloop_input = _terminal_values(preparation, fallback=(skeleton.input,))[-1]
    stored_values = _terminal_values(finalization, fallback=(skeleton.output,))
    preparation_conversion: tuple[TileOp, ...] = ()
    if any(
        operation.primitive not in {TilePrimitive.CONVERT, TilePrimitive.VIEW} for operation in preparation
    ) and not any(operation.primitive is TilePrimitive.CONVERT for operation in preparation):
        converted_input = f"{mainloop_input}.mainloop_bf16"
        preparation_conversion = (
            TileOp(
                primitive=TilePrimitive.CONVERT,
                stage=TileProgramStage.PREPARATION,
                inputs=(mainloop_input,),
                outputs=(converted_input,),
                attributes=(("dtype", DType.BF16.value),),
            ),
        )
        mainloop_input = converted_input
    stores = _storage_operations(finalization, stored_values, skeleton.output_layout)
    operations = (*preparation, *preparation_conversion, *finalization, *stores)
    value_layouts = _infer_value_layouts(skeleton, operations)
    tile_program = optimize_tile_program(
        operations,
        required_outputs=(mainloop_input,),
        value_layouts=value_layouts,
    )
    return GemmProgram(
        input=skeleton.input,
        weight=skeleton.weight,
        output=skeleton.output,
        shape=skeleton.shape,
        accumulation_dtype=skeleton.accumulation_dtype,
        input_layout=skeleton.input_layout,
        output_layout=skeleton.output_layout,
        tile_program=tile_program,
        mainloop_input=mainloop_input,
        stored_values=stored_values,
    )


def _lower_attachment(attachment: Attachment) -> TileOp:
    try:
        primitive = _ATTACHMENT_PRIMITIVES[attachment.operation]
    except KeyError as error:
        raise TileProgramError(f"unsupported GEMM attachment {attachment.operation!r}") from error
    stage = (
        TileProgramStage.PREPARATION
        if attachment.site is AttachmentSite.GEMM_PROLOGUE
        else TileProgramStage.FINALIZATION
    )
    attributes = attachment.attributes
    if attachment.operation == "pairwise_rope_q":
        attributes = (*attributes, ("segment", "query"))
    elif attachment.operation == "pairwise_rope_k":
        attributes = (*attributes, ("segment", "key"))
    return TileOp(
        primitive=primitive,
        stage=stage,
        inputs=attachment.inputs,
        outputs=attachment.outputs,
        attributes=attributes,
    )


def _validate_attachment_sites(skeleton: GemmSkeleton) -> None:
    for attachment in skeleton.prologue:
        if attachment.site is not AttachmentSite.GEMM_PROLOGUE:
            raise TileProgramError(f"preparation attachment {attachment.operation!r} has site {attachment.site.value}")
    for attachment in skeleton.epilogue:
        if attachment.site is not AttachmentSite.GEMM_EPILOGUE:
            raise TileProgramError(f"finalization attachment {attachment.operation!r} has site {attachment.site.value}")


def _validate_composition(
    skeleton: GemmSkeleton,
    preparation: tuple[TileOp, ...],
    finalization: tuple[TileOp, ...],
) -> None:
    if any(
        operation.primitive in {TilePrimitive.PAIRWISE_MAP, TilePrimitive.PAIRWISE_SWIGLU} for operation in finalization
    ):
        if skeleton.shape[1] % 2:
            raise TileProgramError("pairwise SwiGLU requires an even GEMM N dimension")
        if skeleton.output_layout not in {"row_major_mn", "row_major_mn_pair_reduced"}:
            raise TileProgramError("pairwise SwiGLU requires a row-major output layout")
    if any(
        operation.primitive in {TilePrimitive.PAIRWISE_LINEAR_MAP, TilePrimitive.PAIRWISE_ROPE, TilePrimitive.PARTITION}
        for operation in finalization
    ):
        if skeleton.shape[1] % 2:
            raise TileProgramError("pairwise RoPE requires an even GEMM N dimension")
        if skeleton.output_layout != "fa3_bshd_last_dimension_contiguous":
            raise TileProgramError("QKV partition and RoPE require the FA3 BSHD boundary layout")
    if any(operation.primitive is TilePrimitive.SCALE_ROW for operation in preparation):
        if skeleton.input_layout != "row_major_mk":
            raise TileProgramError("preparation row scaling requires row_major_mk input")
    valid_preparation_deliveries = {"tile", "row", "feature"}
    for operation in preparation:
        if operation.primitive not in {TilePrimitive.ADD, TilePrimitive.SUBTRACT, TilePrimitive.MULTIPLY}:
            continue
        delivery = dict(operation.attributes).get("input.1_delivery", "tile")
        if delivery not in valid_preparation_deliveries:
            raise TileProgramError(f"unsupported preparation operand delivery {delivery!r}")
    if any(
        operation.primitive in {TilePrimitive.PARTIAL_SUM, TilePrimitive.PARTIAL_SUM_SQUARE} for operation in preparation
    ):
        raise TileProgramError("partial reductions are only supported in GEMM finalization")


def _terminal_values(operations: tuple[TileOp, ...], *, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if not operations:
        return fallback
    last_output: dict[str, int] = {}
    last_input: dict[str, int] = {}
    for index, operation in enumerate(operations):
        for value in operation.outputs:
            last_output[value] = index
        for value in operation.inputs:
            last_input[value] = index
    terminals = tuple(value for value, output_index in last_output.items() if output_index >= last_input.get(value, -1))
    return terminals or fallback


def _infer_value_layouts(skeleton: GemmSkeleton, operations: tuple[TileOp, ...]) -> dict[str, str]:
    assert skeleton.input_layout is not None
    assert skeleton.output_layout is not None
    layouts = {
        skeleton.input: skeleton.input_layout,
        skeleton.output: skeleton.output_layout,
    }
    for operation in operations:
        if operation.primitive in {TilePrimitive.PARTIAL_SUM, TilePrimitive.PARTIAL_SUM_SQUARE}:
            output_layout = "row_partial_fp32"
        elif operation.stage is TileProgramStage.PREPARATION:
            output_layout = skeleton.input_layout
        else:
            output_layout = skeleton.output_layout
        layouts.update((output, output_layout) for output in operation.outputs)
    return layouts


def _storage_operations(
    finalization: tuple[TileOp, ...],
    stored_values: tuple[str, ...],
    output_layout: str,
) -> tuple[TileOp, ...]:
    fp32_values = {
        output
        for operation in finalization
        if operation.primitive in {TilePrimitive.PARTIAL_SUM, TilePrimitive.PARTIAL_SUM_SQUARE}
        for output in operation.outputs
    }
    operations: list[TileOp] = []
    for value in stored_values:
        stored_input = value
        if value not in fp32_values:
            stored_input = f"{value}.store_bf16"
            operations.append(
                TileOp(
                    primitive=TilePrimitive.CONVERT,
                    stage=TileProgramStage.FINALIZATION,
                    inputs=(value,),
                    outputs=(stored_input,),
                    attributes=(("dtype", DType.BF16.value),),
                )
            )
        operations.append(
            TileOp(
                primitive=TilePrimitive.STORE,
                stage=TileProgramStage.FINALIZATION,
                inputs=(stored_input,),
                outputs=(),
                attributes=(("destination", value), ("layout", output_layout)),
            )
        )
    return tuple(operations)
