# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose accepted tile-lifetime skeletons across one dense Llama block."""

from dataclasses import dataclass

from tile_lifetime.attention import compile_attention_region, select_hopper_attention_config
from tile_lifetime.compiler import RMSScalePlacement
from tile_lifetime.ir import (
    DType,
    LinearOp,
    PairwiseSwiGLUOp,
    QKVProjectionOp,
    ResidualAddOp,
    RMSNormOp,
    RoPEOp,
    ScaledDotProductAttentionOp,
    TensorGraph,
    TensorValue,
    ViewOp,
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
)

H100_GEMM_TILE_SHAPE = (128, 256, 64)


@dataclass(frozen=True)
class _DenseRegion:
    input_view: ViewOp
    qkv: QKVProjectionOp
    rope: RoPEOp
    attention: ScaledDotProductAttentionOp
    attention_view: ViewOp
    output_projection: LinearOp
    attention_residual: ResidualAddOp
    mlp_norm: RMSNormOp
    gate_up: LinearOp
    swiglu: PairwiseSwiGLUOp
    down_projection: LinearOp
    mlp_residual: ResidualAddOp
    next_norm: RMSNormOp
    next_input_view: ViewOp
    next_qkv: QKVProjectionOp
    next_rope: RoPEOp


def compile_dense_transformer_region(
    graph: TensorGraph,
    *,
    numerical_policy: NumericalPolicy,
    rms_scale_placement: RMSScalePlacement = RMSScalePlacement.CONSUMER_PROLOGUE,
) -> RegionPlan:
    """Compile one dense Llama block through the following QKV/RoPE boundary."""
    if numerical_policy is NumericalPolicy.BITWISE_EXACT:
        raise ValueError("the dense structural prototype requires the rounding-reorder numerical policy")
    region = _recover_dense_region(graph)
    return _dense_plan(graph, region, rms_scale_placement=rms_scale_placement)


def _recover_dense_region(graph: TensorGraph) -> _DenseRegion:
    attentions = tuple(operation for operation in graph.operations if isinstance(operation, ScaledDotProductAttentionOp))
    if len(attentions) != 1:
        raise ValueError(f"expected one attention operation, found {len(attentions)}")
    attention = attentions[0]

    rope = graph.producer(attention.query)
    qkv = graph.producer(attention.value)
    if not isinstance(rope, RoPEOp) or graph.producer(attention.key) != rope:
        raise ValueError("attention Q and K are not produced by one RoPE operation")
    if not isinstance(qkv, QKVProjectionOp) or graph.producer(rope.query) != qkv or graph.producer(rope.key) != qkv:
        raise ValueError("attention Q, K, and V do not trace back to one QKV projection")
    input_view = graph.producer(qkv.input)
    if not isinstance(input_view, ViewOp):
        raise ValueError("initial QKV input is not a zero-cost BSH view")

    attention_view = _single_consumer(graph, attention.output, ViewOp)
    output_projection = _single_consumer(graph, attention_view.output, LinearOp)
    attention_residual = _single_consumer(graph, output_projection.output, ResidualAddOp)
    if input_view.input not in (attention_residual.left, attention_residual.right):
        raise ValueError("attention output projection does not add the original residual stream")
    mlp_norm = _consumer_of_type(graph, attention_residual.output, RMSNormOp)
    gate_up = _single_consumer(graph, mlp_norm.output, LinearOp)
    swiglu = _single_consumer(graph, gate_up.output, PairwiseSwiGLUOp)
    down_projection = _single_consumer(graph, swiglu.output, LinearOp)
    mlp_residual = _single_consumer(graph, down_projection.output, ResidualAddOp)
    if attention_residual.output not in (mlp_residual.left, mlp_residual.right):
        raise ValueError("MLP down projection does not add the attention residual stream")
    next_norm = _consumer_of_type(graph, mlp_residual.output, RMSNormOp)
    next_input_view = _single_consumer(graph, next_norm.output, ViewOp)
    next_qkv = _single_consumer(graph, next_input_view.output, QKVProjectionOp)
    next_rope = _shared_qk_consumer(graph, next_qkv)

    expected_operations = {
        operation.id
        for operation in (
            input_view,
            qkv,
            rope,
            attention,
            attention_view,
            output_projection,
            attention_residual,
            mlp_norm,
            gate_up,
            swiglu,
            down_projection,
            mlp_residual,
            next_norm,
            next_input_view,
            next_qkv,
            next_rope,
        )
    }
    if expected_operations != {operation.id for operation in graph.operations}:
        raise ValueError("dense structural prototype does not permit additional semantic operations")
    return _DenseRegion(
        input_view=input_view,
        qkv=qkv,
        rope=rope,
        attention=attention,
        attention_view=attention_view,
        output_projection=output_projection,
        attention_residual=attention_residual,
        mlp_norm=mlp_norm,
        gate_up=gate_up,
        swiglu=swiglu,
        down_projection=down_projection,
        mlp_residual=mlp_residual,
        next_norm=next_norm,
        next_input_view=next_input_view,
        next_qkv=next_qkv,
        next_rope=next_rope,
    )


def _single_consumer(graph: TensorGraph, value: TensorValue, expected_type: type) -> object:
    consumers = graph.consumers(value)
    if len(consumers) != 1 or not isinstance(consumers[0], expected_type):
        raise ValueError(f"{value.name} does not have one {expected_type.__name__} consumer")
    return consumers[0]


def _consumer_of_type(graph: TensorGraph, value: TensorValue, expected_type: type) -> object:
    consumers = tuple(operation for operation in graph.consumers(value) if isinstance(operation, expected_type))
    if len(consumers) != 1:
        raise ValueError(f"{value.name} does not have one {expected_type.__name__} consumer")
    return consumers[0]


def _shared_qk_consumer(graph: TensorGraph, projection: QKVProjectionOp) -> RoPEOp:
    query_consumer = _single_consumer(graph, projection.query, RoPEOp)
    key_consumer = _single_consumer(graph, projection.key, RoPEOp)
    if query_consumer != key_consumer:
        raise ValueError("next Q and K do not share one RoPE operation")
    assert isinstance(query_consumer, RoPEOp)
    return query_consumer


def _dense_plan(
    graph: TensorGraph,
    region: _DenseRegion,
    *,
    rms_scale_placement: RMSScalePlacement,
) -> RegionPlan:
    attention_plan = compile_attention_region(graph, numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER)
    attention_config = select_hopper_attention_config(region.attention)
    first_qkv = _qkv_rope_gemm(
        region.qkv,
        region.rope,
        input_value=region.qkv.input.name,
        backend="quack_sm90_rope_posfreq",
        input_layout="bsh_contiguous",
        output_layout=attention_config.input_layout,
        physical_tile_shape=H100_GEMM_TILE_SHAPE,
        cluster_shape=_qkv_cluster_shape(region.qkv.output.shape[-1]),
        pingpong=False,
    )

    mlp_scaled = f"{region.attention_residual.output.name}_times_{region.mlp_norm.gamma.name}"
    mlp_partials = f"{region.attention_residual.output.name}_rms_partials"
    mlp_inverse_rms = f"{region.attention_residual.output.name}_inverse_rms"
    output_projection = _residual_rms_gemm(
        region.output_projection,
        region.attention_residual,
        region.mlp_norm,
        residual=region.input_view.input.name,
        scaled_output=mlp_scaled,
        partials=mlp_partials,
    )
    mlp_reduction = _rms_reduction(region.mlp_norm, mlp_partials, mlp_inverse_rms)
    gate_up_attachments = _gate_up_rms_attachments(
        region,
        scaled_input=mlp_scaled,
        inverse_rms=mlp_inverse_rms,
        scale_placement=rms_scale_placement,
    )
    gate_up = GemmSkeleton(
        name=f"gate_up_{rms_scale_placement.value}_rms_scale_pairwise_swiglu",
        input=mlp_scaled,
        weight=region.gate_up.weight.name,
        output=region.swiglu.output.name,
        shape=(region.gate_up.output.shape[0], region.gate_up.output.shape[1], region.gate_up.input.shape[1]),
        accumulation_dtype=region.gate_up.accumulation_dtype,
        backend=(
            "quack_sm90_fp32_a_transform_swiglu_dead_preact"
            if rms_scale_placement is RMSScalePlacement.CONSUMER_PROLOGUE
            else "quack_sm90_rstd_swiglu_dead_preact"
        ),
        input_layout="row_major_mk",
        output_layout="row_major_mn",
        physical_tile_shape=H100_GEMM_TILE_SHAPE,
        cluster_shape=_prologue_cluster_shape(region.gate_up.output.shape[1]),
        pingpong=False,
        prologue=gate_up_attachments[0],
        epilogue=gate_up_attachments[1],
    )

    next_scaled = f"{region.mlp_residual.output.name}_times_{region.next_norm.gamma.name}"
    next_partials = f"{region.mlp_residual.output.name}_rms_partials"
    next_inverse_rms = f"{region.mlp_residual.output.name}_inverse_rms"
    down_projection = _residual_rms_gemm(
        region.down_projection,
        region.mlp_residual,
        region.next_norm,
        residual=region.attention_residual.output.name,
        scaled_output=next_scaled,
        partials=next_partials,
    )
    next_reduction = _rms_reduction(region.next_norm, next_partials, next_inverse_rms)
    next_qkv = _qkv_rope_gemm(
        region.next_qkv,
        region.next_rope,
        input_value=next_scaled,
        backend=(
            "quack_sm90_fp32_a_transform_rope_posfreq"
            if rms_scale_placement is RMSScalePlacement.CONSUMER_PROLOGUE
            else "quack_sm90_rstd_rope_posfreq"
        ),
        input_layout="row_major_mk",
        output_layout="fa3_bshd_last_dimension_contiguous",
        physical_tile_shape=H100_GEMM_TILE_SHAPE,
        cluster_shape=_qkv_cluster_shape(region.next_qkv.output.shape[-1]),
        pingpong=False,
        prologue=_next_qkv_prologue(
            region,
            scaled_input=next_scaled,
            inverse_rms=next_inverse_rms,
            scale_placement=rms_scale_placement,
        ),
        epilogue_prefix=_next_qkv_epilogue_prefix(
            region,
            scaled_input=next_scaled,
            inverse_rms=next_inverse_rms,
            scale_placement=rms_scale_placement,
        ),
    )

    skeletons = (
        first_qkv,
        *attention_plan.skeletons,
        output_projection,
        mlp_reduction,
        gate_up,
        down_projection,
        next_reduction,
        next_qkv,
    )
    materializations = (
        _record(
            region.input_view.output,
            MaterializationDisposition.ALIAS,
            "zero-cost BSH view of the residual input",
            alias_of=region.input_view.input.name,
        ),
        *_qkv_rope_records(region.qkv, region.rope, "FA3 input boundary"),
        *attention_plan.materializations,
        _record(
            region.attention_view.output,
            MaterializationDisposition.ALIAS,
            "zero-cost flattening view of the contiguous attention output",
            alias_of=region.attention_view.input.name,
        ),
        _record(
            region.output_projection.output, MaterializationDisposition.EPILOGUE_ONLY, "consumed by residual epilogue"
        ),
        _record(
            region.attention_residual.output,
            MaterializationDisposition.MATERIALIZE,
            "saved residual stream for the MLP down-projection epilogue",
        ),
        _record(
            region.mlp_norm.output,
            _rms_disposition(rms_scale_placement),
            f"inverse-RMS scale is applied in the gate/up GEMM {rms_scale_placement.value}",
        ),
        _synthetic_record(mlp_scaled, region.mlp_norm.output, MaterializationDisposition.MATERIALIZE, "gate/up input"),
        _synthetic_record(
            mlp_partials,
            region.mlp_norm.output,
            MaterializationDisposition.PARTIAL_REDUCTION_ONLY,
            "small FP32 row-statistic buffer",
            shape=_rms_partial_shape(region.mlp_norm),
            dtype=region.mlp_norm.reduction_dtype,
        ),
        _record(region.gate_up.output, MaterializationDisposition.EPILOGUE_ONLY, "consumed by pairwise SwiGLU"),
        _record(region.swiglu.output, MaterializationDisposition.MATERIALIZE, "down-projection input"),
        _record(
            region.down_projection.output, MaterializationDisposition.EPILOGUE_ONLY, "consumed by residual epilogue"
        ),
        _record(
            region.mlp_residual.output,
            MaterializationDisposition.MATERIALIZE,
            "saved residual stream for the following block",
        ),
        _record(
            region.next_norm.output,
            _rms_disposition(rms_scale_placement),
            f"inverse-RMS scale is applied in the next QKV GEMM {rms_scale_placement.value}",
        ),
        _synthetic_record(
            next_scaled, region.next_norm.output, MaterializationDisposition.MATERIALIZE, "next QKV input"
        ),
        _synthetic_record(
            next_partials,
            region.next_norm.output,
            MaterializationDisposition.PARTIAL_REDUCTION_ONLY,
            "small FP32 row-statistic buffer",
            shape=_rms_partial_shape(region.next_norm),
            dtype=region.next_norm.reduction_dtype,
        ),
        _record(
            region.next_input_view.output,
            MaterializationDisposition.ALIAS,
            "zero-cost BSH prologue view",
            alias_of=region.next_input_view.input.name,
        ),
        *_qkv_rope_records(region.next_qkv, region.next_rope, "following block boundary"),
    )
    rewrites = (
        _explanation("fuse_initial_qkv_rope", "unrotated Q/K remain in the initial QKV epilogue"),
        *attention_plan.rewrites,
        _explanation(
            "compose_attention_residual_rms_swiglu",
            f"RMS {rms_scale_placement.value} scaling and pairwise SwiGLU share the gate/up GEMM",
        ),
        _explanation(
            "compose_mlp_residual_rms_next_qkv",
            f"RMS {rms_scale_placement.value} scaling and RoPE attach to the following QKV GEMM",
        ),
    )
    return RegionPlan(skeletons=skeletons, materializations=materializations, rewrites=rewrites)


def _qkv_rope_gemm(
    projection: QKVProjectionOp,
    rope: RoPEOp,
    *,
    input_value: str,
    backend: str,
    input_layout: str,
    output_layout: str,
    physical_tile_shape: tuple[int, int, int] | None = None,
    cluster_shape: tuple[int, int, int] | None = None,
    pingpong: bool | None = None,
    prologue: tuple[Attachment, ...] = (),
    epilogue_prefix: tuple[Attachment, ...] = (),
) -> GemmSkeleton:
    batch, sequence, hidden = projection.input.shape
    return GemmSkeleton(
        name=f"{projection.output.name}.qkv_rope",
        input=input_value,
        weight=projection.weight.name,
        output=projection.output.name,
        shape=(batch * sequence, projection.weight.shape[1], hidden),
        accumulation_dtype=projection.accumulation_dtype,
        backend=backend,
        input_layout=input_layout,
        output_layout=output_layout,
        physical_tile_shape=physical_tile_shape,
        cluster_shape=cluster_shape,
        pingpong=pingpong,
        prologue=prologue,
        epilogue=(
            *epilogue_prefix,
            Attachment(
                operation="partition_qkv_segment_views_bshd",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(projection.output.name,),
                outputs=(projection.query.name, projection.key.name, projection.value.name),
            ),
            Attachment(
                operation="pairwise_rope_q",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(projection.query.name, rope.sine.name, rope.cosine.name),
                outputs=(rope.output.name,),
            ),
            Attachment(
                operation="pairwise_rope_k",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(projection.key.name, rope.sine.name, rope.cosine.name),
                outputs=(rope.key_output.name,),
            ),
        ),
    )


def _gate_up_rms_attachments(
    region: _DenseRegion,
    *,
    scaled_input: str,
    inverse_rms: str,
    scale_placement: RMSScalePlacement,
) -> tuple[tuple[Attachment, ...], tuple[Attachment, ...]]:
    scale = Attachment(
        operation="scale_row",
        site=(
            AttachmentSite.GEMM_PROLOGUE
            if scale_placement is RMSScalePlacement.CONSUMER_PROLOGUE
            else AttachmentSite.GEMM_EPILOGUE
        ),
        inputs=(
            scaled_input if scale_placement is RMSScalePlacement.CONSUMER_PROLOGUE else region.gate_up.output.name,
            inverse_rms,
        ),
        outputs=(
            (
                region.mlp_norm.output.name
                if scale_placement is RMSScalePlacement.CONSUMER_PROLOGUE
                else region.gate_up.output.name
            ),
        ),
    )
    swiglu = Attachment(
        operation="pairwise_swiglu",
        site=AttachmentSite.GEMM_EPILOGUE,
        inputs=(region.gate_up.output.name,),
        outputs=(region.swiglu.output.name,),
    )
    if scale_placement is RMSScalePlacement.CONSUMER_PROLOGUE:
        return (scale,), (swiglu,)
    return (), (scale, swiglu)


def _next_qkv_prologue(
    region: _DenseRegion,
    *,
    scaled_input: str,
    inverse_rms: str,
    scale_placement: RMSScalePlacement,
) -> tuple[Attachment, ...]:
    attachments: list[Attachment] = []
    view_input = scaled_input
    if scale_placement is RMSScalePlacement.CONSUMER_PROLOGUE:
        attachments.append(
            Attachment(
                operation="scale_row",
                site=AttachmentSite.GEMM_PROLOGUE,
                inputs=(scaled_input, inverse_rms),
                outputs=(region.next_norm.output.name,),
            )
        )
        view_input = region.next_norm.output.name
    attachments.append(
        Attachment(
            operation="alias_reshape_bsh",
            site=AttachmentSite.GEMM_PROLOGUE,
            inputs=(view_input,),
            outputs=(region.next_input_view.output.name,),
        )
    )
    return tuple(attachments)


def _next_qkv_epilogue_prefix(
    region: _DenseRegion,
    *,
    scaled_input: str,
    inverse_rms: str,
    scale_placement: RMSScalePlacement,
) -> tuple[Attachment, ...]:
    del scaled_input
    if scale_placement is RMSScalePlacement.CONSUMER_PROLOGUE:
        return ()
    return (
        Attachment(
            operation="scale_row",
            site=AttachmentSite.GEMM_EPILOGUE,
            inputs=(region.next_qkv.output.name, inverse_rms),
            outputs=(region.next_qkv.output.name,),
        ),
    )


def _rms_disposition(scale_placement: RMSScalePlacement) -> MaterializationDisposition:
    if scale_placement is RMSScalePlacement.CONSUMER_PROLOGUE:
        return MaterializationDisposition.PROLOGUE_ONLY
    return MaterializationDisposition.EPILOGUE_ONLY


def _prologue_cluster_shape(output_width: int) -> tuple[int, int, int]:
    return (1, 2 if output_width >= 16_384 else 1, 1)


def _qkv_cluster_shape(output_width: int) -> tuple[int, int, int]:
    return (1, 2 if output_width >= 6_144 else 1, 1)


def _residual_rms_gemm(
    projection: LinearOp,
    residual_add: ResidualAddOp,
    norm: RMSNormOp,
    *,
    residual: str,
    scaled_output: str,
    partials: str,
) -> GemmSkeleton:
    return GemmSkeleton(
        name=f"{projection.output.name}.residual_rms_partials",
        input=projection.input.name,
        weight=projection.weight.name,
        output=scaled_output,
        shape=(projection.output.shape[0], projection.output.shape[1], projection.input.shape[1]),
        accumulation_dtype=projection.accumulation_dtype,
        backend="coda_cute_h100",
        input_layout="row_major_mk",
        output_layout="row_major_mn",
        physical_tile_shape=H100_GEMM_TILE_SHAPE,
        cluster_shape=(1, 1, 1),
        pingpong=False,
        epilogue=(
            Attachment(
                operation="residual_add",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(projection.output.name, residual),
                outputs=(residual_add.output.name,),
            ),
            Attachment(
                operation="multiply_gamma",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(residual_add.output.name, norm.gamma.name),
                outputs=(scaled_output,),
            ),
            Attachment(
                operation="partial_sum_square",
                site=AttachmentSite.GEMM_EPILOGUE,
                inputs=(residual_add.output.name,),
                outputs=(partials,),
            ),
        ),
    )


def _rms_reduction(norm: RMSNormOp, partials: str, inverse_rms: str) -> ReductionSkeleton:
    return ReductionSkeleton(
        name=f"{norm.output.name}.combine_rms_partials",
        input=partials,
        output=inverse_rms,
        operator=f"rsqrt(sum / {norm.input.shape[norm.axis]} + {norm.epsilon})",
        reduction_dtype=norm.reduction_dtype,
    )


def _rms_partial_shape(norm: RMSNormOp) -> tuple[int, int]:
    rows, hidden = norm.output.shape
    tile_columns = H100_GEMM_TILE_SHAPE[1]
    return rows, (hidden + tile_columns - 1) // tile_columns


def _qkv_rope_records(
    projection: QKVProjectionOp,
    rope: RoPEOp,
    boundary: str,
) -> tuple[MaterializationRecord, ...]:
    return (
        _record(
            projection.output,
            MaterializationDisposition.MATERIALIZE,
            f"packed QKV storage at the {boundary}",
        ),
        _record(projection.query, MaterializationDisposition.EPILOGUE_ONLY, "unrotated Q stays in the QKV epilogue"),
        _record(projection.key, MaterializationDisposition.EPILOGUE_ONLY, "unrotated K stays in the QKV epilogue"),
        _record(
            rope.output,
            MaterializationDisposition.ALIAS,
            f"rotated Q segment of packed QKV storage at the {boundary}",
            alias_of=projection.output.name,
        ),
        _record(
            rope.key_output,
            MaterializationDisposition.ALIAS,
            f"rotated K segment of packed QKV storage at the {boundary}",
            alias_of=projection.output.name,
        ),
        _record(
            projection.value,
            MaterializationDisposition.ALIAS,
            f"V segment of packed QKV storage at the {boundary}",
            alias_of=projection.output.name,
        ),
    )


def _record(
    value: TensorValue,
    disposition: MaterializationDisposition,
    reason: str,
    *,
    alias_of: str | None = None,
) -> MaterializationRecord:
    return MaterializationRecord(
        value=value.name,
        shape=value.shape,
        dtype=value.dtype,
        disposition=disposition,
        reason=reason,
        alias_of=alias_of,
    )


def _synthetic_record(
    name: str,
    like: TensorValue,
    disposition: MaterializationDisposition,
    reason: str,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: DType | None = None,
) -> MaterializationRecord:
    return MaterializationRecord(
        value=name,
        shape=like.shape if shape is None else shape,
        dtype=like.dtype if dtype is None else dtype,
        disposition=disposition,
        reason=reason,
    )


def _explanation(name: str, benefit: str) -> RewriteExplanation:
    return RewriteExplanation(
        name=name,
        applied=True,
        original_fragment=("ordinary semantic Transformer graph",),
        transformed_fragment=(benefit,),
        semantic_properties=("tile-local attachments", "row-reduction partials", "zero-cost contiguous views"),
        legality_checks=("all moved values have the required single consumers", "finite-precision reorder is permitted"),
        estimated_benefit=benefit,
        numerical_equivalence=NumericalEquivalence.ALGEBRAICALLY_EXACT,
        numerical_effect="attachments may consume FP32 accumulators before BF16 stores and therefore change rounding",
    )
