# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan the dense region exclusively from erased Flow operations."""

from dataclasses import dataclass, replace

from tile_lifetime.compiler import RowScalePlacement
from tile_lifetime.dense_flow import (
    ErasedDenseFlowProgram,
    FlowContract,
    FlowDomainRestriction,
    FlowFold,
    FlowMap,
    FlowMapIteration,
    FlowValue,
    validate_erased_dense_flow,
)
from tile_lifetime.gemm_program import GENERIC_H100_GEMM_BACKEND
from tile_lifetime.plan import (
    Attachment,
    AttachmentSite,
    GemmSkeleton,
    MaterializationDisposition,
    MaterializationRecord,
    NumericalEquivalence,
    ReductionSkeleton,
    RegionPlan,
    RewriteExplanation,
    StreamingAttentionSkeleton,
)
from tile_lifetime.semantic_erasure import validate_plan_semantic_erasure
from tile_lifetime.tensor_program import ScalarExpression, ScalarExpressionKind, serialize_scalar_expression

H100_GEMM_TILE_SHAPE = (128, 256, 64)
GENERATED_STREAMING_BACKEND = "h100_streaming_contract_fold"


@dataclass(frozen=True)
class _ErasedDenseRegion:
    input_view: FlowMap
    initial_projection: FlowContract
    initial_partition: FlowMap
    initial_pair_maps: tuple[FlowMap, FlowMap]
    score_contract: FlowContract
    score_map: FlowMap
    score_domain: FlowDomainRestriction
    maximum_fold: FlowFold
    center_map: FlowMap
    exponential_map: FlowMap
    sum_fold: FlowFold
    value_contract: FlowContract
    normalize_map: FlowMap
    attention_view: FlowMap
    output_contract: FlowContract
    first_add: FlowMap
    first_square: FlowMap
    first_fold: FlowFold
    first_row_finalize: FlowMap
    first_feature_scale: FlowMap
    first_normalized_scale: FlowMap
    gate_contract: FlowContract
    pairwise_map: FlowMap
    down_contract: FlowContract
    second_add: FlowMap
    second_square: FlowMap
    second_fold: FlowFold
    second_row_finalize: FlowMap
    second_feature_scale: FlowMap
    second_normalized_scale: FlowMap
    next_view: FlowMap
    next_projection: FlowContract
    next_partition: FlowMap
    next_pair_maps: tuple[FlowMap, FlowMap]


def compile_erased_dense_transformer_region(
    erased: ErasedDenseFlowProgram,
    *,
    row_scale_placement: RowScalePlacement,
) -> RegionPlan:
    """Generate the eight-skeleton plan after machine-checking semantic erasure."""
    validate_erased_dense_flow(erased)
    region = _recover_erased_region(erased.operations)
    plan = replace(
        _plan_erased_region(region, row_scale_placement=row_scale_placement),
        semantic_erasure_report=erased.report,
    )
    validate_plan_semantic_erasure(plan)
    return plan


def _recover_erased_region(operations) -> _ErasedDenseRegion:
    producer_by_value = {output.name: operation for operation in operations for output in _operation_outputs(operation)}
    consumers_by_value: dict[str, list[object]] = {}
    for operation in operations:
        for input_value in _operation_inputs(operation):
            consumers_by_value.setdefault(input_value.name, []).append(operation)

    score_domain = _one(
        (operation for operation in operations if isinstance(operation, FlowDomainRestriction)),
        "normalized weighted Fold domain restriction",
    )
    score_map = _producer(producer_by_value, score_domain.input, FlowMap, "score Map")
    score_contract = _producer(producer_by_value, score_map.inputs[0], FlowContract, "score Contract")
    maximum_fold = _one(
        (
            operation
            for operation in consumers_by_value.get(score_domain.output.name, ())
            if isinstance(operation, FlowFold) and operation.reducer.value == "maximum"
        ),
        "maximum Fold",
    )
    center_map = _one(
        (
            operation
            for operation in consumers_by_value.get(score_domain.output.name, ())
            if isinstance(operation, FlowMap) and maximum_fold.output in operation.inputs
        ),
        "center Map",
    )
    exponential_map = _one(
        (
            operation
            for operation in consumers_by_value.get(center_map.outputs[0].name, ())
            if isinstance(operation, FlowMap)
            and any(expression.kind is ScalarExpressionKind.EXP for expression in operation.expressions)
        ),
        "exponential Map",
    )
    sum_fold = _one(
        (
            operation
            for operation in consumers_by_value.get(exponential_map.outputs[0].name, ())
            if isinstance(operation, FlowFold) and operation.reducer.value == "sum"
        ),
        "sum Fold",
    )
    value_contract = _one(
        (
            operation
            for operation in consumers_by_value.get(exponential_map.outputs[0].name, ())
            if isinstance(operation, FlowContract)
        ),
        "weighted-value Contract",
    )
    normalize_map = _one(
        (
            operation
            for operation in consumers_by_value.get(value_contract.output.name, ())
            if isinstance(operation, FlowMap) and sum_fold.output in operation.inputs
        ),
        "normalized weighted Fold finalization",
    )

    initial_pair_q = _producer(producer_by_value, score_contract.inputs[0], FlowMap, "query pair Map")
    initial_pair_k = _producer(producer_by_value, score_contract.inputs[1], FlowMap, "key pair Map")
    if any(operation.iteration is not FlowMapIteration.ADJACENT_PAIR for operation in (initial_pair_q, initial_pair_k)):
        raise ValueError("score Contract inputs are not produced by adjacent-pair Maps")
    query_partition_output = _one(
        (
            input_value
            for input_value in initial_pair_q.inputs
            if isinstance(producer_by_value.get(input_value.name), FlowMap)
            and producer_by_value[input_value.name].iteration is FlowMapIteration.PARTITION
        ),
        "query projection partition output",
    )
    key_partition_output = _one(
        (
            input_value
            for input_value in initial_pair_k.inputs
            if isinstance(producer_by_value.get(input_value.name), FlowMap)
            and producer_by_value[input_value.name].iteration is FlowMapIteration.PARTITION
        ),
        "key projection partition output",
    )
    initial_partition = _producer(producer_by_value, query_partition_output, FlowMap, "initial partition Map")
    if producer_by_value.get(key_partition_output.name) is not initial_partition:
        raise ValueError("query and key pair Maps do not share one projection partition")
    initial_projection = _producer(
        producer_by_value,
        initial_partition.inputs[0],
        FlowContract,
        "initial projection Contract",
    )
    input_view = _producer(producer_by_value, initial_projection.inputs[0], FlowMap, "initial input view")

    attention_view = _one(
        (
            operation
            for operation in consumers_by_value.get(normalize_map.outputs[0].name, ())
            if isinstance(operation, FlowMap) and operation.iteration is FlowMapIteration.VIEW
        ),
        "attention output view",
    )
    output_contract = _one(
        (
            operation
            for operation in consumers_by_value.get(attention_view.outputs[0].name, ())
            if isinstance(operation, FlowContract)
        ),
        "attention output Contract",
    )
    first = _recover_fold_scale_contract(output_contract, producer_by_value, consumers_by_value)
    gate_contract = _consumer_contract(first.normalized_scale.outputs[0], consumers_by_value, "first scaled Contract")
    pairwise_map = _one(
        (
            operation
            for operation in consumers_by_value.get(gate_contract.output.name, ())
            if isinstance(operation, FlowMap) and operation.iteration is FlowMapIteration.ADJACENT_PAIR
        ),
        "expanded projection pairwise Map",
    )
    down_contract = _consumer_contract(pairwise_map.outputs[0], consumers_by_value, "pairwise consumer Contract")
    second = _recover_fold_scale_contract(down_contract, producer_by_value, consumers_by_value)
    next_view = _one(
        (
            operation
            for operation in consumers_by_value.get(second.normalized_scale.outputs[0].name, ())
            if isinstance(operation, FlowMap) and operation.iteration is FlowMapIteration.VIEW
        ),
        "following projection input view",
    )
    next_projection = _consumer_contract(next_view.outputs[0], consumers_by_value, "following projection Contract")
    next_partition = _one(
        (
            operation
            for operation in consumers_by_value.get(next_projection.output.name, ())
            if isinstance(operation, FlowMap) and operation.iteration is FlowMapIteration.PARTITION
        ),
        "following projection partition",
    )
    next_pairs = tuple(
        operation
        for output in next_partition.outputs
        for operation in consumers_by_value.get(output.name, ())
        if isinstance(operation, FlowMap) and operation.iteration is FlowMapIteration.ADJACENT_PAIR
    )
    if len(next_pairs) != 2:
        raise ValueError(f"following partition must feed two adjacent-pair Maps, found {len(next_pairs)}")
    _validate_generic_dataflow(
        initial_projection,
        initial_partition,
        (initial_pair_q, initial_pair_k),
        score_contract,
        score_map,
        score_domain,
        maximum_fold,
        exponential_map,
        sum_fold,
        value_contract,
        normalize_map,
        pairwise_map,
        next_projection,
        next_partition,
        next_pairs,
    )
    return _ErasedDenseRegion(
        input_view,
        initial_projection,
        initial_partition,
        (initial_pair_q, initial_pair_k),
        score_contract,
        score_map,
        score_domain,
        maximum_fold,
        center_map,
        exponential_map,
        sum_fold,
        value_contract,
        normalize_map,
        attention_view,
        output_contract,
        first.add,
        first.square,
        first.fold,
        first.row_finalize,
        first.feature_scale,
        first.normalized_scale,
        gate_contract,
        pairwise_map,
        down_contract,
        second.add,
        second.square,
        second.fold,
        second.row_finalize,
        second.feature_scale,
        second.normalized_scale,
        next_view,
        next_projection,
        next_partition,
        next_pairs,
    )


@dataclass(frozen=True)
class _FlowFoldScaleContract:
    add: FlowMap
    square: FlowMap
    fold: FlowFold
    row_finalize: FlowMap
    feature_scale: FlowMap
    normalized_scale: FlowMap


def _recover_fold_scale_contract(
    producer_contract: FlowContract,
    producer_by_value: dict[str, object],
    consumers_by_value: dict[str, list[object]],
) -> _FlowFoldScaleContract:
    add = _one(
        (
            operation
            for operation in consumers_by_value.get(producer_contract.output.name, ())
            if isinstance(operation, FlowMap) and _single_expression_kind(operation, ScalarExpressionKind.ADD)
        ),
        "tile-local add after Contract",
    )
    square = _one(
        (
            operation
            for operation in consumers_by_value.get(add.outputs[0].name, ())
            if isinstance(operation, FlowMap) and _is_square_map(operation)
        ),
        "pointwise square before Fold",
    )
    fold = _one(
        (
            operation
            for operation in consumers_by_value.get(square.outputs[0].name, ())
            if isinstance(operation, FlowFold) and operation.reducer.value == "sum"
        ),
        "sum Fold after square",
    )
    row_finalize = _one(
        (operation for operation in consumers_by_value.get(fold.output.name, ()) if isinstance(operation, FlowMap)),
        "row-scalar Fold finalization",
    )
    feature_scale = _one(
        (
            operation
            for operation in consumers_by_value.get(add.outputs[0].name, ())
            if isinstance(operation, FlowMap)
            and operation is not square
            and _single_expression_kind(operation, ScalarExpressionKind.MULTIPLY)
        ),
        "feature-scale Map",
    )
    normalized_scale = _one(
        (
            operation
            for operation in consumers_by_value.get(feature_scale.outputs[0].name, ())
            if isinstance(operation, FlowMap) and row_finalize.outputs[0] in operation.inputs
        ),
        "row-scale Map",
    )
    if producer_by_value.get(normalized_scale.outputs[0].name) is not normalized_scale:
        raise ValueError("row-scale Map output is not present in generic Flow dataflow")
    return _FlowFoldScaleContract(add, square, fold, row_finalize, feature_scale, normalized_scale)


def _operation_inputs(operation) -> tuple[FlowValue, ...]:
    if isinstance(operation, FlowFold):
        return (operation.input,)
    if isinstance(operation, FlowDomainRestriction):
        return (operation.input,)
    return operation.inputs


def _operation_outputs(operation) -> tuple[FlowValue, ...]:
    if isinstance(operation, FlowContract):
        return (operation.output,)
    if isinstance(operation, FlowFold):
        return (operation.output,)
    if isinstance(operation, FlowDomainRestriction):
        return (operation.output,)
    return operation.outputs


def _producer(producers: dict[str, object], value: FlowValue, expected_type, description: str):
    operation = producers.get(value.name)
    if not isinstance(operation, expected_type):
        actual = "none" if operation is None else type(operation).__name__
        raise ValueError(f"{description} producer is {actual}, expected {expected_type.__name__}")
    return operation


def _one(candidates, description: str):
    matches = tuple(candidates)
    if len(matches) != 1:
        raise ValueError(f"expected one {description}, found {len(matches)}")
    return matches[0]


def _consumer_contract(
    value: FlowValue,
    consumers_by_value: dict[str, list[object]],
    description: str,
) -> FlowContract:
    return _one(
        (operation for operation in consumers_by_value.get(value.name, ()) if isinstance(operation, FlowContract)),
        description,
    )


def _single_expression_kind(operation: FlowMap, kind: ScalarExpressionKind) -> bool:
    return len(operation.expressions) == 1 and operation.expressions[0].kind is kind


def _is_square_map(operation: FlowMap) -> bool:
    if not _single_expression_kind(operation, ScalarExpressionKind.MULTIPLY):
        return False
    expression = operation.expressions[0]
    return len(expression.operands) == 2 and expression.operands[0] == expression.operands[1]


def _validate_generic_dataflow(
    initial_projection: FlowContract,
    initial_partition: FlowMap,
    initial_pairs: tuple[FlowMap, FlowMap],
    score_contract: FlowContract,
    score_map: FlowMap,
    score_domain: FlowDomainRestriction,
    maximum_fold: FlowFold,
    exponential_map: FlowMap,
    sum_fold: FlowFold,
    value_contract: FlowContract,
    normalize_map: FlowMap,
    pairwise_map: FlowMap,
    next_projection: FlowContract,
    next_partition: FlowMap,
    next_pairs: tuple[FlowMap, FlowMap],
) -> None:
    if initial_partition.iteration is not FlowMapIteration.PARTITION:
        raise ValueError("initial contraction is not followed by a generic partition Map")
    if initial_partition.inputs != (initial_projection.output,):
        raise ValueError("initial partition does not consume the initial Contract")
    if any(operation.iteration is not FlowMapIteration.ADJACENT_PAIR for operation in initial_pairs):
        raise ValueError("initial pair transforms are not generic adjacent-pair Maps")
    if score_contract.inputs[:1] != (initial_pairs[0].outputs[0],):
        raise ValueError("score Contract does not consume the first pairwise output")
    if score_map.inputs != (score_contract.output,) or score_domain.input != score_map.outputs[0]:
        raise ValueError("score Map and DomainRestriction are disconnected")
    if maximum_fold.input != score_domain.output or maximum_fold.reducer.value != "maximum":
        raise ValueError("normalized exponential does not begin with a maximum Fold")
    if sum_fold.input != exponential_map.outputs[0] or sum_fold.reducer.value != "sum":
        raise ValueError("normalized exponential does not contain a sum Fold")
    if value_contract.inputs[0] != exponential_map.outputs[0]:
        raise ValueError("value Contract does not consume normalized-exponential contributions")
    if normalize_map.inputs != (value_contract.output, sum_fold.output):
        raise ValueError("normalized weighted Fold does not finalize by its sum state")
    if pairwise_map.iteration is not FlowMapIteration.ADJACENT_PAIR or len(pairwise_map.expressions) != 1:
        raise ValueError("expanded projection is not finalized by one generic adjacent-pair Map")
    if next_partition.inputs != (next_projection.output,) or next_partition.iteration is not FlowMapIteration.PARTITION:
        raise ValueError("following projection does not feed a generic partition Map")
    if any(operation.iteration is not FlowMapIteration.ADJACENT_PAIR for operation in next_pairs):
        raise ValueError("following pair transforms are not generic adjacent-pair Maps")


def _plan_erased_region(region: _ErasedDenseRegion, *, row_scale_placement: RowScalePlacement) -> RegionPlan:
    initial_projection = _projection_pairwise_linear_contract(
        region.initial_projection,
        region.initial_partition,
        region.initial_pair_maps,
        input_value=region.initial_projection.inputs[0].name,
        input_layout="bsh_contiguous",
        cluster_shape=_projection_cluster_shape(region.initial_projection.output.shape[-1]),
    )
    streaming = _streaming_skeleton(region)

    first_scaled = region.first_feature_scale.outputs[0]
    first_partials = f"{region.first_square.inputs[0].name}.fold_partials"
    first_row_scalar = region.first_row_finalize.outputs[0]
    output_contract = _contract_maps_fold(
        region.output_contract,
        region.first_add,
        region.first_feature_scale,
        region.first_square,
        partials=first_partials,
    )
    first_reduction = _fold_reduction(region.first_fold, region.first_row_finalize, first_partials)
    gate_contract = GemmSkeleton(
        name=f"contract_{row_scale_placement.value}_row_scale_pairwise_map",
        input=first_scaled.name,
        weight=region.gate_contract.inputs[1].name,
        output=region.pairwise_map.outputs[0].name,
        shape=_gemm_shape(region.gate_contract),
        accumulation_dtype=region.gate_contract.accumulation_dtype,
        backend=GENERIC_H100_GEMM_BACKEND,
        input_layout="row_major_mk",
        output_layout="row_major_mn",
        physical_tile_shape=H100_GEMM_TILE_SHAPE,
        cluster_shape=_prologue_cluster_shape(region.gate_contract.output.shape[1]),
        pingpong=False,
        prologue=_row_scale_preparation(
            first_scaled,
            first_row_scalar,
            region.first_normalized_scale.outputs[0],
            row_scale_placement,
        ),
        epilogue=(
            *_row_scale_finalization(region.gate_contract.output, first_row_scalar, row_scale_placement),
            Attachment(
                "pairwise_map",
                AttachmentSite.GEMM_EPILOGUE,
                (region.gate_contract.output.name,),
                (region.pairwise_map.outputs[0].name,),
                (("expression_ast", serialize_scalar_expression(region.pairwise_map.expressions[0])),),
            ),
        ),
    )

    second_scaled = region.second_feature_scale.outputs[0]
    second_partials = f"{region.second_square.inputs[0].name}.fold_partials"
    second_row_scalar = region.second_row_finalize.outputs[0]
    down_contract = _contract_maps_fold(
        region.down_contract,
        region.second_add,
        region.second_feature_scale,
        region.second_square,
        partials=second_partials,
    )
    second_reduction = _fold_reduction(region.second_fold, region.second_row_finalize, second_partials)
    next_contract = _projection_pairwise_linear_contract(
        region.next_projection,
        region.next_partition,
        region.next_pair_maps,
        input_value=second_scaled.name,
        input_layout="row_major_mk",
        cluster_shape=_projection_cluster_shape(region.next_projection.output.shape[-1]),
        prologue=(
            *_row_scale_preparation(
                second_scaled,
                second_row_scalar,
                region.second_normalized_scale.outputs[0],
                row_scale_placement,
            ),
            Attachment(
                "view",
                AttachmentSite.GEMM_PROLOGUE,
                (
                    (
                        region.second_normalized_scale.outputs[0].name
                        if row_scale_placement is RowScalePlacement.CONSUMER_PROLOGUE
                        else second_scaled.name
                    ),
                ),
                (region.next_view.outputs[0].name,),
            ),
        ),
        epilogue_prefix=_row_scale_finalization(
            region.next_projection.output,
            second_row_scalar,
            row_scale_placement,
        ),
    )

    skeletons = (
        initial_projection,
        streaming,
        output_contract,
        first_reduction,
        gate_contract,
        down_contract,
        second_reduction,
        next_contract,
    )
    materializations = _materializations(
        region,
        first_scaled,
        first_partials,
        second_scaled,
        second_partials,
        row_scale_placement,
    )
    rewrites = (
        _explanation("attach_partition_and_pairwise_linear_maps"),
        _explanation("derive_streaming_normalized_weighted_fold"),
        _explanation("place_maps_and_partial_folds_around_contracts"),
    )
    return RegionPlan(skeletons, materializations, rewrites)


def _projection_pairwise_linear_contract(
    contract: FlowContract,
    partition: FlowMap,
    pair_maps: tuple[FlowMap, FlowMap],
    *,
    input_value: str,
    input_layout: str,
    cluster_shape: tuple[int, int, int],
    prologue: tuple[Attachment, ...] = (),
    epilogue_prefix: tuple[Attachment, ...] = (),
) -> GemmSkeleton:
    batch, sequence, hidden = contract.inputs[0].shape
    pair_attachments = tuple(
        Attachment(
            "pairwise_linear_map",
            AttachmentSite.GEMM_EPILOGUE,
            tuple(value.name for value in operation.inputs),
            tuple(value.name for value in operation.outputs),
            tuple(
                (f"expression_ast.{index}", serialize_scalar_expression(expression))
                for index, expression in enumerate(operation.expressions)
            ),
        )
        for operation in pair_maps
    )
    return GemmSkeleton(
        name="contract_partition_pairwise_linear_maps",
        input=input_value,
        weight=contract.inputs[1].name,
        output=contract.output.name,
        shape=(batch * sequence, contract.inputs[1].shape[1], hidden),
        accumulation_dtype=contract.accumulation_dtype,
        backend=GENERIC_H100_GEMM_BACKEND,
        input_layout=input_layout,
        output_layout="fa3_bshd_last_dimension_contiguous",
        physical_tile_shape=H100_GEMM_TILE_SHAPE,
        cluster_shape=cluster_shape,
        pingpong=False,
        prologue=prologue,
        epilogue=(
            *epilogue_prefix,
            Attachment(
                "partition",
                AttachmentSite.GEMM_EPILOGUE,
                (contract.output.name,),
                tuple(value.name for value in partition.outputs),
                partition.attributes,
            ),
            *pair_attachments,
        ),
    )


def _streaming_skeleton(region: _ErasedDenseRegion) -> StreamingAttentionSkeleton:
    query, key = region.score_contract.inputs
    value = region.value_contract.inputs[1]
    output = region.normalize_map.outputs[0]
    scale = _score_scale(region.score_map.expressions[0], region.score_contract.output.name)
    causal = region.score_domain.predicate.kind is ScalarExpressionKind.LESS_EQUAL
    head_dimension = query.shape[-1]
    query_block_size = 192 if head_dimension == 64 else 128
    key_value_block_size = 128 if causal else (192 if head_dimension == 64 else 176)
    query_heads = query.shape[2]
    key_value_heads = key.shape[2]
    return StreamingAttentionSkeleton(
        name="streaming_normalized_weighted_fold",
        query=query.name,
        key=key.name,
        value=value.name,
        output=output.name,
        score_value=region.score_domain.output.name,
        probability_value=region.exponential_map.outputs[0].name,
        query_block_size=query_block_size,
        key_value_block_size=key_value_block_size,
        head_dimension=head_dimension,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        causal=causal,
        scale=scale,
        backend=GENERATED_STREAMING_BACKEND,
        input_layout="fa3_bshd_last_dimension_contiguous",
        output_layout="bshd_contiguous",
        pipeline_stages=2,
        producer_threads=32,
        consumer_threads=(query_block_size // 64) * 128,
        pack_gqa=query_heads != key_value_heads,
        mma_pv_is_rs=causal,
        intra_warpgroup_overlap=True,
        persistent_scheduler=True,
        register_estimate=168 if head_dimension == 128 else None,
        online_state=(
            region.maximum_fold.output.name,
            region.sum_fold.output.name,
            region.value_contract.output.name,
        ),
        attachments=(
            Attachment(
                "score_map",
                AttachmentSite.ATTENTION_SCORE_TRANSFORM,
                (region.score_contract.output.name,),
                (region.score_map.outputs[0].name,),
            ),
            Attachment(
                "domain_restriction",
                AttachmentSite.ATTENTION_SCORE_TRANSFORM,
                (region.score_map.outputs[0].name,),
                (region.score_domain.output.name,),
            ),
            Attachment(
                "online_fold_update",
                AttachmentSite.ATTENTION_ONLINE_UPDATE,
                (region.score_domain.output.name, value.name),
                (
                    region.maximum_fold.output.name,
                    region.sum_fold.output.name,
                    region.value_contract.output.name,
                ),
            ),
            Attachment(
                "fold_finalize",
                AttachmentSite.ATTENTION_OUTPUT_TRANSFORM,
                (region.value_contract.output.name, region.sum_fold.output.name),
                (output.name,),
            ),
        ),
    )


def _contract_maps_fold(
    contract: FlowContract,
    add: FlowMap,
    feature_scale: FlowMap,
    square: FlowMap,
    *,
    partials: str,
) -> GemmSkeleton:
    contract_output = contract.output
    forwarded = next(value for value in add.inputs if value != contract_output)
    feature_vector = next(value for value in feature_scale.inputs if value != square.inputs[0])
    return GemmSkeleton(
        name="contract_maps_and_fold_partials",
        input=contract.inputs[0].name,
        weight=contract.inputs[1].name,
        output=feature_scale.outputs[0].name,
        shape=_gemm_shape(contract),
        accumulation_dtype=contract.accumulation_dtype,
        backend=GENERIC_H100_GEMM_BACKEND,
        input_layout="row_major_mk",
        output_layout="row_major_mn",
        physical_tile_shape=H100_GEMM_TILE_SHAPE,
        cluster_shape=(1, 1, 1),
        pingpong=False,
        epilogue=(
            Attachment(
                "add",
                AttachmentSite.GEMM_EPILOGUE,
                (contract_output.name, forwarded.name),
                (add.outputs[0].name,),
            ),
            Attachment(
                "multiply",
                AttachmentSite.GEMM_EPILOGUE,
                (add.outputs[0].name, feature_vector.name),
                (feature_scale.outputs[0].name,),
                (("input.1_delivery", "row"),),
            ),
            Attachment(
                "partial_sum_square",
                AttachmentSite.GEMM_EPILOGUE,
                (square.inputs[0].name,),
                (partials,),
            ),
        ),
    )


def _fold_reduction(fold: FlowFold, finalize: FlowMap, partials: str) -> ReductionSkeleton:
    return ReductionSkeleton(
        name="combine_fold_partials",
        input=partials,
        output=finalize.outputs[0].name,
        operator=_render_scalar_expression(finalize.expressions[0], {fold.output.name: "sum"}),
        reduction_dtype=fold.accumulation_dtype,
    )


def _row_scale_preparation(
    input_value: FlowValue,
    row_scalar: FlowValue,
    output_value: FlowValue,
    placement: RowScalePlacement,
) -> tuple[Attachment, ...]:
    if placement is not RowScalePlacement.CONSUMER_PROLOGUE:
        return ()
    return (
        Attachment(
            "scale_row",
            AttachmentSite.GEMM_PROLOGUE,
            (input_value.name, row_scalar.name),
            (output_value.name,),
        ),
    )


def _row_scale_finalization(
    contract_output: FlowValue,
    row_scalar: FlowValue,
    placement: RowScalePlacement,
) -> tuple[Attachment, ...]:
    if placement is RowScalePlacement.CONSUMER_PROLOGUE:
        return ()
    return (
        Attachment(
            "scale_row",
            AttachmentSite.GEMM_EPILOGUE,
            (contract_output.name, row_scalar.name),
            (contract_output.name,),
        ),
    )


def _materializations(
    region: _ErasedDenseRegion,
    first_scaled: FlowValue,
    first_partials: str,
    second_scaled: FlowValue,
    second_partials: str,
    placement: RowScalePlacement,
) -> tuple[MaterializationRecord, ...]:
    output: list[MaterializationRecord] = [
        _record(
            region.input_view.outputs[0],
            MaterializationDisposition.ALIAS,
            alias_of=region.input_view.inputs[0].name,
        ),
        *_projection_records(region.initial_projection, region.initial_partition, region.initial_pair_maps),
        *_streaming_records(region),
        _record(
            region.attention_view.outputs[0],
            MaterializationDisposition.ALIAS,
            alias_of=region.attention_view.inputs[0].name,
        ),
        _record(region.output_contract.output, MaterializationDisposition.EPILOGUE_ONLY),
        _record(region.first_add.outputs[0], MaterializationDisposition.MATERIALIZE),
        _record(region.first_normalized_scale.outputs[0], _scale_disposition(placement)),
        _record(first_scaled, MaterializationDisposition.MATERIALIZE),
        MaterializationRecord(
            first_partials,
            _partial_shape(region.first_square.inputs[0]),
            region.first_fold.accumulation_dtype,
            MaterializationDisposition.PARTIAL_REDUCTION_ONLY,
            "partial Fold buffer",
        ),
        _record(region.gate_contract.output, MaterializationDisposition.EPILOGUE_ONLY),
        _record(region.pairwise_map.outputs[0], MaterializationDisposition.MATERIALIZE),
        _record(region.down_contract.output, MaterializationDisposition.EPILOGUE_ONLY),
        _record(region.second_add.outputs[0], MaterializationDisposition.MATERIALIZE),
        _record(region.second_normalized_scale.outputs[0], _scale_disposition(placement)),
        _record(second_scaled, MaterializationDisposition.MATERIALIZE),
        MaterializationRecord(
            second_partials,
            _partial_shape(region.second_square.inputs[0]),
            region.second_fold.accumulation_dtype,
            MaterializationDisposition.PARTIAL_REDUCTION_ONLY,
            "partial Fold buffer",
        ),
        _record(region.next_view.outputs[0], MaterializationDisposition.ALIAS, alias_of=region.next_view.inputs[0].name),
        *_projection_records(region.next_projection, region.next_partition, region.next_pair_maps),
    ]
    return tuple(output)


def _projection_records(
    contract: FlowContract,
    partition: FlowMap,
    pair_maps: tuple[FlowMap, FlowMap],
) -> tuple[MaterializationRecord, ...]:
    return (
        _record(contract.output, MaterializationDisposition.MATERIALIZE),
        *(_record(value, MaterializationDisposition.EPILOGUE_ONLY) for value in partition.outputs[:2]),
        _record(
            pair_maps[0].outputs[0],
            MaterializationDisposition.ALIAS,
            alias_of=contract.output.name,
        ),
        _record(
            pair_maps[1].outputs[0],
            MaterializationDisposition.ALIAS,
            alias_of=contract.output.name,
        ),
        _record(partition.outputs[2], MaterializationDisposition.ALIAS, alias_of=contract.output.name),
    )


def _streaming_records(region: _ErasedDenseRegion) -> tuple[MaterializationRecord, ...]:
    return (
        _record(region.score_domain.output, MaterializationDisposition.INTERNAL_ATTENTION_STATE),
        _record(region.exponential_map.outputs[0], MaterializationDisposition.INTERNAL_ATTENTION_STATE),
        _record(region.maximum_fold.output, MaterializationDisposition.INTERNAL_ATTENTION_STATE),
        _record(region.sum_fold.output, MaterializationDisposition.INTERNAL_ATTENTION_STATE),
        _record(region.value_contract.output, MaterializationDisposition.INTERNAL_ATTENTION_STATE),
        _record(region.normalize_map.outputs[0], MaterializationDisposition.MATERIALIZE),
    )


def _record(
    value: FlowValue,
    disposition: MaterializationDisposition,
    *,
    alias_of: str | None = None,
) -> MaterializationRecord:
    return MaterializationRecord(
        value.name,
        value.shape,
        value.dtype,
        disposition,
        "derived from generic Flow",
        alias_of,
    )


def _scale_disposition(placement: RowScalePlacement) -> MaterializationDisposition:
    if placement is RowScalePlacement.CONSUMER_PROLOGUE:
        return MaterializationDisposition.PROLOGUE_ONLY
    return MaterializationDisposition.EPILOGUE_ONLY


def _partial_shape(value: FlowValue) -> tuple[int, int]:
    return value.shape[0], (value.shape[1] + H100_GEMM_TILE_SHAPE[1] - 1) // H100_GEMM_TILE_SHAPE[1]


def _gemm_shape(contract: FlowContract) -> tuple[int, int, int]:
    return contract.output.shape[0], contract.output.shape[1], contract.inputs[0].shape[1]


def _projection_cluster_shape(output_width: int) -> tuple[int, int, int]:
    return (1, 2 if output_width >= 6_144 else 1, 1)


def _prologue_cluster_shape(output_width: int) -> tuple[int, int, int]:
    return (1, 2 if output_width >= 16_384 else 1, 1)


def _score_scale(expression: ScalarExpression, score_name: str) -> float:
    if expression.kind is not ScalarExpressionKind.MULTIPLY:
        raise ValueError("score Map must multiply the score Contract by a scalar")
    left, right = expression.operands
    literal = right if left.input_name == score_name else left
    if literal.kind is not ScalarExpressionKind.CONSTANT or not isinstance(literal.constant, float):
        raise ValueError("score Map scale is not a floating-point literal")
    return literal.constant


def _render_scalar_expression(
    expression: ScalarExpression,
    aliases: dict[str, str],
    parent_precedence: int = 0,
) -> str:
    if expression.kind is ScalarExpressionKind.INPUT:
        assert expression.input_name is not None
        return aliases.get(expression.input_name, expression.input_name)
    if expression.kind is ScalarExpressionKind.CONSTANT:
        return str(expression.constant)
    if expression.kind is ScalarExpressionKind.RSQRT:
        return f"rsqrt({_render_scalar_expression(expression.operands[0], aliases)})"
    operators = {
        ScalarExpressionKind.ADD: ("+", 10),
        ScalarExpressionKind.SUBTRACT: ("-", 10),
        ScalarExpressionKind.MULTIPLY: ("*", 20),
        ScalarExpressionKind.DIVIDE: ("/", 20),
    }
    operator = operators.get(expression.kind)
    if operator is None:
        raise ValueError(f"unsupported row-finalization expression {expression.kind.value}")
    symbol, precedence = operator
    left = _render_scalar_expression(expression.operands[0], aliases, precedence)
    right = _render_scalar_expression(expression.operands[1], aliases, precedence + 1)
    rendered = f"{left} {symbol} {right}"
    return f"({rendered})" if precedence < parent_precedence else rendered


def _explanation(name: str) -> RewriteExplanation:
    return RewriteExplanation(
        name,
        True,
        ("generic Flow",),
        ("bounded tile-lifetime skeletons",),
        ("Map/Contract/Fold/DomainRestriction dataflow",),
        ("machine-checked semantic-name erasure",),
        "avoid activation-sized intermediates",
        NumericalEquivalence.ALGEBRAICALLY_EXACT,
        "finite-precision order is controlled by the selected row-scale policy",
    )
