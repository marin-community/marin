# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""First-principles lowering of a semantic MoE region into generic EP stages."""

from dataclasses import dataclass, field
from math import ceil, isfinite

from tile_lifetime.expert_parallel_plan import (
    BufferLifetime,
    ExchangeRelationProjection,
    ExchangeRowMode,
    ExpertCapacityPolicy,
    ExpertMaterializationSchedule,
    ExpertOverlapPolicy,
    ExpertOwnership,
    ExpertParallelPlan,
    ExpertParallelSchedule,
    ExpertParallelStage,
    ExpertParallelStageKind,
    ExpertSegmentContract,
    GateUpLayoutContract,
    GateUpPhysicalLayout,
    MapFoldSemantics,
    PipelineDepth,
    ReadinessGranularity,
    RouteRelation,
    TileFlowEdge,
    TileStorage,
    TransportSelection,
    TransportSemantics,
    WorkerPool,
)
from tile_lifetime.ir import (
    DType,
    LinearOp,
    RoutedExpertMLPOp,
    SharedExpertMLPOp,
    TensorGraph,
    TopKRouterOp,
    WeightedExpertCombineOp,
)
from tile_lifetime.plan import (
    MaterializationDisposition,
    MaterializationRecord,
    NumericalEquivalence,
    NumericalPolicy,
    RewriteExplanation,
)
from tile_lifetime.relation import compile_ordered_relation_fold
from tile_lifetime.tensor_program import (
    ScalarExpressionKind,
    scalar_binary,
    scalar_constant,
    scalar_input,
    scalar_unary,
)


@dataclass(frozen=True)
class ExpertParallelConfig:
    """Bounded generic schedule choices for expert-parallel execution."""

    expert_parallel_size: int
    segment_padding: int = 256
    receiver_capacity_factor: float = 1.25
    contraction_tile: tuple[int, int, int] = (256, 256, 64)
    swiglu_tile: tuple[int, int] = (128, 128)
    exchange_tile: tuple[int, int] = (128, 512)
    scatter_tile: tuple[int, int] = (16, 1024)
    exchange_workers: int = 56
    exchange_worker_candidates: tuple[int, ...] = (12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 80, 96)
    contraction_producer_workers: int = 1
    contraction_consumer_workers: int = 1
    transform_workers: int = 1
    minibatch_size: int = 2048
    macrobatch_size: int = 65_536
    contraction_pipeline_depth: int = 6
    transform_pipeline_depth: int = 3
    exchange_pipeline_depth: int = 4
    exchange_implementation: str = "deepep"
    exchange_implementation_candidates: tuple[str, ...] = ("deepep", "ragged_all_to_all")
    forward_transport: TransportSelection = field(
        default_factory=lambda: TransportSelection(
            "deepep_dispatch",
            TransportSemantics.PAYLOAD_PERMUTATION,
        )
    )
    reverse_transport: TransportSelection = field(
        default_factory=lambda: TransportSelection(
            "all_to_all_single",
            TransportSemantics.PAYLOAD_PERMUTATION,
        )
    )
    merge_implementation: str = "generated_source_ordered_fold"
    segmented_contraction_implementation: str = "standalone_sm100_grouped_gemm"
    segmented_contraction_candidates: tuple[str, ...] = ("standalone_sm100_grouped_gemm", "ragged_dot")
    exchange_row_mode: ExchangeRowMode = ExchangeRowMode.COALESCED_TOKEN_OWNER
    gate_up_layout: GateUpPhysicalLayout = GateUpPhysicalLayout.CONCATENATED_E_2I_K
    gate_up_layout_candidates: tuple[GateUpPhysicalLayout, ...] = (
        GateUpPhysicalLayout.CONCATENATED_E_2I_K,
        GateUpPhysicalLayout.INTERLEAVED_E_2I_K,
        GateUpPhysicalLayout.SEPARATE_E_I_K,
    )
    overlap_policy: ExpertOverlapPolicy = ExpertOverlapPolicy.SHARED_WITH_ASYNC_DISPATCH
    overlap_policy_candidates: tuple[ExpertOverlapPolicy, ...] = (
        ExpertOverlapPolicy.SHARED_WITH_ASYNC_DISPATCH,
        ExpertOverlapPolicy.SEQUENTIAL,
    )
    materialization_schedule: ExpertMaterializationSchedule = ExpertMaterializationSchedule.TILE_FLOW_BOUNDARIES
    materialization_schedule_candidates: tuple[ExpertMaterializationSchedule, ...] = (
        ExpertMaterializationSchedule.TILE_FLOW_BOUNDARIES,
        ExpertMaterializationSchedule.COARSE_ACTIVATION_BOUNDARIES,
    )


class ExpertParallelLegalityError(ValueError):
    """Structured rejection of semantic or schedule conditions."""

    def __init__(self, reasons: tuple[str, ...]):
        self.reasons = reasons
        super().__init__("; ".join(reasons))


@dataclass(frozen=True)
class _Region:
    router_projection: LinearOp
    router: TopKRouterOp
    shared: SharedExpertMLPOp
    routed: RoutedExpertMLPOp
    combine: WeightedExpertCombineOp


def compile_expert_parallel_region(
    graph: TensorGraph,
    *,
    config: ExpertParallelConfig,
    numerical_policy: NumericalPolicy,
) -> ExpertParallelPlan:
    """Lower a bounded semantic MoE region without selecting a fused megakernel."""
    region = _recover_region(graph)
    reasons = _legality_reasons(region, config=config, numerical_policy=numerical_policy)
    if reasons:
        raise ExpertParallelLegalityError(reasons)
    return _build_plan(region, config=config)


def _recover_region(graph: TensorGraph) -> _Region:
    routers = tuple(operation for operation in graph.operations if isinstance(operation, TopKRouterOp))
    shared = tuple(operation for operation in graph.operations if isinstance(operation, SharedExpertMLPOp))
    routed = tuple(operation for operation in graph.operations if isinstance(operation, RoutedExpertMLPOp))
    combines = tuple(operation for operation in graph.operations if isinstance(operation, WeightedExpertCombineOp))
    reasons: list[str] = []
    if len(routers) != 1:
        reasons.append(f"expected one top-k router, found {len(routers)}")
    if len(shared) != 1:
        reasons.append(f"expected one shared expert MLP, found {len(shared)}")
    if len(routed) != 1:
        reasons.append(f"expected one routed expert MLP, found {len(routed)}")
    if len(combines) != 1:
        reasons.append(f"expected one weighted combine, found {len(combines)}")
    if reasons:
        raise ExpertParallelLegalityError(tuple(reasons))

    router = routers[0]
    shared_mlp = shared[0]
    routed_mlp = routed[0]
    combine = combines[0]
    router_projection = graph.producer(router.logits)
    if not isinstance(router_projection, LinearOp):
        reasons.append("router logits are not produced by an ordinary linear projection")
    elif router_projection.input != shared_mlp.input:
        reasons.append("router and expert MLPs do not consume the same token tensor")
    if shared_mlp.input != routed_mlp.input:
        reasons.append("shared and routed expert MLPs do not consume the same token tensor")
    if routed_mlp.expert_indices != router.expert_indices:
        reasons.append("routed MLP does not consume the recovered global expert indices")
    if combine.shared != shared_mlp.output or combine.routed != routed_mlp.output:
        reasons.append("weighted combine does not consume the recovered shared and routed outputs")
    if combine.route_weights != router.output:
        reasons.append("weighted combine does not consume the recovered route weights")
    expected = {
        operation.id
        for operation in (router_projection, router, shared_mlp, routed_mlp, combine)
        if operation is not None
    }
    if expected != {operation.id for operation in graph.operations}:
        reasons.append("generic EP region does not permit additional semantic operations")
    if reasons:
        raise ExpertParallelLegalityError(tuple(reasons))
    assert isinstance(router_projection, LinearOp)
    return _Region(router_projection, router, shared_mlp, routed_mlp, combine)


def _legality_reasons(
    region: _Region,
    *,
    config: ExpertParallelConfig,
    numerical_policy: NumericalPolicy,
) -> tuple[str, ...]:
    reasons: list[str] = []
    global_experts = region.router.logits.shape[1]
    routed_experts = region.routed.gate_weight.shape[0]
    if numerical_policy is NumericalPolicy.BITWISE_EXACT:
        reasons.append("segmented contractions and scatter reduction may reorder floating-point operations")
    if config.expert_parallel_size <= 0:
        reasons.append("expert-parallel size must be positive")
    elif global_experts % config.expert_parallel_size != 0:
        reasons.append("global expert count must be divisible by expert-parallel size")
    if routed_experts != global_experts:
        reasons.append(f"routed weights must have the global expert axis ({global_experts}), got {routed_experts}")
    if config.segment_padding <= 0:
        reasons.append("segment padding must be positive")
    if not isfinite(config.receiver_capacity_factor) or config.receiver_capacity_factor < 1.0:
        reasons.append("receiver capacity factor must be finite and at least 1.0")
    for name, tile in (
        ("contraction", config.contraction_tile),
        ("SwiGLU", config.swiglu_tile),
        ("exchange", config.exchange_tile),
        ("scatter", config.scatter_tile),
    ):
        if any(dimension <= 0 for dimension in tile):
            reasons.append(f"{name} tile dimensions must be positive")
    if config.minibatch_size <= 0 or (
        config.segment_padding > 0 and config.minibatch_size % config.segment_padding != 0
    ):
        reasons.append("minibatch size must be positive and divisible by segment padding")
    if config.macrobatch_size <= 0 or config.macrobatch_size % config.minibatch_size != 0:
        reasons.append("macrobatch size must be a positive multiple of minibatch size")
    workers = (
        config.exchange_workers,
        config.contraction_producer_workers,
        config.contraction_consumer_workers,
        config.transform_workers,
    )
    if any(workers_count <= 0 for workers_count in workers):
        reasons.append("every worker pool must contain at least one worker")
    if not config.exchange_worker_candidates or any(worker <= 0 for worker in config.exchange_worker_candidates):
        reasons.append("exchange-worker candidates must be non-empty and positive")
    elif config.exchange_workers not in config.exchange_worker_candidates:
        reasons.append("selected exchange worker count must be one of the declared candidates")
    depths = (
        config.contraction_pipeline_depth,
        config.transform_pipeline_depth,
        config.exchange_pipeline_depth,
    )
    if any(depth <= 0 for depth in depths):
        reasons.append("every pipeline depth must be positive")
    if not config.exchange_implementation_candidates:
        reasons.append("at least one exchange implementation candidate is required")
    elif config.exchange_implementation not in config.exchange_implementation_candidates:
        reasons.append("selected exchange implementation must be one of the declared candidates")
    if config.forward_transport.semantics is not TransportSemantics.PAYLOAD_PERMUTATION:
        reasons.append("forward transport must only permute payload; semantic reduction belongs to a compiler stage")
    if config.reverse_transport.semantics is not TransportSemantics.PAYLOAD_PERMUTATION:
        reasons.append("reverse transport must only permute payload; semantic reduction belongs to a compiler stage")
    if not config.merge_implementation:
        reasons.append("merge implementation must be explicit")
    if not config.segmented_contraction_candidates:
        reasons.append("at least one segmented-contraction candidate is required")
    elif config.segmented_contraction_implementation not in config.segmented_contraction_candidates:
        reasons.append("selected segmented contraction must be one of the declared candidates")
    if config.gate_up_layout not in config.gate_up_layout_candidates:
        reasons.append("selected gate/up layout must be one of the declared layout candidates")
    if config.overlap_policy not in config.overlap_policy_candidates:
        reasons.append("selected overlap policy must be one of the declared candidates")
    if config.materialization_schedule not in config.materialization_schedule_candidates:
        reasons.append("selected materialization schedule must be one of the declared candidates")
    if region.shared.accumulation_dtype is not DType.FP32 or region.routed.accumulation_dtype is not DType.FP32:
        reasons.append("expert contractions must accumulate in FP32")
    activation_and_weights = (
        region.shared.input.dtype,
        region.shared.gate_weight.dtype,
        region.shared.up_weight.dtype,
        region.shared.down_weight.dtype,
        region.routed.gate_weight.dtype,
        region.routed.up_weight.dtype,
        region.routed.down_weight.dtype,
    )
    if any(dtype is not DType.BF16 for dtype in activation_and_weights):
        reasons.append("the first generic EP lowering requires BF16 activations and expert weights")
    return tuple(reasons)


def _build_plan(region: _Region, *, config: ExpertParallelConfig) -> ExpertParallelPlan:
    tokens, hidden = region.shared.input.shape
    global_experts, intermediate, _ = region.routed.gate_weight.shape
    local_experts = global_experts // config.expert_parallel_size
    assignments = tokens * region.router.top_k
    receiver_capacity = max(local_experts, ceil(config.receiver_capacity_factor * assignments))
    padded_local_capacity = receiver_capacity + local_experts * (config.segment_padding - 1)
    prefix = region.combine.output.name
    route_relation = RouteRelation(
        name=f"{prefix}.routes",
        token_count=tokens,
        slots_per_token=region.router.top_k,
        global_expert_count=global_experts,
        token_column="token",
        slot_column="slot",
        global_expert_column="global_expert",
        weight_column="weight",
        source_expert_indices=region.router.expert_indices.name,
        source_weights=region.router.output.name,
    )
    ownership = ExpertOwnership(
        global_expert_count=global_experts,
        expert_parallel_size=config.expert_parallel_size,
        local_expert_count=local_experts,
        owner_rank_expression=f"global_expert // {local_experts}",
        local_expert_expression=f"global_expert % {local_experts}",
    )
    segments = ExpertSegmentContract(
        keys=("local_expert",),
        stable_order=("token", "slot"),
        segment_count=local_experts,
        padding_quantum=config.segment_padding,
        padded_token=-1,
        padded_weight=0.0,
    )
    capacity = ExpertCapacityPolicy(
        capacity_factor=config.receiver_capacity_factor,
        receiver_assignment_capacity=receiver_capacity,
        padded_local_capacity=padded_local_capacity,
        overflow_policy="reject selected plan and run exact fallback before expert contraction",
    )
    exchange_projections = (
        ExchangeRelationProjection(
            mode=ExchangeRowMode.ASSIGNMENT,
            grouping_keys=("token", "slot", "owner_rank", "local_expert"),
            activation_rows="one activation row per route assignment",
            metadata_rows="one metadata row per route assignment",
            receiver_expansion="identity followed by local expert grouping",
        ),
        ExchangeRelationProjection(
            mode=ExchangeRowMode.COALESCED_TOKEN_OWNER,
            grouping_keys=("source_token", "owner_rank"),
            activation_rows="one activation row per distinct (source_token, owner_rank)",
            metadata_rows="all route slots and local experts for the coalesced activation row",
            receiver_expansion="expand route slots then group by local expert",
        ),
    )
    selected_exchange_projection = next(
        projection for projection in exchange_projections if projection.mode is config.exchange_row_mode
    )
    gate_up_layout = GateUpLayoutContract(
        semantic_weight_layout="separate [global_expert, intermediate, hidden] gate/up",
        semantic_output_layout="separate gate and up values",
        selected=config.gate_up_layout,
        candidates=config.gate_up_layout_candidates,
        legalization=(
            "concatenate to [local_expert, 2*intermediate, hidden]"
            if config.gate_up_layout is GateUpPhysicalLayout.CONCATENATED_E_2I_K
            else (
                "interleave to [local_expert, 2*intermediate, hidden]"
                if config.gate_up_layout is GateUpPhysicalLayout.INTERLEAVED_E_2I_K
                else "retain separate gate/up tensors in [local_expert, 2, intermediate, hidden]"
            )
        ),
    )
    schedule = _schedule(config)
    merge_program = compile_ordered_relation_fold(
        partition_count=config.expert_parallel_size,
        accumulation_dtype=DType.FP32,
        output_dtype=region.combine.output.dtype,
    ).tile_program
    names = _names(prefix)
    stages = _stages(region, names=names, exchange_projection=selected_exchange_projection, gate_up=gate_up_layout)
    tile_flows = _tile_flows(
        region,
        names=names,
        assignments=assignments,
        receiver_capacity=receiver_capacity,
        padded_local_capacity=padded_local_capacity,
        local_experts=local_experts,
        hidden=hidden,
        intermediate=intermediate,
        config=config,
    )
    buffers = _derive_buffer_lifetimes(stages, tile_flows)
    materializations = _derive_materializations(tile_flows)
    return ExpertParallelPlan(
        route_relation=route_relation,
        ownership=ownership,
        segments=segments,
        capacity=capacity,
        exchange_projections=exchange_projections,
        selected_exchange_projection=selected_exchange_projection,
        gate_up_layout=gate_up_layout,
        schedule=schedule,
        map_fold_semantics=_map_fold_semantics(),
        merge_program=merge_program,
        stages=stages,
        tile_flows=tile_flows,
        buffers=buffers,
        materializations=materializations,
        rewrites=(_explanation(region, config=config),),
    )


def _map_fold_semantics() -> MapFoldSemantics:
    left = scalar_input("left")
    right = scalar_input("right")
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
    return MapFoldSemantics(
        pair_map=scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_binary(ScalarExpressionKind.MULTIPLY, left, sigmoid),
            right,
        ),
        fold_contribution=scalar_binary(
            ScalarExpressionKind.MULTIPLY,
            scalar_input("value"),
            scalar_input("weight"),
        ),
        fold_update=scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_input("state"),
            scalar_input("contribution"),
        ),
        post_fold_map=scalar_binary(
            ScalarExpressionKind.ADD,
            scalar_input("folded"),
            scalar_input("base"),
        ),
        explicit_rounding_functions=frozenset({"fold_contribution", "fold_update"}),
    )


def _schedule(config: ExpertParallelConfig) -> ExpertParallelSchedule:
    exchange_stages = (
        ExpertParallelStageKind.PROJECT_EXCHANGE_ROWS,
        ExpertParallelStageKind.FORWARD_EXCHANGE,
        ExpertParallelStageKind.EXPAND_LOCAL_ASSIGNMENTS,
        ExpertParallelStageKind.REVERSE_EXCHANGE,
        ExpertParallelStageKind.WEIGHTED_SCATTER_REDUCE,
    )
    contraction_stages = (
        ExpertParallelStageKind.SHARED_DENSE_GATE_UP,
        ExpertParallelStageKind.ROUTED_SEGMENTED_GATE_UP,
        ExpertParallelStageKind.SHARED_DENSE_DOWN,
        ExpertParallelStageKind.ROUTED_SEGMENTED_DOWN,
    )
    transform_stages = (
        ExpertParallelStageKind.ROUTE_RELATION,
        ExpertParallelStageKind.EXPERT_OWNERSHIP,
        ExpertParallelStageKind.LEGALIZE_GATE_UP_LAYOUT,
        ExpertParallelStageKind.GROUP_BY_OWNER,
        ExpertParallelStageKind.SEGMENT_BY_LOCAL_EXPERT,
        ExpertParallelStageKind.PAD_LOCAL_SEGMENTS,
        ExpertParallelStageKind.SHARED_PAIRWISE_SWIGLU,
        ExpertParallelStageKind.ROUTED_PAIRWISE_SWIGLU,
        ExpertParallelStageKind.SHARED_ADD,
    )
    return ExpertParallelSchedule(
        expert_parallel_size=config.expert_parallel_size,
        segment_padding=config.segment_padding,
        contraction_tile=config.contraction_tile,
        swiglu_tile=config.swiglu_tile,
        exchange_tile=config.exchange_tile,
        scatter_tile=config.scatter_tile,
        worker_pools=(
            WorkerPool("exchange", config.exchange_workers, exchange_stages),
            WorkerPool("contraction_producer", config.contraction_producer_workers, contraction_stages),
            WorkerPool("contraction_consumer", config.contraction_consumer_workers, contraction_stages),
            WorkerPool("local_transform", config.transform_workers, transform_stages),
        ),
        minibatch_size=config.minibatch_size,
        macrobatch_size=config.macrobatch_size,
        pipelines=(
            PipelineDepth("exchange", exchange_stages, config.exchange_pipeline_depth),
            PipelineDepth("contraction", contraction_stages, config.contraction_pipeline_depth),
            PipelineDepth("local_transform", transform_stages, config.transform_pipeline_depth),
        ),
        exchange_implementation=config.exchange_implementation,
        exchange_implementation_candidates=config.exchange_implementation_candidates,
        forward_transport=config.forward_transport,
        reverse_transport=config.reverse_transport,
        merge_implementation=config.merge_implementation,
        segmented_contraction_implementation=config.segmented_contraction_implementation,
        segmented_contraction_candidates=config.segmented_contraction_candidates,
        exchange_worker_candidates=config.exchange_worker_candidates,
        overlap_policy=config.overlap_policy,
        overlap_policy_candidates=config.overlap_policy_candidates,
        materialization_schedule=config.materialization_schedule,
        materialization_schedule_candidates=config.materialization_schedule_candidates,
    )


def _names(prefix: str) -> dict[str, str]:
    keys = (
        "route_keys",
        "route_weights",
        "owned_routes",
        "local_gate_up_weight",
        "local_down_weight",
        "segmented_routes",
        "segmented_weights",
        "inverse_permutation",
        "segment_sizes",
        "exchange_x_rows",
        "exchange_route_metadata",
        "exchange_owner_sizes",
        "received_exchange_x",
        "received_exchange_metadata",
        "received_x",
        "received_routes",
        "local_segmented_x",
        "local_segmented_routes",
        "local_inverse_permutation",
        "local_segment_sizes",
        "padded_local_x",
        "padded_local_routes",
        "padded_local_segment_sizes",
        "shared_gate_up",
        "routed_gate_up",
        "routed_segment_output",
        "returned_routes",
        "routed_scatter_output",
    )
    return {key: f"{prefix}.{key}" for key in keys}


def _stage(
    name: str,
    kind: ExpertParallelStageKind,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    operation: str,
    provenance: tuple[str, ...],
) -> ExpertParallelStage:
    return ExpertParallelStage(name, kind, inputs, outputs, operation, provenance)


def _stages(
    region: _Region,
    *,
    names: dict[str, str],
    exchange_projection: ExchangeRelationProjection,
    gate_up: GateUpLayoutContract,
) -> tuple[ExpertParallelStage, ...]:
    return (
        _stage(
            "form_route_relation",
            ExpertParallelStageKind.ROUTE_RELATION,
            (region.router.expert_indices.name, region.router.output.name),
            (names["route_keys"], names["route_weights"]),
            "flatten (token, slot, global_expert, weight) assignments",
            ("TopKRouterOp",),
        ),
        _stage(
            "map_expert_ownership",
            ExpertParallelStageKind.EXPERT_OWNERSHIP,
            (names["route_keys"],),
            (names["owned_routes"],),
            "map each global expert to a contiguous (owner_rank, local_expert)",
            ("RoutedExpertMLPOp global expert axis",),
        ),
        _stage(
            "legalize_local_expert_weights",
            ExpertParallelStageKind.LEGALIZE_GATE_UP_LAYOUT,
            (region.routed.gate_weight.name, region.routed.up_weight.name, region.routed.down_weight.name),
            (names["local_gate_up_weight"], names["local_down_weight"]),
            gate_up.legalization,
            ("global expert ownership", "semantic separate gate/up weight layout"),
        ),
        _stage(
            "group_routes_by_owner",
            ExpertParallelStageKind.GROUP_BY_OWNER,
            (names["owned_routes"], names["route_weights"]),
            (
                names["segmented_routes"],
                names["segmented_weights"],
                names["inverse_permutation"],
                names["segment_sizes"],
            ),
            "stable sort/group assignments by owner_rank for transport",
            ("TopKRouterOp", "ExpertOwnership"),
        ),
        _stage(
            "project_exchange_rows",
            ExpertParallelStageKind.PROJECT_EXCHANGE_ROWS,
            (
                region.shared.input.name,
                names["segmented_routes"],
                names["segmented_weights"],
                names["segment_sizes"],
            ),
            (names["exchange_x_rows"], names["exchange_route_metadata"], names["exchange_owner_sizes"]),
            f"project route relation by {exchange_projection.grouping_keys}: {exchange_projection.activation_rows}",
            ("owner-grouped route relation", exchange_projection.mode.value),
        ),
        _stage(
            "forward_exchange",
            ExpertParallelStageKind.FORWARD_EXCHANGE,
            (names["exchange_x_rows"], names["exchange_route_metadata"], names["exchange_owner_sizes"]),
            (names["received_exchange_x"], names["received_exchange_metadata"]),
            "exchange projected activation rows and route metadata to owner ranks",
            ("expert ownership", "exchange relation projection"),
        ),
        _stage(
            "expand_local_assignments",
            ExpertParallelStageKind.EXPAND_LOCAL_ASSIGNMENTS,
            (names["received_exchange_x"], names["received_exchange_metadata"]),
            (names["received_x"], names["received_routes"]),
            exchange_projection.receiver_expansion,
            ("forward exchange",),
        ),
        _stage(
            "segment_by_local_expert",
            ExpertParallelStageKind.SEGMENT_BY_LOCAL_EXPERT,
            (names["received_x"], names["received_routes"]),
            (
                names["local_segmented_x"],
                names["local_segmented_routes"],
                names["local_inverse_permutation"],
                names["local_segment_sizes"],
            ),
            "stable group received assignments by local_expert",
            ("received route metadata", "expert ownership"),
        ),
        _stage(
            "pad_local_segments",
            ExpertParallelStageKind.PAD_LOCAL_SEGMENTS,
            (names["local_segmented_x"], names["local_segmented_routes"], names["local_segment_sizes"]),
            (names["padded_local_x"], names["padded_local_routes"], names["padded_local_segment_sizes"]),
            "pad each receiver-local expert segment once using zero activation rows",
            ("SegmentBy(local_expert)",),
        ),
        _stage(
            "shared_dense_gate_up",
            ExpertParallelStageKind.SHARED_DENSE_GATE_UP,
            (region.shared.input.name, region.shared.gate_weight.name, region.shared.up_weight.name),
            (names["shared_gate_up"],),
            f"dense FP32-accumulating gate/up contraction in {gate_up.selected.value} layout",
            ("SharedExpertMLPOp gate/up",),
        ),
        _stage(
            "routed_segmented_gate_up",
            ExpertParallelStageKind.ROUTED_SEGMENTED_GATE_UP,
            (
                names["padded_local_x"],
                names["local_gate_up_weight"],
                names["padded_local_segment_sizes"],
            ),
            (names["routed_gate_up"],),
            f"segmented gate/up contraction in {gate_up.selected.value} layout using local group sizes",
            ("RoutedExpertMLPOp gate/up",),
        ),
        _stage(
            "shared_pairwise_swiglu",
            ExpertParallelStageKind.SHARED_PAIRWISE_SWIGLU,
            (names["shared_gate_up"],),
            (region.shared.hidden.name,),
            "silu(gate) * up",
            ("SharedExpertMLPOp activation",),
        ),
        _stage(
            "routed_pairwise_swiglu",
            ExpertParallelStageKind.ROUTED_PAIRWISE_SWIGLU,
            (names["routed_gate_up"],),
            (region.routed.hidden.name,),
            "silu(gate) * up independently within each expert segment",
            ("RoutedExpertMLPOp activation",),
        ),
        _stage(
            "shared_dense_down",
            ExpertParallelStageKind.SHARED_DENSE_DOWN,
            (region.shared.hidden.name, region.shared.down_weight.name),
            (region.shared.output.name,),
            "dense FP32-accumulating down contraction",
            ("SharedExpertMLPOp down",),
        ),
        _stage(
            "routed_segmented_down",
            ExpertParallelStageKind.ROUTED_SEGMENTED_DOWN,
            (region.routed.hidden.name, names["local_down_weight"], names["padded_local_segment_sizes"]),
            (names["routed_segment_output"],),
            "segmented down contraction using the same local-expert group sizes",
            ("RoutedExpertMLPOp down",),
        ),
        _stage(
            "reverse_exchange",
            ExpertParallelStageKind.REVERSE_EXCHANGE,
            (
                names["routed_segment_output"],
                names["padded_local_routes"],
                names["local_inverse_permutation"],
                names["inverse_permutation"],
            ),
            (names["returned_routes"],),
            "reverse the owner exchange and restore source assignment order",
            ("forward exchange relation",),
        ),
        _stage(
            "weighted_scatter_reduce",
            ExpertParallelStageKind.WEIGHTED_SCATTER_REDUCE,
            (names["returned_routes"], names["route_weights"]),
            (names["routed_scatter_output"],),
            "scatter-add route_weight * routed_output by token",
            ("WeightedExpertCombineOp",),
        ),
        _stage(
            "shared_add",
            ExpertParallelStageKind.SHARED_ADD,
            (names["routed_scatter_output"], region.shared.output.name),
            (region.combine.output.name,),
            "add shared expert output to the routed scatter reduction",
            ("WeightedExpertCombineOp",),
        ),
    )


def _edge(
    value: str,
    shape: tuple[int, ...],
    dtype: DType,
    producer: str,
    consumers: tuple[str, ...],
    layout: str,
    readiness: ReadinessGranularity,
    *,
    alias_of: str | None = None,
    tile_shape: tuple[int, ...] | None = None,
    storage: TileStorage = TileStorage.GLOBAL_BUFFER,
) -> TileFlowEdge:
    return TileFlowEdge(
        value,
        shape,
        dtype,
        producer,
        consumers,
        layout,
        tile_shape,
        storage,
        readiness,
        len(consumers),
        alias_of,
    )


def _tile_flows(
    region: _Region,
    *,
    names: dict[str, str],
    assignments: int,
    receiver_capacity: int,
    padded_local_capacity: int,
    local_experts: int,
    hidden: int,
    intermediate: int,
    config: ExpertParallelConfig,
) -> tuple[TileFlowEdge, ...]:
    tokens = region.shared.input.shape[0]
    top_k = region.router.top_k
    output_name = region.combine.output.name
    if config.exchange_row_mode is ExchangeRowMode.COALESCED_TOKEN_OWNER:
        exchange_row_capacity = tokens * min(top_k, config.expert_parallel_size)
    else:
        exchange_row_capacity = assignments
    received_exchange_capacity = receiver_capacity
    if config.gate_up_layout is GateUpPhysicalLayout.SEPARATE_E_I_K:
        local_gate_up_shape = (local_experts, 2, intermediate, hidden)
        shared_gate_up_shape = (tokens, 2, intermediate)
        routed_gate_up_shape = (padded_local_capacity, 2, intermediate)
    else:
        local_gate_up_shape = (local_experts, 2 * intermediate, hidden)
        shared_gate_up_shape = (tokens, 2 * intermediate)
        routed_gate_up_shape = (padded_local_capacity, 2 * intermediate)
    return (
        _edge(
            region.shared.input.name,
            region.shared.input.shape,
            DType.BF16,
            "region_input",
            ("project_exchange_rows", "shared_dense_gate_up"),
            "token_hidden_row_major",
            ReadinessGranularity.TOKEN_TILE,
            alias_of=region.shared.input.name,
            tile_shape=config.exchange_tile,
            storage=TileStorage.INPUT_ALIAS,
        ),
        _edge(
            names["route_keys"],
            (assignments, 3),
            DType.INT32,
            "form_route_relation",
            ("map_expert_ownership",),
            "relation_token_slot_global_expert",
            ReadinessGranularity.RELATION_ROW,
            storage=TileStorage.RELATION_BUFFER,
        ),
        _edge(
            names["route_weights"],
            (assignments,),
            DType.FP32,
            "form_route_relation",
            ("group_routes_by_owner", "weighted_scatter_reduce"),
            "relation_assignment_order",
            ReadinessGranularity.RELATION_ROW,
            alias_of=region.router.output.name,
            storage=TileStorage.INPUT_ALIAS,
        ),
        _edge(
            names["owned_routes"],
            (assignments, 4),
            DType.INT32,
            "map_expert_ownership",
            ("group_routes_by_owner",),
            "relation_token_slot_owner_local_expert",
            ReadinessGranularity.RELATION_ROW,
            storage=TileStorage.RELATION_BUFFER,
        ),
        _edge(
            names["local_gate_up_weight"],
            local_gate_up_shape,
            DType.BF16,
            "legalize_local_expert_weights",
            ("routed_segmented_gate_up",),
            config.gate_up_layout.value,
            ReadinessGranularity.EXPERT_SEGMENT,
            tile_shape=config.contraction_tile,
            storage=TileStorage.GLOBAL_BUFFER,
        ),
        _edge(
            names["local_down_weight"],
            (local_experts, intermediate, hidden),
            DType.BF16,
            "legalize_local_expert_weights",
            ("routed_segmented_down",),
            "local_expert_intermediate_hidden",
            ReadinessGranularity.EXPERT_SEGMENT,
            tile_shape=config.contraction_tile,
            storage=TileStorage.GLOBAL_BUFFER,
        ),
        _edge(
            names["segmented_routes"],
            (assignments, 4),
            DType.INT32,
            "group_routes_by_owner",
            ("project_exchange_rows",),
            "owner_grouped_route_relation",
            ReadinessGranularity.EXPERT_SEGMENT,
            storage=TileStorage.RELATION_BUFFER,
        ),
        _edge(
            names["segmented_weights"],
            (assignments,),
            DType.FP32,
            "group_routes_by_owner",
            ("project_exchange_rows",),
            "owner_grouped_route_weights",
            ReadinessGranularity.EXPERT_SEGMENT,
            storage=TileStorage.RELATION_BUFFER,
        ),
        _edge(
            names["inverse_permutation"],
            (assignments,),
            DType.INT32,
            "group_routes_by_owner",
            ("reverse_exchange",),
            "source_assignment_to_segment_position",
            ReadinessGranularity.RELATION_ROW,
            storage=TileStorage.RELATION_BUFFER,
        ),
        _edge(
            names["segment_sizes"],
            (config.expert_parallel_size,),
            DType.INT32,
            "group_routes_by_owner",
            ("project_exchange_rows",),
            "owner_route_counts",
            ReadinessGranularity.MINIBATCH,
            storage=TileStorage.RELATION_BUFFER,
        ),
        _edge(
            names["exchange_x_rows"],
            (exchange_row_capacity, hidden),
            DType.BF16,
            "project_exchange_rows",
            ("forward_exchange",),
            f"{config.exchange_row_mode.value}_send_rows",
            ReadinessGranularity.MINIBATCH,
            tile_shape=config.exchange_tile,
        ),
        _edge(
            names["exchange_route_metadata"],
            (assignments, 4),
            DType.INT32,
            "project_exchange_rows",
            ("forward_exchange",),
            f"{config.exchange_row_mode.value}_route_metadata",
            ReadinessGranularity.MINIBATCH,
            storage=TileStorage.RELATION_BUFFER,
        ),
        _edge(
            names["exchange_owner_sizes"],
            (config.expert_parallel_size,),
            DType.INT32,
            "project_exchange_rows",
            ("forward_exchange",),
            f"{config.exchange_row_mode.value}_owner_send_sizes",
            ReadinessGranularity.MINIBATCH,
            storage=TileStorage.RELATION_BUFFER,
        ),
        _edge(
            names["received_exchange_x"],
            (received_exchange_capacity, hidden),
            DType.BF16,
            "forward_exchange",
            ("expand_local_assignments",),
            f"received_{config.exchange_row_mode.value}_rows",
            ReadinessGranularity.MINIBATCH,
            tile_shape=config.exchange_tile,
        ),
        _edge(
            names["received_exchange_metadata"],
            (receiver_capacity, 4),
            DType.INT32,
            "forward_exchange",
            ("expand_local_assignments",),
            "received_route_slot_local_expert_metadata",
            ReadinessGranularity.MINIBATCH,
            storage=TileStorage.RELATION_BUFFER,
        ),
        _edge(
            names["received_x"],
            (receiver_capacity, hidden),
            DType.BF16,
            "expand_local_assignments",
            ("segment_by_local_expert",),
            "received_assignment_rows",
            ReadinessGranularity.MINIBATCH,
            tile_shape=config.contraction_tile,
        ),
        _edge(
            names["received_routes"],
            (receiver_capacity, 4),
            DType.INT32,
            "expand_local_assignments",
            ("segment_by_local_expert",),
            "received_assignment_relation",
            ReadinessGranularity.MINIBATCH,
            storage=TileStorage.RELATION_BUFFER,
        ),
        _edge(
            names["local_segmented_x"],
            (receiver_capacity, hidden),
            DType.BF16,
            "segment_by_local_expert",
            ("pad_local_segments",),
            "local_expert_segment_mk",
            ReadinessGranularity.EXPERT_SEGMENT,
            tile_shape=config.contraction_tile,
        ),
        _edge(
            names["local_segmented_routes"],
            (receiver_capacity, 4),
            DType.INT32,
            "segment_by_local_expert",
            ("pad_local_segments",),
            "local_expert_segment_relation",
            ReadinessGranularity.EXPERT_SEGMENT,
            storage=TileStorage.RELATION_BUFFER,
        ),
        _edge(
            names["local_inverse_permutation"],
            (receiver_capacity,),
            DType.INT32,
            "segment_by_local_expert",
            ("reverse_exchange",),
            "received_assignment_to_local_segment_position",
            ReadinessGranularity.RELATION_ROW,
            storage=TileStorage.RELATION_BUFFER,
        ),
        _edge(
            names["local_segment_sizes"],
            (local_experts,),
            DType.INT32,
            "segment_by_local_expert",
            ("pad_local_segments",),
            "local_expert_group_sizes",
            ReadinessGranularity.EXPERT_SEGMENT,
            storage=TileStorage.RELATION_BUFFER,
        ),
        _edge(
            names["padded_local_x"],
            (padded_local_capacity, hidden),
            DType.BF16,
            "pad_local_segments",
            ("routed_segmented_gate_up",),
            "padded_local_expert_segment_mk",
            ReadinessGranularity.PADDED_SEGMENT_TILE,
            tile_shape=config.contraction_tile,
        ),
        _edge(
            names["padded_local_routes"],
            (padded_local_capacity, 4),
            DType.INT32,
            "pad_local_segments",
            ("reverse_exchange",),
            "padded_local_expert_segment_relation",
            ReadinessGranularity.PADDED_SEGMENT_TILE,
            storage=TileStorage.RELATION_BUFFER,
        ),
        _edge(
            names["padded_local_segment_sizes"],
            (local_experts,),
            DType.INT32,
            "pad_local_segments",
            ("routed_segmented_gate_up", "routed_segmented_down"),
            "padded_local_expert_group_sizes",
            ReadinessGranularity.EXPERT_SEGMENT,
            storage=TileStorage.RELATION_BUFFER,
        ),
        _edge(
            names["shared_gate_up"],
            shared_gate_up_shape,
            DType.BF16,
            "shared_dense_gate_up",
            ("shared_pairwise_swiglu",),
            config.gate_up_layout.value,
            ReadinessGranularity.OUTPUT_TILE,
            tile_shape=config.contraction_tile,
        ),
        _edge(
            names["routed_gate_up"],
            routed_gate_up_shape,
            DType.BF16,
            "routed_segmented_gate_up",
            ("routed_pairwise_swiglu",),
            config.gate_up_layout.value,
            ReadinessGranularity.OUTPUT_TILE,
            tile_shape=config.contraction_tile,
        ),
        _edge(
            region.shared.hidden.name,
            (tokens, intermediate),
            DType.BF16,
            "shared_pairwise_swiglu",
            ("shared_dense_down",),
            "token_intermediate_row_major",
            ReadinessGranularity.ROW_TILE,
            tile_shape=config.swiglu_tile,
        ),
        _edge(
            region.routed.hidden.name,
            (padded_local_capacity, intermediate),
            DType.BF16,
            "routed_pairwise_swiglu",
            ("routed_segmented_down",),
            "local_expert_segment_mk",
            ReadinessGranularity.ROW_TILE,
            tile_shape=config.swiglu_tile,
        ),
        _edge(
            region.shared.output.name,
            (tokens, hidden),
            DType.BF16,
            "shared_dense_down",
            ("shared_add",),
            "token_hidden_row_major",
            ReadinessGranularity.OUTPUT_TILE,
            tile_shape=config.contraction_tile,
        ),
        _edge(
            names["routed_segment_output"],
            (padded_local_capacity, hidden),
            DType.BF16,
            "routed_segmented_down",
            ("reverse_exchange",),
            "local_expert_segment_mn",
            ReadinessGranularity.OUTPUT_TILE,
            tile_shape=config.contraction_tile,
        ),
        _edge(
            names["returned_routes"],
            (assignments, hidden),
            DType.BF16,
            "reverse_exchange",
            ("weighted_scatter_reduce",),
            "source_assignment_hidden",
            ReadinessGranularity.MINIBATCH,
            tile_shape=config.exchange_tile,
        ),
        _edge(
            names["routed_scatter_output"],
            (tokens, hidden),
            DType.BF16,
            "weighted_scatter_reduce",
            ("shared_add",),
            "token_hidden_row_major",
            ReadinessGranularity.TOKEN_TILE,
            tile_shape=config.scatter_tile,
        ),
        _edge(
            output_name,
            (tokens, hidden),
            DType.BF16,
            "shared_add",
            ("region_output",),
            "token_hidden_row_major",
            ReadinessGranularity.TOKEN_TILE,
            tile_shape=config.scatter_tile,
        ),
    )


def _derive_buffer_lifetimes(
    stages: tuple[ExpertParallelStage, ...], edges: tuple[TileFlowEdge, ...]
) -> tuple[BufferLifetime, ...]:
    stage_index = {stage.name: index for index, stage in enumerate(stages)}

    def last_consumer(edge: TileFlowEdge) -> str:
        if "region_output" in edge.consumers:
            return "region_output"
        return max(edge.consumers, key=stage_index.__getitem__)

    return tuple(
        BufferLifetime(
            value=edge.value,
            shape=edge.shape,
            dtype=edge.dtype,
            logical_layout=edge.logical_layout,
            tile_shape=edge.tile_shape,
            storage=edge.storage,
            live_from=edge.producer,
            live_until=last_consumer(edge),
            alias_of=edge.alias_of,
        )
        for edge in edges
    )


def _derive_materializations(edges: tuple[TileFlowEdge, ...]) -> tuple[MaterializationRecord, ...]:
    return tuple(
        MaterializationRecord(
            value=edge.value,
            shape=edge.shape,
            dtype=edge.dtype,
            disposition=(
                MaterializationDisposition.ALIAS if edge.alias_of is not None else MaterializationDisposition.MATERIALIZE
            ),
            reason=f"tile flow {edge.producer} -> {', '.join(edge.consumers)} at {edge.readiness.value} readiness",
            alias_of=edge.alias_of,
        )
        for edge in edges
    )


def _explanation(region: _Region, *, config: ExpertParallelConfig) -> RewriteExplanation:
    return RewriteExplanation(
        name="lower_global_moe_to_generic_expert_parallel_dataflow",
        applied=True,
        original_fragment=(
            "global top-k expert indices and weights",
            "shared gated MLP",
            "global routed gated MLP",
            "weighted top-k combine plus shared output",
        ),
        transformed_fragment=(
            "RouteRelation(token, slot, global_expert, weight)",
            "ExpertOwnership(global_expert -> owner_rank, local_expert)",
            f"GroupBy(owner_rank) then ProjectExchangeRows({config.exchange_row_mode.value})",
            "forward Exchange -> expand -> SegmentBy(local_expert) -> PadLocalSegments",
            "segmented gate/up -> pairwise SwiGLU -> segmented down",
            "ReverseExchange -> weighted ScatterReduce -> shared add",
        ),
        semantic_properties=(
            "expert computations are independent before weighted token reduction",
            "global expert weights partition into contiguous owner-local views",
            "gate and up contractions share the same segmented route relation",
            "receiver-local padding rows use token=-1 and weight=0 and cannot affect the output",
        ),
        legality_checks=(
            "routed weights carry the full global expert axis",
            "global expert count divides the expert-parallel size",
            "receiver capacity overflow is guarded by an exact fallback",
            "exchange and contraction candidates implement the declared layouts",
            f"segment padding is {config.segment_padding}",
            "FP32 contraction accumulation and FP32 route weights are preserved",
        ),
        estimated_benefit=(
            "makes communication, segmentation, contraction, and reduction boundaries independently selectable "
            "while retaining tile-granular readiness"
        ),
        numerical_equivalence=NumericalEquivalence.ALGEBRAICALLY_EXACT,
        numerical_effect=(
            "segmented contraction tiling and weighted scatter-reduction order may differ from the source graph"
        ),
    )
