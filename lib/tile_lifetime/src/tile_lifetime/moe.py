# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the opaque Mixture-of-Kittens comparison oracle.

This module is deliberately excluded from Shuttle synthesis paths. It exists
only to describe and validate the pinned complete-kernel baseline.
"""

from dataclasses import dataclass
from enum import StrEnum
from math import ceil, isfinite

from shuttle.ir import DType
from tile_lifetime.ir import (
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
    OpaqueMoKOracleSkeleton,
    PaddedExpertSchedule,
    PersistentTaskPlacement,
    PersistentTaskRole,
    PersistentWorkerRole,
    ReadinessEvent,
    RegionPlan,
    RewriteExplanation,
)

EXPERT_PADDING = 256
SUPPORTED_EXPERT_PARALLEL_SIZES = frozenset({4, 8, 16, 32, 64})
MOK_BACKEND_REVISION = "3e1cf43ab93ad040afed52a45ab03cb490ffe4be"


class MoERoutedPrecision(StrEnum):
    """Routed-expert storage and matrix-multiply precision."""

    BF16 = "bf16"
    MXFP8 = "mxfp8"


@dataclass(frozen=True)
class MoKOracleConfig:
    """Finite physical choices exposed by the pinned complete MoK oracle."""

    expert_parallel_size: int
    communication_sm_count: int = 40
    minibatch_size: int = 4096
    macrobatch_size: int = 131_072
    schedule_capacity_multiplier: float = 0.5
    routed_precision: MoERoutedPrecision = MoERoutedPrecision.BF16


class MoELegalityError(ValueError):
    """Structured rejection of a graph or physical configuration."""

    def __init__(self, reasons: tuple[str, ...]):
        self.reasons = reasons
        super().__init__("; ".join(reasons))


@dataclass(frozen=True)
class _MoERegion:
    router_projection: LinearOp
    router: TopKRouterOp
    shared: SharedExpertMLPOp
    routed: RoutedExpertMLPOp
    combine: WeightedExpertCombineOp


def compile_mok_oracle_region(
    graph: TensorGraph,
    *,
    config: MoKOracleConfig,
    numerical_policy: NumericalPolicy,
) -> RegionPlan:
    """Build an oracle-backed plan that must not be counted as synthesis."""
    region = _recover_region(graph)
    reasons = _legality_reasons(region, config=config, numerical_policy=numerical_policy)
    if reasons:
        raise MoELegalityError(reasons)
    return _build_plan(region, config=config)


def _recover_region(graph: TensorGraph) -> _MoERegion:
    routers = tuple(operation for operation in graph.operations if isinstance(operation, TopKRouterOp))
    shared_experts = tuple(operation for operation in graph.operations if isinstance(operation, SharedExpertMLPOp))
    routed_experts = tuple(operation for operation in graph.operations if isinstance(operation, RoutedExpertMLPOp))
    combines = tuple(operation for operation in graph.operations if isinstance(operation, WeightedExpertCombineOp))
    reasons: list[str] = []
    if len(routers) != 1:
        reasons.append(f"expected one top-k router, found {len(routers)}")
    if len(shared_experts) != 1:
        reasons.append(f"expected one shared expert MLP, found {len(shared_experts)}")
    if len(routed_experts) != 1:
        reasons.append(f"expected one routed expert MLP, found {len(routed_experts)}")
    if len(combines) != 1:
        reasons.append(f"expected one weighted expert combine, found {len(combines)}")
    if reasons:
        raise MoELegalityError(tuple(reasons))

    router = routers[0]
    shared = shared_experts[0]
    routed = routed_experts[0]
    combine = combines[0]
    router_projection = graph.producer(router.logits)
    if not isinstance(router_projection, LinearOp):
        reasons.append("top-k logits are not produced by one ordinary linear projection")
    elif router_projection.input != shared.input:
        reasons.append("router projection and expert MLPs do not consume the same token tensor")
    if shared.input != routed.input:
        reasons.append("shared and routed experts do not consume the same token tensor")
    if routed.expert_indices != router.expert_indices:
        reasons.append("routed experts do not consume the recovered top-k indices")
    if combine.shared != shared.output or combine.routed != routed.output or combine.route_weights != router.output:
        reasons.append("expert combine does not consume the recovered shared, routed, and router-weight values")
    expected_operation_ids = {
        operation.id for operation in (router_projection, router, shared, routed, combine) if operation is not None
    }
    if expected_operation_ids != {operation.id for operation in graph.operations}:
        reasons.append("MoK structural prototype does not permit additional semantic operations")
    if reasons:
        raise MoELegalityError(tuple(reasons))
    assert isinstance(router_projection, LinearOp)
    return _MoERegion(
        router_projection=router_projection,
        router=router,
        shared=shared,
        routed=routed,
        combine=combine,
    )


def _legality_reasons(
    region: _MoERegion,
    *,
    config: MoKOracleConfig,
    numerical_policy: NumericalPolicy,
) -> tuple[str, ...]:
    tokens, hidden = region.shared.input.shape
    intermediate = region.shared.gate_weight.shape[0]
    local_experts = region.routed.gate_weight.shape[0]
    global_experts = region.router.logits.shape[1]
    reasons: list[str] = []
    if numerical_policy is NumericalPolicy.BITWISE_EXACT:
        reasons.append("persistent grouped execution changes finite-precision reduction and conversion order")
    if config.expert_parallel_size not in SUPPORTED_EXPERT_PARALLEL_SIZES:
        reasons.append("expert-parallel size must be one of 4, 8, 16, 32, or 64")
    elif global_experts != local_experts * config.expert_parallel_size:
        reasons.append(
            f"router has {global_experts} experts but {local_experts} local experts across "
            f"{config.expert_parallel_size} ranks require {local_experts * config.expert_parallel_size}"
        )
    if tokens < 512 or tokens % EXPERT_PADDING != 0:
        reasons.append("local token count must be at least 512 and divisible by 256")
    if hidden % EXPERT_PADDING != 0:
        reasons.append("hidden size must be divisible by 256")
    if intermediate % EXPERT_PADDING != 0:
        reasons.append("intermediate size must be divisible by 256")
    if not 0 < region.router.top_k <= 255:
        reasons.append("top-k must be in [1, 255]")
    if config.communication_sm_count <= 0 or config.communication_sm_count % 2 != 0:
        reasons.append("communication SM count must be positive and divisible by the two-CTA cluster size")
    if config.minibatch_size <= 0 or config.minibatch_size % EXPERT_PADDING != 0:
        reasons.append("minibatch size must be positive and divisible by 256")
    if config.macrobatch_size <= 0 or config.macrobatch_size % config.minibatch_size != 0:
        reasons.append("macrobatch size must be a positive multiple of minibatch size")
    if not isfinite(config.schedule_capacity_multiplier) or config.schedule_capacity_multiplier <= 0:
        reasons.append("schedule capacity multiplier must be positive and finite")
    if region.shared.accumulation_dtype is not DType.FP32 or region.routed.accumulation_dtype is not DType.FP32:
        reasons.append("shared and routed expert GEMMs must accumulate in FP32")
    if region.shared.input.dtype is not DType.BF16:
        reasons.append("the initial MoK skeleton requires BF16 activations")
    shared_weight_dtypes = (
        region.shared.gate_weight.dtype,
        region.shared.up_weight.dtype,
        region.shared.down_weight.dtype,
    )
    if any(dtype is not DType.BF16 for dtype in shared_weight_dtypes):
        reasons.append("shared gate, up, and down weights must all have BF16 dtype")
    routed_weight_dtypes = (
        region.routed.gate_weight.dtype,
        region.routed.up_weight.dtype,
        region.routed.down_weight.dtype,
    )
    if config.routed_precision is MoERoutedPrecision.MXFP8:
        reasons.append("MXFP8 scale tensor semantics are not modeled by the first MoK compiler slice")
    elif any(dtype is not DType.BF16 for dtype in routed_weight_dtypes):
        reasons.append("BF16 routed precision requires routed gate, up, and down weights to all have BF16 dtype")
    return tuple(reasons)


def _build_plan(region: _MoERegion, *, config: MoKOracleConfig) -> RegionPlan:
    tokens, hidden = region.shared.input.shape
    top_k = region.router.top_k
    local_experts, intermediate, _ = region.routed.gate_weight.shape
    global_experts = region.router.logits.shape[1]
    capacity_factor = max(2, ceil(config.expert_parallel_size * config.schedule_capacity_multiplier))
    capacity = tokens * top_k * capacity_factor
    prefix = region.combine.output.name
    names = {
        "all_gather": f"{prefix}.all_gathered_expert_indices",
        "peer_rank": f"{prefix}.schedule_peer_rank",
        "peer_token": f"{prefix}.schedule_peer_token_index",
        "padded_tokens": f"{prefix}.padded_token_count",
        "tokens_per_expert": f"{prefix}.tokens_per_local_expert",
        "dispatch_send": f"{prefix}.dispatch_send",
        "routed_input": f"{prefix}.routed_input",
        "combine_receive": f"{prefix}.combine_receive",
    }
    schedule = PaddedExpertSchedule(
        all_gathered_expert_indices=names["all_gather"],
        peer_rank=names["peer_rank"],
        peer_token_index=names["peer_token"],
        padded_token_count=names["padded_tokens"],
        tokens_per_local_expert=names["tokens_per_expert"],
        capacity=capacity,
        capacity_factor=capacity_factor,
        expert_padding=EXPERT_PADDING,
    )
    events = _readiness_events(prefix)
    tasks = _task_roles(region, names=names, events=events)
    gemm_k = 64 if config.routed_precision is MoERoutedPrecision.BF16 else 128
    skeleton = OpaqueMoKOracleSkeleton(
        name="persistent_shared_routed_expert_forward",
        input=region.shared.input.name,
        output=region.combine.output.name,
        router_logits=region.router.logits.name,
        expert_indices=region.router.expert_indices.name,
        router_weights=region.router.output.name,
        top_k=top_k,
        normalize_router_weights=region.router.normalize_weights,
        routed_precision=config.routed_precision.value,
        local_token_count=tokens,
        hidden_size=hidden,
        intermediate_size=intermediate,
        global_experts=global_experts,
        local_experts=local_experts,
        shared_experts=1,
        expert_parallel_size=config.expert_parallel_size,
        shared_gate_weight=region.shared.gate_weight.name,
        shared_up_weight=region.shared.up_weight.name,
        shared_down_weight=region.shared.down_weight.name,
        routed_gate_weight=region.routed.gate_weight.name,
        routed_up_weight=region.routed.up_weight.name,
        routed_down_weight=region.routed.down_weight.name,
        shared_gate_buffer=region.shared.gate.name,
        shared_up_buffer=region.shared.up.name,
        shared_hidden_buffer=region.shared.hidden.name,
        shared_output_buffer=region.shared.output.name,
        dispatch_send_buffer=names["dispatch_send"],
        routed_input_buffer=names["routed_input"],
        routed_gate_buffer=region.routed.gate.name,
        routed_up_buffer=region.routed.up.name,
        routed_hidden_buffer=region.routed.hidden.name,
        routed_output_buffer=region.routed.output.name,
        combine_receive_buffer=names["combine_receive"],
        swiglu_operation="pairwise_swiglu",
        schedule=schedule,
        readiness_events=events,
        task_roles=tasks,
        worker_roles=(
            PersistentWorkerRole(
                name="communication_cta",
                count=config.communication_sm_count,
                responsibilities=("dispatch", "combine", "overlap adjacent macrobatches"),
            ),
            PersistentWorkerRole(
                name="gemm_consumer_warpgroup",
                count=1,
                responsibilities=("WGMMA", "GEMM epilogue"),
            ),
            PersistentWorkerRole(
                name="gemm_producer_warpgroup",
                count=1,
                responsibilities=("TMA loads", "cluster launch-control scheduling"),
            ),
        ),
        communication_sm_count=config.communication_sm_count,
        minibatch_size=config.minibatch_size,
        macrobatch_size=config.macrobatch_size,
        cluster_size=2,
        threads_per_cluster_block=256,
        grouped_gemm_tile=(256, 256, gemm_k),
        swiglu_tile=(128, 128),
        dispatch_tile=(128, 512),
        combine_tile=(16, 1024),
        backend="mixture_of_kittens_sm100_sm103",
        backend_revision=MOK_BACKEND_REVISION,
    )
    materializations = _materializations(
        region,
        names=names,
        capacity=capacity,
        expert_parallel_size=config.expert_parallel_size,
        hidden=hidden,
        intermediate=intermediate,
    )
    explanation = RewriteExplanation(
        name="recover_persistent_shared_routed_expert_program",
        applied=True,
        original_fragment=(
            "router_logits = x @ router_weight",
            "expert_indices, router_weights = top_k(router_logits)",
            "shared = down_shared(silu(gate_shared(x)) * up_shared(x))",
            "routed = down_expert(silu(gate_expert(x)) * up_expert(x), expert_indices)",
            "output = shared + sum(router_weights * routed, axis=top_k)",
        ),
        transformed_fragment=(
            "all-gather top-k indices and build 256-padded expert segments",
            "dispatch routed tokens while shared gate/up GEMMs execute",
            "run shared and routed pairwise SwiGLU and down GEMMs as readiness permits",
            "combine routed outputs and router weights while persistent compute clusters progress",
        ),
        semantic_properties=(
            "each routed expert evaluation is independent before top-k combination",
            "gate and up projections share the same expert-grouped token rows",
            "SwiGLU is pairwise-local",
            "top-k combination is a per-token weighted reduction",
        ),
        legality_checks=(
            "global experts partition evenly across the expert-parallel ranks",
            "tokens, hidden, and intermediate dimensions satisfy the bounded tile family",
            "per-expert schedules use 256-row padded contiguous segments",
            "minibatch and macrobatch sizes satisfy persistent scheduling constraints",
            "shared and routed GEMMs accumulate in FP32",
        ),
        estimated_benefit=(
            "overlaps dispatch/combine communication with shared and routed expert computation and exposes "
            "tile readiness without separate framework operator launches"
        ),
        numerical_equivalence=NumericalEquivalence.ALGEBRAICALLY_EXACT,
        numerical_effect=(
            "grouped GEMM tiling and fused BF16 SwiGLU boundaries may reorder floating-point accumulation "
            "and conversion relative to the semantic graph"
        ),
    )
    return RegionPlan(skeletons=(skeleton,), materializations=materializations, rewrites=(explanation,))


def _readiness_events(prefix: str) -> tuple[ReadinessEvent, ...]:
    return (
        ReadinessEvent(
            name=f"{prefix}.x_routed_ready",
            producers=("dispatch",),
            consumers=("routed_gate_gemm", "routed_up_gemm"),
            granularity="minibatch",
        ),
        ReadinessEvent(
            name=f"{prefix}.gate_up_tile_ready",
            producers=("shared_gate_gemm", "shared_up_gemm", "routed_gate_gemm", "routed_up_gemm"),
            consumers=("shared_swiglu", "routed_swiglu"),
            granularity="output_tile",
        ),
        ReadinessEvent(
            name=f"{prefix}.hidden_row_block_ready",
            producers=("shared_swiglu", "routed_swiglu"),
            consumers=("shared_down_gemm", "routed_down_gemm"),
            granularity="row_block",
        ),
        ReadinessEvent(
            name=f"{prefix}.y_routed_ready",
            producers=("routed_down_gemm",),
            consumers=("combine",),
            granularity="minibatch",
        ),
        ReadinessEvent(
            name=f"{prefix}.y_routed_done",
            producers=("combine",),
            consumers=("dispatch", "routed_down_gemm"),
            granularity="macrobatch_buffer_reuse",
        ),
    )


def _task_roles(
    region: _MoERegion,
    *,
    names: dict[str, str],
    events: tuple[ReadinessEvent, ...],
) -> tuple[PersistentTaskRole, ...]:
    event = {item.name.rsplit(".", 1)[-1]: item.name for item in events}
    return (
        PersistentTaskRole(
            name="dispatch",
            placement=PersistentTaskPlacement.COMMUNICATION_SM,
            inputs=(region.shared.input.name, names["peer_rank"], names["peer_token"]),
            outputs=(names["dispatch_send"], names["routed_input"]),
            waits_for=(event["y_routed_done"],),
            signals=(event["x_routed_ready"],),
        ),
        PersistentTaskRole(
            name="shared_gate_gemm",
            placement=PersistentTaskPlacement.CLUSTER,
            inputs=(region.shared.input.name, region.shared.gate_weight.name),
            outputs=(region.shared.gate.name,),
            signals=(event["gate_up_tile_ready"],),
        ),
        PersistentTaskRole(
            name="shared_up_gemm",
            placement=PersistentTaskPlacement.CLUSTER,
            inputs=(region.shared.input.name, region.shared.up_weight.name),
            outputs=(region.shared.up.name,),
            signals=(event["gate_up_tile_ready"],),
        ),
        PersistentTaskRole(
            name="shared_swiglu",
            placement=PersistentTaskPlacement.CTA_LOCAL,
            inputs=(region.shared.gate.name, region.shared.up.name),
            outputs=(region.shared.hidden.name,),
            waits_for=(event["gate_up_tile_ready"],),
            signals=(event["hidden_row_block_ready"],),
        ),
        PersistentTaskRole(
            name="shared_down_gemm",
            placement=PersistentTaskPlacement.CLUSTER,
            inputs=(region.shared.hidden.name, region.shared.down_weight.name),
            outputs=(region.shared.output.name,),
            waits_for=(event["hidden_row_block_ready"],),
        ),
        PersistentTaskRole(
            name="routed_gate_gemm",
            placement=PersistentTaskPlacement.CLUSTER,
            inputs=(names["routed_input"], region.routed.gate_weight.name),
            outputs=(region.routed.gate.name,),
            waits_for=(event["x_routed_ready"],),
            signals=(event["gate_up_tile_ready"],),
        ),
        PersistentTaskRole(
            name="routed_up_gemm",
            placement=PersistentTaskPlacement.CLUSTER,
            inputs=(names["routed_input"], region.routed.up_weight.name),
            outputs=(region.routed.up.name,),
            waits_for=(event["x_routed_ready"],),
            signals=(event["gate_up_tile_ready"],),
        ),
        PersistentTaskRole(
            name="routed_swiglu",
            placement=PersistentTaskPlacement.CTA_LOCAL,
            inputs=(region.routed.gate.name, region.routed.up.name),
            outputs=(region.routed.hidden.name,),
            waits_for=(event["gate_up_tile_ready"],),
            signals=(event["hidden_row_block_ready"],),
        ),
        PersistentTaskRole(
            name="routed_down_gemm",
            placement=PersistentTaskPlacement.CLUSTER,
            inputs=(region.routed.hidden.name, region.routed.down_weight.name),
            outputs=(region.routed.output.name,),
            waits_for=(event["hidden_row_block_ready"], event["y_routed_done"]),
            signals=(event["y_routed_ready"],),
        ),
        PersistentTaskRole(
            name="combine",
            placement=PersistentTaskPlacement.COMMUNICATION_SM,
            inputs=(region.routed.output.name, region.router.output.name, names["peer_rank"], names["peer_token"]),
            outputs=(names["combine_receive"], region.combine.output.name),
            waits_for=(event["y_routed_ready"],),
            signals=(event["y_routed_done"],),
        ),
    )


def _materializations(
    region: _MoERegion,
    *,
    names: dict[str, str],
    capacity: int,
    expert_parallel_size: int,
    hidden: int,
    intermediate: int,
) -> tuple[MaterializationRecord, ...]:
    tokens = region.shared.input.shape[0]
    top_k = region.router.top_k

    def record(value: str, shape: tuple[int, ...], dtype: DType, reason: str) -> MaterializationRecord:
        return MaterializationRecord(
            value=value,
            shape=shape,
            dtype=dtype,
            disposition=MaterializationDisposition.MATERIALIZE,
            reason=reason,
        )

    return (
        record(region.router.logits.name, region.router.logits.shape, region.router.logits.dtype, "top-k router input"),
        record(
            region.router.expert_indices.name,
            region.router.expert_indices.shape,
            region.router.expert_indices.dtype,
            "all-gathered routing metadata",
        ),
        record(region.router.output.name, region.router.output.shape, DType.FP32, "weighted expert combine input"),
        record(names["all_gather"], (expert_parallel_size, tokens, top_k), DType.INT32, "symmetric route gather"),
        record(names["peer_rank"], (capacity,), DType.INT32, "bounded padded expert schedule"),
        record(names["peer_token"], (capacity,), DType.INT32, "bounded padded expert schedule"),
        record(names["padded_tokens"], (1,), DType.INT32, "runtime padded row count"),
        record(
            names["tokens_per_expert"],
            (region.routed.gate_weight.shape[0],),
            DType.INT32,
            "256-padded local-expert segments",
        ),
        record(names["dispatch_send"], (tokens * top_k, hidden), DType.BF16, "symmetric dispatch workspace"),
        record(names["routed_input"], (capacity, hidden), DType.BF16, "expert-grouped dispatched rows"),
        record(region.shared.gate.name, (tokens, intermediate), DType.BF16, "shared gate forward context"),
        record(region.shared.up.name, (tokens, intermediate), DType.BF16, "shared up forward context"),
        record(region.shared.hidden.name, (tokens, intermediate), DType.BF16, "shared down-projection input"),
        record(region.shared.output.name, (tokens, hidden), DType.BF16, "shared contribution to final epilogue"),
        record(region.routed.gate.name, (capacity, intermediate), DType.BF16, "routed gate forward context"),
        record(region.routed.up.name, (capacity, intermediate), DType.BF16, "routed up forward context"),
        record(region.routed.hidden.name, (capacity, intermediate), DType.BF16, "routed down-projection input"),
        record(region.routed.output.name, (capacity, hidden), DType.BF16, "routed down output before combine"),
        record(names["combine_receive"], (tokens * top_k, hidden), DType.BF16, "symmetric combine workspace"),
        record(region.combine.output.name, region.combine.output.shape, region.combine.output.dtype, "region output"),
    )
