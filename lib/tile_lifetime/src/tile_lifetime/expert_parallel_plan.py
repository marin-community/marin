# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral plan contracts for first-principles expert parallelism."""

from dataclasses import dataclass
from enum import StrEnum

from shuttle.ir import DType
from tile_lifetime.plan import MaterializationRecord, RewriteExplanation
from tile_lifetime.tensor_program import ScalarExpression
from tile_lifetime.tile_program import TileProgram


class ExpertParallelStageKind(StrEnum):
    """Semantic or tiled stage in a generic expert-parallel lowering."""

    ROUTE_RELATION = "route_relation"
    EXPERT_OWNERSHIP = "expert_ownership"
    LEGALIZE_GATE_UP_LAYOUT = "legalize_gate_up_layout"
    GROUP_BY_OWNER = "group_by_owner"
    PROJECT_EXCHANGE_ROWS = "project_exchange_rows"
    FORWARD_EXCHANGE = "forward_exchange"
    EXPAND_LOCAL_ASSIGNMENTS = "expand_local_assignments"
    SEGMENT_BY_LOCAL_EXPERT = "segment_by_local_expert"
    PAD_LOCAL_SEGMENTS = "pad_local_segments"
    SHARED_DENSE_GATE_UP = "shared_dense_gate_up"
    ROUTED_SEGMENTED_GATE_UP = "routed_segmented_gate_up"
    SHARED_PAIRWISE_SWIGLU = "shared_pairwise_swiglu"
    ROUTED_PAIRWISE_SWIGLU = "routed_pairwise_swiglu"
    SHARED_DENSE_DOWN = "shared_dense_down"
    ROUTED_SEGMENTED_DOWN = "routed_segmented_down"
    REVERSE_EXCHANGE = "reverse_exchange"
    WEIGHTED_SCATTER_REDUCE = "weighted_scatter_reduce"
    SHARED_ADD = "shared_add"


class ReadinessGranularity(StrEnum):
    """Smallest unit that permits a tile-flow consumer to start."""

    RELATION_ROW = "relation_row"
    EXPERT_SEGMENT = "expert_segment"
    PADDED_SEGMENT_TILE = "padded_segment_tile"
    MINIBATCH = "minibatch"
    OUTPUT_TILE = "output_tile"
    ROW_TILE = "row_tile"
    TOKEN_TILE = "token_tile"


class TileStorage(StrEnum):
    """Storage/address-space contract selected for a tile-flow edge."""

    INPUT_ALIAS = "input_alias"
    SHARDED_PARAMETER_VIEW = "sharded_parameter_view"
    RELATION_BUFFER = "relation_buffer"
    GLOBAL_BUFFER = "global_buffer"


class ExchangeRowMode(StrEnum):
    """Relation projection used to form activation exchange rows."""

    ASSIGNMENT = "assignment"
    COALESCED_TOKEN_OWNER = "coalesced_token_owner"


class GateUpPhysicalLayout(StrEnum):
    """Physical gate/up weight and contraction-output organization."""

    SEPARATE_E_I_K = "separate_e_i_k"
    CONCATENATED_E_2I_K = "concatenated_e_2i_k"
    INTERLEAVED_E_2I_K = "interleaved_e_2i_k"


class ExpertOverlapPolicy(StrEnum):
    """Overlap choice for shared compute and routed-token dispatch."""

    SEQUENTIAL = "sequential"
    SHARED_WITH_ASYNC_DISPATCH = "shared_with_async_dispatch"


class ExpertMaterializationSchedule(StrEnum):
    """Materialization policy at expert-stage boundaries."""

    TILE_FLOW_BOUNDARIES = "tile_flow_boundaries"
    COARSE_ACTIVATION_BOUNDARIES = "coarse_activation_boundaries"


class TransportSemantics(StrEnum):
    """Semantic work performed by one communication implementation."""

    PAYLOAD_PERMUTATION = "payload_permutation"
    PAYLOAD_PERMUTATION_AND_REDUCTION = "payload_permutation_and_reduction"


@dataclass(frozen=True)
class TransportSelection:
    """Physical transport choice and the semantic work it subsumes."""

    implementation: str
    semantics: TransportSemantics


@dataclass(frozen=True)
class RouteRelation:
    """Relational view of each token-to-expert assignment."""

    name: str
    token_count: int
    slots_per_token: int
    global_expert_count: int
    token_column: str
    slot_column: str
    global_expert_column: str
    weight_column: str
    source_expert_indices: str
    source_weights: str


@dataclass(frozen=True)
class ExpertOwnership:
    """Contiguous mapping from global experts to EP ranks and local experts."""

    global_expert_count: int
    expert_parallel_size: int
    local_expert_count: int
    owner_rank_expression: str
    local_expert_expression: str

    def owner(self, global_expert: int) -> tuple[int, int]:
        """Return the contiguous owner-rank and local-expert coordinates."""
        if not 0 <= global_expert < self.global_expert_count:
            raise ValueError(f"global expert must be in [0, {self.global_expert_count}), got {global_expert}")
        return divmod(global_expert, self.local_expert_count)


@dataclass(frozen=True)
class ExpertSegmentContract:
    """Grouping and padding contract for routed assignments."""

    keys: tuple[str, ...]
    stable_order: tuple[str, ...]
    segment_count: int
    padding_quantum: int
    padded_token: int
    padded_weight: float


@dataclass(frozen=True)
class ExpertCapacityPolicy:
    """Bounded receive capacity with an exact-semantics overflow guard."""

    capacity_factor: float
    receiver_assignment_capacity: int
    padded_local_capacity: int
    overflow_policy: str


@dataclass(frozen=True)
class ExchangeRelationProjection:
    """One legal mapping from route assignments to exchanged activation rows."""

    mode: ExchangeRowMode
    grouping_keys: tuple[str, ...]
    activation_rows: str
    metadata_rows: str
    receiver_expansion: str


@dataclass(frozen=True)
class GateUpLayoutContract:
    """Legalization from separate semantic weights to a physical contraction layout."""

    semantic_weight_layout: str
    semantic_output_layout: str
    selected: GateUpPhysicalLayout
    candidates: tuple[GateUpPhysicalLayout, ...]
    legalization: str


@dataclass(frozen=True)
class WorkerPool:
    """A finite pool assigned to one or more generic stages."""

    name: str
    workers: int
    stages: tuple[ExpertParallelStageKind, ...]


@dataclass(frozen=True)
class PipelineDepth:
    """Pipeline depth associated with one generic stage family."""

    name: str
    stages: tuple[ExpertParallelStageKind, ...]
    depth: int


@dataclass(frozen=True)
class ExpertParallelSchedule:
    """Bounded physical choices without binding to one megakernel implementation."""

    expert_parallel_size: int
    segment_padding: int
    contraction_tile: tuple[int, int, int]
    swiglu_tile: tuple[int, int]
    exchange_tile: tuple[int, int]
    scatter_tile: tuple[int, int]
    worker_pools: tuple[WorkerPool, ...]
    minibatch_size: int
    macrobatch_size: int
    pipelines: tuple[PipelineDepth, ...]
    exchange_implementation: str
    exchange_implementation_candidates: tuple[str, ...]
    forward_transport: TransportSelection
    reverse_transport: TransportSelection
    merge_implementation: str
    segmented_contraction_implementation: str
    segmented_contraction_candidates: tuple[str, ...]
    exchange_worker_candidates: tuple[int, ...]
    overlap_policy: ExpertOverlapPolicy
    overlap_policy_candidates: tuple[ExpertOverlapPolicy, ...]
    materialization_schedule: ExpertMaterializationSchedule
    materialization_schedule_candidates: tuple[ExpertMaterializationSchedule, ...]


@dataclass(frozen=True)
class ExpertParallelStage:
    """One separately inspectable stage in the expert-parallel dataflow."""

    name: str
    kind: ExpertParallelStageKind
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    operation: str
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class MapFoldSemantics:
    """Backend-neutral scalar bodies assigned to generic Map/Fold skeletons."""

    pair_map: ScalarExpression
    fold_contribution: ScalarExpression
    fold_update: ScalarExpression
    post_fold_map: ScalarExpression
    explicit_rounding_functions: frozenset[str]


@dataclass(frozen=True)
class TileFlowEdge:
    """A logical tiled value flowing between expert-parallel stages."""

    value: str
    shape: tuple[int, ...]
    dtype: DType
    producer: str
    consumers: tuple[str, ...]
    logical_layout: str
    tile_shape: tuple[int, ...] | None
    storage: TileStorage
    readiness: ReadinessGranularity
    fanout: int
    alias_of: str | None = None


@dataclass(frozen=True)
class BufferLifetime:
    """Buffer allocation and live interval derived from a tile-flow edge."""

    value: str
    shape: tuple[int, ...]
    dtype: DType
    logical_layout: str
    tile_shape: tuple[int, ...] | None
    storage: TileStorage
    live_from: str
    live_until: str
    alias_of: str | None


@dataclass(frozen=True)
class ExpertParallelPlan:
    """Generic EP stages, tile flow, and derived storage decisions."""

    route_relation: RouteRelation
    ownership: ExpertOwnership
    segments: ExpertSegmentContract
    capacity: ExpertCapacityPolicy
    exchange_projections: tuple[ExchangeRelationProjection, ...]
    selected_exchange_projection: ExchangeRelationProjection
    gate_up_layout: GateUpLayoutContract
    schedule: ExpertParallelSchedule
    map_fold_semantics: MapFoldSemantics
    merge_program: TileProgram
    stages: tuple[ExpertParallelStage, ...]
    tile_flows: tuple[TileFlowEdge, ...]
    buffers: tuple[BufferLifetime, ...]
    materializations: tuple[MaterializationRecord, ...]
    rewrites: tuple[RewriteExplanation, ...]

    def stage(self, kind: ExpertParallelStageKind) -> ExpertParallelStage:
        """Return the unique stage of one kind."""
        matches = tuple(stage for stage in self.stages if stage.kind is kind)
        if len(matches) != 1:
            raise KeyError(f"expected one {kind.value} stage, found {len(matches)}")
        return matches[0]

    def flows_to(self, stage: str) -> tuple[TileFlowEdge, ...]:
        """Return all tile-flow edges consumed by a stage."""
        return tuple(edge for edge in self.tile_flows if stage in edge.consumers)
