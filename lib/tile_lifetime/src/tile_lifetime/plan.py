# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspectable execution-plan representation."""

from dataclasses import dataclass
from enum import StrEnum

from shuttle.ir import DType


class NumericalPolicy(StrEnum):
    """Finite-precision transformations permitted by a compilation."""

    BITWISE_EXACT = "bitwise_exact"
    ALLOW_ROUNDING_REORDER = "allow_rounding_reorder"


class NumericalEquivalence(StrEnum):
    """Numerical relationship between source and transformed programs."""

    BITWISE_EXACT = "bitwise_exact"
    ALGEBRAICALLY_EXACT = "algebraically_exact"


@dataclass(frozen=True)
class SemanticLoweringStep:
    """One named frontend construct erased into generic semantic primitives."""

    source_semantic: str
    generic_primitives: tuple[str, ...]


@dataclass(frozen=True)
class SemanticErasureReport:
    """Machine-readable evidence that scheduling consumed only generic algebra."""

    source_semantics: tuple[str, ...]
    lowering_steps: tuple[SemanticLoweringStep, ...]
    scheduling_keys: tuple[str, ...]
    validation_errors: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        """Whether the erased program passed structural name-erasure validation."""
        return not self.validation_errors


class AttachmentSite(StrEnum):
    """Tile lifetime where an operation executes."""

    GEMM_PROLOGUE = "gemm_prologue"
    GEMM_EPILOGUE = "gemm_epilogue"
    ATTENTION_SCORE_TRANSFORM = "attention_score_transform"
    ATTENTION_ONLINE_UPDATE = "attention_online_update"
    ATTENTION_OUTPUT_TRANSFORM = "attention_output_transform"
    AUXILIARY_REDUCTION = "auxiliary_reduction"
    MATERIALIZED_TRANSFORM = "materialized_transform"


class MaterializationDisposition(StrEnum):
    """Physical disposition of a logical or synthesized value."""

    MATERIALIZE = "materialize"
    ALIAS = "alias"
    PROLOGUE_ONLY = "prologue_only"
    EPILOGUE_ONLY = "epilogue_only"
    PARTIAL_REDUCTION_ONLY = "partial_reduction_only"
    INTERNAL_ATTENTION_STATE = "internal_attention_state"


@dataclass(frozen=True)
class Attachment:
    """A semantic operation placed in a skeleton's tile lifetime."""

    operation: str
    site: AttachmentSite
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class GemmSkeleton:
    """Fixed GEMM mainloop with a programmable epilogue."""

    name: str
    input: str
    weight: str
    output: str
    shape: tuple[int, int, int]
    accumulation_dtype: DType
    backend: str | None = None
    input_layout: str | None = None
    output_layout: str | None = None
    physical_tile_shape: tuple[int, int, int] | None = None
    cluster_shape: tuple[int, int, int] | None = None
    pingpong: bool | None = None
    prologue: tuple[Attachment, ...] = ()
    epilogue: tuple[Attachment, ...] = ()


@dataclass(frozen=True)
class ReductionSkeleton:
    """Small reduction over tile partial statistics."""

    name: str
    input: str
    output: str
    operator: str
    reduction_dtype: DType
    auxiliary_inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class StreamingAttentionSkeleton:
    """Finite-family exact online-softmax attention skeleton."""

    name: str
    query: str
    key: str
    value: str
    output: str
    score_value: str
    probability_value: str
    query_block_size: int
    key_value_block_size: int
    head_dimension: int
    query_heads: int
    key_value_heads: int
    causal: bool
    scale: float
    backend: str
    input_layout: str
    output_layout: str
    pipeline_stages: int
    producer_threads: int
    consumer_threads: int
    pack_gqa: bool
    mma_pv_is_rs: bool
    intra_warpgroup_overlap: bool
    persistent_scheduler: bool
    register_estimate: int | None
    online_state: tuple[str, ...]
    attachments: tuple[Attachment, ...] = ()


@dataclass(frozen=True)
class TransformSkeleton:
    """Fallback materialized tensor transformation."""

    name: str
    operation: str
    inputs: tuple[str, ...]
    output: str


class ScanNumericalContract(StrEnum):
    """Finite-precision relationship between a scan lowering and its source."""

    SOURCE_ORDERED = "source_ordered"
    ORDERED_FP = "ordered_fp"
    BOUNDED_REASSOCIATION = "bounded_reassociation"
    REAL_ALGEBRA_EQUIVALENT = "real_algebra_equivalent"


class StatefulScanExecutionForm(StrEnum):
    """Physical traversal selected for one ordered state program."""

    RECURRENT = "recurrent"
    CHUNKWISE = "chunkwise"


class ChunkSummaryRepresentation(StrEnum):
    """Representation used to carry a state transform across chunk boundaries."""

    NONE = "none"
    FULL_AFFINE = "full_affine"
    FACTORED_AFFINE = "factored_affine"


class StateTransitionStructure(StrEnum):
    """Physical factor family recovered from an affine state update."""

    DIAGONAL = "diagonal"
    DIAGONAL_PLUS_LOW_RANK = "diagonal_plus_low_rank"
    GENERAL_AFFINE = "general_affine"


@dataclass(frozen=True)
class StatefulScanSkeleton:
    """One physical candidate for an ordered stateful computation."""

    name: str
    ordered_axis: str
    length: int
    state: str
    state_shape: tuple[int, ...]
    state_dtype: DType
    output: str
    execution_form: StatefulScanExecutionForm
    chunk_size: int
    summary_representation: ChunkSummaryRepresentation
    transition_structure: StateTransitionStructure
    maximum_update_rank: int
    backend: str
    backend_revision: str | None
    state_layout: str
    materialized_values: tuple[str, ...]
    numerical_contract: ScanNumericalContract
    numerical_effect: str


class PersistentTaskPlacement(StrEnum):
    """Physical worker domain assigned to one persistent MoE task."""

    COMMUNICATION_SM = "communication_sm"
    CLUSTER = "cluster"
    CTA_LOCAL = "cta_local"


@dataclass(frozen=True)
class PersistentTaskRole:
    """One visible task in a bounded persistent schedule."""

    name: str
    placement: PersistentTaskPlacement
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    waits_for: tuple[str, ...] = ()
    signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class PersistentWorkerRole:
    """A specialized worker group used by a persistent schedule."""

    name: str
    count: int
    responsibilities: tuple[str, ...]


@dataclass(frozen=True)
class ReadinessEvent:
    """Counted, generation-scoped readiness connecting physical tasks."""

    name: str
    producers: tuple[str, ...]
    consumers: tuple[str, ...]
    granularity: str
    required_arrivals: int | tuple[int, ...] | str = 1
    generation: str = "region_invocation"
    memory_order_scope: str = "device"


@dataclass(frozen=True)
class BoundedBuffer:
    """Finite storage whose reuse is tied to consumer completion."""

    name: str
    item_domain: str
    capacity_items: int
    size_bytes: int
    producer: str
    consumers: tuple[str, ...]
    reuse_after: str
    placement: str


@dataclass(frozen=True)
class PaddedExpertSchedule:
    """Bounded dispatch schedule grouped into padded local-expert segments."""

    all_gathered_expert_indices: str
    peer_rank: str
    peer_token_index: str
    padded_token_count: str
    tokens_per_local_expert: str
    capacity: int
    capacity_factor: int
    expert_padding: int


@dataclass(frozen=True)
class OpaqueMoKOracleSkeleton:
    """Complete MoK kernel contract retained only as a comparison oracle."""

    name: str
    input: str
    output: str
    router_logits: str
    expert_indices: str
    router_weights: str
    top_k: int
    normalize_router_weights: bool
    routed_precision: str
    local_token_count: int
    hidden_size: int
    intermediate_size: int
    global_experts: int
    local_experts: int
    shared_experts: int
    expert_parallel_size: int
    shared_gate_weight: str
    shared_up_weight: str
    shared_down_weight: str
    routed_gate_weight: str
    routed_up_weight: str
    routed_down_weight: str
    shared_gate_buffer: str
    shared_up_buffer: str
    shared_hidden_buffer: str
    shared_output_buffer: str
    dispatch_send_buffer: str
    routed_input_buffer: str
    routed_gate_buffer: str
    routed_up_buffer: str
    routed_hidden_buffer: str
    routed_output_buffer: str
    combine_receive_buffer: str
    swiglu_operation: str
    schedule: PaddedExpertSchedule
    readiness_events: tuple[ReadinessEvent, ...]
    task_roles: tuple[PersistentTaskRole, ...]
    worker_roles: tuple[PersistentWorkerRole, ...]
    communication_sm_count: int
    minibatch_size: int
    macrobatch_size: int
    cluster_size: int
    threads_per_cluster_block: int
    grouped_gemm_tile: tuple[int, int, int]
    swiglu_tile: tuple[int, int]
    dispatch_tile: tuple[int, int]
    combine_tile: tuple[int, int]
    backend: str
    backend_revision: str


ExecutionSkeleton = (
    GemmSkeleton
    | ReductionSkeleton
    | StreamingAttentionSkeleton
    | TransformSkeleton
    | StatefulScanSkeleton
    | OpaqueMoKOracleSkeleton
)


@dataclass(frozen=True)
class MaterializationRecord:
    """Disposition selected for one logical or synthesized value."""

    value: str
    shape: tuple[int, ...]
    dtype: DType
    disposition: MaterializationDisposition
    reason: str
    alias_of: str | None = None


@dataclass(frozen=True)
class RewriteExplanation:
    """Structured proof and tradeoff record for one rewrite."""

    name: str
    applied: bool
    original_fragment: tuple[str, ...]
    transformed_fragment: tuple[str, ...]
    semantic_properties: tuple[str, ...]
    legality_checks: tuple[str, ...]
    estimated_benefit: str
    numerical_equivalence: NumericalEquivalence
    numerical_effect: str
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegionPlan:
    """Selected skeleton sequence, materializations, and rewrite evidence."""

    skeletons: tuple[ExecutionSkeleton, ...]
    materializations: tuple[MaterializationRecord, ...]
    rewrites: tuple[RewriteExplanation, ...]
    semantic_erasure_report: SemanticErasureReport | None = None

    @property
    def activation_materializations(self) -> tuple[MaterializationRecord, ...]:
        """Return activation-sized values written to global memory."""
        return tuple(
            record
            for record in self.materializations
            if record.disposition is MaterializationDisposition.MATERIALIZE and len(record.shape) >= 2
        )

    @property
    def sequence_squared_materializations(self) -> tuple[MaterializationRecord, ...]:
        """Return materialized logical scores or probabilities from attention skeletons."""
        values = {
            value
            for skeleton in self.skeletons
            if isinstance(skeleton, StreamingAttentionSkeleton)
            for value in (skeleton.score_value, skeleton.probability_value)
        }
        return tuple(
            record
            for record in self.materializations
            if record.value in values and record.disposition is MaterializationDisposition.MATERIALIZE
        )

    def materialization(self, value: str) -> MaterializationRecord:
        """Return the disposition for one named value."""
        matches = tuple(record for record in self.materializations if record.value == value)
        if len(matches) != 1:
            raise KeyError(f"expected one materialization record for {value!r}, found {len(matches)}")
        return matches[0]
