# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Research compiler for Transformer tile-lifetime execution plans."""

from tile_lifetime.attention import (
    AttentionPartial,
    compile_attention_region,
    finalize_attention_partial,
    merge_attention_partials,
    summarize_attention_partial,
)
from tile_lifetime.compiler import RMSScalePlacement, compile_region
from tile_lifetime.dense_region import compile_dense_transformer_region
from tile_lifetime.expert_parallel import (
    ExpertParallelConfig,
    ExpertParallelLegalityError,
    compile_expert_parallel_region,
)
from tile_lifetime.expert_parallel_plan import (
    ExchangeRowMode,
    ExpertMaterializationSchedule,
    ExpertOverlapPolicy,
    ExpertParallelPlan,
    ExpertParallelStageKind,
    GateUpPhysicalLayout,
    ReadinessGranularity,
    TileStorage,
)
from tile_lifetime.gated_delta_scan import (
    GatedDeltaScanCompilation,
    chunkwise_gated_delta_reference,
    compile_gated_delta_scan,
    recurrent_gated_delta_reference,
    summarize_gated_delta_chunk,
)
from tile_lifetime.ir import DType, TensorGraph
from tile_lifetime.kimi_delta_scan import (
    KimiDeltaScanCompilation,
    chunkwise_kimi_delta_reference,
    compile_kimi_delta_scan,
    recurrent_kimi_delta_reference,
    summarize_kimi_delta_chunk,
)
from tile_lifetime.moe import (
    MoELegalityError,
    MoERoutedPrecision,
    MoKCompilerConfig,
    compile_mok_expert_parallel_region,
)
from tile_lifetime.moe_recovery import MoESemanticRecoveryError, RecoveredMoERegion, recover_moe_region
from tile_lifetime.pipeline import (
    compile_stablehlo_attention_region,
    compile_stablehlo_coda_fa3_program,
    compile_stablehlo_dense_transformer_region,
    compile_stablehlo_rms_region,
    recover_stablehlo_moe_region,
)
from tile_lifetime.plan import (
    AttachmentSite,
    ChunkSummaryRepresentation,
    ExpertParallelMoESkeleton,
    GemmSkeleton,
    MaterializationDisposition,
    NumericalPolicy,
    ReductionSkeleton,
    ScanNumericalContract,
    StatefulScanExecutionForm,
    StatefulScanSkeleton,
    StateTransitionStructure,
    StreamingAttentionSkeleton,
    TransformSkeleton,
)
from tile_lifetime.qkv_rope import compile_qkv_rope_attention_region
from tile_lifetime.relation import (
    RelationPlan,
    RelationPlanError,
    build_expert_parallel_relation_plan,
    build_relation_plan,
)
from tile_lifetime.routed_attention import (
    build_routed_attention_relation,
    execute_kv_major_attention,
    execute_query_major_attention,
    make_causal_block_relation,
    routed_attention_reference,
)
from tile_lifetime.routed_attention_plan import (
    RoutedAttentionOrientation,
    RoutedAttentionPhysicalPlan,
    RoutedAttentionPlanConfig,
    compile_bounded_kv_major_candidate,
    compile_routed_attention_candidates,
)
from tile_lifetime.runtime import (
    PlanRuntimeError,
    RuntimeBufferSpec,
    RuntimeDiagnostic,
    RuntimeDiagnosticCode,
    RuntimeResult,
    TensorBinding,
    execute_region_plan,
    required_input_specs,
    validate_region_plan,
)
from tile_lifetime.stateful_scan import (
    AffineChunkSummary,
    AffineStateTransform,
    ChunkAlgebra,
    LogicalAxis,
    ScanPrimitive,
    ScanPrimitiveKind,
    ScanValue,
    ScanValueRole,
    StatefulScan,
    TensorExpression,
    TensorExpressionKind,
    apply_affine_transform,
    binary_expression,
    compose_affine_transforms,
    contract_expression,
    explain_stateful_scan,
    input_expression,
    summarize_affine_sequence,
    unary_expression,
)
from tile_lifetime.stateful_scan_planner import compile_affine_scan_candidates
from tile_lifetime.stateful_scan_recovery import (
    AffineTensorExpression,
    AppliedLinearMap,
    FactoredAffineChunkSummary,
    RecoveredAffineStateUpdate,
    StateLinearTerm,
    apply_factored_affine_chunk,
    execute_recurrent_factored_affine,
    recover_affine_state_update,
    summarize_factored_affine_chunk,
)
from tile_lifetime.swiglu import compile_swiglu_region
