# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Research compiler for Transformer tile-lifetime execution plans."""

from tile_lifetime.attention import (
    AttentionPartial,
    NormalizedAttentionPartial,
    compile_attention_region,
    finalize_attention_partial,
    finalize_normalized_attention_partial,
    merge_attention_partials,
    merge_normalized_attention_partials,
    normalize_attention_partial,
    summarize_attention_partial,
)
from tile_lifetime.compiler import RMSScalePlacement, RowScalePlacement, compile_erased_dense_program, compile_region
from tile_lifetime.dense_algebra import DenseSemanticErasureError, erase_dense_semantics
from tile_lifetime.dense_flow import (
    DenseFlowOperation,
    ErasedDenseFlowProgram,
    FlowContract,
    FlowDomainRestriction,
    FlowFold,
    FlowMap,
    FlowMapIteration,
    FlowValue,
    dense_flow_scheduling_keys,
    erase_dense_transformer_semantics,
    pairwise_product_expression,
    pairwise_silu_product_expression,
    validate_erased_dense_flow,
)
from tile_lifetime.dense_flow_planner import compile_erased_dense_transformer_region
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
    MapFoldSemantics,
    ReadinessGranularity,
    TileStorage,
    TransportSelection,
    TransportSemantics,
)
from tile_lifetime.gated_delta_scan import (
    GatedDeltaScanCompilation,
    chunkwise_gated_delta_reference,
    compile_gated_delta_scan,
    recurrent_gated_delta_reference,
    summarize_gated_delta_chunk,
)
from tile_lifetime.gemm_program import GENERIC_H100_GEMM_BACKEND, GemmProgram, compile_gemm_program
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
    MoKOracleConfig,
    compile_mok_oracle_region,
)
from tile_lifetime.moe_recovery import MoESemanticRecoveryError, RecoveredMoERegion, recover_moe_region
from tile_lifetime.msa_frontend import MSA_INPUT_NAMES, MSADebugConfig, export_debug_msa, msa_region
from tile_lifetime.msa_recovery import (
    NaturalProjectedRoutedAttentionCompilation,
    ProjectedRoutedAttentionRecoveryError,
    RecoveredProjectedRoutedAttentionProgram,
    compile_natural_projected_routed_attention,
    recover_projected_routed_attention_program,
)
from tile_lifetime.pipeline import (
    compile_stablehlo_attention_region,
    compile_stablehlo_dense_transformer_region,
    compile_stablehlo_expert_parallel_region,
    compile_stablehlo_projected_routed_attention_program,
    compile_stablehlo_rms_attention_program,
    compile_stablehlo_rms_region,
    compile_stablehlo_routed_attention_program,
    compile_stablehlo_streaming_attention_program,
    recover_stablehlo_moe_region,
    recover_stablehlo_projected_routed_attention_program,
    recover_stablehlo_routed_attention_program,
)
from tile_lifetime.plan import (
    AttachmentSite,
    ChunkSummaryRepresentation,
    GemmSkeleton,
    MaterializationDisposition,
    NumericalPolicy,
    OpaqueMoKOracleSkeleton,
    ReductionSkeleton,
    ScanNumericalContract,
    SemanticErasureReport,
    SemanticLoweringStep,
    StatefulScanExecutionForm,
    StatefulScanSkeleton,
    StateTransitionStructure,
    StreamingAttentionSkeleton,
    TransformSkeleton,
)
from tile_lifetime.qkv_rope import compile_qkv_rope_attention_region
from tile_lifetime.relation import (
    OrderedRelationFoldProgram,
    RelationPlan,
    RelationPlanError,
    build_expert_parallel_relation_plan,
    build_partitioned_merge_rows,
    build_relation_plan,
    compile_ordered_relation_fold,
)
from tile_lifetime.routed_attention import (
    IndexDomainRestriction,
    ProjectedBlockSelectionProgram,
    RelationSelectionProgram,
    SelectionOutputOrder,
    SelectionResult,
    SelectionSemantics,
    SelectionTieBreak,
    UnderfilledSelectionPolicy,
    build_grouped_routed_attention_relation,
    build_routed_attention_relation,
    execute_kv_major_attention,
    execute_projected_block_selection,
    execute_query_major_attention,
    execute_relation_selection,
    make_causal_block_relation,
    routed_attention_reference,
)
from tile_lifetime.routed_attention_frontend import (
    ROUTED_ATTENTION_INPUT_NAMES,
    RoutedAttentionDebugConfig,
    export_debug_routed_attention,
    routed_attention_region,
)
from tile_lifetime.routed_attention_plan import (
    BoundedKVReusePlan,
    BoundedKVReuseWave,
    QueryMajorBlockIndexPlan,
    RoutedAttentionOrientation,
    RoutedAttentionPhysicalPlan,
    RoutedAttentionPlanConfig,
    RoutedStreamingAttentionCompilation,
    bounded_kv_reuse_plan,
    compile_bounded_kv_major_candidate,
    compile_routed_attention_candidates,
    compile_routed_streaming_attention_candidates,
    query_major_block_index_plan,
)
from tile_lifetime.routed_attention_recovery import (
    NaturalRoutedAttentionCompilation,
    RecoveredRoutedAttentionProgram,
    RoutedAttentionRecoveryError,
    compile_natural_routed_attention,
    recover_routed_attention_program,
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
from tile_lifetime.semantic_erasure import (
    ErasedTensorProgram,
    SemanticErasureError,
    build_tensor_erasure_report,
    semantic_erasure_errors,
    tensor_program_scheduling_keys,
    validate_erased_tensor_program,
    validate_plan_semantic_erasure,
)
from tile_lifetime.sm100_projected_routed_lowering import (
    LoweredAffineIndexDomain,
    SM100ProjectedRoutedCandidate,
    SM100ProjectedRoutedCandidateSet,
    lower_sm100_projected_routed_candidates,
)
from tile_lifetime.sm100_routed_lowering import (
    SM100RelationOrientation,
    SM100RoutedSchedule,
    SM100RoutedStreamingLowering,
    default_sm100_routed_schedules,
    lower_sm100_routed_streaming_program,
)
from tile_lifetime.sm100_selection_lowering import (
    SM100ProjectedSelectionLowering,
    SM100SelectionSchedule,
    SM100SelectionStrategy,
    default_sm100_selection_schedules,
    lower_sm100_projected_selection,
)
from tile_lifetime.stablehlo_scan_recovery import (
    StableHLOScanRecoveryError,
    StableHLOStatefulScanCompilation,
    compile_stablehlo_stateful_scan,
    stateful_scan_scheduling_keys,
    validate_stateful_scan_semantic_erasure,
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
    solve_factored_affine_chunk,
    summarize_factored_affine_chunk,
)
from tile_lifetime.stateful_scan_reference import (
    STATEFUL_SCAN_INPUT_NAMES,
    ScanDecayAxes,
    StatefulScanDebugConfig,
    export_debug_stateful_scan,
    stateful_scan_region,
)
from tile_lifetime.streaming_attention import (
    AttentionScoreAxis,
    OnlineAttentionState,
    ScoreInputSpec,
    ScoreMapSpec,
    StreamingAttentionProgram,
    StreamingAttentionStage,
    StreamingTileSchedule,
    add_score_bias,
    apply_arbitrary_score_mask,
    apply_causal_score_mask,
    apply_tanh_softcap,
    build_attention_tensor_program,
    derive_streaming_attention,
    execute_streaming_attention,
    execute_tensor_program,
    scaled_score_map,
    streaming_attention_from_semantic_operation,
)
from tile_lifetime.swiglu import compile_swiglu_region
from tile_lifetime.tensor_program import (
    AxisIndexMap,
    ContractPrimitive,
    FoldPrimitive,
    FoldReducer,
    MapPrimitive,
    ProgramValue,
    ScalarExpression,
    ScalarExpressionKind,
    TensorAxis,
    TensorProgram,
)
from tile_lifetime.tile_program import (
    TileOp,
    TilePrimitive,
    TileProgram,
    TileProgramError,
    TileProgramStage,
    ValueLifetime,
    optimize_tile_program,
)
from tile_lifetime.tiled_fold_finalize import (
    FoldDenominatorPolicy,
    FoldFeatureLayout,
    FoldPartialAddressing,
    FoldPartialOrder,
    FoldPhysicalAxis,
    FoldReassociationPolicy,
    FoldScalarReduction,
    TiledFoldAxes,
    TiledFoldFinalizeProgram,
    TiledFoldFinalizeSchedule,
    TiledFoldFinalizeSemantics,
    TiledFoldInputLayout,
    deterministic_weighted_sum_fold_program,
    evaluate_tiled_fold_finalize,
    normalized_exponential_fold_program,
)
