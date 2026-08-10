# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Research compiler for Transformer tile-lifetime execution plans."""

from tile_lifetime.attention import (
    AttentionPartial,
    NormalizedAttentionPartial,
    finalize_attention_partial,
    finalize_normalized_attention_partial,
    merge_attention_partials,
    merge_normalized_attention_partials,
    normalize_attention_partial,
    summarize_attention_partial,
)
from tile_lifetime.autodiff import (
    BackwardTensorProgram,
    DifferentiatedTensorProgram,
    differentiate_scalar_expression,
    differentiate_tensor_program,
    extract_backward_tensor_program,
    scalar_expression_vjp,
)
from tile_lifetime.compiler import (
    compile_erased_dense_program,
)
from tile_lifetime.contract_map_chain import (
    BoundCastScalarMap,
    ContractMapChainValue,
    RankTwoContractShape,
    TwoContractMapForwardResult,
    TwoContractMapReverseResult,
    TwoContractMapTrainingProgram,
    execute_two_contract_map_forward,
    execute_two_contract_map_reverse,
    form_two_contract_map_training_program,
)
from tile_lifetime.cuda_contract_map_chain_codegen import (
    ContractMapChainSourceAudit,
    GeneratedCudaContractMapChainFfi,
    audit_cuda_contract_map_chain_source,
    generate_cuda_contract_map_chain_ffi,
)
from tile_lifetime.cuda_normalized_exp_contract_forward_codegen import (
    GeneratedCudaNormalizedExpContractForwardFfi,
    generate_cuda_normalized_exp_contract_forward_ffi,
)
from tile_lifetime.cuda_normalized_exp_contract_reverse_codegen import (
    GeneratedCudaNormalizedExpContractReverseFfi,
    generate_cuda_normalized_exp_contract_reverse_ffi,
)
from tile_lifetime.cuda_prepared_contract_codegen import (
    GeneratedCudaPreparedContract,
    PreparedContractOperand,
    PreparedContractOperandDelivery,
    PreparedContractSourceAudit,
    audit_cuda_prepared_contract_source,
    generate_cuda_prepared_contract,
)
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
from tile_lifetime.expert_parallel import (
    ExpertParallelConfig,
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
from tile_lifetime.expert_parallel_training import (
    ExpertParallelTrainingPlan,
    ExpertParallelTrainingStage,
    ExpertParallelTrainingStageKind,
    derive_expert_parallel_training_plan,
)
from tile_lifetime.expert_parallel_training_runtime import (
    BackwardBufferContract,
    DistributedExpertBackwardABI,
    DistributedExpertBackwardResult,
    ExpertBackwardRankABI,
    derive_distributed_expert_backward_abi,
    verify_distributed_expert_backward_abi,
)
from tile_lifetime.ffi_command_buffer import DirectLaunchFfiPhysicalCandidate
from tile_lifetime.fold_placement import (
    FoldAttachment,
    FoldAttachmentSite,
    FoldResultDisposition,
    OwnerTileAvailability,
    attach_fold_to_owner_preparation,
    verify_owner_preparation_fold_attachment,
)
from tile_lifetime.gated_delta_scan import (
    compile_gated_delta_scan,
)
from tile_lifetime.gemm_program import (
    GENERIC_H100_GEMM_BACKEND,
    GemmProgram,
    compile_gemm_program,
)
from tile_lifetime.ir import DType
from tile_lifetime.jax_collective_transport import (
    JaxCollectiveExecutionPlan,
    build_jax_collective_execution_plan,
    execute_jax_collective_completion,
)
from tile_lifetime.jax_contract_map_chain_ffi import (
    call_cuda_contract_map_chain_forward_ffi,
    call_cuda_contract_map_chain_reverse_ffi,
    register_cuda_contract_map_chain_ffi,
)
from tile_lifetime.jax_normalized_exp_contract_forward_ffi import (
    call_cuda_normalized_exp_contract_forward_ffi,
    register_cuda_normalized_exp_contract_forward_ffi,
)
from tile_lifetime.jax_normalized_exp_contract_reverse_ffi import (
    call_cuda_normalized_exp_contract_reverse_ffi,
    register_cuda_normalized_exp_contract_reverse_ffi,
)
from tile_lifetime.jax_streaming_attention_backward_ffi import (
    CompiledStreamingAttentionBackwardFfi,
    GeneratedStreamingAttentionBackwardFfi,
    StreamingAttentionBackwardFfiBuffer,
    StreamingAttentionBackwardResultPolicy,
    StreamingAttentionBackwardStatePolicy,
    StreamingAttentionLogSumExpEncoding,
    TritonAotKernelPlan,
    call_streaming_attention_backward_ffi,
    call_streaming_attention_training_ffi,
    compile_streaming_attention_backward_ffi,
    generate_streaming_attention_backward_ffi,
    register_streaming_attention_backward_ffi,
)
from tile_lifetime.jax_streaming_attention_forward_ffi import (
    CompiledStreamingAttentionForwardFfi,
    GeneratedStreamingAttentionForwardFfi,
    call_streaming_attention_forward_ffi,
    compile_streaming_attention_forward_ffi,
    generate_streaming_attention_forward_ffi,
    register_streaming_attention_forward_ffi,
)
from tile_lifetime.kimi_delta_scan import (
    compile_kimi_delta_scan,
)
from tile_lifetime.linear_pair_map import (
    LinearPairMapTrainingProgram,
    PairMapSavePolicy,
    PairMapVjpProgram,
    build_linear_pair_map_program,
    compile_linear_pair_map_training,
    pair_silu_product_expression,
    pair_tanh_product_expression,
)
from tile_lifetime.msa_recovery import (
    NaturalProjectedRoutedAttentionCompilation,
    ProjectedRoutedAttentionRecoveryError,
    RecoveredProjectedRoutedAttentionProgram,
    compile_natural_projected_routed_attention,
    recover_projected_routed_attention_program,
)
from tile_lifetime.normalized_exp_contract_training import (
    IndexedFoldSelection,
    NormalizedExpContractTrainingExecution,
    NormalizedExpContractTrainingProgram,
    build_normalized_exp_contract_training_program,
    execute_normalized_exp_contract_training,
    tanh_soft_cap_score_expression,
)
from tile_lifetime.pipeline import (
    FrontendProvenance,
    FrontendSourceKind,
    compile_stablehlo_dense_transformer_region,
    compile_stablehlo_projected_routed_attention_program,
    compile_stablehlo_routed_attention_program,
    recover_stablehlo_projected_routed_attention_program,
    recover_stablehlo_routed_attention_program,
)
from tile_lifetime.plan import (
    AttachmentSite,
    ChunkSummaryRepresentation,
    GemmSkeleton,
    MaterializationDisposition,
    NumericalPolicy,
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
from tile_lifetime.row_normalization_training import (
    GeneratedContractSkeleton,
    GeneratedFoldSkeleton,
    GeneratedMapSkeleton,
    RowNormalizationAxisFoldPrograms,
    RowNormalizationSavePolicy,
    RowNormalizationTrainingPlan,
    RowStatisticKind,
    RowStatisticScalePlacement,
    build_row_normalization_axis_fold_programs,
    build_row_normalized_contract_program,
    compile_row_normalization_training,
    lower_row_normalization_axis_folds,
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
from tile_lifetime.shared_reverse_fusion import (
    OwnerComputeComponent,
    OwnerComputeTraversal,
    SharedReverseFusionDisposition,
    SharedReverseFusionPlan,
    plan_shared_producer_reverse_fusion,
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
from tile_lifetime.stablehlo_row_normalization_backward import (
    RecoveredStableHLORowNormalizationBackward,
    StableHLORowNormalizationBackwardCompilation,
    StableHLORowNormalizationBackwardError,
    StableHLORowNormalizationBackwardFfiCompilation,
    compile_stablehlo_row_normalization_backward,
    compile_stablehlo_row_normalization_backward_ffi,
    recover_stablehlo_row_normalization_backward,
)
from tile_lifetime.stablehlo_scan_recovery import (
    StableHLOScanRecoveryError,
    StableHLOStatefulScanCompilation,
    StatefulScanProvenance,
    StatefulScanSourceKind,
    compile_natural_affine_scan,
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
from tile_lifetime.streaming_attention import (
    AttentionScoreAxis,
    OnlineAttentionState,
    ScoreInputSpec,
    ScoreMapSpec,
    StreamingAttentionExecution,
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
    execute_streaming_attention_with_state,
    execute_tensor_program,
    scaled_score_map,
    streaming_attention_from_semantic_operation,
)
from tile_lifetime.streaming_attention_backward import (
    StreamingAttentionBackwardDomainTraversal,
    StreamingAttentionBackwardExecution,
    StreamingAttentionBackwardFoldOrder,
    StreamingAttentionBackwardMaximumVJP,
    StreamingAttentionBackwardProgram,
    StreamingAttentionBackwardProvenance,
    StreamingAttentionBackwardReassociation,
    StreamingAttentionBackwardStage,
    StreamingAttentionBackwardTileSchedule,
    StreamingAttentionBackwardWorkEstimate,
    derive_streaming_attention_backward,
    derive_streaming_attention_backward_fusion_plan,
    derive_streaming_attention_backward_tile_schedule,
    estimate_streaming_attention_backward_work,
    execute_streaming_attention_backward,
    verify_streaming_attention_backward_score_map_vjp,
)
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
from tile_lifetime.xla_axis_fold_pipeline_ffi import (
    AxisFoldPipelineHloInput,
    AxisFoldPipelineHloReplacementAudit,
    AxisFoldPipelineHloReplacementPlan,
    audit_axis_fold_pipeline_hlo_replacement,
    plan_axis_fold_pipeline_hlo_replacement,
    replace_axis_fold_pipeline_hlo_with_custom_call,
)
from tile_lifetime.xla_low_rank_gated_product import (
    LowRankGatedProductForwardPlan,
    LowRankGatedProductReversePlan,
    LowRankGatedProductTrainingReport,
    RankTwoContractPlan,
    recover_low_rank_gated_product_training,
)
from tile_lifetime.xla_low_rank_gated_product_ffi import (
    GeneratedLowRankContractMapCallAudit,
    GeneratedLowRankContractMapFamily,
    GeneratedLowRankContractMapTrainingAudit,
    GeneratedLowRankContractMapTrainingPlan,
    LowRankContractMapCallAudit,
    LowRankContractMapForwardHloReplacementPlan,
    LowRankContractMapReverseHloReplacementPlan,
    LowRankContractMapTrainingHloReplacementAudit,
    LowRankContractMapTrainingHloReplacementPlan,
    audit_generated_low_rank_contract_map_training,
    audit_low_rank_contract_map_training_hlo_replacement,
    mutate_forward_hidden_scalar_program,
    plan_generated_low_rank_contract_map_training,
    plan_low_rank_contract_map_training_hlo_replacements,
    replace_generated_low_rank_contract_map_training,
    replace_low_rank_contract_map_training_hlo_regions_with_custom_calls,
)
from tile_lifetime.xla_normalized_exp_contract_forward import (
    NormalizedExpContractForwardHloRegion,
    NormalizedExpContractForwardHloReplacementAudit,
    NormalizedExpContractForwardHloReplacementPlan,
    audit_normalized_exp_contract_forward_hlo_replacement,
    plan_normalized_exp_contract_forward_hlo_replacement,
    recover_normalized_exp_contract_forward_hlo_region,
    replace_normalized_exp_contract_forward_hlo_region_with_custom_call,
)
from tile_lifetime.xla_normalized_exp_contract_reverse import (
    NormalizedExpContractReverseHloRegion,
    NormalizedExpContractReverseHloReplacementAudit,
    NormalizedExpContractReverseHloReplacementPlan,
    NormalizedExpContractReverseRecoveryReport,
    NormalizedExpReverseContract,
    audit_normalized_exp_contract_reverse_hlo_replacement,
    plan_normalized_exp_contract_reverse_hlo_replacement,
    recover_normalized_exp_contract_reverse_hlo_regions,
    replace_normalized_exp_contract_reverse_hlo_region_with_custom_call,
)
from tile_lifetime.xla_streaming_attention_backward_ffi import (
    StreamingReverseHloProvenance,
    StreamingReverseHloRegionReplacementAudit,
    StreamingReverseHloRegionReplacementPlan,
    StreamingReverseHloReplacementPlan,
    StreamingReverseHloRole,
    StreamingReverseHloValue,
    audit_streaming_attention_backward_region_replacement,
    plan_streaming_attention_backward_hlo_region_replacement,
    plan_streaming_attention_backward_hlo_replacement,
    replace_streaming_attention_backward_entry_with_custom_call,
    replace_streaming_attention_backward_region_with_custom_call,
)
from tile_lifetime.xla_streaming_attention_training_regions import (
    StreamingAttentionTrainingRegionAudit,
    StreamingAttentionTrainingRegionPlan,
    StreamingForwardHloProvenance,
    StreamingForwardHloRegionReplacementPlan,
    StreamingForwardHloRole,
    StreamingForwardHloValue,
    audit_streaming_attention_training_region_replacement,
    plan_streaming_attention_training_regions,
    replace_streaming_attention_training_regions_with_custom_calls,
)
