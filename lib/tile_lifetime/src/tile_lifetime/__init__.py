# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Research compiler for Transformer tile-lifetime execution plans."""

from tile_lifetime.attention import compile_attention_region
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
from tile_lifetime.ir import DType, TensorGraph
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
    ExpertParallelMoESkeleton,
    GemmSkeleton,
    MaterializationDisposition,
    NumericalPolicy,
    ReductionSkeleton,
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
from tile_lifetime.swiglu import compile_swiglu_region
