# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-static tests for the clean SM100 source extraction."""

from __future__ import annotations

import inspect
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from clean_routed_streaming_emitter import (  # noqa: E402
    GENERATED_PHYSICAL_CLASS,
    GENERATED_RELATION_BUILDER_CLASS,
    GENERATED_RELATION_SCHEDULER_MODULE,
    DomainRestrictionKind,
    ExtractedSM100Sources,
    PartialMergeScheduleKind,
    PartialStateMergeProgram,
    PartialValueDType,
    SM100RelationEncoding,
    audit_python_source,
    emitter_plan_from_lowering,
    import_extracted_python_sources,
    render_partial_merge_cuda,
    render_relation_builder_source,
    render_relation_scheduler_source,
)

from tile_lifetime.ir import DType  # noqa: E402
from tile_lifetime.relation import build_relation_plan  # noqa: E402
from tile_lifetime.sm100_routed_lowering import SM100RelationOrientation  # noqa: E402
from tile_lifetime.tensor_program import TensorAxis  # noqa: E402
from tile_lifetime.tiled_fold_finalize import (  # noqa: E402
    FoldFeatureLayout,
    FoldPartialAddressing,
    FoldPhysicalAxis,
    TiledFoldAxes,
    TiledFoldInputLayout,
    deterministic_weighted_sum_fold_program,
)


def _lowering(
    *,
    causal: bool,
    score_scale: float = 128**-0.5,
    output_scale: float = 1.0,
    relation_shift: int = 0,
) -> SimpleNamespace:
    destination = np.mod(np.arange(64, dtype=np.int32).reshape(8, 8) * 3 + 1 + relation_shift, 8)
    relation = build_relation_plan(
        destination,
        np.ones(destination.shape, dtype=np.float32),
        destination_rank_by_item=np.zeros(8, dtype=np.int32),
        destination_local_item_by_item=np.arange(8, dtype=np.int32),
        padding_quantum=1,
    )
    return SimpleNamespace(
        schedule=SimpleNamespace(
            orientation=SM100RelationOrientation.RIGHT_MAJOR,
            partial_state_representation="log_normalizer_normalized_value",
            partial_merge_threads=256,
            partial_merge_tile_rows=8,
            packed_left_rows=128,
            right_block_size=128,
            right_stages=2,
        ),
        score_map=SimpleNamespace(causal=causal, scale=score_scale, softcap=None),
        head_group_size=4,
        key_value_heads=2,
        query_length=8,
        selected_count=4,
        output_scale=output_scale,
        relation=relation,
        query_tokens_per_task=32,
    )


def test_emitter_plan_changes_domain_restriction_without_changing_physical_family() -> None:
    causal = emitter_plan_from_lowering(_lowering(causal=True), paged_key_value=True)
    unrestricted = emitter_plan_from_lowering(_lowering(causal=False), paged_key_value=True)

    assert causal.domain_restriction.kind is DomainRestrictionKind.CAUSAL
    assert unrestricted.domain_restriction.kind is DomainRestrictionKind.UNRESTRICTED
    assert causal.domain_restriction.accepts(3, 3)
    assert not causal.domain_restriction.accepts(3, 4)
    assert unrestricted.domain_restriction.accepts(3, 4)
    assert causal.physical_class == unrestricted.physical_class
    assert causal.physical_constructor | {"causal": False} == unrestricted.physical_constructor
    assert not causal.external_semantic_kernels


def test_emitter_plan_attaches_generic_relation_event_schedule() -> None:
    baseline = emitter_plan_from_lowering(_lowering(causal=False), paged_key_value=True)
    mutated = emitter_plan_from_lowering(
        _lowering(causal=False, relation_shift=1),
        paged_key_value=True,
    )

    assert baseline.event_schedule.program_fingerprint == mutated.event_schedule.program_fingerprint
    assert baseline.event_schedule.runtime_fingerprint != mutated.event_schedule.runtime_fingerprint
    assert baseline.event_schedule.resource_buffer.capacity == 2
    assert {family.name for family in baseline.event_schedule.program.task_families} == {
        "right_resource_stage",
        "grouped_contract_fold_body",
        "partial_state_fold_finalize",
    }
    assert all(
        forbidden not in family.name
        for family in baseline.event_schedule.program.task_families
        for forbidden in ("query", "key", "value", "attention", "moe", "expert")
    )


def test_score_domain_and_finalization_mutations_reuse_one_physical_class() -> None:
    baseline = emitter_plan_from_lowering(
        _lowering(causal=True, score_scale=128**-0.5, output_scale=1.0),
        paged_key_value=False,
    )
    mutated = emitter_plan_from_lowering(
        _lowering(causal=False, score_scale=0.125, output_scale=0.5),
        paged_key_value=False,
    )

    assert baseline.physical_class == GENERATED_PHYSICAL_CLASS
    assert mutated.physical_class == GENERATED_PHYSICAL_CLASS
    assert baseline.normalized_exp_fold.score_scale != mutated.normalized_exp_fold.score_scale
    assert baseline.domain_restriction != mutated.domain_restriction
    assert baseline.partial_merge.output_scale != mutated.partial_merge.output_scale
    assert baseline.physical_constructor | {"causal": False} == mutated.physical_constructor


def test_partial_merge_matches_explicit_source_order_reference() -> None:
    program = PartialStateMergeProgram(
        representation="log_normalizer_normalized_value",
        output_scale=0.5,
        value_dtype=PartialValueDType.FP32,
    )
    log_normalizer = np.array([[0.0, -1.0], [np.log(3.0), -np.inf]], dtype=np.float32)
    normalized_value = np.array(
        [
            [[2.0, 4.0], [5.0, 7.0]],
            [[10.0, 14.0], [100.0, 200.0]],
        ],
        dtype=np.float32,
    )

    merged = program.merge_numpy(log_normalizer, normalized_value)

    np.testing.assert_allclose(merged, np.array([[4.0, 5.75], [2.5, 3.5]], dtype=np.float32), rtol=1e-6)


def test_relation_encoding_uses_group_then_selected_slot_and_kv_major_key() -> None:
    encoding = SM100RelationEncoding(
        source_domain="query_token",
        destination_domain="key_value_block",
        key_value_heads=4,
        selected_count=16,
    )

    assert encoding.edge_group(35) == 2
    assert encoding.edge_selected_slot(35) == 3
    assert encoding.right_task_key(35, 19) == (2, 19)


def test_static_audit_rejects_forbidden_attention_helpers() -> None:
    source = """
from src.common.softmax import SoftmaxSm100
from src.common.mask import AttentionMask

def run():
    return sparse_atten_func(SoftmaxSm100(), AttentionMask())
"""

    audit = audit_python_source(source)

    assert not audit.clean
    assert set(audit.forbidden_dependencies) == {
        "AttentionMask",
        "SoftmaxSm100",
        "sparse_atten_func",
        "src.common.mask",
        "src.common.softmax",
    }


def test_static_audit_accepts_generic_fold_domain_and_physical_calls() -> None:
    source = """
from shuttle_sm100_generated_semantics import DomainRestrictionSm100, NormalizedExpFoldSm100
from cutlass.cute.nvgpu import tcgen05

def issue():
    return tcgen05.make_tmem_copy()

def update():
    return issue()
"""

    audit = audit_python_source(source)

    assert audit.clean
    assert ("update", "issue") in audit.local_call_edges
    assert "tcgen05" in audit.required_physical_tokens


def test_generated_merge_is_compiler_owned_and_deterministic() -> None:
    source = render_partial_merge_cuda(
        PartialStateMergeProgram(
            representation="log_normalizer_normalized_value",
            output_scale=1.0,
            value_dtype=PartialValueDType.FP32,
        )
    )

    assert "shuttle_merge_normalized_exp_partials" in source
    assert "for (int partial = 0; partial < valid_partials; ++partial)" in source
    assert "split_counts" in source
    assert "query_heads_per_key_value_head" in source
    assert "real_col_to_stg128_float_col" in source
    assert "const int row = blockIdx.x" in source
    assert "__shared__ float partial_weights[32]" in source
    assert "if (threadIdx.x == 0)" in source
    assert source.count("expf(") == 1
    assert "real_feature += blockDim.x" in source
    assert "#include <math_constants.h>" in source
    assert "1.00000000000000000e+00f" in source
    assert 'module.def("merge"' in source
    assert "SparseAttentionForwardCombine" not in source
    assert "atomic" not in source.lower()


def test_partial_value_storage_policy_changes_generated_load_not_physical_family() -> None:
    lowering = _lowering(causal=False)
    bf16_plan = emitter_plan_from_lowering(
        lowering,
        paged_key_value=False,
        partial_value_dtype=PartialValueDType.BF16,
    )
    fp32_plan = emitter_plan_from_lowering(
        lowering,
        paged_key_value=False,
        partial_value_dtype=PartialValueDType.FP32,
    )
    bf16_source = render_partial_merge_cuda(bf16_plan.partial_merge)
    fp32_source = render_partial_merge_cuda(fp32_plan.partial_merge)

    assert bf16_plan.physical_class == fp32_plan.physical_class
    assert bf16_plan.physical_constructor == fp32_plan.physical_constructor
    assert bf16_plan.partial_value_dtype is PartialValueDType.BF16
    assert fp32_plan.partial_value_dtype is PartialValueDType.FP32
    assert "const __nv_bfloat16* normalized_value" in bf16_source
    assert "__bfloat162float(normalized_value[value_index])" in bf16_source
    assert "real_col_to_stg128_half_col(real_feature)" in bf16_source
    assert "normalized_value.size(3) % 32 == 0" in bf16_source
    assert "const float* normalized_value" in fp32_source
    assert "real_col_to_stg128_float_col(real_feature)" in fp32_source
    assert "numerator += partial_weights[partial] * normalized_value[value_index]" in fp32_source
    assert "normalized_value.size(3) % 16 == 0" in fp32_source


def test_warp_rows_merge_is_a_schedule_candidate_for_the_same_physical_family() -> None:
    lowering = _lowering(causal=False)
    row_plan = emitter_plan_from_lowering(
        lowering,
        paged_key_value=False,
        partial_merge_schedule=PartialMergeScheduleKind.ROW_BLOCK,
    )
    warp_plan = emitter_plan_from_lowering(
        lowering,
        paged_key_value=False,
        partial_merge_schedule=PartialMergeScheduleKind.WARP_ROWS,
    )
    row_source = render_partial_merge_cuda(row_plan.partial_merge)
    warp_source = render_partial_merge_cuda(warp_plan.partial_merge)

    assert row_plan.physical_class == warp_plan.physical_class
    assert row_plan.physical_constructor == warp_plan.physical_constructor
    assert row_plan.partial_merge.rows_per_block == 1
    assert warp_plan.partial_merge.rows_per_block == 8
    assert "__shared__ float partial_weights[8][32]" in warp_source
    assert "const int row = blockIdx.x * 8 + warp" in warp_source
    assert "if (lane == 0)" in warp_source
    assert "__syncwarp()" in warp_source
    assert "real_feature += 32" in warp_source
    assert "__syncthreads()" not in warp_source
    assert warp_source.count("expf(") == 1
    assert "row_count + 8 - 1" in warp_source
    assert row_source != warp_source


def test_tiled_fold_finalizer_exposes_generic_vector_staging_and_fixed_tree_order() -> None:
    lowering = _lowering(causal=False)
    lowering.selected_count = 16
    plan = emitter_plan_from_lowering(
        lowering,
        paged_key_value=False,
        partial_merge_schedule=PartialMergeScheduleKind.TILED_PIPELINED,
    )

    source = render_partial_merge_cuda(plan.partial_merge)

    assert plan.partial_merge.partial_extent == lowering.selected_count
    assert plan.partial_merge.rows_per_block == 8
    assert plan.partial_merge.feature_tile == 128
    assert plan.partial_merge.pipeline_stages == 4
    assert plan.partial_merge.pipeline_buffers == 2
    assert plan.partial_merge.vector_bytes == 16
    assert "copy_global_to_shared_16" in source
    assert "cp.async.cg.shared.global" in source
    assert "fixed_warp_max" in source
    assert "fixed_warp_sum" in source
    assert "staged_value[kRowsPerBlock][kPipelineBuffers][kPipelineStages][kFeatureTile]" in source
    assert "constexpr int kFeaturesPerLane = 4;" in source
    assert "const int next_buffer = (current_buffer + 1) % kPipelineBuffers;" in source
    assert source.index("&staged_value[warp][next_buffer]") < source.index("const float contribution0")
    assert source.index("const float contribution0") < source.rindex("wait_for_all_async_groups();")
    assert "for (int partial_base = 0; partial_base < kPartialExtent;" in source
    assert "__shfl_sync(0xffffffffu, local_weight, partial)" in source
    assert "__floats2bfloat162_rn" in source
    assert 'module.def("merge_out"' in source
    assert "SparseAttentionForwardCombine" not in source
    assert "atomic" not in source.lower()


def test_tiled_fold_one_buffer_schedule_serializes_reuse_after_generated_ast() -> None:
    lowering = _lowering(causal=False)
    lowering.selected_count = 16
    ping_pong = emitter_plan_from_lowering(
        lowering,
        paged_key_value=False,
        partial_merge_schedule=PartialMergeScheduleKind.TILED_PIPELINED,
    ).partial_merge
    assert ping_pong.generic_program is not None
    no_overlap_program = replace(
        ping_pong.generic_program,
        schedule=replace(
            ping_pong.generic_program.schedule,
            feature_tile=64,
            shared_buffers=1,
        ),
    )
    no_overlap = replace(
        ping_pong,
        feature_tile=64,
        pipeline_buffers=1,
        generic_program=no_overlap_program,
    )

    source = render_partial_merge_cuda(no_overlap)

    assert "constexpr int kFeatureTile = 64;" in source
    assert "constexpr int kPipelineBuffers = 1;" in source
    assert "constexpr int kFeaturesPerLane = 2;" in source
    assert "Serialized one-buffer schedule" in source
    assert "One-buffer ablation" in source
    assert "Ping-pong schedule" not in source
    assert "&staged_value[warp][next_buffer]" not in source
    assert source.index("const float contribution0") < source.rindex("&staged_value[warp][0]")
    assert source.index("const float contribution0") < source.rindex("wait_for_all_async_groups();")


def test_tiled_fold_finalizer_generates_attention_and_non_attention_semantics_from_one_skeleton() -> None:
    lowering = _lowering(causal=False)
    lowering.selected_count = 16
    attention = emitter_plan_from_lowering(
        lowering,
        paged_key_value=False,
        partial_merge_schedule=PartialMergeScheduleKind.TILED_PIPELINED,
    ).partial_merge
    assert attention.generic_program is not None
    indexed_axes = TiledFoldAxes(
        partial=TensorAxis(0, 6, "route_slot"),
        row=attention.generic_program.schedule.axes.row,
        feature=attention.generic_program.schedule.axes.feature,
    )
    indexed_layout = TiledFoldInputLayout(
        addressing=FoldPartialAddressing.INDEXED,
        value_axis_order=(FoldPhysicalAxis.SOURCE, FoldPhysicalAxis.FEATURE),
        scalar_axis_order=(FoldPhysicalAxis.ROW, FoldPhysicalAxis.PARTIAL),
        index_axis_order=(FoldPhysicalAxis.ROW, FoldPhysicalAxis.PARTIAL),
        feature_layout=FoldFeatureLayout.CONTIGUOUS,
    )
    weighted_semantics = deterministic_weighted_sum_fold_program(
        replace(
            attention.generic_program.schedule,
            axes=indexed_axes,
            partial_addressing=FoldPartialAddressing.INDEXED,
            partial_lanes=1,
            input_layout=indexed_layout,
        ),
        partial_value_dtype=DType.BF16,
        output_dtype=DType.BF16,
    )
    weighted = replace(
        attention,
        representation="weighted_vector_sum",
        partial_extent=6,
        generic_program=weighted_semantics,
    )

    attention_source = render_partial_merge_cuda(attention)
    weighted_source = render_partial_merge_cuda(weighted)

    for mechanism in (
        "copy_global_to_shared_16",
        "cp.async.cg.shared.global",
        "staged_value[kRowsPerBlock][kPipelineBuffers][kPipelineStages][kFeatureTile]",
        "const int next_buffer = (current_buffer + 1) % kPipelineBuffers;",
        "for (int partial_base = 0; partial_base < kPartialExtent;",
        "__floats2bfloat162_rn",
    ):
        assert mechanism in attention_source
        assert mechanism in weighted_source
    assert "const float common = fixed_warp_max" in attention_source
    assert "const float denominator = fixed_warp_sum(local_weight);" in attention_source
    assert "const float common = 0.0f;" in weighted_source
    assert "const float denominator = 1.0f;" in weighted_source
    assert "partial_metadata[row * kPartialExtent + partial]" in weighted_source
    assert "__fmul_rn" in weighted_source
    assert "__fadd_rn" in weighted_source
    assert attention_source != weighted_source


def test_extracted_modules_are_registered_before_dataclass_execution(tmp_path: Path) -> None:
    sources = SimpleNamespace(
        semantic_source="""
from dataclasses import dataclass

@dataclass
class SemanticState:
    value: int
""",
        physical_source="""
from shuttle_sm100_generated_semantics import SemanticState

class RoutedStreamingFoldContractSm100:
    state = SemanticState(3)

    def __call__(self):
        return self.state.value
""",
    )

    module = import_extracted_python_sources(cast(ExtractedSM100Sources, sources), msa_root=tmp_path)

    assert module.RoutedStreamingFoldContractSm100.state.value == 3
    assert "return self.state.value" in inspect.getsource(module.RoutedStreamingFoldContractSm100.__call__)


def test_relation_helpers_expose_generic_physical_interfaces() -> None:
    scheduler = render_relation_scheduler_source(
        """
class SparseAttentionSchedule: pass
class SparseAttentionScheduleModel: pass
SPARSE_SCHEDULE_MODEL = SparseAttentionScheduleModel()
def prepare_sparse_fwd_schedule_and_split(): pass
"""
    )
    builder = render_relation_builder_source(
        """
from src.sm100.prepare_scheduler import SparseAttentionSchedule, SPARSE_SCHEDULE_MODEL
class SparseK2qCsrBuilderSm100: pass
"""
    )

    assert "SparseAttention" not in scheduler
    assert "prepare_sparse_fwd_schedule_and_split" not in scheduler
    assert GENERATED_RELATION_BUILDER_CLASS in builder
    assert GENERATED_RELATION_SCHEDULER_MODULE in builder
    assert "SparseAttentionSchedule" not in builder


def test_first_extracted_template_rejects_left_major() -> None:
    lowering = _lowering(causal=True)
    lowering.schedule.orientation = SM100RelationOrientation.LEFT_MAJOR

    with pytest.raises(ValueError, match="right-major"):
        emitter_plan_from_lowering(lowering, paged_key_value=True)
