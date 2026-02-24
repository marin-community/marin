# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
from pathlib import Path

from marin.profiling.ingest import summarize_trace
from marin.profiling.query import compare_profile_summaries, query_profile_summary
from marin.profiling.report import build_markdown_report
from marin.profiling.schema import PROFILE_SUMMARY_SCHEMA_VERSION, profile_summary_from_dict


def test_summarize_trace_produces_deterministic_breakdown_and_hot_ops(tmp_path: Path) -> None:
    trace_path = tmp_path / "perfetto_trace.json.gz"
    _write_trace(trace_path, step_durations=[100, 110, 120, 130, 140, 150], softmax_duration=60)

    summary = summarize_trace(trace_path, warmup_steps=2, hot_op_limit=10)

    assert summary.schema_version == PROFILE_SUMMARY_SCHEMA_VERSION
    assert summary.trace_overview.num_complete_events == 13
    assert summary.step_time.all_steps.count == 6
    assert summary.step_time.steady_state_steps.count == 4
    assert summary.step_time.steady_state_steps.median == 135.0

    breakdown = summary.time_breakdown
    assert breakdown.total_duration == 310.0
    assert breakdown.compute.total_duration == 120.0
    assert breakdown.communication.total_duration == 60.0
    assert breakdown.host.total_duration == 80.0
    assert breakdown.stall.total_duration == 50.0
    assert breakdown.other.total_duration == 0.0

    hot_ops = summary.hot_ops
    assert hot_ops[0].name == "fusion.1"
    assert hot_ops[0].exclusive_duration == 60.0
    assert hot_ops[1].name == "softmax"
    assert hot_ops[1].exclusive_duration == 60.0

    assert summary.communication_ops[0].collective == "all-reduce"
    assert summary.communication_ops[0].total_duration == 60.0
    assert len(summary.optimization_candidates) > 0


def test_query_and_compare_helpers(tmp_path: Path) -> None:
    before_trace = tmp_path / "before_trace.json.gz"
    after_trace = tmp_path / "after_trace.json.gz"
    _write_trace(before_trace, step_durations=[100, 110, 120, 130, 140, 150], softmax_duration=60)
    _write_trace(after_trace, step_durations=[90, 100, 110, 120, 130, 140], softmax_duration=40)

    before = summarize_trace(before_trace, warmup_steps=2, hot_op_limit=10)
    after = summarize_trace(after_trace, warmup_steps=2, hot_op_limit=10)

    query_result = query_profile_summary(before, "What are the top 3 ops by exclusive time?", top_k=3)
    assert query_result["query_type"] == "top_ops"
    assert len(query_result["results"]) == 3
    assert query_result["results"][0]["name"] == "fusion.1"

    compare_result = compare_profile_summaries(before, after, top_k=5)
    assert compare_result["step_time"]["steady_state_median_delta"] == -10.0
    assert compare_result["regressed_ops"] == []
    assert compare_result["improved_ops"][0]["name"] == "softmax"
    assert compare_result["improved_ops"][0]["delta"] < 0

    round_tripped = profile_summary_from_dict(json.loads(before.to_json()))
    assert round_tripped.schema_version == PROFILE_SUMMARY_SCHEMA_VERSION
    assert round_tripped.hot_ops[0].name == before.hot_ops[0].name

    report = build_markdown_report(before, top_k=2)
    assert "# Profile Report (profile_summary.v1)" in report
    assert "## Time Breakdown (`exclusive_duration_per_track`)" in report
    assert "## Pre-Op Gaps" in report
    assert "## Gap Context (By Region)" in report
    assert "## Hierarchical Regions" in report
    assert "Inclusive %" in report
    assert "Exclusive %" in report
    assert "## Optimization Candidates" in report


def test_semantic_family_share_is_bounded_with_global_breakdown(tmp_path: Path) -> None:
    trace_path = tmp_path / "global_breakdown_trace.json.gz"
    _write_trace(trace_path, step_durations=[100, 110, 120, 130, 140, 150], softmax_duration=60)

    summary = summarize_trace(trace_path, warmup_steps=2, hot_op_limit=10, breakdown_mode="exclusive_global")
    assert summary.time_breakdown.duration_basis == "exclusive_duration_global_timeline"
    assert summary.semantic_families
    assert all(0.0 <= family.share_of_total <= 1.0 for family in summary.semantic_families)


def test_gap_before_specific_op_and_hierarchical_regions(tmp_path: Path) -> None:
    trace_path = tmp_path / "hierarchical_trace.json.gz"
    target = "_linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_combined.1"
    payload = {
        "displayTimeUnit": "ns",
        "traceEvents": [
            {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:TPU:0"}},
            {"ph": "M", "pid": 1, "tid": 2, "name": "thread_name", "args": {"name": "XLA Ops"}},
            {"ph": "X", "pid": 1, "tid": 2, "name": "train_step=>forward=>matmul", "ts": 0, "dur": 50},
            {
                "ph": "X",
                "pid": 1,
                "tid": 2,
                "name": target,
                "ts": 100,
                "dur": 20,
                "args": {"tf_op": "jit(_train_step)/apply_rotary_embedding/linear_softmax_cross_entropy_loss_bwd"},
            },
            {"ph": "X", "pid": 1, "tid": 2, "name": "train_step=>backward=>all-reduce.1", "ts": 140, "dur": 10},
        ],
    }
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    summary = summarize_trace(trace_path, warmup_steps=0, hot_op_limit=20)
    assert len(summary.gap_before_ops) > 0
    top_gap = summary.gap_before_ops[0]
    assert top_gap.name == target
    assert top_gap.total_gap_duration == 50.0
    assert top_gap.max_gap_duration == 50.0

    region_paths = {region.path for region in summary.hierarchical_regions}
    assert "train_step" in region_paths
    assert "_train_step" in region_paths
    assert "_train_step=>apply_rotary_embedding" not in region_paths
    assert "_train_step=>apply_rotary_embedding=>linear_softmax_cross_entropy_loss_bwd" in region_paths

    gap_query = query_profile_summary(
        summary,
        "what is the gap before _linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu_combined.1?",
    )
    assert gap_query["query_type"] == "pre_op_gap"
    assert gap_query["match"]["name"] == target
    assert (
        gap_query["context"][0]["region_path"]
        == "_train_step=>apply_rotary_embedding=>linear_softmax_cross_entropy_loss_bwd"
    )

    region_query = query_profile_summary(summary, "show hierarchical regions", top_k=5)
    assert region_query["query_type"] == "hierarchical_regions"
    assert len(region_query["results"]) > 0
    assert "inclusive_share_of_total" in region_query["results"][0]
    assert "exclusive_share_of_total" in region_query["results"][0]

    context_query = query_profile_summary(summary, "show context for op copy.1", top_k=5)
    assert context_query["query_type"] == "gap_region_context"
    natural_language_context_query = query_profile_summary(
        summary,
        "copy.1 is noisy; can you contextualize that in hierarchical annotation?",
        top_k=5,
    )
    assert natural_language_context_query["query_type"] == "gap_region_context"
    assert natural_language_context_query["target"] == "copy.1"


def test_hierarchy_blacklist_prefers_semantic_parts(tmp_path: Path) -> None:
    trace_path = tmp_path / "blacklist_trace.json.gz"
    payload = {
        "displayTimeUnit": "ns",
        "traceEvents": [
            {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:TPU:0"}},
            {"ph": "M", "pid": 1, "tid": 2, "name": "thread_name", "args": {"name": "XLA Ops"}},
            {
                "ph": "X",
                "pid": 1,
                "tid": 2,
                "name": "custom_kernel.1",
                "ts": 0,
                "dur": 100,
                "args": {"tf_op": "jit(_train_step)/shard_map/apply_rotary_embedding/pallas_call"},
            },
        ],
    }
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    summary = summarize_trace(trace_path, warmup_steps=0, hot_op_limit=10)
    paths = {region.path for region in summary.hierarchical_regions}
    assert "_train_step" in paths
    assert "_train_step=>apply_rotary_embedding" in paths
    assert not any("shard_map" in path for path in paths)
    assert not any("pallas_call" in path for path in paths)
    by_path = {region.path: region for region in summary.hierarchical_regions}
    assert by_path["_train_step"].inclusive_duration == 100.0
    assert by_path["_train_step"].exclusive_duration == 0.0
    assert by_path["_train_step=>apply_rotary_embedding"].exclusive_duration == 100.0


def test_dynamic_donated_prefix_is_trimmed_not_dropped(tmp_path: Path) -> None:
    trace_path = tmp_path / "dynamic_donated_trace.json.gz"
    payload = {
        "displayTimeUnit": "ns",
        "traceEvents": [
            {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:TPU:0"}},
            {"ph": "M", "pid": 1, "tid": 2, "name": "thread_name", "args": {"name": "XLA Ops"}},
            {
                "ph": "X",
                "pid": 1,
                "tid": 2,
                "name": "copy.564",
                "ts": 0,
                "dur": 100,
                "args": {"tf_op": "dynamic_donated[1][0][0].model.params.blocks[0].attn.w_k:"},
            },
            {
                "ph": "X",
                "pid": 1,
                "tid": 2,
                "name": "copy.564",
                "ts": 200,
                "dur": 50,
                "args": {"tf_op": "dynamic_donated[1][0][0].model.params.blocks[0].attn.w_k:"},
            },
        ],
    }
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    summary = summarize_trace(trace_path, warmup_steps=0, hot_op_limit=10)
    assert len(summary.gap_region_contexts) > 0
    assert summary.gap_region_contexts[0].region_path == "copy(model.params.blocks[0].attn.w_k)"


def test_copy_gap_context_does_not_double_wrap_copy_label(tmp_path: Path) -> None:
    trace_path = tmp_path / "copy_gap_trace.json.gz"
    payload = {
        "displayTimeUnit": "ns",
        "traceEvents": [
            {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:TPU:0"}},
            {"ph": "M", "pid": 1, "tid": 2, "name": "thread_name", "args": {"name": "XLA Ops"}},
            {"ph": "X", "pid": 1, "tid": 2, "name": "copy.1", "ts": 0, "dur": 10},
            {"ph": "X", "pid": 1, "tid": 2, "name": "copy.1", "ts": 30, "dur": 10},
        ],
    }
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    summary = summarize_trace(trace_path, warmup_steps=0, hot_op_limit=10)
    assert summary.gap_region_contexts
    assert summary.gap_region_contexts[0].op_name == "copy.1"
    assert summary.gap_region_contexts[0].region_path == "copy"


def _write_trace(path: Path, *, step_durations: list[float], softmax_duration: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "displayTimeUnit": "ns",
        "traceEvents": [
            {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:TPU:0"}},
            {"ph": "M", "pid": 1, "tid": 1, "name": "thread_name", "args": {"name": "Steps"}},
            {"ph": "M", "pid": 1, "tid": 2, "name": "thread_name", "args": {"name": "XLA Ops"}},
            {"ph": "M", "pid": 2, "name": "process_name", "args": {"name": "/host:CPU"}},
            {"ph": "M", "pid": 2, "tid": 1, "name": "thread_name", "args": {"name": "main"}},
        ],
    }

    for idx, duration in enumerate(step_durations):
        payload["traceEvents"].append(
            {"ph": "X", "pid": 1, "tid": 1, "name": str(idx), "ts": idx * 1000, "dur": duration}
        )

    payload["traceEvents"].extend(
        [
            # Nested pair to validate exclusive-duration handling.
            {"ph": "X", "pid": 1, "tid": 2, "name": "fusion.1", "ts": 0, "dur": 100},
            {"ph": "X", "pid": 1, "tid": 2, "name": "all-reduce.1", "ts": 20, "dur": 40},
            {"ph": "X", "pid": 1, "tid": 2, "name": "softmax", "ts": 120, "dur": softmax_duration},
            {"ph": "X", "pid": 1, "tid": 2, "name": "dependency-wait", "ts": 190, "dur": 30},
            {"ph": "X", "pid": 1, "tid": 2, "name": "psum.1", "ts": 230, "dur": 20},
            {"ph": "X", "pid": 2, "tid": 1, "name": "python_host_compute", "ts": 0, "dur": 80},
            {"ph": "X", "pid": 2, "tid": 1, "name": "threading wait", "ts": 90, "dur": 20},
        ]
    )

    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
