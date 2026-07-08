# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import marin.profiling.cli as cli_module
import marin.profiling.ingest as ingest_module
import marin.profiling.xplane as xplane_module
import pytest
from marin.profiling.ingest import summarize_profile_artifact, summarize_trace
from marin.profiling.query import compare_profile_summaries, query_profile_summary
from marin.profiling.report import build_markdown_report
from marin.profiling.schema import PROFILE_SUMMARY_SCHEMA_VERSION, profile_summary_from_dict
from marin.profiling.trace_summary import trace_quality_warnings
from marin.profiling.xplane import (
    XPROF_TABLE_TOOLS,
    _xspace_message_class,
    export_xplane_tables,
    summarize_xplane,
    summarize_xplane_tables,
)
from rigging.filesystem import StoragePath


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
    assert "## Trace Overview" in report
    assert "## Time Breakdown (`exclusive_duration_per_track`)" in report
    assert "## Pre-Op Gaps" in report
    assert "## Gap Context (Region-First)" in report
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


def test_global_stall_uses_compute_window_gaps(tmp_path: Path) -> None:
    trace_path = tmp_path / "global_stall_window_trace.json.gz"
    payload = {
        "displayTimeUnit": "ns",
        "traceEvents": [
            {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:TPU:0"}},
            {"ph": "M", "pid": 1, "tid": 2, "name": "thread_name", "args": {"name": "XLA Ops"}},
            {"ph": "M", "pid": 2, "name": "process_name", "args": {"name": "/host:CPU"}},
            {"ph": "M", "pid": 2, "tid": 1, "name": "thread_name", "args": {"name": "main"}},
            # Outside compute window and should not contribute to global-window stall accounting.
            {"ph": "X", "pid": 2, "tid": 1, "name": "python_host_compute", "ts": 0, "dur": 80},
            {"ph": "X", "pid": 1, "tid": 2, "name": "fusion.1", "ts": 100, "dur": 20},
            {"ph": "X", "pid": 1, "tid": 2, "name": "all-reduce.1", "ts": 130, "dur": 10},
            {"ph": "X", "pid": 1, "tid": 2, "name": "fusion.2", "ts": 180, "dur": 20},
        ],
    }
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    summary = summarize_trace(trace_path, warmup_steps=0, hot_op_limit=20, breakdown_mode="exclusive_global")
    breakdown = summary.time_breakdown
    assert breakdown.duration_basis == "exclusive_duration_global_timeline"
    assert breakdown.total_duration == 100.0
    assert breakdown.compute.total_duration == 40.0
    assert breakdown.communication.total_duration == 10.0
    assert breakdown.host.total_duration == 0.0
    assert breakdown.stall.total_duration == 50.0


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


def test_gap_before_marker_iota_is_attributed_to_payload_op(tmp_path: Path) -> None:
    trace_path = tmp_path / "marker_iota_trace.json.gz"
    payload = {
        "displayTimeUnit": "ns",
        "traceEvents": [
            {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:TPU:0"}},
            {"ph": "M", "pid": 1, "tid": 2, "name": "thread_name", "args": {"name": "XLA Ops"}},
            {"ph": "X", "pid": 1, "tid": 2, "name": "fusion.0", "ts": 0, "dur": 10},
            {"ph": "X", "pid": 1, "tid": 2, "name": "iota.296", "ts": 100, "dur": 1},
            {"ph": "X", "pid": 1, "tid": 2, "name": "dot.1", "ts": 101, "dur": 20},
        ],
    }
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    summary = summarize_trace(trace_path, warmup_steps=0, hot_op_limit=10)
    assert summary.gap_before_ops
    top_gap = summary.gap_before_ops[0]
    assert top_gap.name == "dot.1"
    assert top_gap.payload_op == "dot.1"
    assert top_gap.marker_op == "iota.296"
    assert top_gap.total_gap_duration == 90.0

    gap_query = query_profile_summary(summary, "what is the gap before iota.296?", top_k=3)
    assert gap_query["query_type"] == "pre_op_gap"
    assert gap_query["match"] is not None
    assert gap_query["match"]["name"] == "dot.1"
    assert gap_query["match"]["marker_op"] == "iota.296"


def test_trace_quality_warning_flags_suspected_truncation_cap() -> None:
    suspected, warnings = trace_quality_warnings(num_complete_events=1_000_000)
    assert suspected is True
    assert warnings
    assert "1,000,000" in warnings[0]

    suspected_small, warnings_small = trace_quality_warnings(num_complete_events=999_999)
    assert suspected_small is False
    assert warnings_small == []


def test_gap_marker_payload_resolution_does_not_cross_second_idle_gap(tmp_path: Path) -> None:
    trace_path = tmp_path / "marker_second_gap_trace.json.gz"
    payload = {
        "displayTimeUnit": "ns",
        "traceEvents": [
            {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:TPU:0"}},
            {"ph": "M", "pid": 1, "tid": 2, "name": "thread_name", "args": {"name": "XLA Ops"}},
            {"ph": "X", "pid": 1, "tid": 2, "name": "fusion.0", "ts": 0, "dur": 10},
            {"ph": "X", "pid": 1, "tid": 2, "name": "iota.296", "ts": 100, "dur": 1},
            {"ph": "X", "pid": 1, "tid": 2, "name": "iota.297", "ts": 101, "dur": 1},
            {"ph": "X", "pid": 1, "tid": 2, "name": "dot.1", "ts": 130, "dur": 20},
        ],
    }
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    summary = summarize_trace(trace_path, warmup_steps=0, hot_op_limit=10)
    assert summary.gap_before_ops
    top_gap = summary.gap_before_ops[0]
    assert top_gap.name == "iota.296"
    assert top_gap.payload_op == "iota.296"
    assert top_gap.marker_op == "iota.296"


def test_gpu_stream_threads_and_nccl_ops(tmp_path: Path) -> None:
    """GPU traces use 'Stream #N' threads for ops and NCCL naming for collectives.

    Step markers come from host-side StepTraceAnnotation events (step_num in args)
    rather than the TPU-style 'Steps' thread with numeric event names.
    """
    trace_path = tmp_path / "gpu_trace.json.gz"
    payload = {
        "displayTimeUnit": "ns",
        "traceEvents": [
            # GPU device process with stream-based threads (no "XLA Ops" thread).
            {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:GPU:0"}},
            {"ph": "M", "pid": 1, "tid": 10, "name": "thread_name", "args": {"name": "Stream #0(compute)"}},
            {"ph": "M", "pid": 1, "tid": 11, "name": "thread_name", "args": {"name": "Stream #1(nccl)"}},
            # Host process with step annotations.
            {"ph": "M", "pid": 2, "name": "process_name", "args": {"name": "/host:CPU"}},
            {"ph": "M", "pid": 2, "tid": 1, "name": "thread_name", "args": {"name": "python3"}},
            # Step annotations on host (as produced by jax.profiler.StepTraceAnnotation).
            {"ph": "X", "pid": 2, "tid": 1, "name": "train", "ts": 0, "dur": 500, "args": {"step_num": "0"}},
            {"ph": "X", "pid": 2, "tid": 1, "name": "train", "ts": 500, "dur": 400, "args": {"step_num": "1"}},
            {"ph": "X", "pid": 2, "tid": 1, "name": "train", "ts": 900, "dur": 350, "args": {"step_num": "2"}},
            # Compute ops on Stream #0.
            {"ph": "X", "pid": 1, "tid": 10, "name": "fusion.1", "ts": 10, "dur": 100},
            {"ph": "X", "pid": 1, "tid": 10, "name": "custom-call.2", "ts": 120, "dur": 80},
            # NCCL collective on Stream #1.
            {"ph": "X", "pid": 1, "tid": 11, "name": "ncclDevKernel_AllGather_RING_LL", "ts": 200, "dur": 50},
            {"ph": "X", "pid": 1, "tid": 11, "name": "ncclDevKernel_ReduceScatter_RING_LL", "ts": 260, "dur": 40},
        ],
    }
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    summary = summarize_trace(trace_path, warmup_steps=1, hot_op_limit=10)

    # Step markers detected via host-side step_num fallback.
    assert summary.step_time.all_steps.count == 3
    assert summary.step_time.steady_state_steps.count == 2

    # Ops from Stream threads are recognized (not empty like the old code would produce).
    assert len(summary.hot_ops) > 0
    op_names = {op.name for op in summary.hot_ops}
    assert "fusion.1" in op_names

    # NCCL collectives are classified.
    assert len(summary.communication_ops) > 0
    collective_kinds = {op.collective for op in summary.communication_ops}
    assert "all-gather" in collective_kinds
    assert "reduce-scatter" in collective_kinds

    # Gap analysis works on stream threads.
    assert len(summary.gap_before_ops) > 0


def test_direct_xplane_timeline_parser_recovers_perfetto_style_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(xplane_module, "_try_summarize_xprof_tables", lambda *args, **kwargs: None)
    xplane_path = tmp_path / "profile.xplane.pb"
    _write_xplane(xplane_path)

    summary = summarize_xplane(xplane_path, warmup_steps=1, hot_op_limit=10)

    assert summary.source_format == "xplane_pb"
    assert summary.trace_overview.display_time_unit == "us"
    assert summary.trace_overview.num_complete_events == 6
    assert summary.trace_overview.num_processes == 1
    assert summary.trace_overview.num_threads == 2
    assert summary.step_time.all_steps.count == 2
    assert summary.step_time.steady_state_steps.median == 120.0

    hot_op_names = {op.name for op in summary.hot_ops}
    assert "fusion.1" in hot_op_names
    assert "dot.1" in hot_op_names

    assert summary.gap_before_ops
    top_gap = summary.gap_before_ops[0]
    assert top_gap.name == "dot.1"
    assert top_gap.payload_op == "dot.1"
    assert top_gap.marker_op == "iota.296"
    assert top_gap.total_gap_duration == 70.0

    region_paths = {region.path for region in summary.hierarchical_regions}
    assert "train_step" in region_paths
    assert "train_step=>block_0=>matmul" in region_paths
    assert summary.trace_provenance.run_ids == ["7"]


def test_profile_dir_prefers_xplane_over_capped_perfetto_trace(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(xplane_module, "_try_summarize_xprof_tables", lambda *args, **kwargs: None)
    profile_dir = tmp_path / "artifact" / "plugins" / "profile" / "2026_05_11_12_00_00"
    profile_dir.mkdir(parents=True)
    _write_xplane(profile_dir / "host.xplane.pb")
    (profile_dir / "perfetto_trace.json.gz").write_bytes(b"not a valid gzip trace")

    summary = summarize_profile_artifact(tmp_path / "artifact", warmup_steps=1, hot_op_limit=10)

    assert summary.source_format == "xplane_pb"
    assert summary.trace_overview.num_complete_events == 6


def test_profile_dir_falls_back_to_perfetto_for_multiple_xplane_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(xplane_module, "_try_summarize_xprof_tables", lambda *args, **kwargs: None)
    profile_dir = tmp_path / "artifact" / "plugins" / "profile" / "2026_05_11_12_00_00"
    profile_dir.mkdir(parents=True)
    _write_xplane(profile_dir / "host-0.xplane.pb")
    _write_xplane(profile_dir / "host-1.xplane.pb")
    _write_trace(profile_dir / "perfetto_trace.json.gz", step_durations=[100, 120, 140], softmax_duration=30)

    summary = summarize_profile_artifact(tmp_path / "artifact", warmup_steps=1, hot_op_limit=10)

    assert summary.source_format == "perfetto_trace_json"
    assert summary.step_time.steady_state_steps.median == 130.0


def test_profile_dir_falls_back_to_perfetto_when_xplane_is_malformed(tmp_path: Path) -> None:
    profile_dir = tmp_path / "artifact" / "plugins" / "profile" / "2026_05_11_12_00_00"
    profile_dir.mkdir(parents=True)
    (profile_dir / "host.xplane.pb").write_bytes(b"\xff")
    _write_trace(profile_dir / "perfetto_trace.json.gz", step_durations=[100, 120, 140], softmax_duration=30)

    summary = summarize_profile_artifact(tmp_path / "artifact", warmup_steps=1, hot_op_limit=10)

    assert summary.source_format == "perfetto_trace_json"
    assert summary.step_time.steady_state_steps.median == 130.0


def test_xplane_summary_honors_breakdown_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(xplane_module, "_try_summarize_xprof_tables", lambda *args, **kwargs: None)
    xplane_path = tmp_path / "profile.xplane.pb"
    _write_xplane(xplane_path)

    summary = summarize_xplane(xplane_path, warmup_steps=0, hot_op_limit=10, breakdown_mode="exclusive_global")

    assert summary.time_breakdown.duration_basis == "exclusive_duration_global_timeline"


def test_xplane_summary_merges_timeline_regions_with_xprof_aggregates(tmp_path: Path, monkeypatch) -> None:
    xplane_path = tmp_path / "profile.xplane.pb"
    _write_xplane(xplane_path)
    table_dir = tmp_path / "tables"
    _write_xprof_tables(table_dir)
    table_summary = summarize_xplane_tables(
        table_dir,
        xplane_path=xplane_path,
        warmup_steps=1,
        hot_op_limit=10,
    )
    monkeypatch.setattr(xplane_module, "_try_summarize_xprof_tables", lambda *args, **kwargs: table_summary)

    summary = summarize_xplane(xplane_path, warmup_steps=1, hot_op_limit=10)

    assert summary.source_format == "xplane_pb"
    assert summary.trace_overview.duration_basis == "exclusive_duration_per_track+xprof_aggregate_tables"
    assert summary.step_time.steady_state_steps.median == 2_000.0
    assert summary.time_breakdown.duration_basis == "xprof_overview_step_time_us"
    assert summary.time_breakdown.total_duration == 3_000.0
    assert any(op.name == "xprof_kernel" for op in summary.hot_ops)
    assert any(op.name == "dot.1" for op in summary.hot_ops)
    assert summary.communication_ops[0].collective == "all-gather"
    assert summary.gap_before_ops[0].name == "dot.1"
    assert any(region.path == "train_step=>block_0=>matmul" for region in summary.hierarchical_regions)
    assert any("Step timing was augmented from xprof" in warning for warning in summary.trace_overview.quality_warnings)


def test_xplane_timeline_parser_separates_multiple_planes_with_reused_line_ids(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(xplane_module, "_try_summarize_xprof_tables", lambda *args, **kwargs: None)
    xplane_path = tmp_path / "multi_host.xplane.pb"
    _write_multi_plane_xplane(xplane_path)

    summary = summarize_xplane(xplane_path, warmup_steps=0, hot_op_limit=10)

    assert summary.trace_overview.num_processes == 2
    assert summary.trace_overview.num_threads == 2
    assert summary.trace_overview.num_complete_events == 2
    assert {op.name for op in summary.hot_ops} == {"fusion.host0", "fusion.host1"}
    assert summary.time_breakdown.total_duration == 30.0


def test_summarize_cli_profile_dir_uses_xplane_default(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(xplane_module, "_try_summarize_xprof_tables", lambda *args, **kwargs: None)
    artifact_dir = tmp_path / "artifact" / "plugins" / "profile" / "2026_05_11_12_00_00"
    artifact_dir.mkdir(parents=True)
    _write_xplane(artifact_dir / "host.xplane.pb")
    output_path = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_summary.py",
            "summarize",
            "--profile-dir",
            str(tmp_path / "artifact"),
            "--warmup-steps",
            "1",
            "--output",
            str(output_path),
        ],
    )

    cli_module.main()

    assert capsys.readouterr().out.strip() == str(output_path)
    summary = profile_summary_from_dict(json.loads(output_path.read_text(encoding="utf-8")))
    assert summary.source_format == "xplane_pb"
    assert summary.trace_overview.num_complete_events == 6


def test_summarize_cli_xplane_file_honors_breakdown_mode(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(xplane_module, "_try_summarize_xprof_tables", lambda *args, **kwargs: None)
    xplane_path = tmp_path / "profile.xplane.pb"
    output_path = tmp_path / "summary.json"
    _write_xplane(xplane_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_summary.py",
            "summarize",
            "--xplane-file",
            str(xplane_path),
            "--breakdown-mode",
            "exclusive_global",
            "--output",
            str(output_path),
        ],
    )

    cli_module.main()

    assert capsys.readouterr().out.strip() == str(output_path)
    summary = profile_summary_from_dict(json.loads(output_path.read_text(encoding="utf-8")))
    assert summary.time_breakdown.duration_basis == "exclusive_duration_global_timeline"


def test_summarize_cli_run_target_honors_download_root(tmp_path: Path, monkeypatch, capsys) -> None:
    seen_download_root = []
    seen_run_target = []
    output_path = tmp_path / "summary.json"
    download_root = tmp_path / "downloads"

    def download_profile_dir(
        run_target: str,
        *,
        entity: str | None = None,
        project: str | None = None,
        download_root: Path | None = None,
    ):
        seen_download_root.append(download_root)
        seen_run_target.append((run_target, entity, project))
        return SimpleNamespace(profile_dir=tmp_path / "artifact", run_metadata=None)

    def summarize_artifact(*args, **kwargs):
        trace_path = tmp_path / "trace.json.gz"
        _write_trace(trace_path, step_durations=[100], softmax_duration=10)
        return summarize_trace(trace_path, warmup_steps=0)

    monkeypatch.setattr(cli_module, "download_profile_dir_for_run", download_profile_dir)
    monkeypatch.setattr(cli_module, "summarize_profile_artifact", summarize_artifact)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_summary.py",
            "summarize",
            "--run-target",
            "entity/project/run",
            "--download-root",
            str(download_root),
            "--output",
            str(output_path),
        ],
    )

    cli_module.main()

    assert capsys.readouterr().out.strip() == str(output_path)
    assert seen_download_root == [download_root]
    assert seen_run_target == [("entity/project/run", None, None)]


def test_download_profile_dir_for_run_mirrors_remote_log_dir(tmp_path: Path, monkeypatch) -> None:
    source = "memory://wandb/logs/trainer-456/profiler"
    trace_dir = StoragePath(source) / "plugins/profile/2026_05_11_12_00_00"
    trace_dir.mkdirs()
    (trace_dir / "perfetto_trace.json.gz").write_bytes(b"trace")

    fake_run = SimpleNamespace(
        path=["entity", "project", "run-123"],
        id="run-123",
        config={"trainer": {"id": "trainer-456", "log_dir": "memory://wandb/logs"}},
        summary={},
    )

    class FakeApi:
        def run(self, run_path: str):
            assert run_path == "entity/project/run-123"
            return fake_run

    monkeypatch.setattr(ingest_module.wandb, "Api", lambda: FakeApi())

    downloaded = ingest_module.download_profile_dir_for_run(
        "entity/project/run-123",
        download_root=tmp_path / "downloads",
    )

    assert downloaded.profile_dir == tmp_path / "downloads" / "trainer-456" / "profiler"
    assert downloaded.run_metadata.run_id == "run-123"
    assert (downloaded.profile_dir / "plugins" / "profile" / "2026_05_11_12_00_00" / "perfetto_trace.json.gz").exists()


def test_summarize_xplane_with_installed_xprof_exports_tables(tmp_path: Path) -> None:
    pytest.importorskip("xprof")
    xplane_path = tmp_path / "profile.xplane.pb"
    output_dir = tmp_path / "xprof_tables"
    _write_xplane(xplane_path)

    summary = summarize_xplane(xplane_path, output_dir=output_dir, warmup_steps=1, hot_op_limit=10)

    assert summary.source_format == "xplane_pb"
    assert summary.trace_overview.duration_basis.endswith("+xprof_aggregate_tables")
    assert output_dir.exists()
    assert (output_dir / "overview_page.json").exists()
    assert summary.trace_overview.num_complete_events == 6


def test_export_xplane_tables_accepts_text_and_counts_trace_events(tmp_path: Path, monkeypatch) -> None:
    xplane_path = tmp_path / "profile.xplane.pb"
    xplane_path.write_bytes(b"fake xplane bytes")

    raw_to_tool_data_module = ModuleType("xprof.convert.raw_to_tool_data")
    convert_module = ModuleType("xprof.convert")
    xprof_module = ModuleType("xprof")
    seen_tools: list[str] = []

    def xspace_to_tool_data(paths, tool, options):
        assert paths == [str(xplane_path)]
        assert options == {"use_saved_result": False}
        seen_tools.append(tool)
        if tool == "trace_viewer@":
            return b'{"returnedEventsSize":1000000}', "application/json"
        if tool == "overview_page":
            return '{"cols":[],"rows":[]}', "application/json"
        return b'{"cols":[],"rows":[]}', "application/json"

    raw_to_tool_data_module.__dict__["xspace_to_tool_data"] = xspace_to_tool_data
    convert_module.__dict__["raw_to_tool_data"] = raw_to_tool_data_module
    xprof_module.__dict__["convert"] = convert_module
    monkeypatch.setitem(sys.modules, "xprof", xprof_module)
    monkeypatch.setitem(sys.modules, "xprof.convert", convert_module)
    monkeypatch.setitem(sys.modules, "xprof.convert.raw_to_tool_data", raw_to_tool_data_module)

    export = export_xplane_tables(xplane_path, tmp_path / "tables", count_trace_events=True)

    assert export.trace_event_count == 1_000_000
    assert export.table_sizes["overview_page"] == len('{"cols":[],"rows":[]}')
    assert (export.output_dir / "overview_page.json").read_text(encoding="utf-8") == '{"cols":[],"rows":[]}'
    assert set(XPROF_TABLE_TOOLS).issubset(seen_tools)
    assert "trace_viewer@" in seen_tools


def test_xplane_table_summary_ignores_non_table_entries_and_handles_large_kernel_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "tables"
    output_dir.mkdir()
    xplane_path = tmp_path / "profile.xplane.pb"
    xplane_path.write_bytes(b"fake xplane bytes")

    (output_dir / "overview_page.json").write_text(
        json.dumps(
            [
                {"rows": [{"c": [{"v": "ignored"}]}]},
                {
                    "cols": [
                        {"id": "stepnum"},
                        {"id": "stepTimeMs"},
                        {"id": "deviceCollectivesTimeMs"},
                        {"id": "deviceComputeTimeMs"},
                        {"id": "infeedTimeMs"},
                        {"id": "otherTimeMs"},
                    ],
                    "rows": [
                        {"c": [{"v": 0}, {"v": 1250.0}, {"v": 250.0}, {"v": 700.0}, {"v": 100.0}, {"v": 200.0}]},
                        {"c": [{"v": 1}, {"v": 1500.0}, {"v": 300.0}, {"v": 800.0}, {"v": 150.0}, {"v": 250.0}]},
                    ],
                    "p": {"bottleneck": "collectives"},
                },
            ]
        ),
        encoding="utf-8",
    )
    kernel_rows = [
        {
            "c": [
                {"v": rank},
                {"v": f"ncclAllGather_{rank}" if rank % 10 == 0 else f"_lambda_kernel_{rank}"},
                {"v": 1000.0 + rank},
                {"v": rank + 1},
                {"v": rank % 3 == 0},
            ]
        }
        for rank in range(5000)
    ]
    (output_dir / "kernel_stats.json").write_text(
        json.dumps(
            {
                "cols": [
                    {"id": "rank"},
                    {"id": "kernel_name"},
                    {"id": "total_duration_us"},
                    {"id": "occurrences"},
                    {"id": "is_kernel_using_tensor_core"},
                ],
                "rows": kernel_rows,
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_xplane_tables(
        output_dir,
        xplane_path=xplane_path,
        warmup_steps=1,
        hot_op_limit=20,
        trace_event_count=1_000_000,
    )

    assert summary.source_format == "xplane_pb_xprof_tables"
    assert summary.trace_overview.suspected_truncation is True
    assert any("xprof bottleneck: collectives" in warning for warning in summary.trace_overview.quality_warnings)
    assert summary.trace_overview.num_complete_events == 5000
    assert summary.step_time.all_steps.count == 2
    assert summary.step_time.steady_state_steps.median == 1_500_000.0
    assert summary.time_breakdown.communication.total_duration == 550_000.0
    assert summary.time_breakdown.compute.total_duration == 1_500_000.0
    assert summary.hot_ops[0].name == "_lambda_kernel_4999"
    assert summary.hot_ops[0].exclusive_duration == 5999.0
    assert summary.communication_ops[0].collective == "all-gather"
    assert summary.communication_ops[0].total_duration > 0
    assert summary.semantic_families
    assert summary.optimization_candidates


def test_xplane_table_summary_rejects_malformed_table_json(tmp_path: Path) -> None:
    output_dir = tmp_path / "tables"
    output_dir.mkdir()
    xplane_path = tmp_path / "profile.xplane.pb"
    xplane_path.write_bytes(b"fake xplane bytes")
    (output_dir / "overview_page.json").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed xprof table JSON"):
        summarize_xplane_tables(output_dir, xplane_path=xplane_path)


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


def _write_xplane(path: Path) -> None:
    xspace = _xspace_message_class()()
    xspace.hostnames.append("host-0")
    plane = xspace.planes.add()
    plane.id = 1
    plane.name = "/device:TPU:0"

    plane.stat_metadata[1].id = 1
    plane.stat_metadata[1].name = "tf_op"
    plane.stat_metadata[2].id = 2
    plane.stat_metadata[2].name = "long_name"
    plane.stat_metadata[3].id = 3
    plane.stat_metadata[3].name = "run_id"

    _add_xplane_event_metadata(plane, 1, "0")
    _add_xplane_event_metadata(plane, 2, "1")
    _add_xplane_event_metadata(plane, 3, "%fusion.1 = f32[8,8] fusion()", display_name="fusion.1")
    _add_xplane_event_metadata(plane, 4, "%iota.296 = s32[8] iota()", display_name="iota.296")
    _add_xplane_event_metadata(plane, 5, "%dot.1 = f32[8,8] dot()", display_name="dot.1")
    _add_xplane_event_metadata(plane, 6, "all-reduce.1")

    steps = plane.lines.add()
    steps.id = 1
    steps.name = "Steps"
    _add_xplane_event(steps, 1, offset_us=0, duration_us=100)
    _add_xplane_event(steps, 2, offset_us=100, duration_us=120)

    ops = plane.lines.add()
    ops.id = 2
    ops.name = "XLA Ops"
    fusion = _add_xplane_event(ops, 3, offset_us=0, duration_us=10)
    _add_xplane_stat(fusion, 1, "train_step/block_0/fusion")
    _add_xplane_stat(fusion, 2, "%fusion.1 = f32[8,8] fusion()")
    _add_xplane_stat(fusion, 3, 7)
    _add_xplane_event(ops, 6, offset_us=10, duration_us=20)
    _add_xplane_event(ops, 4, offset_us=100, duration_us=1)
    dot = _add_xplane_event(ops, 5, offset_us=101, duration_us=30)
    _add_xplane_stat(dot, 1, "train_step/block_0/matmul")
    _add_xplane_stat(dot, 2, "%dot.1 = f32[8,8] dot(f32[8,8] %lhs, f32[8,8] %rhs)")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(xspace.SerializeToString())


def _write_multi_plane_xplane(path: Path) -> None:
    xspace = _xspace_message_class()()
    for plane_index in range(2):
        plane = xspace.planes.add()
        plane.id = plane_index + 1
        plane.name = f"/device:TPU:{plane_index}"
        _add_xplane_event_metadata(
            plane,
            1,
            f"%fusion.host{plane_index} = f32[8,8] fusion()",
            display_name=f"fusion.host{plane_index}",
        )

        ops = plane.lines.add()
        ops.id = 7
        ops.name = "XLA Ops"
        _add_xplane_event(ops, 1, offset_us=plane_index * 100, duration_us=10 + plane_index * 10)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(xspace.SerializeToString())


def _write_xprof_tables(output_dir: Path) -> None:
    output_dir.mkdir(parents=True)
    (output_dir / "overview_page.json").write_text(
        json.dumps(
            {
                "cols": [
                    {"id": "stepnum"},
                    {"id": "stepTimeMs"},
                    {"id": "deviceCollectivesTimeMs"},
                    {"id": "deviceComputeTimeMs"},
                    {"id": "infeedTimeMs"},
                    {"id": "otherTimeMs"},
                ],
                "rows": [
                    {"c": [{"v": 0}, {"v": 1.0}, {"v": 0.1}, {"v": 0.7}, {"v": 0.1}, {"v": 0.1}]},
                    {"c": [{"v": 1}, {"v": 2.0}, {"v": 0.2}, {"v": 1.4}, {"v": 0.2}, {"v": 0.2}]},
                ],
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "kernel_stats.json").write_text(
        json.dumps(
            {
                "cols": [
                    {"id": "rank"},
                    {"id": "kernel_name"},
                    {"id": "total_duration_us"},
                    {"id": "occurrences"},
                ],
                "rows": [
                    {"c": [{"v": 1}, {"v": "xprof_kernel"}, {"v": 5_000.0}, {"v": 5}]},
                    {"c": [{"v": 2}, {"v": "ncclAllGather"}, {"v": 800.0}, {"v": 2}]},
                ],
            }
        ),
        encoding="utf-8",
    )


def _add_xplane_event_metadata(plane, metadata_id: int, name: str, *, display_name: str = "") -> None:
    metadata = plane.event_metadata[metadata_id]
    metadata.id = metadata_id
    metadata.name = name
    metadata.display_name = display_name


def _add_xplane_event(line, metadata_id: int, *, offset_us: int, duration_us: int):
    event = line.events.add()
    event.metadata_id = metadata_id
    event.offset_ps = offset_us * 1_000_000
    event.duration_ps = duration_us * 1_000_000
    return event


def _add_xplane_stat(event, metadata_id: int, value) -> None:
    stat = event.stats.add()
    stat.metadata_id = metadata_id
    if isinstance(value, str):
        stat.str_value = value
    else:
        stat.uint64_value = value
