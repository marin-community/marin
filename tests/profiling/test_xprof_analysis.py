# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from marin.profiling import xprof_analysis


def test_canonical_hybrid_shell_family_classifies_expected_paths() -> None:
    assert (
        xprof_analysis.canonical_hybrid_shell_family(
            "jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call"
        )
        == "dispatch_shard_shell"
    )
    assert (
        xprof_analysis.canonical_hybrid_shell_family(
            "jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/closed_call/shard_map"
        )
        == "ad_wrapper_shell"
    )
    assert (
        xprof_analysis.canonical_hybrid_shell_family("jit(_train_step)/HackableDecoderLayer/reshape") == "layout_shell"
    )
    assert (
        xprof_analysis.canonical_hybrid_shell_family(
            "jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/add_any"
        )
        == "residual_add_shell"
    )


def test_compare_xprof_named_rows_normalizes_and_aggregates_families() -> None:
    before = {
        "jit(_train_step)/HackableDecoderLayer/reshape": 1.0,
        "jit(_train_step)/HackableDecoderLayer/add_any": 0.5,
    }
    after = {
        "jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/pallas_call": 7.0,
        "jit(_train_step)/HackableDecoderLayer/reshape": 4.0,
        "jit(_train_step)/transpose(jvp(HackableTransformer))/HackableDecoderLayer/add_any": 2.5,
    }

    result = xprof_analysis.compare_xprof_named_rows(
        before,
        after,
        top_k=10,
        normalize_positive_deltas_ms=20.0,
        family_classifier=xprof_analysis.canonical_hybrid_shell_family,
    )

    assert result["positive_delta_total"] == 12.5
    family_rows = {row["family"]: row for row in result["family_positive_deltas"]}
    assert family_rows["dispatch_shard_shell"]["delta"] == 7.0
    assert family_rows["layout_shell"]["delta"] == 3.0
    assert family_rows["residual_add_shell"]["delta"] == 2.5
    assert abs(family_rows["dispatch_shard_shell"]["normalized_ms"] - (7.0 / 12.5) * 20.0) < 1e-6

    top_row = result["positive_deltas"][0]
    assert top_row["family"] == "dispatch_shard_shell"
    assert abs(top_row["normalized_ms"] - (7.0 / 12.5) * 20.0) < 1e-6


def test_parse_framework_op_stats_rows_handles_table_payload() -> None:
    payload = [
        {
            "cols": [
                {"id": "operation"},
                {"id": "type"},
                {"id": "total_self_time"},
            ],
            "rows": [
                {"c": [{"v": "op_a"}, {"v": "pallas_call"}, {"v": 12.5}]},
                {"c": [{"v": "op_b"}, {"v": "dot_general"}, {"v": 8.0}]},
            ],
        }
    ]
    rows = xprof_analysis.parse_framework_op_stats_rows(payload)
    assert rows == [
        {"operation": "op_a", "type": "pallas_call", "total_self_time": 12.5},
        {"operation": "op_b", "type": "dot_general", "total_self_time": 8.0},
    ]


def test_build_xprof_comparison_report_uses_framework_and_category_payloads(monkeypatch) -> None:
    before_xplane = Path("before.xplane.pb")
    after_xplane = Path("after.xplane.pb")

    framework_before = [
        {
            "cols": [
                {"id": "operation"},
                {"id": "total_self_time"},
            ],
            "rows": [
                {"c": [{"v": "jit(_train_step)/HackableDecoderLayer/reshape"}, {"v": 2.0}]},
            ],
        }
    ]
    framework_after = [
        {
            "cols": [
                {"id": "operation"},
                {"id": "total_self_time"},
            ],
            "rows": [
                {
                    "c": [
                        {
                            "v": (
                                "jit(_train_step)/jvp(HackableTransformer)"
                                "/HackableDecoderLayer/closed_call/shard_map/pallas_call"
                            )
                        },
                        {"v": 8.0},
                    ]
                },
                {"c": [{"v": "jit(_train_step)/HackableDecoderLayer/reshape"}, {"v": 5.0}]},
            ],
        }
    ]
    category_before = {"byCategory": {"children": [{"name": "custom-call", "metrics": {"rawTime": 10.0}}]}}
    category_after = {
        "byCategory": {
            "children": [
                {"name": "custom-call", "metrics": {"rawTime": 30.0}},
                {"name": "all-gather", "metrics": {"rawTime": 5.0}},
            ]
        }
    }

    def fake_convert(xplane_paths, *, tool, params=None):
        target = Path(xplane_paths[0]).name
        if tool == "framework_op_stats":
            return (framework_before if target == before_xplane.name else framework_after), "application/json"
        if tool == "op_profile":
            return (category_before if target == before_xplane.name else category_after), "application/json"
        raise AssertionError(tool)

    monkeypatch.setattr(xprof_analysis, "convert_xplane_tool_data", fake_convert)

    result = xprof_analysis.build_xprof_comparison_report(
        before_xplane=before_xplane,
        after_xplane=after_xplane,
        top_k=5,
        normalize_positive_deltas_ms=10.0,
    )

    framework_top = result["framework_op_stats"]["positive_deltas"][0]
    assert framework_top["family"] == "dispatch_shard_shell"
    assert framework_top["delta"] == 8.0

    category_rows = result["op_profile_category"]["positive_deltas"]
    assert category_rows[0]["name"] == "custom-call"
    assert category_rows[0]["delta"] == 20.0
    assert result["normalize_positive_deltas_ms"] == 10.0
    assert result["derived_metrics"]["framework_family_normalized_ms"]["dispatch_shard_shell"] > 0
    assert result["derived_metrics"]["op_profile_category_normalized_ms"]["custom-call"] > 0
    assert result["derived_metrics"]["idle_normalized_ms"] is None
