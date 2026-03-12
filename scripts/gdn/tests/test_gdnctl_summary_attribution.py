# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import pytest


def _load_gdnctl_module():
    gdnctl_path = Path(__file__).resolve().parents[1] / "gdnctl.py"
    spec = importlib.util.spec_from_file_location("gdnctl", gdnctl_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gdnctl = _load_gdnctl_module()


def _summary_with_ce_forward_divisor() -> dict[str, object]:
    ce_bwd_root = gdnctl.PROFILE_SUMMARY_CE_BWD_ROOT_HOT_PATH
    return {
        "hot_ops": [
            {
                "count": 8,
                "total_duration": 21600.0,
                "tf_op_path": f"jit(_train_step)/jvp()/shard_map/jit({gdnctl.PROFILE_SUMMARY_CE_FORWARD_HOT_PATH[:-1]}:",
            },
            {
                "count": 8,
                "total_duration": 16000.0,
                "tf_op_path": f"jit(_train_step)/{gdnctl.PROFILE_SUMMARY_FORWARD_CLOSED_CALL_HOT_PATH}",
            },
            {
                "count": 8,
                "total_duration": 8000.0,
                "tf_op_path": f"jit(_train_step)/{gdnctl.PROFILE_SUMMARY_BACKWARD_CLOSED_CALL_HOT_PATH}",
            },
            {
                "count": 8,
                "total_duration": 4000.0,
                "tf_op_path": "jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/shard_map/pallas_call:",
            },
            {
                "count": 8,
                "total_duration": 12000.0,
                "tf_op_path": (
                    "jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/add:"
                ),
            },
            {
                "count": 8,
                "total_duration": 6000.0,
                "tf_op_path": "jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/reshape:",
            },
            {
                "count": 504,
                "total_duration": 24000.0,
                "tf_op_path": f"jit(_train_step)/{ce_bwd_root}/while/body/closed_call/dot_general:",
            },
        ],
        "hierarchical_regions": [
            {
                "path": gdnctl.PROFILE_SUMMARY_CE_BWD_WHILE_REGION_PATH,
                "inclusive_duration": 72000.0,
                "exclusive_duration": 0.0,
            },
            {
                "path": gdnctl.PROFILE_SUMMARY_CONDITIONAL_REGION_PATH,
                "inclusive_duration": 16.0,
                "exclusive_duration": 16.0,
            },
        ],
    }


def test_profile_summary_attribution_extracts_tracked_and_remainder_budgets() -> None:
    summary = _summary_with_ce_forward_divisor()

    metrics = gdnctl._profile_summary_attribution(
        summary,
        step_duration_ms=100.0,
        upper_bound_step_ms=40.0,
        gdn_layer_fraction=0.75,
        gdn_layers_per_block=3,
        gdn_block_size=4,
        top_k=4,
    )

    assert metrics["per_step_divisor"] == 8
    assert math.isclose(metrics["forward_closed_call_ms"], 2.0)
    assert math.isclose(metrics["backward_closed_call_ms"], 1.0)
    assert math.isclose(metrics["ce_forward_pallas_ms"], 2.7)
    assert math.isclose(metrics["ce_attributed_while_ms"], 9.0)
    assert math.isclose(metrics["while_ms"], 9.0)
    assert math.isclose(metrics["conditional_ms"], 0.002)
    assert math.isclose(metrics["kernel_budget_ms"], 3.0)
    assert math.isclose(metrics["control_budget_ms"], 9.002)
    assert math.isclose(metrics["train_path_budget_ms"], 12.002)
    assert math.isclose(metrics["step_duration_ms"], 100.0)
    assert math.isclose(metrics["remainder_budget_ms"], 87.998)
    assert math.isclose(metrics["upper_bound_gap_ms"], 60.0)
    assert math.isclose(metrics["gap_explained_by_train_path"], 12.002 / 60.0)
    assert math.isclose(metrics["ad_shell_budget_ms"], 2.75)
    assert [row["path"] for row in metrics["ad_shell_topk"]] == [
        "HackableDecoderLayer/closed_call/shard_map:",
        "HackableDecoderLayer/reshape:",
        "HackableDecoderLayer/shard_map/pallas_call:",
    ]

    remainder_bucket_ms = metrics["remainder_bucket_ms"]
    assert isinstance(remainder_bucket_ms, dict)
    assert math.isclose(remainder_bucket_ms["CE forward pallas_call"], 2.7)
    assert math.isclose(remainder_bucket_ms["HackableDecoderLayer/closed_call/shard_map:"], 1.5)
    assert math.isclose(remainder_bucket_ms["HackableDecoderLayer/reshape:"], 0.75)
    assert math.isclose(remainder_bucket_ms["HackableDecoderLayer/shard_map/pallas_call:"], 0.5)
    assert "CE backward dot_general" not in remainder_bucket_ms

    ce_while_bucket_ms = metrics["ce_while_bucket_ms"]
    assert isinstance(ce_while_bucket_ms, dict)
    assert math.isclose(ce_while_bucket_ms["CE backward dot_general"], 3.0)

    remainder_topk = metrics["remainder_topk"]
    assert isinstance(remainder_topk, list)
    assert [row["path"] for row in remainder_topk] == [
        "CE forward pallas_call",
        "HackableDecoderLayer/closed_call/shard_map:",
        "HackableDecoderLayer/reshape:",
        "HackableDecoderLayer/shard_map/pallas_call:",
    ]


def test_profile_summary_step_divisor_falls_back_to_gcd_without_ce_forward() -> None:
    summary = {
        "hot_ops": [
            {"count": 48, "total_duration": 1000.0, "tf_op_path": "jit(_train_step)/foo:"},
            {"count": 288, "total_duration": 2000.0, "tf_op_path": "jit(_train_step)/bar:"},
            {"count": 3024, "total_duration": 3000.0, "tf_op_path": "jit(_train_step)/baz:"},
        ],
        "hierarchical_regions": [],
    }

    assert gdnctl._profile_summary_step_divisor(summary) == 48


def test_profile_summary_bucket_deltas_only_report_positive_deltas() -> None:
    deltas = gdnctl._profile_summary_bucket_deltas(
        {"decoder shell": 5.0, "CE forward pallas_call": 2.7},
        {"decoder shell": 1.0, "CE forward pallas_call": 3.0},
        top_k=4,
    )

    assert deltas == [{"path": "decoder shell", "ms": 4.0}]


def test_profile_summary_positive_delta_budget_aggregates_bucketwise_uplift() -> None:
    delta_budget = gdnctl._profile_summary_positive_delta_budget(
        {"shell/a": 5.0, "shell/b": 1.0},
        {"shell/a": 1.5, "shell/c": 9.0},
    )

    assert math.isclose(delta_budget, 4.5)


def test_summary_attribution_emits_delta_budget_comparison(tmp_path: Path) -> None:
    candidate_summary = _summary_with_ce_forward_divisor()
    baseline_summary = {
        "hot_ops": [
            {
                "count": 8,
                "total_duration": 4000.0,
                "tf_op_path": (
                    "jit(_train_step)/jvp(HackableTransformer)/HackableDecoderLayer/closed_call/shard_map/add:"
                ),
            },
        ],
        "hierarchical_regions": [],
    }
    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline.json"
    output_path = tmp_path / "comparison.json"
    candidate_path.write_text(json.dumps(candidate_summary), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline_summary), encoding="utf-8")

    args = argparse.Namespace(
        summary=candidate_path,
        baseline_summary=baseline_path,
        step_duration_ms=100.0,
        baseline_step_duration_ms=40.0,
        upper_bound_step_ms=40.0,
        gdn_layer_fraction=0.75,
        baseline_gdn_layer_fraction=0.0,
        gdn_layers_per_block=3,
        baseline_gdn_layers_per_block=0,
        gdn_block_size=4,
        baseline_gdn_block_size=4,
        top_k=4,
        output=output_path,
    )

    assert gdnctl.cmd_summary_attribution(args) == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    comparison = payload["comparison"]
    assert comparison["remainder_delta_budget_ms"] == pytest.approx(4.95)
    assert comparison["decoder_layer_shell_delta_budget_ms"] == pytest.approx(2.25)
    assert comparison["ad_shell_delta_budget_ms"] == pytest.approx(2.25)
    assert comparison["sharding_shell_delta_budget_ms"] == pytest.approx(1.5)
    assert comparison["layout_shell_delta_budget_ms"] == pytest.approx(0.75)
    assert comparison["gap_explained_by_decoder_layer_shell_delta"] == pytest.approx(2.25 / 60.0)
    assert payload["remainder_delta_topk"][0] == {"path": "CE forward pallas_call", "ms": 2.7}
