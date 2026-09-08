# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import fsspec
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_60m_fixed_aggregate_phase_order_results_20260726 as analysis,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("p041", 41),
        ("t9_p041_u_st_pr0", 41),
        ("pinlin_calvin_xu/data_mixture/p041-u-st-pr0", 41),
        ("xp041_u_st_pr0", None),
        ("p41_u_st_pr0", None),
    ],
)
def test_run_index_requires_a_complete_panel_token(name: str, expected: int | None) -> None:
    assert analysis.run_index(name) == expected


def test_table9_metric_by_index_uses_candidate_provenance_and_recovery_source_name() -> None:
    manifest = pd.DataFrame({"candidate_id": ["candidate-zero", "candidate-one"]})
    runs = [
        SimpleNamespace(
            name="t9_p000_candidate",
            id="run-zero",
            state="finished",
            created_at="2026-07-26T00:00:00Z",
            summary={analysis.TABLE9_KEY: 1.1},
            config={
                "checkpoint_path": "gs://bucket/checkpoints/p000_candidate-abc/hf/step-4576",
                "provenance": {"candidate_id": "candidate-zero"},
            },
        ),
        SimpleNamespace(
            name="t9-p001-recovery",
            id="run-one",
            state="finished",
            created_at="2026-07-26T00:01:00Z",
            summary={analysis.TABLE9_KEY: 1.2},
            config={
                "checkpoint_path": "gs://bucket/checkpoints/p001_candidate-def/hf/step-4576",
                "provenance": {"source_run_name": "p001"},
            },
        ),
    ]

    selected = analysis.table9_metric_by_index(runs, manifest)

    assert selected[0]["candidate_id"] == "candidate-zero"
    assert selected[1]["candidate_id"] == "candidate-one"
    assert selected[0]["provenance_mode"] == "candidate_id"
    assert selected[1]["provenance_mode"] == "source_run_name_and_checkpoint_index"
    assert selected[1]["checkpoint_root"] == "gs://bucket/checkpoints/p001_candidate-def"


def test_read_final_uncheatable_requires_the_final_checkpoint_step() -> None:
    checkpoint_root = "memory://phase-order/p000_candidate-abc"
    metrics_path = f"{checkpoint_root}/checkpoints/eval_metrics.jsonl"
    with fsspec.open(metrics_path, "wt") as handle:
        handle.write(json.dumps({"step": 100, analysis.UNCHEATABLE_KEY: 1.2}) + "\n")
        handle.write(json.dumps({"step": analysis.EXPECTED_CHECKPOINT_STEP, analysis.UNCHEATABLE_KEY: 1.1}) + "\n")

    final = analysis.read_final_uncheatable(checkpoint_root)

    assert final == {
        "value": 1.1,
        "step": analysis.EXPECTED_CHECKPOINT_STEP,
        "metrics_path": metrics_path,
    }


def test_read_final_uncheatable_rejects_conflicting_duplicate_steps() -> None:
    checkpoint_root = "memory://phase-order/p001_candidate-def"
    metrics_path = f"{checkpoint_root}/checkpoints/eval_metrics.jsonl"
    with fsspec.open(metrics_path, "wt") as handle:
        handle.write(json.dumps({"step": analysis.EXPECTED_CHECKPOINT_STEP, analysis.UNCHEATABLE_KEY: 1.1}) + "\n")
        handle.write(json.dumps({"step": analysis.EXPECTED_CHECKPOINT_STEP, analysis.UNCHEATABLE_KEY: 1.2}) + "\n")

    with pytest.raises(ValueError, match="conflicting"):
        analysis.read_final_uncheatable(checkpoint_root)


def test_add_same_seed_control_deltas_rejects_duplicate_controls() -> None:
    observed = pd.DataFrame(
        [
            {
                "candidate_id": "control-a",
                "anchor_id": "anchor",
                "seed_block": 0,
                "is_control": True,
                "uncheatable_bpb": 1.0,
                "table9_macro_bpb": 1.1,
            },
            {
                "candidate_id": "control-b",
                "anchor_id": "anchor",
                "seed_block": 0,
                "is_control": True,
                "uncheatable_bpb": 1.0,
                "table9_macro_bpb": 1.1,
            },
        ]
    )

    with pytest.raises(ValueError, match="one tied control"):
        analysis.add_same_seed_control_deltas(observed)


def test_pair_decomposition_rejects_duplicate_or_missing_orientations() -> None:
    observed = pd.DataFrame(
        [
            {
                "anchor_id": "anchor",
                "pair_id": "pair",
                "replicate_index": 0,
                "seed_block": 0,
                "sign": sign,
                "is_control": False,
                "direction_family": "family",
                "direction_id": "direction",
                "hypothesis": "hypothesis",
                "phase_tv": 0.1,
                "candidate_id": f"candidate-{index}",
                "uncheatable_bpb": value,
                "uncheatable_bpb_same_seed_control": 1.0,
                "table9_macro_bpb": float("nan"),
                "table9_macro_bpb_same_seed_control": float("nan"),
            }
            for index, (sign, value) in enumerate([("plus", 0.9), ("plus", 0.8), ("minus", 1.1)])
        ]
    )

    with pytest.raises(ValueError, match="one plus and one minus"):
        analysis.pair_decomposition(observed)


def test_pair_decomposition_computes_odd_and_even_effects() -> None:
    observed = pd.DataFrame(
        [
            {
                "anchor_id": "anchor",
                "pair_id": "pair",
                "replicate_index": 0,
                "seed_block": 0,
                "sign": sign,
                "is_control": False,
                "direction_family": "family",
                "direction_id": "direction",
                "hypothesis": "hypothesis",
                "phase_tv": 0.1,
                "candidate_id": f"candidate-{sign}",
                "uncheatable_bpb": value,
                "uncheatable_bpb_same_seed_control": 1.0,
                "table9_macro_bpb": float("nan"),
                "table9_macro_bpb_same_seed_control": float("nan"),
            }
            for sign, value in [("plus", 0.9), ("minus", 1.05)]
        ]
    )

    pair = analysis.pair_decomposition(observed).iloc[0]

    assert pair["order_half_effect_plus_minus"] == pytest.approx(-0.075)
    assert pair["symmetric_asymmetry_cost"] == pytest.approx(-0.025)
    assert pair["best_orientation_delta"] == pytest.approx(-0.1)


def test_direction_summary_keeps_phase_tv_separate() -> None:
    pairs = pd.DataFrame(
        [
            {
                "target": "uncheatable",
                "anchor_id": "anchor",
                "direction_family": "mechanistic",
                "direction_id": "direction",
                "hypothesis": "hypothesis",
                "phase_tv": phase_tv,
                "order_half_effect_plus_minus": -phase_tv,
                "symmetric_asymmetry_cost": phase_tv / 10,
                "best_orientation_delta": -phase_tv,
            }
            for phase_tv in [0.12, 0.24]
        ]
    )

    summary = analysis.direction_summary(pairs)

    assert summary["phase_tv"].tolist() == [0.12, 0.24]
    assert summary["complete_pairs"].tolist() == [1, 1]


def test_markdown_table_formats_booleans_as_booleans() -> None:
    table = analysis.markdown_table(pd.DataFrame({"agrees": [True, False]}), ["agrees"])

    assert "True" in table
    assert "False" in table
    assert "1.000000" not in table
