# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the Snowball/Grug tournament selection."""

import importlib.util
import sys
from pathlib import Path

PAIRWISE_PATH = Path(__file__).with_name("pairwise.py")
SPEC = importlib.util.spec_from_file_location("campaign_pairwise", PAIRWISE_PATH)
assert SPEC is not None and SPEC.loader is not None
pairwise = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pairwise
SPEC.loader.exec_module(pairwise)


def statistic(
    model: str,
    benchmark: str,
    score: float,
    adjusted_sem: float,
    tracker_score: float | None = None,
) -> dict[str, object]:
    return {
        "model": model,
        "benchmark": benchmark,
        "score": score,
        "tracker_score": score if tracker_score is None else tracker_score,
        "adjusted_sem": adjusted_sem,
    }


def test_pairwise_outcomes_use_adjusted_difference_confidence_interval() -> None:
    candidates = ["Grug-A", "Grug-B", "Snowball-C"]
    benchmarks = ["clear", "uncertain"]
    statistics = {
        ("Grug-A", "clear"): statistic("Grug-A", "clear", 0.8, 0.02),
        ("Grug-B", "clear"): statistic("Grug-B", "clear", 0.7, 0.02),
        ("Snowball-C", "clear"): statistic("Snowball-C", "clear", 0.6, 0.02),
        ("Grug-A", "uncertain"): statistic("Grug-A", "uncertain", 0.55, 0.04),
        ("Grug-B", "uncertain"): statistic("Grug-B", "uncertain", 0.5, 0.04),
        ("Snowball-C", "uncertain"): statistic("Snowball-C", "uncertain", 0.45, 0.04),
    }

    outcomes, pairs, summary = pairwise.pairwise_rows(benchmarks, candidates, statistics)

    assert [row["outcome"] for row in outcomes] == ["left", "tie", "left", "tie", "left", "tie"]
    assert [(row["model"], row["mean_pairwise_win_rate"]) for row in summary] == [
        ("Grug-A", 0.75),
        ("Grug-B", 0.5),
        ("Snowball-C", 0.25),
    ]
    assert next(row for row in pairs if row["model"] == "Grug-A" and row["opponent"] == "Grug-B") == {
        "model": "Grug-A",
        "opponent": "Grug-B",
        "wins": 1,
        "losses": 0,
        "ties": 1,
        "benchmarks": 2,
        "win_rate": 0.75,
    }


def test_pairwise_outcomes_use_full_precision_score_not_tracker_display() -> None:
    candidates = ["Grug-Linear", "Grug-Recovery"]
    benchmarks = ["tau3-pi"]
    statistics = {
        ("Grug-Linear", "tau3-pi"): statistic("Grug-Linear", "tau3-pi", 0.05, 0.012477446057244202, tracker_score=0.05),
        ("Grug-Recovery", "tau3-pi"): statistic(
            "Grug-Recovery", "tau3-pi", 0.020710059171597635, 0.00817730339839839, tracker_score=0.021
        ),
    }

    outcomes, _, summary = pairwise.pairwise_rows(benchmarks, candidates, statistics)

    rounded_difference = 0.05 - 0.021
    threshold = 1.96 * (0.012477446057244202**2 + 0.00817730339839839**2) ** 0.5
    assert rounded_difference < threshold < float(outcomes[0]["score_difference"])
    assert outcomes[0]["outcome"] == "left"
    assert summary[0]["model"] == "Grug-Linear"


def test_candidate_panel_excludes_nonselective_benchmarks_and_requires_complete_scores(tmp_path: Path) -> None:
    tracker = tmp_path / "TRACKER.md"
    tracker.write_text(
        "| Model | math500 | financebench | piqa | triviaqa | tau3-pi |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| Metric | accuracy | accuracy | accuracy | accuracy | accuracy |\n"
        "| open-athena/Snowball-A | 0.900 (s3://bucket/a-math/results) | CRASHED | CRASHED | CRASHED | "
        "0.500 (s3://bucket/a-tau/results) |\n"
        "| open-athena/Grug-B | 0.800 (s3://bucket/b-math/results) | CRASHED | CRASHED | CRASHED | RUNNING |\n"
        "| Qwen/baseline | 0.990 (s3://bucket/qwen-math/results) | CRASHED | CRASHED | CRASHED | "
        "0.990 (s3://bucket/qwen-tau/results) |\n"
    )

    try:
        pairwise.candidate_panel(tracker)
    except ValueError as error:
        assert "open-athena/Grug-B / tau3-pi" in str(error)
        assert "financebench" not in str(error)
    else:
        raise AssertionError("an incomplete included benchmark must fail the tournament")


def test_top_two_model_card_reports_direct_result_and_exclusions(tmp_path: Path) -> None:
    candidates = ["open-athena/Grug-Champion", "open-athena/Grug-Runner-Up"]
    benchmarks = ["math500"]
    statistics = {
        (candidates[0], "math500"): statistic(candidates[0], "math500", 0.8, 0.02),
        (candidates[1], "math500"): statistic(candidates[1], "math500", 0.7, 0.02),
    }
    outcomes, pairs, summary = pairwise.pairwise_rows(benchmarks, candidates, statistics)
    cells = {
        model: {
            benchmark: pairwise.critical_difference.Cell(
                model, benchmark, score, f"s3://bucket/{model}/{benchmark}/results"
            )
            for benchmark, score in {
                "math500": 0.8 if model == candidates[0] else 0.7,
                "financebench": 0.5,
                "piqa": 0.6,
                "triviaqa": 0.4,
            }.items()
        }
        for model in candidates
    }

    pairwise.write_top_two_model_card(
        tmp_path,
        benchmarks,
        cells,
        outcomes,
        pairs,
        summary,
        statistics,
        "tracker-hash",
        "recovered-hash",
    )

    report = (tmp_path / "TOP_TWO_MODEL_CARD.md").read_text()
    assert "**1-0-0**" in report
    assert "| math500 | 0.800 [0.761, 0.839] | 0.700 [0.661, 0.739] | +10.0 pp | Champion |" in report
    assert "## Benchmarks excluded from selection" in report
    assert (tmp_path / "top_two_head_to_head.csv").exists()
