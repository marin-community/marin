#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select a Snowball-lineage checkpoint by uncertainty-aware paired win rate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

# The campaign analysis is intentionally executable without installing it as a
# package. Reuse the critical-difference tool's audited sample-statistics path.
STATISTICS_PATH = Path(__file__).parents[1] / "critical_difference" / "plot.py"
STATISTICS_SPEC = importlib.util.spec_from_file_location("campaign_critical_difference", STATISTICS_PATH)
if STATISTICS_SPEC is None or STATISTICS_SPEC.loader is None:
    raise ImportError(f"could not load campaign statistics module from {STATISTICS_PATH}")
critical_difference: ModuleType = importlib.util.module_from_spec(STATISTICS_SPEC)
sys.modules[STATISTICS_SPEC.name] = critical_difference
STATISTICS_SPEC.loader.exec_module(critical_difference)

CI_MULTIPLIER = 1.96
EXCLUDED_BENCHMARKS = frozenset({"financebench", "piqa", "triviaqa"})
type TournamentRow = dict[str, float | int | str]


def is_candidate(model: str) -> bool:
    return model.rsplit("/", 1)[-1].lower().startswith(("snowball-", "grug-"))


def candidate_panel(
    tracker: Path,
) -> tuple[list[str], list[str], dict[str, dict[str, Any]]]:
    """Return the complete candidate panel used by the release tournament."""
    benchmark_order, cells = critical_difference.tracker_cells(tracker)
    benchmarks = [benchmark for benchmark in benchmark_order if benchmark not in EXCLUDED_BENCHMARKS]
    candidates = [model for model in cells if is_candidate(model)]
    if len(candidates) < 2:
        raise ValueError(f"fewer than two Snowball/Grug rows in {tracker}")
    missing = [
        f"{model} / {benchmark}" for model in candidates for benchmark in benchmarks if benchmark not in cells[model]
    ]
    if missing:
        raise ValueError("candidate tournament requires complete observed scores; missing:\n" + "\n".join(missing))
    return benchmarks, candidates, cells


def pairwise_rows(
    benchmarks: list[str],
    candidates: list[str],
    statistics: dict[tuple[str, str], dict[str, object]],
) -> tuple[list[TournamentRow], list[TournamentRow], list[TournamentRow]]:
    """Score benchmark pairs, treating differences within their 95% CI as ties."""
    outcomes: list[TournamentRow] = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            for benchmark in benchmarks:
                left_row = statistics[(left, benchmark)]
                right_row = statistics[(right, benchmark)]
                difference = float(left_row["score"]) - float(right_row["score"])
                threshold = CI_MULTIPLIER * math.hypot(float(left_row["adjusted_sem"]), float(right_row["adjusted_sem"]))
                outcome = "tie"
                if difference > threshold:
                    outcome = "left"
                elif difference < -threshold:
                    outcome = "right"
                outcomes.append(
                    {
                        "left_model": left,
                        "right_model": right,
                        "benchmark": benchmark,
                        "left_score": float(left_row["score"]),
                        "right_score": float(right_row["score"]),
                        "score_difference": difference,
                        "difference_95_threshold": threshold,
                        "outcome": outcome,
                    }
                )

    pairs: list[TournamentRow] = []
    for model in candidates:
        for opponent in candidates:
            if model == opponent:
                continue
            relevant = [
                row for row in outcomes if {str(row["left_model"]), str(row["right_model"])} == {model, opponent}
            ]
            wins = sum(
                (row["outcome"] == "left" and row["left_model"] == model)
                or (row["outcome"] == "right" and row["right_model"] == model)
                for row in relevant
            )
            losses = sum(
                (row["outcome"] == "right" and row["left_model"] == model)
                or (row["outcome"] == "left" and row["right_model"] == model)
                for row in relevant
            )
            ties = len(relevant) - wins - losses
            pairs.append(
                {
                    "model": model,
                    "opponent": opponent,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "benchmarks": len(relevant),
                    "win_rate": (wins + 0.5 * ties) / len(relevant),
                }
            )

    summary: list[TournamentRow] = []
    for model in candidates:
        opponents = [row for row in pairs if row["model"] == model]
        wins = sum(int(row["wins"]) for row in opponents)
        losses = sum(int(row["losses"]) for row in opponents)
        ties = sum(int(row["ties"]) for row in opponents)
        summary.append(
            {
                "model": model,
                "mean_pairwise_win_rate": sum(float(row["win_rate"]) for row in opponents) / len(opponents),
                "pooled_win_rate": (wins + 0.5 * ties) / (wins + losses + ties),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "comparisons": wins + losses + ties,
            }
        )
    summary.sort(
        key=lambda row: (-float(row["mean_pairwise_win_rate"]), -float(row["pooled_win_rate"]), str(row["model"]))
    )
    return outcomes, pairs, summary


def write_csv(path: Path, rows: list[TournamentRow]) -> None:
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    candidates: list[str],
    benchmarks: list[str],
    pairs: list[TournamentRow],
    summary: list[TournamentRow],
    winner: str,
    leaders: list[str],
    tracker_hash: str,
) -> None:
    ids = {model: f"S{index + 1}" for index, model in enumerate(candidates)}
    lines = [
        "# Snowball-lineage tournament",
        "",
        f"Top checkpoint: `{winner}`." if len(leaders) == 1 else f"Joint first: {', '.join(f'`{m}`' for m in leaders)}.",
        "",
        f"Source tracker SHA-256: `{tracker_hash}`. The tournament requires complete observed scores for all "
        f"{len(candidates)} candidates on {len(benchmarks)} benchmarks. FinanceBench, PIQA, and TriviaQA are excluded. "
        "Comparisons use full-precision aggregates recovered from the durable trial evidence; tracker values are "
        "three-decimal presentation values only. "
        "A benchmark is a win only when the adjusted 95% confidence interval for its paired score difference "
        "excludes zero; otherwise it is a tie. The adjusted SEM includes the policy's infrastructure-failure "
        "tolerance factor of `1 / sqrt(0.9)`.",
        "",
        "Finalist comparison: [`TOP_TWO_MODEL_CARD.md`](TOP_TWO_MODEL_CARD.md).",
        "",
        "| Rank | Model | Mean paired win rate | Pooled win rate | W-L-T | Benchmark comparisons |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        rank = 1 + sum(
            float(other["mean_pairwise_win_rate"]) > float(row["mean_pairwise_win_rate"]) + 1e-12 for other in summary
        )
        lines.append(
            f"| {rank} | {ids[str(row['model'])]} | {100 * float(row['mean_pairwise_win_rate']):.1f}% | "
            f"{100 * float(row['pooled_win_rate']):.1f}% | {row['wins']}-{row['losses']}-{row['ties']} | "
            f"{row['comparisons']} |"
        )
    lines += [
        "",
        "| Model | " + " | ".join(ids.values()) + " |",
        "| --- | " + " | ".join("---:" for _ in candidates) + " |",
    ]
    pair_lookup = {(str(row["model"]), str(row["opponent"])): row for row in pairs}
    for model in candidates:
        values = [
            (
                "—"
                if model == opponent
                else f"{100 * float(pair_lookup[(model, opponent)]['win_rate']):.1f}% "
                f"({pair_lookup[(model, opponent)]['wins']}-{pair_lookup[(model, opponent)]['losses']}-"
                f"{pair_lookup[(model, opponent)]['ties']})"
            )
            for opponent in candidates
        ]
        lines.append(f"| {ids[model]} | " + " | ".join(values) + " |")
    lines += ["", "| ID | Model |", "| --- | --- |"]
    lines += [f"| {ids[model]} | {model} |" for model in candidates]
    lines += [
        "",
        "Each benchmark receives equal weight within a model pair. A win scores 1, a tie 0.5, and a loss 0. "
        f"The selection criterion is the unweighted mean of each candidate's {len(candidates) - 1} pairwise rates. "
        "The matrix shows `win rate (wins-losses-ties)`.",
        "",
    ]
    path.write_text("\n".join(lines))


def top_two_rows(
    benchmarks: list[str],
    outcomes: list[TournamentRow],
    summary: list[TournamentRow],
    statistics: dict[tuple[str, str], dict[str, object]],
) -> list[TournamentRow]:
    """Normalize benchmark outcomes to tournament rank order."""
    champion = str(summary[0]["model"])
    runner_up = str(summary[1]["model"])
    rows: list[TournamentRow] = []
    for benchmark in benchmarks:
        outcome = next(
            row
            for row in outcomes
            if row["benchmark"] == benchmark
            and {str(row["left_model"]), str(row["right_model"])} == {champion, runner_up}
        )
        champion_is_left = outcome["left_model"] == champion
        champion_stat = statistics[(champion, benchmark)]
        runner_up_stat = statistics[(runner_up, benchmark)]
        champion_score = float(champion_stat["score"])
        runner_up_score = float(runner_up_stat["score"])
        champion_radius = CI_MULTIPLIER * float(champion_stat["adjusted_sem"])
        runner_up_radius = CI_MULTIPLIER * float(runner_up_stat["adjusted_sem"])
        normalized_outcome = str(outcome["outcome"])
        if not champion_is_left:
            normalized_outcome = {"left": "runner_up", "right": "champion", "tie": "tie"}[normalized_outcome]
        else:
            normalized_outcome = {"left": "champion", "right": "runner_up", "tie": "tie"}[normalized_outcome]
        rows.append(
            {
                "benchmark": benchmark,
                "champion_score": champion_score,
                "champion_ci_low": max(0.0, champion_score - champion_radius),
                "champion_ci_high": min(1.0, champion_score + champion_radius),
                "runner_up_score": runner_up_score,
                "runner_up_ci_low": max(0.0, runner_up_score - runner_up_radius),
                "runner_up_ci_high": min(1.0, runner_up_score + runner_up_radius),
                "champion_minus_runner_up": champion_score - runner_up_score,
                "difference_95_threshold": float(outcome["difference_95_threshold"]),
                "outcome": normalized_outcome,
            }
        )
    return rows


def write_top_two_model_card(
    output_dir: Path,
    benchmarks: list[str],
    cells: dict[str, dict[str, Any]],
    outcomes: list[TournamentRow],
    pairs: list[TournamentRow],
    summary: list[TournamentRow],
    statistics: dict[tuple[str, str], dict[str, object]],
    tracker_hash: str,
    recovered_hash: str,
) -> None:
    """Write a self-contained comparison of the tournament finalists."""
    champion = str(summary[0]["model"])
    runner_up = str(summary[1]["model"])
    rows = top_two_rows(benchmarks, outcomes, summary, statistics)
    write_csv(output_dir / "top_two_head_to_head.csv", rows)
    direct = next(row for row in pairs if row["model"] == champion and row["opponent"] == runner_up)
    champion_summary = summary[0]
    runner_up_summary = summary[1]
    decisive = [row for row in rows if row["outcome"] != "tie"]
    excluded = [benchmark for benchmark in cells[champion] if benchmark in EXCLUDED_BENCHMARKS]
    champion_advantages = ", ".join(str(row["benchmark"]) for row in decisive if row["outcome"] == "champion")
    runner_up_advantages = ", ".join(str(row["benchmark"]) for row in decisive if row["outcome"] == "runner_up")

    lines = [
        "# Snowball tournament finalists: head-to-head model card",
        "",
        "This card compares the first- and second-ranked checkpoints from the completed uncertainty-aware "
        "Snowball/Grug tournament. It uses observed scores only.",
        "",
        "## Models",
        "",
        "| Tournament rank | Checkpoint | Mean pairwise win rate | Cohort W-L-T |",
        "| ---: | --- | ---: | ---: |",
        f"| 1 | [{champion}](https://huggingface.co/{champion}) | "
        f"{float(champion_summary['mean_pairwise_win_rate']):.3f} | "
        f"{champion_summary['wins']}-{champion_summary['losses']}-{champion_summary['ties']} |",
        f"| 2 | [{runner_up}](https://huggingface.co/{runner_up}) | "
        f"{float(runner_up_summary['mean_pairwise_win_rate']):.3f} | "
        f"{runner_up_summary['wins']}-{runner_up_summary['losses']}-{runner_up_summary['ties']} |",
        "",
        "## Direct result",
        "",
        f"The champion wins the direct matchup **{direct['wins']}-{direct['losses']}-{direct['ties']}** "
        f"across {direct['benchmarks']} benchmarks, for a paired win rate of "
        f"**{float(direct['win_rate']):.3f}**. Its decisive-win margin is "
        f"**{int(direct['wins']) - int(direct['losses']):+d} benchmark**. The tournament title comes from "
        "performance against the entire candidate cohort, not from a dominant direct result against the runner-up.",
        "",
        "A benchmark is decisive only when `abs(score difference) > 1.96 x hypot(adjusted SEMs)`. "
        "The adjusted SEM includes the policy factor `1 / sqrt(0.9)`; otherwise the benchmark is a tie.",
        "",
        "## Decisive benchmarks",
        "",
        "| Benchmark | Champion | Runner-up | Margin | Winner |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in decisive:
        winner = "Champion" if row["outcome"] == "champion" else "Runner-up"
        lines.append(
            f"| {row['benchmark']} | {float(row['champion_score']):.3f} | "
            f"{float(row['runner_up_score']):.3f} | "
            f"{100 * float(row['champion_minus_runner_up']):+.1f} pp | {winner} |"
        )
    lines += [
        "",
        "## Complete selection panel",
        "",
        "| Benchmark | Champion score (95% CI) | Runner-up score (95% CI) | Margin | Decision |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        decision = {"champion": "Champion", "runner_up": "Runner-up", "tie": "Tie"}[str(row["outcome"])]
        lines.append(
            f"| {row['benchmark']} | {float(row['champion_score']):.3f} "
            f"[{float(row['champion_ci_low']):.3f}, {float(row['champion_ci_high']):.3f}] | "
            f"{float(row['runner_up_score']):.3f} "
            f"[{float(row['runner_up_ci_low']):.3f}, {float(row['runner_up_ci_high']):.3f}] | "
            f"{100 * float(row['champion_minus_runner_up']):+.1f} pp | {decision} |"
        )
    lines += [
        "",
        "## Benchmarks excluded from selection",
        "",
        "FinanceBench and PIQA were excluded as non-discriminating within the candidate cohort. TriviaQA was "
        "excluded because its answer extraction was overly sensitive to response format. Their point estimates are "
        "shown for completeness but did not affect either rank.",
        "",
        "| Benchmark | Champion | Runner-up | Margin |",
        "| --- | ---: | ---: | ---: |",
    ]
    for benchmark in excluded:
        champion_score = float(cells[champion][benchmark].score)
        runner_up_score = float(cells[runner_up][benchmark].score)
        lines.append(
            f"| {benchmark} | {champion_score:.3f} | {runner_up_score:.3f} | "
            f"{100 * (champion_score - runner_up_score):+.1f} pp |"
        )
    lines += [
        "",
        "## Interpretation and limitations",
        "",
        f"The champion's decisive advantages are {champion_advantages}; the runner-up's are "
        f"{runner_up_advantages}. "
        "A statistical tie is not evidence that two checkpoints are equivalent; it means this evaluation did not "
        "resolve their difference at the configured threshold. Benchmark metrics share a `[0, 1]` direction but are "
        "not otherwise interchangeable, so raw score margins should be interpreted per benchmark.",
        "",
        "## Provenance",
        "",
        f"- Tracker SHA-256: `{tracker_hash}`",
        f"- Recovered-metrics SHA-256: `{recovered_hash}`",
        f"- Selection panel: {len(benchmarks)} benchmarks",
        f"- Candidate cohort: {len(summary)} checkpoints",
        "- Detailed machine-readable comparison: `top_two_head_to_head.csv`",
        "- Tournament source: `snowball_benchmark_pair_outcomes.csv` and `snowball_pairwise_summary.csv`",
        "",
    ]
    (output_dir / "TOP_TWO_MODEL_CARD.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--recovered-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks, candidates, cells = candidate_panel(args.tracker)
    recovered_bytes = args.recovered_metrics.read_bytes()
    recovered = json.loads(recovered_bytes)
    recovered_hash = hashlib.sha256(recovered_bytes).hexdigest()
    statistics_rows = critical_difference.collect_statistics(
        candidates,
        benchmarks,
        cells,
        args.output_dir / "cell_statistics.csv",
        recovered,
        recovered_hash,
    )
    statistics = {(str(row["model"]), str(row["benchmark"])): row for row in statistics_rows}
    outcomes, pairs, summary = pairwise_rows(benchmarks, candidates, statistics)
    write_csv(args.output_dir / "snowball_benchmark_pair_outcomes.csv", outcomes)
    write_csv(args.output_dir / "snowball_pairwise_head_to_head.csv", pairs)
    write_csv(args.output_dir / "snowball_pairwise_summary.csv", summary)

    tracker_hash = hashlib.sha256(args.tracker.read_bytes()).hexdigest()
    winner = str(summary[0]["model"])
    top_rate = float(summary[0]["mean_pairwise_win_rate"])
    leaders = [
        str(row["model"])
        for row in summary
        if math.isclose(float(row["mean_pairwise_win_rate"]), top_rate, rel_tol=0, abs_tol=1e-12)
    ]
    selection = {
        "winner": winner,
        "joint_leaders": leaders,
        "tie_break": "pooled win rate, then model ID" if len(leaders) > 1 else None,
        "criterion": f"mean of {len(candidates) - 1} uncertainty-aware paired win rates",
        "tie_rule": "tie unless abs(score difference) > 1.96 * hypot(adjusted SEMs)",
        "score_precision": "full-precision task aggregate; tracker score is presentation-only",
        "sem_adjustment": "1 / sqrt(0.9)",
        "tracker_sha256": tracker_hash,
        "recovered_metrics_sha256": recovered_hash,
        "candidates": candidates,
        "benchmarks": benchmarks,
        "excluded_benchmarks": sorted(EXCLUDED_BENCHMARKS),
    }
    (args.output_dir / "snowball_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    write_report(
        args.output_dir / "snowball_pairwise_win_rates.md",
        candidates,
        benchmarks,
        pairs,
        summary,
        winner,
        leaders,
        tracker_hash,
    )
    write_top_two_model_card(
        args.output_dir,
        benchmarks,
        cells,
        outcomes,
        pairs,
        summary,
        statistics,
        tracker_hash,
        recovered_hash,
    )
    print(f"selected {winner}; tracker SHA-256 {tracker_hash}")


if __name__ == "__main__":
    main()
