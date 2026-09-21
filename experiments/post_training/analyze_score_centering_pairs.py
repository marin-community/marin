# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare paired training seeds using both evaluations at the final checkpoint.

The two passes at one checkpoint reduce evaluator noise; they are not two
independent training seeds. The Student-t interval uses one difference per seed.

Example::

    python -m experiments.post_training.analyze_score_centering_pairs \
      --evaluations results/score_centering_qwen_cap105_evals.csv \
      --repeats results/score_centering_terminal_repeat_evals.csv \
      --pair exploratory:17:r24:r25 \
      --pair exploratory:18:r27:r28 \
      --pair exploratory:19:r30:r31 \
      --output-pairs /tmp/pairs.csv --output-summary /tmp/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import t

PAIR_FIELDS = (
    "cohort",
    "seed",
    "control",
    "centered",
    "control_step0",
    "centered_step0",
    "control_scheduled",
    "centered_scheduled",
    "control_final",
    "centered_final",
    "scheduled_difference",
    "final_difference",
    "mean_terminal_difference",
    "step0_difference",
    "baseline_adjusted_difference",
)
SUMMARY_FIELDS = (
    "cohort",
    "seeds",
    "mean_terminal_difference",
    "terminal_ci95_low",
    "terminal_ci95_high",
    "mean_baseline_adjusted_difference",
    "baseline_adjusted_ci95_low",
    "baseline_adjusted_ci95_high",
)


def _rows(paths: list[Path]) -> list[dict[str, str]]:
    result = []
    for path in paths:
        with path.open(newline="") as stream:
            result.extend(csv.DictReader(stream))
    return result


def _interval(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, math.nan, math.nan
    half_width = t.ppf(0.975, len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def summarize(
    evaluations: list[dict[str, str]],
    repeats: list[dict[str, str]],
    pairs: list[str],
    step: int,
    *,
    dataset: str = "all",
    questions: int = 756,
) -> tuple[list[dict], list[dict]]:
    evals = {}
    memberships = set()
    for row in evaluations:
        if row["dataset"] != dataset or int(row["step"]) not in (0, step):
            continue
        key = row["run"], int(row["step"])
        if key in evals:
            raise ValueError(f"duplicate {dataset} evaluation for {key}")
        if int(row["questions"]) != questions:
            raise ValueError(f"{key}: expected {questions} held-out questions")
        memberships.add(row["membership_sha256"])
        evals[key] = row
    if len(memberships) != 1:
        raise ValueError("held-out prompt or ground-truth membership differs")
    repeated = {}
    for row in repeats:
        if row["run"] in repeated:
            raise ValueError(f"duplicate terminal repeat for {row['run']}")
        repeated[row["run"]] = row

    output = []
    seen_seeds = set()
    for spec in pairs:
        parts = spec.split(":")
        if len(parts) != 4:
            raise ValueError(f"pair must be COHORT:SEED:CONTROL:CENTERED, got {spec!r}")
        cohort, seed, control, centered = parts
        if not all(parts) or (cohort, seed) in seen_seeds or control == centered:
            raise ValueError(f"invalid or duplicate pair {spec!r}")
        seen_seeds.add((cohort, seed))
        for run in (control, centered):
            if (run, 0) not in evals or (run, step) not in evals or run not in repeated:
                raise ValueError(f"{run}: missing step-zero, final, or repeated evaluation")
            if int(repeated[run]["final_all"]) != int(evals[run, step]["completed_correct"]):
                raise ValueError(f"{run}: saved final responses disagree with final repeat")
        zero_control = int(evals[control, 0]["completed_correct"])
        zero_centered = int(evals[centered, 0]["completed_correct"])
        scheduled_control = int(repeated[control]["scheduled_all"])
        scheduled_centered = int(repeated[centered]["scheduled_all"])
        final_control = int(repeated[control]["final_all"])
        final_centered = int(repeated[centered]["final_all"])
        scheduled_difference = scheduled_centered - scheduled_control
        final_difference = final_centered - final_control
        mean_difference = (scheduled_difference + final_difference) / 2
        zero_difference = zero_centered - zero_control
        output.append(
            {
                "cohort": cohort,
                "seed": seed,
                "control": control,
                "centered": centered,
                "control_step0": zero_control,
                "centered_step0": zero_centered,
                "control_scheduled": scheduled_control,
                "centered_scheduled": scheduled_centered,
                "control_final": final_control,
                "centered_final": final_centered,
                "scheduled_difference": scheduled_difference,
                "final_difference": final_difference,
                "mean_terminal_difference": mean_difference,
                "step0_difference": zero_difference,
                "baseline_adjusted_difference": mean_difference - zero_difference,
            }
        )

    by_cohort = defaultdict(list)
    for row in output:
        by_cohort[row["cohort"]].append(row)
        by_cohort["all"].append(row)
    summaries = []
    for cohort, rows in sorted(by_cohort.items()):
        terminal = _interval([row["mean_terminal_difference"] for row in rows])
        adjusted = _interval([row["baseline_adjusted_difference"] for row in rows])
        summaries.append(
            dict(
                zip(
                    SUMMARY_FIELDS,
                    (cohort, len(rows), *terminal, *adjusted),
                    strict=True,
                )
            )
        )
    return output, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", action="append", required=True, type=Path)
    parser.add_argument("--repeats", action="append", required=True, type=Path)
    parser.add_argument("--pair", action="append", required=True, metavar="COHORT:SEED:CONTROL:CENTERED")
    parser.add_argument("--step", type=int, default=40)
    parser.add_argument("--dataset", default="all", help="evaluation subset to compare, such as core-math")
    parser.add_argument("--questions", type=int, default=756, help="required size of the selected evaluation subset")
    parser.add_argument("--output-pairs", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    args = parser.parse_args()
    pair_rows, summary_rows = summarize(
        _rows(args.evaluations),
        _rows(args.repeats),
        args.pair,
        args.step,
        dataset=args.dataset,
        questions=args.questions,
    )
    for path, fields, rows in (
        (args.output_pairs, PAIR_FIELDS, pair_rows),
        (args.output_summary, SUMMARY_FIELDS, summary_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
