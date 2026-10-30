# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report held-out empirical interval coverage and width together."""

import json
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rigging.filesystem.buckets import filesystem_for

from experiments.evaluation.mini_policy.calibration import DEST, METHODS
from experiments.evaluation.mini_policy.experiment import read_bytes


def main():
    folds = [json.loads(read_bytes(DEST + f"/fold-{fold}/calibration.json")) for fold in range(4)]
    rows = []
    for method in METHODS:
        for budget in (210, 840):
            values = [
                (fold, next(row for row in fold["results"] if row["method"] == method and row["budget"] == budget))
                for fold in folds
            ]
            n = sum(len(fold["test_models"]) for fold, _ in values)
            coverage = sum(row["benchmark_coverage"] * len(fold["test_models"]) for fold, row in values) / n
            width = sum(row["benchmark_mean_interval_width"] * len(fold["test_models"]) for fold, row in values) / n
            covered = [flag for _, row in values for flag in row["tournament_covered"]]
            widths = [
                high - low
                for _, row in values
                for low, high in zip(row["tournament_lower"], row["tournament_upper"], strict=True)
            ]
            rows.append(
                {
                    "method": method,
                    "budget": budget,
                    "heldout_models": n,
                    "benchmark_coverage": coverage,
                    "benchmark_mean_width": width,
                    "tournament_coverage": float(np.mean(covered)),
                    "tournament_mean_width": float(np.mean(widths)),
                }
            )
    lines = [
        "# Nested interval validation",
        "",
        (
            "Empirical 90% interval audit on 18 held-out models. Task selection, "
            "preprocessing, predictor fitting, and IRT dimensionality selection exclude "
            "the held-out family at both inner and outer levels. Inner three-fold family "
            "CV supplies calibration residuals; per-family maxima prevent related "
            "checkpoints from counting as independent calibration units."
        ),
        "",
        (
            "Intervals use the empirical 90th percentile of family-maximum absolute "
            "benchmark errors. Benchmark-score and SEM bounds are propagated "
            "conservatively through the full tournament rule against the outer fold's "
            "fixed training anchors and its held-out models. Each fold therefore uses the"
            " same 18-model reference cohort. These are empirical CV intervals, not a "
            "distribution-free coverage guarantee. The training folds contain only seven "
            "or nine independent families."
        ),
        "",
        (
            "| Method | Tasks | Benchmark coverage | Mean benchmark width (pp) | "
            "Tournament coverage | Mean tournament width (pp) |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['budget']} | "
            f"{100*row['benchmark_coverage']:.1f}% | {100*row['benchmark_mean_width']:.1f} | "
            f"{100*row['tournament_coverage']:.1f}% | {100*row['tournament_mean_width']:.1f} |"
        )
    lines += [
        "",
        (
            "Coverage must be read together with interval width. A wide interval can "
            "cover the truth while providing little help with checkpoint selection. These"
            " intervals cover uncertainty in predicting the frozen historical target; "
            "they do not establish coverage after a harness change or on fresh stochastic"
            " generations. Whole-suite runtime savings and prospective deployment remain "
            "separate requirements."
        ),
        "",
        f"Per-fold models, calibration predictions, radii, actual outcomes, and IRT fits: `{DEST}`.",
        "",
    ]
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    for name, payload in [
        ("INTERVAL_VALIDATION.md", "\n".join(lines).encode()),
        ("interval-validation.json", json.dumps(rows, indent=2, allow_nan=False).encode()),
    ]:
        (output / name).write_bytes(payload)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    positions = np.arange(len(METHODS))
    for budget, offset, color in [(210, -0.18, "#0072B2"), (840, 0.18, "#D55E00")]:
        subset = [next(row for row in rows if row["method"] == method and row["budget"] == budget) for method in METHODS]
        for axis, key in zip(axes, ["tournament_coverage", "tournament_mean_width"], strict=True):
            axis.bar(
                positions + offset, [100 * row[key] for row in subset], width=0.34, color=color, label=f"{budget} tasks"
            )
    axes[0].axhline(90, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Held-out models covered (%)")
    axes[1].set_ylabel("Mean tournament interval width (pp)")
    for axis in axes:
        axis.set_xticks(
            positions, ["Random", "Relevance", "Greedy CV", "IRT static", "IRT adaptive"], rotation=25, ha="right"
        )
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle("Empirical interval coverage and usefulness must be assessed together")
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(output / f"interval-validation.{extension}", dpi=180)
    plt.close(fig)
    for path in output.iterdir():
        if path.is_file():
            fs, key = filesystem_for(DEST + "/report/" + path.name)
            fs.pipe_file(key, path.read_bytes())
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
