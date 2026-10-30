# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize held-out subset comparisons; do not imply runtime validation."""

import json
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rigging.filesystem.buckets import filesystem_for

from experiments.evaluation.mini_policy.experiment import read_bytes

BASE = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/study-20261030-v3"
DEST = BASE + "-report"
METHODS = ["random", "regularized", "mi_mrmr", "mirt_static", "mirt_adaptive"]
LABELS = ["Stratified random + ridge", "Regularized selection", "MI + diversity", "IRT static", "IRT adaptive"]


def main():
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    rows = []
    fits = []
    manifests = {}
    for mode, suffix in [("family", ""), ("later_checkpoint", "-chronological")]:
        root = BASE + suffix
        settings = json.loads(read_bytes(root + "/settings.json"))
        summary = json.loads(read_bytes(root + "/summary.json"))
        for fold in range(len(settings["splits"])):
            fit = json.loads(read_bytes(root + f"/fold-{fold}-irt-fit.json"))
            if not fit["converged"] or not all(row["converged"] for row in fit["dimension_cv"]):
                raise ValueError("Comparison includes unconverged IRT candidates")
            fits.append({"validation": mode, "fold": fold, **fit})
        for method in METHODS:
            for budget in [210, 840]:
                results = [row for row in summary["results"] if row["method"] == method and row["budget"] == budget]
                by_fold = []
                observations = []
                for result in results:
                    seed = result["seed"]
                    for fold in range(len(settings["splits"])):
                        record = json.loads(read_bytes(root + f"/fold-{fold}-{method}-{seed}-{budget}.json"))
                        by_fold.append({"fold": fold, "seed": seed, "metrics": record["metrics"]})
                        observations.extend(item["observed_task_fraction"] for item in record["probe_manifests"])
                        for item in record["probe_manifests"]:
                            if len(item["selected"]) != budget or len(set(item["selected"])) != budget:
                                raise ValueError("Invalid distinct-task budget")
                            manifests.setdefault((mode, method, seed, fold, item["model"]), {})[budget] = item[
                                "selected"
                            ]
                row = {
                    "validation": mode,
                    "method": method,
                    "budget": budget,
                    "seeds": len(results),
                    "observed_task_fraction": float(np.mean(observations)),
                    "folds": by_fold,
                }
                for metric in ["score_mae", "tournament_spearman", "decisive_agreement", "tie_agreement"]:
                    values = [result["metrics"][metric] for result in results]
                    row[metric] = float(np.mean(values))
                    row[metric + "_seed_stdev"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                row["all_fold_winners_retained_top3"] = all(fold["metrics"]["winner_retained_top3"] for fold in by_fold)
                row["passes_quality_point_gates"] = all(
                    fold["metrics"]["tournament_spearman"] >= 0.95
                    and fold["metrics"]["decisive_agreement"] >= 0.95
                    and fold["metrics"]["winner_retained_top3"]
                    for fold in by_fold
                )
                rows.append(row)
    for nested in manifests.values():
        if nested[840][:210] != nested[210]:
            raise ValueError("Micro is not the nested mini prefix")
    (output / "results.json").write_text(json.dumps({"results": rows, "irt_fits": fits}, indent=2))
    lines = [
        "# Mini/micro historical subset validation",
        "",
        (
            "These are historical held-out prediction results for the frozen 18-model "
            "cohort. Runtime savings and interval calibration are pending. No method is "
            "validated for policy use."
        ),
        "",
        (
            "The family panel holds out all six Snowball/Grug checkpoints together. The "
            "later-checkpoint panel holds out four later Snowball/Grug checkpoints while "
            "retaining fixed baseline anchors; it is not a strict chronological split of "
            "every baseline family."
        ),
        "",
        (
            "Each random seed produces a complete out-of-fold prediction panel; its "
            "metrics are computed separately, then averaged across the three fixed seeds."
            " Predictions are not averaged before computing metrics. Other methods use "
            "one prespecified seed. Per-fold metrics are preserved in `results.json`."
        ),
        "",
        "| Validation | Method | Tasks | Score MAE (pp) | Rank Spearman | Decisive agreement | All winners in top 3 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['validation']} | {row['method']} | "
            f"{row['budget']} | {100*row['score_mae']:.2f} | "
            f"{row['tournament_spearman']:.3f} | {100*row['decisive_agreement']:.1f}% | "
            f"{row['all_fold_winners_retained_top3']} |"
        )
    lines += [
        "",
        "## Unseen Snowball/Grug family",
        "",
        "All six related checkpoints are held out together. Values below average random seeds where applicable.",
        "",
        "| Method | Tasks | Rank Spearman | Decisive agreement |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["validation"] != "family":
            continue
        heldout = [fold["metrics"] for fold in row["folds"] if fold["fold"] == 0]
        rho = np.mean([fold["tournament_spearman"] for fold in heldout])
        agreement = np.mean([fold["decisive_agreement"] for fold in heldout])
        lines.append(f"| {row['method']} | {row['budget']} | {rho:.3f} | {100*agreement:.1f}% |")
    lines += [
        "",
        "## Interpretation and remaining checks",
        "",
        (
            "Decisive agreement counts the benchmark/model-pair outcomes that the full "
            "reference calls a win or loss and that the predicted tournament labels "
            "correctly; predicting a tie counts as a miss. The rule uses a threshold of "
            "1.96 times the combined SEM, expanded by 1/sqrt(0.9). Rank Spearman compares"
            " the resulting tournament scores. Winner retention requires every model tied"
            " for the highest reference tournament score to remain in the predicted top "
            "three, in every fold."
        ),
        "",
        (
            "The proposed point gates require at least 0.95 rank Spearman and 95% "
            "decisive agreement, plus winner retention in every held-out tournament. The "
            "remaining gates require 70% measured GPU-hour savings and approximately 90% "
            "empirical coverage for nominal 90% prediction intervals. These gates apply "
            "separately to micro and mini."
        ),
        "",
        (
            "A good pooled correlation can hide errors among related checkpoints. "
            "Acceptance requires the proposed rank, decisive-comparison, and "
            "winner-retention gates in the held-out panels, plus measured cost and "
            "calibrated prediction intervals. No runtime or interval gate has been tested"
            " by this report."
        ),
        "",
        (
            "Every IRT candidate and selected refit in both validation panels terminated "
            "successfully under SciPy L-BFGS-B, using gradient tolerance 1e-7, relative "
            "objective tolerance 1e-9, and a 6,000-iteration ceiling. Convergence "
            "describes optimization only. The candidate pool has 41,706 tasks; counting "
            "repeated executions and model requests remains part of the runtime audit. "
            "The task pool excludes the five agreed benchmarks; the target is the "
            "unchanged equal-weight 23-benchmark tournament with the combined 95% "
            "uncertainty tie rule. Missing historical responses are imputed from training"
            " models for ridge prediction and ignored when inferring IRT ability; "
            "selected-task coverage is recorded separately."
        ),
        "",
        (
            "The historical source audit includes grading replays, later source score "
            "updates, and native Harbor reconstructions. Config/pin comparability and "
            "fresh end-to-end subset validation remain open. The later-checkpoint panel "
            "contains only four models; a Spearman value of 1.0 there is weak evidence of"
            " generalization."
        ),
        "",
        (
            "Budget counts refer to distinct tasks. Every prescribed repeat remains "
            "attached to its task. The micro manifest is a prefix of mini for every fixed"
            " selector; adaptive paths are nested per evaluated model."
        ),
        "",
        f"Exact comparison outputs and frozen inputs: `{BASE}` and `{BASE}-chronological`.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    specs = [
        ("score_mae", "Benchmark score MAE (pp)", 100, None),
        ("tournament_spearman", "Tournament rank Spearman", 1, 0.95),
        ("decisive_agreement", "Decisive comparison agreement", 100, 95),
    ]
    x = np.arange(len(METHODS))
    for i, mode in enumerate(["family", "later_checkpoint"]):
        for j, (metric, label, multiplier, gate) in enumerate(specs):
            ax = axes[i, j]
            for budget, offset, color in [(210, -0.18, "#4477AA"), (840, 0.18, "#228833")]:
                selected = [
                    next(
                        row
                        for row in rows
                        if row["validation"] == mode and row["method"] == method and row["budget"] == budget
                    )
                    for method in METHODS
                ]
                ax.bar(
                    x + offset,
                    [row[metric] * multiplier for row in selected],
                    width=0.34,
                    label=f"{budget} tasks",
                    color=color,
                )
            if gate is not None:
                ax.axhline(gate, color="#AA3377", linestyle="--", linewidth=1, label="Proposed gate")
            ax.set_xticks(x, ["Random", "Regularized", "MI", "IRT static", "IRT adaptive"], rotation=30, ha="right")
            ax.set_ylabel(label)
            ax.set_title(mode.replace("_", " ").title())
            ax.grid(axis="y", alpha=0.2)
            if gate is not None:
                ax.set_ylim(0, 1.02 * multiplier)
            if i == 0 and j == 0:
                ax.legend()
    figure.suptitle(
        "Historical held-out prediction: mini 840 and micro 210\nRuntime and interval validation pending", fontsize=14
    )
    for extension in ["png", "pdf", "svg"]:
        figure.savefig(output / f"heldout-comparison.{extension}", dpi=180)
    plt.close(figure)
    for path in output.iterdir():
        if not path.is_file():
            continue
        fs, key = filesystem_for(DEST + "/" + path.name)
        fs.pipe_file(key, path.read_bytes())
    print(
        json.dumps(
            {
                "destination": DEST,
                "rows": len(rows),
                "passes_quality_point_gates": sum(row["passes_quality_point_gates"] for row in rows),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
