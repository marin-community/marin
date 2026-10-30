# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report cost-aware selector results without treating proxy costs as runtime measurements."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from rigging.filesystem.buckets import filesystem_for

from experiments.evaluation.mini_policy.experiment import read_bytes

BASE = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/study-20261030-v6-21x21"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default=BASE)
    args = parser.parse_args()
    base = args.study
    rows = []
    for mode, suffix in [("family", ""), ("later_checkpoint", "-chronological")]:
        results = json.loads(read_bytes(base + suffix + "/summary.json"))["results"]
        for method in dict.fromkeys(result["method"] for result in results):
            for budget in [210, 840]:
                subset = [row for row in results if row["method"] == method and row["budget"] == budget]
                summary = {"validation": mode, "method": method, "tasks": budget, "seeds": len(subset)}
                for metric in ["score_mae", "tournament_spearman", "decisive_agreement", "tie_agreement"]:
                    values = np.asarray([row["metrics"][metric] for row in subset], dtype=float)
                    summary[metric] = float(values.mean()) if np.isfinite(values).all() else None
                summary["winner_retained_top3"] = all(row["metrics"]["winner_retained_top3"] for row in subset)
                rows.append(summary)
    lines = [
        "# Cost-aware subset comparison",
        "",
        (
            "Historical validation on the frozen 18-model cohort. Runtime proxies affect "
            "selection; actual whole-suite GPU-hour savings and prediction-interval "
            "coverage remain unmeasured."
        ),
        "",
        (
            "The family panel holds related models together in four outer folds. The "
            "later-checkpoint panel tests four later Snowball checkpoints with fixed "
            "baseline anchors. Metrics are computed per random seed before averaging; all"
            " other methods use one fixed seed. An undefined Spearman value means the "
            "predicted tournament scores are constant and does not satisfy a ranking "
            "gate."
        ),
        "",
        "| Validation | Method | Tasks | MAE (pp) | Rank Spearman | Decisive agreement |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        rho = "undefined" if row["tournament_spearman"] is None else f"{row['tournament_spearman']:.3f}"
        agreement = "undefined" if row["decisive_agreement"] is None else f"{100*row['decisive_agreement']:.1f}%"
        lines.append(
            f"| {row['validation']} | {row['method']} | "
            f"{row['tasks']} | {100*row['score_mae']:.2f} | "
            f"{rho} | {agreement} |"
        )
    lines += [
        "",
        "## Method definitions",
        "",
        (
            "`regularized` is the initial ridge-relevance/diversity heuristic; `mi_mrmr` "
            "uses empirical mutual information and a diversity penalty. They are retained"
            " as exploratory comparators."
        ),
        "",
        (
            "`greedy_cv` selects features using three inner family-grouped folds and the "
            "full tournament uncertainty rule. Its loss is the error rate on decisive "
            "benchmark comparisons plus 0.05 times benchmark score MAE. Candidate "
            "predictions use exact rank-one ridge updates. At each step it chooses the "
            "greatest validation improvement; when no candidate improves the discrete "
            "loss, it minimizes validation loss to fill the prescribed budget."
        ),
        "",
        (
            "`greedy_cv_cost` divides positive improvement by the benchmark cost proxy. "
            "That proxy is the published mean evaluator wall time divided by unique "
            "benchmark task count. It includes prescribed repetitions in the parent "
            "timing but is not measured marginal GPU cost or a server-startup model."
        ),
        "",
        (
            "IRT cost variants divide task Fisher information by the same cost proxy. "
            "Coverage variants reserve one item from each of reasoning, coding, "
            "instruction/structured answers, long context, and interactive agents per "
            "21-step block until the cumulative minimum is met. Micro retains at least 10"
            " items per domain; mini retains at least 40. The micro selection remains a "
            "prefix of mini."
        ),
        "",
        (
            "All candidates come from the same agreed 21-benchmark pool. The target "
            "is the equally weighted tournament over those same 21 benchmarks. Test-family "
            "outcomes are excluded from task selection, predictor fitting, and "
            "dimensionality selection. The later-checkpoint panel is not a strict "
            "chronological split of baseline release dates."
        ),
        "",
        (
            "No method meets the proposed 0.95 rank and 95% decisive-comparison gates. "
            "Cost weighting generally reduces prediction quality on this cohort. This is "
            "evidence against deploying these fitted selectors, pending the remaining "
            "audits; it is not a claim that runtime proxies equal measured savings."
        ),
        "",
        f"Full-precision source predictions and manifests: `{base}` and `{base}-chronological`.",
    ]
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    for name, payload in [
        ("COST_COMPARISON.md", ("\n".join(lines) + "\n").encode()),
        ("cost-comparison.json", json.dumps(rows, indent=2, allow_nan=False).encode()),
    ]:
        (output / name).write_bytes(payload)
        fs, path = filesystem_for(base + "-report/" + name)
        fs.pipe_file(path, payload)
    print(json.dumps({"rows": len(rows), "destination": base + "-report"}), flush=True)


if __name__ == "__main__":
    main()
