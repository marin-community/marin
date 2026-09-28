# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Analyze verified grid selections; no fitted curve chooses an optimum."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from experiments.domain_phase_mix import starcoder_tpp10 as experiment


def verified_measurements(plan: dict, path: Path) -> dict[str, float]:
    experiment.validate_plan(plan)
    expected = {r["run_name"]: r for r in plan["runs"]}
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected) or {r["run_name"] for r in rows} != set(expected):
        raise ValueError("Measurements must cover the submitted run set exactly once")
    result = {}
    for row in rows:
        request = expected[row["run_name"]]
        if (
            row["plan_sha256"] != plan["plan_sha256"]
            or row["config_fingerprint"] != request["fingerprint"]
            or row["status"] != "succeeded"
            or int(row["step"]) != request["total_steps"] - 1
            or row["metric"] != plan["primary_metric"]
        ):
            raise ValueError(f"Unverified endpoint: {row['run_name']}")
        value = float(row["value"])
        if not math.isfinite(value):
            raise ValueError("Nonfinite endpoint loss")
        result[row["run_name"]] = value
    return result


def analyze(plan: dict, values: dict[str, float]) -> dict:
    if plan["stage"] not in ("pilot", "dense"):
        raise ValueError("Selection analysis requires a complete pilot or dense grid")
    grid = sorted({r["percent"] for r in plan["runs"]})
    expected_grid = list(experiment.PILOT_GRID if plan["stage"] == "pilot" else experiment.DENSE_GRID)
    if grid != expected_grid:
        raise ValueError("Submission does not cover its prespecified common grid")
    return {
        "plan_sha256": plan["plan_sha256"],
        "stage": plan["stage"],
        **analyze_common_grid(plan["runs"], values, grid),
    }


def analyze_common_grid(runs: list[dict], values: dict[str, float], grid: list[int]) -> dict:
    """Compare measured selections after averaging trainer seeds within each subset."""
    names = [r["run_name"] for r in runs]
    if len(names) != len(set(names)) or set(names) != set(values):
        raise ValueError("Common-grid measurements must cover each distinct run exactly once")
    if grid != sorted(set(grid)) or {r["percent"] for r in runs} != set(grid):
        raise ValueError("Common-grid coordinates must be complete, unique and ascending")
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Common-grid measurements must be finite")
    index = {(r["arm"], r["subset_seed"], r["trainer_seed"], r["percent"]): values[r["run_name"]] for r in runs}
    expected = {
        (arm, subset, seed, percent)
        for arm, subset in [("target", None), ("unmatched", None)] + [("matched", d) for d in experiment.SUBSET_SEEDS]
        for seed in (experiment.TRAINER_SEEDS[:1] if arm == "target" else experiment.TRAINER_SEEDS)
        for percent in grid
        if arm != "matched" or percent != 0
    }
    if len(index) != len(runs) or set(index) != expected:
        raise ValueError("Common-grid measurements omit or duplicate an arm, subset, seed or coordinate")

    def curve(arm: str, subset: int | None, seed: int) -> np.ndarray:
        return np.array(
            [
                index[("unmatched", None, seed, 0)] if arm == "matched" and p == 0 else index[(arm, subset, seed, p)]
                for p in grid
            ]
        )

    def selected(y: np.ndarray) -> int:
        return int(np.argmin(y))  # ascending grid: exact ties choose smaller p

    target = curve("target", None, experiment.TRAINER_SEEDS[0])
    unmatched_seeds = np.array([curve("unmatched", None, s) for s in experiment.TRAINER_SEEDS])
    unmatched = unmatched_seeds.mean(axis=0)
    unmatched_index = selected(unmatched)
    unmatched_regret = float(target[unmatched_index] - target.min())
    matched = np.array([[curve("matched", d, s) for s in experiment.TRAINER_SEEDS] for d in experiment.SUBSET_SEEDS])
    per_subset = []
    for d, seeded_curves in zip(experiment.SUBSET_SEEDS, matched, strict=True):
        y = seeded_curves.mean(axis=0)
        chosen = selected(y)
        regret = float(target[chosen] - target.min())
        excess_error = (y - y.min()) - (target - target.min())
        rho = spearmanr(y, target).statistic if np.ptp(y) and np.ptp(target) else None
        per_subset.append(
            {
                "subset_seed": d,
                "selected_percent": grid[chosen],
                "target_regret_bpb": regret,
                "paired_regret_difference_bpb": regret - unmatched_regret,
                "trainer_seed_selections": [grid[selected(c)] for c in seeded_curves],
                "excess_curve_rmse_bpb": float(np.sqrt(np.mean(excess_error**2))),
                "spearman": float(rho) if rho is not None else None,
            }
        )
    return {
        "grid_percent": grid,
        "target_selected_percent": grid[selected(target)],
        "unmatched_selected_percent": grid[unmatched_index],
        "unmatched_target_regret_bpb": unmatched_regret,
        "unmatched_excess_curve_rmse_bpb": float(
            np.sqrt(np.mean(((unmatched - unmatched.min()) - (target - target.min())) ** 2))
        ),
        "unmatched_spearman": (
            float(spearmanr(unmatched, target).statistic) if np.ptp(unmatched) and np.ptp(target) else None
        ),
        "unmatched_trainer_seed_selections": [grid[selected(c)] for c in unmatched_seeds],
        "matched_subsets": per_subset,
        "mean_paired_regret_difference_bpb": float(np.mean([r["paired_regret_difference_bpb"] for r in per_subset])),
        "pooled_subset_selection_secondary_percent": grid[selected(matched.mean(axis=(0, 1)))],
        "scope": (
            "Descriptive, conditional on one target trainer seed and one finite parent. "
            "Subset comparisons share target and unmatched curves; no independent-replicate or significance claim."
        ),
        "curves": {
            "target": target.tolist(),
            "unmatched": unmatched.tolist(),
            "matched_subset_means": matched.mean(axis=1).tolist(),
        },
    }


def plot_result(
    result: dict, output: Path, *, title: str | None = None, metric_label: str = "Programming-languages BPB"
) -> None:
    x = np.array(result["grid_percent"]) / 100
    curves = result["curves"]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    if title:
        figure.suptitle(title)
    for axis, excess in zip(axes, (False, True), strict=True):
        for key, color, label in (("target", "#333333", "Target"), ("unmatched", "#0072B2", "Unmatched proxy")):
            y = np.array(curves[key])
            if excess:
                y -= y.min()
            axis.plot(x, y, "o-", ms=3, color=color, label=label)
            i = np.argmin(y)
            axis.plot(x[i], y[i], "*", ms=10, color=color)
        for i, mean in enumerate(curves["matched_subset_means"]):
            y = np.array(mean)
            if excess:
                y -= y.min()
            axis.plot(
                x,
                y,
                color="#D55E00",
                alpha=0.65,
                linewidth=1.2,
                marker=".",
                label="Matched subsets (3)" if i == 0 else None,
            )
            j = np.argmin(y)
            axis.plot(x[j], y[j], "*", ms=8, color="#D55E00")
        axis.set(
            xlabel="StarCoder token fraction, p",
            ylabel="BPB above own grid minimum" if excess else metric_label,
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    figure.savefig(output / "curves.png", dpi=180)
    figure.savefig(output / "curves.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    values = verified_measurements(plan, args.measurements)
    result = experiment.calibration_summary(plan, values) if plan["stage"] == "calibration" else analyze(plan, values)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if plan["stage"] != "calibration":
        plot_result(result, args.output)


if __name__ == "__main__":
    main()
