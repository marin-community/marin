# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze paired crossover experiments using measured loss and execution costs."""

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path
from statistics import mean, median

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
TARGETS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
COST = {4: "single_horizon_execution_seconds", 8: "update_seconds"}


def target_crossing(rows, target, active):
    # Zero-update draft cost includes the fixed copy/check cost from its first measured endpoint.
    overhead = rows[0][COST[4]] - rows[0]["update_seconds"] if active == 4 else 0.0
    previous = {"horizon": 0, "fresh_gain": 0.0, COST[active]: overhead}
    for row in rows:
        if row["fresh_gain"] is None:
            return {"status": "nonfinite"}
        if row["fresh_gain"] >= target:
            fraction = (target - previous["fresh_gain"]) / (row["fresh_gain"] - previous["fresh_gain"])
            assert 0 <= fraction <= 1
            return {
                "status": "crossed",
                "observed_update_bracket": [previous["horizon"], row["horizon"]],
                "observed_cost_bracket": [previous[COST[active]], row[COST[active]]],
                "interpolated_updates": previous["horizon"] + fraction * (row["horizon"] - previous["horizon"]),
                "interpolated_cost": previous[COST[active]] + fraction * (row[COST[active]] - previous[COST[active]]),
            }
        previous = row
    return {"status": "not_observed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="crossover-k4-4096-v1.json")
    parser.add_argument("--prefix", default="crossover")
    args = parser.parse_args()
    source = ROOT / f"measurements/{args.prefix}-summary.json"
    s = json.loads(source.read_text())
    assert s["status"] == "audited"
    protocol = json.loads((ROOT / args.protocol).read_text())
    assert hashlib.sha256((ROOT / args.protocol).read_bytes()).hexdigest() == s["protocol_sha256"]
    for x in s["sources"]:
        assert hashlib.sha256((ROOT / x["path"]).read_bytes()).hexdigest() == x["sha256"]
    targets_to_check = protocol.get("common_loss_targets", TARGETS)
    horizons = protocol["horizons"]
    lookup = {(r["seed"], r["trial"], r["active_experts"], r["horizon"]): r for r in s["records"]}
    assert len(lookup) == len(s["records"]) == 3 * 4 * 2 * len(horizons)
    pairs = []
    equal = []
    targets = []
    dominance = []
    for seed in protocol["seeds"]:
        for trial in protocol["trial_ids"]:
            curves = {k: [lookup[seed, trial, k, h] for h in horizons] for k in (4, 8)}
            assert all(r["fresh_gain"] is not None and r["failure_step"] is None for r in curves[8])
            native_cost = [0] + [r[COST[8]] for r in curves[8]]
            native_gain = [0] + [r["fresh_gain"] for r in curves[8]]
            assert all(b > a for a, b in pairwise(native_cost))
            for d in curves[4]:
                c = d[COST[4]]
                covered = c <= native_cost[-1] and d["fresh_gain"] is not None and d["failure_step"] is None
                ng = float(np.interp(c, native_cost, native_gain)) if covered else None
                equal.append(
                    {
                        "seed": seed,
                        "trial": trial,
                        "horizon": d["horizon"],
                        "draft_cost": c,
                        "covered_by_native_measurements": covered,
                        "draft_gain": d["fresh_gain"],
                        "native_interpolated_gain": ng,
                        "draft_equal_cost_gain_advantage": d["fresh_gain"] - ng if covered else None,
                    }
                )
                if d["fresh_gain"] is not None and d["failure_step"] is None:
                    for n in curves[8]:
                        relation = (
                            "draft_dominates"
                            if d["fresh_gain"] > n["fresh_gain"] and c < n[COST[8]]
                            else ("native_dominates" if n["fresh_gain"] > d["fresh_gain"] and n[COST[8]] < c else None)
                        )
                        if relation:
                            dominance.append(
                                {
                                    "seed": seed,
                                    "trial": trial,
                                    "draft_horizon": d["horizon"],
                                    "native_horizon": n["horizon"],
                                    "relation": relation,
                                    "draft_gain": d["fresh_gain"],
                                    "native_gain": n["fresh_gain"],
                                    "draft_cost": c,
                                    "native_cost": n[COST[8]],
                                }
                            )
            for target in targets_to_check:
                x = {k: target_crossing(curves[k], target, k) for k in (4, 8)}
                ratio = (
                    x[4]["interpolated_cost"] / x[8]["interpolated_cost"]
                    if all(v["status"] == "crossed" for v in x.values())
                    else None
                )
                targets.append(
                    {
                        "seed": seed,
                        "trial": trial,
                        "target": target,
                        "draft": x[4],
                        "native": x[8],
                        "draft_over_native_interpolated_cost": ratio,
                    }
                )
            pairs.append({"seed": seed, "trial": trial, "curves": curves})
    trajectory = []
    for h in horizons:
        draft = [lookup[s, t, 4, h] for s in protocol["seeds"] for t in protocol["trial_ids"]]
        native = [lookup[s, t, 8, h] for s in protocol["seeds"] for t in protocol["trial_ids"]]
        eq = [r for r in equal if r["horizon"] == h and r["covered_by_native_measurements"]]
        valid = [r for r in draft if r["fresh_gain"] is not None]
        advantages = [r["draft_equal_cost_gain_advantage"] for r in eq]
        trajectory.append(
            {
                "horizon": h,
                "accepted": sum(r["accepted"] for r in draft),
                "trials": 12,
                "finite_draft_endpoints": len(valid),
                "mean_draft_gain": mean(r["fresh_gain"] for r in valid) if valid else None,
                "mean_native_gain": mean(r["fresh_gain"] for r in native),
                "median_minimal_cost_ratio": (
                    median(
                        d[COST[4]] / n[COST[8]] for d, n in zip(draft, native, strict=True) if d["failure_step"] is None
                    )
                    if any(d["failure_step"] is None for d in draft)
                    else None
                ),
                "equal_cost_covered_pairs": len(eq),
                "draft_better_equal_cost_pairs": sum(v > 0 for v in advantages),
                "median_equal_cost_gain_advantage": median(advantages) if advantages else None,
                "equal_cost_advantage_quartiles": np.quantile(advantages, [0.25, 0.75]).tolist() if advantages else None,
                "per_seed_median_advantage": {
                    str(seed): median(r["draft_equal_cost_gain_advantage"] for r in eq if r["seed"] == seed)
                    for seed in protocol["seeds"]
                    if any(r["seed"] == seed for r in eq)
                },
            }
        )
    sustained = []
    for i in range(len(trajectory) - 2):
        window = trajectory[i : i + 3]
        if all(r["equal_cost_covered_pairs"] == 12 and r["median_equal_cost_gain_advantage"] > 0 for r in window):
            sustained.append([r["horizon"] for r in window])
    target_summary = []
    for t in targets_to_check:
        group = [r for r in targets if r["target"] == t]
        ratios = [
            r["draft_over_native_interpolated_cost"]
            for r in group
            if r["draft_over_native_interpolated_cost"] is not None
        ]
        target_summary.append(
            {
                "target": t,
                "draft_crossed": sum(r["draft"]["status"] == "crossed" for r in group),
                "native_crossed": sum(r["native"]["status"] == "crossed" for r in group),
                "paired_crossings": len(ratios),
                "draft_cheaper_pairs": sum(v < 1 for v in ratios),
                "median_cost_ratio": median(ratios) if ratios else None,
            }
        )
    result = {
        "status": "audited",
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": (
            f"Twelve paired branches,3checkpoints and4datawindows. Fixed{max(horizons)}horizon; "
            "no adaptive selection. Shared windows/checkpoints limit independence."
        ),
        "cost_basis": (
            "Draft updates+initialcopy+startingcheck+candidatecheck; native update execu"
            "tion. Both8H100. Data preload/compile/restore/extra checks/monitoring exclu"
            "ded and separately reported."
        ),
        "interpolation": (
            "Linear gain-vs-cost interpolation inside native observed cost range; no equ"
            "al-cost extrapolation. Common targets use first recorded crossing and adjac"
            "ent preceding sample, not exact first true crossing. At draft zero-update a"
            "nchor the first endpoint fixed overhead is retained."
        ),
        "sustained_rule": (
            "Positive paired median equal-cost advantage at three consecutive declared h"
            "orizons with all12pairs bracketed. This describes observed curves, not stat"
            "istical significance or indefinite advantage."
        ),
        "first_sustained_window": sustained[0] if sustained else None,
        "all_sustained_windows": sustained,
        "trajectory": trajectory,
        "target_summary": target_summary,
        "targets": targets,
        "equal_cost_records": equal,
        "measured_endpoint_dominance": dominance,
        "pairs": pairs,
    }
    (ROOT / f"measurements/{args.prefix}-efficiency.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    colors = {8: "#17678c", 4: "#c36622"}
    for k, label in [(8, "Native 8: updates"), (4, "Draft 4: updates + minimal copy/checks")]:
        for p in pairs:
            rs = [r for r in p["curves"][k] if r["fresh_gain"] is not None]
            axes[0].plot(
                [r[COST[k]] * 8 / 3600 for r in rs], [r["fresh_gain"] for r in rs], color=colors[k], alpha=0.12, lw=0.8
            )
        xs = []
        ys = []
        for h in horizons:
            rs = [
                lookup[seed, trial, k, h]
                for seed in protocol["seeds"]
                for trial in protocol["trial_ids"]
                if lookup[seed, trial, k, h]["fresh_gain"] is not None
            ]
            xs.append(mean(r[COST[k]] for r in rs) * 8 / 3600 if rs else float("nan"))
            ys.append(mean(r["fresh_gain"] for r in rs) if rs else float("nan"))
        axes[0].plot(xs, ys, "o-", ms=3, color=colors[k], label=label)
        floor = horizons.index(1674)
        axes[0].scatter([xs[floor]], [ys[floor]], marker="s", s=55, color=colors[k])
        axes[0].annotate(
            "LR floor", (xs[floor], ys[floor]), xytext=(3, 8 if k == 8 else -16), textcoords="offset points", fontsize=8
        )
    valid = [r for r in trajectory if r["equal_cost_covered_pairs"] == 12]
    hs = [r["horizon"] for r in valid]
    med = [r["median_equal_cost_gain_advantage"] for r in valid]
    axes[1].fill_between(
        hs,
        [r["equal_cost_advantage_quartiles"][0] for r in valid],
        [r["equal_cost_advantage_quartiles"][1] for r in valid],
        alpha=0.2,
        color=colors[4],
        label="Across-window interquartile range",
    )
    axes[1].plot(hs, med, "o-", ms=4, color=colors[4], label="Paired median")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].axvline(1674, color="gray", ls=":", label="Original LR schedule ends")
    axes[0].set(
        xlabel="Execution cost (H100 GPU-hours)",
        ylabel="Fresh-loss reduction (nats/token)",
        title=f"Measured {max(horizons):,}-update continuation",
    )
    axes[1].set(
        xlabel="Draft updates from step 1,024",
        ylabel="Draft gain minus interpolated native gain",
        title="Equal execution cost: positive favors draft",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(ROOT / f"measurements/{args.prefix}-efficiency.svg")
    fig.savefig(ROOT / f"measurements/{args.prefix}-efficiency.png", dpi=160)
    print(
        json.dumps(
            {
                "first_sustained_window": result["first_sustained_window"],
                "final": trajectory[-1],
                "targets": target_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
