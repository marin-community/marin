# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Post-hoc paired cost/loss interpolation and explicitly conditional trend scenarios."""

import hashlib
import json
from pathlib import Path
from statistics import mean, median

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
HORIZONS = (64, 128, 256, 512)


def main():
    path = ROOT / "measurements/frontier-screen-summary.json"
    data = json.loads(path.read_text())
    assert data["status"] == "audited"
    for source in data["sources"]:
        assert hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() == source["sha256"]
    lookup = {(r["seed"], r["trial"], r["active_experts"], r["horizon"]): r for r in data["records"]}
    costs = {4: "single_horizon_execution_seconds", 8: "update_seconds"}
    pairs = []
    for seed in range(3):
        for trial in range(4):
            rows = {k: [lookup[seed, trial, k, h] for h in HORIZONS] for k in (4, 8)}
            draft, native = rows[4][-1], rows[8][-1]
            earlier_native = rows[8][-2]
            fraction = (draft["fresh_gain"] - earlier_native["fresh_gain"]) / (
                native["fresh_gain"] - earlier_native["fresh_gain"]
            )
            assert 0 < fraction < 1
            native_matching_steps = 256 + fraction * 256
            native_matching_time = earlier_native["update_seconds"] + fraction * (
                native["update_seconds"] - earlier_native["update_seconds"]
            )
            c = draft[costs[4]]
            cost_fraction = (c - earlier_native["update_seconds"]) / (
                native["update_seconds"] - earlier_native["update_seconds"]
            )
            assert 0 < cost_fraction < 1
            native_equal_cost_gain = earlier_native["fresh_gain"] + cost_fraction * (
                native["fresh_gain"] - earlier_native["fresh_gain"]
            )
            trends = []
            for lo, hi in ((128, 256), (256, 512)):
                endpoints = {k: (lookup[seed, trial, k, lo], lookup[seed, trial, k, hi]) for k in (4, 8)}
                slopes = {
                    k: (b["fresh_gain"] - a["fresh_gain"]) / (b[costs[k]] - a[costs[k]])
                    for k, (a, b) in endpoints.items()
                }
                assert all(v > 0 for v in slopes.values())
                # For sensitivity, both historical slopes are anchored at the final measured endpoint.
                intercepts = {k: rows[k][-1]["fresh_gain"] - slopes[k] * rows[k][-1][costs[k]] for k in (4, 8)}
                crossover = (intercepts[8] - intercepts[4]) / (slopes[4] - slopes[8]) if slopes[4] > slopes[8] else None
                if crossover is not None and crossover <= max(c, native["update_seconds"]):
                    crossover = None
                projected_steps = {}
                if crossover is not None:
                    for k in (4, 8):
                        a, b = endpoints[k]
                        seconds_per_step = (b[costs[k]] - a[costs[k]]) / (hi - lo)
                        projected_steps[str(k)] = 512 + (crossover - rows[k][-1][costs[k]]) / seconds_per_step
                # A fixed illustrative future cost budget: twice native's measured512-update cost.
                budget = 2 * native["update_seconds"]
                trends.append(
                    {
                        "slope_interval": [lo, hi],
                        "draft_gain_per_second": slopes[4],
                        "native_gain_per_second": slopes[8],
                        "draft_over_native_marginal_efficiency": slopes[4] / slopes[8],
                        "future_crossover_seconds": crossover,
                        "crossover_projected_update_counts": projected_steps,
                        "crossover_within_original_schedule": (
                            bool(projected_steps) and all(1024 + n <= 2698 for n in projected_steps.values())
                        ),
                        "illustrative_cost_budget_seconds": budget,
                        "projected_draft_gain": slopes[4] * budget + intercepts[4],
                        "projected_native_gain": slopes[8] * budget + intercepts[8],
                    }
                )
            pairs.append(
                {
                    "seed": seed,
                    "trial": trial,
                    "draft512_accepted": draft["accepted"],
                    "draft512_gain": draft["fresh_gain"],
                    "native512_gain": native["fresh_gain"],
                    "draft512_update_seconds": draft["update_seconds"],
                    "draft512_minimal_seconds": c,
                    "native512_update_seconds": native["update_seconds"],
                    "native_interpolated_matching_steps": native_matching_steps,
                    "native_interpolated_matching_seconds": native_matching_time,
                    "draft_update_cost_over_interpolated_native": draft["update_seconds"] / native_matching_time,
                    "draft_minimal_cost_over_interpolated_native": c / native_matching_time,
                    "native_interpolated_equal_cost_gain": native_equal_cost_gain,
                    "native_equal_cost_gain_advantage": native_equal_cost_gain - draft["fresh_gain"],
                    "trends": trends,
                }
            )
    cells = []
    for h in HORIZONS:
        a = [lookup[s, t, 4, h] for s in range(3) for t in range(4)]
        b = [lookup[s, t, 8, h] for s in range(3) for t in range(4)]
        cells.append(
            {
                "horizon": h,
                "accepted": sum(r["accepted"] for r in a),
                "trials": 12,
                "draft_mean_gain": mean(r["fresh_gain"] for r in a),
                "native_mean_gain": mean(r["fresh_gain"] for r in b),
                "median_update_cost_ratio": median(
                    x["update_seconds"] / y["update_seconds"] for x, y in zip(a, b, strict=True)
                ),
                "median_minimal_cost_ratio": median(x[costs[4]] / y[costs[8]] for x, y in zip(a, b, strict=True)),
                "pooled_gain_per_update_second_ratio": (
                    (sum(x["fresh_gain"] for x in a) / sum(x["update_seconds"] for x in a))
                    / (sum(y["fresh_gain"] for y in b) / sum(y["update_seconds"] for y in b))
                ),
                "pooled_gain_per_minimal_second_ratio": (
                    (sum(x["fresh_gain"] for x in a) / sum(x[costs[4]] for x in a))
                    / (sum(y["fresh_gain"] for y in b) / sum(y[costs[8]] for y in b))
                ),
            }
        )
    trends = []
    for i in range(2):
        ts = [p["trends"][i] for p in pairs]
        crosses = [t["future_crossover_seconds"] for t in ts if t["future_crossover_seconds"] is not None]
        steps = [t["crossover_projected_update_counts"]["4"] for t in ts if t["future_crossover_seconds"] is not None]
        trends.append(
            {
                "slope_interval": ts[0]["slope_interval"],
                "draft_marginal_efficiency_better_pairs": sum(
                    t["draft_over_native_marginal_efficiency"] > 1 for t in ts
                ),
                "median_marginal_efficiency_ratio": median(t["draft_over_native_marginal_efficiency"] for t in ts),
                "finite_future_crossover_pairs": len(crosses),
                "crossover_within_original_schedule_pairs": sum(t["crossover_within_original_schedule"] for t in ts),
                "conditional_median_crossover_seconds": median(crosses) if crosses else None,
                "conditional_crossover_seconds_range": [min(crosses), max(crosses)] if crosses else None,
                "conditional_median_draft_updates_at_crossover": median(steps) if steps else None,
                "draft_better_at_illustrative_budget_pairs": sum(
                    t["projected_draft_gain"] > t["projected_native_gain"] for t in ts
                ),
            }
        )
    result = {
        "analysis_type": "Post-hoc descriptive analysis; no new GPU runs and no change to frozen selection.",
        "source": str(path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope": (
            "Twelve paired independent checkpoint/data-window branches on the1.03B model"
            ", shared three seeds. Full-model fresh loss at endpoints. Four experts reta"
            "in native total mixture weight; earlier forward gain0.75 configuration is d"
            "istinct."
        ),
        "cost_basis": (
            "Both arms use8H100s. Draft minimal cost=update execution+initial copy+start"
            "ing check+candidate check. Native cost=update execution. GPU-hours=seconds*"
            "8/3600. Excludes data loading, restore, compilation, fresh monitoring and o"
            "ther research overhead. No total FLOP count or complete production wall-clo"
            "ck measurement."
        ),
        "interpolation": (
            "For each pair, linearly interpolate native loss gain and execution cost bet"
            "ween measured256 and512 endpoints to match draft512 gain. No extrapolation "
            "in this comparison; actual first target crossing remains unknown."
        ),
        "projection": (
            "Extend constant gain/second slopes from128\u2013256 or256\u2013512, anchored at each "
            "arm measured512 endpoint. Both are sensitivity scenarios, not fitted pretra"
            "ining scaling laws or validated forecasts. Conditional crossover statistics"
            " omit no-crossover pairs explicitly counted; no population confidence inter"
            "val."
        ),
        "efficiency_aggregation": (
            "Pooled cumulative gain/second ratios divide summed endpoint gains by summed"
            " costs within each arm, then compare arms. All finite endpoints, including "
            "rejected proposals, are included. This is proposal quality per cost; it doe"
            "s not credit a rejected block as an accepted training improvement. Paired c"
            "ost ratios use medians and therefore need not equal ratios of pooled means."
        ),
        "cells": cells,
        "matched_loss": {
            "pairs": 12,
            "draft_more_costly_pairs": sum(p["draft_minimal_cost_over_interpolated_native"] > 1 for p in pairs),
            "median_native_matching_steps": median(p["native_interpolated_matching_steps"] for p in pairs),
            "median_update_cost_ratio": median(p["draft_update_cost_over_interpolated_native"] for p in pairs),
            "median_minimal_cost_ratio": median(p["draft_minimal_cost_over_interpolated_native"] for p in pairs),
            "mean_native_equal_cost_gain_advantage": mean(p["native_equal_cost_gain_advantage"] for p in pairs),
        },
        "trend_sensitivity": trends,
        "pairs": pairs,
    }
    out = ROOT / "measurements/four-expert-efficiency.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    colors = {4: "#c36622", 8: "#17678c"}
    for k, label in [(8, "Native: update execution"), (4, "4 experts: updates + minimal checks/copy")]:
        for seed in range(3):
            for trial in range(4):
                rr = [lookup[seed, trial, k, h] for h in HORIZONS]
                axes[0].plot(
                    [r[costs[k]] * 8 / 3600 for r in rr],
                    [r["fresh_gain"] for r in rr],
                    color=colors[k],
                    alpha=0.13,
                    lw=0.8,
                )
        xs = [mean(lookup[s, t, k, h][costs[k]] for s in range(3) for t in range(4)) for h in HORIZONS]
        ys = [mean(lookup[s, t, k, h]["fresh_gain"] for s in range(3) for t in range(4)) for h in HORIZONS]
        axes[0].plot([x * 8 / 3600 for x in xs], ys, "o-", color=colors[k], label=label)
        for x, y, h in zip(xs, ys, HORIZONS, strict=True):
            axes[0].annotate(str(h), (x * 8 / 3600, y), xytext=(3, 5), textcoords="offset points", fontsize=8)
        axes[1].plot(xs, ys, "o-", color=colors[k], label=label)
        for idx, style in [(1, ":"), (2, "--")]:
            slope = (ys[idx + 1] - ys[idx]) / (xs[idx + 1] - xs[idx])
            end = 250
            axes[1].plot([xs[-1], end], [ys[-1], ys[-1] + slope * (end - xs[-1])], style, color=colors[k], alpha=0.85)
    axes[0].set(
        xlabel="Eight-H100 execution cost (GPU-hours)",
        ylabel="Full-model fresh-loss reduction (nats/token)",
        title="Measured endpoints: equal update counts differ in quality",
    )
    axes[1].axvline(
        max(mean(lookup[s, t, k, 512][costs[k]] for s in range(3) for t in range(4)) for k in (4, 8)),
        color="gray",
        lw=0.7,
    )
    axes[1].set(
        xlabel="Execution cost (seconds on eight H100s)",
        ylabel="Full-model fresh-loss reduction (nats/token)",
        title="Conditional extensions; not validated forecasts",
    )
    axes[1].text(
        0.03,
        0.96,
        ("Dotted:128\u2013256 slope; dashed:256\u2013512 slope\nBoth anchored at measured512 end" "point"),
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.savefig(ROOT / "measurements/four-expert-efficiency.svg")
    fig.savefig(ROOT / "measurements/four-expert-efficiency.png", dpi=160)
    print(json.dumps({k: result[k] for k in ["cells", "matched_loss", "trend_sensitivity"]}, indent=2))


if __name__ == "__main__":
    main()
