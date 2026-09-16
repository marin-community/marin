# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "matplotlib==3.10.8", "tabulate==0.9.0"]
# ///
"""Render the frozen TPP40 calibration, completion and conditional-gap evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent / "reference_outputs"
OUTPUT = BASE / "tpp40_frontier_gap_20260909"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#A8ADB3"


def frontier(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [group.nsmallest(int(np.ceil(len(group) * 0.2)), "prediction") for _, group in frame.groupby("context")]
    )


def score(frame: pd.DataFrame) -> dict:
    error = frame.measured - frame.prediction
    return {"n": len(frame), "rmse": float(np.sqrt(np.mean(error**2))), "optimism": float(error.mean())}


def main() -> None:
    data = pd.read_csv(OUTPUT / "inputs/outcomes.csv")
    oof = pd.read_csv(OUTPUT / "tpp40_oof_predictions.csv")
    old = pd.read_csv(BASE / "two_phase_hpr_transfer_20260907/aggregate_replacement/predictions.csv")
    old = old[old.context.ne("final") & old.scored & old.objective.eq("uncheatable")]
    old = old.rename(columns={"predicted": "prediction"})
    mariner = frontier(old[old.model.eq("aggregate") & old.tied])
    transfer = frontier(old[old.model.eq("hpr_aggregate_replacement") & ~old.tied])
    old_hpr = frontier(old[old.model.eq("hpr") & ~old.tied])
    current = frontier(oof[~oof.tied])
    scores = pd.DataFrame(
        [
            {"design": "Completed 520-row", "model": "MARINER, tied", **score(mariner)},
            {"design": "Completed 520-row", "model": "HPR, asymmetric", **score(old_hpr)},
            {"design": "Completed 520-row", "model": "HPR + MARINER, asymmetric", **score(transfer)},
            {"design": "Partial TPP40", "model": "HPR, asymmetric", **score(current)},
        ]
    )
    scores.to_csv(OUTPUT / "pooled_frontier_metrics.csv", index=False)
    current.to_csv(OUTPUT / "tpp40_predicted_frontier_rows.csv", index=False)
    metric = "eval/uncheatable_eval/bpb"
    observed = data.sort_values(metric).head(8)[["order", "run_name", "region", "tied", metric]]
    observed.to_csv(OUTPUT / "best_observed.csv", index=False)
    opt = pd.read_csv(OUTPUT / "optimum_summary.csv")
    full = json.loads((OUTPUT / "optima/full.json").read_text())["results"]["tied"]
    policy_rows = []
    for context in ("outer0", "outer1", "outer2", "east5", "europe"):
        record = json.loads((OUTPUT / "optima" / f"{context}.json").read_text())["results"]["tied"]
        tv = float(np.abs(np.asarray(record["weights"])[0] - np.asarray(full["weights"])[0]).sum() / 2)
        policy_rows.append({"context": context, "tv_from_full_tied_optimum": tv})
    pd.DataFrame(policy_rows).to_csv(OUTPUT / "optimum_weight_stability.csv", index=False)
    cross_region = []
    for context in ("east5", "europe"):
        frame = pd.read_csv(OUTPUT / "fits" / context / "predictions.csv")
        cross_region.append({"training_region": context, **score(frame[~frame.in_train])})
    pd.DataFrame(cross_region).to_csv(OUTPUT / "region_transfer_metrics.csv", index=False)
    proportional = float(data.loc[data.order.eq(0), metric].item())
    unimax = float(data.loc[data.order.eq(1), metric].item())
    best = float(data[~data.tied][metric].min())

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "text.usetex": False,
            "font.family": "DejaVu Sans",
        }
    )
    figure, axes = plt.subplots(
        1, 3, figsize=(16.8, 5.4), layout="constrained", gridspec_kw={"width_ratios": [0.85, 1.2, 1]}
    )
    ax = axes[0]
    ax.bar([0, 1], [42, 238], color="#E1E5E9", label="Planned total", width=0.58)
    ax.bar([0, 1], [2, 156], color=BLUE, label="Completed", width=0.58)
    for x, actual, total in ((0, 2, 42), (1, 156, 238)):
        ax.text(x, total + 8, f"{actual} / {total}", ha="center", fontweight="bold")
    ax.set(
        xticks=[0, 1],
        xticklabels=["Single phase", "Two phase"],
        ylim=(0, 285),
        ylabel="Policies",
        title="A. The missing comparison",
    )
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.5,
        -0.14,
        "Only proportional and Unimax are tied.\n0 of 39 deletion runs completed.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    ax = axes[1]
    asym = oof[~oof.tied]
    ax.scatter(asym.prediction, asym.measured, color=GRAY, s=22, alpha=0.62, label="Other held-out two-phase runs")
    ax.scatter(current.prediction, current.measured, color=BLUE, s=29, label="Predicted best 20% in each fold")
    tied = oof[oof.tied]
    ax.scatter(tied.prediction, tied.measured, marker="D", color=ORANGE, s=55, label="Held-out Unimax")
    bounds = [
        min(oof.prediction.min(), oof.measured.min()) - 0.004,
        max(oof.prediction.max(), oof.measured.max()) + 0.004,
    ]
    ax.plot(bounds, bounds, color="#555555", linewidth=1, linestyle="--", zorder=0)
    ax.set(
        xlim=bounds,
        ylim=bounds,
        xlabel="Held-out prediction (BPB)",
        ylabel="Measured BPB",
        title="B. Frontier error: 0.0080 BPB",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.text(
        0.5,
        -0.19,
        "Above the diagonal = optimistic prediction.\nFrontier points were selected without their outcomes.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    ax = axes[2]
    labels = ["Proportional", "Unimax", "Best observed\ntwo phase", "HPR prediction\n1p = 2p"]
    values = [proportional, unimax, best, full["prediction"]]
    colors = ["#52565C", "#52565C", BLUE, ORANGE]
    for index, (value, color) in enumerate(zip(values, colors, strict=True)):
        ax.scatter(value, index, color=color, s=60, marker="o" if index < 3 else "x", linewidth=2)
        ax.text(value + 0.0018, index, f"{value:.4f}", va="center", fontsize=10)
    ax.axhspan(2.55, 3.45, color=ORANGE, alpha=0.065)
    ax.set(
        yticks=range(4),
        yticklabels=labels,
        ylim=(3.6, -0.6),
        xlim=(0.83, 0.927),
        xlabel="Uncheatable BPB (lower is better)",
        title="C. Observed and extrapolated",
    )
    ax.text(
        0.5,
        -0.14,
        "The optimized point is unmeasured and\nfar outside the observed tied support.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )
    figure.suptitle("TPP40: promising observed mixtures, insufficient evidence about the optimum gap", fontsize=16)
    figure.savefig(OUTPUT / "frontier_diagnostic.png", dpi=170, bbox_inches="tight")
    figure.savefig(OUTPUT / "frontier_diagnostic.svg", bbox_inches="tight")
    figure.savefig(OUTPUT / "frontier_diagnostic.pdf", bbox_inches="tight")
    plt.close(figure)

    source_parity = json.loads((OUTPUT / "mariner_source_parity.json").read_text())
    snapshot = json.loads((OUTPUT / "snapshot.json").read_text())
    mismatches = [
        name
        for name, expected in snapshot["sha256"].items()
        if hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest() != expected
    ]
    assert not mismatches
    comparisons = []
    for context in ("full", "outer0", "outer1", "outer2", "east5", "europe"):
        a = pd.read_csv(OUTPUT / "archive/initial_execution/fits" / context / "predictions.csv")
        b = pd.read_csv(OUTPUT / "fits" / context / "predictions.csv")
        error = float(np.max(np.abs(a.prediction.to_numpy() - b.prediction.to_numpy())))
        assert error == 0
        comparisons.append({"context": context, "prediction_replay_error_after_formatting": error})
    raw_files = list((OUTPUT / "raw").rglob("record.json"))
    collection_window = [
        datetime.fromtimestamp(value, UTC).isoformat()
        for value in (min(p.stat().st_mtime for p in raw_files), max(p.stat().st_mtime for p in raw_files))
    ]
    validation = {
        "snapshot_files_verified": len(snapshot["sha256"]),
        "snapshot_hash_failures": mismatches,
        "formatting_replay": comparisons,
        "mariner_source_parity": source_parity,
        "collection_cache_write_window_utc": collection_window,
        "timestamp_note": "snapshot.started_utc is the resumed collector invocation; cached rows began earlier.",
        "completed_unique_final_endpoints": len(data),
        "scored_unique_oof_endpoints": len(oof),
        "scientific_result": "Independent MARINER comparison blocked by two tied rows; conditional HPR gap unsupported.",
    }
    (OUTPUT / "VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n")
    outer_opt = opt[opt.context.str.startswith("outer") & opt["mode"].eq("tied")]
    report = f"""# TPP40 does not yet identify the one-phase versus two-phase optimum gap

9 September 2026. Fieldbook: `exp_01m24qbk7ha94xj8zqfpvkf81y`.

The completed TPP40 data contain a promising two-phase mixture, but cannot yet
support the requested comparison between independently calibrated one-phase
MARINER and a two-phase optimum. The frozen snapshot has **158 completed runs:
156 asymmetric policies and only two tied policies**, proportional and Unimax.
None of the 39 planned domain deletions has completed. The remaining tied
reference is also pending. Completion is therefore highly unbalanced by policy
class; 158/280 does not mean half of the information needed for this comparison.

![Completion, held-out calibration and raw optimum comparison](frontier_diagnostic.png)

## What is already measured

| Policy | Uncheatable BPB |
|---|---:|
| Proportional, single phase | {proportional:.6f} |
| Unimax, single phase | {unimax:.6f} |
| Best completed two-phase policy, run 200 | {best:.6f} |

The observed improvement over Unimax is **{unimax-best:.6f} BPB**. This compares
the best of 156 asymmetric runs with two single-phase baselines. It does not
separate aggregate mixture quality from temporal ordering, does not compare
optimized policy classes on equal search budgets, and has no matched-seed
uncertainty estimate. The two next-best asymmetric results are runs 168 and
178; all three are from Europe. No region adjustment is applied.

## Frontier calibration

The frontier is the lowest predicted 20% within each outer held-out fold and
policy class, selected before reading those outcomes. Positive optimism means
measured minus predicted BPB: the model predicted a result that was too good.

{scores.to_markdown(index=False, floatfmt='.6f')}

The first three rows use the complete 520-row, approximately 300M/6B-token
development design and its existing aggregate-grouped OOF fits. They assess
different policy classes and are not a paired model leaderboard. The latest
standalone MARINER source has identical fitting and response definitions to
that benchmark's snapshot; the changes concern the uncapped optimizer interface.
Replaying all four saved aggregate fits with the latest prediction code agrees
within {max(source_parity['prediction_replay_max_errors'].values()):.2g} BPB.

The old MARINER single-phase selections have held-pool optimism ranging from
-0.000187 to +0.009433 BPB; the HPR/MARINER two-phase selections from +0.000074
to +0.007227. Better mean calibration does not certify the freely optimized
surface beyond those held-out pools.

On this frontier definition, plain HPR also has lower error and optimism than
the HPR/MARINER replacement on the same asymmetric rows. Improving aggregate
prediction averaged over the full design did not improve this frontier check.

At TPP40, HPR's three selected held-out asymmetric policies are optimistic by
0.003426, 0.003228 and 0.005438 BPB. Their held-fold selection regrets are
recorded separately in `tpp40_frontier_metrics.csv`. Unimax is held out once:
predicted 0.890204, measured 0.898002, optimism **0.007797**. There is no
independent held-out single-phase frontier. TPP40 asymmetric OOF RMSE across
all 156 rows is **0.010372 BPB**; the frontier RMSE is **0.007965 BPB**.

## The conditional HPR calculation

Because two tied observations cannot identify the 39-bucket MARINER procedure,
no two-row MARINER fit or transported single-phase loss surface was substituted.
The diagnostic fits the existing HPR estimator on completed TPP40 outcomes and
compares the best found points of H(w,w) and H(w0,w1). This is a shared-model
comparison whose single-phase response depends mostly on assumptions learned
from asymmetric rows. It is not the requested independently calibrated baseline.

| Fit | Best found tied BPB | Best found two-phase BPB | Predicted gap |
|---|---:|---:|---:|
| All 158 completed rows | {full['prediction']:.6f} | {full['prediction']:.6f} | 0.000000 |

All six fits (full, three outer refits, and two regional sensitivities) choose
a tied policy for both searches. This is a **conditional zero gap, not evidence
that the true two-phase advantage is zero**. The predicted full-fit optimum is
{unimax-full['prediction']:.6f} BPB below the best measured tied baseline and
{best-full['prediction']:.6f} below the best measured policy of either class.
It requires {full['max_epochs']:.2f} maximum simulated epochs, is
{full['support']['convex_hull_policy_tv']:.3f} in phase-weighted TV from the hull
of all completed policies, and {full['tied_training_support']['convex_hull_policy_tv']:.3f}
from the hull of the two observed tied policies. Such a point has no local
calibration evidence. Coordinate-wise epoch range membership would not repair
that joint-support problem.

The outer refits' minima range from {outer_opt.predicted_bpb.min():.6f} to
{outer_opt.predicted_bpb.max():.6f}; including regional fits widens this to
{opt.predicted_bpb.min():.6f} to {opt.predicted_bpb.max():.6f}. These are deletion
sensitivities, not confidence intervals. See `optimum_weight_stability.csv`
for policy movement, and `optimum_summary.csv` for every raw result. No
conditional bootstrap or statistical-significance claim is justified here.

## Procedure and verification

Every response is computed from the same seven Uncheatable components with
the standalone's fixed byte weights and agrees with the logged aggregate to
less than 3e-6 BPB. Every completed observation has SUCCESS status and verified
checkpoint metadata at step 27335; identical repeated final log records count
once. Collection read only small existing files from the two assigned regional
roots. The source snapshot preserves 890 file hashes. Collection ran around
20:56 to 20:59 PDT; the snapshot's start field describes the resumed invocation.

The materialized phase fraction is 21856/27336, with 14,331,936,768 tokens and
358,304,128 total parameters. Simulated pools grow with the experiment budget,
so designed epoch coordinates are preserved; this is not a ninefold increase
in epochs. Coordinates use declared pool token counts and normalized source
weights, as in the existing surrogate, rather than reconstructing individual
sample draws or rounding every policy to a proposed runtime grid.

Three outcome-free aggregate-neighborhood folds keep matching aggregate
coordinates together and pin proportional calibration in every training set.
Inner folds, HPR shapes and hierarchical ridge are reselected using training
data only. Seven component scores are aggregated before the HPR fit; no
phase-boundary outcomes enter this open-loop predictor. The original HPR grid,
nonnegative head and TV term are retained. No KL penalty, epoch cap or added
output cap is used. Eight starts per tied optimization and nine per two-phase
optimization are checked by SLSQP, including the tied solution in the latter;
softmax L-BFGS supplies a solver sensitivity. Analytic values match HPR's own
predictor to 3.4e-16 and directional derivatives to 1.1e-8. This checks the
implementation and local search, not global optimality or scientific validity.

East5 contributes 61 rows up to order 91; Europe contributes 97 up to order
222. Their policy distributions differ. Holding out the opposite region yields
RMSE 0.011472 (train East5) and 0.011766 (train Europe); the Europe fit retains
the East5 proportional calibration point. These mix policy-support and hardware
effects. The existing one-pair bridge's unweighted aggregate delta was +0.002214
Europe minus East5; it narrowly failed its strict numerical threshold and was
allowed under a separate operational decision. That is not an independently
estimated regional correction and is not subtracted from these byte-weighted
scores. The strongest completed mixtures are all in Europe, so comparisons
should preserve region/seed matching where possible.

## What would resolve the question

First, use the pending deletion/reference rows when they finish, but recognize
that they chiefly identify behavior around proportional mixing. They do not
provide the missing single-phase frontier by themselves. The most informative
matched comparison is a TPP40 tied counterpart of a strong two-phase mixture:
use a=(21856/27336)w0+(5480/27336)w1 in both phases, with the same horizon,
exposure convention, evaluation and matched training/data seeds. The difference
then isolates phase ordering at that aggregate.

Pair that evidence with TPP40 measurements of a strong MARINER single-phase
proposal and nearby single-phase candidates. Only after checking their local
calibration should an independently fitted MARINER optimum be compared with a
two-phase optimum, with full-refit uncertainty and prospective matched-seed
validation. Current completed outcomes are development evidence. No language
model training or evaluation jobs were launched in this audit.

## Reproduce

Run `uv run --python 3.12` on the adjacent scripts, in order:
`collect_tpp40_surrogate_snapshot_20260909.py`,
`analyze_tpp40_frontier_gap_20260909.py fit`,
`analyze_tpp40_frontier_gap_20260909.py optimize`,
`analyze_tpp40_frontier_gap_20260909.py summarize`, and
`report_tpp40_frontier_gap_20260909.py`.
The collector reuses its frozen snapshot, and fitting/optimization validate
source/input hashes before reusing completed cells. Choose a new artifact root
for a later completion snapshot. The report also checks prediction parity with
the initial pre-format execution. Exact protocols, memberships, predictions,
optimizer traces, raw metric files and metadata are included here.
"""
    (OUTPUT / "REPORT.md").write_text(report)
    print(scores.to_string(index=False))
    print(OUTPUT / "REPORT.md")


if __name__ == "__main__":
    main()
