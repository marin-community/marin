# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""How much does the surrogate generalize as you move AWAY from the training data?

The out-of-fold scores everyone quotes (kernel ho ~0.94 on humaneval) are measured where the
held-out runs are drawn from the SAME Dirichlet design as training -- that is interpolation, not
extrapolation. This figure puts both regimes on one axis: prediction error vs distance-to-nearest-
training-run, from the dense in-distribution core (the 800 swarm runs, out-of-fold) out through the
53 deliberately-extreme off-design probe runs (~2x the median neighbour distance), whose outcomes we
also know. It answers the skeptic's question -- "how far can we push before the fit stops holding" --
as a curve rather than two disconnected numbers.

Target: humaneval bpb (complete for both the swarm and every probe run). LOWER is better.

In-distribution side: computed here (5-fold out-of-fold, per-run |error|, predicted sd, and nearest-
neighbour Hellinger distance to that run's training fold). Off-design side: reused verbatim from
gp_ood_coverage.json (its 53-run reconstruction is verified to 9.4e-15 against the pre-registered
kernel predictions), so this module does not re-derive the probe mixtures.
"""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gp_inversion_study import load_all
from gp_surrogate import fit_gp, predict_gp
from sklearn.model_selection import KFold

GRUG = Path("scratch/mixture_features/grug")
FIG = Path("scratch/mixture_features/report/figs3/f43_generalization_vs_distance.png")
CODE_TASK = "logprob_humaneval_10shot"
N_FOLDS = 5
GROUP_COLORS = {
    "twobucket": "#b02a37",
    "epochrep": "#8a5e00",
    "transect": "#0f7a3d",
    "harm100b": "#2b4fbf",
}


def humaneval_target() -> np.ndarray:
    train = pd.read_parquet(GRUG / "train_runs.parquet")
    out = []
    for ej in train.evals:
        ev = json.loads(ej)
        e = ev.get(CODE_TASK)
        out.append(float(e["bpb"]) if isinstance(e, dict) and "bpb" in e else np.nan)
    return np.array(out)


def in_distribution(d2: np.ndarray, y: np.ndarray) -> dict:
    """5-fold out-of-fold: per-run |error|, predicted sd, nearest-train Hellinger distance."""
    n = len(y)
    miss = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    nn = np.full(n, np.nan)
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=0).split(np.arange(n)):
        f = fit_gp(d2[np.ix_(tr, tr)], y[tr])
        mu, s = predict_gp(f, d2[np.ix_(te, tr)], include_noise=True)
        miss[te] = np.abs(mu - y[te])
        sd[te] = s
        nn[te] = np.sqrt(np.clip(d2[np.ix_(te, tr)].min(axis=1), 0.0, None))  # nearest-train Hellinger
    return {"abs_err": miss, "sd": sd, "nn_hellinger": nn}


def offdesign() -> list[dict]:
    with open(GRUG / "gp_ood_coverage.json") as fh:
        d = json.load(fh)
    return [
        {"group": r["group"], "abs_err": abs(r["miss"]), "sd": r["gp_sd"], "nn_hellinger": r["nn_hellinger"]}
        for r in d["humaneval"]["per_run"]
    ]


def _binned(x, yv, edges):
    idx = np.digitize(x, edges) - 1
    xs, med, lo, hi = [], [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= 5:
            xs.append(0.5 * (edges[b] + edges[b + 1]))
            q = np.percentile(yv[m], [25, 50, 75])
            lo.append(q[0])
            med.append(q[1])
            hi.append(q[2])
    return np.array(xs), np.array(med), np.array(lo), np.array(hi)


def main():
    d2, _zmacro, *_ = load_all()  # need only d2; target reloaded as humaneval below
    y = humaneval_target()
    ok = np.isfinite(y)
    d2, y = d2[np.ix_(ok, ok)], y[ok]

    ind = in_distribution(d2, y)
    off = offdesign()
    p95 = float(np.quantile(ind["nn_hellinger"], 0.95))
    with open(GRUG / "gp_ood_coverage.json") as fh:
        noise = json.load(fh)["humaneval"]["noise_sd"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 7.2), gridspec_kw={"width_ratios": [1.55, 1]})

    # ---- LEFT: |error| vs distance ----
    ax.scatter(ind["nn_hellinger"], ind["abs_err"], s=10, c="#9fb0c8", alpha=0.5, label="800 swarm runs (out-of-fold)")
    edges = np.linspace(ind["nn_hellinger"].min(), ind["nn_hellinger"].max(), 9)
    bx, bm, bl, bh = _binned(ind["nn_hellinger"], ind["abs_err"], edges)
    ax.plot(bx, bm, "-", color="#2b4fbf", lw=2.5, label="in-dist median |error|")
    ax.fill_between(bx, bl, bh, color="#2b4fbf", alpha=0.15)
    # predicted sd trend (does uncertainty track the error?)
    _, sm, _, _ = _binned(ind["nn_hellinger"], ind["sd"], edges)
    ax.plot(bx, sm, "--", color="#0f7a3d", lw=2.0, label="in-dist median predicted sd")
    for g in GROUP_COLORS:
        pts = [r for r in off if r["group"] == g]
        if pts:
            ax.scatter(
                [r["nn_hellinger"] for r in pts],
                [r["abs_err"] for r in pts],
                s=60,
                c=GROUP_COLORS[g],
                edgecolor="k",
                lw=0.4,
                zorder=5,
                label=f"off-design: {g} (n={len(pts)})",
            )
    off_sd = np.mean([r["sd"] for r in off])
    ax.axvline(p95, color="k", ls=":", lw=1.5)
    ax.text(p95, ax.get_ylim()[1] * 0.96, " in-dist p95\n (edge of support)", fontsize=10, va="top")
    ax.axhline(noise, color="#888", ls="-", lw=1, alpha=0.7)
    ax.text(ax.get_xlim()[1], noise, " seed floor", fontsize=9, color="#555", va="bottom", ha="right")
    ax.set_xlabel("nearest-training-run Hellinger distance  (further right = more extrapolation)", fontsize=12)
    ax.set_ylabel("|prediction error|  (humaneval bpb)", fontsize=12)
    ax.set_title(
        "Generalization decays with distance from the training data\n"
        "error explodes off-support while predicted uncertainty barely moves",
        fontsize=12.5,
    )
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.25)

    # ---- RIGHT: calibration (coverage@2sd) vs distance ----
    all_nn = np.concatenate([ind["nn_hellinger"], [r["nn_hellinger"] for r in off]])
    all_z = np.concatenate([ind["abs_err"] / ind["sd"], [r["abs_err"] / r["sd"] for r in off]])
    qedges = np.quantile(all_nn, np.linspace(0, 1, 8))
    cx, cov = [], []
    for b in range(len(qedges) - 1):
        m = (all_nn >= qedges[b]) & (all_nn < qedges[b + 1]) if b < len(qedges) - 2 else (all_nn >= qedges[b])
        if m.sum() >= 5:
            cx.append(0.5 * (qedges[b] + qedges[b + 1]))
            cov.append(100 * np.mean(all_z[m] < 2.0))
    ax2.plot(cx, cov, "o-", color="#b02a37", lw=2.2, ms=7)
    ax2.axhline(95, color="#0f7a3d", ls="--", lw=1.5)
    ax2.text(cx[0], 95, "nominal 95%", fontsize=9.5, color="#0f7a3d", va="bottom")
    ax2.axvline(p95, color="k", ls=":", lw=1.5)
    ax2.set_xlabel("nearest-training-run Hellinger distance", fontsize=12)
    ax2.set_ylabel("% of runs inside their 95% interval", fontsize=12)
    ax2.set_title("...and the error bars stop covering:\nthe model does not know it is extrapolating", fontsize=12.5)
    ax2.set_ylim(0, 103)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    print(f"wrote {FIG}")

    summary = {
        "target": "humaneval_bpb",
        "n_in_distribution": len(y),
        "n_offdesign": len(off),
        "in_dist_median_nn_hellinger": float(np.median(ind["nn_hellinger"])),
        "in_dist_p95_nn_hellinger": p95,
        "offdesign_mean_nn_hellinger": float(np.mean([r["nn_hellinger"] for r in off])),
        "in_dist_median_abs_err": float(np.nanmedian(ind["abs_err"])),
        "offdesign_mean_abs_err": float(np.mean([r["abs_err"] for r in off])),
        "in_dist_median_sd": float(np.nanmedian(ind["sd"])),
        "offdesign_mean_sd": float(off_sd),
        "err_growth_factor": float(np.mean([r["abs_err"] for r in off]) / np.nanmedian(ind["abs_err"])),
        "sd_growth_factor": float(off_sd / np.nanmedian(ind["sd"])),
    }
    with open(GRUG / "generalization_vs_distance.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
