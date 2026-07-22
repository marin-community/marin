# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tier-1 test: does the surrogate generalize to novel CONTENT far from the training data?

f43 showed the model failing off-support, but its off-support points are all epoch-stress probes, so
they confound content-extrapolation with the epoch axis the content features cannot encode. This
isolates the content question using ONLY the 800 swarm runs -- every one of which is at normal epoch
counts, so there is no epoch confound.

Method: leave-one-content-CLUSTER-out. Cluster the 800 runs in content space (agglomerative on the
Hellinger distance the kernel uses), then hold out whole clusters and predict them from the rest.
Removing a whole neighbourhood pushes each held-out run's distance-to-nearest-retained-run UP -- into
the gap on f43's x-axis between the dense core (~0.31) and the epoch probes (~0.44) -- but with real,
normal-epoch runs. If |error| stays flat as that held-out distance grows, content extrapolation is
robust and f43's explosion was purely the epoch axis. If it climbs, the model is fragile to novel
content too.

Random 5-fold is the short-distance reference (interpolation); LOCO is the extrapolation stress.
"""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from generalization_vs_distance import humaneval_target
from gp_inversion_study import load_all
from gp_surrogate import fit_gp, predict_gp
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import KFold

GRUG = Path("scratch/mixture_features/grug")
FIG = Path("scratch/mixture_features/report/figs3/f44_content_extrapolation.png")
N_CLUSTERS = 20  # ~40 runs/cluster; coarse enough to push held-out distance into f43's gap
N_FOLDS = 5
SEED = 0


def _holdout_predict(d2, y, hold_mask):
    """Predict the held-out runs from the retained ones; return per-run (err, sd, nn-to-retained)."""
    tr = np.where(~hold_mask)[0]
    te = np.where(hold_mask)[0]
    f = fit_gp(d2[np.ix_(tr, tr)], y[tr])
    mu, sd = predict_gp(f, d2[np.ix_(te, tr)], include_noise=True)
    nn = np.sqrt(np.clip(d2[np.ix_(te, tr)].min(axis=1), 0.0, None))
    return te, np.abs(mu - y[te]), sd, mu, nn


def evaluate(d2, y, labels):
    n = len(y)
    # --- random 5-fold (interpolation reference) ---
    r_err = np.full(n, np.nan)
    r_nn = np.full(n, np.nan)
    r_pred = np.full(n, np.nan)
    for _tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.arange(n)):
        m = np.zeros(n, bool)
        m[te] = True
        idx, err, _sd, mu, nn = _holdout_predict(d2, y, m)
        r_err[idx], r_nn[idx], r_pred[idx] = err, nn, mu
    # --- leave-one-content-cluster-out (extrapolation stress) ---
    c_err = np.full(n, np.nan)
    c_nn = np.full(n, np.nan)
    c_pred = np.full(n, np.nan)
    for c in np.unique(labels):
        m = labels == c
        idx, err, _sd, mu, nn = _holdout_predict(d2, y, m)
        c_err[idx], c_nn[idx], c_pred[idx] = err, nn, mu
    return {
        "random": {"err": r_err, "nn": r_nn, "pred": r_pred},
        "loco": {"err": c_err, "nn": c_nn, "pred": c_pred},
    }


def _binmed(x, yv, edges):
    xs, md = [], []
    for b in range(len(edges) - 1):
        m = (x >= edges[b]) & (x < edges[b + 1])
        if m.sum() >= 8:
            xs.append(0.5 * (edges[b] + edges[b + 1]))
            md.append(np.median(yv[m]))
    return np.array(xs), np.array(md)


def main():
    d2, zmacro, *_ = load_all()
    hev = humaneval_target()
    ok = np.isfinite(hev)
    d2 = d2[np.ix_(ok, ok)]
    targets = {"zmacro": zmacro[ok], "humaneval": hev[ok]}

    labels = AgglomerativeClustering(n_clusters=N_CLUSTERS, metric="precomputed", linkage="average").fit_predict(
        np.sqrt(np.clip(d2, 0.0, None))
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    summary = {"n": int(d2.shape[0]), "n_clusters": N_CLUSTERS, "targets": {}}

    for ax, (name, y) in zip(axes, targets.items(), strict=False):
        res = evaluate(d2, y, labels)
        rnd, loco = res["random"], res["loco"]
        # headline: does the model still rank held-out content? (LOCO vs random Spearman)
        s_rnd = float(spearmanr(rnd["pred"], y).statistic)
        s_loco = float(spearmanr(loco["pred"], y).statistic)

        ax.scatter(rnd["nn"], rnd["err"], s=9, c="#9fb0c8", alpha=0.5, label="random 5-fold (interpolation)")
        ax.scatter(loco["nn"], loco["err"], s=14, c="#b02a37", alpha=0.55, label="leave-content-cluster-out")
        allnn = np.concatenate([rnd["nn"], loco["nn"]])
        edges = np.linspace(np.nanmin(allnn), np.nanmax(allnn), 10)
        bx, bm = _binmed(loco["nn"], loco["err"], edges)
        ax.plot(bx, bm, "-", color="#7a0f1a", lw=2.6, label="LOCO median |error|")
        rx, rm = _binmed(rnd["nn"], rnd["err"], edges)
        ax.plot(rx, rm, "-", color="#2b4fbf", lw=2.2, label="random median |error|")
        p95 = float(np.nanquantile(rnd["nn"], 0.95))
        ax.axvline(p95, color="k", ls=":", lw=1.3)
        ax.text(p95, ax.get_ylim()[1] * 0.97, " random-CV p95", fontsize=9, va="top")
        ax.set_xlabel("nearest-RETAINED-run Hellinger distance", fontsize=11.5)
        ax.set_ylabel(f"|prediction error|  ({name})", fontsize=11.5)
        ax.set_title(
            f"{name}: content extrapolation (no epoch confound)\n"
            f"held-out-cluster Spearman {s_loco:.3f}  vs  random-fold {s_rnd:.3f}",
            fontsize=11.5,
        )
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.25)

        far = loco["nn"] > p95  # runs pushed beyond the random-CV support edge
        summary["targets"][name] = {
            "random_fold_spearman": s_rnd,
            "loco_spearman": s_loco,
            "loco_rmse": float(np.sqrt(np.nanmean(loco["err"] ** 2))),
            "random_rmse": float(np.sqrt(np.nanmean(rnd["err"] ** 2))),
            "random_median_abs_err": float(np.nanmedian(rnd["err"])),
            "loco_median_abs_err": float(np.nanmedian(loco["err"])),
            "loco_beyond_p95_n": int(far.sum()),
            "loco_beyond_p95_median_abs_err": float(np.nanmedian(loco["err"][far])) if far.any() else None,
            "loco_max_nn_hellinger": float(np.nanmax(loco["nn"])),
            "random_max_nn_hellinger": float(np.nanmax(rnd["nn"])),
        }

    fig.tight_layout()
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    with open(GRUG / "content_extrapolation.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"wrote {FIG}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
