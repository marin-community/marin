# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contour / projection views of the mixture surrogate: what does the fitted function LOOK like?

Every earlier figure in this campaign reports the surrogate as a scalar (a Spearman, an
RMSE). This builds the picture behind those scalars: interpretable 2D slices through the
168-bucket simplex on which we can put the MEASURED function and the FITTED functions side
by side on one shared color scale, and see where the methods agree, where they disagree,
and where nobody has data.

Two slices (both built exactly like build_f17/build_f18: pin a group's total share per
phase, keep within-group proportions at their token-proportional anchor ratios, renormalize
everything else proportionally so each phase still sums to 1):

  SLICE A  x = code_adjacent group phase-0 share, y = its phase-1 share (the f18 axes --
           the axis the campaign already showed carries real signal, rho +0.54 vs realized).
  SLICE B  x = code_adjacent overall share, y = cluster c05's overall share, both pinned
           equal in the two phases. c05 is picked FROM THE DATA: of the 33 non-code
           clusters it has by far the largest realized spread (sd 0.157, reaching 0.80 of
           the mix) and the strongest marginal association with zmacro (Spearman +0.52) --
           the only other group whose axis the swarm genuinely explores.

Six panels per slice, the first three on ONE shared color scale:

  1 EMPIRICAL   the 800 real runs binned by their OWN (x, y) coordinates; per-bin mean
                outcome, per-bin run count printed in the cell. This is the reference
                "true function", not a scatter plot.
  2 KERNEL      the campaign's frozen Hellinger kernel ridge (gamma_factor 0.25, alpha 0.1).
  3 WEIGHTS     ridge on the raw 2x168 bucket proportions -- the RegMix-style linear
                baseline -- fit on the same 800 runs.
  4 KERNEL - EMPIRICAL   and  5 WEIGHTS - EMPIRICAL   (shared diverging scale): the fit.
  6 GP SD       posterior sd of the GP with the same Hellinger kernel: the uncertainty map.

Plus one 3D `plot_surface` view of the kernel over slice A, and a printed fit table.

Honesty machinery, all of it required because comparing methods off-support is meaningless:
  - grid cells whose nearest train run is farther than the train p95 nearest-neighbour
    Hellinger distance are greyed out and excluded from every number;
  - empirical bins with fewer than MIN_BIN_N runs are hatched and excluded;
  - the quantitative table reports, next to each RMSE, the binning NOISE FLOOR
    (sqrt(mean of within-bin var / n)) -- the RMSE a perfect model would still show;
  - a second table block scores each method's OUT-OF-FOLD per-run predictions averaged
    into the same bins. Surface-vs-empirical mixes model error with slice-projection error
    (runs in a bin have arbitrary within-group composition; the slice fixes it at anchor
    ratios); the OOF-binned block does not, so the gap between the two blocks is the
    projection cost.

Targets: zmacro (e_zmacro_english, z-scored, LOWER = better) and logprob_humaneval_10shot
bpb. Train runs only; the 40-run holdout is never touched.

Writes report/figs3/f35_function_contours_zmacro.png, f36_function_contours_humaneval.png,
grug/function_contours.json and grug/function_contours.md, and registers both figures in
report/figs3/manifest3.json.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401  (registers the '3d' projection)
import numpy as np
import pandas as pd
from gp_inversion_study import load_all
from gp_surrogate import fit_gp, krr_predict, predict_gp
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRATCH = REPO_ROOT / "scratch" / "mixture_features"
GRUG = SCRATCH / "grug"
FIGS = SCRATCH / "report" / "figs3"
BUCKETS_TABLE = SCRATCH / "grug_histograms" / "buckets_table.parquet"

CODE_CLUSTERS = (1, 2, 6)  # validation batch 3 code_adjacent group (15 buckets, 24.4% of tokens)
SECOND_CLUSTERS = (5,)  # largest-spread non-code cluster; see module docstring
GAMMA_FACTOR, ALPHA = 0.25, 0.1  # frozen_model_hyperparams.json: 4_hellinger_kernel_k1000
RIDGE_ALPHAS = np.logspace(-3, 3, 25)  # grug_fit's ridge grid
N_GRID = 48  # model-surface resolution
BIN_GRID = 10  # empirical binning resolution
MIN_BIN_N = 8  # bins below this are hatched and excluded from every number
SUPPORT_PCT = 95  # off-support rule: NN Hellinger distance above the train p95
RANGE_FACTOR = 3.0  # each axis sweeps 0 .. 3x the group's anchor share
CORR_F0, CORR_F1 = 0.7987, 0.2013  # corrected phase token fractions (validation batch 2)
N_SPLITS, SEED = 5, 0  # campaign CV protocol (single repeat here; used only for OOF context)
HUMANEVAL = "logprob_humaneval_10shot"

CMAP = plt.get_cmap("RdYlGn_r")  # green = lower = better (campaign convention)
CMAP_RESID = plt.get_cmap("PuOr_r")
CMAP_SD = plt.get_cmap("viridis")
INK, MUTED, LINE = "#0b0b0b", "#52514e", "#d9d7d2"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.labelcolor": INK,
        "text.color": INK,
    }
)


# ---------------------------------------------------------------------------
# Mixture algebra (same construction as build_f17 / build_f18)
# ---------------------------------------------------------------------------


def pinned_weights(p_tok: np.ndarray, masks: list[np.ndarray], shares: list[np.ndarray]) -> np.ndarray:
    """(n_pts, 2, n_buckets) mixtures with each masked group's per-phase total pinned.

    Within every group and within the remainder, proportions stay at the token-proportional
    anchor ratios, so each phase sums to 1 exactly. ``shares[i]`` is (n_pts, 2).
    """
    n = shares[0].shape[0]
    rest = ~np.logical_or.reduce(masks)
    w = np.zeros((n, 2, len(p_tok)))
    total = np.zeros((n, 2))
    for mask, share in zip(masks, shares, strict=True):
        w[:, :, mask] = share[:, :, None] * (p_tok[mask] / p_tok[mask].sum())
        total = total + share
    w[:, :, rest] = np.clip(1.0 - total, 0.0, None)[:, :, None] * (p_tok[rest] / p_tok[rest].sum())
    return w


def mixture_sqrt_hists(w: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(n, 2, k) per-phase sqrt mixture histograms; ``v`` is the (k, n_buckets) basis."""
    return np.sqrt(np.clip(np.stack([w[:, p, :] @ v.T for p in range(2)], axis=1), 0.0, None))


def d2_to_train(sq: np.ndarray, sq_train: np.ndarray) -> np.ndarray:
    """(n, n_train) mean-over-phase squared Hellinger distance."""
    d = np.zeros((sq.shape[0], sq_train.shape[0]))
    for p in range(sq.shape[1]):
        d += np.clip(1.0 - sq[:, p, :] @ sq_train[:, p, :].T, 0.0, None)
    return d / sq.shape[1]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def flat_weights(w: np.ndarray) -> np.ndarray:
    """(n, 2 * n_buckets) raw per-phase bucket proportions: the RegMix-style feature vector."""
    return w.reshape(len(w), -1)


def weights_ridge_predict(w_train: np.ndarray, y: np.ndarray, w_star: np.ndarray) -> np.ndarray:
    model = RidgeCV(alphas=RIDGE_ALPHAS).fit(flat_weights(w_train), y)
    return model.predict(flat_weights(w_star))


def oof_predictions(d2: np.ndarray, w: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Out-of-fold predictions for both methods on identical folds (context, and the projection check)."""
    n = len(y)
    out = {"kernel": np.full(n, np.nan), "weights_ridge": np.full(n, np.nan)}
    for tr, te in KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(np.arange(n)):
        d_tr = d2[np.ix_(tr, tr)]
        gamma = GAMMA_FACTOR / float(np.median(d_tr[~np.eye(len(tr), dtype=bool)]))
        out["kernel"][te] = krr_predict(d_tr, d2[np.ix_(te, tr)], y[tr], gamma, ALPHA)
        out["weights_ridge"][te] = weights_ridge_predict(w[tr], y[tr], w[te])
    return out


# ---------------------------------------------------------------------------
# Slices
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Slice:
    """One interpretable 2D cut through the simplex."""

    key: str
    title: str
    xlabel: str
    ylabel: str
    xmax: float
    ymax: float
    anchor: tuple[float, float]
    run_x: np.ndarray
    run_y: np.ndarray
    masks: list[np.ndarray]
    per_phase: bool  # True: axes are the two PHASE shares of one group; False: two groups' overall shares

    def shares(self, xs: np.ndarray, ys: np.ndarray) -> list[np.ndarray]:
        if self.per_phase:
            return [np.stack([xs, ys], axis=1)]
        return [np.repeat(xs[:, None], 2, axis=1), np.repeat(ys[:, None], 2, axis=1)]

    def weights(self, p_tok: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return pinned_weights(p_tok, self.masks, self.shares(xs, ys))


def build_slices(p_tok: np.ndarray, cluster_id: np.ndarray, w: np.ndarray) -> list[Slice]:
    code = np.isin(cluster_id, CODE_CLUSTERS)
    second = np.isin(cluster_id, SECOND_CLUSTERS)
    p_code, p_second = float(p_tok[code].sum()), float(p_tok[second].sum())
    overall_code = CORR_F0 * w[:, 0, code].sum(axis=1) + CORR_F1 * w[:, 1, code].sum(axis=1)
    overall_second = CORR_F0 * w[:, 0, second].sum(axis=1) + CORR_F1 * w[:, 1, second].sum(axis=1)
    return [
        Slice(
            key="A_code_phase",
            title="SLICE A  code_adjacent group: phase-0 share vs phase-1 share",
            xlabel="code_adjacent phase-0 share (first 80% of tokens)",
            ylabel="code_adjacent phase-1 share (last 20% of tokens)",
            xmax=RANGE_FACTOR * p_code,
            ymax=RANGE_FACTOR * p_code,
            anchor=(p_code, p_code),
            run_x=w[:, 0, code].sum(axis=1),
            run_y=w[:, 1, code].sum(axis=1),
            masks=[code],
            per_phase=True,
        ),
        Slice(
            key="B_code_vs_c05",
            title="SLICE B  code_adjacent share vs c05 (largest-spread web cluster) share",
            xlabel="code_adjacent overall share (both phases pinned equal)",
            ylabel="c05 group overall share (both phases pinned equal)",
            xmax=RANGE_FACTOR * p_code,
            ymax=RANGE_FACTOR * p_second,
            anchor=(p_code, p_second),
            run_x=overall_code,
            run_y=overall_second,
            masks=[code, second],
            per_phase=False,
        ),
    ]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def bin_empirical(sl: Slice, y: np.ndarray) -> dict:
    """Bin the real runs by their own coordinates: per-bin mean outcome, count, and noise."""
    ex = np.linspace(0.0, sl.xmax, BIN_GRID + 1)
    ey = np.linspace(0.0, sl.ymax, BIN_GRID + 1)
    ix = np.clip(np.digitize(sl.run_x, ex) - 1, -1, BIN_GRID - 1)
    iy = np.clip(np.digitize(sl.run_y, ey) - 1, -1, BIN_GRID - 1)
    inside = (sl.run_x <= sl.xmax) & (sl.run_y <= sl.ymax)
    mean = np.full((BIN_GRID, BIN_GRID), np.nan)
    var = np.full((BIN_GRID, BIN_GRID), np.nan)
    count = np.zeros((BIN_GRID, BIN_GRID), dtype=int)
    for j in range(BIN_GRID):
        for i in range(BIN_GRID):
            sel = inside & (ix == i) & (iy == j)
            count[j, i] = int(sel.sum())
            if count[j, i] > 0:
                mean[j, i] = float(y[sel].mean())
            if count[j, i] > 1:
                var[j, i] = float(y[sel].var(ddof=1))
    return {
        "edges_x": ex,
        "edges_y": ey,
        "centers_x": 0.5 * (ex[:-1] + ex[1:]),
        "centers_y": 0.5 * (ey[:-1] + ey[1:]),
        "mean": mean,
        "var": var,
        "count": count,
        "bin_of_run": (ix, iy, inside),
        "n_in_window": int(inside.sum()),
    }


def surfaces(sl: Slice, data: dict, y: np.ndarray, coords: tuple[np.ndarray, np.ndarray]) -> dict:
    """Predict every method plus the support map at the given (xx, yy) coordinate mesh."""
    xx, yy = coords
    w_grid = sl.weights(data["p_tok"], xx.ravel(), yy.ravel())
    feasible = np.ones_like(xx, dtype=bool) if sl.per_phase else (xx + yy <= 1.0)
    ok = feasible.ravel()  # cells where the two pinned shares can coexist at all
    assert np.allclose(w_grid[ok].sum(axis=2), 1.0, atol=1e-12), "grid mixtures do not sum to 1 per phase"
    d2_star = d2_to_train(mixture_sqrt_hists(w_grid, data["v"]), data["sq_train"])
    kernel = krr_predict(data["d2"], d2_star, y, data["gamma"], ALPHA).reshape(xx.shape)
    ridge = weights_ridge_predict(data["w"], y, w_grid).reshape(xx.shape)
    gp = data["gp"]
    _, sd = predict_gp(gp, d2_star, include_noise=False)
    nn = np.sqrt(d2_star.min(axis=1)).reshape(xx.shape)
    mask = feasible & (nn <= data["support_radius"])
    return {
        "kernel": np.where(feasible, kernel, np.nan),
        "weights_ridge": np.where(feasible, ridge, np.nan),
        "gp_sd": np.where(feasible, sd.reshape(xx.shape), np.nan),
        "nn_dist": np.where(feasible, nn, np.nan),
        "in_support": mask,
        "feasible": feasible,
        "gp_hyper": {k: float(gp[k]) for k in ("sigma_f2", "sigma_n2", "gamma", "ridge_equivalent_alpha")},
    }


def bin_oof(sl: Slice, emp: dict, oof: np.ndarray) -> np.ndarray:
    """Per-bin mean of out-of-fold per-run predictions: the same projection as the empirical panel."""
    ix, iy, inside = emp["bin_of_run"]
    out = np.full((BIN_GRID, BIN_GRID), np.nan)
    for j in range(BIN_GRID):
        for i in range(BIN_GRID):
            sel = inside & (ix == i) & (iy == j)
            if sel.sum() > 0:
                out[j, i] = float(oof[sel].mean())
    return out


def score(pred: np.ndarray, emp: dict, usable: np.ndarray) -> dict:
    """Pearson / Spearman / RMSE of a prediction field against the binned empirical means."""
    a, b = pred[usable], emp["mean"][usable]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    return {
        "n_bins": int(ok.sum()),
        "pearson": float(pearsonr(a, b).statistic),
        "spearman": float(spearmanr(a, b).statistic),
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "bias": float(np.mean(a - b)),
    }


def evaluate(sl: Slice, data: dict, y: np.ndarray, oof: dict[str, np.ndarray]) -> dict:
    """Everything one slice needs: fine model surfaces, binned empirical, bin-level scores."""
    ax_x = np.linspace(0.0, sl.xmax, N_GRID)
    ax_y = np.linspace(0.0, sl.ymax, N_GRID)
    fine = surfaces(sl, data, y, np.meshgrid(ax_x, ax_y))
    emp = bin_empirical(sl, y)
    bins = surfaces(sl, data, y, np.meshgrid(emp["centers_x"], emp["centers_y"]))

    usable = (emp["count"] >= MIN_BIN_N) & bins["in_support"]
    counted = emp["count"][usable]
    noise = float(np.sqrt(np.nanmean(emp["var"][usable] / np.maximum(counted, 1))))
    ai = int(np.clip(np.digitize([sl.anchor[0]], emp["edges_x"])[0] - 1, 0, BIN_GRID - 1))
    aj = int(np.clip(np.digitize([sl.anchor[1]], emp["edges_y"])[0] - 1, 0, BIN_GRID - 1))
    scores = {
        "surface_vs_empirical": {m: score(bins[m], emp, usable) for m in ("kernel", "weights_ridge")},
        "oof_binned_vs_empirical": {m: score(bin_oof(sl, emp, v), emp, usable) for m, v in oof.items()},
        "kernel_vs_weights_ridge_surface": {
            "pearson": float(pearsonr(bins["kernel"][usable], bins["weights_ridge"][usable]).statistic),
            "spearman": float(spearmanr(bins["kernel"][usable], bins["weights_ridge"][usable]).statistic),
            "rms_difference": float(np.sqrt(np.mean((bins["kernel"][usable] - bins["weights_ridge"][usable]) ** 2))),
        },
        "binning_noise_floor_rmse": noise,
        "n_usable_bins": int(usable.sum()),
        "n_runs_in_usable_bins": int(counted.sum()),
        "n_runs_in_window": emp["n_in_window"],
        "frac_grid_off_support": float(1.0 - fine["in_support"][fine["feasible"]].mean()),
        "frac_grid_infeasible": float(1.0 - fine["feasible"].mean()),
        "empirical_range": [float(np.nanmin(emp["mean"][usable])), float(np.nanmax(emp["mean"][usable]))],
        "surface_range_in_support": {
            m: [float(np.nanmin(fine[m][fine["in_support"]])), float(np.nanmax(fine[m][fine["in_support"]]))]
            for m in ("kernel", "weights_ridge")
        },
        "anchor_bin": {
            "empirical_mean": float(emp["mean"][aj, ai]),
            "n_runs": int(emp["count"][aj, ai]),
            "kernel": float(bins["kernel"][aj, ai]),
            "weights_ridge": float(bins["weights_ridge"][aj, ai]),
        },
    }
    return {
        "slice": sl,
        "ax_x": ax_x,
        "ax_y": ax_y,
        "fine": fine,
        "emp": emp,
        "bins": bins,
        "usable": usable,
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _frame(ax, ev: dict, title: str, labels: bool) -> None:
    sl = ev["slice"]
    ax.set_xlim(0, sl.xmax)
    ax.set_ylim(0, sl.ymax)
    ax.set_title(title, fontsize=8.8, pad=5)
    if labels:
        ax.set_xlabel(sl.xlabel, fontsize=6.8)
        ax.set_ylabel(sl.ylabel, fontsize=6.8)
    ax.tick_params(labelsize=6.5, colors=MUTED)
    for s in ax.spines.values():
        s.set_color(LINE)


def _hatch_cells(ax, edges_x: np.ndarray, edges_y: np.ndarray, flags: np.ndarray, hatch: str) -> None:
    """Exact per-cell hatching (contourf would smooth the region across cell boundaries)."""
    for j, i in zip(*np.nonzero(flags), strict=True):
        ax.add_patch(
            plt.Rectangle(
                (edges_x[i], edges_y[j]),
                edges_x[i + 1] - edges_x[i],
                edges_y[j + 1] - edges_y[j],
                facecolor="none",
                edgecolor="#3a3a3a",
                hatch=hatch,
                linewidth=0.0,
                zorder=3.5,
            )
        )


def _mark_off_support(ax, ev: dict) -> None:
    """Grey wash + boundary contour over cells no run is close to (and infeasible cells)."""
    fine = ev["fine"]
    step_x = ev["ax_x"][1] - ev["ax_x"][0]
    step_y = ev["ax_y"][1] - ev["ax_y"][0]
    edges_x = np.append(ev["ax_x"] - step_x / 2, ev["ax_x"][-1] + step_x / 2)
    edges_y = np.append(ev["ax_y"] - step_y / 2, ev["ax_y"][-1] + step_y / 2)
    off = np.where(~fine["in_support"], 1.0, np.nan)
    ax.pcolormesh(
        edges_x, edges_y, off, cmap=plt.get_cmap("Greys"), vmin=0, vmax=1.6, alpha=0.75, zorder=3, rasterized=True
    )
    _hatch_cells(ax, edges_x, edges_y, ~fine["in_support"], "////")
    if (~fine["in_support"]).any() and fine["in_support"].any():
        ax.contour(
            ev["ax_x"],
            ev["ax_y"],
            fine["in_support"].astype(float),
            levels=[0.5],
            colors=INK,
            linewidths=0.8,
            linestyles=":",
            zorder=5,
        )


def _overlay_runs(ax, ev: dict) -> None:
    sl = ev["slice"]
    ax.scatter(sl.run_x, sl.run_y, s=3, c="#00000066", linewidths=0, zorder=4)
    ax.scatter(
        [sl.anchor[0]], [sl.anchor[1]], marker="X", s=70, facecolors="white", edgecolors=INK, linewidths=1.2, zorder=6
    )


def draw_empirical(ax, ev: dict, norm, labels: bool) -> None:
    emp = ev["emp"]
    shown = np.where(emp["count"] > 0, emp["mean"], np.nan)
    ax.pcolormesh(emp["edges_x"], emp["edges_y"], shown, cmap=CMAP, norm=norm, zorder=2, rasterized=True)
    _hatch_cells(ax, emp["edges_x"], emp["edges_y"], (emp["count"] > 0) & ~ev["usable"], "xxx")
    for j, cy in enumerate(emp["centers_y"]):
        for i, cx in enumerate(emp["centers_x"]):
            if emp["count"][j, i] > 0:
                ax.text(
                    cx, cy, str(emp["count"][j, i]), ha="center", va="center", fontsize=5.2, color="#111111", zorder=5
                )
    _overlay_runs(ax, ev)
    _frame(
        ax, ev, f"1. EMPIRICAL: {emp['n_in_window']} real runs binned\ncell = mean outcome, number = run count", labels
    )


def draw_surface(ax, ev: dict, key: str, norm, title: str, labels: bool) -> None:
    ax.pcolormesh(
        ev["ax_x"], ev["ax_y"], ev["fine"][key], cmap=CMAP, norm=norm, shading="nearest", zorder=2, rasterized=True
    )
    ax.contour(
        ev["ax_x"],
        ev["ax_y"],
        np.ma.masked_invalid(ev["fine"][key]),
        levels=9,
        colors=INK,
        linewidths=0.35,
        alpha=0.45,
        zorder=2.5,
    )
    _mark_off_support(ax, ev)
    _overlay_runs(ax, ev)
    _frame(ax, ev, title, labels)


def draw_residual(ax, ev: dict, key: str, norm, title: str, labels: bool) -> None:
    emp = ev["emp"]
    resid = np.where(ev["usable"], ev["bins"][key] - emp["mean"], np.nan)
    ax.pcolormesh(emp["edges_x"], emp["edges_y"], resid, cmap=CMAP_RESID, norm=norm, zorder=2, rasterized=True)
    _overlay_runs(ax, ev)
    _frame(ax, ev, title, labels)


def draw_sd(ax, ev: dict, norm, labels: bool) -> None:
    ax.pcolormesh(
        ev["ax_x"],
        ev["ax_y"],
        ev["fine"]["gp_sd"],
        cmap=CMAP_SD,
        norm=norm,
        shading="nearest",
        zorder=2,
        rasterized=True,
    )
    ax.contour(
        ev["ax_x"],
        ev["ax_y"],
        ev["fine"]["nn_dist"],
        levels=[ev["support_radius"]],
        colors="white",
        linewidths=0.9,
        linestyles=":",
        zorder=5,
    )
    _overlay_runs(ax, ev)
    _frame(ax, ev, "6. GP posterior sd (uncertainty map)\ndotted = off-support boundary", labels)


def draw_3d(ax, ev: dict, norm, target_label: str) -> None:
    xx, yy = np.meshgrid(ev["ax_x"], ev["ax_y"])
    z = np.where(ev["fine"]["in_support"], ev["fine"]["kernel"], np.nan)
    ax.plot_surface(xx, yy, z, cmap=CMAP, norm=norm, linewidth=0, antialiased=True, rstride=1, cstride=1, shade=False)
    ax.set_xlabel("phase-0 share", fontsize=6.5, labelpad=-4)
    ax.set_ylabel("phase-1 share", fontsize=6.5, labelpad=-4)
    ax.set_zlabel(target_label, fontsize=6.5, labelpad=-6)
    ax.tick_params(labelsize=5.5, colors=MUTED, pad=-2)
    ax.view_init(elev=26, azim=-128)
    ax.set_title("3D: kernel surface over slice A\n(in-support region only)", fontsize=9)


def fit_table_text(evals: list[dict], oof_global: dict, target_label: str) -> str:
    lines = [f"FIT TABLE -- target: {target_label}", ""]
    lines.append(f"{'slice':<8}{'method':<19}{'block':<19}{'bins':>5}{'pearson':>9}{'spearman':>10}{'rmse':>9}")
    lines.append("-" * 79)
    for ev in evals:
        sc = ev["scores"]
        key = ev["slice"].key.split("_")[0]
        for block in ("surface_vs_empirical", "oof_binned_vs_empirical"):
            for method, s in sc[block].items():
                lines.append(
                    f"{key:<8}{method:<19}{block.split('_vs_')[0]:<19}{s['n_bins']:>5}"
                    f"{s['pearson']:>+9.3f}{s['spearman']:>+10.3f}{s['rmse']:>9.4f}"
                )
        lines.append(
            f"{key:<8}{'(binning noise)':<19}{'floor':<19}{sc['n_usable_bins']:>5}{'':>9}{'':>10}"
            f"{sc['binning_noise_floor_rmse']:>9.4f}"
        )
        kw = sc["kernel_vs_weights_ridge_surface"]
        lines.append(
            f"{key:<8}{'kernel vs wridge':<19}{'surface agreement':<19}{'':>5}{kw['pearson']:>+9.3f}"
            f"{kw['spearman']:>+10.3f}{kw['rms_difference']:>9.4f}"
        )
        lines.append(
            f"{key:<8}off-support {sc['frac_grid_off_support']:.0%} of feasible grid; "
            f"{sc['n_usable_bins']} usable bins hold {sc['n_runs_in_usable_bins']} runs"
        )
        lines.append("")
    lines.append(
        "global out-of-fold Spearman over all 800 runs (slice-independent):  "
        + "   ".join(f"{m} {v:+.3f}" for m, v in oof_global.items())
    )
    lines.append("LightGBM-on-weights (the literal RegMix model) NOT run: no libgomp on this host.")
    return "\n".join(lines)


def row_norms(ev: dict) -> tuple[Normalize, TwoSlopeNorm]:
    """One value scale shared by panels 1-3 of a row, one diverging scale shared by panels 4-5."""
    pool = np.concatenate(
        [ev["emp"]["mean"][ev["usable"]]]
        + [ev["fine"][m][ev["fine"]["in_support"]] for m in ("kernel", "weights_ridge")]
    )
    norm = Normalize(vmin=float(np.nanpercentile(pool, 1)), vmax=float(np.nanpercentile(pool, 99)))
    resid = np.concatenate(
        [np.abs(ev["bins"][m][ev["usable"]] - ev["emp"]["mean"][ev["usable"]]) for m in ("kernel", "weights_ridge")]
    )
    rmax = max(float(np.nanpercentile(resid, 98)), 1e-6)
    return norm, TwoSlopeNorm(vcenter=0.0, vmin=-rmax, vmax=rmax)


def build_figure(evals: list[dict], oof_global: dict, target_label: str, better: str, out: Path, caption: str) -> None:
    sd_pool = np.concatenate([ev["fine"]["gp_sd"][ev["fine"]["feasible"]] for ev in evals])
    sd_norm = Normalize(vmin=float(np.nanmin(sd_pool)), vmax=float(np.nanpercentile(sd_pool, 99)))

    fig = plt.figure(figsize=(22.5, 13.8))
    gs = fig.add_gridspec(
        3, 6, height_ratios=[1.0, 1.0, 0.78], left=0.045, right=0.955, top=0.875, bottom=0.155, hspace=0.62, wspace=0.42
    )
    norms = []

    for row, ev in enumerate(evals):
        norm, rnorm = row_norms(ev)
        norms.append(norm)
        axes = [fig.add_subplot(gs[row, c]) for c in range(6)]
        draw_empirical(axes[0], ev, norm, labels=True)
        draw_surface(
            axes[1], ev, "kernel", norm, "2. Hellinger KERNEL ridge\nfrozen: gamma_f 0.25, alpha 0.1", labels=False
        )
        draw_surface(
            axes[2], ev, "weights_ridge", norm, "3. WEIGHTS ridge on raw proportions\nRegMix-style linear", labels=False
        )
        draw_residual(axes[3], ev, "kernel", rnorm, "4. kernel MINUS empirical\nusable bins only", labels=False)
        draw_residual(
            axes[4], ev, "weights_ridge", rnorm, "5. weights ridge MINUS empirical\nusable bins only", labels=False
        )
        draw_sd(axes[5], ev, sd_norm, labels=False)
        axes[0].text(0.0, 1.34, ev["slice"].title, transform=axes[0].transAxes, fontsize=12, fontweight="semibold")
        axes[0].text(
            0.0,
            1.24,
            "all six panels share these axes; panels 1-3 share one color scale, 4-5 another",
            transform=axes[0].transAxes,
            fontsize=7.5,
            color=MUTED,
        )
        for ax, mappable, label in (
            (axes[2], plt.cm.ScalarMappable(norm=norm, cmap=CMAP), f"{target_label}, {better}"),
            (axes[4], plt.cm.ScalarMappable(norm=rnorm, cmap=CMAP_RESID), "prediction - empirical"),
            (axes[5], plt.cm.ScalarMappable(norm=sd_norm, cmap=CMAP_SD), "GP posterior sd"),
        ):
            cb = fig.colorbar(mappable, ax=ax, fraction=0.05, pad=0.03, aspect=22)
            cb.set_label(label, fontsize=6.5)
            cb.ax.tick_params(labelsize=6)
            cb.outline.set_edgecolor(LINE)

    ax3d = fig.add_subplot(gs[2, 0:2], projection="3d")
    draw_3d(ax3d, evals[0], norms[0], target_label)
    ax_txt = fig.add_subplot(gs[2, 2:6])
    ax_txt.axis("off")
    ax_txt.text(
        0.0,
        1.0,
        fit_table_text(evals, oof_global, target_label),
        transform=ax_txt.transAxes,
        fontsize=7.2,
        family="DejaVu Sans Mono",
        va="top",
        ha="left",
        color=INK,
    )

    legend = [
        Line2D([], [], marker="o", ls="", mfc="#000000", mec="none", ms=4, label="train runs (own coordinates)"),
        Line2D([], [], marker="X", ls="", mfc="white", mec=INK, ms=8, label="token-proportional anchor"),
        Line2D([], [], color=INK, lw=0.9, ls=":", label=f"off-support boundary (NN Hellinger > train p{SUPPORT_PCT})"),
        Line2D([], [], marker="s", ls="", mfc="#bdbdbd", mec=INK, ms=8, label="grey+hatch = off-support / < 8 runs"),
    ]
    fig.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.045, 0.045),
        ncol=4,
        fontsize=8,
        frameon=False,
        columnspacing=2.2,
        handletextpad=0.6,
    )
    fig.suptitle(
        f"What the mixture surrogate actually looks like -- {target_label}", x=0.045, y=0.972, ha="left", fontsize=15
    )
    fig.text(0.045, 0.008, caption, fontsize=7.2, color=MUTED, ha="left", va="bottom", wrap=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_data() -> dict:
    d2, y_zmacro, v, w, h_train = load_all()
    runs = pd.read_parquet(GRUG / "train_runs.parquet")
    assert len(runs) == len(y_zmacro), "load_all dropped runs; the parquet order would no longer align"
    y_humaneval = np.array([json.loads(e)[HUMANEVAL]["bpb"] for e in runs["evals"]], dtype=np.float64)
    bt = pd.read_parquet(BUCKETS_TABLE)
    buckets = sorted(bt["bucket"].tolist())
    assert len(buckets) == w.shape[2], "bucket count mismatch between buckets_table and the weight matrix"
    bt = bt.set_index("bucket").loc[buckets]
    p_tok = bt["total_tokens"].to_numpy(dtype=np.float64)
    p_tok /= p_tok.sum()
    n = len(y_zmacro)
    gamma = GAMMA_FACTOR / float(np.median(d2[~np.eye(n, dtype=bool)]))
    support_radius = float(np.percentile(np.sqrt((d2 + np.eye(n) * 1e9).min(axis=1)), SUPPORT_PCT))
    return {
        "d2": d2,
        "v": v,
        "w": w,
        "sq_train": np.sqrt(np.clip(h_train, 0.0, None)),
        "p_tok": p_tok,
        "cluster_id": bt["cluster_id"].to_numpy(),
        "gamma": gamma,
        "support_radius": support_radius,
        "targets": {"zmacro": y_zmacro, "humaneval": y_humaneval},
    }


def run_target(
    data: dict,
    name: str,
    y: np.ndarray,
    slices: list[Slice],
    out_png: Path,
    target_label: str,
    better: str,
    caption: str,
) -> dict:
    data = {**data, "gp": fit_gp(data["d2"], y)}  # posterior sd only; hyperparameters are target-specific
    oof = oof_predictions(data["d2"], data["w"], y)
    oof_global = {m: float(spearmanr(v, y).statistic) for m, v in oof.items()}
    evals = [evaluate(sl, data, y, oof) for sl in slices]
    for ev in evals:
        ev["support_radius"] = data["support_radius"]
    build_figure(evals, oof_global, target_label, better, out_png, caption)
    return {
        "target": name,
        "target_label": target_label,
        "global_oof_spearman": oof_global,
        "gp_hyperparameters": evals[0]["fine"]["gp_hyper"],
        "slices": {ev["slice"].key: ev["scores"] for ev in evals},
    }


CAPTION = (
    "Within each row, panels 1-3 share ONE color scale so the measured and fitted surfaces are directly comparable, "
    "and panels 4-5 share a second (diverging) scale; the two rows are scaled independently because their value "
    "ranges differ (read each row's colorbar). Panel 1 is MEASURED (real runs binned by their own coordinates, cell "
    "number = run count); panels 2-3 and 6 are MODEL OUTPUT over a constructed grid. Grid cells whose nearest real "
    "run is farther than the train p95 nearest-neighbour Hellinger distance are greyed and hatched and are excluded "
    "from every number in the table; empirical bins with fewer than 8 runs are hatched and likewise excluded. RMSE "
    "must be read against the binning noise floor printed beneath it -- the RMSE a perfect model would still show, "
    "given how few runs land in a bin. TWO CAVEATS THAT DECIDE HOW TO READ PANELS 4-5. (i) Bins holding 8-15 runs "
    "are noisy. (ii) More important: the empirical panel is a MARGINAL view (a bin averages runs whose other 150+ "
    "coordinates vary and, in a Dirichlet swarm, covary with the axis), while the model surfaces are a CONDITIONAL "
    "ceteris-paribus cut with every other bucket held at anchor ratios. They answer different questions, so "
    "surface-vs-empirical disagreement is not by itself model error. The out-of-fold block in the table -- each "
    "method's held-out per-run predictions averaged into the SAME bins -- removes that projection and is the fair "
    "model score."
)


def main() -> None:
    data = load_data()
    slices = build_slices(data["p_tok"], data["cluster_id"], data["w"])
    results = [
        run_target(
            data,
            "zmacro",
            data["targets"]["zmacro"],
            slices,
            FIGS / "f35_function_contours_zmacro.png",
            "zmacro (z-scored bpb)",
            "lower = better",
            CAPTION,
        ),
        run_target(
            data,
            "humaneval",
            data["targets"]["humaneval"],
            slices,
            FIGS / "f36_function_contours_humaneval.png",
            "humaneval 10-shot bpb",
            "lower = better",
            CAPTION,
        ),
    ]
    payload = {
        "protocol": {
            "n_runs": len(data["targets"]["zmacro"]),
            "grid": N_GRID,
            "bin_grid": BIN_GRID,
            "min_bin_n": MIN_BIN_N,
            "support_percentile": SUPPORT_PCT,
            "support_radius_hellinger": data["support_radius"],
            "kernel": {"gamma_factor": GAMMA_FACTOR, "alpha": ALPHA, "gamma": data["gamma"]},
            "weights_ridge": "RidgeCV(logspace(-3,3,25)) on the raw 2x168 per-phase bucket proportions",
            "lightgbm_baseline": "NOT run: this host lacks libgomp, so lightgbm cannot load",
            "code_clusters": list(CODE_CLUSTERS),
            "second_group_clusters": list(SECOND_CLUSTERS),
        },
        "results": results,
    }
    (GRUG / "function_contours.json").write_text(json.dumps(payload, indent=1) + "\n")
    (GRUG / "function_contours.md").write_text(render_markdown(payload))
    print("wrote", GRUG / "function_contours.json", "and", GRUG / "function_contours.md")
    register_manifest()


def render_markdown(payload: dict) -> str:
    p = payload["protocol"]
    out = [
        "# Function contours: what the mixture surrogate looks like on interpretable slices",
        "",
        f"- {p['n_runs']} train runs; model grid {p['grid']}x{p['grid']}, empirical bins {p['bin_grid']}x"
        f"{p['bin_grid']}, min {p['min_bin_n']} runs/bin.",
        f"- off-support rule: nearest-neighbour Hellinger distance to any train run > train p{p['support_percentile']}"
        f" = {p['support_radius_hellinger']:.3f}; off-support cells are excluded from every number below.",
        f"- kernel = frozen Hellinger kernel ridge (gamma_factor {p['kernel']['gamma_factor']}, alpha "
        f"{p['kernel']['alpha']}); weights ridge = {p['weights_ridge']}.",
        f"- LightGBM-on-weights (the literal RegMix model): {p['lightgbm_baseline']}.",
        "- slice A axes are the code_adjacent group's two PHASE shares (f18 axes); slice B axes are the "
        "code_adjacent and c05 overall shares with both phases pinned equal.",
        "",
        "Two blocks per slice, and they answer different questions. `surface_vs_empirical` scores the CONDITIONAL "
        "slice (every other bucket pinned at anchor ratios) against a MARGINAL measurement (a bin averages runs "
        "whose other coordinates vary freely and covary with the axis) -- a mismatch there is as likely to be "
        "confounding in the swarm design as model error. `oof_binned_vs_empirical` scores each method's held-out "
        "per-run predictions averaged into the SAME bins; that removes the projection and is the fair model score.",
        "",
    ]
    for res in payload["results"]:
        out.append(f"## {res['target_label']}")
        out.append("")
        out.append(
            "- global out-of-fold Spearman over all 800 runs: "
            + ", ".join(f"{m} {v:+.3f}" for m, v in res["global_oof_spearman"].items())
        )
        out.append("")
        out.append("| slice | method | block | bins | Pearson | Spearman | RMSE | bias |")
        out.append("|---|---|---|---|---|---|---|---|")
        for key, sc in res["slices"].items():
            for block in ("surface_vs_empirical", "oof_binned_vs_empirical"):
                for method, s in sc[block].items():
                    out.append(
                        f"| {key} | {method} | {block} | {s['n_bins']} | {s['pearson']:+.3f} | "
                        f"{s['spearman']:+.3f} | {s['rmse']:.4f} | {s['bias']:+.4f} |"
                    )
            out.append(
                f"| {key} | (binning noise floor) | - | {sc['n_usable_bins']} | | | "
                f"{sc['binning_noise_floor_rmse']:.4f} | |"
            )
            kw = sc["kernel_vs_weights_ridge_surface"]
            out.append(
                f"| {key} | kernel vs weights ridge | surface agreement | | {kw['pearson']:+.3f} | "
                f"{kw['spearman']:+.3f} | {kw['rms_difference']:.4f} | |"
            )
        out.append("")
        for key, sc in res["slices"].items():
            ab = sc["anchor_bin"]
            sr = sc["surface_range_in_support"]
            out.append(
                f"- {key}: {sc['frac_grid_off_support']:.1%} of the feasible grid is off-support "
                f"({sc['frac_grid_infeasible']:.1%} of the raw grid is infeasible); {sc['n_usable_bins']} usable bins "
                f"hold {sc['n_runs_in_usable_bins']} of {sc['n_runs_in_window']} in-window runs; empirical range over "
                f"usable bins {sc['empirical_range'][0]:.4f} .. {sc['empirical_range'][1]:.4f}."
            )
            out.append(
                f"  - in-support surface range: kernel {sr['kernel'][0]:.4f} .. {sr['kernel'][1]:.4f}, "
                f"weights ridge {sr['weights_ridge'][0]:.4f} .. {sr['weights_ridge'][1]:.4f}."
            )
            out.append(
                f"  - anchor bin ({ab['n_runs']} runs): empirical {ab['empirical_mean']:+.4f}, kernel "
                f"{ab['kernel']:+.4f}, weights ridge {ab['weights_ridge']:+.4f}."
            )
        out.append("")
        out.extend(findings(res))
        out.append("")
    return "\n".join(out) + "\n"


def findings(res: dict) -> list[str]:
    """Read the table back out as prose, with every number pulled from the scores."""
    keys = list(res["slices"])
    lines = ["### what the numbers say", ""]
    fair = {
        k: {m: res["slices"][k]["oof_binned_vs_empirical"][m]["spearman"] for m in ("kernel", "weights_ridge")}
        for k in keys
    }
    winner = "kernel" if all(v["kernel"] > v["weights_ridge"] for v in fair.values()) else "mixed"
    lines.append(
        f"- fair (out-of-fold, projection removed) ranking: **{winner}**; per slice, kernel vs weights ridge "
        + "; ".join(f"{k} {v['kernel']:+.3f} / {v['weights_ridge']:+.3f}" for k, v in fair.items())
        + f" -- consistent with the global out-of-fold Spearman "
        f"({res['global_oof_spearman']['kernel']:+.3f} vs {res['global_oof_spearman']['weights_ridge']:+.3f})."
    )
    for k in keys:
        sc = res["slices"][k]
        surf = sc["surface_vs_empirical"]
        kw = sc["kernel_vs_weights_ridge_surface"]
        lines.append(
            f"- {k}: the SURFACES track the binned empirical with Spearman kernel {surf['kernel']['spearman']:+.3f} / "
            f"weights ridge {surf['weights_ridge']['spearman']:+.3f} over {sc['n_usable_bins']} usable bins. The two "
            f"surfaces order the slice very similarly (Pearson {kw['pearson']:+.3f}, Spearman "
            f"{kw['spearman']:+.3f}) but differ in LEVEL by RMS {kw['rms_difference']:.4f}, "
            f"{kw['rms_difference'] / sc['binning_noise_floor_rmse']:.1f}x the binning noise floor "
            f"({sc['binning_noise_floor_rmse']:.4f})."
        )
        lines.append(
            f"  - level offsets vs the bins: kernel {surf['kernel']['bias']:+.4f} "
            f"({abs(surf['kernel']['bias']) / sc['binning_noise_floor_rmse']:.1f}x the noise floor), weights ridge "
            f"{surf['weights_ridge']['bias']:+.4f} "
            f"({abs(surf['weights_ridge']['bias']) / sc['binning_noise_floor_rmse']:.1f}x). A negative offset means "
            "the method scores anchor-ratio grid mixtures better than the average real run at the same group "
            "coordinate -- a nonlinear 'spread the remaining mass evenly' effect the linear model cannot express. "
            "Every grid cell here is inside the train p95 NN Hellinger radius, so this is interpolation in the "
            "kernel's own metric, but the binned empirical cannot adjudicate it: the Dirichlet swarm holds no run "
            "at anchor within-group ratios."
        )
    return lines


def register_manifest() -> None:
    path = FIGS / "manifest3.json"
    manifest = json.loads(path.read_text())
    source = (
        "train_runs.parquet (800) + buckets_table.parquet + the frozen K=1000 content basis; builder "
        "experiments/datakit/mixture_features/function_contours.py"
    )
    for fname, target in (
        ("f35_function_contours_zmacro.png", "zmacro"),
        ("f36_function_contours_humaneval.png", "humaneval 10-shot bpb"),
    ):
        manifest[fname] = {
            "message": (
                f"What the fitted mixture surrogate LOOKS like against {target}, on two interpretable 2D slices "
                "(slice A: the code_adjacent group's phase-0 vs phase-1 share, the f18 axes; slice B: code_adjacent "
                "vs c05, the largest-spread non-code cluster). Per slice: the binned EMPIRICAL surface from the 800 "
                "real runs (with per-bin counts), the Hellinger kernel-ridge surface, the RegMix-style weights-ridge "
                "surface -- all three on one shared color scale -- plus both methods' residuals against the "
                "empirical bins and the GP posterior-sd map, with off-support cells greyed out and excluded. Bottom "
                "row: a 3D view of the kernel surface over slice A and the quantitative fit table (Pearson/Spearman/"
                "RMSE against the binned empirical, next to the binning noise floor)."
            ),
            "data_source": source,
        }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("updated", path)


if __name__ == "__main__":
    main()
