# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "joblib", "matplotlib"]
# ///
"""Validation batch 3 for the grug surrogate (codex terminal-state items, train-only).

Three analyses on the 800 TRAINING runs (QUARANTINE_test_labels.parquet stays closed).
Machinery is reused verbatim from grug_validation_batch2 (loaders, target, folds, kernel
paths, corrected phase fractions 0.7987/0.2013 via apply_corrected_phase_constants).

  1  HETEROSKEDASTICITY (diagnostic, not headline): kernel OOF residuals on
     zmacro_english_20; Brown-Forsythe (Levene, center=median) across (a) the 17
     max-dose-cluster groups from batch2's LODO, (b) max-epoch bands (<4, 4-10, >10),
     (c) dose-concentration (max dose weight) terciles, BH-FDR across the 3 families;
     plus Spearman of |residual| vs predicted value / max epoch / support distance
     (Hellinger distance to the nearest train neighbor of the run's own OOF folds).
     Figure f13_heteroskedasticity.png.
  2  PER-GROUP EPOCH DISCOUNT delta_g: extend the in-collapse discount
     h_eff = sum_j w_j r_delta(e_j) V[:,j] with ONE delta per cluster-group (3 groups:
     code-adjacent, web/text, tail+small; groups derived from cluster content profiles
     matched to the named dolma3/dolmino reference histograms). Ridge head on
     [h1000 | h_eff]; the 3-dim delta fit by coordinate descent on the train GCV score
     (the grug_fit.predict_h_eff pattern), same 15 folds. Compared against single-delta,
     the no-discount hist baseline and hist+hinge (both recomputed, cross-checked against
     hinge_zmacro_corrected_f.json). Figure f14_delta_groups.png.
  3  METRIC-REGISTRY-LITE: one row per eval task in the union of the 800 train runs'
     evals (55 bpb + 8 accuracy-only), with family, coverage, signal std, per-task
     kernel/ridge OOF Spearman (per_task_cv.parquet), leave-one-out-macro coherence,
     zmacro_english_20 membership and selection rationale. Written to
     metric_registry.parquet + registry.md.

Outputs: scratch/mixture_features/grug/validation_batch3.{json,md},
scratch/mixture_features/grug/metric_registry.parquet + registry.md, figures +
manifest3.json under scratch/mixture_features/report/figs3/.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402

import featurize  # noqa: E402
import grug_fit as gf  # noqa: E402
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from grug_target_analysis import build_task_matrix, task_family  # noqa: E402
from grug_validation_batch2 import (  # noqa: E402
    BLUE,
    CORR_F0,
    GREEN,
    INK,
    LINE,
    MUTED,
    ORANGE,
    apply_corrected_phase_constants,
    cluster_name,
    kernel_oof,
)
from grug_validation_checks import (  # noqa: E402
    REF_KERNEL,
    TARGET_NAME,
    build_zmacro_target,
    per_fold_spearman,
)
from retrodiction import SEED, _sq_hellinger  # noqa: E402
from scipy.stats import levene, spearmanr, wilcoxon  # noqa: E402
from sklearn.model_selection import RepeatedKFold  # noqa: E402

logger = logging.getLogger("grug_validation_batch3")

N_SPLITS, N_REPEATS = 5, 3
SCRATCH = gf.SCRATCH
GRUG_DIR = gf.GRUG_DIR
FIG_DIR = SCRATCH / "report" / "figs3"
DOMAIN_HIST_DIR = SCRATCH / "domain_histograms"
CORRECTED_F_REF = GRUG_DIR / "hinge_zmacro_corrected_f.json"

# Analysis 1
EPOCH_BANDS = (4.0, 10.0)  # max-epoch bands: <4, 4-10, >10
MIN_GROUP_N = 5  # groups smaller than this are excluded from the Levene test (reported anyway)
N_BINS = 10

# Analysis 2
GROUP_NAMES = ("code_adjacent", "web_text", "tail_small")
# named reference domains counted as "code-adjacent" content (code + formal math)
CODE_REFS = frozenset(
    {"dolma3_stack_edu", "dolmino_stack_edu_fim", "dolmino_synth_code", "dolma3_finemath_3plus", "dolmino_synth_math"}
)
SMALL_TOKEN_SHARE = 0.01  # clusters below 1% of swarm tokens join the tail group
MAX_SWEEPS = 4
REF_HIST_PERFOLD = 0.7396  # no-discount hist ridge, per-fold mean (validation_checks reference)
REF_HIST_HINGE_PERFOLD = 0.7665  # hist+hinge with corrected f (hinge_zmacro_corrected_f.json)


# ---------------------------------------------------------------------------
# Analysis 1: heteroskedasticity program
# ---------------------------------------------------------------------------


def bh_adjust(ps: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values."""
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, ps[i] * m / rank)
        adj[i] = prev
    return adj.tolist()


def support_distance(d2: np.ndarray, folds: list) -> np.ndarray:
    """Per-run Hellinger distance to the nearest TRAIN neighbor of its own OOF folds.

    For each repeat the run sits in exactly one test fold; distance = min over that
    fold's train rows of sqrt(mean-phase squared Hellinger), averaged over the 3 repeats.
    """
    n = d2.shape[0]
    dist = np.zeros((N_REPEATS, n))
    for fid, (tr, te) in enumerate(folds):
        tr, te = np.asarray(tr), np.asarray(te)
        dist[fid // N_SPLITS, te] = np.sqrt(d2[np.ix_(te, tr)].min(axis=1))
    return dist.mean(axis=0)


def group_variance_family(resid: np.ndarray, labels: np.ndarray, family: str) -> dict:
    """Brown-Forsythe test over a run->group labeling + per-group residual spread."""
    names = sorted(set(labels.tolist()))
    per_group = {}
    tested = []
    for g in names:
        r = resid[labels == g]
        per_group[str(g)] = {
            "n": len(r),
            "resid_sd": float(r.std(ddof=1)) if len(r) >= 2 else float("nan"),
            "abs_resid_median": float(np.median(np.abs(r))),
        }
        if len(r) >= MIN_GROUP_N:
            tested.append(r)
    stat, p = levene(*tested, center="median")
    sds = [v["resid_sd"] for v in per_group.values() if v["n"] >= MIN_GROUP_N]
    return {
        "family": family,
        "n_groups": len(names),
        "n_groups_tested": len(tested),
        "brown_forsythe_stat": float(stat),
        "p_value": float(p),
        "sd_ratio_max_min": float(max(sds) / min(sds)),
        "per_group": per_group,
    }


def binned_curve(x: np.ndarray, v: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    order = np.argsort(x)
    return [
        {
            "x_mean": float(x[b].mean()),
            "v_mean": float(v[b].mean()),
            "v_ci95": float(1.96 * v[b].std(ddof=1) / np.sqrt(len(b))),
            "n": len(b),
        }
        for b in np.array_split(order, n_bins)
    ]


def figure_f13(groupings, cont) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.6), constrained_layout=True)

    # top row: residual spread per group, one panel per family
    panel_specs = [
        ("max_dose_cluster", "max-dose cluster (batch2 LODO groups)"),
        ("max_epoch_band", "max-epoch band"),
        ("dose_concentration_tercile", "max dose weight tercile"),
    ]
    for ax, (fam, title) in zip(axes[0], panel_specs, strict=True):
        g = groupings[fam]
        labels, data = zip(*sorted(g["_data"].items(), key=lambda kv: np.median(np.abs(kv[1]))), strict=True)
        bp = ax.boxplot(data, vert=True, showfliers=False, widths=0.55, patch_artist=True, medianprops={"color": INK})
        for box in bp["boxes"]:
            box.set(facecolor="#dce9f9", edgecolor=BLUE, linewidth=0.9)
        many = len(labels) > 6
        ax.set_xticks(
            range(1, len(labels) + 1),
            [f"{n} n={len(d)}" for n, d in zip(labels, data, strict=True)],
            fontsize=5.5 if many else 7.5,
            rotation=90 if many else 0,
        )
        ax.axhline(0, color=LINE, lw=0.9)
        ax.set_title(title, fontsize=8.5)
        ax.annotate(
            f"BF p={g['p_value']:.2g} (BH {g['p_bh']:.2g})\nSD ratio {g['sd_ratio_max_min']:.2f}",
            xy=(0.02, 0.97),
            xycoords="axes fraction",
            va="top",
            fontsize=7,
            color=INK,
        )
        ax.set_ylabel("kernel OOF residual" if fam == "max_dose_cluster" else "")

    # bottom row: |residual| vs continuous covariates
    for ax, key, color, xlabel in zip(
        axes[1],
        ("predicted", "max_epoch", "support_distance"),
        (BLUE, GREEN, ORANGE),
        (
            "predicted zmacro_english_20 (OOF)",
            "max simulated epochs (corrected f)",
            "Hellinger dist. to nearest train neighbor",
        ),
        strict=True,
    ):
        c = cont[key]
        x, v = c["_xy"]
        ax.scatter(x, v, s=4, alpha=0.2, color=color, edgecolors="none", rasterized=True)
        bx = [b["x_mean"] for b in c["bins"]]
        bv = [b["v_mean"] for b in c["bins"]]
        be = [b["v_ci95"] for b in c["bins"]]
        ax.errorbar(
            bx, bv, yerr=be, fmt="o", ms=4.5, color=INK, mfc=color, mec=INK, mew=0.6, lw=1.1, capsize=2, zorder=3
        )
        ax.set_title(f"|residual| vs {key.replace('_', ' ')}\nSpearman {c['spearman']:+.3f} (p={c['p_value']:.2g})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("|kernel OOF residual|" if key == "predicted" else "")
    fig.suptitle(
        "Heteroskedasticity of the kernel OOF residuals (800 train runs, zmacro_english_20): "
        "dots = 10 equal-count bins (95% CI)",
        fontsize=9,
    )
    fig.savefig(FIG_DIR / "f13_heteroskedasticity.png", dpi=180)
    plt.close(fig)


def analysis_1(d2_hell, w, e, y, buckets_table, buckets, folds, n_jobs) -> dict:
    oof = kernel_oof(d2_hell, y, folds, n_jobs)
    pred = np.nanmean(oof, axis=0)
    resid = y - pred
    aresid = np.abs(resid)

    # grouping a: max-dose cluster (the batch2 LODO groups, corrected fractions)
    dose = CORR_F0 * w[:, 0, :] + (1.0 - CORR_F0) * w[:, 1, :]
    cluster_of_bucket = buckets_table.set_index("bucket").loc[buckets, "cluster_id"].to_numpy()
    run_cluster = np.array([cluster_name(c) for c in cluster_of_bucket[dose.argmax(axis=1)]])
    # grouping b: max-epoch band
    max_e = e.max(axis=1)
    lo, hi = EPOCH_BANDS
    band = np.where(max_e < lo, f"<{lo:g}", np.where(max_e <= hi, f"{lo:g}-{hi:g}", f">{hi:g}"))
    # grouping c: dose-concentration tercile
    conc = dose.max(axis=1)
    t1, t2 = np.quantile(conc, [1 / 3, 2 / 3])
    terc = np.where(conc <= t1, "t1_diffuse", np.where(conc <= t2, "t2_mid", "t3_concentrated"))

    groupings = {
        "max_dose_cluster": group_variance_family(resid, run_cluster, "max_dose_cluster"),
        "max_epoch_band": group_variance_family(resid, band, "max_epoch_band"),
        "dose_concentration_tercile": group_variance_family(resid, terc, "dose_concentration_tercile"),
    }
    fams = list(groupings)
    for fam, p_bh in zip(fams, bh_adjust([groupings[f]["p_value"] for f in fams]), strict=True):
        groupings[fam]["p_bh"] = p_bh
    for fam, labels in (
        ("max_dose_cluster", run_cluster),
        ("max_epoch_band", band),
        ("dose_concentration_tercile", terc),
    ):
        groupings[fam]["_data"] = {g: resid[labels == g] for g in sorted(set(labels.tolist()))}

    sup = support_distance(d2_hell, folds)
    cont = {}
    for key, x in (("predicted", pred), ("max_epoch", max_e), ("support_distance", sup)):
        rho, p = spearmanr(aresid, x)
        cont[key] = {"spearman": float(rho), "p_value": float(p), "bins": binned_curve(x, aresid), "_xy": (x, aresid)}

    figure_f13(groupings, cont)
    for fam in fams:
        groupings[fam].pop("_data")
    for c in cont.values():
        c.pop("_xy")
    return {
        "note": (
            "residual = y - mean-over-repeats kernel OOF prediction (same 15 folds as every phase-2 "
            "comparison); Brown-Forsythe = Levene center='median' over groups with n >= "
            f"{MIN_GROUP_N}; BH-FDR across the 3 grouping families; support distance = per-repeat "
            "min Hellinger distance to the fold's train rows, averaged over repeats; epoch/dose "
            "quantities use corrected fractions 0.7987/0.2013"
        ),
        "resid_sd_overall": float(resid.std(ddof=1)),
        "groupings": groupings,
        "abs_resid_spearman": {k: {kk: vv for kk, vv in v.items() if kk != "bins"} for k, v in cont.items()},
        "binned": {k: v["bins"] for k, v in cont.items()},
    }


# ---------------------------------------------------------------------------
# Analysis 2: per-cluster-group epoch discount
# ---------------------------------------------------------------------------


def load_named_reference_hists() -> dict[str, np.ndarray]:
    """K=1000 token histograms of the 39 named dolma3/dolmino reference domains."""
    meta = json.loads((DOMAIN_HIST_DIR / "_meta.json").read_text())
    lookup = np.load(SCRATCH / "basis" / "lookup_5000_to_1000.npy")
    out = {}
    for name, dmeta in meta["domains"].items():
        df = pd.read_parquet(DOMAIN_HIST_DIR / dmeta["parquet"])
        h = np.zeros(1000)
        np.add.at(h, lookup[df["cluster_id"].to_numpy()], df["token_count"].to_numpy(dtype=np.float64))
        out[name] = h / h.sum()
    return out


def cluster_delta_groups(buckets: list[str], buckets_table: pd.DataFrame, v1000: np.ndarray) -> tuple[dict, dict]:
    """Assign each of the 168 buckets to one of the 3 delta groups via its cluster.

    Cluster profile = token-weighted mean of its buckets' K=1000 composition columns;
    each profile is matched to the named dolma3/dolmino reference histograms by
    Bhattacharyya affinity. code_adjacent = nearest reference in CODE_REFS (code + formal
    math); tail_small = the -1 tail cluster plus clusters holding < SMALL_TOKEN_SHARE of
    swarm tokens; web_text = the rest.
    """
    bt = buckets_table.set_index("bucket").loc[buckets]
    cluster_of_bucket = bt["cluster_id"].to_numpy()
    tokens = bt["total_tokens"].to_numpy(dtype=np.float64)
    total = tokens.sum()
    refs = load_named_reference_hists()
    doc = {}
    group_of_cluster = {}
    for c in sorted(set(cluster_of_bucket.tolist())):
        m = cluster_of_bucket == c
        p = v1000[:, m] @ tokens[m]
        p /= p.sum()
        aff = {name: float(np.sqrt(p * q).sum()) for name, q in refs.items()}
        top = sorted(aff, key=aff.get, reverse=True)[:3]
        share = float(tokens[m].sum() / total)
        if c == -1 or share < SMALL_TOKEN_SHARE:
            group = "tail_small"
        elif top[0] in CODE_REFS:
            group = "code_adjacent"
        else:
            group = "web_text"
        group_of_cluster[c] = group
        doc[cluster_name(c)] = {
            "group": group,
            "token_share": share,
            "n_buckets": int(m.sum()),
            "top3_reference_affinity": {t: aff[t] for t in top},
        }
    masks = {g: np.array([group_of_cluster[c] == g for c in cluster_of_bucket], dtype=np.float64) for g in GROUP_NAMES}
    for g in GROUP_NAMES:
        if masks[g].sum() == 0:
            raise ValueError(f"delta group {g} is empty")
    return masks, doc


def group_heff_bank(w, e, v1000, masks) -> dict[str, dict[float, np.ndarray]]:
    """Precomputed flat h_eff contribution of each (group, delta): sums to h_eff exactly."""
    bank = {}
    for g, m in masks.items():
        bank[g] = {}
        for delta in gf.DELTA_GRID:
            r = gf.r_delta(e, float(delta)) * m[None, :]
            bank[g][float(delta)] = gf.flat(np.stack([(w[:, p, :] * r) @ v1000.T for p in range(2)], axis=1))
    # exactness check: group contributions must reassemble the single-delta h_eff
    d0 = float(gf.DELTA_GRID[0])
    whole = gf.flat(gf.h_eff_features(w, e, v1000, d0))
    np.testing.assert_allclose(sum(bank[g][d0] for g in GROUP_NAMES), whole, rtol=1e-10, atol=1e-12)
    return bank


def predict_h_eff_grouped(h1000, bank, y, tr, te, init_delta: float) -> tuple[np.ndarray, dict[str, float]]:
    """Ridge on [h1000 | sum_g h_eff_g(delta_g)]; per-group delta by coordinate descent on GCV."""
    deltas = {g: float(init_delta) for g in GROUP_NAMES}

    def fit(dv: dict[str, float]) -> tuple[np.ndarray, float]:
        heff = bank[GROUP_NAMES[0]][dv[GROUP_NAMES[0]]].copy()
        for g in GROUP_NAMES[1:]:
            heff += bank[g][dv[g]]
        x = np.concatenate([h1000, heff], axis=1)
        return gf.ridge_gcv(x[tr], y[tr], x[te])

    best_pred, best_score = fit(deltas)
    for _sweep in range(MAX_SWEEPS):
        changed = False
        for g in GROUP_NAMES:
            for delta in gf.DELTA_GRID:
                delta = float(delta)
                if delta == deltas[g]:
                    continue
                trial = dict(deltas, **{g: delta})
                pred, score = fit(trial)
                if score < best_score:
                    best_pred, best_score, deltas = pred, score, trial
                    changed = True
        if not changed:
            break
    return best_pred, deltas


def _a2_fold(fid: int, tr, te, h1000, h1000_hinge, bank, w, e, v1000, y) -> dict:
    t0 = time.monotonic()
    tr, te = np.asarray(tr), np.asarray(te)
    p_hist = gf.predict_ridge(h1000, y, tr, te)
    p_hinge = gf.predict_ridge(h1000_hinge, y, tr, te)
    p_single, delta_single = gf.predict_h_eff(h1000, w, e, v1000, y, tr, te)
    p_group, deltas = predict_h_eff_grouped(h1000, bank, y, tr, te, delta_single)
    out = {
        "fold_id": fid,
        "hist": per_fold_spearman(p_hist, y[te]),
        "hist_hinge": per_fold_spearman(p_hinge, y[te]),
        "single_delta": per_fold_spearman(p_single, y[te]),
        "group_delta": per_fold_spearman(p_group, y[te]),
        "delta_single": float(delta_single),
        "deltas_group": deltas,
    }
    logger.info("A2 fold %d done %.1fs deltas=%s", fid + 1, time.monotonic() - t0, deltas)
    return out


def r_dstar(delta: float) -> float:
    return (1.0 - delta) / delta


def figure_f14(per_fold: dict[str, np.ndarray], deltas_group: list[dict], delta_single: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), constrained_layout=True)
    ax = axes[0]
    xs = np.arange(1, 16)
    comps = [
        ("single_delta", "hist", BLUE, "single-delta - hist"),
        ("group_delta", "hist", GREEN, "group-delta - hist"),
        ("group_delta", "single_delta", ORANGE, "group-delta - single-delta"),
    ]
    ax.axhline(0, color=LINE, lw=0.9)
    for a, b, color, label in comps:
        d = per_fold[a] - per_fold[b]
        ax.plot(xs, d, "o", ms=4.2, color=color, label=f"{label} (mean {d.mean():+.4f})")
    hh = per_fold["hist_hinge"] - per_fold["hist"]
    ax.axhline(hh.mean(), color=MUTED, ls="--", lw=1.1)
    ax.text(15.2, hh.mean(), f"hist+hinge margin {hh.mean():+.4f}", fontsize=7, color=MUTED, va="center")
    ax.set_xlabel("fold")
    ax.set_ylabel("paired per-fold Spearman delta")
    ax.set_title("epoch-discount variants vs the no-discount hist ridge")
    ax.legend(fontsize=7)

    ax = axes[1]
    grid = [float(d) for d in gf.DELTA_GRID]
    rng = np.random.default_rng(5)
    for gi, (g, color) in enumerate(zip(GROUP_NAMES, (ORANGE, BLUE, MUTED), strict=True)):
        vals = np.array([d[g] for d in deltas_group])
        ax.scatter(np.full(15, gi) + rng.uniform(-0.13, 0.13, 15), vals, s=18, color=color, alpha=0.75)
        ax.plot([gi - 0.22, gi + 0.22], [np.median(vals)] * 2, color=INK, lw=1.4)
    ax.scatter(np.full(15, len(GROUP_NAMES)) + rng.uniform(-0.13, 0.13, 15), delta_single, s=18, color=GREEN, alpha=0.75)
    ax.plot([len(GROUP_NAMES) - 0.22, len(GROUP_NAMES) + 0.22], [np.median(delta_single)] * 2, color=INK, lw=1.4)
    ax.set_xticks(range(len(GROUP_NAMES) + 1), [*GROUP_NAMES, "single (all)"], fontsize=7.5)
    ax.set_yscale("log")
    ax.set_yticks(grid, [f"{d:g}" for d in grid], fontsize=7)
    ax.set_ylabel("fitted delta (log grid)")
    ax.set_title("fitted delta per group (bar = median)")
    fig.suptitle("Per-cluster-group epoch discount delta_g vs single delta (ridge on [h1000 | h_eff])", fontsize=9)
    fig.savefig(FIG_DIR / "f14_delta_groups.png", dpi=180)
    plt.close(fig)


def _paired(per_fold: dict[str, np.ndarray], a: str, b: str) -> dict:
    d = per_fold[a] - per_fold[b]
    return {
        "a": a,
        "b": b,
        "d_mean": float(d.mean()),
        "d_std": float(d.std()),
        "wins": int((d > 0).sum()),
        "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
    }


def analysis_2(h1000, w, e, v1000, y, buckets, buckets_table, folds, n_jobs) -> dict:
    masks, group_doc = cluster_delta_groups(buckets, buckets_table, v1000)
    bank = group_heff_bank(w, e, v1000, masks)
    h1000_hinge = np.concatenate([h1000, gf.hinge_epoch_features(w, e)], axis=1)

    out = joblib.Parallel(n_jobs=n_jobs, backend="threading")(
        joblib.delayed(_a2_fold)(fid, tr, te, h1000, h1000_hinge, bank, w, e, v1000, y)
        for fid, (tr, te) in enumerate(folds)
    )
    out.sort(key=lambda d: d["fold_id"])
    per_fold = {k: np.array([o[k] for o in out]) for k in ("hist", "hist_hinge", "single_delta", "group_delta")}
    deltas_group = [o["deltas_group"] for o in out]
    delta_single = np.array([o["delta_single"] for o in out])

    # cross-check the recomputed baselines against the stored corrected-f reference
    ref = json.loads(CORRECTED_F_REF.read_text())
    checks = {
        "hist_mean_recomputed": float(per_fold["hist"].mean()),
        "hist_mean_stored": float(np.mean(ref["hist"])),
        "hist_hinge_mean_recomputed": float(per_fold["hist_hinge"].mean()),
        "hist_hinge_mean_stored": float(np.mean(ref["hist+hinge_true_f"])),
    }
    if abs(checks["hist_mean_recomputed"] - checks["hist_mean_stored"]) > 1e-6:
        raise AssertionError(f"hist baseline mismatch vs stored reference: {checks}")
    if abs(checks["hist_hinge_mean_recomputed"] - checks["hist_hinge_mean_stored"]) > 1e-6:
        raise AssertionError(f"hist+hinge baseline mismatch vs stored reference: {checks}")

    med_delta = {g: float(np.median([d[g] for d in deltas_group])) for g in GROUP_NAMES}
    figure_f14(per_fold, deltas_group, delta_single)
    return {
        "note": (
            "linear-surrogate track: ridge on [h1000 | h_eff]; group deltas by coordinate descent "
            f"over the {len(gf.DELTA_GRID)}-point delta grid on the train GCV score (predict_h_eff "
            "pattern), initialized at the fold's single-delta winner; epochs use corrected fractions "
            "0.7987/0.2013; baselines recomputed on the same folds and cross-checked against "
            "hinge_zmacro_corrected_f.json"
        ),
        "delta_groups": group_doc,
        "group_bucket_counts": {g: int(masks[g].sum()) for g in GROUP_NAMES},
        "group_token_share": {
            g: float(sum(v["token_share"] for v in group_doc.values() if v["group"] == g)) for g in GROUP_NAMES
        },
        "per_fold": {k: v.tolist() for k, v in per_fold.items()},
        "means": {k: float(v.mean()) for k, v in per_fold.items()},
        "baseline_crosscheck": checks,
        "paired": {
            "single_vs_hist": _paired(per_fold, "single_delta", "hist"),
            "group_vs_hist": _paired(per_fold, "group_delta", "hist"),
            "group_vs_single": _paired(per_fold, "group_delta", "single_delta"),
            "group_vs_hist_hinge": _paired(per_fold, "group_delta", "hist_hinge"),
            "hinge_vs_hist": _paired(per_fold, "hist_hinge", "hist"),
        },
        "fitted_delta": {
            "single_per_fold": delta_single.tolist(),
            "single_median": float(np.median(delta_single)),
            "group_per_fold": deltas_group,
            "group_median": med_delta,
            "group_median_r_dstar": {g: r_dstar(d) for g, d in med_delta.items()},
            "r_dstar_interpretation": (
                "R_D* = (1-delta)/delta: the effective number of extra epochs a bucket's data keeps "
                "paying for before repeats stop helping -- larger R_D* means repeats of that group's "
                "data retain value longer (delta -> 0: no discount; delta = 1: repeats worthless)"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Analysis 3: metric-registry-lite
# ---------------------------------------------------------------------------


def loo_macro_coherence(ymat: np.ndarray, j: int) -> float:
    """Spearman of task j with the leave-one-out macro over the 800 runs."""
    macro_all = np.nanmean(ymat, axis=1)
    n_obs = (~np.isnan(ymat)).sum(axis=1)
    yj = ymat[:, j]
    m = ~np.isnan(yj)
    loo = (macro_all[m] * n_obs[m] - yj[m]) / (n_obs[m] - 1)
    return float(spearmanr(yj[m], loo).statistic)


def selection_rationale(task: str, family: str, has_bpb: bool, in_target: bool) -> str:
    if not has_bpb:
        return "excluded: accuracy-only metric (no bpb); target is a bpb z-macro"
    if in_target:
        return "in target: english family, member of the frozen zmacro_english_20 list"
    if family == "multilingual":
        return (
            "excluded: multilingual family (belebele/include) -- weakly predictable, separate factor "
            "(target repair 2026-07-15)"
        )
    if family in ("code", "math"):
        return f"excluded: single-task {family} family, idiosyncratic axis"
    return "excluded: english bpb task outside the frozen 20-task list"


def analysis_3(runs: pd.DataFrame, ymat, tasks, rec) -> dict:
    per_task = pd.read_parquet(GRUG_DIR / "per_task_cv.parquet").set_index("task")
    evs = [json.loads(e) for e in runs["evals"]]
    all_tasks = sorted({k for ev in evs for k in ev})
    target_set = set(rec["task_list"])
    zmacro_rel = json.loads((GRUG_DIR / "target_candidates.json").read_text())["reliability"]["e_zmacro_english"]

    rows = []
    for t in all_tasks:
        has_bpb = t in per_task.index
        metric = "bpb" if has_bpb else "acc"
        vals = np.array([ev[t][metric] for ev in evs if t in ev and metric in ev[t]], dtype=np.float64)
        family = task_family(t)
        in_target = t in target_set
        row = {
            "task": t,
            "family": family,
            "metric": metric,
            "n_runs_observed": len(vals),
            "signal_mean": float(vals.mean()),
            "signal_std": float(vals.std(ddof=1)),
            "in_zmacro_english_20": in_target,
            "selection_rationale": selection_rationale(t, family, has_bpb, in_target),
            "target_splithalf_reliability_sb": zmacro_rel["reliability_sb_spearman"] if in_target else np.nan,
        }
        if has_bpb:
            pt = per_task.loc[t]
            j = tasks.index(t)
            row.update(
                kernel_oof_spearman=float(pt["kernel_spearman"]),
                kernel_oof_spearman_std=float(pt["kernel_spearman_std"]),
                ridge_oof_spearman=float(pt["ridge_spearman"]),
                task_cluster=int(pt["cluster"]),
                coherence_loo_macro=loo_macro_coherence(ymat, j),
            )
        else:
            row.update(
                kernel_oof_spearman=np.nan,
                kernel_oof_spearman_std=np.nan,
                ridge_oof_spearman=np.nan,
                task_cluster=-1,
                coherence_loo_macro=np.nan,
            )
        rows.append(row)
    reg = pd.DataFrame(rows)
    reg.to_parquet(GRUG_DIR / "metric_registry.parquet", index=False)
    write_registry_md(reg, zmacro_rel)
    return {
        "note": (
            "one row per task in the union of the 800 train runs' evals column: 55 bpb + "
            f"{int((reg['metric'] == 'acc').sum())} accuracy-only = {len(reg)} tasks (the seedpanel "
            "'60-task readout' is the HF evals column; the local union over 800 runs is 63). "
            "Per-task split-half reliability is NOT computable from run-level scalars (the "
            "target_report machinery splits task LISTS); the target-level Spearman-Brown "
            "reliability is attached to the 20 member tasks instead, and coherence_loo_macro "
            "(Spearman with the leave-one-out macro) is the per-task signal-quality proxy"
        ),
        "n_tasks": len(reg),
        "n_bpb": int((reg["metric"] == "bpb").sum()),
        "n_in_target": int(reg["in_zmacro_english_20"].sum()),
        "zmacro_reliability": zmacro_rel,
        "families": reg.groupby("family")["task"].count().to_dict(),
        "kernel_spearman_by_family": {
            f: float(s) for f, s in reg.groupby("family")["kernel_oof_spearman"].mean().items()
        },
        "artifacts": ["metric_registry.parquet", "registry.md"],
    }


def write_registry_md(reg: pd.DataFrame, zmacro_rel: dict) -> None:
    lines = []
    A = lines.append
    A("# Grug metric registry (lite; 800 train runs, holdout untouched)\n")
    A(
        f"{len(reg)} eval tasks = union of the 800 train runs' `evals` column "
        f"({int((reg['metric'] == 'bpb').sum())} bpb, {int((reg['metric'] == 'acc').sum())} accuracy-only)."
    )
    A(
        f"Target zmacro_english_20: split-half Spearman-Brown reliability "
        f"{zmacro_rel['reliability_sb_spearman']:.3f} (implied max Spearman "
        f"{zmacro_rel['max_spearman_sqrt_rel']:.3f}). Per-task split-half reliability is not computable "
        "from run-level scalars; `coherence_loo_macro` (Spearman with the leave-one-out macro) is the "
        "per-task proxy. Kernel/ridge columns = per-task OOF Spearman from per_task_cv.parquet "
        "(Hellinger kernel K=1000 / hist-ridge K=1000, RepeatedKFold(5,3,seed 0)).\n"
    )
    A("| task | family | metric | n | signal std | kernel rho | ridge rho | coherence | in target | rationale |")
    A("|------|--------|--------|---|-----------|------------|-----------|-----------|-----------|-----------|")
    reg = reg.sort_values(["in_zmacro_english_20", "family", "kernel_oof_spearman"], ascending=[False, True, False])
    for _, r in reg.iterrows():
        k = f"{r['kernel_oof_spearman']:+.3f}" if np.isfinite(r["kernel_oof_spearman"]) else "n/a"
        rd = f"{r['ridge_oof_spearman']:+.3f}" if np.isfinite(r["ridge_oof_spearman"]) else "n/a"
        co = f"{r['coherence_loo_macro']:+.3f}" if np.isfinite(r["coherence_loo_macro"]) else "n/a"
        A(
            f"| {r['task']} | {r['family']} | {r['metric']} | {r['n_runs_observed']} | "
            f"{r['signal_std']:.4f} | {k} | {rd} | {co} | {'Y' if r['in_zmacro_english_20'] else '-'} | "
            f"{r['selection_rationale']} |"
        )
    (GRUG_DIR / "registry.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Verdicts, report, manifest
# ---------------------------------------------------------------------------


def make_verdicts(res: dict) -> dict:
    v = {}
    if "analysis_1" in res:
        a = res["analysis_1"]
        large_groups = [(f, g) for f, g in a["groupings"].items() if g["p_bh"] < 0.05 and g["sd_ratio_max_min"] > 1.5]
        large_cont = {k: c["spearman"] for k, c in a["abs_resid_spearman"].items() if abs(c["spearman"]) >= 0.2}
        small_sig = {
            k: c for k, c in a["abs_resid_spearman"].items() if c["p_value"] < 0.05 and abs(c["spearman"]) < 0.2
        }
        if not large_groups and not large_cont:
            extra = (
                " (only a small significant trend: "
                + "; ".join(
                    f"|resid| vs {k} Spearman {c['spearman']:+.3f}, p={c['p_value']:.2g}" for k, c in small_sig.items()
                )
                + ")"
                if small_sig
                else ""
            )
            v["analysis_1"] = (
                "no unexplained large effects that would change recommendations: no grouping family has "
                "BH-adjusted Brown-Forsythe p < 0.05 with residual-SD ratio > 1.5, and no |residual| "
                "Spearman reaches 0.2" + extra
            )
        else:
            parts = [
                f"{f}: BF BH-p {g['p_bh']:.3g}, group residual-SD ratio {g['sd_ratio_max_min']:.2f}"
                for f, g in large_groups
            ]
            parts += [f"|resid| vs {k} Spearman {rho:+.3f}" for k, rho in large_cont.items()]
            v["analysis_1"] = "heteroskedasticity detected -- " + "; ".join(parts)
    if "analysis_2" in res:
        a = res["analysis_2"]
        g_vs_s = a["paired"]["group_vs_single"]
        beats = g_vs_s["d_mean"] > 0 and g_vs_s["wilcoxon_p"] < 0.05
        rd = a["fitted_delta"]["group_median_r_dstar"]
        v["analysis_2"] = (
            f"group-wise delta {'BEATS' if beats else 'does NOT beat'} the global delta "
            f"({g_vs_s['d_mean']:+.4f}, {g_vs_s['wins']}/15, p={g_vs_s['wilcoxon_p']:.4f}); "
            f"group delta {a['means']['group_delta']:.4f} vs single {a['means']['single_delta']:.4f} vs "
            f"hist {a['means']['hist']:.4f} vs hist+hinge {a['means']['hist_hinge']:.4f}; "
            "median R_D* " + ", ".join(f"{g} {rd[g]:.1f}" for g in GROUP_NAMES)
        )
    if "analysis_3" in res:
        a = res["analysis_3"]
        v["analysis_3"] = (
            f"registry written: {a['n_tasks']} tasks ({a['n_bpb']} bpb), {a['n_in_target']} in "
            f"zmacro_english_20; per-family mean kernel rho "
            + ", ".join(f"{f} {s:+.3f}" for f, s in sorted(a["kernel_spearman_by_family"].items()))
        )
    return v


def write_report(res: dict) -> str:
    lines = []
    A = lines.append
    A("# Grug validation batch 3 (codex terminal-state items; 800 train runs only)\n")
    A(f"Target {TARGET_NAME} (frozen train z-stats), folds RepeatedKFold(5,3,seed 0), kernel =")
    A(f"Hellinger kernel ridge K=1000 (reference {REF_KERNEL} per-fold mean). Corrected phase")
    A(f"fractions {CORR_F0:.4f}/{1 - CORR_F0:.4f} applied to every epoch/dose computation.\n")

    if "analysis_1" in res:
        a = res["analysis_1"]
        A("## 1. Heteroskedasticity program (diagnostic, not headline)\n")
        A(f"- overall kernel OOF residual SD {a['resid_sd_overall']:.4f}")
        A("| grouping family | groups (tested) | Brown-Forsythe p | BH-FDR p | SD ratio max/min |")
        A("|-----------------|-----------------|------------------|----------|------------------|")
        for f, g in a["groupings"].items():
            A(
                f"| {f} | {g['n_groups']} ({g['n_groups_tested']}) | {g['p_value']:.4g} | {g['p_bh']:.4g} | "
                f"{g['sd_ratio_max_min']:.2f} |"
            )
        A("")
        A("| covariate | Spearman(|resid|, x) | p |")
        A("|-----------|----------------------|---|")
        for k, c in a["abs_resid_spearman"].items():
            A(f"| {k} | {c['spearman']:+.4f} | {c['p_value']:.3g} |")
        A("")
        A(f"- {a['note']}")
        A(f"- verdict: {res['verdicts']['analysis_1']}")
        A("- figure: figs3/f13_heteroskedasticity.png\n")

    if "analysis_2" in res:
        a = res["analysis_2"]
        A("## 2. Per-cluster-group epoch discount (3-dim delta vs global delta)\n")
        A("- cluster groups (from content profiles matched to named dolma3/dolmino references):")
        for g in GROUP_NAMES:
            members = sorted(c for c, d in a["delta_groups"].items() if d["group"] == g)
            A(
                f"  - {g}: {a['group_bucket_counts'][g]} buckets, "
                f"{a['group_token_share'][g]:.1%} of tokens -- {', '.join(members)}"
            )
        A("")
        A("| model | per-fold Spearman mean |")
        A("|-------|------------------------|")
        for k in ("hist", "hist_hinge", "single_delta", "group_delta"):
            A(f"| {k} | {a['means'][k]:.4f} |")
        A("")
        A("| comparison | d mean | wins/15 | Wilcoxon p |")
        A("|------------|--------|---------|------------|")
        for name, d in a["paired"].items():
            A(f"| {name} | {d['d_mean']:+.4f} | {d['wins']}/15 | {d['wilcoxon_p']:.4f} |")
        A("")
        fd = a["fitted_delta"]
        A(
            "- fitted delta (median over 15 folds): "
            + ", ".join(f"{g} {fd['group_median'][g]:g} (R_D* {fd['group_median_r_dstar'][g]:.1f})" for g in GROUP_NAMES)
            + f"; single {fd['single_median']:g}"
        )
        A(f"- {fd['r_dstar_interpretation']}")
        A(f"- {a['note']}")
        A(f"- verdict: {res['verdicts']['analysis_2']}")
        A("- figure: figs3/f14_delta_groups.png\n")

    if "analysis_3" in res:
        a = res["analysis_3"]
        A("## 3. Metric registry (lite)\n")
        A(f"- {a['note']}")
        A(
            f"- {a['n_tasks']} tasks ({a['n_bpb']} bpb), {a['n_in_target']} in the target; families "
            f"{a['families']}; artifacts: metric_registry.parquet, registry.md"
        )
        A(f"- verdict: {res['verdicts']['analysis_3']}\n")
    return "\n".join(lines)


MANIFEST_TEMPLATES = {
    "f13_heteroskedasticity.png": {
        "message": (
            "Where the kernel's OOF errors are larger: residual spread by max-dose cluster, "
            "max-epoch band and dose concentration (Brown-Forsythe, BH-FDR), plus |residual| against "
            "predicted value, max epoch and distance to the training support."
        ),
        "data_source": (
            "kernel OOF residuals recomputed on the 800 train runs (RepeatedKFold(5,3,seed 0)); "
            "groups from corrected-dose argmax clusters and corrected epoch table"
        ),
    },
    "f14_delta_groups.png": {
        "message": (
            "Does one epoch-discount rate per content group (code-adjacent / web-text / tail+small) "
            "beat a single global rate? Paired per-fold margins over the no-discount hist ridge, and "
            "the fitted delta per group."
        ),
        "data_source": (
            "ridge on [h1000 | h_eff(delta_g)] with per-group delta by coordinate descent on train "
            "GCV; groups from cluster content profiles vs named dolma3/dolmino reference histograms; "
            "corrected phase fractions"
        ),
    },
}


def update_manifest(fig_names: list[str]) -> None:
    path = FIG_DIR / "manifest3.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    for f in fig_names:
        manifest[f] = MANIFEST_TEMPLATES[f]
    path.write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyses", default="1,2,3", help="comma list of analyses to run")
    args = ap.parse_args()
    todo = set(args.analyses.split(","))
    n_jobs = joblib.cpu_count()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    apply_corrected_phase_constants()
    p0, p1 = gf.phase_step_split()
    assert (p0, p1) == (38_144, 9_615) and gf.TOTAL_STEPS == 47_759

    hists, views, _centroids, _rff_means, _rff_order, buckets_table = gf.load_grug_artifacts()
    buckets = [h.domain for h in hists]
    v1000, order = featurize.composition_matrix(hists, k=1000, views=views)
    assert order == buckets
    v1000 = np.asarray(v1000)

    runs = pd.read_parquet(gf.TRAIN_RUNS)
    w = gf.weight_matrix(runs, buckets)
    rec = json.loads((GRUG_DIR / "target_candidates.json").read_text())["recommended_target"]
    y = build_zmacro_target(runs, rec)
    ymat, tasks = build_task_matrix(runs)

    e, summ = gf.stage_b_epochs(w, buckets, buckets_table)
    assert abs(summ["phase_token_fractions"][0] - 0.7987) < 5e-4
    hphase = gf.per_phase_hist(w, v1000)
    h1000 = gf.flat(hphase)
    d2_hell = _sq_hellinger(hphase)
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    folds = list(rkf.split(np.arange(len(y))))

    out_path = GRUG_DIR / "validation_batch3.json"
    res: dict = json.loads(out_path.read_text()) if out_path.exists() else {}
    res.setdefault(
        "meta",
        {
            "target": TARGET_NAME,
            "reference_kernel_perfold": REF_KERNEL,
            "reference_hist_perfold": REF_HIST_PERFOLD,
            "reference_hist_hinge_perfold": REF_HIST_HINGE_PERFOLD,
            "corrected_phase_fractions": [CORR_F0, 1.0 - CORR_F0],
            "compute_notes": (
                "8-core budget: no cuts to the specified scope; group-delta fits limited to "
                f"{MAX_SWEEPS} coordinate-descent sweeps over the standard 8-point delta grid"
            ),
        },
    )
    figs = []

    if "1" in todo:
        t0 = time.monotonic()
        res["analysis_1"] = analysis_1(d2_hell, w, e, y, buckets_table, buckets, folds, n_jobs)
        figs.append("f13_heteroskedasticity.png")
        logger.info("analysis 1 done %.0fs", time.monotonic() - t0)
    if "2" in todo:
        t0 = time.monotonic()
        res["analysis_2"] = analysis_2(h1000, w, e, v1000, y, buckets, buckets_table, folds, n_jobs)
        figs.append("f14_delta_groups.png")
        logger.info("analysis 2 done %.0fs", time.monotonic() - t0)
    if "3" in todo:
        t0 = time.monotonic()
        res["analysis_3"] = analysis_3(runs, ymat, tasks, rec)
        logger.info("analysis 3 done %.0fs", time.monotonic() - t0)

    res["verdicts"] = make_verdicts(res)
    out_path.write_text(json.dumps(res, indent=2, default=gf.json_default))
    (GRUG_DIR / "validation_batch3.md").write_text(write_report(res))
    update_manifest(figs)
    print(write_report(res))
    logger.info("wrote %s", GRUG_DIR / "validation_batch3.md")


if __name__ == "__main__":
    main()
