# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "joblib"]
# ///
"""Holdout & sample-efficiency analysis for parametric scaling-law forms.

Bootstrap results are cached to holdout_plots/bootstrap_cache.pkl so that plots
can be iterated on without rerunning computation.  New functional forms are
automatically detected and only the missing models are fitted.

Produces:
  Table 1   – Models ranked by held-out Huber loss at n_train=48
  Figure 1  – Learning curves overview (all + top 5)
  Figure 1b – Learning curves per-model (small multiples with IQR)
  Figure 2  – Overfitting gap (train vs test RMSE, small multiples)
  Figure 3  – Stability of predicted optimum (box plots at n_train=48)
  Figure 4  – 2D heatmaps of predicted BPB with isoloss contours

Usage:
  uv run holdout_analysis.py                   # all models
  uv run holdout_analysis.py LogQuad           # subset (substring match)
  uv run holdout_analysis.py --plots-only      # skip bootstrap, just regenerate plots
"""

import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from joblib import Parallel, delayed
from pathlib import Path
import matplotlib.colors as mcolors
from scipy.optimize import minimize as sp_minimize

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from scaling_models import (  # noqa: E402
    MODELS,
    SC_EPOCH_MULT,
    build_features,
    cv_metrics,
    fit_linear,
    make_2d_grid,
    make_slice_for_kind,
)

sys.stdout.reconfigure(line_buffering=True)

script_dir = Path(__file__).parent

# =========================================================================
# Data loading
# =========================================================================
_csv_name = os.environ.get("DATA_CSV", "two_phase_starcoder.csv")
TARGET = "eval/paloma/dolma_100_programing_languages/bpb"

df = pd.read_csv(script_dir / _csv_name)
df = df[df["status"] == "completed"].copy()

y = df[TARGET].values
N = len(df)
p0_sc = df["phase_0_starcoder"].values
p1_sc = df["phase_1_starcoder"].values

slice_grid = np.linspace(0.002, 0.998, 300)

print(f"Loaded {N} samples from {_csv_name}")
print(f"Target: {TARGET}")
print(f"y range: [{y.min():.4f}, {y.max():.4f}], median={np.median(y):.4f}")

_out_name = os.environ.get("HOLDOUT_OUT_DIR", "holdout_plots")
OUT_DIR = script_dir / _out_name
OUT_DIR.mkdir(exist_ok=True)
CACHE_FILE = OUT_DIR / "bootstrap_cache.pkl"

# -------------------------------------------------------------------------
# Matplotlib style
# -------------------------------------------------------------------------
mpl.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{opensans}\renewcommand{\familydefault}{\sfdefault}" r"\usepackage{amssymb}",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def clean_name(name: str) -> str:
    """Strip LaTeX braces from model names for display."""
    return name.replace("{-}", "-").replace("{", "").replace("}", "")


# =========================================================================
# Configuration
# =========================================================================
# Generate train sizes: 8 evenly spaced values from ~10% to ~87% of N
_lo, _hi = max(12, N // 10), int(N * 0.87)
TRAIN_SIZES = sorted(set(int(round(v)) for v in np.linspace(_lo, _hi, 8)))
B = 200  # bootstrap iterations per training size
REFERENCE_N = TRAIN_SIZES[-1]  # use largest train size for Table 1 and Figure 3
N_JOBS = -1  # use all cores (M3 Max)
SEED_BASE = 42
HEATMAP_RES = 80  # grid resolution for 2D heatmaps

# Predictions outside this range are treated as divergent (failed fit)
MAX_PRED = 10.0 * float(np.max(y))

# Config dict for cache validation
BOOTSTRAP_CONFIG = {
    "train_sizes": TRAIN_SIZES,
    "B": B,
    "seed_base": SEED_BASE,
    "max_pred": MAX_PRED,
}

# =========================================================================
# CLI: model selection + --plots-only / --quick flags
# =========================================================================
plots_only = "--plots-only" in sys.argv
quick_mode = "--quick" in sys.argv
cli_filters = [a for a in sys.argv[1:] if not a.startswith("--")]

if cli_filters:
    selected = []
    for m in MODELS:
        if any(f.lower() in m.name.lower() for f in cli_filters):
            selected.append(m)
    if not selected:
        print(f"No models match filters: {cli_filters}")
        print("Available:", [m.name for m in MODELS])
        sys.exit(1)
    RUN_MODELS = selected
else:
    RUN_MODELS = list(MODELS)

print(f"Models to evaluate ({len(RUN_MODELS)}):")
for m in RUN_MODELS:
    print(f"  {clean_name(m.name)}")


# =========================================================================
# Huber loss
# =========================================================================
def huber_loss(y_true, y_pred, delta):
    """Element-wise Huber loss, averaged."""
    r = y_true - y_pred
    abs_r = np.abs(r)
    return float(np.mean(np.where(abs_r <= delta, 0.5 * r**2, delta * (abs_r - 0.5 * delta))))


# Compute reference delta from full-data Linear model residuals
_X_weight_for_delta = build_features("weight", p0_sc, p1_sc)
_linear_pred, _ = fit_linear(_X_weight_for_delta, y)
_linear_resid = y - _linear_pred(_X_weight_for_delta)
_mad = np.median(np.abs(_linear_resid - np.median(_linear_resid)))
HUBER_DELTA = 1.345 * _mad / 0.6745
print(f"\nHuber delta = {HUBER_DELTA:.6f} (from Linear MAD={_mad:.6f})")


# =========================================================================
# Feature-type dispatch
# =========================================================================
FEATURE_KINDS = {m.name: m.feature_kind for m in MODELS}


# =========================================================================
# Cache management
# =========================================================================
def _load_cache():
    """Load cached bootstrap results.  Returns (config, model_results) or (None, {})."""
    if not CACHE_FILE.exists():
        return None, {}
    try:
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        return cache.get("config"), cache.get("models", {})
    except Exception as e:
        print(f"  Warning: cache load failed ({e}), recomputing all models")
        return None, {}


def _save_cache(config, model_results):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"config": config, "models": model_results}, f)


# =========================================================================
# Single bootstrap iteration
# =========================================================================
def _run_one_iter(n_train, seed, model_specs, all_p0, all_p1, all_y, delta, sg, max_pred):
    """Run one bootstrap iteration for all models at a given n_train."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(all_y))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    y_train, y_test = all_y[train_idx], all_y[test_idx]
    p0_train, p1_train = all_p0[train_idx], all_p1[train_idx]
    p0_test, p1_test = all_p0[test_idx], all_p1[test_idx]

    fail = {
        "test_huber": np.nan,
        "test_rmse": np.nan,
        "train_huber": np.nan,
        "train_rmse": np.nan,
        "opt_p1": np.nan,
        "success": False,
    }

    results = {}
    for name, fit_fn, kind in model_specs:
        X_train = build_features(kind, p0_train, p1_train)
        X_test = build_features(kind, p0_test, p1_test)

        try:
            pred_fn, _ = fit_fn(X_train, y_train)
            test_pred = pred_fn(X_test)
            train_pred = pred_fn(X_train)

            # Sanity check: reject divergent predictions (exp overflow, etc.)
            if (
                np.any(np.abs(test_pred) > max_pred)
                or np.any(np.isnan(test_pred))
                or np.any(np.abs(train_pred) > max_pred)
            ):
                results[name] = fail
                continue

            Xs = make_slice_for_kind(kind, sg)
            slice_pred = pred_fn(Xs)
            valid_mask = np.isfinite(slice_pred) & (np.abs(slice_pred) < max_pred)
            if valid_mask.any():
                opt_p1 = float(sg[valid_mask][np.argmin(slice_pred[valid_mask])])
            else:
                opt_p1 = np.nan

            results[name] = {
                "test_huber": huber_loss(y_test, test_pred, delta),
                "test_rmse": float(np.sqrt(np.mean((y_test - test_pred) ** 2))),
                "train_huber": huber_loss(y_train, train_pred, delta),
                "train_rmse": float(np.sqrt(np.mean((y_train - train_pred) ** 2))),
                "opt_p1": opt_p1,
                "success": True,
            }
        except Exception:
            results[name] = dict(fail)
    return results


# =========================================================================
# Bootstrap: load cache, compute missing, merge & save
# =========================================================================
model_specs = [(m.name, m.fit_fn, m.feature_kind) for m in RUN_MODELS]
model_names = [m.name for m in RUN_MODELS]
model_colors = {m.name: m.color for m in RUN_MODELS}

sg = np.array(slice_grid)

cached_config, cached_models = _load_cache()
config_matches = cached_config == BOOTSTRAP_CONFIG

if config_matches:
    cached_names = set(cached_models.keys())
    needed_names = [n for n in model_names if n not in cached_names]
    print(f"\nCache: {len(cached_names)} models cached, {len(needed_names)} to compute")
else:
    if cached_config is not None:
        print("\nCache: config mismatch, recomputing all models")
    else:
        print("\nCache: no cache found, computing all models")
    cached_models = {}
    needed_names = list(model_names)

# Run bootstrap only for uncached models
if needed_names and not plots_only and not quick_mode:
    needed_specs = [(m.name, m.fit_fn, m.feature_kind) for m in RUN_MODELS if m.name in needed_names]

    print(f"Running bootstrap: {len(TRAIN_SIZES)} sizes x {B} iters x {len(needed_specs)} models")
    t0 = time.time()

    # Collect per-model results: {model_name: {n_train: {metrics}}}
    new_model_results = {name: {} for name in needed_names}

    for si, n_train in enumerate(TRAIN_SIZES):
        seeds = [SEED_BASE + si * B + b for b in range(B)]

        iter_results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_run_one_iter)(
                n_train,
                seed,
                needed_specs,
                p0_sc,
                p1_sc,
                y,
                HUBER_DELTA,
                sg,
                MAX_PRED,
            )
            for seed in seeds
        )

        for name in needed_names:
            new_model_results[name][n_train] = {
                "test_huber": np.array([r[name]["test_huber"] for r in iter_results]),
                "test_rmse": np.array([r[name]["test_rmse"] for r in iter_results]),
                "train_huber": np.array([r[name]["train_huber"] for r in iter_results]),
                "train_rmse": np.array([r[name]["train_rmse"] for r in iter_results]),
                "opt_p1": np.array([r[name]["opt_p1"] for r in iter_results]),
                "success": np.array([r[name]["success"] for r in iter_results]),
            }

        elapsed = time.time() - t0
        print(f"  n_train={n_train:3d}  done  ({elapsed:.1f}s elapsed)")

    total_time = time.time() - t0
    print(f"Bootstrap complete in {total_time:.1f}s")

    # Merge into cache and save
    cached_models.update(new_model_results)
    _save_cache(BOOTSTRAP_CONFIG, cached_models)
    print(f"Cache saved ({len(cached_models)} models total)")

elif quick_mode:
    print("\n--- Quick mode: skipping bootstrap, will use CV metrics for ranking ---")

elif plots_only and not config_matches:
    print("ERROR: --plots-only but no valid cache exists. Run without --plots-only first.")
    sys.exit(1)

elif needed_names and plots_only:
    missing = [clean_name(n) for n in needed_names]
    print(f"Warning: --plots-only but {len(missing)} models are not cached: {missing}")
    print("  These models will be skipped in plots.")
    model_names = [n for n in model_names if n not in needed_names]
    RUN_MODELS = [m for m in RUN_MODELS if m.name not in needed_names]
    model_specs = [(m.name, m.fit_fn, m.feature_kind) for m in RUN_MODELS]

if not quick_mode:
    # Build all_results[n_train][model_name] from cache for plotting
    all_results = {}
    for n_train in TRAIN_SIZES:
        size_results = {}
        for name in model_names:
            size_results[name] = cached_models[name][n_train]
        all_results[n_train] = size_results

METRICS = {
    "huber": {
        "test_key": "test_huber",
        "train_key": "train_huber",
        "label": f"Huber ($\\delta$={HUBER_DELTA:.4f})",
        "short": "Huber",
    },
    "rmse": {"test_key": "test_rmse", "train_key": "train_rmse", "label": "RMSE", "short": "RMSE"},
}

if not quick_mode:
    # =====================================================================
    # Compute rankings for both metrics (requires bootstrap data)
    # =====================================================================
    ref = all_results[REFERENCE_N]

    def _compute_rankings(metric_key):
        """Compute model rankings sorted by median test metric at REFERENCE_N.

        Uses median (not mean) for ranking so the sort order matches the
        median values displayed in the learning-curve plots.
        """
        rows = []
        for name in model_names:
            d = ref[name]
            mask = d["success"]
            if mask.sum() == 0:
                rows.append((name, np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                continue
            test_vals = d[metric_key][mask]
            train_key = metric_key.replace("test_", "train_")
            train_vals = d[train_key][mask]
            opt_vals = d["opt_p1"][mask]
            median_test = float(np.median(test_vals))
            rows.append(
                (
                    name,
                    median_test,
                    float(np.std(test_vals)),
                    float(np.median(test_vals)),
                    float(np.median(train_vals)),
                    median_test / max(float(np.median(train_vals)), 1e-12),
                    float(np.mean(opt_vals)),
                    float(np.std(opt_vals)),
                    float(mask.mean()) * 100,
                )
            )
        rows.sort(key=lambda x: x[1])
        return rows

    rankings_huber = _compute_rankings("test_huber")
    rankings_rmse = _compute_rankings("test_rmse")

    # Print tables for both metrics
    for metric_name, rankings in [("Huber", rankings_huber), ("RMSE", rankings_rmse)]:
        print(f"\n{'=' * 100}")
        print(f"Models ranked by held-out {metric_name} (n_train={REFERENCE_N})")
        print(f"{'=' * 100}")
        header = (
            f"{'Model':<22} {metric_name + '(test)':>14} {metric_name + '(train)':>14}"
            f" {'Overfit':>8} {'Opt p1':>14} {'Conv%':>6}"
        )
        print(header)
        print("-" * 100)
        for name, m_mean, m_std, _, m_train, overfit, opt_mean, opt_std, conv in rankings:
            cn = clean_name(name)
            print(
                f"{cn:<22} {m_mean:>7.5f}+-{m_std:<5.5f} {m_train:>12.5f}"
                f" {overfit:>8.2f}x {opt_mean:>6.3f}+-{opt_std:<5.3f} {conv:>5.0f}%"
            )

    top_models_huber = [r[0] for r in rankings_huber[:5]]
    top_models_rmse = [r[0] for r in rankings_rmse[:5]]
    print(f"\nTop 5 by Huber: {[clean_name(n) for n in top_models_huber]}")
    print(f"Top 5 by RMSE:  {[clean_name(n) for n in top_models_rmse]}")


# =========================================================================
# Helpers for dual-metric plotting
# =========================================================================
def _learning_curve_stats(name, metric_key="test_huber"):
    """Extract median + IQR learning curve for a given test metric."""
    medians, q25, q75 = [], [], []
    for n_train in TRAIN_SIZES:
        d = all_results[n_train][name]
        mask = d["success"]
        vals = d[metric_key][mask]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            medians.append(np.nan)
            q25.append(np.nan)
            q75.append(np.nan)
        else:
            medians.append(float(np.median(vals)))
            q25.append(float(np.percentile(vals, 25)))
            q75.append(float(np.percentile(vals, 75)))
    return medians, q25, q75


def _plot_fig1(metric_id, rankings, top_models, suffix):
    """Figure 1: Learning curves overview (all + top 5)."""
    mi = METRICS[metric_id]
    test_key, label = mi["test_key"], mi["label"]

    n_panels = 2 if len(model_names) > 5 else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    panels = []
    if n_panels == 2:
        panels.append((0, model_names, False, f"All {len(model_names)} models"))
        panels.append((1, top_models, True, f"Top {min(5, len(model_names))}"))
    else:
        panels.append((0, model_names, True, "All models"))

    for ax_idx, subset, show_ci, title in panels:
        ax = axes[ax_idx]
        for name in subset:
            medians, q25, q75 = _learning_curve_stats(name, test_key)
            color = model_colors[name]
            lw = 2.5 if show_ci else 1.2
            ax.plot(TRAIN_SIZES, medians, "o-", color=color, lw=lw, ms=4, label=clean_name(name))
            if show_ci:
                ax.fill_between(TRAIN_SIZES, q25, q75, color=color, alpha=0.15)
        ax.set_xlabel("$n_{\\mathrm{train}}$")
        if ax_idx == 0:
            ax.set_ylabel(f"Held-out {label}, median")
        ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Collect handles/labels from the left (all models) panel for a shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"Learning Curves: Held-out {label} vs Training Set Size", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(5, len(labels)),
        framealpha=0.9,
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.22 if len(labels) > 10 else 0.15)
    out = OUT_DIR / f"fig1_learning_curves_{suffix}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def _adaptive_ylim(y_maxes, factor=1.5):
    """Compute a shared y-limit that works for the majority of subplots.

    Models whose max y exceeds `factor` * (75th percentile of all maxes) are
    treated as outliers and get their own y-scale.  Returns (shared_ylim,
    outlier_set) where outlier_set contains indices of outlier subplots.
    """
    arr = np.array(y_maxes)
    q75 = float(np.percentile(arr, 75))
    threshold = factor * q75
    outliers = set(int(i) for i, v in enumerate(arr) if v > threshold)
    shared_ylim = float(np.max(arr[arr <= threshold])) if np.any(arr <= threshold) else float(np.max(arr))
    # Add 5% padding
    shared_ylim *= 1.05
    return shared_ylim, outliers


def _plot_fig1b(metric_id, rankings, suffix):
    """Figure 1b: Learning curves per-model (small multiples)."""
    mi = METRICS[metric_id]
    test_key, short = mi["test_key"], mi["short"]

    n_all = len(model_names)
    ncols = min(5, n_all)
    nrows = (n_all + ncols - 1) // ncols
    ranked_names = [r[0] for r in rankings]

    # Pre-compute stats and y-ranges per model
    stats_per_model = []
    y_maxes = []
    for name in ranked_names:
        medians, q25, q75 = _learning_curve_stats(name, test_key)
        stats_per_model.append((medians, q25, q75))
        finite_q75 = [v for v in q75 if np.isfinite(v)]
        y_maxes.append(max(finite_q75) if finite_q75 else 0.0)

    shared_ylim, outlier_idxs = _adaptive_ylim(y_maxes)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows), squeeze=False, sharex=True)
    for idx, name in enumerate(ranked_names):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        medians, q25, q75 = stats_per_model[idx]
        color = model_colors.get(name, "gray")
        ax.plot(TRAIN_SIZES, medians, "o-", color=color, lw=2, ms=4)
        ax.fill_between(TRAIN_SIZES, q25, q75, color=color, alpha=0.2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if idx in outlier_idxs:
            ax.set_title(clean_name(name) + r"  $\ast$ diff. scale", fontsize=9, color="firebrick")
            for spine in ax.spines.values():
                spine.set_edgecolor("firebrick")
                spine.set_linewidth(1.5)
        else:
            ax.set_ylim(0, shared_ylim)
            ax.set_title(clean_name(name), fontsize=10)
        if c == 0:
            ax.set_ylabel(f"{short} (median)")
        if r == nrows - 1:
            ax.set_xlabel("$n_{\\mathrm{train}}$")

    for idx in range(n_all, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Learning Curves per Model — {short} (median + IQR)", fontsize=13, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / f"fig1b_learning_curves_detail_{suffix}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def _plot_fig2(metric_id, rankings, suffix):
    """Figure 2: Overfitting gap (small multiples), train vs test."""
    mi = METRICS[metric_id]
    test_key, train_key, short = mi["test_key"], mi["train_key"], mi["short"]

    n_show = len(rankings)
    show_models = [r[0] for r in rankings]
    ncols = min(5, n_show)
    nrows = (n_show + ncols - 1) // ncols

    # Pre-compute stats and y-ranges per model
    all_stats = []
    y_maxes = []
    for name in show_models:
        train_meds, test_meds = [], []
        train_q25, train_q75, test_q25, test_q75 = [], [], [], []
        for n_train in TRAIN_SIZES:
            d = all_results[n_train][name]
            mask = d["success"]
            tr = d[train_key][mask]
            te = d[test_key][mask]
            tr, te = tr[np.isfinite(tr)], te[np.isfinite(te)]
            train_meds.append(float(np.median(tr)) if len(tr) else np.nan)
            test_meds.append(float(np.median(te)) if len(te) else np.nan)
            train_q25.append(float(np.percentile(tr, 25)) if len(tr) else np.nan)
            train_q75.append(float(np.percentile(tr, 75)) if len(tr) else np.nan)
            test_q25.append(float(np.percentile(te, 25)) if len(te) else np.nan)
            test_q75.append(float(np.percentile(te, 75)) if len(te) else np.nan)
        all_stats.append((train_meds, test_meds, train_q25, train_q75, test_q25, test_q75))
        all_q75 = [v for v in train_q75 + test_q75 if np.isfinite(v)]
        y_maxes.append(max(all_q75) if all_q75 else 0.0)

    shared_ylim, outlier_idxs = _adaptive_ylim(y_maxes)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows), squeeze=False, sharex=True)
    for idx, name in enumerate(show_models):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        train_meds, test_meds, train_q25, train_q75, test_q25, test_q75 = all_stats[idx]

        ax.plot(TRAIN_SIZES, train_meds, "s--", color="royalblue", lw=1.5, ms=3, label="Train" if idx == 0 else None)
        ax.fill_between(TRAIN_SIZES, train_q25, train_q75, color="royalblue", alpha=0.12)
        ax.plot(TRAIN_SIZES, test_meds, "o-", color="crimson", lw=1.5, ms=3, label="Test" if idx == 0 else None)
        ax.fill_between(TRAIN_SIZES, test_q25, test_q75, color="crimson", alpha=0.12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if idx in outlier_idxs:
            ax.set_title(clean_name(name) + r"  $\ast$ diff. scale", fontsize=9, color="firebrick")
            for spine in ax.spines.values():
                spine.set_edgecolor("firebrick")
                spine.set_linewidth(1.5)
        else:
            ax.set_ylim(0, shared_ylim)
            ax.set_title(clean_name(name), fontsize=10)
        if c == 0:
            ax.set_ylabel(f"{short} (median)")
        if r == nrows - 1:
            ax.set_xlabel("$n_{\\mathrm{train}}$")

    for idx in range(n_show, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    _leg = [
        Line2D([], [], color="royalblue", ls="--", marker="s", ms=4, lw=1.5, label="Train"),
        Line2D([], [], color="crimson", ls="-", marker="o", ms=4, lw=1.5, label="Test"),
    ]
    fig.legend(handles=_leg, loc="upper right", framealpha=0.9, fontsize=10)
    fig.suptitle(f"Overfitting Gap: Train vs Test {short}", fontsize=13, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / f"fig2_overfitting_gap_{suffix}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


if not quick_mode:
    # =====================================================================
    # Generate figures 1, 1b, 2 for both Huber and RMSE
    # =====================================================================
    for metric_id, rnk, top in [
        ("huber", rankings_huber, top_models_huber),
        ("rmse", rankings_rmse, top_models_rmse),
    ]:
        short = METRICS[metric_id]["short"]
        print(f"\nGenerating figures ({short})...")
        _plot_fig1(metric_id, rnk, top, metric_id)
        _plot_fig1b(metric_id, rnk, metric_id)
        _plot_fig2(metric_id, rnk, metric_id)


# =========================================================================
# Cache full-data fits so each model is fit exactly once (avoids redundant
# grid-search in PCEQ when regenerating multiple plots).
# =========================================================================
_full_fit_cache: dict[str, tuple] = {}


def _get_full_fit(m):
    """Return (pred_fn, params) for model *m* fit on all training data, cached."""
    if m.name not in _full_fit_cache:
        X_full = build_features(m.feature_kind, p0_sc, p1_sc)
        _full_fit_cache[m.name] = m.fit_fn(X_full, y)
    return _full_fit_cache[m.name]


def _refine_optimum(pred_fn, feature_kind, p0_init, p1_init):
    """Refine a grid-argmin optimum using L-BFGS-B on [0.001, 0.999]²."""

    def _obj(x):
        X = build_features(feature_kind, np.array([x[0]]), np.array([x[1]]))
        return float(pred_fn(X)[0])

    result = sp_minimize(
        _obj, [p0_init, p1_init], method="L-BFGS-B", bounds=[(0.001, 0.999), (0.001, 0.999)]
    )
    if result.success:
        return float(result.x[0]), float(result.x[1]), float(result.fun)
    return p0_init, p1_init, _obj([p0_init, p1_init])


if not quick_mode:
    # =====================================================================
    # Figure 3: Stability of predicted optimum + BPB sanity check
    # (uses Huber ranking for ordering)
    # =====================================================================
    print("\nGenerating Figure 3: Stability of predicted optimum...")

    ref_data = all_results[REFERENCE_N]
    ranked_names = [r[0] for r in rankings_huber]

    opt_data = []
    for name in ranked_names:
        d = ref_data[name]
        mask = d["success"]
        vals = d["opt_p1"][mask]
        vals = vals[np.isfinite(vals)]
        opt_data.append((name, vals, float(np.median(vals)) if len(vals) else np.nan))

    observed_best_idx = int(np.argmin(y))
    observed_best_p1 = float(p1_sc[observed_best_idx])
    observed_best_bpb = float(np.min(y))

    predicted_bpb_at_opt = {}
    for m in RUN_MODELS:
        median_p1 = next(med for n, _, med in opt_data if n == m.name)
        if not np.isfinite(median_p1):
            predicted_bpb_at_opt[m.name] = np.nan
            continue
        try:
            pred_fn, _ = _get_full_fit(m)
            Xs = make_slice_for_kind(m.feature_kind, sg)
            slice_pred = pred_fn(Xs)
            predicted_bpb_at_opt[m.name] = float(np.interp(median_p1, sg, slice_pred))
        except Exception:
            predicted_bpb_at_opt[m.name] = np.nan

    fig_width = max(8, 1.3 * len(opt_data))
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(fig_width, 8), gridspec_kw={"height_ratios": [3, 2]}, sharex=True)

    bp_data = [vals for _, vals, _ in opt_data]
    bp_labels = [clean_name(n) for n, _, _ in opt_data]
    bp_colors = [model_colors.get(n, "gray") for n, _, _ in opt_data]
    x_pos = np.arange(1, len(opt_data) + 1)

    bplot = ax3a.boxplot(
        bp_data,
        labels=bp_labels,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color="black", lw=1.5),
        flierprops=dict(marker="o", ms=3, alpha=0.5),
    )
    for patch, color in zip(bplot["boxes"], bp_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    ax3a.axhline(
        observed_best_p1, color="red", ls="--", lw=2, zorder=0, label=f"Observed best $p_1$={observed_best_p1:.3f}"
    )
    ax3a.set_ylabel("Predicted optimal $p_1^{\\mathrm{sc}}$")
    ax3a.set_title(f"Stability of Predicted Optimum ($n_{{\\mathrm{{train}}}}$={REFERENCE_N}, B={B})")
    ax3a.legend(loc="upper left", framealpha=0.9)

    bpb_vals = [predicted_bpb_at_opt.get(n, np.nan) for n, _, _ in opt_data]
    bar_colors = ["#2ca02c" if b <= observed_best_bpb else "#d62728" for b in bpb_vals]

    ax3b.bar(x_pos, bpb_vals, color=bar_colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax3b.axhline(
        observed_best_bpb, color="red", ls="--", lw=2, zorder=0, label=f"Observed best BPB={observed_best_bpb:.4f}"
    )
    ax3b.set_ylabel("Predicted BPB at median $p_1^*$")
    ax3b.set_title("Sanity Check: predicted BPB at median optimum (green $=$ improves on observed)")
    ax3b.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax3b.set_xlim(0.3, len(opt_data) + 0.7)

    finite_bpbs = [b for b in bpb_vals if np.isfinite(b)]
    if finite_bpbs:
        ylo = min(min(finite_bpbs), observed_best_bpb) - 0.005
        yhi = max(max(finite_bpbs), observed_best_bpb) + 0.005
        ax3b.set_ylim(ylo, yhi)

    for i, bpb in enumerate(bpb_vals):
        if np.isfinite(bpb):
            ax3b.text(x_pos[i], bpb + 0.0005, f"{bpb:.4f}", ha="center", va="bottom", fontsize=7)

    plt.setp(ax3b.get_xticklabels(), rotation=30, ha="right")

    fig3.tight_layout()
    out3 = OUT_DIR / "fig3_stability.png"
    fig3.savefig(out3)
    plt.close(fig3)
    print(f"  Saved {out3}")

    # =================================================================
    # Figure 3b: Stability + Predicted vs Actual BPB at 2D optima
    # =================================================================
    _optima_csv_3b = script_dir / "two_phase_starcoder_4.csv"
    if _optima_csv_3b.exists():
        print("\nGenerating Figure 3b: Stability + predicted vs actual BPB...")
        _odf = pd.read_csv(_optima_csv_3b)
        _oruns = _odf[_odf["run_id"] >= 90016]
        _bpb_col = "eval/paloma/dolma_100_programing_languages/bpb"
        _actuals_3b: list[tuple[float, float, float]] = []
        for _, _r in _oruns.iterrows():
            if _bpb_col in _r and pd.notna(_r[_bpb_col]):
                _actuals_3b.append(
                    (
                        float(_r["phase_0_starcoder"]),
                        float(_r["phase_1_starcoder"]),
                        float(_r[_bpb_col]),
                    )
                )

        # Compute 2D predicted optimum + match to actual for each model
        _model_info_3b: list[dict] = []
        for m in RUN_MODELS:
            info: dict = {"name": m.name, "model": m}
            try:
                pred_fn, _ = _get_full_fit(m)
                _g3b = np.linspace(0.005, 0.995, HEATMAP_RES)
                X_grid, P0, P1 = make_2d_grid(_g3b, _g3b, kind=m.feature_kind)
                Z = pred_fn(X_grid).reshape(P0.shape)
                idx = int(np.argmin(Z))
                info["opt_p0"] = float(P0.ravel()[idx])
                info["opt_p1"] = float(P1.ravel()[idx])
                info["pred_bpb"] = float(Z.ravel()[idx])

                # Match to actual
                info["actual_bpb"] = None
                for _ap0, _ap1, _abpb in _actuals_3b:
                    if abs(_ap0 - info["opt_p0"]) < 0.02 and abs(_ap1 - info["opt_p1"]) < 0.02:
                        info["actual_bpb"] = _abpb
                        break
            except Exception:
                info["opt_p0"] = np.nan
                info["opt_p1"] = np.nan
                info["pred_bpb"] = np.nan
                info["actual_bpb"] = None

            # Get bootstrap p1 distribution from ref_data
            d = ref_data.get(m.name)
            if d is not None:
                mask = d["success"]
                vals = d["opt_p1"][mask]
                info["p1_vals"] = vals[np.isfinite(vals)]
            else:
                info["p1_vals"] = np.array([])

            _model_info_3b.append(info)

        # Sort by ascending actual BPB (None last)
        _model_info_3b.sort(
            key=lambda t: (
                t["actual_bpb"] is None,
                t["actual_bpb"] if t["actual_bpb"] is not None else 999.0,
            )
        )

        n3b = len(_model_info_3b)
        fig_w_3b = max(8, 1.3 * n3b)
        fig3b, (ax3b_top, ax3b_bot) = plt.subplots(
            2,
            1,
            figsize=(fig_w_3b, 9),
            gridspec_kw={"height_ratios": [2, 3]},
            sharex=True,
        )
        x3b = np.arange(1, n3b + 1)
        labels_3b = [clean_name(t["name"]) for t in _model_info_3b]

        # --- Top panel: box plots of bootstrap p1, sorted by actual BPB ---
        bp_data_3b = [t["p1_vals"] for t in _model_info_3b]
        bp_colors_3b = [model_colors.get(t["name"], "gray") for t in _model_info_3b]
        bplot_3b = ax3b_top.boxplot(
            bp_data_3b,
            labels=labels_3b,
            patch_artist=True,
            widths=0.6,
            medianprops=dict(color="black", lw=1.5),
            flierprops=dict(marker="o", ms=3, alpha=0.5),
        )
        for patch, color in zip(bplot_3b["boxes"], bp_colors_3b):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)

        ax3b_top.axhline(
            observed_best_p1,
            color="red",
            ls="--",
            lw=2,
            zorder=0,
            label=f"Observed best $p_1$={observed_best_p1:.3f}",
        )
        ax3b_top.set_ylabel("Predicted optimal $p_1^{\\mathrm{sc}}$")
        ax3b_top.set_title(f"Stability of Predicted Optimum ($n_{{\\mathrm{{train}}}}$={REFERENCE_N}, B={B})")
        ax3b_top.legend(loc="upper left", framealpha=0.9, fontsize=8)

        # --- Bottom panel: predicted vs actual BPB ---
        # Bars for actual BPB; diamond markers for predicted BPB.
        # Bar color encodes two things via a two-step scheme:
        #   - Hue: green if actual < best_observed, red if worse
        #   - Saturation/alpha modulated by |actual - predicted|
        # Connecting line shows the prediction gap.
        pred_vals = np.array([t["pred_bpb"] if t["pred_bpb"] is not None else np.nan for t in _model_info_3b])
        actual_vals = np.array([t["actual_bpb"] if t["actual_bpb"] is not None else np.nan for t in _model_info_3b])

        bar_width = 0.35
        # Actual bars (solid)
        for i in range(n3b):
            a_bpb = actual_vals[i]
            p_bpb = pred_vals[i]
            if not np.isfinite(a_bpb):
                continue

            # Base color: green if actual < best observed, red if worse
            if a_bpb < observed_best_bpb:
                base_color = "#2ca02c"  # green — found better!
            else:
                base_color = "#d62728"  # red — worse than best observed

            ax3b_bot.bar(
                x3b[i] - bar_width / 2,
                a_bpb,
                bar_width,
                color=base_color,
                alpha=0.75,
                edgecolor="black",
                linewidth=0.5,
                label="Actual" if i == 0 else None,
            )

            # Predicted bars (hatched, lighter)
            if np.isfinite(p_bpb):
                ax3b_bot.bar(
                    x3b[i] + bar_width / 2,
                    p_bpb,
                    bar_width,
                    color="steelblue",
                    alpha=0.45,
                    edgecolor="black",
                    linewidth=0.5,
                    hatch="//",
                    label="Predicted" if i == 0 else None,
                )

                # Connecting line between bar tops
                ax3b_bot.plot(
                    [x3b[i] - bar_width / 2, x3b[i] + bar_width / 2],
                    [a_bpb, p_bpb],
                    color="gray",
                    lw=0.8,
                    ls="-",
                    alpha=0.6,
                )

                # Delta annotation
                delta = a_bpb - p_bpb
                ann_y = max(a_bpb, p_bpb) + 0.001
                ax3b_bot.text(
                    x3b[i],
                    ann_y,
                    f"{delta:+.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="green" if delta < 0 else "red",
                    fontweight="bold",
                )

        ax3b_bot.axhline(
            observed_best_bpb,
            color="red",
            ls="--",
            lw=2,
            zorder=0,
            label=f"Best observed={observed_best_bpb:.4f}",
        )

        # Y-axis limits from the data
        all_bpb_3b = np.concatenate(
            [
                pred_vals[np.isfinite(pred_vals)],
                actual_vals[np.isfinite(actual_vals)],
            ]
        )
        if len(all_bpb_3b):
            ylo_3b = min(float(np.min(all_bpb_3b)), observed_best_bpb) - 0.008
            yhi_3b = max(float(np.max(all_bpb_3b)), observed_best_bpb) + 0.008
            ax3b_bot.set_ylim(ylo_3b, yhi_3b)

        ax3b_bot.set_ylabel("BPB at predicted optimum")
        ax3b_bot.set_title(
            "Predicted vs Actual BPB "
            "(green = actual beats best observed, "
            "red = actual worse, $\\Delta$ = actual $-$ predicted)"
        )
        ax3b_bot.legend(loc="upper right", framealpha=0.9, fontsize=8)
        ax3b_bot.set_xlim(0.3, n3b + 0.7)
        plt.setp(ax3b_bot.get_xticklabels(), rotation=30, ha="right")

        fig3b.tight_layout()
        out3b = OUT_DIR / "fig3b_stability_predicted_vs_actual.png"
        fig3b.savefig(out3b)
        plt.close(fig3b)
        print(f"  Saved {out3b}")
    else:
        print(f"\nSkipping Figure 3b: {_optima_csv_3b} not found (run --analyze-local first)")


# =========================================================================
# Cross-validation metrics table (runs in both quick and full mode)
# =========================================================================
print("\n" + "=" * 90)
print("Cross-Validation Metrics (5-fold CV on full data)")
print("=" * 90)
print(f"{'Model':<25} {'R2':>7} {'RMSE':>7} {'Spearman':>9} {'RMSE_bot':>9}")
print("-" * 90)

cv_results_dict = {}
for m in RUN_MODELS:
    X_full = build_features(m.feature_kind, p0_sc, p1_sc)
    cv_res = cv_metrics(m.fit_fn, X_full, y, n_folds=5, seed=42)
    cv_results_dict[m.name] = cv_res
    print(
        f"{clean_name(m.name):<25} {cv_res['R2']:>7.4f} {cv_res['RMSE']:>7.4f} "
        f"{cv_res['Spearman']:>9.4f} {cv_res['RMSE_bot']:>9.4f}"
    )

print("=" * 90)

# In quick mode, derive rankings from CV metrics (for fig 4 ordering and fig 5 top models)
if quick_mode:
    _cv_ranked = sorted(cv_results_dict.items(), key=lambda x: -x[1]["R2"])
    rankings_huber = [(name, cv["RMSE"], 0.0, cv["RMSE"], cv["RMSE"], 1.0, 0.0, 0.0, 100.0) for name, cv in _cv_ranked]
    top_models_huber = [name for name, _ in _cv_ranked[:5]]
    print(f"\nTop 5 by CV R²: {[clean_name(n) for n in top_models_huber]}")


# =========================================================================
# Figure 4: 2D heatmaps of predicted BPB with isoloss contours
# =========================================================================
print("\nGenerating Figure 4: 2D heatmaps...")

g = np.linspace(0.005, 0.995, HEATMAP_RES)

ranked_model_info = []
rank_order = {r[0]: i for i, r in enumerate(rankings_huber)}
for m in RUN_MODELS:
    ranked_model_info.append((rank_order.get(m.name, 999), m))
ranked_model_info.sort()

n_models = len(ranked_model_info)
ncols4 = min(4, n_models)
nrows4 = (n_models + ncols4 - 1) // ncols4

cell_size = 4.0
fig4, axes4 = plt.subplots(
    nrows4,
    ncols4,
    figsize=(cell_size * ncols4 + 1.8, cell_size * nrows4),
    squeeze=False,
    constrained_layout=True,
)

vmin_global = float(np.min(y)) - 0.05
vmax_global = float(np.percentile(y, 90))
n_contour_levels = 15

last_scatter = None
for mi, (_, m) in enumerate(ranked_model_info):
    row, col = divmod(mi, ncols4)
    ax = axes4[row][col]

    n_params = None
    try:
        pred_fn, params = _get_full_fit(m)
        n_params = len(params)
        X_grid, P0, P1 = make_2d_grid(g, g, kind=m.feature_kind)
        Z = pred_fn(X_grid).reshape(P0.shape)
        Z_clipped = np.clip(Z, vmin_global, vmax_global + 0.5)

        ax.pcolormesh(
            P0,
            P1,
            Z_clipped,
            cmap="RdYlGn_r",
            vmin=vmin_global,
            vmax=vmax_global,
            shading="gouraud",
        )
        cs = ax.contour(P0, P1, Z_clipped, levels=n_contour_levels, colors="black", linewidths=0.4, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")

        opt_idx = int(np.argmin(Z))
        opt_p0 = float(P0.ravel()[opt_idx])
        opt_p1_val = float(P1.ravel()[opt_idx])
        opt_bpb = float(Z.ravel()[opt_idx])
        ax.plot(
            opt_p0, opt_p1_val, marker="*", ms=14, color="gold", markeredgecolor="black", markeredgewidth=1.2, zorder=6
        )

        ax.annotate(
            f"({opt_p0:.2f}, {opt_p1_val:.2f})\nBPB={opt_bpb:.3f}",
            xy=(opt_p0, opt_p1_val),
            xytext=(0.97, 0.03),
            textcoords="axes fraction",
            fontsize=7,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            zorder=7,
        )

    except Exception as e:
        ax.text(0.5, 0.5, f"Fit failed:\n{e}", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="red")

    last_scatter = ax.scatter(
        p0_sc,
        p1_sc,
        c=y,
        cmap="RdYlGn_r",
        vmin=vmin_global,
        vmax=vmax_global,
        s=40,
        marker="o",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
    )

    title = clean_name(m.name)
    if n_params is not None:
        title += f" ({n_params}p)"
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if row == nrows4 - 1:
        ax.set_xlabel("Phase 0 StarCoder weight")
    if col == 0:
        ax.set_ylabel("Phase 1 StarCoder weight")

for idx in range(n_models, nrows4 * ncols4):
    row, col = divmod(idx, ncols4)
    axes4[row][col].set_visible(False)

if last_scatter is not None:
    fig4.colorbar(last_scatter, ax=axes4.ravel().tolist(), label="BPB", shrink=0.7, pad=0.03, aspect=30)

fig4.suptitle(
    "Predicted BPB over Weight Space (full-data fit, $\\bigstar$ = predicted optimum)",
    fontsize=13,
)
out4 = OUT_DIR / "fig4_heatmaps.png"
fig4.savefig(out4)
plt.close(fig4)
print(f"  Saved {out4}")


# =========================================================================
# Figure 4b: Heatmaps with predicted vs actual BPB at predicted optima
# =========================================================================
# Load actual BPB for predicted-optima baseline runs from the v4 analysis CSV.
_optima_csv = script_dir / "two_phase_starcoder_4.csv"
if _optima_csv.exists():
    print("\nGenerating Figure 4b: Heatmaps with predicted vs actual BPB...")
    _optima_df = pd.read_csv(_optima_csv)
    # Build lookup: (p0_sc, p1_sc) -> actual BPB (for optima runs, run_id >= 90016)
    _optima_runs = _optima_df[_optima_df["run_id"] >= 90016].copy()
    _actuals_lookup: list[tuple[float, float, float]] = []
    for _, _row in _optima_runs.iterrows():
        _bpb_col = "eval/paloma/dolma_100_programing_languages/bpb"
        if _bpb_col in _row and pd.notna(_row[_bpb_col]):
            _actuals_lookup.append(
                (float(_row["phase_0_starcoder"]), float(_row["phase_1_starcoder"]), float(_row[_bpb_col]))
            )

    # First pass: compute predicted optima and match to actuals for sorting
    _model_optima: list[tuple[str, object, float, float, float, float | None]] = []
    for _, m in ranked_model_info:
        try:
            pred_fn, _ = _get_full_fit(m)
            X_grid, P0, P1 = make_2d_grid(g, g, kind=m.feature_kind)
            Z = pred_fn(X_grid).reshape(P0.shape)
            idx = int(np.argmin(Z))
            opt_p0 = float(P0.ravel()[idx])
            opt_p1_val = float(P1.ravel()[idx])
            opt_bpb = float(Z.ravel()[idx])

            # Find closest actual within L1 < 0.02
            best_dist, actual_bpb = 0.02, None
            for _ap0, _ap1, _abpb in _actuals_lookup:
                dist = abs(_ap0 - opt_p0) + abs(_ap1 - opt_p1_val)
                if dist < best_dist:
                    best_dist = dist
                    actual_bpb = _abpb
            _model_optima.append((m.name, m, opt_p0, opt_p1_val, opt_bpb, actual_bpb))
        except Exception:
            _model_optima.append((m.name, m, 0.0, 0.0, 0.0, None))

    # Sort by ascending actual BPB (models without actuals go last)
    _model_optima.sort(key=lambda t: (t[5] is None, t[5] if t[5] is not None else 999.0))

    n_models_4b = len(_model_optima)
    ncols4b = min(4, n_models_4b)
    nrows4b = (n_models_4b + ncols4b - 1) // ncols4b

    fig4b, axes4b = plt.subplots(
        nrows4b,
        ncols4b,
        figsize=(cell_size * ncols4b + 1.8, cell_size * nrows4b),
        squeeze=False,
        constrained_layout=True,
    )

    last_scatter_4b = None
    for mi, (mname, m, opt_p0, opt_p1_val, opt_bpb, actual_bpb) in enumerate(_model_optima):
        row, col = divmod(mi, ncols4b)
        ax = axes4b[row][col]

        n_params_4b = None
        try:
            pred_fn, params_4b = _get_full_fit(m)
            n_params_4b = len(params_4b)
            X_grid, P0, P1 = make_2d_grid(g, g, kind=m.feature_kind)
            Z = pred_fn(X_grid).reshape(P0.shape)
            Z_clipped = np.clip(Z, vmin_global, vmax_global + 0.5)

            ax.pcolormesh(
                P0,
                P1,
                Z_clipped,
                cmap="RdYlGn_r",
                vmin=vmin_global,
                vmax=vmax_global,
                shading="gouraud",
            )
            cs = ax.contour(P0, P1, Z_clipped, levels=n_contour_levels, colors="black", linewidths=0.4, alpha=0.6)
            ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")

            # Star colored by actual BPB on the same colormap as the heatmap
            if actual_bpb is not None:

                _cmap = plt.get_cmap("RdYlGn_r")
                _norm = mcolors.Normalize(vmin=vmin_global, vmax=vmax_global)
                star_color = _cmap(_norm(actual_bpb))
                label_text = f"({opt_p0:.2f}, {opt_p1_val:.2f})\nPred={opt_bpb:.3f}\nActual={actual_bpb:.3f}"
            else:
                star_color = "gray"
                label_text = f"({opt_p0:.2f}, {opt_p1_val:.2f})\nPred={opt_bpb:.3f}\nActual=N/A"

            ax.plot(
                opt_p0,
                opt_p1_val,
                marker="*",
                ms=14,
                color=star_color,
                markeredgecolor="black",
                markeredgewidth=1.2,
                zorder=6,
            )

            ax.annotate(
                label_text,
                xy=(opt_p0, opt_p1_val),
                xytext=(0.97, 0.03),
                textcoords="axes fraction",
                fontsize=6.5,
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                zorder=7,
            )

        except Exception as e:
            ax.text(
                0.5, 0.5, f"Fit failed:\n{e}", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="red"
            )

        last_scatter_4b = ax.scatter(
            p0_sc,
            p1_sc,
            c=y,
            cmap="RdYlGn_r",
            vmin=vmin_global,
            vmax=vmax_global,
            s=40,
            marker="o",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )

        title_4b = clean_name(mname)
        if n_params_4b is not None:
            title_4b += f" ({n_params_4b}p)"
        ax.set_title(title_4b, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if row == nrows4b - 1:
            ax.set_xlabel("Phase 0 StarCoder weight")
        if col == 0:
            ax.set_ylabel("Phase 1 StarCoder weight")

    for idx in range(n_models_4b, nrows4b * ncols4b):
        row, col = divmod(idx, ncols4b)
        axes4b[row][col].set_visible(False)

    if last_scatter_4b is not None:
        fig4b.colorbar(last_scatter_4b, ax=axes4b.ravel().tolist(), label="BPB", shrink=0.7, pad=0.03, aspect=30)

    fig4b.suptitle(
        "Predicted vs Actual BPB at Predicted Optima ($\\bigstar$ colored by actual BPB, sorted by ascending actual)",
        fontsize=12,
    )
    out4b = OUT_DIR / "fig4b_heatmaps_predicted_vs_actual.png"
    fig4b.savefig(out4b)
    plt.close(fig4b)
    print(f"  Saved {out4b}")
else:
    print(f"\nSkipping Figure 4b: {_optima_csv} not found (run --analyze-local first)")


# =========================================================================
# Figure 6: Optimum trajectory over training sizes
# Shows how each model's predicted 2D optimum drifts as training data grows.
# 8 points from TRAIN_SIZES (one seed each) + 1 full-data point = 9 total.
# =========================================================================
print("\nGenerating Figure 6: Optimum trajectory over training sizes...")

_cmap6 = plt.get_cmap("RdYlGn_r")
_norm6 = mcolors.Normalize(vmin=vmin_global, vmax=vmax_global)

n_models_6 = len(ranked_model_info)
ncols6 = min(4, n_models_6)
nrows6 = (n_models_6 + ncols6 - 1) // ncols6

fig6, axes6 = plt.subplots(
    nrows6,
    ncols6,
    figsize=(cell_size * ncols6 + 1.8, cell_size * nrows6),
    squeeze=False,
    constrained_layout=True,
)

last_scatter_6 = None
for mi, (_, m) in enumerate(ranked_model_info):
    row, col = divmod(mi, ncols6)
    ax = axes6[row][col]

    # Compute optima at each training size (1 seed per size) + full data
    optima: list[tuple[float, float, float]] = []

    for si, n_train in enumerate(TRAIN_SIZES):
        seed = SEED_BASE + si * B
        rng = np.random.RandomState(seed)
        idx = rng.permutation(N)
        train_idx = idx[:n_train]

        try:
            X_train = build_features(m.feature_kind, p0_sc[train_idx], p1_sc[train_idx])
            y_train = y[train_idx]
            pred_fn_sub, _ = m.fit_fn(X_train, y_train)
            X_grid, P0, P1 = make_2d_grid(g, g, kind=m.feature_kind)
            Z_sub = pred_fn_sub(X_grid).reshape(P0.shape)
            grid_idx = int(np.argmin(Z_sub))
            p0_init = float(P0.ravel()[grid_idx])
            p1_init = float(P1.ravel()[grid_idx])
            optima.append(_refine_optimum(pred_fn_sub, m.feature_kind, p0_init, p1_init))
        except Exception:
            optima.append((np.nan, np.nan, np.nan))

    # Full-data optimum
    try:
        pred_fn_full, _ = _get_full_fit(m)
        X_grid, P0, P1 = make_2d_grid(g, g, kind=m.feature_kind)
        Z_full = pred_fn_full(X_grid).reshape(P0.shape)
        Z_clipped = np.clip(Z_full, vmin_global, vmax_global + 0.5)
        grid_idx = int(np.argmin(Z_full))
        p0_init = float(P0.ravel()[grid_idx])
        p1_init = float(P1.ravel()[grid_idx])
        optima.append(_refine_optimum(pred_fn_full, m.feature_kind, p0_init, p1_init))

        # Render heatmap + contours from full-data fit
        ax.pcolormesh(P0, P1, Z_clipped, cmap="RdYlGn_r", vmin=vmin_global, vmax=vmax_global, shading="gouraud")
        cs = ax.contour(P0, P1, Z_clipped, levels=n_contour_levels, colors="black", linewidths=0.4, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
    except Exception as e:
        ax.text(0.5, 0.5, f"Fit failed:\n{e}", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="red")
        optima.append((np.nan, np.nan, np.nan))

    # Training data scatter with transparent edges
    last_scatter_6 = ax.scatter(
        p0_sc, p1_sc, c=y, cmap="RdYlGn_r", vmin=vmin_global, vmax=vmax_global,
        s=30, marker="o", edgecolors=(0, 0, 0, 0.25), linewidths=0.6, zorder=3,
    )

    # Draw arrows between consecutive valid optima
    for i in range(len(optima) - 1):
        if np.isnan(optima[i][0]) or np.isnan(optima[i + 1][0]):
            continue
        ax.annotate(
            "", xy=(optima[i + 1][0], optima[i + 1][1]),
            xytext=(optima[i][0], optima[i][1]),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2, alpha=0.7),
            zorder=7,
        )

    # Draw numbered points colored by predicted BPB
    for i, (p0_opt, p1_opt, bpb_opt) in enumerate(optima):
        if np.isnan(p0_opt):
            continue
        color = _cmap6(_norm6(bpb_opt))
        ax.plot(p0_opt, p1_opt, "o", ms=12, color=color, markeredgecolor="black", markeredgewidth=0.8, zorder=8)
        label = str(i + 1) if i < len(TRAIN_SIZES) else "F"
        ax.text(p0_opt, p1_opt, label, ha="center", va="center", fontsize=6, fontweight="bold", color="white", zorder=9)

    # Star on final (full-data) optimum
    if not np.isnan(optima[-1][0]):
        ax.plot(
            optima[-1][0], optima[-1][1], marker="*", ms=16,
            color="gold", markeredgecolor="black", markeredgewidth=1.2, zorder=10,
        )

    n_params_6 = None
    try:
        _, params_6 = _get_full_fit(m)
        n_params_6 = len(params_6)
    except Exception:
        pass
    title_6 = clean_name(m.name)
    if n_params_6 is not None:
        title_6 += f" ({n_params_6}p)"
    ax.set_title(title_6, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if row == nrows6 - 1:
        ax.set_xlabel("Phase 0 StarCoder weight")
    if col == 0:
        ax.set_ylabel("Phase 1 StarCoder weight")

for idx in range(n_models_6, nrows6 * ncols6):
    row, col = divmod(idx, ncols6)
    axes6[row][col].set_visible(False)

if last_scatter_6 is not None:
    fig6.colorbar(last_scatter_6, ax=axes6.ravel().tolist(), label="BPB", shrink=0.7, pad=0.03, aspect=30)

# Build TRAIN_SIZES legend string
_ts_labels = ", ".join(f"{i + 1}={n}" for i, n in enumerate(TRAIN_SIZES))
fig6.suptitle(
    f"Predicted Optimum Trajectory over Training Size ({_ts_labels}, F=full $n$={N})",
    fontsize=11,
)
out6 = OUT_DIR / "fig6_optimum_trajectory.png"
fig6.savefig(out6)
plt.close(fig6)
print(f"  Saved {out6}")


# =========================================================================
# Figure 5: 3-panel slice predictions (from debug_epoch_features_v3)
# =========================================================================
print("\nGenerating Figure 5: 3-panel slice predictions...")

# Fit all models on full data and get slice predictions
fitted_models = []
for m in RUN_MODELS:
    pred_fn, params = _get_full_fit(m)

    # Build slice features
    Xs = make_slice_for_kind(m.feature_kind, slice_grid)
    preds_slice = pred_fn(Xs)
    label = m.label_fn(params)

    best_i = np.argmin(preds_slice)
    fitted_models.append((m.name, pred_fn, params, label, preds_slice, m.color, m.linestyle, cv_results_dict[m.name]))

# Extract actual data on the p0=0 slice (100% Nemotron in phase 0)
mask = df["phase_0_nemotron_full"].round(4) == 1.0
df_slice = df[mask].sort_values("phase_1_starcoder")
x_actual = df_slice["phase_1_starcoder"].values
y_actual = df_slice[TARGET].values
actual_best_i = np.argmin(y_actual)

# Create 3-panel plot
fig5, axes5 = plt.subplots(1, 3, figsize=(36, 9))

XLABEL = r"$p$ = StarCoder fraction in Phase 1"
YLABEL = r"BPB (eval/paloma/dolma\_100\_programing\_languages)"
NOTATION_LINE = (
    r"\small $p = p_1^{\mathrm{sc}}$,\; $L = \ln(\mathrm{sc\_epochs}_1)$," r"\; (w{-}e) = weight{-}epoch features"
)
TOP_MODELS_FIG5 = set(top_models_huber)

for panel, (ax, xlim, ylim, title) in enumerate(
    zip(
        axes5,
        [(0, 1), (0.1, 0.55), (0.1, 0.55)],
        [(0.85, 1.75), (0.88, 0.97), (0.88, 0.97)],
        ["Full range", "Zoomed: all models", "Zoomed: top 5"],
    )
):
    ax.scatter(x_actual, y_actual, s=50, c="black", zorder=10, label="Actual data")

    for name, _, params, label, preds_s, color, ls, cv_m in fitted_models:
        if panel == 2 and name not in TOP_MODELS_FIG5:
            continue
        if panel == 0:
            lbl = clean_name(name)
        elif panel == 2:
            lbl = rf"{clean_name(name)}: {label}"
        else:
            lbl = label
        ax.plot(slice_grid, preds_s, label=lbl, linewidth=2.0, color=color, linestyle=ls)

    ax.set_xlabel(XLABEL, fontsize=14)
    ax.set_ylabel(YLABEL, fontsize=14)
    ax.set_title(rf"Slice: 100\% Nemotron in Phase 0 --- {title}", fontsize=15)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=12)

    # Secondary x-axis showing epoch scale
    secax = ax.secondary_xaxis("top", functions=(lambda w: w * SC_EPOCH_MULT, lambda e: e / SC_EPOCH_MULT))
    secax.set_xlabel(r"StarCoder epochs in Phase 1", fontsize=12)
    secax.tick_params(labelsize=10)

    if panel == 0:
        ax.legend(fontsize=9, loc="upper right", ncol=2)
    elif panel == 2:
        ax.plot([], [], " ", label=NOTATION_LINE)
        ax.legend(fontsize=11, loc="upper left", framealpha=0.95)

fig5.tight_layout()
out5 = OUT_DIR / "fig5_slice_predictions.png"
fig5.savefig(out5, dpi=300, bbox_inches="tight")
plt.close(fig5)
print(f"  Saved {out5}")


print(f"\nAll outputs saved to {OUT_DIR}/")
print("Done.")
