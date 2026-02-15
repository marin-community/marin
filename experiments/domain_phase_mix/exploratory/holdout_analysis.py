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

import pickle
import sys
import time
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from joblib import Parallel, delayed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from debug_epoch_features_v3 import (  # noqa: E402
    EPS,
    MODELS,
    SC_EPOCH_MULT,
    X_weight,
    feature_kind,
    fit_linear,
    make_2d_grid,
    make_slice_vdom,
    make_slice_weight,
    make_slice_wt_epoch,
    p0_sc,
    p1_sc,
    slice_grid,
    y,
)

sys.stdout.reconfigure(line_buffering=True)

script_dir = Path(__file__).parent
OUT_DIR = script_dir / "holdout_plots"
OUT_DIR.mkdir(exist_ok=True)
CACHE_FILE = OUT_DIR / "bootstrap_cache.pkl"

# -------------------------------------------------------------------------
# Matplotlib style
# -------------------------------------------------------------------------
mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": (
        r"\usepackage{opensans}\renewcommand{\familydefault}{\sfdefault}"
        r"\usepackage{amssymb}"
    ),
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
})


def clean_name(name: str) -> str:
    """Strip LaTeX braces from model names for display."""
    return name.replace("{-}", "-").replace("{", "").replace("}", "")


# =========================================================================
# Configuration
# =========================================================================
TRAIN_SIZES = [12, 18, 24, 30, 36, 42, 48, 54]
B = 200  # bootstrap iterations per training size
REFERENCE_N = 48  # training size for Table 1 and Figure 3
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
# CLI: model selection + --plots-only flag
# =========================================================================
plots_only = "--plots-only" in sys.argv
cli_filters = [a for a in sys.argv[1:] if not a.startswith("--")]

if cli_filters:
    selected = []
    for name, fit_fn, X_data, label_fn, color, ls in MODELS:
        if any(f.lower() in name.lower() for f in cli_filters):
            selected.append((name, fit_fn, X_data, label_fn, color, ls))
    if not selected:
        print(f"No models match filters: {cli_filters}")
        print("Available:", [n for n, *_ in MODELS])
        sys.exit(1)
    RUN_MODELS = selected
else:
    RUN_MODELS = list(MODELS)

print(f"Models to evaluate ({len(RUN_MODELS)}):")
for name, *_ in RUN_MODELS:
    print(f"  {clean_name(name)}")


# =========================================================================
# Huber loss
# =========================================================================
def huber_loss(y_true, y_pred, delta):
    """Element-wise Huber loss, averaged."""
    r = y_true - y_pred
    abs_r = np.abs(r)
    return float(np.mean(np.where(abs_r <= delta, 0.5 * r**2, delta * (abs_r - 0.5 * delta))))


# Compute reference delta from full-data Linear model residuals
_linear_pred, _ = fit_linear(X_weight, y)
_linear_resid = y - _linear_pred(X_weight)
_mad = np.median(np.abs(_linear_resid - np.median(_linear_resid)))
HUBER_DELTA = 1.345 * _mad / 0.6745
print(f"\nHuber delta = {HUBER_DELTA:.6f} (from Linear MAD={_mad:.6f})")


# =========================================================================
# Feature-type dispatch helpers
# =========================================================================
FEATURE_KINDS = {}
for _name, _fit_fn, _X_data, *_rest in MODELS:
    FEATURE_KINDS[_name] = feature_kind(_X_data)


def _make_slice_for_kind(kind, g):
    if kind == "weight":
        return make_slice_weight(g)
    elif kind == "wt_epoch":
        return make_slice_wt_epoch(g)
    return make_slice_vdom(g)


def _build_features(kind, p0_vals, p1_vals):
    """Build feature matrix from raw proportions for the given feature kind."""
    if kind == "weight":
        return np.column_stack([p0_vals, p1_vals])
    elif kind == "wt_epoch":
        return np.column_stack([
            p0_vals, p1_vals,
            np.log(SC_EPOCH_MULT * p0_vals + EPS),
            np.log(SC_EPOCH_MULT * p1_vals + EPS),
        ])
    else:
        return np.column_stack([
            0.5 * (1 - p0_vals), 0.5 * p0_vals,
            0.5 * (1 - p1_vals), 0.5 * p1_vals,
        ])


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

    fail = {"test_huber": np.nan, "test_rmse": np.nan, "train_rmse": np.nan,
            "opt_p1": np.nan, "success": False}

    results = {}
    for name, fit_fn, kind in model_specs:
        X_train = _build_features(kind, p0_train, p1_train)
        X_test = _build_features(kind, p0_test, p1_test)

        try:
            pred_fn, _ = fit_fn(X_train, y_train)
            test_pred = pred_fn(X_test)
            train_pred = pred_fn(X_train)

            # Sanity check: reject divergent predictions (exp overflow, etc.)
            if (np.any(np.abs(test_pred) > max_pred)
                    or np.any(np.isnan(test_pred))
                    or np.any(np.abs(train_pred) > max_pred)):
                results[name] = fail
                continue

            Xs = _make_slice_for_kind(kind, sg)
            slice_pred = pred_fn(Xs)
            valid_mask = np.isfinite(slice_pred) & (np.abs(slice_pred) < max_pred)
            if valid_mask.any():
                opt_p1 = float(sg[valid_mask][np.argmin(slice_pred[valid_mask])])
            else:
                opt_p1 = np.nan

            results[name] = {
                "test_huber": huber_loss(y_test, test_pred, delta),
                "test_rmse": float(np.sqrt(np.mean((y_test - test_pred) ** 2))),
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
model_specs = [(name, fit_fn, FEATURE_KINDS[name]) for name, fit_fn, *_ in RUN_MODELS]
model_names = [name for name, *_ in RUN_MODELS]
model_colors = {name: color for name, _, _, _, color, _ in RUN_MODELS}

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
if needed_names and not plots_only:
    needed_specs = [(n, fn, FEATURE_KINDS[n])
                    for n, fn, *_ in RUN_MODELS if n in needed_names]

    print(f"Running bootstrap: {len(TRAIN_SIZES)} sizes x {B} iters x {len(needed_specs)} models")
    t0 = time.time()

    # Collect per-model results: {model_name: {n_train: {metrics}}}
    new_model_results = {name: {} for name in needed_names}

    for si, n_train in enumerate(TRAIN_SIZES):
        seeds = [SEED_BASE + si * B + b for b in range(B)]

        iter_results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_run_one_iter)(
                n_train, seed, needed_specs, p0_sc, p1_sc, y, HUBER_DELTA, sg, MAX_PRED,
            )
            for seed in seeds
        )

        for name in needed_names:
            new_model_results[name][n_train] = {
                "test_huber": np.array([r[name]["test_huber"] for r in iter_results]),
                "test_rmse": np.array([r[name]["test_rmse"] for r in iter_results]),
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

elif plots_only and not config_matches:
    print("ERROR: --plots-only but no valid cache exists. Run without --plots-only first.")
    sys.exit(1)

elif needed_names and plots_only:
    missing = [clean_name(n) for n in needed_names]
    print(f"Warning: --plots-only but {len(missing)} models are not cached: {missing}")
    print("  These models will be skipped in plots.")
    model_names = [n for n in model_names if n not in needed_names]
    RUN_MODELS = [(n, f, x, l, c, s)
                  for n, f, x, l, c, s in RUN_MODELS if n not in needed_names]
    model_specs = [(n, f, FEATURE_KINDS[n]) for n, f, *_ in RUN_MODELS]

# Build all_results[n_train][model_name] from cache for plotting
all_results = {}
for n_train in TRAIN_SIZES:
    size_results = {}
    for name in model_names:
        size_results[name] = cached_models[name][n_train]
    all_results[n_train] = size_results


# =========================================================================
# Table 1: Models ranked at n_train=REFERENCE_N
# =========================================================================
print(f"\n{'=' * 100}")
print(f"TABLE 1: Models ranked by held-out Huber loss (n_train={REFERENCE_N})")
print(f"{'=' * 100}")
header = (
    f"{'Model':<22} {'Huber(test)':>14} {'RMSE(test)':>12} {'RMSE(train)':>12}"
    f" {'Overfit':>8} {'Opt p1':>14} {'Conv%':>6}"
)
print(header)
print("-" * 100)

ref = all_results[REFERENCE_N]
rankings = []
for name in model_names:
    d = ref[name]
    mask = d["success"]
    if mask.sum() == 0:
        rankings.append((name, np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        continue
    h_vals = d["test_huber"][mask]
    t_rmse = d["test_rmse"][mask]
    tr_rmse = d["train_rmse"][mask]
    opt_vals = d["opt_p1"][mask]
    rankings.append((
        name,
        float(np.mean(h_vals)), float(np.std(h_vals)),
        float(np.mean(t_rmse)),
        float(np.mean(tr_rmse)),
        float(np.mean(t_rmse)) / max(float(np.mean(tr_rmse)), 1e-12),
        float(np.mean(opt_vals)), float(np.std(opt_vals)),
        float(mask.mean()) * 100,
    ))

rankings.sort(key=lambda x: x[1])

for name, h_mean, h_std, rmse_test, rmse_train, overfit, opt_mean, opt_std, conv in rankings:
    cn = clean_name(name)
    print(
        f"{cn:<22} {h_mean:>7.5f}+-{h_std:<5.5f} {rmse_test:>10.5f}"
        f" {rmse_train:>12.5f} {overfit:>8.2f}x"
        f" {opt_mean:>6.3f}+-{opt_std:<5.3f} {conv:>5.0f}%"
    )

top_models = [r[0] for r in rankings[:min(5, len(rankings))]]
print(f"\nTop models by held-out Huber: {[clean_name(n) for n in top_models]}")


# =========================================================================
# Helper: extract learning curve stats for one model
# =========================================================================
def _learning_curve_stats(name):
    medians, q25, q75 = [], [], []
    for n_train in TRAIN_SIZES:
        d = all_results[n_train][name]
        mask = d["success"]
        vals = d["test_huber"][mask]
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


# =========================================================================
# Figure 1: Learning curves overview (all + top 5)
# =========================================================================
print("\nGenerating Figure 1: Learning curves (overview)...")

n_panels = 2 if len(model_names) > 5 else 1
fig1, axes1 = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
if n_panels == 1:
    axes1 = [axes1]

panel_configs = []
if n_panels == 2:
    panel_configs.append((0, model_names, False, f"All {len(model_names)} models"))
    panel_configs.append((1, top_models, True, f"Top {min(5, len(model_names))}"))
else:
    panel_configs.append((0, model_names, True, "All models"))

for ax_idx, subset, show_ci, title in panel_configs:
    ax = axes1[ax_idx]
    for name in subset:
        medians, q25, q75 = _learning_curve_stats(name)
        color = model_colors[name]
        lw = 2.5 if show_ci else 1.2
        ax.plot(TRAIN_SIZES, medians, "o-", color=color, lw=lw, ms=4,
                label=clean_name(name))
        if show_ci:
            ax.fill_between(TRAIN_SIZES, q25, q75, color=color, alpha=0.15)

    ax.set_xlabel("$n_{\\mathrm{train}}$")
    if ax_idx == 0:
        ax.set_ylabel(f"Held-out Huber loss, median ($\\delta$={HUBER_DELTA:.4f})")
    ax.set_title(title)
    ax.legend(framealpha=0.9, loc="upper right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

fig1.suptitle("Learning Curves: Held-out Huber Loss vs Training Set Size", fontsize=13, y=1.02)
fig1.tight_layout()
out1 = OUT_DIR / "fig1_learning_curves.png"
fig1.savefig(out1)
plt.close(fig1)
print(f"  Saved {out1}")


# =========================================================================
# Figure 1b: Learning curves per-model (small multiples)
# =========================================================================
print("Generating Figure 1b: Learning curves (per-model)...")

n_all = len(model_names)
ncols1b = min(5, n_all)
nrows1b = (n_all + ncols1b - 1) // ncols1b

fig1b, axes1b = plt.subplots(nrows1b, ncols1b, figsize=(3.5 * ncols1b, 3.2 * nrows1b),
                              squeeze=False, sharex=True)

# Sort models by ranking order for consistent layout
ranked_names = [r[0] for r in rankings]

for idx, name in enumerate(ranked_names):
    row, col = divmod(idx, ncols1b)
    ax = axes1b[row][col]

    medians, q25, q75 = _learning_curve_stats(name)
    color = model_colors.get(name, "gray")

    ax.plot(TRAIN_SIZES, medians, "o-", color=color, lw=2, ms=4)
    ax.fill_between(TRAIN_SIZES, q25, q75, color=color, alpha=0.2)

    ax.set_title(clean_name(name), fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if col == 0:
        ax.set_ylabel("Huber (median)")
    if row == nrows1b - 1:
        ax.set_xlabel("$n_{\\mathrm{train}}$")

for idx in range(n_all, nrows1b * ncols1b):
    row, col = divmod(idx, ncols1b)
    axes1b[row][col].set_visible(False)

fig1b.suptitle("Learning Curves per Model (median + IQR)", fontsize=13, y=1.02)
fig1b.tight_layout()
out1b = OUT_DIR / "fig1b_learning_curves_detail.png"
fig1b.savefig(out1b)
plt.close(fig1b)
print(f"  Saved {out1b}")


# =========================================================================
# Figure 2: Overfitting gap (small multiples)
# =========================================================================
print("Generating Figure 2: Overfitting gap...")

n_show = len(rankings)
show_models = [r[0] for r in rankings[:n_show]]
ncols2 = min(5, n_show)
nrows2 = (n_show + ncols2 - 1) // ncols2

fig2, axes2 = plt.subplots(nrows2, ncols2, figsize=(3.5 * ncols2, 3.5 * nrows2),
                            squeeze=False, sharex=True)

for idx, name in enumerate(show_models):
    row, col = divmod(idx, ncols2)
    ax = axes2[row][col]

    train_meds, test_meds = [], []
    train_q25, train_q75, test_q25, test_q75 = [], [], [], []
    for n_train in TRAIN_SIZES:
        d = all_results[n_train][name]
        mask = d["success"]
        tr = d["train_rmse"][mask]
        te = d["test_rmse"][mask]
        tr, te = tr[np.isfinite(tr)], te[np.isfinite(te)]
        train_meds.append(float(np.median(tr)) if len(tr) else np.nan)
        test_meds.append(float(np.median(te)) if len(te) else np.nan)
        train_q25.append(float(np.percentile(tr, 25)) if len(tr) else np.nan)
        train_q75.append(float(np.percentile(tr, 75)) if len(tr) else np.nan)
        test_q25.append(float(np.percentile(te, 25)) if len(te) else np.nan)
        test_q75.append(float(np.percentile(te, 75)) if len(te) else np.nan)

    ax.plot(TRAIN_SIZES, train_meds, "s--", color="royalblue", lw=1.5, ms=3,
            label="Train" if idx == 0 else None)
    ax.fill_between(TRAIN_SIZES, train_q25, train_q75, color="royalblue", alpha=0.12)
    ax.plot(TRAIN_SIZES, test_meds, "o-", color="crimson", lw=1.5, ms=3,
            label="Test" if idx == 0 else None)
    ax.fill_between(TRAIN_SIZES, test_q25, test_q75, color="crimson", alpha=0.12)

    ax.set_title(clean_name(name), fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if col == 0:
        ax.set_ylabel("RMSE (median)")
    if row == nrows2 - 1:
        ax.set_xlabel("$n_{\\mathrm{train}}$")

for idx in range(n_show, nrows2 * ncols2):
    row, col = divmod(idx, ncols2)
    axes2[row][col].set_visible(False)

_leg_handles = [
    Line2D([], [], color="royalblue", ls="--", marker="s", ms=4, lw=1.5, label="Train"),
    Line2D([], [], color="crimson", ls="-", marker="o", ms=4, lw=1.5, label="Test"),
]
fig2.legend(handles=_leg_handles, loc="upper right", framealpha=0.9, fontsize=10)
fig2.suptitle("Overfitting Gap: Train (dashed) vs Test (solid) RMSE", fontsize=13, y=1.02)
fig2.tight_layout()
out2 = OUT_DIR / "fig2_overfitting_gap.png"
fig2.savefig(out2)
plt.close(fig2)
print(f"  Saved {out2}")


# =========================================================================
# Figure 3: Stability of predicted optimum + BPB sanity check
# =========================================================================
print("Generating Figure 3: Stability of predicted optimum...")

ref_data = all_results[REFERENCE_N]

# Sort models by ranking order (increasing held-out loss)
ranked_names = [r[0] for r in rankings]

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

# Compute predicted BPB at each model's median predicted optimum (full-data fit)
predicted_bpb_at_opt = {}
for name, fit_fn, X_data, *_rest in RUN_MODELS:
    kind = FEATURE_KINDS[name]
    median_p1 = next(m for n, _, m in opt_data if n == name)
    if not np.isfinite(median_p1):
        predicted_bpb_at_opt[name] = np.nan
        continue
    try:
        pred_fn, _ = fit_fn(X_data, y)
        Xs = _make_slice_for_kind(kind, sg)
        slice_pred = pred_fn(Xs)
        # Interpolate BPB at median_p1
        predicted_bpb_at_opt[name] = float(np.interp(median_p1, sg, slice_pred))
    except Exception:
        predicted_bpb_at_opt[name] = np.nan

fig_width = max(8, 1.3 * len(opt_data))
fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(fig_width, 8),
                                   gridspec_kw={"height_ratios": [3, 2]},
                                   sharex=True)

bp_data = [vals for _, vals, _ in opt_data]
bp_labels = [clean_name(n) for n, _, _ in opt_data]
bp_colors = [model_colors.get(n, "gray") for n, _, _ in opt_data]
x_pos = np.arange(1, len(opt_data) + 1)

# Panel A: Box plots of predicted p1
bplot = ax3a.boxplot(bp_data, labels=bp_labels, patch_artist=True, widths=0.6,
                     medianprops=dict(color="black", lw=1.5),
                     flierprops=dict(marker="o", ms=3, alpha=0.5))
for patch, color in zip(bplot["boxes"], bp_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.4)

ax3a.axhline(observed_best_p1, color="red", ls="--", lw=2, zorder=0,
             label=f"Observed best $p_1$={observed_best_p1:.3f}")
ax3a.set_ylabel("Predicted optimal $p_1^{\\mathrm{sc}}$")
ax3a.set_title(f"Stability of Predicted Optimum ($n_{{\\mathrm{{train}}}}$={REFERENCE_N}, B={B})")
ax3a.legend(loc="upper left", framealpha=0.9)

# Panel B: Predicted BPB at median predicted optimum (sanity check)
bpb_vals = [predicted_bpb_at_opt.get(n, np.nan) for n, _, _ in opt_data]
bar_colors = ["#2ca02c" if b <= observed_best_bpb else "#d62728"
              for b in bpb_vals]

ax3b.bar(x_pos, bpb_vals, color=bar_colors, alpha=0.7, edgecolor="black", linewidth=0.5)
ax3b.axhline(observed_best_bpb, color="red", ls="--", lw=2, zorder=0,
             label=f"Observed best BPB={observed_best_bpb:.4f}")
ax3b.set_ylabel("Predicted BPB at median $p_1^*$")
ax3b.set_title("Sanity Check: predicted BPB at median optimum (green $=$ improves on observed)")
ax3b.legend(loc="upper right", framealpha=0.9, fontsize=8)
ax3b.set_xlim(0.3, len(opt_data) + 0.7)

# Zoom y-axis to show the differences clearly
finite_bpbs = [b for b in bpb_vals if np.isfinite(b)]
if finite_bpbs:
    ylo = min(min(finite_bpbs), observed_best_bpb) - 0.005
    yhi = max(max(finite_bpbs), observed_best_bpb) + 0.005
    ax3b.set_ylim(ylo, yhi)

# Annotate each bar with the BPB value
for i, bpb in enumerate(bpb_vals):
    if np.isfinite(bpb):
        ax3b.text(x_pos[i], bpb + 0.0005, f"{bpb:.4f}", ha="center", va="bottom",
                  fontsize=7)

plt.setp(ax3b.get_xticklabels(), rotation=30, ha="right")

fig3.tight_layout()
out3 = OUT_DIR / "fig3_stability.png"
fig3.savefig(out3)
plt.close(fig3)
print(f"  Saved {out3}")


# =========================================================================
# Figure 4: 2D heatmaps of predicted BPB with isoloss contours
# =========================================================================
print("Generating Figure 4: 2D heatmaps...")

g = np.linspace(0.005, 0.995, HEATMAP_RES)

# Sort models by ranking order (increasing held-out loss)
ranked_model_info = []
rank_order = {r[0]: i for i, r in enumerate(rankings)}
for name, fit_fn, X_data, label_fn, color, ls in RUN_MODELS:
    ranked_model_info.append((rank_order.get(name, 999), name, fit_fn, X_data))
ranked_model_info.sort()

n_models = len(ranked_model_info)
ncols4 = min(4, n_models)
nrows4 = (n_models + ncols4 - 1) // ncols4

cell_size = 4.2
fig4, axes4 = plt.subplots(
    nrows4, ncols4,
    figsize=(cell_size * ncols4 + 1.8, cell_size * nrows4 + 1.0),
    squeeze=False,
    gridspec_kw={"hspace": 0.35, "wspace": 0.3},
)

vmin_global = float(np.min(y)) - 0.05
vmax_global = float(np.percentile(y, 90))
n_contour_levels = 15

last_scatter = None
for mi, (_, name, fit_fn, X_data) in enumerate(ranked_model_info):
    row, col = divmod(mi, ncols4)
    ax = axes4[row][col]
    kind = FEATURE_KINDS[name]

    opt_p0, opt_p1_val, opt_bpb = np.nan, np.nan, np.nan

    try:
        pred_fn, _ = fit_fn(X_data, y)
        X_grid, P0, P1 = make_2d_grid(g, g, kind=kind)
        Z = pred_fn(X_grid).reshape(P0.shape)
        Z_clipped = np.clip(Z, vmin_global, vmax_global + 0.5)

        # Filled contour
        ax.contourf(P0, P1, Z_clipped, levels=n_contour_levels,
                     cmap="RdYlGn_r", vmin=vmin_global, vmax=vmax_global,
                     extend="both")
        # Isoloss contour lines with labels
        cs = ax.contour(P0, P1, Z_clipped, levels=n_contour_levels,
                        colors="black", linewidths=0.4, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")

        # Predicted optimum
        opt_idx = int(np.argmin(Z))
        opt_p0 = float(P0.ravel()[opt_idx])
        opt_p1_val = float(P1.ravel()[opt_idx])
        opt_bpb = float(Z.ravel()[opt_idx])
        ax.plot(opt_p0, opt_p1_val, marker="*", ms=14, color="gold",
                markeredgecolor="black", markeredgewidth=1.2, zorder=6)

        # Label the optimum with mixture values and predicted BPB
        ax.annotate(
            f"({opt_p0:.2f}, {opt_p1_val:.2f})\nBPB={opt_bpb:.3f}",
            xy=(opt_p0, opt_p1_val),
            xytext=(0.97, 0.03), textcoords="axes fraction",
            fontsize=7, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            zorder=7,
        )

    except Exception as e:
        ax.text(0.5, 0.5, f"Fit failed:\n{e}", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="red")

    # Training runs overlay: conspicuous bordered circles
    last_scatter = ax.scatter(p0_sc, p1_sc, c=y, cmap="RdYlGn_r",
                              vmin=vmin_global, vmax=vmax_global,
                              s=40, edgecolors="white", linewidths=1.2, zorder=5)

    ax.set_title(clean_name(name), fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if row == nrows4 - 1:
        ax.set_xlabel("Phase 0 StarCoder weight")
    if col == 0:
        ax.set_ylabel("Phase 1 StarCoder weight")

# Hide unused subplots
for idx in range(n_models, nrows4 * ncols4):
    row, col = divmod(idx, ncols4)
    axes4[row][col].set_visible(False)

# Shared colorbar placed on the right, outside the subplots
if last_scatter is not None:
    fig4.colorbar(last_scatter, ax=axes4.ravel().tolist(), label="BPB",
                  shrink=0.7, pad=0.03, aspect=30)

fig4.suptitle(
    "Predicted BPB over Weight Space (full-data fit, $\\bigstar$ = predicted optimum)",
    fontsize=13,
)
out4 = OUT_DIR / "fig4_heatmaps.png"
fig4.savefig(out4)
plt.close(fig4)
print(f"  Saved {out4}")


print(f"\nAll outputs saved to {OUT_DIR}/")
print("Done.")
