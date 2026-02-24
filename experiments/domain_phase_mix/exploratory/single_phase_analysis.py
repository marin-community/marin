# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "joblib"]
# ///
"""Fit CEQ-SUM Soft (and low-param comparison models) on single-phase 18-domain CC data.

Loads single_phase_epoch.csv and/or single_phase_no_epoch.csv, constructs DatasetSpec
objects for each, and produces:
  1. Learning curve plots (bootstrap train vs test RMSE/Huber, small multiples)
  2. Bar charts: natural (proportional) weights vs CEQ-SUM Soft predicted optimal weights

Usage:
  uv run single_phase_analysis.py                    # both epoch and no_epoch
  uv run single_phase_analysis.py --epoch             # epoch only
  uv run single_phase_analysis.py --no-epoch          # no_epoch only
  uv run single_phase_analysis.py --plots-only        # regenerate plots from cache
  uv run single_phase_analysis.py CEQ CES             # filter models by name
"""

import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
warnings.filterwarnings("ignore")

from general_scaling_models import GENERAL_MODELS, DatasetSpec  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent

# =========================================================================
# Constants
# =========================================================================
# 18 CC topics from dolma3_pool
COMMON_CRAWL_TOPICS = [
    "adult_content",
    "art_and_design",
    "crime_and_law",
    "education_and_jobs",
    "electronics_and_hardware",
    "entertainment",
    "fashion_and_beauty",
    "finance_and_business",
    "food_and_dining",
    "games",
    "health",
    "history_and_geography",
    "home_and_hobbies",
    "industrial",
    "literature",
    "politics",
    "religion",
    "science_math_and_technology",
]

# Number of quality tiers per topic (determines approximate token count)
_TIERS_PER_TOPIC = {
    "adult_content": 13,
    "art_and_design": 15,
    "crime_and_law": 18,
    "education_and_jobs": 17,
    "electronics_and_hardware": 15,
    "entertainment": 15,
    "fashion_and_beauty": 14,
    "finance_and_business": 18,
    "food_and_dining": 16,
    "games": 16,
    "health": 18,
    "history_and_geography": 16,
    "home_and_hobbies": 15,
    "industrial": 17,
    "literature": 16,
    "politics": 18,
    "religion": 17,
    "science_math_and_technology": 16,
}

# ~28.1B tokens per partition (average from dolma3_pool.py)
_TOKENS_PER_PARTITION_B = 28.1

DOMAIN_NAMES = list(COMMON_CRAWL_TOPICS)
M = len(DOMAIN_NAMES)
EXPERIMENT_BUDGET = 1_000_000_000

TARGET = os.environ.get("TARGET", "eval/paloma/c4_en/bpb")

# Bootstrap configuration
TRAIN_SIZES = list(range(25, 56, 5))  # [25, 30, 35, 40, 45, 50, 55] — 7 sizes
B = int(os.environ.get("B", "50"))
SEED_BASE = 42
TIMEOUT_S = 120.0
N_JOBS = int(os.environ.get("N_JOBS", "-1"))

# Low-param models suitable for R=62, M=18 (all have <= 25 params)
LOW_PARAM_MODELS = {
    "Linear",        # 19p
    "LogLinear",     # 19p
    "CES",           # 21p
    "CEQ-SUM soft",  # 23p
    "CEQ-SUM hinge", # 23p
    "CEQ-SUM(k)",    # 24p
    "NCEQ",          # 24p
    "NCEQ(k)",       # 25p
    "FM-CEQ",        # 23p
}

# =========================================================================
# Matplotlib style
# =========================================================================
mpl.rcParams.update(
    {
        "text.usetex": False,
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


# =========================================================================
# Data loading
# =========================================================================
def _domain_sizes_b() -> np.ndarray:
    """Approximate token count per domain in billions."""
    return np.array([_TIERS_PER_TOPIC[t] * _TOKENS_PER_PARTITION_B for t in COMMON_CRAWL_TOPICS])


def compute_natural_weights() -> np.ndarray:
    """Proportional weights based on domain token counts."""
    sizes = _domain_sizes_b()
    return sizes / sizes.sum()


def load_single_phase_spec(csv_path: Path, target: str, epoch: bool) -> DatasetSpec:
    """Load single-phase CSV and construct DatasetSpec (N=1, M=18)."""
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "completed"].reset_index(drop=True)
    R = len(df)

    weight_cols = [f"phase_0_{d}" for d in DOMAIN_NAMES]
    W = df[weight_cols].values[:, np.newaxis, :]  # (R, 1, 18)
    y = df[target].values.astype(float)

    valid = ~np.isnan(y)
    if not valid.all():
        print(f"  Dropping {(~valid).sum()} rows with NaN target")
        W, y = W[valid], y[valid]

    # Epoch multipliers: C[d] = target_budget / domain_tokens[d]
    domain_tokens = _domain_sizes_b() * 1e9  # convert to raw tokens
    total_cc = float(domain_tokens.sum())
    target_budget = total_cc if epoch else float(EXPERIMENT_BUDGET)
    C = (target_budget / domain_tokens).reshape(1, M)  # (1, 18)

    label = "epoch" if epoch else "no_epoch"
    return DatasetSpec(
        weights=W,
        y=y,
        epoch_multipliers=C,
        domain_names=DOMAIN_NAMES,
        phase_names=["phase_0"],
        small_domains=None,  # all 18 domains
        name=f"single_phase_{label}",
    )


# =========================================================================
# Huber loss
# =========================================================================
def huber_loss(y_true, y_pred, delta):
    """Element-wise Huber loss, averaged."""
    r = y_true - y_pred
    abs_r = np.abs(r)
    return float(
        np.mean(np.where(abs_r <= delta, 0.5 * r**2, delta * (abs_r - 0.5 * delta)))
    )


# =========================================================================
# Single bootstrap iteration
# =========================================================================
def _run_one_iter(
    n_train: int,
    seed: int,
    model_names_list: list[str],
    full_weights: np.ndarray,
    full_y: np.ndarray,
    epoch_multipliers: np.ndarray,
    domain_names: list[str],
    phase_names: list[str],
    small_domains: list[int],
    delta: float,
    max_pred: float,
) -> dict[str, dict]:
    """Run one bootstrap iteration for all models at a given n_train."""
    warnings.filterwarnings("ignore")

    _script_dir = str(Path(__file__).resolve().parent)
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)

    from general_scaling_models import GENERAL_MODELS as _GM

    model_map = {m.name: m for m in _GM}

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(full_y))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    train_spec = DatasetSpec(
        weights=full_weights[train_idx],
        y=full_y[train_idx],
        epoch_multipliers=epoch_multipliers,
        domain_names=domain_names,
        phase_names=phase_names,
        small_domains=small_domains,
        name="train",
    )
    y_test = full_y[test_idx]
    W_test = full_weights[test_idx]
    W_train = full_weights[train_idx]

    fail = {
        "test_rmse": np.nan,
        "train_rmse": np.nan,
        "test_huber": np.nan,
        "train_huber": np.nan,
        "n_params": 0,
        "success": False,
    }

    results = {}
    for name in model_names_list:
        model = model_map.get(name)
        if model is None or not model.applicable(train_spec):
            results[name] = dict(fail)
            continue

        try:
            pred_fn, info = model.fit_fn(train_spec)
        except Exception:
            results[name] = dict(fail)
            continue

        try:
            test_pred = pred_fn(W_test)
            train_pred = pred_fn(W_train)

            if (
                np.any(np.abs(test_pred) > max_pred)
                or np.any(np.isnan(test_pred))
                or np.any(np.abs(train_pred) > max_pred)
                or np.any(np.isnan(train_pred))
            ):
                results[name] = dict(fail)
                continue

            results[name] = {
                "test_rmse": float(np.sqrt(np.mean((y_test - test_pred) ** 2))),
                "train_rmse": float(np.sqrt(np.mean((train_spec.y - train_pred) ** 2))),
                "test_huber": huber_loss(y_test, test_pred, delta),
                "train_huber": huber_loss(train_spec.y, train_pred, delta),
                "n_params": info.get("n_params", 0),
                "success": True,
            }
        except Exception:
            results[name] = dict(fail)

    return results


# =========================================================================
# Cache management
# =========================================================================
def _load_cache(cache_file: Path):
    if not cache_file.exists():
        return None, {}
    try:
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        return cache.get("config"), cache.get("models", {})
    except Exception as e:
        print(f"  Warning: cache load failed ({e}), recomputing all models")
        return None, {}


def _save_cache(cache_file: Path, config, model_results):
    with open(cache_file, "wb") as f:
        pickle.dump({"config": config, "models": model_results}, f)


# =========================================================================
# Plotting: overfitting gap (small multiples)
# =========================================================================
def _adaptive_ylim(y_maxes, factor=1.5):
    arr = np.array(y_maxes)
    q75 = float(np.percentile(arr, 75))
    threshold = factor * q75
    outliers = set(int(i) for i, v in enumerate(arr) if v > threshold)
    non_outlier = arr[arr <= threshold]
    shared_ylim = float(np.max(non_outlier)) if len(non_outlier) > 0 else float(np.max(arr))
    shared_ylim *= 1.05
    return shared_ylim, outliers


def _plot_overfitting_gap(metric_id, all_results, model_names_list, train_sizes, n_params_map, out_dir, suffix=""):
    """Plot train vs test metric per model as small multiples."""
    test_key = f"test_{metric_id}"
    train_key = f"train_{metric_id}"
    label = metric_id.upper()

    max_ts = train_sizes[-1]
    sort_key = {}
    for name in model_names_list:
        d = all_results[max_ts][name]
        mask = d["success"]
        vals = d[test_key][mask]
        vals = vals[np.isfinite(vals)]
        sort_key[name] = float(np.median(vals)) if len(vals) > 0 else np.inf

    ranked_names = sorted(model_names_list, key=lambda n: sort_key[n])

    ncols = min(5, len(ranked_names))
    nrows = (len(ranked_names) + ncols - 1) // ncols

    all_stats = []
    y_maxes = []
    for name in ranked_names:
        train_meds, test_meds = [], []
        train_q25, train_q75, test_q25, test_q75 = [], [], [], []
        for n_train in train_sizes:
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

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows), squeeze=False, sharex=True
    )

    for idx, name in enumerate(ranked_names):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        train_meds, test_meds, train_q25, train_q75, test_q25, test_q75 = all_stats[idx]

        ax.plot(train_sizes, train_meds, "s--", color="royalblue", lw=1.5, ms=3,
                label="Train" if idx == 0 else None)
        ax.fill_between(train_sizes, train_q25, train_q75, color="royalblue", alpha=0.12)
        ax.plot(train_sizes, test_meds, "o-", color="crimson", lw=1.5, ms=3,
                label="Test" if idx == 0 else None)
        ax.fill_between(train_sizes, test_q25, test_q75, color="crimson", alpha=0.12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        np_val = n_params_map.get(name, "?")
        np_str = f"{np_val}p"
        if idx in outlier_idxs:
            ax.set_title(f"{name} ({np_str}) * diff. scale", fontsize=8, color="firebrick")
            for spine in ax.spines.values():
                spine.set_edgecolor("firebrick")
                spine.set_linewidth(1.5)
        else:
            ax.set_ylim(0, shared_ylim)
            ax.set_title(f"{name} ({np_str})", fontsize=9)

        if c == 0:
            ax.set_ylabel(f"{label} (median)")
        if r == nrows - 1:
            ax.set_xlabel("n_train")

    for idx in range(len(ranked_names), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    _leg = [
        Line2D([], [], color="royalblue", ls="--", marker="s", ms=4, lw=1.5, label="Train"),
        Line2D([], [], color="crimson", ls="-", marker="o", ms=4, lw=1.5, label="Test"),
    ]
    fig.legend(handles=_leg, loc="upper right", framealpha=0.9, fontsize=10)
    fig.suptitle(f"Overfitting Gap: Train vs Test {label}", fontsize=13, y=1.02)
    fig.tight_layout()
    out = out_dir / f"overfitting_gap_{suffix}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# =========================================================================
# Plotting: weight comparison bar chart
# =========================================================================
def _plot_weight_comparison(natural_w, optimal_w, optimal_bpb, domain_names, out_dir, label=""):
    """Grouped bar chart: natural vs predicted optimal weights."""
    x = np.arange(len(domain_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, natural_w, width, label="Natural", color="#3b5a6e", edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, optimal_w, width, label="Ours (CEQ-SUM Soft)", color="#d4738a", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Weight")
    ax.set_xlabel("Domain")
    ax.set_xticks(x)
    short_names = [n.replace("_", " ") for n in domain_names]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_title(f"Natural vs Predicted Optimal Weights ({label}, pred BPB={optimal_bpb:.4f})", fontsize=12)

    fig.tight_layout()
    out = out_dir / "optimal_weights.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# =========================================================================
# Optimal weight finding (18D simplex optimization)
# =========================================================================
def find_optimal_weights(pred_fn, n_restarts=200, seed=42):
    """Find weight vector on the M-simplex that minimizes pred_fn."""
    best_val, best_w = np.inf, None
    rng = np.random.RandomState(seed)

    for _ in range(n_restarts):
        w0 = rng.dirichlet(np.ones(M))

        def obj(w):
            W = w.reshape(1, 1, M)
            return float(pred_fn(W)[0])

        res = minimize(
            obj, w0,
            method="SLSQP",
            bounds=[(1e-4, 1.0)] * M,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        if res.success and np.isfinite(res.fun) and res.fun < best_val:
            best_val, best_w = float(res.fun), res.x.copy()

    if best_w is None:
        print("  WARNING: optimization failed on all restarts")
        return np.ones(M) / M, np.nan

    best_w = np.maximum(best_w, 0.0)
    best_w /= best_w.sum()
    return best_w, best_val


# =========================================================================
# Run analysis for one mode (epoch or no_epoch)
# =========================================================================
def run_analysis(
    spec: DatasetSpec,
    run_models: list,
    out_dir: Path,
    label: str,
    plots_only: bool = False,
):
    """Run full analysis pipeline for one DatasetSpec."""
    out_dir.mkdir(exist_ok=True)
    cache_file = out_dir / "bootstrap_cache.pkl"

    y = spec.y
    max_pred = 10.0 * float(np.max(y))

    print(f"\n{'='*60}")
    print(f"Analysis: {label}")
    print(f"{'='*60}")
    print(f"  R={spec.R}, N={spec.N}, M={spec.M}")
    print(f"  Target: {TARGET}")
    print(f"  y range: [{y.min():.4f}, {y.max():.4f}], median={np.median(y):.4f}")

    # Compute Huber delta from Linear model residuals
    linear_model = next((m for m in GENERAL_MODELS if m.name == "Linear"), None)
    if linear_model is not None:
        try:
            _linear_pred_fn, _ = linear_model.fit_fn(spec)
            _linear_resid = y - _linear_pred_fn(spec.weights)
            _mad = np.median(np.abs(_linear_resid - np.median(_linear_resid)))
            huber_delta = 1.345 * _mad / 0.6745
        except Exception:
            huber_delta = 1.0
    else:
        huber_delta = 1.0
    print(f"  Huber delta = {huber_delta:.6f}")

    # Filter to low-param models and fit on full data
    valid_models = [m for m in run_models if m.name in LOW_PARAM_MODELS and m.applicable(spec)]
    model_names = [m.name for m in valid_models]
    print(f"\n  Models ({len(model_names)}):")
    for name in model_names:
        print(f"    {name}")

    print("\n  Fitting on full data...")
    n_params_map: dict[str, int] = {}
    full_fit_cache: dict[str, tuple] = {}
    for model in valid_models:
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(model.fit_fn, spec)
                pred_fn, info = future.result(timeout=TIMEOUT_S)
            np_count = info.get("n_params", 0)
            n_params_map[model.name] = np_count
            full_fit_cache[model.name] = (pred_fn, info)
            print(f"    {model.name:30s}  n_params={np_count}")
        except FuturesTimeoutError:
            print(f"    {model.name:30s}  TIMEOUT ({TIMEOUT_S}s)")
            model_names = [n for n in model_names if n != model.name]
        except Exception as e:
            print(f"    {model.name:30s}  FAILED: {e}")
            model_names = [n for n in model_names if n != model.name]

    if not model_names:
        print("  No models passed param filter, skipping bootstrap")
        return

    # Bootstrap
    bootstrap_config = {
        "train_sizes": TRAIN_SIZES,
        "B": B,
        "seed_base": SEED_BASE,
        "max_pred": max_pred,
        "target": TARGET,
        "timeout_s": TIMEOUT_S,
    }

    cached_config, cached_models = _load_cache(cache_file)
    config_matches = cached_config == bootstrap_config

    if config_matches:
        cached_names = set(cached_models.keys())
        needed_names = [n for n in model_names if n not in cached_names]
        print(f"\n  Cache: {len(cached_names)} models cached, {len(needed_names)} to compute")
    else:
        if cached_config is not None:
            print("\n  Cache: config mismatch, recomputing all models")
        else:
            print("\n  Cache: no cache found, computing all models")
        cached_models = {}
        needed_names = list(model_names)

    if needed_names and not plots_only:
        print(f"  Running bootstrap: {len(TRAIN_SIZES)} sizes x {B} iters x {len(needed_names)} models")
        t0 = time.time()

        new_model_results: dict[str, dict] = {name: {} for name in needed_names}

        _w = spec.weights
        _y = spec.y
        _em = spec.epoch_multipliers
        _dn = list(spec.domain_names)
        _pn = list(spec.phase_names)
        _sd = list(spec.small_domains) if spec.small_domains else []

        for si, n_train in enumerate(TRAIN_SIZES):
            seeds = [SEED_BASE + si * B + b for b in range(B)]

            iter_results = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(_run_one_iter)(
                    n_train, seed, needed_names,
                    _w, _y, _em, _dn, _pn, _sd,
                    huber_delta, max_pred,
                )
                for seed in seeds
            )

            for name in needed_names:
                new_model_results[name][n_train] = {
                    "test_rmse": np.array([r[name]["test_rmse"] for r in iter_results]),
                    "train_rmse": np.array([r[name]["train_rmse"] for r in iter_results]),
                    "test_huber": np.array([r[name]["test_huber"] for r in iter_results]),
                    "train_huber": np.array([r[name]["train_huber"] for r in iter_results]),
                    "n_params": np.array([r[name]["n_params"] for r in iter_results]),
                    "success": np.array([r[name]["success"] for r in iter_results]),
                }

            elapsed = time.time() - t0
            print(f"    n_train={n_train:3d}  done  ({elapsed:.1f}s elapsed)")

        total_time = time.time() - t0
        print(f"  Bootstrap complete in {total_time:.1f}s")

        cached_models.update(new_model_results)
        _save_cache(cache_file, bootstrap_config, cached_models)
        print(f"  Cache saved ({len(cached_models)} models total)")

    elif plots_only and not config_matches:
        print("  ERROR: --plots-only but no valid cache exists. Run without --plots-only first.")
        return

    elif needed_names and plots_only:
        print(f"  Warning: --plots-only but {len(needed_names)} models not cached, skipping them")
        model_names = [n for n in model_names if n not in needed_names]

    # Build all_results from cache
    all_results: dict[int, dict[str, dict]] = {}
    for n_train in TRAIN_SIZES:
        size_results = {}
        for name in model_names:
            if name in cached_models and n_train in cached_models[name]:
                size_results[name] = cached_models[name][n_train]
            else:
                size_results[name] = {
                    "test_rmse": np.array([np.nan] * B),
                    "train_rmse": np.array([np.nan] * B),
                    "test_huber": np.array([np.nan] * B),
                    "train_huber": np.array([np.nan] * B),
                    "success": np.array([False] * B),
                }
        all_results[n_train] = size_results

    # Generate overfitting gap plots
    print("\n  Generating overfitting gap plots...")
    for metric_id, suffix in [("rmse", "rmse"), ("huber", "huber")]:
        _plot_overfitting_gap(
            metric_id, all_results, model_names, TRAIN_SIZES, n_params_map, out_dir, suffix
        )

    # Print summary table
    print(f"\n  {'Model':30s} {'RMSE(test)':>12s} {'RMSE(train)':>12s} {'n_params':>9s} {'Conv%':>6s}")
    print("  " + "-" * 75)

    ref = all_results[TRAIN_SIZES[-1]]
    rows = []
    for name in model_names:
        d = ref[name]
        mask = d["success"]
        te = d["test_rmse"][mask]
        tr = d["train_rmse"][mask]
        te, tr = te[np.isfinite(te)], tr[np.isfinite(tr)]
        med_te = float(np.median(te)) if len(te) > 0 else np.inf
        med_tr = float(np.median(tr)) if len(tr) > 0 else np.inf
        conv = float(mask.mean()) * 100
        np_val = n_params_map.get(name, "?")
        rows.append((name, med_te, med_tr, np_val, conv))

    rows.sort(key=lambda x: x[1])
    for name, med_te, med_tr, np_val, conv in rows:
        te_s = f"{med_te:.6f}" if np.isfinite(med_te) else "N/A"
        tr_s = f"{med_tr:.6f}" if np.isfinite(med_tr) else "N/A"
        print(f"  {name:30s} {te_s:>12s} {tr_s:>12s} {str(np_val):>9s} {conv:>5.0f}%")

    # Find optimal weights for CEQ-SUM soft
    ceq_name = "CEQ-SUM soft"
    if ceq_name in full_fit_cache:
        print(f"\n  Finding optimal weights ({ceq_name})...")
        pred_fn, _ = full_fit_cache[ceq_name]
        optimal_w, optimal_bpb = find_optimal_weights(pred_fn)
        natural_w = compute_natural_weights()

        print(f"  Predicted optimal BPB: {optimal_bpb:.4f}")
        print(f"\n  {'Domain':35s} {'Natural':>8s} {'Optimal':>8s} {'Ratio':>8s}")
        print("  " + "-" * 62)
        for i, name in enumerate(DOMAIN_NAMES):
            ratio = optimal_w[i] / natural_w[i] if natural_w[i] > 0 else 0
            print(f"  {name:35s} {natural_w[i]:8.4f} {optimal_w[i]:8.4f} {ratio:8.2f}x")

        _plot_weight_comparison(natural_w, optimal_w, optimal_bpb, DOMAIN_NAMES, out_dir, label)
    else:
        print(f"\n  WARNING: {ceq_name} not in full_fit_cache, skipping optimal weight search")

    print(f"\n  All outputs saved to {out_dir}/")


# =========================================================================
# Main
# =========================================================================
def main():
    cli_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    plots_only = "--plots-only" in sys.argv
    run_epoch = "--epoch" in sys.argv
    run_no_epoch = "--no-epoch" in sys.argv

    # Default: run both
    if not run_epoch and not run_no_epoch:
        run_epoch = True
        run_no_epoch = True

    # Filter models by name if CLI args provided
    if cli_args:
        run_models = [
            m for m in GENERAL_MODELS
            if any(f.lower() in m.name.lower() for f in cli_args)
        ]
        if not run_models:
            print(f"No models match filters: {cli_args}")
            print("Available:", [m.name for m in GENERAL_MODELS if m.name in LOW_PARAM_MODELS])
            sys.exit(1)
    else:
        run_models = [m for m in GENERAL_MODELS if m.name in LOW_PARAM_MODELS]

    print(f"Candidate models: {len(run_models)}")
    print(f"Target: {TARGET}")
    print(f"Modes: {'epoch' if run_epoch else ''} {'no_epoch' if run_no_epoch else ''}")

    if run_epoch:
        csv_path = SCRIPT_DIR / "single_phase_epoch.csv"
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found")
        else:
            spec = load_single_phase_spec(csv_path, TARGET, epoch=True)
            out_dir = SCRIPT_DIR / "single_phase_plots_epoch"
            run_analysis(spec, run_models, out_dir, "epoch", plots_only)

    if run_no_epoch:
        csv_path = SCRIPT_DIR / "single_phase_no_epoch.csv"
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found")
        else:
            spec = load_single_phase_spec(csv_path, TARGET, epoch=False)
            out_dir = SCRIPT_DIR / "single_phase_plots_no_epoch"
            run_analysis(spec, run_models, out_dir, "no_epoch", plots_only)

    print("\nDone.")


if __name__ == "__main__":
    main()
