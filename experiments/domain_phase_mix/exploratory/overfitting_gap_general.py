# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "joblib"]
# ///
"""Overfitting gap analysis for generalized scaling models on 2-phase starcoder data.

Bootstrap results are cached to overfitting_gap_general_plots/bootstrap_cache.pkl.
Only missing models are fitted when new models are added to GENERAL_MODELS.

Produces:
  Figure 1  – Overfitting gap (train vs test RMSE, small multiples)
  Figure 2  – Overfitting gap (train vs test Huber, small multiples)

Usage:
  uv run overfitting_gap_general.py                   # all models
  uv run overfitting_gap_general.py CEQ               # subset (substring match)
  uv run overfitting_gap_general.py --plots-only      # skip bootstrap, regenerate plots
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
warnings.filterwarnings("ignore")

from general_scaling_models import GENERAL_MODELS, DatasetSpec  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent

# =========================================================================
# Constants
# =========================================================================
TARGET_BUDGET = 5_729_908_864_777  # Nemotron full token count
NEMOTRON_TOKENS = 5_729_908_864_777
STARCODER_TOKENS = 217_000_000_000
PHASE_FRACS = np.array([0.5, 0.5])
DOMAIN_NAMES = ["nemotron_full", "starcoder"]
PHASE_NAMES = ["phase_0", "phase_1"]

TARGET = os.environ.get("TARGET", "eval/paloma/dolma_100_programing_languages/bpb")
_csv_name = os.environ.get("DATA_CSV", "two_phase_starcoder.csv")

# =========================================================================
# Data loading
# =========================================================================


def build_starcoder_spec(csv_path: Path, target: str) -> DatasetSpec:
    """Load two_phase_starcoder.csv and construct DatasetSpec (N=2, M=2).

    Builds the weight array and epoch_multipliers manually rather than using
    load_dataset_spec, because the CSV contains phase_K_domain_epochs columns
    that _extract_phase_domain_columns would incorrectly treat as domains.
    """
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "completed"].reset_index(drop=True)
    R = len(df)

    weights = np.zeros((R, 2, 2))
    weights[:, 0, 0] = df["phase_0_nemotron_full"].values
    weights[:, 0, 1] = df["phase_0_starcoder"].values
    weights[:, 1, 0] = df["phase_1_nemotron_full"].values
    weights[:, 1, 1] = df["phase_1_starcoder"].values

    y = df[target].values.astype(float)
    valid = ~np.isnan(y)
    if not valid.all():
        print(f"  Dropping {(~valid).sum()} rows with NaN target")
        weights = weights[valid]
        y = y[valid]

    domain_tokens = np.array([NEMOTRON_TOKENS, STARCODER_TOKENS])
    epoch_mults = np.array(
        [[pf * TARGET_BUDGET / dt for dt in domain_tokens] for pf in PHASE_FRACS]
    )

    return DatasetSpec(
        weights=weights,
        y=y,
        epoch_multipliers=epoch_mults,
        domain_names=DOMAIN_NAMES,
        phase_names=PHASE_NAMES,
        small_domains=[1],
        name="two_phase_starcoder",
    )


full_spec = build_starcoder_spec(SCRIPT_DIR / _csv_name, TARGET)
N_DATA = full_spec.R
y = full_spec.y

print(f"Loaded {N_DATA} samples from {_csv_name}")
print(f"Target: {TARGET}")
print(f"y range: [{y.min():.4f}, {y.max():.4f}], median={np.median(y):.4f}")

# =========================================================================
# Output directory
# =========================================================================
_out_name = os.environ.get("OUT_DIR", "overfitting_gap_general_plots")
OUT_DIR = SCRIPT_DIR / _out_name
OUT_DIR.mkdir(exist_ok=True)
CACHE_FILE = OUT_DIR / "bootstrap_cache.pkl"

# =========================================================================
# Matplotlib style
# =========================================================================
mpl.rcParams.update(
    {
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
# Configuration
# =========================================================================
TRAIN_SIZES = list(range(1, 21))
B = 200
SEED_BASE = 42
TIMEOUT_S = 30.0
N_JOBS = int(os.environ.get("N_JOBS", "-1"))
MAX_PRED = 10.0 * float(np.max(y))

BOOTSTRAP_CONFIG = {
    "train_sizes": TRAIN_SIZES,
    "B": B,
    "seed_base": SEED_BASE,
    "max_pred": MAX_PRED,
    "target": TARGET,
    "timeout_s": TIMEOUT_S,
}

# =========================================================================
# CLI
# =========================================================================
plots_only = "--plots-only" in sys.argv
cli_filters = [a for a in sys.argv[1:] if not a.startswith("--")]

if cli_filters:
    selected = [
        m
        for m in GENERAL_MODELS
        if any(f.lower() in m.name.lower() for f in cli_filters)
    ]
    if not selected:
        print(f"No models match filters: {cli_filters}")
        print("Available:", [m.name for m in GENERAL_MODELS])
        sys.exit(1)
    RUN_MODELS = selected
else:
    RUN_MODELS = list(GENERAL_MODELS)

print(f"\nModels to evaluate ({len(RUN_MODELS)}):")
for m in RUN_MODELS:
    print(f"  {m.name}")


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


# Compute Huber delta from Linear model residuals on full data
_linear_model = next(m for m in GENERAL_MODELS if m.name == "Linear")
_linear_pred_fn, _ = _linear_model.fit_fn(full_spec)
_linear_resid = y - _linear_pred_fn(full_spec.weights)
_mad = np.median(np.abs(_linear_resid - np.median(_linear_resid)))
HUBER_DELTA = 1.345 * _mad / 0.6745
print(f"\nHuber delta = {HUBER_DELTA:.6f} (from Linear MAD={_mad:.6f})")

# =========================================================================
# n_params: fit each model once on full data to get structural parameter count
# =========================================================================
print("\nCollecting n_params from full-data fits...")
n_params_map: dict[str, int] = {}
for model in RUN_MODELS:
    if not model.applicable(full_spec):
        continue
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(model.fit_fn, full_spec)
            _, info = future.result(timeout=60.0)
        n_params_map[model.name] = info.get("n_params", 0)
        print(f"  {model.name:30s}  n_params={n_params_map[model.name]}")
    except FuturesTimeoutError:
        print(f"  {model.name:30s}  TIMEOUT (60s)")
    except Exception as e:
        print(f"  {model.name:30s}  FAILED: {e}")


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
    """Run one bootstrap iteration for all models at a given n_train.

    Models are looked up by name inside the worker to avoid pickling closures
    (loky backend pickles all arguments).
    """
    warnings.filterwarnings("ignore")

    # Ensure import paths are set in the worker process (loky doesn't inherit sys.path)
    _script_dir = str(Path(__file__).resolve().parent)
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)

    # Import and build model map inside the worker process
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
        model = model_map[name]
        if not model.applicable(train_spec):
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
                "train_rmse": float(
                    np.sqrt(np.mean((train_spec.y - train_pred) ** 2))
                ),
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
def _load_cache():
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
# Bootstrap execution
# =========================================================================
model_names = [m.name for m in RUN_MODELS]

cached_config, cached_models = _load_cache()
config_matches = cached_config == BOOTSTRAP_CONFIG

if config_matches:
    cached_names = set(cached_models.keys())
    needed_names = [n for n in model_names if n not in cached_names]
    print(
        f"\nCache: {len(cached_names)} models cached, {len(needed_names)} to compute"
    )
else:
    if cached_config is not None:
        print("\nCache: config mismatch, recomputing all models")
    else:
        print("\nCache: no cache found, computing all models")
    cached_models = {}
    needed_names = list(model_names)

if needed_names and not plots_only:
    needed_model_names = [n for n in needed_names]

    print(
        f"Running bootstrap: {len(TRAIN_SIZES)} sizes x {B} iters x {len(needed_model_names)} models"
    )
    t0 = time.time()

    new_model_results: dict[str, dict] = {name: {} for name in needed_model_names}

    # Pre-extract arrays for pickling (loky can't pickle DatasetSpec closures)
    _w = full_spec.weights
    _y = full_spec.y
    _em = full_spec.epoch_multipliers
    _dn = list(full_spec.domain_names)
    _pn = list(full_spec.phase_names)
    _sd = list(full_spec.small_domains) if full_spec.small_domains else []

    for si, n_train in enumerate(TRAIN_SIZES):
        seeds = [SEED_BASE + si * B + b for b in range(B)]

        iter_results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_run_one_iter)(
                n_train,
                seed,
                needed_model_names,
                _w,
                _y,
                _em,
                _dn,
                _pn,
                _sd,
                HUBER_DELTA,
                MAX_PRED,
            )
            for seed in seeds
        )

        for name in needed_model_names:
            new_model_results[name][n_train] = {
                "test_rmse": np.array(
                    [r[name]["test_rmse"] for r in iter_results]
                ),
                "train_rmse": np.array(
                    [r[name]["train_rmse"] for r in iter_results]
                ),
                "test_huber": np.array(
                    [r[name]["test_huber"] for r in iter_results]
                ),
                "train_huber": np.array(
                    [r[name]["train_huber"] for r in iter_results]
                ),
                "n_params": np.array(
                    [r[name]["n_params"] for r in iter_results]
                ),
                "success": np.array(
                    [r[name]["success"] for r in iter_results]
                ),
            }

        elapsed = time.time() - t0
        print(f"  n_train={n_train:3d}  done  ({elapsed:.1f}s elapsed)")

    total_time = time.time() - t0
    print(f"Bootstrap complete in {total_time:.1f}s")

    cached_models.update(new_model_results)
    _save_cache(BOOTSTRAP_CONFIG, cached_models)
    print(f"Cache saved ({len(cached_models)} models total)")

elif plots_only and not config_matches:
    print(
        "ERROR: --plots-only but no valid cache exists. Run without --plots-only first."
    )
    sys.exit(1)

elif needed_names and plots_only:
    missing = needed_names
    print(
        f"Warning: --plots-only but {len(missing)} models are not cached: {missing}"
    )
    print("  These models will be skipped in plots.")
    model_names = [n for n in model_names if n not in needed_names]
    RUN_MODELS = [m for m in RUN_MODELS if m.name not in needed_names]

# Build all_results[n_train][model_name] from cache for plotting
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


# =========================================================================
# Plotting helpers
# =========================================================================
def _adaptive_ylim(y_maxes, factor=1.5):
    """Compute shared y-limit with outlier detection."""
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

    # Sort by median test metric at largest train size
    max_ts = train_sizes[-1]
    sort_key = {}
    for name in model_names_list:
        d = all_results[max_ts][name]
        mask = d["success"]
        vals = d[test_key][mask]
        vals = vals[np.isfinite(vals)]
        sort_key[name] = float(np.median(vals)) if len(vals) > 0 else np.inf

    ranked_names = sorted(model_names_list, key=lambda n: sort_key[n])

    ncols = 5
    nrows = (len(ranked_names) + ncols - 1) // ncols

    # Pre-compute stats and y-ranges
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

    # Hide empty subplots
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
# Generate plots
# =========================================================================
print("\nGenerating overfitting gap plots...")

for metric_id, suffix in [("rmse", "rmse"), ("huber", "huber")]:
    _plot_overfitting_gap(
        metric_id, all_results, model_names, TRAIN_SIZES, n_params_map, OUT_DIR, suffix
    )

# Print summary table at n_train=20
print(f"\n{'='*80}")
print(f"Model rankings by median test RMSE at n_train={TRAIN_SIZES[-1]}")
print(f"{'='*80}")
print(f"{'Model':30s} {'RMSE(test)':>12s} {'RMSE(train)':>12s} {'n_params':>9s} {'Conv%':>6s}")
print("-" * 80)

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
    print(f"{name:30s} {te_s:>12s} {tr_s:>12s} {str(np_val):>9s} {conv:>5.0f}%")

print(f"\nAll outputs saved to {OUT_DIR}/")
print("Done.")
